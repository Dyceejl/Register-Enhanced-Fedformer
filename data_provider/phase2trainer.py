# run_phase2_training.py - Complete Implementation
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from torch.utils.data import DataLoader, ConcatDataset
from phase2_models import (
    FEDformerRegisterImputation,
    MultiDomainDataset,
    load_phase1_extractors,
    create_base_config,
    DOMAIN_CONFIG,
    DATASET_CONFIGS,
)
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import warnings

warnings.filterwarnings("ignore")


class WeightedMSELoss(nn.Module):
    """Weighted MSE loss for imputation"""

    def __init__(self, missing_weight=2.0):
        super(WeightedMSELoss, self).__init__()
        self.missing_weight = missing_weight

    def forward(self, pred, true, mask):
        mse = (pred - true) ** 2
        weights = mask + (1 - mask) * self.missing_weight
        weighted_mse = mse * weights
        return weighted_mse.mean()


class Phase2Trainer:
    """Phase 2 Training Pipeline"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
        )

        # Multi-domain dataset handler
        self.domain_handler = MultiDomainDataset(DOMAIN_CONFIG)

        # Load Phase 1 extractors
        print("Loading Phase 1 trained models...")
        self.base_config = create_base_config()
        self.domain_extractors = load_phase1_extractors(
            args.phase1_checkpoint_path, args.datasets, self.base_config
        )

        # Create Phase 2 model
        print("Creating Phase 2 model...")
        self.model = self._build_model()

        # Loss functions
        self.imputation_criterion = WeightedMSELoss(args.missing_weight)
        self.domain_criterion = nn.CrossEntropyLoss()

        # Optimizer and scheduler
        self.optimizer = self._select_optimizer()
        self.scheduler = self._select_scheduler()

        # Early stopping
        self.early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    def _build_model(self):
        """Build Phase 2 model"""
        # Use largest dataset config as base
        config = self.base_config
        config.c_out = max([DATASET_CONFIGS[d]["c_out"] for d in self.args.datasets])

        model = FEDformerRegisterImputation(
            configs=config,
            domain_extractors=self.domain_extractors,
            register_size=self.args.register_size,
            num_register_tokens=self.args.num_register_tokens,
        )

        return model.to(self.device)

    def _select_optimizer(self):
        """Select optimizer"""
        if self.args.freeze_extractors:
            # Only train register and decoder
            params_to_train = []
            params_to_train.extend(self.model.ts_register.parameters())
            params_to_train.extend(self.model.decoder.parameters())
            params_to_train.extend(self.model.domain_classifier.parameters())
        else:
            # Train all parameters
            params_to_train = self.model.parameters()

        return optim.Adam(params_to_train, lr=self.args.learning_rate)

    def _select_scheduler(self):
        """Select learning rate scheduler"""
        return optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.args.lr_step_size, gamma=self.args.lr_gamma
        )

    def create_missing_mask(self, data, missing_ratio=0.2):
        """Create missing values mask"""
        mask = torch.ones_like(data)
        B, L, C = data.shape

        # Random missing pattern
        missing_indices = torch.rand(B, L, C) < missing_ratio
        mask[missing_indices] = 0

        # Create data with missing values
        data_missing = data.clone()
        data_missing[missing_indices] = float("nan")

        return data_missing, mask

    def get_multi_domain_data_loader(self, flag="train"):
        """Create multi-domain data loader"""
        all_datasets = []
        dataset_domain_map = []

        for dataset_name in self.args.datasets:
            # Create args for this dataset
            dataset_args = argparse.Namespace(**vars(self.args))
            dataset_config = DATASET_CONFIGS.get(dataset_name, {})

            # Update dataset-specific configs
            for key, value in dataset_config.items():
                setattr(dataset_args, key, value)

            dataset_args.data = dataset_name
            dataset_args.data_path = f"{dataset_name}.csv"

            # Get dataset
            dataset, _ = data_provider(dataset_args, flag)
            all_datasets.append(dataset)

            # Map dataset to domain
            domain_id = self.domain_handler.get_domain_id(dataset_name)
            dataset_domain_map.extend([domain_id] * len(dataset))

        # Combine all datasets
        combined_dataset = ConcatDataset(all_datasets)

        # Create data loader
        data_loader = DataLoader(
            combined_dataset,
            batch_size=self.args.batch_size,
            shuffle=(flag == "train"),
            num_workers=self.args.num_workers,
            drop_last=True,
        )

        return data_loader, dataset_domain_map

    def train_epoch(self, data_loader, dataset_domain_map, epoch, phase="pretrain"):
        """Train one epoch"""
        self.model.train()

        total_loss = 0.0
        total_imputation_loss = 0.0
        total_register_loss = 0.0
        total_domain_loss = 0.0

        batch_idx = 0

        for batch_data in data_loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data

            # Move to device
            batch_x = batch_x.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)

            # Get domain ID for this batch
            start_idx = batch_idx * self.args.batch_size
            end_idx = start_idx + batch_x.shape[0]
            if end_idx <= len(dataset_domain_map):
                batch_domain_ids = dataset_domain_map[start_idx:end_idx]
                # Use majority domain for this batch
                domain_id = max(set(batch_domain_ids), key=batch_domain_ids.count)
            else:
                domain_id = 0  # Default to first domain

            domain_labels = torch.tensor(
                [domain_id] * batch_x.shape[0], device=self.device
            )

            # Create missing values
            batch_x_missing, mask = self.create_missing_mask(
                batch_x, missing_ratio=self.args.missing_ratio
            )
            mask = mask.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            outputs = self.model(
                batch_x_missing, batch_x_mark, mask, domain_id=domain_id, mode=phase
            )

            # Calculate losses
            imputation_loss = self.imputation_criterion(
                outputs["imputed"], batch_x, mask
            )

            register_loss = outputs["register_loss"]

            domain_loss = self.domain_criterion(outputs["domain_pred"], domain_labels)

            # Combined loss
            if phase == "pretrain":
                # Focus on register learning
                total_batch_loss = (
                    imputation_loss
                    + self.args.register_weight * register_loss
                    + self.args.domain_weight * domain_loss
                )
            else:
                # Focus on imputation performance
                total_batch_loss = (
                    imputation_loss + 0.1 * register_loss + 0.1 * domain_loss
                )

            # Backward pass
            total_batch_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.max_grad_norm
            )

            self.optimizer.step()

            # Accumulate losses
            total_loss += total_batch_loss.item()
            total_imputation_loss += imputation_loss.item()
            total_register_loss += register_loss.item()
            total_domain_loss += domain_loss.item()

            batch_idx += 1

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch}, Batch {batch_idx}, "
                    f"Total Loss: {total_batch_loss.item():.6f}, "
                    f"Imputation: {imputation_loss.item():.6f}, "
                    f"Register: {register_loss.item():.6f}, "
                    f"Domain: {domain_loss.item():.6f}"
                )

        avg_losses = {
            "total": total_loss / batch_idx,
            "imputation": total_imputation_loss / batch_idx,
            "register": total_register_loss / batch_idx,
            "domain": total_domain_loss / batch_idx,
        }

        return avg_losses

    def validate(self, dataset_name):
        """Validate on a specific dataset"""
        self.model.eval()

        # Create validation dataset
        val_args = argparse.Namespace(**vars(self.args))
        dataset_config = DATASET_CONFIGS.get(dataset_name, {})
        for key, value in dataset_config.items():
            setattr(val_args, key, value)

        val_args.data = dataset_name
        val_args.data_path = f"{dataset_name}.csv"

        _, val_loader = data_provider(val_args, "val")
        domain_id = self.domain_handler.get_domain_id(dataset_name)

        total_loss = 0.0
        total_mae = 0.0
        total_missing_mae = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_data in val_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # Create missing values
                batch_x_missing, mask = self.create_missing_mask(
                    batch_x, missing_ratio=self.args.missing_ratio
                )
                mask = mask.to(self.device)

                # Forward pass
                outputs = self.model(
                    batch_x_missing,
                    batch_x_mark,
                    mask,
                    domain_id=domain_id,
                    mode="inference",
                )

                # Calculate metrics
                loss = self.imputation_criterion(outputs["imputed"], batch_x, mask)

                # MAE for missing values only
                missing_mask = (1 - mask).bool()
                if missing_mask.sum() > 0:
                    mae_missing = torch.abs(
                        outputs["imputed"][missing_mask] - batch_x[missing_mask]
                    ).mean()
                    total_missing_mae += mae_missing.item()

                # Overall MAE
                mae_all = torch.abs(outputs["imputed"] - batch_x).mean()

                total_loss += loss.item()
                total_mae += mae_all.item()
                num_batches += 1

        return {
            "loss": total_loss / num_batches,
            "mae": total_mae / num_batches,
            "missing_mae": total_missing_mae / num_batches
            if total_missing_mae > 0
            else 0,
        }

    def train(self):
        """Main training loop"""
        print("Starting Phase 2 training...")

        # Create multi-domain data loader
        train_loader, dataset_domain_map = self.get_multi_domain_data_loader("train")
        val_loader, _ = self.get_multi_domain_data_loader("val")

        best_val_loss = float("inf")

        # Phase 1: Pretrain register
        print("=" * 50)
        print("Phase 2.1: Register Pretraining")
        print("=" * 50)

        for epoch in range(self.args.pretrain_epochs):
            # Train
            train_losses = self.train_epoch(
                train_loader, dataset_domain_map, epoch, phase="pretrain"
            )

            # Validate on first dataset
            val_metrics = self.validate(self.args.datasets[0])

            print(f"Pretrain Epoch {epoch + 1}/{self.args.pretrain_epochs}:")
            print(
                f"  Train - Total: {train_losses['total']:.6f}, "
                f"Imputation: {train_losses['imputation']:.6f}, "
                f"Register: {train_losses['register']:.6f}, "
                f"Domain: {train_losses['domain']:.6f}"
            )
            print(
                f"  Val - Loss: {val_metrics['loss']:.6f}, "
                f"MAE: {val_metrics['mae']:.6f}, "
                f"Missing MAE: {val_metrics['missing_mae']:.6f}"
            )

            # Save checkpoint
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                self.save_checkpoint(epoch, "pretrain_best.pth")

        # Phase 2: Fine-tune
        print("=" * 50)
        print("Phase 2.2: Joint Fine-tuning")
        print("=" * 50)

        # Optionally unfreeze extractors
        if not self.args.keep_extractors_frozen:
            print("Unfreezing domain extractors...")
            for extractor in self.model.domain_extractors.values():
                for param in extractor.parameters():
                    param.requires_grad = True

            # Recreate optimizer with all parameters
            self.optimizer = self._select_optimizer()

        # Reset early stopping
        self.early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.finetune_epochs):
            # Train
            train_losses = self.train_epoch(
                train_loader, dataset_domain_map, epoch, phase="finetune"
            )

            # Validate on all datasets
            val_loss_total = 0
            for dataset_name in self.args.datasets:
                val_metrics = self.validate(dataset_name)
                val_loss_total += val_metrics["loss"]
                print(
                    f"  {dataset_name} - Loss: {val_metrics['loss']:.6f}, "
                    f"MAE: {val_metrics['mae']:.6f}, "
                    f"Missing MAE: {val_metrics['missing_mae']:.6f}"
                )

            avg_val_loss = val_loss_total / len(self.args.datasets)

            print(f"Finetune Epoch {epoch + 1}/{self.args.finetune_epochs}:")
            print(
                f"  Train - Total: {train_losses['total']:.6f}, "
                f"Imputation: {train_losses['imputation']:.6f}"
            )
            print(f"  Avg Val Loss: {avg_val_loss:.6f}")

            # Learning rate scheduling
            self.scheduler.step()

            # Early stopping
            self.early_stopping(avg_val_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_checkpoint(epoch, "finetune_best.pth")

        print("Phase 2 training completed!")
        return best_val_loss

    def save_checkpoint(self, epoch, filename):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "args": self.args,
        }

        os.makedirs(self.args.checkpoints, exist_ok=True)
        torch.save(checkpoint, os.path.join(self.args.checkpoints, filename))
        print(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint_path = os.path.join(self.args.checkpoints, filename)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Checkpoint loaded: {filename}")
        return checkpoint["epoch"]


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Register-Enhanced FEDformer Training"
    )

    # Basic settings
    parser.add_argument("--is_training", type=int, default=1, help="status")
    parser.add_argument(
        "--model_id", type=str, default="phase2_register", help="model id"
    )
    parser.add_argument(
        "--model", type=str, default="FEDformerRegister", help="model name"
    )

    # Data settings
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["ETTh1", "ETTh2", "Traffic", "Weather"],
        help="datasets to train on",
    )
    parser.add_argument(
        "--root_path", type=str, default="./data/", help="root path of data file"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )
    parser.add_argument(
        "--phase1_checkpoint_path",
        type=str,
        default="D:/FEDformer/checkpoints/",
        help="path to Phase 1 checkpoints",
    )

    # Phase 1 compatibility
    parser.add_argument("--task_name", type=str, default="imputation", help="task name")
    parser.add_argument("--features", type=str, default="M", help="M: multivariate")
    parser.add_argument("--target", type=str, default="OT", help="target feature")
    parser.add_argument(
        "--freq", type=str, default="h", help="freq for time features encoding"
    )
    parser.add_argument(
        "--embed", type=str, default="timeF", help="time features encoding"
    )

    # Model parameters
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument(
        "--pred_len", type=int, default=0, help="prediction sequence length"
    )
    parser.add_argument(
        "--individual", action="store_true", default=False, help="individual"
    )
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument(
        "--moving_avg", type=int, default=25, help="window size of moving average"
    )
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument("--dropout", type=float, default=0.05, help="dropout")
    parser.add_argument(
        "--embed_type",
        type=int,
        default=0,
        help="0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding",
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="whether to output attention in encoder",
    )

    # FEDformer specific
    parser.add_argument(
        "--version",
        type=str,
        default="Fourier",
        help="for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]",
    )
    parser.add_argument(
        "--mode_select",
        type=str,
        default="random",
        help="for FEDformer, there are two mode_select to choose, options: [random, low]",
    )
    parser.add_argument(
        "--modes", type=int, default=64, help="modes to be selected random 64"
    )
    parser.add_argument("--L", type=int, default=3, help="ignore level")
    parser.add_argument("--base", type=str, default="legendre", help="mwt base")
    parser.add_argument(
        "--cross_activation", type=str, default="tanh", help="mwt cross activation"
    )

    # Register-specific parameters
    parser.add_argument(
        "--register_size", type=int, default=128, help="size of TS-Register"
    )
    parser.add_argument(
        "--num_register_tokens", type=int, default=3, help="number of register tokens"
    )
    parser.add_argument(
        "--register_weight", type=float, default=1.0, help="weight for register loss"
    )
    parser.add_argument(
        "--domain_weight",
        type=float,
        default=0.5,
        help="weight for domain classification loss",
    )

    # Training parameters
    parser.add_argument(
        "--num_workers", type=int, default=4, help="data loader num workers"
    )
    parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
    parser.add_argument(
        "--pretrain_epochs", type=int, default=5, help="register pretrain epochs"
    )
    parser.add_argument(
        "--finetune_epochs", type=int, default=15, help="joint finetune epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="early stopping patience"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="mse", help="loss function")
    parser.add_argument(
        "--lradj", type=str, default="type1", help="adjust learning rate"
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )
    parser.add_argument(
        "--inverse", action="store_true", help="inverse output data", default=False
    )
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0,1,2,3", help="device ids of multile gpus"
    )

    # Imputation specific
    parser.add_argument(
        "--missing_ratio", type=float, default=0.2, help="missing ratio"
    )
    parser.add_argument(
        "--missing_weight",
        type=float,
        default=2.0,
        help="weight for missing values in loss",
    )

    # Phase 2 specific training options
    parser.add_argument(
        "--freeze_extractors",
        action="store_true",
        default=True,
        help="freeze Phase 1 extractors during register pretraining",
    )
    parser.add_argument(
        "--keep_extractors_frozen",
        action="store_true",
        default=False,
        help="keep extractors frozen during fine-tuning",
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="gradient clipping norm"
    )
    parser.add_argument(
        "--lr_step_size", type=int, default=10, help="step size for lr scheduler"
    )
    parser.add_argument(
        "--lr_gamma", type=float, default=0.5, help="gamma for lr scheduler"
    )

    # Resume training
    parser.add_argument(
        "--resume_checkpoint", type=str, default=None, help="checkpoint to resume from"
    )

    args = parser.parse_args()

    print("Args in experiment:")
    print(args)

    # Set device
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # Create trainer and start training
    trainer = Phase2Trainer(args)

    if args.resume_checkpoint:
        trainer.load_checkpoint(args.resume_checkpoint)

    best_val_loss = trainer.train()

    print(f"Training completed! Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
