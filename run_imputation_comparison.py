#!/usr/bin/env python3
"""
Clean Register Ablation Study for Multi-Domain Time Series Imputation
Fixed version without duplicate functions and import errors
"""

import argparse
import copy
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime
import torch.nn as nn
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Clean imports - only what we need
from Register_Analysis import analyze_register_system
from configs.imputation_config import ImputationConfig, LightImputationConfig
from imputation_trainer import (
    ImputationDataset,
    ROSEStyleTrainer,
    create_domain_datasets,
    train_approach_2,  # This calls train_approach_no_register
)
from multi_domain_fedformer import (
    MultiDomainFEDformerWithoutRegister,
    MultiDomainFEDformerWithRegister,
)
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_approach_no_register(configs, train_datasets=None, test_datasets=None):
    """
    Modified to accept custom datasets for data scarcity experiments
    """
    print("\n" + "=" * 80)
    print("TRAINING NO-REGISTER VERSION")
    print("=" * 80)

    # Use provided datasets or create new ones
    if train_datasets is None or test_datasets is None:
        print("\nCreating domain-separate datasets...")
        dataset_type = getattr(configs, "dataset_type", "original")
        train_datasets, test_datasets = create_domain_datasets(
            configs, dataset_type=dataset_type
        )
    else:
        print("\nUsing provided datasets...")

    print(f"Datasets created for {len(train_datasets)} domains:")
    for domain_name, dataset in train_datasets.items():
        sample = dataset[0] if hasattr(dataset, "__getitem__") else dataset.dataset[0]
        print(f"  {domain_name:12s}: {len(dataset)} samples")

    # Rest of your existing training code...
    model = MultiDomainFEDformerWithoutRegister(configs)
    trainer = NoRegisterTrainer(model, configs)

    # Training phases
    if configs.two_phase_training:
        pretrain_epochs = configs.pretrain_epochs
        finetune_epochs = configs.finetune_epochs
    else:
        pretrain_epochs = 0
        finetune_epochs = configs.train_epochs

    # Training tracking
    best_loss = float("inf")
    train_losses = []

    # Phase 1: Frequency Pre-training
    if pretrain_epochs > 0:
        print(f"\nPHASE 1: FREQUENCY PRE-TRAINING ({pretrain_epochs} epochs)")
        for epoch in range(pretrain_epochs):
            pretrain_loss, _ = trainer.train_epoch_frequency_pretrain(train_datasets)
            train_losses.append(("pretrain", pretrain_loss, 0, 0))

            if pretrain_loss < best_loss and pretrain_loss > 0:
                best_loss = pretrain_loss
                print(f"  Epoch {epoch + 1}: Loss = {pretrain_loss:.6f}")

    # Phase 2: Imputation Fine-tuning
    if finetune_epochs > 0:
        print(f"\nPHASE 2: IMPUTATION FINE-TUNING ({finetune_epochs} epochs)")
        for epoch in range(finetune_epochs):
            finetune_loss, _ = trainer.train_epoch_imputation_finetune(train_datasets)
            train_losses.append(("finetune", finetune_loss, 0, 0))

            if finetune_loss < best_loss and finetune_loss > 0:
                best_loss = finetune_loss
                print(f"  Epoch {epoch + 1}: Loss = {finetune_loss:.6f}")

    # Evaluation
    test_results = trainer.evaluate_domain_separate(test_datasets)

    return model, train_losses, test_results


class NoRegisterTrainer:
    """
    Trainer for no-register model (identical to ROSEStyleTrainer except register handling)
    """

    def __init__(self, model, configs):
        self.model = model.to(device)
        self.configs = configs
        self.criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()

        # Single optimizer (no register optimizer needed)
        self.main_optimizer = torch.optim.Adam(
            model.parameters(),
            lr=configs.learning_rate,
        )

        # Loss weights (same as register version)
        self.reconstruction_weight = getattr(configs, "reconstruction_weight", 1.0)
        self.register_weight = 0.0  # Always 0 for no-register version
        self.imputation_weight = getattr(configs, "imputation_weight", 1.0)

    def train_epoch_frequency_pretrain(self, train_datasets):
        """Frequency pre-training without register (identical structure to register version)"""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {
            "reconstruction_loss": [],
            "register_loss": [],  # Will always be 0
            "total_loss": [],
        }

        print("  Domain-wise pre-training (NO REGISTER):")

        for domain_name, domain_dataset in train_datasets.items():
            try:
                domain_imputation = ImputationDataset(
                    domain_dataset,
                    missing_rate=self.configs.missing_rate,
                    missing_pattern=self.configs.missing_pattern,
                )

                domain_loader = DataLoader(
                    domain_imputation,
                    batch_size=self.configs.batch_size,
                    shuffle=True,
                    num_workers=self.configs.num_workers,
                    drop_last=True,
                )
            except Exception as e:
                print(f"    Error creating dataset for {domain_name}: {e}")
                continue

            domain_losses = []
            domain_recon_losses = []
            processed_batches = 0

            for batch_idx, batch_data in enumerate(domain_loader):
                try:
                    x_missing, x_mark, mask, x_target = batch_data

                    x_missing = x_missing.to(device)
                    x_mark = x_mark.to(device)
                    mask = mask.to(device)
                    x_target = x_target.to(device)

                    if torch.isnan(x_missing).all() or mask.sum() == 0:
                        continue

                    self.main_optimizer.zero_grad()

                    # Forward pass (no register loss returned)
                    _, _, register_loss, reconstruction_loss = self.model(
                        x_missing,
                        x_mark,
                        mask,
                        domain_name,
                        training_phase="frequency_pretrain",
                    )

                    if torch.isnan(reconstruction_loss):
                        continue

                    # Only reconstruction loss (no register component)
                    total_loss = self.reconstruction_weight * reconstruction_loss

                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        continue

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.main_optimizer.step()

                    domain_losses.append(total_loss.item())
                    domain_recon_losses.append(reconstruction_loss.item())
                    processed_batches += 1

                except Exception as e:
                    print(f"    Error in {domain_name} batch {batch_idx}: {e}")
                    continue

                if batch_idx >= getattr(self.configs, "max_batches_per_domain", 1000):
                    break

            if domain_losses:
                avg_domain_loss = sum(domain_losses) / len(domain_losses)
                avg_recon_loss = sum(domain_recon_losses) / len(domain_recon_losses)

                epoch_losses.append(avg_domain_loss)
                epoch_metrics["reconstruction_loss"].append(avg_recon_loss)
                epoch_metrics["register_loss"].append(0.0)  # Always 0
                epoch_metrics["total_loss"].append(avg_domain_loss)

                print(
                    f"    {domain_name:12s}: Loss={avg_domain_loss:.6f}, Batches={processed_batches}"
                )
            else:
                print(f"    {domain_name:12s}: No valid batches processed")

        if epoch_losses:
            return sum(epoch_losses) / len(epoch_losses), epoch_metrics
        else:
            return 0.0, epoch_metrics

    def train_epoch_imputation_finetune(self, train_datasets):
        """Imputation fine-tuning without register (identical structure to register version)"""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {
            "imputation_loss": [],
            "register_loss": [],  # Will always be 0
            "mae_loss": [],
            "total_loss": [],
        }

        print("  Domain-wise fine-tuning (NO REGISTER):")

        for domain_name, domain_dataset in train_datasets.items():
            try:
                domain_imputation = ImputationDataset(
                    domain_dataset,
                    missing_rate=self.configs.missing_rate,
                    missing_pattern=self.configs.missing_pattern,
                )

                domain_loader = DataLoader(
                    domain_imputation,
                    batch_size=self.configs.batch_size,
                    shuffle=True,
                    num_workers=self.configs.num_workers,
                    drop_last=True,
                )
            except Exception as e:
                print(f"    Error creating dataset for {domain_name}: {e}")
                continue

            domain_losses = []
            domain_imputation_losses = []
            domain_mae_losses = []
            processed_batches = 0

            for batch_idx, batch_data in enumerate(domain_loader):
                try:
                    x_missing, x_mark, mask, x_target = batch_data

                    x_missing = x_missing.to(device)
                    x_mark = x_mark.to(device)
                    mask = mask.to(device)
                    x_target = x_target.to(device)

                    if torch.isnan(x_missing).all() or mask.sum() == 0:
                        continue

                    self.main_optimizer.zero_grad()

                    # Forward pass (no register loss)
                    dec_out, reconstructed, register_loss = self.model(
                        x_missing,
                        x_mark,
                        mask,
                        domain_name,
                        training_phase="imputation_finetune",
                    )

                    # Imputation loss only on missing values
                    missing_mask = (~mask).bool()
                    if missing_mask.any():
                        imputation_loss = self.criterion(
                            dec_out[missing_mask], x_target[missing_mask]
                        )
                        mae_loss = self.mae_criterion(
                            dec_out[missing_mask], x_target[missing_mask]
                        )
                    else:
                        imputation_loss = self.criterion(dec_out, x_target) * 0.1
                        mae_loss = self.mae_criterion(dec_out, x_target) * 0.1

                    if torch.isnan(imputation_loss) or torch.isnan(mae_loss):
                        continue

                    # Only imputation loss (no register component)
                    total_loss = self.imputation_weight * imputation_loss

                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        continue

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.main_optimizer.step()

                    domain_losses.append(total_loss.item())
                    domain_imputation_losses.append(imputation_loss.item())
                    domain_mae_losses.append(mae_loss.item())
                    processed_batches += 1

                except Exception as e:
                    print(f"    Error in {domain_name} batch {batch_idx}: {e}")
                    continue

                if batch_idx >= getattr(self.configs, "max_batches_per_domain", 1000):
                    break

            if domain_losses:
                avg_domain_loss = sum(domain_losses) / len(domain_losses)
                avg_imputation_loss = sum(domain_imputation_losses) / len(
                    domain_imputation_losses
                )
                avg_mae_loss = sum(domain_mae_losses) / len(domain_mae_losses)

                epoch_losses.append(avg_domain_loss)
                epoch_metrics["imputation_loss"].append(avg_imputation_loss)
                epoch_metrics["mae_loss"].append(avg_mae_loss)
                epoch_metrics["register_loss"].append(0.0)  # Always 0
                epoch_metrics["total_loss"].append(avg_domain_loss)

                print(
                    f"    {domain_name:12s}: Loss={avg_domain_loss:.6f}, Batches={processed_batches}"
                )
            else:
                print(f"    {domain_name:12s}: No valid batches processed")

        if epoch_losses:
            return sum(epoch_losses) / len(epoch_losses), epoch_metrics
        else:
            return 0.0, epoch_metrics

    def evaluate_domain_separate(self, test_datasets):
        """Evaluate without register (identical to register version)"""
        self.model.eval()
        results = {}

        print("  Domain-wise evaluation (NO REGISTER):")

        with torch.no_grad():
            for domain_name, test_dataset in test_datasets.items():
                try:
                    test_imputation = ImputationDataset(
                        test_dataset,
                        missing_rate=self.configs.test_missing_rate,
                        missing_pattern=self.configs.test_missing_pattern,
                    )

                    test_loader = DataLoader(
                        test_imputation,
                        batch_size=self.configs.batch_size,
                        shuffle=False,
                        num_workers=self.configs.num_workers,
                    )
                except Exception as e:
                    print(f"    Error creating test dataset for {domain_name}: {e}")
                    results[domain_name] = (float("inf"), float("inf"))
                    continue

                total_mse = 0
                total_mae = 0
                processed_batches = 0

                for batch_idx, batch_data in enumerate(test_loader):
                    try:
                        x_missing, x_mark, mask, x_target = batch_data

                        x_missing = x_missing.to(device)
                        x_mark = x_mark.to(device)
                        mask = mask.to(device)
                        x_target = x_target.to(device)

                        if torch.isnan(x_missing).all():
                            continue

                        dec_out, reconstructed, _ = self.model(
                            x_missing,
                            x_mark,
                            mask,
                            domain_name,
                            training_phase="imputation_finetune",
                        )

                        missing_mask = (~mask).bool()
                        if missing_mask.any():
                            mse = F.mse_loss(
                                dec_out[missing_mask], x_target[missing_mask]
                            )
                            mae = F.l1_loss(
                                dec_out[missing_mask], x_target[missing_mask]
                            )
                        else:
                            mse = F.mse_loss(dec_out, x_target)
                            mae = F.l1_loss(dec_out, x_target)

                        if not (
                            torch.isnan(mse)
                            or torch.isnan(mae)
                            or torch.isinf(mse)
                            or torch.isinf(mae)
                        ):
                            total_mse += mse.item()
                            total_mae += mae.item()
                            processed_batches += 1

                    except Exception as e:
                        print(
                            f"    Error evaluating {domain_name} batch {batch_idx}: {e}"
                        )
                        continue

                    if batch_idx >= getattr(
                        self.configs, "max_eval_batches_per_domain", 1000
                    ):
                        break

                if processed_batches > 0:
                    avg_mse = total_mse / processed_batches
                    avg_mae = total_mae / processed_batches
                    results[domain_name] = (avg_mse, avg_mae)
                    print(
                        f"    {domain_name:12s}: MSE={avg_mse:.6f}, MAE={avg_mae:.6f}"
                    )
                else:
                    results[domain_name] = (float("inf"), float("inf"))
                    print(f"    {domain_name:12s}: No valid batches")

        return results

    def save_checkpoint(self, epoch, loss, filepath, metrics=None):
        """Save checkpoint for no-register model"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "main_optimizer_state_dict": self.main_optimizer.state_dict(),
                "loss": loss,
                "metrics": metrics,
                "config": self.configs.__dict__
                if hasattr(self.configs, "__dict__")
                else {},
                "ablation_note": "No register system",
            }

            torch.save(checkpoint, filepath)
            return True
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            return False


def train_approach_3_complete_rose(configs):
    """
    Complete ROSE-style multi-domain training implementation:
    - Decomposed frequency learning pre-training
    - Domain-separate processing (no padding)
    - Two-phase training strategy
    - Proper loss function handling
    """
    print("\n" + "=" * 80)
    print("TRAINING APPROACH 3: COMPLETE ROSE-STYLE MULTI-DOMAIN IMPUTATION")
    print("=" * 80)

    # Print configuration
    if hasattr(configs, "print_config"):
        configs.print_config()

    # Set random seed for reproducibility
    set_seed(configs.seed if hasattr(configs, "seed") else 42)

    # Create datasets - now domain-separate (no padding needed!)
    print("\nCreating domain-separate datasets...")
    # Use dataset_type from configs if set, otherwise default to "original"
    dataset_type = getattr(configs, "dataset_type", "original")
    train_datasets, test_datasets = create_domain_datasets(
        configs, dataset_type=dataset_type
    )

    print(f"Datasets created for {len(train_datasets)} domains:")
    for domain_name, dataset in train_datasets.items():
        sample = dataset[0]
        print(f"  {domain_name:12s}: {sample[0].shape} (native dimensions)")

    # Create model
    print(f"\nInitializing ROSE-style model...")
    model = MultiDomainFEDformerWithRegister(configs)

    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Register system diagnostics
    print(f"  Register system: {model.register_system.register.shape}")
    print(f"  Domain embeddings: {len(model.domain_embeddings)} domains")

    # Create trainer
    print(f"\nInitializing trainer...")
    trainer = ROSEStyleTrainer(model, configs)

    # Setup training phases
    if configs.two_phase_training:
        pretrain_epochs = configs.pretrain_epochs
        finetune_epochs = configs.finetune_epochs
    else:
        pretrain_epochs = 0
        finetune_epochs = configs.train_epochs

    print(f"\nTraining phases:")
    print(f"  Phase 1 (Frequency Pre-training): {pretrain_epochs} epochs")
    print(f"  Phase 2 (Imputation Fine-tuning): {finetune_epochs} epochs")
    print(f"  Total training epochs: {configs.train_epochs}")

    # Training tracking
    best_loss = float("inf")
    train_losses = []
    training_history = {
        "pretrain_losses": [],
        "finetune_losses": [],
        "pretrain_metrics": [],
        "finetune_metrics": [],
    }

    # =================================================================
    # PHASE 1: FREQUENCY RECONSTRUCTION PRE-TRAINING
    # =================================================================

    if pretrain_epochs > 0:
        print(f"\n{'=' * 60}")
        print(f"PHASE 1: FREQUENCY RECONSTRUCTION PRE-TRAINING")
        print(f"{'=' * 60}")
        print("Following ROSE Section 3.2 & 3.4")

        model.register_system.set_fine_tuning_mode(False)  # Pre-training mode

        for epoch in range(pretrain_epochs):
            epoch_start_time = time.time()

            print(f"\nPre-training Epoch {epoch + 1}/{pretrain_epochs}")
            print("-" * 50)

            # Train epoch with frequency reconstruction
            pretrain_loss, pretrain_metrics = trainer.train_epoch_frequency_pretrain(
                train_datasets
            )

            # Track training
            train_losses.append(("pretrain", pretrain_loss, 0, 0))
            training_history["pretrain_losses"].append(pretrain_loss)
            training_history["pretrain_metrics"].append(pretrain_metrics)

            epoch_time = time.time() - epoch_start_time

            print(f"Pre-training Results:")
            print(f"  Average Loss: {pretrain_loss:.6f}")
            print(f"  Epoch Time: {epoch_time:.1f}s")

            if pretrain_metrics["reconstruction_loss"]:
                avg_recon = sum(pretrain_metrics["reconstruction_loss"]) / len(
                    pretrain_metrics["reconstruction_loss"]
                )
                avg_register = sum(pretrain_metrics["register_loss"]) / len(
                    pretrain_metrics["register_loss"]
                )
                print(f"  Avg Reconstruction Loss: {avg_recon:.6f}")
                print(f"  Avg Register Loss: {avg_register:.6f}")

            # Save best pre-training model
            if pretrain_loss < best_loss and pretrain_loss > 0:
                best_loss = pretrain_loss
                improvement = True

                # Save checkpoint
                checkpoint_path = os.path.join(
                    configs.save_path, "approach3_rose_pretrain_best.pth"
                )
                if trainer.save_checkpoint(
                    epoch, pretrain_loss, checkpoint_path, pretrain_metrics
                ):
                    print(f"  → New best pre-training model saved!")

            # Progress indicator
            progress = (epoch + 1) / pretrain_epochs * 100
            print(f"  Pre-training Progress: {progress:.1f}%")

        print(f"\nPre-training Phase Completed!")
        print(f"Best pre-training loss: {best_loss:.6f}")

    # =================================================================
    # PHASE 2: IMPUTATION FINE-TUNING
    # =================================================================

    if finetune_epochs > 0:
        print(f"\n{'=' * 60}")
        print(f"PHASE 2: IMPUTATION FINE-TUNING")
        print(f"{'=' * 60}")
        print("Focus on missing value prediction performance")

        # Switch to fine-tuning mode
        model.register_system.set_fine_tuning_mode(True)

        # Fine-tuning tracking
        finetune_best_loss = best_loss
        patience_counter = 0

        for epoch in range(finetune_epochs):
            epoch_start_time = time.time()
            global_epoch = pretrain_epochs + epoch + 1

            print(
                f"\nFine-tuning Epoch {epoch + 1}/{finetune_epochs} (Global: {global_epoch}/{configs.train_epochs})"
            )
            print("-" * 50)

            # Train epoch for imputation
            finetune_loss, finetune_metrics = trainer.train_epoch_imputation_finetune(
                train_datasets
            )

            # Track training
            train_losses.append(("finetune", finetune_loss, 0, 0))
            training_history["finetune_losses"].append(finetune_loss)
            training_history["finetune_metrics"].append(finetune_metrics)

            epoch_time = time.time() - epoch_start_time

            print(f"Fine-tuning Results:")
            print(f"  Average Loss: {finetune_loss:.6f}")
            print(f"  Epoch Time: {epoch_time:.1f}s")

            if finetune_metrics["imputation_loss"]:
                avg_imp = sum(finetune_metrics["imputation_loss"]) / len(
                    finetune_metrics["imputation_loss"]
                )
                avg_mae = sum(finetune_metrics["mae_loss"]) / len(
                    finetune_metrics["mae_loss"]
                )
                print(f"  Avg Imputation Loss: {avg_imp:.6f}")
                print(f"  Avg MAE: {avg_mae:.6f}")

            # Model improvement tracking
            if finetune_loss < finetune_best_loss and finetune_loss > 0:
                improvement_pct = (
                    (finetune_best_loss - finetune_loss) / finetune_best_loss
                ) * 100
                finetune_best_loss = finetune_loss
                patience_counter = 0

                # Save best fine-tuned model
                checkpoint_path = os.path.join(
                    configs.save_path, "approach3_rose_best.pth"
                )
                if trainer.save_checkpoint(
                    global_epoch, finetune_loss, checkpoint_path, finetune_metrics
                ):
                    print(f"  → New best! Improved by {improvement_pct:.2f}%")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{configs.patience}")

                # Early stopping
                if patience_counter >= configs.patience:
                    print(
                        f"\nEarly stopping triggered after {epoch + 1} fine-tuning epochs"
                    )
                    print(f"No improvement for {patience_counter} consecutive epochs")
                    break

            # Progress indicator
            progress = (epoch + 1) / finetune_epochs * 100
            print(f"  Fine-tuning Progress: {progress:.1f}%")

        print(f"\nFine-tuning Phase Completed!")
        print(f"Best fine-tuning loss: {finetune_best_loss:.6f}")

    # =================================================================
    # EVALUATION PHASE
    # =================================================================

    print(f"\n{'=' * 60}")
    print("EVALUATION PHASE")
    print(f"{'=' * 60}")

    # Evaluate on test sets (domain-separate)
    test_results = trainer.evaluate_domain_separate(test_datasets)

    # Print detailed results
    print(f"\nTest Results (Domain-Separate Evaluation):")
    print(f"{'Domain':<12} {'MSE':<12} {'MAE':<12}")
    print("-" * 40)

    total_mse, total_mae = 0, 0
    valid_domains = 0

    for domain_name, (mse, mae) in test_results.items():
        if mse != float("inf"):
            print(f"{domain_name:<12} {mse:<12.6f} {mae:<12.6f}")
            total_mse += mse
            total_mae += mae
            valid_domains += 1
        else:
            print(f"{domain_name:<12} {'Failed':<12} {'Failed':<12}")

    if valid_domains > 0:
        avg_mse = total_mse / valid_domains
        avg_mae = total_mae / valid_domains
        print("-" * 40)
        print(f"{'AVERAGE':<12} {avg_mse:<12.6f} {avg_mae:<12.6f}")

    # Save training history
    history_path = os.path.join(
        configs.save_path, "approach3_rose_training_history.json"
    )
    try:
        with open(history_path, "w") as f:
            # Convert any tensors to floats for JSON serialization
            serializable_history = {}
            for key, value in training_history.items():
                if isinstance(value, list):
                    serializable_history[key] = [
                        float(v) if isinstance(v, (int, float)) else v for v in value
                    ]
                else:
                    serializable_history[key] = value

            json.dump(
                {
                    "config": configs.__dict__ if hasattr(configs, "__dict__") else {},
                    "training_history": serializable_history,
                    "test_results": test_results,
                    "final_metrics": {
                        "best_loss": float(
                            min(
                                best_loss,
                                finetune_best_loss
                                if finetune_epochs > 0
                                else best_loss,
                            )
                        ),
                        "total_epochs": len(train_losses),
                        "pretrain_epochs": pretrain_epochs,
                        "finetune_epochs": finetune_epochs,
                    },
                },
                f,
                indent=2,
            )
        print(f"\nTraining history saved to: {history_path}")
    except Exception as e:
        print(f"Could not save training history: {e}")

    print(f"\nAll results saved to: {configs.save_path}")
    print("=" * 60)

    return model, train_losses, test_results


def run_complete_ablation_study():
    """Run complete register ablation study"""

    print("=" * 100)
    print("REGISTER ABLATION STUDY FOR MULTI-DOMAIN TIME SERIES IMPUTATION")
    print("=" * 100)
    print("Comparing IDENTICAL models with and without register system")
    print()

    # Set seed for reproducibility
    set_seed(42)

    # Create base configuration
    base_config = LightImputationConfig()

    # Create identical configurations for both approaches
    config_no_register = copy.deepcopy(base_config)
    config_with_register = copy.deepcopy(base_config)

    # Create separate save paths with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_path = base_config.save_path

    config_no_register.save_path = os.path.join(
        base_save_path, f"no_register_{timestamp}"
    )
    config_with_register.save_path = os.path.join(
        base_save_path, f"with_register_{timestamp}"
    )

    print(f"Configuration:")
    print(f"  Base config: LightImputationConfig")
    print(f"  Total epochs: {base_config.train_epochs}")
    print(f"  Pretrain epochs: {base_config.pretrain_epochs}")
    print(f"  Finetune epochs: {base_config.finetune_epochs}")
    print(f"  Batch size: {base_config.batch_size}")
    print(f"  Learning rate: {base_config.learning_rate}")
    print()
    print(f"Results will be saved to:")
    print(f"  No Register: {config_no_register.save_path}")
    print(f"  With Register: {config_with_register.save_path}")
    print()

    # ============================================================================
    # PHASE 1: TRAIN WITHOUT REGISTER (Approach 2)
    # ============================================================================

    print("PHASE 1: TRAINING WITHOUT REGISTER")
    print("=" * 60)

    try:
        model_no_register, losses_no_register, results_no_register = train_approach_2(
            config_no_register
        )
        print(f"✓ Approach 2 (No Register) completed successfully")
        print(f"  Final results: {len(results_no_register)} domains")
    except Exception as e:
        print(f"✗ Approach 2 (No Register) failed: {e}")
        return None

    # ============================================================================
    # PHASE 2: TRAIN WITH REGISTER (Approach 3)
    # ============================================================================

    print("\nPHASE 2: TRAINING WITH REGISTER")
    print("=" * 60)

    try:
        model_with_register, losses_with_register, results_with_register = (
            train_approach_3_complete_rose(config_with_register)
        )
        print(f"✓ Approach 3 (With Register) completed successfully")
        print(f"  Final results: {len(results_with_register)} domains")
    except Exception as e:
        print(f"✗ Approach 3 (With Register) failed: {e}")
        return None

    # ============================================================================
    # PHASE 3: COMPARISON AND ANALYSIS
    # ============================================================================

    print("\n" + "=" * 100)
    print("REGISTER ABLATION RESULTS COMPARISON")
    print("=" * 100)

    # Domain-wise comparison
    print(
        f"\n{'Domain':<12} {'No Register':<20} {'With Register':<20} {'Register Impact':<15}"
    )
    print(
        f"{'':12} {'MSE':<9} {'MAE':<9} {'MSE':<9} {'MAE':<9} {'MSE %':<7} {'MAE %':<7}"
    )
    print("-" * 85)

    total_mse_no, total_mae_no = 0, 0
    total_mse_with, total_mae_with = 0, 0
    num_domains = 0
    improvements = []

    for domain in sorted(results_no_register.keys()):
        if domain in results_with_register:
            mse_no, mae_no = results_no_register[domain]
            mse_with, mae_with = results_with_register[domain]

            # Skip failed domains
            if mse_no == float("inf") or mse_with == float("inf"):
                print(f"{domain:<12} {'Failed':<20} {'Failed':<20} {'N/A':<15}")
                continue

            # Calculate improvements (positive = register helps)
            mse_improvement = ((mse_no - mse_with) / mse_no * 100) if mse_no > 0 else 0
            mae_improvement = ((mae_no - mae_with) / mae_no * 100) if mae_no > 0 else 0

            improvements.append(mse_improvement)

            print(
                f"{domain:<12} {mse_no:<9.6f} {mae_no:<9.6f} {mse_with:<9.6f} {mae_with:<9.6f} {mse_improvement:<7.2f} {mae_improvement:<7.2f}"
            )

            total_mse_no += mse_no
            total_mae_no += mae_no
            total_mse_with += mse_with
            total_mae_with += mae_with
            num_domains += 1

    # Calculate averages and summary
    if num_domains > 0:
        avg_mse_no = total_mse_no / num_domains
        avg_mae_no = total_mae_no / num_domains
        avg_mse_with = total_mse_with / num_domains
        avg_mae_with = total_mae_with / num_domains

        avg_mse_improvement = (
            ((avg_mse_no - avg_mse_with) / avg_mse_no * 100) if avg_mse_no > 0 else 0
        )
        avg_mae_improvement = (
            ((avg_mae_no - avg_mae_with) / avg_mae_no * 100) if avg_mae_no > 0 else 0
        )

        print("-" * 85)
        print(
            f"{'AVERAGE':<12} {avg_mse_no:<9.6f} {avg_mae_no:<9.6f} {avg_mse_with:<9.6f} {avg_mae_with:<9.6f} {avg_mse_improvement:<7.2f} {avg_mae_improvement:<7.2f}"
        )

        # Summary statistics
        print(f"\n{'=' * 60}")
        print("SUMMARY STATISTICS")
        print(f"{'=' * 60}")

        positive_improvements = [x for x in improvements if x > 0]
        negative_improvements = [x for x in improvements if x < 0]

        print(
            f"Domains where register helps: {len(positive_improvements)}/{num_domains}"
        )
        print(
            f"Domains where register hurts: {len(negative_improvements)}/{num_domains}"
        )
        print(f"Average MSE improvement: {avg_mse_improvement:.2f}%")
        print(f"Average MAE improvement: {avg_mae_improvement:.2f}%")

        if improvements:
            print(f"Best improvement: {max(improvements):.2f}%")
            print(f"Worst degradation: {min(improvements):.2f}%")
            print(f"Standard deviation: {np.std(improvements):.2f}%")

        # Conclusion
        print(f"\n{'=' * 60}")
        print("CONCLUSION")
        print(f"{'=' * 60}")

        if avg_mse_improvement > 5:
            conclusion = "REGISTER PROVIDES SIGNIFICANT BENEFIT"
            recommendation = "Use register system for production"
        elif avg_mse_improvement > 1:
            conclusion = "REGISTER PROVIDES MODEST BENEFIT"
            recommendation = "Consider register system if computational cost acceptable"
        elif avg_mse_improvement > -1:
            conclusion = "REGISTER IMPACT IS NEGLIGIBLE"
            recommendation = "Use simpler no-register version for efficiency"
        else:
            conclusion = "REGISTER MAY BE HARMFUL"
            recommendation = "Avoid register system or investigate implementation"

        print(f"Overall conclusion: {conclusion}")
        print(f"Recommendation: {recommendation}")

        # Save comparison results
        comparison_results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "epochs": base_config.train_epochs,
                "pretrain_epochs": base_config.pretrain_epochs,
                "finetune_epochs": base_config.finetune_epochs,
                "batch_size": base_config.batch_size,
                "learning_rate": base_config.learning_rate,
            },
            "results_no_register": results_no_register,
            "results_with_register": results_with_register,
            "summary": {
                "num_domains": num_domains,
                "avg_mse_improvement": avg_mse_improvement,
                "avg_mae_improvement": avg_mae_improvement,
                "domains_improved": len(positive_improvements),
                "domains_degraded": len(negative_improvements),
            },
            "conclusion": conclusion,
            "recommendation": recommendation,
        }

        # Save to both directories
        for save_path in [config_no_register.save_path, config_with_register.save_path]:
            comparison_file = os.path.join(save_path, "ablation_comparison.json")
            try:
                os.makedirs(save_path, exist_ok=True)
                with open(comparison_file, "w") as f:
                    json.dump(comparison_results, f, indent=2)
                print(f"Comparison saved to: {comparison_file}")
            except Exception as e:
                print(f"Error saving to {comparison_file}: {e}")

        print(f"\n{'=' * 100}")
        print("REGISTER ABLATION STUDY COMPLETED SUCCESSFULLY")
        print(f"{'=' * 100}")

        return comparison_results

    else:
        print("ERROR: No valid domains found for comparison")
        return None


# Add this to your run_imputation_comparison.py


def run_enhanced_ablation_study():
    """Run ablation study with register domain exploration"""

    print("=" * 100)
    print("ENHANCED REGISTER ABLATION STUDY WITH DOMAIN EXPLORATION")
    print("=" * 100)

    # Set seed for reproducibility
    set_seed(42)

    # Create configurations
    base_config = LightImputationConfig()
    config_no_register = copy.deepcopy(base_config)
    config_with_register = copy.deepcopy(base_config)

    # Create separate save paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_path = base_config.save_path

    config_no_register.save_path = os.path.join(
        base_save_path, f"no_register_{timestamp}"
    )
    config_with_register.save_path = os.path.join(
        base_save_path, f"with_register_{timestamp}"
    )

    # Phase 1: Train without register
    print("PHASE 1: TRAINING WITHOUT REGISTER")
    print("=" * 60)

    model_no_register, losses_no_register, results_no_register = train_approach_2(
        config_no_register
    )

    # Phase 2: Train with register
    print("\nPHASE 2: TRAINING WITH REGISTER")
    print("=" * 60)

    model_with_register, losses_with_register, results_with_register = (
        train_approach_3_complete_rose(config_with_register)
    )

    # Phase 3: Register Domain Exploration
    print("\nPHASE 3: REGISTER DOMAIN EXPLORATION")
    print("=" * 60)

    # Create test datasets for analysis
    _, test_datasets = create_domain_datasets(config_with_register)

    # Analyze register system behavior
    analysis_path = os.path.join(config_with_register.save_path, "register_analysis")
    register_report = analyze_register_system(
        model_with_register, test_datasets, analysis_path
    )

    # Phase 4: Enhanced comparison with register insights
    print("\nPHASE 4: ENHANCED COMPARISON WITH REGISTER INSIGHTS")
    print("=" * 60)

    # Standard performance comparison
    print(
        f"\n{'Domain':<12} {'No Register':<20} {'With Register':<20} {'Register Impact':<15} {'Register Usage':<15}"
    )
    print(
        f"{'':12} {'MSE':<9} {'MAE':<9} {'MSE':<9} {'MAE':<9} {'MSE %':<7} {'MAE %':<7} {'Specialization':<15}"
    )
    print("-" * 110)

    total_mse_no, total_mae_no = 0, 0
    total_mse_with, total_mae_with = 0, 0
    num_domains = 0
    improvements = []

    for domain in sorted(results_no_register.keys()):
        if domain in results_with_register:
            mse_no, mae_no = results_no_register[domain]
            mse_with, mae_with = results_with_register[domain]

            # Skip failed domains
            if mse_no == float("inf") or mse_with == float("inf"):
                continue

            # Calculate improvements
            mse_improvement = ((mse_no - mse_with) / mse_no * 100) if mse_no > 0 else 0
            mae_improvement = ((mae_no - mae_with) / mae_no * 100) if mae_no > 0 else 0

            # Get register specialization for this domain
            specialization = "N/A"
            if domain in register_report["domain_analysis"]:
                spec_entropy = register_report["domain_analysis"][domain][
                    "specialization_entropy"
                ]
                if spec_entropy < 1.0:
                    specialization = "High"
                elif spec_entropy < 2.0:
                    specialization = "Medium"
                else:
                    specialization = "Low"

            print(
                f"{domain:<12} {mse_no:<9.6f} {mae_no:<9.6f} {mse_with:<9.6f} {mae_with:<9.6f} {mse_improvement:<7.2f} {mae_improvement:<7.2f} {specialization:<15}"
            )

            improvements.append(mse_improvement)
            total_mse_no += mse_no
            total_mae_no += mae_no
            total_mse_with += mse_with
            total_mae_with += mae_with
            num_domains += 1

    # Enhanced summary with register insights
    if num_domains > 0:
        avg_mse_improvement = (
            ((total_mse_no - total_mse_with) / total_mse_no * 100)
            if total_mse_no > 0
            else 0
        )
        avg_mae_improvement = (
            ((total_mae_no - total_mae_with) / total_mae_no * 100)
            if total_mae_no > 0
            else 0
        )

        print("-" * 110)
        print(
            f"{'AVERAGE':<12} {total_mse_no / num_domains:<9.6f} {total_mae_no / num_domains:<9.6f} {total_mse_with / num_domains:<9.6f} {total_mae_with / num_domains:<9.6f} {avg_mse_improvement:<7.2f} {avg_mae_improvement:<7.2f}"
        )

        # Register insights
        print(f"\n{'=' * 80}")
        print("REGISTER SYSTEM INSIGHTS")
        print(f"{'=' * 80}")

        print(
            f"Register Utilization: {register_report['summary']['register_utilization']:.1%}"
        )
        print(
            f"Total Registers Used: {register_report['summary']['total_registers_used']}/{register_report['register_system_config']['register_size']}"
        )
        print(
            f"Average Domain Specialization: {register_report['summary']['average_specialization']:.3f}"
        )

        if register_report["summary"]["most_specialized_domain"]:
            print(
                f"Most Specialized Domain: {register_report['summary']['most_specialized_domain']}"
            )
            print(
                f"Least Specialized Domain: {register_report['summary']['least_specialized_domain']}"
            )

        # Correlation analysis between specialization and improvement
        specializations = []
        domain_improvements = []

        for domain in sorted(results_no_register.keys()):
            if (
                domain in results_with_register
                and domain in register_report["domain_analysis"]
            ):
                mse_no, _ = results_no_register[domain]
                mse_with, _ = results_with_register[domain]

                if mse_no != float("inf") and mse_with != float("inf"):
                    improvement = (
                        ((mse_no - mse_with) / mse_no * 100) if mse_no > 0 else 0
                    )
                    specialization = register_report["domain_analysis"][domain][
                        "specialization_entropy"
                    ]

                    domain_improvements.append(improvement)
                    specializations.append(specialization)

        # Calculate correlation
        if len(specializations) > 2:
            correlation = np.corrcoef(specializations, domain_improvements)[0, 1]
            print(f"Specialization-Improvement Correlation: {correlation:.3f}")

            if abs(correlation) > 0.5:
                if correlation > 0:
                    insight = "Higher specialization correlates with better performance"
                else:
                    insight = "Lower specialization correlates with better performance"
                print(f"Key Insight: {insight}")

        # Enhanced conclusion with register insights
        print(f"\n{'=' * 80}")
        print("ENHANCED CONCLUSION")
        print(f"{'=' * 80}")

        if avg_mse_improvement > 5:
            conclusion = "REGISTER PROVIDES SIGNIFICANT BENEFIT"
            if register_report["summary"]["register_utilization"] > 0.8:
                explanation = "High register utilization suggests diverse domain needs"
            else:
                explanation = "Low register utilization suggests focused specialization"
        elif avg_mse_improvement > 1:
            conclusion = "REGISTER PROVIDES MODEST BENEFIT"
            explanation = f"Register utilization: {register_report['summary']['register_utilization']:.1%}"
        elif avg_mse_improvement > -1:
            conclusion = "REGISTER IMPACT IS NEGLIGIBLE"
            explanation = "Consider simpler architecture or different register design"
        else:
            conclusion = "REGISTER MAY BE HARMFUL"
            explanation = (
                "Register system might be interfering with domain-specific learning"
            )

        print(f"Performance: {conclusion}")
        print(f"Explanation: {explanation}")

        # Actionable recommendations
        print(f"\nRecommendations:")
        if avg_mse_improvement > 3:
            print("✓ Use register system in production")
            if register_report["summary"]["register_utilization"] < 0.5:
                print("✓ Consider reducing register size for efficiency")
        elif avg_mse_improvement > 0:
            print("? Consider register system if computational cost is acceptable")
            print("? Experiment with different register configurations")
        else:
            print("✗ Avoid register system or redesign the approach")
            print("✗ Focus on domain-specific architectures instead")

        # Save enhanced results
        enhanced_results = {
            "timestamp": datetime.now().isoformat(),
            "performance_comparison": {
                "results_no_register": results_no_register,
                "results_with_register": results_with_register,
                "avg_mse_improvement": avg_mse_improvement,
                "avg_mae_improvement": avg_mae_improvement,
            },
            "register_analysis": register_report,
            "insights": {
                "conclusion": conclusion,
                "explanation": explanation,
                "specialization_correlation": correlation
                if "correlation" in locals()
                else None,
            },
            "visualizations_generated": [
                "register_usage_heatmap.png",
                "register_similarity_matrix.png",
                "domain_embeddings_tsne.png",
                "register_analysis_report.json",
            ],
        }

        # Save to both directories
        for save_path in [config_no_register.save_path, config_with_register.save_path]:
            enhanced_file = os.path.join(save_path, "enhanced_ablation_results.json")
            try:
                os.makedirs(save_path, exist_ok=True)
                with open(enhanced_file, "w") as f:
                    json.dump(enhanced_results, f, indent=2)
                print(f"Enhanced results saved to: {enhanced_file}")
            except Exception as e:
                print(f"Error saving to {enhanced_file}: {e}")

        print(f"\n{'=' * 100}")
        print("ENHANCED REGISTER ABLATION STUDY COMPLETED")
        print(f"{'=' * 100}")
        print(f"Check {analysis_path}/ for detailed register visualizations")

        return enhanced_results

    return None


def run_dimensionality_sweep(configs):
    """Run training on new datasets for dimensionality analysis"""

    print("=" * 80)
    print("DIMENSIONALITY SWEEP EXPERIMENT")
    print("=" * 80)
    print("Training on new datasets to analyze complexity threshold")

    # SET FLAG FOR NEW DATASETS
    configs.dataset_type = "dimensionality_sweep"

    # Create datasets for dimensionality sweep
    train_datasets, test_datasets = create_domain_datasets(
        configs, dataset_type="dimensionality_sweep"
    )

    print(f"New datasets for dimensionality analysis:")
    for domain_name, dataset in train_datasets.items():
        sample = dataset[0]
        print(f"  {domain_name:15s}: {sample[0].shape} features")

    # Train no-register approach
    print("\n" + "=" * 60)
    print("TRAINING NO-REGISTER MODELS ON NEW DATASETS")
    print("=" * 60)

    configs_no_reg = copy.deepcopy(configs)
    configs_no_reg.dataset_type = "dimensionality_sweep"
    configs_no_reg.save_path = os.path.join(
        configs.save_path, "dimensionality_no_register"
    )

    model_no_reg, losses_no_reg, results_no_reg = train_approach_no_register(
        configs_no_reg
    )

    # Train with-register approach
    print("\n" + "=" * 60)
    print("TRAINING WITH-REGISTER MODELS ON NEW DATASETS")
    print("=" * 60)

    configs_with_reg = copy.deepcopy(configs)
    configs_with_reg.dataset_type = "dimensionality_sweep"
    configs_with_reg.save_path = os.path.join(
        configs.save_path, "dimensionality_with_register"
    )

    model_with_reg, losses_with_reg, results_with_reg = train_approach_3_complete_rose(
        configs_with_reg
    )

    # ADD THE REGISTER ANALYSIS HERE:
    print("\nPHASE 3: REGISTER ANALYSIS ACROSS DIMENSIONALITIES")
    print("=" * 60)

    analysis_path = os.path.join(configs_with_reg.save_path, "register_analysis")
    register_report = analyze_register_system(
        model_with_reg, test_datasets, analysis_path
    )

    # Save dimensionality sweep results
    dimensionality_results = {
        "timestamp": datetime.now().isoformat(),
        "experiment_type": "dimensionality_sweep",
        "datasets_tested": list(train_datasets.keys()),
        "results_no_register": results_no_reg,
        "results_with_register": results_with_reg,
        "register_analysis": register_report,  # ADD THIS LINE TOO
        "config": configs.__dict__ if hasattr(configs, "__dict__") else {},
    }

    # Save results
    results_path = os.path.join(configs.save_path, "dimensionality_sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(dimensionality_results, f, indent=2)

    print(f"\nDimensionality sweep results saved to: {results_path}")
    return model_no_reg, model_with_reg, dimensionality_results


def run_simple_comprehensive_ablation():
    """Simple comprehensive ablation using existing functions"""

    print("=" * 100)
    print("COMPREHENSIVE ABLATION STUDY (EXTENDED TRAINING)")
    print("=" * 100)
    print("25 epochs (15 pretrain + 10 finetune) on ALL datasets")

    set_seed(42)
    base_config = LightImputationConfig()

    # Create configurations
    config_no_reg = copy.deepcopy(base_config)
    config_with_reg = copy.deepcopy(base_config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config_no_reg.save_path = os.path.join(
        base_config.save_path, f"comprehensive_no_reg_{timestamp}"
    )
    config_with_reg.save_path = os.path.join(
        base_config.save_path, f"comprehensive_with_reg_{timestamp}"
    )

    print(f"Training both approaches on 13 domains (7 original + 6 sweep)")
    print(f"Results will be saved to separate directories")

    config_no_reg.dataset_type = "all"  # or "comprehensive"
    config_with_reg.dataset_type = "all"

    # Phase 1: No register
    print("\n" + "=" * 80)
    print("PHASE 1: NO-REGISTER (25 EPOCHS)")
    print("=" * 80)

    model_no_reg, losses_no_reg, results_no_reg = train_approach_2(config_no_reg)

    # Phase 2: With register
    print("\n" + "=" * 80)
    print("PHASE 2: WITH-REGISTER (25 EPOCHS)")
    print("=" * 80)

    model_with_reg, losses_with_reg, results_with_reg = train_approach_3_complete_rose(
        config_with_reg
    )

    # Phase 3: Register analysis
    print("\n" + "=" * 80)
    print("PHASE 3: REGISTER ANALYSIS")
    print("=" * 80)

    _, test_datasets = create_domain_datasets(config_with_reg, dataset_type="all")
    analysis_path = os.path.join(config_with_reg.save_path, "register_analysis")
    register_report = analyze_register_system(
        model_with_reg, test_datasets, analysis_path
    )

    # Phase 4: Simple comparison
    print("\n" + "=" * 100)
    print("COMPREHENSIVE COMPARISON RESULTS")
    print("=" * 100)

    print(
        f"{'Domain':<20} {'Features':<8} {'No Register':<12} {'With Register':<12} {'Improvement':<12}"
    )
    print(f"{'':20} {'':8} {'MSE':<12} {'MSE':<12} {'%':<12}")
    print("-" * 80)

    improvements = []
    for domain in sorted(results_no_reg.keys()):
        if domain in results_with_reg:
            mse_no = results_no_reg[domain][0]
            mse_with = results_with_reg[domain][0]

            if mse_no != float("inf") and mse_with != float("inf"):
                improvement = ((mse_no - mse_with) / mse_no) * 100
                improvements.append(improvement)

                # Get feature count
                domain_features = {
                    "ETTh1": 7,
                    "ETTh2": 7,
                    "ETTm1": 7,
                    "ETTm2": 7,
                    "weather": 21,
                    "traffic": 865,
                    "electricity": 321,
                    "beijing_pm25": 8,
                    "air_quality": 12,
                    "bike_sharing": 15,
                    "appliances_energy": 29,
                    "electric_power": 53,
                    "pamap2": 55,
                }
                features = domain_features.get(domain, "?")

                print(
                    f"{domain:<20} {features:<8} {mse_no:<12.6f} {mse_with:<12.6f} {improvement:<+12.2f}"
                )

    if improvements:
        avg_improvement = np.mean(improvements)
        positive_count = len([x for x in improvements if x > 0])
        print("-" * 80)
        print(f"Summary: {positive_count}/{len(improvements)} domains improved")
        print(f"Average improvement: {avg_improvement:.2f}%")
        print(
            f"Register utilization: {register_report['summary']['register_utilization']:.1%}"
            if register_report
            else "Analysis failed"
        )

    return results_no_reg, results_with_reg, register_report


def create_data_scarcity_datasets(configs, data_fraction=1.0, seed=42):
    """
    Create datasets with limited data per domain for data scarcity experiments

    Args:
        configs: Configuration object
        data_fraction: Fraction of data to use (0.1 = 10%, 0.25 = 25%, etc.)
        seed: Random seed for reproducible sampling
    """
    import random

    random.seed(seed)
    np.random.seed(seed)

    # Get full datasets
    train_datasets, test_datasets = create_domain_datasets(
        configs, dataset_type="original"
    )

    # Sample fraction of training data per domain
    limited_train_datasets = {}

    for domain_name, full_dataset in train_datasets.items():
        dataset_size = len(full_dataset)
        sample_size = max(1, int(dataset_size * data_fraction))

        # Random sampling of indices
        indices = list(range(dataset_size))
        sampled_indices = random.sample(indices, sample_size)

        # Create subset dataset
        limited_dataset = torch.utils.data.Subset(full_dataset, sampled_indices)
        limited_train_datasets[domain_name] = limited_dataset

        print(
            f"  {domain_name:12s}: {dataset_size} -> {sample_size} samples ({data_fraction * 100:.1f}%)"
        )

    return limited_train_datasets, test_datasets


def train_single_domain_baseline(configs, domain_name, train_dataset, test_dataset):
    """
    Train single-domain FEDformer baseline for comparison
    """
    from multi_domain_fedformer import MultiDomainFEDformerWithoutRegister

    # Create single-domain model (same architecture as your multi-domain)
    single_domain_model = MultiDomainFEDformerWithoutRegister(configs)

    # Only keep the embedding for this specific domain
    domain_embedding = single_domain_model.domain_embeddings[domain_name]
    single_domain_model.domain_embeddings = nn.ModuleDict(
        {domain_name: domain_embedding}
    )

    # Move to device
    single_domain_model = single_domain_model.to(device)

    # Create trainer (reuse NoRegisterTrainer)
    trainer = NoRegisterTrainer(single_domain_model, configs)

    # Training data for this domain only
    single_domain_datasets = {domain_name: train_dataset}

    print(f"Training single-domain model for {domain_name}...")

    # Training loop (simplified)
    best_loss = float("inf")
    for epoch in range(configs.train_epochs):
        # Train epoch
        train_loss, _ = trainer.train_epoch_imputation_finetune(single_domain_datasets)

        if train_loss < best_loss and train_loss > 0:
            best_loss = train_loss
            print(f"  Epoch {epoch + 1}: Loss = {train_loss:.6f}")

    # Save single-domain model
    try:
        os.makedirs(configs.save_path, exist_ok=True)
        model_path = os.path.join(
            configs.save_path, f"single_{domain_name}_{configs.train_epochs}epochs.pth"
        )
        torch.save(single_domain_model.state_dict(), model_path)
        print(f"Single-domain model saved: {model_path}")
    except Exception as e:
        print(f"Failed to save single-domain model: {e}")

    # Evaluate
    test_datasets_single = {domain_name: test_dataset}
    results = trainer.evaluate_domain_separate(test_datasets_single)

    return single_domain_model, results[domain_name]


def create_adjusted_config_for_scarcity(base_config, data_fraction, domain_sizes=None):
    """
    Adjust configuration parameters for data scarcity experiments

    Args:
        base_config: LightImputationConfig instance
        data_fraction: Fraction of data (0.1, 0.25, 0.5, 0.75, 1.0)
        domain_sizes: Dict of domain sizes for batch size calculation
    """
    import copy

    adjusted_config = copy.deepcopy(base_config)
    adjusted_config.two_phase_training = False

    # REDUCED epochs for faster training
    if data_fraction <= 0.1:  # 10% data
        adjusted_config.train_epochs = 10
        adjusted_config.pretrain_epochs = 0
        adjusted_config.finetune_epochs = 10

    elif data_fraction <= 0.25:  # 25% data
        adjusted_config.train_epochs = 12
        adjusted_config.pretrain_epochs = 0
        adjusted_config.finetune_epochs = 12

    elif data_fraction <= 0.5:  # 50% data
        adjusted_config.train_epochs = 15
        adjusted_config.pretrain_epochs = 0
        adjusted_config.finetune_epochs = 15

    else:  #  100% data
        adjusted_config.train_epochs = 20
        adjusted_config.pretrain_epochs = 0
        adjusted_config.finetune_epochs = 20

    # 2. Adjust batch size for smaller datasets
    if domain_sizes and data_fraction < 1.0:
        # Estimate minimum domain size after sampling
        min_domain_size = min(domain_sizes.values())
        estimated_samples = int(min_domain_size * data_fraction)

        # Ensure at least 6-8 batches per epoch
        target_batches = 8
        suggested_batch_size = max(4, estimated_samples // target_batches)

        # Don't go below 8 or above original batch size
        adjusted_config.batch_size = max(
            8, min(adjusted_config.batch_size, suggested_batch_size)
        )

    # 3. Adjust learning rate slightly for small datasets
    if data_fraction <= 0.1:
        adjusted_config.learning_rate = (
            base_config.learning_rate * 0.8
        )  # Slightly lower
    elif data_fraction <= 0.25:
        adjusted_config.learning_rate = base_config.learning_rate * 0.9
    # else keep original learning rate

    # 4. Prevent overfitting on small datasets
    if data_fraction <= 0.25:
        adjusted_config.dropout = min(
            0.2, adjusted_config.dropout * 1.2
        )  # Increase dropout

    print(f"Adjusted config for {data_fraction * 100:.0f}% data:")
    print(
        f"  Epochs: {adjusted_config.train_epochs} ({adjusted_config.pretrain_epochs}+{adjusted_config.finetune_epochs})"
    )
    print(f"  Batch size: {adjusted_config.batch_size}")
    print(f"  Learning rate: {adjusted_config.learning_rate}")
    print(f"  Patience: {adjusted_config.patience}")
    print(f"  Dropout: {adjusted_config.dropout}")

    return adjusted_config


def run_data_scarcity_experiment_with_adjusted_configs(base_config):
    """
    Data scarcity experiment with properly adjusted configurations
    """
    print("=" * 80)
    print("DATA SCARCITY EXPERIMENT WITH ADJUSTED CONFIGS")
    print("=" * 80)

    # Data fractions to test
    data_fractions = [0.1, 0.25, 0.5, 1.0]  # Start with 4 points

    # Get domain sizes for batch size calculation
    print("Getting dataset sizes...")
    temp_train_datasets, temp_test_datasets = create_domain_datasets(base_config)
    domain_sizes = {name: len(dataset) for name, dataset in temp_train_datasets.items()}

    print("Domain sizes:")
    for name, size in domain_sizes.items():
        print(f"  {name:12s}: {size} samples")

    # Store results
    multi_domain_results = {}
    single_domain_results = {}

    for fraction in data_fractions:
        print(f"\n{'=' * 60}")
        print(f"TESTING WITH {fraction * 100:.0f}% TRAINING DATA")
        print(f"{'=' * 60}")

        # Create adjusted configs for this data fraction
        configs_multi = create_adjusted_config_for_scarcity(
            base_config, fraction, domain_sizes
        )
        configs_single = create_adjusted_config_for_scarcity(
            base_config, fraction, domain_sizes
        )

        # Create limited datasets
        print(f"\nCreating {fraction * 100:.0f}% data samples:")
        train_datasets_limited, test_datasets = create_data_scarcity_datasets(
            base_config, data_fraction=fraction, seed=42
        )

        # Train multi-domain model
        print(f"\nTraining multi-domain model...")
        try:
            model_multi, _, results_multi = train_approach_no_register(
                configs_multi, train_datasets_limited, test_datasets
            )
            multi_domain_results[fraction] = results_multi
            print(f"Multi-domain training completed")

            # Save multi-domain model
            os.makedirs(configs_multi.save_path, exist_ok=True)
            model_path = os.path.join(
                configs_multi.save_path, f"multi_domain_{fraction * 100:.0f}pct.pth"
            )
            torch.save(model_multi.state_dict(), model_path)
            print(f"Multi-domain model saved: {model_path}")

        except Exception as e:
            print(f"Multi-domain training failed: {e}")
            multi_domain_results[fraction] = {}

        # Train single-domain baselines
        print(f"\nTraining single-domain baselines...")
        single_results_this_fraction = {}

        for domain_name in train_datasets_limited.keys():
            print(f"  Training {domain_name}...")

            try:
                _, domain_result = train_single_domain_baseline(
                    configs_single,
                    domain_name,
                    train_datasets_limited[domain_name],
                    test_datasets[domain_name],
                )
                single_results_this_fraction[domain_name] = domain_result
                print(f"    Success: MSE={domain_result[0]:.6f}")

            except Exception as e:
                print(f"    Failed: {e}")
                single_results_this_fraction[domain_name] = (float("inf"), float("inf"))

        single_domain_results[fraction] = single_results_this_fraction

        # Quick comparison
        print(f"\nQuick comparison at {fraction * 100:.0f}% data:")
        for domain_name in single_results_this_fraction.keys():
            if domain_name in results_multi:
                multi_mse = results_multi[domain_name][0]
                single_mse = single_results_this_fraction[domain_name][0]

                if single_mse != float("inf") and multi_mse != float("inf"):
                    improvement = ((single_mse - multi_mse) / single_mse) * 100
                    winner = "Multi" if improvement > 0 else "Single"
                    print(f"  {domain_name}: {winner} wins by {abs(improvement):.1f}%")

    return multi_domain_results, single_domain_results


# Add these debugging functions to run_imputation_comparison.py


def debug_model_complexity(model_multi, model_single_example):
    """Compare model complexities to identify architectural issues"""

    print("=" * 60)
    print("MODEL COMPLEXITY ANALYSIS")
    print("=" * 60)

    # Count parameters
    multi_params = sum(p.numel() for p in model_multi.parameters())
    single_params = sum(p.numel() for p in model_single_example.parameters())

    print(f"Multi-domain parameters: {multi_params:,}")
    print(f"Single-domain parameters: {single_params:,}")
    print(f"Parameter ratio (Multi/Single): {multi_params / single_params:.2f}")

    # Check domain embeddings
    if hasattr(model_multi, "domain_embeddings"):
        print(f"\nDomain embeddings: {len(model_multi.domain_embeddings)} domains")
        for domain_name, embedding in model_multi.domain_embeddings.items():
            domain_params = sum(p.numel() for p in embedding.parameters())
            print(f"  {domain_name}: {domain_params:,} parameters")

    # Check frequency learning components
    if hasattr(model_multi, "freq_learning"):
        freq_params = sum(p.numel() for p in model_multi.freq_learning.parameters())
        print(f"Frequency learning parameters: {freq_params:,}")

    print(
        f"\nComplexity ratio suggests: {'Over-parameterized' if multi_params / single_params > 2 else 'Reasonable'}"
    )


def debug_training_strategy_simplified(base_config):
    """Test simplified multi-domain training without frequency pretraining"""

    print("=" * 80)
    print("DEBUGGING: SIMPLIFIED MULTI-DOMAIN TRAINING")
    print("=" * 80)
    print("Testing multi-domain WITHOUT frequency pretraining")

    # Create simplified config
    configs_simplified = copy.deepcopy(base_config)
    configs_simplified.two_phase_training = False  # Skip frequency pretraining
    configs_simplified.train_epochs = 10  # Same as single-domain
    configs_simplified.save_path = os.path.join(
        base_config.save_path, "debug_simplified_multidomain"
    )

    print(
        f"Simplified training: {configs_simplified.train_epochs} epochs (no pretraining)"
    )

    # Test on limited domains first
    test_domains = ["ETTh1", "ETTh2"]  # Just 2 similar domains
    data_fraction = 0.25  # 25% data

    print(f"Testing on {len(test_domains)} domains with {data_fraction * 100}% data")

    # Create limited datasets
    train_datasets_limited, test_datasets = create_data_scarcity_datasets(
        base_config, data_fraction=data_fraction, seed=42
    )

    # Filter to test domains only
    train_datasets_test = {
        k: v for k, v in train_datasets_limited.items() if k in test_domains
    }
    test_datasets_test = {k: v for k, v in test_datasets.items() if k in test_domains}

    print(f"Filtered datasets: {list(train_datasets_test.keys())}")

    # Train simplified multi-domain model
    print("\nTraining simplified multi-domain...")
    model_simplified, _, results_multi_simplified = train_approach_no_register(
        configs_simplified, train_datasets_test, test_datasets_test
    )

    # Train single-domain baselines
    print("\nTraining single-domain baselines...")
    results_single_simplified = {}

    for domain_name in test_domains:
        print(f"Training single-domain {domain_name}...")
        _, result = train_single_domain_baseline(
            configs_simplified,
            domain_name,
            train_datasets_test[domain_name],
            test_datasets_test[domain_name],
        )
        results_single_simplified[domain_name] = result

    # Compare results
    print(f"\n{'=' * 60}")
    print("SIMPLIFIED MULTI-DOMAIN vs SINGLE-DOMAIN COMPARISON")
    print(f"{'=' * 60}")
    print(
        f"{'Domain':<10} {'Multi-MSE':<12} {'Single-MSE':<12} {'Winner':<10} {'Margin':<10}"
    )
    print("-" * 60)

    for domain in test_domains:
        if domain in results_multi_simplified and domain in results_single_simplified:
            multi_mse = results_multi_simplified[domain][0]
            single_mse = results_single_simplified[domain][0]

            if single_mse != float("inf") and multi_mse != float("inf"):
                if multi_mse < single_mse:
                    winner = "Multi"
                    margin = f"{((single_mse - multi_mse) / single_mse) * 100:+.1f}%"
                else:
                    winner = "Single"
                    margin = f"{((multi_mse - single_mse) / single_mse) * 100:+.1f}%"

                print(
                    f"{domain:<10} {multi_mse:<12.6f} {single_mse:<12.6f} {winner:<10} {margin:<10}"
                )

    return model_simplified, results_multi_simplified, results_single_simplified


def debug_data_leakage(train_datasets, test_datasets):
    """Check for potential data leakage or inconsistent splits"""

    print("=" * 60)
    print("DATA LEAKAGE ANALYSIS")
    print("=" * 60)

    for domain_name in train_datasets.keys():
        train_dataset = train_datasets[domain_name]
        test_dataset = test_datasets[domain_name]

        print(f"\n{domain_name}:")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")

        # Check first few samples to ensure different data
        try:
            train_sample = train_dataset[0][0]  # First sequence
            test_sample = test_dataset[0][0]  # First sequence

            # Check if identical (potential leakage)
            if torch.allclose(
                torch.FloatTensor(train_sample), torch.FloatTensor(test_sample)
            ):
                print(f"  ⚠️  WARNING: Identical first samples detected!")
            else:
                print(f"  ✓ Train/test data appears distinct")

            print(f"  Train shape: {train_sample.shape}")
            print(f"  Test shape: {test_sample.shape}")

        except Exception as e:
            print(f"  Error checking samples: {e}")


def debug_loss_components(model, train_datasets, configs, num_batches=5):
    """Analyze individual loss components to identify training issues"""

    print("=" * 60)
    print("LOSS COMPONENT ANALYSIS")
    print("=" * 60)

    model.eval()
    device = next(model.parameters()).device

    domain_losses = {}

    for domain_name, domain_dataset in list(train_datasets.items())[
        :2
    ]:  # Just 2 domains
        print(f"\nAnalyzing {domain_name} losses...")

        try:
            from imputation_trainer import ImputationDataset

            domain_imputation = ImputationDataset(domain_dataset, missing_rate=0.2)

            domain_loader = DataLoader(
                domain_imputation,
                batch_size=8,  # Small batch for debugging
                shuffle=False,
            )

            batch_losses = []

            with torch.no_grad():
                for batch_idx, batch_data in enumerate(domain_loader):
                    if batch_idx >= num_batches:
                        break

                    x_missing, x_mark, mask, x_target = batch_data
                    x_missing = x_missing.to(device)
                    x_mark = x_mark.to(device)
                    mask = mask.to(device)
                    x_target = x_target.to(device)

                    # Forward pass
                    try:
                        if hasattr(model, "_forward_imputation"):
                            # Multi-domain model
                            dec_out, reconstructed, register_loss = (
                                model._forward_imputation(
                                    x_missing, x_mark, mask, domain_name
                                )
                            )
                        else:
                            # Fallback
                            dec_out, reconstructed, register_loss = model(
                                x_missing,
                                x_mark,
                                mask,
                                domain_name,
                                "imputation_finetune",
                            )

                        # Calculate losses
                        missing_mask = (~mask).bool()
                        if missing_mask.any():
                            mse_loss = F.mse_loss(
                                dec_out[missing_mask], x_target[missing_mask]
                            )
                            mae_loss = F.l1_loss(
                                dec_out[missing_mask], x_target[missing_mask]
                            )
                        else:
                            mse_loss = F.mse_loss(dec_out, x_target)
                            mae_loss = F.l1_loss(dec_out, x_target)

                        batch_losses.append(
                            {
                                "mse": mse_loss.item(),
                                "mae": mae_loss.item(),
                                "register": register_loss.item()
                                if register_loss is not None
                                else 0.0,
                            }
                        )

                    except Exception as e:
                        print(f"    Error in batch {batch_idx}: {e}")
                        continue

            if batch_losses:
                avg_mse = np.mean([b["mse"] for b in batch_losses])
                avg_mae = np.mean([b["mae"] for b in batch_losses])
                avg_register = np.mean([b["register"] for b in batch_losses])

                print(f"  Average MSE: {avg_mse:.6f}")
                print(f"  Average MAE: {avg_mae:.6f}")
                print(f"  Average Register Loss: {avg_register:.6f}")

                domain_losses[domain_name] = {
                    "mse": avg_mse,
                    "mae": avg_mae,
                    "register": avg_register,
                }
            else:
                print(f"  No valid batches processed")

        except Exception as e:
            print(f"  Error processing {domain_name}: {e}")

    return domain_losses


def run_complete_debugging_suite(base_config):
    """Run all debugging steps systematically"""

    print("=" * 100)
    print("COMPLETE MULTI-DOMAIN DEBUGGING SUITE")
    print("=" * 100)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_config = copy.deepcopy(base_config)
    debug_config.save_path = os.path.join(
        base_config.save_path, f"debugging_{timestamp}"
    )

    # Step 1: Test simplified training
    print("\nSTEP 1: Testing simplified multi-domain training...")
    model_simplified, results_multi, results_single = (
        debug_training_strategy_simplified(debug_config)
    )

    # Step 2: Model complexity analysis
    print("\nSTEP 2: Analyzing model complexity...")
    # Create example single-domain model for comparison
    from multi_domain_fedformer import MultiDomainFEDformerWithoutRegister

    model_single_example = MultiDomainFEDformerWithoutRegister(debug_config)
    debug_model_complexity(model_simplified, model_single_example)

    # Step 3: Data leakage check
    print("\nSTEP 3: Checking for data leakage...")
    train_datasets, test_datasets = create_domain_datasets(debug_config)
    debug_data_leakage(train_datasets, test_datasets)

    # Step 4: Loss component analysis
    print("\nSTEP 4: Analyzing loss components...")
    # Create small dataset for analysis
    train_limited, _ = create_data_scarcity_datasets(debug_config, data_fraction=0.1)
    loss_analysis = debug_loss_components(model_simplified, train_limited, debug_config)

    # Summary
    print(f"\n{'=' * 80}")
    print("DEBUGGING SUMMARY")
    print(f"{'=' * 80}")
    print("1. Simplified training results: See comparison above")
    print("2. Model complexity: Check parameter ratios")
    print("3. Data integrity: Check for warnings")
    print("4. Loss components: Check for anomalies")
    print(f"\nDebugging results saved to: {debug_config.save_path}")

    return {
        "simplified_results": (results_multi, results_single),
        "loss_analysis": loss_analysis,
        "debug_path": debug_config.save_path,
    }


# Update your main function
def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-domain FEDformer Imputation"
    )
    parser.add_argument(
        "--approach", type=int, choices=[1, 2, 3], help="Approach to run"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="light",
        choices=["full", "light"],
        help="Configuration type",
    )
    parser.add_argument(
        "--ablation", action="store_true", help="Run register ablation study"
    )
    parser.add_argument(
        "--enhanced-ablation",
        action="store_true",
        help="Run enhanced ablation with register exploration",
    )
    parser.add_argument(  # ADD THIS
        "--dimensionality-sweep",  # ADD THIS
        action="store_true",  # ADD THIS
        help="Run dimensionality sweep experiment",  # ADD THIS
    )  # ADD THIS
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_path", type=str, default="./results/", help="Save path")

    parser.add_argument(
        "--comprehensive", action="store_true", help="Run comprehensive ablation study"
    )

    parser.add_argument(
        "--data-scarcity",
        action="store_true",
        help="Run data scarcity experiment (Approach 1)",
    )

    parser.add_argument("--debug", action="store_true", help="Run debugging suite")

    parser.add_argument(
        "--thesis-eval",
        action="store_true",
        help="Run complete thesis evaluation framework",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    # CREATE CONFIG AND TIMESTAMP EARLY
    if args.config == "light":
        configs = LightImputationConfig()
    else:
        configs = ImputationConfig()

    if args.save_path:
        configs.save_path = args.save_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # NOW USE THEM IN CONDITIONS

    if args.thesis_eval:
        print("Starting thesis evaluation framework...")
        from evaluation_framework import ThesisEvaluationFramework
        from configs.test_imputation_config import (
            TestImputationConfig,
        )  # Use test config

        configs = TestImputationConfig()  # Changed from LightImputationConfig
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        configs.save_path = os.path.join(configs.save_path, f"thesis_test_{timestamp}")

        framework = ThesisEvaluationFramework(configs)
        results = framework.run_complete_evaluation()
        print("Thesis evaluation test completed!")

    elif args.data_scarcity:
        print("Starting Data Scarcity Experiment (Approach 1)...")
        configs.save_path = os.path.join(
            configs.save_path, f"data_scarcity_{timestamp}"
        )

        # Run the experiment
        multi_results, single_results = (
            run_data_scarcity_experiment_with_adjusted_configs(configs)
        )

        print("Data scarcity experiment completed!")
        print(f"Results saved to: {configs.save_path}")

    elif args.debug:
        print("Starting debugging suite...")
        debug_results = run_complete_debugging_suite(configs)
        print("Debugging completed!")

    elif args.comprehensive:
        print("Starting comprehensive ablation study...")
        results = run_simple_comprehensive_ablation()
        print("Comprehensive ablation completed!")

    elif args.enhanced_ablation:
        # Run enhanced ablation study with register exploration
        print("Starting Enhanced Register Ablation Study with Domain Exploration...")
        results = run_enhanced_ablation_study()
        print("Enhanced ablation study completed!")

    elif args.dimensionality_sweep:  # FIXED: Now it's elif
        configs.save_path = os.path.join(
            configs.save_path, f"dimensionality_sweep_{timestamp}"
        )
        print("Starting dimensionality sweep experiment...")
        results = run_dimensionality_sweep(configs)
        print(f"Dimensionality sweep completed! Results saved to: {configs.save_path}")

    elif args.ablation:
        # Run standard ablation study
        results = run_complete_ablation_study()

    elif args.approach:
        # Run single approach
        if args.approach == 3:
            configs.dataset_type = "dimensionality_sweep"
            configs.save_path = os.path.join(
                configs.save_path, f"rose_with_register_{timestamp}"
            )
            model, losses, results = train_approach_3_complete_rose(configs)

            # Add register exploration for single approach
            _, test_datasets = create_domain_datasets(
                configs
            )  # Uses default "original"
            analysis_path = os.path.join(configs.save_path, "register_analysis")
            register_report = analyze_register_system(
                model, test_datasets, analysis_path
            )
            print(
                f"Register analysis completed! Check {analysis_path}/ for visualizations"
            )

        elif args.approach == 2:
            configs.save_path = os.path.join(
                configs.save_path, f"rose_no_register_{timestamp}"
            )
            model, losses, results = train_approach_no_register(configs)

    # CHANGE your final else statement to include the new option
    else:
        print(
            "Please specify --approach, --ablation, --enhanced-ablation, --dimensionality-sweep, or --data-scarcity"
        )
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()  # Call the main() function that parses arguments
    sys.exit(exit_code)
