import datetime
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os
import random
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from data_provider.data_loader import (
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom,
)
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_missing_mask(data, missing_rate=0.15, pattern="random"):
    """Create missing value patterns for imputation testing"""
    if len(data.shape) == 2:  # [seq_len, features]
        L, C = data.shape
        mask = torch.ones_like(data, dtype=torch.bool)

        if pattern == "random":
            num_missing = int(L * C * missing_rate)
            flat_mask = mask.reshape(-1)  # Changed from view to reshape
            indices = torch.randperm(len(flat_mask))[:num_missing]
            flat_mask[indices] = False
            mask = flat_mask.reshape(L, C)  # Changed from view to reshape

        elif pattern == "block":
            # Missing entire time steps
            num_missing_steps = int(L * missing_rate)
            if num_missing_steps > 0:
                start_idx = torch.randint(0, L - num_missing_steps + 1, (1,)).item()
                mask[start_idx : start_idx + num_missing_steps, :] = False

        elif pattern == "consecutive":
            # Consecutive missing per feature
            for c in range(C):
                if torch.rand(1) < missing_rate:
                    missing_len = torch.randint(1, max(1, int(L * 0.3)), (1,)).item()
                    if missing_len < L:
                        start_idx = torch.randint(0, L - missing_len + 1, (1,)).item()
                        mask[start_idx : start_idx + missing_len, c] = False

    return mask


class ImputationDataset(Dataset):
    """Enhanced wrapper with normalization for fair comparison"""

    def __init__(
        self,
        base_dataset,
        missing_rate=0.15,
        missing_pattern="random",
        shared_scaler=None,
    ):
        self.base_dataset = base_dataset
        self.missing_rate = missing_rate
        self.missing_pattern = missing_pattern

        if shared_scaler is not None:
            self.scaler = shared_scaler
            print(f"    Using shared normalization scaler")
        else:
            self._setup_normalization()

    def __len__(self):
        return len(self.base_dataset)

    def _setup_normalization(self):
        """Setup normalization statistics from the dataset"""
        from sklearn.preprocessing import StandardScaler

        print(f"  Computing normalization statistics...")
        all_data = []

        # Sample data to compute statistics (use first 1000 samples)
        sample_size = min(1000, len(self.base_dataset))
        for i in range(sample_size):
            seq_x, _, _, _ = self.base_dataset[i]
            if torch.is_tensor(seq_x):
                seq_x = seq_x.detach().cpu().numpy()
            all_data.append(seq_x)

        # Flatten for scaler
        data_array = np.array(all_data)  # [N, L, C]
        original_shape = data_array.shape
        data_flat = data_array.reshape(-1, data_array.shape[-1])  # [N*L, C]

        # Fit scaler
        self.scaler = StandardScaler()
        self.scaler.fit(data_flat)

        print(f"  Normalization fitted on {len(all_data)} samples")

    def _normalize_data(self, data):
        """Normalize data using fitted scaler"""
        original_shape = data.shape
        data_flat = data.reshape(-1, data.shape[-1])
        data_normalized = self.scaler.transform(data_flat)
        return data_normalized.reshape(original_shape)

    def _denormalize_data(self, data):
        """Denormalize data for evaluation"""
        original_shape = data.shape
        data_flat = data.reshape(-1, data.shape[-1])
        data_denormalized = self.scaler.inverse_transform(data_flat)
        return data_denormalized.reshape(original_shape)

    def __getitem__(self, index):
        try:
            seq_x, seq_y, seq_x_mark, seq_y_mark = self.base_dataset[index]

            # CRITICAL FIX: Ensure all tensors are float32
            seq_x = (
                torch.FloatTensor(seq_x)
                if not torch.is_tensor(seq_x)
                else seq_x.float()
            )
            seq_x_mark = (
                torch.FloatTensor(seq_x_mark)
                if not torch.is_tensor(seq_x_mark)
                else seq_x_mark.float()
            )

            # NORMALIZE DATA
            seq_x_np = seq_x.detach().cpu().numpy()
            seq_x_normalized = self._normalize_data(seq_x_np)
            seq_x = torch.FloatTensor(seq_x_normalized)

            # Create missing mask
            mask = create_missing_mask(seq_x, self.missing_rate, self.missing_pattern)
            mask = mask.bool()  # Ensure boolean

            # Apply missing values
            seq_x_missing = seq_x.clone().float()
            seq_x_missing[~mask] = float("nan")

            # CRITICAL FIX: Return all tensors with correct dtypes
            return (
                seq_x_missing.float(),  # float32
                seq_x_mark.float(),  # float32
                mask.bool(),  # boolean
                seq_x.float(),  # float32 (target)
            )

        except Exception as e:
            print(f"    Error in __getitem__({index}): {e}")
            # Safe fallback
            seq_shape = (96, 7)  # Default shape
            seq_x = torch.zeros(seq_shape, dtype=torch.float32)
            seq_x_mark = torch.zeros((seq_shape[0], 4), dtype=torch.float32)
            mask = torch.ones(seq_shape, dtype=torch.bool)
            return seq_x, seq_x_mark, mask, seq_x


def create_domain_datasets(configs, dataset_type="original"):
    # Check config flag if dataset_type not provided
    if dataset_type is None:
        dataset_type = getattr(configs, "dataset_type", "original")

    if dataset_type == "all" or dataset_type == "comprehensive":
        # Use all domains - combine both sets
        domain_configs = {
            # Original domains
            "ETTh1": {"data_path": "ETT/ETTh1.csv", "dataset_class": Dataset_ETT_hour},
            "ETTh2": {"data_path": "ETT/ETTh2.csv", "dataset_class": Dataset_ETT_hour},
            "ETTm1": {
                "data_path": "ETT/ETTm1.csv",
                "dataset_class": Dataset_ETT_minute,
            },
            "ETTm2": {
                "data_path": "ETT/ETTm2.csv",
                "dataset_class": Dataset_ETT_minute,
            },
            "weather": {
                "data_path": "weather/weather.csv",
                "dataset_class": Dataset_Custom,
            },
            "traffic": {
                "data_path": "traffic/traffic_with_date.csv",
                "dataset_class": Dataset_Custom,
            },
            "electricity": {
                "data_path": "electricity/electricity.csv",
                "dataset_class": Dataset_Custom,
            },
            # Dimensionality sweep domains
            "beijing_pm25": {
                "data_path": "beijing_pm25/beijing_pm25.csv",
                "dataset_class": Dataset_Custom,
            },
            "air_quality": {
                "data_path": "air_quality/AirQualityUCI.csv",
                "dataset_class": Dataset_Custom,
            },
            "bike_sharing": {
                "data_path": "bike_sharing/bike_sharing.csv",
                "dataset_class": Dataset_Custom,
            },
            "appliances_energy": {
                "data_path": "appliances_energy/appliances_energy.csv",
                "dataset_class": Dataset_Custom,
            },
            "electric_power": {
                "data_path": "electric_power/power_engineered.csv",
                "dataset_class": Dataset_Custom,
            },
            "pamap2": {
                "data_path": "pamap2/pamap2_52features.csv",
                "dataset_class": Dataset_Custom,
            },
        }
    elif dataset_type == "original":
        # Your existing domain configs
        domain_configs = {
            "ETTh1": {"data_path": "ETT/ETTh1.csv", "dataset_class": Dataset_ETT_hour},
            "ETTh2": {"data_path": "ETT/ETTh2.csv", "dataset_class": Dataset_ETT_hour},
            "ETTm1": {
                "data_path": "ETT/ETTm1.csv",
                "dataset_class": Dataset_ETT_minute,
            },
            "ETTm2": {
                "data_path": "ETT/ETTm2.csv",
                "dataset_class": Dataset_ETT_minute,
            },
            "weather": {
                "data_path": "weather/weather.csv",
                "dataset_class": Dataset_Custom,
            },
            "traffic": {
                "data_path": "traffic/traffic_with_date.csv",
                "dataset_class": Dataset_Custom,
            },
            "electricity": {
                "data_path": "electricity/electricity.csv",
                "dataset_class": Dataset_Custom,
            },
        }

    elif dataset_type == "dimensionality_sweep":
        domain_configs = {
            "beijing_pm25": {
                "data_path": "beijing_pm25/beijing_pm25.csv",
                "dataset_class": Dataset_Custom,
                "features": 8,  # Changed from 5 to match actual (8-1 for target)
            },
            "air_quality": {
                "data_path": "air_quality/AirQualityUCI.csv",
                "dataset_class": Dataset_Custom,
                "features": 12,  # Changed from 9 to match actual (12-1 for target)
            },
            "bike_sharing": {
                "data_path": "bike_sharing/bike_sharing.csv",
                "dataset_class": Dataset_Custom,
                "features": 15,  # Changed from 12 to match actual (15-1 for target)
            },
            "appliances_energy": {
                "data_path": "appliances_energy/appliances_energy.csv",
                "dataset_class": Dataset_Custom,
                "features": 29,  # Changed from 25 to match actual (29-1 for target)
            },
            "electric_power": {
                "data_path": "electric_power/power_engineered.csv",
                "dataset_class": Dataset_Custom,
                "features": 53,  # Changed from 50 to match actual (53-1 for target)
            },
            "pamap2": {
                "data_path": "pamap2/pamap2_52features.csv",
                "dataset_class": Dataset_Custom,
                "features": 55,  # Changed from 52 to match actual (55-1 for target)
            },
        }

    train_datasets = {}
    test_datasets = {}

    for domain_name, domain_config in domain_configs.items():
        # Create train dataset
        train_dataset = domain_config["dataset_class"](
            root_path=configs.root_path,
            flag="train",
            size=[configs.seq_len, 0, 0],
            features=configs.features,  # This should be "M"
            target=configs.target,
            data_path=domain_config["data_path"],
            timeenc=1 if configs.embed == "timeF" else 0,
            freq=configs.freq,
        )

        # Create test dataset
        test_dataset = domain_config["dataset_class"](
            root_path=configs.root_path,
            flag="test",
            size=[configs.seq_len, 0, 0],
            features=configs.features,
            target=configs.target,
            data_path=domain_config["data_path"],
            timeenc=1 if configs.embed == "timeF" else 0,
            freq=configs.freq,
        )

        # Debug: Check the shape of first sample
        sample = train_dataset[0]

        train_datasets[domain_name] = train_dataset
        test_datasets[domain_name] = test_dataset

    return train_datasets, test_datasets


def train_approach_2(configs):
    """
    Train Approach 2: Multi-domain FEDformer WITHOUT register (Clean Implementation)
    This calls the domain-separate no-register implementation for fair comparison
    """
    print("=" * 80)
    print("TRAINING APPROACH 2: MULTI-DOMAIN WITHOUT REGISTER")
    print("=" * 80)
    print("Using domain-separate training (identical to Approach 3 except NO register)")

    # Import the clean no-register implementation
    try:
        from run_imputation_comparison import train_approach_no_register

        # Call the clean implementation
        return train_approach_no_register(configs)
    except ImportError as e:
        print(f"Error importing train_approach_no_register: {e}")
        print("Please ensure run_imputation_comparison.py is in the same directory")
        raise


class ROSEStyleTrainer:
    """
    FIXED: Complete ROSE-style trainer with robust error handling
    """

    def __init__(self, model, configs):
        self.model = model.to(device)
        self.configs = configs
        self.criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()

        # Separate optimizers for main model and register system
        self.main_optimizer = torch.optim.Adam(
            [p for n, p in model.named_parameters() if "register_system" not in n],
            lr=configs.learning_rate,
        )
        self.register_optimizer = torch.optim.Adam(
            model.register_system.parameters(),
            lr=configs.learning_rate * 1.5,
        )

        # Loss weights following ROSE paper
        self.reconstruction_weight = getattr(configs, "reconstruction_weight", 1.0)
        self.register_weight = getattr(configs, "register_weight", 0.1)
        self.imputation_weight = getattr(configs, "imputation_weight", 1.0)
        self.prediction_weight = getattr(configs, "prediction_weight", 1.0)

    def train_epoch_frequency_pretrain(self, train_datasets):
        """
        FIXED: Phase 1 training with robust error handling
        """
        self.model.train()
        epoch_losses = []
        epoch_metrics = {
            "reconstruction_loss": [],
            "register_loss": [],
            "total_loss": [],
        }

        print("  Domain-wise pre-training:")

        for domain_name, domain_dataset in train_datasets.items():
            print(f"    Processing domain: {domain_name}")

            # Create domain-specific imputation dataset
            try:
                domain_imputation = ImputationDataset(
                    domain_dataset,
                    missing_rate=self.configs.missing_rate,
                    missing_pattern=self.configs.missing_pattern,
                )

                # Standard DataLoader
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
            domain_register_losses = []
            processed_batches = 0

            # Process batches for this domain
            for batch_idx, batch_data in enumerate(domain_loader):
                try:
                    x_missing, x_mark, mask, x_target = batch_data

                    # Move to device
                    x_missing = x_missing.to(device)
                    x_mark = x_mark.to(device)
                    mask = mask.to(device)
                    x_target = x_target.to(device)

                    # Skip batches with all NaN or no missing values
                    if torch.isnan(x_missing).all() or mask.sum() == 0:
                        continue

                    # Zero gradients
                    self.main_optimizer.zero_grad()
                    self.register_optimizer.zero_grad()

                    # Forward pass for frequency pre-training
                    _, _, register_loss, reconstruction_loss = self.model(
                        x_missing,
                        x_mark,
                        mask,
                        domain_name,
                        training_phase="frequency_pretrain",
                    )

                    # Check for valid losses
                    if torch.isnan(reconstruction_loss) or torch.isnan(register_loss):
                        print(f"    NaN loss detected in batch {batch_idx}")
                        continue

                    # ROSE pre-training loss (Equation 12)
                    total_loss = (
                        self.reconstruction_weight * reconstruction_loss
                        + self.register_weight * register_loss
                    )

                    # Check for valid total loss
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        continue

                    # Backward pass
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                    # Update parameters
                    self.main_optimizer.step()
                    self.register_optimizer.step()

                    # Track metrics
                    domain_losses.append(total_loss.item())
                    domain_recon_losses.append(reconstruction_loss.item())
                    domain_register_losses.append(register_loss.item())
                    processed_batches += 1

                except Exception as e:
                    print(f"    Error in {domain_name} batch {batch_idx}: {e}")
                    continue

                # Break early for testing (limit batches per domain)
                if batch_idx >= getattr(self.configs, "max_batches_per_domain", 1000):
                    break

            # Calculate domain averages
            if domain_losses:
                avg_domain_loss = sum(domain_losses) / len(domain_losses)
                avg_recon_loss = sum(domain_recon_losses) / len(domain_recon_losses)
                avg_register_loss = sum(domain_register_losses) / len(
                    domain_register_losses
                )

                epoch_losses.append(avg_domain_loss)
                epoch_metrics["reconstruction_loss"].append(avg_recon_loss)
                epoch_metrics["register_loss"].append(avg_register_loss)
                epoch_metrics["total_loss"].append(avg_domain_loss)

                print(
                    f"    {domain_name:12s}: Loss={avg_domain_loss:.6f}, Batches={processed_batches}"
                )
            else:
                print(f"    {domain_name:12s}: No valid batches processed")

        # Return epoch averages
        if epoch_losses:
            return sum(epoch_losses) / len(epoch_losses), epoch_metrics
        else:
            return 0.0, epoch_metrics

    def train_epoch_imputation_finetune(self, train_datasets):
        """
        FIXED: Phase 2 training with robust error handling
        """
        self.model.train()
        epoch_losses = []
        epoch_metrics = {
            "imputation_loss": [],
            "register_loss": [],
            "mae_loss": [],
            "total_loss": [],
        }

        print("  Domain-wise fine-tuning:")

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
            domain_register_losses = []
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
                    self.register_optimizer.zero_grad()

                    # Forward pass for imputation
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
                        # If no missing values, use small loss on all values
                        imputation_loss = self.criterion(dec_out, x_target) * 0.1
                        mae_loss = self.mae_criterion(dec_out, x_target) * 0.1

                    # Check for valid losses
                    if (
                        torch.isnan(imputation_loss)
                        or torch.isnan(mae_loss)
                        or torch.isnan(register_loss)
                    ):
                        continue

                    # Fine-tuning loss
                    total_loss = (
                        self.imputation_weight * imputation_loss
                        + 0.1
                        * self.register_weight
                        * register_loss  # Reduced register weight
                    )

                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        continue

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                    self.main_optimizer.step()
                    self.register_optimizer.step()

                    domain_losses.append(total_loss.item())
                    domain_imputation_losses.append(imputation_loss.item())
                    domain_mae_losses.append(mae_loss.item())
                    domain_register_losses.append(register_loss.item())
                    processed_batches += 1

                except Exception as e:
                    print(f"    Error in {domain_name} batch {batch_idx}: {e}")
                    continue

                # Break early for testing
                if batch_idx >= getattr(self.configs, "max_batches_per_domain", 1000):
                    break

            if domain_losses:
                avg_domain_loss = sum(domain_losses) / len(domain_losses)
                avg_imputation_loss = sum(domain_imputation_losses) / len(
                    domain_imputation_losses
                )
                avg_mae_loss = sum(domain_mae_losses) / len(domain_mae_losses)
                avg_register_loss = sum(domain_register_losses) / len(
                    domain_register_losses
                )

                epoch_losses.append(avg_domain_loss)
                epoch_metrics["imputation_loss"].append(avg_imputation_loss)
                epoch_metrics["mae_loss"].append(avg_mae_loss)
                epoch_metrics["register_loss"].append(avg_register_loss)
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
        """
        FIXED: Evaluate each domain separately with robust error handling
        """
        self.model.eval()
        results = {}

        print("  Domain-wise evaluation:")

        with torch.no_grad():
            for domain_name, test_dataset in test_datasets.items():
                print(f"    Evaluating {domain_name}...")

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

                        # Forward pass for imputation
                        dec_out, reconstructed, _ = self.model(
                            x_missing,
                            x_mark,
                            mask,
                            domain_name,
                            training_phase="imputation_finetune",
                        )

                        # Evaluate only on missing values
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

                    # Limit evaluation batches for testing
                    if batch_idx >= getattr(
                        self.configs, "max_eval_batches_per_domain", 1000
                    ):
                        break

                if processed_batches > 0:
                    avg_mse = total_mse / processed_batches
                    avg_mae = total_mae / processed_batches
                    results[domain_name] = (avg_mse, avg_mae)
                    print(
                        f"    {domain_name:12s}: MSE={avg_mse:.6f}, MAE={avg_mae:.6f}, Batches={processed_batches}"
                    )
                else:
                    results[domain_name] = (float("inf"), float("inf"))
                    print(f"    {domain_name:12s}: No valid batches")

        return results

    def save_checkpoint(self, epoch, loss, filepath, metrics=None):
        """Save comprehensive checkpoint with error handling"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "main_optimizer_state_dict": self.main_optimizer.state_dict(),
                "register_optimizer_state_dict": self.register_optimizer.state_dict(),
                "loss": loss,
                "metrics": metrics,
                "config": self.configs.__dict__
                if hasattr(self.configs, "__dict__")
                else {},
            }

            torch.save(checkpoint, filepath)
            return True
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            return False

    def load_checkpoint(self, filepath):
        """Load checkpoint with error handling"""
        try:
            checkpoint = torch.load(filepath, map_location=device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.main_optimizer.load_state_dict(checkpoint["main_optimizer_state_dict"])
            self.register_optimizer.load_state_dict(
                checkpoint["register_optimizer_state_dict"]
            )

            return (
                checkpoint["epoch"],
                checkpoint["loss"],
                checkpoint.get("metrics", {}),
            )
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return 0, float("inf"), {}


class ImputationFocusedTrainer:
    """
    Focused trainer for imputation tasks (no frequency pretraining)
    """

    def __init__(self, model, configs):
        self.model = model.to(device)
        self.configs = configs
        self.criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()

        # Single optimizer for imputation task
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=configs.learning_rate,
            weight_decay=1e-5,  # Add weight decay for regularization
        )

        # Scheduler for better convergence
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.7, patience=5
        )

    def train_epoch_imputation_only(self, train_datasets):
        """
        Train ONLY on imputation task - FIXED VERSION with better NaN handling
        """
        self.model.train()
        epoch_losses = []

        print("  Direct imputation training:")

        for domain_name, domain_dataset in train_datasets.items():
            if len(domain_dataset) == 0:
                print(f"    {domain_name:12s}: Empty dataset, skipping")
                continue

            try:
                domain_loader = DataLoader(
                    domain_dataset,
                    batch_size=self.configs.batch_size,
                    shuffle=True,
                    num_workers=self.configs.num_workers,
                    drop_last=True,
                )
            except Exception as e:
                print(f"    Error creating dataloader for {domain_name}: {e}")
                continue

            domain_losses = []
            processed_batches = 0
            max_batches = getattr(self.configs, "max_batches_per_domain", 1000)

            for batch_idx, batch_data in enumerate(domain_loader):
                if batch_idx >= max_batches:
                    break

                try:
                    x_missing, x_mark, mask, x_target = batch_data

                    # Move to device and ensure float32
                    x_missing = x_missing.to(device).float()
                    x_mark = x_mark.to(device).float()
                    mask = mask.to(device)
                    x_target = x_target.to(device).float()

                    # CRITICAL: Skip batches with all NaN or no valid data
                    if torch.isnan(x_missing).all() or torch.isnan(x_target).all():
                        continue

                    # Check if mask has any missing values to impute
                    missing_mask = (~mask).bool()
                    if not missing_mask.any():
                        continue  # Skip if no missing values

                    self.optimizer.zero_grad()

                    # Forward pass with error handling
                    try:
                        imputation_out, reconstructed, _ = self.model(
                            x_missing, x_mark, mask, domain_name
                        )
                    except Exception as forward_error:
                        print(
                            f"      Forward pass error in {domain_name}: {forward_error}"
                        )
                        continue

                    # CRITICAL: Check for NaN in model output
                    if torch.isnan(imputation_out).any():
                        print(f"      NaN detected in model output for {domain_name}")
                        continue

                    # Loss computation with better NaN handling
                    try:
                        # Primary loss: missing value reconstruction
                        if missing_mask.any():
                            # Only compute loss on valid (non-NaN) missing positions
                            missing_values = x_target[missing_mask]
                            imputed_values = imputation_out[missing_mask]

                            # Remove any remaining NaN pairs
                            valid_pairs = ~(
                                torch.isnan(missing_values)
                                | torch.isnan(imputed_values)
                            )

                            if valid_pairs.any():
                                valid_missing = missing_values[valid_pairs]
                                valid_imputed = imputed_values[valid_pairs]

                                primary_loss = self.criterion(
                                    valid_imputed, valid_missing
                                )
                            else:
                                continue  # Skip this batch if no valid pairs
                        else:
                            # Fallback: full reconstruction loss
                            valid_full = ~(
                                torch.isnan(x_target) | torch.isnan(imputation_out)
                            )
                            if valid_full.any():
                                primary_loss = self.criterion(
                                    imputation_out[valid_full], x_target[valid_full]
                                )
                            else:
                                continue

                        # Secondary loss: observed value consistency (optional)
                        observed_mask = mask.bool()
                        if observed_mask.any():
                            observed_values = x_target[observed_mask]
                            observed_imputed = imputation_out[observed_mask]

                            valid_observed = ~(
                                torch.isnan(observed_values)
                                | torch.isnan(observed_imputed)
                            )
                            if valid_observed.any():
                                consistency_loss = (
                                    self.criterion(
                                        observed_imputed[valid_observed],
                                        observed_values[valid_observed],
                                    )
                                    * 0.1
                                )  # 10% weight
                            else:
                                consistency_loss = torch.tensor(0.0, device=device)
                        else:
                            consistency_loss = torch.tensor(0.0, device=device)

                        # Total loss
                        total_loss = primary_loss + consistency_loss

                        # CRITICAL: Check for NaN/inf loss before backward pass
                        if torch.isnan(total_loss) or torch.isinf(total_loss):
                            print(f"      Invalid loss detected: {total_loss.item()}")
                            continue

                        # Backward pass
                        total_loss.backward()

                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0
                        )

                        self.optimizer.step()

                        domain_losses.append(total_loss.item())
                        processed_batches += 1

                    except Exception as loss_error:
                        print(
                            f"      Loss computation error in {domain_name}: {loss_error}"
                        )
                        continue

                except Exception as batch_error:
                    print(
                        f"    Error in {domain_name} batch {batch_idx}: {batch_error}"
                    )
                    continue

            if domain_losses:
                avg_domain_loss = sum(domain_losses) / len(domain_losses)
                epoch_losses.append(avg_domain_loss)
                print(
                    f"    {domain_name:12s}: Loss={avg_domain_loss:.6f}, Batches={processed_batches}"
                )
            else:
                print(f"    {domain_name:12s}: No valid batches processed")

        if epoch_losses:
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)

            # Update learning rate scheduler
            try:
                self.scheduler.step(avg_epoch_loss)
            except:
                pass  # Ignore scheduler errors

            return avg_epoch_loss
        else:
            return float("inf")

    def evaluate_domain_separate(self, test_datasets):
        """Robust domain-wise evaluation for imputation, avoids NaN losses."""
        self.model.eval()
        results = {}

        print("  Domain-wise evaluation:")

        with torch.no_grad():
            for domain_name, test_dataset in test_datasets.items():
                try:
                    test_imputation = ImputationDataset(
                        test_dataset,
                        missing_rate=getattr(self.configs, "test_missing_rate", 0.1),
                        missing_pattern=getattr(
                            self.configs, "test_missing_pattern", "random"
                        ),
                    )

                    test_loader = DataLoader(
                        test_imputation,
                        batch_size=getattr(self.configs, "batch_size", 32),
                        shuffle=False,
                        num_workers=getattr(self.configs, "num_workers", 0),
                    )

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

                            imputation_out, reconstructed, _ = self.model(
                                x_missing, x_mark, mask, domain_name
                            )

                            # Replace NaNs in target to prevent NaN loss
                            x_target_clean = torch.where(
                                torch.isnan(x_target), imputation_out, x_target
                            )

                            # Missing mask fallback
                            missing_mask = (~mask).bool()
                            if missing_mask.sum() == 0:
                                missing_mask = torch.ones_like(mask, dtype=torch.bool)

                            mse = F.mse_loss(
                                imputation_out[missing_mask],
                                x_target_clean[missing_mask],
                            )
                            mae = F.l1_loss(
                                imputation_out[missing_mask],
                                x_target_clean[missing_mask],
                            )

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
                        print(f"    {domain_name:12s}: No valid batches processed")

                except Exception as e:
                    print(f"    Error creating test dataset for {domain_name}: {e}")
                    results[domain_name] = (float("inf"), float("inf"))

        return results
