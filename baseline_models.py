import torch
import torch.nn as nn
import numpy as np
from pypots.imputation import BRITS, SAITS
import os
import json
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PyPOTSDatasetAdapter:
    def __init__(self, base_dataset, missing_rate=0.15, missing_pattern="random"):
        self.base_dataset = base_dataset
        self.missing_rate = missing_rate
        self.missing_pattern = missing_pattern

    def __len__(self):
        return len(self.base_dataset)

    def convert_to_pypots_format(self, data_dict):
        X_list = []
        for i in range(len(self.base_dataset)):
            seq_x, seq_y, seq_x_mark, seq_y_mark = self.base_dataset[i]
            if torch.is_tensor(seq_x):
                seq_x = seq_x.detach().cpu().numpy()
            X_list.append(seq_x)
        X = np.array(X_list)
        if hasattr(self, "apply_missing"):
            X = self.apply_missing(X)
        return {"X": X}


class InputAdapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        B, L, C = x.shape
        x = x.reshape(-1, C)
        x = self.linear(x)
        return x.reshape(B, L, -1)


class BRITSWrapper:
    def __init__(self, configs):
        self.configs = configs
        self.model = None
        
        from sklearn.preprocessing import MinMaxScaler  
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        self.fitted = False
        self.fixed_dim = getattr(configs, "fixed_input_dim", None)
        self.input_adapters = {}

    def _get_or_create_adapter(self, in_dim):
        if self.fixed_dim is None:
            return None
        if in_dim not in self.input_adapters:
            adapter = InputAdapter(in_dim, self.fixed_dim).to(device)
            self.input_adapters[in_dim] = adapter
            print(f"    Created new input adapter: {in_dim} -> {self.fixed_dim}")
        return self.input_adapters[in_dim]

    def fit(self, train_datasets):
        print("Training BRITS model...")
        all_train_data = []
        for domain_name, dataset in train_datasets.items():
            print(f"  Processing {domain_name}...")
            for i in range(min(len(dataset), 1000)):
                seq_x, _, _, _ = dataset[i]
                if torch.is_tensor(seq_x):
                    seq_x = seq_x.detach().cpu().numpy()
                all_train_data.append(seq_x)
        if not all_train_data:
            raise ValueError("No training data found")
        X_train = np.array(all_train_data)
        print(f"  Training data shape: {X_train.shape}")
        original_shape = X_train.shape
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_train_flat = self.scaler.fit_transform(X_train_flat)
        X_train = X_train_flat.reshape(original_shape)
        n_steps, n_features = X_train.shape[1], X_train.shape[2]
        adapter = self._get_or_create_adapter(n_features)
        if adapter is not None and n_features != self.fixed_dim:
            with torch.no_grad():
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
                X_train = adapter(X_train_tensor).detach().cpu().numpy()
            n_features = self.fixed_dim
        self.model = BRITS(
            n_steps=n_steps,
            n_features=n_features,
            rnn_hidden_size=getattr(self.configs, "rnn_hidden_size", 64),
            epochs=min(getattr(self.configs, "train_epochs", 50), 50),
            patience=getattr(self.configs, "patience", 10),
            batch_size=getattr(self.configs, "batch_size", 32),
            device=device,
        )
        X_train_missing = self._create_missing_data(X_train, self.configs.missing_rate)
        pypots_data = {"X": X_train_missing}
        self.model.fit(pypots_data)
        self.fitted = True
        print("BRITS training completed!")

    def evaluate_domain(self, domain_name, test_dataset):
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        print(
            f"  Evaluating {self.__class__.__name__.replace('Wrapper', '')} on {domain_name}..."
        )

        test_data = []
        original_data = []

        for i in range(min(len(test_dataset), 500)):
            seq_x, _, _, _ = test_dataset[i]
            if torch.is_tensor(seq_x):
                seq_x = seq_x.detach().cpu().numpy()
            original_data.append(seq_x.copy())
            test_data.append(seq_x)

        if not test_data:
            return float("inf"), float("inf")

        X_test = np.array(test_data)
        X_original = np.array(original_data)

        # Normalize test data
        original_shape = X_test.shape
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])

        try:
            X_test_flat = self.scaler.transform(X_test_flat)
        except Exception:
            self.scaler = StandardScaler().fit(X_test_flat)
            X_test_flat = self.scaler.transform(X_test_flat)

        X_test = X_test_flat.reshape(original_shape)

        # Apply same normalization to original
        X_orig_flat = X_original.reshape(-1, X_original.shape[-1])
        X_orig_flat = self.scaler.transform(X_orig_flat)
        X_original_norm = X_orig_flat.reshape(original_shape)

        # Handle input adapter if needed
        adapter = self._get_or_create_adapter(X_test.shape[-1])
        if adapter is not None:
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                X_test = adapter(X_test_tensor).detach().cpu().numpy()

                X_original_tensor = torch.tensor(
                    X_original_norm, dtype=torch.float32
                ).to(device)
                X_original_norm = adapter(X_original_tensor).detach().cpu().numpy()

        # Create missing mask for evaluation
        missing_rate = getattr(self.configs, "test_missing_rate", 0.2)
        missing_mask = np.random.rand(*X_test.shape) < missing_rate
        missing_mask = missing_mask.astype(bool)

        # Apply missing values
        X_test_missing = X_test.copy()
        X_test_missing[missing_mask] = np.nan

        try:
            # Impute using model
            pypots_test_data = {"X": X_test_missing}

            # FIXED: Handle different return formats from PyPOTS
            impute_result = self.model.impute(pypots_test_data)

            print(f"    [DEBUG] Impute result type: {type(impute_result)}")

            # Handle different possible return formats
            if isinstance(impute_result, dict):
                # Dictionary format: {"imputation": array}
                if "imputation" in impute_result:
                    imputed_data = impute_result["imputation"]
                elif "X" in impute_result:
                    imputed_data = impute_result["X"]
                else:
                    # Take the first value if it's a dict
                    imputed_data = list(impute_result.values())[0]
            elif isinstance(impute_result, (list, tuple)):
                # List/tuple format: [imputed_data, ...]
                imputed_data = impute_result[0]
            else:
                # Direct array format
                imputed_data = impute_result

            print(
                f"    [DEBUG] Extracted imputed_data type: {type(imputed_data)}, shape: {getattr(imputed_data, 'shape', 'no shape')}"
            )

            # Convert to numpy if needed
            if torch.is_tensor(imputed_data):
                imputed_data = imputed_data.detach().cpu().numpy()

            # Ensure arrays are numpy arrays
            imputed_data = np.asarray(imputed_data)
            X_original_norm = np.asarray(X_original_norm)

            print(
                f"    [DEBUG] Final shapes - imputed: {imputed_data.shape}, original: {X_original_norm.shape}, mask: {missing_mask.shape}"
            )
            print(
                f"    [DEBUG] Missing ratio: {missing_mask.sum() / missing_mask.size:.3f}"
            )

            # Handle shape mismatches
            if imputed_data.shape != X_original_norm.shape:
                print(f"    [WARNING] Shape mismatch - attempting to fix")

                # Try to match dimensions
                target_shape = X_original_norm.shape

                if imputed_data.ndim == X_original_norm.ndim:
                    # Same number of dimensions, truncate to minimum
                    min_dims = [
                        min(imputed_data.shape[i], target_shape[i])
                        for i in range(len(target_shape))
                    ]

                    if len(min_dims) == 3:
                        imputed_data = imputed_data[
                            : min_dims[0], : min_dims[1], : min_dims[2]
                        ]
                        X_original_norm = X_original_norm[
                            : min_dims[0], : min_dims[1], : min_dims[2]
                        ]
                        missing_mask = missing_mask[
                            : min_dims[0], : min_dims[1], : min_dims[2]
                        ]
                    else:
                        # Flatten and compare
                        min_size = min(
                            imputed_data.size, X_original_norm.size, missing_mask.size
                        )
                        imputed_data = imputed_data.flatten()[:min_size]
                        X_original_norm = X_original_norm.flatten()[:min_size]
                        missing_mask = missing_mask.flatten()[:min_size]
                else:
                    # Different dimensions, try flattening
                    min_size = min(
                        imputed_data.size, X_original_norm.size, missing_mask.size
                    )
                    imputed_data = imputed_data.flatten()[:min_size]
                    X_original_norm = X_original_norm.flatten()[:min_size]
                    missing_mask = missing_mask.flatten()[:min_size]

            # Calculate metrics only on missing positions
            if missing_mask.any():
                try:
                    # Ensure all arrays have the same shape before indexing
                    if (
                        imputed_data.shape
                        == X_original_norm.shape
                        == missing_mask.shape
                    ):
                        imputed_missing = imputed_data[missing_mask]
                        original_missing = X_original_norm[missing_mask]

                        # Check for valid values
                        if len(imputed_missing) > 0 and len(original_missing) > 0:
                            # Remove any remaining NaN values
                            valid_mask = ~(
                                np.isnan(imputed_missing) | np.isnan(original_missing)
                            )

                            if valid_mask.any():
                                imputed_clean = imputed_missing[valid_mask]
                                original_clean = original_missing[valid_mask]

                                mse = np.mean((imputed_clean - original_clean) ** 2)
                                mae = np.mean(np.abs(imputed_clean - original_clean))

                                print(
                                    f"    [SUCCESS] MSE: {mse:.6f}, MAE: {mae:.6f} (on {len(imputed_clean)} valid points)"
                                )
                                return mse, mae
                            else:
                                print(
                                    f"    [WARNING] No valid points after cleaning NaN values"
                                )
                                return 1.0, 1.0
                        else:
                            print(f"    [WARNING] No missing values extracted")
                            return 1.0, 1.0
                    else:
                        print(
                            f"    [ERROR] Shape mismatch persists: {imputed_data.shape} vs {X_original_norm.shape} vs {missing_mask.shape}"
                        )
                        return float("inf"), float("inf")

                except Exception as idx_error:
                    print(f"    [ERROR] Indexing failed: {idx_error}")
                    return float("inf"), float("inf")
            else:
                print(f"    [WARNING] No missing values to evaluate")
                # Use full data comparison as fallback
                try:
                    mse = np.mean((imputed_data - X_original_norm) ** 2)
                    mae = np.mean(np.abs(imputed_data - X_original_norm))
                    print(f"    [FALLBACK] MSE: {mse:.6f}, MAE: {mae:.6f} (full data)")
                    return mse, mae
                except:
                    return 1.0, 1.0

        except Exception as e:
            print(f"    [ERROR] Evaluation failed: {e}")
            import traceback

            traceback.print_exc()
            return float("inf"), float("inf")

    def _create_missing_data(self, data, missing_rate):
        data_missing = data.copy()
        mask = np.random.rand(*data.shape) < missing_rate
        data_missing[mask] = np.nan
        return data_missing

    def _create_missing_mask(self, data, missing_rate):
        return np.random.rand(*data.shape) < missing_rate


class SAITSWrapper:
    def __init__(self, configs):
        self.configs = configs
        self.model = None
        
        from sklearn.preprocessing import MinMaxScaler  
        self.scaler = MinMaxScaler(feature_range=(0, 1))
                                   
        self.fitted = False
        self.fixed_dim = getattr(configs, "fixed_input_dim", None)
        self.input_adapters = {}

    def _get_or_create_adapter(self, in_dim):
        if self.fixed_dim is None:
            return None
        if in_dim not in self.input_adapters:
            adapter = InputAdapter(in_dim, self.fixed_dim).to(device)
            self.input_adapters[in_dim] = adapter
            print(f"    Created new input adapter: {in_dim} -> {self.fixed_dim}")
        return self.input_adapters[in_dim]

    def fit(self, train_datasets):
        print("Training SAITS model...")
        all_train_data = []
        for domain_name, dataset in train_datasets.items():
            print(f"  Processing {domain_name}...")
            for i in range(min(len(dataset), 1000)):
                seq_x, _, _, _ = dataset[i]
                if torch.is_tensor(seq_x):
                    seq_x = seq_x.detach().cpu().numpy()
                all_train_data.append(seq_x)
        if not all_train_data:
            raise ValueError("No training data found")
        X_train = np.array(all_train_data)
        print(f"  Training data shape: {X_train.shape}")
        original_shape = X_train.shape
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        X_train_flat = self.scaler.fit_transform(X_train_flat)
        X_train = X_train_flat.reshape(original_shape)
        n_steps, n_features = X_train.shape[1], X_train.shape[2]
        adapter = self._get_or_create_adapter(n_features)
        if adapter is not None and n_features != self.fixed_dim:
            with torch.no_grad():
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
                X_train = adapter(X_train_tensor).detach().cpu().numpy()
            n_features = self.fixed_dim
        self.model = SAITS(
            n_steps=n_steps,
            n_features=n_features,
            n_layers=getattr(self.configs, "n_layers", 2),
            d_model=getattr(self.configs, "d_model", 128),
            n_heads=getattr(self.configs, "n_heads", 4),
            d_k=getattr(self.configs, "d_k", 32),
            d_v=getattr(self.configs, "d_v", 32),
            d_ffn=getattr(self.configs, "d_ffn", 256),
            epochs=min(getattr(self.configs, "train_epochs", 50), 50),
            patience=getattr(self.configs, "patience", 10),
            batch_size=getattr(self.configs, "batch_size", 32),
            device=device,
        )
        X_train_missing = self._create_missing_data(X_train, self.configs.missing_rate)
        pypots_data = {"X": X_train_missing}
        self.model.fit(pypots_data)
        self.fitted = True
        print("SAITS training completed!")

    def evaluate_domain(self, domain_name, test_dataset):
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        print(
            f"  Evaluating {self.__class__.__name__.replace('Wrapper', '')} on {domain_name}..."
        )

        test_data = []
        original_data = []

        for i in range(min(len(test_dataset), 500)):
            seq_x, _, _, _ = test_dataset[i]
            if torch.is_tensor(seq_x):
                seq_x = seq_x.detach().cpu().numpy()
            original_data.append(seq_x.copy())
            test_data.append(seq_x)

        if not test_data:
            return float("inf"), float("inf")

        X_test = np.array(test_data)
        X_original = np.array(original_data)

        # Normalize test data
        original_shape = X_test.shape
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])

        try:
            X_test_flat = self.scaler.transform(X_test_flat)
        except Exception:
            self.scaler = StandardScaler().fit(X_test_flat)
            X_test_flat = self.scaler.transform(X_test_flat)

        X_test = X_test_flat.reshape(original_shape)

        # Apply same normalization to original
        X_orig_flat = X_original.reshape(-1, X_original.shape[-1])
        X_orig_flat = self.scaler.transform(X_orig_flat)
        X_original_norm = X_orig_flat.reshape(original_shape)

        # Handle input adapter if needed
        adapter = self._get_or_create_adapter(X_test.shape[-1])
        if adapter is not None:
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                X_test = adapter(X_test_tensor).detach().cpu().numpy()

                X_original_tensor = torch.tensor(
                    X_original_norm, dtype=torch.float32
                ).to(device)
                X_original_norm = adapter(X_original_tensor).detach().cpu().numpy()

        # Create missing mask for evaluation
        missing_rate = getattr(self.configs, "test_missing_rate", 0.2)
        missing_mask = np.random.rand(*X_test.shape) < missing_rate
        missing_mask = missing_mask.astype(bool)

        # Apply missing values
        X_test_missing = X_test.copy()
        X_test_missing[missing_mask] = np.nan

        try:
            # Impute using model
            pypots_test_data = {"X": X_test_missing}

            # FIXED: Handle different return formats from PyPOTS
            impute_result = self.model.impute(pypots_test_data)

            print(f"    [DEBUG] Impute result type: {type(impute_result)}")

            # Handle different possible return formats
            if isinstance(impute_result, dict):
                # Dictionary format: {"imputation": array}
                if "imputation" in impute_result:
                    imputed_data = impute_result["imputation"]
                elif "X" in impute_result:
                    imputed_data = impute_result["X"]
                else:
                    # Take the first value if it's a dict
                    imputed_data = list(impute_result.values())[0]
            elif isinstance(impute_result, (list, tuple)):
                # List/tuple format: [imputed_data, ...]
                imputed_data = impute_result[0]
            else:
                # Direct array format
                imputed_data = impute_result

            print(
                f"    [DEBUG] Extracted imputed_data type: {type(imputed_data)}, shape: {getattr(imputed_data, 'shape', 'no shape')}"
            )

            # Convert to numpy if needed
            if torch.is_tensor(imputed_data):
                imputed_data = imputed_data.detach().cpu().numpy()

            # Ensure arrays are numpy arrays
            imputed_data = np.asarray(imputed_data)
            X_original_norm = np.asarray(X_original_norm)

            print(
                f"    [DEBUG] Final shapes - imputed: {imputed_data.shape}, original: {X_original_norm.shape}, mask: {missing_mask.shape}"
            )
            print(
                f"    [DEBUG] Missing ratio: {missing_mask.sum() / missing_mask.size:.3f}"
            )

            # Handle shape mismatches
            if imputed_data.shape != X_original_norm.shape:
                print(f"    [WARNING] Shape mismatch - attempting to fix")

                # Try to match dimensions
                target_shape = X_original_norm.shape

                if imputed_data.ndim == X_original_norm.ndim:
                    # Same number of dimensions, truncate to minimum
                    min_dims = [
                        min(imputed_data.shape[i], target_shape[i])
                        for i in range(len(target_shape))
                    ]

                    if len(min_dims) == 3:
                        imputed_data = imputed_data[
                            : min_dims[0], : min_dims[1], : min_dims[2]
                        ]
                        X_original_norm = X_original_norm[
                            : min_dims[0], : min_dims[1], : min_dims[2]
                        ]
                        missing_mask = missing_mask[
                            : min_dims[0], : min_dims[1], : min_dims[2]
                        ]
                    else:
                        # Flatten and compare
                        min_size = min(
                            imputed_data.size, X_original_norm.size, missing_mask.size
                        )
                        imputed_data = imputed_data.flatten()[:min_size]
                        X_original_norm = X_original_norm.flatten()[:min_size]
                        missing_mask = missing_mask.flatten()[:min_size]
                else:
                    # Different dimensions, try flattening
                    min_size = min(
                        imputed_data.size, X_original_norm.size, missing_mask.size
                    )
                    imputed_data = imputed_data.flatten()[:min_size]
                    X_original_norm = X_original_norm.flatten()[:min_size]
                    missing_mask = missing_mask.flatten()[:min_size]

            # Calculate metrics only on missing positions
            if missing_mask.any():
                try:
                    # Ensure all arrays have the same shape before indexing
                    if (
                        imputed_data.shape
                        == X_original_norm.shape
                        == missing_mask.shape
                    ):
                        imputed_missing = imputed_data[missing_mask]
                        original_missing = X_original_norm[missing_mask]

                        # Check for valid values
                        if len(imputed_missing) > 0 and len(original_missing) > 0:
                            # Remove any remaining NaN values
                            valid_mask = ~(
                                np.isnan(imputed_missing) | np.isnan(original_missing)
                            )

                            if valid_mask.any():
                                imputed_clean = imputed_missing[valid_mask]
                                original_clean = original_missing[valid_mask]

                                mse = np.mean((imputed_clean - original_clean) ** 2)
                                mae = np.mean(np.abs(imputed_clean - original_clean))

                                print(
                                    f"    [SUCCESS] MSE: {mse:.6f}, MAE: {mae:.6f} (on {len(imputed_clean)} valid points)"
                                )
                                return mse, mae
                            else:
                                print(
                                    f"    [WARNING] No valid points after cleaning NaN values"
                                )
                                return 1.0, 1.0
                        else:
                            print(f"    [WARNING] No missing values extracted")
                            return 1.0, 1.0
                    else:
                        print(
                            f"    [ERROR] Shape mismatch persists: {imputed_data.shape} vs {X_original_norm.shape} vs {missing_mask.shape}"
                        )
                        return float("inf"), float("inf")

                except Exception as idx_error:
                    print(f"    [ERROR] Indexing failed: {idx_error}")
                    return float("inf"), float("inf")
            else:
                print(f"    [WARNING] No missing values to evaluate")
                # Use full data comparison as fallback
                try:
                    mse = np.mean((imputed_data - X_original_norm) ** 2)
                    mae = np.mean(np.abs(imputed_data - X_original_norm))
                    print(f"    [FALLBACK] MSE: {mse:.6f}, MAE: {mae:.6f} (full data)")
                    return mse, mae
                except:
                    return 1.0, 1.0

        except Exception as e:
            print(f"    [ERROR] Evaluation failed: {e}")
            import traceback

            traceback.print_exc()
            return float("inf"), float("inf")

    def _create_missing_data(self, data, missing_rate):
        data_missing = data.copy()
        mask = np.random.rand(*data.shape) < missing_rate
        data_missing[mask] = np.nan
        return data_missing

    def _create_missing_mask(self, data, missing_rate):
        return np.random.rand(*data.shape) < missing_rate

    def _apply_missing_mask(self, data, mask, fill_value=np.nan):
        data_with_missing = data.copy()
        data_with_missing[mask] = fill_value
        return data_with_missing


def train_baseline_models(configs, train_datasets, test_datasets):
    """Train and evaluate all baseline models"""

    print("=" * 80)
    print("TRAINING BASELINE MODELS (BRITS & SAITS)")
    print("=" * 80)

    results = {}
    models = {}

    # Train BRITS
    try:
        print("\n1. Training BRITS...")
        brits_model = BRITSWrapper(configs)
        brits_model.fit(train_datasets)
        models["BRITS"] = brits_model

        # Evaluate BRITS
        brits_results = {}
        for domain_name, test_dataset in test_datasets.items():
            mse, mae = brits_model.evaluate_domain(domain_name, test_dataset)
            brits_results[domain_name] = (mse, mae)

        results["BRITS"] = brits_results
        print("BRITS training and evaluation completed!")

    except Exception as e:
        print(f"BRITS training failed: {e}")
        results["BRITS"] = {
            domain: (float("inf"), float("inf")) for domain in test_datasets.keys()
        }

    # Train SAITS
    try:
        print("\n2. Training SAITS...")
        saits_model = SAITSWrapper(configs)
        saits_model.fit(train_datasets)
        models["SAITS"] = saits_model

        # Evaluate SAITS
        saits_results = {}
        for domain_name, test_dataset in test_datasets.items():
            mse, mae = saits_model.evaluate_domain(domain_name, test_dataset)
            saits_results[domain_name] = (mse, mae)

        results["SAITS"] = saits_results
        print("SAITS training and evaluation completed!")

    except Exception as e:
        print(f"SAITS training failed: {e}")
        results["SAITS"] = {
            domain: (float("inf"), float("inf")) for domain in test_datasets.keys()
        }

    return models, results


def compare_all_models(fedformer_results, baseline_results, save_path):
    """Compare FedFormer results with baseline models"""

    print("\n" + "=" * 100)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("=" * 100)

    # Header
    print(
        f"{'Domain':<15} {'FedFormer':<20} {'BRITS':<20} {'SAITS':<20} {'Best Model':<15}"
    )
    print(
        f"{'':15} {'MSE':<9} {'MAE':<9} {'MSE':<9} {'MAE':<9} {'MSE':<9} {'MAE':<9} {'Winner':<15}"
    )
    print("-" * 120)

    comparison_results = {}
    model_wins = {"FedFormer": 0, "BRITS": 0, "SAITS": 0}

    for domain in sorted(fedformer_results.keys()):
        if domain in baseline_results.get(
            "BRITS", {}
        ) and domain in baseline_results.get("SAITS", {}):
            # Get results
            fed_mse, fed_mae = fedformer_results[domain]
            brits_mse, brits_mae = baseline_results["BRITS"][domain]
            saits_mse, saits_mae = baseline_results["SAITS"][domain]

            # Skip failed results
            if any(x == float("inf") for x in [fed_mse, brits_mse, saits_mse]):
                print(
                    f"{domain:<15} {'Failed':<20} {'Failed':<20} {'Failed':<20} {'N/A':<15}"
                )
                continue

            # Find best model (lowest MSE)
            mse_scores = {"FedFormer": fed_mse, "BRITS": brits_mse, "SAITS": saits_mse}
            best_model = min(mse_scores, key=mse_scores.get)
            model_wins[best_model] += 1

            # Print results
            print(
                f"{domain:<15} {fed_mse:<9.6f} {fed_mae:<9.6f} "
                f"{brits_mse:<9.6f} {brits_mae:<9.6f} "
                f"{saits_mse:<9.6f} {saits_mae:<9.6f} "
                f"{best_model:<15}"
            )

            # Store detailed comparison
            comparison_results[domain] = {
                "FedFormer": {"MSE": fed_mse, "MAE": fed_mae},
                "BRITS": {"MSE": brits_mse, "MAE": brits_mae},
                "SAITS": {"MSE": saits_mse, "MAE": saits_mae},
                "best_model": best_model,
            }

    # Summary
    print("-" * 120)
    print(f"SUMMARY:")
    print(f"  FedFormer wins: {model_wins['FedFormer']} domains")
    print(f"  BRITS wins: {model_wins['BRITS']} domains")
    print(f"  SAITS wins: {model_wins['SAITS']} domains")

    total_domains = sum(model_wins.values())
    if total_domains > 0:
        print(f"\nWin rates:")
        for model, wins in model_wins.items():
            print(f"  {model}: {wins / total_domains * 100:.1f}%")

    # Save detailed comparison
    try:
        os.makedirs(save_path, exist_ok=True)
        comparison_file = os.path.join(save_path, "model_comparison.json")

        full_comparison = {
            "timestamp": datetime.now().isoformat(),
            "model_wins": model_wins,
            "detailed_results": comparison_results,
            "summary": {
                "total_domains": total_domains,
                "fedformer_win_rate": model_wins["FedFormer"] / total_domains
                if total_domains > 0
                else 0,
                "brits_win_rate": model_wins["BRITS"] / total_domains
                if total_domains > 0
                else 0,
                "saits_win_rate": model_wins["SAITS"] / total_domains
                if total_domains > 0
                else 0,
            },
        }

        with open(comparison_file, "w") as f:
            json.dump(full_comparison, f, indent=2)

        print(f"\nDetailed comparison saved to: {comparison_file}")

    except Exception as e:
        print(f"Error saving comparison: {e}")

    return comparison_results
