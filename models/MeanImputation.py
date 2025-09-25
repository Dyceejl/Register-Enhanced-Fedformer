# models/MeanImputation.py
import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    """
    Mean Imputation Model - Simple baseline for time series imputation
    Follows the same interface as FEDformer models for easy comparison
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # Store config parameters
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.output_attention = getattr(configs, 'output_attention', False)

        # Strategy for mean computation
        self.strategy = getattr(configs, 'mean_strategy', 'feature')  # 'global', 'feature', 'window'
        self.window_size = getattr(configs, 'window_size', 24)

        # Buffers to store computed means
        self.register_buffer('feature_means', torch.zeros(self.enc_in))
        self.register_buffer('global_mean', torch.tensor(0.0))
        self.register_buffer('is_fitted', torch.tensor(False))

    def forward(self, x_enc, x_mark_enc, mask=None):
        """
        Forward pass for Mean Imputation

        Args:
            x_enc: Input sequences [B, L, C] with NaN for missing values
            x_mark_enc: Time features [B, L, time_features] (unused)
            mask: Binary mask [B, L, C] (1 for observed, 0 for missing)

        Returns:
            imputed_data: Imputed sequences [B, L, C]
            imputed_data: Same as above (for compatibility)
        """
        B, L, C = x_enc.shape

        # Initialize with input data
        imputed_x = x_enc.clone()

        # Compute means if not fitted
        if not self.is_fitted:
            self._compute_means(x_enc, mask)

        # Apply imputation based on strategy
        if self.strategy == 'global':
            imputed_x = self._global_mean_imputation(imputed_x, mask)
        elif self.strategy == 'feature':
            imputed_x = self._feature_mean_imputation(imputed_x, mask)
        elif self.strategy == 'window':
            imputed_x = self._window_mean_imputation(imputed_x, mask)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Return in same format as FEDformer
        if self.output_attention:
            return imputed_x, imputed_x, None
        else:
            return imputed_x, imputed_x

    def _compute_means(self, x_enc, mask=None):
        """Compute means from observed values"""
        B, L, C = x_enc.shape

        # Handle missing values
        if mask is not None:
            # Use mask (0 = missing, 1 = observed)
            observed_data = x_enc.clone()
            observed_data[mask == 0] = float('nan')
        else:
            observed_data = x_enc

        # Compute feature-wise means
        feature_means = torch.zeros(C, device=x_enc.device)
        all_values = []

        for c in range(C):
            feature_data = observed_data[:, :, c].flatten()
            valid_data = feature_data[~torch.isnan(feature_data)]

            if len(valid_data) > 0:
                feature_means[c] = valid_data.mean()
                all_values.append(valid_data)
            else:
                feature_means[c] = 0.0

        # Compute global mean
        if all_values:
            global_mean = torch.cat(all_values).mean()
        else:
            global_mean = torch.tensor(0.0, device=x_enc.device)

        # Update buffers
        self.feature_means.data = feature_means
        self.global_mean.data = global_mean
        self.is_fitted.data = torch.tensor(True)

    def _global_mean_imputation(self, x_enc, mask):
        """Replace missing values with global mean"""
        imputed = x_enc.clone()

        if mask is not None:
            missing_mask = (mask == 0)
            imputed[missing_mask] = self.global_mean
        else:
            nan_mask = torch.isnan(imputed)
            imputed[nan_mask] = self.global_mean

        return imputed

    def _feature_mean_imputation(self, x_enc, mask):
        """Replace missing values with feature-wise means"""
        imputed = x_enc.clone()

        for c in range(self.enc_in):
            if mask is not None:
                missing_mask = (mask[:, :, c] == 0)
            else:
                missing_mask = torch.isnan(imputed[:, :, c])

            if missing_mask.any():
                imputed[:, :, c][missing_mask] = self.feature_means[c]

        return imputed

    def _window_mean_imputation(self, x_enc, mask):
        """Replace missing values with local window means"""
        imputed = x_enc.clone()
        B, L, C = x_enc.shape

        for b in range(B):
            for c in range(C):
                sequence = imputed[b, :, c]

                if mask is not None:
                    missing_indices = torch.where(mask[b, :, c] == 0)[0]
                else:
                    missing_indices = torch.where(torch.isnan(sequence))[0]

                for idx in missing_indices:
                    # Define window around missing point
                    start = max(0, idx - self.window_size // 2)
                    end = min(L, idx + self.window_size // 2 + 1)

                    # Get window values (excluding the missing point itself)
                    window_values = sequence[start:end]
                    if mask is not None:
                        window_mask = mask[b, start:end, c]
                        window_values = window_values[window_mask == 1]
                    else:
                        window_values = window_values[~torch.isnan(window_values)]

                    # Use window mean or fall back to feature mean
                    if len(window_values) > 0:
                        imputed[b, idx, c] = window_values.mean()
                    else:
                        imputed[b, idx, c] = self.feature_means[c]

        return imputed

    def fit_on_data_loader(self, data_loader):
        """
        Pre-compute means from training data loader
        """
        print("Computing means from training data...")

        all_values = {c: [] for c in range(self.enc_in)}
        global_values = []

        self.eval()
        with torch.no_grad():
            for batch_x, _, batch_x_mark, _ in data_loader:
                batch_x = batch_x.float()

                for c in range(self.enc_in):
                    feature_data = batch_x[:, :, c].flatten()
                    valid_data = feature_data[~torch.isnan(feature_data)]

                    if len(valid_data) > 0:
                        all_values[c].append(valid_data)
                        global_values.append(valid_data)

        # Compute final means
        feature_means = torch.zeros(self.enc_in)
        for c in range(self.enc_in):
            if all_values[c]:
                feature_means[c] = torch.cat(all_values[c]).mean()
            else:
                feature_means[c] = 0.0

        global_mean = torch.cat(global_values).mean() if global_values else torch.tensor(0.0)

        self.feature_means.data = feature_means.to(self.feature_means.device)
        self.global_mean.data = global_mean.to(self.global_mean.device)
        self.is_fitted.data = torch.tensor(True)

        print(f"Computed means: {feature_means.numpy()}")
        print(f"Global mean: {global_mean.item():.4f}")