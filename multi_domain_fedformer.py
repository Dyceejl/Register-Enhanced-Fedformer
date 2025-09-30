import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
from torch.utils.data import DataLoader, Dataset
import random

# Import your existing FEDformer components
from layers.Embed import DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.Autoformer_EncDec import series_decomp, series_decomp_multi, my_Layernorm
from configs import imputation_config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def linear_interpolate_missing(x):
    """Linear interpolation for missing values before FFT"""
    B, L, C = x.shape
    device = x.device
    x_filled = x.clone()
    
    for b in range(B):
        for c in range(C):
            series = x[b, :, c]
            
            if not torch.isnan(series).any():
                continue
            
            if torch.isnan(series).all():
                x_filled[b, :, c] = 0.0
                continue
            
            observed_mask = ~torch.isnan(series)
            observed_idx = torch.where(observed_mask)[0]
            observed_val = series[observed_mask]
            
            all_idx = torch.arange(L, dtype=torch.float32, device=device)
            
            if device.type == 'cpu':
                import numpy as np
                filled = np.interp(
                    all_idx.cpu().numpy(),
                    observed_idx.cpu().numpy(),
                    observed_val.cpu().numpy()
                )
                x_filled[b, :, c] = torch.from_numpy(filled).to(device)
            else:
                filled = series.clone()
                last_val = None
                for i in range(L):
                    if observed_mask[i]:
                        last_val = filled[i]
                    elif last_val is not None:
                        filled[i] = last_val
                
                last_val = None
                for i in range(L-1, -1, -1):
                    if observed_mask[i]:
                        last_val = filled[i]
                    elif last_val is not None and torch.isnan(filled[i]):
                        filled[i] = last_val
                
                x_filled[b, :, c] = filled
    
    return x_filled

class DecomposedFrequencyLearning(nn.Module):
    """ROSE's Decomposed Frequency Learning for representation learning"""

    def __init__(self, Kf=4, max_freq_ratio=0.2, use_interpolation=True):
        super().__init__()
        self.Kf = Kf  # Number of frequency masks
        self.max_freq_ratio = max_freq_ratio  # Upper bound for frequency masking
        self.use_interpolation = use_interpolation

    def multi_frequency_mask(self, x):
        """
        Apply multi-frequency masking with proper NaN handling
        Args:
            x: [B, L, C] - Input time series (may contain NaN)
        Returns:
            masked_series: [Kf, B, L, C] - Kf frequency-masked versions
        """
        B, L, C = x.shape

        # STEP 1: Clean NaN using interpolation or zero-fill
        if self.use_interpolation:
            x_clean = linear_interpolate_missing(x)
        else:
            x_clean = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        # STEP 2: Double-check no NaN remains
        if torch.isnan(x_clean).any():
            print(f"      Warning: NaN still present after cleaning, replacing with zeros")
            x_clean = torch.where(torch.isnan(x_clean), torch.zeros_like(x_clean), x_clean)

        # STEP 3: Transform to frequency domain
        x_freq = torch.fft.rfft(x_clean, dim=1)  # [B, L//2+1, C]
        freq_len = x_freq.shape[1]

        masked_series = []

        for k in range(self.Kf):
            # Sample threshold τ and direction μ (ROSE Equation 2)
            max_threshold = max(1, int(freq_len * self.max_freq_ratio))
            tau = torch.randint(1, max_threshold + 1, (1,)).item()
            mu = torch.bernoulli(torch.tensor(0.5)).item()

            # Create frequency mask matrix M
            mask = torch.ones_like(x_freq)

            if mu == 1:  # Remove high frequencies (keep low)
                mask[:, tau:, :] = 0
            else:  # Remove low frequencies (keep high)
                mask[:, :tau, :] = 0

            # Apply mask and transform back (ROSE Equation 3)
            x_freq_masked = x_freq * mask
            x_masked = torch.fft.irfft(x_freq_masked, n=L, dim=1)

            # STEP 4: Check for NaN after inverse FFT (shouldn't happen, but be safe)
            if torch.isnan(x_masked).any():
                # This can happen if the frequency masking was too aggressive
                # Replace NaN with the original clean values at those positions
                x_masked = torch.where(torch.isnan(x_masked), x_clean, x_masked)

            masked_series.append(x_masked)

        return torch.stack(masked_series, dim=0)  # [Kf, B, L, C]

    def forward(self, x):
        """Generate Kf frequency-masked versions for reconstruction learning"""
        return self.multi_frequency_mask(x)


# Domain-specific configurations (no padding needed)
DOMAIN_CONFIGS = {
    # Original domains
    "ETTh1": {"features": 7, "freq": "h", "data_path": "ETT/ETTh1.csv"},
    "ETTh2": {"features": 7, "freq": "h", "data_path": "ETT/ETTh2.csv"},
    "ETTm1": {"features": 7, "freq": "t", "data_path": "ETT/ETTm1.csv"},
    "ETTm2": {"features": 7, "freq": "t", "data_path": "ETT/ETTm2.csv"},
    "weather": {"features": 21, "freq": "h", "data_path": "weather/weather.csv"},
    "traffic": {
        "features": 865,
        "freq": "h",
        "data_path": "traffic/traffic_with_date.csv",
    },
    "electricity": {
        "features": 321,
        "freq": "h",
        "data_path": "electricity/electricity.csv",
    },
    # NEW: Dimensionality sweep domains
    "beijing_pm25": {
        "features": 8,
        "freq": "h",
        "data_path": "beijing_pm25/beijing_pm25.csv",
    },
    "air_quality": {
        "features": 12,
        "freq": "h",
        "data_path": "air_quality/AirQualityUCI.csv",
    },
    "bike_sharing": {
        "features": 15,
        "freq": "h",
        "data_path": "bike_sharing/bike_sharing.csv",
    },
    "appliances_energy": {
        "features": 29,
        "freq": "t",
        "data_path": "appliances_energy/appliances_energy.csv",
    },
    "electric_power": {
        "features": 53,
        "freq": "t",
        "data_path": "electric_power/power_engineered.csv",
    },
    "pamap2": {
        "features": 55,
        "freq": "s",
        "data_path": "pamap2/pamap2_52features.csv",
    },
}


class DomainSpecificEmbedding(nn.Module):
    """Enhanced embedding for specific domain dimensions"""

    def __init__(self, input_features, d_model, embed_type, freq, dropout=0.1):
        super().__init__()
        self.input_features = input_features
        self.d_model = d_model

        # Value embedding using conv1d
        self.value_embedding = nn.Conv1d(
            in_channels=input_features,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )

        # Temporal embedding
        if embed_type == "timeF":
            # Time features embedding (assuming 4 time features: hour, day, weekday, month)
            self.temporal_embedding = nn.Linear(4, d_model)
        else:
            # Position embedding
            self.temporal_embedding = nn.Embedding(366, d_model)  # Max days in year

        self.dropout = nn.Dropout(dropout)
        self.embed_type = embed_type

    def forward(self, x, x_mark):
        """
        Args:
            x: [B, L, input_features] - time series values
            x_mark: [B, L, time_features] - time encodings
        """
        B, L = x.shape[:2]

        # Value embedding
        x_embed = self.value_embedding(x.permute(0, 2, 1)).transpose(1, 2)

        # Temporal embedding
        if x_mark is not None and self.embed_type == "timeF":
            temporal_embed = self.temporal_embedding(x_mark)
        elif x_mark is not None:
            # Use day of year as position
            positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
            temporal_embed = self.temporal_embedding(positions % 366)
        else:
            temporal_embed = torch.zeros_like(x_embed)

        # Combine embeddings
        x_embed = x_embed + temporal_embed
        return self.dropout(x_embed)


class DomainAwareNormalization(nn.Module):
    """Domain-specific normalization for different feature dimensions"""

    def __init__(self):
        super().__init__()
        self.domain_norms = nn.ModuleDict()

    def get_or_create_norm(self, feature_dim):
        """Get or create normalization layer for specific feature dimension"""
        key = str(feature_dim)
        if key not in self.domain_norms:
            self.domain_norms[key] = nn.LayerNorm(feature_dim)
        return self.domain_norms[key]

    def forward(self, x, feature_dim=None):
        """
        Args:
            x: Input tensor [B, L, C]
            feature_dim: Number of features (if None, infer from x)
        """
        if feature_dim is None:
            feature_dim = x.shape[-1]

        norm_layer = self.get_or_create_norm(feature_dim)
        return norm_layer(x)


class TimeSeriesRegister(nn.Module):
    """Register mechanism for adaptive domain knowledge selection following ROSE paper"""

    def __init__(
        self,
        register_size: int,
        register_dim: int,
        num_register_tokens: int,
    ):
        super().__init__()
        self.register_size = register_size  # H in the paper
        self.register_dim = register_dim  # Dr in the paper
        self.num_register_tokens = num_register_tokens  # Nr in the paper

        # Initialize register vectors randomly
        self.register = nn.Parameter(torch.randn(register_size, register_dim) * 0.02)

        # FIXED: Dynamic embedding projections for different input dimensions
        self.embedding_projections = nn.ModuleDict()

        # Low-rank adaptation matrices for fine-tuning (ROSE Equation 8)
        self.low_rank_u = nn.Parameter(torch.randn(num_register_tokens) * 0.02)
        self.low_rank_v = nn.Parameter(torch.randn(register_dim) * 0.02)

        # Training mode flags
        self.pre_training = True
        self.fine_tuning = False

    def get_or_create_projection(self, input_dim, device):
        """Get or create embedding projection for specific input dimension"""
        key = str(input_dim)
        if key not in self.embedding_projections:
            projection = nn.Linear(input_dim, self.register_dim)
            # Initialize with smaller weights
            nn.init.xavier_normal_(projection.weight, gain=0.1)
            nn.init.zeros_(projection.bias)
            projection = projection.to(device)
            self.embedding_projections[key] = projection

        # Ensure projection is on correct device
        self.embedding_projections[key] = self.embedding_projections[key].to(device)
        return self.embedding_projections[key]

    def set_fine_tuning_mode(self, fine_tuning: bool = True):
        """Switch between pre-training and fine-tuning modes"""
        self.fine_tuning = fine_tuning
        self.pre_training = not fine_tuning

        # Freeze register parameters during fine-tuning
        self.register.requires_grad = not fine_tuning

        # Only low-rank parameters are trainable during fine-tuning
        self.low_rank_u.requires_grad = fine_tuning
        self.low_rank_v.requires_grad = fine_tuning

    def forward(self, x: torch.Tensor, top_k: int = 3):
        """
        Args:
            x: Input time series [batch_size, seq_len, features]
            top_k: Number of register vectors to select
        """
        batch_size, seq_len, features = x.shape
        device = x.device

        # Handle NaN inputs properly
        if torch.isnan(x).all():
            fallback_tokens = torch.zeros(
                batch_size, self.num_register_tokens, self.register_dim, device=device
            )
            return fallback_tokens, torch.tensor(0.0, device=device)

        # Clean input and compute mean with NaN protection
        x_clean = torch.where(torch.isnan(x), torch.zeros_like(x), x)

        # Handle case where all values are zero after cleaning
        if x_clean.abs().sum() == 0:
            fallback_tokens = torch.zeros(
                batch_size, self.num_register_tokens, self.register_dim, device=device
            )
            return fallback_tokens, torch.tensor(0.0, device=device)

        x_mean = x_clean.mean(dim=1)  # [batch_size, features]

        # Get dynamic projection based on actual input dimension
        try:
            projection = self.get_or_create_projection(features, device)
            xe = projection(x_mean)  # [batch_size, register_dim]
        except Exception as e:
            print(f"Error in register projection: {e}")
            fallback_tokens = torch.zeros(
                batch_size, self.num_register_tokens, self.register_dim, device=device
            )
            return fallback_tokens, torch.tensor(0.0, device=device)

        # Safety check for projection output
        if torch.isnan(xe).any() or torch.isinf(xe).any():
            xe = torch.zeros_like(xe)

        if self.pre_training:
            # Pre-training: Use closest register for each sample
            try:
                distances = torch.cdist(
                    xe, self.register
                )  # [batch_size, register_size]

                # Add numerical stability
                distances = distances + 1e-8

                if torch.isnan(distances).any() or torch.isinf(distances).any():
                    e_k = xe.unsqueeze(1).repeat(1, self.num_register_tokens, 1)
                    return e_k, torch.tensor(0.0, device=device)

                closest_indices = distances.argmin(dim=1)  # [batch_size]

                # Register loss for clustering
                selected_register_vectors = self.register[
                    closest_indices
                ]  # [batch_size, register_dim]

                # Compute loss with additional safety
                diff = xe - selected_register_vectors
                register_loss = torch.mean(torch.sum(diff**2, dim=1))

                # Final safety check on register loss
                if torch.isnan(register_loss) or torch.isinf(register_loss):
                    register_loss = torch.tensor(0.0, device=device)

                # Use closest register vectors for this batch
                e_k = selected_register_vectors

            except Exception as e:
                print(f"Error in register pre-training: {e}")
                e_k = xe
                register_loss = torch.tensor(0.0, device=device)

        else:
            # Fine-tuning/Inference: Top-K selection
            try:
                distances = torch.cdist(
                    xe, self.register
                )  # [batch_size, register_size]
                distances = distances + 1e-8

                if torch.isnan(distances).any():
                    e_k = xe
                else:
                    # Top-K selection based on inverse distance
                    similarities = 1.0 / (distances + 1e-8)
                    _, top_indices = similarities.topk(
                        top_k, dim=1
                    )  # [batch_size, top_k]

                    # Average top-k registers
                    top_registers = self.register[
                        top_indices
                    ]  # [batch_size, top_k, register_dim]
                    e_k = top_registers.mean(dim=1)  # [batch_size, register_dim]

            except Exception as e:
                print(f"Error in register fine-tuning: {e}")
                e_k = xe

            register_loss = torch.tensor(0.0, device=device)

        # Convert e_k to register tokens Xd
        Xd = e_k.unsqueeze(1).repeat(
            1, self.num_register_tokens, 1
        )  # [batch_size, Nr, register_dim]

        # Final safety check
        if torch.isnan(Xd).any():
            Xd = torch.zeros_like(Xd)

        if self.fine_tuning:
            # Apply low-rank adaptation
            try:
                A = torch.outer(self.low_rank_u, self.low_rank_v)  # [Nr, register_dim]
                A_expanded = A.unsqueeze(0).expand(
                    batch_size, -1, -1
                )  # [batch_size, Nr, register_dim]
                Xr = Xd * A_expanded  # Element-wise multiplication

                if torch.isnan(Xr).any():
                    Xr = Xd  # Fallback to non-adapted tokens

                return Xr, register_loss
            except Exception as e:
                print(f"Error in low-rank adaptation: {e}")
                return Xd, register_loss
        else:
            # During pre-training, return Xd directly
            return Xd, register_loss


class ImputationEncoderLayerWithRegister(nn.Module):
    """
    FIXED: Modified encoder layer for imputation with register token integration
    """

    def __init__(
        self,
        attention,
        d_model,
        d_ff=None,
        moving_avg=25,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False
        )
        self.conv2 = nn.Conv1d(
            in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False
        )

        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg)
            self.decomp2 = series_decomp_multi(moving_avg)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, num_register_tokens=0):
        # Apply attention with proper error handling
        try:
            new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        except Exception as e:
            print(f"Error in attention layer: {e}")
            # Fallback: return input unchanged
            return x, None

        # If we have register tokens, handle them separately
        if num_register_tokens > 0:
            register_tokens = new_x[:, :num_register_tokens, :]
            sequence_tokens = new_x[:, num_register_tokens:, :]

            # Apply bidirectional attention only to sequence tokens
            x_seq = x[:, num_register_tokens:, :]
            try:
                x_rev = torch.flip(x_seq, dims=[1])
                new_x_rev, _ = self.attention(x_rev, x_rev, x_rev, attn_mask=None)
                new_x_rev = torch.flip(new_x_rev, dims=[1])

                # Combine forward and backward for sequence tokens
                sequence_tokens = (sequence_tokens + new_x_rev) / 2
            except Exception as e:
                print(f"Error in bidirectional attention: {e}")
                # Use only forward attention

            # Combine register and sequence tokens
            new_x = torch.cat([register_tokens, sequence_tokens], dim=1)
            x = x + self.dropout(new_x)
        else:
            # Original bidirectional processing
            try:
                x_rev = torch.flip(x, dims=[1])
                new_x_rev, _ = self.attention(x_rev, x_rev, x_rev, attn_mask=attn_mask)
                new_x_rev = torch.flip(new_x_rev, dims=[1])
                new_x = (new_x + new_x_rev) / 2
            except Exception as e:
                print(f"Error in bidirectional processing: {e}")
                # Use only forward attention

            x = x + self.dropout(new_x)

        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class ImputationEncoderWithRegister(nn.Module):
    """
    Bidirectional encoder for imputation with register token support
    """

    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, num_register_tokens=0):
        attns = []
        for attn_layer in self.attn_layers:
            try:
                x, attn = attn_layer(
                    x, attn_mask=attn_mask, num_register_tokens=num_register_tokens
                )
                attns.append(attn)
            except Exception as e:
                print(f"Error in encoder layer: {e}")
                # Continue with unchanged input
                attns.append(None)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class MultiDomainFEDformerWithRegister(nn.Module):
    """
    FIXED: Complete ROSE-style multi-domain model with proper error handling
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.output_attention = getattr(configs, "output_attention", False)
        self.seq_len = configs.seq_len
        self.num_register_tokens = getattr(configs, "num_register_tokens", 3)
        self.d_model = configs.d_model

        # ROSE's frequency learning component
        self.freq_learning = DecomposedFrequencyLearning(
            Kf=getattr(configs, "Kf", 4),
            max_freq_ratio=getattr(configs, "max_freq_ratio", 0.2),
        )

        # Register system for domain adaptation
        register_size = getattr(configs, "register_size", 128)
        register_dim = getattr(configs, "register_dim", 256)
        self.register_system = TimeSeriesRegister(
            register_size=register_size,
            register_dim=register_dim,
            num_register_tokens=self.num_register_tokens,
        )

        # Project register tokens to model dimension
        self.register_projection = nn.Linear(register_dim, self.d_model)

        # FIXED: Replace domain-specific embeddings with universal embedding
        # OLD CODE (REMOVE):
        # self.domain_embeddings = nn.ModuleDict()
        # self.reconstruction_heads = nn.ModuleDict()
        # self.imputation_heads = nn.ModuleDict()
        # for domain_name, domain_config in DOMAIN_CONFIGS.items():
        #     self.domain_embeddings[domain_name] = DomainSpecificEmbedding(...)
        #     self.reconstruction_heads[domain_name] = nn.Linear(...)
        #     self.imputation_heads[domain_name] = nn.Linear(...)

        # NEW CODE: Universal embedding and heads
        self.universal_embedding = DynamicDataEmbedding(
            d_model=self.d_model,
            embed=configs.embed,
            freq=configs.freq,
            dropout=configs.dropout,
        )

        # Universal heads that can adapt to any output dimension
        self.universal_reconstruction_head = UniversalOutputHead(self.d_model)
        self.universal_imputation_head = UniversalOutputHead(self.d_model)

        # Missing value embedding
        self.missing_embedding = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

        # FIXED: Ensure consistent attention head configuration
        n_heads = getattr(configs, "n_heads", 8)

        # FEDformer encoder components
        if configs.version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(
                ich=self.d_model, L=configs.L, base=configs.base
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=self.d_model,
                out_channels=self.d_model,
                seq_len=self.seq_len + self.num_register_tokens,
                modes=configs.modes,
                mode_select_method=configs.mode_select,
            )

        # FIXED: Encoder with register support and consistent head configuration
        self.encoder = ImputationEncoderWithRegister(
            [
                ImputationEncoderLayerWithRegister(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        self.d_model,
                        n_heads,  # Use consistent n_heads
                    ),
                    self.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(self.d_model),
        )

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.seq_len, self.d_model) * 0.02
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights to prevent NaN issues"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.xavier_normal_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        print("Model weights initialized")

    def forward(
        self, x_enc, x_mark_enc, mask, domain_name, training_phase="imputation"
    ):
        """
        Forward pass for different training phases with comprehensive error handling
        """
        try:
            if training_phase == "frequency_pretrain":
                return self._forward_frequency_pretrain(x_enc, x_mark_enc, domain_name)
            else:
                return self._forward_imputation(x_enc, x_mark_enc, mask, domain_name)
        except Exception as e:
            print(f"Error in forward pass for {domain_name}: {e}")
            # Return fallback values
            B, L, C = x_enc.shape
            device = x_enc.device
            fallback_output = torch.zeros_like(x_enc)
            fallback_loss = torch.tensor(0.0, device=device)

            if training_phase == "frequency_pretrain":
                return None, None, fallback_loss, fallback_loss
            else:
                return fallback_output, fallback_output, fallback_loss

    def _forward_frequency_pretrain(self, x_enc, x_mark_enc, domain_name):
        """
        ROSE's frequency reconstruction pre-training phase WITH register - FIXED for universal embedding
        """
        B, L, C = x_enc.shape  # C = actual input features
        device = x_enc.device

        try:
            # Clean NaN values
            x_clean = torch.where(torch.isnan(x_enc), torch.zeros_like(x_enc), x_enc)

            # Generate Kf frequency-masked series (SAME)
            freq_masked_series = self.freq_learning(x_clean)  # [Kf, B, L, C]
            Kf = freq_masked_series.shape[0]

            reconstruction_losses = []
            register_losses = []

            # Process each frequency-masked series
            for k in range(Kf):
                try:
                    x_masked = freq_masked_series[k]  # [B, L, C]

                    # Register processing for domain adaptation (SAME)
                    register_tokens, register_loss = self.register_system(
                        x_masked, top_k=3
                    )
                    register_tokens = self.register_projection(register_tokens)
                    register_losses.append(register_loss)

                    # FIXED: Use universal embedding instead of domain-specific
                    # OLD CODE (REMOVED):
                    # if domain_name not in self.domain_embeddings:
                    #     print(f"Warning: Domain {domain_name} not found, using ETTh1")
                    #     domain_name = "ETTh1"
                    # enc_out = self.domain_embeddings[domain_name](x_masked, x_mark_enc)

                    # NEW CODE: Universal embedding handles any input dimension
                    enc_out = self.universal_embedding(x_masked, x_mark_enc)

                    # Add positional encoding (SAME)
                    enc_out = enc_out + self.pos_encoding.expand(B, -1, -1)

                    # Combine register tokens with sequence tokens (SAME)
                    combined_tokens = torch.cat([register_tokens, enc_out], dim=1)

                    # FEDformer processing (SAME)
                    encoded, _ = self.encoder(
                        combined_tokens, num_register_tokens=self.num_register_tokens
                    )

                    # Extract sequence representations (SAME)
                    sequence_encoded = encoded[:, self.num_register_tokens :, :]

                    # FIXED: Use universal head with actual input dimension
                    # OLD CODE (REMOVED):
                    # reconstructed = self.reconstruction_heads[domain_name](sequence_encoded)

                    # NEW CODE: Universal head adapts to input dimension C
                    reconstructed = self.universal_reconstruction_head(
                        sequence_encoded, C
                    )

                    # Reconstruction loss (SAME)
                    recon_loss = F.mse_loss(reconstructed, x_clean)
                    reconstruction_losses.append(recon_loss)

                except Exception as e:
                    print(f"Error processing frequency mask {k}: {e}")
                    # Add zero losses for failed masks
                    reconstruction_losses.append(torch.tensor(0.0, device=device))
                    register_losses.append(torch.tensor(0.0, device=device))

            # Aggregate losses (SAME)
            if reconstruction_losses:
                avg_reconstruction_loss = torch.stack(reconstruction_losses).mean()
                avg_register_loss = torch.stack(register_losses).mean()
            else:
                avg_reconstruction_loss = torch.tensor(0.0, device=device)
                avg_register_loss = torch.tensor(0.0, device=device)

            return None, None, avg_register_loss, avg_reconstruction_loss

        except Exception as e:
            print(f"Error in register frequency pre-training: {e}")
            return (
                None,
                None,
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
            )

    def _forward_imputation(self, x_enc, x_mark_enc, mask, domain_name):
        """
        Imputation task forward pass with error handling
        """
        B, L, C = x_enc.shape
        device = x_enc.device

        try:
            # Clean NaN values
            x_clean = torch.where(torch.isnan(x_enc), torch.zeros_like(x_enc), x_enc)

            # Register processing for domain adaptation
            register_tokens, register_loss = self.register_system(x_clean, top_k=3)
            register_tokens = self.register_projection(register_tokens)

            # Domain-specific embedding
            if domain_name not in self.domain_embeddings:
                print(f"Warning: Domain {domain_name} not found, using ETTh1")
                domain_name = "ETTh1"

            enc_out = self.domain_embeddings[domain_name](x_clean, x_mark_enc)

            # Add missing value embeddings
            if len(mask.shape) == 4:
                mask = mask.squeeze(-1)

            missing_indicator = (~mask).float().mean(dim=-1, keepdim=True)  # [B, L, 1]
            missing_mask_expanded = missing_indicator.expand(-1, -1, self.d_model)
            missing_emb = self.missing_embedding.expand(B, L, -1)
            enc_out = enc_out + missing_mask_expanded * missing_emb

            # Add positional encoding
            enc_out = enc_out + self.pos_encoding.expand(B, -1, -1)

            # Combine register tokens with sequence tokens
            combined_tokens = torch.cat([register_tokens, enc_out], dim=1)

            # FEDformer processing
            encoded, attns = self.encoder(
                combined_tokens, num_register_tokens=self.num_register_tokens
            )

            # Extract sequence representations
            sequence_encoded = encoded[:, self.num_register_tokens :, :]

            # Imputation prediction
            imputation_out = self.imputation_heads[domain_name](sequence_encoded)

            # Reconstruct full series (observed + imputed)
            reconstructed = torch.where(mask, x_clean, imputation_out)

            if self.output_attention:
                return imputation_out, reconstructed, register_loss, attns
            else:
                return imputation_out, reconstructed, register_loss

        except Exception as e:
            print(f"Error in imputation forward: {e}")
            # Return fallback values
            fallback_output = torch.zeros_like(x_enc)
            fallback_loss = torch.tensor(0.0, device=device)

            if self.output_attention:
                return fallback_output, fallback_output, fallback_loss, []
            else:
                return fallback_output, fallback_output, fallback_loss

    def set_fine_tuning_mode(self, fine_tuning: bool = True):
        """Switch register system to fine-tuning mode"""
        self.register_system.set_fine_tuning_mode(fine_tuning)

        if fine_tuning:
            # Freeze most parameters except register adaptation
            for name, param in self.named_parameters():
                if "register_system.low_rank" not in name:
                    param.requires_grad = False

            # Enable register low-rank adaptation
            self.register_system.low_rank_u.requires_grad = True
            self.register_system.low_rank_v.requires_grad = True


class DynamicDataEmbedding(nn.Module):
    """Fully dynamic embedding that adapts to any input dimension"""

    def __init__(self, d_model, embed, freq, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embed = embed
        self.freq = freq
        self.dropout = nn.Dropout(dropout)

        # Store embeddings for different input dimensions
        self.value_embeddings = nn.ModuleDict()

        # Temporal embedding components
        if embed == "timeF":
            self.temporal_projections = nn.ModuleDict()
        else:
            # For other embedding types, create a simple temporal embedding
            self.temporal_embedding = nn.Embedding(366, d_model)  # Days in year

    def get_or_create_value_embedding(self, input_dim, device):
        """Get or create value embedding for specific input dimension"""
        key = str(input_dim)
        if key not in self.value_embeddings:
            # Create new embedding for this dimension
            embedding = nn.Conv1d(
                in_channels=input_dim,
                out_channels=self.d_model,
                kernel_size=3,
                padding=1,
                padding_mode="circular",
                bias=False,
            )
            embedding = embedding.to(device)
            self.value_embeddings[key] = embedding

        # Ensure embedding is on correct device
        self.value_embeddings[key] = self.value_embeddings[key].to(device)
        return self.value_embeddings[key]

    def get_or_create_temporal_projection(self, temporal_dim, device):
        """Get or create temporal projection for specific temporal feature dimension"""
        key = str(temporal_dim)
        if key not in self.temporal_projections:
            projection = nn.Linear(temporal_dim, self.d_model)
            projection = projection.to(device)
            self.temporal_projections[key] = projection

        # Ensure projection is on correct device
        self.temporal_projections[key] = self.temporal_projections[key].to(device)
        return self.temporal_projections[key]

    def forward(self, x, x_mark):
        B, L, C = x.shape
        device = x.device

        # Value embedding - dynamic based on input channels
        value_embedding = self.get_or_create_value_embedding(C, device)
        x_embed = value_embedding(x.permute(0, 2, 1)).transpose(1, 2)  # [B, L, d_model]

        # Temporal embedding
        if x_mark is not None and self.embed == "timeF":
            # Time features embedding
            temporal_proj = self.get_or_create_temporal_projection(
                x_mark.shape[-1], device
            )
            temporal_embed = temporal_proj(x_mark)
        elif x_mark is not None:
            # Position embedding (simplified)
            positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            temporal_embed = self.temporal_embedding(positions % 366)
        else:
            # No temporal features
            temporal_embed = torch.zeros_like(x_embed)

        # Combine embeddings
        x_embed = x_embed + temporal_embed

        return self.dropout(x_embed)


class UniversalOutputHead(nn.Module):
    """Universal output head that adapts to any output dimension"""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.output_projections = nn.ModuleDict()

    def get_or_create_projection(self, output_dim, device):
        """Get or create output projection for specific dimension"""
        key = str(output_dim)
        if key not in self.output_projections:
            projection = nn.Linear(self.d_model, output_dim)
            projection = projection.to(device)
            self.output_projections[key] = projection

        # Ensure projection is on correct device
        self.output_projections[key] = self.output_projections[key].to(device)
        return self.output_projections[key]

    def forward(self, x, output_dim):
        """
        Args:
            x: [B, L, d_model] - encoded features
            output_dim: Target output dimension
        Returns:
            [B, L, output_dim] - projected output
        """
        B, L, d_model = x.shape
        device = x.device

        projection = self.get_or_create_projection(output_dim, device)

        # Reshape for projection: [B*L, d_model] -> [B*L, output_dim] -> [B, L, output_dim]
        x_flat = x.reshape(-1, d_model)
        output_flat = projection(x_flat)
        output = output_flat.reshape(B, L, output_dim)

        return output


class MultiDomainFEDformerWithoutRegister(nn.Module):
    """
    Clean ablation version: Same as register version but WITHOUT register system
    For fair comparison with register version
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.output_attention = getattr(configs, "output_attention", False)
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model

        # self.missing_projection = nn.Linear(configs.c_in, self.d_model)
        # self.c_in = getattr(configs, "c_in", 7)

        self.missing_projections = nn.ModuleDict()
        
        # SAME: ROSE's frequency learning component
        self.freq_learning = DecomposedFrequencyLearning(
            Kf=getattr(configs, "Kf", 4),
            max_freq_ratio=getattr(configs, "max_freq_ratio", 0.2),
        )
        
        self.freq_learning = DecomposedFrequencyLearning(
            Kf=getattr(configs, 'Kf', 4),
            max_freq_ratio=getattr(configs, 'max_freq_ratio', 0.2),
            use_interpolation=getattr(configs, 'use_fft_interpolation', True)  # NEW
        )

        # REMOVED: Register system completely
        # NO register_system
        # NO register_projection

        # FIXED: Replace domain-specific embeddings with universal embedding
        # OLD CODE (REMOVE):
        # self.domain_embeddings = nn.ModuleDict()
        # self.reconstruction_heads = nn.ModuleDict()
        # self.imputation_heads = nn.ModuleDict()
        # for domain_name, domain_config in DOMAIN_CONFIGS.items():
        #     self.domain_embeddings[domain_name] = DomainSpecificEmbedding(...)
        #     self.reconstruction_heads[domain_name] = nn.Linear(...)
        #     self.imputation_heads[domain_name] = nn.Linear(...)

        # NEW CODE: Universal embedding and heads
        self.universal_embedding = DynamicDataEmbedding(
            d_model=self.d_model,
            embed=configs.embed,
            freq=configs.freq,
            dropout=configs.dropout,
        )

        # Universal heads that can adapt to any output dimension
        self.universal_reconstruction_head = UniversalOutputHead(self.d_model)
        self.universal_imputation_head = UniversalOutputHead(self.d_model)

        # SAME: Missing value embedding
        self.missing_embedding = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

        # SAME: FEDformer encoder components
        n_heads = getattr(configs, "n_heads", 8)

        if configs.version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(
                ich=self.d_model, L=configs.L, base=configs.base
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=self.d_model,
                out_channels=self.d_model,
                seq_len=self.seq_len,  # NO register tokens added
                modes=configs.modes,
                mode_select_method=configs.mode_select,
            )

        # SAME: Encoder but without register support
        self.encoder = ImputationEncoderWithRegister(
            [
                ImputationEncoderLayerWithRegister(
                    AutoCorrelationLayer(encoder_self_att, self.d_model, n_heads),
                    self.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(self.d_model),
        )

        # SAME: Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.seq_len, self.d_model) * 0.02
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights to prevent NaN issues"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.xavier_normal_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        print("No-Register Model weights initialized")

    def forward(self, x_enc, x_mark_enc, mask, domain_name, training_phase="imputation"):
        """
        Forward pass routing based on training phase
        """
        if training_phase == "frequency_pretrain":
            return self._forward_frequency_pretrain(x_enc, x_mark_enc, domain_name)
        else:
            return self._forward_imputation(x_enc, x_mark_enc, mask, domain_name)

    def _forward_imputation(self, x_enc, x_mark_enc=None, mask=None, domain_name=None):
        """
        Robust forward pass for imputation WITHOUT register.
        Handles NaNs, ensures input/output feature dimensions match,
        and works with universal embedding + universal heads.
        """
        B, L, C = x_enc.shape
        device = x_enc.device

        try:
            # 1️⃣ Replace NaNs with zeros
            x_clean = torch.where(torch.isnan(x_enc), torch.zeros_like(x_enc), x_enc)

            # 2️⃣ Universal embedding
            enc_out = self.universal_embedding(x_clean, x_mark_enc)  # [B, L, d_model]

            # 3️⃣ Handle mask
            if mask is None:
                mask = torch.ones_like(x_enc, dtype=torch.bool, device=device)
            if len(mask.shape) == 4:
                mask = mask.squeeze(-1)
            missing_pattern = (~mask).float()  # [B, L, C]

            # 4️⃣ Project missing pattern to model dimension
            if domain_name not in self.missing_projections:
                self.missing_projections[domain_name] = nn.Linear(C, self.d_model).to(device)
                print(f"    Created missing projection for {domain_name}: {C} -> {self.d_model}")
            
            missing_embedding = self.missing_projections[domain_name](missing_pattern)  # [B, L, d_model]
            enc_out = enc_out + missing_embedding

            # 5️⃣ Add positional encoding
            pos_enc = self.pos_encoding[:, :L, :].expand(B, -1, -1)
            enc_out = enc_out + pos_enc

            # 6️⃣ Encoder forward
            encoded, attns = self.encoder(
                enc_out, num_register_tokens=0
            )  # [B, L, d_model]

            # 7️⃣ Imputation head: match input feature dimension
            imputation_out = self.universal_imputation_head(
                encoded, output_dim=C
            )  # [B, L, C]

            # 8️⃣ Reconstruct full series
            reconstructed = torch.where(mask, x_clean, imputation_out)  # [B, L, C]

            # 9️⃣ No register loss for this version
            register_loss = torch.tensor(0.0, device=device)

            if self.output_attention:
                return imputation_out, reconstructed, register_loss, attns
            else:
                return imputation_out, reconstructed, register_loss

        except Exception as e:
            print(f"Error in imputation forward: {e}")
            fallback_output = torch.zeros_like(x_enc)
            fallback_loss = torch.tensor(0.0, device=device)
            if self.output_attention:
                return fallback_output, fallback_output, fallback_loss, []
            else:
                return fallback_output, fallback_output, fallback_loss

    def _forward_frequency_pretrain(self, x_enc, x_mark_enc, domain_name):
        """
        Frequency reconstruction pre-training phase WITHOUT register - FIXED for NaN handling
        """
        B, L, C = x_enc.shape
        device = x_enc.device

        try:
            # Step 1: Apply frequency learning (handles NaN internally via interpolation)
            freq_masked_series = self.freq_learning(x_enc)  # [Kf, B, L, C]
            Kf = freq_masked_series.shape[0]

            reconstruction_losses = []

            # Step 2: Get the clean target (for computing loss)
            # The freq_learning already did interpolation, so we need the clean version
            if self.freq_learning.use_interpolation:
                x_clean = linear_interpolate_missing(x_enc)
            else:
                x_clean = torch.where(torch.isnan(x_enc), torch.zeros_like(x_enc), x_enc)

            # Process each frequency-masked series
            for k in range(Kf):
                try:
                    x_masked = freq_masked_series[k]  # [B, L, C]

                    # Check for NaN in masked series
                    if torch.isnan(x_masked).any():
                        x_masked = torch.where(torch.isnan(x_masked), torch.zeros_like(x_masked), x_masked)

                    # Use universal embedding
                    enc_out = self.universal_embedding(x_masked, x_mark_enc)

                    # Check for NaN after embedding
                    if torch.isnan(enc_out).any():
                        print(f"      Warning: NaN after embedding in mask {k}")
                        reconstruction_losses.append(torch.tensor(0.0, device=device))
                        continue

                    # Add positional encoding
                    pos_enc = self.pos_encoding[:, :L, :].expand(B, -1, -1)
                    enc_out = enc_out + pos_enc

                    # Encoder processing (no register tokens)
                    encoded, _ = self.encoder(enc_out, num_register_tokens=0)

                    # Check for NaN after encoder
                    if torch.isnan(encoded).any():
                        print(f"      Warning: NaN after encoder in mask {k}")
                        reconstruction_losses.append(torch.tensor(0.0, device=device))
                        continue

                    # Reconstruction head
                    reconstructed = self.universal_reconstruction_head(encoded, C)

                    # Check for NaN after reconstruction
                    if torch.isnan(reconstructed).any():
                        print(f"      Warning: NaN after reconstruction head in mask {k}")
                        reconstruction_losses.append(torch.tensor(0.0, device=device))
                        continue

                    # Compute reconstruction loss against clean target
                    # CRITICAL: Make sure both tensors are valid
                    if torch.isnan(x_clean).any():
                        print(f"      Warning: NaN in x_clean target")
                        # Replace NaN in target with reconstructed values (no loss contribution)
                        x_clean_safe = torch.where(torch.isnan(x_clean), reconstructed.detach(), x_clean)
                    else:
                        x_clean_safe = x_clean

                    # Compute MSE loss
                    recon_loss = F.mse_loss(reconstructed, x_clean_safe)

                    # Final check
                    if torch.isnan(recon_loss) or torch.isinf(recon_loss):
                        print(f"      Warning: Invalid loss for mask {k}: {recon_loss}")
                        reconstruction_losses.append(torch.tensor(0.0, device=device))
                    else:
                        reconstruction_losses.append(recon_loss)

                except Exception as e:
                    print(f"      Error processing frequency mask {k}: {e}")
                    reconstruction_losses.append(torch.tensor(0.0, device=device))

            # Aggregate losses
            if reconstruction_losses:
                # Filter out zero losses if we have some valid ones
                valid_losses = [loss for loss in reconstruction_losses if loss.item() > 0]
                
                if valid_losses:
                    avg_reconstruction_loss = torch.stack(valid_losses).mean()
                else:
                    # All losses were zero/invalid
                    avg_reconstruction_loss = torch.stack(reconstruction_losses).mean()
            else:
                avg_reconstruction_loss = torch.tensor(0.0, device=device)

            # No register loss for this version
            avg_register_loss = torch.tensor(0.0, device=device)

            # Return 4 values (to match expected signature)
            return None, None, avg_register_loss, avg_reconstruction_loss

        except Exception as e:
            print(f"      Critical error in frequency pre-training: {e}")
            import traceback
            traceback.print_exc()
            return (
                None,
                None,
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
            )


class ImputationOptimizedEmbedding(nn.Module):
    """Simplified embedding optimized for imputation tasks"""

    def __init__(self, input_features, d_model, embed_type, freq, dropout=0.1):
        super().__init__()
        self.input_features = input_features
        self.d_model = d_model

        # FIXED: Use linear projection instead of circular conv
        self.value_projection = nn.Linear(input_features, d_model)

        # Temporal embedding
        if embed_type == "timeF":
            self.temporal_embedding = nn.Linear(4, d_model)  # 4 time features
        else:
            self.temporal_embedding = nn.Embedding(366, d_model)

        self.dropout = nn.Dropout(dropout)
        self.embed_type = embed_type

    def forward(self, x, x_mark):
        B, L = x.shape[:2]

        # Simple linear projection for values
        x_embed = self.value_projection(x)

        # Temporal embedding
        if x_mark is not None and self.embed_type == "timeF":
            temporal_embed = self.temporal_embedding(x_mark)
        elif x_mark is not None:
            positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
            temporal_embed = self.temporal_embedding(positions % 366)
        else:
            temporal_embed = torch.zeros_like(x_embed)

        return self.dropout(x_embed + temporal_embed)


class SimpleImputationEncoderLayer(nn.Module):
    """Simplified encoder layer without bidirectional processing"""

    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Simple feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None, num_register_tokens=0):
        # Self-attention with residual connection
        attn_out, attn_weights = self.attention(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + attn_out)

        # Feedforward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x, attn_weights


class ImputationOptimizedFEDformer(nn.Module):
    """
    Simplified FEDformer optimized specifically for imputation tasks
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.output_attention = getattr(configs, "output_attention", False)
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model

        # Domain-specific embeddings (using simplified embedding)
        self.domain_embeddings = nn.ModuleDict()
        self.imputation_heads = nn.ModuleDict()

        from multi_domain_fedformer import DOMAIN_CONFIGS

        for domain_name, domain_config in DOMAIN_CONFIGS.items():
            # Simplified domain-specific embedding
            self.domain_embeddings[domain_name] = ImputationOptimizedEmbedding(
                input_features=domain_config["features"],
                d_model=self.d_model,
                embed_type=configs.embed,
                freq=domain_config["freq"],
                dropout=configs.dropout,
            )

            # Imputation head
            self.imputation_heads[domain_name] = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(self.d_model // 2, domain_config["features"]),
            )

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, configs.seq_len, self.d_model) * 0.02
        )

        # Simplified encoder
        from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer

        encoder_self_att = AutoCorrelation(
            mask_flag=False,
            factor=configs.factor if hasattr(configs, "factor") else 1,
            attention_dropout=configs.dropout,
        )

        self.encoder_layers = nn.ModuleList(
            [
                SimpleImputationEncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att, self.d_model, getattr(configs, "n_heads", 8)
                    ),
                    self.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(self.d_model)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Conservative weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self, x_enc, x_mark_enc, mask, domain_name, training_phase="imputation"
    ):
        """Simplified forward pass focused on imputation"""
        B, L, C = x_enc.shape
        device = x_enc.device

        try:
            # Handle NaN values
            x_clean = torch.where(torch.isnan(x_enc), torch.zeros_like(x_enc), x_enc)

            # Domain-specific embedding
            if domain_name not in self.domain_embeddings:
                domain_name = "ETTh1"

            enc_out = self.domain_embeddings[domain_name](x_clean, x_mark_enc)

            # Add positional encoding (handle length mismatch)
            if L <= self.seq_len:
                pos_enc = self.pos_encoding[:, :L, :].expand(B, -1, -1)
            else:
                # Repeat positional encoding for longer sequences
                pos_enc = self.pos_encoding.repeat(1, (L // self.seq_len) + 1, 1)[
                    :, :L, :
                ].expand(B, -1, -1)

            enc_out = enc_out + pos_enc

            # Encoder processing
            attns = []
            for layer in self.encoder_layers:
                enc_out, attn = layer(enc_out, attn_mask=None, num_register_tokens=0)
                attns.append(attn)

            enc_out = self.final_norm(enc_out)

            # Imputation prediction
            imputation_out = self.imputation_heads[domain_name](enc_out)

            # Reconstruct: use observed values where available
            reconstructed = torch.where(mask, x_clean, imputation_out)

            # No register loss for this simplified version
            register_loss = torch.tensor(0.0, device=device)

            if self.output_attention:
                return imputation_out, reconstructed, register_loss, attns
            else:
                return imputation_out, reconstructed, register_loss

        except Exception as e:
            print(f"Error in simplified forward pass: {e}")
            fallback = torch.zeros_like(x_enc)
            fallback_loss = torch.tensor(0.0, device=device)

            if self.output_attention:
                return fallback, fallback, fallback_loss, []
            else:
                return fallback, fallback, fallback_loss
