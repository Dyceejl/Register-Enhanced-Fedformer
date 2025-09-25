import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.Autoformer_EncDec import series_decomp, series_decomp_multi, my_Layernorm
import math
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ImputationEncoderLayer(nn.Module):
    """
    Modified encoder layer for imputation with bidirectional processing
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
        super(ImputationEncoderLayer, self).__init__()
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

    def forward(self, x, attn_mask=None):
        # Bidirectional attention (forward and backward)
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)

        # Reverse sequence for backward pass
        x_rev = torch.flip(x, dims=[1])
        new_x_rev, _ = self.attention(x_rev, x_rev, x_rev, attn_mask=attn_mask)
        new_x_rev = torch.flip(new_x_rev, dims=[1])

        # Combine forward and backward
        new_x = (new_x + new_x_rev) / 2

        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class ImputationEncoder(nn.Module):
    """
    Bidirectional encoder for imputation
    """

    def __init__(self, attn_layers, norm_layer=None):
        super(ImputationEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class ImputationDecoder(nn.Module):
    """
    Decoder for reconstruction with missing value focus
    """

    def __init__(self, d_model, c_out, seq_len, moving_avg=25):
        super(ImputationDecoder, self).__init__()

        if isinstance(moving_avg, list):
            self.decomp = series_decomp_multi(moving_avg)
        else:
            self.decomp = series_decomp(moving_avg)

        # Reconstruction layers
        self.seasonal_projection = nn.Linear(d_model, c_out)
        self.trend_projection = nn.Linear(d_model, c_out)

        # Additional refinement layers
        self.refinement = nn.Sequential(
            nn.Conv1d(c_out, c_out * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(c_out * 2, c_out, kernel_size=3, padding=1),
        )

    def forward(self, enc_out, original_data, mask):
        """
        Args:
            enc_out: Encoded representation [B, L, d_model]
            original_data: Original input with missing values [B, L, c_out]
            mask: Binary mask (1 for observed, 0 for missing) [B, L, c_out]
        """
        B, L, _ = enc_out.shape

        # Decompose encoded representation
        seasonal_part, trend_part = self.decomp(enc_out)

        # Project to output dimension
        seasonal_out = self.seasonal_projection(seasonal_part)
        trend_out = self.trend_projection(trend_part)

        # Combine seasonal and trend
        reconstructed = seasonal_out + trend_out

        # Apply refinement
        reconstructed_refined = self.refinement(
            reconstructed.transpose(1, 2)
        ).transpose(1, 2)

        # For observed values, blend with original data
        # For missing values, use pure reconstruction
        output = mask * original_data + (1 - mask) * reconstructed_refined

        return output, reconstructed_refined


class FEDformerImputation(nn.Module):
    """
    FEDformer adapted for time series imputation
    """

    def __init__(self, configs):
        super(FEDformerImputation, self).__init__()
        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.output_attention = getattr(configs, "output_attention", False)

        # Embedding for input with missing values
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        # Missing value embedding
        self.missing_embedding = nn.Parameter(torch.randn(1, 1, configs.d_model))

        # Attention mechanism selection
        if configs.version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, L=configs.L, base=configs.base
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                seq_len=self.seq_len,
                modes=configs.modes,
                mode_select_method=configs.mode_select,
            )

        # Bidirectional encoder
        self.encoder = ImputationEncoder(
            [
                ImputationEncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att, configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
        )

        # Reconstruction decoder
        self.decoder = ImputationDecoder(
            configs.d_model, configs.c_out, self.seq_len, configs.moving_avg
        )

        # Position encoding for missing values
        self.pos_encoding = nn.Parameter(torch.randn(1, self.seq_len, configs.d_model))

    def forward(self, x_enc, x_mark_enc, mask):
        """
        Args:
            x_enc: Input time series [B, L, C] with NaN for missing values
            x_mark_enc: Time features [B, L, time_features]
            mask: Binary mask [B, L, C] (1 for observed, 0 for missing)
        """
        B, L, C = x_enc.shape

        # Handle missing values by replacing NaN with 0
        x_enc_clean = torch.where(torch.isnan(x_enc), torch.zeros_like(x_enc), x_enc)

        # Embedding
        enc_out = self.enc_embedding(x_enc_clean, x_mark_enc)

        # Add missing value embeddings
        # Fix mask dimensions
        if len(mask.shape) == 4:
            mask = mask.squeeze(-1)  # Remove extra dimension if present

        # Create missing mask for embedding - average across features
        missing_indicator = (1 - mask).float().mean(dim=-1, keepdim=True)  # [B, L, 1]
        missing_mask = missing_indicator.expand(
            -1, -1, enc_out.size(-1)
        )  # [B, L, d_model]
        missing_emb = self.missing_embedding.expand(B, L, -1)
        enc_out = enc_out + missing_mask * missing_emb

        # Add positional encoding
        enc_out = enc_out + self.pos_encoding.expand(B, -1, -1)

        # Encode
        enc_out, attns = self.encoder(enc_out)

        # Decode and reconstruct
        dec_out, reconstructed = self.decoder(enc_out, x_enc_clean, mask)

        if self.output_attention:
            return dec_out, reconstructed, attns
        else:
            return dec_out, reconstructed

    def get_seasonal_trend(self, x_enc, x_mark_enc, mask):
        """
        Extract seasonal and trend components from the input time series.
        Args:
            x_enc: Input time series [B, L, C] with NaN for missing values
            x_mark_enc: Time features [B, L, time_features]
            mask: Binary mask [B, L, C] (1 for observed, 0 for missing)
        Returns:
            seasonal_part: Seasonal component [B, L, d_model]
            trend_part: Trend component [B, L, d_model]
        """
        # Embedding
        x_enc_clean = torch.where(torch.isnan(x_enc), torch.zeros_like(x_enc), x_enc)
        enc_out = self.enc_embedding(x_enc_clean, x_mark_enc)

        # Add missing value embeddings
        if len(mask.shape) == 4:
            mask = mask.squeeze(-1)
        missing_indicator = (1 - mask).float().mean(dim=-1, keepdim=True)
        missing_mask = missing_indicator.expand(-1, -1, enc_out.size(-1))
        missing_emb = self.missing_embedding.expand(x_enc.shape[0], x_enc.shape[1], -1)
        enc_out = enc_out + missing_mask * missing_emb
        enc_out = enc_out + self.pos_encoding.expand(x_enc.shape[0], -1, -1)

        # Encode
        enc_out, attns = self.encoder(enc_out)

        # Decompose
        seasonal_part, trend_part = self.decoder.decomp(enc_out)
        return seasonal_part, trend_part
