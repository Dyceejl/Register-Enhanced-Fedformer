# phase2_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.FEDformer_Imputation import FEDformerImputation
from layers.Embed import DataEmbedding_wo_pos
from layers.FourierCorrelation import FourierBlock
from layers.Autoformer_EncDec import series_decomp, series_decomp_multi, my_Layernorm
import os
import math


class DomainFeatureExtractor(nn.Module):
    """Extract domain-specific features from Phase 1 trained models"""

    def __init__(self, trained_fedformer_path, configs):
        super(DomainFeatureExtractor, self).__init__()

        # Load trained Phase 1 model
        self.fedformer = FEDformerImputation(configs)
        checkpoint = torch.load(trained_fedformer_path, map_location="cpu")
        self.fedformer.load_state_dict(checkpoint)

        # Freeze all parameters
        for param in self.fedformer.parameters():
            param.requires_grad = False

        self.fedformer.eval()

    def forward(self, x_enc, x_mark_enc, mask):
        """Extract domain-specific features"""
        with torch.no_grad():
            # Get encoded representation from Phase 1 model
            B, L, C = x_enc.shape

            # Handle missing values
            x_enc_clean = torch.where(
                torch.isnan(x_enc), torch.zeros_like(x_enc), x_enc
            )

            # Embedding
            enc_out = self.fedformer.enc_embedding(x_enc_clean, x_mark_enc)

            # Add missing value embeddings
            if len(mask.shape) == 4:
                mask = mask.squeeze(-1)
            missing_indicator = (1 - mask).float().mean(dim=-1, keepdim=True)
            missing_mask = missing_indicator.expand(-1, -1, enc_out.size(-1))
            missing_emb = self.fedformer.missing_embedding.expand(B, L, -1)
            enc_out = enc_out + missing_mask * missing_emb

            # Add positional encoding
            enc_out = enc_out + self.fedformer.pos_encoding.expand(B, -1, -1)

            # Encode with bidirectional encoder
            enc_out, _ = self.fedformer.encoder(enc_out)

            # Extract domain features (mean pooling over sequence)
            domain_features = enc_out.mean(dim=1)  # [B, d_model]

        return domain_features, enc_out


class TSRegister(nn.Module):
    """Time Series Register for domain-specific knowledge storage"""

    def __init__(self, register_size=128, d_model=512, num_register_tokens=3):
        super(TSRegister, self).__init__()

        self.register_size = register_size  # H
        self.d_model = d_model  # Dr
        self.num_register_tokens = num_register_tokens  # Nr

        # Initialize register
        self.register = nn.Parameter(torch.randn(register_size, d_model))

        # Data projection for clustering
        self.data_projection = nn.Linear(d_model, d_model)

        # Low-rank adaptation for fine-tuning
        self.low_rank_u = nn.Parameter(torch.randn(num_register_tokens))
        self.low_rank_v = nn.Parameter(torch.randn(d_model))

    def get_closest_cluster(self, x_encoded):
        """Find closest cluster center for given encoding"""
        # Project data
        xe = self.data_projection(x_encoded)  # [B, d_model]

        # Calculate distances to all register centers
        distances = torch.cdist(
            xe.unsqueeze(1), self.register.unsqueeze(0)
        )  # [B, 1, H]
        closest_idx = distances.argmin(dim=-1).squeeze(1)  # [B]

        return closest_idx, distances.min(dim=-1)[0].squeeze(1)

    def get_register_tokens(self, x_encoded, mode="pretrain", k=3):
        """Get register tokens based on input encoding"""
        B = x_encoded.shape[0]

        if mode == "pretrain":
            # Use closest cluster center
            closest_idx, distances = self.get_closest_cluster(x_encoded)
            selected_centers = self.register[closest_idx]  # [B, d_model]

            # Replicate to create multiple register tokens
            register_tokens = selected_centers.unsqueeze(1).expand(
                -1, self.num_register_tokens, -1
            )

            return register_tokens, closest_idx, distances

        elif mode == "finetune":
            # Use Top-K selection
            xe = self.data_projection(x_encoded)
            similarities = torch.cosine_similarity(
                xe.unsqueeze(1), self.register.unsqueeze(0), dim=-1
            )  # [B, H]

            top_k_values, top_k_indices = similarities.topk(k, dim=-1)  # [B, k]
            selected_centers = self.register[top_k_indices]  # [B, k, d_model]

            # Apply low-rank adaptation
            adaptation_matrix = torch.outer(
                self.low_rank_u, self.low_rank_v
            )  # [Nr, d_model]
            adapted_centers = selected_centers + adaptation_matrix.unsqueeze(0)

            return adapted_centers, top_k_indices, top_k_values

    def compute_register_loss(self, x_encoded, closest_idx):
        """Compute clustering loss for register training"""
        xe = self.data_projection(x_encoded)
        selected_centers = self.register[closest_idx]

        # L2 loss between data embedding and closest center
        register_loss = torch.norm(xe - selected_centers, p=2, dim=-1).mean()

        return register_loss


class RegisterEnhancedDecoder(nn.Module):
    """Enhanced decoder that uses both original features and register tokens"""

    def __init__(self, d_model, c_out, seq_len, num_register_tokens=3, moving_avg=25):
        super(RegisterEnhancedDecoder, self).__init__()

        self.num_register_tokens = num_register_tokens

        # Decomposition layers
        if isinstance(moving_avg, list):
            self.decomp = series_decomp_multi(moving_avg)
        else:
            self.decomp = series_decomp(moving_avg)

        # Enhanced feature combination
        self.feature_fusion = nn.Linear(
            d_model * 2, d_model
        )  # Combine original + register

        # Reconstruction layers
        self.seasonal_projection = nn.Linear(d_model, c_out)
        self.trend_projection = nn.Linear(d_model, c_out)

        # Refinement layers
        self.refinement = nn.Sequential(
            nn.Conv1d(c_out, c_out * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(c_out * 2, c_out, kernel_size=3, padding=1),
        )

    def forward(self, encoded_features, register_tokens, original_data, mask):
        """
        Args:
            encoded_features: [B, L, d_model] from encoder
            register_tokens: [B, Nr, d_model] from register
            original_data: [B, L, c_out] original input
            mask: [B, L, c_out] binary mask
        """
        B, L, d_model = encoded_features.shape

        # Expand register tokens to sequence length
        expanded_register = register_tokens.mean(dim=1, keepdim=True).expand(
            -1, L, -1
        )  # [B, L, d_model]

        # Combine original features with register knowledge
        combined_features = torch.cat(
            [encoded_features, expanded_register], dim=-1
        )  # [B, L, 2*d_model]
        fused_features = self.feature_fusion(combined_features)  # [B, L, d_model]

        # Decompose into seasonal and trend
        seasonal_part, trend_part = self.decomp(fused_features)

        # Project to output dimension
        seasonal_out = self.seasonal_projection(seasonal_part)
        trend_out = self.trend_projection(trend_part)

        # Combine components
        reconstructed = seasonal_out + trend_out

        # Apply refinement
        reconstructed_refined = self.refinement(
            reconstructed.transpose(1, 2)
        ).transpose(1, 2)

        # Smart blending: keep observed values, impute missing values
        output = mask * original_data + (1 - mask) * reconstructed_refined

        return output, reconstructed_refined


class FEDformerRegisterImputation(nn.Module):
    """Phase 2: Register-Enhanced FEDformer for Multi-Domain Imputation"""

    def __init__(
        self, configs, domain_extractors, register_size=128, num_register_tokens=3
    ):
        super(FEDformerRegisterImputation, self).__init__()

        self.configs = configs
        self.domain_extractors = domain_extractors
        self.num_domains = len(domain_extractors)

        # TS-Register
        self.ts_register = TSRegister(
            register_size=register_size,
            d_model=configs.d_model,
            num_register_tokens=num_register_tokens,
        )

        # Enhanced decoder
        self.decoder = RegisterEnhancedDecoder(
            d_model=configs.d_model,
            c_out=configs.c_out,
            seq_len=configs.seq_len,
            num_register_tokens=num_register_tokens,
            moving_avg=getattr(configs, "moving_avg", [24]),
        )

        # Domain classifier for automatic domain detection
        self.domain_classifier = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model // 2),
            nn.ReLU(),
            nn.Linear(configs.d_model // 2, self.num_domains),
        )

    def forward(self, x_enc, x_mark_enc, mask, domain_id=None, mode="pretrain"):
        """
        Args:
            x_enc: [B, L, C] input time series
            x_mark_enc: [B, L, T] time features
            mask: [B, L, C] binary mask
            domain_id: domain identifier (0, 1, 2, ...)
            mode: 'pretrain', 'finetune', or 'inference'
        """
        B, L, C = x_enc.shape

        # Feature extraction using domain-specific extractors
        if domain_id is not None:
            # Use specific domain extractor
            domain_extractor = self.domain_extractors[domain_id]
            domain_features, encoded_features = domain_extractor(
                x_enc, x_mark_enc, mask
            )
        else:
            # Use all extractors and find best match
            all_features = []
            all_encoded = []

            for extractor in self.domain_extractors.values():
                feat, enc = extractor(x_enc, x_mark_enc, mask)
                all_features.append(feat)
                all_encoded.append(enc)

            # Stack and take mean (could be improved with attention)
            domain_features = torch.stack(all_features).mean(dim=0)
            encoded_features = torch.stack(all_encoded).mean(dim=0)

        # Get register tokens
        if mode == "pretrain":
            register_tokens, closest_idx, distances = (
                self.ts_register.get_register_tokens(domain_features, mode="pretrain")
            )

            # Compute register loss
            register_loss = self.ts_register.compute_register_loss(
                domain_features, closest_idx
            )

        elif mode in ["finetune", "inference"]:
            register_tokens, selected_idx, similarities = (
                self.ts_register.get_register_tokens(
                    domain_features, mode="finetune", k=3
                )
            )
            register_loss = torch.tensor(0.0, device=x_enc.device)

        # Enhanced decoding with register knowledge
        output, reconstructed = self.decoder(
            encoded_features, register_tokens, x_enc, mask
        )

        # Domain classification (auxiliary task)
        domain_pred = self.domain_classifier(domain_features)

        return {
            "imputed": output,
            "reconstructed": reconstructed,
            "register_loss": register_loss,
            "domain_pred": domain_pred,
            "domain_features": domain_features,
        }


class MultiDomainDataset:
    """Dataset handler for multi-domain training"""

    def __init__(self, domain_datasets):
        """
        Args:
            domain_datasets: dict like {
                'energy': [dataset1, dataset2, ...],
                'transport': [dataset3, ...],
                'weather': [dataset4, ...]
            }
        """
        self.domain_datasets = domain_datasets
        self.domain_to_id = {
            domain: idx for idx, domain in enumerate(domain_datasets.keys())
        }
        self.id_to_domain = {idx: domain for domain, idx in self.domain_to_id.items()}

    def get_domain_id(self, dataset_name):
        """Get domain ID from dataset name"""
        for domain, datasets in self.domain_datasets.items():
            if dataset_name in datasets:
                return self.domain_to_id[domain]
        return 0  # Default to first domain

    def get_domain_name(self, domain_id):
        """Get domain name from ID"""
        return self.id_to_domain.get(domain_id, "unknown")


# Domain configuration based on your setup
DOMAIN_CONFIG = {
    "energy": ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity"],
    "transport": ["Traffic"],
    "weather": ["Weather"],
}

# Dataset-specific configurations
DATASET_CONFIGS = {
    "ETTh1": {
        "enc_in": 7,
        "c_out": 7,
        "modes": 64,
        "dropout": 0.05,
        "batch_size": 16,
        "train_epochs": 10,
    },
    "ETTh2": {
        "enc_in": 7,
        "c_out": 7,
        "modes": 64,
        "dropout": 0.05,
        "batch_size": 16,
        "train_epochs": 10,
    },
    "ETTm1": {
        "enc_in": 7,
        "c_out": 7,
        "modes": 64,
        "dropout": 0.05,
        "batch_size": 16,
        "train_epochs": 10,
    },
    "ETTm2": {
        "enc_in": 7,
        "c_out": 7,
        "modes": 64,
        "dropout": 0.05,
        "batch_size": 16,
        "train_epochs": 10,
    },
    "Weather": {
        "enc_in": 21,
        "c_out": 21,
        "modes": 64,
        "dropout": 0.05,
        "batch_size": 16,
        "train_epochs": 10,
    },
    "Traffic": {
        "enc_in": 862,
        "c_out": 862,
        "modes": 64,
        "dropout": 0.05,
        "batch_size": 8,
        "train_epochs": 5,
    },
    "Electricity": {
        "enc_in": 321,
        "c_out": 321,
        "modes": 32,
        "dropout": 0.1,
        "batch_size": 8,
        "train_epochs": 20,
    },
}


def load_phase1_extractors(checkpoint_base_path, datasets, configs):
    """Load Phase 1 trained models as feature extractors"""
    extractors = {}

    for i, dataset in enumerate(datasets):
        # Create config for this dataset
        dataset_config = configs.copy()
        dataset_specific = DATASET_CONFIGS.get(dataset, {})
        for key, value in dataset_specific.items():
            setattr(dataset_config, key, value)

        # Load checkpoint
        checkpoint_path = os.path.join(
            checkpoint_base_path,
            f"imputation_test_FEDformer_Fourier_{dataset}_missing0.2",
            "checkpoint.pth",
        )

        if os.path.exists(checkpoint_path):
            print(f"Loading Phase 1 model for {dataset} from {checkpoint_path}")
            extractors[i] = DomainFeatureExtractor(checkpoint_path, dataset_config)
        else:
            print(f"Warning: Checkpoint not found for {dataset} at {checkpoint_path}")

    return extractors


def create_base_config():
    """Create base configuration"""

    class Config:
        def __init__(self):
            # Common parameters
            self.seq_len = 96
            self.d_model = 512
            self.n_heads = 8
            self.e_layers = 2
            self.d_ff = 2048
            self.activation = "gelu"
            self.learning_rate = 0.0001
            self.patience = 5
            self.missing_ratio = 0.2
            self.missing_weight = 2.0
            self.version = "Fourier"
            self.mode_select = "random"
            self.embed = "timeF"
            self.freq = "h"
            self.moving_avg = [24]

            # Will be set per dataset
            self.enc_in = 7
            self.c_out = 7
            self.modes = 64
            self.dropout = 0.05
            self.batch_size = 16
            self.train_epochs = 10

    return Config()
