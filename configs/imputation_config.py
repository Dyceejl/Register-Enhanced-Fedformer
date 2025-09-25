# Update your ImputationConfig class with these complete ROSE-style parameters


class ImputationConfig:
    """Complete configuration for ROSE-style multi-domain imputation"""

    def __init__(self):
        # Data paths
        self.root_path = "./data/"
        self.save_path = "./checkpoints/"

        # Model architecture - adjusted for domain-separate training
        self.seq_len = 96  # Sequence length for imputation
        self.label_len = 0  # Not used for imputation
        self.pred_len = 0  # Not used for imputation

        self.d_model = 512
        self.d_ff = 2048
        self.n_heads = 8
        self.e_layers = 2
        self.dropout = 0.05
        self.activation = "gelu"
        self.moving_avg = 25

        # FEDformer specific
        self.version = "Fourier"
        self.mode_select = "random"
        self.modes = 64
        self.L = 1
        self.base = "legendre"

        # Embedding
        self.embed = "timeF"
        self.freq = "h"
        self.features = "M"  # Multivariate
        self.target = "OT"

        # ====== ROSE-SPECIFIC PARAMETERS ======

        # Decomposed Frequency Learning (Section 3.2)
        self.Kf = 4  # Number of frequency masks
        self.max_freq_ratio = 0.2  # Frequency masking range (τ upper bound)

        # Time Series Register (Section 3.3)
        self.register_size = 128  # H: Number of register vectors
        self.register_dim = 256  # Dr: Register vector dimension
        self.num_register_tokens = 3  # Nr: Number of register tokens

        # Training Strategy (Section 3.4)
        self.two_phase_training = False
        self.pretrain_epochs = None  # Will be set as train_epochs // 3
        self.finetune_epochs = None  # Will be set as remaining epochs

        # Loss Weights (ROSE Equation 12)
        self.reconstruction_weight = 1.0  # Weight for frequency reconstruction
        self.register_weight = 0.1  # Weight for register clustering
        self.imputation_weight = 1.0  # Weight for imputation task
        self.prediction_weight = 1.0  # Weight for prediction (if used)

        # ====== TRAINING PARAMETERS ======

        self.batch_size = 32
        self.learning_rate = 1e-4
        self.train_epochs = 50  # Reduced for faster experimentation
        self.num_workers = 0

        # Early stopping and checkpointing
        self.patience = 8
        self.save_best = True
        self.save_frequency = 5  # Save checkpoint every N epochs

        # ====== MISSING VALUE CONFIGURATION ======

        # Training missing patterns
        self.missing_rate = 0.2
        self.missing_pattern = "random"

        # Test missing patterns
        self.test_missing_rate = 0.2
        self.test_missing_pattern = "random"

        # ====== EVALUATION PARAMETERS ======

        self.output_attention = False
        self.use_amp = False  # Automatic Mixed Precision
        self.use_gpu = True

        # ====== DOMAIN-SPECIFIC SETTINGS ======

        # Domain configurations (loaded from DOMAIN_CONFIGS)
        self.domain_configs = {
            "ETTh1": {"features": 7, "freq": "h"},
            "ETTh2": {"features": 7, "freq": "h"},
            "ETTm1": {"features": 7, "freq": "t"},
            "ETTm2": {"features": 7, "freq": "t"},
            "weather": {"features": 21, "freq": "h"},
            "traffic": {"features": 865, "freq": "h"},
            "electricity": {"features": 321, "freq": "h"},
        }

        # Auto-adjust training epochs for two-phase training
        self._setup_training_phases()

    def _setup_training_phases(self):
        """Automatically configure two-phase training"""
        if self.two_phase_training:
            if self.pretrain_epochs is None:
                self.pretrain_epochs = max(1, self.train_epochs // 3)
            if self.finetune_epochs is None:
                self.finetune_epochs = self.train_epochs - self.pretrain_epochs

            print(f"Two-phase training configured:")
            print(f"  Pre-training: {self.pretrain_epochs} epochs")
            print(f"  Fine-tuning: {self.finetune_epochs} epochs")
            print(f"  Total: {self.train_epochs} epochs")
        else:
            self.pretrain_epochs = 0
            self.finetune_epochs = self.train_epochs

    def get_domain_config(self, domain_name):
        """Get configuration for specific domain"""
        return self.domain_configs.get(domain_name, {"features": 7, "freq": "h"})

    def print_config(self):
        """Print complete configuration"""
        print("\n" + "=" * 60)
        print("ROSE-STYLE MULTI-DOMAIN IMPUTATION CONFIGURATION")
        print("=" * 60)

        print(f"Model Architecture:")
        print(f"  d_model: {self.d_model}, d_ff: {self.d_ff}")
        print(f"  n_heads: {self.n_heads}, e_layers: {self.e_layers}")
        print(f"  seq_len: {self.seq_len}, dropout: {self.dropout}")

        print(f"\nROSE Components:")
        print(f"  Frequency masks (Kf): {self.Kf}")
        print(f"  Max freq ratio: {self.max_freq_ratio}")
        print(f"  Register size: {self.register_size}")
        print(f"  Register tokens: {self.num_register_tokens}")

        print(f"\nTraining Strategy:")
        print(f"  Two-phase training: {self.two_phase_training}")
        print(f"  Pre-training epochs: {self.pretrain_epochs}")
        print(f"  Fine-tuning epochs: {self.finetune_epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.learning_rate}")

        print(f"\nLoss Weights:")
        print(f"  Reconstruction: {self.reconstruction_weight}")
        print(f"  Register: {self.register_weight}")
        print(f"  Imputation: {self.imputation_weight}")

        print(f"\nDomain Information:")
        for domain, config in self.domain_configs.items():
            print(
                f"  {domain:12s}: {config['features']:3d} features, freq='{config['freq']}'"
            )

        print("=" * 60)


class LightImputationConfig(ImputationConfig):
    """Lighter configuration for quick experimentation with customizable training phases"""

    def __init__(self):
        super().__init__()

        # Reduced model size for faster training
        self.d_model = 256
        self.d_ff = 1024
        self.n_heads = 8
        self.e_layers = 1

        # Smaller register
        self.register_size = 64
        self.register_dim = 128

        # Fewer frequency masks
        self.Kf = 2

        # EXTENDED TRAINING FOR COMPREHENSIVE STUDY
        self.train_epochs = 10
        self.pretrain_epochs = 0
        self.finetune_epochs = 10

        # Override the automatic phase setup
        self.two_phase_training = False

        # Verify the math
        assert self.pretrain_epochs + self.finetune_epochs == self.train_epochs, (
            f"Phase mismatch: {self.pretrain_epochs} + {self.finetune_epochs} != {self.train_epochs}"
        )

    def _setup_training_phases(self):
        """Override parent method since we set phases manually"""
        pass  # Do nothing - phases are set in __init__
