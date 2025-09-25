"""
Test configuration with reduced epochs for faster evaluation
"""

from configs.imputation_config import LightImputationConfig


class TestImputationConfig(LightImputationConfig):
    """Reduced configuration for testing thesis evaluation"""

    def __init__(self):
        super().__init__()

        # Reduced epochs for faster testing
        self.train_epochs = 5  # Reduced from 25
        self.pretrain_epochs = 3  # Reduced from 15
        self.finetune_epochs = 2  # Reduced from 10

        # Faster training settings
        self.batch_size = 16  # Reduced from 32 (if memory allows)
        self.patience = 3  # Reduced early stopping patience

        # Limit data processing for faster runs
        self.max_batches_per_domain = 50  # Reduced from 1000
        self.max_eval_batches_per_domain = 25  # Reduced from 1000

        self.fixed_input_dim = 64

        print("Using TestImputationConfig with reduced epochs for faster evaluation")
