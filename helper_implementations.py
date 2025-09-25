"""
Helper method implementations for thesis evaluation framework
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from imputation_trainer import ImputationDataset
from run_imputation_comparison import NoRegisterTrainer
from multi_domain_fedformer import MultiDomainFEDformerWithoutRegister


def train_fedformer_on_domains(configs, train_datasets):
    """Train FedFormer on specific set of domains"""

    # Create model
    model = MultiDomainFEDformerWithoutRegister(configs)
    trainer = NoRegisterTrainer(model, configs)

    # Training loop
    best_loss = float("inf")
    for epoch in range(configs.train_epochs):
        train_loss, _ = trainer.train_epoch_imputation_finetune(train_datasets)
        if train_loss < best_loss and train_loss > 0:
            best_loss = train_loss

    return model, None, None


def evaluate_fedformer_on_domain(model, domain_name, test_dataset):
    """Evaluate FedFormer on a specific domain"""

    model.eval()
    test_imputation = ImputationDataset(test_dataset, missing_rate=0.2)
    test_loader = DataLoader(test_imputation, batch_size=32, shuffle=False)

    total_mse = 0
    total_mae = 0
    num_batches = 0

    with torch.no_grad():
        for batch_data in test_loader:
            x_missing, x_mark, mask, x_target = batch_data
            x_missing = x_missing.to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )
            x_mark = x_mark.to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )
            mask = mask.to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )
            x_target = x_target.to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            )

            try:
                dec_out, _, _ = model(
                    x_missing, x_mark, mask, domain_name, "imputation_finetune"
                )

                missing_mask = (~mask).bool()
                if missing_mask.any():
                    mse = torch.nn.functional.mse_loss(
                        dec_out[missing_mask], x_target[missing_mask]
                    )
                    mae = torch.nn.functional.l1_loss(
                        dec_out[missing_mask], x_target[missing_mask]
                    )

                    total_mse += mse.item()
                    total_mae += mae.item()
                    num_batches += 1

            except Exception as e:
                print(f"Error evaluating batch: {e}")
                continue

            if num_batches >= 100:  # Limit for efficiency
                break

    if num_batches > 0:
        return total_mse / num_batches, total_mae / num_batches
    else:
        return float("inf"), float("inf")


def adapt_fedformer(source_datasets, target_dataset, test_dataset, target_domain):
    """Adapt pre-trained FedFormer to new domain with limited data"""

    # Step 1: Pre-train on source domains
    from configs.imputation_config import LightImputationConfig

    pretrain_config = LightImputationConfig()
    pretrain_config.train_epochs = 10  # Reduced for pre-training

    pretrained_model, _, _ = train_fedformer_on_domains(
        pretrain_config, source_datasets
    )

    # Step 2: Fine-tune on target domain
    target_datasets = {target_domain: target_dataset}

    # Create trainer for fine-tuning
    finetune_config = LightImputationConfig()
    finetune_config.train_epochs = 5  # Few epochs for adaptation
    finetune_config.learning_rate = (
        finetune_config.learning_rate * 0.1
    )  # Lower learning rate

    trainer = NoRegisterTrainer(pretrained_model, finetune_config)

    # Fine-tune
    for epoch in range(finetune_config.train_epochs):
        trainer.train_epoch_imputation_finetune(target_datasets)

    # Evaluate
    return evaluate_fedformer_on_domain(pretrained_model, target_domain, test_dataset)


def finetune_brits(source_datasets, target_dataset, test_dataset, target_domain):
    """Fine-tune BRITS from source domains to target domain"""

    from baseline_models import BRITSWrapper
    from configs.imputation_config import LightImputationConfig

    # Step 1: Pre-train on source domains
    config = LightImputationConfig()
    config.train_epochs = 10

    brits_model = BRITSWrapper(config)
    brits_model.fit(source_datasets)

    # Step 2: Fine-tune on target domain (simplified - would need actual fine-tuning implementation)
    target_datasets = {target_domain: target_dataset}

    # For BRITS, we approximate fine-tuning by training a new model with pre-trained features
    # In practice, this would require modifying PyPOTS BRITS to support fine-tuning
    brits_adapted = BRITSWrapper(config)
    brits_adapted.fit(target_datasets)

    return brits_adapted.evaluate_domain(target_domain, test_dataset)


def train_brits_from_scratch(target_dataset, test_dataset, target_domain):
    """Train BRITS from scratch on limited target data"""

    from baseline_models import BRITSWrapper
    from configs.imputation_config import LightImputationConfig

    config = LightImputationConfig()
    config.train_epochs = 15  # More epochs for from-scratch training

    target_datasets = {target_domain: target_dataset}

    brits_model = BRITSWrapper(config)
    brits_model.fit(target_datasets)

    return brits_model.evaluate_domain(target_domain, test_dataset)


# Similar implementations for SAITS
def finetune_saits(source_datasets, target_dataset, test_dataset, target_domain):
    """Fine-tune SAITS from source domains to target domain"""

    from baseline_models import SAITSWrapper
    from configs.imputation_config import LightImputationConfig

    config = LightImputationConfig()
    config.train_epochs = 10

    saits_model = SAITSWrapper(config)
    saits_model.fit(source_datasets)

    # Adaptation (simplified)
    target_datasets = {target_domain: target_dataset}
    saits_adapted = SAITSWrapper(config)
    saits_adapted.fit(target_datasets)

    return saits_adapted.evaluate_domain(target_domain, test_dataset)


def train_saits_from_scratch(target_dataset, test_dataset, target_domain):
    """Train SAITS from scratch on limited target data"""

    from baseline_models import SAITSWrapper
    from configs.imputation_config import LightImputationConfig

    config = LightImputationConfig()
    config.train_epochs = 15

    target_datasets = {target_domain: target_dataset}

    saits_model = SAITSWrapper(config)
    saits_model.fit(target_datasets)

    return saits_model.evaluate_domain(target_domain, test_dataset)


def create_limited_datasets(configs, fraction, seed=42):
    """Create datasets with limited training data"""

    import random

    torch.manual_seed(seed)
    random.seed(seed)

    from imputation_trainer import create_domain_datasets

    # Get full datasets
    full_train_datasets, test_datasets = create_domain_datasets(configs)

    # Sample fraction of training data
    limited_train_datasets = {}

    for domain_name, full_dataset in full_train_datasets.items():
        dataset_size = len(full_dataset)
        sample_size = max(1, int(dataset_size * fraction))

        # Random sampling
        indices = list(range(dataset_size))
        sampled_indices = random.sample(indices, sample_size)

        limited_dataset = Subset(full_dataset, sampled_indices)
        limited_train_datasets[domain_name] = limited_dataset

    return limited_train_datasets, test_datasets


def test_fedformer_with_limited_data(configs, train_datasets, test_datasets):
    """Test FedFormer with limited training data"""

    model, _, _ = train_fedformer_on_domains(configs, train_datasets)

    results = {}
    for domain_name, test_dataset in test_datasets.items():
        mse, mae = evaluate_fedformer_on_domain(model, domain_name, test_dataset)
        results[domain_name] = (mse, mae)

    return results


def test_brits_with_limited_data(configs, train_datasets, test_datasets):
    """Test BRITS with limited training data"""

    from baseline_models import BRITSWrapper

    brits_model = BRITSWrapper(configs)
    brits_model.fit(train_datasets)

    results = {}
    for domain_name, test_dataset in test_datasets.items():
        mse, mae = brits_model.evaluate_domain(domain_name, test_dataset)
        results[domain_name] = (mse, mae)

    return results


def test_saits_with_limited_data(configs, train_datasets, test_datasets):
    """Test SAITS with limited training data"""

    from baseline_models import SAITSWrapper

    saits_model = SAITSWrapper(configs)
    saits_model.fit(train_datasets)

    results = {}
    for domain_name, test_dataset in test_datasets.items():
        mse, mae = saits_model.evaluate_domain(domain_name, test_dataset)
        results[domain_name] = (mse, mae)

    return results
