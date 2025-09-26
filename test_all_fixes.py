# test_all_fixes.py - Verify all three models work before full comparison


def test_all_models():
    """Test FedFormer, BRITS, and SAITS separately"""

    from configs.imputation_config import LightImputationConfig
    from imputation_trainer import create_domain_datasets, ImputationDataset
    import copy
    import torch

    print("=" * 80)
    print("🧪 TESTING ALL MODEL FIXES")
    print("=" * 80)

    # --- CONFIGS ---
    configs = LightImputationConfig()
    configs.train_epochs = 1  # quick test
    configs.batch_size = 32  # small batch for fast test
    configs.num_workers = 0  # avoid multiprocessing issues
    configs.test_missing_rate = 0.2
    configs.test_missing_pattern = "random"

    # Get datasets
    train_datasets, test_datasets = create_domain_datasets(configs)
    test_domain = "ETTh1"

    print(f"Testing on domain: {test_domain}")
    print(f"Dataset size: {len(train_datasets[test_domain])}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- TEST 1: FEDFORMER ---
    print("\n" + "=" * 40)
    print("1. TESTING FEDFORMER EVALUATION")
    print("=" * 40)

    try:
        from imputation_trainer import ImputationFocusedTrainer
        from multi_domain_fedformer import MultiDomainFEDformerWithoutRegister

        # Create model and trainer
        model = MultiDomainFEDformerWithoutRegister(configs).to(device)
        trainer = ImputationFocusedTrainer(model, configs)

        # Shared normalization scaler for train/test
        temp_dataset = ImputationDataset(train_datasets[test_domain], missing_rate=0.2)
        scaler = temp_dataset.scaler

        train_dataset_with_scaler = ImputationDataset(
            train_datasets[test_domain], missing_rate=0.2, shared_scaler=scaler
        )
        test_dataset_with_scaler = ImputationDataset(
            test_datasets[test_domain], missing_rate=0.2, shared_scaler=scaler
        )

        # Quick training
        single_domain_train = {test_domain: train_dataset_with_scaler}
        epoch_loss = trainer.train_epoch_imputation_only(single_domain_train)
        print(f"  Training loss: {epoch_loss:.6f}")

        # Evaluation
        test_single_domain = {test_domain: test_dataset_with_scaler}
        results = trainer.evaluate_domain_separate(test_single_domain)
        mse, mae = results[test_domain]

        if mse != float("inf"):
            print(f"  ✅ FedFormer: MSE={mse:.6f}, MAE={mae:.6f}")
        else:
            print(f"  ❌ FedFormer evaluation failed")
            return False

    except Exception as e:
        print(f"  ❌ FedFormer test failed: {e}")
        return False

    # --- TEST 2: BRITS ---
    print("\n" + "=" * 40)
    print("2. TESTING BRITS (Single Domain)")
    print("=" * 40)

    try:
        from baseline_models import BRITSWrapper

        brits_configs = copy.deepcopy(configs)
        brits_configs.train_epochs = 2  # must be ≥ patience
        brits_configs.patience = 1  # small for quick test

        brits_model = BRITSWrapper(brits_configs)
        single_domain_train = {test_domain: train_datasets[test_domain]}
        brits_model.fit(single_domain_train)

        # Evaluate
        mse, mae = brits_model.evaluate_domain(test_domain, test_datasets[test_domain])
        if mse != float("inf"):
            print(f"  ✅ BRITS: MSE={mse:.6f}, MAE={mae:.6f}")
        else:
            print(f"  ❌ BRITS evaluation failed")
            return False

    except Exception as e:
        print(f"  ❌ BRITS test failed: {e}")
        return False

    # --- TEST 3: SAITS ---
    print("\n" + "=" * 40)
    print("3. TESTING SAITS (Single Domain)")
    print("=" * 40)

    try:
        from baseline_models import SAITSWrapper

        saits_configs = copy.deepcopy(configs)
        saits_configs.train_epochs = 2  # quick test
        saits_configs.patience = 1

        saits_model = SAITSWrapper(saits_configs)
        saits_model.fit(single_domain_train)

        # Evaluate
        mse, mae = saits_model.evaluate_domain(test_domain, test_datasets[test_domain])
        if mse != float("inf"):
            print(f"  ✅ SAITS: MSE={mse:.6f}, MAE={mae:.6f}")
        else:
            print(f"  ❌ SAITS evaluation failed")
            return False

    except Exception as e:
        print(f"  ❌ SAITS test failed: {e}")
        return False

    print("\n" + "=" * 80)
    print("🎉 ALL MODEL TESTS PASSED!")
    print("=" * 80)
    print("All three models can train and evaluate successfully.")
    print("Ready for full comparison!")

    return True


if __name__ == "__main__":
    success = test_all_models()
    if success:
        print("\n🚀 Run the full comparison:")
        print("   python run_imputation_comparison.py --fixed-comparison")
    else:
        print("\n⚠️ Fix the failing model before running full comparison.")
