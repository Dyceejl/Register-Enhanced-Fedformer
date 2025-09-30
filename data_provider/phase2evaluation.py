# evaluate_phase2.py - Evaluation script for Phase 2 model
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from phase2_models import (
    FEDformerRegisterImputation,
    MultiDomainDataset,
    load_phase1_extractors,
    create_base_config,
    DOMAIN_CONFIG,
    DATASET_CONFIGS,
)
from data_provider.data_factory import data_provider
from utils.metrics import metric
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore")


class Phase2Evaluator:
    """Phase 2 Model Evaluation"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
        )

        # Multi-domain dataset handler
        self.domain_handler = MultiDomainDataset(DOMAIN_CONFIG)

        # Load Phase 1 extractors
        print("Loading Phase 1 trained models...")
        self.base_config = create_base_config()
        self.domain_extractors = load_phase1_extractors(
            args.phase1_checkpoint_path, args.datasets, self.base_config
        )

        # Create and load Phase 2 model
        print("Creating and loading Phase 2 model...")
        self.model = self._build_and_load_model()

    def _build_and_load_model(self):
        """Build and load trained Phase 2 model"""
        config = self.base_config
        config.c_out = max([DATASET_CONFIGS[d]["c_out"] for d in self.args.datasets])

        model = FEDformerRegisterImputation(
            configs=config,
            domain_extractors=self.domain_extractors,
            register_size=self.args.register_size,
            num_register_tokens=self.args.num_register_tokens,
        )

        # Load trained weights
        checkpoint_path = os.path.join(self.args.checkpoints, self.args.checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])

        model.to(self.device)
        model.eval()

        print(f"Model loaded from: {checkpoint_path}")
        return model

    def create_missing_mask(self, data, missing_ratio=0.2, pattern="random"):
        """Create different missing patterns for evaluation"""
        mask = torch.ones_like(data)
        B, L, C = data.shape

        if pattern == "random":
            # Random missing pattern
            missing_indices = torch.rand(B, L, C) < missing_ratio

        elif pattern == "block":
            # Block missing pattern
            missing_indices = torch.zeros(B, L, C, dtype=torch.bool)
            for b in range(B):
                for c in range(C):
                    block_size = int(L * missing_ratio)
                    start_idx = torch.randint(0, L - block_size + 1, (1,)).item()
                    missing_indices[b, start_idx : start_idx + block_size, c] = True

        elif pattern == "seasonal":
            # Seasonal missing pattern (simulate sensor failures during specific periods)
            missing_indices = torch.zeros(B, L, C, dtype=torch.bool)
            seasonal_period = 24  # Daily pattern
            for b in range(B):
                for c in range(C):
                    # Create missing blocks at regular intervals
                    for start in range(0, L, seasonal_period):
                        if torch.rand(1) < missing_ratio:
                            end = min(start + seasonal_period // 4, L)
                            missing_indices[b, start:end, c] = True

        mask[missing_indices] = 0

        # Create data with missing values
        data_missing = data.clone()
        data_missing[missing_indices] = float("nan")

        return data_missing, mask

    def evaluate_dataset(
        self, dataset_name, missing_patterns=["random"], missing_ratios=[0.1, 0.2, 0.3]
    ):
        """Evaluate on a specific dataset with different missing patterns and ratios"""
        print(f"\nEvaluating on {dataset_name}...")

        # Create dataset
        eval_args = argparse.Namespace(**vars(self.args))
        dataset_config = DATASET_CONFIGS.get(dataset_name, {})
        for key, value in dataset_config.items():
            setattr(eval_args, key, value)

        eval_args.data = dataset_name
        eval_args.data_path = f"{dataset_name}.csv"

        _, test_loader = data_provider(eval_args, "test")
        domain_id = self.domain_handler.get_domain_id(dataset_name)

        results = {}

        for pattern in missing_patterns:
            results[pattern] = {}

            for missing_ratio in missing_ratios:
                print(f"  Pattern: {pattern}, Missing Ratio: {missing_ratio}")

                pattern_results = self._evaluate_pattern(
                    test_loader, domain_id, pattern, missing_ratio
                )
                results[pattern][missing_ratio] = pattern_results

                print(
                    f"    MAE: {pattern_results['mae']:.4f}, "
                    f"Missing MAE: {pattern_results['missing_mae']:.4f}, "
                    f"RMSE: {pattern_results['rmse']:.4f}"
                )

        return results

    def _evaluate_pattern(self, test_loader, domain_id, pattern, missing_ratio):
        """Evaluate specific pattern and ratio"""
        total_mae = 0.0
        total_missing_mae = 0.0
        total_rmse = 0.0
        total_missing_rmse = 0.0
        total_mape = 0.0
        total_missing_mape = 0.0

        all_predictions = []
        all_true_values = []
        all_missing_mask = []

        num_batches = 0

        with torch.no_grad():
            for batch_data in test_loader:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # Create missing values with specific pattern
                batch_x_missing, mask = self.create_missing_mask(
                    batch_x, missing_ratio=missing_ratio, pattern=pattern
                )
                mask = mask.to(self.device)

                # Forward pass
                outputs = self.model(
                    batch_x_missing,
                    batch_x_mark,
                    mask,
                    domain_id=domain_id,
                    mode="inference",
                )

                predictions = outputs["imputed"]

                # Calculate metrics
                mae = torch.abs(predictions - batch_x).mean().item()
                rmse = torch.sqrt(torch.mean((predictions - batch_x) ** 2)).item()

                # Calculate MAPE (avoid division by zero)
                mape = (
                    torch.mean(
                        torch.abs((predictions - batch_x) / (batch_x + 1e-8))
                    ).item()
                    * 100
                )

                # Metrics for missing values only
                missing_mask = (1 - mask).bool()
                if missing_mask.sum() > 0:
                    missing_mae = (
                        torch.abs(predictions[missing_mask] - batch_x[missing_mask])
                        .mean()
                        .item()
                    )
                    missing_rmse = torch.sqrt(
                        torch.mean(
                            (predictions[missing_mask] - batch_x[missing_mask]) ** 2
                        )
                    ).item()
                    missing_mape = (
                        torch.mean(
                            torch.abs(
                                (predictions[missing_mask] - batch_x[missing_mask])
                                / (batch_x[missing_mask] + 1e-8)
                            )
                        ).item()
                        * 100
                    )
                else:
                    missing_mae = missing_rmse = missing_mape = 0.0

                total_mae += mae
                total_rmse += rmse
                total_mape += mape
                total_missing_mae += missing_mae
                total_missing_rmse += missing_rmse
                total_missing_mape += missing_mape

                # Store for analysis
                all_predictions.append(predictions.cpu().numpy())
                all_true_values.append(batch_x.cpu().numpy())
                all_missing_mask.append(missing_mask.cpu().numpy())

                num_batches += 1

        return {
            "mae": total_mae / num_batches,
            "rmse": total_rmse / num_batches,
            "mape": total_mape / num_batches,
            "missing_mae": total_missing_mae / num_batches,
            "missing_rmse": total_missing_rmse / num_batches,
            "missing_mape": total_missing_mape / num_batches,
            "predictions": np.concatenate(all_predictions, axis=0),
            "true_values": np.concatenate(all_true_values, axis=0),
            "missing_mask": np.concatenate(all_missing_mask, axis=0),
        }

    def evaluate_domain_classification(self):
        """Evaluate domain classification accuracy"""
        print("\nEvaluating domain classification...")

        all_predictions = []
        all_true_labels = []

        for dataset_name in self.args.datasets:
            # Create dataset
            eval_args = argparse.Namespace(**vars(self.args))
            dataset_config = DATASET_CONFIGS.get(dataset_name, {})
            for key, value in dataset_config.items():
                setattr(eval_args, key, value)

            eval_args.data = dataset_name
            eval_args.data_path = f"{dataset_name}.csv"

            _, test_loader = data_provider(eval_args, "test")
            true_domain_id = self.domain_handler.get_domain_id(dataset_name)

            with torch.no_grad():
                for batch_data in test_loader:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                    batch_x = batch_x.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)

                    # Create missing values
                    batch_x_missing, mask = self.create_missing_mask(
                        batch_x, missing_ratio=0.2
                    )
                    mask = mask.to(self.device)

                    # Forward pass
                    outputs = self.model(
                        batch_x_missing,
                        batch_x_mark,
                        mask,
                        domain_id=None,
                        mode="inference",  # Don't provide domain_id
                    )

                    # Get domain predictions
                    domain_pred = torch.softmax(outputs["domain_pred"], dim=1)
                    predicted_domains = torch.argmax(domain_pred, dim=1)

                    all_predictions.extend(predicted_domains.cpu().numpy())
                    all_true_labels.extend([true_domain_id] * batch_x.shape[0])

        # Calculate accuracy
        accuracy = np.mean(np.array(all_predictions) == np.array(all_true_labels))

        # Generate classification report
        domain_names = [
            self.domain_handler.get_domain_name(i) for i in range(len(DOMAIN_CONFIG))
        ]
        report = classification_report(
            all_true_labels,
            all_predictions,
            target_names=domain_names,
            output_dict=True,
        )

        print(f"Domain Classification Accuracy: {accuracy:.4f}")
        print("\nPer-domain classification report:")
        for domain_name in domain_names:
            if domain_name in report:
                metrics = report[domain_name]
                print(
                    f"{domain_name}: Precision={metrics['precision']:.3f}, "
                    f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}"
                )

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "predictions": all_predictions,
            "true_labels": all_true_labels,
        }

    def analyze_register_utilization(self):
        """Analyze how register tokens are utilized"""
        print("\nAnalyzing register utilization...")

        register_usage = {}

        for dataset_name in self.args.datasets:
            # Create dataset
            eval_args = argparse.Namespace(**vars(self.args))
            dataset_config = DATASET_CONFIGS.get(dataset_name, {})
            for key, value in dataset_config.items():
                setattr(eval_args, key, value)

            eval_args.data = dataset_name
            eval_args.data_path = f"{dataset_name}.csv"

            _, test_loader = data_provider(eval_args, "test")
            domain_id = self.domain_handler.get_domain_id(dataset_name)

            cluster_assignments = []
            distances = []

            with torch.no_grad():
                for batch_data in test_loader:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
                    batch_x = batch_x.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)

                    # Create missing values
                    batch_x_missing, mask = self.create_missing_mask(
                        batch_x, missing_ratio=0.2
                    )
                    mask = mask.to(self.device)

                    # Get domain features
                    domain_extractor = self.model.domain_extractors[domain_id]
                    domain_features, _ = domain_extractor(
                        batch_x_missing, batch_x_mark, mask
                    )

                    # Get closest clusters
                    closest_idx, cluster_distances = (
                        self.model.ts_register.get_closest_cluster(domain_features)
                    )

                    cluster_assignments.extend(closest_idx.cpu().numpy())
                    distances.extend(cluster_distances.cpu().numpy())

            register_usage[dataset_name] = {
                "cluster_assignments": cluster_assignments,
                "distances": distances,
                "unique_clusters": np.unique(cluster_assignments),
                "cluster_usage_frequency": np.bincount(
                    cluster_assignments, minlength=self.args.register_size
                ),
            }

        return register_usage

    def visualize_results(self, results, save_path="./evaluation_results/"):
        """Create visualizations for evaluation results"""
        os.makedirs(save_path, exist_ok=True)

        # 1. Missing ratio vs performance
        plt.figure(figsize=(15, 10))

        datasets = list(results.keys())
        patterns = (
            ["random", "block", "seasonal"]
            if "random" in results[datasets[0]]
            else ["random"]
        )
        missing_ratios = [0.1, 0.2, 0.3]

        for i, pattern in enumerate(patterns):
            plt.subplot(2, 2, i + 1)

            for dataset in datasets:
                if pattern in results[dataset]:
                    maes = [
                        results[dataset][pattern][ratio]["missing_mae"]
                        for ratio in missing_ratios
                    ]
                    plt.plot(missing_ratios, maes, marker="o", label=dataset)

            plt.xlabel("Missing Ratio")
            plt.ylabel("MAE on Missing Values")
            plt.title(f"Performance vs Missing Ratio ({pattern.title()} Pattern)")
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(
            os.path.join(save_path, "missing_ratio_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # 2. Cross-domain performance heatmap
        domain_performance = {}
        for dataset in datasets:
            domain_name = self.domain_handler.get_domain_name(
                self.domain_handler.get_domain_id(dataset)
            )
            if "random" in results[dataset] and 0.2 in results[dataset]["random"]:
                domain_performance[dataset] = results[dataset]["random"][0.2][
                    "missing_mae"
                ]

        if domain_performance:
            plt.figure(figsize=(10, 6))
            datasets_list = list(domain_performance.keys())
            values = list(domain_performance.values())

            plt.bar(datasets_list, values)
            plt.xlabel("Dataset")
            plt.ylabel("Missing MAE")
            plt.title("Cross-Domain Imputation Performance")
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                os.path.join(save_path, "cross_domain_performance.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def generate_report(self, results, domain_classification_results, register_usage):
        """Generate comprehensive evaluation report"""
        report = {
            "summary": {
                "model_info": {
                    "register_size": self.args.register_size,
                    "num_register_tokens": self.args.num_register_tokens,
                    "evaluated_datasets": self.args.datasets,
                },
                "domain_classification_accuracy": domain_classification_results[
                    "accuracy"
                ],
                "average_performance": {},
            },
            "detailed_results": results,
            "domain_classification": domain_classification_results,
            "register_analysis": register_usage,
        }

        # Calculate average performance across datasets
        avg_metrics = {"mae": 0, "missing_mae": 0, "rmse": 0, "missing_rmse": 0}
        count = 0

        for dataset in results:
            if "random" in results[dataset] and 0.2 in results[dataset]["random"]:
                metrics = results[dataset]["random"][0.2]
                for key in avg_metrics:
                    avg_metrics[key] += metrics[key]
                count += 1

        if count > 0:
            for key in avg_metrics:
                avg_metrics[key] /= count

        report["summary"]["average_performance"] = avg_metrics

        return report

    def run_comprehensive_evaluation(self):
        """Run complete evaluation pipeline"""
        print("Starting comprehensive Phase 2 evaluation...")

        # 1. Imputation performance evaluation
        missing_patterns = ["random", "block", "seasonal"]
        missing_ratios = [0.1, 0.2, 0.3]

        imputation_results = {}
        for dataset in self.args.datasets:
            imputation_results[dataset] = self.evaluate_dataset(
                dataset, missing_patterns, missing_ratios
            )

        # 2. Domain classification evaluation
        domain_results = self.evaluate_domain_classification()

        # 3. Register utilization analysis
        register_usage = self.analyze_register_utilization()

        # 4. Generate visualizations
        self.visualize_results(imputation_results)

        # 5. Generate comprehensive report
        final_report = self.generate_report(
            imputation_results, domain_results, register_usage
        )

        # 6. Save results
        import json

        results_dir = "./evaluation_results/"
        os.makedirs(results_dir, exist_ok=True)

        with open(os.path.join(results_dir, "evaluation_report.json"), "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj

            json.dump(convert_numpy(final_report), f, indent=2)

        print(f"\nEvaluation completed! Results saved to {results_dir}")
        print(
            f"Average Missing MAE: {final_report['summary']['average_performance']['missing_mae']:.4f}"
        )
        print(
            f"Domain Classification Accuracy: {final_report['summary']['domain_classification_accuracy']:.4f}"
        )

        return final_report


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Register-Enhanced FEDformer Evaluation"
    )

    # Basic settings
    parser.add_argument(
        "--model_id", type=str, default="phase2_register", help="model id"
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="finetune_best.pth",
        help="checkpoint filename",
    )

    # Data settings
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["ETTh1", "ETTh2", "Traffic", "Weather"],
        help="datasets to evaluate on",
    )
    parser.add_argument(
        "--root_path", type=str, default="./data/", help="root path of data file"
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/phase2/",
        help="location of model checkpoints",
    )
    parser.add_argument(
        "--phase1_checkpoint_path",
        type=str,
        default="D:/FEDformer/checkpoints/",
        help="path to Phase 1 checkpoints",
    )

    # Model parameters (should match training)
    parser.add_argument(
        "--register_size", type=int, default=128, help="size of TS-Register"
    )
    parser.add_argument(
        "--num_register_tokens", type=int, default=3, help="number of register tokens"
    )

    # Evaluation settings
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="data loader num workers"
    )
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")

    # Phase 1 compatibility parameters
    parser.add_argument("--task_name", type=str, default="imputation")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)

    args = parser.parse_args()

    print("Evaluation Args:")
    print(args)

    # Run evaluation
    evaluator = Phase2Evaluator(args)
    results = evaluator.run_comprehensive_evaluation()

    return results


if __name__ == "__main__":
    main()
