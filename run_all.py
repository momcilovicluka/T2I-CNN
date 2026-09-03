"""
Master script to run all experiments for Chapter 4.

Runs 60 CNN experiments (5 T2I x 3 datasets x 4 architectures)
+ 9 tabular baselines (RF, XGBoost, MLP x 3 datasets)
= 69 total experiments.

Usage:
    python run_all.py                        # All 69 experiments
    python run_all.py --dataset breast_cancer # One dataset only
    python run_all.py --baselines           # Just the 9 tabular baselines
    python run_all.py --cnn-only            # Just the 60 CNN experiments
    python run_all.py --dry-run             # Print what would run, don't train
"""

import itertools
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch



DATASETS = ['breast_cancer', 'dry_bean', 'adult_income']
T2I_METHODS = ['naive', 'tinto', 'deepinsight', 'igtd', 's_igtd']
CNN_ARCHITECTURES = ['shallow', 'resnet', 'resnet_scratch', 'vit']
BASELINE_MODELS = ['rf', 'xgboost', 'mlp']

# Dataset-specific configurations
DATASET_CONFIG = {
    'breast_cancer': {'num_classes': 2, 'image_size': 32},
    'dry_bean': {'num_classes': 7, 'image_size': 32},
    'adult_income': {'num_classes': 2, 'image_size': 32},
}


class ProgressTracker:
    """Track experiment progress with ETA and resume support."""

    def __init__(self, total, label="experiments"):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.label = label
        self.start_time = time.time()
        self.times = []

    def start_experiment(self, idx, name):
        self.current_name = name
        self.current_start = time.time()
        elapsed = time.time() - self.start_time
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"[{idx}/{self.total}] {name}")
        print(f"  Elapsed: {self._fmt_time(elapsed)}", end="")

    def end_experiment(self, success=True, f1=None):
        elapsed_exp = time.time() - self.current_start
        self.times.append(elapsed_exp)

        if success:
            self.completed += 1
            avg_time = sum(self.times) / len(self.times)
            remaining = (self.total - self.completed - self.failed) * avg_time
            eta = datetime.now() + timedelta(seconds=remaining)
            f1_str = f" | F1={f1:.4f}" if f1 else ""
            eta_str = eta.strftime("%H:%M")
            print(f"{f1_str} | Done in {elapsed_exp:.0f}s | ETA: {self._fmt_time(remaining)} ({eta_str})")
        else:
            self.failed += 1
            print(f" | FAILED after {elapsed_exp:.0f}s")

        pct = self.completed / self.total * 100 if self.total > 0 else 0
        print(f"  Progress: {self.completed}/{self.total} done, {self.failed} failed ({pct:.0f}%)")

    def summary(self):
        elapsed = time.time() - self.start_time
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"SUMMARY: {self.completed}/{self.total} completed, {self.failed} failed")
        print(f"Total time: {self._fmt_time(elapsed)}")
        if self.times:
            print(f"Avg per experiment: {sum(self.times)/len(self.times):.0f}s")

    def _fmt_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.0f}s"
        if seconds < 3600:
            m = int(seconds // 60)
            s = int(seconds % 60)
            return f"{m}m {s}s"
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"

def create_cnn_model(arch, num_classes):
    """Initialize a CNN model by architecture name.

    IMPORTANT: Pretrained models must use input_channels=3 — they receive
    ImageNet-normalized RGB input (imagenet_normalize repeats the grayscale
    channel to 3). This keeps the original 3-channel conv1 with pretrained
    weights instead of replacing it with a 1-channel averaged version.
    From-scratch models use input_channels=1 (raw grayscale images).
    """
    if arch == 'shallow':
        from src.models.shallow_cnn import ShallowCNN
        return ShallowCNN(num_classes=num_classes)
    elif arch == 'resnet':
        from src.models.resnet_wrapper import ResNetWrapper
        return ResNetWrapper(num_classes=num_classes, pretrained=True, input_channels=3)
    elif arch == 'resnet_scratch':
        from src.models.resnet_wrapper import ResNetWrapper
        return ResNetWrapper(num_classes=num_classes, pretrained=False, input_channels=1)
    elif arch == 'vit':
        from src.models.vit_wrapper import ViTWrapper
        return ViTWrapper(num_classes=num_classes, pretrained=True, input_channels=3)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def run_single_experiment(dataset, t2i_method, cnn_arch, output_dir='results'):
    """Run one CNN experiment: dataset -> T2I -> CNN -> evaluate.

    Returns: dict with all metrics and metadata.
    """
    from src.preprocessing import preprocess_dataset
    from src.t2i import T2ITransformer
    from src.train import (
        prepare_loaders, train_model, compute_class_weights, set_global_seed
    )
    from src.evaluate import evaluate_model

    set_global_seed(42)
    config = DATASET_CONFIG[dataset]
    num_classes = config['num_classes']
    image_size = config['image_size']

    # 1. Load and preprocess dataset
    print(f"  Loading {dataset}...")
    data = preprocess_dataset(dataset)
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

    # 2. Fit T2I transformer on training data only (no leakage)
    print(f"  Fitting T2I: {t2i_method}...")
    t2i = T2ITransformer(method=t2i_method, image_size=image_size)
    t2i.fit(X_train, y_train)

    # 3. Transform all splits to images
    train_imgs = t2i.transform(X_train, y_train)
    val_imgs = t2i.transform(X_val, y_val)
    test_imgs = t2i.transform(X_test, y_test)

    # 4. Create DataLoaders
    train_loader, val_loader = prepare_loaders(
        train_imgs.numpy(), y_train,
        val_imgs.numpy(), y_val,
        batch_size=32,
    )

    # 5. Initialize CNN model
    model = create_cnn_model(cnn_arch, num_classes)

    # 6. Compute class weights for imbalanced datasets
    class_weights = compute_class_weights(y_train)

    # 7. Training config
    train_config = {
        'epochs': 50,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'early_stopping_patience': 15,
        'label_smoothing': 0.1,
        'class_weights': class_weights,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    # 8. Train
    print(f"  Training {cnn_arch}...", flush=True)
    start_time = time.time()
    model, history = train_model(model, train_loader, val_loader, train_config)
    train_time = time.time() - start_time

    # 9. Evaluate on test set
    from torch.utils.data import DataLoader, TensorDataset
    test_loader = DataLoader(
        TensorDataset(test_imgs, torch.tensor(y_test).long()),
        batch_size=32, shuffle=False,
    )
    metrics = evaluate_model(model, test_loader, num_classes=num_classes)

    # 10. Add metadata
    metrics['dataset'] = dataset
    metrics['t2i_method'] = t2i_method
    metrics['cnn_arch'] = cnn_arch
    # Persist train-derived pixel scale for TINTO (used by Grad-CAM figures
    # so displayed images match the scale the CNN was trained on).
    if getattr(t2i.transformer, '_pix_min', None) is not None:
        metrics['t2i_pixel_range'] = [
            float(t2i.transformer._pix_min),
            float(t2i.transformer._pix_max),
        ]
    metrics['train_samples'] = len(X_train)
    metrics['test_samples'] = len(X_test)
    metrics['image_size'] = image_size
    metrics['train_time_sec'] = round(train_time, 1)
    metrics['epochs_trained'] = len(history['train_loss'])
    metrics['final_train_loss'] = history['train_loss'][-1]
    metrics['final_val_loss'] = history['val_loss'][-1]
    metrics['history'] = history

    # 11. Save model weights (needed for Grad-CAM visualization)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    model_file = output_path / f"{dataset}_{t2i_method}_{cnn_arch}_model.pt"
    torch.save(model.state_dict(), model_file)

    # 12. Save results (atomic write)
    # FIX (audit): Was a direct open(result_file, 'w') — a kill mid-write
    # (Ctrl+C, Colab timeout) left a truncated JSON that the resume logic
    # (exists() check) would SKIP forever, silently losing the experiment
    # from aggregate_results and disabling the persisted t2i_pixel_range.
    # Now writes to .json.tmp and atomically renames, matching run_baseline.
    # Note: model .pt is saved first; a kill between the two writes leaves
    # an orphan .pt that the resume re-run simply overwrites (harmless).
    result_file = output_path / f"{dataset}_{t2i_method}_{cnn_arch}.json"

    # Convert numpy types for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    import os
    tmp_file = result_file.with_suffix('.json.tmp')
    with open(tmp_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=to_serializable)
    os.replace(str(tmp_file), str(result_file))

    print(f"  -> {result_file.name}: F1={metrics['f1_macro']:.4f}, "
          f"Acc={metrics['accuracy']:.4f} ({train_time:.0f}s)")

    return metrics


def run_baseline(dataset, model_type, output_dir='results'):
    """Run one tabular baseline experiment.

    Returns: dict with all metrics and metadata.
    """
    from src.preprocessing import preprocess_dataset
    from src.train import set_global_seed

    set_global_seed(42)
    config = DATASET_CONFIG[dataset]
    num_classes = config['num_classes']

    # 1. Load dataset (raw features, no T2I)
    # IMPORTANT: Baselines train on X_train ONLY (same as CNNs) for fairness.
    # Previously they trained on train+val (455 vs 398 samples) which gave
    # baselines ~14% more data — biased against CNNs. Val is used only for
    # early stopping by CNNs, so it must not be training data for baselines.
    print(f"  Loading {dataset}...")
    data = preprocess_dataset(dataset)
    X_train = data['X_train']
    y_train = data['y_train']
    X_test, y_test = data['X_test'], data['y_test']

    # 2. Train and evaluate
    print(f"  Training {model_type}...")
    start_time = time.time()

    if model_type == 'rf':
        from src.baselines.rf import train_and_evaluate
    elif model_type == 'xgboost':
        from src.baselines.xgboost_model import train_and_evaluate
    elif model_type == 'mlp':
        from src.baselines.mlp import train_and_evaluate
    else:
        raise ValueError(f"Unknown baseline: {model_type}")

    metrics = train_and_evaluate(X_train, y_train, X_test, y_test, num_classes)
    train_time = time.time() - start_time

    # 3. Add metadata
    metrics['dataset'] = dataset
    metrics['t2i_method'] = 'none'
    metrics['cnn_arch'] = model_type
    metrics['train_samples'] = len(X_train)
    metrics['test_samples'] = len(X_test)
    metrics['train_time_sec'] = round(train_time, 1)

    # 4. Save results (atomic write)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    result_file = output_path / f"baseline_{dataset}_{model_type}.json"

    def to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    import os
    tmp_file = result_file.with_suffix('.json.tmp')
    with open(tmp_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=to_serializable)
    os.replace(str(tmp_file), str(result_file))

    print(f"  -> {result_file.name}: F1={metrics['f1_macro']:.4f}, "
          f"Acc={metrics['accuracy']:.4f} ({train_time:.0f}s)")

    return metrics


def aggregate_results(output_dir='results'):
    """Aggregate all JSON results into a single CSV file."""
    import pandas as pd

    results_dir = Path(output_dir)
    all_results = []

    for json_file in sorted(results_dir.glob('*.json')):
        if json_file.name.startswith('ablation_'):
            continue  # Skip ablation results
        try:
            with open(json_file) as f:
                data = json.load(f)
            # Flatten for CSV (exclude history and confusion_matrix)
            row = {k: v for k, v in data.items()
                   if k not in ('history', 'classification_report')}
            all_results.append(row)
        except Exception as e:
            print(f"  Warning: Could not load {json_file}: {e}")

    if not all_results:
        print("No results found to aggregate.")
        return

    df = pd.DataFrame(all_results)

    # Select key columns for the summary CSV
    key_cols = ['dataset', 't2i_method', 'cnn_arch', 'accuracy', 'f1_macro',
                'precision_macro', 'recall_macro', 'roc_auc', 'pr_auc',
                'train_time_sec', 'epochs_trained', 'train_samples', 'test_samples']
    key_cols = [c for c in key_cols if c in df.columns]

    csv_path = results_dir / 'all_experiments.csv'
    df[key_cols].to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nAggregated {len(df)} results -> {csv_path}")

    # Print summary table
    print("\n=== Summary: Macro-F1 (%) by T2I Method and Architecture ===\n")
    if 't2i_method' in df.columns and 'cnn_arch' in df.columns:
        pivot = df.pivot_table(
            index=['dataset', 't2i_method'],
            columns='cnn_arch',
            values='f1_macro',
        )
        print((pivot * 100).round(2).to_string())

    return df


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run all experiments')
    parser.add_argument('--cnn-only', action='store_true',
                        help='Run only CNN experiments')
    parser.add_argument('--baselines', action='store_true',
                        help='Run only tabular baselines')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Run for one dataset only')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would run, do not train')
    parser.add_argument('--aggregate', action='store_true',
                        help='Aggregate existing results into CSV')
    args = parser.parse_args()

    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # Clean up any partial .json.tmp files from interrupted previous runs
    import os
    for tmp in results_dir.glob('*.json.tmp'):
        print(f"  Cleaning up partial file: {tmp.name}")
        os.remove(str(tmp))

    if args.aggregate:
        aggregate_results()
        return

    run_baselines = not args.cnn_only
    run_cnn = not args.baselines

    if run_cnn:
        combos = list(itertools.product(DATASETS, T2I_METHODS, CNN_ARCHITECTURES))
        if args.dataset:
            combos = [(d, t, c) for d, t, c in combos if d == args.dataset]

        if args.dry_run:
            print(f"Would run {len(combos)} CNN experiments:")
            for d, t, c in combos:
                print(f"  {d} + {t} + {c}")
        else:
            # Count already-done experiments for resume summary
            done_count = 0
            for dataset, t2i, cnn in combos:
                rf = results_dir / f"{dataset}_{t2i}_{cnn}.json"
                if rf.exists():
                    done_count += 1
            if done_count > 0:
                print(f"Resume: {done_count}/{len(combos)} already done, running {len(combos)-done_count} remaining")

            print(f"Running {len(combos)} CNN experiments...")
            for i, (dataset, t2i, cnn) in enumerate(combos, 1):
                result_file = results_dir / f"{dataset}_{t2i}_{cnn}.json"
                if result_file.exists():
                    print(f"\n[{i}/{len(combos)}] {dataset} + {t2i} + {cnn} — SKIP (done)")
                    continue
                print(f"\n[{i}/{len(combos)}] {dataset} + {t2i} + {cnn}")
                try:
                    run_single_experiment(dataset, t2i, cnn)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    if run_baselines:
        combos = list(itertools.product(DATASETS, BASELINE_MODELS))
        if args.dataset:
            combos = [(d, m) for d, m in combos if d == args.dataset]

        if args.dry_run:
            print(f"\nWould run {len(combos)} baseline experiments:")
            for d, m in combos:
                print(f"  {d} + {m}")
        else:
            done_count = 0
            for dataset, model in combos:
                rf = results_dir / f"baseline_{dataset}_{model}.json"
                if rf.exists():
                    done_count += 1
            if done_count > 0:
                print(f"Resume: {done_count}/{len(combos)} already done, running {len(combos)-done_count} remaining")

            print(f"\nRunning {len(combos)} baseline experiments...")
            for i, (dataset, model) in enumerate(combos, 1):
                result_file = results_dir / f"baseline_{dataset}_{model}.json"
                if result_file.exists():
                    print(f"\n[{i}/{len(combos)}] {dataset} + {model} — SKIP (done)")
                    continue
                print(f"\n[{i}/{len(combos)}] {dataset} + {model}")
                try:
                    run_baseline(dataset, model)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    if not args.dry_run:
        print("\nAll experiments complete! Aggregating results...")
        aggregate_results()


if __name__ == '__main__':
    main()
