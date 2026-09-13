"""
Master script to run all experiments for Chapter 4.

Default grid: 36 CNN experiments (4 T2I x 3 datasets x 3 architectures)
+ 9 tabular baselines (RF, XGBoost, MLP x 3 datasets)
= 45 total experiments.

NOTE (2026-09-03): s_igtd was dropped from the study — its wrapper
computed the supervised between-group distance but never fed it to
the layout optimizer, so its images were bit-identical to igtd's
(probe-verified). See Plan/paper-statement-guide.md PART 13i.

NOTE (2026-09-03): ViT-Base/16 is excluded from the DEFAULT grid — at
~15 s/epoch on GPU and ~830 s/epoch on CPU it is not runnable without
GPU time (free Colab GPU was lost). Re-add it any time with
`--archs shallow,resnet,resnet_scratch,vit`; ARCH_LR['vit'] = 1e-4
remains the correct LR (probe-verified, see ARCH_LR comment).

Usage:
    python run_all.py                          # All 45 experiments
    python run_all.py --dataset breast_cancer  # One dataset only
    python run_all.py --archs shallow,resnet,vit  # Override architectures
    python run_all.py --baselines              # Just the 9 tabular baselines
    python run_all.py --cnn-only               # Just the CNN experiments
    python run_all.py --dry-run                # Print what would run, don't train
"""

import itertools
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch



DATASETS = ['breast_cancer', 'dry_bean', 'adult_income']
T2I_METHODS = ['naive', 'tinto', 'deepinsight', 'igtd']
# Default grid runs CPU-feasible architectures only. ViT-Base/16 (pretrained)
# is kept out of the default (2026-09-03): not runnable without GPU time
# (~830 s/epoch on CPU vs ~15 s/epoch on GPU). Re-add with
# `--archs shallow,resnet,resnet_scratch,vit`.
CNN_ARCHITECTURES = ['shallow', 'resnet', 'resnet_scratch']
ALL_ARCHITECTURES = CNN_ARCHITECTURES + ['vit']
BASELINE_MODELS = ['rf', 'xgboost', 'mlp']

# Dataset-specific configurations
DATASET_CONFIG = {
    'breast_cancer': {'num_classes': 2, 'image_size': 32},
    'dry_bean': {'num_classes': 7, 'image_size': 32},
    'adult_income': {'num_classes': 2, 'image_size': 32},
}

# Per-architecture learning rates.
# FIX (2026-09-03, probe-verified): the shared lr=1e-3 makes pretrained
# ViT-B/16 diverge on sparse T2I inputs — train loss pinned at log(2)
# (~0.698) for 20 epochs, val acc stuck at class priors, F1~0. Fine-tuning
# a pretrained ViT at lr=1e-3 is ~100x the established range (timm
# practice ~1e-5..1e-4). Probe on breast_cancer/tinto: at lr=1e-4 the
# same setup reaches ~0.91 val acc within 5 epochs and 0.42 train loss
# by epoch 8. From-scratch models and pretrained ResNet-18 (BatchNorm
# robustness) converge fine at 1e-3, so only ViT is lowered.
ARCH_LR = {
    'shallow': 1e-3,
    'resnet': 1e-3,
    'resnet_scratch': 1e-3,
    'vit': 1e-4,
}


# NOTE (audit C13/C20): a `ProgressTracker` class used to be defined here. It was
# never instantiated anywhere in the project, and its `end_experiment` method
# referenced `datetime`/`timedelta` without importing them, so it could never
# have run. The inline `[i/N] ... — SKIP (done)` / `ERROR` prints inside
# `main()` already cover progress and resume reporting, so the dead class was
# removed instead of repaired. Do not cite it in the write-up.


# C11 — pretrained vs from-scratch ResNet is confounded.
#
# 'resnet' (pretrained) receives ImageNet-normalised 3-channel input, while
# 'resnet_scratch' receives raw 1-channel grayscale. A delta between them is
# therefore a COMBINED effect of (a) weight initialisation and (b) input
# pipeline, and cannot be attributed to pretraining alone
# (professor-validation 9.1, rephrased in the draft).
#
# --scratch-3ch removes confound (b): the from-scratch arm then gets the same
# 3-channel ImageNet-normalised pipeline, so the two arms differ only in how
# the weights are initialised.
#
# Off by default, so the already-recorded 12 resnet_scratch cells stay exactly
# reproducible. Turning it on INVALIDATES those 12 cells: back them up (they are
# the 1-channel arms) and let the resume logic retrain them, or the write-up
# will mix two different input pipelines under one column.
SCRATCH_3CH = False


def create_cnn_model(arch, num_classes):
    """Initialize a CNN model by architecture name.

    IMPORTANT: Pretrained models must use input_channels=3 — they receive
    ImageNet-normalized RGB input (imagenet_normalize repeats the grayscale
    channel to 3). This keeps the original 3-channel conv1 with pretrained
    weights instead of replacing it with a 1-channel averaged version.
    From-scratch models use input_channels=1 (raw grayscale images).
    NOTE (professor-validation 9.1, 2026-09-03; audit C11): pretrained vs
    from-scratch ResNet therefore differ in BOTH weight init and input
    representation (3ch + ImageNet norm vs 1ch raw gray); any delta is a
    combined effect and must not be attributed to pretraining alone, unless the
    module-level SCRATCH_3CH control described above is enabled.
    """
    if arch == 'shallow':
        from src.models.shallow_cnn import ShallowCNN
        return ShallowCNN(num_classes=num_classes)
    elif arch == 'resnet':
        from src.models.resnet_wrapper import ResNetWrapper
        return ResNetWrapper(num_classes=num_classes, pretrained=True, input_channels=3)
    elif arch == 'resnet_scratch':
        from src.models.resnet_wrapper import ResNetWrapper
        if SCRATCH_3CH:
            # C11 control arm: same input pipeline as the pretrained arm, so the
            # only remaining difference is weight initialisation. The extra
            # attribute is what train/evaluate/gradcam read via
            # uses_imagenet_normalization(); without it a random-init model
            # would silently receive un-normalised input.
            model = ResNetWrapper(num_classes=num_classes, pretrained=False,
                                  input_channels=3)
            model.force_imagenet_norm = True
            return model
        return ResNetWrapper(num_classes=num_classes, pretrained=False, input_channels=1)
    elif arch == 'vit':
        from src.models.vit_wrapper import ViTWrapper
        return ViTWrapper(num_classes=num_classes, pretrained=True, input_channels=3)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def _backup_cell(result_file, results_dir, backup_dir_name='backup_pre_rerun'):
    """Move an existing result JSON (and its weights) aside before a re-run.

    A forced re-run must never destroy the evidence for why it was forced: the
    recorded adult_income/naive/resnet JSON (F1 57.58 %) is the artefact the
    seminar's audit trail refers to, so it is preserved rather than overwritten.
    Mirrored to $RESULTS_SYNC_DIR like every other result.

    Returns the list of file names moved.
    """
    import shutil
    from src.colab_sync import sync_path

    backup_dir = Path(results_dir) / backup_dir_name
    backup_dir.mkdir(parents=True, exist_ok=True)
    moved = []
    candidates = [result_file, result_file.with_name(result_file.stem + '_model.pt')]
    for path in candidates:
        if not path.exists():
            continue
        dest = backup_dir / path.name
        n = 2
        while dest.exists():          # never clobber an earlier backup
            dest = backup_dir / f'{path.stem}_run{n}{path.suffix}'
            n += 1
        try:
            shutil.move(str(path), str(dest))
        except OSError as exc:
            print(f"    ! could not back up {path.name}: {exc}")
            continue
        moved.append(dest.name)
        sync_path(dest)
    return moved


def parse_cells(spec):
    """Parse 'dataset/t2i/arch,dataset/t2i/arch' into a set of triples."""
    cells = set()
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        parts = tuple(p.strip() for p in item.split('/'))
        if len(parts) != 3:
            sys.exit(f"Bad --cells entry {item!r}; expected dataset/t2i/arch, "
                     f"e.g. adult_income/naive/resnet")
        cells.add(parts)
    return cells


def _experiment_is_done(result_file):
    """True only if the result JSON exists AND parses as a complete result.

    FIX (audit): Resume previously treated any existing file as done, so a
    truncated JSON from an interrupted pre-atomic-write run (or any corrupt
    file) was skipped forever. A corrupt file must be treated as not-done so
    the experiment is re-run and overwrites it.
    """
    if not result_file.exists():
        return False
    try:
        with open(result_file) as f:
            data = json.load(f)
        return isinstance(data, dict) and 'dataset' in data and 'f1_macro' in data
    except (json.JSONDecodeError, OSError):
        return False


def run_single_experiment(dataset, t2i_method, cnn_arch, output_dir='results',
                          seed=42, split_seed=None, save_weights=True):
    """Run one CNN experiment: dataset -> T2I -> CNN -> evaluate.

    Args:
        seed: training seed — passed to set_global_seed, so it fixes the model
            initialisation, batch order and any other RNG use.
        split_seed: seed for the stratified train/val/test split. Defaults to
            `seed`, i.e. varying the seed also varies the split, which is the
            honest error bar ("would another split give another answer?").
            Pass 42 explicitly to hold the split fixed and isolate training
            noise only (audit C7). The default of 42 reproduces the recorded
            grid exactly.
        save_weights: persist the state_dict (needed by the Grad-CAM figures).
            Pass False for bulk sweeps — nothing consumes weights there, and a
            ResNet sweep would otherwise write ~1.5 GB.

    On completion the result JSON (and the weights, if saved) are mirrored to
    $RESULTS_SYNC_DIR when that is set, so an interrupted Colab session loses at
    most the one experiment in flight. See src/colab_sync.py.

    Returns: dict with all metrics and metadata.
    """
    from src.preprocessing import preprocess_dataset
    from src.t2i import T2ITransformer
    from src.train import (
        prepare_loaders, train_model, compute_class_weights, set_global_seed
    )
    from src.evaluate import evaluate_model

    set_global_seed(seed)
    config = DATASET_CONFIG[dataset]
    num_classes = config['num_classes']
    image_size = config['image_size']
    cell_start = time.time()

    # 1. Load and preprocess dataset
    # C7: splitting is seeded separately from training, so a seed sweep can
    # choose between "another training run" (split_seed fixed) and "another
    # split" (split_seed = seed). Existing behaviour: seed=42, split_seed=None
    # -> random_state 42, identical to the recorded grid.
    if split_seed is None:
        split_seed = seed
    print(f"  Loading {dataset} (seed={seed}, split_seed={split_seed})...")
    data = preprocess_dataset(dataset, random_state=split_seed)
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

    # 2. Fit T2I transformer on training data only (no leakage)
    # t2i_time_sec covers fit + all transforms (professor-validation
    # 10.2): for TINTO the per-sample file writes dominate wall time,
    # and the runtime figure compares T2I methods, so CNN-only
    # train_time_sec alone would understate the method's cost.
    print(f"  Fitting T2I: {t2i_method}...")
    t2i_start = time.time()
    t2i = T2ITransformer(method=t2i_method, image_size=image_size)
    t2i.fit(X_train, y_train)

    # 3. Transform all splits to images
    train_imgs = t2i.transform(X_train, y_train)
    val_imgs = t2i.transform(X_val, y_val)
    test_imgs = t2i.transform(X_test, y_test)
    t2i_time = time.time() - t2i_start

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
        'lr': ARCH_LR[cnn_arch],
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
    metrics['lr'] = train_config['lr']
    # Reproducibility / provenance (audit C7, C11): every cell now records the
    # seeds that produced it and which input pipeline the from-scratch arm used,
    # so a seed sweep and a 3-channel control run can never be mistaken for the
    # recorded grid.
    metrics['seed'] = seed
    metrics['split_seed'] = split_seed
    metrics['scratch_input'] = ('3ch-imagenet'
                                if (cnn_arch == 'resnet_scratch' and SCRATCH_3CH)
                                else ('1ch-raw'
                                      if cnn_arch == 'resnet_scratch'
                                      else 'default'))
    metrics['train_time_sec'] = round(train_time, 1)
    metrics['t2i_time_sec'] = round(t2i_time, 1)
    metrics['total_time_sec'] = round(time.time() - cell_start, 1)
    metrics['epochs_trained'] = len(history['train_loss'])
    metrics['final_train_loss'] = history['train_loss'][-1]
    metrics['final_val_loss'] = history['val_loss'][-1]
    metrics['history'] = history
    # Checkpoint provenance (post-run validation, 2026-09-13): surface the
    # fields written by train_model at top level, so the screen
    # (scripts/audit_cells.py) and the stability table can read them without
    # walking the whole history. `best_epoch` in particular is what exposes a
    # result frozen on a barely-trained checkpoint.
    for _key in ('best_epoch', 'best_val_loss', 'epochs_run', 'stopped_early',
                 'val_loss_oscillation', 'device'):
        if _key in history:
            metrics[_key] = history[_key]

    # 11. Save model weights (needed for Grad-CAM visualization)
    from src.colab_sync import sync_path
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    model_file = output_path / f"{dataset}_{t2i_method}_{cnn_arch}_model.pt"
    if save_weights:
        torch.save(model.state_dict(), model_file)
        sync_path(model_file)   # durability: retraining a cell is the costly part

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
    sync_path(result_file)   # crash-safe: this cell is now complete and durable

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
    # Baselines have no T2I step; fit+eval IS the whole cell.
    metrics['t2i_time_sec'] = 0.0
    metrics['total_time_sec'] = round(train_time, 1)

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
    from src.colab_sync import sync_path
    sync_path(result_file)

    print(f"  -> {result_file.name}: F1={metrics['f1_macro']:.4f}, "
          f"Acc={metrics['accuracy']:.4f} ({train_time:.0f}s)")

    return metrics


def aggregate_results(output_dir='results'):
    """Aggregate all JSON results into a single CSV file."""
    import pandas as pd

    results_dir = Path(output_dir)
    all_results = []
    stale_backfill = []

    for json_file in sorted(results_dir.glob('*.json')):
        if json_file.name.startswith('ablation_'):
            continue  # Skip ablation results
        try:
            with open(json_file) as f:
                data = json.load(f)
            # C6: a grid cell written before the metrics change carries no
            # `f1_macro_all`, so the CSV column would silently be empty for it.
            # Post-run validation (2026-09-13): this is now loud instead of
            # silent — the first aggregate pass after the Colab run shipped a
            # CSV containing only the 9 baseline rows.
            if data.get('t2i_method') not in (None, 'none'):
                for key in ('f1_macro_all', 'balanced_accuracy'):
                    if key not in data:
                        stale_backfill.append((json_file.name, key))
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

    cnn_rows = int((df['t2i_method'] != 'none').sum()) if 't2i_method' in df.columns else 0
    print(f"  Loaded {len(df)} result dicts "
          f"({cnn_rows} CNN/T2I cells, {len(df) - cnn_rows} baselines)")
    if stale_backfill:
        names = sorted({n for n, _ in stale_backfill})
        print(f"  !! {len(names)} grid cell(s) lack the C6 metrics "
              f"({', '.join(sorted({k for _, k in stale_backfill}))}).")
        print(f"     Run: python scripts/backfill_metrics.py --write")
        print(f"     Affected: {', '.join(names[:5])}"
              f"{f' … +{len(names) - 5} more' if len(names) > 5 else ''}")
    if cnn_rows == 0:
        sys.exit('Refusing to write all_experiments.csv: no CNN/T2I rows were '
                 'found. results/ looks like it is missing the grid JSONs, and '
                 'a baselines-only CSV would silently look complete.')


    # Select key columns for the summary CSV
    # C6: `f1_macro_all` + `balanced_accuracy` are cross-dataset-comparable
    # (macro over ALL classes). The legacy `f1_macro` key is kept for
    # backwards compatibility but holds scikit 'binary' positive-class F1 on
    # breast_cancer/adult_income and true macro-F1 only on dry_bean.
    key_cols = ['dataset', 't2i_method', 'cnn_arch', 'accuracy', 'f1_macro',
                'f1_macro_all', 'balanced_accuracy',
                'precision_macro', 'recall_macro', 'roc_auc', 'pr_auc',
                'train_time_sec', 'epochs_trained', 'train_samples', 'test_samples']
    key_cols = [c for c in key_cols if c in df.columns]

    csv_path = results_dir / 'all_experiments.csv'
    df[key_cols].to_csv(csv_path, index=False, float_format='%.4f')
    from src.colab_sync import sync_path
    sync_path(csv_path)
    print(f"\nAggregated {len(df)} results -> {csv_path}")

    # Print summary table.
    #
    # C6: the stored `f1_macro` column is NOT one comparable metric — for the
    # two binary datasets it is scikit's BINARY positive-class F1, and only for
    # dry_bean is it a true macro average. Print both columns, explicitly
    # labelled, so this console output cannot be mis-transcribed into the
    # write-up as a single cross-dataset comparison.
    print("\n=== Summary: F1 (%) by T2I Method and Architecture ===\n")
    print("NOTE: 'f1_positive_class' = positive-class F1 for the binary")
    print("      datasets breast_cancer (benign) / adult_income (>50K), and")
    print("      macro-F1 for the 7-class dry_bean (legacy key: f1_macro).")
    print("      'f1_macro_all' = macro-F1 over ALL classes — the only")
    print("      cross-dataset-comparable number.\n")
    if 't2i_method' in df.columns and 'cnn_arch' in df.columns:
        for value_col, title in (('f1_macro', 'f1_positive_class / macro (stored: f1_macro)'),
                                 ('f1_macro_all', 'f1_macro_all (comparable)'),
                                 ('balanced_accuracy', 'balanced accuracy')):
            if value_col not in df.columns or df[value_col].isna().all():
                print(f"--- {title} --- (not present; run "
                      f"scripts/backfill_metrics.py --write)\n")
                continue
            pivot = df.pivot_table(
                index=['dataset', 't2i_method'],
                columns='cnn_arch',
                values=value_col,
            )
            print(f"--- {title} ---")
            print((pivot * 100).round(2).to_string())
            print()

    return df


def main():
    import argparse

    global SCRATCH_3CH

    # Live output: a Colab cell gives Python a PIPE, which it block-buffers, so a
    # three-hour grid would print nothing until it finished. See
    # src.ablation._unbuffer_stdout for the same fix in the ablation path.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    from src.colab_sync import describe, sync_tree
    print(describe())

    parser = argparse.ArgumentParser(description='Run all experiments')
    parser.add_argument('--cnn-only', action='store_true',
                        help='Run only CNN experiments')
    parser.add_argument('--baselines', action='store_true',
                        help='Run only tabular baselines')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Run for one dataset only')
    parser.add_argument('--archs', type=str, default=None,
                        help='Comma-separated CNN architectures to run '
                             '(default: %s; add vit to re-include ViT-Base/16, '
                             'e.g. "shallow,resnet,resnet_scratch,vit")'
                             % ','.join(CNN_ARCHITECTURES))
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would run, do not train')
    parser.add_argument('--aggregate', action='store_true',
                        help='Aggregate existing results into CSV')
    parser.add_argument('--seed', type=int, default=42,
                        help='Training seed (default: 42 = the recorded grid). '
                             'Varying it also varies the split unless '
                             '--split-seed is given (audit C7/C8)')
    parser.add_argument('--split-seed', type=int, default=None,
                        help='Hold the train/val/test split at this seed while '
                             '--seed varies, to isolate training noise from '
                             'split noise (audit C7)')
    parser.add_argument('--scratch-3ch', action='store_true',
                        help='C11 control: give resnet_scratch the same '
                             '3-channel ImageNet-normalised input as the '
                             'pretrained arm. INVALIDATES the existing 12 '
                             '1-channel resnet_scratch cells')
    parser.add_argument('--output-dir', default='results',
                        help='Directory for result JSONs (default: results)')
    parser.add_argument('--cells', type=str, default=None,
                        help='Comma-separated dataset/t2i/arch triples to run, '
                             'e.g. adult_income/naive/resnet,adult_income/naive/shallow. '
                             'Use this to re-run a specific cell without touching '
                             'the rest of the grid (post-run validation, 2026-09-13).')
    parser.add_argument('--force', action='store_true',
                        help='Re-run selected cells even when a complete result '
                             'JSON already exists. The existing JSON and its '
                             '_model.pt are moved to results/backup_pre_rerun/ '
                             'first, so the original evidence survives.')
    args = parser.parse_args()

    if args.scratch_3ch:
        SCRATCH_3CH = True
        print('=' * 70)
        print('C11 CONTROL ENABLED: resnet_scratch now uses 3-channel '
              'ImageNet-normalised input.')
        print('The existing 1-channel resnet_scratch results are NOT '
              'comparable. Back them up first:')
        print('  mkdir -p results/backup_scratch_1ch && \\')
        print('    mv results/*resnet_scratch* results/backup_scratch_1ch/')
        print('=' * 70)

    results_dir = Path(args.output_dir)
    results_dir.mkdir(exist_ok=True)

    # Clean up any partial .json.tmp files from interrupted previous runs
    import os
    for tmp in results_dir.glob('*.json.tmp'):
        print(f"  Cleaning up partial file: {tmp.name}")
        os.remove(str(tmp))

    if args.aggregate:
        aggregate_results(output_dir=str(results_dir))
        # Mirror the whole tree: the backfilled per-cell JSONs are only copied
        # here, since the aggregate path returns before the closing sync pass.
        print(f"[sync] mirrored {sync_tree(results_dir)} file(s) from {results_dir}/") 
        return

    run_baselines = not args.cnn_only
    run_cnn = not args.baselines
    if args.cells:
        # --cells names dataset/t2i/arch triples, so the tabular baselines are
        # out of scope by construction: a targeted re-run of a CNN cell must not
        # be able to force-retrain (and overwrite) the baseline JSONs.
        if run_baselines:
            print('--cells given: baselines are out of scope and will not run.')
        run_baselines = False
        run_cnn = True

    if run_cnn:
        archs = CNN_ARCHITECTURES
        if args.archs:
            archs = [a.strip() for a in args.archs.split(',') if a.strip()]
            unknown = [a for a in archs if a not in ALL_ARCHITECTURES]
            if unknown:
                sys.exit(f"Unknown architecture(s): {unknown}. "
                         f"Valid: {ALL_ARCHITECTURES}")
        combos = list(itertools.product(DATASETS, T2I_METHODS, archs))
        if args.dataset:
            combos = [(d, t, c) for d, t, c in combos if d == args.dataset]

        if args.cells:
            wanted = parse_cells(args.cells)
            unknown = [c for c in wanted if c not in set(combos)]
            if unknown:
                valid = sorted({(a, b, c) for a, b, c in combos})
                sys.exit(f"--cells entries not in this run's grid: "
                         f"{['/'.join(u) for u in unknown]}\n"
                         f"Valid combinations here: "
                         f"{['/'.join(v) for v in valid]}")
            combos = [c for c in combos if c in wanted]
            print(f"--cells: {len(combos)} cell(s) selected")

        if args.dry_run:
            print(f"Would run {len(combos)} CNN experiments:")
            for d, t, c in combos:
                print(f"  {d} + {t} + {c}")
            if args.force:
                print("(--force: existing results would be backed up and re-run)")
        else:
            # Count already-done experiments for resume summary
            done_count = 0
            for dataset, t2i, cnn in combos:
                rf = results_dir / f"{dataset}_{t2i}_{cnn}.json"
                if _experiment_is_done(rf):
                    done_count += 1
            if done_count > 0:
                print(f"Resume: {done_count}/{len(combos)} already done, running {len(combos)-done_count} remaining")

            print(f"Running {len(combos)} CNN experiments...")
            for i, (dataset, t2i, cnn) in enumerate(combos, 1):
                result_file = results_dir / f"{dataset}_{t2i}_{cnn}.json"
                if _experiment_is_done(result_file) and not args.force:
                    print(f"\n[{i}/{len(combos)}] {dataset} + {t2i} + {cnn} — SKIP (done)")
                    continue
                print(f"\n[{i}/{len(combos)}] {dataset} + {t2i} + {cnn}")
                if args.force and result_file.exists():
                    moved = _backup_cell(result_file, results_dir)
                    print(f"  backed up to {results_dir}/backup_pre_rerun/: "
                          f"{', '.join(moved) if moved else 'nothing to move'}")
                try:
                    run_single_experiment(dataset, t2i, cnn,
                                          output_dir=str(results_dir),
                                          seed=args.seed,
                                          split_seed=args.split_seed)
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
                if _experiment_is_done(rf):
                    done_count += 1
            if done_count > 0:
                print(f"Resume: {done_count}/{len(combos)} already done, running {len(combos)-done_count} remaining")

            print(f"\nRunning {len(combos)} baseline experiments...")
            for i, (dataset, model) in enumerate(combos, 1):
                result_file = results_dir / f"baseline_{dataset}_{model}.json"
                if _experiment_is_done(result_file):
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
        aggregate_results(output_dir=str(results_dir))
        # Final sweep of the whole tree: picks up the aggregate CSV, any figure
        # and the weights of cells whose per-file mirror was skipped or failed.
        print(f"[sync] mirrored {sync_tree(results_dir)} file(s) from {results_dir}/ in the closing pass")


if __name__ == '__main__':
    main()
