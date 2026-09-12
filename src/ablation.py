"""
Ablation study: prove spatial structure matters for CNN performance.

Three ablations:
1. Pixel Shuffling: destroy spatial structure, measure accuracy drop
2. Feature Ordering: compare original vs random vs correlation-sorted
3. LP-FT vs Direct Fine-Tuning: training strategy comparison

Usage:
    python src/ablation.py --dataset breast_cancer --t2i deepinsight --cnn shallow
    python src/ablation.py --all --dataset breast_cancer
"""

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))


def _json_default(obj):
    """JSON fallback for numpy scalars/arrays (mirrors run_all.py)."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')


def _write_json_atomic(path, obj):
    """Write JSON via temp file + os.replace.

    Audit C19: the three ablation outputs used plain open(...,'w'), so a kill
    mid-write left a truncated JSON behind. run_all.py already cleans up
    `results/*.json.tmp` on startup, so the temp name matches its pattern.
    """
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(str(tmp), str(path))
    # Durability: mirror the finished file immediately, so an interrupted Colab
    # session loses at most the ablation in flight (see src/colab_sync.py).
    from src.colab_sync import sync_path
    sync_path(path)


# Keys that only exist in outputs written by the CURRENT code. Used by the
# resume check below: a file from before the C1/C3/C4 fixes lacks them, so it is
# correctly re-run instead of being trusted as complete.
ABLATION_MARKERS = {
    'pixel_shuffling': ('shuffled_train_f1', 'retrain_drop', 'arm_legend'),
    'feature_ordering': ('correlation_perm', 'ordering_note'),
    'lpft': ('seed', 'arm_note', 'direct_ft_config'),
}


def _ablation_is_current(path, kind):
    """True when an existing ablation JSON was produced by the current code.

    Resume-by-default (matching run_all.py) makes a crashed or timed-out run
    cheap to restart: completed ablations are skipped, and only the missing or
    stale ones are recomputed. Pass --force to recompute regardless.
    """
    if not path.exists():
        return False
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return all(k in data for k in ABLATION_MARKERS[kind])


# ============================================================
# ABLATION 1: Pixel Shuffling
# ============================================================

def shuffle_pixels(images, seed=42):
    """Shuffle pixel positions across all images (same permutation).

    WHY: If CNN relies on spatial structure (feature adjacency),
    shuffling pixels should destroy performance. If accuracy stays
    the same, the CNN is not using spatial relationships — the T2I
    transformation adds no value.

    The shuffle is applied identically to all samples (same permutation),
    preserving per-pixel intensity distributions but destroying spatial
    relationships between features.
    """
    rng = np.random.RandomState(seed)
    N, C, H, W = images.shape
    n_pixels = H * W

    # Create random permutation of pixel positions
    perm = rng.permutation(n_pixels)

    # Apply same permutation to all images
    shuffled = images.copy()
    for i in range(N):
        flat = shuffled[i, 0].reshape(n_pixels)
        flat = flat[perm]
        shuffled[i, 0] = flat.reshape(H, W)

    return shuffled


def run_pixel_shuffling_ablation(dataset, t2i_method, cnn_arch, output_dir='results'):
    """Ablation 1: does the CNN actually NEED the spatial layout?

    Two arms (FIX, audit C3):
    - Arm A (f1_drop): train on ORIGINAL images, evaluate on a shuffled test
      set. This is a distribution-shift / brittleness probe — it only shows
      that a model trained on the original layout does not survive a permuted
      input, which is not the same as showing that the layout carried signal.
    - Arm B (retrain_drop): train AND evaluate on the SAME pixel permutation.
      This is the structure test — if the shuffled-trained model recovers its
      F1, the layout was not required and the CNN is learning from marginal
      pixel statistics alone.
    """
    from src.preprocessing import preprocess_dataset
    from src.t2i import T2ITransformer
    from src.train import prepare_loaders, train_model, compute_class_weights, set_global_seed
    from src.evaluate import evaluate_model
    from run_all import create_cnn_model, DATASET_CONFIG, ARCH_LR

    set_global_seed(42)
    config = DATASET_CONFIG[dataset]
    num_classes = config['num_classes']
    image_size = config['image_size']

    print(f"\n=== Pixel Shuffling Ablation: {dataset} + {t2i_method} + {cnn_arch} ===")

    # Load data and generate images
    data = preprocess_dataset(dataset)
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

    t2i = T2ITransformer(method=t2i_method, image_size=image_size)
    t2i.fit(X_train, y_train)

    train_imgs = t2i.transform(X_train, y_train).numpy()
    val_imgs = t2i.transform(X_val, y_val).numpy()
    test_imgs = t2i.transform(X_test, y_test).numpy()

    # Train model on original images
    print("  Training model on original images...")
    train_loader, val_loader = prepare_loaders(train_imgs, y_train, val_imgs, y_val)
    model = create_cnn_model(cnn_arch, num_classes)
    class_weights = compute_class_weights(y_train)
    config_train = {
        'epochs': 50, 'lr': ARCH_LR[cnn_arch], 'weight_decay': 1e-4,
        'early_stopping_patience': 15, 'label_smoothing': 0.1,
        'class_weights': class_weights,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    model, history = train_model(model, train_loader, val_loader, config_train)

    # Evaluate on original test set
    test_loader = DataLoader(
        TensorDataset(torch.tensor(test_imgs), torch.tensor(y_test).long()),
        batch_size=32, shuffle=False,
    )
    original_metrics = evaluate_model(model, test_loader, num_classes)
    print(f"  Original:  F1={original_metrics['f1_macro']:.4f}, Acc={original_metrics['accuracy']:.4f}")

    # Evaluate on shuffled test set
    shuffled_test = shuffle_pixels(test_imgs, seed=42)
    shuffled_loader = DataLoader(
        TensorDataset(torch.tensor(shuffled_test), torch.tensor(y_test).long()),
        batch_size=32, shuffle=False,
    )
    shuffled_metrics = evaluate_model(model, shuffled_loader, num_classes)
    print(f"  Shuffled:  F1={shuffled_metrics['f1_macro']:.4f}, Acc={shuffled_metrics['accuracy']:.4f}")

    # Compute delta
    f1_drop = original_metrics['f1_macro'] - shuffled_metrics['f1_macro']
    acc_drop = original_metrics['accuracy'] - shuffled_metrics['accuracy']
    print(f"  Delta:     F1={f1_drop:+.4f}, Acc={acc_drop:+.4f}")

    # Arm B (FIX, audit C3): retrain on the shuffled images and evaluate on the
    # shuffled test set, i.e. train and test see the SAME permutation. This is
    # the actual "does spatial structure carry information" test.
    print("  Retraining on shuffled images (arm B)...")
    set_global_seed(42)
    shuffled_train = shuffle_pixels(train_imgs, seed=42)
    shuffled_val = shuffle_pixels(val_imgs, seed=42)
    sh_train_loader, sh_val_loader = prepare_loaders(
        shuffled_train, y_train, shuffled_val, y_val
    )
    model_shuffled = create_cnn_model(cnn_arch, num_classes)
    model_shuffled, _ = train_model(
        model_shuffled, sh_train_loader, sh_val_loader, dict(config_train)
    )
    shuffled_train_metrics = evaluate_model(model_shuffled, shuffled_loader, num_classes)
    print(f"  Shuffled-trained: F1={shuffled_train_metrics['f1_macro']:.4f}, "
          f"Acc={shuffled_train_metrics['accuracy']:.4f}")

    # The honest headline keys on arm B: destroying the layout at TRAIN time is
    # what tells us whether the layout was needed in the first place.
    retrain_drop = original_metrics['f1_macro'] - shuffled_train_metrics['f1_macro']
    if retrain_drop > 0.02:
        conclusion = 'spatial_structure_matters'
    else:
        conclusion = 'layout_not_required_marginals_sufficient'

    result = {
        'ablation': 'pixel_shuffling',
        'dataset': dataset,
        't2i_method': t2i_method,
        'cnn_arch': cnn_arch,
        'original_f1': original_metrics['f1_macro'],
        'shuffled_f1': shuffled_metrics['f1_macro'],
        'f1_drop': f1_drop,
        'original_acc': original_metrics['accuracy'],
        'shuffled_acc': shuffled_metrics['accuracy'],
        'acc_drop': acc_drop,
        # Arm B: trained AND evaluated on the same pixel permutation.
        'shuffled_train_f1': shuffled_train_metrics['f1_macro'],
        'shuffled_train_acc': shuffled_train_metrics['accuracy'],
        'retrain_drop': retrain_drop,
        'conclusion': conclusion,
        'arm_legend': {
            'f1_drop': 'train original -> test shuffled (brittleness / shift)',
            'retrain_drop': 'train shuffled -> test shuffled (structure test)',
        },
    }

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    _write_json_atomic(
        output_path / f'ablation_pixel_shuffling_{dataset}_{t2i_method}.json', result)

    return result


# ============================================================
# ABLATION 2: Feature Ordering
# ============================================================

def correlation_order(X, y):
    """Column order by descending |corr(feature, target)|.

    FIX (audit C1): compute this ONCE from the training split and reuse the
    result for train/val/test. Recomputing it per split was the C1 bug — each
    split then received a DIFFERENT column order, so the T2I layout no longer
    aligned and test feature values were written at train-derived pixel
    positions (2.34% of test pixels differed on breast_cancer, producing the
    spurious 4.48/2.30/2.27 pp "correlation-sorted is worst" result).
    """
    corrs = np.abs(np.array([
        np.corrcoef(X[:, i], y)[0, 1] if np.std(X[:, i]) > 0 else 0.0
        for i in range(X.shape[1])
    ]))
    return np.argsort(-corrs)  # descending


def reorder_features(X, order, y=None, perm=None):
    """Reorder feature columns according to specified strategy.

    Args:
        X: (N, d) feature matrix
        order: 'original', 'random', 'correlation', 'reversed'
        y: labels (fallback for 'correlation' when `perm` is None)
        perm: precomputed permutation for 'correlation' (see correlation_order).
            Pass the SAME permutation to every split, derived from the training
            split. Recomputing per split scrambles the train/test alignment
            (audit C1); the y-based fallback below is only safe for a
            single-split sanity check.

    Returns:
        X_reordered: (N, d) with features reordered
    """
    if order == 'original':
        return X
    elif order == 'reversed':
        return X[:, ::-1]
    elif order == 'random':
        rng = np.random.RandomState(42)  # re-seeded -> same permutation per split
        perm = rng.permutation(X.shape[1])
        return X[:, perm]
    elif order == 'correlation':
        if perm is None:
            if y is None:
                raise ValueError("'correlation' ordering needs a precomputed "
                                 "perm or labels y")
            perm = correlation_order(X, y)
        return X[:, perm]
    else:
        raise ValueError(f"Unknown order: {order}")


def run_feature_ordering_ablation(dataset, t2i_method, cnn_arch, output_dir='results'):
    """Ablation 2: Compare different feature orderings.

    If correlation-sorted > random -> feature arrangement matters.
    If all orderings similar -> CNN is robust to layout changes.
    """
    from src.preprocessing import preprocess_dataset
    from src.t2i import T2ITransformer
    from src.train import prepare_loaders, train_model, compute_class_weights, set_global_seed
    from src.evaluate import evaluate_model
    from run_all import create_cnn_model, DATASET_CONFIG, ARCH_LR

    set_global_seed(42)
    config = DATASET_CONFIG[dataset]
    num_classes = config['num_classes']
    image_size = config['image_size']

    print(f"\n=== Feature Ordering Ablation: {dataset} + {t2i_method} + {cnn_arch} ===")

    data = preprocess_dataset(dataset)
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

    results = {}
    orderings = ['original', 'random', 'correlation', 'reversed']

    # FIX (audit C1): ONE permutation from the training split, applied to every
    # split. Previously each split was sorted by its own labels, so
    # train/val/test received different column orders and the "correlation"
    # cell measured a split misalignment rather than a layout effect.
    corr_perm = correlation_order(X_train, y_train)

    for order in orderings:
        print(f"\n  Ordering: {order}")
        set_global_seed(42)  # Reset for fair comparison

        # Reorder features with the SAME permutation on all three splits
        X_train_r = reorder_features(X_train, order, y_train, perm=corr_perm)
        X_val_r = reorder_features(X_val, order, y_val, perm=corr_perm)
        X_test_r = reorder_features(X_test, order, y_test, perm=corr_perm)

        # Fit T2I on reordered features
        t2i = T2ITransformer(method=t2i_method, image_size=image_size)
        t2i.fit(X_train_r, y_train)

        train_imgs = t2i.transform(X_train_r, y_train).numpy()
        val_imgs = t2i.transform(X_val_r, y_val).numpy()
        test_imgs = t2i.transform(X_test_r, y_test).numpy()

        # Train and evaluate
        train_loader, val_loader = prepare_loaders(train_imgs, y_train, val_imgs, y_val)
        model = create_cnn_model(cnn_arch, num_classes)
        class_weights = compute_class_weights(y_train)
        config_train = {
            'epochs': 50, 'lr': ARCH_LR[cnn_arch], 'weight_decay': 1e-4,
            'early_stopping_patience': 15, 'label_smoothing': 0.1,
            'class_weights': class_weights,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        }
        model, _ = train_model(model, train_loader, val_loader, config_train)

        test_loader = DataLoader(
            TensorDataset(torch.tensor(test_imgs), torch.tensor(y_test).long()),
            batch_size=32, shuffle=False,
        )
        metrics = evaluate_model(model, test_loader, num_classes)
        results[order] = {
            'f1': metrics['f1_macro'],
            'accuracy': metrics['accuracy'],
        }
        print(f"    F1={metrics['f1_macro']:.4f}, Acc={metrics['accuracy']:.4f}")

    # Save
    output = {
        'ablation': 'feature_ordering',
        'dataset': dataset,
        't2i_method': t2i_method,
        'cnn_arch': cnn_arch,
        'results': results,
        'best_ordering': max(results, key=lambda k: results[k]['f1']),
        'worst_ordering': min(results, key=lambda k: results[k]['f1']),
        # Audit C1 traceability: the correlation key is now a single
        # train-derived permutation reused across splits, so for a
        # column-order-invariant transform (DeepInsight) all four orderings are
        # expected to collapse to identical F1.
        'ordering_note': 'correlation permutation computed on train only and '
                         'applied to train/val/test',
        'correlation_perm': corr_perm.tolist(),
    }
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    _write_json_atomic(
        output_path / f'ablation_feature_ordering_{dataset}_{t2i_method}.json', output)

    print(f"\n  Best: {output['best_ordering']}, Worst: {output['worst_ordering']}")
    return output


# ============================================================
# ABLATION 3: LP-FT vs Direct Fine-Tuning
# ============================================================

def run_lpft_ablation(dataset, t2i_method, output_dir='results'):
    """Ablation 3: Compare LP-FT training vs direct fine-tuning.

    LP-FT (current approach):
    Phase 1: freeze backbone, train head
    Phase 2: unfreeze all, train with low LR

    Direct FT:
    Train all layers from the start with uniform LR

    LP-FT should be more stable, especially on small datasets.
    """
    from src.preprocessing import preprocess_dataset
    from src.t2i import T2ITransformer
    from src.train import (
        prepare_loaders, train_model, train_lp_ft,
        compute_class_weights, set_global_seed
    )
    from src.evaluate import evaluate_model
    from src.models.resnet_wrapper import ResNetWrapper
    from run_all import DATASET_CONFIG, ARCH_LR

    set_global_seed(42)
    config = DATASET_CONFIG[dataset]
    num_classes = config['num_classes']
    image_size = config['image_size']

    print(f"\n=== LP-FT Ablation: {dataset} + {t2i_method} ===")

    data = preprocess_dataset(dataset)
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

    # Generate images
    t2i = T2ITransformer(method=t2i_method, image_size=image_size)
    t2i.fit(X_train, y_train)
    train_imgs = t2i.transform(X_train, y_train).numpy()
    val_imgs = t2i.transform(X_val, y_val).numpy()
    test_imgs = t2i.transform(X_test, y_test).numpy()

    class_weights = compute_class_weights(y_train)
    base_config = {
        'epochs': 50, 'weight_decay': 1e-4,
        'label_smoothing': 0.1, 'class_weights': class_weights,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    results = {}

    # --- Direct Fine-Tuning ---
    print("  Training: Direct Fine-Tuning...")
    set_global_seed(42)
    train_loader, val_loader = prepare_loaders(train_imgs, y_train, val_imgs, y_val)
    # pretrained=True needs input_channels=3 (imagenet_normalize produces RGB)
    model_direct = ResNetWrapper(num_classes=num_classes, pretrained=True, input_channels=3)
    # Align with the main table: pretrained ResNet-18 trains at 1e-3 in
    # run_all (ARCH_LR). This cell reproduces the main-table resnet row,
    # so it must use the same LR or the ablation figure contradicts the
    # results table (was 1e-4 -> direct-FT F1 0.935 vs main 0.972 on
    # breast_cancer/deepinsight).
    direct_config = {**base_config, 'lr': ARCH_LR['resnet']}
    model_direct, hist_direct = train_model(model_direct, train_loader, val_loader, direct_config)

    test_loader = DataLoader(
        TensorDataset(torch.tensor(test_imgs), torch.tensor(y_test).long()),
        batch_size=32, shuffle=False,
    )
    direct_metrics = evaluate_model(model_direct, test_loader, num_classes)
    results['direct_ft'] = {
        'f1': direct_metrics['f1_macro'],
        'accuracy': direct_metrics['accuracy'],
        'epochs': len(hist_direct['train_loss']),
    }
    print(f"    Direct FT: F1={direct_metrics['f1_macro']:.4f}, "
          f"Acc={direct_metrics['accuracy']:.4f}, Epochs={len(hist_direct['train_loss'])}")

    # --- LP-FT ---
    print("  Training: LP-FT (Linear Probing + Fine-Tuning)...")
    set_global_seed(42)
    train_loader, val_loader = prepare_loaders(train_imgs, y_train, val_imgs, y_val)
    model_lpft = ResNetWrapper(num_classes=num_classes, pretrained=True, input_channels=3)
    lpft_config = {**base_config, 'lr': 1e-3, 'lr_ft': 1e-4}
    model_lpft, hist_lpft = train_lp_ft(
        model_lpft, train_loader, val_loader, lpft_config,
        lp_epochs=10, ft_epochs=40,
    )

    lpft_metrics = evaluate_model(model_lpft, test_loader, num_classes)
    results['lp_ft'] = {
        'f1': lpft_metrics['f1_macro'],
        'accuracy': lpft_metrics['accuracy'],
        'epochs': len(hist_lpft['train_loss']),
    }
    print(f"    LP-FT:     F1={lpft_metrics['f1_macro']:.4f}, "
          f"Acc={lpft_metrics['accuracy']:.4f}, Epochs={len(hist_lpft['train_loss'])}")

    # --- Summary ---
    f1_diff = results['lp_ft']['f1'] - results['direct_ft']['f1']
    print(f"\n  LP-FT vs Direct: F1={f1_diff:+.4f}")

    output = {
        'ablation': 'lpft_vs_direct',
        'dataset': dataset,
        't2i_method': t2i_method,
        'results': results,
        'lpft_f1_advantage': f1_diff,
        'conclusion': 'lpft_better' if f1_diff > 0.01 else ('direct_better' if f1_diff < -0.01 else 'comparable'),
        # Audit C4 traceability: the direct-FT arm is a SEPARATE run from the
        # main-table resnet cell, not a reproduction of it (same config gave
        # 98.63 here vs 97.22 in the main run on breast_cancer). Record the
        # exact protocol so the figure/table can be reconciled or caveated.
        'seed': 42,
        'direct_ft_config': {
            'lr': ARCH_LR['resnet'], 'epochs': 50,
            'early_stopping_patience': 15, 'label_smoothing': 0.1,
        },
        'lpft_config': {
            'lr': 1e-3, 'lr_ft': 1e-4, 'lp_epochs': 10, 'ft_epochs': 40,
        },
        'arm_note': 'direct_ft is an independent run, not a reproduction of the '
                    'main-table resnet cell',
    }

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    _write_json_atomic(
        output_path / f'ablation_lpft_{dataset}_{t2i_method}.json', output)

    return output


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Ablation study')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name')
    parser.add_argument('--t2i', type=str, default='deepinsight',
                        help='T2I method')
    parser.add_argument('--cnn', type=str, default='shallow',
                        help='CNN architecture')
    parser.add_argument('--all', action='store_true',
                        help='Run all three ablations')
    parser.add_argument('--pixel-shuffle', action='store_true',
                        help='Run pixel shuffling ablation only')
    parser.add_argument('--feature-order', action='store_true',
                        help='Run feature ordering ablation only')
    parser.add_argument('--lpft', action='store_true',
                        help='Run LP-FT comparison only')
    parser.add_argument('--force', action='store_true',
                        help='Recompute even if an up-to-date result already '
                             'exists (default: resume and skip those)')
    args = parser.parse_args()

    from src.colab_sync import describe
    print(describe())

    run_any = args.all or args.pixel_shuffle or args.feature_order or args.lpft
    if not run_any:
        print("Specify --all, --pixel-shuffle, --feature-order, or --lpft")
        parser.print_help()
        return

    def pending(kind, filename):
        """Whether this ablation still needs to run."""
        path = Path('results') / filename
        if args.force:
            return True
        if _ablation_is_current(path, kind):
            print(f"  {filename} — SKIP (already up to date; use --force to redo)")
            return False
        return True

    if args.all or args.pixel_shuffle:
        if pending('pixel_shuffling',
                   f'ablation_pixel_shuffling_{args.dataset}_{args.t2i}.json'):
            run_pixel_shuffling_ablation(args.dataset, args.t2i, args.cnn)

    if args.all or args.feature_order:
        if pending('feature_ordering',
                   f'ablation_feature_ordering_{args.dataset}_{args.t2i}.json'):
            run_feature_ordering_ablation(args.dataset, args.t2i, args.cnn)

    if args.all or args.lpft:
        if pending('lpft', f'ablation_lpft_{args.dataset}_{args.t2i}.json'):
            run_lpft_ablation(args.dataset, args.t2i)


if __name__ == '__main__':
    main()
