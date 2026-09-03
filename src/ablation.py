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
    """Ablation 1: Compare original vs shuffled pixel performance.

    If shuffling drops F1 significantly -> spatial structure matters.
    If shuffling has minimal effect -> CNN ignores spatial layout.
    """
    from src.preprocessing import preprocess_dataset
    from src.t2i import T2ITransformer
    from src.train import prepare_loaders, train_model, compute_class_weights, set_global_seed
    from src.evaluate import evaluate_model
    from run_all import create_cnn_model, DATASET_CONFIG

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
        'epochs': 50, 'lr': 1e-3, 'weight_decay': 1e-4,
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
        'conclusion': 'spatial_structure_matters' if f1_drop > 0.02 else 'minimal_spatial_effect',
    }

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    with open(output_path / f'ablation_pixel_shuffling_{dataset}_{t2i_method}.json', 'w') as f:
        json.dump(result, f, indent=2)

    return result


# ============================================================
# ABLATION 2: Feature Ordering
# ============================================================

def reorder_features(X, order, y=None):
    """Reorder feature columns according to specified strategy.

    Args:
        X: (N, d) feature matrix
        order: 'original', 'random', 'correlation', 'reversed'
        y: labels (needed for 'correlation' ordering)

    Returns:
        X_reordered: (N, d) with features reordered
    """
    if order == 'original':
        return X
    elif order == 'reversed':
        return X[:, ::-1]
    elif order == 'random':
        rng = np.random.RandomState(42)
        perm = rng.permutation(X.shape[1])
        return X[:, perm]
    elif order == 'correlation':
        # Sort features by absolute correlation with target
        corrs = np.abs(np.array([
            np.corrcoef(X[:, i], y)[0, 1] if np.std(X[:, i]) > 0 else 0
            for i in range(X.shape[1])
        ]))
        sorted_idx = np.argsort(-corrs)  # descending
        return X[:, sorted_idx]
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
    from run_all import create_cnn_model, DATASET_CONFIG

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

    for order in orderings:
        print(f"\n  Ordering: {order}")
        set_global_seed(42)  # Reset for fair comparison

        # Reorder features
        X_train_r = reorder_features(X_train, order, y_train)
        X_val_r = reorder_features(X_val, order, y_val)
        X_test_r = reorder_features(X_test, order, y_test)

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
            'epochs': 50, 'lr': 1e-3, 'weight_decay': 1e-4,
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
    }
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    with open(output_path / f'ablation_feature_ordering_{dataset}_{t2i_method}.json', 'w') as f:
        json.dump(output, f, indent=2)

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
    from run_all import DATASET_CONFIG

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
    direct_config = {**base_config, 'lr': 1e-4}
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
    }

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    with open(output_path / f'ablation_lpft_{dataset}_{t2i_method}.json', 'w') as f:
        json.dump(output, f, indent=2)

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
    args = parser.parse_args()

    run_any = args.all or args.pixel_shuffle or args.feature_order or args.lpft

    if args.all or args.pixel_shuffle:
        run_pixel_shuffling_ablation(args.dataset, args.t2i, args.cnn)

    if args.all or args.feature_order:
        run_feature_ordering_ablation(args.dataset, args.t2i, args.cnn)

    if args.all or args.lpft:
        run_lpft_ablation(args.dataset, args.t2i)

    if not run_any:
        print("Specify --all, --pixel-shuffle, --feature-order, or --lpft")
        parser.print_help()


if __name__ == '__main__':
    main()
