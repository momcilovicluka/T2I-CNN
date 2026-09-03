"""
Visualization pipeline for Chapter 4 figures.

Reads experiment results from results/*.json and generates publication-ready figures.

Usage:
    python src/visualize.py                    # All figures
    python src/visualize.py --heatmap-only     # Just main results heatmap
    python src/visualize.py --ablation-only    # Just ablation figures
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent.parent))


DATASETS = ['breast_cancer', 'dry_bean', 'adult_income']
DATASET_LABELS = {
    'breast_cancer': 'Breast Cancer\n(30 features, 2 classes)',
    'dry_bean': 'Dry Bean\n(16 features, 7 classes)',
    'adult_income': 'Adult Income\n(108 features, 2 classes)',
}
T2I_METHODS = ['naive', 'tinto', 'deepinsight', 'igtd']  # s_igtd dropped (duplicates igtd, PART 13i)

# Honest F1 label per dataset (PART 13e resolution): 'macro' is only true
# for the 7-class dry_bean; the two binary datasets store scikit 'binary'
# F1 (positive class only: benign for breast, >50K for adult).
F1_LABEL = {
    'breast_cancer': 'F1 (positive class, %)',
    'dry_bean': 'Macro-F1 (%)',
    'adult_income': 'F1 (positive class, %)',
}
T2I_LABELS = {
    'naive': 'Naive', 'tinto': 'TINTO', 'deepinsight': 'DeepInsight',
    'igtd': 'IGTD',
}
CNN_ARCHS = ['shallow', 'resnet', 'resnet_scratch', 'vit']
CNN_LABELS = {
    'shallow': 'ShallowCNN',
    'resnet': 'ResNet-18\n(pretrained)',
    'resnet_scratch': 'ResNet-18\n(from scratch)',
    'vit': 'ViT-Base',
}
BASELINES = ['rf', 'xgboost', 'mlp']
BASELINE_LABELS = {'rf': 'RF', 'xgboost': 'XGBoost', 'mlp': 'MLP'}


def load_results(results_dir='results'):
    """Load all experiment results from JSON files."""
    results = []
    results_path = Path(results_dir)
    for json_file in sorted(results_path.glob('*.json')):
        if json_file.name.startswith('ablation_'):
            continue
        try:
            with open(json_file) as f:
                data = json.load(f)
            results.append(data)
        except Exception as e:
            print(f"  Warning: Could not load {json_file}: {e}")
    return results


def _load_t2i_pixel_range(results_dir, dataset, t2i_method, cnn_arch):
    """Load the train-derived pixel scale recorded by run_all.py.

    run_all.py saves t2i_pixel_range=[min, max] for TINTO experiments so
    Grad-CAM figures can reproduce the exact scaling the CNN was trained on
    (TINTO caches [0,1] stats on its first transform call — which must be
    the training split). Returns None if absent.
    """
    json_file = Path(results_dir) / f"{dataset}_{t2i_method}_{cnn_arch}.json"
    if not json_file.exists():
        return None
    try:
        with open(json_file) as f:
            data = json.load(f)
        rng = data.get('t2i_pixel_range')
        if rng is not None and len(rng) == 2:
            return (float(rng[0]), float(rng[1]))
    except Exception:
        pass
    return None


def load_ablation_results(results_dir='results', prefix='ablation_'):
    """Load ablation results from JSON files."""
    results = []
    results_path = Path(results_dir)
    for json_file in sorted(results_path.glob(f'{prefix}*.json')):
        try:
            with open(json_file) as f:
                data = json.load(f)
            results.append(data)
        except Exception as e:
            print(f"  Warning: Could not load {json_file}: {e}")
    return results


# ============================================================
# Figure 4.3: Main Results Heatmap
# ============================================================

def plot_main_results_heatmap(results, output_dir='results/figures'):
    """Heatmap: rows=T2I methods, columns=CNN architectures, color=F1.

    One heatmap per dataset. This is the most important figure.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Filter to CNN results only (exclude baselines)
    cnn_results = [r for r in results if r.get('cnn_arch') in CNN_ARCHS]

    for dataset in DATASETS:
        ds_results = [r for r in cnn_results if r['dataset'] == dataset]
        if not ds_results:
            print(f"  No CNN results for {dataset}, skipping heatmap")
            continue

        # Build matrix
        matrix = np.full((len(T2I_METHODS), len(CNN_ARCHS)), np.nan)
        for r in ds_results:
            i = T2I_METHODS.index(r['t2i_method'])
            j = CNN_ARCHS.index(r['cnn_arch'])
            matrix[i, j] = r['f1_macro'] * 100

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))

        # Custom colormap: white (low) -> blue (high)
        cmap = LinearSegmentedColormap.from_list('f1', ['#f0f0f0', '#2166ac', '#053061'])

        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=50, vmax=100)

        # Labels
        ax.set_xticks(range(len(CNN_ARCHS)))
        ax.set_xticklabels([CNN_LABELS[a] for a in CNN_ARCHS], fontsize=10)
        ax.set_yticks(range(len(T2I_METHODS)))
        ax.set_yticklabels([T2I_LABELS[m] for m in T2I_METHODS], fontsize=11)

        # Annotate cells
        for i in range(len(T2I_METHODS)):
            for j in range(len(CNN_ARCHS)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = 'white' if val > 85 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                            fontsize=12, fontweight='bold', color=color)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, label=F1_LABEL[dataset])
        ax.set_title(f'{DATASET_LABELS[dataset]} — {F1_LABEL[dataset]}',
                     fontsize=13, fontweight='bold', pad=12)

        plt.tight_layout()
        path = output_path / f'ch4_heatmap_{dataset}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved: {path.name}")


# ============================================================
# Figure 4.4: Baseline Comparison Bar Chart
# ============================================================

def plot_baseline_comparison(results, output_dir='results/figures'):
    """Grouped bar chart: RF, XGBoost, MLP, best CNN per dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx]

        # Get baseline F1s
        baseline_f1s = []
        baseline_names = []
        for b in BASELINES:
            b_results = [r for r in results if r['dataset'] == dataset
                         and r.get('cnn_arch') == b]
            if b_results:
                baseline_f1s.append(b_results[0]['f1_macro'] * 100)
                baseline_names.append(BASELINE_LABELS[b])

        # Get best CNN F1
        cnn_results = [r for r in results if r['dataset'] == dataset
                       and r.get('cnn_arch') in CNN_ARCHS]
        if cnn_results:
            best_cnn = max(cnn_results, key=lambda r: r['f1_macro'])
            baseline_f1s.append(best_cnn['f1_macro'] * 100)
            baseline_names.append(f"Best CNN\n({T2I_LABELS[best_cnn['t2i_method']]})")

        if not baseline_f1s:
            continue

        colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b2']
        bars = ax.bar(range(len(baseline_f1s)), baseline_f1s,
                      color=colors[:len(baseline_f1s)], edgecolor='white', linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, baseline_f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xticks(range(len(baseline_names)))
        ax.set_xticklabels(baseline_names, fontsize=9)
        ax.set_ylim(0, 105)
        ax.set_ylabel(F1_LABEL[dataset] if idx == 0 else '')
        ax.set_title(DATASET_LABELS[dataset], fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Baseline Comparison: Tabular vs CNN+T2I',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = output_path / 'ch4_baseline_comparison.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ============================================================
# Figure 4.5: Per-Class F1 (Dry Bean)
# ============================================================

def plot_per_class_f1(results, output_dir='results/figures'):
    """Per-class F1 for Dry Bean (7 classes) grouped by T2I method."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = 'dry_bean'
    ds_results = [r for r in results if r['dataset'] == dataset
                  and r.get('cnn_arch') in CNN_ARCHS]
    if not ds_results:
        print(f"  No results for {dataset}, skipping per-class F1")
        return

    # Use best CNN per T2I method
    best_per_method = {}
    for method in T2I_METHODS:
        method_results = [r for r in ds_results if r['t2i_method'] == method]
        if method_results:
            best_per_method[method] = max(method_results, key=lambda r: r['f1_macro'])

    if not best_per_method:
        return

    class_names = ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA']

    # Extract per-class F1 from classification reports
    # For now, use confusion matrix to approximate per-class metrics
    fig, ax = plt.subplots(figsize=(12, 5))

    n_methods = len(best_per_method)
    n_classes = len(class_names)
    x = np.arange(n_classes)
    width = 0.8 / n_methods

    colors = plt.cm.Set2(np.linspace(0, 1, n_methods))

    for idx, (method, data) in enumerate(best_per_method.items()):
        # Parse classification report to get per-class F1
        report = data.get('classification_report', '')
        per_class_f1 = []
        for cls_name in class_names:
            # Try to extract F1 from report
            try:
                lines = report.split('\n')
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5 and cls_name.lower() in line.lower():
                        per_class_f1.append(float(parts[4]))
                        break
                else:
                    per_class_f1.append(0.0)
            except:
                per_class_f1.append(0.0)

        bars = ax.bar(x + idx * width, per_class_f1, width,
                      label=T2I_LABELS[method], color=colors[idx], edgecolor='white')

    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('F1-Score')
    ax.set_title(f'Dry Bean — Per-Class F1 by T2I Method (Best CNN)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = output_path / 'ch4_per_class_f1_dry_bean.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ============================================================
# Figure 4.6: Training Curves
# ============================================================

def plot_training_curves(results, output_dir='results/figures'):
    """Training curves: loss/accuracy vs epoch for each architecture."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        ds_results = [r for r in results if r['dataset'] == dataset
                      and r.get('cnn_arch') in CNN_ARCHS
                      and r.get('history') is not None]
        if not ds_results:
            print(f"  No training histories for {dataset}, skipping")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Use one T2I method (deepinsight) for comparison
        method = 'deepinsight'
        method_results = [r for r in ds_results if r['t2i_method'] == method]

        colors = {'shallow': '#4c72b0', 'resnet': '#55a868',
                  'resnet_scratch': '#c44e52', 'vit': '#8172b2'}

        for r in method_results:
            hist = r['history']
            arch = r['cnn_arch']
            epochs = range(1, len(hist['train_loss']) + 1)

            axes[0].plot(epochs, hist['train_loss'], '--', color=colors.get(arch, 'gray'),
                         alpha=0.7, linewidth=1)
            axes[0].plot(epochs, hist['val_loss'], '-', color=colors.get(arch, 'gray'),
                         label=CNN_LABELS[arch], linewidth=2)

            axes[1].plot(epochs, hist['val_acc'], '-', color=colors.get(arch, 'gray'),
                         label=CNN_LABELS[arch], linewidth=2)

        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss (train: dashed, val: solid)', fontsize=11)
        axes[0].legend(fontsize=9)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Validation Accuracy')
        axes[1].set_title('Validation Accuracy', fontsize=11)
        axes[1].legend(fontsize=9)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        fig.suptitle(f'{DATASET_LABELS[dataset]} — DeepInsight',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        path = output_path / f'ch4_training_curves_{dataset}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved: {path.name}")


# ============================================================
# Figure 4.7: Confusion Matrices
# ============================================================

def plot_confusion_matrices(results, output_dir='results/figures'):
    """Confusion matrices for best CNN per dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx]
        ds_results = [r for r in results if r['dataset'] == dataset
                      and r.get('cnn_arch') in CNN_ARCHS]
        if not ds_results:
            continue

        # Use best overall result
        best = max(ds_results, key=lambda r: r['f1_macro'])
        cm = np.array(best['confusion_matrix'])

        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(f"{DATASET_LABELS[dataset].split(chr(10))[0]}\n"
                     f"({T2I_LABELS[best['t2i_method']]} + {CNN_LABELS[best['cnn_arch']].split(chr(10))[0]})",
                     fontsize=10, fontweight='bold')

        # Annotate
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black',
                        fontsize=12, fontweight='bold')

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Confusion Matrices — Best CNN per Dataset',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = output_path / 'ch4_confusion_matrices.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ============================================================
# Figure 4.8: Ablation Results
# ============================================================

def plot_ablation_results(output_dir='results/figures'):
    """Pixel shuffling and LP-FT ablation results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Pixel Shuffling
    shuffle_results = load_ablation_results('results', 'ablation_pixel_shuffling_')
    if shuffle_results:
        fig, ax = plt.subplots(figsize=(8, 5))

        datasets_with_data = list(set(r['dataset'] for r in shuffle_results))
        n = len(datasets_with_data)
        x = np.arange(n)
        width = 0.35

        original_f1s = []
        shuffled_f1s = []
        labels = []

        for ds in datasets_with_data:
            ds_results = [r for r in shuffle_results if r['dataset'] == ds]
            if ds_results:
                r = ds_results[0]
                original_f1s.append(r['original_f1'] * 100)
                shuffled_f1s.append(r['shuffled_f1'] * 100)
                labels.append(DATASET_LABELS[ds].split('\n')[0])

        bars1 = ax.bar(x - width/2, original_f1s, width, label='Original',
                       color='#4c72b0', edgecolor='white')
        bars2 = ax.bar(x + width/2, shuffled_f1s, width, label='Shuffled',
                       color='#c44e52', edgecolor='white')

        for bar, val in zip(bars1, original_f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        for bar, val in zip(bars2, shuffled_f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('F1 (%)')  # shared axis across datasets: macro for dry_bean, positive-class otherwise (PART 13e)
        ax.set_title('Ablation: Pixel Shuffling (DeepInsight + ShallowCNN)',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        path = output_path / 'ch4_ablation_pixel_shuffling.png'
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved: {path.name}")

    # LP-FT vs Direct FT
    lpft_results = load_ablation_results('results', 'ablation_lpft_')
    if lpft_results:
        fig, ax = plt.subplots(figsize=(8, 5))

        datasets_with_data = list(set(r['dataset'] for r in lpft_results))
        n = len(datasets_with_data)
        x = np.arange(n)
        width = 0.35

        direct_f1s = []
        lpft_f1s = []
        labels = []

        for ds in datasets_with_data:
            ds_results = [r for r in lpft_results if r['dataset'] == ds]
            if ds_results:
                r = ds_results[0]
                direct_f1s.append(r['results']['direct_ft']['f1'] * 100)
                lpft_f1s.append(r['results']['lp_ft']['f1'] * 100)
                labels.append(DATASET_LABELS[ds].split('\n')[0])

        bars1 = ax.bar(x - width/2, direct_f1s, width, label='Direct FT',
                       color='#4c72b0', edgecolor='white')
        bars2 = ax.bar(x + width/2, lpft_f1s, width, label='LP-FT',
                       color='#55a868', edgecolor='white')

        for bar, val in zip(bars1, direct_f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        for bar, val in zip(bars2, lpft_f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('Macro-F1 (%)')
        ax.set_title('Ablation: LP-FT vs Direct Fine-Tuning (ResNet-18 pretrained)',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        path = output_path / 'ch4_ablation_lpft.png'
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved: {path.name}")


# ============================================================
# Figure 4.10: Feature Density vs Performance
# ============================================================

def plot_density_vs_performance(results, output_dir='results/figures'):
    """Scatter: feature density (x) vs F1 (y), colored by method."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Compute feature density for each (dataset, method) combo
    # density = n_features / image_size^2 * 100
    cnn_results = [r for r in results if r.get('cnn_arch') in CNN_ARCHS]
    if not cnn_results:
        print("  No CNN results for density plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    marker_map = {'shallow': 'o', 'resnet': 's', 'resnet_scratch': '^', 'vit': 'D'}
    color_map = {'naive': '#c44e52', 'tinto': '#8172b2', 'deepinsight': '#4c72b0',
                 'igtd': '#55a868'}

    for r in cnn_results:
        # Estimate density from dataset and image_size
        feature_counts = {'breast_cancer': 30, 'dry_bean': 16, 'adult_income': 108}
        n_feat = feature_counts.get(r['dataset'], 30)
        img_size = r.get('image_size', 32)
        density = n_feat / (img_size * img_size) * 100

        ax.scatter(density, r['f1_macro'] * 100,
                   c=color_map.get(r['t2i_method'], 'gray'),
                   marker=marker_map.get(r['cnn_arch'], 'o'),
                   s=80, alpha=0.7, edgecolors='white', linewidth=0.5)

    # Add legend
    from matplotlib.lines import Line2D
    method_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[m],
                             markersize=10, label=T2I_LABELS[m]) for m in T2I_METHODS]
    arch_handles = [Line2D([0], [0], marker=marker_map[a], color='w', markerfacecolor='gray',
                           markersize=10, label=CNN_LABELS[a].split('\n')[0]) for a in CNN_ARCHS]

    leg1 = ax.legend(handles=method_handles, title='T2I Method', loc='lower right', fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=arch_handles, title='Architecture', loc='lower center', fontsize=9)

    ax.set_xlabel('Feature Density (%)')
    ax.set_ylabel('Macro-F1 (%)')
    ax.set_title('Feature Density vs Classification Performance',
                 fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = output_path / 'ch4_density_vs_performance.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ============================================================
# Figure: Grad-CAM Heatmaps
# ============================================================

def plot_gradcam_grid(results_dir='results', output_dir='results/figures', n_samples=2):
    """Generate Grad-CAM heatmaps for each T2I method.

    WHY: The most important interpretability figure. Shows whether
    the CNN actually uses the spatial structure created by T2I methods.
    If DeepInsight's heatmap shows spread-out attention while Naive's
    shows one corner, it proves T2I spatial mapping matters.

    Layout per dataset: rows = T2I methods, columns = original | overlay | heatmap
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    import torch
    sys.path.insert(0, str(Path(__file__).parent))
    from src.preprocessing import preprocess_dataset
    from src.t2i import T2ITransformer
    from src.gradcam import generate_gradcam, overlay_heatmap
    from src.models.shallow_cnn import ShallowCNN
    from src.models.resnet_wrapper import ResNetWrapper

    # Use ShallowCNN for Grad-CAM (most interpretable, standard conv layers)
    arch = 'shallow'
    model_class = ShallowCNN

    for dataset in DATASETS:
        print(f"\nGrad-CAM for {dataset}...")

        # Load data
        data = preprocess_dataset(dataset)
        X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
        y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
        num_classes = len(np.unique(y_train))

        # Sample balanced test samples
        sample_indices = []
        for c in range(num_classes):
            class_indices = np.where(y_test == c)[0]
            chosen = np.random.RandomState(42).choice(
                class_indices, size=min(n_samples, len(class_indices)), replace=False
            )
            sample_indices.extend(chosen)
        sample_indices = sample_indices[:n_samples * num_classes]

        # Prepare figure: rows = methods, columns = original/overlay/heatmap
        n_methods = len(T2I_METHODS)
        n_show = len(sample_indices)
        fig, axes = plt.subplots(
            n_methods, n_show * 3,
            figsize=(4 * n_show * 3, 3.5 * n_methods)
        )
        if n_methods == 1:
            axes = axes.reshape(1, -1)

        for row, method in enumerate(T2I_METHODS):
            # Load model
            model = model_class(num_classes=num_classes)
            model_path = Path(results_dir) / f"{dataset}_{method}_{arch}_model.pt"
            if not model_path.exists():
                print(f"  No model for {method}, skipping")
                for col in range(n_show * 3):
                    axes[row, col].axis('off')
                continue

            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()

            # Fit T2I on train
            t2i = T2ITransformer(method=method, image_size=32)
            t2i.fit(X_train, y_train)

            # Restore TINTO's train-derived pixel scale recorded by run_all.py.
            # FIX (audit): TINTO caches [0,1] scaling stats on its FIRST
            # transform call. run_all.py trained on train-derived scaling and
            # saved it in the results JSON. Transforming test samples first
            # would cache test-derived stats (max 0.23 vs 0.30 for breast
            # cancer) and display differently-scaled images than the CNN saw.
            t2i_pix_range = _load_t2i_pixel_range(results_dir, dataset, method, arch)
            if t2i_pix_range is not None:
                t2i.transformer._pix_min = float(t2i_pix_range[0])
                t2i.transformer._pix_max = float(t2i_pix_range[1])
            elif getattr(t2i.transformer, '_pix_min', None) is None and hasattr(t2i.transformer, '_pix_min'):
                # Legacy results lack the field: seed the cache exactly the way
                # run_all.py did (full training transform) before test samples.
                t2i.transform(X_train, y_train)

            # Transform test samples
            sample_X = X_test[sample_indices]
            sample_y = y_test[sample_indices]
            sample_imgs = t2i.transform(sample_X, sample_y)

            for col_idx, s_idx in enumerate(range(n_show)):
                img = sample_imgs[s_idx]  # (1, 32, 32)
                true_label = sample_y[s_idx]

                # Get prediction
                with torch.no_grad():
                    output = model(img.unsqueeze(0))
                    pred = output.argmax(dim=1).item()
                    conf = torch.softmax(output, dim=1)[0, pred].item()

                # Generate Grad-CAM
                try:
                    heatmap = generate_gradcam(model, img, pred, arch)
                except Exception as e:
                    print(f"  Grad-CAM failed for {method} sample {s_idx}: {e}")
                    axes[row, col_idx * 3].axis('off')
                    axes[row, col_idx * 3 + 1].axis('off')
                    axes[row, col_idx * 3 + 2].axis('off')
                    continue

                # Original image
                orig = img.squeeze().numpy()
                axes[row, col_idx * 3].imshow(orig, cmap='gray', vmin=0, vmax=1)
                axes[row, col_idx * 3].set_title(
                    f"{T2I_LABELS[method]}\nTrue={int(true_label)}, Pred={pred}"
                    f"\nConf={conf:.2f}", fontsize=8
                )
                axes[row, col_idx * 3].axis('off')

                # Overlay
                overlay = overlay_heatmap(orig, heatmap, alpha=0.5)
                axes[row, col_idx * 3 + 1].imshow(overlay)
                axes[row, col_idx * 3 + 1].set_title('Grad-CAM Overlay', fontsize=8)
                axes[row, col_idx * 3 + 1].axis('off')

                # Heatmap only
                axes[row, col_idx * 3 + 2].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
                axes[row, col_idx * 3 + 2].set_title('Heatmap', fontsize=8)
                axes[row, col_idx * 3 + 2].axis('off')

        fig.suptitle(
            f'Grad-CAM: Which pixels does ShallowCNN focus on?\n({DATASET_LABELS[dataset]})',
            fontsize=14, fontweight='bold', y=1.02
        )
        plt.tight_layout()
        path = output_path / f'ch4_gradcam_{dataset}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved: {path.name}")


# ============================================================
# Main
# ============================================================

# ============================================================
# Figure: Class Distribution
# ============================================================

def plot_class_distribution(output_dir='results/figures'):
    """Bar chart showing class distribution for all 3 datasets.

    WHY: Explains why macro-F1 was chosen over accuracy.
    Dry Bean has 7 classes with 3:1 imbalance. Adult Income
    has 3.2:1 imbalance. Without seeing this, readers might
    question why accuracy alone isn't sufficient.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load datasets to get true class distributions
    sys.path.insert(0, str(Path(__file__).parent))
    from src.preprocessing import load_breast_cancer, load_dry_bean, load_adult_income

    loaders = {
        'breast_cancer': load_breast_cancer,
        'dry_bean': load_dry_bean,
        'adult_income': load_adult_income,
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (dataset, loader_fn) in enumerate(loaders.items()):
        ax = axes[idx]
        X, y, _, class_names = loader_fn()
        unique, counts = np.unique(y, return_counts=True)

        colors = plt.cm.Set2(np.linspace(0, 1, len(unique)))
        bars = ax.bar(range(len(unique)), counts, color=colors,
                      edgecolor='white', linewidth=0.5)

        # Add count labels
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Add imbalance ratio annotation
        max_count = counts.max()
        min_count = counts.min()
        ratio = max_count / min_count
        ax.annotate(f'Imbalance: {ratio:.1f}:1', xy=(0.95, 0.95),
                    xycoords='axes fraction', ha='right', va='top',
                    fontsize=10, fontstyle='italic',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

        ax.set_xticks(range(len(unique)))
        ax.set_xticklabels(class_names, rotation=45 if len(class_names) > 4 else 0,
                           ha='right', fontsize=9)
        ax.set_ylabel('Count')
        ax.set_title(DATASET_LABELS[dataset], fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Class Distribution Across Datasets',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = output_path / 'ch3_class_distribution.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ============================================================
# Figure: Runtime Comparison
# ============================================================

def plot_runtime_comparison(results, output_dir='results/figures'):
    """Bar chart: training time per T2I method × architecture.

    WHY: Shows computational cost of each approach. A professor
    might ask 'is the better performance worth the extra compute?'
    Also useful for practical deployment decisions.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cnn_results = [r for r in results if r.get('cnn_arch') in CNN_ARCHS
                   and r.get('train_time_sec') is not None]
    if not cnn_results:
        print("  No CNN results with timing data, skipping runtime plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx]
        ds_results = [r for r in cnn_results if r['dataset'] == dataset]
        if not ds_results:
            continue

        # Build matrix: rows=methods, cols=architectures
        matrix = np.full((len(T2I_METHODS), len(CNN_ARCHS)), np.nan)
        for r in ds_results:
            i = T2I_METHODS.index(r['t2i_method'])
            j = CNN_ARCHS.index(r['cnn_arch'])
            matrix[i, j] = r['train_time_sec']

        # Grouped bar chart
        n_methods = len(T2I_METHODS)
        n_archs = len(CNN_ARCHS)
        x = np.arange(n_methods)
        width = 0.8 / n_archs
        arch_colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b2']

        for j in range(n_archs):
            vals = matrix[:, j]
            bars = ax.bar(x + j * width - 0.4 + width/2, vals, width,
                          label=CNN_LABELS[CNN_ARCHS[j]].split('\n')[0],
                          color=arch_colors[j], edgecolor='white', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([T2I_LABELS[m] for m in T2I_METHODS],
                           rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Training Time (s)' if idx == 0 else '')
        ax.set_title(DATASET_LABELS[dataset].split('\n')[0],
                     fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=8, loc='upper left')

    fig.suptitle('Training Time by T2I Method and Architecture',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = output_path / 'ch4_runtime_comparison.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ============================================================
# Figure: ROC Curves
# ============================================================

def plot_roc_curves(results, output_dir='results/figures'):
    """ROC curves: one per dataset, 5 curves (one per T2I method, best arch).

    WHY: Standard ML paper figure. Shows trade-off between true positive
    rate and false positive rate. More informative than just reporting
    the ROC-AUC scalar.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    method_colors = {'naive': '#c44e52', 'tinto': '#8172b2',
                     'deepinsight': '#4c72b0',                     'igtd': '#55a868'}

    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx]
        ds_results = [r for r in results if r['dataset'] == dataset
                      and r.get('cnn_arch') in CNN_ARCHS]
        if not ds_results:
            continue

        for method in T2I_METHODS:
            method_results = [r for r in ds_results if r['t2i_method'] == method]
            if not method_results:
                continue

            # Use best arch for this method
            best = max(method_results, key=lambda r: r['f1_macro'])

            # Binary classification ROC
            if 'roc_curve' in best:
                fpr = best['roc_curve']['fpr']
                tpr = best['roc_curve']['tpr']
                auc_val = best.get('roc_auc', 0)
                ax.plot(fpr, tpr, color=method_colors.get(method, 'gray'),
                        linewidth=2, label=f"{T2I_LABELS[method]} (AUC={auc_val:.3f})")

            # Multiclass ROC (one-vs-rest, plot macro average)
            elif 'roc_curves_per_class' in best:
                # Average across classes
                all_tpr = []
                all_fpr_combined = np.linspace(0, 1, 100)
                for cls_data in best['roc_curves_per_class'].values():
                    from numpy import interp
                    tpr_interp = interp(all_fpr_combined, cls_data['fpr'], cls_data['tpr'])
                    all_tpr.append(tpr_interp)
                mean_tpr = np.mean(all_tpr, axis=0)
                auc_val = best.get('roc_auc', 0)
                ax.plot(all_fpr_combined, mean_tpr, color=method_colors.get(method, 'gray'),
                        linewidth=2, label=f"{T2I_LABELS[method]} (AUC={auc_val:.3f})")

        # Diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate' if idx == 0 else '')
        ax.set_title(DATASET_LABELS[dataset].split('\n')[0],
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('ROC Curves — Best CNN per T2I Method',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = output_path / 'ch4_roc_curves.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ============================================================
# Figure: Overlap Diagnostics
# ============================================================

def plot_overlap_diagnostics(output_dir='results/figures'):
    """Bar chart: OF% and OP% per T2I method per dataset.

    WHY: Quantifies why TINTO might underperform — higher feature
    overlap means features lose individual identity. This figure
    directly supports the claim that overlap degrades performance.
    Uses compute_overlap_all_methods() from overlap_metrics.py.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).parent))
    from src.t2i.overlap_metrics import compute_overlap_all_methods
    from src.preprocessing import preprocess_dataset

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    all_overlap = {}
    for dataset in DATASETS:
        print(f"  Computing overlap for {dataset}...")
        data = preprocess_dataset(dataset)
        X_train = data['X_train']
        y_train = data['y_train']
        overlap = compute_overlap_all_methods(X_train, y_train, image_size=32)
        all_overlap[dataset] = overlap

    # Plot OF (Percentage of Overlapped Features)
    ax = axes[0]
    n_datasets = len(DATASETS)
    n_methods = len(T2I_METHODS)
    x = np.arange(n_datasets)
    width = 0.8 / n_methods
    method_colors = {'naive': '#c44e52', 'tinto': '#8172b2',
                     'deepinsight': '#4c72b0',                     'igtd': '#55a868'}

    for j, method in enumerate(T2I_METHODS):
        of_vals = [all_overlap[ds].get(method, {}).get('of_percent', 0) for ds in DATASETS]
        ax.bar(x + j * width - 0.4 + width/2, of_vals, width,
               label=T2I_LABELS[method], color=method_colors.get(method, 'gray'),
               edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[ds].split('\n')[0] for ds in DATASETS],
                       fontsize=10)
    ax.set_ylabel('Overlapped Features (%)')
    ax.set_title('Feature Overlap (OF)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Plot OP (Percentage of Overlapped Pixels)
    ax = axes[1]
    for j, method in enumerate(T2I_METHODS):
        op_vals = [all_overlap[ds].get(method, {}).get('op_percent', 0) for ds in DATASETS]
        ax.bar(x + j * width - 0.4 + width/2, op_vals, width,
               label=T2I_LABELS[method], color=method_colors.get(method, 'gray'),
               edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[ds].split('\n')[0] for ds in DATASETS],
                       fontsize=10)
    ax.set_ylabel('Overlapped Pixels (%)')
    ax.set_title('Pixel Overlap (OP)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.suptitle('T2I Image Quality: Feature and Pixel Overlap Diagnostics',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = output_path / 'ch4_overlap_diagnostics.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path.name}")


def main():
    parser = argparse.ArgumentParser(description='Generate Chapter 4 figures')
    parser.add_argument('--heatmap-only', action='store_true')
    parser.add_argument('--ablation-only', action='store_true')
    parser.add_argument('--results-dir', default='results')
    args = parser.parse_args()

    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} experiment results\n")

    if args.ablation_only:
        print("=== Ablation Figures ===")
        plot_ablation_results()
        return

    if args.heatmap_only:
        print("=== Main Results Heatmap ===")
        plot_main_results_heatmap(results)
        return

    # All figures
    print("=== Figure 4.3: Main Results Heatmap ===")
    plot_main_results_heatmap(results)

    print("\n=== Figure 4.4: Baseline Comparison ===")
    plot_baseline_comparison(results)

    print("\n=== Figure 4.5: Per-Class F1 (Dry Bean) ===")
    plot_per_class_f1(results)

    print("\n=== Figure 4.6: Training Curves ===")
    plot_training_curves(results)

    print("\n=== Figure 4.7: Confusion Matrices ===")
    plot_confusion_matrices(results)

    print("\n=== Figure 4.8: Ablation Results ===")
    plot_ablation_results()

    print("\n=== Figure 4.10: Density vs Performance ===")
    plot_density_vs_performance(results)

    print("\n=== New: ROC Curves ===")
    plot_roc_curves(results)

    print("\n=== New: Runtime Comparison ===")
    plot_runtime_comparison(results)

    print("\n=== New: Class Distribution ===")
    plot_class_distribution()

    print("\n=== New: Overlap Diagnostics ===")
    plot_overlap_diagnostics()

    print("\n=== New: Grad-CAM Heatmaps ===")
    plot_gradcam_grid()

    print(f"\nAll figures saved to {Path(args.results_dir) / 'figures'}/")


if __name__ == '__main__':
    main()
