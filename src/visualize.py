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
T2I_METHODS = ['naive', 'tinto', 'deepinsight', 'igtd', 's_igtd']
T2I_LABELS = {
    'naive': 'Naive', 'tinto': 'TINTO', 'deepinsight': 'DeepInsight',
    'igtd': 'IGTD', 's_igtd': 'S-IGTD',
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
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='Macro-F1 (%)')
        ax.set_title(f'{DATASET_LABELS[dataset]} — Macro-F1 (%)',
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
        ax.set_ylabel('Macro-F1 (%)' if idx == 0 else '')
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
        ax.set_ylabel('Macro-F1 (%)')
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
                 'igtd': '#55a868', 's_igtd': '#ccb974'}

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
# Main
# ============================================================

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

    print(f"\nAll figures saved to {Path(args.results_dir) / 'figures'}/")


if __name__ == '__main__':
    main()
