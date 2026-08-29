"""
Visual comparison of T2I methods across all 3 datasets.

Generates a 3x3 grid:
- Rows: Breast Cancer, Dry Bean, Adult Income
- Columns: Naive Reshape, DeepInsight, IGTD

Each cell shows 4 example images per class (or as many as fit).

Output: results/figures/t2i_comparison.png
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.preprocessing import preprocess_dataset
from src.t2i import T2ITransformer


def get_sample_images(X, y, n_per_class=2):
    """Get n_per_class representative samples from each class."""
    classes = np.unique(y)
    samples = {}
    for c in classes:
        idx = np.where(y == c)[0][:n_per_class]
        samples[c] = idx
    return samples


def plot_comparison(dataset_name, image_size=32):
    """Generate comparison figure for one dataset."""
    data = preprocess_dataset(dataset_name)
    X_train = data['X_train']
    y_train = data['y_train']
    classes = np.unique(y_train)
    n_classes = len(classes)

    # Get sample indices
    samples = get_sample_images(X_train, y_train, n_per_class=2)
    total_samples = sum(len(v) for v in samples.values())

    methods = ['naive', 'deepinsight', 'igtd']
    method_labels = ['Naive Reshape', 'DeepInsight', 'IGTD']

    # Create figure: 3 columns (methods) x total_samples rows
    fig, axes = plt.subplots(total_samples, 3, figsize=(9, 2.5 * total_samples))

    for col_idx, (method, label) in enumerate(zip(methods, method_labels)):
        # Fit and transform
        t2i = T2ITransformer(method=method, image_size=image_size)
        t2i.fit(X_train, y_train)
        images = t2i.transform(X_train, y_train)  # (N, 1, H, W)
        images_np = images.numpy()

        # Column title
        axes[0, col_idx].set_title(label, fontsize=14, fontweight='bold')

        # Plot samples
        row = 0
        for c in classes:
            for idx in samples[c]:
                img = images_np[idx, 0]  # (H, W)
                axes[row, col_idx].imshow(img, cmap='gray', vmin=0, vmax=1)
                axes[row, col_idx].set_xticks([])
                axes[row, col_idx].set_yticks([])

                # Add class label on the left
                if col_idx == 0:
                    axes[row, col_idx].set_ylabel(
                        f'Class {int(c)}', fontsize=11, rotation=0, labelpad=50, va='center'
                    )
                row += 1

    # Dataset title
    title_map = {
        'breast_cancer': 'Breast Cancer Wisconsin (30 features)',
        'dry_bean': 'Dry Bean (16 features)',
        'adult_income': 'Adult Income (108 features)',
    }
    fig.suptitle(f'{title_map.get(dataset_name, dataset_name)} — {image_size}×{image_size}',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


def plot_pixel_density_comparison():
    """Show how feature density varies across methods and datasets."""
    datasets = ['breast_cancer', 'dry_bean', 'adult_income']
    dataset_labels = ['Breast Cancer\n(30 feat)', 'Dry Bean\n(16 feat)', 'Adult Income\n(108 feat)']
    methods = ['naive', 'deepinsight', 'igtd']
    method_labels = ['Naive', 'DeepInsight', 'IGTD']
    image_size = 32

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for col_idx, dataset_name in enumerate(datasets):
        data = preprocess_dataset(dataset_name)
        X_train = data['X_train']
        y_train = data['y_train']
        classes = np.unique(y_train)

        # Use first sample from each class
        for row_idx, method in enumerate(methods):
            t2i = T2ITransformer(method=method, image_size=image_size)
            t2i.fit(X_train, y_train)
            images = t2i.transform(X_train, y_train)

            # Pick one sample from the most frequent class
            most_freq_class = classes[np.argmax(np.bincount(y_train))]
            sample_idx = np.where(y_train == most_freq_class)[0][0]
            img = images[sample_idx, 0].numpy()

            # Compute density
            density = (img > 0.01).sum() / img.size * 100

            axes[row_idx, col_idx].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[row_idx, col_idx].set_title(
                f'{method_labels[row_idx]}\nDensity: {density:.1f}%',
                fontsize=11
            )
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

            # Dataset label on the right side of first row
            if row_idx == 0:
                pass  # title already there

    # Add dataset labels on top
    for col_idx, label in enumerate(dataset_labels):
        axes[0, col_idx].set_xlabel('')  # clear
        fig.text(
            0.22 + col_idx * 0.28, 1.01, label,
            ha='center', fontsize=13, fontweight='bold',
            transform=fig.transFigure
        )

    fig.suptitle('Feature Density Comparison: Naive vs Intelligent Arrangement',
                 fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    return fig


def main():
    import os
    os.makedirs('results/figures', exist_ok=True)

    print("Generating T2I method comparison...")

    # Individual dataset comparisons
    for dataset in ['breast_cancer', 'dry_bean', 'adult_income']:
        print(f"  Processing {dataset}...")
        fig = plot_comparison(dataset, image_size=32)
        path = f'results/figures/t2i_comparison_{dataset}.png'
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved: {path}")

    # Density comparison grid
    print("  Generating density comparison...")
    fig = plot_pixel_density_comparison()
    path = 'results/figures/t2i_density_comparison.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")

    print("\nAll figures saved to results/figures/")


if __name__ == '__main__':
    main()
