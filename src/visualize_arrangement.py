"""
Feature-arrangement visualizations for the T2I methods (Chapter 3/4 figures).

These figures do NOT need experiment results — they visualize *how each T2I
method places the D features onto the pixel grid* and *whether the placement
is consistent with feature similarity* (the core claim behind DeepInsight /
TINTO / IGTD: correlated features should end up near each other).

Figures produced (into results/figures/):
  ch3_feature_layout_{dataset}.png
      Dots (or numbered cells when D <= 30) showing where each feature's
      intensity is written in a 32x32 image, per method.
      Source of coordinates:
        - tinto/deepinsight/igtd: the fitted TINTOlib model's feature->pixel
          mapping (_features_mapping row/column or _features_positions).
        - naive: analytic row-major grid placement (pad to ceil(sqrt(D))
          square, then bicubic-scaled to 32x32) — naive has no coordinate
          model, so this mirrors exactly where the naive transform writes.
  ch4_arrangement_quality.png
      For each (dataset, method): scatter of feature-pair Euclidean pixel
      distance (x) vs |Pearson correlation| of the pair on the training set
      (y). Spearman rho printed per panel. A layout that places correlated
      features nearby yields strongly NEGATIVE rho (short distance -> high
      correlation); naive (input column order) shows no such structure.

Usage:
    python src/visualize_arrangement.py                    # all datasets
    python src/visualize_arrangement.py --dataset breast_cancer

Implementation notes:
- We deliberately read the fitted model's coordinate mapping (not our own
  estimate) so the figure shows exactly where the transform places values.
- IGTD in this TINTOlib version assigns the D features to the first D grid
  pixels in row-major order and only *re-orders* which feature occupies each
  slot, so its coordinate map is a narrow strip — the figure makes that
  visible (verified against TINTOlib/igtd.py _fitAlg/_build_features_mapping).
- Adult one-hot categories constant within the training split have NaN
  correlation; those pairs are dropped before computing rho.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import preprocess_dataset
from src.t2i import T2ITransformer

DATASETS = ['breast_cancer', 'dry_bean', 'adult_income']
DATASET_LABELS = {
    'breast_cancer': 'Breast Cancer (30 features)',
    'dry_bean': 'Dry Bean (16 features)',
    'adult_income': 'Adult Income (104 features)',
}
METHODS = ['naive', 'tinto', 'deepinsight', 'igtd']
METHOD_LABELS = {
    'naive': 'Naive', 'tinto': 'TINTO',
    'deepinsight': 'DeepInsight', 'igtd': 'IGTD',
}
METHOD_COLORS = {
    'naive': '#c44e52', 'tinto': '#8172b2',
    'deepinsight': '#4c72b0', 'igtd': '#55a868',
}
IMAGE_SIZE = 32  # fixed in run_all.py for all datasets (no auto-size)


def get_feature_coordinates(method, transformer, n_features, image_size=IMAGE_SIZE):
    """Return (rows, cols) per feature in final-image pixel space.

    naive: analytic placement — feature i sits at cell (i//grid, i%grid) of
    the padded square; cell centers scaled by image_size/grid to match the
    bicubic resize to the final canvas.
    others: the fitted TINTOlib model's feature->pixel mapping, kept in the
    wrapper's get_coordinates() (see src/t2i/*.py).
    """
    if method == 'naive':
        grid = transformer.grid_size
        rows, cols = np.divmod(np.arange(n_features), grid)
        scale = image_size / grid
        return (rows + 0.5) * scale, (cols + 0.5) * scale

    coords = transformer.get_coordinates()
    if coords is None:
        return None
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (n_features, 2):
        return None
    return coords[:, 0], coords[:, 1]  # (row, column)


def _fit_layout(dataset):
    """Fit all methods on the dataset's training split; return fitted list."""
    data = preprocess_dataset(dataset)
    X_train, y_train = data['X_train'], data['y_train']
    fitted = []
    for method in METHODS:
        try:
            t2i = T2ITransformer(method=method, image_size=IMAGE_SIZE)
            t2i.fit(X_train, y_train)
            fitted.append((method, t2i))
        except Exception as e:  # keep other panels readable on failure
            fitted.append((method, e))
    return X_train, fitted


def plot_feature_layout(dataset, output_dir='results/figures'):
    """Dot/number map of feature placement for ONE dataset across methods."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    X_train, fitted = _fit_layout(dataset)
    n_features = X_train.shape[1]
    annotate = n_features <= 30

    fig, axes = plt.subplots(1, len(METHODS), figsize=(4.2 * len(METHODS), 4.2))

    for ax, (method, item) in zip(axes, fitted):
        if isinstance(item, Exception):
            ax.text(0.5, 0.5, f'{METHOD_LABELS[method]} failed:\n{item}',
                    ha='center', va='center', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        rows, cols = get_feature_coordinates(method, item.transformer,
                                             n_features)
        if rows is None:
            ax.text(0.5, 0.5, f'{METHOD_LABELS[method]}\n(no coordinates)',
                    ha='center', va='center', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        ax.scatter(cols, rows, s=42 if annotate else 12, alpha=0.85,
                   color=METHOD_COLORS[method], edgecolors='white',
                   linewidth=0.4, zorder=3)
        if annotate:
            for i, (r, c) in enumerate(zip(rows, cols)):
                ax.annotate(str(i), (c, r), fontsize=4.5, zorder=4,
                            textcoords='offset points', xytext=(2.5, 2.5))

        uniq = len({(round(r, 3), round(c, 3)) for r, c in zip(rows, cols)})
        n_coll = n_features - uniq

        ax.set_xlim(-1.5, IMAGE_SIZE + 1.5)
        ax.set_ylim(IMAGE_SIZE + 1.5, -1.5)  # row 0 at top, image-style
        ax.set_xticks(range(0, IMAGE_SIZE + 1, 8))
        ax.set_yticks(range(0, IMAGE_SIZE + 1, 8))
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_aspect('equal')
        title = (f'{METHOD_LABELS[method]}'
                 + (f'\n{n_coll} features share a pixel' if n_coll else ''))
        ax.set_title(title, fontsize=10, fontweight='bold')

    fig.suptitle(f'{DATASET_LABELS[dataset]} — where each feature is drawn '
                 f'(32×32 image)', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = output_path / f'ch3_feature_layout_{dataset}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {path.name}')


def plot_arrangement_quality(datasets=None, output_dir='results/figures'):
    """Scatter of pair pixel distance vs |Pearson r| (rows = datasets)."""
    from scipy.stats import spearmanr

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ds_list = datasets if datasets is not None else DATASETS
    fig, axes = plt.subplots(len(ds_list), len(METHODS),
                             figsize=(4.0 * len(METHODS), 3.4 * len(ds_list)))
    if len(ds_list) == 1:
        axes = axes.reshape(1, -1)

    for di, dataset in enumerate(ds_list):
        X_train, fitted = _fit_layout(dataset)
        n_features = X_train.shape[1]

        # Pairwise feature correlation on the training split (|r|).
        C = np.corrcoef(X_train.T)
        iu = np.triu_indices(n_features, k=1)
        corr_abs = np.abs(C[iu])

        for mj, (method, item) in enumerate(fitted):
            ax = axes[di, mj]
            if isinstance(item, Exception):
                ax.text(0.5, 0.5, f'{METHOD_LABELS[method]} failed:\n{item}',
                        ha='center', va='center', fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            rows, cols = get_feature_coordinates(method, item.transformer,
                                                 n_features)
            if rows is None:
                ax.text(0.5, 0.5, f'{METHOD_LABELS[method]}\n(no coordinates)',
                        ha='center', va='center', fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            dist = np.sqrt((rows[iu[0]] - rows[iu[1]]) ** 2
                           + (cols[iu[0]] - cols[iu[1]]) ** 2)

            ok = np.isfinite(corr_abs) & np.isfinite(dist)
            rho = spearmanr(dist[ok], corr_abs[ok]).statistic

            ax.scatter(dist[ok], corr_abs[ok], s=6, alpha=0.35,
                       color=METHOD_COLORS[method], edgecolors='none', zorder=2)
            ax.set_xlim(-0.5, IMAGE_SIZE * 1.5)
            ax.set_ylim(-0.02, 1.02)
            if mj == 0:
                ax.set_ylabel(f'{DATASET_LABELS[dataset].split(" (")[0]}\n'
                              f'|Pearson r|', fontsize=9)
            if di == len(ds_list) - 1:
                ax.set_xlabel('Pixel distance', fontsize=9)
            ax.set_title(f"{METHOD_LABELS[method]}\n"
                         f"Spearman ρ = {rho:+.2f} "
                         f"({len(dist[ok])} pairs)", fontsize=9)
            ax.grid(True, alpha=0.2, linewidth=0.5)
            ax.tick_params(labelsize=7)

    fig.suptitle('Feature-pair similarity vs pixel distance — '
                 'does the layout place correlated features together?',
                 fontsize=13, fontweight='bold', y=1.0)
    plt.tight_layout()
    path = output_path / 'ch4_arrangement_quality.png'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {path.name}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=DATASETS, default=None,
                    help='only this dataset (default: all three)')
    ap.add_argument('--output-dir', default='results/figures')
    args = ap.parse_args()

    ds_list = [args.dataset] if args.dataset else DATASETS

    print('=== Feature layout maps ===')
    for dataset in ds_list:
        print(f'  {dataset}: fitting T2I transforms...')
        plot_feature_layout(dataset, args.output_dir)

    print('\n=== Arrangement quality (correlation vs distance) ===')
    plot_arrangement_quality(ds_list, args.output_dir)
    print('\nDone.')


if __name__ == '__main__':
    main()
