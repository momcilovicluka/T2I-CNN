"""One-off post-processing: add cross-dataset-comparable metrics to results.

Audit C6. `src/evaluate.py` stores scikit-learn's F1 under the key `f1_macro`:

  * breast_cancer / adult_income -> BINARY positive-class F1
    (benign for breast_cancer, `>50K` for adult_income)
  * dry_bean                     -> a true macro average over 7 classes

so that one column is not a single comparable metric, and any figure / table
axis that labels it "Macro-F1" is wrong for two of the three datasets.

`src/evaluate.py` now also records, for new runs:

  * `f1_macro_all`      - macro-F1 averaged over ALL classes
  * `balanced_accuracy` - mean per-class recall

This script backfills those two keys into the already-saved results by
recomputing them from each stored `confusion_matrix`. No model is retrained,
and the legacy `f1_macro` key is left untouched so existing consumers keep
working.

Usage (from the repository root):

    python scripts/backfill_metrics.py            # report only (dry run)
    python scripts/backfill_metrics.py --write    # rewrite the JSONs in place

After applying, regenerate the aggregate CSV and the figures:

    python run_all.py --aggregate
    python src/visualize.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def metrics_from_confusion_matrix(cm):
    """Return (f1_macro_all, balanced_accuracy) from a confusion matrix.

    Follows scikit-learn's convention: rows are the true labels, columns the
    predicted ones, classes in sorted order.

    Macro-F1 averages 2*TP / (2*TP + FP + FN) over all classes; balanced
    accuracy averages TP / (TP + FN) over all classes. Both reduce to the
    familiar binary formulas for a 2x2 matrix, but computed over BOTH classes
    (i.e. they are symmetric and do not privilege a positive class).
    """
    cm = np.asarray(cm, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f'confusion matrix is not square: shape={cm.shape}')

    tp = np.diag(cm)
    support = cm.sum(axis=1)   # true class counts
    predicted = cm.sum(axis=0)  # predicted class counts

    with np.errstate(divide='ignore', invalid='ignore'):
        f1_per_class = np.where(
            (2 * tp + (predicted - tp) + (support - tp)) > 0,
            2 * tp / (2 * tp + (predicted - tp) + (support - tp)),
            0.0,
        )
        recall_per_class = np.where(support > 0, tp / support, 0.0)

    return float(np.mean(f1_per_class)), float(np.mean(recall_per_class))


def annotate(obj, counters, path):
    """Recursively add the two metrics to every dict holding a confusion matrix."""
    if isinstance(obj, dict):
        cm = obj.get('confusion_matrix')
        if isinstance(cm, (list, tuple, np.ndarray)):
            try:
                f1_all, bal_acc = metrics_from_confusion_matrix(cm)
            except ValueError as exc:
                print(f'    ! {path}: {exc}')
            else:
                obj['f1_macro_all'] = f1_all
                obj['balanced_accuracy'] = bal_acc
                counters['annotated'] += 1
                # Cross-check: for a balanced-by-construction macro average the
                # stored binary F1 must differ from f1_macro_all unless the two
                # classes are equally well predicted.
                stored = obj.get('f1_macro')
                if stored is not None:
                    delta = abs(float(stored) - f1_all)
                    counters['delta_pp'].append(delta * 100)
                    counters['rows'].append((path, obj.get('dataset'),
                                             obj.get('t2i_method'),
                                             obj.get('cnn_arch') or obj.get('model'),
                                             float(stored), f1_all, bal_acc))
                # Only count once per dict even if nested deeper.
        for value in obj.values():
            annotate(value, counters, path)
    elif isinstance(obj, list):
        for item in obj:
            annotate(item, counters, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--results-dir', default=str(REPO_ROOT / 'results'),
                        help='directory holding the result JSONs (default: results/)')
    parser.add_argument('--write', action='store_true',
                        help='apply the changes; without it this is a dry run')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        sys.exit(f'No such results directory: {results_dir}')

    json_files = sorted(results_dir.glob('*.json'))
    if not json_files:
        sys.exit(f'No JSON files in {results_dir}')

    counters = {'annotated': 0, 'delta_pp': [], 'rows': []}
    touched = []
    skipped = []

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        before = counters['annotated']
        annotate(data, counters, json_file.name)
        if counters['annotated'] == before:
            skipped.append(json_file.name)
            continue

        touched.append(json_file.name)
        if args.write:
            tmp_file = json_file.with_suffix('.json.tmp')
            with open(tmp_file, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(str(tmp_file), str(json_file))

    print(f'\nScanned {len(json_files)} JSON files in {results_dir}')
    print(f'  result dicts with a confusion matrix: {counters["annotated"]}'
          f' (in {len(touched)} files)')
    print(f'  files with no confusion matrix (skipped): {len(skipped)}')

    if counters['rows']:
        print('\n  file                          dataset        method       arch      '
              'stored f1   f1_macro_all  bal_acc')
        for name, dataset, method, arch, stored, f1_all, bal_acc in counters['rows']:
            print(f'  {name:28.28s}  {str(dataset):13.13s}  {str(method):10.10s}  '
                  f'{str(arch):8.8s}  {stored:9.4f}  {f1_all:11.4f}  {bal_acc:7.4f}')

    if counters['delta_pp']:
        deltas = np.array(counters['delta_pp'])
        print(f'\n  |stored f1_macro - f1_macro_all|: mean {deltas.mean():.2f} pp, '
              f'max {deltas.max():.2f} pp')
        print('  (equal to ~0 pp only for dry_bean, where stored f1_macro IS macro;')
        print('   for the binary datasets it is the positive-class F1.)')

    if args.write:
        print(f'\n  WROTE f1_macro_all + balanced_accuracy into {len(touched)} files.')
        print('  Next: python run_all.py --aggregate && python src/visualize.py')
    else:
        print('\n  DRY RUN — nothing was written. Re-run with --write to apply.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
