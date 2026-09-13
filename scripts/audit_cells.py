"""Screen every recorded cell for the checkpoint-selection artifact.

Post-run validation (2026-09-13). The recorded `adult_income/naive/resnet` cell
reports F1 57.58 %, but five repeats of the *identical* configuration give
68.36 +/- 0.82 %. The stored history shows why: validation loss oscillates
between 0.58 and 3.88 (learning rate 1e-3 is unstable for this arm), and early
stopping kept the best checkpoint at **epoch 2 of 50**, so the reported model had
barely trained. Its accuracy (64.70 %) consequently landed *below* the 75.2 %
majority-class rate.

That is a class of artifact you can detect without retraining anything, because
every JSON stores its per-epoch validation loss and its confusion matrix. This
script scans the whole results tree and classifies each cell:

* HIGH  - the run is not a valid summary of its configuration: the selected
          (best-val-loss) checkpoint is in the first few epochs, or test
          accuracy is worse than always predicting the majority class. Either
          way the number should not carry a claim as recorded.
* INFO  - validation loss oscillates a lot, but the run clears the majority
          rate and selected a checkpoint well into training. Adam with label
          smoothing produces this routinely; it is reported for transparency
          and deliberately does NOT trigger a re-run.

Oscillation is therefore informational only: on the recorded tree it appears in
25 of 83 cells, while the two decisive criteria isolate exactly one -- the cell
the seed sweep already contradicted.

Cells whose differences are quoted in the seminar are marked CLAIM, and the
report ends with the exact command to re-run the HIGH ones.

Runs in seconds, trains nothing, writes nothing unless --out is given.

Usage (from the repository root):

    python scripts/audit_cells.py                 # report + results/cell_health.csv
    python scripts/audit_cells.py --no-write      # report only
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
# Running `python scripts/audit_cells.py` puts scripts/ on sys.path, so the sweep's
# cell list (the single source of truth for what carries a claim) imports directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from seed_sweep import DEFAULT_CELLS  # noqa: E402
except ImportError:  # pragma: no cover - defensive
    DEFAULT_CELLS = []

CLAIM_CELLS = {tuple(c.split('/')) for c in DEFAULT_CELLS}

# Thresholds, kept as module constants so they are visible rather than buried.
EARLY_EPOCH_LIMIT = 3        # HIGH: best checkpoint at or before this epoch
OSCILLATION_LIMIT = 5.0      # INFO only: max(val_loss) / min(val_loss)


def cell_from_name(name, group):
    """Recover (dataset, t2i, arch) from a result file name."""
    stem = name[:-5] if name.endswith('.json') else name
    if stem.startswith('baseline_'):
        return None
    stem = stem[len('ablation_'):] if stem.startswith('ablation_') else stem
    for dataset in ('breast_cancer', 'dry_bean', 'adult_income'):
        if not stem.startswith(dataset + '_'):
            continue
        rest = stem[len(dataset) + 1:]
        for arch in ('resnet_scratch', 'resnet', 'shallow'):
            if rest.endswith('_' + arch):
                return (dataset, rest[:-(len(arch) + 1)], arch)
    return (stem, '', '') if group else None


def diagnose(data):
    """Return the diagnostic dict for one result, or None if it has no history."""
    history = data.get('history') or {}
    val_loss = history.get('val_loss')
    if not val_loss:
        return None
    val_loss = np.asarray(val_loss, dtype=float)
    finite = np.isfinite(val_loss)

    best_epoch = data.get('best_epoch')
    if best_epoch is None:
        best_epoch = (int(np.argmin(np.where(finite, val_loss, np.inf))) + 1
                      if finite.any() else None)
    epochs_run = data.get('epochs_run', len(val_loss))
    osc = data.get('val_loss_oscillation')
    if osc is None and finite.any():
        lo = float(val_loss[finite].min())
        osc = float(val_loss[finite].max()) / lo if lo > 0 else float('nan')

    cm = data.get('confusion_matrix')
    majority = accuracy = None
    if cm:
        cm = np.asarray(cm, dtype=float)
        total = cm.sum()
        if total > 0:
            majority = float(cm.sum(axis=1).max() / total)
            accuracy = float(np.trace(cm) / total)
    if accuracy is None:
        accuracy = data.get('accuracy')

    high = []
    info = []
    if best_epoch is not None and best_epoch <= EARLY_EPOCH_LIMIT:
        high.append(f'EARLY_CHECKPOINT(epoch {best_epoch})')
    if majority is not None and accuracy is not None and accuracy < majority:
        high.append('BELOW_MAJORITY')
    if osc is not None and osc == osc and osc > OSCILLATION_LIMIT:
        info.append(f'OSCILLATING(x{osc:.1f})')

    return {
        'best_epoch': best_epoch,
        'epochs_run': epochs_run,
        'stopped_early': data.get('stopped_early'),
        'oscillation': osc,
        'accuracy': accuracy,
        'majority_rate': majority,
        'f1': data.get('f1_macro'),
        'high': high,
        'info': info,
    }


def collect(results_dir):
    """Yield (group, relative path, cell, diagnostic) for every result JSON."""
    groups = [('grid', results_dir.glob('*.json'))]
    seeds = results_dir / 'seeds'
    if seeds.is_dir():
        groups.append(('seed-sweep', seeds.glob('seed*/*.json')))
    other = results_dir / 'scratch3ch'
    if other.is_dir():
        groups.append(('scratch3ch', other.glob('*.json')))

    for group, pattern in groups:
        for path in sorted(pattern):
            if path.name.startswith('ablation_'):
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                print(f'  WARNING: could not read {path}: {exc}')
                continue
            cell = cell_from_name(path.name, group)
            if cell is None or not data.get('t2i_method'):
                continue          # baselines are out of scope for this screen
            diag = diagnose(data)
            if diag is None:
                continue
            yield group, path.relative_to(results_dir).as_posix(), cell, diag


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--results-dir', default=str(REPO_ROOT / 'results'),
                        help='directory holding the result JSONs (default: results/)')
    parser.add_argument('--out', default=str(REPO_ROOT / 'results' / 'cell_health.csv'),
                        help='where to write the machine-readable report')
    parser.add_argument('--no-write', action='store_true',
                        help='print the report but write no file')
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    results_dir = Path(args.results_dir)
    if not results_dir.is_dir():
        sys.exit(f'No such results directory: {results_dir}')

    rows = []
    for group, rel, cell, diag in collect(results_dir):
        key = '/'.join(cell)
        rows.append({'group': group, 'file': rel, 'cell': key,
                     'claim': tuple(cell) in CLAIM_CELLS, **diag})

    if not rows:
        sys.exit(f'No trainable result JSONs found under {results_dir}')

    # Aggregate per (cell, group): a seed sweep contributes five rows per cell,
    # and listing the same cell five times hides which cells actually need work.
    per_cell = defaultdict(list)
    for row in rows:
        per_cell[(row['cell'], row['group'])].append(row)

    cells = []
    for (cell, group), group_rows in per_cell.items():
        high_rows = [r for r in group_rows if r['high']]
        worst = min(high_rows or group_rows, key=lambda r: r['accuracy'] or 1.0)
        cells.append({
            'cell': cell,
            'group': group,
            'claim': group_rows[0]['claim'],
            'runs': len(group_rows),
            'n_high': len(high_rows),
            'high': sorted({f for r in high_rows for f in r['high']}),
            'n_info': sum(1 for r in group_rows if r['info']),
            'worst': worst,
        })

    high_cells = sorted([c for c in cells if c['n_high']],
                        key=lambda c: (not c['claim'], c['cell']))
    info_only = [c for c in cells if not c['n_high'] and c['n_info']]

    print('=' * 78)
    print('CELL HEALTH SCREEN  (checkpoint selection / stability)')
    print('=' * 78)
    print(f'  scanned {len(rows)} runs over {len(cells)} cell/group combinations '
          f'under {results_dir}')
    print(f'  HIGH {len(high_cells)}   INFO-only {len(info_only)}   '
          f'clean {len(cells) - len(high_cells) - len(info_only)}')
    print(f'  HIGH = best checkpoint <= epoch {EARLY_EPOCH_LIMIT}, or accuracy below '
          f'the majority rate')
    print(f'  INFO = validation-loss oscillation > x{OSCILLATION_LIMIT} '
          f'(reported, does not trigger a re-run)')

    if high_cells:
        print('\n  HIGH-severity cells (number is not a valid summary of the config):')
        print('  cell                                   group      runs  best_ep/run  acc     maj     flags')
        for c in high_cells:
            w = c['worst']
            ep = f"{w['best_epoch']}/{w['epochs_run']}" if w['best_epoch'] else '?'
            acc = f"{w['accuracy'] * 100:.2f}" if w['accuracy'] is not None else 'n/a'
            maj = f"{w['majority_rate'] * 100:.2f}" if w['majority_rate'] is not None else 'n/a'
            mark = 'CLAIM ' if c['claim'] else '      '
            print(f"  {mark}{c['cell']:38.38s} {c['group']:10.10s} {c['runs']:5d}  "
                  f"{ep:12s} {acc:7s} {maj:7s} {', '.join(c['high'])}")
    else:
        print('\n  No cell fails the two decisive criteria.')

    if info_only:
        print(f'\n  {len(info_only)} cell(s) oscillate but clear the majority rate and '
              f'selected a late checkpoint;')
        print('  listed in the CSV as information, not as a problem to fix:')
        for c in sorted(info_only, key=lambda c: c['cell'])[:8]:
            print(f"    {c['cell']:38.38s} {c['group']:10.10s} "
                  f"{c['n_info']}/{c['runs']} run(s) oscillating")
        if len(info_only) > 8:
            # ASCII only: this line goes to Windows consoles and piped Colab cells.
            print(f'    ... +{len(info_only) - 8} more')

    todo = [c for c in high_cells if c['claim']]
    print()
    if todo:
        print('  Re-run these HIGH claim-bearing cells three times at the identical')
        print('  configuration, to separate training noise from the recorded value.')
        print('  Each run backs up the previous JSON to results/backup_pre_rerun/.')
        for c in todo:
            print(f'    python run_all.py --cells {c["cell"]} --force '
                  f'--seed 42 --split-seed 42')
        print('\n  Then re-aggregate and regenerate the figures.')
    else:
        print('  Nothing high-severity among the claim-bearing cells — nothing to re-run.')

    if not args.no_write:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['cell', 'group', 'claim', 'runs', 'high_runs',
                             'severity', 'flags', 'worst_best_epoch', 'worst_epochs_run',
                             'worst_accuracy', 'worst_majority_rate', 'worst_f1_macro',
                             'worst_file'])
            for c in sorted(cells, key=lambda c: (c['n_high'] == 0, c['cell'])):
                w = c['worst']
                writer.writerow([
                    c['cell'], c['group'], int(c['claim']), c['runs'], c['n_high'],
                    'HIGH' if c['n_high'] else ('INFO' if c['n_info'] else 'clean'),
                    ' '.join(c['high'] or c['worst']['info']),
                    w['best_epoch'], w['epochs_run'],
                    '' if w['accuracy'] is None else f"{w['accuracy']:.6f}",
                    '' if w['majority_rate'] is None else f"{w['majority_rate']:.6f}",
                    '' if w['f1'] is None else f"{w['f1']:.6f}",
                    w['file'],
                ])
        from src.colab_sync import sync_path
        sync_path(out)
        print(f'\n  Wrote {out}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
