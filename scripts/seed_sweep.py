"""Repeat the claim-bearing cells over several seeds and report mean +/- spread.

Audit C7/C8. Every number in the study currently comes from ONE run, so no
difference between methods carries an uncertainty estimate. On breast_cancer the
test set is 114 rows, which means a single flipped prediction moves F1 by about
0.9 pp — larger than several of the quoted differences. This script produces the
error bars for the few cells that actually carry a claim, instead of re-running
the whole 36-cell grid.

It reuses `run_all.run_single_experiment`, so the training path is provably the
same one that produced the recorded results; nothing is duplicated here.

WHAT ONE SEED MEANS
By default the seed drives BOTH the training RNG and the train/val/test split,
so the reported spread answers "would another split plus another training run
give another answer?" — the honest error bar. Pass --split-seed 42 to hold the
split fixed and measure training noise alone (tighter, but it cannot tell you
whether the conclusion depends on the particular partition).

Usage (from the repository root):

    # Colab: use %cd (a `!cd` line does NOT persist to the next `!` line,
    # so `!python scripts/...` would then run in the wrong directory).
    %cd /content/T2I-CNN

    # dry run: show what would run and what it costs
    python scripts/seed_sweep.py --dry-run

    # the default set: 7 claim-bearing cells x 5 seeds
    python scripts/seed_sweep.py

    # cheaper: only the three adult/naive cells that carry the headline claim
    python scripts/seed_sweep.py --cells adult_income/naive/resnet,adult_income/naive/resnet_scratch,adult_income/naive/shallow

Results land in results/seeds/seed<N>/ so the recorded seed-42 grid in
results/ is never touched. A summary table is printed and written to
results/seed_summary.csv.

NOTE: this is CPU-heavy (see the per-phase budget in
Plan/critical-audit-findings.md). On a GPU session the same sweep takes minutes.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# The cells whose differences are actually quoted in the draft. Widen or narrow
# with --cells; there is no point spending seeds on cells nobody compares.
DEFAULT_CELLS = [
    'adult_income/naive/resnet',          # headline: negative transfer, the one cell the chapter leans on
    'adult_income/naive/resnet_scratch',  # its from-scratch counterpart (-11.40 pp claim)
    'adult_income/naive/shallow',         # third arm of the same three-way comparison
    'adult_income/tinto/shallow',         # "TINTO/DeepInsight below naive on Adult" claim
    'adult_income/deepinsight/shallow',   # ditto
    'dry_bean/naive/shallow',             # best cell in the whole study (93.99 %)
    'breast_cancer/tinto/shallow',        # "CNN+T2I reaches the tabular baseline" claim
]

# Pairs whose DIFFERENCE is the claim, not whose absolute value is.
COMPARISONS = [
    ('adult_income/naive/resnet', 'adult_income/naive/resnet_scratch',
     'pretrained vs from-scratch ResNet on Adult/naive (negative transfer)'),
    ('adult_income/naive/shallow', 'adult_income/tinto/shallow',
     'naive vs TINTO on Adult with ShallowCNN (collision cost)'),
]

METRICS = ['f1_macro', 'f1_macro_all', 'balanced_accuracy', 'accuracy']


def parse_cells(spec):
    cells = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            continue
        parts = item.split('/')
        if len(parts) != 3:
            sys.exit(f"Bad --cells entry {item!r}; expected dataset/t2i/arch")
        cells.append(tuple(parts))
    return cells


def summarize(rows, cells, seeds):
    """rows: {(cell, seed): metrics_dict} -> summary dict keyed by (cell, metric)."""
    summary = {}
    for cell in cells:
        key = '/'.join(cell)
        present = [rows[(key, s)] for s in seeds if (key, s) in rows]
        if not present:
            continue
        for metric in METRICS:
            vals = np.array([r[metric] for r in present
                             if metric in r and r[metric] is not None], dtype=float)
            if vals.size == 0:
                continue
            summary[(key, metric)] = {
                'n': int(vals.size),
                'mean': float(vals.mean()),
                'std': float(vals.std(ddof=1)) if vals.size > 1 else float('nan'),
                'min': float(vals.min()),
                'max': float(vals.max()),
            }
    return summary


def main():
    # Live output: a Colab cell gives Python a PIPE, which it block-buffers, so a
    # two-hour sweep would print nothing between runs. See
    # src.ablation._unbuffer_stdout for the same fix elsewhere.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--cells', default=','.join(DEFAULT_CELLS),
                        help='comma-separated dataset/t2i/arch cells')
    parser.add_argument('--seeds', default='42,43,44,45,46',
                        help='comma-separated training seeds (default: 42-46)')
    parser.add_argument('--split-seed', type=int, default=None,
                        help='hold the data split at this seed (e.g. 42) to '
                             'measure training noise only; default varies it')
    parser.add_argument('--out', default=str(REPO_ROOT / 'results' / 'seeds'),
                        help='root output directory (default: results/seeds)')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the plan without training anything')
    parser.add_argument('--force', action='store_true',
                        help='retrain even when a complete result JSON already '
                             'exists (default: skip finished cells)')
    parser.add_argument('--no-restore', action='store_true',
                        help='do not copy the $RESULTS_SYNC_DIR mirror back '
                             'into results/ before checking what is done')
    args = parser.parse_args()

    cells = parse_cells(args.cells)
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    if not cells or not seeds:
        sys.exit('Need at least one cell and one seed.')

    import run_all
    from run_all import DATASET_CONFIG, run_single_experiment, _experiment_is_done
    from src.colab_sync import describe, get_sync_dir, restore, sync_path

    print(describe())

    # Colab VMs are ephemeral: a previous session's results live only in the
    # $RESULTS_SYNC_DIR mirror (Drive). Without pulling them back first, the
    # local results/seeds/ tree looks empty and every cell retrains "from the
    # beginning" even though the work is already done. Restore once at startup
    # so the cached checks below see the full history. Existing local files
    # win over the mirror, so a partial local run is never clobbered.
    if not args.no_restore and get_sync_dir() is not None:
        copied, kept = restore(dest='results')
        if copied or kept:
            print(f'[restore] copied {copied}, kept {kept} existing file(s) '
                  f'from the mirror')

    # Validate early: a typo here would otherwise fail after the first training.
    for dataset, t2i, arch in cells:
        if dataset not in DATASET_CONFIG:
            sys.exit(f'Unknown dataset {dataset!r}. Valid: {sorted(DATASET_CONFIG)}')
        if t2i not in run_all.T2I_METHODS:
            sys.exit(f'Unknown T2I method {t2i!r}. Valid: {run_all.T2I_METHODS}')
        if arch not in run_all.ALL_ARCHITECTURES:
            sys.exit(f'Unknown architecture {arch!r}. Valid: {run_all.ALL_ARCHITECTURES}')

    print(f'Seeds:  {seeds}')
    print(f'Split:  ' + ('fixed at 42 (training noise only)'
                         if args.split_seed is not None
                         else 'varies with the seed (split + training noise)'))
    print(f'Cells ({len(cells)}):')
    for c in cells:
        print(f'  {" / ".join(c)}')
    print(f'\nTotal runs: {len(cells) * len(seeds)}', end='')
    if args.force:
        print(' (--force: retraining everything)')
    else:
        print(' (finished cells are skipped)')
    if args.dry_run:
        todo = cached = 0
        for seed in seeds:
            for cell in cells:
                out = Path(args.out) / f'seed{seed}'
                if args.force:
                    done = False
                else:
                    done = _experiment_is_done(out / f'{cell[0]}_{cell[1]}_{cell[2]}.json')
                cached += bool(done)
                todo += (not done)
                print(f'  seed {seed}: {"/".join(cell)}{"  (cached — will skip)" if done else ""}')
        print(f'\nDRY RUN — nothing trained. {todo} to run, {cached} cached (skipped).')
        if todo == 0:
            print('ALL CACHED — nothing to do.')
        return 0

    rows = {}
    skipped = ran = 0
    for seed in seeds:
        out_dir = Path(args.out) / f'seed{seed}'
        out_dir.mkdir(parents=True, exist_ok=True)
        # Drop stale partial writes from a killed run so they can never be
        # mistaken for results (run_single_experiment writes atomically, but
        # an old .tmp from an earlier version may still linger).
        for tmp in out_dir.glob('*.json.tmp'):
            print(f'  Cleaning up partial file: {tmp.name}')
            tmp.unlink()
        for cell in cells:
            dataset, t2i, arch = cell
            key = '/'.join(cell)
            result_file = out_dir / f'{dataset}_{t2i}_{arch}.json'

            if not args.force and _experiment_is_done(result_file):
                print(f'  [seed {seed}] {key} — cached, skipping')
                skipped += 1
            else:
                print(f'\n{"=" * 70}\n  [seed {seed}] {key}\n{"=" * 70}')
                try:
                    # save_weights=False: a sweep keeps only numbers, and 35
                    # ResNet state_dicts would be ~1.5 GB written (and mirrored
                    # to Drive) for nothing.
                    run_single_experiment(dataset, t2i, arch,
                                          output_dir=str(out_dir),
                                          seed=seed,
                                          split_seed=args.split_seed,
                                          save_weights=False)
                except Exception as exc:  # keep the sweep going
                    print(f'  ERROR on {key} @ seed {seed}: {exc}')
                    import traceback
                    traceback.print_exc()
                    continue
                ran += 1

            try:
                with open(result_file) as f:
                    rows[(key, seed)] = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                print(f'  WARNING: could not read {result_file.name}: {exc}')
                continue

    print(f'\nSweep done: {ran} trained, {skipped} skipped (cached).')
    if not rows:
        sys.exit('No results collected.')

    summary = summarize(rows, cells, seeds)

    # --- report ---
    print(f'\n\n{"=" * 78}')
    print('SEED SWEEP SUMMARY  (metrics in %, mean ± sample sd)')
    print(f'{"=" * 78}')
    for cell in cells:
        key = '/'.join(cell)
        print(f'\n{key}')
        for metric in METRICS:
            st = summary.get((key, metric))
            if not st:
                continue
            if st['n'] > 1 and st['std'] == st['std']:
                spread = f"± {st['std'] * 100:.2f}"
            else:
                spread = '± n/a (1 seed)'
            print(f"  {metric:18s} n={st['n']}  {st['mean'] * 100:6.2f} {spread}"
                  f"   [{st['min'] * 100:.2f}, {st['max'] * 100:.2f}]")

    print(f'\n{"=" * 78}')
    print('DIFFERENCES THAT CARRY A CLAIM')
    print('(descriptive: delta ± 2 standard errors. NOT a significance test —')
    print(' with 5 seeds this is an interval you should describe as approximate.)')
    print(f'{"=" * 78}')
    for a, b, label in COMPARISONS:
        for metric in ['f1_macro', 'balanced_accuracy']:
            sa, sb = summary.get((a, metric)), summary.get((b, metric))
            if not sa or not sb or sa['n'] < 2 or sb['n'] < 2:
                continue
            delta = sa['mean'] - sb['mean']
            se = float(np.sqrt(sa['std'] ** 2 / sa['n'] + sb['std'] ** 2 / sb['n']))
            lo, hi = delta - 2 * se, delta + 2 * se
            verdict = 'resolved (interval excludes 0)' if lo * hi > 0 else 'UNRESOLVED (interval spans 0)'
            print(f'\n  {label}')
            print(f'    {metric}: {delta * 100:+.2f} pp  ± {2 * se * 100:.2f} (2 se)  '
                  f'-> {verdict}')
            print(f'      {a}: {sa["mean"] * 100:.2f} ± {sa["std"] * 100:.2f}')
            print(f'      {b}: {sb["mean"] * 100:.2f} ± {sb["std"] * 100:.2f}')

    # --- CSV ---
    csv_path = Path(args.out).parent / 'seed_summary.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['cell', 'metric', 'n', 'mean', 'std', 'min', 'max'])
        for cell in cells:
            key = '/'.join(cell)
            for metric in METRICS:
                st = summary.get((key, metric))
                if not st:
                    continue
                writer.writerow([key, metric, st['n'],
                                 f"{st['mean']:.6f}",
                                 f"{st['std']:.6f}" if st['std'] == st['std'] else '',
                                 f"{st['min']:.6f}", f"{st['max']:.6f}"])
    sync_path(csv_path)
    print(f'\nWrote {csv_path}')
    print('Report the mean ± std (and the per-seed values) in the tables; quote the')
    print('UNRESOLVED rows as "not distinguishable from noise" rather than as findings.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
