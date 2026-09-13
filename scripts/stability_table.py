"""Build the stability table (mean +/- spread over seeds) for the seminar.

Post-run validation (2026-09-13). The seminar reports two tables instead of one:
the main results table is one run per cell (seed 42, single 70/10/20 split), and
this is the separate table that says how stable the claim-bearing cells are.

It reads `results/seeds/seed<42..46>/*.json` -- the untouched output of
`scripts/seed_sweep.py` -- so nothing is retrained and the numbers cannot drift
from the sweep. Two things come out:

* per-cell statistics for every metric (mean, sample sd, min, max);
* the differences that actually carry a claim, computed PAIRED by seed: both arms
  of a pair are trained at the same seed with the same split, so pairing removes
  the split-to-split variance that dominates the unpaired comparison. The
  interval is mean +/- 2 standard errors (5 seeds), which is an approximate
  interval to describe, not a significance test.

Usage (from the repository root):

    python scripts/stability_table.py                  # writes results/stability_table.{md,csv}
    python scripts/stability_table.py --no-write       # print only
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from seed_sweep import COMPARISONS, METRICS, summarize  # noqa: E402

# Human labels for the metrics, so the table can be pasted into the draft as is.
METRIC_LABELS = {
    'f1_macro': 'F1 (positive class / macro)',
    'f1_macro_all': 'macro-F1 over all classes',
    'balanced_accuracy': 'balanced accuracy',
    'accuracy': 'accuracy',
}


def cell_key(stem):
    """'adult_income_naive_resnet' -> 'adult_income/naive/resnet'.

    The seed sweep keys its comparisons ('dataset/t2i/arch') while the files are
    named with underscores, and both 'breast_cancer' and 'resnet_scratch' contain
    one, so a blind replace('_', '/') would corrupt them.
    """
    for dataset in ('breast_cancer', 'dry_bean', 'adult_income'):
        if not stem.startswith(dataset + '_'):
            continue
        rest = stem[len(dataset) + 1:]
        for arch in ('resnet_scratch', 'resnet', 'shallow'):
            if rest.endswith('_' + arch):
                return f'{dataset}/{rest[:-(len(arch) + 1)]}/{arch}'
    return stem


def load_seed_rows(seeds_root):
    """{(cell, seed): metrics} for every seed result on disk."""
    rows = {}
    for path in sorted(Path(seeds_root).glob('seed*/*.json')):
        seed = int(path.parent.name.replace('seed', ''))
        try:
            with open(path) as f:
                rows[(cell_key(path.stem), seed)] = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f'  WARNING: could not read {path}: {exc}')
    return rows


def paired_delta(rows, cell_a, cell_b, metric, seeds):
    """Paired difference a - b over the seeds present for both cells."""
    deltas = []
    for seed in seeds:
        a, b = rows.get((cell_a, seed)), rows.get((cell_b, seed))
        if not a or not b:
            continue
        va, vb = a.get(metric), b.get(metric)
        if va is None or vb is None:
            continue
        deltas.append(float(va) - float(vb))
    if not deltas:
        return None
    deltas = np.asarray(deltas, dtype=float)
    n = deltas.size
    mean = float(deltas.mean())
    sd = float(deltas.std(ddof=1)) if n > 1 else float('nan')
    se = sd / math.sqrt(n) if n > 1 and sd == sd else float('nan')
    half = 2 * se if se == se else float('nan')
    if n < 2 or half != half:
        verdict = 'single seed'
    elif (mean - half) * (mean + half) > 0:
        verdict = 'resolved (interval excludes 0)'
    else:
        verdict = 'UNRESOLVED (interval spans 0)'
    return {'n': n, 'mean': mean, 'sd': sd, 'half': half,
            'lo': mean - half if half == half else float('nan'),
            'hi': mean + half if half == half else float('nan'),
            'verdict': verdict}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--seeds-root', default=str(REPO_ROOT / 'results' / 'seeds'),
                        help='directory holding seed<N>/ (default: results/seeds)')
    parser.add_argument('--seeds', default='42,43,44,45,46',
                        help='seeds to read (default: 42-46)')
    parser.add_argument('--out', default=str(REPO_ROOT / 'results' / 'stability_table.md'),
                        help='markdown output path')
    parser.add_argument('--no-write', action='store_true',
                        help='print the table but write no file')
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    rows = load_seed_rows(args.seeds_root)
    if not rows:
        sys.exit(f'No seed results under {args.seeds_root}. '
                 f'Run scripts/seed_sweep.py first.')

    cells = sorted({cell for cell, _ in rows})
    if not all(c.count('/') == 2 for c in cells):
        sys.exit(f'Unexpected seed-file names: '
                 f'{[c for c in cells if c.count(chr(47)) != 2]}')
    summary = summarize(rows, [tuple(c.split('/')) for c in cells], seeds)

    lines = []
    lines.append('# Stability table (seeds %s)' % ', '.join(str(s) for s in seeds))
    lines.append('')
    lines.append('Companion to the main results table, which is one run per cell '
                 '(seed 42, single\n70/10/20 stratified split). Each cell below was '
                 're-run over five seeds with the\nidentical code path; the seed drives '
                 'both the training RNG and the split, so the\nspread answers "would '
                 'another split plus another training run have answered\ndifferently?". '
                 'n = number of completed runs, sd = sample standard deviation.')
    lines.append('')
    lines.append('| Cell | Metric | n | Mean | SD | Min | Max |')
    lines.append('|---|---|---|---|---|---|---|')
    for cell in cells:
        for metric in METRICS:
            st = summary.get((cell, metric))
            if not st:
                continue
            sd = f"{st['std'] * 100:.2f}" if st['std'] == st['std'] else 'n/a'
            lines.append(f"| {cell} | {METRIC_LABELS.get(metric, metric)} | {st['n']} | "
                         f"{st['mean'] * 100:.2f} | {sd} | {st['min'] * 100:.2f} | "
                         f"{st['max'] * 100:.2f} |")
    lines.append('')
    lines.append('All values in %. The metric label follows the stored semantics: '
                 'for the two binary\ndatasets `f1_macro` is the positive-class F1 '
                 '(audit C6), so `macro-F1 over all classes`\nis the only '
                 'cross-dataset-comparable column.')
    lines.append('')
    lines.append('## Differences that carry a claim (paired by seed)')
    lines.append('')
    lines.append('Both arms of a pair run at the same seed on the same split, so the '
                 'difference is\ncomputed per seed and then averaged; that removes the '
                 'split-to-split variance that\ndominates an unpaired comparison. The '
                 'interval is mean ± 2 standard errors over the\navailable seeds -- an '
                 'approximate interval to describe, not a significance test.')
    lines.append('')
    lines.append('| Comparison | Metric | n | Δ (pp) | ± 2 se | 95 %-ish interval | Verdict |')
    lines.append('|---|---|---|---|---|---|---|')
    for cell_a, cell_b, label in COMPARISONS:
        for metric in ('f1_macro', 'f1_macro_all', 'balanced_accuracy'):
            st = paired_delta(rows, cell_a, cell_b, metric, seeds)
            if st is None:
                continue
            if st['half'] == st['half']:
                half = f"{st['half'] * 100:.2f}"
                interval = f"[{st['lo'] * 100:+.2f}, {st['hi'] * 100:+.2f}]"
            else:
                half = interval = 'n/a'
            lines.append(f"| {label} | {METRIC_LABELS.get(metric, metric)} | {st['n']} | "
                         f"{st['mean'] * 100:+.2f} | {half} | {interval} | "
                         f"{st['verdict']} |")
    lines.append('')
    lines.append('Unresolved rows are quoted in the text as "not distinguishable from '
                 'noise", never as\nfindings. Cells outside this table are single-split '
                 'runs, so differences smaller than\nroughly one percentage point are '
                 'not interpreted anywhere in the write-up.')
    lines.append('')

    text = '\n'.join(lines)

    def console_safe(s):
        """Keep a Windows console (cp1252) from killing the report.

        The markdown file is UTF-8 and keeps its Δ/±; only the console copy is
        transliterated, so running this in a plain cmd.exe window still works.
        """
        for fancy, plain in (('Δ', 'delta '), ('±', '+/-'), ('–', '-'), ('—', '-')):
            s = s.replace(fancy, plain)
        enc = getattr(sys.stdout, 'encoding', None) or 'utf-8'
        try:
            return s.encode(enc, 'replace').decode(enc, 'replace')
        except (LookupError, UnicodeError):
            return s.encode('ascii', 'replace').decode('ascii')

    print(console_safe(text))

    if not args.no_write:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding='utf-8')
        csv_path = out.with_suffix('.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['cell', 'metric', 'n', 'mean_pct', 'sd_pct', 'min_pct', 'max_pct'])
            for cell in cells:
                for metric in METRICS:
                    st = summary.get((cell, metric))
                    if not st:
                        continue
                    writer.writerow([cell, metric, st['n'],
                                     f"{st['mean'] * 100:.2f}",
                                     f"{st['std'] * 100:.2f}" if st['std'] == st['std'] else '',
                                     f"{st['min'] * 100:.2f}", f"{st['max'] * 100:.2f}"])
            writer.writerow([])
            writer.writerow(['paired_by_seed', 'metric', 'n', 'delta_pp', 'plusminus_2se_pp',
                             'interval_lo_pp', 'interval_hi_pp', 'verdict'])
            for cell_a, cell_b, label in COMPARISONS:
                for metric in ('f1_macro', 'f1_macro_all', 'balanced_accuracy'):
                    st = paired_delta(rows, cell_a, cell_b, metric, seeds)
                    if st is None:
                        continue
                    writer.writerow([label, metric, st['n'], f"{st['mean'] * 100:+.2f}",
                                     f"{st['half'] * 100:.2f}" if st['half'] == st['half'] else '',
                                     f"{st['lo'] * 100:+.2f}" if st['lo'] == st['lo'] else '',
                                     f"{st['hi'] * 100:+.2f}" if st['hi'] == st['hi'] else '',
                                     st['verdict']])
        from src.colab_sync import sync_path
        sync_path(out)
        sync_path(csv_path)
        print(f'\nWrote {out}')
        print(f'Wrote {csv_path}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
