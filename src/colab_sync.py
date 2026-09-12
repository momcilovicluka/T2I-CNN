"""Mirror finished results to a durable location (e.g. mounted Google Drive).

WHY
Colab VMs are ephemeral and `results/` is gitignored, so a disconnect, an idle
timeout or a crash destroys hours of training. Every experiment that finishes now
mirrors its own output the moment it is written, which makes the long runs both
durable and resumable: whatever survived in the mirror can be copied back, and
the existing resume logic skips it instead of retraining.

HOW
    export RESULTS_SYNC_DIR=/content/drive/MyDrive/t2i-results

With the variable unset — the local default — every function here is a no-op, so
a normal local run is completely unaffected. The relative path under the
repository is preserved, so `results/seeds/seed43/x.json` mirrors to
`$RESULTS_SYNC_DIR/results/seeds/seed43/x.json` and can be copied straight back.

Save everything (results, weights, figures) at the end of a session:

    python -m src.colab_sync
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

SYNC_DIR_ENV = 'RESULTS_SYNC_DIR'
REPO_ROOT = Path(__file__).resolve().parent.parent

# Only warn once per distinct failure, so a full disk does not flood a 2 h log.
_warned = set()


def get_sync_dir():
    """The mirror directory, or None when syncing is switched off."""
    raw = os.environ.get(SYNC_DIR_ENV, '').strip()
    return Path(raw) if raw else None


def describe():
    """Status line printed at the start of every long run.

    A sync that is configured but NOT writable is the dangerous case: the run
    looks protected and is not, so that is reported loudly rather than silently.
    """
    target = get_sync_dir()
    if target is None:
        return (f'[sync] OFF - set {SYNC_DIR_ENV}=<dir> to mirror results as they '
                f'finish. results/ is gitignored, so an ephemeral VM loses them.')
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / '.sync_write_test'
        probe.write_text('ok')
        probe.unlink()
    except OSError as exc:
        return (f'[sync] *** NOT WRITABLE *** {target} ({exc}) - results are NOT '
                f'being saved. Mount Drive or fix {SYNC_DIR_ENV} before a long run.')
    return f'[sync] ON - mirroring finished files to {target}'


def sync_path(path):
    """Mirror one finished file to the sync directory.

    Never raises. A dead drive, an unmounted path or a full disk must not kill a
    two-hour run; the failure is reported once and training continues.
    """
    target = get_sync_dir()
    if target is None:
        return False

    path = Path(path)
    if not path.exists() or not path.is_file():
        return False

    resolved = path.resolve()
    try:
        rel = resolved.relative_to(REPO_ROOT)
    except ValueError:
        # Outside the repo (an absolute output dir): keep the basename only.
        rel = Path(path.name)

    dest = target / rel
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(path), str(dest))
        return True
    except OSError as exc:
        key = f'{type(exc).__name__}: {exc}'
        if key not in _warned:
            _warned.add(key)
            print(f'[sync] WARNING: could not mirror {path}: {exc}')
        return False


def sync_tree(root='results'):
    """Mirror an entire tree. Returns the number of files copied."""
    if get_sync_dir() is None:
        return 0
    root = Path(root)
    if not root.exists():
        return 0
    return sum(1 for p in sorted(root.rglob('*')) if p.is_file() and sync_path(p))


def sync_figures(figures_dir='results/figures'):
    """Mirror a whole figure directory after a plotting script finishes.

    The per-experiment hooks cover result JSONs, weights and the summary CSV, but
    the figure scripts write many files in one go and are called explicitly, so
    they mirror their own output here rather than leaving it to the closing pass.
    """
    return sync_tree(figures_dir)


def restore(sync_dir=None, dest='results'):
    """Copy a previous session's mirror back into the working tree.

    Returns (copied, skipped): existing local files are never overwritten, so a
    partially completed local run wins over a stale mirror.
    """
    src = Path(sync_dir) if sync_dir else get_sync_dir()
    if src is None or not src.exists():
        return (0, 0)

    # The mirror may be the sync root itself or its parent (i.e. contain results/).
    candidates = [src / dest, src]
    source_root = next((c for c in candidates if c.exists()), None)
    if source_root is None:
        return (0, 0)
    if source_root.name != Path(dest).name:
        source_root = source_root / Path(dest).name
        if not source_root.exists():
            return (0, 0)

    dest_root = Path(dest)
    copied = skipped = 0
    for p in sorted(source_root.rglob('*')):
        if not p.is_file():
            continue
        rel = p.relative_to(source_root)
        out = dest_root / rel
        if out.exists():
            skipped += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(p), str(out))
        copied += 1
    return (copied, skipped)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--root', default='results',
                        help='tree to mirror (default: results)')
    parser.add_argument('--restore', action='store_true',
                        help='copy the mirror back into the working tree instead')
    parser.add_argument('--sync-dir', default=None,
                        help='override %s' % SYNC_DIR_ENV)
    args = parser.parse_args()

    if args.sync_dir:
        os.environ[SYNC_DIR_ENV] = args.sync_dir

    print(describe())

    if args.restore:
        copied, skipped = restore(dest=args.root)
        print(f'[restore] copied {copied}, kept {skipped} existing file(s)')
        print('[restore] re-run the pipeline: completed experiments now resume instead of retraining')
        return 0

    if get_sync_dir() is None:
        print(f'[sync] nothing to do - {SYNC_DIR_ENV} is not set')
        return 1

    count = sync_tree(args.root)
    print(f'[sync] mirrored {count} file(s) from {args.root}/')
    return 0


if __name__ == '__main__':
    sys.exit(main())
