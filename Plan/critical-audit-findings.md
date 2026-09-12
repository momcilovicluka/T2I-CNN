# Critical audit findings — code, results, figures and draft (2026-09-11)

Scope: a hostile-reader (professor) pass over `run_all.py`, `src/**`, `results/**`,
`Plan/*.md` and the generated figures, looking for anything a reviewer could
falsify, contradict, or refuse to accept.

Status legend: **CONFIRMED** = reproduced against the current code; **LIKELY** =
code reading, verifiable by inspection; **DOCUMENTED** = already known and
already stated as a limitation (still listed so the list is complete).

Companion documents: `Plan/professor-validation.md` (per-pass audit log),
`Plan/paper-statement-guide.md` (claim wording), `Plan/final-writeup-plan.md`
("no correctness problems" — **that verdict is superseded by this file**).

---

## 0. Priority summary

| ID | Severity | Area | Confirmed? | Needs rerun? | Rerun scope |
|----|----------|------|-----------|--------------|-------------|
| C1 | **CRITICAL (integrity)** | Feature-ordering ablation | yes (reproduced) | yes, small | 3 ordering ablations + 1 figure |
| C2 | **HIGH (figure wrong)** | Overlap-diagnostics figure | yes (reproduced) | no | 1 figure |
| C3 | HIGH (method) | Pixel-shuffle ablation design | code reading | yes, small | 3 pixel-shuffle ablations |
| C4 | HIGH (reproducibility) | LP-FT cell vs main table | yes (from JSONs) | yes, small | 3 LP-FT ablations |
| C5 | MEDIUM (integrity) | Hard-coded OF/OP for naive/IGTD | yes (reproduced) | no | 1 figure |
| C6 | MEDIUM (metric) | `f1_macro` holds binary F1 / labels | yes | no (recompute) | the 54 JSONs (post-process) |
| C7 | MEDIUM (statistics) | One split, no variance; dead `cross_validate` | yes | optional, costly | chosen cells |
| C8 | MEDIUM (claim) | Headline negative transfer = one run | yes | recommended | 1 cell × 5 seeds |
| C9 | MEDIUM (consistency) | IGTD "has collisions" vs OF=OP=0 | yes | no | text only |
| C10 | MEDIUM (consistency) | Density figure has no density axis | yes | no | 1 figure + text |
| C11 | MEDIUM (method) | Pretrained-vs-scratch confound | documented | optional, big | 12 resnet_scratch cells |
| C12 | LOW/MED (leak) | One-hot fitted before split | documented | only if changed | adult (whole grid) |
| C13 | LOW | `datetime`/`timedelta` not imported | yes | no | — |
| C14 | LOW | `get_model` misses `resnet_scratch` | yes | no | — |
| C15 | LOW | Grad-CAM ignores `target_class` | yes | no | 3 Grad-CAM figures if changed |
| C16 | LOW | Deprecated `cm.get_cmap`; unpinned deps | yes | no | — |
| C17 | LOW | Duplicate `return`, two meanings of "density" | yes | no | — |
| C18 | LOW | TINTO pixel-cache depends on call order | code reading | no | — |
| C19 | LOW | Ablation JSON writes are not atomic | yes | no | — |
| C20 | LOW | Dead code inventory | yes | no | — |
| C21 | LOW (hygiene) | Repo clutter / duplicate artefacts | yes | no | — |

Estimated aggregate rerun cost if everything in "needs rerun" is done:
**≈ 2.5–3.5 CPU hours** (ablations only) — the 36-cell main grid does **not**
need to be re-run unless C11 or C12 is changed.

---

## Phase 1 implementation status (2026-09-11)

Phase 1 (audit items C1–C5) is implemented in the working tree. **No code was
executed** — the Colab run below is still required. Line numbers quoted in the
findings below refer to the pre-fix revision; the functions are named, so they
remain findable after the edits.

| Item | Change made | State | Still required |
|------|-------------|-------|----------------|
| C1 | `src/ablation.py`: new `correlation_order()`; `reorder_features(..., perm=)`; `run_feature_ordering_ablation` computes ONE train permutation and applies it to all splits; ordering JSON gains `ordering_note` + `correlation_perm`. `src/visualize.py` figure footnote rewritten. Draft §6.6/§7/§8 rewritten. | done | Re-run 3 ordering ablations; confirm four identical bars; regenerate `ch4_ablation_feature_ordering.png` |
| C3 | `src/ablation.py`: pixel-shuffle gains arm B (train shuffled → test shuffled) with `shuffled_train_f1` / `retrain_drop`; `conclusion` now keys on arm B. Draft §6.6/§7/§8 restructured into arm A / arm B. *(2026-09-12: the two abstracts, §4.7 item 1 and the chapter-5 listing commentary still carried the arm-A-only inference — "a drop proves the CNN uses the spatial arrangement" — and were qualified too.)* | done | Re-run 3 pixel-shuffle ablations; fill the `[UNETI ...]` numbers |
| C4 | `src/ablation.py`: LP-FT JSON gains `seed`, `direct_ft_config`, `lpft_config`, `arm_note`. `professor-validation.md` §12.3b correction added; draft LP-FT caveat added. | done | Re-run 3 LP-FT ablations and reconcile 98.63 vs 97.22 |
| C2 | `src/visualize.py`: OF/OP annotations moved inside the per-method loop; ylim computed across all methods. | done | Regenerate `ch4_overlap_diagnostics.png` |
| C5 | `src/t2i/overlap_metrics.py`: IGTD OF/OP measured from its fitted coordinates; naive annotated (`note`, `n_active_pixels` = D). | done | Regenerate the overlap figure |

Main grid: **not touched** — the 36 CNN + 9 baseline cells and their JSONs are
unchanged, so `run_all.py` needs no re-run for Phase 1.

Colab sequence after pulling these edits (bash cells; ~45–70 min total):

```bash
for ds in breast_cancer dry_bean adult_income; do
  python src/ablation.py --dataset $ds --t2i deepinsight --cnn shallow --all
done
python src/visualize.py --ablation-only          # 3 ablation figures
python -c "import src.visualize as v; v.plot_overlap_diagnostics()"
```

`--all` runs pixel-shuffle, feature-ordering and LP-FT for each dataset;
`--ablation-only` regenerates the three ablation figures but NOT the overlap
figure, hence the one-liner (or the full `python src/visualize.py`, which also
re-runs Grad-CAM).

Expected outcome of the C1 fix: all four orderings collapse to identical F1
(96.45 / 93.37 / 66.53 %), i.e. the ordering ablation becomes an invariance
(negative) result for DeepInsight; the optional `--t2i naive` run is what would
show a real ordering effect.

---

## Phase 2 implementation status (2026-09-12)

Phase 2 (audit items C6, C10, C13–C21) is implemented in the working tree.
**No code was executed.** Nothing here requires retraining — the only new file
is `scripts/backfill_metrics.py`, a post-processing script over existing JSONs.

| Item | Change made | State | Still required |
|------|-------------|-------|----------------|
| C6 (metrics) | `src/evaluate.py`: new runs also record `f1_macro_all` (macro over ALL classes) and `balanced_accuracy`; the legacy `f1_macro` key is untouched. New `scripts/backfill_metrics.py` recomputes the same two keys for the 45 already-saved result dicts that carry a `confusion_matrix` (the 9 ablation JSONs store only `f1`/`accuracy` — no confusion matrix — so they are skipped). | done | Run `python scripts/backfill_metrics.py --write`, then `python run_all.py --aggregate` |
| C6 (labels) | `run_all.py`: console summary now prints three explicitly titled tables (`f1_macro`, `f1_macro_all`, `balanced_accuracy`) with a header that says which dataset's F1 is positive-class and which is macro; `key_cols` gains `f1_macro_all` + `balanced_accuracy`. `src/visualize.py::plot_density_vs_performance` y-label is now `F1_LABEL[dataset]` instead of a hard-coded `Macro-F1 (%)`. | done | Regenerate `ch4_density_vs_performance.png` |
| C9 (text) | Draft §6.1.3/§6.6/§8 already rewritten in Phase 1 to describe IGTD's one-row strip instead of "collisions" (naive/IGTD OF = OP = 0 is correct — the collisions belong to TINTO/DeepInsight). Verified by grep: no remaining "uz kolizije" for IGTD. | done | none |
| C10 (figure) | `src/visualize.py`: suptitle retitled to *"Classification Performance by T2I Method and Architecture (per-dataset feature density annotated in panel titles)"*. `_titles.txt` `density_perf` row updated. Draft §6.4 bullet + figure-index row updated. | done | Regenerate the figure |
| C13 + C20 | `run_all.py`: the dead, un-importable `ProgressTracker` class was deleted (never instantiated; referenced `datetime`/`timedelta` that were never imported). `src/t2i/__init__.py`: `compute_optimal_image_size` / `auto_size` marked "reference utility only — not part of the final protocol" (all reported images are 32×32). `src/train.py`: unused helpers marked in a header note. `src/t2i/s_igtd.py`: already marked NOT PART OF THE STUDY. | done | Say the same in chapter 5 when listing the code |
| C14 | `src/models/__init__.py::get_model` now maps `resnet_scratch` (with `pretrained=False`, `input_channels=1` defaults) and `verify_all_models` covers it. | done | — |
| C15 | `src/gradcam.py`: `target_class` is now actually used via `ClassifierOutputTarget`, with a fallback to the previous `targets=None` behaviour if the helper is unavailable. | done | Regenerate the 3 Grad-CAM figures and say in the caption that the map explains the *true* class |
| C16 | `src/gradcam.py`: `cm.get_cmap('jet')` → `matplotlib.colormaps['jet']`. `src/baselines/xgboost_model.py`: `use_label_encoder=False` is now only passed when the xgboost major version is < 2. `requirements.txt`: reproducibility header, freeze instructions, the missing **TINTOlib** entry added, and the unused deps flagged. | done | Generate `requirements-lock.txt` from the final Colab run |
| C17 | `src/visualize_t2i.py`: unreachable duplicate `return fig` deleted; the per-panel figure-of-merit renamed to "non-zero pixels: X %" (it is pixel coverage, not the study's feature density) and the suptitle retitled accordingly. | done | Regenerate with `python src/visualize_t2i.py` (~10 min CPU) |
| C19 | `src/ablation.py`: new `_write_json_atomic()` (temp file + `os.replace`, numpy-aware `default=`), used by all three ablation outputs. Temp names match `results/*.json.tmp`, which `run_all.py` already cleans on startup. | done | — |
| C21 | `.gitignore`: `results*.zip`, `SLR/`, `.freebuff/`, `_figure_preview.html` added. The 4 `results*.zip` (334–828 MB), `SLR/` and the SLR `.docx` are still deliberately **untracked**. | done | Decide whether the SLR `.docx` belongs in this repo at all (it is the literature review, not the seminar results paper) |

### C6 backfill — measured effect (dry run, 2026-09-12)

`python scripts/backfill_metrics.py` (dry run: reads and prints, writes nothing)
scanned 54 JSON files: **45 carry a `confusion_matrix`** (36 CNN cells + 9
baselines), the 9 ablation JSONs do not and are skipped.

The recomputation is self-validating: for **dry_bean** the stored `f1_macro`
*already is* a macro average, and `f1_macro_all` reproduces it exactly in all
15 rows (0.9379/0.9379, 0.9328/0.9328, 0.9399/0.9399, …). For the two binary
datasets the two numbers diverge exactly as predicted — mean |gap| **3.35 pp**,
max **11.40 pp**.

Numbers that matter for the write-up:

| cell | stored F1 (positive class) | `f1_macro_all` | `balanced_accuracy` |
|---|---|---|---|
| adult_income / naive / resnet (pretrained) | 57.58 % | 63.68 % | 75.41 % |
| adult_income / naive / resnet_scratch (best CNN) | 68.98 % | 77.67 % | 82.38 % |
| adult_income / igtd / shallow | 68.52 % | 77.23 % | 82.16 % |
| adult_income / xgboost (tabular baseline) | 71.43 % | 79.64 % | 83.98 % |

Three consequences:

1. The headline negative-transfer gap **survives a change of metric but changes
   size**: −11.40 pp in positive-class F1 vs **−13.99 pp** in macro-F1. The
   draft now quotes both and names the metric, so the claim cannot be attacked
   as metric-shopping.
2. The *right* statistic for "is this run better than predicting the majority
   class?" is **balanced accuracy: 75.41 % against a 75.2 % majority rate** —
   i.e. at chance for the minority class. That is a much stronger and cleaner
   statement than the accuracy comparison, and it should be quoted together
   with the C8 multi-seed repeat.
3. The "T2I is lossy vs XGBoost" finding is metric-robust but narrower:
   71.43 vs 68.98 (−2.45 pp) becomes 79.64 vs 77.67 (**−1.97 pp**) in macro-F1.

### Colab sequence for Phase 2 (no training)

```bash
# 0. syntax check
python -m py_compile src/visualize.py src/visualize_t2i.py src/ablation.py \
    src/evaluate.py src/gradcam.py src/train.py src/models/__init__.py \
    src/t2i/__init__.py src/baselines/xgboost_model.py run_all.py

# 1. backfill the two comparable metrics into the 45 existing JSONs (dry run first)
python scripts/backfill_metrics.py
python scripts/backfill_metrics.py --write

# 2. refresh the aggregate CSV and the console summary
python run_all.py --aggregate

# 3. regenerate ALL figures (t2i script is the slow one, ~10 min)
python src/visualize.py
python src/visualize_arrangement.py
python src/visualize_pipeline.py
python src/visualize_t2i.py
```

If the Phase-1 ablation re-runs are done in the same session, run steps 1–3
*after* them so the aggregate CSV picks up the new ablation keys too.

---

## Phase 3 implementation status (2026-09-12)

Phase 3 is the optional hardening layer: seed repeats (C7/C8), the rigorous
pretrained-vs-scratch control (C11) and the naive ordering add-on (C1). **None
of it repairs a defect** — C7/C8/C11 are already written down as limitations.
What it buys is turning those limitations into measurements.

| Item | Change made | State | Still required |
|------|-------------|-------|----------------|
| C7/C8 | New `scripts/seed_sweep.py`. It repeats the 7 claim-bearing cells over seeds 42–46 by calling the **same** `run_all.run_single_experiment`, so the training path is provably identical and nothing is duplicated; results go to `results/seeds/seed<N>/` so the recorded seed-42 grid is never touched; it prints mean ± sample sd per cell plus a delta ± 2 se verdict for each *claimed difference*, and writes `results/seed_summary.csv`. `run_single_experiment` gained `seed` and `split_seed`; `run_all.py` gained `--seed`, `--split-seed`, `--output-dir`. Defaults remain 42, so the recorded grid reproduces exactly; every JSON now records `seed`/`split_seed`. | done | Run the sweep in Colab; fill the `[UNETI ...]` placeholders |
| C7 (`cross_validate`) | Decision: **keep, explicitly marked**. The train.py header note now states the variance requirement it was written for is served by `seed_sweep.py`. Wiring it into a `--cv` path would duplicate what the sweep already does via the identical training path, and it retrains 5 folds per cell. | done | — |
| C11 | `run_all.py` gained the `SCRATCH_3CH` flag + `--scratch-3ch`. With it on, `resnet_scratch` is built with `input_channels=3` (conv1 architecture-identical to the pretrained arm) plus `force_imagenet_norm`. Normalisation is now decided by **one** predicate, `src/train.uses_imagenet_normalization()`, shared by `train_model`, `evaluate_model` and `generate_gradcam` — previously three independent `getattr(model, 'pretrained', False)` checks that could silently disagree, which is exactly the failure mode that would have made the control arm look fine while scoring un-normalised test images. | done | Back up the 12 `resnet_scratch` JSONs, then re-run those 12 cells with the flag |
| C1 add-on | `src/visualize.py`: the feature-ordering figure now keys on `(dataset, method)` and draws one panel per method, so the naive run appears beside DeepInsight. This also fixed a **latent bug**: the old `by_ds = {r['dataset']: r}` silently dropped one method once two existed, so the naive control would have overwritten the DeepInsight bars in the figure. The footnote is conditional on which panels are present. | done | Run `--t2i naive`; regenerate the figure |
| Durability | New `src/colab_sync.py` + hooks in `run_all.py`, `_write_json_atomic` and `seed_sweep.py`: each experiment mirrors itself to `$RESULTS_SYNC_DIR` the moment it is written, so a Colab disconnect costs at most the experiment in flight instead of hours. Additive and off by default (unset variable = no-op). | done | Set the variable per session; `python -m src.colab_sync --restore` after a reconnect |

Verification performed locally (no training): `py_compile` on all touched files;
model construction checked — default `resnet_scratch` = 1ch/no-norm (unchanged),
flag on = 3ch + ImageNet norm with `pretrained=False`, `resnet` and `shallow`
unchanged; `run_all.py --dry-run` with and without the flag; `seed_sweep.py
--dry-run`; and both the one-panel and two-panel paths of the ordering figure
rendered against synthetic ablation data into a temp directory.

### Measured time budget

Aggregated from the `train_time_sec` / `t2i_time_sec` fields of the 45 saved
result JSONs (CPU; the ablations store no timings, so their rows are
extrapolated from the matching main-grid cell):

| scope | time |
|---|---|
| main grid, 45 cells (training only) | 12,955 s = **3.60 h** |
| — of which adult_income | 9,529 s = 159 min (74 %) |
| — of which the two ResNet variants | 11,939 s = 199 min (92 %) |
| — of which breast_cancer (9 CNN cells) | 120 s = 2 min |
| T2I generation, all cells | ~582 s = ~10 min |
| **Phase 1** ablations (feature-ordering ×3, pixel-shuffling ×3, LP-FT ×3) | **≈ 80 min** (LP-FT alone ≈ 52 min) |
| **Phase 2** (backfill + aggregate) | seconds |
| **Phase 2** figures (4 scripts) | ≈ 15–30 min, zero training |
| **Phase 3** C1 naive ordering | ≈ 15 min |
| **Phase 3** C7/C8 seed sweep (7 cells × 4 extra seeds) | ≈ 2.2 h |
| **Phase 3** C11 control (12 cells) | ≈ 1.8 h |

This **supersedes the earlier 2.5–3.5 h estimate for Phase 1**, which was
guessed rather than derived. Fresh full run with everything on: ≈ 9.5–10 h CPU;
resuming from the existing grid, ≈ 6 h. On a Colab GPU the ResNet-dominated items
(seed sweep, C11) drop by roughly an order of magnitude, which is the difference
between Phase 3 being a day and being an hour.

### Colab sequence for Phase 3

```bash
# 0. durability FIRST (Python cell, before any long run):
#    import os; os.environ['RESULTS_SYNC_DIR'] = '/content/drive/MyDrive/t2i-results'
#    See "Crash-safe long runs" above. Every job prints [sync] ON/OFF as line 1.

# C1 add-on — the positive counterpart to the invariance result (~15 min)
for ds in breast_cancer dry_bean adult_income; do
  python src/ablation.py --dataset $ds --t2i naive --feature-order
done

# C7/C8 — error bars on the cells that carry a claim (~2.2 h CPU)
python scripts/seed_sweep.py --dry-run        # confirm plan and cost first
python scripts/seed_sweep.py                  # 7 cells x seeds 42-46
# cheaper: just the headline trio
python scripts/seed_sweep.py \
  --cells adult_income/naive/resnet,adult_income/naive/resnet_scratch,adult_income/naive/shallow

# C11 — the rigorous controlled comparison, ONLY if the confound is challenged (~1.8 h CPU)
mkdir -p results/backup_scratch_1ch
mv results/*resnet_scratch* results/backup_scratch_1ch/   # keep the 1-channel arms
python run_all.py --archs resnet_scratch --scratch-3ch

# aggregate and regenerate figures LAST, so they pick up the new files
python run_all.py --aggregate
python src/visualize.py --ablation-only
python src/visualize_t2i.py
```

`--seed` / `--split-seed` default to 42, so nothing above changes the recorded
grid unless a flag is passed. `seed_sweep.py` never writes into `results/*.json`.

### Crash-safe long runs (`RESULTS_SYNC_DIR`)

Phases 1 and 3 are hours of CPU, `results/` is gitignored, and a Colab VM is
ephemeral — a disconnect or an idle timeout destroys work that cannot be
recovered from git. Every experiment now mirrors its own output the instant it is
written, via `src/colab_sync.py`, so an interrupted session loses at most the one
experiment that was in flight.

Enable it once per session, in a **Python** cell (so every later `%%bash` cell
inherits it):

```python
import os
os.environ['RESULTS_SYNC_DIR'] = '/content/drive/MyDrive/t2i-results'
```

With the variable unset every sync call is a no-op, so local runs are unchanged.
Each long job prints its status as its first line, which is what makes the
protection visible rather than assumed:

- `[sync] ON - mirroring finished files to ...` — protected.
- `[sync] OFF - set RESULTS_SYNC_DIR=...` — results live only on the VM.
- `[sync] *** NOT WRITABLE *** ...` — **the dangerous case**: it looks protected
  and is not (Drive not mounted, full, or a path that cannot be created). Fix it
  before starting, not after.

What is covered: every CNN cell and baseline JSON, each of the three ablation
JSONs (`_write_json_atomic` mirrors on the way out), the state_dict of every cell
that saves weights, and `results/seed_summary.csv`. Mirroring never raises — a
dead drive cannot kill a two-hour run, it warns once per distinct error.

**Recovering after a disconnect.** Mount Drive, then:

```bash
python -m src.colab_sync --restore      # copy the mirror back into results/
python run_all.py                       # resume skips every completed cell
for ds in breast_cancer dry_bean adult_income; do
  python src/ablation.py --dataset $ds --t2i deepinsight --cnn shallow --all
done
```

`--restore` never overwrites a local file, so a partly finished local run wins
over a stale mirror. Ablation runs now resume by default, and an existing JSON is
skipped **only** if it carries the keys the current code produces
(`ABLATION_MARKERS`) — a pre-fix file is therefore correctly re-run instead of
being trusted; `--force` recomputes regardless. `run_all.py` and `seed_sweep.py`
were already resume-aware.

To save everything at the end of a session (weights and figures included):

```bash
python -m src.colab_sync
```

That exits non-zero when the sync directory is unset, deliberately: in a `%%bash`
cell it stops you from believing a session was backed up when it was not.

---

## C1 — CRITICAL: the feature-ordering ablation is invalid (per-split permutation)

**Where.** `src/ablation.py:172` (`reorder_features`, `order == 'correlation'`),
called at `src/ablation.py:215-217`.

```python
elif order == 'correlation':
    corrs = np.abs(np.array([
        np.corrcoef(X[:, i], y)[0, 1] if np.std(X[:, i]) > 0 else 0
        for i in range(X.shape[1])
    ]))
    sorted_idx = np.argsort(-corrs)
    return X[:, sorted_idx]
...
X_train_r = reorder_features(X_train, order, y_train)   # sorted by y_train
X_val_r   = reorder_features(X_val,   order, y_val)     # sorted by y_val
X_test_r  = reorder_features(X_test,  order, y_test)    # sorted by y_test
```

**Core problem.** The permutation is recomputed from each split's *own* labels.
Train, validation and test therefore receive **three different column orders**.
The T2I transformer is fitted on the train permutation, then applied to the test
permutation, so every test feature value is written to the pixel coordinate that
belongs to whatever train column happens to sit at that index. The network is
evaluated on a scrambled layout.

**Evidence (reproduced).** On `breast_cancer` (30 features):

| check | result |
|---|---|
| same permutation applied to fit and transform | images bit-identical for `original`, `random`, `reversed`, `correlation` (max |diff| = 0.0) |
| train permutation vs test permutation | **not equal** (`ptr != pte`) |
| test images, train-perm fit + test-perm transform | differ on **2.34 % of pixels**, max diff 0.79 |

**Why it matters.** The draft and both audit docs build a causal story on top of
this artefact:

- `Plan/seminar2-rad-nacrt.md` §6.6 / §7 / §8 — "jedini poredak koji zaista
  menja raspored jeste sortiranje po korelaciji… dosledno škodi (−2,27 do
  −4,48 pp)".
- `Plan/professor-validation.md` §12.2 — "correlation-sorted is worst
  (91.97/91.07/64.26)".
- `src/visualize.py` feature-ordering footnote — "only the correlation-sorted
  key genuinely changes the layout".

All three are **false**. DeepInsight (as wrapped) is provably column-order
invariant; the drop is the misalignment above. A professor who re-runs
`reorder_features` with a single permutation will get four identical bars and
conclude the ablation — and the chapter that rests on it — is wrong.

**Fix (code).** Derive one permutation from the training split and reuse it for
every split. Replace the split-dependent branch with an explicit permutation
argument:

```python
def correlation_order(X, y):
    """Single ordering of columns by |corr(feature, y)| on TRAIN only."""
    corrs = np.abs([np.corrcoef(X[:, i], y)[0, 1] if np.std(X[:, i]) > 0 else 0.0
                    for i in range(X.shape[1])])
    return np.argsort(-np.asarray(corrs))

def apply_order(X, order, perm=None, y=None):
    if order == 'original':   return X
    if order == 'reversed':   return X[:, ::-1]
    if order == 'random':
        return X[:, np.random.RandomState(42).permutation(X.shape[1])]
    if order == 'correlation':
        return X[:, perm]
    raise ValueError(order)
```

Inside `run_feature_ordering_ablation` (before the ordering loop):

```python
perm = correlation_order(X_train, y_train)     # train-derived, fixed
...
X_train_r = apply_order(X_train, order, perm, y_train)
X_val_r   = apply_order(X_val,   order, perm, y_val)
X_test_r  = apply_order(X_test,  order, perm, y_test)
```

**Expected outcome after the fix.** All four orderings produce identical images
and identical F1 (the invariance is the real finding). Either:

1. **Preferred:** report the invariance as the result, and delete the false
   "correlation hurts" interpretation from §6.6/§7/§8 and from the figure note.
2. **Optional add-on:** run the same ablation with `--t2i naive` — naive is
   genuinely order-dependent (row-major placement), so it demonstrates a real
   ordering effect. `Plan/final-writeup-plan.md` §3.3 already proposed this.

**Re-run required: YES.**
- `python src/ablation.py --dataset breast_cancer --t2i deepinsight --cnn shallow --feature-order`
- same for `dry_bean`, `adult_income`
- then `python src/visualize.py --ablation-only`
- Rewrite the ordering prose in `Plan/seminar2-rad-nacrt.md` §6.6/§7/§8 and the
  §12.2/§12.3 lines in `Plan/professor-validation.md`.

Cost: 4 trainings × 3 datasets at ShallowCNN (median 4 s / 73 s / 301 s per
cell) → **~25–35 min**. Do **not** touch the main grid.

---

## C2 — HIGH: overlap-diagnostics figure annotates the wrong bars

**Where.** `src/visualize.py:2286` and `:2326`.

```python
for j, method in enumerate(T2I_METHODS):
    of_vals = [...]
    bars = ax.bar(...)
    ...
for bar, val in zip(ax.patches, of_vals):   # <-- OUTSIDE the j loop
    ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0.4),
            f'{val:.1f}', ...)
```

**Core problem.** After the loop, `of_vals` holds the **last** method's values
(IGTD = all zeros) and `ax.patches` holds all **12** bars (4 methods × 3
datasets). `zip` therefore labels the *first three* bars — the **naive** group —
with IGTD's zeros, and never labels TINTO/DeepInsight, which carry the only
non-zero OF/OP in the study.

**Evidence (reproduced).** With injected values I get:

```
axes 0 (OF): patches=12, texts=[('0.0', -0.3), ('0.0', 0.7), ('0.0', 1.7)]
axes 1 (OP): patches=12, texts=[('0.0', -0.3), ('0.0', 0.7), ('0.0', 1.7)]
```

**Why it matters.** The figure is cited in §6.4 as the quantitative source of
the TINTO/DeepInsight collision numbers (78/104 and 70/104 on adult). Those
numbers are not printed anywhere on the figure; the only printed numbers are
`0.0` on the zero-height naive bars. A reviewer reading the figure cannot trace
the claim.

**Fix.** Move the annotation inside the per-method loop, using that loop's own
`bars`:

```python
for j, method in enumerate(T2I_METHODS):
    of_vals = [all_overlap[ds].get(method, {}).get('of_percent', 0) for ds in DATASETS]
    bars = ax.bar(..., fill=not is_zero_method)
    if is_zero_method:
        for bar in bars:
            bar.set_hatch('/'); bar.set_alpha(0.4)
    for bar, val in zip(bars, of_vals):
        ax.text(bar.get_x() + bar.get_width()/2, max(bar.get_height(), 0.4),
                f'{val:.1f}', ha='center', va='bottom', fontsize=7.5,
                fontweight='bold', color='#333333' if val > 0 else '#777777')
```

Repeat identically for the OP panel with `op_vals`.

**Re-run required: YES (figure only).**
`python src/visualize.py` (or delete `results/figures/ch4_overlap_diagnostics.png`
and regenerate). No numeric rerun: `compute_overlap_all_methods` is called live.

---

## C3 — HIGH: the pixel-shuffle ablation measures brittleness, not structure use

**Where.** `src/ablation.py:31` (`shuffle_pixels`), `:60`
(`run_pixel_shuffling_ablation`). It trains on **original** images and
evaluates on **shuffled** ones:

```python
model, history = train_model(model, train_loader, val_loader, config_train)  # original
...
shuffled_metrics = evaluate_model(model, shuffled_loader, num_classes)       # shuffled
```

**Core problem.** This is a train/test distribution-shift test: the model is
asked to classify an input transform it never saw. It answers "is the network
brittle to a permuted pixel order", not "does the network exploit spatial
structure when learning". The figure's own conclusion string
(`'spatial_structure_matters'`) and the draft §6.6/§7/§8 then over-claim.

**Evidence.** Dry Bean F1 93.37 → 8.16 (accuracy 27.07 % ≈ majority 26.1 %) is
consistent with uniform output collapse, i.e. a model confronted with
out-of-distribution input.

**Why it matters.** A professor will say: "you destroyed the structure at test
time; of course it fails — retrain on shuffled images and see whether it can
still reach the same F1. That is the ablation." Interpreted strictly, the
current design cannot separate "the CNN needs spatial structure" from "the CNN
is not invariant to arbitrary pixel permutations".

**Fix.** Train and evaluate two arms:

```python
# Arm A (keep): train original, test original  -> reference
# Arm B (add):  train shuffled, test shuffled  -> does structure still help?
train_sh = shuffle_pixels(train_imgs, seed=42)
val_sh   = shuffle_pixels(val_imgs,   seed=42)
test_sh  = shuffle_pixels(test_imgs,  seed=42)
loader_tr, loader_va = prepare_loaders(train_sh, y_train, val_sh, y_val)
model_sh, _ = train_model(model_sh, loader_tr, loader_va, config_train)   # same seed
shuffled_metrics = evaluate_model(model_sh, shuffled_loader, num_classes)
```

Report both. If the shuffled-trained model recovers near-baseline F1, the
honest conclusion is "the CNN does not rely on the *specific* layout, only on
the marginal pixel statistics" — which is a *different, more defensible*
finding, and it must replace the current headline.

Keep the current arm's numbers in the JSON under distinct keys
(`original_f1`, `shuffled_eval_f1`, `shuffled_train_f1`) so no consumer breaks.

**Re-run required: YES, small.**
`python src/ablation.py --dataset {ds} --t2i deepinsight --cnn shallow --pixel-shuffle`
×3 datasets. Cost: one extra ShallowCNN training per dataset → **~10–15 min**.
Then `python src/visualize.py --ablation-only` and update §6.6/§7/§8.

---

## C4 — HIGH: the same cell has two different F1 values (main table vs LP-FT)

**Where.** `src/ablation.py:273` (`run_lpft_ablation`, direct-FT arm) vs
`run_all.py::run_single_experiment`.

**Evidence (from the result JSONs).** Same dataset, method, architecture, LR and
seed:

| cell | main table | ablation `direct_ft` | delta |
|---|---|---|---|
| breast_cancer / deepinsight / resnet | **97.22 %** | **98.63 %** | **+1.41 pp** |
| dry_bean / deepinsight / resnet | 93.79 % | 93.74 % | −0.05 pp |
| adult_income / deepinsight / resnet | 66.19 % | 66.44 % | +0.25 pp |

**Core problem.** `Plan/professor-validation.md` §12.3 claims the ablation cells
"reproduce the main table almost exactly — bit-identical on breast". That is
true for the *ordering* `original` cell (96.45 = 96.45) but **false** for the
LP-FT cell. The draft (§6.6 LP-FT table) prints 98.63 % for a configuration the
main table (§6.1.3) prints as 97.22 %.

**Why it matters.** Paper numbers must be traceable to one run. Two different
values for the same experimental configuration, in the same document, with the
reproducibility note explicitly saying they match, is exactly the kind of thing
that destroys trust in the whole table. It also hints that the pipeline is not
seed-deterministic at this scale — which is *load-bearing* for C8 (the
"deterministic seed 42" claim attached to the −11.40 pp headline).

**Fix.** Do all of the following:

1. Re-run the LP-FT ablation from the current commit on the current machine
   (`python src/ablation.py --dataset {ds} --t2i deepinsight --lpft`, ×3).
2. If the direct-FT arm still differs from `all_experiments.csv`, that is
   CPU non-determinism — then **remove** the "bit-identical / reproduces the
   main table" claim from `plan/professor-validation.md` §12.3 and from the
   draft's reproducibility note, and replace it with: "ablation arms are
   separate independent runs; their 'original' / direct-FT cells are reported
   as run, not as reproductions".
3. **Never** present the ablation direct-FT number as a second estimate of the
   main-table cell. State explicitly in the caption that the LP-FT comparison is
   internally paired (direct vs LP-FT within the same run).
4. To remove the ambiguity entirely, seed a single explicit run per arm and
   log `torch`/CPU versions in the JSON (see C7's fix).

**Re-run required: YES, small.**
3 LP-FT ablations (2 ResNet-18 trainings each): breast ~30 s, dry_bean ~15 min,
adult ~50 min → **~1–1.2 h**. No main-grid rerun.

---

## C5 — MEDIUM: OF/OP for naive and IGTD are hard-coded, not measured

**Where.** `src/t2i/overlap_metrics.py:97` and `:103`.

```python
results['naive'] = {'OF': 0.0, 'OP': 0.0, ..., 'n_active_pixels': 0, ...}
results['igtd']  = {'OF': 0.0, 'OP': 0.0, ..., 'n_active_pixels': 0, ...}
```

**Core problem.** Two of the four methods in the study never have their overlap
computed — the values are asserted. `n_active_pixels: 0` is also wrong metadata
for a method that occupies D pixels. The figure then hatches these bars as
"zero by design", which is honest presentation but advertises that the numbers
are not measurement.

**Evidence (reproduced).** IGTD coordinates are a degenerate strip
(row 0, columns 0…D−1) for both breast (D=30) and dry_bean (D=16); measured
overlap from those coordinates is OF = OP = 0 %. So the hard-coded value is
*correct*, but only by accident of the current library version. For reference,
the computed values for the two projection methods on breast are
DeepInsight OF 6.67 / OP 3.45 and TINTO OF 13.33 / OP 7.14.

**Why it matters.** "You hard-coded two of four bars to zero" is a serious
review remark, even when the value is right. It also silently breaks if TINTOlib
changes IGTD's placement (the guide already warns IGTD can move to a narrow
strip).

**Fix.** Compute instead of asserting:

```python
# IGTD: use the fitted coordinate map (same path as deepinsight/tinto)
try:
    from .igtd import IGTD
    ig = IGTD(image_size=image_size); ig.fit(X_train, y_train)
    coords = ig.get_coordinates()
    base = compute_overlap(coords, image_size) if coords is not None else None
    results['igtd'] = ({**base, 'of_percent': base['OF'], 'op_percent': base['OP']}
                       if base else {'OF': 0.0, 'OP': 0.0, 'of_percent': 0.0,
                                     'op_percent': 0.0, 'error': 'no coordinates'})
except Exception as e:
    results['igtd'] = {'OF': 0.0, 'OP': 0.0, 'of_percent': 0.0,
                       'op_percent': 0.0, 'error': str(e)}

# Naive: pixel collisions are undefined for a padded 1-feature-per-cell grid
# scaled by bicubic; say so explicitly rather than emitting a bare 0.
results['naive'] = {'OF': 0.0, 'OP': 0.0, 'of_percent': 0.0, 'op_percent': 0.0,
                    'n_features': X_train.shape[1],
                    'n_active_pixels': X_train.shape[1],   # every feature has a cell
                    'n_overlapped_features': 0, 'n_overlapped_pixels': 0,
                    'note': 'pad+reshape: one feature per cell before resize'}
```

**Re-run required: NO** (no model retraining). Regenerate
`results/figures/ch4_overlap_diagnostics.png` only. Cost: 1 IGTD fit per dataset
(~seconds–1 min each).

---

## C6 — MEDIUM: `f1_macro` stores binary positive-class F1; labels disagree in two places

**Where.**
- `src/evaluate.py::_compute_metrics` — `avg = 'macro' if num_classes > 2 else 'binary'`,
  stored under the key **`f1_macro`** (same for `precision_macro` / `recall_macro`).
- `run_all.py:431` — prints `"=== Summary: Macro-F1 (%) by T2I Method and Architecture ==="`.
- `src/visualize.py:1223` — `plot_density_vs_performance` hard-codes
  `ax.set_ylabel('Macro-F1 (%)')` on **all three** panels, even though
  `F1_LABEL` (`visualize.py:65`) correctly distinguishes per dataset.

**Core problem.** For `breast_cancer` and `adult_income` the stored value is
F1 of class 1 (benign = **majority** for breast; `>50K` = **minority** for
adult). For `dry_bean` it is true macro-F1. The key name, the aggregate printout
and the density-figure axis therefore mislabel two thirds of the study and imply
a cross-dataset comparability that does not exist.

**Why it matters.** (a) A reviewer reading `all_experiments.csv` sees a column
called `f1_macro` and will treat the numbers as comparable — they are not.
(b) The density figure axis is simply wrong for breast/adult while every other
figure uses `F1_LABEL`. This was already flagged in `professor-validation.md`
§9.4 but the *figure* half was never fixed.

**Fix (no retraining).**

1. Add, in a one-off post-processing script, `f1_binary_macro` and
   `balanced_accuracy` to every CNN/baseline/ablation JSON, recomputed from the
   saved `confusion_matrix`:

   ```python
   cm = np.array(row['confusion_matrix'])
   tp, fn, fp, tn = cm[1,1], cm[1,0], cm[0,1], cm[0,0]
   f1_pos = 2*tp / (2*tp + fp + fn)
   f1_neg = 2*tn / (2*tn + fn + fp)
   row['f1_binary_macro'] = (f1_pos + f1_neg) / 2
   row['balanced_accuracy'] = 0.5 * (tp/(tp+fn) + tn/(tn+fp))
   ```

2. Point the density-, per-class-, heatmap- and baseline-y-axis labels at
   `F1_LABEL[dataset]` (fix `visualize.py:1223` to
   `ax.set_ylabel(F1_LABEL[dataset], fontsize=10)`).
3. Change `run_all.py:431` to print the honest header, and add the new column to
   `aggregate_results`' `key_cols`.
4. In the tables, state the positive class explicitly: "F1(benign)" for breast,
   "F1(>50K)" for adult.
5. Do **not** rename the existing `f1_macro` key (it is consumed by
   `visualize.py`, `ablation.py`, `run_all.py`); add the new keys alongside.

**Re-run required: NO.** Post-processing script over the 54 JSONs + regenerate
figures.

---

## C7 — MEDIUM: single split, no uncertainty, and an unused CV implementation

**Where.** `src/train.py:294` (`cross_validate`) is fully implemented but **never
called** anywhere. `run_all.py` trains each cell once. No `± std`, no CI, no
significance test anywhere in the results.

**Core problem.** Every "method X beats Y by 0.5 pp" statement is a point
estimate from a single 70/10/20 split. With 114 breast test rows, one flipped
prediction moves F1 by ≈0.9 pp — i.e. several claimed differences are inside one
prediction's worth of noise. Shipping an unused CV function makes it look like
the mitigation was written and then abandoned.

**Why it matters.** This is the single most predictable professor question
("how do you know that difference is real?"), and it is already listed as a
limitation — but "listed as a limitation" and "shipped an unused CV function"
together read worse than either alone.

**Fix (pick one; both are honest):**

- **Cheap / recommended:** run the three or four cells that carry a *claim*
  (e.g. `breast_cancer/deepinsight` vs `tinto` on shallow; the adult
  `naive/resnet` negative-transfer cell; the adult TINTO-vs-naive gap) over 5
  seeds (42–46), report mean ± std, and state for the rest: "single split;
  differences below X pp are not interpreted". Cost ≈ 5 × per-cell time for
  ≤ 6 cells.
- **Thorough:** call the existing `cross_validate` for a subset and report
  mean ± std per method. Note that `cross_validate` retrains 5 folds per cell,
  so scope it to 1–2 datasets.

Also either (a) wire `cross_validate` into a `--cv` path and document it, or
(b) delete it; leaving it dead is the worst of both.

**Re-run required: YES if adopted.** Scope as above; the main grid stays.

---

## C8 — MEDIUM: the headline negative-transfer result is a single run

**Where.** `results/adult_income_naive_resnet.json` — F1 57.58 %, accuracy
64.70 %, **17 epochs** (early-stopped). Compare the sibling cells
`adult_income/naive/resnet_scratch` 68.98 % / 81.05 % and
`adult_income/naive/shallow` 68.94 % / 80.75 %.

**Core problem.** Accuracy 64.70 % is **below the 75.2 % majority rate**, so this
run is worse than predicting the majority class. The draft makes it the headline
"The pretrained backbone fails / negative transfer" result and the guide calls
it "deterministic (seed 42)" — but C4 shows the pipeline is not bit-reproducible
across runs, so the determinism claim is unsupported.

**Why it matters.** If a single unlucky early-stop produced 57.58 % and a repeat
gives ~68 %, the chapter's main transfer-learning claim collapses. If it is
stable, the claim is strong — but you cannot know without repeats. This is a
five-minute question to ask and a one-line rebuttal to be missing.

**Fix.** Re-run the cell with seeds 42–46 (LR 1e-3, everything else fixed),
report mean ± std and the full confusion matrix per seed. Add the outcome to
§6.3 and the discussion. If the spread is small, the claim stands and gains
error bars; if it is large, reframe to "unstable on this configuration".

**Re-run required: YES.** 5 trainings of pretrained ResNet-18 on adult
(~20–28 min each) → **~2 h**, or ~1 h if run on GPU.

---

## C9 — MEDIUM: draft says IGTD has collisions; the diagnostics say it has none

**Where.** `Plan/seminar2-rad-nacrt.md` §6.1.3 ("traka po redosledu i kolizije")
and §8 ("traka bez prostornog grupisanja, uz kolizije"); vs
`src/t2i/overlap_metrics.py` (IGTD OF = OP = 0) and the hatched "zero by design"
IGTD bars.

**Core problem.** The text attributes pixel collisions to IGTD; the computed /
hard-coded diagnostics (and my direct measurement — IGTD places all D features
on row 0, unique columns) show **zero** collisions. IGTD's real defect is the
degenerate one-row strip (no 2-D grouping), which is a different explanation.

**Fix.** Replace "uz kolizije" with the accurate mechanism: "all features are
compressed into a single row (one-pixel-tall strip), so the CNN sees no 2-D
neighbourhood structure". Keep collisions for DeepInsight/TINTO only (adult
78/104 and 70/104).

**Re-run required: NO.** Text edit only.

---

## C10 — MEDIUM: "density vs performance" figure has no density axis

**Where.** `src/visualize.py:1133-1256` (`plot_density_vs_performance`); `_titles.txt` line 6.

**Core problem.** The figure plots F1 grouped by T2I method × architecture; the
density is constant within a dataset and only appears in the panel title. It is
no longer a density-vs-performance plot. Two artefacts still describe it as one:
the figure's suptitle "Feature Density vs Classification Performance", and
`_titles.txt` (`density_perf: Feature Density (%) | Macro-F1 (%) | ...`), which
looks like it feeds the document's axis descriptions.

**Fix.** Retitle to something truthful — "Classification Performance by T2I
Method and Architecture (per-dataset feature density annotated)" — and update
`_titles.txt` and the draft §6.4 sentence that calls it "gustina naspram F1 po
ćeliji". Either that, or switch the panel to a genuine density axis by plotting
all three datasets in one axes (density is 2.9 / 1.6 / 10.5 %, only three
distinct x-values).

**Re-run required: NO** (or figure-only if retitled).

---

## C11 — MEDIUM (documented): pretrained vs from-scratch ResNet is confounded

**Where.** `run_all.py::create_cnn_model` — `resnet` → 3-channel + ImageNet
normalisation; `resnet_scratch` → 1-channel raw grayscale. Documented in
`professor-validation.md` §9.1 and rephrased in the draft (b4e9bdd).

**Core problem (still).** The transfer-delta figure's caption and §6.3 frame the
difference as "ImageNet transfer", while the two arms also differ in input
representation. `cvpr`-grade reviewers treat that as a controlled experiment
failure; for a seminar the rephrasing is acceptable **only if** the figure axis
is renamed too ("Δ between the two ResNet input pipelines") and the caption
repeats the caveat.

**Fix (choose).**
- **Cheap:** rename the figure axis/caption and add the caveat to §6.3 (text
  only, no rerun).
- **Rigorous:** make `resnet_scratch` use `input_channels=3` +
  ImageNet normalisation so the arms differ only in weight init. This requires
  the `run_all.py` change **and** a 12-cell rerun.

**Re-run required: only for the rigorous option.** 12 `resnet_scratch` cells:
breast 3×16 s, dry_bean 3×~435 s (~22 min), adult 3×~1276 s (~64 min) →
**~1.5 h**. Delete only those 12 JSONs + `.pt` files and let the resume logic
re-train them; the deterministic cells are skipped.

---

## C12 — LOW/MEDIUM (documented): one-hot vocabulary fitted before the split

**Where.** `src/preprocessing.py:163` (`pd.get_dummies` on the concatenated
frame) runs before `preprocess()` at `:214` splits.

**Core problem.** Category vocabulary is influenced by test rows. Impact here is
near zero (all categories appear in train after the `?` fix), and it is common
practice — but it is trivially avoidable.

**Fix.** Encode on train+val, `reindex(columns=..., fill_value=0)` for test.

**Re-run required: YES, whole adult column of the grid** if changed (the encoded
feature matrix changes). Do **not** change this unless you are already rerunning;
otherwise keep it in the limitations paragraph (as currently written).

---

## C13–C21 — code quality, robustness, hygiene

### C13 — `run_all.py::ProgressTracker` references undefined names
`run_all.py:101` uses `datetime.now()` / `timedelta` but the module imports only
`time`. The class is never instantiated today (dead), so it never fires — but a
reviewer scanning the file will see an unimportable code path. **Fix:** add
`from datetime import datetime, timedelta` or delete `ProgressTracker` (the
inline prints already cover progress). No rerun.

### C14 — `src/models/__init__.py::get_model` omits `resnet_scratch`
`get_model` maps only `shallow`/`resnet`/`vit` (`:13-15`). `run_all.py` bypasses
it with its own factory, so nothing breaks now, but the file's public factory is
wrong. **Fix:** add `'resnet_scratch': ResNetWrapper` with
`preset=dict(pretrained=False, input_channels=1)`, or delete `get_model` /
`verify_all_models` as unused. No rerun.

### C15 — Grad-CAM ignores `target_class`
`src/gradcam.py:43` accepts `target_class` and `:75` calls
`cam(..., targets=None)`, so the heatmap is always for the **predicted** class;
the parameter is dead and the figure's `True=…, Pred=…` title invites the reader
to assume the map explains the true class. **Fix:** either use it
(`from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget;
cam(..., targets=[ClassifierOutputTarget(int(target_class))])` and decide
pred vs true explicitly), or drop the parameter and rename the panel to
"Grad-CAM (predicted class)". If changed, regenerate the 3 Grad-CAM figures.
No retraining needed.

### C16 — deprecated colormap API and unpinned dependencies
`src/gradcam.py:114` uses `cm.get_cmap('jet')` (removed in matplotlib ≥ 3.9).
`requirements.txt` pins only lower bounds (`matplotlib>=3.7`, `xgboost>=2.0`,
`torch>=2.0`), and `src/baselines/xgboost_model.py` passes
`use_label_encoder=False`, which newer XGBoost versions ignore or reject.
**Fix:** use `matplotlib.colormaps['jet']`; pin exact versions used for the
final run (torch, scikit-learn, xgboost, TINTOlib, matplotlib) in
`requirements.txt` or a `requirements-lock.txt`; verify the XGBoost call against
the installed version. No rerun (unless a version bump changes results — hence
pin).

### C17 — duplicate `return`; two different "density" definitions
`src/visualize_t2i.py:147-148` has an unreachable second `return fig`. The same
file labels `Density: {x}%` where `x = fraction of pixels > 0.01` — a *different*
quantity from the study's `n_features / H·W` density. The draft separates them in
§6.4, but the figures should too. **Fix:** delete the dead return; rename the
per-panel label to "non-zero pixels: X %". Figure-only rerun if changed.

### C18 — TINTO pixel scale depends on call order
`src/t2i/tinto.py:110`: `_pix_min/_pix_max` are captured on the **first**
`transform()` call and reused thereafter. Correctness currently relies entirely
on `run_all.py` always transforming the train split first; a future caller that
transforms val/test first would silently produce differently-scaled images.
`visualize.py` already works around it via the persisted `t2i_pixel_range`.
**Fix:** make the scale an explicit `fit`-time artefact (compute it from a
train-transform inside `fit`, or require an explicit `set_pixel_range`). No
rerun (behaviour on the existing call order is unchanged).

### C19 — ablation JSONs are not written atomically
`src/ablation.py:144, 263, 385` use plain `open(...,'w') + json.dump`, while
`run_all.py` uses `.json.tmp` + `os.replace`. A kill mid-write leaves a truncated
ablation JSON. **Fix:** reuse the atomic-write helper for all three ablation
outputs. No rerun.

### C20 — dead code inventory
`src/train.py`: `save_checkpoint`, `load_checkpoint`, `zscore_normalize`,
`cross_validate` (see C7) — none called. `src/models/__init__.py`: `get_model`,
`verify_all_models` — unused. `run_all.py`: `ProgressTracker` (see C13).
`src/t2i/__init__.py`: `compute_optimal_image_size` / `auto_size` — the pathway
exists but `run_all.py` never sets `auto_size=True`, so all images are 32×32.
`src/t2i/s_igtd.py`: kept for reference, unregistered (documented). **Fix:**
delete or explicitly mark each as "reference / not used in the final protocol"
and say so in chapter 5. A professor reading chapter 5 code listings should not
find functions that never execute.

### C21 — repository hygiene
Untracked duplicates: `results.zip`, `results (1-3).zip`, `01_eda.ipynb` vs
`01_eda_executed.ipynb`, empty `New folder*`, `checkpoints/` (empty?),
`__pycache__/`, and the `.docx` — note the `.docx` is the **SLR literature
review**, not the seminar results paper, so it does not belong next to
`seminar2-rad-nacrt.md` without a name that says so. `final-writeup-plan.md`
§3.1 already lists the zips. **Fix:** archive or delete; add build artefacts to
`.gitignore`. No rerun.

---

## Verified clean (so the list is not one-sided)

Worth keeping on record because they are the things a professor checks first:

- No leakage in the main path: `StandardScaler` fit on train only; T2I
  coordinate mapping fit on train only; class weights from `y_train` only;
  baselines train on `X_train` only and are evaluated on the same `X_test`.
- Result files are complete and internally consistent: 36 CNN + 9 baseline +
  9 ablation JSONs; confusion matrices square and summing to the test sizes;
  `all_experiments.csv` agrees with the JSONs; atomic writes with
  corrupt-file-aware resume in `run_all.py`.
- Ranges are plausible: breast 95.0–97.2, dry bean 90.3–94.0, adult 57.6–69.0
  CNN vs XGBoost 71.43 (the honest "transformation is lossy" finding).
- Overlap coordinates are 0-indexed and in `[0, 31]`, so `int(round(...))` in
  `compute_overlap` has no off-by-one (checked: breast DeepInsight OF 6.67 /
  OP 3.45, TINTO 13.33 / 7.14).
- Naive normalisation and the train-derived clipping are consistent across
  splits (the old per-split clip bug is genuinely fixed).

---

## Ordered action plan

**Phase 1 — integrity fixes that need small reruns (~2 h + figure regen)**
1. C1 fix `reorder_features` → single train-derived permutation; re-run 3
   ordering ablations; rewrite §6.6/§7/§8 and §12.2/§12.3. *(~30 min)*
2. C3 extend pixel-shuffle to a shuffled-train arm; re-run 3 ablations. *(~15 min)*
3. C4 re-run 3 LP-FT ablations; reconcile the 97.22 vs 98.63 discrepancy and
   fix the §12.3 reproducibility claim. *(~1 h)*
4. C2 + C5 fix the overlap figure annotation and compute IGTD/naive overlap;
   regenerate the figure. *(~5 min)*

**Phase 2 — no-rerun fixes (same day)** — *implemented 2026-09-12, see the
Phase 2 status section above*
5. C6 add `f1_binary_macro` + `balanced_accuracy` from saved confusion matrices;
   fix `F1_LABEL` usage in `plot_density_vs_performance` and the `run_all.py`
   summary header.
6. C9, C10 fix the IGTD-collision wording and the density-figure
   title/`_titles.txt`.
7. C13–C17, C19 delete/fix the dead code paths, deprecated API and non-atomic
   ablation writes.
8. C20, C21 clean the repo.

**Phase 3 — optional hardening (only if time allows / professor pushes)**
9. C7 seed-repeat the claim-bearing cells; either wire or delete
   `cross_validate`.
10. C8 five seeds on the adult `naive/resnet` headline cell. *(~1–2 h)*
11. C11 rigorous option — 3-channel from-scratch ResNet; 12-cell rerun. *(~1.5 h)*
12. C1 add-on — run the ordering ablation with `--t2i naive` to demonstrate a
    genuine ordering effect. *(~10 min)*

**Overall rerun verdict.** The **36-cell main grid does not need to be
re-run** for C1–C10. It needs re-running only if you adopt the rigorous C11
option (12 `resnet_scratch` cells) or C12 (whole adult column). All figures must
be regenerated after the Phase-1/2 fixes:
`python src/visualize.py`, `python src/visualize_arrangement.py`,
`python src/visualize_pipeline.py` (and `python src/visualize_t2i.py` only if
C17's label change is made — that script takes ~10 min CPU).
