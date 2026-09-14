# Results validation — post-run check, 14 Sept 2026

Scope: the `results/` tree after the finalisation run (repeats of the flagged cell,
C6 backfill, aggregate, stability table, figures). Every statement below was checked
against the files on disk, not from memory. Read-only checks except where marked
**fixed here**.

---

## 1. Status summary

| # | Check | Verdict |
|---|---|---|
| 1 | Inventory: 36 grid + 9 baselines + 12 ablations = **57 root JSONs**, all parse | **PASS** |
| 2 | The flagged cell now records the stable value (`f1_macro` 0.6908, acc 0.8198, `best_epoch` 33, 48 epochs) | **PASS** |
| 3 | The anomalous run is preserved as evidence in `results/backup_pre_rerun/` (57.58 %, `best_epoch` 2, 17 epochs, `val_loss_oscillation` 6.71) | **PASS** |
| 4 | Checkpoint provenance written by the new code (V1 machinery works) | **PASS** — see §2 |
| 5 | C6 backfill (`f1_macro_all`, `balanced_accuracy`) in the 36 grid JSONs | **FAILED, then fixed here** — 1/36 → 45/45 |
| 6 | `results/all_experiments.csv` matches the grid | **FAILED, then fixed here** — 10 rows → 45 (36 CNN + 9 baselines) |
| 7 | `results/stability_table.md` present and identical to the numbers in §6.7 / Table 6.5 | **PASS** |
| 8 | All 30 figures present and populated (no blank panels) | **PASS** — ink 4.4–58.6 % |
| 9 | `scripts/audit_cells.py` screen | **PASS but stale in the text** — see §4 |
| 10 | The three repeats confirm the epoch-2 artefact is a one-off | **FAILED** — see §3, the important one |
| 11 | Draft's baseline numbers == recorded baseline numbers | **FAILED for RF only** — see §5 |

---

## 2. The flagged cell: what the repeat experiment actually returned

`results/adult_income_naive_resnet.json` plus the eight files in
`results/backup_pre_rerun/` are **nine records of the same cell** (seed 42,
`split_seed` 42). They fall into exactly two groups:

| Group | files | `f1_macro` | accuracy | `best_epoch` | `epochs_run` | `val_loss[0]` | distinct `train_time_sec` |
|---|---|---|---|---|---|---|---|
| A | 6 | 0.5757938089544307 | 0.6470 | 2 | 17 | 0.6486 | 457.1 / 460.3 / 464.4 / 465.0 / 465.4 / 466.5 |
| B | 3 (+ sweep seed 42) | 0.6908194233687405 | 0.8198 | 33 | 48 | 0.6917 | 1283.3 / 1286.4 / 1297.5 / 1271.8 |

Both facts matter:

* **Inside a group the runs are numerically identical** — same `final_train_loss`
  (0.507118571627, 15 digits), same `roc_auc`, same confusion matrix, while
  `train_time_sec`, `total_time_sec` and `t2i_time_sec` all differ. So these are
  genuinely separate trainings, and the pipeline *is* reproducible within a run
  environment. (This also corrects audit C8's "not bit-reproducible across runs":
  it is not reproducible *across environments*, not across runs.)
* **The two groups differ from the first validation epoch** (0.6486 vs 0.6917), so
  the split is not merely "a different checkpoint was selected" — the whole training
  trajectory differs. Same seed, same split sizes (31654/9045), same lr.

**Consequence.** The planned sentence "the cell was re-run three times and the
artefact did not recur" **cannot be written**. Six of the nine runs reproduce
57.58 %, so the artefact *does* recur, deterministically, in one environment.

The honest statement, which is also stronger evidence for the draft's existing
conclusion:

> The cell is numerically unstable at lr 1e-3: its validation loss has a spurious
> minimum at epoch 2, and whether that minimum is the last one depends on the
> environment the run happens in. Re-running the identical configuration nine times
> produced 57.58 % (six times) and 69.08 % (three times), each group bit-identical
> within itself. No single-run value for this cell may therefore be quoted; the
> five-seed sweep is the reference.

The draft's **conclusions survive** (57.58 % is not reported as a finding; Table 6.3
uses the sweep; §6.7 calls it a checkpoint-selection artefact). What is now wrong is:

1. `Plan/critical-audit-findings.md:319` — "targeted re-run outstanding" → the re-run
   was done and returned a two-group result.
2. `Plan/professor-validation.md:699` — "One re-run is outstanding: the flagged cell,
   three times at seed 42…" → same.
3. `Plan/critical-audit-findings.md:784` — "The repeats do not scatter around 57.58 %"
   is true of the *sweep*, but the sentence reads as if the cell never produces
   57.58 %; needs the two-group figure added.
4. `Plan/seminar2-rad-nacrt.md` §6.7 ("Artefakt izbora modela") — should state the
   direct-repeat evidence (6× vs 3×) and say explicitly that the cell is
   environment-sensitive, not that it was a single unlucky run.
5. `Plan/seminar2-rad-nacrt.md:1230` header note "jedna verzija koda na jednoj mašini"
   — undermined by §2 above; soften to "jedna verzija koda; ćelije koje nose tvrdnje
   ponovljene su i kroz semena".

---

## 3. Free post-processing gaps (fixed here, and how to redo it on Colab)

The finalisation had not applied two of the four free steps:

* **C6 backfill (V2).** 35 of 36 grid JSONs had no `f1_macro_all` / `balanced_accuracy`,
  so every `f1_macro_all` / `balanced_accuracy` column in the CSV would have been
  empty for 35 of 36 CNN rows.
  **Fixed here:** `python scripts/backfill_metrics.py --write` → *"WROTE … into 45 files"*.
* **Aggregate (V3).** `all_experiments.csv` held only 10 rows (9 baselines + the one
  re-run cell).
  **Fixed here:** `python run_all.py --aggregate` → 45 rows, 36 CNN, no row missing
  `f1_macro_all`. The flagged cell reads acc 0.8198 / F1 0.6908 / F1-all 0.7818.

Colab equivalent (safe, no training, seconds; **order matters — backfill before aggregate**):

```bash
%%bash
cd /content/T2I-CNN
git pull
python -u scripts/backfill_metrics.py --write
python -u run_all.py --aggregate
python -u scripts/stability_table.py
```

Do **not** re-run figure generation on Colab — that is what produced the six blank
figures (three Grad-CAM grids, `ch4_roc_curves`, `ch4_confusion_matrices`,
`ch4_baseline_comparison`). The PNGs now on disk are the populated set.

---

## 4. Screen status changed meaning

`scripts/audit_cells.py` now reports **HIGH 0** (`INFO-only 11, clean 44`) because the
flagged cell has been replaced by a healthy run; the anomaly lives in
`backup_pre_rerun/`, which the screen does not scan by design.

`Plan/seminar2-rad-nacrt.md` §6.7 still says the screen "pronalazi **tačno ovu ćeliju
i nijednu drugu**". After the re-run that is no longer what it does. Reword to: the
screen flags nothing in the final set, and the evidence for the artefact is the
preserved 57.58 % run plus the two-group repeat.

---

## 5. RF baseline numbers in the draft are not in the results

Re-running the baselines locally with the current code (`class_weight='balanced'`,
`random_state=42`, train on `X_train` only — `src/baselines/rf.py`) reproduces the
**draft's** numbers exactly and **not** the stored JSONs:

| dataset | stored JSON (`results/baseline_*_rf.json`) | current code, fresh run | draft §6.1 |
|---|---|---|---|
| breast_cancer | acc 93.86 / F1 95.10 | acc 95.61 / F1 **96.55** | 96,55 (95,61) |
| dry_bean | acc 91.81 / F1 93.10 | acc 92.32 / F1 **93.48** | 93,48 (92,32) |
| adult_income | acc 84.27 / F1 69.99 | acc 85.52 / F1 **67.84** | 67,84 (85,52) |

XGBoost and MLP, by contrast, match the stored JSONs to 0.00–0.01 pp in the draft
(so the draft is faithful to the results everywhere except RF). No configuration I
tried reproduces the stored RF values — not `class_weight=None`, not training on
train+val, not split seeds 0/1/42/123. The adult confusion matrices differ
systematically (stored: 583 FN / 840 FP; fresh: 860 FN / 450 FP), so this is not the
0.04–0.12 pp environment noise seen on XGB/MLP.

**Why it matters.** On breast the recorded results make **XGBoost (95.83) the best
baseline, not RF (95.10)**. That flips the sentence in four places:

* `Plan/seminar2-rad-nacrt.md:1261` (per-dataset baseline line),
* `:1265` ("+0,67 pp iznad RF baselajna (96,55)"),
* `:1325` (§6.2 Breast Cancer bullet: "+0,67 pp prema najboljem baselajnu 96,55 % (RF)"),
* `:1668` (§7, "paritetna … sa najboljim baselajnom (RF 96,55 %)"),
* plus the two abstract mentions (`:48`, `:84`) of "97,22 % naspram RF 96,55 %",
* and `:1309`/`:1670` ("RF ima najvišu tačnost (85,52 %)" → 84,27 %, still the
  highest of the three, so only the number changes).

With the recorded values the breast margins become **+1,39 pp** (97.22 − 95.83, XGBoost),
and dry_bean/adult conclusions are unchanged (XGBoost is the best baseline there in
both readings).

**Two ways to close it — pick one, they are not compatible:**

* **(A) Re-run the three RF baselines on Colab** (seconds): delete the RF JSONs, run
  `python run_all.py --baselines`, then `--aggregate`. If Colab reproduces 95.10 /
  93.10 / 69.99, the stored artefacts were right and the draft needs the text edits
  above. If it reproduces the fresh numbers, the stored RF JSONs were stale and the
  draft is already correct.
* **(B) Adopt the recorded values now** and make the text edits above. Conservative —
  it reports what the run produced — but it reports a baseline whose configuration is
  not reproducible with the current code (which is exactly the "was it a fair
  baseline?" question a professor may ask).

Recommendation: **(A)** — it is a one-minute, no-training check that turns an
ambiguous artefact into a definitive one, and it also puts all nine baselines on a
single code path.

---

## 6. Minor / housekeeping

* `src/visualize.py` has an **uncommitted** `safe_savefig` helper (atomic write +
  retry, for the OneDrive/AV file-lock `Errno 22` seen during figure regeneration).
  It is functional and was used for the current PNGs; it should either be committed
  or reverted deliberately — not left dangling.
* `Plan/professor-review.md` is untracked (the PDF review). Keep it tracked or delete
  it; an untracked review file invites a stale copy.
* `ch4_baseline_comparison.png` will move with the RF decision above — regenerate the
  figures **after** the baseline question is settled.
* Nothing else in the tree is modified; no training was run by this validation.

---

## 7. What to re-run, in order

1. `python run_all.py --baselines` (delete the 9 baseline JSONs first) → settles §5.
   Cost: **~1 min on Colab**, no GPU.
2. `python scripts/backfill_metrics.py --write` → `python run_all.py --aggregate` →
   `python scripts/stability_table.py` (§3). Cost: seconds.
3. If step 1 changes the RF numbers: apply the text edits in §5, and regenerate the
   figures (already-populated set; do this locally, not on Colab).
4. Text-only edits from §2 and §4 — no re-run needed.

No grid cell, ablation, seed-sweep or C11 run needs repeating.

---

## 8. Postscript — the runs are done, this validation is closed (14 Sep, Colab)

Everything in §7 has now been executed in Colab (Python 3.13.15, `scikit-learn`
1.9.1, one runtime) and the whole results tree was re-downloaded and compared
against the local copy.

1. **§5 (RF baseline) — resolved in favour of the recorded files.** The three RF
   baselines were re-run after moving the old files to `results/backup_rf_before/`,
   and they reproduced the stored values exactly: **95.10 / 93.10 / 69.99 %**, with
   the same accuracy and confusion matrices. The 96.55 / 93.48 / 67.84 % observed
   earlier came from running the same code under `scikit-learn` 1.7.2 (Python
   3.10.11), not from a stale record. So the stored JSONs were correct, the draft's
   RF numbers were stale, and **the §5 text edits were the right fix** — they are
   now applied to `Plan/seminar2-rad-nacrt.md` (four places plus both abstracts).
   On Breast Cancer the best baseline is XGBoost (95.83 %), so the CNN margin is
   **+1.39 pp**, not +0.67 pp.
   A new limitation is recorded in the draft (§6.12 item 8) and in
   `Plan/rezultati-i-diskusija.md`: the RF baseline is library-version sensitive, so
   the version belongs in the protocol description.
2. **§3 (C6 metrics) — applied.** `backfill_metrics.py --write` wrote `f1_macro_all`
   and `balanced_accuracy` into 45 files; `run_all.py --aggregate` produced a
   **45-row CSV (36 CNN + 9 baselines) with no missing metrics**; `stability_table.md`
   and `.csv` were regenerated and match the values already quoted in the draft.
3. **Figures — untouched and still correct.** All 30 PNGs in the download are
   byte-identical to the locally regenerated set (the run never called
   `visualize.py`), so the 2026-09-14 blank-panel fix survives intact.
4. **Whole-tree diff:** 200 of 204 files identical after normalising line endings.
   The only differences are the three `baseline_*_rf.json` (`train_time_sec` only)
   and `all_experiments.csv` (same reason). **No metric changed anywhere.** The only
   text consequence is the baselines' total training time, 21.1 s → **20.6 s**.
5. **Local tree replaced with the Colab copy**, including the new
   `results/backup_rf_before/`; nothing that existed only locally was lost (the
   comparison found zero local-only files).

Closed with this: the two stale status lines flagged in §2 —
`critical-audit-findings.md:319` and `professor-validation.md:699`.

Still not done, and not required for the write-up: the C11-rigorous multi-split
seed sweep over every cell.
