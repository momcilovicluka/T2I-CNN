# Final audit — state of code, results, and write-up (2026-09-04)

Scope: verify all visuals exist, confirm code quality/commenting, confirm every
result/metric/figure has an explanation in the markdowns, and list what remains
before the seminar text is finished. Outcome: no correctness problems in the
results or pipeline; one real figure-code bug found and fixed; four visuals were
missing and are now generated.

## 1. Verified complete

- **Results (54 JSONs + CSV).** Fresh invariant pass: 36 CNN + 9 baseline + 9
  ablation files; all metrics in [0,1], no NaN; every cell lr=1e-3 (uniform
  protocol); confusion matrices square and sum to test sizes; one shared
  train/test split per dataset across CNN and baseline cells; CSV agrees with
  every JSON; values sit inside the predicted bands (breast 95.0-97.2,
  dry bean 90.3-94.0, adult 57.6-69.0 CNN vs XGB 71.43); headline adult
  naive+pretrained 57.58 / -11.40 pp negative transfer confirmed
  (professor-validation.md sections 9-12).
- **Draft chapter 6 tables** match all_experiments.csv cell-for-cell (39/39
  checked, 0 mismatches).
- **Code quality.** No TODO/FIXME/XXX markers anywhere under src/; all modules
  carry docstrings; the design-decision comments (WHY/FIX/PART markers in
  naive.py, tinto.py, train.py, resnet_wrapper.py, run_all.py) are in place;
  listings 5.1-5.11 were verified line-by-line against the repository
  (checkbox in Prilog B).
- **Metric/result explanations.** Section 3.5 defines accuracy, precision,
  recall, F1 (+macro/positive-class convention), ROC-AUC and PR-AUC;
  chapter 6 reports and explains every result with per-figure prose and
  baselines; ablation section 6.6 explains pixel-shuffle, ordering
  (permutation invariance) and LP-FT findings; chapter 7 discussion and
  chapter 8 conclusion summarize interpretation.
- **Visuals.** 30/30 PNGs in results/figures are valid and non-trivial
  (checked with PIL). Every figure referenced in the draft now exists for
  every dataset (wildcard patterns expanded).

## 2. Resolved this turn

- **Bug (visualize_t2i.py): density grid out of bounds.** plot_pixel_density_
  comparison() built a (3,3) subplot grid but iterated 4 T2I methods as rows,
  so the density figure could never be produced (IndexError on the 4th row).
  Fixed minimally: grid rows = len(methods), columns = len(datasets). Compiles;
  script runs clean end-to-end.
- **Missing visuals generated.** python src/visualize_t2i.py now emits
  t2i_comparison_{breast_cancer,dry_bean,adult_income}.png and
  t2i_density_comparison.png (fits the four T2I transformers on each train
  set; ~10 min CPU). All four valid.
- Commit of the visualize_t2i.py fix is still pending (see 3.1).

## 3. Open items before/while writing the seminar

### 3.1 Code/cleanup (small, do now)
- [ ] Commit the visualize_t2i.py density-grid fix (repo style, unsigned).
- [ ] Remove/archive duplicate result zips in the repo root
      (results.zip, "results (1-3).zip" — untracked copies; keep the exported
      one the draft's results/ came from) and the duplicate EDA notebooks
      (01_eda.ipynb vs 01_eda_executed.ipynb) if one supersedes the other.

### 3.2 Draft/write-up gaps (the actual remaining writing work)
- [x] **Sažetak / apstrakt + ključne reči** — added (2026-09-04): Serbian
      summary with keywords and an English abstract, placed between the
      title page and chapter 1.
- [ ] **Example T2I figures have no body prose.** t2i_comparison_{dataset}.png
      are listed only in Prilog A; the write-up should give them figure
      numbers and captions and point to them where methods are introduced
      (section 3.2, likely as Slika 3.5-3.7 or placed in 6.4 — decide
      numbering once) plus one sentence per figure (how naive's row-major
      layout vs DeepInsight/TINTO projections vs IGTD strips read).
- [ ] **t2i_density_comparison.png role vs ch4_density_vs_performance.png.**
      Both density-related; decide which is illustrative (per-sample grid,
      t2i_density_comparison) vs quantitative (density-vs-performance
      scatter) and say so where each is cited (6.4 / Prilog A).
- [ ] **Grad-CAM region interpretation.** Figure layout is described in 6.4;
      the per-dataset verbal reading of activation regions is still deferred
      to final redaction after visual inspection of the PNGs (honest framing
      per guide PART 15d).
- [ ] **pipeline_diagram.png is not referenced in the md** (draft embeds
      Mermaid + ASCII, Slika 5.1/5.2). It exists for Word/LaTeX insertion;
      regenerate with src/visualize_pipeline.py at final code state and
      insert as the pipeline figure in the document.
- [ ] **References.** Move Notebook/references.bib into section 9.1 with
      IEEE numbering per first citation (the only unticked Prilog B item).

### 3.3 Optional (only if a professor probes deeper)
- [ ] Naive-method feature-ordering ablation (run the ordering ablation with
      t2i=naive): because DeepInsight's layout is permutation-invariant, the
      ordering ablation is meaningful only for the correlation-sorted key; a
      naive run would demonstrate the permutation effect visually. Not
      required by the current claims; costs ~1-2 CPU hours.

## 4. Bottom line
Everything a professor can check in the code and numbers is verified and
explained in the markdowns. The remaining work is writing, not fixing:
abstract, example-figure prose, Grad-CAM region notes, reference migration,
pipeline-PNG insertion in the final document — plus two small cleanups and the
pending commit of the density-grid fix.
