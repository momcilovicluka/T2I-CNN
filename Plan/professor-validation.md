# Professor's Validation: Tabular-to-Image Pipeline

## Perspective: ML University Professor Reviewing Seminar Work

---

## 1. Methodology Validity

### ✅ PASS: No Data Leakage
- StandardScaler: fit on train only, transform on val/test ✓
- TINTOlib MinMaxScaler: fit on train only, transform on test ✓
- T2I coordinate mapping: fit on train only ✓
- Class weights: computed from train distribution only ✓

**Professor's note:** This is the most common mistake in student projects.
You avoided it. Good.

### ✅ PASS: Stratified Splits
- train_test_split with stratify=y ensures class proportions are preserved
- Two-stage split (train+val→test, then train→val) is correct
- Same random_state (42) ensures reproducibility

### ⚠️ CONCERN: Adult Income Re-splitting
The UCI Adult dataset has an official train/test split. You combine them
and re-split with your own stratified split.

**Professor would ask:** "Why not use the official split?"
**Answer:** Official split is not stratified. Your re-split is more
statistically sound. This is acceptable but should be noted in the paper.

---

## 2. Experimental Design

### ✅ PASS: Fair Comparison Setup
- All methods use same train/val/test splits
- All methods use same preprocessing (StandardScaler)
- All methods produce same output format (N, 1, 32, 32)
- All methods normalized to [0, 1]
- Same CNN architectures applied to all method outputs

**Professor's note:** This is a well-controlled experiment. The only
variable changing is the T2I method. Everything else is held constant.

### ⚠️ CONCERN: Naive Method Is Fundamentally Different
Naive creates 97%+ zero pixels. DeepInsight/IGTD create ~10-30% non-zero
pixels (feature positions only).

**Professor would ask:** "Are you comparing feature arrangement or
feature density? These are confounded."

**Honest answer:** Both. Naive is a legitimate baseline showing what
happens without intelligent arrangement. The paper should acknowledge
this confound explicitly.

### ⚠️ CONCERN: No Hyperparameter Tuning
The same hyperparameters (lr=1e-3, epochs=50, patience=10) are used
for all experiments.

**Professor would ask:** "Did you tune hyperparameters per method?
Different methods may need different learning rates."

**Honest answer:** For a seminar, fixed hyperparameters are acceptable.
The goal is comparison, not optimization. But the paper should state
that hyperparameters were not tuned per method.

---

## 3. Metric Validity

### ✅ PASS: Macro-F1 as Primary Metric
Correctly handles class imbalance. Dry Bean (6.8:1) and Adult Income
(3.2:1) would inflate accuracy if accuracy were the only metric.

### ✅ PASS: Multiple Metrics Reported
Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix.
This is comprehensive and allows readers to assess different aspects.

### ⚠️ CONCERN: Label Smoothing Affects Loss Values
Label smoothing (0.1) changes the loss landscape. The reported training
loss will be lower than standard cross-entropy, making it harder to
compare with published results that don't use smoothing.

**Professor would ask:** "Is your loss comparable to baselines?"

**Answer:** Not directly. But all methods in this experiment use the
same smoothing, so relative comparison is fair.

### ⚠️ CONCERN: Class Weights Affect Loss Magnitude
Class weights change the loss scale. A model with class weights will
have different absolute loss values than one without.

**Professor would ask:** "Can you compare your results with papers
that don't use class weights?"

**Answer:** Not directly. But within this experiment, all methods
use the same weights, so comparison is fair.

---

## 4. Statistical Validity

### 🔴 ISSUE: Single Train/Test Split
Each experiment is run once on one fixed split. No cross-validation,
no confidence intervals, no statistical tests.

**Professor would ask:** "How do you know the differences are
statistically significant? Could they be due to random split?"

**Honest answer:** This is a limitation. For a seminar, a single
split is often acceptable. But the paper should acknowledge this
and ideally report mean±std over 3-5 runs with different seeds.

### 🔴 ISSUE: No Variance Reporting
Results are single numbers. No standard deviation, no error bars.

**Professor would ask:** "What's the variance across runs?
Is the difference between DeepInsight and IGTD real or noise?"

**Recommendation:** Run each experiment 3-5 times with different
random seeds. Report mean ± std. This is standard practice.

---

## 5. Reproducibility

### ✅ PASS: Random Seeds Set
- `set_global_seed(42)` sets random, numpy, torch, CUDA seeds
- `torch.backends.cudnn.deterministic = True` for GPU reproducibility
- `torch.backends.cudnn.benchmark = False` disables auto-tuner

### ⚠️ CONCERN: TINTOlib Internal Randomness
TINTOlib uses `random_seed=42` (and our wrappers pass it explicitly).
DeepInsight/TINTO projections use PCA (deterministic), but IGTD's
iterative swap optimization may still have non-deterministic
components across different hardware (guide PART 15c).

**Professor would ask:** "Are results identical on different machines?"

**Answer:** Not guaranteed for TINTOlib methods. Naive is fully
deterministic. The paper should note that TINTOlib results may vary
slightly across platforms.

### ✅ PASS: Data Saved
- Raw data files in `data/` directory
- Preprocessing is deterministic (same random_state)
- Results saved to JSON files

---

## 6. Model Architecture Concerns

### ⚠️ CONCERN: Shallow CNN vs ResNet-18 Capacity Mismatch
Shallow CNN: ~620K parameters (measured 618,178)
ResNet-18: ~11M parameters

**Professor would ask:** "Are you comparing T2I methods or model
capacity? ResNet has 50x more parameters."

**Honest answer:** Both. The experiment shows how each T2I method
performs with different model capacities. But the paper should
acknowledge that capacity is a confound.

### ⚠️ CONCERN: Pretrained ResNet on Synthetic Images
ResNet-18 was trained on natural images (ImageNet). Our images are
synthetic (sparse, grayscale-based). Feature transfer may be poor.

**Professor would ask:** "Is transfer learning actually beneficial
here, or are you just adding complexity?"

**Honest answer:** Unknown until experiments run. The experiment
is designed to answer this question. If ResNet performs worse than
shallow CNN, that's a valid finding.

### ⚠️ CONCERN: ViT on 32×32 Images
ViT with patch_size=16 on 32×32 images = 4 patches. Very few
tokens for attention. May underperform.

**Professor would ask:** "Is ViT appropriate for such small images?"

**Answer:** This is an open question. The experiment will show.
If ViT performs poorly, it's because 32×32 is too small for
patch-based attention — a valid finding.

---

## 7. What a Professor Would Criticize

### 1. "Why only 3 datasets?"
**Response:** Deliberate choice for diversity: binary (breast cancer),
multiclass (dry bean), mixed features (adult income). More datasets
would strengthen the paper but aren't necessary for a seminar.

### 2. "Why not use cross-validation?"
**Response:** Single split is a limitation acknowledged in the paper.
Cross-validation would be stronger but is beyond seminar scope.

### 3. "Your results might not generalize."
**Response:** Correct. The paper should state this as a limitation.
Future work could test on more datasets and domains.

### 4. "Naive is not a fair comparison."
**Response:** Naive is a baseline, not a competitor. It shows the
minimum performance without intelligent arrangement. The comparison
is: naive < deepinsight/igtd = better arrangement helps.

### 5. "Class weights change the loss — are you cheating?"
**Response:** No. All methods use the same weights. The comparison
is fair. Without weights, all methods would default to majority
class, making the comparison meaningless.

### 6. "Why bicubic and not nearest?"
**Response:** Literature shows bicubic produces better gradients for
CNNs. Nearest creates blocky artifacts. This is a design choice,
not a bug. Documented in the code.

### 7. "Your images are mostly zeros — CNN can't learn from that."
**Response:** Correct for naive. DeepInsight/IGTD also create sparse
images but place features intelligently. The CNN learns from the
non-zero pixels. This is a known limitation of the approach.

 smoothing reduces overfitting but also reduces accuracy."
**Response:** Label smoothing trades training accuracy for test
generalization. All methods use the same smoothing, so comparison
is fair.

---

## 8. Final Verdict

### What Would Pass
- Clean preprocessing pipeline
- No data leakage
- Fair experimental comparison
- Comprehensive metrics
- Reproducible with seeds

### What Would Need Improvement for a Journal Paper
- Cross-validation (3-5 folds)
- Confidence intervals (mean +/- std over multiple runs)
- Statistical significance tests (paired t-test or Wilcoxon)
- More datasets (5-10)
- Hyperparameter tuning per method

### What Is Acceptable for a Seminar
- Single split
- Fixed hyperparameters
- 3 datasets
- 3 T2I methods x 3 CNNs = 9 experiments
- Macro-F1 as primary metric

### Bottom Line
The pipeline is methodologically sound for a seminar paper. The
main limitations (single split, no cross-validation, confounded
naive baseline) should be acknowledged in the paper but do not
invalidate the work. The implementation is clean and the comparison
is fair.


## 9. Red-team review (2026-09-03) — pre-results audit of the final protocol

Auditor stance: ML professor reading the FINAL code and the claims the draft
will make, before the 36-cell run is used for results. Supersedes section 8's
verdict where they conflict (section 8 predates the ViT removal, the fixed
TINTO/naive pipelines and the resnet_scratch input design).

### 9.1 FINDING 1 — CRITICAL: the resnet vs resnet_scratch comparison is confounded
Evidence (code):
- run_all.py create_cnn_model(): resnet -> ResNetWrapper(pretrained=True,
  input_channels=3); resnet_scratch -> ResNetWrapper(pretrained=False,
  input_channels=1).
- train.py train_model()/evaluate.py evaluate_model(): ImageNet normalization
  applied iff model.pretrained is True. So resnet gets 3-channel
  ImageNet-normalized RGB input; resnet_scratch gets raw 1-channel [0,1]
  grayscale. The wrapper replaces conv1 for input_channels=1 (random init).
Why a professor would object: RQ2 ("does ImageNet transfer help?") requires
the pretrained and from-scratch model to differ ONLY in weight initialization.
Here they also differ in input channels AND normalization statistics, so any
delta mixes "pretrained vs random init" with "RGB+norm vs gray-raw". The draft
currently claims otherwise (§3.3: "Ova dva režima razdvajaju efekat kapaciteta
od efekta pretreniranosti") and Slika 6.1 caption says the archs "razlikuju se
samo po pretreniranosti" — both statements are false under the current code.
Integrity impact: the headline RQ2 delta figure would not support the clean
"transfer learning effect" claim it is captioned with. Shallow/resnet rows in
the heatmap and all other comparisons remain valid; only pretrained-vs-scratch
inference is affected.
Fix plan (recommended): train resnet_scratch on the SAME input pipeline —
input_channels=3 + ImageNet normalization — so the two ResNet-18s differ only
in weight init. Concretely:
  1. run_all.py create_cnn_model(): resnet_scratch -> input_channels=3
     (pretrained=False keeps random init).
  2. train.py / evaluate.py: trigger ImageNet normalization from a channel
     flag (e.g., model.imagenet_input = input_channels==3) instead of
     model.pretrained, so normalization applies to resnet_scratch too but NOT
     to shallow (1-channel local model; no matched-arch claim).
  3. resnet_wrapper.py: set the flag; keep conv1 replacement only for the 1ch
     case (shallow-style), never for the 3ch scratch.
  4. Rerun scope: delete only the 12 resnet_scratch result JSONs + model.pt
     files; resume logic re-runs exactly those cells (deterministic seed 42 —
     untouched shallow/resnet/baseline cells reproduce identically and are
     correctly skipped as done).
  5. Docs: create_cnn_model docstring, guide PART 4.2/9a wording, workflow
     bug-table row 2, draft §3.3/§3.4/§5.7/listing 5.6.
  Alternative (weaker, not recommended): keep 1-channel scratch and rephrase
  every claim to "pretrained RGB+ImageNet ResNet vs ResNet from scratch on
  single-channel images" — a professor would still press for the controlled
  version, and the delta figure loses its core meaning.

  **RESOLUTION (2026-09-03):** decision — KEEP the 1-channel raw-gray
  from-scratch ResNet. No code change, no rerun; claims were rephrased
  instead: draft §3.3, §5.7, §6.3/Slika 6.1 now state the delta is the
  COMBINED effect of pretraining and input representation, and
  `run_all.py::create_cnn_model` carries the same caveat in its docstring.
  Watchlist item 1 is updated accordingly.

### 9.2 FINDING 2 — MEDIUM (already documented): one split, no variance
Single 70/10/20 split, one seed; all methods share it (fair), but no
mean±std, so small deltas cannot be distinguished from noise. Draft has this
as a limitation (PART 1.1, §7). Acceptable for a seminar IF claims are phrased
per-run, not as general superiority. Optional hardening (post-run, cheap):
3-5 repeats of the best-vs-runner-up cell only with different seeds, report
mean±std as supporting evidence.

### 9.3 FINDING 3 — MEDIUM (documented): untuned baselines
RF/XGBoost/MLP use sklearn/xgboost defaults; CNN hyperparameters are also
fixed across methods (never tuned per dataset). The comparison is therefore
"reference CNN protocol vs reference tabular methods" — fair as a reference
benchmark, but the paper must NOT claim CNNs outperform *tuned* tabular
models. Professor's likely probe: "would XGBoost with 30 random-search iters
beat your best T2I-CNN?" Optional post-run: light tuning of the 3 baselines
per dataset (~minutes each) and re-report as the honest bar.

### 9.4 FINDING 4 — LOW/MEDIUM: F1 label semantics asymmetry
f1_macro key stores scikit *binary* F1 (positive class) for breast_cancer and
adult_income, macro-F1 for dry_bean. Positive class is the MAJORITY benign
(357/212) for breast and the MINORITY >50K for adult. Documented honestly
(PART 13e, F1_LABEL in visualize.py), but a professor will note the binary
"positive-class F1" is asymmetric across datasets and favors breast numbers.
Fix (no rerun needed): report, alongside the existing metric, binary macro-F1
and balanced accuracy derived from the SAVED classification_report /
confusion_matrix (evaluate_model stores both per run); state explicitly in
tables that breast F1 = F1(benign).

### 9.5 FINDING 5 — LOW: one-hot encoder fitted before the split
get_dummies is applied to the FULL dataset in the loaders; preprocess() splits
afterwards. Category vocabulary is therefore informed by test rows. Impact is
near-zero here (all adult categories occur in train after the '?' fix; breast
and dry_bean have no categoricals) and it is a common practice, but a strict
professor may flag it. If the suite is ever re-run anyway, encode on
train+val and apply the same columns to test; otherwise record it as a
limitation note.

### 9.6 LOW/process notes (no action expected)
- ResNet-18 weights download on first run (torchvision/HF); document that a
  network fetch is part of setup for reproducibility.
- All CNN hyperparameters are protocol-fixed (never per-method tuned) — this
  is a strength for comparability and a stated limitation for optimality.
- ViT: excluded from the main grid (documented 2026-09-03); re-addable via
  --archs flag. Any future ViT table must not be compared to the 36-cell grid
  numbers directly.
- Code key name f1_macro for a binary positive-class value is confusing on
  inspection; the JSON consumers (visualize.py F1_LABEL, guide PART 13e) label
  it honestly, so no code change required.

### 9.7 Expected-results watchlist (what to check the moment the run finishes)
1. Any resnet-vs-resnet_scratch margin (per the 2026-09-03 rephrase decision)
   is the COMBINED effect of pretraining and input representation (3ch+norm
   vs 1ch gray). Report the delta as such; never attribute its magnitude to
   pretraining alone, and do not claim a clean transfer-learning effect.

2. Adult income: watch for any CNN pinned near the ~75/25 prior (acc ≈ 0.75
   with F1 near 0) — a symptom, not a result; class weights + report cards
   should prevent it, but verify each cell's final val loss is below ~0.69
   (log(2) for binary) for the binary sets.
3. Dry bean 7-class: check per-class F1 spread for classes collapsed by the
   CNN; confusion matrix should confirm whether minority classes are merely
   dropped or genuinely separable.
4. Training time sanity: adult resnet/resnet_scratch cells are the long pole
   (est. ~15-20 min each CPU); anything wildly outside 5-60x the shallow cell
   on the same dataset warrants a look at the log.
5. If the fixed transfer delta is NEGATIVE for most cells, that is a real,
   interesting finding ("ImageNet features do not transfer to synthetic T2I")
   — report it as such rather than as a failure; do not cherry-pick cells.

### 9.8 Verdict (supersedes section 8)
Methodology is sound and the fixes documented in PART 15 made the pipelines
fair. One claim exceeded what the code supports — the clean
pretraining-effect decomposition (9.1). Resolved 2026-09-03 by rephrasing
every affected claim to a combined effect (no code change, no rerun).
Everything else is an acceptable, documented limitation for a seminar and
must simply be stated as such in the text.


## 10. Second-pass professor audit (2026-09-03) — comparison fairness & runtime honesty

Focused on what the FIRST pass did not cover: cross-family comparison
fairness (CNN vs baselines) and the meaning of the runtime figure.

### 10.1 FINDING — imbalance handling is asymmetric between CNNs and baselines (MEDIUM)
Code evidence: src/baselines/rf.py RandomForestClassifier (no
class_weight), src/baselines/xgboost_model.py XGBClassifier (no
scale_pos_weight / sample weights), src/baselines/mlp.py MLPClassifier
(no sample_weight). The CNN training loss uses inverse-frequency class
weights (compute_class_weights, train.py). On adult_income (~3:1) and
dry_bean (~6.8:1) a default-weighted RF/XGB sacrifices minority recall
exactly where class-weighted CNNs are protected, tilting the
CNN-vs-baseline comparison (ch4_baseline_comparison) in the CNN favor.
These are not "untuned defaults" in a neutral sense: they are defaults
WITHOUT the imbalance remedy applied to the other family. Fix (9-cell
rerun only, seconds-to-minutes per cell): RF class_weight='balanced';
XGB per-sample weights from sklearn compute_class_weight('balanced');
MLP sample_weight likewise; keep seeds. Update the "baselines untuned"
wording to "default hyperparameters with balanced class weighting" in
guide PART 13b and the draft.

### 10.2 FINDING — runtime figure ignores T2I generation time (MEDIUM/LOW)
[RESOLVED 2026-09-03: run_all.py records t2i_time_sec + total_time_sec per
cell (baselines: total==fit+eval); visualize.py uses total with fallback.]
train_time_sec wraps ONLY train_model() (run_all.py step 8). The T2I
fit/transform (steps 2-3), which for TINTO writes one file per sample,
is excluded — yet the figure's purpose (ch4_runtime_comparison,
"Training Time by T2I Method and Architecture") is to compare the cost
of the T2I approaches. As recorded, a slow T2I transform would be
invisible and the naive-vs-TINTO cost gap understated. Fix (additive,
no invalidation): record t2i_time_sec and total_time_sec per CNN cell
(steps 2-3 and the whole run_single_experiment) alongside
train_time_sec; runtime figure uses total_time_sec with a fallback to
train_time_sec only if the field is absent (old cells). Do this before
the final suite runs so every cell carries the field.

### 10.3 FINDING — confusion matrices lack class labels (LOW, presentational)
[RESOLVED 2026-09-03: per-dataset class-name ticks from loader class_names.]
ch4_confusion_matrices plots raw integer cells with no row/column tick
labels; for dry_bean's 7 classes this is unreadable without the class
legend. Fix in visualize.py: per-dataset class-name ticks (binary
labels from the loaders; dry_bean 7 names). No rerun needed.

### 10.4 VERIFIED CLEAN (no action)
- CNN and baseline F1/precision/recall share ONE implementation
  (_compute_metrics, average='macro' only for >2 classes else
  'binary'), so cross-family bars compare the same quantity per
  dataset; label semantics are annotated per dataset (PART 13e).
- Baselines train on X_train ONLY and evaluate the SAME X_test rows as
  CNNs; the scaler is fit on train and identical for both families;
  val never enters baseline training (fix 9b, 6692c79).
- No per-method hyperparameter tuning anywhere (CNNs and baselines
  alike): fixed protocol is a stated comparability strength.
- Atomic result writes and corrupt-file-aware resume verified.
- visualize.BASELINES keys match run_all baseline cnn_arch values; the
  baseline bar chart is correctly wired.

### 10.5 Minor notes (mostly documented; one fixed)
[FIXED 2026-09-03: dry-bean per-class F1 parser now reads numeric report
rows by class id — the old name-based parser produced all-zero bars.]
- adult_income: a category present only in test produces a
  constant-zero training column; sklearn StandardScaler handles
  zero-variance columns (scale->1) so this is benign; mention once.
- MLP's internal early_stopping holds out its own 10% validation
  inside X_train (validation_fraction=0.1): it sees ~90% of the CNN's
  training rows. On adult this is ~3.2k rows; state it in the baseline
  description or set validation_fraction to use a seeded split.
- Final numbers must come from ONE machine/run (CPU-only; the earlier
  GPU results were lost and CPU/GPU cells are not interchangeable in a
  single table). State the environment (torch version, CPU) in the
  reproducibility section; note that ResNet-18 ImageNet weights are
  downloaded on first run (network needed to reproduce).
- Dry-bean per-class F1 figure (ch4_per_class_f1_dry_bean) parses the
  saved classification_report by class NAME; if dry_bean labels reach
  sklearn as integers, the parser silently yields 0.0 bars. Verify
  class-name presence in the report on the FIRST finished dry_bean
  result before trusting that figure.

### 10.6 Plan
A. Baseline class weighting (10.1): edit rf.py / xgboost_model.py /
   mlp.py; re-run only `python run_all.py --baselines` (9 fast cells);
   update guide PART 13b + draft wording.
B. Per-cell time fields (10.2): edit run_all.py (steps 2-3 timing,
   whole-cell total); no rerun needed if added before the suite; if the
   suite already started, cells finished without total_time_sec can
   stay (runtime figure falls back to train_time_sec) or be deleted and
   resumed.
C. Confusion-matrix labels (10.3): visualize.py only.
D. Verify the dry-bean per-class parser against the first real result
   (10.5) before using figure 4.5 in the paper.
