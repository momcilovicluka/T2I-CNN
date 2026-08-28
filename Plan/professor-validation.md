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
Correctly handles class imbalance. Dry Bean (6.6:1) and Adult Income
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
TINTOlib uses `random_seed=42`, but t-SNE and iterative optimization
may have non-deterministic components across different hardware.

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
Shallow CNN: ~200K parameters
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
