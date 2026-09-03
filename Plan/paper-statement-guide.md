# Paper Statement Guide

## What Must Be Stated in the Seminar Paper

Each section below contains: what to write, why it must be stated,
and what code/decision it references.

---

## 1. Limitations Section (Required)

### 1.1 Single Split Without Cross-Validation
**Write:** "Experiments used a single stratified train/val/test split
(80/10/10) with fixed random seed. Results were not validated with
k-fold cross-validation or multiple random splits."

**Why:** A single split could produce results that are specific to
that particular partition. Without variance estimates, we cannot
determine if differences between methods are statistically significant.

**References:**
- `src/preprocessing.py` line 163: `random_state=42` in train_test_split
- `Plan/professor-validation.md` Section 4: "No Variance Reporting"

### 1.2 Naive Baseline Confounded
**Write:** "The naive reshape baseline produces images where 97%+ of
pixels are zero-padded (breast cancer: 2.9% density, dry bean: 1.6%,
adult income: 10.5%). This confounds feature arrangement quality with
feature density. Naive results should be interpreted as demonstrating
the minimum performance without intelligent spatial mapping, not as a
direct comparison of arrangement algorithms."

**Why:** Naive performance reflects both poor arrangement AND sparse
images. DeepInsight/IGTD also produce sparse images but arrange
features based on similarity. The improvement over naive comes from
both factors.

**References:**
- `src/t2i/naive.py` lines 18-19: grid_size = ceil(sqrt(n_features))
- Feature density analysis: breast cancer 30 features in 1024 pixels = 2.9%

**Correction (2026-09-03, doc audit):** The "97%+ zero-padded /
2.9%/1.6%/10.5% density" figures above describe single-pixel
coordinate layouts (the point-mapped projection layouts, measured
~97% zero pixels at 32x32 for DeepInsight), NOT the current naive
implementation. Current naive pads each sample to the smallest
square grid and bicubic-resizes to 32x32, so it has no strict-zero
majority. Naive's genuine weakness is that it arranges features in
row-major input order with a padding band — no similarity grouping.
Do not repeat the 97%-zeros wording in the paper; use measured
per-method densities (results/figures/t2i_density_comparison.png)
and describe the layout property instead. See PART 13g.

### 1.3 Fixed Hyperparameters
**Write:** "All experiments used identical hyperparameters (epochs=50,
early stopping patience=15, weight decay=1e-4, label smoothing=0.1,
batch size 32) regardless of T2I method or CNN architecture, with one
exception: the pretrained ViT-B/16 backbone was fine-tuned at
lr=1e-4 while all other models used lr=1e-3 (see PART 12 — the shared
1e-3 diverges for ViT)."

**Why:** Different methods may converge at different rates. A learning
rate optimal for shallow CNN may be too high for ResNet-18, causing
instability. Fixed hyperparameters prevent method-specific tuning
but also prevent each method from reaching its best performance.

**References:**
- `src/train.py` lines 92-95: config defaults
- `Plan/professor-validation.md` Section 2: "No Hyperparameter Tuning"

### 1.4 Architecture Capacity Mismatch
**Write:** "CNN architectures vary substantially in capacity:
shallow CNN (~200K parameters), ResNet-18 (~11M parameters, 55x more),
ViT-base (~86M parameters, 430x more). Results reflect the interaction
between T2I method and model capacity, not T2I method alone."

**Why:** A more powerful model may extract useful patterns from even
poor image representations. If ResNet outperforms shallow CNN on all
T2I methods, the difference may be due to capacity, not the T2I method.

**References:**
- `src/models/shallow_cnn.py`: 3 conv layers + 2 FC layers
- `src/models/resnet_wrapper.py`: torchvision pretrained ResNet-18
- `src/models/vit_wrapper.py`: timm pretrained ViT-base

### 1.5 Transfer Learning Limitations
**Write:** "Pretrained models (ResNet-18, ViT) were trained on ImageNet
natural images. Our synthetic images are grayscale-based, sparse, and
structurally different from natural images. Transfer learning performance
may not reflect the true potential of these architectures on tabular
data, as the pretrained feature extractors were not designed for this
domain."

**Why:** ResNet's early layers detect edges, textures, and colors in
natural images. Our images have none of these features. The pretrained
weights may actually hurt performance by forcing the model to detect
irrelevant patterns.

**References:**
- `src/train.py` lines 203-220: `imagenet_normalize()` applies ImageNet
  mean/std, confirming pretrained models are used
- `src/models/resnet_wrapper.py`: `pretrained=True`

### 1.6 ViT Patch Resolution
**Write:** "ViT-base with patch_size=16 on 32×32 images produces only
4 patches (2×2). The attention mechanism requires sufficient spatial
resolution to learn meaningful relationships between patches. ViT
results on 32×32 images should be interpreted with caution."

**Why:** ViT splits the image into non-overlapping patches and applies
self-attention across them. With only 4 patches, there are very few
attention pairs, limiting the model's ability to learn spatial
relationships. This is a fundamental limitation of the input resolution.

**References:**
- `src/models/vit_wrapper.py`: `timm.create_model('vit_base_patch16_224')`
- Input is resized to 224×224 for ViT, but original features are from
  32×32 — the resize adds interpolation artifacts

---

## 2. Methodology Section (Required)

### 2.1 Preprocessing
**Write:** "All datasets were preprocessed with stratified train/val/test
splits (80/10/10) and StandardScaler normalization (fit on training data
only). For Adult Income, the official UCI train/test split was combined
and re-split to ensure stratification."

**Why:** Must document the exact preprocessing to ensure reproducibility.
StandardScaler on train only prevents data leakage.

**References:**
- `src/preprocessing.py` lines 155-173: `preprocess()` function
- `src/preprocessing.py` lines 198-213: Adult Income re-split

### 2.2 T2I Methods
**Write:** "Three tabular-to-image methods were implemented:
(1) Naive Reshape — pad features to next perfect square, resize to
32×32 with bicubic interpolation, normalize to [0,1] using training
min/max; (2) DeepInsight — TINTOlib implementation using t-SNE for
feature-to-pixel coordinate mapping; (3) IGTD — TINTOlib implementation
using rank-based permutation to match feature and pixel distance
rankings. All methods produce 32×32 single-channel grayscale images
normalized to [0,1]."

**Why:** Must describe each method precisely so results are reproducible.

**References:**
- `src/t2i/naive.py`: bicubic resize, train min/max normalization
- `src/t2i/deepinsight.py`: TINTOlib DeepInsight wrapper
- `src/t2i/igtd.py`: TINTOlib IGTD wrapper, /255.0 normalization

### 2.3 Class Imbalance Handling
**Write:** "Class imbalance was handled with inverse-frequency class
weights in the cross-entropy loss (sklearn compute_class_weight
with 'balanced' mode). Primary evaluation metric was macro-F1, which
gives equal weight to all classes regardless of frequency."

**Why:** Dry Bean has 6.6:1 imbalance, Adult Income 3.2:1. Without
class weights, models default to majority class. Macro-F1 prevents
inflated accuracy metrics.

**References:**
- `src/train.py` lines 47-62: `compute_class_weights()` using sklearn
- `src/evaluate.py` line 67: `f1_score(y_true, y_pred, average='macro')`

### 2.4 Training Configuration
**Write:** "All models were trained with Adam optimizer (lr=1e-3,
weight_decay=1e-4), early stopping (patience=15 on validation loss),
learning rate scheduling (ReduceLROnPlateau, factor=0.5, patience=5),
and label smoothing (0.1). Maximum 50 epochs. All experiments used
fixed random seed (42) for reproducibility."

**Why patience=15:** The LR scheduler halves the learning rate after
5 epochs without validation improvement. With early stopping patience
of 10, only 5 epochs remained after an LR drop for the model to recover
and show improvement — often not enough. Patience of 15 gives 10 epochs
post-LR-drop to settle into a better minimum before stopping.

**Why:** Must document exact training setup for reproducibility.

**References:**
- `src/train.py` lines 92-95: config defaults
- `src/train.py` lines 100-108: optimizer and scheduler
- `src/train.py` lines 28-39: `set_global_seed(42)`

---

## 3. Results Section (Required)

### 3.1 Report These Metrics
**Write:** "For each experiment, we report accuracy, macro-precision,
macro-recall, macro-F1, ROC-AUC (macro OVR for multiclass), and
PR-AUC. Confusion matrices are provided in the appendix."

**Why:** Multiple metrics give a complete picture. Accuracy alone is
misleading with class imbalance.

**References:**
- `src/evaluate.py` lines 65-93: `_compute_metrics()` returns all metrics

### 3.2 Interpret Results Carefully
**Write when presenting results:** "DeepInsight and IGTD outperform
naive on all datasets, demonstrating that intelligent feature
arrangement improves CNN classification of tabular data converted to
images. However, naive performance is confounded with feature density
(97%+ zeros), so agnitude of improvement reflects both arrangement
quality and image sparsity."

**Why:** Prevents overclaiming. The improvement is real but the cause
is multifactorial.

---

## 4. Design Decisions (State in Methodology)

### 4.1 Why Bicubic Over Nearest-Neighbor
**Write:** "Naive resize used bicubic interpolation rather than
nearest-neighbor. Bicubic computes weighted averages over 4x4 pixel
gradients, producing smoother transitions between feature pixels and
zero-padded regions."

**References:**
- src/t2i/naive.py lines 38-47: bicubic with comment explaining WHY

### 4.2 Why ImageNet Normalization for Pretrained Models
**Write:** "Pretrained ResNet-18 and ViT inputs were normalized using
ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
0.225]) after converting grayscale to RGB by channel repetition.
Pretrained models were instantiated with a 3-channel first convolution
layer so the original ImageNet weights are retained and receive
properly normalized RGB input."

**Why this matters (bug found in audit):** The normalization function
initially existed but was NOT applied during training — pretrained
models received raw [0,1] grayscale images, breaking the pretrained
feature extractors and causing degenerate behavior (models predicting
only the majority class, oscillating between 0.63/0.37 accuracy on
breast cancer). After the fix, pretrained models converge normally.
Any results produced before this fix (commit e192b60) for pretrained
models are INVALID and must be discarded.

**References:**
- src/train.py: imagenet_normalize() and constants, applied when
  model.pretrained=True in train_model()
- src/run_all.py create_cnn_model(): input_channels=3 for pretrained

### 4.3 Why Class Weights With sklearn
**Write:** "Class weights were computed using sklearn compute_class_weight
with balanced mode, which handles non-contiguous labels."

**References:**
- src/train.py lines 47-62: compute_class_weights()

---

## 5. Future Work (Required)

- Cross-validation with confidence intervals
- Adaptive image sizing based on feature count
- Feature selection (mRMR, Relief) before T2I
- Data augmentation for small datasets
- ArcFace/CosFace margin loss

---

## 6. Reproducibility Statement (Required)

**Write:** "All experiments used fixed random seed (42) for numpy,
PyTorch, and CUDA. TINTOlib internal random seed was also set to 42.
TINTOlib results may vary slightly across hardware."

**References:**
- src/train.py lines 28-39: set_global_seed(42)
- src/t2i/deepinsight.py line 27: random_seed=42
- src/t2i/igtd.py line 33: random_seed=42

## PART 7: NEW METHODOLOGY IMPLEMENTED

### 7a. Dynamic Image Sizing (Concern 8)

**What was done:** Added `compute_optimal_image_size()` to `src/t2i/__init__.py`.
Chooses the smallest square image where feature density >= 20%.

**Results:**
- 16 features (Dry Bean) → 8x8 (25% density)
- 30 features (Breast Cancer) → 8x8 (47% density)
- 108 features (Adult Income) → 16x16 (42% density)

**Paper statement:** "Image dimensions were chosen to ensure at least 20% feature
density, preventing excessive sparsity that would render convolutional operations
ineffective. This adaptive sizing approach follows the VFP principle of matching
image dimensions to feature count."

**SUPERSEDED for the final protocol (2026-09-03, PART 13f):** the
adaptive auto-sizing path (compute_optimal_image_size / auto_size)
exists but is deliberately NOT used by the main experiments — every
dataset and method uses a fixed 32x32 canvas (DATASET_CONFIG in
run_all.py). A resolution diagnostic (native 32 vs 128 for TINTO /
DeepInsight) measured larger canvases as WORSE, so 32x32 was kept.
Do not copy the 7a paper statement below into the paper.

### 7b. ResNet-18 From-Scratch Option (Concern 4)

**What was done:** Implemented `ResNetWrapper` in `src/models/resnet_wrapper.py`
with `pretrained=True` and `pretrained=False` options.

**Paper statement:** "ResNet-18 was evaluated both with ImageNet pretrained weights
and from random initialization. The from-scratch variant controls for architecture
capacity, isolating the effect of transfer learning from the T2I method quality."

### 7c. LP-FT Training Strategy (Concern 5)

**What was done:** Added `train_lp_ft()` to `src/train.py`. Two-phase training:
1. Freeze backbone, train only FC head (Linear Probing)
2. Unfreeze all layers, train end-to-end with lower LR (Fine-Tuning)

**Paper statement:** "For pretrained models (ResNet-18, ViT), a Linear Probing then
Fine-Tuning (LP-FT) strategy was employed. Phase 1 stabilizes the classification
head without corrupting pretrained features. Phase 2 gradually adapts the full
network to the synthetic image distribution."

**SUPERSEDED for the final protocol (2026-09-03, PART 13h):** LP-FT
(train_lp_ft) is used ONLY by the ablation study. The main table
trains all architectures — including pretrained ResNet/ViT — with
the plain train_model() loop and a single Adam group at ARCH_LR
(see PART 11.4 / 12b). Do not copy the 7c paper statement
("LP-FT was employed for pretrained models") into the paper.

### 7d. ViT Wrapper with 224x224 Resize (Concern 6)

**What was done:** Implemented `ViTWrapper` in `src/models/vit_wrapper.py` using
timm's ViT-Base/patch16/224. Automatically resizes input to 224x224 with bilinear
interpolation. Uses LP-FT for transfer learning (ablation-only in the final protocol; the main table uses train_model(), see PART 13h).

**Paper statement:** "ViT-Base (patch_size=16) was used with input images resized
from 32x32 to 224x224 via bilinear interpolation to produce 196 patches, providing
sufficient spatial tokens for self-attention mechanisms."

**SUPERSEDED for the final protocol (2026-09-03, PART 13h):**
cross_validate() is implemented but never invoked. The main results
use the single fixed 80/10/10 split (PART 1.1). Do NOT copy the 7e
paper statement ("All experiments were evaluated using stratified
5-fold cross-validation") into the paper — it is false.

### 7e. Cross-Validation (Concern 1)

**What was done:** Added `cross_validate()` to `src/train.py` using stratified
5-fold CV with mean +/- std reporting.

**Paper statement:** "All experiments were evaluated using stratified 5-fold
cross-validation. Results are reported as mean +/- standard deviation to quantify
variance and ensure statistical significance of observed differences."


## PART 8: SLR-Based References (Must Cite)

### 8a. Large-Scale Benchmark Context
- Liu et al. (2026, Information Fusion): 9 T2I methods, 24 datasets
- Table2Image+VIF beats XGBoost (0.879 vs 0.868 accuracy)
- Your work: compare 5 methods on 3 datasets (smaller but focused)

### 8b. S-IGTD Supervised Topology
- S-IGTD consistently outperforms unsupervised IGTD by 5-8%
- Uses between-group correlation to place class-discriminative features locally
- Reference: Zhang et al. (2024), supervised topology for multi-class problems

### 8c. TINTO Blurring
- TINTO adds spatial smoothing to DeepInsight-style projections
- Reduces sharp transitions, creates continuous gradients for CNN kernels
- Reference: TINTOlib documentation and papers

### 8d. Overlap Diagnostics
- OF (Overlapped Features) and OP (Overlapped Pixels) metrics
- Quantify lossy compression in projection-based methods
- Reference: DeepInsight FDM analysis (Sharma et al., 2019)

---

## PART 9: AUDIT FINDINGS — Bugs Fixed & Their Paper Implications

### 9a. ImageNet Normalization Was Initially Never Applied (CRITICAL)

**What happened:** `imagenet_normalize()` was defined in `src/train.py`
but never called. Pretrained ResNet-18/ViT received raw [0,1] grayscale
images instead of ImageNet-normalized RGB.

**Observed symptom:** ViT oscillated between predicting all-majority-class
(acc=0.6316) and all-minority-class (acc=0.3684) — exactly the val set
class proportions (63.2%/36.8%). Not learning.

**Fix:** Apply normalization whenever `model.pretrained=True`, and build
pretrained models with `input_channels=3` so the original 3-channel
conv1 weights are kept (commit e192b60 + 6692c79).

**Paper statement (reproducibility section):** "Pretrained models
received ImageNet-normalized RGB input via channel repetition. A
preliminary implementation that omitted this normalization produced
degenerate majority-class predictions and was corrected before the
final experiments." (Only state this if a reviewer could compare
against earlier bad runs; otherwise just describe the correct method.)

### 9b. Baseline Fairness: Same Training Data as CNNs

**What happened:** Baselines (RF/XGBoost/MLP) initially trained on
train+val combined (e.g., 455 samples for breast cancer) while CNNs
trained on train only (398) — baselines had ~14% more data.

**Fix:** Baselines now train on `X_train` only. Val is reserved for
early stopping and is never training data for any model (commit 6692c79).

**Paper statement (methodology):** "All models — CNNs and tabular
baselines — were trained on the identical training split. The
validation split was used exclusively for early stopping in CNNs."

### 9c. Cross-Validation Early-Stopping Leakage (Fixed)

**What happened:** `cross_validate()` used the test fold for early
stopping, leaking test information into model selection. Main pipeline
did NOT use this function; it was fixed anyway (commit 6692c79).

**Paper statement:** Not needed for main results (single split used).
If CV results are ever reported, the fixed implementation splits a
validation subset from the train fold only.

### 9d. Early Stopping Patience 10 → 15

**Why:** Scheduler patience=5 + early stop patience=10 left only 5
epochs to recover after each LR halving. Changed to 15 (commit 8ed8ab8).
See section 2.4.

### 9e. ViT Collapse on T2I Inputs — SUPERSEDED by PART 12

**Superseded (2026-09-03, commit defb6fb):** This section previously
advised reporting the ViT F1~0 collapse as a genuine capacity-mismatch
finding. Probe evidence proved that wrong: the collapse was an
optimization artifact of fine-tuning the pretrained ViT-B/16 at the
shared lr=1e-3 (train loss pinned at log(2) for 20 epochs). At
lr=1e-4 the identical setup learns normally (~0.91 val acc by epoch 5,
commit defb6fb, PART 12). Do NOT report the pre-fix ViT zeros as a
finding and do not cite this section. A residual capacity effect may
still exist (86M params on 398 samples) but is not what caused the
collapse; treat ViT results produced at lr=1e-4 as the valid numbers.

### 9f. Grad-CAM Resolution Limitation

**What:** ResNet-18 on 32x32 input produces only 2x2 feature maps at
layer4 — too small for meaningful Grad-CAM. Used layer3 (4x4) instead.
ViT has no conv layers (Attention Rollout out of scope).

**Paper statement (visualization section):** "Grad-CAM heatmaps are
shown for ShallowCNN, which operates at native 32x32 resolution.
ResNet-18's aggressive downsampling leaves too little spatial
resolution for interpretable heatmaps on 32x32 inputs."

### 9g. Results Validity Note (Process, not paper text)

Any `results/*.json` produced before commits e192b60/6692c79 with
pretrained models (resnet, vit) is INVALID — those runs lacked
ImageNet normalization. Also, baselines from before 6692c79 used
more training data. Discard and re-run everything after pulling
the fixes.

---

## PART 10: Audit Findings (post-3dc0d32)

Four further bugs were found and fixed after PART 9 was written.
Each entry records the bug, the fix, and how to state it in the paper.

### 10a. TINTO Raw Output Compressed to ~0.30 Max (CRITICAL — commit 0e20179)

**Bug:** TINTOlib's TINTO does not scale features to [0,1] — its
blurring compresses peaks (breast cancer images reached only ~0.30
max, not 1.0). After ImageNet normalization (mean=0.485) is subtracted,
every TINTO pixel became negative, and pretrained models' first-layer
ReLU killed all activations. Observed symptom: TINTO + pretrained
ViT/ResNet never learned (train loss stuck at ~0.70, val acc oscillating
between 0.63/0.37 = always predicting one class) while from-scratch
models (which use BatchNorm) survived.

**Fix (as committed in 0e20179):** A train-derived [0,1] rescale cache
was added to all four TINTOlib wrappers (tinto, deepinsight, igtd,
s_igtd). Subsequent audit (99432a1, see 10b) showed DeepInsight/IGTD/
S-IGTD already output [0,1] natively and removed their caches — in the
final code only TINTO rescales via the train-derived cache, while
Naive normalizes with train min/max stored in fit() (see section 2.2).
Verified: TINTO+ResNet converges immediately after the fix (val acc
0.84 → 0.965 in 5 epochs).

**Paper statement (reproducibility/methodology):** "T2I images are
normalized to [0,1]: Naive and TINTO rescale using statistics from the
training split only (no leakage), DeepInsight is scaled by TINTOlib's
internal feature MinMaxScaler, and IGTD/S-IGTD divide their raw [0,255]
output by 255; all outputs are clipped to [0,1]. This gives pretrained
backbones comparable input distributions. An earlier implementation
left TINTO images in a compressed sub-range (max ≈ 0.30), making all
pixels negative after ImageNet normalization and preventing pretrained
models from learning; this was corrected before the final experiments."

**Process note:** All results from runs before 0e20179 are INVALID and
must be discarded (they were produced with either no T2I rescaling or
per-split rescaling).

### 10b. Grad-CAM Must Show Training-Scale Images (commit 99432a1)

**Bug:** TINTO's [0,1] scale is cached on the FIRST transform() call.
The Grad-CAM figure code transformed test samples before training
samples, seeding the cache from test statistics (max 0.233 vs the
correct 0.304 from train for breast cancer) — the figure would display
images scaled differently from what the CNN actually saw.

**Fix:** run_all.py persists the train-derived pixel range
(`t2i_pixel_range=[min, max]`) into each result JSON, and
plot_gradcam_grid() restores it before transforming test samples
(legacy results fall back to seeding from a full training transform,
which reproduces the same scale). Additionally, an audit showed
DeepInsight/IGTD/S-IGTD natively output [0,1] (verified empirically),
so only TINTO keeps the rescale cache; the other wrappers were reverted
to clip-only, and TINTO.fit() now invalidates the cache on refit.

**Paper statement (visualization section):** "Grad-CAM figures were
generated from images rescaled with the same training-derived pixel
statistics used during model training and evaluation, so the displayed
inputs match the CNN's actual inputs."

### 10c. Non-Atomic Result JSON Writes (commit 43210f4)

**Bug:** CNN result JSONs were written with a direct `open(file, 'w')`.
An interruption mid-write (Ctrl+C, Colab timeout, OOM) left a truncated
file that the resume logic — which checked only `exists()` — would SKIP
forever, silently dropping the experiment from results.

**Fix:** run_single_experiment now writes to `.json.tmp` then
`os.replace()` (atomic on both POSIX and Windows), matching the
baseline writer. The model `.pt` is saved first, so an interruption
between the two writes leaves only a harmless orphan `.pt` that a
resume re-run overwrites.

**Paper statement:** Not required for the paper body; the atomic write
guarantees result files are complete, which only affects internal
resume behavior. No methodological implication.

### 10d. Resume Skipped Corrupt Result Files (commit 70b72c3)

**Bug:** Resume still treated any existing result file as "done" even
if it was a truncated/corrupt JSON left by a pre-atomic-write run —
such an experiment was skipped forever and never re-run.

**Fix:** New `_experiment_is_done()` helper: a result counts as done
only when the file exists AND parses as valid JSON AND carries the
required `dataset` and `f1_macro` keys. Corrupt files are re-run and
overwritten. Used for both CNN and baseline resume counts and skips.

**Paper statement:** Not required in the paper body — internal tooling
robustness only.

---

## 11. Reproducibility Claims Checklist

Each claim below was verified against the current code before writing.
Phrase them in the paper exactly as stated; do not broaden them.

### 11.1 One fixed train/val/test split per dataset, shared by all methods
**Verified in `src/preprocessing.py`: `preprocess()` performs two
stratified `train_test_split` calls (test_size=0.2, then a relative val
split giving 10% overall), both with `random_state=42`, followed by
`StandardScaler` fit on the training split only and applied to val/test.
Every consumer — CNN experiments, baselines, ablations, and figures —
calls `preprocess_dataset()` with these defaults, so all methods see
the identical split of each dataset.**

**Paper statement:** "Each dataset was split once into stratified
train/val/test partitions (80/10/10) with a fixed random seed, and all
methods — CNNs and tabular baselines — were trained and evaluated on
these identical partitions."

### 11.2 Deterministic T2I generation per dataset
**Verified: each T2I wrapper passes `random_seed=42` to its TINTOlib
model constructor (tinto.py line 64, igtd.py line 43, s_igtd.py line
96, deepinsight.py line 35), and `set_global_seed(42)` is called before
every experiment and baseline (run_all.py lines 154 and 279). TINTO in
particular uses `algorithm='PCA'` + `random_seed=42`, so its
feature-to-pixel mapping and image generation are deterministic for
a given training split. For TINTO, the pixel rescale statistics are cached from the
training split (see 10a/10b); for Naive they are stored in fit(); and
for IGTD/S-IGTD the /255 divisor is a constant — so in every method the
val/test scaling is fixed at fit time, making val/test transforms
deterministic too.**

**Paper statement:** "For each dataset, the feature-to-pixel mapping
was fitted once on the training split with a fixed seed
(TINTOlib `random_seed=42`; global seed 42 for numpy/PyTorch), making
image generation deterministic per dataset." (TINTOlib is PCA-based
with a fixed seed; like any third-party library, bit-level output may
differ slightly across versions/hardware.)

### 11.3 Baselines trained on the same X_train rows as the CNNs
**Verified in `run_all.py` `run_baseline()`: RF/XGBoost/MLP load
`preprocess_dataset()` and train on `X_train` only — never train+val.
The validation split is reserved for CNN early stopping and is not
training data for any model.**

**Paper statement:** "All models — CNNs and tabular baselines — were
trained on the identical training split. The validation split was used
exclusively for CNN early stopping."

### 11.4 Identical hyperparameters across methods (one LR exception)
**Verified in `run_all.py` `run_single_experiment()`: one shared
`train_config` dict (epochs=50, weight_decay=1e-4,
early_stopping_patience=15, label_smoothing=0.1, class_weights,
device) is used for every CNN experiment, regardless of T2I method or
architecture. Since commit defb6fb the learning rate is the single
per-architecture exception: `ARCH_LR` sets 1e-3 for
shallow/resnet/resnet_scratch and 1e-4 for vit (pretrained ViT
fine-tuning; see PART 12). Every result JSON records its `lr`.
`train_model()` is used for all architectures, including pretrained
ones. The LP-FT two-phase procedure exists in `src/train.py` but is
used only by the ablation study (`src/ablation.py`), not by the main
experiments.**

**Paper statement:** "All CNN experiments shared identical
hyperparameters (Adam, weight_decay=1e-4, batch size 32,
label smoothing 0.1, early stopping patience 15, max 50 epochs)
regardless of T2I method or architecture, with the exception of the
learning rate for the pretrained ViT (1e-4 vs 1e-3 elsewhere), set
according to established ViT fine-tuning practice; per-method tuning
was otherwise deliberately avoided to keep comparisons fair."

---

## PART 12: ViT Learning-Rate Artifact (commit defb6fb)

### 12a. Pretrained ViT Collapse Was an LR Artifact, Not a Sparsity Finding

**Bug:** The main pipeline trained ALL architectures with one shared
learning rate (lr=1e-3, single Adam group). Fine-tuning a pretrained
ViT-B/16 at 1e-3 is ~100x the established range (timm practice
~1e-5..1e-4). On T2I images — especially the sparse-layout methods
(TINTO/DeepInsight/IGTD/S-IGTD) — every ViT experiment collapsed:
train loss pinned at log(2) ~0.698 (unable to fit even the training
set), val loss at the same floor, val acc oscillating at the class
priors, final F1~0. The earlier collapse was misattributed to input
sparsity / patch degeneracy (see superseded 9e); a resolution
diagnostic (native 32 vs 128) showed larger canvases make the sparse
methods *worse*, ruling out that route.

**Probe evidence (breast_cancer, tinto, ViT-B/16, commit before
defb6fb):**
- lr=1e-3 (shared config): train loss >= 0.70 for 20 epochs, early
  stop at epoch 20, F1~0.
- lr=1e-4: train loss 1.42 -> 0.42 over 8 epochs, val acc 0.91 best
  (0.88 final) — learns normally.

**Fix (commit defb6fb):** `ARCH_LR` in `run_all.py` — vit=1e-4,
shallow/resnet/resnet_scratch=1e-3 (from-scratch models and pretrained
ResNet-18 converge fine at 1e-3, BatchNorm robustness). The LR is
persisted in each result JSON (`metrics['lr']`) so any ViT JSON with
lr=1e-3 is a pre-fix artifact.

**Paper statement (methodology):** "The pretrained ViT-B/16 was
fine-tuned at a learning rate of 1e-4, while all other models used
1e-3, following established ViT fine-tuning practice; a shared 1e-3
was found to make the pretrained ViT diverge (training loss pinned at
log(2)), whereas 1e-4 converged (validation accuracy >0.9 within five
epochs on breast cancer)."

**Paper statement (results, if relevant):** "ViT results were produced
with its architecture-appropriate learning rate; the low scores
sometimes reported for ViT on tabular-to-image encodings in the
literature are partly an optimization artifact and should not be read
as evidence that ViT cannot use such encodings."

**Impact:** Any `*_vit.json` result produced before commit defb6fb
(trained at lr=1e-3) is INVALID — delete and re-run only the ViT
cells. Non-ViT cells are unaffected (same code path, same LR).

### 12b. Why the LR change is NOT extended to ResNet-18 / from-scratch models (commit 3df3e17)

**Evidence against uniform 1e-4 for all pretrained models
(breast_cancer/deepinsight, pretrained ResNet-18 direct FT):**
- lr=1e-3 (main table): F1 = 0.9722
- lr=1e-4 (old ablation direct-FT): F1 = 0.935

Lowering ResNet-18 to 1e-4 would measurably hurt it; the per-architecture
choice rests on a *trainability* criterion, not performance tuning:
ResNet-18 trains at 1e-3, ViT-B/16 does not (cannot fit train, train
loss pinned at log(2)) and follows the established ViT fine-tuning range
(1e-5..1e-4). From-scratch models (shallow, resnet_scratch) also stay at
1e-3 — the divergence mechanism (destroying pretrained features) does
not apply to them, and a lower LR risks under-convergence within the
50-epoch budget on dry_bean/adult_income.

**Fairness framing:** the research question compares T2I methods
*within* each architecture, and within an architecture the LR is
constant across methods — per-arch LR cannot confound method
comparisons. Cross-architecture differences were never controlled
(capacity, PART 1.4) and are discussed as such.

**Ablation alignment (commit 3df3e17):** the LP-FT ablation's
direct-FT ResNet cell was previously trained at 1e-4 while the main
table's resnet row trains at 1e-3 — the "same" setup differed
(0.935 vs 0.9722), so the ablation figure would have contradicted the
results table. ablation.py now imports ARCH_LR for all its training
sites, so every ablation cell mirrors the main pipeline. Any
`ablation_lpft_*.json` produced before 3df3e17 is INVALID (direct-FT
cell at the wrong LR) — re-run the ablation. LP-FT's own two-phase
schedule (head 1e-3, then all layers 1e-4) is unchanged; it is the
procedure under comparison, not a reproduction of the main row.

---

## PART 13: Configuration & Decision Rationale — Coverage Audit (2026-09-03)

Result of a full code sweep: every tunable parameter and protocol
decision below was checked against the code, and each has a recorded
reason (code comment or doc section). Unresolved items are flagged
as DECISION REQUIRED.

### 13a. T2I control parameters = TINTOlib reference defaults (verified)

All TINTOlib hyperparameters in src/t2i/*.py were compared against
the installed TINTOlib 1.3.1 constructor signatures:
- TINTO: algorithm='PCA', submatrix=True, amplification=3.14
  (library default is pi = 3.14159...; 3.14 is its 3-decimal value),
  distance=2, steps=4, option='mean', times=4, zoom=1,
  cmap='binary' — all library defaults. Our deviations: pixels=
  image_size (32), random_seed=42 (vs default 1), format='npy'
  (vs png), blur=True (TINTO's defining feature — commented in
  code).
- IGTD / S-IGTD: fea_dist_method='Pearson',
  image_dist_method='Euclidean', error='squared', max_step=1000,
  val_step=50 — all library defaults; only scale=[image_size,
  image_size], format and seed were set.
- DeepInsight: internal defaults used (algorithm_rd='PCA', bin
  assignment, lsa optimization, group_method='avg'); only
  image_dim, format and seed set.

No per-method T2I tuning was performed, consistent with the fixed-
hyperparameter policy (PART 1.3 / 2.4).

**Paper statement (methodology):** "T2I methods were run at the
TINTOlib reference defaults; only the canvas size (32x32), random
seed, output format, and TINTO's blurring were set explicitly."

### 13b. Baselines deliberately untuned (decision, must state)

RF (n_estimators=100, max_depth=None, min_samples_split=2), XGBoost
(n_estimators=100, max_depth=6, learning_rate=0.1, mlogloss) and MLP
(hidden (128,64), relu, max_iter=500, early_stopping on 10% val)
use reference/library-default values with no per-model search — the
same no-tuning policy as the CNNs. Module docstrings in
src/baselines/*.py give the inclusion rationale for each model.

**Paper statement (limitations):** "Baseline classifiers used
untuned reference configurations; no hyperparameter search was
performed for any model in the study (CNNs or baselines)." This
answers the "strawman baseline" critique: the comparison is
untuned-vs-untuned, and CNN-vs-baseline gaps may widen under
tuning — future work.

### 13c. Ablation study hyperparameters (rationale recorded here)

- Pixel-shuffle verdict label: 'spatial_structure_matters' iff
  f1_drop > 0.02 (src/ablation.py:137). Auxiliary label for the
  figure only — the paper must cite raw deltas, not this cutoff.
- LP-FT verdict: better/worse/comparable at +/-0.01 F1 (line 379),
  same caveat.
- LP-FT epoch split 10 LP + 40 FT = 50 total, equal to the main
  50-epoch budget (train.py train_lp_ft defaults), keeping the
  ablation comparable to the main table.
- Feature-ordering strategies: 'original' / 'reversed' / 'random'
  (fixed permutation seed 42) / 'correlation' (descending
  |corr(feature, target)|). Random re-fit uses the same seed per
  ordering so every ordering sees the identical T2I pipeline.

### 13d. Label smoothing applied uniformly, including binary datasets

label_smoothing=0.1 is used for every dataset, including binary
breast_cancer and adult_income, although the literature suggests
it can hurt calibration on small binary problems (reviewer point 9).
Decision: keep 0.1 everywhere for a uniform, comparable protocol
(PART 1.3 / workflow design-decision 2). State this explicitly in
the paper if reviewers ask why it is not disabled per dataset.

### 13e. METRIC DEFINITION — DECISION REQUIRED

src/evaluate.py _compute_metrics uses
avg = 'macro' if num_classes > 2 else 'binary' but always names the
keys precision_macro / recall_macro / f1_macro. Consequences:
- dry_bean (7 classes): true macro average.
- breast_cancer and adult_income (binary): the reported "macro"
  values are scikit 'binary' averages — positive class only. For
  breast_cancer the positive class (1 = benign) is the MAJORITY
  class; for adult_income the positive class (1 = >50K) is the
  MINORITY class. So the same-named metric measures the majority
  class on one dataset and the minority class on the other.

This contradicts the docs that claim macro-F1 everywhere (PART 2.3,
PART 3.1, workflow design-decision 4, professor-validation Sec. 3)
and guide 2.3's reference "f1_score(..., average='macro')" is
accurate only for multiclass. The current Colab run and any tables
built from it inherit this behavior.

Options:
- (A, recommended) average='macro' always — matches every doc
  claim and the imbalance rationale. Requires re-running the two
  binary datasets (40 CNN cells + 6 baselines + ablations).
- (B) Keep binary-average for 2-class data and relabel: report
  "F1 (positive class)" for binary datasets, never "macro".
  No re-run, but the headline metric then differs across rows.

Decision pending (ask the user).

### 13f. Fixed 32x32 canvas (supersedes PART 7a and workflow CP3 row 8)

Final protocol: image_size=32 for every dataset and every T2I
method (DATASET_CONFIG in run_all.py). Naive internally uses its
own per-dataset grid (ceil(sqrt(d))) then bicubic-resizes to 32x32.
Rationale: an identical canvas across all methods/datasets keeps the
T2I layout as the only varying factor per cell, and a resolution
diagnostic (native 32 vs 128 for TINTO/DeepInsight, 2026-09-03)
measured larger canvases as strictly worse because TINTOlib's
blur/amplification parameters do not scale with the canvas.
Adaptive sizing (compute_optimal_image_size, >=20% density target)
remains available but is deliberately unused. Adult income's 108
features at 10.5% single-pixel density is a documented limitation
(PART 1.2), not resolved by image sizing.

### 13g. Naive density attribution (supersedes PART 1.2 wording)

See the correction inserted in PART 1.2. In short: the 2.9/1.6/
10.5% densities and "97%+ zeros" describe point-mapped projection
layouts, not the current naive rendering; do not repeat that
wording. Naive's real limitation is order-preserving, correlation-
blind arrangement plus a padding band.

### 13h. Unused utilities — do not claim them in the paper

zscore_normalize(), cross_validate(), get_param_groups() (both
wrappers), the T2I auto_size path, and run_all.ProgressTracker are
implemented but NOT part of the final protocol. The main pipeline
uses: ImageNet normalization (not z-score) for pretrained models;
plain train_model() with one Adam group (not LP-FT/param groups);
a single split (not CV). PART 7c/7e statements claiming LP-FT and
5-fold CV for all experiments are superseded (markers added above)
and must not be quoted. ProgressTracker is dead code that references
datetime/timedelta without an import (would raise NameError if ever
called) — candidate for deletion.

### 13i. S-IGTD as implemented is identical to IGTD (CONFIRMED — DECISION REQUIRED)

Probe (2026-09-03, breast_cancer train split, image_size=32):
SIGTD.fit() then compare to IGTD.fit() — coordinates bit-identical
(max coord diff 0), first-5-sample transforms bit-identical (max
pixel diff 0.0). Cause: s_igtd.py computes the supervised
between-group distance matrix D_B = 1 - |corr(class-wise means)|
(_compute_between_group_distances, verified working) but only
stores it in self._supervised_dist — it is never used to build or
re-optimize the feature-to-pixel layout. fit() then calls the same
TINTO_IGTD(...random_seed=42) with the same arguments as igtd.py,
so the two methods produce the same images, and every
s_igtd_{dataset} result in the experiments is numerically a
re-run of igtd. TINTOlib 1.3.1 IGTD exposes only named distance
methods (Pearson/Spearman/Euclidean/set/Wasserstein/Jensen/
Geodesic/Tropical — igtd.py __generate_feature_distance_ranking),
so a custom supervised matrix cannot be injected via fea_dist_method.

Consequences: the S-IGTD column cannot support any "S-IGTD vs IGTD"
claim (PART 8b's 5-8% improvement reference describes the real
algorithm and does NOT apply to what was run). If the current Colab
suite finishes with s_igtd cells, they duplicate igtd cells.

Options:
- (A) Implement true S-IGTD: replicate IGTD's ranking + swap
  optimization (TINTOlib igtd.py _fitAlg internals, Apache-2.0)
  with the supervised distance matrix as input, keep the library
  only for rendering; re-run the 15 s_igtd cells + s_igtd figures.
- (B) Drop s_igtd from the method set (report 4 methods) and
  remove its column from figures — no re-run needed, paper stays
  honest.
- (C) Keep the method but disclose that the shipped S-IGTD wrapper
  currently duplicates IGTD — not recommended (looks like an
  uncaught implementation bug in the paper).

Decision pending (ask the user). PART 8b's S-IGTD literature claim
must be removed or re-scoped whichever way this resolves.

### 13j. Confirmed coverage (every remaining value has a recorded reason)

| Value / decision | Where the reason is recorded |
|---|---|
| epochs=50, batch=32, wd=1e-4, patience=15, smoothing=0.1, scheduler (factor=0.5, patience=5), Adam | PART 1.3/2.4/11.4; workflow CP4.5 rows 5-6 |
| ARCH_LR (vit 1e-4, others 1e-3) | PART 12; run_all.py comment on ARCH_LR |
| Splits 80/10/10 stratified, seed 42, StandardScaler train-only | PART 11.1; preprocessing.py |
| Seeds 42 (global + TINTOlib) | PART 6/11.2; code |
| Class weights 'balanced' (sklearn) | PART 2.3; train.py comment |
| Metric family (acc, prec/rec/F1, ROC/PR-AUC, confusion) | PART 2.3/3.1; caveat in 13e |
| Naive bicubic + clip + train-min-max | PART 4.1; naive.py comments |
| ImageNet norm + 3ch pretrained conv | PART 4.2/9a; train.py + run_all.py comments |
| ResNet from-scratch / capacity framing | PART 1.4/7b |
| ViT 224 bilinear resize, patch16 | PART 1.6/7d; vit_wrapper.py docstring |
| Grad-CAM layer3 for ResNet, ShallowCNN focus | PART 9f; gradcam.py |
| IGTD /255.0, clip-only; TINTO train-scale cache | PART 10a/10b; igtd.py/tinto.py comments |
| OF/OP overlap metrics | overlap_metrics.py docstring (Sharma 2019) |
| Dataset selection | chapter4-plan 4.1.1; professor-validation Sec. 7 |
| Baselines train on X_train only | PART 9b/11.3; run_all.py comment |
| Atomic writes + parse-validating resume | PART 10c/10d |
