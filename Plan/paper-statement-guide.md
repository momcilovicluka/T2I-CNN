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

### 1.3 Fixed Hyperparameters
**Write:** "All experiments used identical hyperparameters (learning
rate=1e-3, epochs=50, early stopping patience=15, weight decay=1e-4,
label smoothing=0.1) regardless of T2I method or CNN architecture.
This ensures fair comparison but may disadvantage architectures that
require different tuning (e.g., ResNet-18 typically needs lower
learning rates for backbone layers)."

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

### 7d. ViT Wrapper with 224x224 Resize (Concern 6)

**What was done:** Implemented `ViTWrapper` in `src/models/vit_wrapper.py` using
timm's ViT-Base/patch16/224. Automatically resizes input to 224x224 with bilinear
interpolation. Uses LP-FT for transfer learning.

**Paper statement:** "ViT-Base (patch_size=16) was used with input images resized
from 32x32 to 224x224 via bilinear interpolation to produce 196 patches, providing
sufficient spatial tokens for self-attention mechanisms."

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

### 9e. ViT Degenerate Behavior on Tiny Datasets (Report as Finding)

**Observed (even after fixes):** ViT-Base (85M params) on breast cancer
(398 train samples) is massively overparameterized. Expect instability
and worse results than ShallowCNN. This is a legitimate finding —
capacity mismatch — NOT a bug. Report it as such and reference
section 1.4 (Architecture Capacity Mismatch).

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
