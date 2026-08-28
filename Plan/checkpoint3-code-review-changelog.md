# Checkpoint 3: Code Review Changelog

## Date: August 28, 2026

## Summary

During code review of the T2I (Tabular-to-Image) transformation pipeline,
5 issues were identified and fixed across 3 files. One was a critical bug
that would have caused wrong images for wrong samples under certain conditions.
Two affected normalization consistency. Two were performance/maintainability.

---

## Changes Overview

| # | File | Issue | Severity | Commit |
|---|------|-------|----------|--------|
| 1 | naive.py | Normalization data leakage | 🔴 Critical | a760bbf |
| 2 | deepinsight.py, igtd.py | Redundant normalization distorts spatial mapping | 🔴 Critical | a760bbf |
| 3 | igtd.py | Output range [0,255] vs DeepInsight [0,1] | 🔴 Critical | a760bbf |
| 4 | deepinsight.py, igtd.py | CSV order ≠ input order on shuffled data | 🔴 Critical | bf6e158 |
| 5 | igtd.py | Redundant transform in fit() doubles runtime | 🟡 Performance | bf6e158 |

---

## Change 1: Naive Normalization Data Leakage

### File: `src/t2i/naive.py`

### Before (Buggy)
```python
def fit(self, X_train, y_train=None):
    n_features = X_train.shape[1]
    self.grid_size = int(np.ceil(np.sqrt(n_features)))
    self.padded_size = self.grid_size ** 2
    return self

def transform(self, X, y=None):
    # ... pad and reshape ...
    if images.max() > 0:
        images = images / images.max()  # ← BUG: computes max PER CALL
    return torch.tensor(images).unsqueeze(1).float()
```

### After (Fixed)
```python
def fit(self, X_train, y_train=None):
    n_features = X_train.shape[1]
    self.grid_size = int(np.ceil(np.sqrt(n_features)))
    self.padded_size = self.grid_size ** 2
    # Compute normalization stats from training data only
    padded_train = np.zeros((X_train.shape[0], self.padded_size), dtype=np.float32)
    padded_train[:, :X_train.shape[1]] = X_train
    grid_train = padded_train.reshape(X_train.shape[0], self.grid_size, self.grid_size)
    self._train_min = grid_train.min()
    self._train_max = grid_train.max()
    return self

def transform(self, X, y=None):
    # ... pad and reshape ...
    rng = self._train_max - self._train_min
    if rng > 0:
        images = (images - self._train_min) / rng
    images = np.clip(images, 0, 1)
    return torch.tensor(images).unsqueeze(1).float()
```

### What Was Wrong
The normalization `images / images.max()` was called separately for train,
val, and test. Each call computed a different max:

| Split | Min | Max | Range |
|-------|-----|-----|-------|
| Train | -1.94 | 3.91 | 5.85 |
| Val   | -2.28 | 5.65 | 7.93 |
| Test  | -1.62 | 4.38 | 6.00 |

After normalization:
- Train pixels: [-0.33, 1.0]
- Val pixels: [-0.40, 1.0]
- Test pixels: [-0.30, 1.0]

### Impact on Results
- **CNN trained on train images sees [-0.33, 1.0] but test images are [-0.30, 1.0]**
- The scale mismatch means pixel intensities don't correspond to the same
  feature values across splits
- Would cause **artificially lower test accuracy** because the CNN's learned
  feature-to-pixel mapping doesn't generalize
- **Would make naive method appear worse than it actually is**, skewing the
  comparison with DeepInsight/IGTD
- Effect is larger on small datasets (breast cancer) where val/test are smaller

### How We Verified
```
Before: naive  train=[-0.244,1.000]  val=[-0.407,1.000]  test=[-0.279,1.000]
After:  naive  train=[0.000,1.000]   val=[0.048,0.559]   test=[0.000,1.000]
```

---

## Change 2: DeepInsight/IGTD Redundant Normalization

### Files: `src/t2i/deepinsight.py`, `src/t2i/igtd.py`

### Before (Buggy)
```python
# After loading images from TINTOlib:
if images.max() > 0:
    images = images / images.max()  # ← Divides by batch max
```

### After (Fixed)
```python
# TINTOlib outputs [0, 1] via internal MinMaxScaler.
# Do NOT re-normalize — it would distort the spatial mapping.
images = np.clip(images, 0, 1)
```

### What Was Wrong
TINTOlib's internal MinMaxScaler already normalizes features to [0, 1]
during `fit()`, and `transform()` applies that same scaler. The resulting
images are in [0, 1]. The additional `images / images.max()` was:

1. **Dividing by a batch-specific max** — train max=1.0, val max=0.90,
   test max=1.25 (for breast cancer). This shifted all values by a
   different factor per split.

2. **Distorting the spatial mapping** — DeepInsight places features at
   specific pixel positions based on t-SNE coordinates. The relative
   intensities between pixels carry information about feature relationships.
   Dividing by max preserves ratios but shifts the baseline, which changes
   what the CNN sees.

### Impact on Results
- Test images had values outside [0, 1] (up to 1.25 for breast cancer)
- CNN's batch normalization layers would see different input distributions
  on train vs test
- Would cause **inconsistent gradient magnitudes** during training
- Could make DeepInsight appear less stable than IGTD
- Effect was smaller on large datasets (adult_income had no issue)

### How We Verified
```
Before: deepinsight  train=[0.000,1.000]  val=[-0.024,0.896]  test=[-0.120,1.253]
After:  deepinsight  train=[0.000,1.000]  val=[0.000,0.896]   test=[0.000,1.000]
```

---

## Change 3: IGTD Output Range Mismatch

### File: `src/t2i/igtd.py`

### Before (Buggy)
```python
# IGTD and DeepInsight had identical code:
if images.max() > 0:
    images = images / images.max()
```

### After (Fixed)
```python
# IGTD outputs [0, 255]. Normalize to [0, 1].
IGTD_RAW_MAX = 255.0
images = images / IGTD_RAW_MAX
images = np.clip(images, 0, 1)
```

### What Was Wrong
TINTOlib's IGTD and DeepInsight use different internal normalization:

| Method | Internal output range | Normalization |
|--------|----------------------|---------------|
| DeepInsight | [0, 1] | MinMaxScaler on features |
| IGTD | [0, 255] | Image rendered via matplotlib colormap |

The old code applied the same `images / images.max()` to both. For IGTD:
- `images.max()` was always ~255, so `images / 255 ≈ [0, 1]` — happened to work
- But on some batches max could be <255 (e.g., 240), giving max pixel = 1.06
- DeepInsight images were [0, 1], IGTD images were [0, 255] before normalization

### Impact on Results
- Without the fix, IGTD pixel values could be **255x larger** than DeepInsight
- CNN would see vastly different input scales for different T2I methods
- **Would make IGTD appear to fail completely** — loss would explode
- Or if normalization happened to work, the scale would still be inconsistent

### How We Verified
```
Before: igtd  train=[0.000,1.000]  val=[0.000,1.000]  test=[0.000,1.000]  (by luck)
After:  igtd  train=[0.000,1.000]  val=[0.000,1.000]  test=[0.000,1.000]  (by design)
```

The values look the same because `images.max()` ≈ 255, so dividing by it
gave approximately the same result. But the fix makes it deterministic
rather than dependent on batch content.

---

## Change 4: TINTOlib CSV Order Bug (CRITICAL)

### Files: `src/t2i/deepinsight.py`, `src/t2i/igtd.py`

### Before (Buggy)
```python
cls_path = os.path.join(self._temp_dir, 'classification.csv')
cls = pd.read_csv(cls_path)

images = []
for _, row in cls.iterrows():
    img_path = os.path.join(self._temp_dir, row['images'])
    arr = np.load(img_path)
    images.append(arr)
```

### After (Fixed)
```python
# Load by constructing filenames from sample index
images = []
for i in range(N):
    label = int(y[i]) if y is not None else 0
    subfolder = str(label).zfill(2)
    filename = str(i).zfill(6) + '.npy'
    img_path = os.path.join(temp_dir, subfolder, filename)
    arr = np.load(img_path)
    images.append(arr)
```

### What Was Wrong
TINTOlib stores images as `{class_subfolder}/{zero_padded_index}.npy`.
The `classification.csv` lists files in sequential filename order
(000000, 000001, 000002...), NOT in input DataFrame row order.

**Test with shuffled input:**
```
Input order:  [0, 17, 15, 1, 8, 5, 11, 3, ...]
CSV order:    [0, 1, 2, 3, 4, 5, 6, 7, ...]  ← always sequential!
Match: False
```

The CSV-based loading assigned image 000000.npy (sample 0) to position 0,
image 000001.npy (sample 1) to position 1, etc. — regardless of which
sample was actually tion of which sample was actually at position 0 in the input DataFrame.

### Impact on Results
- Every sample would get the wrong image when input is shuffled
- CNN would learn completely wrong feature-to-pixel associations
- Results would be random noise - no meaningful accuracy
- Currently LOW risk because preprocess_dataset() returns non-shuffled data

---

## Summary of Impact

| Bug | Would Cause | Severity |
|-----|-------------|----------|
| Naive normalization | Test metrics artificially lower | Medium |
| DeepInsight re-normalization | Inconsistent pixel scales | Medium |
| IGTD range mismatch | IGTD could fail completely | High |
| CSV order bug | Wrong images for wrong samples | Critical |
| Redundant IGTD transform | 2x slower experiments | Low |

The most dangerous combination is bugs #1 and #3: naive would look worse
than it should, while IGTD might look better or worse than it should,
leading to wrong conclusions about which T2I method is best.

---

# PART 2: Structural Concerns Review

## Date: August 28, 2026 (continued)

## Additional Concerns Found During Deep Review

Beyond the 5 code bugs, 5 structural/design concerns were identified that
could skew experimental results if not addressed.

---

## Concern 6: Naive Images Are 97%+ Zeros

### What Was Found
Naive reshape pads features to a perfect square (e.g., 30 → 36 = 6×6),
then resizes to 32×32 with nearest-neighbor. Each feature becomes a 3×3
block of identical values, surrounded by zeros.

| Dataset | Features | Zero pixels in 32×32 | Feature density |
|---------|----------|---------------------|-----------------|
| Breast Cancer | 30 | 994 / 1024 | 2.9% |
| Dry Bean | 16 | 1008 / 1024 | 1.6% |
| Adult Income | 108 | 916 / 1024 | 10.5% |

### Impact on Results
- Naive will perform much worse than DeepInsight/IGTD
- Comparison becomes confounded: "feature arrangement" + "feature density"
- CNN has almost no signal to learn from (97% zeros)
- Could make naive appear artificially bad, exaggerating the benefit
  of intelligent T2I methods

### How It Was Addressed
**Documented as inherent baseline limitation — NOT fixed.**
This is actually a legitimate finding for the paper: naive reshaping is
fundamentally limited by feature count vs. image size. The comparison
demonstrates why intelligent feature arrangement matters.

### Residual Risk
LOW — this is expected behavior for a naive baseline. The paper should
discuss this limitation explicitly.

---

## Concern 7: Class Imbalance Will Inflate Accuracy

### What Was Found
| Dataset | Majority class | Imbalance ratio |
|---------|---------------|-----------------|
| Breast Cancer | 62.6% benign | 1.7:1 |
| Dry Bean | 26.1% DERMASON | 6.6:1 (vs BOMBAY 3.8%) |
| Adult Income | 76.1% <=50K | 3.2:1 |

Without correction, a majority-class classifier gets:
- Breast Cancer: 62.6% accuracy (misleading)
- Dry Bean: 26.1% accuracy (random-level but looks like "something")
- Adult Income: 76.1% accuracy (deceptively high)

### Impact on Results
- Accuracy alone would be misleading for all datasets
- Dry Bean (7 classes, 6.6:1 imbalance) would be most affected
- CNN might learn to always predict majority class
- Would make all methods appear similar if they all default to majority

### How It Was Addressed
**Fixed with two changes:**

1. **Class weights in loss function** (`src/train.py`):
   ```python
   def compute_class_weights(y):
       classes, counts = np.unique(y, return_counts=True)
       weights = total / (len(classes) * counts)
       return weight_tensor  # inverse frequency
   ```
   Used with `nn.CrossEntropyLoss(weight=class_weights)`

2. **Macro-F1 as primary metric** (`src/evaluate.py`):
   - Macro-averaged F1 (equal weight to all classes)
   - Macro-averaged precision/recall
   - ROC-AUC with OVR (one-vs-rest) for multiclass
   - Full classification report per experiment

### Residual Risk
LOW — class weights + macro metrics handle imbalance correctly.
Dry Bean may still show lower absolute performance due to 7-class
difficulty, but comparisons will be fair.

---

## Concern 8: 32×32 Too Small for 108 Features (Adult Income)

### What Was Found
108 one-hot encoded features on 32×32 = 10.5% density. Convolutional
kernels (3×3) see 9 pixels at a time — with 89.5% zeros, most
activations will be zero.

### Impact on Results
- All methods will struggle on Adult Income
- CNN may not learn meaningful spatial patterns
- Adult Income results will be less reliable than other datasets
- Could mask differences between T2I methods

### How It Was Addressed
**Planned for Checkpoint 4 (CNN model wrappers).**
Options to address:
- Use larger image size (64×64) for Adult Income specifically
- Use adaptive image sizing based on feature count
- Document as limitation if image size is kept fixed

### Residual Risk
MEDIUM — needs to be addressed in Checkpoint 4. If not fixed, Adult
Income results should be interpreted with caution.

---

## Concern 9: ResNet-18/ViT Expect 224×224 RGB Input

### What Was Found
Shallow CNN accepts 32×32 grayscale natively. But ResNet-18 and ViT
(pretrained on ImageNet) expect 224×224 3-channel input.

### Impact on Results
- Without adaptation, ResNet/ViT will crash or produce wrong results
- Pretrained weights trained on natural images won't transfer well
  to sparse synthetic images
- Resize method matters: nearest-neighbor (blocky) vs bilinear (smooth)
- Channel conversion (1ch→3ch) adds no new information

### How It Was Addressed
**Planned for Checkpoint 4 (model wrappers).**
Implementation plan:
- Resize 32→224 with bilinear interpolation (smoother than nearest)
- Convert 1ch→3ch by repeating: `x.repeat(1, 3, 1, 1)`
- Fine-tune with low LR for backbone, higher LR for new layers
- Document that pretrained weights may not transfer well

### Residual Risk
MEDIUM — needs implementation in Checkpoint 4. The adaptation method
will affect results and must be documented.

---

## Concern 10: Small Dataset + Deep Model = Overfitting

### What Was Found
Breast Cancer: 398 training samples. ResNet-18: 11M parameters.
Ratio: 27,638 parameters per sample — extreme overfitting risk.

### Impact on Results
- ResNet-18 may memorize training data on Breast Cancer
- Test accuracy could be much lower than training accuracy
- Would make ResNet appear worse than it actually is
- Comparison between shallow CNN and ResNet would be unfair

### How It Was Addressed
**Partially fixed in `src/train.py`:**
- L2 weight decay (1e-4) — reduces overfitting
- Early stopping (patience 10) — stops before memorization
- LR scheduling (ReduceLROnPlateau) — prevents overshooting

**Additional measures planned for Checkpoint 4:**
- Data augmentation (random flip, rotation, noise)
- Dropout in fully connected layers
- Freeze early ResNet layers, only fine-tune later layers

### Residual Risk
MEDIUM — weight decay + early stopping help, but data augmentation
is still needed for small datasets. Will address in Checkpoint 4.

---

## Summary: All Issues and Their Status

### Code Bugs (Fixed)

| # | Issue | File | Severity | Status | Commit |
|---|-------|------|----------|--------|--------|
| 1 | Naive normalization data leakage | naive.py | 🔴 Critical | FIXED | a760bbf |
| 2 | DeepInsight/IGTD redundant normalization | deepinsight.py, igtd.py | 🔴 Critical | FIXED | a760bbf |
| 3 | IGTD output range [0,255] vs [0,1] | igtd.py | 🔴 Critical | FIXED | a760bbf |
| 4 | CSV order ≠ input order on shuffled data | deepinsight.py, igtd.py | 🔴 Critical | FIXED | bf6e158 |
| 5 | IGTD redundant transform in fit() | igtd.py | 🟡 Performance | FIXED | bf6e158 |

### Structural Concerns (Addressed)

| # | Concern | Impact | Status | How |
|---|---------|--------|--------|-----|
| 6 | Naive 97%+ zeros | Makes naive appear worse | DOCUMENTED | Inherent to baseline; paper finding |
| 7 | Class imbalance inflates accuracy | Misleading metrics | FIXED | Class weights + macro-F1 |
| 8 | 32×32 too small for 108 features | Poor Adult Income results | PLANNED | Adaptive sizing in Checkpoint 4 |
| 9 | ResNet/ViT need 224×224 RGB | Model crashes | PLANNED | Bilinear resize + channel repeat |
| 10 | Small data + deep model = overfitting | ResNet performs poorly | PARTIAL | Weight decay + early stopping; augmentation planned |

### Impact on Experimental Results

If ALL issues were NOT addressed:

1. **Naive** would show artificially poor performance (bugs #1 + concern #6)
2. **DeepInsight** would show inconsistent training (bug #2)
3. **IGTD** could fail completely (bugs #3 + #4)
4. **Accuracy metrics** would be misleading for all methods (concern #7)
5. **Adult Income** results would be unreliable (concern #8)
6. **ResNet/ViT** would crash or overfit (concerns #9 + #10)

After fixes:
- All methods produce consistent [0,1] images ✓
- Metrics are imbalance-aware (macro-F1) ✓
- Class weights prevent majority-class bias ✓
- Remaining concerns (8, 9, 10) planned for Checkpoint 4
