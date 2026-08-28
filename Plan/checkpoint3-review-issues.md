# Checkpoint 3 Code Review — Issues Found

## 🔴 CRITICAL: TINTOlib CSV order doesn't match input order when data is shuffled

**File:** deepinsight.py, igtd.py (both use classification.csv for loading)
**Severity:** Would cause WRONG images for wrong samples if input is shuffled
**Evidence:** Shuffled input [0,17,15,1,...] -> CSV outputs [0,1,2,3,...] (always sequential)

**Root cause:** TINTOlib filenames are always `000000.npy`, `000001.npy`, etc. regardless
of input row order. The CSV lists them in sequential filename order, not input order.

**Current risk:** LOW — our `preprocess_dataset()` returns data in non-shuffled order from
`train_test_split`. But this is a latent bug that could cause silent wrong results.

**Fix:** Load images by constructing filenames from original indices, not from CSV order.

## 🟡 PERFORMANCE: IGTD fit() runs redundant transform to get min/max

**File:** igtd.py lines 46-55
**Impact:** Doubles IGTD fit time (fit + full transform just for statistics)

**Fix:** Compute min/max from training data directly before calling TINTOlib, or accept
the existing min/max from TINTOlib's internal normalization.

## 🟡 CODE DUPLICATION: deepinsight.py and igtd.py share ~20 identical lines

**File:** deepinsight.py lines 29-56, igtd.py lines 58-82
**Impact:** Maintenance burden, inconsistency risk

**Fix:** Extract shared TINTOlib loading logic into a helper function.

## 🟢 MINOR: Unnecessary temp directory in IGTD fit()

**File:** igtd.py lines 46-55
**Impact:** Creates and deletes temp dir during fit, leaves artifacts on crash

**Fix:** Addressed by the performance fix above.

---

## Fix Plan

### Fix 1: Replace CSV-based loading with index-based loading (CRITICAL)

In both deepinsight.py and igtd.py, replace:
```python
cls = pd.read_csv(cls_path)
for _, row in cls.iterrows():
    arr = np.load(os.path.join(tmp, row['images']))
```

With:
```python
# Load by original sample index from filename
images = []
for i in range(N):
    label = int(y[i]) if y is not None else 0
    subfolder = str(label).zfill(2)
    filename = str(i).zfill(6) + '.npy'
    img_path = os.path.join(tmp, subfolder, filename)
    arr = np.load(img_path)
    images.append(arr)
```

### Fix 2: Extract shared helper function

Create a helper that both deepinsight.py and igtd.py call:
```python
def _load_tinto_images(temp_dir, N, y):
    """Load images from TINTOlib output directory."""
    images = []
    for i in range(N):
        label = int(y[i]) if y is not None else 0
        subfolder = str(label).zfill(2)
        filename = str(i).zfill(6) + '.npy'
        img_path = os.path.join(temp_dir, subfolder, filename)
        arr = np.load(img_path)
        images.append(arr)
    return np.stack(images)
```

### Fix 3: Remove redundant transform in IGTD fit()

Replace the full transform + CSV read for min/max with:
- Use TINTOlib's internal normalization range (already [0,1] from MinMaxScaler)
- Or: use the known IGTD output range [0, 255] and always normalize by /255.0
