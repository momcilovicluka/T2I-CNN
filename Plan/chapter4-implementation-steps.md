# Chapter 4 Implementation Steps

> **STATUS 2026-09-03:** STEP 2 (S-IGTD wrapper) was later DROPPED from
the study — its supervised distance was never used, so it duplicated
IGTD (paper-statement-guide PART 13i). Final method set: naive, TINTO,
DeepInsight, IGTD. Experiment counts below (60/69) are superseded by
48 CNN + 9 baselines = 57.

Split into 7 independent steps. Each produces working, testable code.
Step 0 updates existing code before anything new.

---

## STEP 0: Update Existing Code
**Goal:** Make current code work with the new 5-method experiment matrix.
**Time:** ~30 min
**Dependencies:** None

### What to change:

1. **`src/t2i/__init__.py`** — Add TINTO and S-IGTD to METHODS dict
   - Add `from .tinto import TINTO` and `from .s_igtd import SIGTD`
   - Add `'tinto': TINTO` and `'s_igtd': SIGTD` to METHODS dict
   - Update `verify_all_transformers()` to test all 5 methods

2. **`run_all.py`** — Update T2I_METHODS list
   - Change `T2I_METHODS = ['naive', 'deepinsight', 'igtd']`
   - To `T2I_METHODS = ['naive', 'tinto', 'deepinsight', 'igtd', 's_igtd']`
   - Update print statements and help text

3. **`src/visualize_t2i.py`** — Add TINTO and S-IGTD to comparison plots
   - Add TINTO and S-IGTD to the methods list
   - Update figure grid (now 5 columns instead of 3)

### Validation:
```bash
python -c "from src.t2i import T2ITransformer; print(T2ITransformer.METHODS.keys())"
# Should show: dict_keys(['naive', 'tinto', 'deepinsight', 'igtd', 's_igtd'])
```

### Files modified: 3 (t2i/__init__.py, run_all.py, visualize_t2i.py)

---

## STEP 1: Implement TINTO Wrapper
**Goal:** Add TINTO as 4th T2I method.
**Time:** ~20 min
**Dependencies:** Step 0

### What to build:

1. **`src/t2i/tinto.py`** — TINTO wrapper (~40 lines)
   - Same pattern as deepinsight.py (uses TINTOlib)
   - Key difference: uses `tintolib.TINTO()` with blurring parameters
   - TINTOlib applies artistic blurring filter to smooth spatial patterns
   - After TINTOlib generates images, load with `_load_tinto_images()`
   - Normalize to [0, 1] same as other methods

### TINTOlib API:
```python
from tintolib import TINTO
model = TINTO(image_size=32, random_seed=42)
model.fit(df_train)  # df with features + target column last
model.transform(df_test, temp_dir)
# Output: .npy files in temp_dir/{class}/{index}.npy
```

### Validation:
```python
from src.t2i import T2ITransformer
t = T2ITransformer(method='tinto', image_size=32)
t.fit(X_train, y_train)
images = t.transform(X_train, y_train)
print(f'TINTO: {images.shape}, range=[{images.min():.3f}, {images.max():.3f}]')
```

### Files created: 1 (src/t2i/tinto.py)

---

## STEP 2: Implement S-IGTD Wrapper
**Goal:** Add S-IGTD (supervised IGTD) as 5th T2I method.
**Time:** ~30 min
**Dependencies:** Step 0

### What to build:

1. **`src/t2i/s_igtd.py`** — S-IGTD wrapper (~50 lines)
   - Uses TINTOlib's IGTD but with supervised distance computation
   - Key difference from IGTD: computes between-group correlation using class labels
   - Distance matrix uses class-wise means: D_B(i,j) = 1 - |corr([mu_i_1,...,mu_i_C], [mu_j_1,...,mu_j_C])|
   - This places class-discriminative features in local neighborhoods
   - After TINTOlib generates images, load with `_load_tinto_images()`

### TINTOlib API for S-IGTD:
```python
from tintolib import SIGTD  # or S_IGTD
model = SIGTD(image_size=32, random_seed=42)
model.fit(df_train)  # df with features + target column last
model.transform(df_test, temp_dir)
```

### If TINTOlib doesn't have S-IGTD directly:
- Use IGTD with correlation distance (1 - |corr|) instead of Euclidean
- This is a reasonable approximation for supervised topology
- Document this as a limitation

### Validation:
```python
from src.t2i import T2ITransformer
t = T2ITransformer(method='s_igtd', image_size=32)
t.fit(X_train, y_train)  # y_train used for class-aware distances
images = t.transform(X_train, y_train)
print(f'S-IGTD: {images.shape}, range=[{images.min():.3f}, {images.max():.3f}]')
```

### Files created: 1 (src/t2i/s_igtd.py)

---

## STEP 3: Implement Overlap Diagnostics
**Goal:** Compute OF/OP metrics for projection-based methods.
**Time:** ~20 min
**Dependencies:** None (can run in parallel with Steps 1-2)

### What to build:

1. **`src/t2i/overlap_metrics.py`** — Overlap diagnostic functions (~30 lines)
   - `compute_overlap(coordinates, image_size)` → dict with OF, OP
   - coordinates: dict mapping feature_index → (x, y) pixel coordinate
   - OF = (features at occupied pixels / total features) × 100
   - OP = (pixels with >1 feature / total active pixels) × 100
   - IGTD/S-IGTD: always return OF=0, OP=0 (collision-free by design)

2. **Update `src/t2i/deepinsight.py` and `tinto.py`** — Store coordinate maps
   - After fit(), save the feature→pixel coordinate mapping
   - Expose via `transformer.get_coordinates()` method
   - This enables overlap computation without re-running T2I

### Validation:
```python
from src.t2i.overlap_metrics import compute_overlap
# After fitting DeepInsight:
coords = t.transformer.get_coordinates()
metrics = compute_overlap(coords, image_size=32)
print(f"OF={metrics['of']:.1f}%, OP={metrics['op']:.1f}%")
```

### Files created: 1, files modified: 2

---

## STEP 4: Implement Baselines
**Goal:** Working RF, XGBoost, MLP baselines.
**Time:** ~20 min
**Dependencies:** None (can run in parallel)

### What to build:

1. **`src/baselines/rf.py`** — Random Forest (~15 lines)
   - `RandomForestClassifier(n_estimators=100, random_state=42)`
   - Fit on X_train, predict on X_test
   - Return metrics via `evaluate_tabular()`

2. **`src/baselines/xgboost_model.py`** — XGBoost (~15 lines)
   - `XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False)`
   - Same fit/predict/evaluate pattern

3. **`src/baselines/mlp.py`** — MLP (~15 lines)
   - `MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)`
   - Same fit/predict/evaluate pattern

### All baselines share:
- Input: raw tabular features (NOT images)
- Use `src/evaluate.py`'s `evaluate_tabular()` for metrics
- Save results to `results/baseline_{dataset}_{model}.json`

### Validation:
```python
from src.baselines.rf import train_and_evaluate
result = train_and_evaluate(X_train, y_train, X_test, y_test)
print(f"RF F1: {result['f1_macro']:.4f}")
```

### Files modified: 3 (rf.py, xgboost_model.py, mlp.py)

---

## STEP 5: Build Experiment Runner (run_all.py)
**Goal:** Automated pipeline for all 69 experiments.
**Time:** ~45 min
**Dependencies:** Steps 0-4

### What to build:

1. **`run_all.py`** — Full experiment runner
   - `run_single_experiment(dataset, t2i, cnn)`:
     a. Load dataset → preprocess
     b. Fit T2I on train → transform train/val/test
     c. Create DataLoaders
     d. Initialize CNN model
     e. Train with early stopping
     f. Evaluate on test set
     g. Save to `results/{dataset}_{t2i}_{cnn}.json`
   - `run_baseline(dataset, model)`:
     a. Load dataset → preprocess (raw features)
     b. Train baseline model
     c. Evaluate on test set
     d. Save to `results/baseline_{dataset}_{model}.json`

2. **Results CSV generation** — `results/all_experiments.csv`
   - Aggregate all JSON results into one CSV
   - Columns: dataset, t2i_method, cnn_arch, accuracy, f1_macro, precision, recall, roc_auc, pr_auc

### Command line:
```bash
python run_all.py                        # All 69 experiments
python run_all.py --dataset breast_cancer # One dataset only
python run_all.py --baselines           # Just the 9 baselines
python run_all.py --dry-run             # Print what would run, don't train
```

### Validation:
```bash
python run_all.py --dry-run
# Should print: "Would run 60 CNN experiments + 9 baselines"
```

### Files created/modified: 1 (run_all.py)

---

## STEP 6: Build Ablation Study
**Goal:** Prove spatial structure matters for CNN performance.
**Time:** ~30 min
**Dependencies:** Steps 0-4

### What to build:

1. **`src/ablation.py`** — Full ablation study

   **Ablation 1: Pixel Shuffling**
   - Take trained model + T2I images
   - Shuffle pixel positions (fixed seed)
   - Evaluate: accuracy should drop significantly
   - If it doesn't → CNN isn't using spatial structure

   **Ablation 2: Feature Ordering**
   - Generate images with different feature orderings:
     a. Original (DeepInsight/IGTD layout)
     b. Random ordering
     c. Sorted by correlation with target
     d. Reversed ordering
   - Compare perform
   - Compare performance across orderings

   **Ablation 3: LP-FT vs Direct Fine-Tuning**
   - Train ResNet-18 with LP-FT (current approach)
   - Train ResNet-18 with direct fine-tuning (no freezing)
   - Compare: LP-FT should be more stable

### Output:
- ablation_pixel_shuffling.json
- ablation_feature_ordering.json
- ablation_lp_ft_comparison.json

### Command line:
```bash
python src/ablation.py --dataset breast_cancer --t2i deepinsight --cnn resnet
python src/ablation.py --all  # Run all ablations
```

### Files created/modified: 1 (src/ablation.py)

---

## STEP 7: Build Visualization Pipeline
**Goal:** Generate all figures for Chapter 4.
**Time:** ~30 min
**Dependencies:** Steps 5-6 (needs experiment results)

### What to build:

1. **`src/visualize.py`** — Figure generation script

   **Figure 4.2: T2I Comparison Grid** (already done)
   - 5 columns (Naive, TINTO, DeepInsight, IGTD, S-IGTD) x 3 rows (datasets)

   **Figure 4.3: Main Results Heatmap**
   - Rows: T2I methods (5)
   - Columns: CNN architectures (4)
   - Cell color: Macro-F1
   - One heatmap per dataset -> 3 figures

   **Figure 4.4: Baseline Comparison Bar Chart**
   - Grouped bars: RF, XGBoost, MLP, best CNN per dataset

   **Figure 4.5: Per-Class F1 (Dry Bean)**
   - Grouped bar chart: 5 T2I methods x 7 classes

   **Figure 4.6: Training Curves**
   - Loss/accuracy vs epoch for each architecture

   **Figure 4.7: Confusion Matrices**
   - Best CNN per dataset

   **Figure 4.8: Ablation Results**
   - Pixel shuffling: original vs shuffled
   - Feature ordering: 4 orderings

   **Figure 4.9: Grad-CAM Heatmaps**
   - Input image + Grad-CAM overlay
   - Best/worst T2I methods

   **Figure 4.10: Feature Density vs Performance**
   - Scatter plot: density (x) vs F1 (y), colored by method

### Output:
- All figures saved to results/figures/

### Files created/modified: 1 (src/visualize.py)

---

## Implementation Order (Dependency Graph)

```
STEP 0 (Update existing)
    |
    +-- STEP 1 (TINTO wrapper)
    |       |
    +-- STEP 2 (S-IGTD wrapper)
    |       |
    +-- STEP 3 (Overlap diagnostics) -- can run in parallel with 1,2
    |
    +-- STEP 4 (Baselines) -- can run in parallel with 1,2,3
    |
    +-- STEP 5 (Experiment runner) -- needs 0-4
            |
            +-- STEP 6 (Ablation) -- needs 0-4
            |
            +-- STEP 7 (Visualization) -- needs 5,6
```

### Parallelizable:
- Steps 1, 2, 3, 4 can all run in parallel after Step 0
- Steps 6 and 7 can run in parallel after Step 5

### Estimated Time:
- Step 0: 30 min
- Steps 1-4: ~90 min (parallel) or ~2 hours (sequential)
- Step 5: 45 min
- Steps 6-7: ~60 min (parallel)
- **Total: ~4-5 hours of coding** (not including experiment runtime)

---

## Quick Reference: Files to Create/Modify

| Step | Create | Modify |
|------|--------|--------|
| 0 | -- | t2i/__init__.py, run_all.py, visualize_t2i.py |
| 1 | t2i/tinto.py | -- |
| 2 | t2i/s_igtd.py | -- |
| 3 | t2i/overlap_metrics.py | t2i/deepinsight.py, t2i/tinto.py |
| 4 | -- | baselines/rf.py, baselines/xgboost_model.py, baselines/mlp.py |
| 5 | -- | run_all.py |
| 6 | ablation.py (full rewrite) | -- |
| 7 | visualize.py (full rewrite) | -- |
