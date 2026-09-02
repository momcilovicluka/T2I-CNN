# Chapter 4: Experimental Results and Analysis

## Plan for Seminar Paper — Chapter 4

---

## 4.1 Experimental Setup

### 4.1.1 Datasets

**Table 1: Dataset Overview**

| Dataset | Samples | Features | Classes | Imbalance | Split (Train/Val/Test) |
|---------|---------|----------|---------|-----------|----------------------|
| Breast Cancer Wisconsin | 569 | 30 | 2 (malignant/benign) | 1.7:1 | 398/57/114 |
| Dry Bean | 13,611 | 16 | 7 | 6.6:1 | 9,527/1,361/2,723 |
| Adult Income | 48,842 | 108 (after encoding) | 2 (≤50K/>50K) | 3.2:1 | 34,188/4,885/9,769 |

**Text to write:**
- Describe each dataset: source (UCI/sklearn), domain, feature types (numerical/categorical)
- Justify dataset selection: covers binary/multi-class, small/large, balanced/imbalanced
- Explain preprocessing: StandardScaler (fit on train), stratified split, one-hot encoding
- Note Adult Income: 14 raw features → 108 after one-hot encoding

**Code reference:** `src/preprocessing.py`

### 4.1.2 Tabular-to-Image Methods

**Table 2: T2I Methods Summary**

| Method | Algorithm | Feature Mapping | Image Size | Pixel Density |
|--------|-----------|----------------|------------|---------------|
| Naive Reshape | Pad + reshape + bicubic resize | Sequential (row-major) | Adaptive | 25–47% |
| DeepInsight | t-SNE → pixel coordinates | Manifold-based | 32×32 | Varies |
| IGTD | Rank-based permutation | Distance-preserving | 32×32 | Varies |

**Text to write:**
- Describe each method's algorithm (1 paragraph each)
- Naive: pad to next perfect square, reshape to grid, bicubic upscale to 32×32
- DeepInsight: compute feature-feature correlation matrix, t-SNE to 2D, assign pixel coordinates, fill with normalized values
- IGTD: rank feature-feature distances, rank pixel-pixel distances, minimize Frobenius norm via iterative swapping
- Explain adaptive image sizing: minimum 20% feature density (Section 3.2)
- State: "All methods produce single-channel grayscale images normalized to [0,1]"

**Figures:**
- Figure 4.1: Side-by-side comparison of one sample from each method per dataset (already generated: `t2i_comparison_*.png`)

### 4.1.3 CNN Architectures

**Table 3: CNN Architecture Summary**

| Architecture | Parameters | Input Size | Training Strategy |
|-------------|-----------|------------|-------------------|
| ShallowCNN | ~620K | Adaptive (8×8 to 32×32) | End-to-end, lr=1e-3 |
| ResNet-18 (pretrained) | ~11M | 32×32 | LP-FT: head lr=1e-3, backbone lr=1e-4 |
| ResNet-18 (from scratch) | ~11M | 32×32 | End-to-end, lr=1e-3 |
| ViT-Base/16 (pretrained) | ~86M | 224×224 | LP-FT: head lr=1e-3, backbone lr=1e-5 |

**Text to write:**
- Describe each architecture (1 paragraph each)
- ShallowCNN: 3 conv blocks + adaptive pooling + FC classifier. No pretrained features. Fair baseline.
- ResNet-18: 18-layer residual network. Pretrained on ImageNet. Grayscale input via weight averaging. LP-FT training.
- ViT-Base: patch_size=16, 196 attention tokens at 224×224. Bilinear resize from 32×32. LP-FT training.
- Explain LP-FT: Phase 1 (Linear Probing: freeze backbone, train head), Phase 2 (Fine-Tune: unfreeze all, low LR)
- State parameter counts explicitly (capacity mismatch concern)

### 4.1.4 Training Configuration

**Table 4: Training Hyperparameters**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | Adam | Standard choice, adaptive learning rates |
| Learning rate | 1e-3 | Balanced for all architectures |
| Weight decay | 1e-4 | L2 regularization |
| Label smoothing | 0.1 | Reduces overconfident predictions (small datasets) |
| Early stopping patience | 10 epochs | Prevents overfitting |
| Max epochs | 50 | Most models converge before this |
| Batch size | 32 | Standard for small-medium datasets |
| Class weights | Inverse frequency | Handles class imbalance |
| Primary metric | Macro-F1 | Equal weight to all classes |

**Text to write:**
- Justify each hyperparameter choice
- Explain class weighting: Dry Bean has 6.6:1 imbalance, without weights majority-class classifier gets 26% accuracy (random-level)
- Explain macro-F1: accuracy is misleading with imbalance (Adult Income: majority class = 76.1%)
- State: "Hyperparameters were fixed across all experiments to ensure fair comparison"

### 4.1.5 Baselines

**Text to write:**
- 3 tabular baselines: Random Forest, XGBoost, MLP
- Purpose: establish upper bound for what's achievable without image transformation
- If CNN + T2I outperforms baselines → the transformation adds value
- If baselines outperform CNN → transformation may be lossy

### 4.1.6 Evaluation Protocol

**Text to write:**
- Single stratified 80/10/10 split (train/val/test)
- Early stopping on validation loss
- Final evaluation on held-out test set
- Metrics: Accuracy, Macro-F1, Precision, Recall, ROC-AUC, PR-AUC
- **Limitation:** No cross-validation, no confidence intervals (state in Limitations)

---

## 4.2 Main Results

### 4.2.1 CNN Classification Results

**Table 5: Classification Results — Macro-F1 (%) by T2I Method and Architecture**

This is the **most important table** in the paper. Structure:

| Dataset | T2I Method | ShallowCNN | ResNet-18 (pt) | ResNet-18 (scr) | ViT |
|---------|-----------|------------|----------------|-----------------|-----|
| Breast Cancer | Naive | xmm.xx | xmm.xx | xmm.xx | xmm.xx |
| | DeepInsight | xmm.xx | xmm.xx | xmm.xx | xmm.xx |
| | IGTD | xmm.xx | xmm.xx | xmm.xx | xmm.xx |
| Dry Bean | Naive | xmm.xx | xmm.xx | xmm.xx | xmm.xx |
| | DeepInsight | xmm.xx | xmm.xx | xmm.xx | xmm.xx |
| | IGTD | xmm.xx | xmm.xx | xmm.xx | xmm.xx |
| Adult Income | Naive | xmm.xx | xmm.xx | xmm.xx | xmm.xx |
| | DeepInsight | xmm.xx | xmm.xx | xmm.xx | xmm.xx |
| | IGTD | xmm.xx | xmm.xx | xmm.xx | xmm.xx |

**Text to write (per dataset, 1 paragraph each):**
- Which T2I method performed best? By how much?
- Which architecture performed best?
- Did transfer learning help or hurt?
- Did the from-scratch ResNet differ significantly from pretrained?

### 4.2.2 Baseline Comparison

**Table 6: Baseline Comparison — Macro-F1 (%)**

| Dataset | RF | XGBoost | MLP | Best CNN |
|---------|-----|---------|-----|---------|
| Breast Cancer | xmm.xx | xmm.xx | xmm.xx | xmm.xx |
| Dry Bean | xmm.xx | xmm.xx | xmm.xx | xmm.xx |
| Adult Income | xmm.xx | xmm.xx | xmm.xx | xmm.xx |

**Text to write:**
- Compare best CNN result against each baseline
- If baselines outperform CNNs → discuss why (image transformation loses information)
- If CNNs outperform → the spatial structure provides value
- Key insight: "Tabular methods have direct access to all features, while CNNs must learn spatial patterns from transformed representations"

### 4.2.3 Feature Density Analysis

**Figure 4.2: Feature Density vs. Performance**

- X-axis: feature density (%)
- Y-axis: Macro-F1
- One point per (dataset, method) combination
- Color by T2I method
- Show correlation (or lack thereof) between density and performance

**Text to write:**
- Does higher density always mean better performance?
- Naive has lowest density AND lowest performance (expected)
- DeepInsight/IGTD have similar density but may differ in performance → arrangement matters

---

## 4.3 Detailed Analysis

### 4.3.1 Per-Class Performance

**Figure 4.3: Per-class F1 scores for Dry Bean (7 classes)**

- Grouped bar chart: one group per T2I method, bars = classes
- Show which classes are hardest to classify
- BOMBAY class (3.8% of data) likely worst performing → class imbalance effect

**Text to write:**
- Which classes benefit most from intelligent T2I?
- Does class weighting help minority classes?

### 4.3.2 Confusion Matrices

**Figure 4.4: Confusion matrices for best CNN per dataset**

- 3 confusion matrices (one per dataset)
- Show misclassification patterns
- For Dry Bean: which classes get confused most?

### 4.3.3 Training Dynamics

**Figure 4.5: Training curves**

- 2 panels per dataset: loss vs epoch, accuracy vs epoch
- Overlay: ShallowCNN vs ResNet-18 vs ViT (for one T2I method)
- Show convergence speed, overfitting, early stopping

**Text to write:**
- Which architecture converges fastest?
- Does transfer learning (ResNet/ViT pretrained)

- Show convergence speed, overfitting, early stopping

**Text to write:**
- Which architecture converges fastest?
- Does transfer learning (ResNet/ViT pretrained) show faster initial convergence?
- Where does overfitting occur?

### 4.3.4 Transfer Learning Analysis

**Table 7: Transfer Learning Impact**

| Dataset | Model | Pretrained F1 | From-Scratch F1 | Delta |
|---------|-------|---------------|-----------------|-------|
| Breast Cancer | ResNet-18 | x | x | x |
| Dry Bean | ResNet-18 | x | x | x |
| Adult Income | ResNet-18 | x | x | x |

**Text to write:**
- Does ImageNet pretraining help on synthetic tabular images?
- Expected: pretrained may hurt on naive, but help on DeepInsight/IGTD

## 4.4 Ablation Study

### 4.4.1 Pixel Shuffling

**Table 8: Pixel Shuffling Ablation**

| Dataset | T2I Method | Original F1 | Shuffled F1 | Delta (%) |
|---------|-----------|-------------|-------------|-----------|
| Breast Cancer | DeepInsight | x | x | x |
| Dry Bean | DeepInsight | x | x | x |
| Adult Income | DeepInsight | x | x | x |

**Text to write:**
- If shuffling degrades performance -> spatial structure matters
- This is the key experiment validating the T2I approach

### 4.4.2 Feature Ordering

**Figure 4.6:** 4 bars per dataset: original, random, sorted-by-correlation, reversed

### 4.4.3 LP-FT vs Direct Fine-Tuning

**Table 9:** Compare LP-FT with direct fine-tuning across datasets

## 4.5 Visualization Analysis

### 4.5.1 Grad-CAM

**Figure 4.7:** Grad-CAM heatmaps showing which pixels CNN attends to

### 4.5.2 T2I Comparison

**Figure 4.8:** Feature density comparison grid (already generated)

## 4.6 Summary of Key Findings

**Table 10: Summary**

| Finding | Evidence |
|---------|----------|
| Intelligent T2I outperforms naive | Table 5 |
| Spatial structure matters | Table 8 (shuffling ablation) |
| Transfer learning mixed results | Table 7 |
| Class imbalance affects minorities | Figure 4.3 |
| T2I x architecture interaction | Table 5 |

## Writing Order

1. 4.1 Experimental Setup
2. 4.2 Main Results (Tables 5-6)
3. 4.4 Ablation Study (Table 8)
4. 4.3 Detailed Analysis (Figures 4.3-4.7)
5. 4.5 Visualization (Grad-CAM, density grid)
6. 4.6 Summary (Table 10)

## What Must Be Implemented Before Writing

### Must Have
1. `run_all.py` - trains all 27 CNN + 9 baseline configs
2. `src/baselines/*.py` - implement RF, XGBoost, MLP (currently stubs)
3. `src/ablation.py` - pixel shuffling, feature ordering (currently stubs)

### Should Have
4. `src/visualize.py` - Grad-CAM, training curves, confusion matrices
5. Per-class metrics in results

### Nice to Have
6. Cross-validation results (mean +/- std)
7. Statistical significance tests

## Expected Runtime

| Group | Count | GPU | CPU |
|-------|-------|-----|-----|
| Breast Cancer CNN | 12 | ~20min | ~2h |
| Dry Bean CNN | 12 | ~2h | ~12h |
| Adult Income CNN | 12 | ~4h | ~24h |
| Baselines | 9 | ~5min | ~15min |
| Ablation | 6 | ~20min | ~2h |
| **Total** | **~51** | **~7h** | **~40h** |
