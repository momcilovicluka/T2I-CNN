# Seminar 2: Technical Implementation Workflow

## Overview

This is the step-by-step implementation plan for everything **before** writing the paper.
Each checkpoint produces concrete artifacts (files, results, graphs) that feed directly into the seminar.

---

## CHECKPOINT 0: Environment Setup
**Goal:** Reproducible Python environment with all dependencies

### Steps:
1. Create project directory structure:
```
seminar2/
├── data/                  # Raw datasets
│   ├── breast_cancer/
│   ├── dry_bean/
│   └── adult_income/
├── images/                # Generated T2I images
│   ├── naive/
│   ├── deepinsight/
│   └── igtd/
├── src/
│   ├── preprocessing.py   # Dataset loading + cleaning
│   ├── t2i/               # Tabular-to-image methods
│   │   ├── naive.py
│   │   ├── deepinsight.py
│   │   └── igtd.py
│   ├── models/            # CNN architectures
│   │   ├── shallow_cnn.py
│   │   ├── resnet_wrapper.py
│   │   └── vit_wrapper.py
│   ├── baselines/         # Tabular ML models
│   │   ├── rf.py
│   │   ├── xgboost_model.py
│   │   └── mlp.py
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Metrics computation
│   ├── ablation.py        # Ablation experiments
│   └── visualize.py       # Grad-CAM, charts, confusion matrices
├── results/               # Saved metrics, CSVs, figures
├── notebooks/             # Jupyter notebooks for exploration
├── checkpoints/           # Saved model weights
├── requirements.txt
└── run_all.py             # Master script to run everything
```

2. Install dependencies:
```bash
pip install torch torchvision timm tintolib scikit-learn xgboost \
            pandas numpy matplotlib seaborn grad-cam umap-learn
```

3. Verify: `python -c "import torch; print(torch.cuda.is_available())"`

### Artifact: `requirements.txt`, working directory structure

---

## CHECKPOINT 1: Dataset Acquisition & EDA
**Goal:** All 3 datasets downloaded, cleaned, and profiled

### Steps:
1. **Breast Cancer Wisconsin** — `sklearn.datasets.load_breast_cancer()`
   - 569 samples, 30 features, binary (malignant/benign)
   - No missing values, all numerical

2. **Dry Bean** — UCI ML Repository
   - ~13,000 samples, 16 features, 7 classes
   - Download from: https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset
   - CSV: `DryBeanDataset/Dry_Bean_Dataset.csv`

3. **Adult Income** — UCI / Kaggle
   - ~48,000 samples, 14 features (6 categorical + 8 numerical), binary
   - Download from: https://archive.ics.uci.edu/ml/datasets/Adult

4. For each dataset, create a profiling notebook:
   - Class distribution (bar chart)
   - Feature distributions (histograms)
   - Correlation heatmap
   - Missing values report
   - Summary statistics table

### Artifact: `data/` folder with all CSVs, `notebooks/01_eda.ipynb`

### Checkpoint validation:
```bash
python -c "from src.preprocessing import load_all_datasets; load_all_datasets()"
```

---

## CHECKPOINT 2: Preprocessing Pipeline
**Goal:** Unified preprocessing that outputs clean numerical feature matrices + labels

### Steps:
1. Implement `src/preprocessing.py`:
   - `load_breast_cancer()` → returns X (569×30), y (569,), feature_names
   - `load_dry_bean()` → returns X (~13K×16), y (~13K,), feature_names
   - `load_adult_income()` → returns X (~48K×14), y (~48K,), feature_names
   - For Adult: one-hot encode categorical features, track original feature names

2. Common preprocessing function:
   ```python
   def preprocess(X, y, test_size=0.2, val_size=0.1, random_state=42):
       # Stratified split: train / val / test
       # StandardScaler fit on train only
       # Return X_train, X_val, X_test, y_train, y_val, y_test
   ```

3. Verify dimensions make sense after encoding (Adult will expand from 14 → ~100+ features)

### Artifact: `src/preprocessing.py`, each dataset returns (X_train, X_val, X_test, y_train, y_val, y_test)

### Checkpoint validation:
```bash
python -c "
from src.preprocessing import preprocess_dataset
for name in ['breast_cancer', 'dry_bean', 'adult_income']:
    data = preprocess_dataset(name)
    print(f'{name}: train={data[0].shape}, classes={len(set(data[3]))}')
"
```

---

## CHECKPOINT 3: T2I Method Implementation
**Goal:** Three working tabular-to-image transformation methods

### Step 3a: Naive Reshape (baseline)
- Take feature vector [x1, x2, ..., xn]
- Pad to next perfect square if needed (e.g., 30 → 36 = 6×6)
- Reshape to square grid: 6×6 → single-channel grayscale image
- Output: `images/naive/{dataset}/{sample_id}.png` (or tensor)
- **Don't save as PNG** — keep as tensors in memory for speed

### Step 3b: DeepInsight (via TINTOlib)
- Use `tintolib` library: `pip install tintolib`
- Pipeline: t-SNE on feature correlations → map features to 2D pixel coordinates
- Handle edge cases: features that collapse to same point
- Output: sparse-to-dense pixel intensity images
- **Key parameter:** image_size (try 32×32, 64×64)

### Step 3c: IGTD
- Also from TINTOlib or implement from scratch
- Algorithm: rank-based permutation to match feature distance rankings with pixel distance rankings
- Frobenius norm minimization via greedy iterative swapping
- Output: dense grayscale images

### For each method, create a generator class:
```python
class T2ITransformer:
    def __init__(self, method='naive', image_size=32):
        self.method = method
        self.image_size = image_size

    def fit(self, X_train):
        """Fit on training data only (for DeepInsight/IGTD coordinate mapping)"""

    def transform(self, X):
        """Transform feature vectors to image tensors"""
        # Returns: torch.Tensor of shape (N, 1, H, W)
```

### Artifact: `src/t2i/naive.py`, `src/t2i/deepinsight.py`, `src/t2i/igtd.py`

### Checkpoint validation:
```python
from src.t2i import T2ITransformer
import torch

# Test with breast cancer (smallest dataset)
X_train, _, _, _, _, _ = preprocess_dataset('breast_cancer')

for method in ['naive', 'deepinsight', 'igtd']:
    t = T2ITransformer(method=method, image_size=32)
    t.fit(X_train)
    images = t.transform(X_train)
    print(f'{method}: {images.shape}')  # Should be (569, 1, 32, 32)
```

---

## CHECKPOINT 4: CNN Model Implementation
**Goal:** Three working CNN architectures ready for training

### Step 4a: Shallow CNN (from scratch)
```
Input (1, 32, 32)
→ Conv2d(1, 32, 3, padding=1) → BN → ReLU → MaxPool
→ Conv2d(32, 64, 3, padding=1) → BN → ReLU → MaxPool
→ Conv2d(64, 128, 3, padding=1) → BN → ReLU → AdaptiveAvgPool(4)
→ Flatten → Linear(128*4*4, 256) → ReLU → Dropout(0.5) → Linear(256, num_classes)
```

### Step 4b: ResNet-18 Transfer Learning
- Load `torchvision.models.resnet18(pretrained=True)`
- Replace first conv: `Conv2d(1, 64, 7, ...)` → keeps 1 channel grayscale input
- OR convert 1ch → 3ch by repeating: `x.repeat(1, 3, 1, 1)`
- Replace final FC: `Linear(512, num_classes)`
- **Freeze** first N layers, **fine-tune** last layers
- Training strategy: low LR for backbone, higher LR for new layers

### Step 4c: ViT Transfer Learning
- Use `timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)`
- Adapt input: resize images to 224×224, convert 1ch → 3ch
- Or use `timm.create_model('vit_small_patch16_224', ...)` for smaller model
- Same freeze/fine-tune strategy as ResNet

### Common model interface:
```python
class BaseModel(nn.Module):
    def __init__(self, num_classes, input_channels=1, input_size=32):
        ...

    def forward(self, x):
        ...
```

### Artifact: `src/models/shallow_cnn.py`, `src/models/resnet_wrapper.py`, `src/models/vit_wrapper.py`

### Checkpoint validation:
```python
from src.models import ShallowCNN, ResNetWrapper, ViTWrapper

x = torch.randn(8, 1, 32, 32)  # batch of 8 grayscale images
for ModelClass in [ShallowCNN, ResNetWrapper, ViTWrapper]:
    model = ModelClass(num_classes=2)
    out = model(x)
    print(f'{ModelClass.__name__}: {out.shape}')  # (8, 2)
    params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {params:,}')
```

---

## CHECKPOINT 5: Training & Evaluation Pipeline
**Goal:** Single script that trains any model on any dataset and logs results

### `src/train.py`:
```python
def train_model(model, train_loader, val_loader, config):
    """
    config = {
        'epochs': 50,
        'lr': 1e-3,
        'optimizer': 'adam',
        'early_stopping_patience': 10,
        'device': 'cuda' / 'cpu'
    }
    Returns: trained model, training_history (loss, metrics per epoch)
    """
```

- Early stopping based on validation loss
- Save best model checkpoint
- Log training loss, val loss, val accuracy per epoch
- Return history dict for plotting

### `src/evaluate.py`:
```python
def evaluate_model(model, test_loader, num_classes):
    """
    Returns dict:
    {
        'accuracy': float,
        'precision': float,
        'recall': float,
        'f1': float,
        'roc_auc': float (binary) or 'roc_auc_ovr': float (multiclass),
        'pr_auc': float,
        'confusion_matrix': np.array,
        'classification_report': str
    }
    """
```

### Artifact: `src/train.py`, `src/evaluate.py`

### Checkpoint validation:
```bash
# Quick smoke test on breast cancer + naive + shallow CNN
python -c "
from src.train import train_model
from src.evaluate import evaluate_model
from src.preprocessing import preprocess_dataset
from src.t2i import T2ITransformer
from src.models import ShallowCNN

data = preprocess_dataset('breast_cancer')
t = T2ITransformer('naive', 32)
t.fit(data[0])
# Transform all splits...
# Train for 5 epochs as smoke test
# Evaluate and print metrics
"
```

---

## CHECKPOINT 6: Run All Experiments
**Goal:** Complete 27 CNN experiments + 9 tabular baselines, saved to results/

### `run_all.py`:
```python
experiments = [
    (dataset, t2i_method, cnn_arch)
    for dataset in ['breast_cancer', 'dry_bean', 'adult_income']
    for t2i_method in ['naive', 'deepinsight', 'igtd']
    for cnn_arch in ['shallow', 'resnet', 'vit']
]

for dataset, t2i, cnn in experiments:
    run_experiment(dataset, t2i, cnn)
    # Saves: results/{dataset}_{t2i}_{cnn}.json
```

### For each of the 27 experiments:
1. Load dataset -> preprocess
2. Fit T2I transformer on train split
3. Transform train/val/test to images
4. Create DataLoaders (batch_size=32 or 64)
5. Initialize CNN model
6. Train with early stopping
7. Evaluate on test set
8. Save metrics to `results/` JSON + model checkpoint

### Tabular baselines (9 experiments):
```python
baselines = [
    (dataset, model_type)
    for dataset in ['breast_cancer', 'dry_bean', 'adult_income']
    for model_type in ['rf', 'xgboost', 'mlp']
]
```
- RF/XGBoost: standard sklearn/xgboost fit/predict
- MLP: `sklearn.neural_network.MLPClassifier` or small PyTorch MLP
- Save metrics to `results/baseline_{dataset}_{model}.json`

### Expected runtime estimates:
| Dataset | T2I Method | CNN | Est. Time |
|---------|-----------|-----|-----------|
| Breast Cancer (569 samples) | Any | Shallow | ~2 min |
| Breast Cancer | Any | ResNet | ~5 min |
| Breast Cancer | Any | ViT | ~10 min |
| Dry Bean (13K samples) | Any | Shallow | ~10 min |
| Dry Bean | Any | ResNet | ~20 min |
| Dry Bean | Any | ViT | ~30 min |
| Adult Income (48K samples) | Any | Shallow | ~15 min |
| Adult Income | Any | ResNet | ~30 min |
| Adult Income | Any | ViT | ~45 min |

**Total estimated: ~4-6 hours on GPU, ~24-48 hours on CPU**

### Artifact: `results/all_experiments.csv`, saved model checkpoints

### Checkpoint validation:
```bash
# Verify all 36 results files exist
ls results/*.json | wc -l  # Should be 36 (27 CNN + 9 baselines)
python -c "
import pandas as pd
df = pd.read_csv('results/all_experiments.csv')
print(f'Experiments: {len(df)}')
print(df.pivot_table(index='dataset', columns='t2i_method', values='accuracy'))
"
```

---

## CHECKPOINT 7: Ablation Study
**Goal:** Prove spatial structure matters for CNN performance

### Take best combo from Checkpoint 6 (e.g., deepinsight + resnet + breast_cancer)

### Ablation 1: Pixel Shuffling
```python
# Train normally -> get accuracy = 87%
# Then on test set:
# 1. Shuffle pixel positions (fixed random seed) -> accuracy should drop
# 2. Reverse pixel order -> accuracy should drop
# 3. Transpose image (swap x/y) -> accuracy may stay similar
```

### Ablation 2: Feature Correlation Impact
```python
# 1. Full features -> baseline accuracy
# 2. Remove top 20% correlated features -> measure change
# 3. Only keep top 20% correlated features -> measure change
# 4. Randomly select features (same count) -> control
```

### Ablation 3: Feature Ordering
```python
# 1. Original ordering -> baseline
# 2. Sort by correlation with target -> does targeted ordering help?
# 3. Random ordering -> should degrade
# 4. Reverse ordering -> tests if specific order matters
```

### Artifact: `results/ablation_*.json`, `src/ablation.py`

### Checkpoint validation:
```bash
python src/ablation.py --dataset breast_cancer --t2i deepinsight --cnn resnet
# Should print comparison table showing accuracy drops when structure is destroyed
```

---

## CHECKPOINT 8: Visualization & Figures
**Goal:** Publication-ready figures for the seminar paper

### `src/visualize.py` generates:

1. **Main comparison heatmap** (most important figure)
   - Rows: T2I methods (naive, deepinsight, igtd)
   - Columns: CNN architectures (shallow, resnet, vit)
   - Cell color: accuracy
   - One heatmap per dataset -> 3 figures

2. **CNN vs. Tabular baselines bar chart**
   - Grouped bars: RF, XGBoost, MLP, best CNN per dataset
   - One chart per dataset -> 3 figures

3. **Training curves**
   - Loss vs. epoch for each experiment
   - Overlay: shallow CNN vs. ResNet vs. ViT -> 3 figures

4. **Confusion matrices**
   - Heatmap of confusion matrix for each experiment
   - Pick best/worst per dataset -> ~6 figures

5. **Ablation results bar chart**
   - Grouped bars: original vs. shuffled vs. reduced vs. reordered
   - Shows accuracy drop when spatial structure is destroyed

6. **Grad-CAM visualizations** (if ResNet experiment works)
   - Input image + Grad-CAM heatmap overlay
   - Shows which "pixel regions" the CNN attends to
   - 2-3 compelling examples

7. **Dataset profiling figures** (from Checkpoint 1 EDA)
   - Correlation heatmaps, class distributions

### Artifact: `results/figures/` folder with all PNGs, `src/visualize.py`

### Checkpoint validation:
```bash
python src/visualize.py
ls results/figures/*.png  # Should have 15-20 figures
```

---

## Implementation Order & Dependencies

```
CHECKPOINT 0 (Environment)
        |
        v
CHECKPOINT 1 (Datasets + EDA)
        |
        v
CHECKPOINT 2 (Preprocessing)
        |
        +----------------------+
        v                      v
CHECKPOINT 3 (T2I methods)   CHECKPOINT 4 (CNN models)
        |                      |
        +----------+-----------+
                   v
        CHECKPOINT 5 (Train + Eval pipeline)
                   |
                   v
        CHECKPOINT 6 (Run all 36 experiments)
                   |
        +----------+-----------+
        v                      v
CHECKPOINT 7              CHECKPOINT 8
(Ablation study)          (Visualizations)
        |                      |
        +----------+-----------+
                   v
          READY FOR PAPER WRITING
```

---

## Key Decisions to Make Early

1. **Image size:** 32x32 (fast) vs 64x64 (more detail)?
   - Recommendation: **32x32** for shallow CNN, **224x224** for ResNet/ViT (they expect this)
   - For ResNet/ViT: naive/IGTD images upsampled with bilinear interpolation

2. **1-channel vs 3-channel input:**
   - Shallow CNN: 1 channel (grayscale)
   - ResNet/ViT: convert to 3 channels by repeating: `x.repeat(1, 3, 1, 1)`
   - Alternative: some papers use multi-channel for different feature groups

3. **Batch size:** 32 for small datasets, 64 for large
4. **Early stopping patience:** 10 epochs
5. **Max epochs:** 50 (most will converge before then)
6. **Learning rates:**
   - Shallow CNN: 1e-3
   - ResNet/ViT backbone: 1e-4, new layers: 1e-3

---

## Quick Reference: Commands per Checkpoint

```bash
# Checkpoint 0
pip install -r requirements.txt

# Checkpoint 1
python -c "from src.preprocessing import download_datasets; download_datasets()"

# Checkpoint 2
python -c "from src.preprocessing impo
rt all_datasets; verify_all()"

# Checkpoint 3
python -c "from src.t2i import verify_all_transformers; verify_all_transformers()"

# Checkpoint 4
python -c "from src.models import verify_all_models; verify_all_models()"

# Checkpoint 5
python src/train.py --smoke-test  # 5-epoch quick run

# Checkpoint 6
python run_all.py  # Full experiment grid

# Checkpoint 7
python src/ablation.py  # Ablation experiments

# Checkpoint 8
python src/visualize.py  # Generate all figures
"""

---

# WORKFLOW UPDATE: Changes Made & Remaining Work

## Date: August 28, 2026

## Changes Made (Checkpoints 1-3 Complete)

### Checkpoint 0: Environment Setup ✓
- Git initialized, project structure created
- requirements.txt, .gitignore, run_all.py
- Commit: 328bfbf

### Checkpoint 1: Dataset Acquisition ✓
- Breast Cancer (sklearn), Dry Bean (UCI), Adult Income (UCI)
- preprocessing.py: load, encode, stratified split, StandardScaler
- EDA notebook with 10 profiling figures
- Commits: 151d35b, b2e64a8

### Checkpoint 2: Preprocessing Pipeline ✓
- Unified preprocess_dataset() returns dict with all splits
- Verified: all 3 datasets load correctly
- Part of commit: 151d35b

### Checkpoint 3: T2I Methods ✓
- naive.py: custom pad+reshape+bicubic resize
- deepinsight.py: TINTOlib wrapper (t-SNE manifold)
- igtd.py: TINTOlib wrapper (rank-based permutation)
- __init__.py: T2ITransformer unified interface + _load_tinto_images helper
- Commit: 079a363

### Checkpoint 3: Code Review & Bug Fixes ✓
5 bugs fixed, 5 structural concerns addressed:

| # | Issue | Fix | Commit |
|---|-------|-----|--------|
| 1 | Naive normalization data leakage | Train min/max in fit() | a760bbf |
| 2 | DeepInsight/IGTD redundant normalization | Removed, added clip | a760bbf |
| 3 | IGTD output range [0,255] mismatch | Divide by 255.0 | a760bbf |
| 4 | CSV order bug | Index-based loading | bf6e158 |
| 5 | IGTD redundant fit transform | Removed | bf6e158 |
| 6 | Naive 97%+ zeros | Documented as limitation | — |
| 7 | Class imbalance | Class weights + macro-F1 | 79e79c9 |
| 8 | 32x32 too small for 108 feat | PLANNED for CP4 | — |
| 9 | ResNet/ViT need 224x224 | PLANNED for CP4 | — |
| 10 | Small data overfitting | Weight decay + early stop + label smoothing | 79e79c9, 8b556ad |

### Literature-Based Improvements ✓
- Bicubic interpolation (concern 9): naive.py uses Image.BICUBIC
- Label smoothing (concern 10): CrossEntropyLoss(label_smoothing=0.1)
- Z-score normalization (concern 3): zscore_normalize() utility
- Alignment assertion (concern 4): FileNotFoundError in _load_tinto_images
- Commit: 8b556ad

---

## Remaining Work: Checkpoint 4 (CNN Models)

### Must Implement

1. **ShallowCNN** (src/models/shallow_cnn.py)
   - 3-layer CNN from scratch
   - Input: (1, 32, 32) grayscale
   - Architecture: Conv(1→32)→BN→ReLU→Pool → Conv(32→64)→BN→ReLU→Pool → Conv(64→128)→BN→ReLU→Pool → FC(128*4*4→256)→Dropout→FC(256→C)
   - Dropout(0.5) for regularization

2. **ResNetWrapper** (src/models/resnet_wrapper.py)
   - Load pretrained torchvision.models.resnet18
   - ADAPT INPUT: resize 32→224 with bicubic, repeat 1ch→3ch
   - Replace first conv: Conv2d(1, 64, 7) OR use repeat approach
   - Replace final FC: Linear(512, num_classes)
   - Freeze early layers, fine-tune later layers
   - Two learning rate groups: backbone (1e-4), head (1e-3)

3. **ViTWrapper** (src/models/vit_wrapper.py)
   - Use timm.create_model('vit_base_patch16_224', pretrained=True)
   - Same input adaptation as ResNet
   - Replace head: num_classes output
   - Same freeze/fine-tune strategy

### Must Handle

4. **Adaptive image sizing** (Concern 8)
   - Adult Income: 108 features → use 64x64 or 128x128 instead of 32x32
   - Or: use feature selection (Relief/mRMR) to reduce to ~50 features
   - Decision needed before running experiments

5. **Data augmentation** (Concern 10)
   - For small datasets (breast cancer: 398 samples)
   - Random horizontal/vertical flip
   - Random rotation (±15°)
   - Random noise (Gaussian, σ=0.01)
   - Apply AFTER T2I transformation (augment images, not features)

6. **run_all.py integration**
   - Connect preprocessing → T2I → CNN → evaluate pipeline
   - Save results to JSON with all metrics
   - Handle class weights computation per dataset

### Should Implement

7. **Gradient-weighted class activation mapping (Grad-CAM)**
   - Visualize which pixels CNN attends to
   - Compare across T2I methods
   - Shows if CNN learns meaningful spatial patterns

8. **Training curves logging**
   - Save loss/accuracy per epoch for plotting
   - matplotlib figures for paper

### Nice to Have

9. **Feature selection before T2I** (Concern 8)
   - Run mRMR/Relief to select top-K features
   - Reduces sparsity for high-dimensional datasets
   - Would help Adult Income significantly

10. **ArcFace/CosFace margin loss** (Concern 10)
    - Alternative to cross-entropy for small datasets
    - Projects features onto unit hypersphere
    - May improve breast cancer results

---

## Experiment Matrix (Final)

| Dataset | T2I Method | CNN | Notes |
|---------|-----------|-----|-------|
| Breast Cancer | naive | shallow | 398 train, 30 features |
| Breast Cancer | naive | resnet | Transfer learning |
| Breast Cancer | naive | vit | Transfer learning |
| Breast Cancer | deepinsight | shallow | t-SNE mapping |
| Breast Cancer | deepinsight | resnet | Transfer learning |
| Breast Cancer | deepinsight | vit | Transfer learning |
| Breast Cancer | igtd | shallow | Rank-based mapping |
| Breast Cancer | igtd | resnet | Transfer learning |
| Breast Cancer | igtd | vit | Transfer learning |
| Dry Bean | naive | shallow | 9527 train, 16 features |
| Dry Bean | naive | resnet | Transfer learning |
| Dry Bean | naive | vit | Transfer learning |
| Dry Bean | deepinsight | shallow | t-SNE mapping |
| Dry Bean | deepinsight | resnet | Transfer learning |
| Dry Bean | deepinsight | vit | Transfer learning |
| Dry Bean | igtd | shallow | Rank-based mapping |
| Dry Bean | igtd | resnet | Transfer learning |
| Dry Bean | igtd | vit | Transfer learning |
| Adult Income | naive | shallow | 34188 train, 108 features |
| Adult Income | naive | resnet | Transfer learning |
| Adult Income | naive | vit | Transfer learning |
| Adult Income | deepinsight | shallow | t-SNE mapping |
| Adult Income | deepinsight | resnet | Transfer learning |
| Adult Income | deepinsight | vit | Transfer learning |
| Adult Income | igtd | shallow | Rank-based mapping |
| Adult Income | igtd | resnet | Transfer learning |
| Adult Income | igtd | vit | Transfer learning |

Total: 27 CNN experiments + 9 baselines (RF, XGBoost, MLP) = 36 experiments

---

## Key Design Decisions (Documented)

1. **Why bicubic over nearest**: Produces smoother gradients, preserves edges better when upscaling sparse matrices. Nearest creates blocky artifacts. (Literature: Concern 9)

2. **Why label smoothing (0.1)**: Reduces overconfident predictions, improves generalization on small datasets. Especially important for breast cancer (398 samples). (Literature: Concern 10)

3. **Why class weights**: Inverse-frequency weighting prevents majority-class bias. Dry Bean 6.6:1 imbalance would otherwise inflate accuracy. (Concern 7)

4. **Why macro-F1 over accuracy**: Accuracy is misleading with class imbalance. Macro-F1 gives equal weight to all classes. (Concern 7)

5. **Why index-based loading**: TINTOlib CSV order doesn't match input order when data is shuffled. Loading by filename index ensures correct sample-to-image mapping. (Bug #4)

6. **Why /255.0 for IGTD**: IGTD outputs [0,255] via matplotlib colormap, DeepInsight outputs [0,1] via MinMaxScaler. Explicit /255.0 makes normalization deterministic rather than dependent on batch max. (Bug #3)

7. **Why clamp(0,1)**: Out-of-distribution test samples can have values outside training range. Clipping ensures consistent pixel range across all methods and splits. (Bug #2)
