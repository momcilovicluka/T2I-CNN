# Google Colab Setup

## Cell 1: Clone repo and install dependencies

```python
# Clone the repo (replace with your GitHub URL)
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO

# Install dependencies
!pip install torch torchvision timm tintolib xgboost grad-cam scikit-learn pandas matplotlib seaborn
```

## Cell 2: Verify GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("WARNING: No GPU! Go to Runtime > Change runtime type > GPU")
```

## Cell 3: Run all experiments

```python
# Run all 69 experiments (~1-2 hours on Colab T4/A100)
!python run_all.py
```

## Cell 4: Generate all figures

```python
# Generate all 15+ figures
!python src/visualize.py

# List generated figures
!ls -la results/figures/
```

## Cell 5: Run ablation study

```python
# Run ablation experiments (~10 min)
!python src/ablation.py --all
```

## Cell 6: Download results

```python
# Option A: Download as zip
!zip -r results.zip results/
from google.colab import files
files.download('results.zip')

# Option B: Save to Google Drive (run this instead of Option A)
from google.colab import drive
drive.mount('/content/drive')
!cp -r results/ /content/drive/MyDrive/seminar2_results/
```

## Cell 7: Quick summary

```python
import pandas as pd
df = pd.read_csv('results/all_experiments.csv')
print(f"Total experiments: {len(df)}")
print(f"\nMacro-F1 by T2I method and dataset:")
print(df.pivot_table(index=['dataset', 't2i_method'], columns='cnn_arch', values='f1_macro').round(4).to_string())
```

## Expected runtime on Colab

| Dataset | Time (T4) | Time (A100) |
|---------|-----------|-------------|
| Breast Cancer | ~10 min | ~3 min |
| Dry Bean | ~30 min | ~10 min |
| Adult Income | ~1-2 hours | ~30 min |
| All 69 experiments | ~1.5-2 hours | ~45 min |
| Figures + Ablation | ~15 min | ~10 min |
| **Total** | **~2-2.5 hours** | **~1 hour** |
