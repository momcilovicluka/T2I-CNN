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
# Default grid (2026-09-03): 36 CNN (4 T2I x 3 datasets x 3 archs)
# + 9 baselines = 45 experiments. ViT-Base/16 is DEFERRED (needs GPU,
# ~830 s/epoch on CPU); re-add only on a GPU runtime with:
#   !python run_all.py --archs shallow,resnet,resnet_scratch,vit
# CPU runtime: run in a terminal with nohup and re-attach, or keep this
# cell running and re-run to resume after disconnects (resume support).
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

## Expected runtime (2026-09-03, ViT deferred)

| Dataset | CPU (laptop) | CPU (Colab) | GPU (T4, if ViT re-added) |
|---------|-------------|-------------|--------------------------|
| Breast Cancer (12 CNN cells) | ~3 min | ~15 min | ~2 min |
| Dry Bean (12 CNN cells) | ~40-60 min | ~2-3 h | ~10 min |
| Adult Income (12 CNN cells) | ~2-3 h | ~5-8 h | ~30 min |
| 9 baselines | ~15 min | ~40 min | ~15 min |
| All 45 experiments | **~3.5-5 h** | **~8-12 h** | ~1 h |
| ViT only (12 cells, re-added) | days (not feasible) | days | ~2-3 h |
| Figures + Ablation | ~15 min | ~15 min | ~10 min |

Runtime estimates for dry_bean/adult are extrapolated from measured
breast_cancer CPU epochs (shallow ~0.1 s, resnet ~0.3 s,
resnet_scratch ~0.4 s per epoch at 32x32).
