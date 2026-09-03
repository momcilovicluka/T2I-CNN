# Checkpoint 3: T2I Method Implementation — Detailed Plan

## Key Discovery: TINTOlib saves images to disk, not tensors

TINTOlib API:
```python
model = IGTD(scale=[6,6], fea_dist_method='Pearson')
model.fit(dataframe_with_target_last)
model.transform(dataframe_with_target_last, output_folder)
# Output: PNG files in output_folder
```

**Implication:** Our wrapper must:
1. Convert numpy arrays → DataFrame (target col last)
2. Call TINTOlib fit/transform (saves PNGs to temp dir)
3. Load PNGs back as tensors
4. Return torch.Tensor (N, 1, H, W)

---

## Architecture Decision: Use TINTOlib for DeepInsight + IGTD

**Why TINTOlib over custom implementation:**
- Published, cited library (SoftwareX 2025, Information Fusion 2026)
- Validated against benchmarks
- Handles edge cases (collapsing features, normalization)
- We cite it in the paper — strengthens methodology
- Saves weeks of debugging custom implementations

**Naive Reshape: implement from scratch** (trivial — just pad + reshape)

---

## Step-by-step Implementation

### Step 1: Install TINTOlib

```bash
pip install TINTOlib
```

TINTOlib pulls in: scikit-learn, numpy, pandas, matplotlib, Pillow, tensorflow (for some methods), etc.

### Step 2: Implement `src/t2i/naive.py`

Algorithm:
1. `fit(X_train)`: Compute number of features → find next perfect square → determine grid size
2. `transform(X)`: For each sample:
   - Pad feature vector with zeros to next perfect square length
   - Reshape to (H, W) where H = W = ceil(sqrt(n_features))
   - Resize to target image_size x image_size using nearest interpolation
   - Return as tensor (N, 1, H, W)

```python
class NaiveReshape:
    def __init__(self, image_size=32):
        self.image_size = image_size
        self.grid_size = None  # determined in fit()

    def fit(self, X_train):
        n_features = X_train.shape[1]
        self.grid_size = int(np.ceil(np.sqrt(n_features)))
        self.padded_size = self.grid_size ** 2
        return self

    def transform(self, X):
        N = X.shape[0]
        # Pad each sample to perfect square
        padded = np.zeros((N, self.padded_size), dtype=np.float32)
        padded[:, :X.shape[1]] = X
        # Reshape to square grid
        images = padded.reshape(N, self.grid_size, self.grid_size)
        # Resize to target size using nearest neighbor
        # ... (use PIL or cv2)
        # Add channel dimension: (N, 1, H, W)
        return torch.tensor(images).unsqueeze(1).float()
```

### Step 3: Implement `src/t2i/deepinsight.py`

DeepInsight algorithm (from original paper + TINTOlib):
1. Compute pairwise feature correlation matrix
2. Apply t-SNE (or PCA) to map features to 2D coordinates
3. Create bounding box around 2D points
4. Map each feature to nearest pixel position
5. For each sample: place feature values at their pixel positions
6. Handle collisions (multiple features → same pixel): average or max

TINTOlib implementation:
```python
from TINTOlib.deepinsight import DeepInsight

class DeepInsightWrapper:
    def __init__(self, image_size=32):
        self.image_size = image_size
        self.model = DeepInsight(
            image_dim=[image_size, image_size],
            problem='classification',
            verbose=False,
            zoom=1,
            format='npy'  # Save as numpy instead of PNG for speed
        )
        self.temp_dir = None

    def fit(self, X_train, y_train):
        # Convert to DataFrame with target LAST
        df = pd.DataFrame(X_train)
        df['target'] = y_train
        # Fit TINTOlib
        self.model.fit(df)
        return self

    def transform(self, X, y):
        # Create temp dir
        self.temp_dir = tempfile.mkdtemp()
        df = pd.DataFrame(X)
        df['target'] = y
        self.model.transform(df, self.temp_dir)
        # Load generated images as tensors
        images = self._load_images()
        # Cleanup
        shutil.rmtree(self.temp_dir)
        return images

    def _load_images(self):
        # Load .npy or .png files from temp_dir
        # Stack into (N, 1, H, W) tensor
        ...
```

**Important:** TINTOlib needs `y` (labels) in the DataFrame. We must pass labels through.
**Edge case:** Our wrapper API currently only takes X. We need to modify the T2ITransformer
interface to also accept y in transform().

### Step 4: Implement `src/t2i/igtd.py`

IGTD algorithm:
1. Compute pairwise feature distance matrix (Pearson correlation by default)
2. Compute pairwise pixel distance matrix on the image grid
3. Rank features by distance, rank pixels by distance
4. Iteratively swap feature assignments to minimize Frobenius norm
   between rank matrices
5. Assign each feature to its final pixel position
6. For each sample: place feature values at assigned pixel positions

TINTOlib implementation:
```python
from TINTOlib.igtd import IGTD

class IGTDWrapper:
    def __init__(self, image_size=32):
        self.image_size = image_size
        # scale = [rows, cols] — must fit all features
        grid = int(np.ceil(np.sqrt(image_size**2)))  # or use image_size directly
        self.model = IGTD(
            scale=[image_size, image_size],
            fea_dist_method='Pearson',
            image_dist_method='Euclidean',
            error='squared',
            max_step=1000,
            val_step=50,
            verbose=False,
            format='npy',
            zoom=1,
        )
```

Same fit/transform pattern as DeepInsight.

### Step 5: Update `src/t2i/__init__.py`

Modify T2ITransformer to pass labels:

```python
class T2ITransformer:
    def __init__(self, method='naive', image_size=32):
        ...

    def fit(self, X_train, y_train=None):
        """Fit on training data."""
        self.transformer.fit(X_train, y_train)
        return self

    def transform(self, X, y=None):
        """Transform to images. y needed for TINTOlib methods."""
        return self.transformer.transform(X, y)
```

### Step 6: Handle image size mismatch

TINTOlib methods may produce images of different sizes than requested.
Our pipeline needs consistent sizes. Strategy:

- **Naive Reshape:** directly produces target image_size
- **DeepInsight/IGTD:** TINTOlib's `image_dim` parameter controls output size
  - Set `image_dim=[image_size, image_size]`
  - If output differs, resize with PIL to exact target

### Step 7: Handle train-only fit

Critical: fit() must only see training data (no data leakage).

- Naive reshape: fit() only computes grid size — no leakage risk
- DeepInsight: fit() learns feature→pixel mapping from training correlations
- IGTD: fit() optimizes feature placement from training distance matrix

Both DeepInsight and IGTD transform() must use the same mapping learned in fit().

### Step 8: Add `image_size` as parameter

Support both 32x32 (for shallow CNN) and 224x224 (for ResNet/ViT):
- DeepInsight/IGTD: set image_dim to target size
- Naive: reshape to 32x32, then upsample to 224x224 with bilinear

---

## Verification Plan

After implementation, run:

```python
from src.t2i import T2ITransformer
from src.preprocessing import preprocess_dataset

data = preprocess_dataset('breast_cancer')
X_train = data['X_train']
y_train = data['y_train']

for method in ['naive', 'deepinsight', 'igtd']:
    t = T2ITransformer(method=method, image_size=32)
    t.fit(X_train, y_train)
    images = t.transform(X_train, y_train)
    print(f'{method}: {images.shape}')
    # Expected: (398, 1, 32, 32)
    assert images.shape == (398, 1, 32, 32)
    assert images.min() >= 0  # pixel values in [0, 1] or [0, 255]
    assert images.max() <= 255  # if not normalized
    print(f'  pixel range: [{images.min():.1f}, {images.max():.1f}]')
```

---

## Potential Issues & Solutions

1. **TINTOlib requires TensorFlow** for some methods
   - DeepInsight and IGTD don't need TF (they use sklearn)
   - If TF is pulled in as dependency, it's ~500MB but harmless

2. **TINTOlib's output format**
   - Use `format='npy'` to save as numpy arrays (faster than PNG)
   - Load with `np.load()` instead of PIL

INTOlib saves images to disk, not tensors

2. **TINTOlib output format**
   - Use format='npy' to save as numpy arrays (faster than PNG)
   - Load with np.load() instead of PIL

3. **Memory for large datasets**
   - Adult Income: 34K samples x 108 features -> 34K images x 32x32 x 4 bytes = ~1.4GB
   - This is fine for RAM, but temp disk space needed for TINTOlib

4. **Feature count vs image size**
   - 108 features (Adult) needs at least 11x11 = 121 pixels
   - image_size=32 -> 32x32 = 1024 pixels - plenty of room
   - IGTD scale=[32,32] works for all our datasets

5. **TINTOlib expects target column LAST in DataFrame**
   - Our preprocessing returns X and y separately
   - Wrapper concatenates them: df = pd.DataFrame(X); df['target'] = y

6. **Reproducibility**
   - Set random_seed in TINTOlib models
   - Set torch manual seeds
   - Save fitted model state for reproducibility
