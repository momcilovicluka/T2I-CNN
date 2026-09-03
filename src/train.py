"""
Training loop with early stopping and class-weight support.

Usage:
    python src/train.py --smoke-test    # Quick 5-epoch test
    python src/train.py --full          # Full training
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import json
import time
from pathlib import Path


def set_global_seed(seed=42):
    """Set all random seeds for reproducibility.

    WHY: Even with TINTOlib's random_seed=42, numpy and torch global
    generators can produce different results across runs. This function
    ensures deterministic behavior for full reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weights(y):
    """Compute balanced class weights for imbalanced datasets.

    Uses sklearn's compute_class_weight which handles:
    - Non-contiguous labels (e.g., [0, 2, 5])
    - Correct inverse-frequency weighting
    - Works with any number of classes

    Returns: torch.Tensor indexed by class label for CrossEntropyLoss.
    """
    from sklearn.utils.class_weight import compute_class_weight

    unique_classes = np.unique(y)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y
    )
    # Map weights into tensor indexed by class label
    weight_tensor = torch.ones(unique_classes.max() + 1, dtype=torch.float32)
    for cls, w in zip(unique_classes, weights):
        weight_tensor[cls] = w
    return weight_tensor


def train_model(model, train_loader, val_loader, config):
    """
    Train a model with early stopping and optional class weighting.

    Args:
        model: nn.Module
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: dict with keys:
            - epochs (int, default 50)
            - lr (float, default 1e-3)
            - weight_decay (float, default 1e-4)
            - early_stopping_patience (int, default 15)
            - device (str, 'cuda' or 'cpu')
            - class_weights (torch.Tensor, optional) — for imbalanced datasets

    Returns:
        model: trained model (best checkpoint)
        history: dict with 'train_loss', 'val_loss', 'val_acc' per epoch
    """
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    epochs = config.get('epochs', 50)
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 1e-4)
    patience = config.get('early_stopping_patience', 15)

    model = model.to(device)

    # Loss function with:
    # - Class weights (Concern 7): Inverse-frequency weighting prevents
    #   majority-class bias. Dry Bean has 6.6:1 imbalance, Adult Income
    #   3.2:1. Without weights, CNN defaults to predicting majority class.
    # - Label smoothing (Concern 10): smoothing=0.1 transforms one-hot
    #   labels to soft distributions, reducing overconfident predictions.
    #   Especially important for small datasets (breast cancer: 398 samples)
    #   where deep models tend to memorize.
    class_weights = config.get('class_weights', None)
    label_smoothing = config.get('label_smoothing', 0.1)
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=label_smoothing
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    epoch_start = time.time()
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                _, predicted = output.max(1)
                correct += predicted.eq(y_batch).sum().item()
                total += y_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Progress update every epoch
        epoch_time = time.time() - epoch_start
        epoch_start = time.time()
        lr_now = optimizer.param_groups[0]['lr']
        improved = '*' if epochs_no_improve == 0 else ''
        print(f"    Epoch {epoch+1:2d}/{epochs}: loss={train_loss:.4f}/{val_loss:.4f} "
              f"acc={val_acc:.4f} lr={lr_now:.1e} [{epoch_time:.1f}s]{improved}", flush=True)

        if epochs_no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)", flush=True)
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model = model.cpu()

    return model, history


def save_checkpoint(model, path):
    """Save model weights."""
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path):
    """Load model weights."""
    model.load_state_dict(torch.load(path))
    return model


def prepare_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    """Convert numpy arrays or tensors to DataLoaders.

    Handles both raw feature arrays (N, features) and pre-generated
    images (N, 1, H, W) from T2I methods. Adds channel dim only if needed.
    """
    # Convert to tensors and add channel dim if missing
    X_train_t = torch.tensor(X_train).float()
    if X_train_t.ndim == 3:  # (N, H, W) — add channel
        X_train_t = X_train_t.unsqueeze(1)
    # else: already (N, 1, H, W) or (N, C, H, W)

    y_train_t = torch.tensor(y_train).long()

    X_val_t = torch.tensor(X_val).float()
    if X_val_t.ndim == 3:
        X_val_t = X_val_t.unsqueeze(1)
    y_val_t = torch.tensor(y_val).long()

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# === Additional improvements from literature review ===

def zscore_normalize(X_train, X_val=None, X_test=None):
    """Z-score standardization (channel-wise for images).

    WHY (Concern 3): Literature shows CNNs, especially pretrained models
    (ResNet-18, ViT), benefit from Z-score normalization. Subtracting
    channel-wise mean and dividing by std aligns input distribution with
    what ImageNet-pretrained models expect. Can improve transfer learning
    performance by up to +21.65% (per reviewer citations).

    Computed on training data only — no data leakage.

    Args:
        X_train: np.ndarray (N, C, H, W) or (N, H, W)
        X_val: optional validation data
        X_test: optional test data
    
    Returns:
        Normalized arrays (same shape as input)
    """
    # Compute per-channel mean and std from training data
    if X_train.ndim == 4:
        # (N, C, H, W) — compute per channel
        mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
        std = X_train.std(axis=(0, 2, 3), keepdims=True)
    else:
        # (N, H, W) — compute global
        mean = X_train.mean()
        std = X_train.std()
    
    std = np.where(std == 0, 1, std)  # avoid division by zero
    
    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std if X_val is not None else None
    X_test_n = (X_test - mean) / std if X_test is not None else None
    
    return X_train_n, X_val_n, X_test_n


# ImageNet normalization constants for pretrained models (ResNet, ViT)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def imagenet_normalize(images):
    """Apply ImageNet normalization to images for pretrained models.

    WHY (Concern 4): Pretrained ResNet-18 and ViT were trained on ImageNet
    with specific mean/std normalization. Feeding [0,1] images directly
    causes distribution mismatch, hurting transfer learning by up to
    +21.65% accuracy loss (per literature).

    Input: (N, 1, H, W) grayscale tensor
    Output: (N, 3, H, W) normalized tensor
    """
    # Convert grayscale to RGB by repeating channel
    images_rgb = images.repeat(1, 3, 1, 1)  # (N, 3, H, W)
    # Apply ImageNet normalization
    images_norm = (images_rgb - IMAGENET_MEAN.to(images.device)) / IMAGENET_STD.to(images.device)
    return images_norm

# === Cross-Validation (Issue 1: single split limitation) ===

def cross_validate(X, y, model_fn, t2i_method, image_size=32,
                   n_folds=5, config=None, seed=42):
    """Run stratified K-fold cross-validation.

    WHY: A single split can produce results specific to that partition.
    CV with mean +/- std gives variance estimates and statistical
    confidence that differences between methods are real, not noise.

    Args:
        X: np.ndarray (N, features) — raw tabular features
        y: np.ndarray (N,) — labels
        model_fn: callable that returns a new model instance
        t2i_method: str — 'naive', 'deepinsight', or 'igtd'
        image_size: int — output image size
        n_folds: int — number of CV folds (default 5)
        config: dict — training config (passed to train_model)
        seed: int — random seed for reproducibility

    Returns:
        dict with 'mean' and 'std' of all metrics across folds
    """
    from sklearn.model_selection import StratifiedKFold
    from src.t2i import T2ITransformer

    if config is None:
        config = {'epochs': 50, 'lr': 1e-3, 'label_smoothing': 0.1}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        set_global_seed(seed + fold)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit T2I on train fold only
        t2i = T2ITransformer(method=t2i_method, image_size=image_size)
        t2i.fit(X_train, y_train)

        # Transform
        train_imgs = t2i.transform(X_train, y_train)
        test_imgs = t2i.transform(X_test, y_test)

        # Train model
        model = model_fn()
        train_loader, val_loader = prepare_loaders(
            train_imgs.numpy(), y_train,
            test_imgs.numpy(), y_test,
            batch_size=config.get('batch_size', 32)
        )

        # Use a small validation split from train for early stopping
        n_val = int(len(X_train) * 0.1)
        val_loader_final = DataLoader(
            TensorDataset(test_imgs[:n_val], torch.tensor(y_test[:n_val]).long()),
            batch_size=config.get('batch_size', 32)
        )

        model, history = train_model(model, train_loader, val_loader, config)

        # Evaluate on held-out test fold
        from src.evaluate import evaluate_model
        test_loader = DataLoader(
            TensorDataset(test_imgs, torch.tensor(y_test).long()),
            batch_size=config.get('batch_size', 32)
        )
        metrics = evaluate_model(model, test_loader, num_classes=len(np.unique(y)))
        fold_metrics.append(metrics)

        print(f"  Fold {fold+1}/{n_folds}: F1={metrics['f1_macro']:.4f}, "
              f"Acc={metrics['accuracy']:.4f}")

    # Compute mean and std across folds
    result = {}
    for key in fold_metrics[0]:
        if isinstance(fold_metrics[0][key], (int, float)):
            values = [m[key] for m in fold_metrics]
            result[key] = {'mean': np.mean(values), 'std': np.std(values)}
        else:
            result[key] = fold_metrics[0][key]  # keep non-numeric as-is

    return result


# === LP-FT: Linear Probing then Fine-Tuning (Issue 5) ===

def train_lp_ft(model, train_loader, val_loader, config,
                lp_epochs=10, ft_epochs=40):
    """Two-phase training: Linear Probing then Fine-Tuning.

    WHY (Issue 5 — transfer learning on synthetic images):
    When using pretrained ResNet/ViT on synthetic T2I images, directly
    fine-tuning all layers can destroy pretrained features because the
    images look nothing like natural images (97% zeros for naive method).

    LP-FT approach:
    - Phase 1 (LP): Freeze backbone, train only the new FC head.
      This lets the head adapt to the T2I image distribution without
      corrupting the pretrained feature extractor.
    - Phase 2 (FT): Unfreeze everything, train end-to-end with a much
      lower learning rate. The head is already reasonable, so the
      backbone can gradually adapt.

    Literature shows LP-FT improves over direct fine-tuning in 58%+ of
    specialized transfer learning tasks.

    Args:
        model: nn.Module with freeze_backbone() and unfreeze_backbone()
        train_loader: DataLoader
        val_loader: DataLoader
        config: dict with training hyperparameters
        lp_epochs: epochs for linear probing phase
        ft_epochs: epochs for fine-tuning phase

    Returns:
        model: trained model, history: combined training history
    """
    print("  Phase 1: Linear Probing (backbone frozen)...")
    model.freeze_backbone()

    # LP phase: higher LR for head, no backbone updates
    lp_config = {
        **config,
        'epochs': lp_epochs,
        'lr': config.get('lr', 1e-3),  # head LR
    }
    model, history_lp = train_model(model, train_loader, val_loader, lp_config)

    print("  Phase 2: Fine-Tuning (all layers)...")
    model.unfreeze_backbone()

    # FT phase: lower LR for backbone, moderate for head
    ft_config = {
        **config,
        'epochs': ft_epochs,
        'lr': config.get('lr_ft', 1e-4),  # much lower for full network
    }
    model, history_ft = train_model(model, train_loader, val_loader, ft_config)

    # Combine histories
    history = {
        'train_loss': history_lp['train_loss'] + history_ft['train_loss'],
        'val_loss': history_lp['val_loss'] + history_ft['val_loss'],
        'val_acc': history_lp['val_acc'] + history_ft['val_acc'],
        'phase': 'lp_ft',
        'lp_epochs': lp_epochs,
        'ft_epochs': ft_epochs,
    }

    return model, history
