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
            - early_stopping_patience (int, default 10)
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
    patience = config.get('early_stopping_patience', 10)

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
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

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

        if epochs_no_improve >= patience:
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
