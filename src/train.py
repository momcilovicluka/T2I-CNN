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
import json
import time
from pathlib import Path


def compute_class_weights(y):
    """Compute inverse-frequency class weights for imbalanced datasets.
    
    Returns: torch.Tensor of shape (num_classes,) to use with CrossEntropyLoss.
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = total / (len(classes) * counts)
    # Convert to tensor indexed by class
    weight_tensor = torch.zeros(classes.max() + 1, dtype=torch.float32)
    for c, w in zip(classes, weights):
        weight_tensor[c] = w
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
    """Convert numpy arrays to DataLoaders."""
    # Add channel dim: (N, C, H, W) — C=1 for grayscale
    X_train_t = torch.tensor(X_train).unsqueeze(1).float()
    y_train_t = torch.tensor(y_train).long()
    X_val_t = torch.tensor(X_val).unsqueeze(1).float()
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
