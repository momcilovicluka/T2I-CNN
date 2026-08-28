"""
Training loop with early stopping.

Usage:
    python src/train.py --smoke-test    # Quick 5-epoch test
    python src/train.py --full          # Full training
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
import time
from pathlib import Path


def train_model(model, train_loader, val_loader, config):
    """
    Train a model with early stopping.

    Args:
        model: nn.Module
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: dict with keys:
            - epochs (int, default 50)
            - lr (float, default 1e-3)
            - optimizer (str, default 'adam')
            - early_stopping_patience (int, default 10)
            - device (str, 'cuda' or 'cpu')

    Returns:
        model: trained model (best checkpoint)
        history: dict with 'train_loss', 'val_loss', 'val_acc' per epoch
    """
    raise NotImplementedError


def save_checkpoint(model, path):
    """Save model weights."""
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path):
    """Load model weights."""
    model.load_state_dict(torch.load(path))
    return model
