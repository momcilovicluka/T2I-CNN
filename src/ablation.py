"""
Ablation study: prove spatial structure matters for CNN performance.

Usage:
    python src/ablation.py --dataset breast_cancer --t2i deepinsight --cnn resnet
"""

import argparse
import json


def pixel_shuffling_ablation(model, test_loader, image_size=32):
    """
    Evaluate effect of shuffling pixel positions.
    Should show significant accuracy drop if spatial structure matters.
    """
    raise NotImplementedError


def feature_correlation_ablation(X_train, y_train, X_test, y_test):
    """
    Evaluate effect of removing/reducing correlated features.
    """
    raise NotImplementedError


def feature_ordering_ablation(model, X, y, image_size=32):
    """
    Evaluate effect of changing feature ordering in the image.
    """
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--t2i', type=str, default='deepinsight')
    parser.add_argument('--cnn', type=str, default='resnet')
    args = parser.parse_args()
    print(f"Ablation study: {args.dataset} + {args.t2i} + {args.cnn}")
    raise NotImplementedError


if __name__ == '__main__':
    main()
