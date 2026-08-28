"""
Master script to run all experiments.

Usage:
    python run_all.py                 # Run all 36 experiments
    python run_all.py --cnn-only      # Run only 27 CNN experiments
    python run_all.py --baselines     # Run only 9 tabular baselines
    python run_all.py --dataset breast_cancer  # Run for one dataset
"""

import itertools
import json
from pathlib import Path


DATASETS = ['breast_cancer', 'dry_bean', 'adult_income']
T2I_METHODS = ['naive', 'deepinsight', 'igtd']
CNN_ARCHITECTURES = ['shallow', 'resnet', 'vit']
BASELINE_MODELS = ['rf', 'xgboost', 'mlp']


def run_single_experiment(dataset, t2i_method, cnn_arch, output_dir='results'):
    """Run one CNN experiment: dataset -> T2I -> CNN -> evaluate."""
    raise NotImplementedError


def run_baseline(dataset, model_type, output_dir='results'):
    """Run one tabular baseline experiment."""
    raise NotImplementedError


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn-only', action='store_true')
    parser.add_argument('--baselines', action='store_true')
    parser.add_argument('--dataset', type=str, default=None)
    args = parser.parse_args()

    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    if not args.baselines:
        combos = list(itertools.product(DATASETS, T2I_METHODS, CNN_ARCHITECTURES))
        if args.dataset:
            combos = [(d, t, c) for d, t, c in combos if d == args.dataset]
        print(f"Running {len(combos)} CNN experiments...")
        for dataset, t2i, cnn in combos:
            print(f"  {dataset} + {t2i} + {cnn}")
            run_single_experiment(dataset, t2i, cnn)

    if not args.cnn_only:
        combos = list(itertools.product(DATASETS, BASELINE_MODELS))
        if args.dataset:
            combos = [(d, m) for d, m in combos if d == args.dataset]
        print(f"Running {len(combos)} baseline experiments...")
        for dataset, model in combos:
            print(f"  {dataset} + {model}")
            run_baseline(dataset, model)

    print("All experiments complete!")


if __name__ == '__main__':
    main()
