"""
Dataset loading and preprocessing pipeline.

Handles:
- Loading Breast Cancer Wisconsin, Dry Bean, Adult Income datasets
- Categorical encoding (one-hot for Adult)
- Stratified train/val/test splits
- StandardScaler normalization (fit on train only)
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer as _load_bc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_breast_cancer():
    """Load Breast Cancer Wisconsin dataset from sklearn.

    Returns:
        X: np.ndarray (569, 30) - all numerical features
        y: np.ndarray (569,) - binary labels (0=malignant, 1=benign)
        feature_names: list of str
        class_names: list of str
    """
    data = _load_bc()
    X = data.data.astype(np.float32)
    y = data.target
    feature_names = list(data.feature_names)
    class_names = list(data.target_names)
    return X, y, feature_names, class_names


def load_dry_bean():
    """Load Dry Bean dataset.

    Downloads from UCI if not present locally.
    ~13,000 samples, 16 features (all numerical), 7 classes.

    Returns:
        X: np.ndarray - numerical features
        y: np.ndarray - integer labels
        feature_names: list of str
        class_names: list of str
    """
    csv_path = DATA_DIR / "dry_bean" / "Dry_Bean_Dataset.csv"

    if not csv_path.exists():
        _download_dry_bean(csv_path)

    df = pd.read_csv(csv_path)

    # Last column is 'Class'
    y_raw = df["Class"].values
    X = df.drop(columns=["Class"]).values.astype(np.float32)

    # Encode labels to integers
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    feature_names = [c for c in df.columns if c != "Class"]
    class_names = list(le.classes_)

    return X, y, feature_names, class_names


def _download_dry_bean(save_path):
    """Download Dry Bean dataset from UCI (xlsx -> csv conversion)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00602/DryBeanDataset.zip"
    print("Downloading Dry Bean dataset from UCI...")

    try:
        import urllib.request
        import zipfile
        import io

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        response = urllib.request.urlopen(req, timeout=60)
        z = zipfile.ZipFile(io.BytesIO(response.read()))

        # UCI zip contains .xlsx, not .csv - read with pandas and convert
        xlsx_name = "DryBeanDataset/Dry_Bean_Dataset.xlsx"
        print(f"Reading {xlsx_name} from zip...")
        with z.open(xlsx_name) as f:
            df = pd.read_excel(f, engine="openpyxl")

        df.to_csv(save_path, index=False)
        print(f"Saved {len(df)} rows to {save_path}")

    except Exception as e:
        print(f"Auto-download failed: {e}")
        print(f"Please download manually from:")
        print(f"  https://archive.ics.uci.edu/dataset/602/dry+bean+dataset")
        print(f"  and place Dry_Bean_Dataset.csv at: {save_path}")
        raise


def load_adult_income():
    """Load Adult Income dataset.

    Downloads from UCI if not present locally.
    ~45,000 samples after removing rows with '?' missing values,
    14 raw features (8 categorical + 6 numerical), binary.

    Returns:
        X: np.ndarray - one-hot encoded features
        y: np.ndarray - binary labels (0=<=50K, 1=>50K)
        feature_names: list of str (one-hot expanded)
        class_names: list of str
    """
    train_path = DATA_DIR / "adult_income" / "adult.data"
    test_path = DATA_DIR / "adult_income" / "adult.test"

    if not train_path.exists():
        _download_adult_income(train_path.parent)

    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    # Load train set (no header)
    # FIX (audit 2026-09-03): na_values must be '?' NOT ' ?'. With
    # skipinitialspace=True pandas strips the space before matching, so
    # ' ?' never matched and all rows with '?' were kept as literal '?'
    # one-hot categories instead of being dropped (dropna was a no-op;
    # load returned 48,842 rows incl. ~3,620 missing-value rows).
    df_train = pd.read_csv(train_path, header=None, names=columns,
                           na_values="?", skipinitialspace=True)

    # Load test set (has trailing period in labels, no header)
    df_test = pd.read_csv(test_path, header=None, names=columns,
                          na_values="?", skipinitialspace=True)

    # Drop rows with missing values
    df_train = df_train.dropna()
    df_test = df_test.dropna()

    # Combine for consistent encoding, then split back
    df_train["_split"] = "train"
    df_test["_split"] = "test"
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Encode target
    df["income"] = df["income"].str.replace(".", "", regex=False)
    df["income"] = (df["income"] == ">50K").astype(int)
    y = df["income"].values

    # One-hot encode categorical features
    categorical_cols = ["workclass", "education", "marital-status",
                        "occupation", "relationship", "race", "sex",
                        "native-country"]
    numerical_cols = ["age", "fnlwgt", "education-num",
                      "capital-gain", "capital-loss", "hours-per-week"]

    df_encoded = pd.get_dummies(df[numerical_cols + categorical_cols],
                                columns=categorical_cols, dtype=np.float32)
    X = df_encoded.values.astype(np.float32)
    feature_names = list(df_encoded.columns)

    # Split back using the _split column
    split_mask = df["_split"].values
    X_train_raw = X[split_mask == "train"]
    y_train_raw = y[split_mask == "train"]
    X_test_raw = X[split_mask == "test"]
    y_test_raw = y[split_mask == "test"]

    # Merge train+test, we'll re-split later with our own stratified split
    X = np.vstack([X_train_raw, X_test_raw])
    y = np.concatenate([y_train_raw, y_test_raw])

    class_names = ["<=50K", ">50K"]

    return X, y, feature_names, class_names


def _download_adult_income(save_path):
    """Download Adult Income dataset from UCI."""
    save_path.mkdir(parents=True, exist_ok=True)

    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
    files = {
        "adult.data": base_url + "adult.data",
        "adult.test": base_url + "adult.test",
    }

    import urllib.request

    for fname, url in files.items():
        dest = save_path / fname
        if dest.exists():
            continue
        print(f"Downloading {fname}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            response = urllib.request.urlopen(req, timeout=30)
            with open(dest, "wb") as f:
                f.write(response.read())
            print(f"  Saved to {dest}")
        except Exception as e:
            print(f"  Failed: {e}")
            print(f"  Please download manually from: {url}")
            print(f"  and place at: {dest}")
            raise


def preprocess(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Stratified train/val/test split + StandardScaler normalization.

    Args:
        X: np.ndarray of features
        y: np.ndarray of labels
        test_size: fraction for test set
        val_size: fraction for validation set (from remaining after test)
        random_state: random seed for reproducibility

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Second split: train vs val (adjust val_size relative to temp)
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, stratify=y_temp, random_state=random_state
    )

    # StandardScaler: fit on train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_dataset(name, test_size=0.2, val_size=0.1, random_state=42):
    """
    End-to-end: load + preprocess a dataset by name.

    Args:
        name: 'breast_cancer', 'dry_bean', or 'adult_income'

    Returns:
        dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
                        feature_names, num_classes, class_names
    """
    loaders = {
        "breast_cancer": load_breast_cancer,
        "dry_bean": load_dry_bean,
        "adult_income": load_adult_income,
    }

    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(loaders)}")

    X, y, feature_names, class_names = loaders[name]()

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess(
        X, y, test_size=test_size, val_size=val_size, random_state=random_state
    )

    num_classes = len(np.unique(y))

    print(f"[{name}]")
    print(f"  Features: {X_train.shape[1]} (after encoding)")
    print(f"  Classes: {num_classes} -> {class_names}")
    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": feature_names,
        "num_classes": num_classes,
        "class_names": class_names,
    }


def download_datasets():
    """Download all datasets to data/ directory."""
    print("Ensuring all datasets are available...")

    # Breast Cancer comes from sklearn, no download needed
    print("[breast_cancer] Available via sklearn (no download needed)")

    # Dry Bean
    dry_bean_csv = DATA_DIR / "dry_bean" / "Dry_Bean_Dataset.csv"
    if not dry_bean_csv.exists():
        _download_dry_bean(dry_bean_csv)
    else:
        print("[dry_bean] Already downloaded")

    # Adult Income
    adult_dir = DATA_DIR / "adult_income"
    if not (adult_dir / "adult.data").exists():
        _download_adult_income(adult_dir)
    else:
        print("[adult_income] Already downloaded")

    print("\nAll datasets ready!")


def verify_all():
    """Verify all datasets load and preprocess correctly."""
    for name in ["breast_cancer", "dry_bean", "adult_income"]:
        try:
            data = preprocess_dataset(name)
            assert data["X_train"].shape[0] > 0, "Empty training set"
            assert data["X_train"].ndim == 2, "X should be 2D"
            assert data["y_train"].ndim == 1, "y should be 1D"
            assert data["num_classes"] >= 2, "Need at least 2 classes"
            print(f"  PASS: {name}\n")
        except Exception as e:
            print(f"  FAIL: {name} - {e}\n")
            raise


if __name__ == "__main__":
    download_datasets()
    print()
    verify_all()
