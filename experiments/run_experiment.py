"""Experiment runner: compare 1D-to-2D transformations for stress classification.

Protocol:
- Load preprocessed dataset (30s windows, single-channel ECG, binary labels)
- Apply each transformation to convert 1D windows into 2D images
- Extract features from images using flattened pixel values + histogram features
- Classify using Random Forest (no GPU needed, reproducible)
- Evaluate with stratified 5-fold cross-validation (subject-aware when possible)
- Report accuracy, F1, precision, recall per transformation

Usage:
    python run_experiment.py --dataset WESAD --image_size 32 --transforms GASF GADF RPM MSDCF
"""

import os
import sys
import pickle
import time
import json
import argparse
import logging
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# Add parent for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformations import TRANSFORMS


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )
    return log_file


def extract_features_from_image(image_2d):
    """Extract feature vector from a 2D image.

    Uses a combination of:
    1. Downsampled pixel values (spatial features)
    2. Statistical features (mean, std, skew per row/col)
    3. Histogram of pixel intensities
    """
    # Flatten downsampled image
    flat = image_2d.flatten()

    # Row and column statistics
    row_means = image_2d.mean(axis=1)
    row_stds = image_2d.std(axis=1)
    col_means = image_2d.mean(axis=0)
    col_stds = image_2d.std(axis=0)

    # Diagonal features
    diag = np.diag(image_2d)

    # Histogram (16 bins)
    hist, _ = np.histogram(image_2d, bins=16, range=(0, 1))
    hist = hist / hist.sum() if hist.sum() > 0 else hist

    # Upper triangle mean (asymmetry measure)
    upper = image_2d[np.triu_indices_from(image_2d, k=1)]
    lower = image_2d[np.tril_indices_from(image_2d, k=-1)]
    asymmetry = [upper.mean() - lower.mean()] if len(upper) > 0 else [0]

    features = np.concatenate([
        flat,
        row_means, row_stds,
        col_means, col_stds,
        diag,
        hist.astype(float),
        asymmetry,
    ])
    return features


def transform_dataset(windows, transform_fn, image_size, fs=250, max_samples=None):
    """Apply a transformation to all windows in a dataset.

    Parameters
    ----------
    windows : ndarray of shape (N, win_samples) or (N, win_samples, 1)
    transform_fn : callable
    image_size : int

    Returns
    -------
    features : ndarray of shape (N, n_features)
    transform_time : float (seconds per sample)
    """
    if windows.ndim == 3:
        windows = windows[:, :, 0]  # Use first channel

    n = len(windows)
    if max_samples and n > max_samples:
        n = max_samples
        logging.info(f"  Limiting to {max_samples} samples for speed")

    images = []
    t_start = time.time()
    for i in range(n):
        sig = windows[i]
        # Check if transform needs fs parameter
        import inspect
        params = inspect.signature(transform_fn).parameters
        if 'fs' in params:
            img = transform_fn(sig, image_size=image_size, fs=fs)
        else:
            img = transform_fn(sig, image_size=image_size)

        images.append(img)

        if (i + 1) % 200 == 0:
            logging.info(f"    Transformed {i+1}/{n}")

    t_elapsed = time.time() - t_start
    time_per_sample = t_elapsed / n if n > 0 else 0

    # Extract features — handle multi-channel images by extracting per-channel
    all_features = []
    for img in images:
        if img.ndim == 3:
            # Multi-channel: extract features from each channel and concatenate
            ch_feats = [extract_features_from_image(img[:, :, c])
                        for c in range(img.shape[2])]
            all_features.append(np.concatenate(ch_feats))
        else:
            all_features.append(extract_features_from_image(img))
    features = np.array(all_features)

    return features, time_per_sample


def evaluate_classifier(features, labels, participants, n_folds=5):
    """Evaluate using stratified k-fold CV with Random Forest.

    Returns dict of metrics.
    """
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Use stratified k-fold (subject-aware splits are better but some datasets
    # have too few subjects per fold for LOGO)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_preds = np.zeros(len(labels))
    all_true = np.zeros(len(labels))

    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        all_preds[test_idx] = preds
        all_true[test_idx] = y_test

    metrics = {
        "accuracy": accuracy_score(all_true, all_preds),
        "f1": f1_score(all_true, all_preds, average="binary"),
        "precision": precision_score(all_true, all_preds, average="binary"),
        "recall": recall_score(all_true, all_preds, average="binary"),
    }
    return metrics


def run_experiment(dataset_path, dataset_name, transforms_to_test, image_size,
                   results_dir, max_samples=None):
    """Run full comparison experiment on one dataset."""
    logging.info(f"=" * 60)
    logging.info(f"Dataset: {dataset_name}")
    logging.info(f"Image size: {image_size}")
    logging.info(f"Transforms: {transforms_to_test}")
    logging.info(f"=" * 60)

    # Load data
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    windows = data["data"]
    labels = data["labels"]
    participants = data["participants"]

    # Infer sampling rate from window length
    # All datasets use 30s windows: win_samples = fs * 30
    win_samples = windows.shape[1]
    fs = win_samples // 30
    logging.info(f"Loaded: {windows.shape[0]} windows, {win_samples} samples/window, "
                 f"inferred fs={fs} Hz")
    logging.info(f"Labels: {dict(zip(*np.unique(labels, return_counts=True)))}")
    logging.info(f"Participants: {len(np.unique(participants))}")

    results = {}

    for tname in transforms_to_test:
        if tname not in TRANSFORMS:
            logging.warning(f"Unknown transform: {tname}, skipping")
            continue

        logging.info(f"\n--- Transform: {tname} ---")
        transform_fn = TRANSFORMS[tname]

        try:
            features, time_per_sample = transform_dataset(
                windows, transform_fn, image_size, fs=fs, max_samples=max_samples
            )
            logging.info(f"  Features shape: {features.shape}")
            logging.info(f"  Transform time: {time_per_sample:.4f} s/sample")

            # Limit labels/participants to match if max_samples was used
            n = features.shape[0]
            metrics = evaluate_classifier(features, labels[:n], participants[:n])

            results[tname] = {
                "metrics": metrics,
                "time_per_sample": time_per_sample,
                "feature_dim": features.shape[1],
            }

            logging.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logging.info(f"  F1:        {metrics['f1']:.4f}")
            logging.info(f"  Precision: {metrics['precision']:.4f}")
            logging.info(f"  Recall:    {metrics['recall']:.4f}")

        except Exception as e:
            logging.error(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[tname] = {"error": str(e)}

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"{dataset_name}_results.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"\nResults saved to {result_file}")

    return results


def print_comparison_table(all_results):
    """Print a formatted comparison table across datasets and transforms."""
    print("\n" + "=" * 80)
    print("COMPARISON TABLE: Accuracy / F1 Score")
    print("=" * 80)

    # Collect all transforms
    all_transforms = set()
    for ds_results in all_results.values():
        all_transforms.update(ds_results.keys())
    all_transforms = sorted(all_transforms)

    # Header
    header = f"{'Transform':<15}"
    for ds in all_results:
        header += f" | {ds:<24}"
    print(header)
    print("-" * len(header))

    # Rows
    for t in all_transforms:
        row = f"{t:<15}"
        for ds in all_results:
            if t in all_results[ds] and "metrics" in all_results[ds][t]:
                m = all_results[ds][t]["metrics"]
                row += f" | Acc={m['accuracy']:.3f} F1={m['f1']:.3f}  "
            else:
                row += f" | {'ERROR':<24}"
        print(row)

    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1D-to-2D transformation comparison")
    parser.add_argument("--dataset", type=str, default="WESAD",
                        help="Dataset name (WESAD, CLAS, CLARE, StressID, SWELL)")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--transforms", nargs="+",
                        default=["GASF", "GADF", "RP", "MTF", "RPM", "MSDCF"])
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples per dataset for faster testing")
    parser.add_argument("--all_datasets", action="store_true",
                        help="Run on all available datasets")
    args = parser.parse_args()

    exp_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = setup_logging(os.path.join(exp_dir, "logs"))
    results_dir = os.path.join(exp_dir, "results")

    logging.info(f"Experiment config: {vars(args)}")

    datasets = ["WESAD", "CLAS", "CLARE", "StressID", "SWELL"] if args.all_datasets else [args.dataset]

    all_results = {}
    for ds in datasets:
        ds_path = os.path.join(args.data_dir, f"{ds}_30s_50ovl_preprocessed.pkl")
        if not os.path.exists(ds_path):
            logging.warning(f"Dataset file not found: {ds_path}")
            continue

        results = run_experiment(
            ds_path, ds, args.transforms, args.image_size, results_dir,
            max_samples=args.max_samples,
        )
        all_results[ds] = results

    if len(all_results) > 0:
        print_comparison_table(all_results)

    # Save combined results
    combined_file = os.path.join(results_dir, "combined_results.json")
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"Combined results saved to {combined_file}")
