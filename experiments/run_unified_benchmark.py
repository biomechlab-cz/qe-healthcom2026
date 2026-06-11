"""
Unified benchmark: 3 baseline transforms (GASF, Spectrogram, RPM) vs 3 novel transforms
(ASPAG, MSGK, PSSAM) across all preprocessed datasets.

PCA(200) + RandomForest(500, balanced) with 5-fold stratified CV.
Paired t-tests: each novel vs each baseline.
Results saved to results/unified_benchmark.json.
"""

import os
import sys
import glob
import pickle
import re
import time
import json
import logging
from datetime import datetime

import numpy as np
from scipy.signal import stft
from scipy.ndimage import zoom
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from novel_transforms import NOVEL_TRANSFORMS

IMAGE_SIZE = 64
N_FOLDS = 5
PCA_COMPONENTS = 200
RF_ESTIMATORS = 500


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging():
    os.makedirs(os.path.join(os.path.dirname(__file__), "logs"), exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(os.path.dirname(__file__), "logs", f"unified_benchmark_{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )
    logging.info(f"Log file: {log_path}")


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_win_sec(filename):
    """Extract window length in seconds from a filename like DATASET_30s_50ovl_preprocessed.pkl."""
    basename = os.path.basename(filename)
    match = re.search(r"_(\d+)s_", basename)
    if match:
        return int(match.group(1))
    return 30  # fallback


# ---------------------------------------------------------------------------
# Baseline transforms
# ---------------------------------------------------------------------------

def _paa(signal, target_len):
    """Piecewise Aggregate Approximation."""
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    if n == target_len:
        return signal.copy()
    if n < target_len:
        return np.interp(np.linspace(0, n - 1, target_len), np.arange(n), signal)
    return np.array(
        [signal[int(i * n / target_len): int((i + 1) * n / target_len)].mean()
         for i in range(target_len)]
    )


def transform_gasf(signal, image_size=IMAGE_SIZE, fs=None):
    """Gramian Angular Summation Field via pyts."""
    from pyts.image import GramianAngularField
    sig = np.asarray(signal, dtype=float).reshape(1, -1)
    gaf = GramianAngularField(image_size=image_size, method="summation")
    return gaf.fit_transform(sig).squeeze()


def transform_spectrogram(signal, image_size=IMAGE_SIZE, fs=250):
    """Short-time Fourier transform spectrogram, log-scaled and zoomed to image_size x image_size."""
    signal = np.asarray(signal, dtype=float)
    nperseg = max(16, min(256, len(signal) // 8))
    _, _, Zxx = stft(signal, fs=fs, nperseg=nperseg)
    S = np.abs(Zxx)
    S = np.log1p(S)
    # Zoom to image_size x image_size
    zoom_y = image_size / S.shape[0]
    zoom_x = image_size / S.shape[1]
    S = zoom(S, (zoom_y, zoom_x))
    mn, mx = S.min(), S.max()
    if mx > mn:
        S = (S - mn) / (mx - mn)
    return S.astype(float)


def transform_rpm(signal, image_size=IMAGE_SIZE, fs=None):
    """Recurrence Plot Matrix: PAA reduce, outer pairwise subtraction, normalize to [0,1]."""
    signal = np.asarray(signal, dtype=float)
    reduced = _paa(signal, image_size)
    M = np.abs(reduced[:, np.newaxis] - reduced[np.newaxis, :])
    mn, mx = M.min(), M.max()
    if mx > mn:
        M = (M - mn) / (mx - mn)
    return M.astype(float)


BASELINE_TRANSFORMS = {
    "GASF": transform_gasf,
    "Spectrogram": transform_spectrogram,
    "RPM": transform_rpm,
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(image_2d):
    """Flatten + row/col stats + diagonal + histogram."""
    img = np.asarray(image_2d, dtype=float)
    flat = img.flatten()
    row_means = img.mean(axis=1)
    row_stds = img.std(axis=1)
    col_means = img.mean(axis=0)
    col_stds = img.std(axis=0)
    diag = np.diag(img)
    hist, _ = np.histogram(img, bins=16, range=(0, 1))
    hist = hist / (hist.sum() + 1e-10)
    return np.concatenate([flat, row_means, row_stds, col_means, col_stds, diag, hist.astype(float)])


def transform_all(windows, transform_fn, fs, transform_name, accepts_fs):
    """Apply transform_fn to every window (N, win_samples) and extract features.

    Returns (feature_matrix, time_per_sample).
    """
    n = len(windows)
    features_list = []
    t0 = time.time()
    for i, sig in enumerate(windows):
        try:
            if accepts_fs:
                img = transform_fn(sig, image_size=IMAGE_SIZE, fs=fs)
            else:
                img = transform_fn(sig, image_size=IMAGE_SIZE)
        except Exception as exc:
            logging.warning(f"    Sample {i} failed ({transform_name}): {exc}. Using zeros.")
            img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=float)
        features_list.append(extract_features(img))
        if (i + 1) % 500 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            logging.info(f"    {transform_name}: {i+1}/{n} samples ({elapsed:.1f}s elapsed)")
    tps = (time.time() - t0) / max(n, 1)
    return np.array(features_list), tps


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def evaluate_per_fold(features, labels):
    """PCA(200) + RF(500, balanced), 5-fold stratified CV.

    Returns (mean_acc, mean_f1, fold_accs, fold_f1s).
    """
    n_components = min(PCA_COMPONENTS, features.shape[0] - 1, features.shape[1])
    pipe = Pipeline([
        ("pca", PCA(n_components=n_components, random_state=42)),
        ("clf", RandomForestClassifier(
            n_estimators=RF_ESTIMATORS,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )),
    ])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_accs, fold_f1s = [], []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        logging.info(f"      Fold {fold_i+1}/{N_FOLDS}: acc={acc:.4f} f1={f1:.4f}")

    return np.mean(fold_accs), np.mean(fold_f1s), np.array(fold_accs), np.array(fold_f1s)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def significance_label(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return "ns"


def run_paired_ttests(fold_data, novel_names, baseline_names):
    """Paired t-tests: each novel vs each baseline for accuracy and F1."""
    results = {}
    for novel in novel_names:
        if novel not in fold_data:
            continue
        results[novel] = {}
        for baseline in baseline_names:
            if baseline not in fold_data:
                continue
            n_accs = fold_data[novel]["acc"]
            b_accs = fold_data[baseline]["acc"]
            n_f1s = fold_data[novel]["f1"]
            b_f1s = fold_data[baseline]["f1"]
            _, p_acc = ttest_rel(n_accs, b_accs)
            _, p_f1 = ttest_rel(n_f1s, b_f1s)
            results[novel][baseline] = {
                "acc_diff": float(n_accs.mean() - b_accs.mean()),
                "acc_p": float(p_acc),
                "acc_sig": significance_label(p_acc),
                "f1_diff": float(n_f1s.mean() - b_f1s.mean()),
                "f1_p": float(p_f1),
                "f1_sig": significance_label(p_f1),
            }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_logging()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data", "processed")
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Discover preprocessed files
    pattern = os.path.join(data_dir, "*_preprocessed.pkl")
    pkl_files = sorted(glob.glob(pattern))
    if not pkl_files:
        logging.error(f"No preprocessed pkl files found matching: {pattern}")
        sys.exit(1)
    logging.info(f"Found {len(pkl_files)} preprocessed dataset(s):")
    for p in pkl_files:
        logging.info(f"  {os.path.basename(p)}")

    # Determine which transforms accept fs
    novel_fns = NOVEL_TRANSFORMS  # dict: name -> fn
    import inspect
    all_transforms = {**BASELINE_TRANSFORMS, **novel_fns}
    accepts_fs_map = {
        name: "fs" in inspect.signature(fn).parameters
        for name, fn in all_transforms.items()
    }

    baseline_names = list(BASELINE_TRANSFORMS.keys())   # ["GASF", "Spectrogram", "RPM"]
    novel_names = list(novel_fns.keys())                 # ["ASPAG", "MSGK", "PSSAM"]
    transform_order = baseline_names + novel_names

    all_results = {}
    start_total = time.time()

    for pkl_path in pkl_files:
        ds_name = os.path.basename(pkl_path).replace("_preprocessed.pkl", "")
        logging.info(f"\n{'='*70}")
        logging.info(f"Dataset: {ds_name}")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        windows_raw = data["data"]   # (N, win_samples, C)
        labels = np.array(data["labels"])

        # Use first channel only
        if windows_raw.ndim == 3:
            windows = windows_raw[:, :, 0]
        else:
            windows = windows_raw

        N, win_samples = windows.shape
        win_sec = parse_win_sec(pkl_path)
        fs = win_samples // win_sec
        n_classes = len(np.unique(labels))

        logging.info(f"  Samples: {N} | win_samples: {win_samples} | win_sec: {win_sec}s | fs: {fs} Hz")
        logging.info(f"  Classes: {dict(zip(*np.unique(labels, return_counts=True)))}")
        logging.info(f"  Channels in file: {windows_raw.shape[2] if windows_raw.ndim == 3 else 1} (using first)")

        ds_result = {"metadata": {"n_samples": N, "win_samples": win_samples, "win_sec": win_sec, "fs": fs, "n_classes": n_classes}}
        fold_data = {}

        for tname in transform_order:
            fn = all_transforms[tname]
            logging.info(f"\n  --- Transform: {tname} ---")
            t_start = time.time()
            try:
                features, tps = transform_all(windows, fn, fs, tname, accepts_fs_map[tname])
                logging.info(f"  Feature matrix: {features.shape} | {tps:.4f} s/sample")

                mean_acc, mean_f1, fold_accs, fold_f1s = evaluate_per_fold(features, labels)

                ds_result[tname] = {
                    "mean_acc": float(mean_acc),
                    "mean_f1": float(mean_f1),
                    "fold_accs": fold_accs.tolist(),
                    "fold_f1s": fold_f1s.tolist(),
                    "time_per_sample_s": float(tps),
                    "feature_dim": features.shape[1],
                }
                fold_data[tname] = {"acc": fold_accs, "f1": fold_f1s}

                logging.info(f"  RESULT {tname}: acc={mean_acc:.4f} (std={fold_accs.std():.4f})  f1={mean_f1:.4f} (std={fold_f1s.std():.4f})")

            except Exception as exc:
                import traceback
                logging.error(f"  FAILED {tname}: {exc}")
                traceback.print_exc()
                ds_result[tname] = {"error": str(exc)}

            logging.info(f"  Wall time for {tname}: {time.time() - t_start:.1f}s")

        # Paired t-tests
        logging.info(f"\n  --- Paired t-tests (novel vs baseline) ---")
        ttests = run_paired_ttests(fold_data, novel_names, baseline_names)
        ds_result["ttest"] = ttests
        for novel, comparisons in ttests.items():
            for baseline, stats in comparisons.items():
                logging.info(
                    f"  {novel} vs {baseline}: "
                    f"acc diff={stats['acc_diff']:+.4f} (p={stats['acc_p']:.4f} {stats['acc_sig']})  "
                    f"f1 diff={stats['f1_diff']:+.4f} (p={stats['f1_p']:.4f} {stats['f1_sig']})"
                )

        all_results[ds_name] = ds_result

    total_time = time.time() - start_total
    logging.info(f"\nTotal runtime: {total_time/60:.1f} min")

    # -----------------------------------------------------------------------
    # Print comprehensive summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("UNIFIED BENCHMARK — PCA(200) + RF(500 balanced) — 5-fold stratified CV — image_size=64")
    print("=" * 110)

    ds_names = list(all_results.keys())
    col_w = 24

    # Header
    header = f"{'Transform':<14}"
    for ds in ds_names:
        short = ds[:col_w - 2]
        header += f" | {short:^{col_w}}"
    print(header)
    print("-" * (14 + (col_w + 3) * len(ds_names)))

    for tname in transform_order:
        row = f"{tname:<14}"
        for ds in ds_names:
            if tname in all_results[ds] and "mean_acc" in all_results[ds][tname]:
                m = all_results[ds][tname]
                cell = f"Acc={m['mean_acc']:.3f} F1={m['mean_f1']:.3f}"
            elif tname in all_results[ds] and "error" in all_results[ds][tname]:
                cell = "ERROR"
            else:
                cell = "---"
            row += f" | {cell:^{col_w}}"
        print(row)

    print("=" * (14 + (col_w + 3) * len(ds_names)))

    # t-test summary
    print("\nPaired t-test summary (novel vs baseline, per dataset):")
    print("-" * 100)
    for ds in ds_names:
        if "ttest" not in all_results[ds]:
            continue
        print(f"\n  {ds}:")
        for novel, comparisons in all_results[ds]["ttest"].items():
            for baseline, stats in comparisons.items():
                print(
                    f"    {novel} vs {baseline:12s}: "
                    f"acc {stats['acc_diff']:+.4f} ({stats['acc_sig']})  "
                    f"f1 {stats['f1_diff']:+.4f} ({stats['f1_sig']})"
                )

    print(f"\nTotal runtime: {total_time/60:.1f} min")

    # -----------------------------------------------------------------------
    # Save JSON (convert numpy arrays to lists for serialization)
    # -----------------------------------------------------------------------
    out_path = os.path.join(results_dir, "unified_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"Results saved to {out_path}")
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
