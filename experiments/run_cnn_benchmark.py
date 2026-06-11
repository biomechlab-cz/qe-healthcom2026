"""
CNN-based benchmark: 3 baseline transforms (GASF, Spectrogram, RPM) vs 3 novel transforms
(ASPAG, MSGK, PSSAM) across all preprocessed datasets.

SimpleCNN with 5-fold stratified CV, early stopping, class-weighted CrossEntropyLoss.
Results saved to results/cnn_benchmark.json.
"""

import os
import sys
import glob
import pickle
import re
import time
import json
import inspect
from datetime import datetime

import numpy as np
from scipy.signal import stft as scipy_stft
from scipy.ndimage import zoom as scipy_zoom
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Ensure transforms from same directory are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from novel_transforms import transform_aspag, transform_msgk, transform_pssam
from pyts.image import GramianAngularField

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}", flush=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_SIZE = 64
N_FOLDS = 5
BATCH_SIZE = 128
EPOCHS = 50
EARLY_STOP_PATIENCE = 7
LR = 1e-3
WEIGHT_DECAY = 1e-4


# ---------------------------------------------------------------------------
# CNN Model
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    """
    Input:  (batch, 1, 64, 64)
    Block1: Conv2d(1,  32, 3, pad=1) -> BN -> ReLU -> MaxPool(2)        -> 32x32x32
    Block2: Conv2d(32, 64, 3, pad=1) -> BN -> ReLU -> MaxPool(2)        -> 64x16x16
    Block3: Conv2d(64,128, 3, pad=1) -> BN -> ReLU -> AdaptiveAvgPool(4)-> 128x4x4
    Flatten -> 2048
    FC:     Linear(2048,128) -> ReLU -> Dropout(0.3) -> Linear(128, n_classes)
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class ImageDataset(Dataset):
    """Wraps pre-computed image arrays (N, 64, 64) and integer labels."""

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        # images: (N, H, W) float32 already in [0,1]
        self.images = torch.from_numpy(images[:, np.newaxis, :, :]).float()  # (N,1,H,W)
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Transform implementations
# ---------------------------------------------------------------------------

def _paa(signal: np.ndarray, target_len: int) -> np.ndarray:
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


def _normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize a 2D array to [0, 1]."""
    mn, mx = img.min(), img.max()
    if mx > mn:
        return (img - mn) / (mx - mn)
    return np.zeros_like(img)


def transform_gasf(signal: np.ndarray, image_size: int = IMAGE_SIZE, fs: float = None) -> np.ndarray:
    """Gramian Angular Summation Field via pyts."""
    sig = np.asarray(signal, dtype=float).reshape(1, -1)
    gaf = GramianAngularField(image_size=image_size, method="summation")
    img = gaf.fit_transform(sig).squeeze()
    return _normalize_image(img)


def transform_spectrogram(signal: np.ndarray, image_size: int = IMAGE_SIZE, fs: float = 250) -> np.ndarray:
    """STFT spectrogram — log-scaled and zoomed to image_size x image_size."""
    signal = np.asarray(signal, dtype=float)
    nperseg = max(16, min(256, len(signal) // 8))
    _, _, Zxx = scipy_stft(signal, fs=fs, nperseg=nperseg)
    S = np.abs(Zxx)
    S = np.log1p(S)
    zoom_y = image_size / S.shape[0]
    zoom_x = image_size / S.shape[1]
    S = scipy_zoom(S, (zoom_y, zoom_x))
    return _normalize_image(S)


def transform_rpm(signal: np.ndarray, image_size: int = IMAGE_SIZE, fs: float = None) -> np.ndarray:
    """Relative Position Matrix: PAA to 64, pairwise diff, normalize."""
    signal = np.asarray(signal, dtype=float)
    reduced = _paa(signal, image_size)
    M = reduced[:, np.newaxis] - reduced[np.newaxis, :]
    return _normalize_image(M)


def _transform_aspag(signal: np.ndarray, image_size: int = IMAGE_SIZE, fs: float = 250) -> np.ndarray:
    return _normalize_image(transform_aspag(signal, image_size=image_size, fs=fs))


def _transform_msgk(signal: np.ndarray, image_size: int = IMAGE_SIZE, fs: float = None) -> np.ndarray:
    return _normalize_image(transform_msgk(signal, image_size=image_size))


def _transform_pssam(signal: np.ndarray, image_size: int = IMAGE_SIZE, fs: float = None) -> np.ndarray:
    return _normalize_image(transform_pssam(signal, image_size=image_size))


# Map transform name -> (function, accepts_fs)
ALL_TRANSFORMS = {
    "GASF":        (transform_gasf,        False),
    "Spectrogram": (transform_spectrogram, True),
    "RPM":         (transform_rpm,         False),
    "ASPAG":       (_transform_aspag,      True),
    "MSGK":        (_transform_msgk,       False),
    "PSSAM":       (_transform_pssam,      False),
}

TRANSFORM_ORDER = ["GASF", "Spectrogram", "RPM", "ASPAG", "MSGK", "PSSAM"]
BASELINE_NAMES  = ["GASF", "Spectrogram", "RPM"]
NOVEL_NAMES     = ["ASPAG", "MSGK", "PSSAM"]


# ---------------------------------------------------------------------------
# Pre-compute transform images for a dataset
# ---------------------------------------------------------------------------

def precompute_images(windows: np.ndarray, transform_fn, accepts_fs: bool, fs: float,
                      transform_name: str) -> np.ndarray:
    """Apply transform to all windows; returns float32 array (N, 64, 64) in [0,1]."""
    N = len(windows)
    images = np.zeros((N, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    t0 = time.time()
    for i, sig in enumerate(windows):
        try:
            if accepts_fs:
                img = transform_fn(sig, image_size=IMAGE_SIZE, fs=fs)
            else:
                img = transform_fn(sig, image_size=IMAGE_SIZE)
            images[i] = img.astype(np.float32)
        except Exception as exc:
            print(f"    [WARN] {transform_name} sample {i} failed: {exc}. Using zeros.")
        if (i + 1) % 1000 == 0 or (i + 1) == N:
            elapsed = time.time() - t0
            print(f"    {transform_name}: {i+1}/{N} images ({elapsed:.1f}s)")
    total = time.time() - t0
    print(f"    {transform_name}: done in {total:.1f}s ({total/max(N,1):.4f} s/sample)")
    return images


# ---------------------------------------------------------------------------
# Class weights helper
# ---------------------------------------------------------------------------

def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Returns normalized inverse-frequency weights as a 1-D tensor."""
    classes, counts = np.unique(labels, return_counts=True)
    n_classes = int(classes.max()) + 1
    weights = np.zeros(n_classes, dtype=np.float32)
    for c, cnt in zip(classes, counts):
        weights[int(c)] = 1.0 / cnt
    weights /= weights.sum()   # normalize
    return torch.from_numpy(weights)


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(y)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_true = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            loss = criterion(out, y)
            running_loss += loss.item() * len(y)
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_true.append(y.cpu().numpy())
    avg_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_true  = np.concatenate(all_true)
    acc = accuracy_score(all_true, all_preds)
    f1  = f1_score(all_true, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1, all_true, all_preds


def train_fold(images: np.ndarray, labels: np.ndarray,
               train_idx: np.ndarray, val_idx: np.ndarray,
               n_classes: int):
    """Train SimpleCNN on one fold, return (val_acc, val_f1, val_true, val_preds)."""
    train_ds = ImageDataset(images[train_idx], labels[train_idx])
    val_ds   = ImageDataset(images[val_idx],   labels[val_idx])

    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)

    # Class weights from training labels
    class_weights = compute_class_weights(labels[train_idx]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = SimpleCNN(n_classes=n_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion)

        if epoch % 5 == 0 or epoch == 1:
            print(f"      Epoch {epoch:2d}/{EPOCHS} | "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"      Early stop at epoch {epoch}")
                break

    # Reload best weights and get final predictions
    if best_state is not None:
        model.load_state_dict(best_state)
    _, val_acc, val_f1, val_true, val_preds = evaluate(model, val_loader, criterion)
    return val_acc, val_f1, val_true, val_preds


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_win_sec(filename: str) -> int:
    basename = os.path.basename(filename)
    match = re.search(r"_(\d+)s_", basename)
    return int(match.group(1)) if match else 30


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def significance_label(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.1:   return "."
    return "ns"


def run_paired_ttests(fold_data: dict) -> dict:
    """Paired t-tests: ASPAG vs each baseline (acc and F1)."""
    results = {}
    for novel in NOVEL_NAMES:
        if novel not in fold_data:
            continue
        results[novel] = {}
        for baseline in BASELINE_NAMES:
            if baseline not in fold_data:
                continue
            n_accs = np.array(fold_data[novel]["fold_accs"])
            b_accs = np.array(fold_data[baseline]["fold_accs"])
            n_f1s  = np.array(fold_data[novel]["fold_f1s"])
            b_f1s  = np.array(fold_data[baseline]["fold_f1s"])
            _, p_acc = ttest_rel(n_accs, b_accs)
            _, p_f1  = ttest_rel(n_f1s,  b_f1s)
            results[novel][baseline] = {
                "acc_diff": float(n_accs.mean() - b_accs.mean()),
                "acc_p":    float(p_acc),
                "acc_sig":  significance_label(p_acc),
                "f1_diff":  float(n_f1s.mean() - b_f1s.mean()),
                "f1_p":     float(p_f1),
                "f1_sig":   significance_label(p_f1),
            }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    data_dir    = os.path.join(script_dir, "..", "data", "processed")
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Discover preprocessed files
    pattern   = os.path.join(data_dir, "*_preprocessed.pkl")
    pkl_files = sorted(glob.glob(pattern))
    if not pkl_files:
        print(f"ERROR: no preprocessed pkl files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(pkl_files)} dataset(s):")
    for p in pkl_files:
        print(f"  {os.path.basename(p)}")

    all_results = {}
    start_total = time.time()

    for pkl_path in pkl_files:
        ds_name = os.path.basename(pkl_path).replace("_preprocessed.pkl", "")
        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name}")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        windows_raw  = np.array(data["data"])   # (N, win_samples, C)
        labels       = np.array(data["labels"], dtype=np.int64)
        participants = np.array(data["participants"])  # subject IDs for GroupKFold

        # Remap labels to contiguous 0..n_classes-1
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map[l] for l in labels], dtype=np.int64)

        # Use first channel only
        windows = windows_raw[:, :, 0] if windows_raw.ndim == 3 else windows_raw

        N, win_samples = windows.shape
        win_sec  = parse_win_sec(pkl_path)
        fs       = win_samples // max(win_sec, 1)
        n_classes = len(unique_labels)
        n_groups = len(np.unique(participants))

        print(f"  Samples={N}  win_samples={win_samples}  win_sec={win_sec}s  fs={fs}Hz  n_classes={n_classes}  subjects={n_groups}")

        # Class distribution
        cls_dist = {}
        for c_orig, c_new in label_map.items():
            cnt = int((labels == c_new).sum())
            cls_dist[str(c_orig)] = cnt
        print(f"  Class distribution (original->count): {cls_dist}")

        ds_result = {
            "metadata": {
                "n_samples":   N,
                "win_samples": win_samples,
                "win_sec":     win_sec,
                "fs":          fs,
                "n_classes":   n_classes,
                "class_dist":  cls_dist,
            }
        }
        fold_data_all = {}  # tname -> {fold_accs, fold_f1s}

        # ---- Pre-compute images for all transforms ----
        cached_images = {}
        for tname in TRANSFORM_ORDER:
            fn, accepts_fs = ALL_TRANSFORMS[tname]
            print(f"\n  [Pre-computing] {tname} ...")
            t0 = time.time()
            imgs = precompute_images(windows, fn, accepts_fs, float(fs), tname)
            cached_images[tname] = imgs
            print(f"  Pre-compute done: {time.time()-t0:.1f}s  shape={imgs.shape}")

        # ---- 5-fold CV per transform (subject-aware GroupKFold when possible) ----
        if n_groups >= N_FOLDS:
            try:
                gkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
                folds = list(gkf.split(windows, labels, groups=participants))
                print(f"  Using StratifiedGroupKFold (subject-aware, {n_groups} subjects)")
            except Exception:
                gkf = GroupKFold(n_splits=min(N_FOLDS, n_groups))
                folds = list(gkf.split(windows, labels, groups=participants))
                print(f"  Using GroupKFold ({n_groups} subjects)")
        else:
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
            folds = list(skf.split(windows, labels))
            print(f"  Using StratifiedKFold (only {n_groups} subjects, too few for GroupKFold)")

        for tname in TRANSFORM_ORDER:
            images = cached_images[tname]
            print(f"\n  --- CNN Training: {tname} ---")
            t_start = time.time()

            fold_accs, fold_f1s = [], []
            per_class_f1_folds  = []  # list of per-class F1 arrays

            for fold_i, (train_idx, val_idx) in enumerate(folds):
                print(f"    Fold {fold_i+1}/{N_FOLDS}")
                try:
                    val_acc, val_f1, val_true, val_preds = train_fold(
                        images, labels, train_idx, val_idx, n_classes
                    )
                    fold_accs.append(val_acc)
                    fold_f1s.append(val_f1)

                    per_class_f1 = f1_score(val_true, val_preds,
                                            labels=list(range(n_classes)),
                                            average=None, zero_division=0)
                    per_class_f1_folds.append(per_class_f1.tolist())

                    print(f"    Fold {fold_i+1} result: acc={val_acc:.4f}  macro_f1={val_f1:.4f}")
                except Exception as exc:
                    import traceback
                    print(f"    [ERROR] Fold {fold_i+1} failed: {exc}")
                    traceback.print_exc()
                    fold_accs.append(0.0)
                    fold_f1s.append(0.0)
                    per_class_f1_folds.append([0.0] * n_classes)

            mean_acc = float(np.mean(fold_accs))
            mean_f1  = float(np.mean(fold_f1s))
            std_acc  = float(np.std(fold_accs))
            std_f1   = float(np.std(fold_f1s))

            # Average per-class F1 across folds
            mean_per_class_f1 = np.mean(per_class_f1_folds, axis=0).tolist()

            print(f"  RESULT {tname}: acc={mean_acc:.4f} (±{std_acc:.4f})  "
                  f"macro_f1={mean_f1:.4f} (±{std_f1:.4f})  "
                  f"wall_time={time.time()-t_start:.1f}s")

            ds_result[tname] = {
                "mean_acc":         mean_acc,
                "std_acc":          std_acc,
                "mean_f1":          mean_f1,
                "std_f1":           std_f1,
                "fold_accs":        [float(v) for v in fold_accs],
                "fold_f1s":         [float(v) for v in fold_f1s],
                "per_class_f1":     mean_per_class_f1,
                "wall_time_s":      float(time.time() - t_start),
            }
            fold_data_all[tname] = {
                "fold_accs": fold_accs,
                "fold_f1s":  fold_f1s,
            }

        # ---- Paired t-tests ----
        print(f"\n  --- Paired t-tests (novel vs baseline) ---")
        ttests = run_paired_ttests(fold_data_all)
        ds_result["ttest"] = ttests
        for novel, comparisons in ttests.items():
            for baseline, stats in comparisons.items():
                print(f"  {novel} vs {baseline}: "
                      f"acc {stats['acc_diff']:+.4f} ({stats['acc_sig']})  "
                      f"f1 {stats['f1_diff']:+.4f} ({stats['f1_sig']})")

        all_results[ds_name] = ds_result

        # Free memory
        del cached_images

    total_time = time.time() - start_total

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("CNN BENCHMARK — SimpleCNN — 5-fold stratified CV — image_size=64 — CPU")
    print("=" * 120)

    ds_names = list(all_results.keys())
    col_w = 22

    # Header row
    header = f"{'Transform':<12} {'n_classes':>9}"
    for ds in ds_names:
        short = ds.replace("_30s_50ovl", "").replace("_5s_50ovl", "").replace("_10s_50ovl", "")
        header += f" | {short[:col_w]:^{col_w}}"
    print(header)
    print("-" * (12 + 10 + (col_w + 3) * len(ds_names)))

    for tname in TRANSFORM_ORDER:
        for ds in ds_names:
            if tname in all_results[ds] and "mean_acc" in all_results[ds][tname]:
                pass  # we'll print per-dataset below
        row = f"{tname:<12} {'':>9}"
        for ds in ds_names:
            entry = all_results[ds].get(tname, {})
            if "mean_acc" in entry:
                cell = f"{entry['mean_acc']:.3f}/{entry['mean_f1']:.3f}"
            elif "error" in entry:
                cell = "ERR"
            else:
                cell = "---"
            row += f" | {cell:^{col_w}}"
        print(row)

    print("=" * (12 + 10 + (col_w + 3) * len(ds_names)))
    print("Columns: Acc/MacroF1 (mean over 5 folds)")

    # Per-dataset n_classes row
    nc_row = f"{'n_classes':<12} {'':>9}"
    for ds in ds_names:
        nc = all_results[ds]["metadata"]["n_classes"]
        nc_row += f" | {nc:^{col_w}}"
    print(nc_row)

    # t-test summary
    print("\nPaired t-test summary (novel vs baseline, per dataset):")
    print("-" * 100)
    for ds in ds_names:
        if "ttest" not in all_results[ds]:
            continue
        print(f"\n  {ds}:")
        for novel, comparisons in all_results[ds]["ttest"].items():
            for baseline, stats in comparisons.items():
                print(f"    {novel} vs {baseline:12s}: "
                      f"acc {stats['acc_diff']:+.4f} ({stats['acc_sig']})  "
                      f"f1 {stats['f1_diff']:+.4f} ({stats['f1_sig']})")

    print(f"\nTotal runtime: {total_time/60:.1f} min")

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    out_path = os.path.join(results_dir, "cnn_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
