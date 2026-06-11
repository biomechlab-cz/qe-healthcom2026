"""1D-to-2D time series transformation implementations.

All transformations take a 1D signal array and return a 2D image array.
Consistent interface: transform(signal, image_size) -> (image_size, image_size) ndarray.
"""

import numpy as np
from pyts.image import GramianAngularField, RecurrencePlot, MarkovTransitionField
from scipy.signal import stft as scipy_stft


def _paa_reduce(signal, target_len):
    """Piecewise Aggregate Approximation to downsample signal."""
    n = len(signal)
    if n == target_len:
        return signal
    if n < target_len:
        # Upsample via linear interpolation
        return np.interp(np.linspace(0, n - 1, target_len), np.arange(n), signal)
    # Downsample: average over segments
    result = np.zeros(target_len)
    for i in range(target_len):
        start = int(i * n / target_len)
        end = int((i + 1) * n / target_len)
        result[i] = signal[start:end].mean()
    return result


# ============================================================
# Baseline Transformations
# ============================================================

def transform_gasf(signal, image_size=64):
    """Gramian Angular Summation Field."""
    gaf = GramianAngularField(image_size=image_size, method='summation')
    return gaf.fit_transform(signal.reshape(1, -1)).squeeze()


def transform_gadf(signal, image_size=64):
    """Gramian Angular Difference Field."""
    gaf = GramianAngularField(image_size=image_size, method='difference')
    return gaf.fit_transform(signal.reshape(1, -1)).squeeze()


def transform_rp(signal, image_size=64):
    """Recurrence Plot."""
    rp = RecurrencePlot(threshold='distance', percentage=20)
    img = rp.fit_transform(signal.reshape(1, -1)).squeeze()
    # Resize to target image_size
    from skimage.transform import resize
    return resize(img, (image_size, image_size), anti_aliasing=True)


def transform_mtf(signal, image_size=64):
    """Markov Transition Field."""
    mtf = MarkovTransitionField(image_size=image_size, n_bins=8)
    return mtf.fit_transform(signal.reshape(1, -1)).squeeze()


def transform_spectrogram(signal, image_size=64, fs=250):
    """STFT spectrogram."""
    nperseg = min(256, len(signal) // 4)
    noverlap = nperseg // 2
    f, t, Zxx = scipy_stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    spec = np.abs(Zxx)
    # Log scale for better dynamic range
    spec = np.log1p(spec)
    # Resize to image_size x image_size
    from skimage.transform import resize
    return resize(spec, (image_size, image_size), anti_aliasing=True)


def transform_cwt_scalogram(signal, image_size=64, fs=250):
    """Continuous Wavelet Transform scalogram using Morlet wavelet."""
    from skimage.transform import resize
    # Reduce signal for computational efficiency
    reduced = _paa_reduce(signal, image_size * 4)
    n = len(reduced)
    # Generate scales
    widths = np.geomspace(2, n / 4, num=image_size)
    # Manual CWT with Morlet-like wavelet (Gabor)
    rows = []
    for w in widths:
        # Create Morlet wavelet: complex sinusoid * Gaussian
        M = min(int(10 * w), n)
        t = np.arange(M) - M / 2
        wavelet = np.exp(1j * 2 * np.pi * t / w) * np.exp(-t**2 / (2 * (w / 2)**2))
        wavelet = np.real(wavelet) / np.sqrt(w)
        conv = np.convolve(reduced, wavelet, mode='same')
        rows.append(np.abs(conv))
    scalogram = np.array(rows)
    scalogram = np.log1p(scalogram)
    return resize(scalogram, (image_size, image_size), anti_aliasing=True)


def transform_rpm(signal, image_size=64):
    """Relative Position Matrix (Chen & Shi, 2019).
    M[i,j] = x[i] - x[j]
    """
    reduced = _paa_reduce(signal, image_size)
    M = reduced[:, None] - reduced[None, :]
    # Normalize to [0, 1] for image compatibility
    if M.max() != M.min():
        M = (M - M.min()) / (M.max() - M.min())
    return M


# ============================================================
# Novel Transformation: Multi-Scale Derivative Coupling Field (MSDCF)
# ============================================================
#
# HYPOTHESIS: Combining temporal position differences (like RPM) with
# rate-of-change information at multiple time scales captures both
# amplitude relationships AND dynamic behavior that single-scale methods miss.
#
# Motivation:
#   - RPM captures pairwise amplitude differences but ignores signal dynamics
#   - GAF encodes angular relationships but at a single temporal resolution
#   - Physiological signals (ECG, EDA) have meaningful features at multiple
#     time scales (individual beats, HRV trends, slow SCR drifts)
#   - The first derivative encodes rate of change, which is diagnostically
#     important (ST segment slope in ECG, SCR rise time in EDA)
#
# Definition:
#   Given signal x(t) of length N:
#   1. Compute x at original scale and its smoothed versions at K scales:
#      x_k(t) = smooth(x, window=2^k), k=0..K-1
#   2. For each scale k, compute derivative: d_k(t) = diff(x_k)
#   3. Compute coupling matrix at each scale:
#      C_k[i,j] = x_k[i] * d_k[j] - x_k[j] * d_k[i]
#      This captures "amplitude-derivative coupling" — how the amplitude at
#      time i relates to the rate of change at time j, antisymmetrically.
#   4. Aggregate across scales: M[i,j] = sum_k w_k * C_k[i,j]
#      where w_k = 1/K (uniform) or learned weights.
#
# The coupling x[i]*d[j] - x[j]*d[i] is inspired by the cross product
# (treating (x, dx) as 2D vectors at each time point). It captures rotational
# dynamics in the phase space of (amplitude, velocity) — regions where the
# signal is "spinning" vs. "traveling straight" in phase space.
#
# Expected advantages:
#   - Captures dynamics (not just static amplitude relationships)
#   - Multi-scale: sensitive to both fast oscillations and slow trends
#   - No discretization or embedding hyperparameters
#   - Computationally O(K * n^2) with K typically 3-4
#
# Likely failure modes:
#   - Derivative amplifies noise — may need signal pre-smoothing
#   - If classification depends purely on frequency content, CWT/STFT may win
#   - Multi-scale aggregation might blur scale-specific discriminative features
#
# Ablation ideas:
#   - Single-scale MSDCF vs. multi-scale
#   - Different coupling functions (additive vs. multiplicative)
#   - Individual scale channels vs. aggregated

def _smooth(signal, window_size):
    """Simple moving average smoothing."""
    if window_size <= 1:
        return signal.copy()
    kernel = np.ones(window_size) / window_size
    # Pad to avoid edge effects
    padded = np.pad(signal, (window_size // 2, window_size // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(signal)]


def transform_msdcf(signal, image_size=64, n_scales=4):
    """Multi-Scale Derivative Coupling Field (novel method).

    Combines amplitude-derivative cross-coupling across multiple time scales.

    Parameters
    ----------
    signal : 1D array
    image_size : int
        Output image dimensions.
    n_scales : int
        Number of smoothing scales (powers of 2).

    Returns
    -------
    2D array of shape (image_size, image_size)
    """
    reduced = _paa_reduce(signal, image_size)

    M = np.zeros((image_size, image_size))

    for k in range(n_scales):
        window = 2 ** k
        x_k = _smooth(reduced, window)
        # Compute derivative (forward difference, padded)
        d_k = np.gradient(x_k)

        # Normalize to unit variance to make scales comparable
        x_std = np.std(x_k)
        d_std = np.std(d_k)
        if x_std > 0:
            x_k = x_k / x_std
        if d_std > 0:
            d_k = d_k / d_std

        # Coupling matrix: cross-product-like antisymmetric coupling
        # C[i,j] = x[i]*d[j] - x[j]*d[i]
        C = x_k[:, None] * d_k[None, :] - x_k[None, :] * d_k[:, None]

        M += C / n_scales

    # Normalize to [0, 1]
    if M.max() != M.min():
        M = (M - M.min()) / (M.max() - M.min())

    return M


def transform_msdcf_multichannel(signal, image_size=64, n_scales=4):
    """Multi-Scale Derivative Coupling Field — multi-channel variant.

    Returns separate scale images stacked as channels instead of aggregating.

    Returns
    -------
    3D array of shape (image_size, image_size, n_scales)
    """
    reduced = _paa_reduce(signal, image_size)
    channels = []

    for k in range(n_scales):
        window = 2 ** k
        x_k = _smooth(reduced, window)
        d_k = np.gradient(x_k)

        x_std = np.std(x_k)
        d_std = np.std(d_k)
        if x_std > 0:
            x_k = x_k / x_std
        if d_std > 0:
            d_k = d_k / d_std

        C = x_k[:, None] * d_k[None, :] - x_k[None, :] * d_k[:, None]

        if C.max() != C.min():
            C = (C - C.min()) / (C.max() - C.min())

        channels.append(C)

    return np.stack(channels, axis=-1)


def transform_segment_stats(signal, image_size=64, fs=250):
    """Segment-statistics image: divide signal into image_size segments,
    compute multiple statistics per segment, arrange as 2D image.

    Each row = one segment, each column = one statistic across segments,
    creating a (image_size x n_stats) image resized to (image_size x image_size).
    """
    from skimage.transform import resize
    n = len(signal)
    seg_len = n // image_size

    stats_matrix = []
    for i in range(image_size):
        seg = signal[i * seg_len:(i + 1) * seg_len]
        if len(seg) == 0:
            stats_matrix.append(np.zeros(8))
            continue
        # Compute 8 statistics per segment
        stats = [
            np.mean(seg),
            np.std(seg),
            np.min(seg),
            np.max(seg),
            np.median(seg),
            np.mean(np.abs(np.diff(seg))),  # mean absolute derivative
            np.std(np.diff(seg)),  # derivative std
            np.sum(np.abs(seg) > np.std(signal)),  # threshold crossings
        ]
        stats_matrix.append(stats)

    img = np.array(stats_matrix)
    # Normalize columns independently
    for c in range(img.shape[1]):
        col = img[:, c]
        if col.max() != col.min():
            img[:, c] = (col - col.min()) / (col.max() - col.min())
    return resize(img, (image_size, image_size), anti_aliasing=True)


def transform_msdcf_segmented(signal, image_size=64, n_scales=4, fs=250):
    """MSDCF applied to segment-level features instead of raw PAA.

    Instead of reducing 7500 samples to 64 via PAA, we:
    1. Divide signal into image_size segments
    2. Compute mean + derivative-mean per segment (preserving dynamics)
    3. Apply MSDCF coupling on these segment-level features

    This preserves more temporal dynamics than raw PAA reduction.
    """
    n = len(signal)
    seg_len = n // image_size

    # Compute segment features: mean amplitude and mean derivative
    seg_means = np.zeros(image_size)
    seg_derivs = np.zeros(image_size)
    for i in range(image_size):
        seg = signal[i * seg_len:(i + 1) * seg_len]
        seg_means[i] = np.mean(seg)
        seg_derivs[i] = np.mean(np.abs(np.diff(seg))) if len(seg) > 1 else 0

    M = np.zeros((image_size, image_size))

    for k in range(n_scales):
        window = 2 ** k
        x_k = _smooth(seg_means, window)
        d_k = _smooth(seg_derivs, window)

        x_std = np.std(x_k)
        d_std = np.std(d_k)
        if x_std > 0:
            x_k = x_k / x_std
        if d_std > 0:
            d_k = d_k / d_std

        C = x_k[:, None] * d_k[None, :] - x_k[None, :] * d_k[:, None]
        M += C / n_scales

    if M.max() != M.min():
        M = (M - M.min()) / (M.max() - M.min())
    return M


def transform_msdf(signal, image_size=64, fs=250):
    """Multi-Scale Derivative Field (MSDF) — novel method, refined.

    Hypothesis: A 2D image where rows represent temporal segments and columns
    represent different signal characterizations (raw stats, derivatives at
    multiple scales, spectral features) captures both local morphology and
    multi-scale dynamics better than any single-perspective encoding.

    Unlike pure pairwise matrix methods (GAF, RPM) which create n×n relationship
    matrices, MSDF creates a segment×feature matrix that preserves the full
    statistical richness of each segment at multiple analysis scales.

    Construction:
    1. Divide signal into image_size temporal segments
    2. For each segment, compute features at K scales:
       - Scale 0: raw statistics (mean, std, min, max, median, energy)
       - Scale 1: first derivative statistics
       - Scale 2: smoothed signal (window=4) derivative statistics
       - Scale 3: smoothed signal (window=16) derivative statistics
       - Plus: zero-crossing rate, peak count, spectral centroid
    3. Arrange as (image_size × n_features) matrix, resize to (image_size × image_size)

    This yields a rich multi-perspective view of each temporal segment.
    """
    from skimage.transform import resize
    n = len(signal)
    seg_len = n // image_size

    feature_rows = []

    for i in range(image_size):
        start = i * seg_len
        end = min((i + 1) * seg_len, n)
        seg = signal[start:end]

        if len(seg) < 4:
            feature_rows.append(np.zeros(24))
            continue

        features = []

        # Scale 0: raw signal statistics
        features.extend([
            np.mean(seg),
            np.std(seg),
            np.min(seg),
            np.max(seg),
            np.median(seg),
            np.sum(seg ** 2) / len(seg),  # energy
        ])

        # Multi-scale derivative features
        for window in [1, 4, 16]:
            if window > 1:
                smoothed = _smooth(seg, window)
            else:
                smoothed = seg
            deriv = np.diff(smoothed) if len(smoothed) > 1 else np.array([0.0])
            features.extend([
                np.mean(deriv),
                np.std(deriv),
                np.mean(np.abs(deriv)),
                np.max(np.abs(deriv)),
            ])

        # Shape features
        mean_val = np.mean(seg)
        zero_crossings = np.sum(np.diff(np.sign(seg - mean_val)) != 0)
        features.append(zero_crossings / len(seg))

        # Local peaks
        if len(seg) > 2:
            peaks = np.sum((seg[1:-1] > seg[:-2]) & (seg[1:-1] > seg[2:]))
        else:
            peaks = 0
        features.append(peaks / max(len(seg), 1))

        # Simple spectral features (using FFT on segment)
        if len(seg) >= 8:
            fft_vals = np.abs(np.fft.rfft(seg))
            fft_freqs = np.fft.rfftfreq(len(seg), 1.0 / fs)
            total_power = np.sum(fft_vals ** 2)
            if total_power > 0:
                spectral_centroid = np.sum(fft_freqs * fft_vals ** 2) / total_power
                spectral_bandwidth = np.sqrt(np.sum((fft_freqs - spectral_centroid) ** 2 * fft_vals ** 2) / total_power)
            else:
                spectral_centroid = 0
                spectral_bandwidth = 0
            # Ratio of low-freq to high-freq power
            mid_idx = len(fft_vals) // 2
            lf_power = np.sum(fft_vals[:mid_idx] ** 2)
            hf_power = np.sum(fft_vals[mid_idx:] ** 2)
            lf_hf_ratio = lf_power / (hf_power + 1e-10)
        else:
            spectral_centroid = 0
            spectral_bandwidth = 0
            lf_hf_ratio = 1

        features.extend([spectral_centroid, spectral_bandwidth, lf_hf_ratio, 0])  # pad to 24

        feature_rows.append(features[:24])

    img = np.array(feature_rows)  # (image_size, 24)

    # Normalize each feature column to [0, 1]
    for c in range(img.shape[1]):
        col = img[:, c]
        rng = col.max() - col.min()
        if rng > 0:
            img[:, c] = (col - col.min()) / rng

    return resize(img, (image_size, image_size), anti_aliasing=True)


def transform_subband_correlation(signal, image_size=64, fs=250, n_bands=None):
    """Sub-band Temporal Correlation Field (STCF) — novel method.

    Hypothesis: Decompose the signal into frequency sub-bands using a filter bank,
    then for each pair of time segments compute the cross-band energy correlation.
    This creates a 2D image that jointly encodes frequency decomposition and
    temporal correlation — information that neither spectrograms (freq vs time)
    nor GAF/RPM (time vs time in amplitude space) capture.

    Construction:
    1. Decompose signal into B frequency sub-bands using bandpass filtering
    2. Divide each sub-band into T temporal segments
    3. For each segment, compute energy in each sub-band -> (T, B) matrix
    4. Compute correlation between all pairs of time segments across their
       sub-band energy profiles: M[i,j] = corr(energy_profile_i, energy_profile_j)

    The resulting T×T matrix encodes "spectral similarity over time" — when two
    time segments have similar frequency content, their correlation is high.

    This is distinct from:
    - Spectrogram: shows absolute freq content at each time (T×F)
    - GAF/RPM: compares amplitudes, not spectral content (T×T in amplitude space)
    - Recurrence Plot: compares phase-space trajectories, not spectral profiles

    STCF shows T×T in spectral-similarity space — a genuinely different perspective.
    """
    from skimage.transform import resize

    if n_bands is None:
        n_bands = min(image_size // 2, 32)

    n = len(signal)
    n_segments = image_size

    # Frequency sub-band decomposition via FFT band slicing
    fft_full = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    n_fft = len(fft_full)

    # Create sub-band boundaries (log-spaced for better low-freq resolution)
    band_edges = np.geomspace(max(0.5, freqs[1]), fs / 2, num=n_bands + 1)

    # Compute sub-band signals via inverse FFT of each band
    subband_signals = []
    for b in range(n_bands):
        mask = (freqs >= band_edges[b]) & (freqs < band_edges[b + 1])
        band_fft = np.zeros_like(fft_full)
        band_fft[mask] = fft_full[mask]
        subband_signals.append(np.fft.irfft(band_fft, n=n))

    # Compute energy in each sub-band for each temporal segment
    seg_len = n // n_segments
    energy_profiles = np.zeros((n_segments, n_bands))

    for t in range(n_segments):
        start = t * seg_len
        end = min((t + 1) * seg_len, n)
        for b in range(n_bands):
            seg = subband_signals[b][start:end]
            energy_profiles[t, b] = np.sum(seg ** 2) / max(len(seg), 1)

    # Normalize energy profiles per band (z-score)
    for b in range(n_bands):
        col = energy_profiles[:, b]
        std = np.std(col)
        if std > 0:
            energy_profiles[:, b] = (col - np.mean(col)) / std

    # Compute spectral correlation matrix: cosine similarity between
    # energy profiles of different time segments
    norms = np.linalg.norm(energy_profiles, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = energy_profiles / norms
    M = normalized @ normalized.T  # cosine similarity matrix

    # Shift to [0, 1]
    M = (M + 1) / 2

    return resize(M, (image_size, image_size), anti_aliasing=True) if M.shape[0] != image_size else M


def transform_multiview_fusion(signal, image_size=64, fs=250):
    """Multi-View Fusion Image (MVFI) — novel composite method.

    Hypothesis: No single 1D-to-2D transformation captures all discriminative
    information. By stacking the 3 most complementary representations as
    RGB channels of a single image, we provide the classifier with
    simultaneous access to:
      - Channel R: Spectrogram (frequency content over time)
      - Channel G: RPM (amplitude relationships)
      - Channel B: STCF (spectral similarity over time)

    This is inspired by Ahmad et al. (2021) who fused GAF+RP+MTF as RGB,
    but we select channels based on complementarity rather than convention:
    spectrogram captures frequency, RPM captures amplitude dynamics, and STCF
    captures cross-time spectral correlation — three orthogonal views.

    The fused image is flattened and all 3 channels' features are jointly
    available to the classifier.
    """
    spec = transform_spectrogram(signal, image_size=image_size, fs=fs)
    rpm = transform_rpm(signal, image_size=image_size)
    stcf = transform_subband_correlation(signal, image_size=image_size, fs=fs)

    # Stack as 3-channel image
    return np.stack([spec, rpm, stcf], axis=-1)


# Registry of all available transformations
TRANSFORMS = {
    "GASF": transform_gasf,
    "GADF": transform_gadf,
    "RP": transform_rp,
    "MTF": transform_mtf,
    "Spectrogram": transform_spectrogram,
    "CWT": transform_cwt_scalogram,
    "RPM": transform_rpm,
    "MSDCF": transform_msdcf,
    "SegStats": transform_segment_stats,
    "MSDCF_seg": transform_msdcf_segmented,
    "MSDF": transform_msdf,
    "STCF": transform_subband_correlation,
    "MVFI": transform_multiview_fusion,
}
