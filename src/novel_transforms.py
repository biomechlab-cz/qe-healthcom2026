"""
Novel 1D-to-2D time series transformations.

Three experimental transforms for encoding univariate signals as 2D images:
  - ASPAG : Analytic Signal Phase-Amplitude Gramian
  - MSGK  : Multi-Scale Gradient Kernel Map
  - PSSAM : Phase-Space Signed Area Matrix

Dependencies: numpy, scipy only.
"""

import numpy as np
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _paa(signal, target_len):
    """Piecewise Aggregate Approximation (PAA) reduction / interpolation.

    Reduces or expands *signal* to *target_len* samples.  When the signal is
    shorter than the target the result is produced by linear interpolation;
    when it is longer each frame is the mean of the corresponding segment.

    Parameters
    ----------
    signal : array_like, shape (n,)
    target_len : int

    Returns
    -------
    ndarray, shape (target_len,)
    """
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    if n == target_len:
        return signal.copy()
    if n < target_len:
        return np.interp(
            np.linspace(0, n - 1, target_len), np.arange(n), signal
        )
    return np.array(
        [
            signal[int(i * n / target_len) : int((i + 1) * n / target_len)].mean()
            for i in range(target_len)
        ]
    )


# ---------------------------------------------------------------------------
# Transform 1: ASPAG
# ---------------------------------------------------------------------------

def transform_aspag(signal, image_size=64, fs=250):
    """Analytic Signal Phase-Amplitude Gramian (ASPAG).

    Encodes the coupling between instantaneous amplitude and phase by forming
    a gramian matrix whose (i, j) entry reflects how coherently two time
    points contribute to the overall amplitude-modulated oscillation.

    Algorithm
    ---------
    1. Compute the analytic signal z(t) = hilbert(signal).
    2. Extract amplitude A(t) = |z(t)| and phase φ(t) = angle(z(t)).
    3. Normalise amplitude: A_bar = (A - A.min()) / (A.max() - A.min() + ε).
    4. PAA-reduce A_bar and φ each to *image_size* samples.
    5. M[i,j] = A_bar[i] * A_bar[j] * cos(φ[i] - φ[j]).
    6. Shift to [0, 1]: M = (M + 1) / 2.

    Parameters
    ----------
    signal : array_like, shape (n,)
    image_size : int, default 64
    fs : float, default 250
        Sampling frequency (currently unused but reserved for future
        band-limited extensions).

    Returns
    -------
    ndarray, shape (image_size, image_size)
    """
    signal = np.asarray(signal, dtype=float)

    z = hilbert(signal)
    A = np.abs(z)
    phi = np.unwrap(np.angle(z))  # unwrap to avoid discontinuities

    A_norm = (A - A.min()) / (A.max() - A.min() + 1e-10)

    # Compute unit-analytic vector at full resolution, PAA after projection
    # Use cos(φ) and sin(φ) components — these are safe to average (unlike raw φ)
    cos_phi = np.cos(phi) * A_norm
    sin_phi = np.sin(phi) * A_norm

    cos_r = _paa(cos_phi, image_size)
    sin_r = _paa(sin_phi, image_size)

    # M[i,j] = cos_r[i]*cos_r[j] + sin_r[i]*sin_r[j]
    #        = A_bar[i]*A_bar[j]*cos(φ[i]-φ[j])  (angle addition formula)
    M = cos_r[:, np.newaxis] * cos_r[np.newaxis, :] + \
        sin_r[:, np.newaxis] * sin_r[np.newaxis, :]

    # Normalise to [0, 1]
    mn, mx = M.min(), M.max()
    if mx > mn:
        M = (M - mn) / (mx - mn)
    return M


# ---------------------------------------------------------------------------
# Transform 2: MSGK
# ---------------------------------------------------------------------------

def transform_msgk(signal, image_size=64, n_scales=6):
    """Multi-Scale Gradient Kernel Map (MSGK).

    Captures multi-resolution dynamics by smoothing the signal at
    exponentially increasing scales and computing the gradient at each scale.
    The resulting feature vectors are compared via a Gaussian (RBF) kernel to
    produce the image.

    Algorithm
    ---------
    1. For k in 0..n_scales-1, σ_k = 2^k:
       - Smooth signal with gaussian_filter1d(signal, σ_k).
       - Compute gradient via np.gradient.
       - PAA-reduce to *image_size* samples.
    2. Stack into feature matrix V of shape (image_size, n_scales).
    3. Normalise each column to zero mean and unit variance.
    4. Pairwise squared distances D[i,j] = ||V[i] - V[j]||².
    5. Bandwidth h = median of sqrt(D[D > 0]).
    6. M[i,j] = exp(-D[i,j] / (2h² + ε)).

    Parameters
    ----------
    signal : array_like, shape (n,)
    image_size : int, default 64
    n_scales : int, default 6

    Returns
    -------
    ndarray, shape (image_size, image_size)
    """
    signal = np.asarray(signal, dtype=float)

    cols = []
    for k in range(n_scales):
        sigma_k = 2.0 ** k
        smoothed = gaussian_filter1d(signal, sigma=sigma_k)
        grad = np.gradient(smoothed)
        cols.append(_paa(grad, image_size))

    V = np.column_stack(cols)  # (image_size, n_scales)

    # Normalise each column
    mu = V.mean(axis=0)
    std = V.std(axis=0)
    std[std < 1e-10] = 1e-10
    V = (V - mu) / std

    # Pairwise squared L2 distances via broadcasting — no Python loops
    diff = V[:, np.newaxis, :] - V[np.newaxis, :, :]  # (N, N, n_scales)
    D = (diff ** 2).sum(axis=-1)                        # (N, N)

    positive = D[D > 0]
    h = np.median(np.sqrt(positive)) if len(positive) > 0 else 1.0

    M = np.exp(-D / (2.0 * h ** 2 + 1e-10))
    return M


# ---------------------------------------------------------------------------
# Transform 3: PSSAM
# ---------------------------------------------------------------------------

def transform_pssam(signal, image_size=64):
    """Phase-Space Signed Area Matrix (PSSAM).

    Represents the signed area swept by the phase-space trajectory (x, dx)
    between any two time indices.  Positive values indicate counter-clockwise
    excursions; negative values indicate clockwise ones, giving the matrix a
    rich signed structure that captures local oscillatory geometry.

    Algorithm
    ---------
    1. PAA-reduce signal to *image_size* points → x.
    2. Compute dx = np.gradient(x).
    3. Precompute element-wise cross-products: c[k] = x[k]*dx[k+1] - x[k+1]*dx[k]
       (using index k and k+1, padded so length equals image_size).
    4. Build cumulative sum: cumsum[j] = Σ_{k=0}^{j-1} c[k].
    5. S[i,j] = 0.5 * (cumsum[max(i,j)] - cumsum[min(i,j)]) — vectorised.
    6. M = tanh(S / (std(S) + ε)).
    7. M = (M + 1) / 2.

    Parameters
    ----------
    signal : array_like, shape (n,)
    image_size : int, default 64

    Returns
    -------
    ndarray, shape (image_size, image_size)
    """
    signal = np.asarray(signal, dtype=float)

    x = _paa(signal, image_size)   # (N,)
    dx = np.gradient(x)            # (N,)

    N = image_size

    # Cross-product terms: c[k] = x[k]*dx[k+1] - x[k+1]*dx[k]
    # For k = N-1 the "next" index wraps; we pad with 0 so cumsum length = N.
    c = np.empty(N, dtype=float)
    c[: N - 1] = x[: N - 1] * dx[1:N] - x[1:N] * dx[: N - 1]
    c[N - 1] = 0.0

    # cumsum[j] = Σ_{k=0}^{j-1} c[k],  cumsum[0] = 0  (length N+1)
    cumsum = np.empty(N + 1, dtype=float)
    cumsum[0] = 0.0
    cumsum[1:] = np.cumsum(c)

    # Vectorised signed-area matrix — no Python loops
    idx = np.arange(N)
    lo = np.minimum(idx[:, np.newaxis], idx[np.newaxis, :])  # (N, N)
    hi = np.maximum(idx[:, np.newaxis], idx[np.newaxis, :])  # (N, N)

    S = 0.5 * (cumsum[hi] - cumsum[lo])  # (N, N)

    # Robust percentile scaling to [0, 1]
    vmin = np.percentile(S, 2)
    vmax = np.percentile(S, 98)
    if vmax <= vmin:
        vmax = vmin + 1e-10
    M = np.clip((S - vmin) / (vmax - vmin), 0, 1)
    return M


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

NOVEL_TRANSFORMS = {
    "ASPAG": transform_aspag,
    "MSGK": transform_msgk,
    "PSSAM": transform_pssam,
}
