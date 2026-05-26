"""
Signal analysis primitives for ISA-PHM data files.

All functions require a `base_dir` — the folder that contains the ISA JSON
investigation's referenced CSVs (relative paths in `dataFiles[].name` resolve
from there). When base_dir is None, functions return a structured
"data not available" record rather than raising — the agent can then surface
this as a capability gap to the user.

CSV assumption: single-column or two-column signal traces (time, value) or a
single value column. If multiple columns are present, the first numeric
column with length > 1 is used. This matches the Milling and most PHM
example datasets.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.stats import kurtosis as scipy_kurtosis


def _resolve_path(base_dir: str, relative_path: str) -> str:
    """Turn an ISA-style './Case_01/Sensor/...' path into an absolute filesystem
    path under base_dir. Handles both forward and backslash separators."""
    cleaned = relative_path.replace("\\", "/").lstrip("./").lstrip("/")
    return os.path.join(base_dir, *cleaned.split("/"))


def _read_signal(abs_path: str, max_rows: int | None = None) -> np.ndarray:
    """Return a 1-D float numpy array of the signal values from a CSV.
    Picks the first numeric column. Raises FileNotFoundError or ValueError."""
    if not os.path.exists(abs_path):
        raise FileNotFoundError(abs_path)

    read_kwargs = {"encoding": "utf-8", "low_memory": False}
    if max_rows:
        read_kwargs["nrows"] = max_rows

    # Try header auto-detect: if first row parses as numeric, treat as headerless
    try:
        df = pd.read_csv(abs_path, header=None, nrows=2, **{k: v for k, v in read_kwargs.items() if k != "nrows"})
    except Exception:
        df = pd.read_csv(abs_path, **read_kwargs)
    else:
        first_row_numeric = True
        for v in df.iloc[0].tolist():
            try:
                float(v)
            except (ValueError, TypeError):
                first_row_numeric = False
                break
        df = pd.read_csv(abs_path, header=None if first_row_numeric else 0, **read_kwargs)

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        raise ValueError(f"No numeric column in CSV: {abs_path}")

    # Prefer a column that isn't a monotonic "time" column
    chosen = None
    for col in numeric_df.columns:
        s = numeric_df[col].dropna()
        if len(s) < 4:
            continue
        diffs = np.diff(s.values[: min(len(s), 1000)])
        if diffs.size and np.all(diffs > 0) and len(numeric_df.columns) > 1:
            continue  # monotonic → likely time
        chosen = col
        break
    if chosen is None:
        chosen = numeric_df.columns[0]

    values = numeric_df[chosen].dropna().to_numpy(dtype=float)
    return values


# ---------------------------------------------------------------------------
# File loading (agent-facing)
# ---------------------------------------------------------------------------

def load_run_csv(base_dir: str | None, relative_path: str, max_rows: int = 5) -> dict:
    """Load the first few rows of a CSV plus summary stats. Used by the
    `load_dataframe_with_meta` equivalent tool."""
    if not base_dir:
        return {
            "status": "data_not_available",
            "message": (
                "No data base directory set. This tool needs the filesystem "
                "location of the CSVs referenced by the ISA-JSON. Set it in "
                "the Streamlit sidebar to enable signal-level tools."
            ),
            "relative_path": relative_path,
        }

    abs_path = _resolve_path(base_dir, relative_path)
    if not os.path.exists(abs_path):
        return {
            "status": "file_missing",
            "message": f"File not found on disk: {abs_path}",
            "relative_path": relative_path,
            "absolute_path": abs_path,
        }

    try:
        df_head = pd.read_csv(abs_path, nrows=max_rows)
        values = _read_signal(abs_path)
    except Exception as e:
        return {
            "status": "read_error",
            "message": f"Could not read CSV: {e}",
            "relative_path": relative_path,
        }

    return {
        "status": "ok",
        "relative_path": relative_path,
        "n_rows": int(len(values)),
        "columns": [str(c) for c in df_head.columns.tolist()],
        "dtypes": {str(c): str(df_head[c].dtype) for c in df_head.columns},
        "head": df_head.head(max_rows).to_dict(orient="list"),
        "stats": {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "rms": float(np.sqrt(np.mean(values ** 2))),
        },
    }


# ---------------------------------------------------------------------------
# Lifecycle features
# ---------------------------------------------------------------------------

def _feature_set(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"rms": None, "kurtosis": None, "crest_factor": None, "peak_to_peak": None, "mean": None, "std": None, "n": 0}
    rms = float(np.sqrt(np.mean(values ** 2)))
    peak = float(np.max(np.abs(values)))
    crest = peak / rms if rms > 0 else float("nan")
    return {
        "rms": rms,
        "kurtosis": float(scipy_kurtosis(values, fisher=True, bias=False)) if values.size > 3 else None,
        "crest_factor": crest,
        "peak_to_peak": float(np.max(values) - np.min(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "n": int(values.size),
    }


def lifecycle_features_for_assay(
    base_dir: str | None,
    data_files: list[str],
) -> dict:
    """Compute RMS/kurtosis/crest/peak-to-peak per run for an assay.
    data_files is the ordered list of relative CSV paths (one per run).
    """
    if not base_dir:
        return {
            "status": "data_not_available",
            "message": "No data base directory set — cannot compute lifecycle features.",
            "n_expected_runs": len(data_files),
        }

    per_run = []
    skipped = []
    for idx, rel in enumerate(data_files, start=1):
        abs_path = _resolve_path(base_dir, rel)
        if not os.path.exists(abs_path):
            skipped.append({"run": idx, "file": rel, "reason": "missing"})
            continue
        try:
            values = _read_signal(abs_path)
        except Exception as e:
            skipped.append({"run": idx, "file": rel, "reason": f"read_error: {e}"})
            continue
        feats = _feature_set(values)
        feats["run"] = idx
        feats["file"] = rel
        per_run.append(feats)

    trend = None
    if len(per_run) >= 3:
        rms_series = [r["rms"] for r in per_run if r["rms"] is not None]
        if len(rms_series) >= 3:
            x = np.arange(len(rms_series))
            slope = float(np.polyfit(x, rms_series, 1)[0])
            trend = {
                "rms_slope": slope,
                "monotonic_increasing": all(b >= a for a, b in zip(rms_series, rms_series[1:])),
                "direction": "increasing" if slope > 0 else ("decreasing" if slope < 0 else "flat"),
            }

    return {
        "status": "ok",
        "n_runs_processed": len(per_run),
        "n_runs_skipped": len(skipped),
        "skipped": skipped[:20],
        "per_run": per_run,
        "trend": trend,
    }


# ---------------------------------------------------------------------------
# Frequency-domain primitives
# ---------------------------------------------------------------------------

def compute_fft(values: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies_hz, magnitude) of the single-sided FFT."""
    n = values.size
    if n == 0:
        return np.array([]), np.array([])
    window = np.hanning(n)
    windowed = values * window
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(spectrum) * 2.0 / n
    return freqs, mag


def compute_psd(values: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD — returns (f, Pxx)."""
    nperseg = min(4096, max(256, values.size // 8))
    f, pxx = sp_signal.welch(values, fs=fs, nperseg=nperseg)
    return f, pxx


def compute_spectrogram(values: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nperseg = min(1024, max(128, values.size // 32))
    f, t, sxx = sp_signal.spectrogram(values, fs=fs, nperseg=nperseg)
    return f, t, sxx


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def detect_outliers_sigma(values: np.ndarray, sigma: float = 3.0) -> dict:
    if values.size == 0:
        return {"mask": np.array([], dtype=bool), "n_outliers": 0}
    mu = float(np.mean(values))
    sd = float(np.std(values))
    if sd == 0:
        return {"mask": np.zeros_like(values, dtype=bool), "n_outliers": 0, "mean": mu, "std": sd}
    mask = np.abs(values - mu) > sigma * sd
    return {
        "mask": mask,
        "n_outliers": int(mask.sum()),
        "mean": mu,
        "std": sd,
        "sigma": sigma,
    }
