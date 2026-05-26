"""
Matplotlib plotting for ISA-PHM signals and lifecycle features.

Each function returns a matplotlib.figure.Figure the caller can render in
Streamlit (`st.pyplot(fig)`) or save to disk. Every plot accepts either raw
numpy arrays or the structured dicts produced by analytics.py so that tool
handlers can pass through whatever they have.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # headless-safe; Streamlit renders figures from this backend.
import matplotlib.pyplot as plt
import numpy as np

from analytics import (
    _read_signal,
    _resolve_path,
    compute_fft,
    compute_psd,
    compute_spectrogram,
    detect_outliers_sigma,
)


def _new_fig(figsize: tuple[float, float] = (9.0, 4.5)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def plot_timeseries(
    base_dir: str,
    relative_path: str,
    fs: float | None = None,
    title: str | None = None,
) -> plt.Figure:
    abs_path = _resolve_path(base_dir, relative_path)
    values = _read_signal(abs_path)
    fig, ax = _new_fig()
    if fs:
        t = np.arange(values.size) / fs
        ax.plot(t, values, linewidth=0.6)
        ax.set_xlabel("Time (s)")
    else:
        ax.plot(values, linewidth=0.6)
        ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    ax.set_title(title or f"Time series — {relative_path}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_frequency_domain(
    base_dir: str,
    relative_path: str,
    fs: float,
    title: str | None = None,
) -> plt.Figure:
    abs_path = _resolve_path(base_dir, relative_path)
    values = _read_signal(abs_path)
    freqs, mag = compute_fft(values, fs)
    fig, ax = _new_fig()
    ax.plot(freqs, mag, linewidth=0.7)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(title or f"FFT — {relative_path}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_power_spectral_density(
    base_dir: str,
    relative_path: str,
    fs: float,
    title: str | None = None,
) -> plt.Figure:
    abs_path = _resolve_path(base_dir, relative_path)
    values = _read_signal(abs_path)
    f, pxx = compute_psd(values, fs)
    fig, ax = _new_fig()
    ax.semilogy(f, pxx, linewidth=0.7)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V²/Hz)")
    ax.set_title(title or f"Power spectral density — {relative_path}")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_spectrogram(
    base_dir: str,
    relative_path: str,
    fs: float,
    title: str | None = None,
) -> plt.Figure:
    abs_path = _resolve_path(base_dir, relative_path)
    values = _read_signal(abs_path)
    f, t, sxx = compute_spectrogram(values, fs)
    fig, ax = _new_fig()
    pcm = ax.pcolormesh(t, f, 10 * np.log10(sxx + 1e-12), shading="gouraud")
    fig.colorbar(pcm, ax=ax, label="Power (dB)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title or f"Spectrogram — {relative_path}")
    fig.tight_layout()
    return fig


def plot_lifecycle(
    per_run_features: Sequence[dict],
    feature: str = "rms",
    title: str | None = None,
) -> plt.Figure:
    """per_run_features: list of dicts with 'run' and feature keys
    (as produced by analytics.lifecycle_features_for_assay)."""
    xs = [r["run"] for r in per_run_features if r.get(feature) is not None]
    ys = [r[feature] for r in per_run_features if r.get(feature) is not None]
    fig, ax = _new_fig(figsize=(8.0, 4.0))
    if xs:
        ax.plot(xs, ys, marker="o", linewidth=1.4)
        if len(xs) >= 3:
            coeffs = np.polyfit(xs, ys, 1)
            trend = np.polyval(coeffs, xs)
            ax.plot(xs, trend, linestyle="--", alpha=0.6, label=f"linear trend (slope={coeffs[0]:.3g})")
            ax.legend()
    ax.set_xlabel("Run index")
    ax.set_ylabel(feature.upper())
    ax.set_title(title or f"Lifecycle — {feature.upper()} per run")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_multi_feature_lifecycle(
    per_run_features: Sequence[dict],
    features: Sequence[str] = ("rms", "kurtosis", "crest_factor"),
    title: str | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(len(features), 1, figsize=(8.0, 2.6 * len(features)), sharex=True)
    if len(features) == 1:
        axes = [axes]
    for ax, feat in zip(axes, features):
        xs = [r["run"] for r in per_run_features if r.get(feat) is not None]
        ys = [r[feat] for r in per_run_features if r.get(feat) is not None]
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=1.2)
        ax.set_ylabel(feat)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Run index")
    fig.suptitle(title or "Lifecycle features per run")
    fig.tight_layout()
    return fig


def plot_outlier_comparison(
    base_dir: str,
    relative_path: str,
    sigma: float = 3.0,
    title: str | None = None,
) -> plt.Figure:
    abs_path = _resolve_path(base_dir, relative_path)
    values = _read_signal(abs_path)
    out = detect_outliers_sigma(values, sigma=sigma)
    mask = out["mask"]
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.5), sharex=True)
    axes[0].plot(values, linewidth=0.5, color="C0")
    if mask.size and mask.any():
        idx = np.where(mask)[0]
        axes[0].scatter(idx, values[idx], color="C3", s=12, label=f"{mask.sum()} outliers")
        axes[0].legend()
    axes[0].set_title(title or f"Raw signal with {sigma}σ outliers — {relative_path}")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)

    cleaned = values.copy()
    if mask.size:
        cleaned[mask] = np.nan
    axes[1].plot(cleaned, linewidth=0.5, color="C2")
    axes[1].set_title("Signal with outliers removed")
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
