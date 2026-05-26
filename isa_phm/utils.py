"""
Shared utilities for the ISA-PHM wrapper.

Contents
--------
ReferenceResolver   — builds a flat @id → object index from the raw ISA-JSON tree
                      and resolves bare {"@id": "..."} references to full objects.
_compute_features   — compute scalar PHM health indicators from a 1-D signal array.
_is_windows_path    — detect Windows absolute paths cross-platform.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
from scipy.stats import kurtosis as _scipy_kurtosis
from scipy.stats import skew as _scipy_skew

from .errors import ExtractionError

logger = logging.getLogger("isa_phm")


# ---------------------------------------------------------------------------
# Reference resolver
# ---------------------------------------------------------------------------


class ReferenceResolver:
    """
    Builds an @id-indexed lookup table from the full raw ISA-JSON dict and
    resolves `{"@id": "..."}` placeholder objects to their full definitions.

    Usage
    -----
    resolver = ReferenceResolver()
    resolver.build(repaired_dict)          # index entire tree (one pass)
    full_obj = resolver.resolve(ref)        # resolve a bare reference
    full_obj = resolver.try_resolve(ref)    # same but returns None if missing
    """

    def __init__(self) -> None:
        self._index: dict[str, dict] = {}
        self._warned_duplicates: set[str] = set()

    # ------------------------------------------------------------------
    # Building the index
    # ------------------------------------------------------------------

    def build(self, root: Any) -> None:
        """Walk the entire ISA-JSON tree and register every full object with @id."""
        self._walk(root)
        logger.debug("ReferenceResolver indexed %d objects.", len(self._index))

    def _walk(self, node: Any) -> None:
        if isinstance(node, dict):
            # A "full object" has @id plus at least one other key.
            # A bare reference like {"@id": "#foo"} has len == 1 → skip.
            if "@id" in node and len(node) > 1:
                id_str = node["@id"]
                if id_str not in self._index:
                    self._index[id_str] = node
                else:
                    # Duplicate @id – keep first occurrence, warn once.
                    existing = self._index[id_str]
                    if existing is not node:
                        logger.warning(
                            "Duplicate @id '%s' — keeping first occurrence.", id_str
                        )
            for value in node.values():
                self._walk(value)
        elif isinstance(node, list):
            for item in node:
                self._walk(item)

    # ------------------------------------------------------------------
    # Resolving references
    # ------------------------------------------------------------------

    def resolve(self, ref: Any) -> dict:
        """
        Resolve a reference to its full object dict.

        Parameters
        ----------
        ref : str | dict | Any
            - str  : treated as a bare @id string.
            - dict with only "@id" key  : bare reference placeholder.
            - dict with "@id" + other keys : already a full object, returned as-is.
            - anything else : returned as-is.

        Raises
        ------
        ExtractionError
            If the @id is not found in the index.
        """
        if isinstance(ref, str):
            return self._lookup(ref)
        if isinstance(ref, dict):
            if len(ref) == 1 and "@id" in ref:
                return self._lookup(ref["@id"])
            # Already a full object (has @id + other keys, or no @id).
            return ref
        # Not a reference — pass through.
        return ref

    def try_resolve(self, ref: Any) -> dict | None:
        """Like resolve() but returns None instead of raising for missing @id."""
        try:
            return self.resolve(ref)
        except ExtractionError:
            return None

    def _lookup(self, id_str: str) -> dict:
        result = self._index.get(id_str)
        if result is None:
            raise ExtractionError(
                f"Unresolvable @id reference: '{id_str}'. "
                f"This @id does not appear as a full object anywhere in the ISA-JSON. "
                f"The file may be truncated or hand-edited. "
                f"Total @id objects indexed: {len(self._index)}."
            )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_annotation_value(self, ref: Any, default: str = "") -> str:
        """Resolve ref and extract annotationValue safely."""
        obj = self.try_resolve(ref)
        if obj is None:
            return default
        return obj.get("annotationValue", default)

    def get_unit_str(self, unit_ref: Any) -> str | None:
        """Resolve a unit reference and return the annotationValue string, or None."""
        if not unit_ref:
            return None
        obj = self.try_resolve(unit_ref)
        if obj is None:
            return None
        val = obj.get("annotationValue") or obj.get("termAccession") or None
        return val if val else None

    def __len__(self) -> int:
        return len(self._index)


# ---------------------------------------------------------------------------
# Signal feature computation
# ---------------------------------------------------------------------------

# Features computed per run for lifecycle analysis
FEATURE_NAMES: tuple[str, ...] = (
    "rms",
    "max",
    "mean",
    "peak2peak",
    "kurtosis",
    "std",
    "crest_factor",
    "skewness",
)


def compute_features(values: np.ndarray) -> dict[str, float | None]:
    """
    Compute scalar health-indicator features from a 1-D signal array.

    Parameters
    ----------
    values : np.ndarray
        Raw signal values (already filtered for NaN, finite).

    Returns
    -------
    dict mapping feature name → float (or None if computation failed).
    """
    if len(values) == 0:
        return {k: None for k in FEATURE_NAMES}

    v = values.astype(np.float64, copy=False)

    rms = float(np.sqrt(np.mean(v ** 2)))
    peak = float(np.max(np.abs(v)))
    peak2peak = float(np.max(v) - np.min(v))
    std = float(np.std(v))

    crest = float(peak / rms) if rms > 1e-12 else None
    kurt = float(_scipy_kurtosis(v, fisher=True))
    skewness = float(_scipy_skew(v))

    return {
        "rms": rms,
        "max": float(np.max(v)),
        "mean": float(np.mean(v)),
        "peak2peak": peak2peak,
        "kurtosis": kurt,
        "std": std,
        "crest_factor": crest,
        "skewness": skewness,
    }


# ---------------------------------------------------------------------------
# Outlier detection and correction
# ---------------------------------------------------------------------------


def detect_outliers(
    values: np.ndarray,
    method: str = "iqr",
    threshold: float = 3.0,
    lower: "float | None" = None,
    upper: "float | None" = None,
) -> tuple[np.ndarray, dict]:
    """
    Compute a boolean outlier mask for a 1-D signal array.

    Parameters
    ----------
    values : np.ndarray
        1-D array of signal values.
    method : "iqr" | "zscore" | "fixed"
        ``"iqr"``    — interquartile range; bounds = Q1/Q3 ± threshold × IQR.
        ``"zscore"`` — z-score; bounds = mean ± threshold × std.
        ``"fixed"``  — explicit bounds only; use ``lower`` and/or ``upper``
                       to define the valid range; ``threshold`` is ignored.
    threshold : float
        IQR multiplier or z-score threshold (default 3.0).
        Ignored when ``method="fixed"``.
    lower : float | None
        Override the computed lower bound.  Values strictly below this are
        flagged as outliers.  ``None`` keeps the statistically computed bound
        (or ``-inf`` for ``method="fixed"``).
    upper : float | None
        Override the computed upper bound.  Values strictly above this are
        flagged as outliers.  ``None`` keeps the statistically computed bound
        (or ``+inf`` for ``method="fixed"``).

    Returns
    -------
    mask : np.ndarray[bool]
        True where a sample is an outlier.
    stats : dict
        ``{"lower_bound": float, "upper_bound": float}``.

    Raises
    ------
    ValueError
        Unknown ``method``.
    """
    v = np.asarray(values, dtype=np.float64)
    if method == "iqr":
        q1 = float(np.percentile(v, 25))
        q3 = float(np.percentile(v, 75))
        iqr = q3 - q1
        lb = q1 - threshold * iqr
        ub = q3 + threshold * iqr
    elif method == "zscore":
        mean = float(np.mean(v))
        std = float(np.std(v))
        lb = mean - threshold * std
        ub = mean + threshold * std
    elif method == "fixed":
        lb = -np.inf
        ub = np.inf
    else:
        raise ValueError(
            f"Unknown outlier method: '{method}'. Use 'iqr', 'zscore', or 'fixed'."
        )
    if lower is not None:
        lb = lower
    if upper is not None:
        ub = upper
    mask = (v < lb) | (v > ub)
    return mask, {"lower_bound": lb, "upper_bound": ub}


def fix_outliers(
    values: np.ndarray,
    mask: np.ndarray,
    strategy: str = "clip",
    bounds: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Apply a correction strategy to outlier-flagged samples.

    Parameters
    ----------
    values : np.ndarray
        Original 1-D signal array.
    mask : np.ndarray[bool]
        Boolean mask from :func:`detect_outliers` (True = outlier).
    strategy : "clip" | "nan" | "drop"
        ``"clip"`` — clamp outliers to [lower_bound, upper_bound].
        ``"nan"``  — replace outliers with NaN.
        ``"drop"`` — remove outlier samples entirely (changes array length).
    bounds : (float, float) | None
        Required when ``strategy="clip"``; typically the
        ``(lower_bound, upper_bound)`` returned by :func:`detect_outliers`.

    Returns
    -------
    np.ndarray
        Corrected signal (same length except for ``"drop"`` strategy).

    Raises
    ------
    ValueError
        Unknown strategy or missing bounds for clip.
    """
    v = np.asarray(values, dtype=np.float64).copy()
    if strategy == "clip":
        if bounds is None:
            raise ValueError("bounds=(lower, upper) is required for strategy='clip'.")
        v = np.clip(v, bounds[0], bounds[1])
    elif strategy == "nan":
        v[mask] = np.nan
    elif strategy == "drop":
        v = v[~mask]
    elif strategy in ("interpolate", "ffill", "bfill"):
        # Replace flagged samples with NaN first, then fill using pandas
        import pandas as pd  # local import — utils has no top-level pandas dep
        s = pd.Series(v)
        s[mask] = np.nan
        if strategy == "interpolate":
            s = s.interpolate(method="linear", limit_direction="both")
        elif strategy == "ffill":
            s = s.ffill().bfill()  # bfill handles leading NaNs
        else:  # bfill
            s = s.bfill().ffill()  # ffill handles trailing NaNs
        v = s.to_numpy(dtype=np.float64)
    else:
        raise ValueError(
            f"Unknown fix strategy: '{strategy}'. "
            "Use 'clip', 'nan', 'drop', 'interpolate', 'ffill', or 'bfill'."
        )
    return v


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_WINDOWS_ABS_RE = re.compile(r"^[A-Za-z]:[/\\]|^\\\\")


def is_windows_absolute_path(path_str: str) -> bool:
    """Return True if path_str looks like a Windows absolute path (C:\\...)."""
    return bool(_WINDOWS_ABS_RE.match(path_str))
