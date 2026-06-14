"""
factor_utils.py — utilities for resolving CSV-backed study factor values.

ISA-PHM stores factor values as either:
  - Plain scalars (strings/numbers) — returned as-is.
  - Relative file paths ending in ``.csv``:
      * Single-cell CSV (1 row × 1 col, no header) → resolved to ``float``.
      * Multi-row CSV (timeseries) → resolved to a summary dict; use
        ``load_factor_csv_dataframe`` to get the full ``pd.DataFrame``.

Detection is purely content-based — no ``valueMode`` metadata is required.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("isa_phm")


def resolve_factor_csv_value(raw_value: Any, data_root: Path | None) -> Any:
    """
    Resolve a factor value, loading and interpreting CSV files when applicable.

    Parameters
    ----------
    raw_value : Any
        The raw factor value as stored in the ISA-JSON (string, number, etc.).
    data_root : Path | None
        Root directory used to resolve relative CSV paths.  When ``None``, the
        raw value is returned unchanged.

    Returns
    -------
    float
        When the path points to a single-cell CSV (scalar CSV).
    dict
        Summary dict when the CSV contains multiple rows (timeseries CSV):
        ``{"type": "timeseries", "path": str, "rows": int, "cols": int,
           "head": list[list]}``
    Any
        The original ``raw_value`` when it is not a ``.csv`` path, when
        ``data_root`` is ``None``, or when the file cannot be read.
    """
    if data_root is None:
        return raw_value
    if not isinstance(raw_value, str):
        return raw_value
    stripped = raw_value.strip()
    if not stripped.lower().endswith(".csv"):
        return raw_value

    csv_path = (data_root / stripped).resolve()
    if not csv_path.exists():
        logger.warning(
            "factor_utils: CSV path does not exist: '%s' (data_root='%s'). "
            "Returning raw value.",
            stripped,
            data_root,
        )
        return raw_value

    try:
        import pandas as pd

        df = pd.read_csv(csv_path, header=None)
    except Exception as exc:
        logger.warning(
            "factor_utils: failed to read CSV '%s': %s. Returning raw value.",
            csv_path,
            exc,
        )
        return raw_value

    if df.shape == (1, 1):
        try:
            return float(df.iloc[0, 0])
        except (ValueError, TypeError):
            return raw_value

    # Multi-row / multi-column — timeseries summary.
    return {
        "type": "timeseries",
        "path": stripped,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "head": df.head(3).values.tolist(),
    }


def load_factor_csv_dataframe(raw_value: str, data_root: Path) -> "pd.DataFrame":
    """
    Load the full DataFrame for a factor value that references a CSV file.

    Parameters
    ----------
    raw_value : str
        Relative CSV path as stored in the ISA-JSON (e.g.
        ``"./Case_01/02_Settings/02_CS/Case_01_CS_run_01.csv"``).
    data_root : Path
        Root directory used to resolve the relative path.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        When the value does not end in ``.csv`` or the file does not exist.
    """
    import pandas as pd

    stripped = raw_value.strip()
    if not stripped.lower().endswith(".csv"):
        raise ValueError(
            f"factor value '{raw_value}' is not a CSV file path."
        )
    csv_path = (data_root / stripped).resolve()
    if not csv_path.exists():
        raise ValueError(
            f"CSV file not found: '{csv_path}' "
            f"(data_root='{data_root}', relative path='{stripped}')."
        )
    return pd.read_csv(csv_path, header=None)
