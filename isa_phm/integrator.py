"""
DataIntegrator: load ISA-PHM measurement files into pandas DataFrames.

Two access modes:
1) Cached single-run loading for interactive usage.
2) Streaming lifecycle feature extraction for many runs.
"""

from __future__ import annotations

import csv
import logging
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Literal, cast

import numpy as np
import pandas as pd

from .errors import AmbiguousRunError, DataFileError, RunNotFoundError
from .factor_utils import resolve_factor_csv_value
from .schemas import AssayModel, DataFile, DataLoadMetadata, RunRecord
from .utils import compute_features

logger = logging.getLogger("isa_phm")

STANDARD_COLUMNS = (
    "time",
    "value",
)

REQUESTED_FILE_TYPES: tuple[str, ...] = ("raw", "processed", "auto")
RESOLVED_FILE_TYPES: tuple[str, ...] = ("raw", "processed")
CSV_BAD_LINE_MODES: tuple[str, ...] = ("error", "warn", "skip")


@dataclass(frozen=True)
class _CSVReadConfig:
    engine: Literal["c", "python"]
    sep: str | None
    encoding: str


@dataclass(frozen=True)
class _CSVReadDecision:
    config: _CSVReadConfig
    source: str


class _FIFOCache:
    """Simple FIFO eviction cache backed by OrderedDict."""

    def __init__(self, maxsize: int = 100) -> None:
        self._maxsize = maxsize
        self._store: OrderedDict = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: tuple) -> pd.DataFrame | None:
        with self._lock:
            return self._store.get(key)

    def put(self, key: tuple, df: pd.DataFrame) -> None:
        with self._lock:
            if key in self._store:
                return
            if len(self._store) >= self._maxsize:
                self._store.popitem(last=False)
            self._store[key] = df

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


class DataIntegrator:
    """Load and cache ISA-PHM data files as normalized DataFrames."""

    def __init__(
        self,
        data_root: Path,
        cache_maxsize: int = 100,
        enable_chunked_large_file_mode: bool = True,
        large_file_threshold_mb: float = 64.0,
        chunk_rows: int = 250_000,
        csv_bad_lines: Literal["error", "warn", "skip"] = "error",
    ) -> None:
        self._data_root = Path(data_root)
        self._cache = _FIFOCache(maxsize=cache_maxsize)
        self._enable_chunked_large_file_mode = enable_chunked_large_file_mode
        self._large_file_threshold_mb = max(0.0, float(large_file_threshold_mb))
        self._chunk_rows = max(1, int(chunk_rows))
        self._csv_bad_lines = self._validate_csv_bad_lines(csv_bad_lines)
        self._load_lock_guard = threading.Lock()
        self._load_locks: dict[tuple, threading.Lock] = {}

    # ------------------------------------------------------------------
    # Public: single-run loading
    # ------------------------------------------------------------------

    def load(
        self,
        assay: AssayModel,
        study_id: str,
        run_id: str | None = None,
        file_type: Literal["raw", "processed", "auto"] = "processed",
    ) -> pd.DataFrame:
        """
        Load a run into a DataFrame with columns [time, value].

        ``file_type='auto'`` prefers processed and falls back to raw.
        """
        df, _ = self.load_with_meta(
            assay=assay,
            study_id=study_id,
            run_id=run_id,
            file_type=file_type,
        )
        return df

    def load_with_meta(
        self,
        assay: AssayModel,
        study_id: str,
        run_id: str | None = None,
        file_type: Literal["raw", "processed", "auto"] = "processed",
    ) -> tuple[pd.DataFrame, DataLoadMetadata]:
        """Load a run and return DataFrame plus resolved file-load metadata."""
        requested_file_type = self._validate_file_type(file_type)
        run = self._resolve_run(assay, run_id)
        data_file, resolved_file_type = self._resolve_data_file(
            assay_id=assay.assay_id,
            run=run,
            requested_file_type=requested_file_type,
        )

        # study_id MUST be part of the key: different studies routinely share an
        # assay_id (e.g. 'se01') and run_id (e.g. 'run_01'), so a key without it
        # collides and returns one study's signal for another study's run.
        cache_key = (study_id, assay.assay_id, run.run_id, resolved_file_type)
        with self._cache_key_lock(cache_key):
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit: %s", cache_key)
                csv_config = cached.attrs.get("csv_config", {})
                return cached, DataLoadMetadata(
                    assay_id=assay.assay_id,
                    run_id=run.run_id,
                    requested_file_type=requested_file_type,
                    resolved_file_type=resolved_file_type,
                    file_path=data_file.path,
                    from_cache=True,
                    csv_engine=csv_config.get("engine"),
                    csv_sep=csv_config.get("sep"),
                    csv_encoding=csv_config.get("encoding"),
                    csv_detection_source=csv_config.get("source"),
                    csv_bad_lines=csv_config.get("on_bad_lines", self._csv_bad_lines),
                )

            df, read_decision = self._read_csv(Path(data_file.path), return_decision=True)
            self._normalize_time(df, run)
            df.attrs["csv_config"] = {
                "engine": read_decision.config.engine,
                "sep": read_decision.config.sep,
                "encoding": read_decision.config.encoding,
                "source": read_decision.source,
                "on_bad_lines": self._csv_bad_lines,
            }
            self._cache.put(cache_key, df)
            return df, DataLoadMetadata(
                assay_id=assay.assay_id,
                run_id=run.run_id,
                requested_file_type=requested_file_type,
                resolved_file_type=resolved_file_type,
                file_path=data_file.path,
                from_cache=False,
                csv_engine=read_decision.config.engine,
                csv_sep=read_decision.config.sep,
                csv_encoding=read_decision.config.encoding,
                csv_detection_source=read_decision.source,
                csv_bad_lines=self._csv_bad_lines,
            )

    # ------------------------------------------------------------------
    # Public: streaming lifecycle features
    # ------------------------------------------------------------------

    def stream_lifecycle_features(
        self,
        assay: AssayModel,
        study_id: str,
        file_type: Literal["raw", "processed", "auto"] = "processed",
    ) -> Generator[dict, None, None]:
        """Yield one lifecycle feature row per run without caching all runs."""
        requested_file_type = self._validate_file_type(file_type)

        for run in assay.runs:
            try:
                data_file, resolved_file_type = self._resolve_data_file(
                    assay_id=assay.assay_id,
                    run=run,
                    requested_file_type=requested_file_type,
                )
                # study_id keeps runs from different studies that share an
                # assay_id/run_id from colliding in the shared frame cache.
                cache_key = (study_id, assay.assay_id, run.run_id, resolved_file_type)
                raw_values = self._load_values_for_features(
                    path=Path(data_file.path),
                    cache_key=cache_key,
                )
            except DataFileError as exc:
                logger.warning(
                    "Skipping run '%s' of assay '%s': %s",
                    run.run_id,
                    assay.assay_id,
                    exc,
                )
                continue

            values = raw_values.dropna().to_numpy(dtype=np.float64)
            features = compute_features(values)
            yield {
                "run_id": run.run_id,
                "run_number": run.run_number,
                "study_id": study_id,
                "assay_id": assay.assay_id,
                **features,
                **{f"fv_{k}": resolve_factor_csv_value(v, self._data_root) for k, v in run.factor_values.items()},
            }

    def _compute_run_features(
        self,
        run: RunRecord,
        assay: AssayModel,
        study_id: str,
        file_type: Literal["raw", "processed", "auto"],
    ) -> dict | None:
        """Helper for threaded lifecycle feature extraction."""
        try:
            data_file, resolved_file_type = self._resolve_data_file(
                assay_id=assay.assay_id,
                run=run,
                requested_file_type=self._validate_file_type(file_type),
            )
            # study_id keeps runs from different studies that share an
            # assay_id/run_id from colliding in the shared frame cache.
            cache_key = (study_id, assay.assay_id, run.run_id, resolved_file_type)
            raw_values = self._load_values_for_features(
                path=Path(data_file.path),
                cache_key=cache_key,
            )
        except DataFileError as exc:
            logger.warning(
                "Skipping run '%s' of assay '%s': %s",
                run.run_id,
                assay.assay_id,
                exc,
            )
            return None

        values = raw_values.dropna().to_numpy(dtype=np.float64)
        features = compute_features(values)
        return {
            "run_id": run.run_id,
            "run_number": run.run_number,
            "study_id": study_id,
            "assay_id": assay.assay_id,
            **features,
            **{f"fv_{k}": resolve_factor_csv_value(v, self._data_root) for k, v in run.factor_values.items()},
        }

    def lifecycle_features_df(
        self,
        assay: AssayModel,
        study_id: str,
        file_type: Literal["raw", "processed", "auto"] = "processed",
        n_workers: int | None = None,
    ) -> pd.DataFrame:
        """Return lifecycle features for all runs as a DataFrame."""
        requested_file_type = self._validate_file_type(file_type)
        effective_workers = (
            n_workers if n_workers is not None else min(32, (os.cpu_count() or 1) + 4)
        )

        if effective_workers == 1 or len(assay.runs) <= 1:
            rows = list(
                self.stream_lifecycle_features(
                    assay=assay,
                    study_id=study_id,
                    file_type=requested_file_type,
                )
            )
        else:
            rows_unordered: list[dict] = []
            with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                futures = {
                    pool.submit(
                        self._compute_run_features,
                        run,
                        assay,
                        study_id,
                        requested_file_type,
                    ): run
                    for run in assay.runs
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        rows_unordered.append(result)
            rows = sorted(rows_unordered, key=lambda r: r["run_number"])

        if not rows:
            logger.warning(
                "No lifecycle features extracted for assay '%s'.",
                assay.assay_id,
            )
            return pd.DataFrame()
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Evict all cached DataFrames."""
        n = len(self._cache)
        self._cache.clear()
        logger.info("DataIntegrator cache cleared (%d entries removed).", n)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_run(self, assay: AssayModel, run_id: str | None) -> RunRecord:
        if run_id is None:
            if len(assay.runs) == 1:
                return assay.runs[0]
            if len(assay.runs) == 0:
                raise DataFileError(
                    f"Assay '{assay.assay_id}' has no runs extracted from the ISA-JSON."
                )
            raise AmbiguousRunError(
                f"Assay '{assay.assay_id}' has {len(assay.runs)} runs. "
                "Specify run_id explicitly. "
                f"Available: {[r.run_id for r in assay.runs]}"
            )

        run = assay.get_run(run_id)
        if run is None:
            available = [r.run_id for r in assay.runs]
            raise RunNotFoundError(
                f"run_id '{run_id}' not found in assay '{assay.assay_id}'. "
                f"Available run IDs: {available}"
            )
        return run

    @staticmethod
    def _validate_file_type(
        file_type: str,
    ) -> Literal["raw", "processed", "auto"]:
        normalized = str(file_type).strip().lower()
        if normalized not in REQUESTED_FILE_TYPES:
            raise DataFileError(
                f"Invalid file_type '{file_type}'. "
                f"Use one of: {REQUESTED_FILE_TYPES}."
            )
        return cast(Literal["raw", "processed", "auto"], normalized)

    @staticmethod
    def _validate_csv_bad_lines(
        csv_bad_lines: str,
    ) -> Literal["error", "warn", "skip"]:
        normalized = str(csv_bad_lines).strip().lower()
        if normalized not in CSV_BAD_LINE_MODES:
            raise DataFileError(
                f"Invalid csv_bad_lines '{csv_bad_lines}'. "
                f"Use one of: {CSV_BAD_LINE_MODES}."
            )
        return cast(Literal["error", "warn", "skip"], normalized)

    @staticmethod
    def _resolve_data_file(
        assay_id: str,
        run: RunRecord,
        requested_file_type: Literal["raw", "processed", "auto"],
    ) -> tuple[DataFile, Literal["raw", "processed"]]:
        if requested_file_type == "auto":
            if run.processed_file is not None and run.processed_file.path:
                return run.processed_file, "processed"
            if run.raw_file is not None and run.raw_file.path:
                return run.raw_file, "raw"
            raise DataFileError(
                f"Assay '{assay_id}', run '{run.run_id}': no data file is recorded "
                "in the ISA-JSON (neither 'processed' nor 'raw')."
            )

        resolved_file_type = cast(
            Literal["raw", "processed"],
            requested_file_type,
        )
        data_file = (
            run.processed_file if resolved_file_type == "processed" else run.raw_file
        )
        if data_file is None or not data_file.path:
            raise DataFileError(
                f"Assay '{assay_id}', run '{run.run_id}': "
                f"no '{resolved_file_type}' data file is recorded in the ISA-JSON. "
                "Use file_type='auto' to prefer processed and fall back to raw."
            )
        return data_file, resolved_file_type

    @staticmethod
    def _normalize_time(df: pd.DataFrame, run: RunRecord) -> None:
        """Scale/shift time column to seconds in-place when sampling freq is known."""
        if "time" not in df.columns:
            raise DataFileError("Cannot normalize time: DataFrame is missing 'time' column.")
        if df.empty:
            raise DataFileError("Cannot normalize time: DataFrame has no rows.")

        t = df["time"]
        t_valid = t.dropna()
        if t_valid.empty:
            raise DataFileError("Cannot normalize time: no valid time values found.")

        fs: float | None = None
        for pv in run.measurement_params:
            if pv.unit and ("hz" in pv.unit.lower()):
                try:
                    val = float(str(pv.value).replace(",", "."))
                    fs = val * 1000.0 if "khz" in pv.unit.lower() else val
                    break
                except (ValueError, TypeError):
                    continue

        t0 = float(t_valid.iloc[0])
        if fs and fs > 0:
            dt_raw = float(t.diff().dropna().abs().median())
            if np.isfinite(dt_raw) and dt_raw > 0:
                scale = (1.0 / fs) / dt_raw
                df["time"] = (t - t0) * scale
                return

        df["time"] = t - t0

    @contextmanager
    def _cache_key_lock(self, cache_key: tuple):
        """Return a per-cache-key lock to avoid duplicate concurrent file loads."""
        with self._load_lock_guard:
            lock = self._load_locks.get(cache_key)
            if lock is None:
                lock = threading.Lock()
                self._load_locks[cache_key] = lock
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    def _should_use_chunked_mode(self, path: Path) -> bool:
        if not self._enable_chunked_large_file_mode:
            return False
        try:
            size_mb = path.stat().st_size / (1024 * 1024)
        except OSError:
            return False
        return size_mb >= self._large_file_threshold_mb

    def _load_values_for_features(self, path: Path, cache_key: tuple) -> pd.Series:
        """
        Load a value series for feature extraction.

        Large files can be streamed in chunks to avoid materializing the full frame.
        """
        with self._cache_key_lock(cache_key):
            cached = self._cache.get(cache_key)
            if cached is not None:
                return self._get_value_column(cached)

            if self._should_use_chunked_mode(path):
                return self._read_values_chunked(path)

            df = self._read_csv(path)
            self._cache.put(cache_key, df)
            return self._get_value_column(df)

    def _read_csv(
        self,
        path: Path,
        *,
        return_decision: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, _CSVReadDecision]:
        """Read a 2-column measurement CSV/TSV into [time, value]."""
        if not path.exists():
            raise DataFileError(
                f"Data file not found: '{path}'. "
                "Set data_root to the directory containing your measurement files."
            )

        logger.debug("Reading CSV: '%s'.", path)
        read_decision = self._detect_csv_read_config(path)
        read_config = read_decision.config
        try:
            df = self._read_csv_frame(path, read_config)
        except Exception as exc:
            raise DataFileError(
                f"Failed to read '{path}' as CSV with detected config "
                f"(engine={read_config.engine}, sep={read_config.sep}, encoding={read_config.encoding}): {exc}"
            ) from exc

        if df.empty:
            raise DataFileError(f"Data file is empty: '{path}'.")

        # Drop header row if first row is non-numeric (e.g. timestamp,value).
        if not _is_numeric_row(df.iloc[0]):
            df = df.iloc[1:].reset_index(drop=True)

        if df.empty:
            raise DataFileError(f"Data file is empty after removing header: '{path}'.")

        if df.shape[1] < 2:
            raise DataFileError(
                f"Data file '{path}' has only {df.shape[1]} column(s). "
                "ISA-PHM one-column rule requires exactly 2 columns: time + value."
            )

        if df.shape[1] > 2:
            logger.warning("'%s' has %d columns; using first two as (time, value).", path, df.shape[1])

        df = df.iloc[:, :2].copy()
        df.columns = ["time", "value"]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["time"] = _to_float_seconds(df["time"])
        if return_decision:
            return df, read_decision
        return df

    def _read_values_chunked(self, path: Path) -> pd.Series:
        """Stream only the value column for large files to reduce memory use."""
        read_decision = self._detect_csv_read_config(path)
        chunks = self._iter_csv_chunks(path, read_decision.config)
        first_chunk = True
        warned_extra_cols = False
        arrays: list[np.ndarray] = []

        for chunk in chunks:
            if chunk.empty:
                continue
            if first_chunk and not _is_numeric_row(chunk.iloc[0]):
                chunk = chunk.iloc[1:].reset_index(drop=True)
            first_chunk = False
            if chunk.empty:
                continue

            if chunk.shape[1] < 2:
                raise DataFileError(
                    f"Data file '{path}' has only {chunk.shape[1]} column(s). "
                    "ISA-PHM one-column rule requires exactly 2 columns: time + value."
                )
            if chunk.shape[1] > 2 and not warned_extra_cols:
                logger.warning(
                    "'%s' has %d columns; using second column as value for chunked feature mode.",
                    path,
                    chunk.shape[1],
                )
                warned_extra_cols = True

            values = pd.to_numeric(chunk.iloc[:, 1], errors="coerce").dropna()
            if not values.empty:
                arrays.append(values.to_numpy(dtype=np.float64))

        if not arrays:
            return pd.Series(dtype=np.float64)
        if len(arrays) == 1:
            return pd.Series(arrays[0], dtype=np.float64)
        return pd.Series(np.concatenate(arrays), dtype=np.float64)

    def _detect_csv_read_config(self, path: Path) -> _CSVReadDecision:
        """Detect a workable CSV read configuration with a small probe pass."""
        # Fast path: C engine with explicit common separators.
        for sep in (",", "\t", ";"):
            for encoding in ("utf-8", "latin-1"):
                config = _CSVReadConfig(engine="c", sep=sep, encoding=encoding)
                if self._probe_csv(path, config):
                    decision = _CSVReadDecision(config=config, source="common-separators")
                    self._log_csv_config(path, decision)
                    return decision

        # Fallback 1: sniff delimiter once, then retry C engine.
        sniff_sep = self._sniff_delimiter(path)
        if sniff_sep:
            for encoding in ("utf-8", "latin-1"):
                config = _CSVReadConfig(engine="c", sep=sniff_sep, encoding=encoding)
                if self._probe_csv(path, config):
                    decision = _CSVReadDecision(config=config, source="sniffed-delimiter")
                    self._log_csv_config(path, decision)
                    return decision

        # Fallback 2: Python engine with auto-detection.
        for encoding in ("utf-8", "latin-1"):
            config = _CSVReadConfig(engine="python", sep=None, encoding=encoding)
            if self._probe_csv(path, config):
                decision = _CSVReadDecision(config=config, source="python-auto")
                self._log_csv_config(path, decision)
                return decision

        raise DataFileError(f"Cannot decode '{path}'. Tried UTF-8 and Latin-1.")

    def _log_csv_config(self, path: Path, decision: _CSVReadDecision) -> None:
        config = decision.config
        if (
            config.encoding != "utf-8"
            or config.engine == "python"
            or self._csv_bad_lines != "error"
        ):
            logger.info(
                "CSV config selected for '%s': engine=%s sep=%r encoding=%s source=%s on_bad_lines=%s",
                path,
                config.engine,
                config.sep,
                config.encoding,
                decision.source,
                self._csv_bad_lines,
            )
        else:
            logger.debug(
                "CSV config selected for '%s': engine=%s sep=%r encoding=%s source=%s on_bad_lines=%s",
                path,
                config.engine,
                config.sep,
                config.encoding,
                decision.source,
                self._csv_bad_lines,
            )

    @staticmethod
    def _sniff_delimiter(path: Path) -> str | None:
        for encoding in ("utf-8", "latin-1"):
            try:
                with path.open("rb") as fh:
                    sample_raw = fh.read(32768)
                sample = sample_raw.decode(encoding, errors="strict")
                if not sample.strip():
                    continue
                return csv.Sniffer().sniff(sample, delimiters=",\t;| ").delimiter
            except (UnicodeDecodeError, csv.Error, OSError):
                continue
        return None

    def _probe_csv(self, path: Path, config: _CSVReadConfig) -> bool:
        try:
            probe = self._read_csv_frame(path, config, nrows=8)
            return probe.shape[1] >= 2
        except UnicodeDecodeError:
            return False
        except Exception:
            return False

    def _read_csv_frame(
        self,
        path: Path,
        config: _CSVReadConfig,
        nrows: int | None = None,
    ) -> pd.DataFrame:
        return pd.read_csv(
            path,
            sep=config.sep,
            engine=config.engine,
            encoding=config.encoding,
            header=None,
            dtype=str,
            on_bad_lines=self._csv_bad_lines,
            nrows=nrows,
        )

    def _iter_csv_chunks(
        self,
        path: Path,
        config: _CSVReadConfig,
    ):
        return pd.read_csv(
            path,
            sep=config.sep,
            engine=config.engine,
            encoding=config.encoding,
            header=None,
            dtype=str,
            on_bad_lines=self._csv_bad_lines,
            chunksize=self._chunk_rows,
        )

    @staticmethod
    def _get_value_column(df: pd.DataFrame) -> pd.Series:
        if "value" in df.columns:
            return df["value"]
        return df.iloc[:, 1]


def _is_numeric_row(row: pd.Series) -> bool:
    """Return True if all values in row parse as numeric."""
    try:
        pd.to_numeric(row, errors="raise")
        return True
    except (ValueError, TypeError):
        return False


def _to_float_seconds(series: pd.Series) -> pd.Series:
    """Convert time column to float seconds where possible."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric
    try:
        return pd.to_timedelta(series).dt.total_seconds()
    except Exception:
        return numeric
