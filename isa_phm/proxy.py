"""
Fluent proxy layer — AD-2 primary API for ISA-PHM datasets.

Hierarchy
---------
ISAWrapper
    QueryNavigator              (investigation-level)
            StudyProxy          (study-level)
                    AssayProxy  (assay/sensor-level)
                            RunProxy   (single-run level)

Entry point::

    wrapper = ISAWrapper("path/to/isa.json", data_root="path/to/data/")
    df = wrapper.study("Case 01").assay("a_st01_se01").load_dataframe()
    fig = wrapper.study("Case 01").assay("a_st01_se01").plot_lifecycle()
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import pandas as pd

from .errors import (
    AmbiguousRunError,
    AssayNotFoundError,
    DataFileError,
    PlotError,
    RunNotFoundError,
    StudyNotFoundError,
    ValidationError,
)
from .schemas import (
    AssayModel,
    AssayOverview,
    AssaySummary,
    DataLoadMetadata,
    InvestigationModel,
    InvestigationOverview,
    MissingValuesReport,
    RunOverview,
    RunRecord,
    RunSummary,
    SemanticField,
    SemanticManifest,
    StudyModel,
    StudyOverview,
    StudySummary,
)

if TYPE_CHECKING:
    from .integrator import DataIntegrator
    from .plotter import ISAPlotter
    from .semantic import SemanticNormalizer

logger = logging.getLogger("isa_phm")
_FAULT_TYPE_RE = re.compile(r"\b(fault|damage|rul|defect|degradation)\b", re.IGNORECASE)
_UNIT_VALUE_RE = re.compile(r"^[A-Za-z]{1,8}$")


# ---------------------------------------------------------------------------
# QueryNavigator  (investigation-level)
# ---------------------------------------------------------------------------

class QueryNavigator:
    """
    Top-level entry point for traversing an ``InvestigationModel``.

    Attached to ``ISAWrapper`` as the primary routing object.

    Parameters
    ----------
    investigation : InvestigationModel
    integrator : DataIntegrator
    plotter : ISAPlotter
    """

    def __init__(
        self,
        investigation: InvestigationModel,
        integrator: "DataIntegrator",
        plotter: "ISAPlotter",
        semantic: "SemanticNormalizer | None" = None,
    ) -> None:
        from .semantic import SemanticNormalizer

        self._inv = investigation
        self._integrator = integrator
        self._plotter = plotter
        self._semantic = semantic or SemanticNormalizer()

        # Build lookup indices.
        self._by_uuid: dict[str, StudyModel] = {}
        self._by_title_lower: dict[str, StudyModel] = {}
        for s in investigation.studies:
            self._by_uuid[s.study_id] = s
            key = s.title.strip().lower()
            existing = self._by_title_lower.get(key)
            if existing is not None and existing.study_id != s.study_id:
                raise ValidationError(
                    "Duplicate normalized study title detected. "
                    f"Both '{existing.title}' ({existing.study_id}) and "
                    f"'{s.title}' ({s.study_id}) map to key '{key}'."
                )
            self._by_title_lower[key] = s

    # Public API mirrors ISAWrapper for convenience.

    def investigation_overview(self) -> InvestigationOverview:
        """Return a high-level summary of the investigation."""
        from .schemas import StudySummary

        study_summaries = [
            StudySummary(
                study_id=s.study_id,
                title=s.title,
                n_assays=len(s.assays),
                n_runs=s.run_count,
                n_factors=len(s.factors),
            )
            for s in self._inv.studies
        ]
        return InvestigationOverview(
            title=self._inv.title,
            description=self._inv.description,
            identifier=self._inv.identifier,
            experiment_type=self._inv.experiment_type,
            n_studies=len(self._inv.studies),
            n_contacts=len(self._inv.contacts),
            studies=study_summaries,
        )

    def list_studies(self) -> list[StudySummary]:
        """Return a summary row for each study."""
        return self.investigation_overview().studies

    def study(self, study_id: str | int) -> "StudyProxy":
        """
        Look up a study by 1-based integer index, UUID, or title (case-insensitive).

        Parameters
        ----------
        study_id : str | int
            1-based integer index, study UUID, or human-readable title.

        Raises
        ------
        StudyNotFoundError
        """
        # Integer index (1-based).
        if isinstance(study_id, int):
            studies = self._inv.studies
            if not (1 <= study_id <= len(studies)):
                raise StudyNotFoundError(
                    f"Study index {study_id} out of range. "
                    f"Valid range: 1–{len(studies)}."
                )
            return StudyProxy(studies[study_id - 1], self._inv, self._integrator, self._plotter, self._semantic)

        # Exact UUID match.
        s = self._by_uuid.get(study_id)
        if s:
            return StudyProxy(s, self._inv, self._integrator, self._plotter, self._semantic)

        # Case-insensitive title match.
        s = self._by_title_lower.get(study_id.strip().lower())
        if s:
            return StudyProxy(s, self._inv, self._integrator, self._plotter, self._semantic)

        available = sorted(self._by_title_lower.keys())
        raise StudyNotFoundError(
            f"Study '{study_id}' not found in investigation '{self._inv.title}'. "
            f"Available study titles: {available}"
        )

    def compare_studies(
        self,
        study_ids: list[str | int],
        assay_id: str | int | None = None,
        assay_group: "AssayGroup | None" = None,
        file_type: Literal["raw", "processed", "auto"] = "raw",
        n_workers: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Load lifecycle features for the same sensor across multiple studies.

        Replaces the manual for-loop pattern. Returns a dict ready to pass
        directly into ``plotter.plot_multi_lifecycle()``.
        Only studies listed in ``study_ids`` are included.

        Parameters
        ----------
        study_ids : list[str | int]
            Study titles or 1-based indices.
        assay_id : str | int | None
            Single sensor — string assay_id or 1-based integer. Mutually
            exclusive with *assay_group*.
        assay_group : AssayGroup | None
            Named sensor group (e.g. ``AssayGroup([1, 2], name="Accel X/Y")``)
            — one lifecycle DataFrame per member, keyed by
            ``"StudyTitle — assay_alias"``.
        file_type : "raw" | "processed" | "auto"
        n_workers : int | None

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys are study titles (or ``"StudyTitle — alias"`` for groups).
        """
        if assay_id is not None and assay_group is not None:
            raise ValueError("Provide either assay_id or assay_group, not both.")
        out: dict[str, pd.DataFrame] = {}
        for sid in study_ids:
            sp = self.study(sid)
            if assay_group is not None:
                grp = sp.assay_group(assay_group)
                out.update(grp.lifecycle_features(file_type=file_type, n_workers=n_workers))
            else:
                aid = assay_id if assay_id is not None else 1
                lc = sp.assay(aid).lifecycle_features(file_type=file_type, n_workers=n_workers)
                out[sp.title] = lc
        return out


# ---------------------------------------------------------------------------
# StudyProxy
# ---------------------------------------------------------------------------

class StudyProxy:
    """
    Study-level proxy.  Provides summaries, factor listings, and assay navigation.

    Acquire via: ``wrapper.study("title or uuid")``
    """

    def __init__(
        self,
        study: StudyModel,
        investigation: InvestigationModel,
        integrator: "DataIntegrator",
        plotter: "ISAPlotter",
        semantic: "SemanticNormalizer",
    ) -> None:
        self._study = study
        self._inv = investigation
        self._integrator = integrator
        self._plotter = plotter
        self._semantic = semantic

        # Assay lookup by filename (assay_id).
        self._assay_by_id: dict[str, AssayModel] = {
            a.assay_id: a for a in study.assays
        }
        self._assay_proxy_cache: dict[str, AssayProxy] = {}

    def _assay_proxy(self, assay_model: AssayModel) -> "AssayProxy":
        proxy = self._assay_proxy_cache.get(assay_model.assay_id)
        if proxy is None:
            proxy = AssayProxy(
                assay_model,
                self._study,
                self._inv,
                self._integrator,
                self._plotter,
                self._semantic,
            )
            self._assay_proxy_cache[assay_model.assay_id] = proxy
        return proxy

    @property
    def has_runs(self) -> bool:
        """True if the study contains more than one run."""
        return self._study.run_count > 1

    @property
    def run_count(self) -> int:
        return self._study.run_count

    @property
    def title(self) -> str:
        return self._study.title

    def overview(self) -> StudyOverview:
        """Return a detailed summary of the study."""
        from .schemas import AssaySummary, FactorSummary

        assay_summaries = [
            AssaySummary(
                assay_id=a.assay_id,
                sensor_id=a.sensor.sensor_id,
                sensor_alias=a.sensor.alias,
                technology_type=a.sensor.technology_type,
                measurement_type=a.sensor.measurement_type,
                n_runs=len(a.runs),
                n_raw_files=len([r for r in a.runs if r.raw_file and r.raw_file.path]),
                n_processed_files=len([r for r in a.runs if r.processed_file and r.processed_file.path]),
            )
            for a in self._study.assays
        ]
        factor_summaries = [
            FactorSummary(
                factor_id=f.factor_id,
                factor_name=f.factor_name,
                factor_type=f.factor_type,
                unit=f.unit,
            )
            for f in self._study.factors
        ]
        return StudyOverview(
            study_id=self._study.study_id,
            title=self._study.title,
            description=self._study.description,
            experiment_type=self._inv.experiment_type,
            n_assays=len(self._study.assays),
            run_count=self._study.run_count,
            assays=assay_summaries,
            factors=factor_summaries,
        )

    def list_assays(self) -> list[AssaySummary]:
        """Return a summary row per assay."""
        return self.overview().assays

    def list_factors(self) -> list[FactorSummary]:
        """Return the factors defined for this study."""
        return self.overview().factors

    def semantic_factors(self) -> list[SemanticField]:
        """Return normalized semantic labels for this study's factors."""
        return self._semantic.normalize_study_factors(self._study)

    def variable_overview(self) -> list[pd.DataFrame]:
        """
        Return a list of DataFrames — one per unique experimental condition.

        Each DataFrame has two columns:
            variable — factor name
            value    — factor value for that condition

        For a single-condition diagnostic study this returns a list with one
        element; for a multi-condition study each element represents a distinct
        operating point or fault configuration.

        Returns
        -------
        list[pd.DataFrame]
        """
        seen_keys: list[tuple] = []
        seen_dicts: list[dict] = []
        for assay in self._study.assays:
            for run in assay.runs:
                key = tuple(sorted(run.factor_values.items()))
                if key not in seen_keys:
                    seen_keys.append(key)
                    seen_dicts.append(dict(run.factor_values))

        if not seen_dicts:
            return [pd.DataFrame(columns=["variable", "value"])]

        return [
            pd.DataFrame(
                [{"variable": k, "value": v} for k, v in fv_dict.items()],
                columns=["variable", "value"],
            )
            for fv_dict in seen_dicts
        ]

    def test_matrix(self) -> pd.DataFrame:
        """
        Return a compact factor-by-condition pivot table.

        Each row is one study factor; the first three columns are:

        ``variable`` — factor name |
        ``type``     — factor type annotation (e.g. "Operating condition") |
        ``unit``     — unit string (empty when not specified)

        Additional columns hold the factor value for each unique experimental
        condition observed in the dataset.  When only a single condition
        exists the data column is named ``"Value"``; for multiple runs
        they are labelled ``"Run 1"``, ``"Run 2"``, and so on.

        Returns
        -------
        pd.DataFrame
        """
        if not self._study.factors:
            return pd.DataFrame(columns=["variable", "type", "unit"])

        # Collect unique factor-value dicts in order of first appearance, paired
        # with the run number of the first run that carries that combination.
        seen_keys: list[tuple] = []
        conditions: list[dict] = []
        run_numbers: list[int] = []
        for assay in self._study.assays:
            for run in assay.runs:
                key = tuple(sorted(run.factor_values.items()))
                if key not in seen_keys:
                    seen_keys.append(key)
                    conditions.append(dict(run.factor_values))
                    run_numbers.append(run.run_number)

        rows = []
        for factor in self._study.factors:
            row: dict = {
                "variable": factor.factor_name,
                "type":     factor.factor_type,
                "unit":     factor.unit or "",
            }
            if len(conditions) == 1:
                row["Value"] = conditions[0].get(factor.factor_name, "")
            else:
                for rn, cond in zip(run_numbers, conditions):
                    row[f"Run {rn}"] = cond.get(factor.factor_name, "")
            rows.append(row)

        return pd.DataFrame(rows)

    def operating_conditions(self) -> pd.DataFrame:
        """
        Subset of :py:meth:`test_matrix` containing only operating-condition factors.

        Returns an empty DataFrame (with correct columns) when no operating
        condition factors are defined.
        """
        df = self.test_matrix()
        if df.empty:
            return df
        mask = df["type"].str.lower().str.contains("operating condition", na=False)
        return df[mask].reset_index(drop=True)

    def fault_conditions(self) -> pd.DataFrame:
        """
        Subset of :py:meth:`test_matrix` containing only fault-specification factors.

        Matches factor types whose annotation contains any of:
        ``fault``, ``damage``, ``rul``, ``defect``, ``degradation``.

        Returns an empty DataFrame (with correct columns) when no fault factors
        are defined.
        """
        df = self.test_matrix()
        if df.empty:
            return df
        mask = df["type"].fillna("").apply(lambda t: bool(_FAULT_TYPE_RE.search(str(t))))
        return df[mask].reset_index(drop=True)

    def get_fault_labels(self, assay_id: str | None = None) -> pd.DataFrame:
        """
        Return fault-related factor values for every (assay, run) pair.

        Useful for building labelled training datasets from fault-degradation
        experiments.

        Parameters
        ----------
        assay_id : str | None
            When provided, only include assays whose ``assay_id`` contains
            this string (case-sensitive substring match).

        Returns
        -------
        pd.DataFrame
            Columns: ``assay_id``, ``run_id``, ``run_number``,
            plus one column per fault-related factor.
            Returns an empty DataFrame with correct columns when no fault
            factors are defined.
        """
        fault_factor_names = [
            f.factor_name
            for f in self._study.factors
            if _FAULT_TYPE_RE.search(f.factor_type or "")
        ]
        base_cols = ["assay_id", "run_id", "run_number"]
        all_cols = base_cols + fault_factor_names

        if not fault_factor_names:
            return pd.DataFrame(columns=base_cols)

        rows = []
        for assay in self._study.assays:
            if assay_id is not None and assay_id not in assay.assay_id:
                continue
            for run in assay.runs:
                row: dict = {
                    "assay_id":   assay.assay_id,
                    "run_id":     run.run_id,
                    "run_number": run.run_number,
                }
                for fname in fault_factor_names:
                    row[fname] = run.factor_values.get(fname)
                rows.append(row)

        if not rows:
            return pd.DataFrame(columns=all_cols)
        return pd.DataFrame(rows, columns=all_cols)

    def plot_sensor_boxplot(
        self,
        file_type: Literal["raw", "processed", "auto"] = "raw",
        outlier_method: str = "fixed",
        outlier_upper: "float | None" = None,
        outlier_lower: "float | None" = None,
        outlier_strategy: str = "drop",
        column: str = "value",
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ):
        """
        Interactive Bokeh boxplot comparing amplitude across all sensor channels.

        Computes box statistics (Q1, median, Q3, whiskers, mean) for each sensor
        one at a time — only summary stats are kept in memory, not raw arrays.
        Returns a ``bokeh.plotting.figure`` that can be shown with::

            from bokeh.plotting import show
            show(fig_box)

        Parameters
        ----------
        file_type : "raw" | "processed" | "auto"
        outlier_method : "iqr" | "zscore" | "fixed"
        outlier_upper : float | None
            Hard upper bound, e.g. ``1e7`` to exclude hardware overflow values.
        outlier_lower : float | None
            Hard lower bound.
        outlier_strategy : "clip" | "nan" | "drop"
        column : str
            DataFrame column holding measurement values (default ``"value"``)
        title : str | None
            Override the auto-generated plot title.
        xlabel : str | None
            Override x-axis label.
        ylabel : str | None
            Override y-axis label.
        width : int | None
            Override figure width in pixels.
        height : int | None
            Override figure height in pixels.

        Returns
        -------
        bokeh.plotting.figure
        """
        try:
            from bokeh.models import ColumnDataSource, HoverTool, Whisker
            from bokeh.plotting import figure
        except ImportError as exc:
            raise ImportError(
                "bokeh is required for interactive boxplots. "
                "Install it with: pip install bokeh"
            ) from exc

        sensors: list[str] = []
        q1s:     list[float] = []
        medians: list[float] = []
        q3s:     list[float] = []
        means:   list[float] = []
        uppers:  list[float] = []
        lowers:  list[float] = []
        unit: str | None = None

        for summary in self.list_assays():
            a_model = self._assay_by_id[summary.assay_id]
            a = self._assay_proxy(a_model)
            if unit is None:
                unit = a._infer_unit()
            try:
                df = a.load_dataframe(file_type=file_type)
                df_clean = a.fix_outliers(
                    df,
                    method=outlier_method,
                    upper=outlier_upper,
                    lower=outlier_lower,
                    strategy=outlier_strategy,
                )
                vals = df_clean[column].dropna()
                q1  = float(vals.quantile(0.25))
                med = float(vals.quantile(0.50))
                q3  = float(vals.quantile(0.75))
                iqr = q3 - q1
                sensors.append(summary.sensor_alias)
                q1s.append(q1)
                medians.append(med)
                q3s.append(q3)
                means.append(float(vals.mean()))
                uppers.append(min(float(vals.max()), q3 + 1.5 * iqr))
                lowers.append(max(float(vals.min()), q1 - 1.5 * iqr))
                del df, df_clean, vals
            except (DataFileError, PlotError, ValueError) as exc:
                logger.warning("Skipping sensor '%s': %s", summary.sensor_alias, exc)

        source = ColumnDataSource(dict(
            x=sensors, q1=q1s, median=medians, q3=q3s,
            mean=means, upper=uppers, lower=lowers,
        ))

        resolved_ylabel = ylabel or (f"Amplitude ({unit})" if unit else "Amplitude")
        p = figure(
            x_range=sensors,
            title=title or f"{self._study.title} \u2014 Sensor Amplitude Comparison",
            width=width or max(600, len(sensors) * 60),
            height=height or 450,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            y_axis_label=resolved_ylabel,
            x_axis_label=xlabel or "Sensor channel",
        )

        # Whiskers (extend to 1.5 \u00d7 IQR or data min/max, whichever comes first)
        p.add_layout(
            Whisker(source=source, base="x", upper="upper", lower="lower",
                    line_color="#555555")
        )

        # IQR box \u2014 two colour halves make the median clearly visible as a boundary
        p.vbar(x="x", top="median", bottom="q1", width=0.6, source=source,
               fill_color="#4C72B0", line_color="black", fill_alpha=0.9,
               legend_label="Q1 \u2013 median")
        p.vbar(x="x", top="q3", bottom="median", width=0.6, source=source,
               fill_color="#6B9FD4", line_color="black", fill_alpha=0.9,
               legend_label="median \u2013 Q3")

        # Mean marker
        p.circle(x="x", y="mean", size=9, source=source,
                 color="orangered", legend_label="Mean")

        p.add_tools(HoverTool(tooltips=[
            ("Sensor",       "@x"),
            ("Median",       "@median{0.00000}"),
            ("Mean",         "@mean{0.00000}"),
            ("Q1",           "@q1{0.00000}"),
            ("Q3",           "@q3{0.00000}"),
            ("Whisker low",  "@lower{0.00000}"),
            ("Whisker high", "@upper{0.00000}"),
        ]))
        p.legend.location = "top_right"
        return p

    def sensor_catalog(self) -> pd.DataFrame:
        """
        Return a DataFrame with one row per sensor channel (assay) in this study.

        Columns: assay_id, sensor_alias, measurement_type, technology_type,
                 technology_platform, n_runs, n_raw_files, n_processed_files,
                 fs_hz (when inferrable), unit (when inferrable).

        Returns
        -------
        pd.DataFrame
        """
        rows = []
        for a_model in self._study.assays:
            a = self._assay_proxy(a_model)
            row: dict = {
                "assay_id": a_model.assay_id,
                "sensor_alias": a_model.sensor.alias,
                "measurement_type": a_model.sensor.measurement_type,
                "technology_type": a_model.sensor.technology_type,
                "technology_platform": a_model.sensor.technology_platform,
                "n_runs": len(a_model.runs),
                "n_raw_files": len([r for r in a_model.runs if r.raw_file and r.raw_file.path]),
                "n_processed_files": len(
                    [r for r in a_model.runs if r.processed_file and r.processed_file.path]
                ),
            }
            fs = a._infer_fs()
            if fs is not None:
                row["fs_hz"] = fs
            unit = a._infer_unit()
            if unit is not None:
                row["unit"] = unit
            rows.append(row)
        return pd.DataFrame(rows)

    def export_labeled_dataset(
        self,
        file_type: Literal["raw", "processed", "auto"] = "raw",
        sensors: "list[str] | None" = None,
        outlier_method: "str | None" = None,
        outlier_upper: "float | None" = None,
        outlier_lower: "float | None" = None,
        outlier_strategy: str = "drop",
    ) -> pd.DataFrame:
        """
        Export all sensor time series annotated with ISA-PHM metadata labels.

        Every row of the returned DataFrame is one sample, tagged with its
        sensor identity and experimental condition — ready for ML pipelines.

        Columns: time, value, assay_id, sensor_alias, measurement_type,
                 run_id, run_number, <factor_name>…

        Parameters
        ----------
        file_type : "raw" | "processed" | "auto"
        sensors : list[str] | None
            Sensor aliases to include.  Defaults to all assays in the study.
        outlier_method : "iqr" | "zscore" | "fixed" | None
            When set, outlier correction is applied before export.
        outlier_upper : float | None
        outlier_lower : float | None
        outlier_strategy : "clip" | "nan" | "drop"

        Returns
        -------
        pd.DataFrame

        Notes
        -----
        Export diagnostics are attached to ``df.attrs['export_summary']`` with:
        ``n_target_assays``, ``n_total_runs``, ``n_loaded_runs``,
        ``n_skipped_runs``, and ``skipped_runs``.
        """
        target_assays = [
            a for a in self._study.assays
            if sensors is None or a.sensor.alias in sensors
        ]
        n_total_runs = sum(len(a.runs) for a in target_assays)
        n_skipped_runs = 0
        skipped_runs: list[dict[str, str]] = []
        frames: list[pd.DataFrame] = []
        for a_model in target_assays:
            a = self._assay_proxy(a_model)
            for run in a_model.runs:
                try:
                    df = self._integrator.load(
                        a_model,
                        study_id=self._study.study_id,
                        run_id=run.run_id,
                        file_type=file_type,
                    )
                except DataFileError as exc:
                    logger.warning(
                        "export_labeled_dataset: skipping assay '%s' run '%s': %s",
                        a_model.assay_id, run.run_id, exc,
                    )
                    n_skipped_runs += 1
                    skipped_runs.append(
                        {
                            "assay_id": a_model.assay_id,
                            "run_id": run.run_id,
                            "reason": str(exc),
                        }
                    )
                    continue
                if outlier_method is not None:
                    df = a.fix_outliers(
                        df,
                        method=outlier_method,
                        upper=outlier_upper,
                        lower=outlier_lower,
                        strategy=outlier_strategy,
                    )
                df = df.copy()
                df["assay_id"] = a_model.assay_id
                df["sensor_alias"] = a_model.sensor.alias
                df["measurement_type"] = a_model.sensor.measurement_type
                df["run_id"] = run.run_id
                df["run_number"] = run.run_number
                for factor_name, factor_val in run.factor_values.items():
                    df[factor_name] = factor_val
                frames.append(df)
        if not frames:
            out = pd.DataFrame()
        else:
            out = pd.concat(frames, ignore_index=True)

        if n_skipped_runs > 0:
            logger.warning(
                "export_labeled_dataset: skipped %d/%d runs due to file errors.",
                n_skipped_runs,
                n_total_runs,
            )
        out.attrs["export_summary"] = {
            "n_target_assays": len(target_assays),
            "n_total_runs": n_total_runs,
            "n_loaded_runs": n_total_runs - n_skipped_runs,
            "n_skipped_runs": n_skipped_runs,
            "skipped_runs": skipped_runs,
        }
        return out

    def load_multi_sensor_dataframe(
        self,
        sensors: "list[str] | None" = None,
        file_type: Literal["raw", "processed", "auto"] = "raw",
        run_id: "str | None" = None,
    ) -> pd.DataFrame:
        """
        Load multiple sensor channels and time-align them into a wide DataFrame.

        For single-run assays ``run_id`` may be omitted.  For multi-run assays
        an explicit ``run_id`` is required when there is more than one run.

        Columns: time, <sensor_alias_1>, <sensor_alias_2>, …

        The merge uses a nearest-neighbour join on ``time`` so sensors sampled
        at slightly different instants can still be aligned.

        Parameters
        ----------
        sensors : list[str] | None
            Sensor aliases to include.  Defaults to all assays in the study.
        file_type : "raw" | "processed" | "auto"
        run_id : str | None
            Run identifier for multi-run assays.

        Returns
        -------
        pd.DataFrame  Wide format: time + one column per sensor alias.
        """
        target_assays = [
            a for a in self._study.assays
            if sensors is None or a.sensor.alias in sensors
        ]
        dfs: list[tuple[str, pd.DataFrame]] = []
        for a_model in target_assays:
            try:
                df = self._integrator.load(
                    a_model,
                    study_id=self._study.study_id,
                    run_id=run_id,
                    file_type=file_type,
                )
                alias = a_model.sensor.alias
                dfs.append((alias, df[["time", "value"]].rename(columns={"value": alias})))
            except (DataFileError, AmbiguousRunError, RunNotFoundError) as exc:
                logger.warning(
                    "load_multi_sensor_dataframe: skipping '%s': %s",
                    a_model.assay_id, exc,
                )
        if not dfs:
            return pd.DataFrame()
        if len(dfs) == 1:
            alias, df = dfs[0]
            return df
        # Nearest-neighbour time merge across all sensor channels.
        _, merged = dfs[0]
        merged = merged.sort_values("time").reset_index(drop=True)
        for _, df in dfs[1:]:
            df = df.sort_values("time").reset_index(drop=True)
            merged = pd.merge_asof(merged, df, on="time", direction="nearest")
        return merged

    def assay(self, assay_id: str | int) -> "AssayProxy":
        """
        Navigate into a specific assay by 1-based integer index, filename, or assay_id.

        Parameters
        ----------
        assay_id : str | int
            1-based integer index, assay_id string (e.g. ``"a_st01_se01"``),
            or a case-insensitive partial match.

        Raises
        ------
        AssayNotFoundError
        """
        # Integer index (1-based).
        if isinstance(assay_id, int):
            assays = self._study.assays
            if not (1 <= assay_id <= len(assays)):
                raise AssayNotFoundError(
                    f"Assay index {assay_id} out of range for study "
                    f"'{self._study.title}'. Valid range: 1–{len(assays)}."
                )
            return self._assay_proxy(assays[assay_id - 1])

        a = self._assay_by_id.get(assay_id)
        if a is None:
            # Try partial / case-insensitive match.
            lower = assay_id.strip().lower()
            for key, val in self._assay_by_id.items():
                if key.lower() == lower or key.lower().endswith(lower):
                    a = val
                    break

        if a is None:
            available = sorted(self._assay_by_id.keys())
            raise AssayNotFoundError(
                f"Assay '{assay_id}' not found in study '{self._study.title}'. "
                f"Available assay IDs: {available}"
            )
        return self._assay_proxy(a)

    def assay_group(self, group: "AssayGroup") -> "_AssayGroupBinding":
        """
        Bind an :class:`AssayGroup` descriptor to this study.

        Returns an internal binding that provides ``lifecycle_features()``
        for a single study.  For multi-study comparisons, call
        :py:meth:`AssayGroup.compare_with` directly on the descriptor
        or use :py:meth:`ISAWrapper.compare_studies`.

        Parameters
        ----------
        group : AssayGroup
            A study-agnostic descriptor, e.g.
            ``AssayGroup([1, 2, 3], name="PE Accelerometer")``.
        """
        return _AssayGroupBinding(group, self)

    def lifecycle_features(
        self,
        group: "AssayGroup",
        file_type: Literal["raw", "processed", "auto"] = "raw",
        n_workers: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Lifecycle features for every assay in *group* within this study.

        Convenience shortcut for
        ``study.assay_group(group).lifecycle_features(...)``.

        Parameters
        ----------
        group : AssayGroup
            Sensor group descriptor.
        file_type : "raw" | "processed" | "auto"
        n_workers : int | None

        Returns
        -------
        dict[str, pd.DataFrame]
            Keyed by ``\"<StudyTitle> \u2014 <sensor alias>\"``.
        """
        return _AssayGroupBinding(group, self).lifecycle_features(
            file_type=file_type, n_workers=n_workers
        )

    def compare_with(
        self,
        other_studies: list["StudyProxy"],
        assay_id: str | int | None = None,
        assay_group: "AssayGroup | None" = None,
        file_type: Literal["raw", "processed", "auto"] = "raw",
        n_workers: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Load lifecycle features for the same sensor across this study and
        *other_studies*. Self is always included as the first entry.

        Parameters
        ----------
        other_studies : list[StudyProxy]
            Additional study proxies to compare against. The result order is
            deterministic: ``[self] + other_studies``.
        assay_id : str | int | None
            Single sensor.  Mutually exclusive with *assay_group*.
        assay_group : AssayGroup | None
            Named sensor group.
        file_type : "raw" | "processed" | "auto"
        n_workers : int | None

        Returns
        -------
        dict[str, pd.DataFrame]
        """
        if assay_id is not None and assay_group is not None:
            raise ValueError("Provide either assay_id or assay_group, not both.")
        out: dict[str, pd.DataFrame] = {}
        for sp in [self] + list(other_studies):
            if assay_group is not None:
                grp = sp.assay_group(assay_group)
                out.update(grp.lifecycle_features(file_type=file_type, n_workers=n_workers))
            else:
                aid = assay_id if assay_id is not None else 1
                lc = sp.assay(aid).lifecycle_features(file_type=file_type, n_workers=n_workers)
                out[sp.title] = lc
        return out

    def plot_sensor_lifecycle_correlation(
        self,
        assay_ids: "list[str | int] | None" = None,
        feature: str = "rms",
        file_type: Literal["raw", "processed", "auto"] = "raw",
        n_workers: int | None = None,
        title: str | None = None,
    ) -> object:
        """
        Correlation heatmap of *feature* across all sensor channels over all runs.

        Each column in the underlying matrix is one assay; each row is one run.
        The heatmap shows how sensors co-vary across the degradation trajectory.

        Parameters
        ----------
        assay_ids : list[str | int] | None
            Subset of assays to include.  None = all assays.
        feature : str
            Lifecycle feature column name (e.g. ``"rms"``, ``"kurtosis"``).
        file_type : "raw" | "processed" | "auto"
        n_workers : int | None

        Notes
        -----
        If one sensor fails to produce lifecycle features, it is skipped and
        a warning is logged. At least two valid sensors are required to build
        a correlation heatmap.
        """
        assay_list = self.list_assays()
        if assay_ids is not None:
            proxies = [self.assay(a) for a in assay_ids]
            labels  = [p.assay_id for p in proxies]
        else:
            proxies = [self._assay_proxy(self._assay_by_id[a.assay_id]) for a in assay_list]
            labels  = [
                a.sensor_alias or a.assay_id   # AssaySummary has sensor_alias
                for a in assay_list
            ]

        matrix: dict[str, pd.Series] = {}
        failures: list[str] = []
        for proxy, label in zip(proxies, labels):
            try:
                lc = proxy.lifecycle_features(file_type=file_type, n_workers=n_workers)
            except Exception as exc:
                msg = f"{label}: {type(exc).__name__}: {exc}"
                failures.append(msg)
                logger.warning(
                    "plot_sensor_lifecycle_correlation: study='%s', assay='%s', "
                    "feature='%s' failed (%s: %s)",
                    self.title,
                    label,
                    feature,
                    type(exc).__name__,
                    exc,
                )
                continue

            if feature not in lc.columns:
                msg = f"{label}: missing feature '{feature}'"
                failures.append(msg)
                logger.warning(
                    "plot_sensor_lifecycle_correlation: study='%s', assay='%s', "
                    "feature='%s' missing from lifecycle columns=%s",
                    self.title,
                    label,
                    feature,
                    list(lc.columns),
                )
                continue

            matrix[label] = lc.set_index("run_number")[feature]

        if len(matrix) < 2:
            if failures:
                detail = "; ".join(failures[:3])
                if len(failures) > 3:
                    detail += f"; +{len(failures) - 3} more"
            else:
                detail = f"Only {len(matrix)} valid sensor(s) available."
            raise PlotError(
                f"Need at least 2 valid sensors for feature '{feature}' in "
                f"study '{self.title}'. {detail}"
            )

        pivot = pd.DataFrame(matrix)
        pivot.columns = [str(c) for c in pivot.columns]
        return self._plotter.plot_correlation(
            pivot,
            columns=list(pivot.columns),
            title=title or f"{self.title} — Sensor Correlation ({feature})",
        )

    def plot_cross_correlation(
        self,
        assay_id_1: str | int,
        assay_id_2: str | int,
        run_id: str | None = None,
        max_lag: int = 200,
        file_type: Literal["raw", "processed", "auto"] = "raw",
        title: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> object:
        """
        Normalised cross-correlation between two sensors over a single run.

        Shows the lag (in samples) at which the two signals are most correlated.
        Useful for detecting that one sensor (e.g. temperature) leads or lags
        another (e.g. vibration) by a fixed number of samples.

        Parameters
        ----------
        assay_id_1, assay_id_2 : str | int
            The two assays to correlate.
        run_id : str | None
            Which run to use.  None = first run of assay_id_1.
        max_lag : int
            Maximum lag in samples to display (±max_lag).
        file_type : "raw" | "processed" | "auto"
        """
        import numpy as np
        from bokeh.plotting import figure as bokeh_figure
        from bokeh.models import Span

        a1 = self.assay(assay_id_1)
        a2 = self.assay(assay_id_2)

        # Select run_id — fall back to first run of assay 1.
        if run_id is None and a1._assay.runs:
            run_id = a1._assay.runs[0].run_id

        df1 = a1.load_dataframe(run_id=run_id, file_type=file_type)
        df2 = a2.load_dataframe(run_id=run_id, file_type=file_type)

        col1 = [c for c in df1.columns if c not in ("packet_ts", "time")][0]
        col2 = [c for c in df2.columns if c not in ("packet_ts", "time")][0]

        v1 = df1[col1].dropna().to_numpy(dtype=float)
        v2 = df2[col2].dropna().to_numpy(dtype=float)

        # Trim to equal length.
        n = min(len(v1), len(v2))
        v1, v2 = v1[:n], v2[:n]

        # Zero-mean normalise.
        v1 = (v1 - v1.mean()) / (v1.std() + 1e-12)
        v2 = (v2 - v2.mean()) / (v2.std() + 1e-12)

        # Full cross-correlation, normalised by n.
        xcorr = np.correlate(v1, v2, mode="full") / n
        lags  = np.arange(-len(v1) + 1, len(v1))

        # Clip to ±max_lag.
        mask   = np.abs(lags) <= max_lag
        lags   = lags[mask]
        xcorr  = xcorr[mask]

        peak_lag = int(lags[np.argmax(np.abs(xcorr))])

        alias1 = a1._assay.sensor.alias or str(assay_id_1)
        alias2 = a2._assay.sensor.alias or str(assay_id_2)

        p = bokeh_figure(
            width=width or 900,
            height=height or 350,
            title=title or f"Cross-correlation: {alias1} vs {alias2}  (peak lag = {peak_lag} samples)",
            x_axis_label="Lag (samples)",
            y_axis_label="Normalised correlation",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p.line(lags.tolist(), xcorr.tolist(), line_width=1.5, color="#4C72B0")
        p.add_layout(Span(location=0, dimension="height", line_color="grey", line_dash="dashed"))
        p.add_layout(Span(location=peak_lag, dimension="height", line_color="firebrick", line_dash="dashed", line_width=1.5))
        p.title.text_font_size = "13pt"
        return p


# ---------------------------------------------------------------------------
# AssayGroup  (study-agnostic named descriptor)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, init=False, repr=False)
class AssayGroup:
    """
    An immutable, study-agnostic descriptor for a logical group of assays.

    ``AssayGroup`` holds only a name and a list of assay identifiers — it
    carries no data and is not bound to any study.  Define it once at the
    top of a notebook and pass it into any study-level method.

    Parameters
    ----------
    assay_ids : list[str | int]
        Assay IDs (strings) or 1-based integer indices.
    name : str
        Human-readable label used in plot legends and dict keys.

    Example
    -------
    ::

        from isa_phm import AssayGroup

        PE_ACCEL = AssayGroup([9, 10, 11], name="PE Accelerometer (X/Y/Z)")
        MEMS     = AssayGroup([21, 22, 23], name="MEMS Accelerometer")

        # Single study — fluent chain:
        s1.lifecycle_features(group=PE_ACCEL, file_type="raw")

        # Multi-study — call compare_with directly on the descriptor:
        PE_ACCEL.compare_with([s1, s2, s3], file_type="raw")

        # Or use the wrapper shorthand:
        wrapper.compare_studies(["Bearing 2_1", "Bearing 2_2"], assay_group=PE_ACCEL)
    """

    name: str
    assay_ids: tuple[str | int, ...]

    def __init__(self, assay_ids: list[str | int], name: str) -> None:
        if not assay_ids:
            raise ValueError("AssayGroup requires at least one assay_id.")
        object.__setattr__(self, "assay_ids", tuple(assay_ids))
        object.__setattr__(self, "name", name)

    def __repr__(self) -> str:
        return f"AssayGroup(name={self.name!r}, assay_ids={list(self.assay_ids)})"

    def compare_with(
        self,
        studies: "list[StudyProxy]",
        file_type: "Literal['raw', 'processed', 'auto']" = "raw",
        n_workers: "int | None" = None,
    ) -> "dict[str, pd.DataFrame]":
        """
        Lifecycle features for this group across all provided studies.

        No study binding required — call directly on the :class:`AssayGroup`
        descriptor.  All studies are explicit: what you pass is exactly
        what appears in the result.

        Parameters
        ----------
        studies : list[StudyProxy]
            Studies to include.
        file_type : "raw" | "processed" | "auto"
        n_workers : int | None

        Returns
        -------
        dict[str, pd.DataFrame]
            Keyed by ``\"<StudyTitle> — <sensor alias>\"``.
        """
        out: dict[str, pd.DataFrame] = {}
        for sp in studies:
            grp = sp.assay_group(self)
            out.update(grp.lifecycle_features(file_type=file_type, n_workers=n_workers))
        return out


# ---------------------------------------------------------------------------
# _AssayGroupBinding  (AssayGroup bound to a specific study — internal)
# ---------------------------------------------------------------------------

class _AssayGroupBinding:
    """
    A live binding of an :class:`AssayGroup` to a specific :class:`StudyProxy`.

    Internal object — acquire via ``study.assay_group(group)`` or
    ``study.lifecycle_features(group=group)``.  Provides
    ``lifecycle_features()`` over all member assays for a single study.

    For multi-study comparisons call :py:meth:`AssayGroup.compare_with`
    directly on the descriptor instead.
    """

    def __init__(self, group: AssayGroup, study: "StudyProxy") -> None:
        self._group   = group
        self._study   = study
        self._proxies = [study.assay(aid) for aid in group.assay_ids]

    def lifecycle_features(
        self,
        file_type: Literal["raw", "processed", "auto"] = "raw",
        n_workers: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Compute lifecycle features for every member assay.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keyed by ``\"<StudyTitle> — <sensor alias>\"``.
        """
        out: dict[str, pd.DataFrame] = {}
        for proxy in self._proxies:
            alias = proxy._assay.sensor.alias or proxy.assay_id
            key   = f"{self._study.title} — {alias}"
            out[key] = proxy.lifecycle_features(file_type=file_type, n_workers=n_workers)
        return out


# ---------------------------------------------------------------------------
# AssayProxy
# ---------------------------------------------------------------------------

class AssayProxy:
    """
    Assay (sensor channel) level proxy.

    Provides data loading, lifecycle feature computation, and all six plots.
    Acquire via: ``wrapper.study("…").assay("a_st01_se01")``
    """

    def __init__(
        self,
        assay: AssayModel,
        study: StudyModel,
        investigation: InvestigationModel,
        integrator: "DataIntegrator",
        plotter: "ISAPlotter",
        semantic: "SemanticNormalizer",
    ) -> None:
        self._assay = assay
        self._study = study
        self._inv = investigation
        self._integrator = integrator
        self._plotter = plotter
        self._semantic = semantic

    # ------------------------------------------------------------------
    # Informational
    # ------------------------------------------------------------------

    @property
    def assay_id(self) -> str:
        return self._assay.assay_id

    @property
    def run_count(self) -> int:
        return len(self._assay.runs)

    def overview(self) -> AssayOverview:
        """Return a detailed summary of this assay."""
        from .schemas import RunSummary

        run_summaries = [
            RunSummary(
                run_id=r.run_id,
                run_number=r.run_number,
                raw_file_path=r.raw_file.path if r.raw_file else None,
                processed_file_path=r.processed_file.path if r.processed_file else None,
                n_measurement_params=len(r.measurement_params),
                n_processing_params=len(r.processing_params),
                factor_values=r.factor_values,
            )
            for r in self._assay.runs
        ]
        return AssayOverview(
            assay_id=self._assay.assay_id,
            sensor_id=self._assay.sensor.sensor_id,
            sensor_alias=self._assay.sensor.alias,
            technology_type=self._assay.sensor.technology_type,
            technology_platform=self._assay.sensor.technology_platform,
            measurement_type=self._assay.sensor.measurement_type,
            n_runs=len(self._assay.runs),
            runs=run_summaries,
        )

    def list_runs(self) -> list[RunSummary]:
        """Return a summary row per run."""
        return self.overview().runs

    def list_measurement_params(self, run_id: str | None = None) -> pd.DataFrame:
        """
        Return the measurement protocol parameters for a run as a DataFrame.

        Columns: ``parameter_name``, ``value``, ``unit``.

        Uses the first run when ``run_id`` is omitted.  Returns an empty
        DataFrame (with correct columns) when no parameters are defined.
        """
        run = self._first_or_run(run_id)
        if run is None or not run.measurement_params:
            return pd.DataFrame(columns=["parameter_name", "value", "unit"])
        rows = [
            {"parameter_name": pv.parameter_name, "value": pv.value, "unit": pv.unit or ""}
            for pv in run.measurement_params
        ]
        return pd.DataFrame(rows, columns=["parameter_name", "value", "unit"])

    def list_processing_params(self, run_id: str | None = None) -> pd.DataFrame:
        """
        Return the processing protocol parameters for a run as a DataFrame.

        Columns: ``parameter_name``, ``value``, ``unit``.

        Uses the first run when ``run_id`` is omitted.  Returns an empty
        DataFrame (with correct columns) when no parameters are defined.
        """
        run = self._first_or_run(run_id)
        if run is None or not run.processing_params:
            return pd.DataFrame(columns=["parameter_name", "value", "unit"])
        rows = [
            {"parameter_name": pv.parameter_name, "value": pv.value, "unit": pv.unit or ""}
            for pv in run.processing_params
        ]
        return pd.DataFrame(rows, columns=["parameter_name", "value", "unit"])

    def semantic_parameters(
        self,
        run_id: str | None = None,
    ) -> dict[str, list[SemanticField]]:
        """Return normalized semantic labels for measurement/processing parameters."""
        run = self._first_or_run(run_id)
        return self._semantic.normalize_assay_parameters(self._assay, run=run)

    def columns(self) -> list[str]:
        """Return the standard DataFrame column names produced by load_dataframe()."""
        from .integrator import STANDARD_COLUMNS

        return list(STANDARD_COLUMNS)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_dataframe(
        self,
        run_id: str | None = None,
        file_type: Literal["raw", "processed", "auto"] = "processed",
    ) -> pd.DataFrame:
        """
        Load a run's measurement file into a DataFrame.

        For single-run assays ``run_id`` may be omitted.

        Parameters
        ----------
        run_id : str | None
            Run identifier.  Omit for single-run (diagnostic) assays.
        file_type : "processed" | "raw" | "auto"
            Which data file to load.  Defaults to ``"processed"``.
            Use ``"auto"`` to prefer processed and fall back to raw.

        Returns
        -------
        pd.DataFrame  (columns from STANDARD_COLUMNS)

        Raises
        ------
        AmbiguousRunError   — multi-run assay and run_id is None.
        RunNotFoundError    — explicit run_id not present.
        DataFileError       — file missing or unreadable (neither raw nor processed available).
        """
        return self._integrator.load(
            self._assay,
            study_id=self._study.study_id,
            run_id=run_id,
            file_type=file_type,
        )

    def load_dataframe_with_meta(
        self,
        run_id: str | None = None,
        file_type: Literal["raw", "processed", "auto"] = "processed",
    ) -> tuple[pd.DataFrame, DataLoadMetadata]:
        """Load data and return DataFrame plus resolved file-load metadata."""
        return self._integrator.load_with_meta(
            self._assay,
            study_id=self._study.study_id,
            run_id=run_id,
            file_type=file_type,
        )

    def lifecycle_features(self, file_type: Literal["raw", "processed", "auto"] = "processed", n_workers: int | None = None) -> pd.DataFrame:
        """
        Compute scalar features for every run (parallel CSV loading for speed).

        Parameters
        ----------
        file_type : "processed" | "raw" | "auto"
        n_workers : int | None
            Number of threads.  None = auto (min(32, cpu_count+4)).  1 = sequential.

        Returns
        -------
        pd.DataFrame with columns:
            run_id, run_number, study_id, assay_id,
            rms, max, mean, peak2peak, kurtosis, std, crest_factor, skewness,
            fv_<factor_name>…
        """
        return self._integrator.lifecycle_features_df(
            self._assay, study_id=self._study.study_id, file_type=file_type, n_workers=n_workers
        )

    def to_ml_dataset(
        self,
        label_column: str | None = None,
        feature_columns: list[str] | None = None,
        file_type: Literal["raw", "processed", "auto"] = "processed",
        n_workers: int | None = None,
    ) -> pd.DataFrame:
        """
        Return a clean, ML-ready DataFrame built from lifecycle features.

        Each row is one run.  Feature columns are the scalar lifecycle statistics
        (rms, kurtosis, etc.).  An optional *label_column* names the regression
        or classification target — pulled from the factor-value columns
        (``fv_*``) already present in the lifecycle DataFrame.

        No modelling is done here.  This method only shapes and cleans the data
        so it can be passed directly to scikit-learn, XGBoost, or any other
        ML framework::

            X = assay.to_ml_dataset(label_column="fv_Bearing Lifetime")
            y = X.pop("target")
            # → X, y ready for sklearn

        Parameters
        ----------
        label_column : str | None
            Factor-value column to use as the target (e.g.
            ``"fv_Bearing Lifetime"``).  When provided it is renamed to
            ``"target"`` and moved to the last column.  When None the
            DataFrame is returned without a target column.
        feature_columns : list[str] | None
            Explicit list of feature columns to keep.  Defaults to the
            eight scalar features: rms, max, mean, peak2peak, kurtosis,
            std, crest_factor, skewness.
        file_type : "processed" | "raw" | "auto"
        n_workers : int | None

        Returns
        -------
        pd.DataFrame
            Columns: run_id, run_number, <features>, [target]
        """
        _DEFAULT_FEATURES = [
            "rms", "max", "mean", "peak2peak",
            "kurtosis", "std", "crest_factor", "skewness",
        ]
        lc = self.lifecycle_features(file_type=file_type, n_workers=n_workers)

        keep = ["run_id", "run_number"]
        feat_cols = feature_columns or _DEFAULT_FEATURES
        keep += [c for c in feat_cols if c in lc.columns]

        if label_column is not None:
            if label_column not in lc.columns:
                available = [c for c in lc.columns if c.startswith("fv_")]
                raise ValueError(
                    f"label_column '{label_column}' not found. "
                    f"Available factor-value columns: {available}"
                )
            keep.append(label_column)

        df = lc[keep].copy()
        df = df.dropna(subset=[c for c in feat_cols if c in df.columns])

        if label_column is not None:
            df = df.rename(columns={label_column: "target"})

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Quality checks
    # ------------------------------------------------------------------

    def missing_values_report(
        self,
        run_id: str | None = None,
        file_type: Literal["raw", "processed", "auto"] = "processed",
    ) -> MissingValuesReport:
        """
        Return a textual report of missing / NaN values.

        For single-run assays omit ``run_id``.
        """
        df = self.load_dataframe(run_id=run_id, file_type=file_type)
        numeric = df.select_dtypes(include=["number"])
        n_rows = len(df)
        n_missing = int(numeric.isnull().sum().sum())
        pct = round(n_missing / max(n_rows * len(numeric.columns), 1) * 100, 2)
        by_col = {col: int(numeric[col].isnull().sum()) for col in numeric.columns}
        return MissingValuesReport(
            n_rows=n_rows,
            n_cols=len(numeric.columns),
            n_missing=n_missing,
            pct_missing=pct,
            by_column=by_col,
        )

    # ------------------------------------------------------------------
    # Navigation into runs
    # ------------------------------------------------------------------

    def run(self, run_id: str) -> "RunProxy":
        """
        Navigate into a specific run.

        Raises
        ------
        RunNotFoundError
        """
        record = self._assay.get_run(run_id)
        if record is None:
            available = [r.run_id for r in self._assay.runs]
            raise RunNotFoundError(
                f"run_id '{run_id}' not found in assay '{self._assay.assay_id}'. "
                f"Available: {available}"
            )
        return RunProxy(record, self._assay, self._study, self._inv, self._integrator, self._plotter)

    # ------------------------------------------------------------------
    # Plots (all six MVP plots)
    # ------------------------------------------------------------------

    def plot_distribution(
        self,
        df: "pd.DataFrame | None" = None,
        run_id: str | None = None,
        column: str = "value",
        bins: int = 50,
        file_type: Literal["raw", "processed", "auto"] = "processed",
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> object:
        """
        Amplitude histogram + KDE for one run.

        Parameters
        ----------
        df : pd.DataFrame | None
            Pre-loaded (and optionally pre-cleaned) DataFrame.  When provided,
            ``run_id`` and ``file_type`` are ignored for data loading but the
            run label is still derived from ``run_id`` if given.
        run_id : str | None
        column : str
        bins : int
        file_type : "processed" | "raw" | "auto"
        """
        if df is None:
            df = self.load_dataframe(run_id=run_id, file_type=file_type)
        label = run_id or (self._assay.runs[0].run_id if self._assay.runs else "run_01")
        return self._plotter.plot_distribution(
            df,
            column=column,
            bins=bins,
            title=title or f"{self._assay.assay_id} / {label} — Distribution of '{column}'",
            xlabel=xlabel,
            ylabel=ylabel,
            width=width,
            height=height,
        )

    def plot_lifecycle(
        self,
        feature: str = "rms",
        file_type: Literal["raw", "processed", "auto"] = "processed",
        n_workers: int | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> object:
        """
        Lifecycle curve — scalar feature over all runs.

        Raises
        ------
        PlotError   If lifecycle data could not be computed.
        """
        lc = self.lifecycle_features(file_type=file_type, n_workers=n_workers)
        if lc.empty:
            raise PlotError(
                f"No lifecycle data available for assay '{self._assay.assay_id}'. "
                f"Ensure processed data files exist and are readable."
            )
        return self._plotter.plot_lifecycle(
            lc,
            feature=feature,
            title=title or f"{self._assay.assay_id} — Lifecycle {feature.upper()}",
            xlabel=xlabel,
            ylabel=ylabel,
            width=width,
            height=height,
        )

    def plot_frequency_domain(
        self,
        df: "pd.DataFrame | None" = None,
        run_id: str | None = None,
        fs: float | None = None,
        column: str = "value",
        file_type: Literal["raw", "processed", "auto"] = "processed",
        log_scale: bool = True,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> object:
        """
        FFT spectrum for one run.

        Parameters
        ----------
        df : pd.DataFrame | None
            Pre-loaded (and optionally pre-cleaned) DataFrame.  Pass ``df_clean``
            here to exclude overflow/outlier rows before the FFT — otherwise those
            values will dominate the spectrum.  When omitted the data is loaded
            from the file.
        run_id : str | None
        fs : float | None
            Sampling frequency in Hz.  Auto-inferred from protocol parameters
            if not provided.
        column : str
        file_type : "processed" | "raw" | "auto"
        log_scale : bool
            True (default) — magnitude in dB; good for spotting fault sidebands.
            False — linear amplitude; easier to read peak values in signal units.
        """
        if fs is None:
            fs = self._infer_fs()
        if df is None:
            df = self.load_dataframe(run_id=run_id, file_type=file_type)
        label = run_id or (self._assay.runs[0].run_id if self._assay.runs else "")
        return self._plotter.plot_frequency_domain(
            df,
            fs=fs,
            column=column,
            log_scale=log_scale,
            title=title or f"{self._assay.assay_id} / {label} — FFT Spectrum",
            xlabel=xlabel,
            ylabel=ylabel,
            width=width,
            height=height,
        )

    def plot_psd(
        self,
        run_id: str | None = None,
        fs: float | None = None,
        column: str = "value",
        nperseg: int = 1024,
        window: str = "hann",
        file_type: Literal["raw", "processed", "auto"] = "processed",
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> object:
        """
        Welch Power Spectral Density for one run.

        Smoother than a raw FFT for noisy or short signals because it averages
        overlapping periodograms (Welch's method).

        Parameters
        ----------
        run_id : str | None
        fs : float | None
            Sampling frequency in Hz.  Auto-inferred from protocol parameters
            if not provided.
        column : str
        nperseg : int
            Length of each Welch segment (default 1024).
        window : str
            Window function name passed to ``scipy.signal.welch`` (default "hann").
        file_type : "processed" | "raw" | "auto"
        """
        if fs is None:
            fs = self._infer_fs()
        if fs is None:
            raise PlotError(
                "fs (sampling frequency) is required for plot_psd. "
                "Pass fs= explicitly or ensure the ISA-JSON protocol parameters "
                "include sampling frequency in Hz."
            )
        df = self.load_dataframe(run_id=run_id, file_type=file_type)
        label = run_id or (self._assay.runs[0].run_id if self._assay.runs else "")
        return self._plotter.plot_power_spectral_density(
            df,
            fs=fs,
            column=column,
            nperseg=nperseg,
            window=window,
            title=title or f"{self._assay.assay_id} / {label} — Welch PSD",
            xlabel=xlabel,
            ylabel=ylabel,
            width=width,
            height=height,
        )

    def plot_spectrogram(
        self,
        run_id: str | None = None,
        fs: float | None = None,
        column: str = "value",
        nperseg: int = 256,
        overlap: float = 0.75,
        file_type: Literal["raw", "processed", "auto"] = "processed",
        title: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> object:
        """
        Short-Time Fourier Transform spectrogram for one run.

        Renders a time × frequency heatmap showing how the signal's frequency
        content changes within a single measurement run.

        Parameters
        ----------
        run_id : str | None
        fs : float | None
            Sampling frequency in Hz.  Auto-inferred if not provided.
        column : str
        nperseg : int
            FFT window length in samples (default 256).
        overlap : float
            Fraction of window overlap 0–1 (default 0.75).
        file_type : "processed" | "raw" | "auto"
        """
        if fs is None:
            fs = self._infer_fs()
        if fs is None:
            raise PlotError(
                "fs (sampling frequency) is required for plot_spectrogram. "
                "Pass fs= explicitly or ensure the ISA-JSON protocol parameters "
                "include sampling frequency in Hz."
            )
        df = self.load_dataframe(run_id=run_id, file_type=file_type)
        label = run_id or (self._assay.runs[0].run_id if self._assay.runs else "")
        return self._plotter.plot_spectrogram(
            df,
            fs=fs,
            column=column,
            nperseg=nperseg,
            overlap=overlap,
            title=title or f"{self._assay.assay_id} / {label} — Spectrogram",
            width=width,
            height=height,
        )

    def plot_waterfall(
        self,
        run_ids: list[str] | None = None,
        fs: float | None = None,
        column: str = "value",
        n_runs: int = 10,
        nperseg: int = 1024,
        offset_scale: float = 0.3,
        file_type: Literal["raw", "processed", "auto"] = "processed",
        title: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> object:
        """
        Waterfall plot — stacked FFT spectra across multiple runs.

        Shows how the frequency content evolves over the bearing lifecycle.

        Parameters
        ----------
        run_ids : list[str] | None
            Specific run IDs to include.  If None, evenly samples *n_runs*
            runs across the full lifecycle.
        fs : float | None
            Sampling frequency in Hz.  Auto-inferred if not provided.
        n_runs : int
            Number of runs to sample when *run_ids* is None (default 10).
        nperseg : int
            FFT window length for Welch smoothing.
        offset_scale : float
            Vertical spacing between traces as a fraction of the PSD range.
        file_type : "processed" | "raw" | "auto"
        """
        if fs is None:
            fs = self._infer_fs()
        if fs is None:
            raise PlotError(
                "fs (sampling frequency) is required for plot_waterfall. "
                "Pass fs= explicitly or ensure the ISA-JSON protocol parameters "
                "include sampling frequency in Hz."
            )
        all_runs = self._assay.runs
        if run_ids is not None:
            selected = [r for r in all_runs if r.run_id in run_ids]
        else:
            step = max(1, len(all_runs) // n_runs)
            selected = all_runs[::step][:n_runs]

        dfs: dict[str, pd.DataFrame] = {}
        for run in selected:
            try:
                df = self.load_dataframe(run_id=run.run_id, file_type=file_type)
                dfs[f"Run {run.run_number}"] = df
            except Exception:
                pass

        return self._plotter.plot_waterfall(
            dfs,
            fs=fs,
            column=column,
            nperseg=nperseg,
            offset_scale=offset_scale,
            title=title or f"{self._assay.assay_id} — Waterfall (FFT evolution)",
            width=width,
            height=height,
        )

    def plot_correlation(
        self,
        columns: list[str] | None = None,
        file_type: Literal["raw", "processed", "auto"] = "processed",
        n_workers: int | None = None,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> object:
        """
        Correlation heatmap of lifecycle scalar features across all runs.

        Raises
        ------
        PlotError   If fewer than 2 runs could be loaded.
        """
        lc = self.lifecycle_features(file_type=file_type, n_workers=n_workers)
        if lc.empty:
            raise PlotError(
                f"No lifecycle data available for assay '{self._assay.assay_id}'."
            )
        if len(lc) < 2:
            raise PlotError(
                f"plot_correlation requires at least 2 runs; "
                f"assay '{self._assay.assay_id}' produced {len(lc)} row(s). "
                f"Use a multi-run dataset."
            )
        return self._plotter.plot_correlation(
            lc,
            columns=columns,
            title=title or f"{self._assay.assay_id} — Feature Correlation",
            xlabel=xlabel,
            ylabel=ylabel,
            width=width,
            height=height,
        )

    def plot_variability(
        self,
        run_ids: list[str] | None = None,
        value_column: str = "value",
        file_type: Literal["raw", "processed", "auto"] = "processed",
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> object:
        """
        Boxplot of sensor amplitude per run.

        Loads all (or the specified subset of) runs and builds a combined DataFrame.
        For large run-count assays, prefer specifying ``run_ids`` to limit memory use.
        """
        target_runs = self._assay.runs
        if run_ids is not None:
            target_runs = [r for r in target_runs if r.run_id in set(run_ids)]

        frames: list[pd.DataFrame] = []
        for run in target_runs:
            try:
                df = self._integrator.load(
                    self._assay,
                    study_id=self._study.study_id,
                    run_id=run.run_id,
                    file_type=file_type,
                )
                df = df.copy()
                df["run_id"] = run.run_id
                frames.append(df)
            except (DataFileError, ValueError) as exc:
                logger.warning("Skipping run '%s': %s", run.run_id, exc)

        if not frames:
            raise PlotError(
                f"No data could be loaded for assay '{self._assay.assay_id}'. "
                f"Check data_root and file paths."
            )

        combined = pd.concat(frames, ignore_index=True)
        return self._plotter.plot_variability(
            combined,
            value_column=value_column,
            group_by="run_id",
            title=title or f"{self._assay.assay_id} — Amplitude Variability",
            xlabel=xlabel,
            ylabel=ylabel,
            width=width,
            height=height,
        )

    def plot_missing_values(
        self,
        run_id: str | None = None,
        file_type: Literal["raw", "processed", "auto"] = "processed",
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> object:
        """Heatmap of NaN presence in a single run's time series."""
        df = self.load_dataframe(run_id=run_id, file_type=file_type)
        label = run_id or (self._assay.runs[0].run_id if self._assay.runs else "")
        return self._plotter.plot_missing_values(
            df,
            title=title or f"{self._assay.assay_id} / {label} — Missing Values",
            xlabel=xlabel,
            ylabel=ylabel,
            width=width,
            height=height,
        )

    def plot_timeseries(
        self,
        run_id: str | None = None,
        file_type: Literal["raw", "processed", "auto"] = "processed",
        outlier_mask: "np.ndarray | None" = None,
        show_outliers: bool = False,
        outlier_method: str = "iqr",
        outlier_threshold: float = 3.0,
        outlier_lower: "float | None" = None,
        outlier_upper: "float | None" = None,
        max_points: int = 10_000,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> object:
        """
        Time-domain waveform for one run.

        Parameters
        ----------
        run_id : str | None
            Omit for single-run (diagnostic) assays.
        file_type : "processed" | "raw" | "auto"
        show_outliers : bool
            When True, auto-detect outliers and highlight them in red.
            Ignored if ``outlier_mask`` is provided explicitly.
        outlier_method : "iqr" | "zscore" | "fixed"
            Detection method used when ``show_outliers=True``.
        outlier_threshold : float
            Multiplier for the chosen method (default 3.0).
        outlier_lower : float | None
            Hard lower bound for outlier detection (overrides computed bound).
        outlier_upper : float | None
            Hard upper bound for outlier detection (overrides computed bound).
            Use e.g. ``outlier_upper=1e7`` to flag only sensor overflow values.
        outlier_mask : np.ndarray | None
            Pre-computed boolean mask (overrides show_outliers).
        max_points : int
            Down-sample to this many points for responsive rendering.
        """
        from .utils import detect_outliers as _detect

        df = self.load_dataframe(run_id=run_id, file_type=file_type)
        label = run_id or (self._assay.runs[0].run_id if self._assay.runs else "")

        if show_outliers and outlier_mask is None:
            values_for_detect = df["value"].fillna(df["value"].median()).to_numpy(dtype=float)
            outlier_mask, _ = _detect(
                values_for_detect,
                method=outlier_method,
                threshold=outlier_threshold,
                lower=outlier_lower,
                upper=outlier_upper,
            )

        return self._plotter.plot_timeseries(
            df,
            outlier_mask=outlier_mask,
            max_points=max_points,
            xlabel=xlabel or "time",
            ylabel=ylabel or self._ylabel(),
            title=title or f"{self._assay.assay_id} / {label} — Waveform",
            width=width,
            height=height,
        )

    def plot_outlier_comparison(
        self,
        df_original: pd.DataFrame,
        df_clean: pd.DataFrame,
        strategy: str = "clip",
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> object:
        """
        Side-by-side before/after waveform after :py:meth:`fix_outliers`.

        Parameters
        ----------
        df_original : pd.DataFrame
            The unmodified DataFrame from :py:meth:`load_dataframe`.
        df_clean : pd.DataFrame
            The corrected DataFrame from :py:meth:`fix_outliers`.
        strategy : str
            Label shown in the plot title (default ``"clip"``).
        """
        return self._plotter.plot_outlier_comparison(
            df_original,
            df_clean,
            strategy=strategy,
            xlabel=xlabel,
            ylabel=ylabel or self._ylabel(),
            title=title or f"{self._assay.assay_id} — Outlier correction ({strategy})",
            width=width,
            height=height,
        )

    # ------------------------------------------------------------------
    # Signal quality: outlier detection & correction
    # ------------------------------------------------------------------

    def detect_outliers(
        self,
        run_id: str | None = None,
        file_type: Literal["raw", "processed", "auto"] = "processed",
        method: str = "iqr",
        threshold: float = 3.0,
        lower: "float | None" = None,
        upper: "float | None" = None,
        column: str = "value",
    ) -> "OutlierReport":
        """
        Detect outliers in a run's signal and return an :class:`OutlierReport`.

        Parameters
        ----------
        run_id : str | None
        file_type : "processed" | "raw" | "auto"
        method : "iqr" | "zscore" | "fixed"
            ``"iqr"``    bounds = Q1/Q3 Â± threshold Ã— IQR.
            ``"zscore"`` bounds = mean Â± threshold Ã— std.
            ``"fixed"``  explicit bounds only; set ``lower`` and/or ``upper``.
        threshold : float
            Multiplier for the chosen method (default 3.0).
        lower : float | None
            Hard lower bound override.  Values below this are outliers.
        upper : float | None
            Hard upper bound override.  Values above this are outliers.
            Example: ``upper=1e7`` to flag only sensor overflow values.
        column : str
            Signal column to inspect (default ``"value"``).

        Returns
        -------
        OutlierReport
        """
        from .schemas import OutlierReport
        from .utils import detect_outliers as _detect

        df = self.load_dataframe(run_id=run_id, file_type=file_type)
        values = df[column].dropna().to_numpy(dtype=float)
        mask, stats = _detect(values, method=method, threshold=threshold, lower=lower, upper=upper)

        n_out = int(mask.sum())
        pct = round(n_out / max(len(mask), 1) * 100, 3)
        return OutlierReport(
            n_outliers=n_out,
            pct_outliers=pct,
            method=method,
            threshold=threshold,
            by_column={column: {**stats, "n_outliers": n_out}},
        )

    def fix_outliers(
        self,
        df: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 3.0,
        lower: "float | None" = None,
        upper: "float | None" = None,
        strategy: str = "clip",
        column: str = "value",
    ) -> pd.DataFrame:
        """
        Return a copy of ``df`` with outliers in ``column`` corrected.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame returned by :py:meth:`load_dataframe`.
        method : "iqr" | "zscore" | "fixed"
            ``"fixed"`` uses explicit ``lower``/``upper`` bounds only.
        threshold : float
        lower : float | None
            Hard lower bound override.  Values below this are treated as outliers.
        upper : float | None
            Hard upper bound override.  Values above this are treated as outliers.
            Example: ``upper=1e7`` to remove only sensor overflow values.
        strategy : "clip" | "nan" | "drop"
            ``"clip"`` — clamp to detection bounds (default).
            ``"nan"``  — replace with NaN.
            ``"drop"`` — remove outlier rows.
        column : str
            Signal column to correct (default ``"value"``).

        Returns
        -------
        pd.DataFrame
            Copy of ``df`` with outliers corrected in ``column``.
        """
        from .utils import detect_outliers as _detect, fix_outliers as _fix

        values = df[column].to_numpy(dtype=float)
        mask, stats = _detect(values, method=method, threshold=threshold, lower=lower, upper=upper)
        fixed = _fix(
            values, mask, strategy=strategy,
            bounds=(stats["lower_bound"], stats["upper_bound"]),
        )

        result = df.copy()
        if strategy == "drop":
            result = result[~mask].reset_index(drop=True)
        else:
            result[column] = fixed
        return result

    def fill_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "interpolate",
    ) -> pd.DataFrame:
        """
        Return a copy of ``df`` with NaN values filled using the chosen strategy.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame returned by :py:meth:`load_dataframe`.
        strategy : "interpolate" | "ffill" | "bfill" | "mean" | "zero"
            ``"interpolate"`` — linear interpolation between adjacent samples (default).
            ``"ffill"``       — forward-fill from the last valid sample.
            ``"bfill"``       — backward-fill from the next valid sample.
            ``"mean"``        — replace every NaN with the column mean.
            ``"zero"``        — replace every NaN with 0.

        Returns
        -------
        pd.DataFrame
            Copy of ``df`` with NaN values filled in all numeric columns.
        """
        result = df.copy()
        numeric_cols = result.select_dtypes(include=["number"]).columns
        if strategy == "interpolate":
            result[numeric_cols] = result[numeric_cols].interpolate(
                method="linear", limit_direction="both"
            )
        elif strategy == "ffill":
            result[numeric_cols] = result[numeric_cols].ffill()
        elif strategy == "bfill":
            result[numeric_cols] = result[numeric_cols].bfill()
        elif strategy == "mean":
            for col in numeric_cols:
                result[col] = result[col].fillna(result[col].mean())
        elif strategy == "zero":
            result[numeric_cols] = result[numeric_cols].fillna(0.0)
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose one of: 'interpolate', 'ffill', 'bfill', 'mean', 'zero'."
            )
        return result

    def sensor_info(self) -> dict:
        """
        Return a dictionary of all known metadata for this sensor channel.

        Keys always present
        -------------------
        sensor_alias, assay_id, measurement_type, technology_type,
        technology_platform, n_runs, n_raw_files, n_processed_files

        Optional keys (resolved from ISA-JSON protocol parameters)
        ----------------------------------------------------------
        fs_hz, unit, measurement_params, factor_names

        Returns
        -------
        dict
        """
        n_raw = len([r for r in self._assay.runs if r.raw_file and r.raw_file.path])
        n_proc = len([r for r in self._assay.runs if r.processed_file and r.processed_file.path])
        info: dict = {
            "sensor_alias": self._assay.sensor.alias,
            "sensor_id": self._assay.sensor.sensor_id,
            "assay_id": self._assay.assay_id,
            "measurement_type": self._assay.sensor.measurement_type,
            "technology_type": self._assay.sensor.technology_type,
            "technology_platform": self._assay.sensor.technology_platform,
            "n_runs": len(self._assay.runs),
            "n_raw_files": n_raw,
            "n_processed_files": n_proc,
        }
        fs = self._infer_fs()
        if fs is not None:
            info["fs_hz"] = fs
        unit = self._infer_unit()
        if unit is not None:
            info["unit"] = unit
        if self._assay.runs:
            info["measurement_params"] = [str(pv) for pv in self._assay.runs[0].measurement_params]
            info["factor_names"] = list(self._assay.runs[0].factor_values.keys())
        return info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # Units that describe protocol/timing metadata, not the measured signal.
    _SKIP_UNITS: frozenset[str] = frozenset(
        {"hz", "khz", "mhz", "min", "s", "ms", "sec", "second", "seconds",
         "hr", "hour", "hours"}
    )

    def _infer_unit(self) -> str | None:
        """
        Attempt to infer the signal measurement unit from protocol parameters.

        Checks ``pv.unit`` (resolved ontology string) first, skipping
        frequency units (Hz/kHz — those describe sampling rate, not the
        measured quantity) and time/duration units (min, s, ms, hr — those
        describe sampling intervals, not the physical signal).

        When no ``pv.unit`` is present, checks whether ``pv.parameter_name``
        contains the word "unit" (case-insensitive) — for example
        ``"Measured Unit"``, ``"Signal Unit"``, ``"Output Unit"`` — and, if
        so, treats ``pv.value`` directly as the unit string.  This handles
        arbitrary user-defined unit parameter names without requiring a fixed
        canonical name.

        Falls back to ``pv.value`` when it looks like a short alphabetical
        unit string (e.g. ``"nm"``).
        """
        if not self._assay.runs:
            return None
        for pv in self._assay.runs[0].measurement_params:
            if pv.unit and pv.unit.lower() not in self._SKIP_UNITS and "hz" not in pv.unit.lower():
                return pv.unit
            if not pv.unit and isinstance(pv.value, str):
                v = pv.value.strip()
                if not v or v.lower() in self._SKIP_UNITS:
                    continue
                # Parameter name contains "unit" → trust the value as-is
                if pv.parameter_name and "unit" in pv.parameter_name.lower():
                    return v
                # Generic fallback: short alphabetical string
                if _UNIT_VALUE_RE.fullmatch(v) is not None:
                    return v
        return None

    def _ylabel(self) -> str:
        """Build a y-axis label: measurement_type (unit) if unit is known."""
        label = self._assay.sensor.measurement_type or "value"
        unit = self._infer_unit()
        return f"{label} ({unit})" if unit else label

    def _infer_fs(self) -> float | None:
        """
        Attempt to infer sampling frequency from the first run's measurement
        protocol parameters.  Returns None if no Hz/kHz parameter is found.
        """
        if not self._assay.runs:
            return None
        for pv in self._assay.runs[0].measurement_params:
            if pv.unit and ("hz" in pv.unit.lower()):
                try:
                    raw_val = str(pv.value).replace(",", ".")
                    val = float(raw_val)
                    if "khz" in pv.unit.lower():
                        return val * 1000.0
                    return val
                except (ValueError, TypeError):
                    continue
        return None

    def _first_or_run(self, run_id: str | None) -> "RunRecord | None":
        """Return the named run, or the first run when run_id is None."""
        if not self._assay.runs:
            return None
        if run_id is None:
            return self._assay.runs[0]
        return self._assay.get_run(run_id)


# ---------------------------------------------------------------------------
# RunProxy
# ---------------------------------------------------------------------------

class RunProxy:
    """
    Single-run level proxy.

    Acquire via: ``wrapper.study("…").assay("…").run("run_01")``
    """

    def __init__(
        self,
        run: RunRecord,
        assay: AssayModel,
        study: StudyModel,
        investigation: InvestigationModel,
        integrator: "DataIntegrator",
        plotter: "ISAPlotter",
    ) -> None:
        self._run = run
        self._assay = assay
        self._study = study
        self._inv = investigation
        self._integrator = integrator
        self._plotter = plotter

    @property
    def run_id(self) -> str:
        return self._run.run_id

    @property
    def run_number(self) -> int:
        return self._run.run_number

    def overview(self) -> RunOverview:
        """Return a detailed summary of this run."""
        return RunOverview(
            run_id=self._run.run_id,
            run_number=self._run.run_number,
            assay_id=self._assay.assay_id,
            study_id=self._study.study_id,
            sensor_alias=self._assay.sensor.alias,
            measurement_type=self._assay.sensor.measurement_type,
            technology_type=self._assay.sensor.technology_type,
            raw_file_path=self._run.raw_file.path if self._run.raw_file else None,
            processed_file_path=(
                self._run.processed_file.path if self._run.processed_file else None
            ),
            factor_values=self._run.factor_values,
            measurement_params=[pv.model_dump() for pv in self._run.measurement_params],
            processing_params=[pv.model_dump() for pv in self._run.processing_params],
        )

    def factor_values(self) -> dict[str, object]:
        """Return study-level factor values for this run."""
        return dict(self._run.factor_values)

    def load_dataframe(self, file_type: Literal["raw", "processed", "auto"] = "processed") -> pd.DataFrame:
        """Load this run's data file into a DataFrame."""
        return self._integrator.load(
            self._assay,
            study_id=self._study.study_id,
            run_id=self._run.run_id,
            file_type=file_type,
        )

    def load_dataframe_with_meta(
        self,
        file_type: Literal["raw", "processed", "auto"] = "processed",
    ) -> tuple[pd.DataFrame, DataLoadMetadata]:
        """Load run data and return DataFrame plus resolved file-load metadata."""
        return self._integrator.load_with_meta(
            self._assay,
            study_id=self._study.study_id,
            run_id=self._run.run_id,
            file_type=file_type,
        )

    def plot_distribution(
        self,
        column: str = "value",
        bins: int = 50,
        file_type: Literal["raw", "processed", "auto"] = "processed",
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> object:
        """Amplitude histogram + KDE."""
        df = self.load_dataframe(file_type=file_type)
        return self._plotter.plot_distribution(
            df,
            column=column,
            bins=bins,
            title=title or f"{self._assay.assay_id} / {self._run.run_id} — Distribution",
            xlabel=xlabel,
            ylabel=ylabel,
            width=width,
            height=height,
        )

    def plot_frequency_domain(
        self,
        fs: float | None = None,
        column: str = "value",
        file_type: Literal["raw", "processed", "auto"] = "processed",
        log_scale: bool = True,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> object:
        """FFT magnitude spectrum."""
        if fs is None:
            for pv in self._run.measurement_params:
                if pv.unit and "hz" in pv.unit.lower():
                    try:
                        v = float(str(pv.value).replace(",", "."))
                        fs = v * 1000 if "khz" in pv.unit.lower() else v
                        break
                    except (ValueError, TypeError):
                        pass
        df = self.load_dataframe(file_type=file_type)
        return self._plotter.plot_frequency_domain(
            df,
            fs=fs,
            column=column,
            log_scale=log_scale,
            title=title or f"{self._assay.assay_id} / {self._run.run_id} — FFT Spectrum",
            xlabel=xlabel,
            ylabel=ylabel,
            width=width,
            height=height,
        )



