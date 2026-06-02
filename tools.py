"""
Tool definitions and executor for the ISA-PHM AI agent.

Metadata tools route through the local ISAParser (which itself uses the
vendored isa_phm wrapper for JSON parse + preprocess). Signal-level tools
(load_run_csv, lifecycle_features, make_plot) route through an ISAWrapper
instance attached to the parser once a data directory is configured.

build_tool_definitions() generates the tool list with the currently loaded
dataset names injected into each description.
execute_tool() dispatches a Claude tool_use call to the right handler.

Plot tools return {"plot_id": str, "plot_kind": str, "message": str} and
register the Bokeh figure in a shared registry; the Streamlit app renders
from that registry after the turn completes via st.bokeh_chart().
"""

from __future__ import annotations

import json
import math
import re
import uuid
from typing import Any

from isa_parser import ISAParser


# ---------------------------------------------------------------------------
# Figure registry — populated by plot tools, consumed by the Streamlit app
# ---------------------------------------------------------------------------
# Keyed by plot_id. Values are dicts {"figure": bokeh.figure | layout,
# "kind": str, "title": str, "backend": "bokeh"}. The Streamlit app
# renders these via st.bokeh_chart() after the turn completes.
FIGURE_REGISTRY: dict[str, dict] = {}


def _register_figure(fig, kind: str, title: str, backend: str = "bokeh") -> str:
    plot_id = f"plot_{uuid.uuid4().hex[:8]}"
    FIGURE_REGISTRY[plot_id] = {
        "figure": fig,
        "kind": kind,
        "title": title,
        "backend": backend,
    }
    return plot_id


def _jsonify(value: Any) -> Any:
    """Recursively convert pandas/numpy/pydantic objects to JSON-safe primitives."""
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if hasattr(value, "model_dump"):
        try:
            return _jsonify(value.model_dump())
        except Exception:
            return str(value)
    if hasattr(value, "to_dict"):
        try:
            return _jsonify(value.to_dict())
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(v) for v in value]
    return str(value)


# ---------------------------------------------------------------------------
# Session working-copy store — populated by process_signal, consumed by
# load_run_csv and single-run make_plot kinds so cleaned data persists across
# prompts. The dict is owned by the Streamlit session (st.session_state) and
# threaded through run_agent -> execute_tool, mirroring how data_dirs flows.
# Keyed by _wc_key(); values are dicts:
#   {"df": DataFrame, "dataset": str, "study": str, "sensor": str,
#    "run_id": str, "transforms": [ {...}, ... ], "raw_stats": {...}}
# ---------------------------------------------------------------------------

def _wc_key(dataset_name: str, study_title: str, sensor_alias: str, run_id: str) -> str:
    return f"{dataset_name}::{study_title}::{sensor_alias}::{run_id}"


def _value_stats(df, column: str = "value") -> dict | None:
    """Summary stats for the signal column (falls back to the first numeric
    column). Includes the NaN count so before/after fills are visible."""
    import numpy as np
    if df is None or len(df.columns) == 0:
        return None
    if column not in df.columns:
        numeric = [c for c in df.columns if df[c].dtype.kind in "fiu"]
        if not numeric:
            return None
        column = numeric[0]
    s = df[column]
    n_missing = int(s.isna().sum())
    vals = s.dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return {"column": str(column), "n": int(len(s)), "n_missing": n_missing}
    return {
        "column": str(column),
        "n": int(len(s)),
        "n_missing": n_missing,
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "rms": float(np.sqrt(np.mean(vals ** 2))),
    }


def _resolve_run_id(parser, assay_proxy, study_title, sensor_alias, relative_path=None):
    """Resolve a single run_id consistently across process_signal, load_run_csv,
    and make_plot. Prefers the run whose CSV matches relative_path; otherwise
    defaults to the LAST run (most degraded — relevant for prognostics)."""
    run_id = None
    if relative_path:
        run_id = parser.get_run_id_for_path(study_title, sensor_alias, relative_path)
    if run_id is None:
        try:
            runs = assay_proxy.list_runs()
            if runs:
                run_id = runs[-1].run_id
        except Exception:
            run_id = None
    return run_id


def build_tool_definitions(dataset_names: list[str]) -> list[dict]:
    """Generate tool definitions with the available dataset names embedded."""
    names_str = ", ".join(f'"{n}"' for n in dataset_names)

    dataset_param = {
        "dataset_name": {
            "type": "string",
            "description": (
                f"The name of the dataset to query. "
                f"Currently loaded datasets: {names_str}."
            ),
            "enum": dataset_names if dataset_names else None,
        }
    }
    # enum=None is invalid for Anthropic tools; drop if empty.
    if dataset_param["dataset_name"].get("enum") is None:
        del dataset_param["dataset_name"]["enum"]

    return [
        # ------------------------------------------------------------------
        # Metadata — ISA hierarchy
        # ------------------------------------------------------------------
        {
            "name": "get_investigation_overview",
            "description": (
                "Top-level overview of an ISA-PHM investigation: description, "
                "experiment type, license, release date, people, publications, "
                "number of studies. Use this first to understand a dataset."
            ),
            "input_schema": {
                "type": "object",
                "properties": dataset_param,
                "required": ["dataset_name"],
            },
        },
        {
            "name": "list_studies",
            "description": (
                "List every study: title, description, number of assays/samples, "
                "sensor aliases, factor names. Use before drilling into a study."
            ),
            "input_schema": {
                "type": "object",
                "properties": dataset_param,
                "required": ["dataset_name"],
            },
        },
        {
            "name": "get_study_details",
            "description": (
                "Detailed metadata for one study (by title): factors with types "
                "and units, characteristic categories, protocols + parameters, "
                "assay summaries, sample count."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    **dataset_param,
                    "study_title": {"type": "string", "description": "Exact study title, e.g. 'Case 1'."},
                },
                "required": ["dataset_name", "study_title"],
            },
        },
        {
            "name": "get_assay_details",
            "description": (
                "Detailed metadata for one assay within a study: measurement "
                "type, technology, data files, per-process parameter values."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    **dataset_param,
                    "study_title": {"type": "string"},
                    "sensor_alias": {"type": "string", "description": "e.g. 'vib_table'."},
                },
                "required": ["dataset_name", "study_title", "sensor_alias"],
            },
        },
        {
            "name": "list_data_files",
            "description": (
                "Data file paths from the metadata, optionally filtered by "
                "study and/or sensor. Returns {study, sensor, file, type}."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    **dataset_param,
                    "study_title": {"type": "string", "description": "Optional filter."},
                    "sensor_alias": {"type": "string", "description": "Optional filter."},
                },
                "required": ["dataset_name"],
            },
        },
        {
            "name": "search_metadata",
            "description": (
                "Case-insensitive keyword search across investigation, studies, "
                "assays, protocols, and people. Use for open-ended questions."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    **dataset_param,
                    "term": {"type": "string"},
                },
                "required": ["dataset_name", "term"],
            },
        },

        # ------------------------------------------------------------------
        # Semantic / PHM helpers
        # ------------------------------------------------------------------
        {
            "name": "ai_context",
            "description": (
                "Normalized PHM view of the investigation: experiment_type, "
                "studies with factor classifications (fault / operating_condition / "
                "degradation / other), assay measurement types, semantic manifest, "
                "and inferred PHM objective (detection / diagnostics / "
                "health assessment / prognosis). ALWAYS prefer this over raw "
                "metadata when answering a PHM question."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    **dataset_param,
                    "include_semantics": {"type": "boolean", "default": True},
                },
                "required": ["dataset_name"],
            },
        },
        {
            "name": "validate_dataset",
            "description": (
                "Run the 5 PHM quality gates from the playbook: 1) structural, "
                "2) file linkage (only if check_files=True + data base dir), "
                "3) PHM semantic (fault/operating-condition presence), "
                "4) PHM ambition (run-count vs inferred objective), "
                "5) merge readiness (only if check_merge=True + merge_target_dataset). "
                "Returns errors/warnings/info + gate_summary + PHM objective. "
                "Call this BEFORE analysis in every non-trivial workflow."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    **dataset_param,
                    "semantic_strict": {"type": "boolean", "default": False},
                    "check_merge": {"type": "boolean", "default": False},
                    "merge_target_dataset": {
                        "type": "string",
                        "description": "Name of the second loaded dataset to compare against for gate 5. Required if check_merge=True.",
                    },
                },
                "required": ["dataset_name"],
            },
        },
        {
            "name": "get_experiment_matrix",
            "description": (
                "Per-study × factor matrix with coverage %, PHM class, unit, and "
                "example raw values. Useful for getting a bird's-eye view of "
                "what was varied across studies. Note: in ISA-PHM, factor "
                "values sometimes point at per-run setting-file paths; the "
                "matrix surfaces that honestly."
            ),
            "input_schema": {
                "type": "object",
                "properties": dataset_param,
                "required": ["dataset_name"],
            },
        },
        {
            "name": "get_run_inventory",
            "description": (
                "Per-sample run inventory for one study: run_index, sample name, "
                "factor values (resolved), characteristics, data-file paths per "
                "assay, plus a continuity check against declared total_runs."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    **dataset_param,
                    "study_title": {"type": "string"},
                },
                "required": ["dataset_name", "study_title"],
            },
        },
        {
            "name": "get_label_coverage",
            "description": (
                "Per-study label coverage: how many runs have fault-label / "
                "degradation / operating-condition factor values populated. Use "
                "before proposing ML labels."
            ),
            "input_schema": {
                "type": "object",
                "properties": dataset_param,
                "required": ["dataset_name"],
            },
        },
        {
            "name": "get_sensor_availability",
            "description": (
                "Per-study assay measurement-type map. Flags sparse mapping "
                "(e.g. Sietze-style asynchronous modalities). Use before saying "
                "'sensor X is missing' — it may be intentional."
            ),
            "input_schema": {
                "type": "object",
                "properties": dataset_param,
                "required": ["dataset_name"],
            },
        },
        {
            "name": "classify_phm_objective",
            "description": (
                "Return the inferred PHM objective class (detection / diagnostics / "
                "health assessment / prognosis) with confidence and reasoning. "
                "Call this FIRST whenever a user asks vague/open-ended questions."
            ),
            "input_schema": {
                "type": "object",
                "properties": dataset_param,
                "required": ["dataset_name"],
            },
        },
        {
            "name": "compare_studies",
            "description": (
                "Side-by-side comparison of multiple studies WITHIN the same "
                "dataset. Returns factors, assays, sample counts, common factor "
                "names across the chosen studies."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    **dataset_param,
                    "study_titles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Two or more study titles from the same dataset.",
                    },
                },
                "required": ["dataset_name", "study_titles"],
            },
        },
        {
            "name": "get_protocol_details",
            "description": (
                "Replication-grade protocol extraction (Student Guide J-series). "
                "Returns everything needed to reproduce one study in a lab: "
                "rig_and_configuration, fault_introduction, operating_conditions, "
                "sensor_placement, sampling_and_acquisition, filter_settings, "
                "amplifier_settings, plus per-assay parameter values. Use for "
                "questions about reproducing or replicating an experiment."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    **dataset_param,
                    "study_title": {"type": "string"},
                },
                "required": ["dataset_name", "study_title"],
            },
        },
        {
            "name": "get_replication_gaps",
            "description": (
                "Score replication completeness for a study. Returns "
                "present_and_sufficient + missing_or_underspecified lists, "
                "an overall readiness rating (full / partial / not_reproducible), "
                "and the single most critical missing field. Use for replication "
                "audits and dataset-quality scoring."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    **dataset_param,
                    "study_title": {"type": "string"},
                },
                "required": ["dataset_name", "study_title"],
            },
        },
        {
            "name": "get_sensor_compatibility",
            "description": (
                "Check whether a sensor is high-rate / low-rate / unknown and "
                "which plot kinds are compatible. CALL THIS BEFORE generating "
                "any FFT/PSD/spectrogram plot — the playbook (§2.3) requires "
                "compatibility verification. Returns sensor_tier, sampling_rate_hz "
                "(if documented), compatible_plots, incompatible_plots with "
                "reasons, and a fallback suggestion."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    **dataset_param,
                    "study_title": {"type": "string"},
                    "sensor_alias": {"type": "string"},
                },
                "required": ["dataset_name", "study_title", "sensor_alias"],
            },
        },
        {
            "name": "compare_datasets_metadata",
            "description": (
                "Metadata-only side-by-side comparison of TWO datasets: "
                "experiment_type, study factor keys, sensor measurement types, "
                "unit strings, n_studies, median runs/study. DOES NOT fuse or "
                "crosswalk factors — cross-dataset fusion is a future capability. "
                "Returns a structured table plus per-dimension COMPATIBLE / "
                "NEEDS_REVIEW / INCOMPATIBLE flags."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "dataset_a": {"type": "string"},
                    "dataset_b": {"type": "string"},
                },
                "required": ["dataset_a", "dataset_b"],
            },
        },

        # ------------------------------------------------------------------
        # Code generation — real wrapper API (no dataset_name required)
        # ------------------------------------------------------------------
        {
            "name": "get_wrapper_api",
            "description": (
                "Return the REAL ISA-PHM wrapper API: the exact classes, methods, "
                "signatures, and one-line docs available in the vendored isa_phm "
                "package (ISAWrapper, StudyProxy, AssayProxy, RunProxy). CALL THIS "
                "BEFORE writing any Python / TensorFlow / PyTorch / scikit-learn code "
                "that uses the wrapper, and use ONLY the methods it returns — never "
                "invent wrapper method names. Optionally filter to one class via "
                "class_name, or to methods whose name contains `search`."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "class_name": {
                        "type": "string",
                        "enum": ["ISAWrapper", "StudyProxy", "AssayProxy", "RunProxy"],
                        "description": "Optional. Drill into a single class.",
                    },
                    "search": {
                        "type": "string",
                        "description": "Optional. Only return methods whose name contains this substring.",
                    },
                },
                "required": [],
            },
        },

        # ------------------------------------------------------------------
        # Signal-level tools (require a data base directory)
        # ------------------------------------------------------------------
        {
            "name": "load_run_csv",
            "description": (
                "Load the head and summary stats of one run's CSV via the "
                "wrapper. Requires the data directory to be configured in the "
                "sidebar. Returns n_rows, columns, dtypes, head, "
                "min/max/mean/std/rms stats, plus 'source' (raw_file or "
                "working_copy) and any 'transforms' already applied. Identify the "
                "run by EITHER study_title + sensor_alias (defaults to the last "
                "run; pass relative_path to pick a specific one) OR relative_path "
                "alone. If a process_signal working copy exists for the resolved "
                "run, the CLEANED data is returned (source='working_copy')."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    **dataset_param,
                    "study_title": {"type": "string", "description": "The study this run belongs to."},
                    "sensor_alias": {"type": "string", "description": "The sensor alias for this run's assay."},
                    "relative_path": {
                        "type": "string",
                        "description": (
                            "Optional. ISA-style relative path of the run's CSV. "
                            "If omitted (with study_title + sensor_alias), defaults "
                            "to the most recent run."
                        ),
                    },
                    "max_rows": {"type": "integer", "default": 5},
                },
                "required": ["dataset_name", "study_title", "sensor_alias"],
            },
        },
        {
            "name": "lifecycle_features",
            "description": (
                "Compute per-run lifecycle features (RMS, kurtosis, crest factor, "
                "peak-to-peak) across an assay. Returns trend slope + "
                "monotonicity flag. Requires data base directory. Use for "
                "prognostic degradation analysis."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    **dataset_param,
                    "study_title": {"type": "string"},
                    "sensor_alias": {"type": "string"},
                },
                "required": ["dataset_name", "study_title", "sensor_alias"],
            },
        },
        {
            "name": "process_signal",
            "description": (
                "Clean/transform ONE run's signal and CACHE the result as a "
                "session 'working copy', so later tools reuse the cleaned data "
                "instead of re-reading the raw file. Requires the dataset's data "
                "directory. Apply outlier correction (fix_outliers) and/or "
                "missing-value filling (fill_missing). The working copy persists "
                "across prompts for this (dataset, study, sensor, run) until "
                "reset, and subsequent load_run_csv and single-run make_plot "
                "calls for the SAME run automatically use it. Defaults to the "
                "last run if relative_path is omitted. Transforms STACK on the "
                "current working copy unless reset_to_raw=true. Returns the "
                "transforms applied plus before/after signal stats."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    **dataset_param,
                    "study_title": {"type": "string"},
                    "sensor_alias": {"type": "string"},
                    "relative_path": {
                        "type": "string",
                        "description": (
                            "Optional ISA-style relative path of the run's CSV. "
                            "If omitted, defaults to the most recent run."
                        ),
                    },
                    "fix_outliers": {
                        "type": "object",
                        "description": (
                            "If present (even as {}), correct outliers in the "
                            "'value' column."
                        ),
                        "properties": {
                            "method": {
                                "type": "string",
                                "enum": ["iqr", "zscore", "fixed"],
                                "default": "iqr",
                            },
                            "threshold": {"type": "number", "default": 3.0},
                            "strategy": {
                                "type": "string",
                                "enum": ["clip", "nan", "drop"],
                                "default": "clip",
                                "description": (
                                    "clip=clamp to bounds, nan=replace with NaN, "
                                    "drop=remove outlier rows."
                                ),
                            },
                            "lower": {
                                "type": "number",
                                "description": "Hard lower bound (for method='fixed' or to override).",
                            },
                            "upper": {
                                "type": "number",
                                "description": "Hard upper bound, e.g. 1e7 to remove sensor overflow.",
                            },
                        },
                    },
                    "fill_missing": {
                        "type": "object",
                        "description": "If present (even as {}), fill NaN values in numeric columns.",
                        "properties": {
                            "strategy": {
                                "type": "string",
                                "enum": ["interpolate", "ffill", "bfill", "mean", "zero"],
                                "default": "interpolate",
                            },
                        },
                    },
                    "reset_to_raw": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "If true, discard any existing working copy for this "
                            "run first (start from the raw file). If true with no "
                            "operations, just clears the working copy."
                        ),
                    },
                },
                "required": ["dataset_name", "study_title", "sensor_alias"],
            },
        },
        {
            "name": "make_plot",
            "description": (
                "Render a Bokeh plot via the ISA-PHM wrapper. Requires the "
                "dataset's data directory to be configured. Returns a plot_id; "
                "the app renders the figure.\n"
                "PLOT KINDS:\n"
                "• Single-run (need sensor_alias; default to last run if "
                "relative_path omitted): timeseries, fft, psd, spectrogram, "
                "distribution, missing_values, outlier_comparison.\n"
                "• Single-sensor across all runs (need sensor_alias): lifecycle "
                "(feature trend, e.g. RMS), correlation (feature-correlation "
                "heatmap), variability (per-run amplitude boxplot), waterfall "
                "(stacked FFT spectra).\n"
                "• Study-wide across sensors (sensor_alias NOT needed): "
                "multi_lifecycle (overlay one feature — default RMS — for every "
                "sensor: the 'Multi-RMS' comparison), sensor_boxplot (amplitude "
                "across all sensors), sensor_lifecycle_correlation (feature "
                "correlation across sensors).\n"
                "• Two-sensor (need sensor_alias AND sensor_alias_2): "
                "cross_correlation."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    **dataset_param,
                    "kind": {
                        "type": "string",
                        "enum": [
                            "timeseries", "fft", "psd", "spectrogram",
                            "distribution", "missing_values",
                            "outlier_comparison", "lifecycle", "correlation",
                            "variability", "waterfall", "multi_lifecycle",
                            "sensor_boxplot", "sensor_lifecycle_correlation",
                            "cross_correlation",
                        ],
                    },
                    "study_title": {"type": "string"},
                    "sensor_alias": {
                        "type": "string",
                        "description": (
                            "Sensor channel. Required for all single-run, "
                            "single-sensor, and two-sensor kinds. Not needed for "
                            "study-wide kinds (multi_lifecycle, sensor_boxplot, "
                            "sensor_lifecycle_correlation)."
                        ),
                    },
                    "sensor_alias_2": {
                        "type": "string",
                        "description": "Second sensor channel. Required only for kind='cross_correlation'.",
                    },
                    "relative_path": {
                        "type": "string",
                        "description": (
                            "Optional. ISA-style relative path of one run's CSV. "
                            "If omitted, single-run kinds default to the most "
                            "recent run. Ignored by multi-run and study-wide kinds."
                        ),
                    },
                    "feature": {
                        "type": "string",
                        "default": "rms",
                        "description": (
                            "Scalar feature for kind in {lifecycle, "
                            "multi_lifecycle, sensor_lifecycle_correlation}: "
                            "rms, kurtosis, crest_factor, peak2peak, mean, std."
                        ),
                    },
                },
                "required": ["dataset_name", "kind", "study_title"],
            },
        },
        # ------------------------------------------------------------------
        # Generic escape hatch — run ANY wrapper method not covered above.
        # ------------------------------------------------------------------
        {
            "name": "call_wrapper_function",
            "description": (
                "Escape hatch: run ANY public method on the ISA-PHM wrapper objects "
                "that has no dedicated tool, so you are never limited to the curated "
                "tools. ALWAYS call get_wrapper_api FIRST to get the exact method "
                "name, owning class, and arguments; then call this. A method that "
                "returns a plot is rendered; a method that returns a table is "
                "summarized (first rows + shape). Signal/plot methods need a data "
                "directory. Does NOT update the cleaned working-copy cache — use "
                "process_signal for cleaning that must persist across prompts."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    **dataset_param,
                    "target": {
                        "type": "string",
                        "enum": ["wrapper", "study", "assay", "run"],
                        "description": (
                            "Which wrapper object owns the method (get_wrapper_api "
                            "groups methods by class): 'wrapper' = ISAWrapper, "
                            "'study' = StudyProxy, 'assay' = AssayProxy, 'run' = RunProxy."
                        ),
                    },
                    "method": {
                        "type": "string",
                        "description": (
                            "Exact public method name from get_wrapper_api, e.g. "
                            "sensor_catalog, operating_conditions, to_ml_dataset."
                        ),
                    },
                    "arguments": {
                        "type": "object",
                        "description": (
                            "Keyword arguments for the method, as a JSON object. "
                            "Omit or pass an empty object for no arguments."
                        ),
                    },
                    "study_title": {
                        "type": "string",
                        "description": "Required for target study/assay/run.",
                    },
                    "sensor_alias": {
                        "type": "string",
                        "description": "Required for target assay/run.",
                    },
                    "relative_path": {
                        "type": "string",
                        "description": "Optional for target run; defaults to the last run.",
                    },
                },
                "required": ["dataset_name", "target", "method"],
            },
        },
    ]


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

def execute_tool(
    tool_name: str,
    tool_input: dict,
    parsers: dict[str, ISAParser],
    data_dirs: dict[str, str] | None = None,
    working_copies: dict | None = None,
) -> str:
    """Dispatch a Claude tool_use call. Always returns JSON.
    data_dirs maps dataset_name → base_dir for signal tools.
    working_copies is the session-scoped cleaned-data store (mutated in place by
    process_signal; read by load_run_csv and single-run make_plot)."""
    data_dirs = data_dirs or {}
    if working_copies is None:
        working_copies = {}

    try:
        # Dataset-independent tools
        if tool_name == "get_wrapper_api":
            result = _wrapper_api_reference(
                tool_input.get("class_name"), tool_input.get("search")
            )
            return json.dumps(result, indent=2, default=str)

        # Two-dataset tools
        if tool_name == "compare_datasets_metadata":
            a = tool_input.get("dataset_a")
            b = tool_input.get("dataset_b")
            if a not in parsers or b not in parsers:
                return json.dumps({"error": f"One or both datasets not loaded. Available: {list(parsers.keys())}"})
            result = _compare_datasets_metadata(parsers[a], parsers[b], a, b)
            return json.dumps(result, indent=2, default=str)

        # Every other tool requires dataset_name
        dataset_name = tool_input.get("dataset_name")
        if dataset_name not in parsers:
            return json.dumps({"error": f"Dataset '{dataset_name}' not found. Available: {list(parsers.keys())}"})

        parser = parsers[dataset_name]
        base_dir = data_dirs.get(dataset_name)

        if tool_name == "get_investigation_overview":
            result = parser.get_investigation_overview()

        elif tool_name == "list_studies":
            result = parser.list_studies()

        elif tool_name == "get_study_details":
            result = parser.get_study_details(tool_input["study_title"])

        elif tool_name == "get_assay_details":
            result = parser.get_assay_details(tool_input["study_title"], tool_input["sensor_alias"])

        elif tool_name == "list_data_files":
            result = parser.list_data_files(
                study_title=tool_input.get("study_title"),
                sensor_alias=tool_input.get("sensor_alias"),
            )

        elif tool_name == "search_metadata":
            result = parser.search_metadata(tool_input["term"])

        elif tool_name == "ai_context":
            result = parser.get_ai_context(
                include_semantics=tool_input.get("include_semantics", True),
            )

        elif tool_name == "validate_dataset":
            merge_target_name = tool_input.get("merge_target_dataset")
            merge_target_parser = parsers.get(merge_target_name) if merge_target_name else None
            result = parser.validate_dataset(
                semantic_strict=tool_input.get("semantic_strict", False),
                check_merge=tool_input.get("check_merge", False),
                merge_target_parser=merge_target_parser,
            )

        elif tool_name == "get_experiment_matrix":
            result = parser.get_experiment_matrix()

        elif tool_name == "get_run_inventory":
            result = parser.get_run_inventory(tool_input["study_title"])

        elif tool_name == "get_label_coverage":
            result = parser.get_label_coverage()

        elif tool_name == "get_sensor_availability":
            result = parser.get_sensor_availability()

        elif tool_name == "classify_phm_objective":
            result = parser.classify_phm_objective()

        elif tool_name == "compare_studies":
            result = parser.compare_studies(tool_input["study_titles"])

        elif tool_name == "get_protocol_details":
            result = parser.get_protocol_details(tool_input["study_title"])

        elif tool_name == "get_replication_gaps":
            result = parser.get_replication_gaps(tool_input["study_title"])

        elif tool_name == "get_sensor_compatibility":
            result = parser.get_sensor_compatibility(
                tool_input["study_title"],
                tool_input["sensor_alias"],
            )

        elif tool_name == "load_run_csv":
            result = _load_run_csv_via_wrapper(parser, tool_input, dataset_name, working_copies)

        elif tool_name == "process_signal":
            result = _process_signal_via_wrapper(parser, tool_input, dataset_name, working_copies)

        elif tool_name == "lifecycle_features":
            result = _lifecycle_features_via_wrapper(
                parser, tool_input["study_title"], tool_input["sensor_alias"]
            )

        elif tool_name == "make_plot":
            result = _make_plot_via_wrapper(parser, tool_input, dataset_name, working_copies)

        elif tool_name == "call_wrapper_function":
            result = _call_wrapper_function(parser, tool_input, dataset_name)

        else:
            result = {"error": f"Unknown tool: {tool_name}"}

    except Exception as e:
        result = {"error": f"{type(e).__name__}: {e}"}

    return json.dumps(result, indent=2, default=str)


# ---------------------------------------------------------------------------
# Compound handlers
# ---------------------------------------------------------------------------

def _compare_datasets_metadata(pa: ISAParser, pb: ISAParser, name_a: str, name_b: str) -> dict:
    ca = pa.get_ai_context(include_semantics=True)
    cb = pb.get_ai_context(include_semantics=True)

    def _median_runs(ctx):
        runs = [s["n_runs"] for s in ctx["studies"]]
        if not runs:
            return 0
        runs.sort()
        return runs[len(runs) // 2]

    rows = []

    def _row(dim, va, vb, compat=None):
        if compat is None:
            compat = "COMPATIBLE" if va == vb else "NEEDS_REVIEW"
        rows.append({"dimension": dim, name_a: va, name_b: vb, "flag": compat})

    _row(
        "experiment_type",
        ca["experiment_type"],
        cb["experiment_type"],
        compat=(
            "COMPATIBLE" if ca["experiment_type"] == cb["experiment_type"]
            else "INCOMPATIBLE"
        ),
    )
    _row("n_studies", ca["n_studies"], cb["n_studies"])
    _row("median_runs_per_study", _median_runs(ca), _median_runs(cb))

    factors_a = set(ca["semantic_manifest"]["all_factor_names"])
    factors_b = set(cb["semantic_manifest"]["all_factor_names"])
    _row(
        "study_factor_keys",
        sorted(factors_a),
        sorted(factors_b),
        compat="COMPATIBLE" if factors_a == factors_b else (
            "NEEDS_REVIEW" if factors_a & factors_b else "INCOMPATIBLE"
        ),
    )

    mtypes_a = set(ca["semantic_manifest"]["all_measurement_types"])
    mtypes_b = set(cb["semantic_manifest"]["all_measurement_types"])
    _row(
        "sensor_measurement_types",
        sorted(mtypes_a),
        sorted(mtypes_b),
        compat="COMPATIBLE" if mtypes_a == mtypes_b else (
            "NEEDS_REVIEW" if mtypes_a & mtypes_b else "INCOMPATIBLE"
        ),
    )

    verdict = "FEASIBLE"
    if any(r["flag"] == "INCOMPATIBLE" for r in rows):
        verdict = "NOT FEASIBLE"
    elif any(r["flag"] == "NEEDS_REVIEW" for r in rows):
        verdict = "NEEDS MANUAL REVIEW"

    return {
        "rows": rows,
        "overall_fusion_verdict": verdict,
        "future_capability_note": (
            "⚠ Cross-dataset factor crosswalk and unit conversion are not "
            "implemented. This tool only reports metadata compatibility "
            "side-by-side; it does NOT produce a merged dataset or a "
            "programmatic GO/NO-GO."
        ),
    }


def _wrapper_api_reference(class_name: str | None = None, search: str | None = None) -> dict:
    """Introspect the real isa_phm wrapper classes so the agent writes code against
    methods that actually exist instead of inventing them."""
    import functools
    import inspect

    try:
        from isa_phm.wrapper import ISAWrapper
        from isa_phm.proxy import StudyProxy, AssayProxy, RunProxy
    except Exception as e:  # pragma: no cover - defensive
        return {"error": f"Could not import wrapper classes: {type(e).__name__}: {e}"}

    classes = {
        "ISAWrapper": ISAWrapper,
        "StudyProxy": StudyProxy,
        "AssayProxy": AssayProxy,
        "RunProxy": RunProxy,
    }
    if class_name:
        match = {k: v for k, v in classes.items() if k.lower() == class_name.lower()}
        if not match:
            return {
                "error": (
                    f"Unknown class '{class_name}'. "
                    f"Known classes: {', '.join(classes)}."
                )
            }
        classes = match

    term = (search or "").lower()

    def _members(cls) -> list[dict]:
        out = []
        for name, member in vars(cls).items():
            if name.startswith("_"):
                continue
            if term and term not in name.lower():
                continue
            if isinstance(member, property):
                fn, kind = member.fget, "property"
            elif isinstance(member, functools.cached_property):
                fn, kind = member.func, "property"
            elif isinstance(member, (staticmethod, classmethod)):
                fn, kind = member.__func__, "method"
            elif inspect.isfunction(member):
                fn, kind = member, "method"
            else:
                continue
            try:
                sig = re.sub(r"\(self(?:,\s*)?", "(", str(inspect.signature(fn)), count=1)
            except (ValueError, TypeError):
                sig = "(...)"
            doc = inspect.getdoc(fn) or ""
            first_line = doc.strip().split("\n", 1)[0] if doc else ""
            out.append({"name": name, "kind": kind, "signature": sig, "doc": first_line})
        return sorted(out, key=lambda m: m["name"])

    return {
        "object_model": (
            "ISAWrapper(path, data_root, strict_validation=False, csv_bad_lines='warn') "
            "-> .study(study_title) -> StudyProxy.assay(assay_filename) "
            "-> AssayProxy.run(run_id) -> RunProxy. "
            "In this app a study is addressed by its title and an assay by its CSV filename."
        ),
        "classes": {name: _members(cls) for name, cls in classes.items()},
        "note": (
            "These are the ONLY wrapper methods that exist — use these exact names and "
            "signatures when generating code. Do not invent methods. Call again with "
            "class_name=... to focus on one class. The wrapper does not train models or "
            "split data; wrap its DataFrame output with tf.keras / sklearn for that."
        ),
    }


def _serialize_wrapper_result(value, dataset_name: str, target: str, method: str) -> dict:
    """Turn an arbitrary wrapper return value into a JSON-safe tool result.
    Plots are registered for rendering; DataFrames/Series are summarized (never
    dumped whole); everything else is JSON-ified with a size cap."""
    import pandas as pd

    module = getattr(type(value), "__module__", "") or ""
    if "bokeh" in module:  # a Bokeh figure — render it like make_plot does
        title = " · ".join(p for p in [dataset_name, target, method] if p)
        plot_id = _register_figure(value, kind=f"wrapper:{method}", title=title)
        return {
            "result_type": "figure",
            "plot_id": plot_id,
            "message": f"{method} returned a plot. The app will render it.",
        }

    if isinstance(value, pd.DataFrame):
        n_rows, n_cols = int(value.shape[0]), int(value.shape[1])
        return {
            "result_type": "dataframe",
            "n_rows": n_rows,
            "n_cols": n_cols,
            "columns": [str(c) for c in value.columns],
            "dtypes": {str(c): str(value[c].dtype) for c in value.columns},
            "preview_rows": _jsonify(value.head(5).to_dict("records")),
            "truncated": n_rows > 5,
            "note": (
                "DataFrame summarized (first 5 rows). For the full table, generate "
                "code with the wrapper (see the code-generation guidance)."
            ),
        }

    if isinstance(value, pd.Series):
        n = int(value.shape[0])
        return {
            "result_type": "series",
            "n": n,
            "preview": _jsonify(value.head(20).to_dict()),
            "truncated": n > 20,
        }

    if value is None:
        return {"result_type": "none", "message": f"{method} returned None."}

    payload = _jsonify(value)
    serialized = json.dumps(payload, default=str)
    cap = 6000
    if len(serialized) > cap:
        return {
            "result_type": "object",
            "truncated": True,
            "value_preview": serialized[:cap] + "…",
            "note": "Result truncated; request a narrower call or generate code for the full object.",
        }
    return {"result_type": "object", "value": payload}


def _call_wrapper_function(parser: ISAParser, tool_input: dict, dataset_name: str) -> dict:
    """Generic dispatch: invoke any public method on a resolved wrapper object
    (ISAWrapper / StudyProxy / AssayProxy / RunProxy). Pairs with get_wrapper_api
    (discover the method) so the agent can reach the whole API, not just the
    curated tools. Read/analysis-only — the wrapper API does not write or delete."""
    import functools
    import inspect

    if parser.wrapper is None:
        return _wrapper_unavailable(parser)

    target = (tool_input.get("target") or "").strip().lower()
    method = tool_input.get("method")
    arguments = tool_input.get("arguments") or {}
    if not isinstance(arguments, dict):
        return {"error": "arguments must be a JSON object of keyword arguments."}
    if not isinstance(method, str) or not method or method.startswith("_"):
        return {"error": "method must be a public method name (no leading underscore)."}

    study_title = tool_input.get("study_title")
    sensor_alias = tool_input.get("sensor_alias")
    relative_path = tool_input.get("relative_path")

    # Resolve the target object.
    if target == "wrapper":
        obj = parser.wrapper
    elif target == "study":
        if study_title is None:
            return {"error": "target 'study' requires study_title."}
        obj, err = _resolve_study_proxy(parser, study_title)
        if err is not None:
            return err
    elif target == "assay":
        if study_title is None or sensor_alias is None:
            return {"error": "target 'assay' requires study_title and sensor_alias."}
        obj, err = _resolve_assay_proxy(parser, study_title, sensor_alias)
        if err is not None:
            return err
    elif target == "run":
        if study_title is None or sensor_alias is None:
            return {"error": "target 'run' requires study_title and sensor_alias."}
        assay_proxy, err = _resolve_assay_proxy(parser, study_title, sensor_alias)
        if err is not None:
            return err
        run_id = _resolve_run_id(parser, assay_proxy, study_title, sensor_alias, relative_path)
        if run_id is None:
            return {"error": "Could not resolve a run for this sensor (no runs found)."}
        try:
            obj = assay_proxy.run(run_id)
        except Exception as e:
            return {"error": f"Could not open run '{run_id}': {type(e).__name__}: {e}"}
    else:
        return {"error": "target must be one of: wrapper, study, assay, run."}

    # Guard: only public methods/properties that actually exist on the resolved
    # object's class. getattr_static avoids triggering descriptors during the check.
    descriptor = inspect.getattr_static(obj, method, None)
    if descriptor is None:
        return {"error": (
            f"'{method}' is not a public member of the {target} object. "
            f"Call get_wrapper_api(class_name=...) to list valid methods."
        )}
    is_property = isinstance(descriptor, (property, functools.cached_property))
    is_method = inspect.isfunction(descriptor) or isinstance(descriptor, (staticmethod, classmethod))
    if not (is_property or is_method):
        return {"error": f"'{method}' is not a callable method on the {target} object."}

    try:
        if is_property:
            value = getattr(obj, method)  # properties take no arguments
        else:
            value = getattr(obj, method)(**arguments)
    except TypeError as e:
        return {"error": (
            f"Bad arguments for {target}.{method}: {e}. "
            f"Check the exact signature via get_wrapper_api."
        )}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    out = _serialize_wrapper_result(value, dataset_name, target, method)
    out.setdefault("target", target)
    out.setdefault("method", method)
    return out


def _wrapper_unavailable(parser: ISAParser) -> dict:
    """Standard response when signal-level operations are not possible yet."""
    status = getattr(parser, "wrapper_status", "not_attached")
    err = getattr(parser, "wrapper_error", "")
    return {
        "status": "data_not_available",
        "wrapper_status": status,
        "message": (
            err
            or "No data directory is configured for this dataset. Set the "
            "'Data directory' field in the sidebar to enable signal-level tools."
        ),
    }


def _resolve_assay_proxy(parser: ISAParser, study_title: str, sensor_alias: str):
    """Return (assay_proxy, error_dict). On success error_dict is None."""
    if parser.wrapper is None:
        return None, _wrapper_unavailable(parser)
    filename = parser.get_assay_filename(study_title, sensor_alias)
    if filename is None:
        return None, {"error": (
            f"Could not map sensor_alias '{sensor_alias}' under study "
            f"'{study_title}' to an assay. Check sensor name spelling."
        )}
    try:
        study_proxy = parser.wrapper.study(study_title)
        assay_proxy = study_proxy.assay(filename)
        return assay_proxy, None
    except Exception as e:
        return None, {"error": f"Wrapper lookup failed: {type(e).__name__}: {e}"}


def _load_run_csv_via_wrapper(
    parser: ISAParser, tool_input: dict, dataset_name: str, working_copies: dict
) -> dict:
    if parser.wrapper is None:
        return _wrapper_unavailable(parser)

    relative_path = tool_input.get("relative_path")
    max_rows = tool_input.get("max_rows", 5)
    study_title = tool_input.get("study_title")
    sensor_alias = tool_input.get("sensor_alias")
    from_working_copy = False
    transforms: list = []

    # If caller provided study + sensor, resolve via wrapper proxies. Use
    # `is not None` (not truthiness): some datasets use an empty sensor alias,
    # which is still a valid, resolvable channel.
    if study_title is not None and sensor_alias is not None:
        assay_proxy, err = _resolve_assay_proxy(parser, study_title, sensor_alias)
        if err is not None:
            return err
        # Resolve consistently with process_signal/make_plot: prefer the run
        # matching relative_path, else default to the last run.
        run_id = _resolve_run_id(parser, assay_proxy, study_title, sensor_alias, relative_path)
        wc = working_copies.get(_wc_key(dataset_name, study_title, sensor_alias, run_id))
        if wc is not None:
            # Serve the cached cleaned data instead of re-reading the raw file.
            df = wc["df"]
            meta = {"source": "working_copy"}
            transforms = wc["transforms"]
            from_working_copy = True
        else:
            try:
                df, meta = assay_proxy.load_dataframe_with_meta(run_id=run_id, file_type="auto")
            except Exception as e:
                return {"status": "read_error", "message": f"{type(e).__name__}: {e}"}
    elif not relative_path:
        return {"error": "load_run_csv needs either (study_title + sensor_alias) or relative_path."}
    else:
        # Fallback: scan the investigation for a run whose data file ends with relative_path.
        run_id = None
        target = relative_path.replace("\\", "/").lstrip("./").lower()
        match = None  # (study_title, assay_filename, run_id)
        for study in parser.wrapper.investigation.studies:
            for assay in study.assays:
                for run in assay.runs:
                    for df_obj in (run.raw_file, run.processed_file):
                        if df_obj is None:
                            continue
                        cand = (df_obj.path or "").replace("\\", "/").lower()
                        if cand.endswith("/" + target) or cand.endswith(target):
                            match = (study.title, assay.assay_id, run.run_id)
                            break
                    if match: break
                if match: break
            if match: break
        if match is None:
            return {"status": "file_missing", "message": f"No assay run matches path '{relative_path}'."}
        s_title, a_id, run_id = match
        try:
            assay_proxy = parser.wrapper.study(s_title).assay(a_id)
            df, meta = assay_proxy.load_dataframe_with_meta(run_id=run_id, file_type="auto")
        except Exception as e:
            return {"status": "read_error", "message": f"{type(e).__name__}: {e}"}

    # Build a JSON-safe summary.
    import numpy as np
    head = df.head(max_rows).to_dict(orient="list")
    # Pick the first numeric column for stats; skip a monotonic "time" column when possible.
    numeric_cols = [c for c in df.columns if df[c].dtype.kind in "fiu"]
    chosen = None
    for c in numeric_cols:
        s = df[c].dropna()
        if len(s) < 4:
            continue
        diffs = np.diff(s.values[: min(len(s), 1000)])
        if diffs.size and np.all(diffs > 0) and len(numeric_cols) > 1:
            continue  # monotonic → likely time column
        chosen = c
        break
    if chosen is None and numeric_cols:
        chosen = numeric_cols[0]

    stats = None
    if chosen is not None:
        values = df[chosen].dropna().to_numpy(dtype=float)
        if values.size:
            stats = {
                "column": str(chosen),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "rms": float(np.sqrt(np.mean(values ** 2))),
            }

    return {
        "status": "ok",
        "run_id": run_id,
        "relative_path": relative_path,
        "source": "working_copy" if from_working_copy else "raw_file",
        "transforms": _jsonify(transforms),
        "n_rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
        "head": _jsonify(head),
        "stats": stats,
        "load_meta": _jsonify(meta),
    }


def _process_signal_via_wrapper(
    parser: ISAParser, tool_input: dict, dataset_name: str, working_copies: dict
) -> dict:
    """Clean/transform one run and cache the result as a session working copy.
    Loads from the existing working copy if present (so transforms stack), else
    from the raw file. Mutates working_copies in place."""
    if parser.wrapper is None:
        return _wrapper_unavailable(parser)

    study_title = tool_input.get("study_title")
    sensor_alias = tool_input.get("sensor_alias")
    # Use `is None` (not truthiness): some datasets use an empty sensor alias,
    # which is still a valid, resolvable channel.
    if study_title is None or sensor_alias is None:
        return {"error": "process_signal requires study_title and sensor_alias."}

    assay_proxy, err = _resolve_assay_proxy(parser, study_title, sensor_alias)
    if err is not None:
        return err

    relative_path = tool_input.get("relative_path")
    run_id = _resolve_run_id(parser, assay_proxy, study_title, sensor_alias, relative_path)
    if run_id is None:
        return {"error": "Could not resolve a run for this sensor (no runs found)."}

    key = _wc_key(dataset_name, study_title, sensor_alias, run_id)
    reset_to_raw = bool(tool_input.get("reset_to_raw", False))
    fix_spec = tool_input.get("fix_outliers")
    fill_spec = tool_input.get("fill_missing")

    # reset_to_raw with no operations → just drop the working copy.
    if reset_to_raw and fix_spec is None and fill_spec is None:
        existed = working_copies.pop(key, None) is not None
        return {
            "status": "ok",
            "run_id": run_id,
            "reset_to_raw": True,
            "transforms": [],
            "message": (
                "Working copy cleared; later tools will read the raw file."
                if existed else "No working copy existed for this run; nothing to reset."
            ),
        }

    existing = None if reset_to_raw else working_copies.get(key)
    if existing is not None:
        df = existing["df"].copy()
        transforms = list(existing["transforms"])
        raw_stats = existing["raw_stats"]
        started_from = "working_copy"
    else:
        try:
            df = assay_proxy.load_dataframe(run_id=run_id, file_type="auto")
        except Exception as e:
            return {"status": "read_error", "message": f"{type(e).__name__}: {e}"}
        transforms = []
        raw_stats = _value_stats(df)
        started_from = "raw"

    before_stats = _value_stats(df)
    col = (before_stats or {}).get("column", "value")
    applied: list = []

    if fix_spec is not None:
        method = fix_spec.get("method", "iqr")
        threshold = fix_spec.get("threshold", 3.0)
        strategy = fix_spec.get("strategy", "clip")
        lower = fix_spec.get("lower")
        upper = fix_spec.get("upper")
        n_outliers = None
        try:
            from isa_phm.utils import detect_outliers as _detect
            if col in df.columns:
                mask, _ = _detect(
                    df[col].to_numpy(dtype=float),
                    method=method, threshold=threshold, lower=lower, upper=upper,
                )
                n_outliers = int(mask.sum())
        except Exception:
            n_outliers = None
        try:
            df = assay_proxy.fix_outliers(
                df, method=method, threshold=threshold,
                lower=lower, upper=upper, strategy=strategy, column=col,
            )
        except Exception as e:
            return {"status": "transform_error", "step": "fix_outliers",
                    "message": f"{type(e).__name__}: {e}"}
        step = {"op": "fix_outliers", "method": method, "threshold": threshold,
                "strategy": strategy, "column": col}
        if lower is not None:
            step["lower"] = lower
        if upper is not None:
            step["upper"] = upper
        if n_outliers is not None:
            step["n_outliers_detected"] = n_outliers
        applied.append(step)
        transforms.append(step)

    if fill_spec is not None:
        strategy = fill_spec.get("strategy", "interpolate")
        try:
            df = assay_proxy.fill_missing_values(df, strategy=strategy)
        except Exception as e:
            return {"status": "transform_error", "step": "fill_missing",
                    "message": f"{type(e).__name__}: {e}"}
        step = {"op": "fill_missing", "strategy": strategy}
        applied.append(step)
        transforms.append(step)

    if not applied:
        return {
            "status": "noop",
            "run_id": run_id,
            "message": (
                "No transform applied. Pass fix_outliers and/or fill_missing "
                "(each may be {} to use defaults), or reset_to_raw=true to clear."
            ),
        }

    after_stats = _value_stats(df)
    working_copies[key] = {
        "df": df,
        "dataset": dataset_name,
        "study": study_title,
        "sensor": sensor_alias,
        "run_id": run_id,
        "transforms": transforms,
        "raw_stats": raw_stats,
    }

    return {
        "status": "ok",
        "run_id": run_id,
        "started_from": started_from,
        "applied_now": _jsonify(applied),
        "transforms_total": _jsonify(transforms),
        "n_rows": int(len(df)),
        "raw_stats": _jsonify(raw_stats),
        "before_stats": _jsonify(before_stats),
        "after_stats": _jsonify(after_stats),
        "message": (
            f"Working copy updated for run '{run_id}' ({len(transforms)} transform(s) "
            f"total). load_run_csv and single-run make_plot for this run now use it "
            f"until reset_to_raw."
        ),
    }


def _lifecycle_features_via_wrapper(parser: ISAParser, study_title: str, sensor_alias: str) -> dict:
    assay_proxy, err = _resolve_assay_proxy(parser, study_title, sensor_alias)
    if err is not None:
        return err
    try:
        lc_df = assay_proxy.lifecycle_features(file_type="auto")
    except Exception as e:
        return {"status": "read_error", "message": f"{type(e).__name__}: {e}"}

    # Trim to a manageable JSON payload.
    per_run = lc_df.to_dict(orient="records")

    # Compute a simple RMS-trend summary if present.
    trend = None
    if "rms" in lc_df.columns and len(lc_df) >= 3:
        import numpy as np
        rms_vals = lc_df["rms"].dropna().to_numpy(dtype=float)
        if rms_vals.size >= 3:
            x = np.arange(rms_vals.size)
            slope = float(np.polyfit(x, rms_vals, 1)[0])
            trend = {
                "rms_slope": slope,
                "monotonic_increasing": bool(all(b >= a for a, b in zip(rms_vals, rms_vals[1:]))),
                "direction": "increasing" if slope > 0 else ("decreasing" if slope < 0 else "flat"),
            }

    return {
        "status": "ok",
        "n_runs_processed": int(len(per_run)),
        "per_run": _jsonify(per_run),
        "trend": trend,
    }


# Study-wide kinds operate on a StudyProxy and span multiple sensors.
_STUDY_PLOT_KINDS = {
    "multi_lifecycle", "sensor_boxplot", "sensor_lifecycle_correlation",
    "cross_correlation",
}
# Single-sensor kinds that span all runs (no run_id needed).
_MULTI_RUN_KINDS = {"lifecycle", "correlation", "variability", "waterfall"}
# Kinds whose `feature` argument is meaningful.
_FEATURE_KINDS = {"lifecycle", "multi_lifecycle", "sensor_lifecycle_correlation"}


def _resolve_study_proxy(parser: ISAParser, study_title: str):
    """Return (study_proxy, error_dict). On success error_dict is None."""
    if parser.wrapper is None:
        return None, _wrapper_unavailable(parser)
    try:
        return parser.wrapper.study(study_title), None
    except Exception as e:
        return None, {"error": f"Could not open study '{study_title}': {type(e).__name__}: {e}"}


def _build_multi_lifecycle(study_proxy, feature: str):
    """Compose {sensor_alias: lifecycle_df} for every sensor in the study and
    overlay them via the wrapper's ISAPlotter.plot_multi_lifecycle. This is the
    'Multi-RMS' comparison. Returns a Bokeh figure, or an error dict."""
    try:
        summaries = study_proxy.list_assays()
    except Exception as e:
        return {"error": f"Could not list sensors: {type(e).__name__}: {e}"}
    if not summaries:
        return {"error": "This study has no assays/sensors to overlay."}

    dfs: dict = {}
    skipped: list[str] = []
    for summ in summaries:
        label = getattr(summ, "sensor_alias", None) or getattr(summ, "assay_id", "?")
        try:
            lc = study_proxy.assay(summ.assay_id).lifecycle_features(file_type="auto")
        except Exception:
            skipped.append(label)
            continue
        if lc is None or lc.empty or feature not in lc.columns:
            skipped.append(label)
            continue
        dfs[label] = lc

    if not dfs:
        return {"error": (
            f"Could not compute '{feature}' lifecycle features for any sensor "
            f"in this study."
        )}
    try:
        from isa_phm.plotter import ISAPlotter
        return ISAPlotter().plot_multi_lifecycle(dfs, feature=feature)
    except Exception as e:
        return {"error": f"plot_multi_lifecycle failed: {type(e).__name__}: {e}"}


# Single-run kinds that can be re-rendered from an in-memory (cleaned) df.
_WORKING_COPY_PLOT_KINDS = {
    "timeseries", "fft", "psd", "spectrogram", "distribution",
    "missing_values", "outlier_comparison",
}


def _render_single_run_plot(assay_proxy, kind: str, working_df, title: str):
    """Render a single-run plot from an in-memory DataFrame (a cleaned working
    copy), so plots reflect process_signal transforms. Methods that accept df=
    are used directly; the rest call the underlying ISAPlotter with the same fs/
    label inference the wrapper applies. Returns a Bokeh figure or an error dict.
    outlier_comparison is handled by the caller (it needs the raw df too)."""
    plotter = assay_proxy._plotter
    if kind == "distribution":
        return assay_proxy.plot_distribution(df=working_df, title=title)
    if kind == "fft":
        return assay_proxy.plot_frequency_domain(
            df=working_df, fs=assay_proxy._infer_fs(), title=title
        )
    if kind == "timeseries":
        return plotter.plot_timeseries(
            working_df, xlabel="time", ylabel=assay_proxy._ylabel(), title=title
        )
    if kind == "missing_values":
        return plotter.plot_missing_values(working_df, title=title)
    fs = assay_proxy._infer_fs()
    if kind == "psd":
        if fs is None:
            return {"error": (
                "psd needs a sampling frequency, which could not be inferred from "
                "the protocol. Re-run make_plot from raw data or add an Hz parameter."
            )}
        return plotter.plot_power_spectral_density(working_df, fs=fs, title=title)
    if kind == "spectrogram":
        if fs is None:
            return {"error": (
                "spectrogram needs a sampling frequency, which could not be inferred "
                "from the protocol."
            )}
        return plotter.plot_spectrogram(working_df, fs=fs, title=title)
    return {"error": f"kind='{kind}' cannot render from a working copy."}


def _make_plot_via_wrapper(
    parser: ISAParser, tool_input: dict, dataset_name: str, working_copies: dict
) -> dict:
    """Render a Bokeh plot via the wrapper. Dispatches across single-run,
    single-sensor-multi-run, study-wide, and two-sensor plot kinds."""
    if parser.wrapper is None:
        return _wrapper_unavailable(parser)

    kind = tool_input["kind"]
    study_title = tool_input.get("study_title")
    sensor_alias = tool_input.get("sensor_alias")
    sensor_alias_2 = tool_input.get("sensor_alias_2")
    relative_path = tool_input.get("relative_path")
    feature = tool_input.get("feature", "rms")

    if not study_title:
        return {"error": "Plotting requires study_title."}

    def _title(*extra: str) -> str:
        parts = [dataset_name, study_title, *extra]
        return " · ".join(p for p in parts if p)

    def _ok(plot_id: str, run_id: str | None = None) -> dict:
        out = {
            "plot_id": plot_id,
            "plot_kind": kind,
            "message": f"{kind} plot registered. The app will render it.",
        }
        if kind in _FEATURE_KINDS:
            out["feature"] = feature
        if run_id is not None:
            out["run_id"] = run_id
        return out

    # ------------------------------------------------------------------
    # Study-wide kinds — operate on a StudyProxy, span multiple sensors.
    # ------------------------------------------------------------------
    if kind in _STUDY_PLOT_KINDS:
        study_proxy, err = _resolve_study_proxy(parser, study_title)
        if err is not None:
            return err
        try:
            if kind == "multi_lifecycle":
                fig = _build_multi_lifecycle(study_proxy, feature)
            elif kind == "sensor_boxplot":
                fig = study_proxy.plot_sensor_boxplot(file_type="auto")
            elif kind == "sensor_lifecycle_correlation":
                fig = study_proxy.plot_sensor_lifecycle_correlation(
                    feature=feature, file_type="auto"
                )
            elif kind == "cross_correlation":
                if not sensor_alias or not sensor_alias_2:
                    return {"error": (
                        "kind='cross_correlation' needs sensor_alias and "
                        "sensor_alias_2 (the two sensors to correlate)."
                    )}
                f1 = parser.get_assay_filename(study_title, sensor_alias)
                f2 = parser.get_assay_filename(study_title, sensor_alias_2)
                if not f1 or not f2:
                    return {"error": (
                        "Could not map one of the sensor aliases to an assay. "
                        "Check the sensor names."
                    )}
                fig = study_proxy.plot_cross_correlation(f1, f2, file_type="auto")
            else:  # pragma: no cover
                return {"error": f"Unhandled study plot kind: {kind}"}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

        if isinstance(fig, dict):  # helper returned an error
            return fig
        suffix = f"{kind} ({feature})" if kind in _FEATURE_KINDS else kind
        return _ok(_register_figure(fig, kind=kind, title=_title(suffix)))

    # ------------------------------------------------------------------
    # Single-sensor kinds — need a sensor_alias. Use `is None` (not
    # truthiness): an empty sensor alias is still a valid channel.
    # ------------------------------------------------------------------
    if sensor_alias is None:
        return {"error": f"kind='{kind}' requires study_title and sensor_alias."}

    assay_proxy, err = _resolve_assay_proxy(parser, study_title, sensor_alias)
    if err is not None:
        return err

    # Multi-run single-sensor kinds — no run_id needed.
    if kind in _MULTI_RUN_KINDS:
        try:
            if kind == "lifecycle":
                fig = assay_proxy.plot_lifecycle(feature=feature, file_type="auto")
            elif kind == "correlation":
                fig = assay_proxy.plot_correlation(file_type="auto")
            elif kind == "variability":
                fig = assay_proxy.plot_variability(file_type="auto")
            elif kind == "waterfall":
                fig = assay_proxy.plot_waterfall(file_type="auto")
            else:  # pragma: no cover
                return {"error": f"Unhandled multi-run kind: {kind}"}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        suffix = f"{kind} ({feature})" if kind in _FEATURE_KINDS else kind
        return _ok(_register_figure(fig, kind=kind, title=_title(sensor_alias, suffix)))

    # Per-run single-sensor kinds — resolve a run_id (default: last run).
    run_id = _resolve_run_id(parser, assay_proxy, study_title, sensor_alias, relative_path)

    wc = working_copies.get(_wc_key(dataset_name, study_title, sensor_alias, run_id))
    suffix = f"{kind} (cleaned)" if wc is not None else kind
    title = _title(sensor_alias, suffix)

    try:
        if wc is not None:
            # Render from the cached cleaned data so transforms are reflected.
            if kind == "outlier_comparison":
                df_raw = assay_proxy.load_dataframe(run_id=run_id, file_type="auto")
                fig = assay_proxy.plot_outlier_comparison(df_raw, wc["df"], title=title)
            else:
                fig = _render_single_run_plot(assay_proxy, kind, wc["df"], title)
        elif kind == "timeseries":
            fig = assay_proxy.plot_timeseries(run_id=run_id, file_type="auto")
        elif kind == "fft":
            fig = assay_proxy.plot_frequency_domain(run_id=run_id, file_type="auto")
        elif kind == "psd":
            fig = assay_proxy.plot_psd(run_id=run_id, file_type="auto")
        elif kind == "spectrogram":
            fig = assay_proxy.plot_spectrogram(run_id=run_id, file_type="auto")
        elif kind == "distribution":
            fig = assay_proxy.plot_distribution(run_id=run_id, file_type="auto")
        elif kind == "missing_values":
            fig = assay_proxy.plot_missing_values(run_id=run_id, file_type="auto")
        elif kind == "outlier_comparison":
            df = assay_proxy.load_dataframe(run_id=run_id, file_type="auto")
            df_clean = assay_proxy.fix_outliers(df)
            fig = assay_proxy.plot_outlier_comparison(df, df_clean)
        else:
            return {"error": f"Unknown plot kind: {kind}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    if isinstance(fig, dict):  # _render_single_run_plot returned an error
        return fig

    return _ok(_register_figure(fig, kind=kind, title=title), run_id=run_id)
