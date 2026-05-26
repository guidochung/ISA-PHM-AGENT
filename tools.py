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
        # Signal-level tools (require a data base directory)
        # ------------------------------------------------------------------
        {
            "name": "load_run_csv",
            "description": (
                "Load the head and summary stats of one run's CSV via the "
                "wrapper. Requires the data directory to be configured in the "
                "sidebar. Returns n_rows, columns, dtypes, head, and "
                "min/max/mean/std/rms stats. If data is not available, returns "
                "status='data_not_available'. Providing study_title and "
                "sensor_alias is optional but speeds up lookup."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    **dataset_param,
                    "relative_path": {
                        "type": "string",
                        "description": "Relative path as recorded in the ISA metadata.",
                    },
                    "study_title": {"type": "string", "description": "Optional. The study this run belongs to."},
                    "sensor_alias": {"type": "string", "description": "Optional. The sensor alias for this run's assay."},
                    "max_rows": {"type": "integer", "default": 5},
                },
                "required": ["dataset_name", "relative_path"],
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
            "name": "make_plot",
            "description": (
                "Render a Bokeh plot via the ISA-PHM wrapper. "
                "kind ∈ {timeseries, fft, psd, spectrogram, lifecycle, "
                "outlier_comparison, waterfall, distribution}. Requires the "
                "dataset's data directory to be configured. For per-run plots "
                "(everything except lifecycle/waterfall), defaults to the last "
                "run if relative_path is omitted. Returns plot_id — the app "
                "renders the figure with st.bokeh_chart()."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    **dataset_param,
                    "kind": {
                        "type": "string",
                        "enum": [
                            "timeseries", "fft", "psd", "spectrogram",
                            "lifecycle", "outlier_comparison",
                            "waterfall", "distribution",
                        ],
                    },
                    "study_title": {"type": "string"},
                    "sensor_alias": {"type": "string"},
                    "relative_path": {
                        "type": "string",
                        "description": (
                            "Optional. ISA-style relative path of one run's CSV. "
                            "If omitted, defaults to the most recent run. Ignored "
                            "for kind='lifecycle' and kind='waterfall'."
                        ),
                    },
                    "feature": {
                        "type": "string",
                        "default": "rms",
                        "description": "For kind='lifecycle': which feature to plot (rms, kurtosis, crest_factor, peak2peak, mean, std).",
                    },
                },
                "required": ["dataset_name", "kind", "study_title", "sensor_alias"],
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
) -> str:
    """Dispatch a Claude tool_use call. Always returns JSON.
    data_dirs maps dataset_name → base_dir for signal tools."""
    data_dirs = data_dirs or {}

    try:
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
            result = _load_run_csv_via_wrapper(parser, tool_input)

        elif tool_name == "lifecycle_features":
            result = _lifecycle_features_via_wrapper(
                parser, tool_input["study_title"], tool_input["sensor_alias"]
            )

        elif tool_name == "make_plot":
            result = _make_plot_via_wrapper(parser, tool_input, dataset_name)

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


def _load_run_csv_via_wrapper(parser: ISAParser, tool_input: dict) -> dict:
    if parser.wrapper is None:
        return _wrapper_unavailable(parser)

    relative_path = tool_input["relative_path"]
    max_rows = tool_input.get("max_rows", 5)
    study_title = tool_input.get("study_title")
    sensor_alias = tool_input.get("sensor_alias")

    # If caller provided study + sensor, resolve via wrapper proxies.
    if study_title and sensor_alias:
        assay_proxy, err = _resolve_assay_proxy(parser, study_title, sensor_alias)
        if err is not None:
            return err
        run_id = parser.get_run_id_for_path(study_title, sensor_alias, relative_path)
        try:
            df, meta = assay_proxy.load_dataframe_with_meta(run_id=run_id, file_type="auto")
        except Exception as e:
            return {"status": "read_error", "message": f"{type(e).__name__}: {e}"}
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
        "n_rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
        "head": _jsonify(head),
        "stats": stats,
        "load_meta": _jsonify(meta),
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


def _make_plot_via_wrapper(parser: ISAParser, tool_input: dict, dataset_name: str) -> dict:
    """Render a Bokeh plot via the wrapper's AssayProxy/RunProxy."""
    if parser.wrapper is None:
        return _wrapper_unavailable(parser)

    kind = tool_input["kind"]
    study_title = tool_input.get("study_title")
    sensor_alias = tool_input.get("sensor_alias")
    relative_path = tool_input.get("relative_path")
    feature = tool_input.get("feature", "rms")

    if not study_title or not sensor_alias:
        return {"error": "Plotting requires study_title and sensor_alias."}

    assay_proxy, err = _resolve_assay_proxy(parser, study_title, sensor_alias)
    if err is not None:
        return err

    def _title(suffix: str) -> str:
        parts = [dataset_name, study_title, sensor_alias, suffix]
        return " · ".join(p for p in parts if p)

    # Lifecycle is multi-run — no relative_path needed.
    if kind == "lifecycle":
        try:
            fig = assay_proxy.plot_lifecycle(feature=feature, file_type="auto")
            plot_id = _register_figure(fig, kind="lifecycle", title=_title(f"Lifecycle ({feature})"))
            return {
                "plot_id": plot_id,
                "plot_kind": "lifecycle",
                "feature": feature,
                "message": "Lifecycle plot registered. The app will render it.",
            }
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    # Per-run plots — need a run_id. If a relative_path was supplied,
    # resolve it. Otherwise default to the last run (the most-degraded one
    # in lifecycle-style datasets).
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

    try:
        if kind == "timeseries":
            fig = assay_proxy.plot_timeseries(run_id=run_id, file_type="auto")
        elif kind == "fft":
            fig = assay_proxy.plot_frequency_domain(run_id=run_id, file_type="auto")
        elif kind == "psd":
            fig = assay_proxy.plot_psd(run_id=run_id, file_type="auto")
        elif kind == "spectrogram":
            fig = assay_proxy.plot_spectrogram(run_id=run_id, file_type="auto")
        elif kind == "outlier_comparison":
            fig = assay_proxy.plot_outlier_comparison(run_id=run_id, file_type="auto")
        elif kind == "waterfall":
            fig = assay_proxy.plot_waterfall(file_type="auto")
        elif kind == "distribution":
            fig = assay_proxy.plot_distribution(run_id=run_id, file_type="auto")
        else:
            return {"error": f"Unknown plot kind: {kind}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

    plot_id = _register_figure(fig, kind=kind, title=_title(kind))
    return {
        "plot_id": plot_id,
        "plot_kind": kind,
        "run_id": run_id,
        "message": f"{kind} plot registered. The app will render it.",
    }
