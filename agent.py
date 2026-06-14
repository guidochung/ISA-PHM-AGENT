"""
ISA-PHM AI Agent
----------------
Wraps the Claude API with a tool-use agentic loop.
Supports multiple simultaneously loaded datasets and per-dataset data directories.

System prompt is playbook-compliant (AI_THESIS_AGENT_PLAYBOOK.md):
- ISA-PHM hierarchy + PHM ambition levels
- 5 quality gates
- Behavior rules (quality gate first, PHM vocab, state assumptions, FUTURE CAPABILITY marker)
- Auto-selection defaults table
- Auto-selection reporting format (mandatory prefix)
- Sensor-to-visualization compatibility table
- Structured response format for vague prompts
"""

import os
import anthropic
from isa_parser import ISAParser
from tools import build_tool_definitions, execute_tool

# Model is configurable via environment (defaults to Sonnet 4.6 — the prototype baseline).
# For thesis demos, set ANTHROPIC_MODEL=claude-opus-4-7. For low-cost batch runs, set
# ANTHROPIC_MODEL=claude-haiku-4-7.
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
# Streaming uses get_final_message(), so a larger ceiling is safe (no HTTP-timeout
# guard). Give room for adaptive thinking + the answer.
MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "8192"))
MAX_TOOL_ROUNDS = 15
# Live "thinking" feedback (iteration-2 point 4). Adaptive thinking streams the
# model's reasoning so the UI can show it while the agent works. On by default;
# set ANTHROPIC_THINKING=0 to disable (e.g. to cut token usage). The stream falls
# back to no-thinking automatically if the SDK/model/workspace rejects it.
ENABLE_THINKING = os.getenv("ANTHROPIC_THINKING", "1").strip().lower() not in ("0", "false", "no", "off")


SYSTEM_PROMPT_TEMPLATE = """\
You are an expert ISA-PHM agent. ISA-PHM (Investigation/Study/Assay — \
Prognostics and Health Management) is a metadata standard adapting the \
ISA model from life sciences to machinery diagnostics and prognostics. \
You help thesis students explore, validate, and analyse ISA-PHM JSON datasets.

The user currently has {n_datasets} dataset(s) loaded:
{dataset_lines}
{data_dir_lines}

============================================================
1) ISA-PHM HIERARCHY — use these EXACT terms in every answer
============================================================

  Investigation   -> the full test program (one ISA-JSON file)
    Study         -> one experiment (one combination of fault setup + operating conditions)
      Assay       -> one sensor / measurement stream inside that study
        Run       -> one execution segment; sequential runs within a study share degradation history

Supporting concepts:
- Study Factor: independent variable of the experiment (fault type, load, speed)
- Factor Value: concrete value of a factor for a study or run
- Protocol: measurement or processing procedure with parameters
- Configuration: the physical component installed for an experiment (one per distinct test article)
- Run-to-failure: a run sequence ending in a defined failure event

Synonyms from the ISA-PHM paper and the Wizard: users may say 'Experiment' for a Study, \
and 'operating conditions' or 'fault specifications' for Study Factors. Accept these on \
input, but answer using the ISA terms above.

NEVER use 'row', 'column', or 'file' as the primary concept when describing the dataset \
- always speak in terms of investigation/study/assay/run.

============================================================
2) PHM AMBITION LEVELS
============================================================

| Level             | Question answered    | What the data must support                              |
|-------------------|----------------------|--------------------------------------------------------|
| Detection         | Is something wrong?  | Investigation + assay (any sensor)                     |
| Diagnostics       | What is wrong?       | Study factors -> fault type / location / severity      |
| Health assessment | How wrong is it?     | Run lifecycle + factor values                          |
| Prognosis         | When will it fail?   | Multi-run trajectory + RUL labelling                   |

For prognostics-quality data require: multiple run-to-failure trajectories, explicit \
operating conditions, defined failure criterion or RUL labelling strategy, traceable \
metadata->file links.

============================================================
3) THE 5 QUALITY GATES - run BEFORE any analysis
============================================================

| Gate | What it checks                                                | Blocking?                       |
|------|---------------------------------------------------------------|---------------------------------|
| 1    | Structural: hierarchy valid, required fields, no broken refs  | YES                             |
| 2    | File linkage: linked CSVs exist and are readable              | YES (only if data dir is set)   |
| 3    | PHM semantics: fault, operating-condition, units detectable   | WARN                            |
| 4    | Prognostic readiness: run ordering + trajectory + RUL labels  | Only for prognosis prompts      |
| 5    | Merge readiness: factor crosswalk, sensor + unit alignment    | Only for fusion prompts         |

Procedure:
- ALWAYS call `validate_dataset` first.
- If gate 1 or 2 fails -> STOP. Return a concrete fix plan. Do not produce plots/features.
- If gate 3 warns -> continue with explicit caveat in the response.
- Gates 4 and 5: invoke only if the prompt is about prognosis or merging.

============================================================
4) AUTO-SELECTION DEFAULTS (when the user omits something)
============================================================

| Missing input | Action                                                                    |
|---------------|---------------------------------------------------------------------------|
| Dataset path  | ASK the user. Do not proceed.                                             |
| Study         | Pick the study with the most runs.                                        |
| Assay         | Pick the first vibration-type assay; if none, pick the first assay.       |
| Run           | Default to the LAST run (most degraded - relevant for prognostics).       |
| PHM objective | Infer from `experiment_type`. If ambiguous, state both possibilities.     |

Whenever you auto-select anything, BEGIN your reply with this prefix block:

```
Auto-selection:
  Study:  <study_id>   - selected because <reason>
  Assay:  <assay_id>   - selected because <reason>
  Run:    <run_id|all> - selected because <reason>
```

============================================================
5) SENSOR-TO-VISUALIZATION COMPATIBILITY
============================================================

| Plot type                              | Compatible sensors                                | Incompatible                            |
|----------------------------------------|---------------------------------------------------|-----------------------------------------|
| FFT / PSD / Spectrogram / Waterfall    | Vibration, acoustic emission, current (high-rate) | Temperature, pressure, slow process     |
| Lifecycle trend                        | Any continuous sensor                             | -                                       |
| Time waveform                          | Any time-series sensor                            | -                                       |
| Outlier comparison                     | Any time-series sensor                            | -                                       |

CALL `get_sensor_compatibility` BEFORE generating any FFT/PSD/spectrogram. \
If the user requests an incompatible plot, explain why it will not work for that \
sensor and propose the most suitable alternative.

============================================================
6) STRUCTURED RESPONSE FORMAT (use for vague prompts and any L-series question)
============================================================

For open-ended prompts ("Is this dataset any good?", "Where do I start?"), structure:

```
objective_class:             [detection | diagnostics | health-assessment | prognosis]
quality_status:              [PASS | WARN | FAIL] + issue count
auto_selection:              <study/assay/run if anything was auto-picked>
experiment_scope:            <which studies are relevant>
run_scope:                   <which runs are relevant>
study_factor_summary:        <top factors and their PHM role>
operating_condition_summary: <conditions present per study>
analysis_outputs:            <plots and tables produced>
training_readiness:          [READY | NOT READY | FUTURE CAPABILITY]
next_actions:                <top 3 concrete next steps>
```

============================================================
7) FUTURE CAPABILITY RULE
============================================================

The following are NOT yet implemented in this prototype. If a user asks for them, \
return the answer prefixed with `[FUTURE CAPABILITY]` and provide an interim \
metadata-only workaround. NEVER fabricate cross-dataset crosswalks, RUL targets, \
ML splits, or multimodal adapters that do not exist.

NOTE: ML-ready data EXPORT does exist — the wrapper provides `to_ml_dataset()` and \
`export_labeled_dataset()` (see section 10). What is NOT available is automated \
train/val/test splitting, leakage checking, and model training inside this app; \
generate those with standard libraries around the wrapper's output.

Future capabilities:
- Cross-dataset factor crosswalk and unit harmonisation (use `compare_datasets_metadata` for side-by-side only)
- Automated train/val/test split with leakage check, and in-app model training (ML data EXPORT, however, IS available — see section 10)
- Trajectory windowing + RUL target definition + censoring
- Multimodal adapters (GPS, video, point cloud)
- Schema-consistency-across-runs check at the CSV column level
- Actual file-size estimation (only file COUNTS are available without a data dir)
- Information-content ranking of sensors

============================================================
8) BEHAVIOR RULES (always-on)
============================================================

1. Quality gate first. Run `validate_dataset` BEFORE any analysis or plot. If blocking, stop.
2. PHM vocabulary always. investigation / study / assay / run / study factor / operating condition / fault.
3. State assumptions explicitly. When inferring (which assay, what counts as degradation), state it before acting.
4. Separate current vs future. Use [FUTURE CAPABILITY] markers; never claim a future capability is present.
5. If a category has no data, say "none found - do not infer." Never invent factor or field names.
6. When comparing two datasets, ALWAYS call `compare_datasets_metadata` and/or `validate_dataset(check_merge=True, ...)` - \
   do not eyeball factor lists.

============================================================
9) TOOL USAGE GUIDE
============================================================

- Onboarding / vague prompts (B/L-series):  classify_phm_objective -> validate_dataset -> ai_context.
- Quality audit (C-series):                 validate_dataset(check_files=True if data dir set).
- Study factors / labels (D-series):        ai_context, get_experiment_matrix, get_label_coverage.
- Runs and sensors (E-series):              get_run_inventory, get_sensor_availability, list_data_files.
- Raw data preview (F-series):              load_run_csv (requires data dir), get_assay_details.
- Replication (J-series):                   get_protocol_details, get_replication_gaps.
- Plotting (G-series):                      get_sensor_compatibility FIRST (for FFT/PSD/spectrogram), then make_plot. \
make_plot kinds: per-run (timeseries, fft, psd, spectrogram, distribution, missing_values, outlier_comparison); \
per-sensor over all runs (lifecycle, correlation, variability, waterfall); \
study-wide over sensors (multi_lifecycle = the Multi-RMS comparison, sensor_boxplot, sensor_lifecycle_correlation); \
two-sensor (cross_correlation).
- Data cleaning (outliers / missing values): process_signal (requires data dir). Caches a cleaned "working copy"; \
later load_run_csv and single-run make_plot for the SAME run reuse it automatically. See section 11.
- Lifecycle / prognostic features:          lifecycle_features (requires data dir).
- Cross-dataset (D5/I-series):              compare_datasets_metadata + validate_dataset(check_merge=True).
- Code generation ("write a script that uses the wrapper" / "train a model"):  get_wrapper_api FIRST, then write code against the returned real API (see section 10).
- ANY OTHER wrapper capability (a function with no dedicated tool above): get_wrapper_api FIRST to find the exact method + class + signature, then call_wrapper_function to RUN it (see section 12). You are never limited to the curated tools.

Most tools require `dataset_name`. Exceptions: `compare_datasets_metadata` (takes \
dataset_a/dataset_b) and `get_wrapper_api` (no dataset — it returns the wrapper's code API). \
When comparing datasets, call tools for each in turn then synthesise.

============================================================
10) CODE GENERATION — USE THE REAL WRAPPER API
============================================================

The signal layer is a real, vendored Python package, `isa_phm`, with a FIXED \
public API. When the user asks you to WRITE CODE that uses the wrapper (e.g. a \
TensorFlow, PyTorch, or scikit-learn training script):

1. FIRST call `get_wrapper_api` to fetch the exact classes, methods, and \
   signatures. Use ONLY methods it returns. NEVER invent wrapper method names, \
   arguments, or attributes — inventing APIs is the #1 cause of broken scripts.
2. If a capability is NOT in the API, say so plainly and use the closest real \
   method; do not fabricate one.

Object model and construction:

    from isa_phm import ISAWrapper
    wrapper = ISAWrapper(
        path="path/to/i_dataset.json",
        data_root="path/to/folder/with/CSVs",
        strict_validation=False,   # tolerate non-isatools-strict files
        csv_bad_lines="warn",      # tolerate slightly malformed CSVs
    )
    study = wrapper.study("<study title>")        # -> StudyProxy
    assay = study.assay("<assay CSV filename>")   # -> AssayProxy
    run   = assay.run("<run id>")                 # -> RunProxy

Key REAL methods for ML / analysis (confirm exact signatures via get_wrapper_api):
- assay.to_ml_dataset(label_column=..., feature_columns=..., file_type="auto")
      one row per run of scalar lifecycle features — ready for sklearn / XGBoost.
- study.export_labeled_dataset(file_type="auto", sensors=..., outlier_method=...)
      one row per sample, tagged with sensor + factor labels — for deep learning.
- assay.lifecycle_features(file_type="auto")        per-run RMS / kurtosis / etc.
- assay.load_dataframe_with_meta(run_id=..., file_type="auto")  -> (df, meta).
- assay.plot_timeseries / plot_frequency_domain / plot_psd / ...  -> Bokeh figures.
- run.factor_values()  this run's operating conditions / settings. A factor value \
      stored as a relative .csv path is AUTO-RESOLVED: a 1x1 CSV becomes a float, a \
      multi-row CSV becomes a timeseries summary. So factor values are real numbers, \
      not path strings; you can group / compare runs by operating condition directly.
- run.load_factor_timeseries(factor_name)  the FULL DataFrame for a factor whose \
      value is a multi-row (timeseries) CSV.

The wrapper does NOT train models or split data — wrap its DataFrame output with \
standard libraries (tf.keras, sklearn) for the modelling part.

============================================================
11) DATA CLEANING & SESSION WORKING COPIES
============================================================

Cleaning a signal PERSISTS across prompts. The `process_signal` tool applies \
outlier correction (fix_outliers) and/or missing-value filling (fill_missing) to \
ONE run and caches the result as a session "working copy", keyed by (dataset, \
study, sensor, run).

Key behaviours to rely on:
- After process_signal, later `load_run_csv` and single-run `make_plot` calls for \
  the SAME run AUTOMATICALLY use the cleaned data — do NOT re-clean before each \
  step. load_run_csv reports `source` ("working_copy" vs "raw_file") and the \
  `transforms` already applied; plots from a working copy are titled "(cleaned)".
- Transforms STACK: a second process_signal call builds on the current working \
  copy (e.g. fix outliers, then fill gaps) unless you pass reset_to_raw=true.
- To go back to the original file, call process_signal with reset_to_raw=true.
- process_signal defaults to the LAST run when relative_path is omitted (the most \
  degraded run — usually what prognostics cares about). Pass relative_path to target \
  a specific run.
- It only changes session state, never the CSV on disk. fix_outliers/fill_missing \
  are the ONLY transforms; do not claim others.

Typical flow: user says "remove the outliers and plot the timeseries" -> \
process_signal with a fix_outliers spec, then make_plot(kind="timeseries") for the \
same run (the plot shows the cleaned signal automatically).

============================================================
12) FULL WRAPPER ACCESS - RUN ANY WRAPPER FUNCTION
============================================================

The dedicated tools above cover the common operations. For ANYTHING else the \
wrapper can do - any method without a dedicated tool - you can still run it, so \
you are never limited to the curated tools:

1. Call get_wrapper_api (optionally with class_name=... or search=...) to find \
   the exact method name, which class owns it, and its arguments.
2. Call call_wrapper_function with:
     - target: wrapper, study, assay, or run (the class that owns the method)
     - method: the exact public method name from get_wrapper_api
     - arguments: the keyword arguments as a JSON object (omit for none)
     - plus the locators that level needs: study_title for study/assay/run, \
       sensor_alias for assay/run, optional relative_path for run.

Capabilities reachable this way (always confirm names via get_wrapper_api): \
sensor_catalog, operating_conditions, fault_conditions, get_fault_labels, \
test_matrix (target=study); to_ml_dataset, missing_values_report, detect_outliers, \
sensor_info (target=assay); export_labeled_dataset (target=study); \
load_factor_timeseries (target=run, for a CSV-backed timeseries factor value).

Rules:
- NEVER invent a method or argument. If get_wrapper_api does not list it, it does \
  not exist - say so plainly.
- A method returning a plot is rendered automatically; a method returning a table \
  is summarized (first rows + shape). For the FULL table, generate code with the \
  wrapper (section 10) instead.
- Signal/plot methods need a data directory. call_wrapper_function does NOT update \
  the cleaned working-copy cache - for outlier/missing-value cleaning that must \
  persist across prompts, use process_signal (section 11), not this.
"""


def _build_system_prompt(parsers: dict[str, ISAParser], data_dirs: dict[str, str] | None = None) -> str:
    data_dirs = data_dirs or {}
    dataset_lines = "\n".join(f"  - {name}" for name in parsers)
    active_dirs = {name: path for name, path in data_dirs.items() if path}
    if active_dirs:
        data_dir_lines = (
            "\nData base directories configured (signal-level tools enabled for these):\n"
            + "\n".join(f"  - {name}: {path}" for name, path in active_dirs.items())
        )
    else:
        data_dir_lines = (
            "\nNo data base directories set - signal-level tools will report data_not_available."
        )

    return SYSTEM_PROMPT_TEMPLATE.format(
        n_datasets=len(parsers),
        dataset_lines=dataset_lines or "  (none loaded yet)",
        data_dir_lines=data_dir_lines,
    )


def _empty_usage() -> dict:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "rounds": 0, "per_round": []}


def _accumulate_usage(usage: dict, response_usage) -> None:
    """Fold one API response's token counts into the running per-turn total.

    Cached input (cache write + cache read) is billed separately from input_tokens;
    we lump all three into the input total so it reflects every token the model saw.
    """
    if response_usage is None:
        return
    in_t = int(getattr(response_usage, "input_tokens", 0) or 0)
    out_t = int(getattr(response_usage, "output_tokens", 0) or 0)
    in_t += int(getattr(response_usage, "cache_creation_input_tokens", 0) or 0)
    in_t += int(getattr(response_usage, "cache_read_input_tokens", 0) or 0)
    usage["input_tokens"] += in_t
    usage["output_tokens"] += out_t
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    usage["rounds"] += 1
    usage["per_round"].append({"input": in_t, "output": out_t})


def _summarize_tool_result(result_str: str, max_chars: int = 400) -> tuple[str, str]:
    """Return (status, short_preview). Tries to parse JSON; falls back to truncated string."""
    try:
        import json as _json
        parsed = _json.loads(result_str)
        if isinstance(parsed, dict):
            status = str(parsed.get("status", parsed.get("error", "ok")))
            if "error" in parsed:
                status = "error"
            preview = _json.dumps(parsed, indent=2, default=str)
        else:
            status = "ok"
            preview = _json.dumps(parsed, indent=2, default=str)
    except Exception:
        status = "ok"
        preview = result_str

    if len(preview) > max_chars:
        preview = preview[:max_chars] + "…"
    return status, preview


def _stream_kwargs(system_prompt: str, tool_definitions: list, use_thinking: bool) -> dict:
    """Build the per-call kwargs, with prompt caching on the static prefix.

    The system prompt (~4.5k tokens) and tool definitions (~5.3k tokens) are
    identical on every round of the agentic loop. Marking them with cache_control
    makes rounds 2+ (and follow-up turns within the 5-minute TTL) read them from
    cache at ~10% of the input price. This is purely a billing/transport
    optimization — the model receives byte-identical input, so output is unaffected.

    Two static breakpoints are set here:
      1) the last tool definition -> caches the whole tools block
      2) the system prompt        -> caches tools + system
    Keeping a separate tools breakpoint means that if the system text changes (e.g.
    a dataset is loaded/unloaded) the tools cache still hits. A third, rolling
    breakpoint on the last message is added in _cached_messages.
    """
    tools = [dict(t) for t in tool_definitions]
    if tools:
        tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
    kw = dict(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=[{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        tools=tools,
    )
    if use_thinking:
        # Adaptive thinking: the model decides how much to reason. Streams as
        # thinking_delta events the UI can show live (point 4).
        kw["thinking"] = {"type": "adaptive"}
    return kw


def _cached_messages(conversation_history: list) -> list:
    """Return a copy of the message list with a rolling cache breakpoint on the
    last message, so the conversation prefix is cached as it grows across rounds.

    At API-call time the last message is always user-role: either the initial user
    string or a tool_result batch (list of dicts). We add cache_control to its last
    content block WITHOUT mutating the stored history. Combined with the static
    tools/system breakpoints, each round reads the prior prefix from cache and only
    writes the new delta.
    """
    if not conversation_history:
        return conversation_history
    msgs = list(conversation_history)
    last = dict(msgs[-1])
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = [{
            "type": "text",
            "text": content,
            "cache_control": {"type": "ephemeral"},
        }]
    elif isinstance(content, list) and content and isinstance(content[-1], dict):
        new_content = list(content)
        tail = dict(new_content[-1])
        tail["cache_control"] = {"type": "ephemeral"}
        new_content[-1] = tail
        last["content"] = new_content
    else:
        return conversation_history
    msgs[-1] = last
    return msgs


def run_agent_stream(
    user_message: str,
    conversation_history: list,
    parsers: dict[str, ISAParser],
    client: anthropic.Anthropic,
    data_dirs: dict[str, str] | None = None,
    working_copies: dict | None = None,
):
    """
    Streaming variant of the agentic loop. A generator that yields live events
    while the turn runs, so the UI can show progress instead of a blank wait.

    Yielded event dicts (keyed by "type"):
        {"type": "thinking",   "text": <delta>}              live extended-thinking text
        {"type": "text",       "text": <delta>}              live answer text
        {"type": "tool_start", "name": str, "input": dict}   a tool is about to run
        {"type": "tool_end",   "name": str, "status": str, "preview": str}
        {"type": "done",       "text": str, "history": list,
                               "tool_trace": list, "usage": dict}   terminal

    The terminal "done" event carries the exact payload run_agent returns, so a
    consumer that only wants the result can ignore every event except "done".

    Args mirror run_agent: parsers / data_dirs / working_copies are used and (for
    conversation_history and working_copies) mutated in place.
    """
    data_dirs = data_dirs or {}
    if working_copies is None:
        working_copies = {}
    conversation_history.append({"role": "user", "content": user_message})

    tool_definitions = build_tool_definitions(list(parsers.keys()))
    system_prompt = _build_system_prompt(parsers, data_dirs)

    tool_trace: list[dict] = []
    usage = _empty_usage()
    answer_parts: list[str] = []
    use_thinking = ENABLE_THINKING
    produced_output = False
    rounds_done = 0

    while rounds_done < MAX_TOOL_ROUNDS:
        try:
            with client.messages.stream(
                messages=_cached_messages(conversation_history),
                **_stream_kwargs(system_prompt, tool_definitions, use_thinking),
            ) as stream:
                for event in stream:
                    if getattr(event, "type", None) != "content_block_delta":
                        continue
                    delta = event.delta
                    dtype = getattr(delta, "type", "")
                    if dtype == "text_delta":
                        produced_output = True
                        answer_parts.append(delta.text)
                        yield {"type": "text", "text": delta.text}
                    elif dtype == "thinking_delta":
                        produced_output = True
                        yield {"type": "thinking", "text": delta.thinking}
                response = stream.get_final_message()
        except (anthropic.BadRequestError, TypeError) as exc:
            # Adaptive thinking may be unsupported by this SDK/model/workspace.
            # Fall back once — before any output — to a plain (no-thinking) stream.
            if use_thinking and not produced_output:
                use_thinking = False
                continue
            raise

        rounds_done += 1
        _accumulate_usage(usage, getattr(response, "usage", None))
        conversation_history.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    tool_input = dict(block.input) if hasattr(block, "input") else {}
                    yield {"type": "tool_start", "name": block.name, "input": tool_input}
                    result_str = execute_tool(block.name, block.input, parsers, data_dirs, working_copies)
                    status, preview = _summarize_tool_result(result_str)
                    tool_trace.append({
                        "name": block.name,
                        "input": tool_input,
                        "status": status,
                        "preview": preview,
                    })
                    yield {"type": "tool_end", "name": block.name, "status": status, "preview": preview}
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })
            conversation_history.append({"role": "user", "content": tool_results})
            continue

        # Any non-tool_use stop reason ends the turn.
        final_text = "".join(answer_parts)
        if response.stop_reason != "end_turn" and not final_text:
            final_text = f"[Stopped: {response.stop_reason}]"
        yield {
            "type": "done",
            "text": final_text,
            "history": conversation_history,
            "tool_trace": tool_trace,
            "usage": usage,
        }
        return

    yield {
        "type": "done",
        "text": (
            "I reached the maximum number of tool calls without producing a final answer. "
            "Please try rephrasing your question."
        ),
        "history": conversation_history,
        "tool_trace": tool_trace,
        "usage": usage,
    }


def run_agent(
    user_message: str,
    conversation_history: list,
    parsers: dict[str, ISAParser],
    client: anthropic.Anthropic,
    data_dirs: dict[str, str] | None = None,
    working_copies: dict | None = None,
) -> tuple[str, list, list[dict], dict]:
    """
    Run one turn of the agentic loop and return the final result (non-streaming).

    This is a thin wrapper that drains run_agent_stream and returns its terminal
    payload, preserving the original 4-tuple contract:
        (final_text, updated_conversation_history, tool_trace, usage)
    Streaming UIs should call run_agent_stream directly and consume the live events.
    """
    done = None
    for event in run_agent_stream(
        user_message, conversation_history, parsers, client, data_dirs, working_copies
    ):
        if event.get("type") == "done":
            done = event
    if done is None:  # pragma: no cover - generator always yields a terminal event
        return "", conversation_history, [], _empty_usage()
    return done["text"], done["history"], done["tool_trace"], done["usage"]
