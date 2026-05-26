# ISA-PHM Conversational Interface

A locally-run Streamlit application that lets a user explore ISA-PHM datasets
through natural-language conversation. A Claude-based agent uses function
calling to invoke methods on the ISA-PHM Data Engineering Wrapper and returns
answers grounded in the source ISA-JSON file and its CSV measurements.

---

## What this is

ISA-PHM is a metadata standard for Prognostics and Health Management
testbench experiments, built on top of the ISA (Investigation, Study, Assay)
framework originally developed for the life sciences. The ISA-PHM Wizard
produces compliant ISA-JSON files; the ISA-PHM Data Engineering Wrapper
parses them and exposes Python methods for inspection and signal processing.
The wrapper is powerful but assumes the user already knows which method to
call. This application closes that gap with a chat interface: the user asks
questions in natural language, an LLM decides which wrapper methods to call,
and the wrapper returns the facts.

The application is built **on top of** the wrapper, not around it. All
parsing, structural validation, CSV loading, feature computation, and
plotting is done by the wrapper. The application contributes the
conversational layer: a system prompt, a tool-calling loop, and a thin
catalogue of tools that route to wrapper methods. The wrapper does the
data work; this application decides which wrapper method to call for a
given user question.

---

## Architecture

The system has four layers. Every user query travels through all of them.

```
+--------------------------------------------------------------+
|  User Interface  —  Streamlit app (app.py)                   |
|  Sidebar (file uploader, data-dir field), chat tab,          |
|  quality-report tab, compare-datasets tab, agent-trace panel |
+--------------------------------------------------------------+
|  Agent Layer  —  Claude API with tool use (agent.py)         |
|  System prompt (≈160 lines), tool-calling loop, trace logger |
+--------------------------------------------------------------+
|  Tool Layer  —  21 tools (tools.py)                          |
|  Metadata tools route through ISAParser                      |
|  Signal tools route through ISAWrapper / AssayProxy / Plotter|
+--------------------------------------------------------------+
|  Data Layer                                                  |
|  ISAParser (in-house, isa_parser.py) — hierarchy queries     |
|  ISAWrapper (vendored, isa_phm/) — CSV + lifecycle + plots   |
+--------------------------------------------------------------+
```

Two data paths run in parallel, and both rely on the wrapper:

- **Metadata path** — always active. Every uploaded ISA-JSON is first run
  through the wrapper's parser and preprocessor, which absorb structural
  defects (missing optional fields, mislocated nodes, ID collisions) and
  return a clean dictionary. The in-house `ISAParser` then walks that
  dictionary to answer metadata questions. Metadata tools return JSON.
- **Signal path** — activated when the user supplies a local data directory.
  At that point the wrapper becomes the execution layer for every
  signal-level operation: `AssayProxy.load_dataframe_with_meta()` for
  raw CSV loading, `lifecycle_features()` for RMS / kurtosis /
  peak-to-peak, and `plot_*()` through `ISAPlotter` for Bokeh figures.
  The application registers each of these as a tool the agent can call.

Keeping the two paths separate is deliberate: metadata operations work
without any file-system access, so the app is usable on a fresh machine for
quick inspection, and signal operations only activate when the user
explicitly points at the CSV folder.

---

## Setup and run

Requirements:

- Python 3.13
- An Anthropic API key with access to the Claude Messages API
- ISA-JSON files and the matching CSV measurements

```bash
cd Prototype
python3.13 -m pip install -r requirements.txt
streamlit run app.py
```

In the Streamlit UI:

1. Paste your Anthropic API key in the sidebar.
2. Upload one or more ISA-JSON files.
3. For each uploaded file, fill in the local path to the directory holding
   that dataset's CSV measurements. Wait for the status pill under the
   field to turn green (`data dir: connected`).
4. Ask questions in the Chat tab.

If no data directory is set, signal-level tools degrade gracefully and
return a `data_not_available` status; metadata-only questions still work.

---

## How the agent works

The agent lives in `agent.py`. It exposes a single function, `run_agent`,
which takes the user message, the conversation history, the dictionary of
loaded parsers, and the optional dictionary of data directories. It returns
the assistant response together with a tool-call trace.

### The tool-calling loop

```
   user message
        |
        v
+---------------+
|  Claude API   |  <-- system prompt + tool definitions + history
+---------------+
        |
   stop_reason == "tool_use"?
        | yes                              | no (end_turn)
        v                                  v
   execute tool                       return text
   append result to history
   (repeat, max 15 rounds)
```

The loop terminates as soon as the model produces a text-only response. In
practice it finishes within four rounds for typical questions. The hard
cap of fifteen rounds is a safety net against runaway sequences.

Every tool call is logged with its name, its input, its outcome status,
and a short preview of its output. The trace is returned to the user
interface and displayed in an expander under each assistant message, so a
reader can inspect exactly which tools the agent called, in what order,
with which arguments, and what each returned.

### The system prompt

The system prompt (`SYSTEM_PROMPT_TEMPLATE` in `agent.py`) is approximately
160 lines and contains seven sections:

1. **ISA-PHM vocabulary** — Investigation, Study, Assay, Run, Study Factor,
   Operating Condition. The model is required to use these terms and is
   instructed not to fall back on generic spreadsheet language (row,
   column, file).
2. **PHM ambition levels** — Detection, Diagnostics, Health assessment,
   Prognosis. Each ambition level has a different data requirement; the
   model uses this table to judge whether a dataset supports the question
   being asked.
3. **Quality gates** — five checks that run before any analysis (see
   below).
4. **Auto-selection defaults** — when the user does not specify a study,
   assay, or run, the system chooses one according to documented rules
   (study with the most runs, first vibration assay, last run).
   Auto-selection is reported to the user in a structured prefix before
   the actual answer.
5. **Sensor-to-visualisation compatibility** — which plot types are
   appropriate for which sensor categories. The model is required to call
   `get_sensor_compatibility` before generating any frequency-domain plot.
6. **Structured response format** — a documented template for vague or
   open-ended prompts. The model fills the template rather than producing
   free-form prose for these prompts.
7. **FUTURE CAPABILITY rule** — a list of capabilities not implemented in
   the application. The model is required to answer such requests with a
   `[FUTURE CAPABILITY]` prefix and a metadata-only workaround.

### Quality gates

Five gates are run by the `validate_dataset` tool. The agent is required
to call this tool before any non-trivial analysis.

| Gate | Checks | On failure |
|------|--------|-----------|
| 1. Structural | Hierarchy valid, required fields present, no broken cross-references | **STOP**. Return a concrete fix plan. No plots, no features. |
| 2. File linkage | Linked CSV files exist on disk and are readable | **STOP** (only when a data directory is set) |
| 3. PHM semantics | Fault type, operating conditions, units are detectable | **WARN**. Continue, but include a caveat in the response. |
| 4. Prognostic readiness | Run ordering, trajectories, RUL labels | Only checked for prognosis prompts |
| 5. Merge readiness | Factor crosswalk, sensor and unit alignment | Only checked when merging or comparing datasets |

Gates 1 and 2 are blocking. Gate 3 produces a warning that the agent must
surface in the response. Gates 4 and 5 are conditional — they run only
when the prompt is in their scope.

---

## Tool catalogue

The agent has 21 tools, defined in `tools.py`. The tools are presented to
the model as a flat list with short, unambiguous descriptions.

| Category | Tool | Purpose |
|----------|------|---------|
| Onboarding | `ai_context` | Compact natural-language summary of a dataset. |
| Metadata | `get_investigation_overview` | Top-level investigation properties. |
| Metadata | `list_studies` | List of studies with id and title. |
| Metadata | `get_study_details` | Full study record. |
| Metadata | `get_assay_details` | Full assay record. |
| Metadata | `list_data_files` | CSV inventory for an assay. |
| Search | `search_metadata` | Free-text search across the parsed structure. |
| Quality | `validate_dataset` | Run the five-gate quality check. |
| Run/sensor | `get_experiment_matrix` | Study factors × studies as a table. |
| Run/sensor | `get_run_inventory` | Inventory of runs per study and assay. |
| Run/sensor | `get_label_coverage` | Coverage of factor labels across runs. |
| Run/sensor | `get_sensor_availability` | Sensor list with metadata per assay. |
| Run/sensor | `get_sensor_compatibility` | Plot-type compatibility for a sensor category. |
| Comparison | `classify_phm_objective` | PHM ambition level for a dataset. |
| Comparison | `compare_studies` | Differences and overlaps between two studies. |
| Comparison | `get_protocol_details` | Protocol record with parameters. |
| Comparison | `get_replication_gaps` | Missing replications across factor cells. |
| Comparison | `compare_datasets_metadata` | Side-by-side metadata comparison. |
| Signal | `load_run_csv` | Load a CSV measurement through the wrapper. |
| Signal | `lifecycle_features` | RMS / kurtosis / peak-to-peak features per run. |
| Signal | `make_plot` | Render a Bokeh figure for an assay. |

Two principles drive the tool-set design:

1. **Keep the catalogue small and semantically differentiated.** Wrong-tool
   selection rises sharply with catalogue size and falls with descriptive
   clarity. Each tool here covers a non-overlapping question category.
2. **Degrade gracefully on signal tools when no data directory is set.**
   Rather than raising, signal tools return a structured response with
   status `data_not_available` and an explanation. The agent surfaces this
   message to the user without rewriting it.

---

## Data layer

### ISAParser (in-house)

`isa_parser.py` defines `ISAParser`, a lightweight read-only wrapper
around a parsed ISA-JSON dictionary. The parser exposes 22 query methods
that walk the dictionary and return JSON-serialisable objects. The
methods are paired 1:1 with the metadata tools in `tools.py`.

`ISAParser` also exposes `attach_wrapper(data_root)`, which initialises an
`ISAWrapper` against a user-specified CSV directory. Once attached, the
parser owns the wrapper instance and exposes two lookup helpers used by
the signal tools: `get_assay_filename(study_title, sensor_alias)` and
`get_run_id_for_path(study_title, sensor_alias, relative_path)`.

The wrapper's preprocessor runs over every uploaded ISA-JSON before the
in-house parser sees it. The preprocessor absorbs structural
inconsistencies in real-world files (missing optional fields, mislocated
nodes, ID collisions), which means the in-house parser can assume a
clean dictionary.

### ISAWrapper (vendored)

The wrapper is the load-bearing data layer of the application. Anything
that touches an ISA-JSON file or a CSV on disk passes through it. The
full ISA-PHM Data Engineering Wrapper is bundled into `isa_phm/`, which
makes the application self-contained and removes the need for a
separate pip install. Modules of interest:

- `wrapper.py` — the `ISAWrapper` class, the entry point to the data path.
- `parser.py` and `preprocessor.py` — the ISA-JSON parser and the
  structural-defect repair pipeline.
- `integrator.py` — `DataIntegrator`, resolves relative file paths and
  loads CSVs with their full metadata.
- `proxy.py` — `AssayProxy`, the per-assay façade exposed by the wrapper.
  Provides `load_dataframe_with_meta()`, `lifecycle_features()`, `plot_*()`.
- `plotter.py` — `ISAPlotter`, the Bokeh-based visualisation module.
- `tools.py` — `ToolRegistry`, the wrapper's own tool registration system
  (not used here; the application registers tools at the top level
  through its own `tools.py`).

---

## User interface

The UI is implemented in `app.py`. Structure:

- **Sidebar**
  - API-key input
  - Multi-file uploader for ISA-JSON files
  - Per-dataset text field for the local CSV directory, plus a
    colour-coded status pill (`connected` / `error` / `not set`)
  - Per-dataset clear-conversation button

- **Main area, three tabs**
  - **Chat** — the primary surface. Renders the conversation, the
    agent-trace expander under each assistant message, and the Bokeh
    figure registry.
  - **Quality Report** — pre-formatted view of the validation result for
    each loaded dataset (equivalent to calling `validate_dataset` once
    per dataset).
  - **Compare Datasets** — side-by-side metadata comparison for two or
    more loaded datasets (calls `compare_datasets_metadata`).

The UI does not implement any analysis itself; every action is the
result of a tool call. This keeps the UI thin and makes the architecture
testable by replaying conversation logs.

### The agent-trace expander

Every assistant message is followed by a collapsible panel listing:

- Tool name
- Tool input (the arguments the model passed)
- Status (`ok`, `error`, `data_not_available`, etc.)
- Short preview of the result (first ~400 characters of the JSON)

---

## Plot rendering

All plots are Bokeh figures, produced by `ISAPlotter` and rendered in
Streamlit via `bokeh.embed.file_html` + `st.components.v1.html`. Two
notes:

- `st.bokeh_chart` was removed in recent Streamlit versions; the
  `streamlit-bokeh` shim is incompatible with Streamlit 1.56. The
  `file_html` + `components.html` route is the supported way to embed
  Bokeh in current Streamlit.
- The figure registry (`tools.FIGURE_REGISTRY`) holds produced figures
  by ID. `make_plot` returns the ID; the UI looks the figure up and
  renders it. This decouples plot production from rendering.

Supported plot kinds: `time`, `fft`, `psd`, `spectrogram`, `waterfall`,
`distribution`, `lifecycle_trend`.

---

## Configuration via environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | — | Required for the agent to call the Claude Messages API. Can also be entered in the sidebar. |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | The Claude model identifier. Override to `claude-opus-4-7` for higher-quality runs or `claude-haiku-4-7` for low-cost batch runs. |

Constants in `agent.py`:

- `MAX_TOKENS = 4096` — per-response budget.
- `MAX_TOOL_ROUNDS = 15` — safety cap on tool-calling iterations.

---

## Smoke test

A scripted smoke test that exercises the metadata path without an LLM
call is provided in `smoke_test.py`. Adjust the `JSON_PATH` constant at
the top of the file to point at a local ISA-PHM JSON before running.

Manual end-to-end smoke test:

```
ISA-JSON file:   any ISA-PHM JSON
Data directory:  the matching CSV folder on disk
Prompt:          "Plot the time series of the first sensor"
Expected:        Interactive Bokeh chart
```

---

## Troubleshooting

**The data-dir status pill stays grey.**
The path you entered is empty or not a directory. The signal tools will
return `data_not_available` until the pill turns green.

**The status pill turns red with "wrapper-init failed".**
The path exists but the wrapper could not bind to it. The most common
cause is that the directory does not contain the CSV files that the
ISA-JSON references. Check the error message under the pill.

**The agent says "I cannot find the file" for a CSV that is on disk.**
The relative paths inside the ISA-JSON are interpreted relative to the
data directory you specified. If the JSON expects
`data/run_01/sensor_a.csv`, the data directory must contain
`data/run_01/`, not the file directly.

**The agent never calls `validate_dataset`.**
This indicates a system-prompt drift. Confirm that the system prompt is
loaded by looking at the agent trace on the first assistant message;
the first entry should be `validate_dataset`.

**Bokeh charts render as empty boxes.**
Most common cause: the wrapper produced a figure with no data points
(because the requested run had no measurements in the requested range).
Check the agent trace for the `make_plot` call and inspect the input
arguments.
