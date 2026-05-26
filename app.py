"""
ISA-PHM AI Interface — Streamlit App
-------------------------------------
Run with:  streamlit run app.py
"""

import os
import anthropic
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from isa_parser import load_from_bytes
from agent import run_agent
import tools as tools_module

load_dotenv()

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="ISA-PHM AI Interface",
    layout="wide",
)

# ------------------------------------------------------------------
# Global styling
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Base typography */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter",
                     sans-serif;
    }

    /* Main container padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Headings */
    h1, h2, h3, h4 {
        letter-spacing: -0.01em;
        font-weight: 600;
    }

    /* Sidebar nav — Claude-style stacked button selector */
    section[data-testid="stSidebar"] [class*="st-key-navbtn_"] {
        margin-bottom: 4px;
    }
    section[data-testid="stSidebar"] [class*="st-key-navbtn_"] button {
        height: 44px;
        border-radius: 10px;
        font-weight: 500;
        font-size: 14px;
        text-align: left;
        justify-content: flex-start;
        padding: 0 14px;
        transition: background 0.15s ease, border-color 0.15s ease,
                    box-shadow 0.15s ease;
    }
    section[data-testid="stSidebar"] [class*="st-key-navbtn_"] button[kind="secondary"] {
        background: transparent;
        border: 1px solid transparent;
        color: #2a313c;
        box-shadow: none;
    }
    section[data-testid="stSidebar"] [class*="st-key-navbtn_"] button[kind="secondary"]:hover {
        background: #eceef1;
        border-color: transparent;
        color: #111827;
    }
    section[data-testid="stSidebar"] [class*="st-key-navbtn_"] button[kind="primary"] {
        background: #ffffff;
        color: #111827;
        border: 1px solid #e0e3e7;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06),
                    0 1px 3px rgba(15, 23, 42, 0.04);
    }
    section[data-testid="stSidebar"] [class*="st-key-navbtn_"] button[kind="primary"]:hover {
        background: #ffffff;
        border-color: #d4d7dc;
        color: #111827;
    }
    section[data-testid="stSidebar"] [class*="st-key-navbtn_"] button[kind="primary"]:focus,
    section[data-testid="stSidebar"] [class*="st-key-navbtn_"] button[kind="primary"]:active {
        background: #ffffff;
        color: #111827;
    }

    /* Status pills */
    .pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 500;
        letter-spacing: 0.02em;
        line-height: 1.6;
    }
    .pill-pass { background: #e7f6ec; color: #1f7a3a; }
    .pill-warn { background: #fff4e0; color: #8a5a00; }
    .pill-fail { background: #fde7e9; color: #a3262f; }
    .pill-none { background: #eef0f3; color: #5b6470; }

    /* Sidebar polish */
    section[data-testid="stSidebar"] {
        background: #fafbfc;
        border-right: 1px solid #eceef1;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.15s ease;
    }

    /* Chat input rounding */
    [data-testid="stChatInput"] {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Session state initialisation
# ------------------------------------------------------------------
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "datasets" not in st.session_state:
    st.session_state.datasets = {}
if "data_dirs" not in st.session_state:
    st.session_state.data_dirs = {}
if "validation_cache" not in st.session_state:
    st.session_state.validation_cache = {}
if "client" not in st.session_state:
    st.session_state.client = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("ANTHROPIC_API_KEY", "")
if "show_api_key_panel" not in st.session_state:
    st.session_state.show_api_key_panel = not bool(st.session_state.api_key)
if "active_view" not in st.session_state:
    st.session_state.active_view = "Chat"


_PILL_CLASS = {
    "pass": "pill-pass",
    "warn": "pill-warn",
    "fail": "pill-fail",
}


def _status_pill(status: str | None) -> str:
    if not status:
        return '<span class="pill pill-none">unknown</span>'
    cls = _PILL_CLASS.get(status, "pill-none")
    return f'<span class="pill {cls}">{status.upper()}</span>'


def _refresh_validation(name: str) -> None:
    parser = st.session_state.datasets.get(name)
    if parser is None:
        return
    try:
        st.session_state.validation_cache[name] = parser.validate_dataset()
    except Exception as e:
        st.session_state.validation_cache[name] = {
            "status": "fail",
            "errors": [f"validate_dataset crashed: {e}"],
            "warnings": [],
            "info": [],
        }


def _render_agent_trace(trace: list[dict]) -> None:
    """Render a collapsible 'Agent trace' panel showing tool calls for this turn."""
    if not trace:
        return
    n = len(trace)
    with st.expander(f"Agent trace ({n} tool call{'s' if n != 1 else ''})", expanded=False):
        for i, step in enumerate(trace, start=1):
            name = step.get("name", "?")
            status = step.get("status", "ok")
            status_lower = str(status).lower()
            if "error" in status_lower or status_lower == "fail":
                badge = '<span class="pill pill-fail">error</span>'
            elif "warn" in status_lower or "not_available" in status_lower or "missing" in status_lower:
                badge = '<span class="pill pill-warn">warn</span>'
            else:
                badge = '<span class="pill pill-pass">ok</span>'
            st.markdown(
                f"**{i}. `{name}`** &nbsp; {badge}",
                unsafe_allow_html=True,
            )
            inp = step.get("input", {})
            if inp:
                st.caption("Input:")
                st.code(_short_json(inp), language="json")
            preview = step.get("preview", "")
            if preview:
                st.caption("Result preview:")
                st.code(preview, language="json")


def _short_json(value) -> str:
    import json as _json
    try:
        return _json.dumps(value, indent=2, default=str)
    except Exception:
        return str(value)


def _render_registered_figure(plot_id: str) -> None:
    """Render a figure from the tools.FIGURE_REGISTRY in the chat stream.

    Handles both Bokeh (wrapper-produced) and Matplotlib (legacy) figures.

    Bokeh rendering: st.bokeh_chart was removed in recent Streamlit versions
    and the streamlit-bokeh shim is unstable, so we serialise the Bokeh figure
    to standalone HTML and embed it via st.components.v1.html. This works on
    every Streamlit version that supports the components API.
    """
    entry = tools_module.FIGURE_REGISTRY.get(plot_id)
    if not entry:
        return
    fig = entry["figure"]
    backend = entry.get("backend", "bokeh")
    title = entry.get("title", plot_id)
    try:
        if backend == "bokeh":
            from bokeh.embed import file_html
            from bokeh.resources import CDN
            html = file_html(fig, CDN, title)
            # Heuristic height: tall enough for axis labels + toolbar.
            height = int(getattr(fig, "height", 0) or 450) + 60
            st.components.v1.html(html, height=height, scrolling=False)
        else:
            st.pyplot(fig)
    except Exception as exc:
        st.error(f"Could not render plot ({title}): {exc}")
    st.caption(f"_{title}_")


def _set_client_if_needed(api_key: str) -> None:
    if not api_key:
        st.session_state.client = None
        return
    if (
        st.session_state.client is None
        or getattr(st.session_state.client, "_api_key", None) != api_key
    ):
        st.session_state.client = anthropic.Anthropic(api_key=api_key)


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ISA-PHM AI Interface")
    st.markdown("---")

    # --- View selector ---
    _views = ["Chat", "Quality Report", "Compare Datasets"]
    for _view in _views:
        _is_active = st.session_state.active_view == _view
        if st.button(
            _view,
            key=f"navbtn_{_view}",
            use_container_width=True,
            type="primary" if _is_active else "secondary",
        ):
            st.session_state.active_view = _view
            st.rerun()

    st.markdown("---")

    # --- API key (collapsible) ---
    if st.session_state.show_api_key_panel:
        st.markdown("**Claude API Key**")
        api_key_input = st.text_input(
            "Anthropic API key",
            type="password",
            value=st.session_state.api_key,
            help="Your key is never stored. It is only kept in memory for this session.",
            label_visibility="collapsed",
        )

        col_save, col_hide = st.columns([3, 2])
        if col_save.button("Save key", use_container_width=True, type="primary"):
            st.session_state.api_key = api_key_input
            _set_client_if_needed(api_key_input)
            if api_key_input:
                st.session_state.show_api_key_panel = False
                st.rerun()
        if st.session_state.api_key:
            if col_hide.button("Hide", use_container_width=True):
                st.session_state.show_api_key_panel = False
                st.rerun()

        if not st.session_state.api_key:
            st.caption("Enter your API key to start chatting.")
    else:
        _set_client_if_needed(st.session_state.api_key)
        col_status, col_edit = st.columns([3, 2])
        col_status.markdown(
            '<span class="pill pill-pass">API key set</span>',
            unsafe_allow_html=True,
        )
        if col_edit.button("Edit", help="Change API key"):
            st.session_state.show_api_key_panel = True
            st.rerun()

    current_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    st.caption(f"Model: `{current_model}`")

    st.markdown("---")

    # --- Datasets ---
    st.markdown("**Datasets**")

    uploaded_file = st.file_uploader(
        "Upload an ISA-PHM JSON file",
        type=["json"],
        help="Upload one or more ISA-PHM JSON files. All loaded datasets are active.",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        if uploaded_file.name not in st.session_state.datasets:
            try:
                parser = load_from_bytes(uploaded_file.read(), source_hint=uploaded_file.name)
                st.session_state.datasets[uploaded_file.name] = parser
                st.session_state.data_dirs[uploaded_file.name] = ""
                _refresh_validation(uploaded_file.name)
            except Exception as e:
                st.error(f"Failed to load **{uploaded_file.name}**: {e}")

    for name in list(st.session_state.datasets.keys()):
        parser = st.session_state.datasets[name]
        validation = st.session_state.validation_cache.get(name)
        status = validation.get("status") if validation else None

        col_name, col_btn = st.columns([8, 1])
        col_name.markdown(
            f"{_status_pill(status)} &nbsp; **{name}**",
            unsafe_allow_html=True,
        )
        if col_btn.button("×", key=f"remove_{name}", help=f"Remove {name}"):
            del st.session_state.datasets[name]
            st.session_state.data_dirs.pop(name, None)
            st.session_state.validation_cache.pop(name, None)
            st.session_state.conversation_history = []
            st.session_state.chat_messages = []
            st.rerun()
            continue

        # Data directory input — enables wrapper-backed signal tools.
        current_dir = st.session_state.data_dirs.get(name, "")
        new_dir = st.text_input(
            "Data directory",
            value=current_dir,
            key=f"datadir_input_{name}",
            placeholder="/path/to/folder/with/CSVs",
            help=(
                "Absolute path to the folder that contains the CSV files "
                "referenced by this ISA-JSON. Required for signal-level "
                "tools (plots, lifecycle features, raw data preview)."
            ),
            label_visibility="collapsed",
        )
        if new_dir != current_dir:
            st.session_state.data_dirs[name] = new_dir
            ok, msg = parser.attach_wrapper(new_dir if new_dir else None)
            if new_dir and not ok:
                st.warning(f"Could not attach data directory: {msg}")
            st.rerun()

        # Wrapper status pill — tells the user whether signal tools are live.
        wstatus = getattr(parser, "wrapper_status", "not_attached")
        if wstatus == "attached":
            wpill = '<span class="pill pill-pass">data dir: connected</span>'
        elif wstatus in ("invalid_data_root", "init_failed", "wrapper_unavailable", "no_source"):
            wpill = '<span class="pill pill-fail">data dir: error</span>'
        else:
            wpill = '<span class="pill pill-none">data dir: not set</span>'
        st.markdown(wpill, unsafe_allow_html=True)
        if wstatus in ("invalid_data_root", "init_failed") and getattr(parser, "wrapper_error", ""):
            st.caption(f"⚠ {parser.wrapper_error}")

    st.markdown("---")

    # --- Export + Clear ---
    if st.session_state.chat_messages:
        from datetime import datetime
        lines = [
            "# ISA-PHM AI Interface — Conversation Export",
            f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Datasets:** {', '.join(st.session_state.datasets.keys()) or '—'}",
            "",
        ]
        for msg in st.session_state.chat_messages:
            role_label = "**You**" if msg["role"] == "user" else "**Assistant**"
            lines.append(f"### {role_label}")
            lines.append(msg["content"])
            lines.append("")

        export_md = "\n".join(lines)
        filename = f"isa_phm_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        st.download_button(
            label="Export conversation",
            data=export_md,
            file_name=filename,
            mime="text/markdown",
            use_container_width=True,
        )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.conversation_history = []
        st.session_state.chat_messages = []
        st.rerun()


# ------------------------------------------------------------------
# Main area — three tabs
# ------------------------------------------------------------------
st.title("ISA-PHM AI Interface")

# Chat input is declared at top-level so Streamlit pins it to the bottom
# of the viewport. We capture it now and consume it inside the Chat view.
_chat_ready = (
    bool(st.session_state.datasets)
    and st.session_state.client is not None
    and st.session_state.active_view == "Chat"
)
user_input = (
    st.chat_input("Ask a question about your dataset(s)…") if _chat_ready else None
)

_active = st.session_state.active_view


# ══════════════════════════════════════════════════════════════════
# VIEW 1 — Chat
# ══════════════════════════════════════════════════════════════════
if _active == "Chat":
    if not st.session_state.datasets:
        st.info("Upload one or more ISA-PHM JSON files in the sidebar to get started.")
    elif st.session_state.client is None:
        st.info("Enter your Claude API key in the sidebar to start chatting.")
    else:
        loaded_names = list(st.session_state.datasets.keys())
        if len(loaded_names) == 1:
            st.caption(f"Active dataset: **{loaded_names[0]}**")
        else:
            st.caption(f"Active datasets: {', '.join(f'**{n}**' for n in loaded_names)}")

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                for plot_id in msg.get("plot_ids", []):
                    _render_registered_figure(plot_id)
                _render_agent_trace(msg.get("tool_trace", []))

        if user_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        pre_plot_ids = set(tools_module.FIGURE_REGISTRY.keys())

                        answer, updated_history, tool_trace = run_agent(
                            user_message=user_input,
                            conversation_history=st.session_state.conversation_history,
                            parsers=st.session_state.datasets,
                            client=st.session_state.client,
                            data_dirs=st.session_state.data_dirs,
                        )
                        st.session_state.conversation_history = updated_history

                        new_plot_ids = [
                            pid for pid in tools_module.FIGURE_REGISTRY.keys()
                            if pid not in pre_plot_ids
                        ]

                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": answer,
                            "plot_ids": new_plot_ids,
                            "tool_trace": tool_trace,
                        })
                        st.markdown(answer)

                        for pid in new_plot_ids:
                            _render_registered_figure(pid)
                        _render_agent_trace(tool_trace)

                    except anthropic.AuthenticationError:
                        err = "Invalid API key. Please check the key entered in the sidebar."
                        st.error(err)
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": err}
                        )
                    except Exception as e:
                        err = f"An error occurred: {e}"
                        st.error(err)
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": err}
                        )

    # Prompt library — collapsed by default, opens via expander
    if st.session_state.datasets and not st.session_state.chat_messages:
        st.markdown("---")
        with st.expander("Show example prompts", expanded=False):
            first_dataset = list(st.session_state.datasets.keys())[0]
            multi = len(st.session_state.datasets) > 1

            sections = [
                ("L. Vague starters (let the agent decide)", [
                    f"Is this dataset any good for PHM? ({first_dataset})",
                    f"I don't know where to start. What would you do first with {first_dataset}?",
                    f"Make useful PHM graphs for {first_dataset} and tell me what stands out.",
                    f"Is {first_dataset} ready to train a PHM model on? What are the blockers?",
                ]),
                ("A. Investigation-level basics", [
                    f"Explain {first_dataset} at investigation level: experiment matrix, sensors, run structure, and which factors are fault vs operating-condition.",
                    f"I am new to {first_dataset}. What experiments were done, under which operating conditions, and what can I use for labels?",
                    f"Give me a PHM-ready dictionary for {first_dataset}: fault fields, operating conditions, degradation/RUL fields, sensor configuration.",
                ]),
                ("B. Study factors and labels", [
                    f"List all study factors in {first_dataset}. For each: name, category (fault / operating condition / other), values, whether it varies between studies or runs.",
                    f"What fault types are in {first_dataset} and how many studies / runs cover each?",
                    f"Can you extract labels from {first_dataset} automatically? What fields would you use?",
                ]),
                ("C. Quality audit", [
                    f"Run a full quality audit on {first_dataset}. Return per gate: pass/warn/fail | issues | fix suggestion.",
                    f"Check if {first_dataset} is ready for prognostics work. Return READY / NOT READY + per-criterion results.",
                    f"Score {first_dataset} (1-5) on metadata completeness, label quality, trajectory quality, and overall readiness.",
                ]),
                ("E. Runs and sensors", [
                    f"Walk me through what happens in a single run in {first_dataset}: what is being measured, when does it start and end, what changes between runs?",
                    f"Which assays / sensors are available for the first study in {first_dataset} and what does each one measure?",
                ]),
                ("F. Raw data, sensor specs, provenance", [
                    f"What is the sampling rate and sensor specification for each assay in {first_dataset}?",
                    f"What PHM level is {first_dataset} best suited for: detection, diagnostics, health assessment, or prognosis?",
                    f"Who created {first_dataset}? Return all contact, affiliation, and publication information available.",
                ]),
                ("G. Visualisations (signal-level — needs data dir)", [
                    f"Generate the standard PHM plot set for the first study in {first_dataset}: time waveform + FFT + PSD + spectrogram + lifecycle trend.",
                    f"I want to see how the degradation evolves across runs in {first_dataset}. Generate the most appropriate visualisation.",
                ]),
                ("J. Replication / protocol extraction", [
                    f"Give me everything I need to reproduce one study from {first_dataset} in a lab.",
                    f"What metadata is missing from {first_dataset} that would prevent me from fully reproducing the experiment?",
                ]),
            ]

            if multi:
                names = list(st.session_state.datasets.keys())
                sections.append((
                    "D. Compare datasets",
                    [
                        f"Compare {names[0]} and {names[1]} side by side: experiment type, factors, sensors, n_studies, n_runs.",
                        f"If I had to pick just one of {names[0]} or {names[1]} for my thesis, which would you recommend and why?",
                        f"Are {names[0]} and {names[1]} mergeable for one PHM benchmark? Run validate_dataset with check_merge=True.",
                    ],
                ))

            cols = st.columns(2)
            for i, (header, prompts) in enumerate(sections):
                with cols[i % 2]:
                    st.markdown(f"**{header}**")
                    for p in prompts:
                        st.markdown(f"- *{p}*")


# ══════════════════════════════════════════════════════════════════
# VIEW 2 — Quality Report
# ══════════════════════════════════════════════════════════════════
elif _active == "Quality Report":
    if not st.session_state.datasets:
        st.info("Upload a dataset in the sidebar to see its quality report.")
    else:
        _GATE_LABELS = {
            "1_structural": "Gate 1 — Structural",
            "2_semantic": "Gate 2 — PHM semantic",
            "3_ambition": "Gate 3 — PHM ambition",
            "4_merge_readiness": "Gate 4 — Merge readiness",
        }

        for name, parser in st.session_state.datasets.items():
            st.subheader(name)

            if name not in st.session_state.validation_cache:
                _refresh_validation(name)
            validation = st.session_state.validation_cache.get(name, {})

            status = validation.get("status", "warn")
            st.markdown(
                f"Overall status: &nbsp; {_status_pill(status)}",
                unsafe_allow_html=True,
            )

            obj = validation.get("phm_objective", {})
            if obj:
                col_obj, col_conf = st.columns(2)
                col_obj.metric("PHM Objective", obj.get("objective", "—").title())
                col_conf.metric("Confidence", obj.get("confidence", "—").title())
                st.caption(f"Reasoning: {obj.get('reasoning', '—')}")
                if obj.get("inconsistency_warning"):
                    st.warning(f"Inconsistency detected: {obj['inconsistency_warning']}")

            st.markdown("**Quality Gates**")

            gates = validation.get("gate_summary", {})
            gate_rows = []
            for key, label in _GATE_LABELS.items():
                result = gates.get(key, "—")
                gate_rows.append({"Gate": label, "Result": result.upper()})

            gate_df = pd.DataFrame(gate_rows)
            st.dataframe(gate_df, use_container_width=True, hide_index=True)

            errors = validation.get("errors", [])
            warnings_list = validation.get("warnings", [])
            info_list = validation.get("info", [])

            if errors:
                with st.expander(f"{len(errors)} error(s) — expand to see fix suggestions", expanded=True):
                    for e in errors:
                        st.error(e)
            else:
                st.success("No errors found.")

            if warnings_list:
                with st.expander(f"{len(warnings_list)} warning(s)", expanded=True):
                    for w in warnings_list:
                        st.warning(w)

            if info_list:
                with st.expander(f"{len(info_list)} info message(s)"):
                    for i in info_list:
                        st.info(i)

            if len(st.session_state.datasets) > 1:
                st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# VIEW 3 — Compare Datasets
# ══════════════════════════════════════════════════════════════════
elif _active == "Compare Datasets":
    names = list(st.session_state.datasets.keys())

    if len(names) < 2:
        st.info("Load at least **two** ISA-PHM datasets in the sidebar to use this view.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            dataset_a = st.selectbox("Dataset A", names, index=0, key="cmp_a")
        with col_b:
            default_b = 1 if len(names) > 1 else 0
            dataset_b = st.selectbox("Dataset B", names, index=default_b, key="cmp_b")

        if dataset_a == dataset_b:
            st.warning("Select two different datasets to compare.")
        else:
            pa = st.session_state.datasets[dataset_a]
            pb = st.session_state.datasets[dataset_b]

            st.markdown("### Side-by-side metadata comparison")
            cmp_result = tools_module._compare_datasets_metadata(pa, pb, dataset_a, dataset_b)

            verdict = cmp_result.get("overall_fusion_verdict", "—")
            if verdict == "FEASIBLE":
                st.success(f"Fusion verdict: **{verdict}**")
            elif verdict == "NOT FEASIBLE":
                st.error(f"Fusion verdict: **{verdict}**")
            else:
                st.warning(f"Fusion verdict: **{verdict}**")

            rows = cmp_result.get("rows", [])
            if rows:
                _FLAG_COLORS = {
                    "COMPATIBLE": "#d4edda",
                    "NEEDS_REVIEW": "#fff3cd",
                    "INCOMPATIBLE": "#f8d7da",
                }

                def _style_flag(val):
                    color = _FLAG_COLORS.get(val, "")
                    return f"background-color: {color}" if color else ""

                cmp_df = pd.DataFrame(rows)
                styled = cmp_df.style.map(_style_flag, subset=["flag"])
                st.dataframe(styled, use_container_width=True, hide_index=True)

            st.caption(cmp_result.get("future_capability_note", ""))

            st.markdown("### Merge readiness check (Gate 5)")
            with st.spinner("Running merge readiness check…"):
                try:
                    merge_val = pa.validate_dataset(
                        check_merge=True,
                        merge_target_parser=pb,
                    )
                    merge_details = merge_val.get("merge_details") or {}
                    gate5 = merge_val.get("gate_summary", {}).get("5_merge_readiness", "—")

                    st.markdown(
                        f"**Gate 5 result:** &nbsp; {_status_pill(gate5 if gate5 != '—' else None)}",
                        unsafe_allow_html=True,
                    )

                    if merge_details:
                        exp_match = merge_details.get("experiment_type_match", False)
                        shared_factors = merge_details.get("shared_factor_names", [])
                        shared_mtypes = merge_details.get("shared_measurement_types", [])
                        overlap_pct = merge_details.get("factor_overlap_pct", 0)

                        col1, col2, col3 = st.columns(3)
                        col1.metric(
                            "Experiment type match",
                            "Yes" if exp_match else "No",
                            delta=None,
                        )
                        col2.metric("Shared factor names", len(shared_factors))
                        col3.metric("Factor overlap", f"{overlap_pct}%")

                        if shared_factors:
                            st.markdown(f"**Shared factors:** {', '.join(shared_factors)}")
                        if shared_mtypes:
                            st.markdown(f"**Shared sensor types:** {', '.join(shared_mtypes)}")

                        only_a = merge_details.get("this_only_factors", [])
                        only_b = merge_details.get("target_only_factors", [])
                        if only_a or only_b:
                            diff_col_a, diff_col_b = st.columns(2)
                            with diff_col_a:
                                st.markdown(f"**Only in {dataset_a}:**")
                                st.write(only_a if only_a else "—")
                            with diff_col_b:
                                st.markdown(f"**Only in {dataset_b}:**")
                                st.write(only_b if only_b else "—")

                    merge_warnings = [
                        w for w in merge_val.get("warnings", [])
                        if "merge" in w.lower() or "gate" in w.lower()
                    ]
                    if merge_warnings:
                        with st.expander("Merge gate warnings"):
                            for w in merge_warnings:
                                st.warning(w)

                except Exception as e:
                    st.error(f"Merge check failed: {e}")
