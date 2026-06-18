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

from isa_parser import load_from_bytes, load_from_file, scan_project_folder
from agent import run_agent, run_agent_stream
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
if "working_copies" not in st.session_state:
    # Session-scoped cleaned-signal store. Keyed by
    # "dataset::study::sensor::run_id"; written by process_signal, read by
    # load_run_csv / make_plot. Threaded into run_agent and mutated in place.
    st.session_state.working_copies = {}
if "validation_cache" not in st.session_state:
    st.session_state.validation_cache = {}
if "client" not in st.session_state:
    st.session_state.client = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "token_usage_total" not in st.session_state:
    st.session_state.token_usage_total = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "turns": 0,
    }
if "project_folder_input" not in st.session_state:
    st.session_state.project_folder_input = ""
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


def _fmt_tokens(n: int) -> str:
    """Human-friendly token count: 1234 -> '1,234', 12345 -> '12.3k'."""
    n = int(n or 0)
    if n < 10_000:
        return f"{n:,}"
    return f"{n / 1000:.1f}k"


def _render_token_usage(usage: dict | None) -> None:
    """Render a compact token-usage line for one assistant turn.

    `usage` is the dict returned by run_agent:
        {"input_tokens", "output_tokens", "total_tokens",
         "rounds", "per_round": [{"input", "output"}, ...]}
    """
    if not usage or not usage.get("total_tokens"):
        return
    in_t = usage.get("input_tokens", 0)
    out_t = usage.get("output_tokens", 0)
    total = usage.get("total_tokens", 0)
    rounds = usage.get("rounds", 0)
    st.caption(
        f"🪙 {_fmt_tokens(total)} tokens this turn "
        f"· {_fmt_tokens(in_t)} in / {_fmt_tokens(out_t)} out "
        f"· {rounds} model call{'s' if rounds != 1 else ''}"
    )
    per_round = usage.get("per_round", [])
    if len(per_round) > 1:
        with st.expander("Token breakdown per model call", expanded=False):
            rows = [
                {
                    "Call": i,
                    "Input": r.get("input", 0),
                    "Output": r.get("output", 0),
                    "Total": r.get("input", 0) + r.get("output", 0),
                }
                for i, r in enumerate(per_round, start=1)
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _reset_token_usage() -> None:
    st.session_state.token_usage_total = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "turns": 0,
    }


def _working_copies_for(dataset_name: str) -> dict:
    """Return the working-copy entries belonging to one dataset."""
    return {
        k: v for k, v in st.session_state.working_copies.items()
        if v.get("dataset") == dataset_name
    }


def _drop_working_copies(dataset_name: str) -> None:
    """Remove all cached cleaned data for one dataset (reset-to-raw)."""
    for k in list(st.session_state.working_copies.keys()):
        if st.session_state.working_copies[k].get("dataset") == dataset_name:
            del st.session_state.working_copies[k]


def _summarize_transforms(transforms: list) -> str:
    """Compact one run's transform list, e.g. 'fix_outliers(iqr); fill_missing(interpolate)'."""
    parts = []
    for t in transforms or []:
        op = t.get("op")
        if op == "fix_outliers":
            parts.append(f"fix_outliers({t.get('method', 'iqr')}/{t.get('strategy', 'clip')})")
        elif op == "fill_missing":
            parts.append(f"fill_missing({t.get('strategy', 'interpolate')})")
        elif op:
            parts.append(op)
    return "; ".join(parts)


def _pick_folder_dialog() -> str | None:
    """Open a native folder picker in a SEPARATE process and return the chosen path.

    Why a subprocess: tkinter must run on the process's main thread, but Streamlit
    runs this script on a worker thread. Calling tkinter directly here crashes the
    whole Streamlit server on macOS ("Connection error"), and the crash happens
    below Python so a try/except can't catch it. Running the dialog in its own
    child process isolates the GUI — if anything goes wrong, only the child dies
    and Streamlit keeps running. Returns the path, or None if cancelled/unavailable.
    """
    import subprocess
    import sys

    script = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "r = tk.Tk(); r.withdraw()\n"
        "try:\n"
        "    r.attributes('-topmost', True); r.lift(); r.focus_force()\n"
        "except Exception:\n"
        "    pass\n"
        "p = filedialog.askdirectory(title='Select project folder')\n"
        "r.destroy()\n"
        "print(p or '')\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except Exception:
        return None
    lines = [ln for ln in (result.stdout or "").splitlines() if ln.strip()]
    return lines[-1].strip() if lines else None


def _load_project_folder(folder: str) -> tuple[bool, str]:
    """Scan a project folder for an ISA-PHM JSON, load it, and auto-connect the
    detected data directory. Returns (success, message)."""
    found = scan_project_folder(folder)
    if not found:
        return False, (
            "No ISA-PHM JSON found in that folder (only non-ISA / wizard files?). "
            "Use the upload + data-directory fields below instead."
        )
    json_path = found["json_path"]
    data_root = found["data_root"]
    name = os.path.basename(json_path)
    if name in st.session_state.datasets:
        return False, f"'{name}' is already loaded."
    try:
        parser = load_from_file(json_path)
    except Exception as e:
        return False, f"Found {name} but failed to load it: {e}"

    ok, msg = parser.attach_wrapper(data_root)
    st.session_state.datasets[name] = parser
    st.session_state.data_dirs[name] = data_root if ok else ""
    _refresh_validation(name)
    if ok:
        return True, f"Loaded **{name}** · data dir connected:\n`{data_root}`"
    return True, (
        f"Loaded **{name}**, but could not auto-connect the data dir ({msg}). "
        f"Set it manually below."
    )


def _short_json(value) -> str:
    import json as _json
    try:
        return _json.dumps(value, indent=2, default=str)
    except Exception:
        return str(value)


def _fit_bokeh_to_column(fig) -> int:
    """Make a Bokeh figure/layout fit the chat column and return the iframe height.

    Two problems this solves:
    * Width: every wrapper plot is built at a fixed 1150px width, which is wider
      than the chat column, so the right side was clipped. We switch every node
      in the layout tree (the top-level column, any intermediate gridplot/row,
      and each plot panel) to ``stretch_width`` so it scales to the column. It is
      not enough to set it on the plot panels alone: a stretch_width plot left
      inside a fixed-size gridplot collapses its frame to zero width in Bokeh
      3.x, so the axes and toolbar still render but the data canvas comes out
      blank (this is what broke the outlier-comparison comparison panels).
    * Height: some tools (e.g. outlier comparison) return a *layout* of stacked
      panels. A layout has no usable ``.height``, so the old fixed heuristic only
      reserved room for one panel and cut off the rest. We sum the heights of the
      actual panels instead, so the whole figure is visible.

    Must be called before ``file_html`` so the serialised HTML reflects the
    responsive sizing.
    """
    try:
        from bokeh.models import LayoutDOM, Plot
        refs = list(fig.references())
    except Exception:
        refs = []
    panels = [m for m in refs if isinstance(m, Plot)]
    if not panels:
        panels = [fig]

    # Scale to the column width instead of overflowing at the fixed 1150px. Every
    # layout node must become stretch_width together — setting it on the panels
    # but leaving an intervening gridplot fixed collapses the panel frames, so the
    # waveform disappears while the axes remain. Plot is itself a LayoutDOM, so
    # this loop also covers the panels.
    for node in refs:
        if isinstance(node, LayoutDOM):
            try:
                node.sizing_mode = "stretch_width"
            except Exception:
                pass
    try:
        fig.sizing_mode = "stretch_width"  # safety net for the top-level layout
    except Exception:
        pass

    # Reserve room for every panel plus its chrome (toolbar/title/axis labels),
    # with a small base margin for any header Div / inter-panel spacing.
    per_panel_chrome = 65
    base_margin = 20
    return base_margin + sum(
        int(getattr(panel, "height", 0) or 400) + per_panel_chrome for panel in panels
    )


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
            # Fit to the column width and size the iframe to all panels before
            # serialising, then keep scrolling enabled as a safety net so a
            # tall layout can never be clipped.
            height = _fit_bokeh_to_column(fig)
            html = file_html(fig, CDN, title)
            st.components.v1.html(html, height=height, scrolling=True)
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

    # --- Session token usage ---
    _tut = st.session_state.token_usage_total
    if _tut["turns"]:
        st.caption(
            f"🪙 Session: **{_fmt_tokens(_tut['total_tokens'])}** tokens "
            f"over {_tut['turns']} turn{'s' if _tut['turns'] != 1 else ''} "
            f"({_fmt_tokens(_tut['input_tokens'])} in / "
            f"{_fmt_tokens(_tut['output_tokens'])} out)"
        )

    st.markdown("---")

    # --- Datasets ---
    st.markdown("**Datasets**")

    # One-step project folder: scan for the ISA-JSON and auto-set its data dir.
    # Apply a folder chosen via the native dialog on the previous run (must run
    # before the text_input is instantiated, or Streamlit rejects the write).
    if "_folder_pick" in st.session_state:
        st.session_state.project_folder_input = st.session_state.pop("_folder_pick")

    st.caption("Project folder — auto-detect the ISA-JSON and its data:")
    col_path, col_browse = st.columns([3, 1])
    folder_val = col_path.text_input(
        "Project folder",
        key="project_folder_input",
        placeholder="/path/to/project (ISA-JSON + CSVs)",
        label_visibility="collapsed",
    )
    if col_browse.button("Browse", help="Browse for a folder (native dialog)", use_container_width=True):
        picked = _pick_folder_dialog()
        if picked:
            st.session_state["_folder_pick"] = picked
            st.rerun()
        else:
            st.warning("Native folder picker unavailable here — paste the path instead.")
    if st.button("Load from folder", use_container_width=True, key="load_from_folder"):
        target = (folder_val or "").strip()
        if not target:
            st.warning("Enter or browse to a project folder first.")
        else:
            ok, msg = _load_project_folder(target)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.warning(msg)

    st.caption("or upload a JSON (set its data directory below):")
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
            _drop_working_copies(name)
            st.session_state.conversation_history = []
            st.session_state.chat_messages = []
            _reset_token_usage()
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

        # Working-copy chip — shows how many runs have cached cleaned data.
        wc_runs = _working_copies_for(name)
        if wc_runs:
            col_chip, col_reset = st.columns([7, 3])
            col_chip.markdown(
                f'<span class="pill pill-pass">working data: '
                f'{len(wc_runs)} run(s) cleaned</span>',
                unsafe_allow_html=True,
            )
            if col_reset.button(
                "Reset to raw", key=f"wc_reset_{name}",
                help="Discard cleaned working copies for this dataset; "
                     "tools will read the raw files again.",
            ):
                _drop_working_copies(name)
                st.rerun()
            transform_summaries = sorted({
                _summarize_transforms(wc["transforms"]) for wc in wc_runs.values()
            })
            st.caption("Applied: " + "; ".join(t for t in transform_summaries if t))

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
        st.session_state.working_copies = {}
        _reset_token_usage()
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
                _render_token_usage(msg.get("usage"))

        if user_input:
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                try:
                    pre_plot_ids = set(tools_module.FIGURE_REGISTRY.keys())

                    # Live streaming (point 4): show the model's thinking and its
                    # tool calls as they happen, and stream the answer token-by-token.
                    # run_agent_stream yields events; the terminal "done" event carries
                    # the final answer/history/trace/usage (same payload as run_agent).
                    status = st.status("Thinking…", expanded=True)
                    thinking_ph = status.empty()
                    answer_ph = st.empty()
                    thinking_parts: list[str] = []
                    answer_parts: list[str] = []
                    done_event: dict | None = None

                    for ev in run_agent_stream(
                        user_message=user_input,
                        conversation_history=st.session_state.conversation_history,
                        parsers=st.session_state.datasets,
                        client=st.session_state.client,
                        data_dirs=st.session_state.data_dirs,
                        working_copies=st.session_state.working_copies,
                    ):
                        etype = ev.get("type")
                        if etype == "thinking":
                            thinking_parts.append(ev["text"])
                            thinking_ph.markdown("🧠 _Thinking…_\n\n" + "".join(thinking_parts))
                        elif etype == "text":
                            answer_parts.append(ev["text"])
                            answer_ph.markdown("".join(answer_parts))
                        elif etype == "tool_start":
                            status.update(label=f"Using tool: {ev['name']}…")
                            status.write(f"🔧 `{ev['name']}`")
                        elif etype == "tool_end":
                            status.write(f"   ↳ {ev.get('status', 'ok')}")
                        elif etype == "done":
                            done_event = ev

                    # Finalize from the terminal event.
                    done_event = done_event or {}
                    answer = done_event.get("text", "".join(answer_parts))
                    tool_trace = done_event.get("tool_trace", [])
                    usage = done_event.get("usage", {})
                    st.session_state.conversation_history = done_event.get(
                        "history", st.session_state.conversation_history
                    )

                    status.update(
                        label=(f"Done — {len(tool_trace)} tool call(s)" if tool_trace else "Done"),
                        state="complete",
                        expanded=False,
                    )
                    if not thinking_parts:
                        thinking_ph.empty()
                    answer_ph.markdown(answer)

                    # Fold this turn's tokens into the running session total.
                    _sess = st.session_state.token_usage_total
                    _sess["input_tokens"] += usage.get("input_tokens", 0)
                    _sess["output_tokens"] += usage.get("output_tokens", 0)
                    _sess["total_tokens"] += usage.get("total_tokens", 0)
                    _sess["turns"] += 1

                    new_plot_ids = [
                        pid for pid in tools_module.FIGURE_REGISTRY.keys()
                        if pid not in pre_plot_ids
                    ]

                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": answer,
                        "plot_ids": new_plot_ids,
                        "tool_trace": tool_trace,
                        "usage": usage,
                    })

                    for pid in new_plot_ids:
                        _render_registered_figure(pid)
                    _render_agent_trace(tool_trace)
                    _render_token_usage(usage)

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
