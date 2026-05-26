"""
JSON-only tool registry for AI/agent integrations.

All tool inputs and outputs are plain JSON-serializable dictionaries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

if TYPE_CHECKING:
    from .wrapper import ISAWrapper


ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class _ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolRegistry:
    """Registry of JSON-in/JSON-out tools backed by :class:`ISAWrapper`."""

    def __init__(self, wrapper: "ISAWrapper") -> None:
        self._wrapper = wrapper
        self._handlers: dict[str, ToolHandler] = {
            "ai_context": self._tool_ai_context,
            "validate_dataset": self._tool_validate_dataset,
            "load_dataframe_with_meta": self._tool_load_dataframe_with_meta,
        }
        self._specs: dict[str, _ToolSpec] = {
            "ai_context": _ToolSpec(
                name="ai_context",
                description="Return metadata-only normalized AI context.",
                input_schema={
                    "include_semantics": "bool (default true)",
                    "include_validation": "bool (default false)",
                },
            ),
            "validate_dataset": _ToolSpec(
                name="validate_dataset",
                description="Run wrapper validation checks and return issue report.",
                input_schema={
                    "check_files": "bool (default true)",
                    "semantic_strict": "bool (default false)",
                    "max_unknown_ratio": "float (default 0.05)",
                    "max_ambiguous_ratio": "float (default 0.0)",
                    "require_override_config": "bool (default false)",
                },
            ),
            "load_dataframe_with_meta": _ToolSpec(
                name="load_dataframe_with_meta",
                description=(
                    "Load one run and return load metadata plus JSON-safe frame summary/head."
                ),
                input_schema={
                    "study_id": "str (required)",
                    "assay_id": "str (required)",
                    "run_id": "str | null (optional)",
                    "file_type": "'raw' | 'processed' | 'auto' (default 'processed')",
                    "head_rows": "int (default 5, max 200)",
                },
            ),
        }

    def list_tools(self) -> list[dict[str, Any]]:
        """Return stable tool metadata for discovery."""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
            }
            for spec in sorted(self._specs.values(), key=lambda s: s.name)
        ]

    def invoke(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Invoke a tool by name and always return a JSON-safe envelope."""
        if not isinstance(name, str) or not name.strip():
            return {
                "tool": str(name),
                "ok": False,
                "error": {
                    "type": "InvalidToolName",
                    "message": "Tool name must be a non-empty string.",
                },
            }
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return {
                "tool": name,
                "ok": False,
                "error": {
                    "type": "InvalidToolArgs",
                    "message": "Tool args must be a JSON object (dict).",
                },
            }

        handler = self._handlers.get(name)
        if handler is None:
            return {
                "tool": name,
                "ok": False,
                "error": {
                    "type": "UnknownTool",
                    "message": f"Unknown tool '{name}'.",
                },
            }
        try:
            result = handler(args)
            return {
                "tool": name,
                "ok": True,
                "result": _json_safe(result),
            }
        except Exception as exc:
            return {
                "tool": name,
                "ok": False,
                "error": {
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                },
            }

    def _tool_ai_context(self, args: dict[str, Any]) -> dict[str, Any]:
        include_semantics = bool(args.get("include_semantics", True))
        include_validation = bool(args.get("include_validation", False))
        return self._wrapper.ai_context(
            include_semantics=include_semantics,
            include_validation=include_validation,
        )

    def _tool_validate_dataset(self, args: dict[str, Any]) -> dict[str, Any]:
        report = self._wrapper.validate_dataset(
            check_files=bool(args.get("check_files", True)),
            semantic_strict=bool(args.get("semantic_strict", False)),
            max_unknown_ratio=float(args.get("max_unknown_ratio", 0.05)),
            max_ambiguous_ratio=float(args.get("max_ambiguous_ratio", 0.0)),
            require_override_config=bool(args.get("require_override_config", False)),
        )
        return report.model_dump()

    def _tool_load_dataframe_with_meta(self, args: dict[str, Any]) -> dict[str, Any]:
        study_id = args.get("study_id")
        assay_id = args.get("assay_id")
        if not isinstance(study_id, str) or not study_id.strip():
            raise ValueError("tool 'load_dataframe_with_meta' requires 'study_id' (str).")
        if not isinstance(assay_id, str) or not assay_id.strip():
            raise ValueError("tool 'load_dataframe_with_meta' requires 'assay_id' (str).")

        run_id_value = args.get("run_id")
        run_id = run_id_value if isinstance(run_id_value, str) else None
        file_type = str(args.get("file_type", "processed"))
        head_rows = int(args.get("head_rows", 5))
        head_rows = max(0, min(head_rows, 200))

        assay = self._wrapper.study(study_id).assay(assay_id)
        df, meta = assay.load_dataframe_with_meta(run_id=run_id, file_type=file_type)

        head_records = _df_head_records(df, head_rows=head_rows)
        numeric_cols = list(df.select_dtypes(include="number").columns)
        value_stats: dict[str, Any] = {}
        if "value" in df.columns:
            series = pd.to_numeric(df["value"], errors="coerce").dropna()
            if not series.empty:
                value_stats = {
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=0)),
                }

        return {
            "metadata": meta.model_dump(),
            "dataframe": {
                "n_rows": int(df.shape[0]),
                "n_columns": int(df.shape[1]),
                "columns": [str(c) for c in df.columns],
                "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
                "numeric_columns": numeric_cols,
                "head": head_records,
                "value_stats": value_stats,
            },
        }


def _df_head_records(df: pd.DataFrame, head_rows: int) -> list[dict[str, Any]]:
    if head_rows <= 0:
        return []
    if df.empty:
        return []
    # Use pandas JSON conversion to avoid NumPy scalar serialization issues.
    return json.loads(df.head(head_rows).to_json(orient="records", date_format="iso"))


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        pass

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)
