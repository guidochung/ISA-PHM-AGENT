from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from isa_phm import ISAWrapper

logger = logging.getLogger("isa_phm.agent_server")


@dataclass(frozen=True)
class AgentServerConfig:
    isa_json: Path
    data_root: Path | None
    strict_validation: bool
    csv_bad_lines: str
    openai_model: str
    openai_api_key: str | None

    @staticmethod
    def from_env() -> "AgentServerConfig":
        isa_json_raw = os.getenv("ISA_PHM_JSON", "").strip()
        if not isa_json_raw:
            raise ValueError(
                "Missing ISA_PHM_JSON environment variable. "
                "Set it to an ISA-JSON file path."
            )
        isa_json = Path(isa_json_raw).expanduser().resolve()
        data_root_raw = os.getenv("ISA_PHM_DATA_ROOT", "").strip()
        data_root = Path(data_root_raw).expanduser().resolve() if data_root_raw else None
        strict_validation = os.getenv("ISA_PHM_STRICT_VALIDATION", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        csv_bad_lines = os.getenv("ISA_PHM_CSV_BAD_LINES", "error").strip().lower()
        openai_model = os.getenv("OPENAI_MODEL", "gpt-5-mini").strip()
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip() or None
        return AgentServerConfig(
            isa_json=isa_json,
            data_root=data_root,
            strict_validation=strict_validation,
            csv_bad_lines=csv_bad_lines,
            openai_model=openai_model,
            openai_api_key=openai_api_key,
        )


def create_wrapper(cfg: AgentServerConfig) -> ISAWrapper:
    return ISAWrapper(
        cfg.isa_json,
        data_root=cfg.data_root,
        strict_validation=cfg.strict_validation,
        csv_bad_lines=cfg.csv_bad_lines,  # keep strict by default
    )


def _openai_tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "ai_context",
                "description": "Return normalized metadata context for the dataset.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "include_semantics": {"type": "boolean"},
                        "include_validation": {"type": "boolean"},
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "validate_dataset",
                "description": "Validate dataset structure, files, and semantics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "check_files": {"type": "boolean"},
                        "semantic_strict": {"type": "boolean"},
                        "max_unknown_ratio": {"type": "number"},
                        "max_ambiguous_ratio": {"type": "number"},
                        "require_override_config": {"type": "boolean"},
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "load_dataframe_with_meta",
                "description": "Load one run and return metadata plus JSON-safe head preview.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "study_id": {"type": "string"},
                        "assay_id": {"type": "string"},
                        "run_id": {"type": "string"},
                        "file_type": {
                            "type": "string",
                            "enum": ["raw", "processed", "auto"],
                        },
                        "head_rows": {"type": "integer", "minimum": 0, "maximum": 200},
                    },
                    "required": ["study_id", "assay_id"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def _parse_tool_arguments(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    text = raw.strip()
    if not text:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return parsed


def _run_openai_tool_loop(
    *,
    wrapper: ISAWrapper,
    user_message: str,
    api_key: str,
    model: str,
    max_round_trips: int = 8,
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai package is not installed. Install optional deps with: "
            "pip install -e \".[agent]\""
        ) from exc

    client = OpenAI(api_key=api_key)
    tools = _openai_tool_specs()

    system_text = (
        "You are an ISA-PHM analysis assistant. "
        "Always use tools for dataset facts instead of inventing values. "
        "Be explicit about missing data and validation issues."
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_message},
    ]

    tool_invocations: list[dict[str, Any]] = []

    for _ in range(max_round_trips):
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
        )
        choice = completion.choices[0]
        msg = choice.message
        tool_calls = msg.tool_calls or []

        if not tool_calls:
            answer_text = msg.content or ""
            return {
                "answer": answer_text,
                "tool_invocations": tool_invocations,
                "finish_reason": choice.finish_reason,
                "model": model,
            }

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            tool_name = tc.function.name
            args = _parse_tool_arguments(tc.function.arguments)
            result = wrapper.call_tool(tool_name, args)
            tool_invocations.append(
                {
                    "tool": tool_name,
                    "args": args,
                    "ok": bool(result.get("ok", False)),
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    return {
        "answer": (
            "Reached maximum tool round trips before a final answer was produced. "
            "Try a more specific question."
        ),
        "tool_invocations": tool_invocations,
        "finish_reason": "max_round_trips",
        "model": model,
    }


def create_app():
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError(
            "fastapi is not installed. Install optional deps with: "
            "pip install -e \".[agent]\""
        ) from exc

    class ToolInvokeRequest(BaseModel):
        name: str
        args: dict[str, Any] = Field(default_factory=dict)

    class ChatRequest(BaseModel):
        message: str
        max_round_trips: int = 8

    cfg = AgentServerConfig.from_env()
    wrapper = create_wrapper(cfg)

    app = FastAPI(
        title="isa-phm-agent-server",
        version="0.1.0",
        description="Tool API + optional OpenAI tool-calling loop for ISA-PHM wrapper.",
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        overview = wrapper.investigation_overview()
        return {
            "ok": True,
            "isa_json": str(cfg.isa_json),
            "n_studies": overview.n_studies,
            "title": overview.title,
        }

    @app.get("/tools")
    def tools() -> dict[str, Any]:
        return {"tools": wrapper.list_tools()}

    @app.post("/tool")
    def invoke_tool(req: ToolInvokeRequest) -> dict[str, Any]:
        return wrapper.call_tool(req.name, req.args)

    @app.post("/chat")
    def chat(req: ChatRequest) -> dict[str, Any]:
        if not cfg.openai_api_key:
            raise HTTPException(
                status_code=503,
                detail=(
                    "OPENAI_API_KEY is not configured. "
                    "Set OPENAI_API_KEY to enable /chat, or use /tool directly."
                ),
            )
        try:
            return _run_openai_tool_loop(
                wrapper=wrapper,
                user_message=req.message,
                api_key=cfg.openai_api_key,
                model=cfg.openai_model,
                max_round_trips=max(1, min(req.max_round_trips, 16)),
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is not installed. Install optional deps with: "
            "pip install -e \".[agent]\""
        ) from exc

    host = os.getenv("ISA_AGENT_HOST", "127.0.0.1")
    port = int(os.getenv("ISA_AGENT_PORT", "8000"))
    uvicorn.run("isa_phm.agent_server:create_app", factory=True, host=host, port=port)


if __name__ == "__main__":
    main()
