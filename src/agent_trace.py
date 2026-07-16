import json
import logging
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

from langchain.agents.middleware import AgentMiddleware
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage

from errors.handler import (
    CURRENT_NODE,
    CURRENT_PAGE_INDEX,
    CURRENT_PDF_PATH,
    CURRENT_RULE_INDEX,
    TaskRotatingFileHandler,
)
from constants import PROJECT_ROOT, SENSITIVE_KEYS




def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


def _resolve_log_path() -> Path:
    configured_path = Path(
        os.getenv("AGENT_TRACE_LOG_PATH", "logs/agent_trace.jsonl")
    )
    if configured_path.is_absolute():
        return configured_path
    return PROJECT_ROOT / configured_path


def _build_trace_logger() -> tuple[logging.Logger, Path]:
    logger = logging.getLogger("document_review.agent_trace")
    if logger.handlers:
        return logger, _resolve_log_path()

    log_path = _resolve_log_path()
    handler = TaskRotatingFileHandler(
        log_path,
        max_bytes=int(
            os.getenv(
                "AGENT_TRACE_LOG_MAX_BYTES",
                str(50 * 1024 * 1024),
            )
        ),
        backup_count=int(os.getenv("AGENT_TRACE_LOG_BACKUP_COUNT", "5")),
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger, log_path


TRACE_ENABLED = _env_flag("AGENT_TRACE_ENABLED", True)
TRACE_LOGGER, TRACE_LOG_PATH = _build_trace_logger()


def _redact_data_url(value: str) -> str:
    if value.startswith("data:image/"):
        header, separator, encoded = value.partition(",")
        encoded_length = len(encoded) if separator else 0
        return f"<{header}; payload omitted; chars={encoded_length}>"
    return value


def _json_safe(value: Any, *, key: str | None = None) -> Any:
    if key and key.lower() in SENSITIVE_KEYS:
        return "<redacted>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _redact_data_url(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, BaseMessage):
        return serialize_message(value)
    if isinstance(value, dict):
        return {
            str(item_key): _json_safe(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(by_alias=True))
        except TypeError:
            return _json_safe(value.model_dump())
    return repr(value)


def serialize_message(message: BaseMessage) -> dict[str, Any]:
    serialized: dict[str, Any] = {
        "type": getattr(message, "type", type(message).__name__),
        "id": getattr(message, "id", None),
        "name": getattr(message, "name", None),
        "content": _json_safe(getattr(message, "content", None)),
    }
    for attribute in (
        "tool_calls",
        "invalid_tool_calls",
        "usage_metadata",
        "response_metadata",
        "tool_call_id",
        "status",
    ):
        attribute_value = getattr(message, attribute, None)
        if attribute_value not in (None, [], {}):
            serialized[attribute] = _json_safe(attribute_value)
    return serialized


def _context_fields() -> dict[str, Any]:
    return {
        "pdf_path": CURRENT_PDF_PATH.get(),
        "rule_index": CURRENT_RULE_INDEX.get(),
        "page_index": CURRENT_PAGE_INDEX.get(),
        "node": CURRENT_NODE.get(),
    }


def trace_event(event: str, **payload: Any) -> None:
    if not TRACE_ENABLED:
        return
    record = {
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "event": event,
        **_context_fields(),
        **_json_safe(payload),
    }
    try:
        TRACE_LOGGER.info(
            json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        )
    except Exception:
        # 调试日志绝不能反过来中断审查流程。
        pass


def agent_trace_config(agent_name: str) -> dict[str, Any]:
    return {
        "tags": [f"agent:{agent_name}"],
        "metadata": {"agent_name": agent_name},
    }


class AgentTraceCallbackHandler(BaseCallbackHandler):
    """把每次模型输入、输出及模型生成的工具调用写入 JSONL。"""

    raise_error = False

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        trace_event(
            "chat_model_start",
            run_id=run_id,
            parent_run_id=parent_run_id,
            model_id=serialized.get("id"),
            model_name=serialized.get("name"),
            tags=tags or [],
            metadata=metadata or {},
            invocation=kwargs,
            message_batches=[
                [serialize_message(message) for message in batch]
                for batch in messages
            ],
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        generations: list[list[Any]] = getattr(response, "generations", [])
        serialized_generations: list[list[dict[str, Any]]] = []
        for batch in generations:
            serialized_batch: list[dict[str, Any]] = []
            for generation in batch:
                message = getattr(generation, "message", None)
                generation_data = {
                    "text": _json_safe(getattr(generation, "text", None)),
                    "generation_info": _json_safe(
                        getattr(generation, "generation_info", None)
                    ),
                }
                if isinstance(message, BaseMessage):
                    generation_data["message"] = serialize_message(message)
                serialized_batch.append(generation_data)
            serialized_generations.append(serialized_batch)

        trace_event(
            "chat_model_end",
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags or [],
            generations=serialized_generations,
            llm_output=_json_safe(getattr(response, "llm_output", None)),
        )

        for batch in generations:
            for generation in batch:
                message = getattr(generation, "message", None)
                for tool_call in getattr(message, "tool_calls", None) or []:
                    trace_event(
                        "model_tool_call",
                        run_id=run_id,
                        parent_run_id=parent_run_id,
                        tags=tags or [],
                        tool_call=tool_call,
                    )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        trace_event(
            "chat_model_error",
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags or [],
            exception_type=type(error).__name__,
            message=str(error),
            traceback="".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        )


AGENT_TRACE_CALLBACK = AgentTraceCallbackHandler()


class AgentToolTraceMiddleware(AgentMiddleware):
    """记录 create_agent 实际执行的本地及 MCP 工具。"""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        tool_call = request.tool_call
        trace_event(
            "tool_start",
            agent_name=self.agent_name,
            tool_call=tool_call,
        )
        try:
            result = handler(request)
        except Exception as exc:
            trace_event(
                "tool_error",
                agent_name=self.agent_name,
                tool_call=tool_call,
                exception_type=type(exc).__name__,
                message=str(exc),
            )
            raise
        trace_event(
            "tool_end",
            agent_name=self.agent_name,
            tool_call=tool_call,
            result=result,
        )
        return result

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        tool_call = request.tool_call
        trace_event(
            "tool_start",
            agent_name=self.agent_name,
            tool_call=tool_call,
        )
        try:
            result = await handler(request)
        except Exception as exc:
            trace_event(
                "tool_error",
                agent_name=self.agent_name,
                tool_call=tool_call,
                exception_type=type(exc).__name__,
                message=str(exc),
            )
            raise
        trace_event(
            "tool_end",
            agent_name=self.agent_name,
            tool_call=tool_call,
            result=result,
        )
        return result
