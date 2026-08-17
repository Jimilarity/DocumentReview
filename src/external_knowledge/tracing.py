"""外部知识模块的可选审查轨迹桥接。"""

from typing import Any


def trace_event(event: str, **payload: Any) -> None:
    """在主审查运行时写入轨迹，不让调试依赖影响知识调用。"""
    try:
        from agent_trace import trace_event as emit_trace_event

        emit_trace_event(event, **payload)
    except Exception:
        return
