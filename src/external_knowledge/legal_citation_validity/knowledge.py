"""调用版本化法条检索 API，为单项审查提供候选法条。"""

import asyncio
import json
import os
import time
from datetime import date
from threading import Lock
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..models import KnowledgeContext, KnowledgeItem
from ..registry import register_knowledge
from ..tracing import trace_event


DEFAULT_BASE_URL = "https://review.zfqp.fun/law-api"
DEFAULT_TOP_K = 3
DEFAULT_VERSION_K = 7
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 2
MAX_QUERY_LENGTH = 300

_cache: dict[str, list[KnowledgeItem]] = {}
_cache_lock = Lock()


def _configured_string_list(config: dict[str, Any], key: str) -> list[str]:
    value = config.get(key, [])
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise TypeError(f"法条检索查询.{key} 必须是非空字符串数组")
    return [item.strip() for item in value]


def _query_config(context: KnowledgeContext) -> dict[str, Any]:
    config = context.review_item.get("法条检索查询", {})
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise TypeError("法条检索查询必须是对象")
    return config


def _as_of(metadata: dict[str, Any], config: dict[str, Any]) -> str | None:
    configured_fields = _configured_string_list(config, "日期字段")
    fallback_fields = ["案发日期", "违法行为发生日期", "立案日期", "结案日期"]
    for field_name in dict.fromkeys([*configured_fields, *fallback_fields]):
        value = metadata.get(field_name)
        if isinstance(value, str):
            try:
                return date.fromisoformat(value.strip()).isoformat()
            except ValueError:
                continue
    return None


def _text_value(metadata: dict[str, Any], field_name: str) -> str | None:
    value = metadata.get(field_name)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _query_text(
    context: KnowledgeContext,
    config: dict[str, Any],
) -> tuple[str, list[str]]:
    metadata = context.metadata
    case_reason = _text_value(metadata, "案由")
    reason_source = "案由"
    if case_reason is None:
        for field_name in ("违法事实", "案件简要情况", "处理结果"):
            case_reason = _text_value(metadata, field_name)
            if case_reason is not None:
                reason_source = field_name
                break
    if case_reason is None:
        raise ValueError("法条检索需要案由")

    case_facts = _text_value(metadata, "案情")
    fact_sources: list[str] = []
    if case_facts is not None:
        fact_sources.append("案情")
    else:
        fragments = []
        for field_name in (
            "违法事实",
            "案件简要情况",
            "处罚具体内容",
            "陈述、申辩处理结果",
            "处理结果",
            "立案日期",
            "结案日期",
        ):
            value = _text_value(metadata, field_name)
            if value is not None:
                fragments.append(f"{field_name}：{value}")
                fact_sources.append(field_name)
        if fragments:
            case_facts = "；".join(fragments)
        else:
            case_facts = case_reason
            fact_sources.append(reason_source)
    if not case_facts:
        raise ValueError("法条检索需要案情")

    parts = [f"案由：{case_reason}", f"案情：{case_facts}"]
    for field_name in _configured_string_list(config, "字段"):
        value = _text_value(metadata, field_name)
        if value is not None:
            parts.append(f"{field_name}：{value}")
    return "\n".join(parts)[:MAX_QUERY_LENGTH], fact_sources


def _positive_int(config: dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TypeError(f"法条检索查询.{key} 必须是正整数")
    return value


def _request(payload: dict[str, Any]) -> dict[str, Any]:
    base_url = os.getenv("LAW_RETRIEVAL_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    endpoint = f"{base_url}/retrieve"
    api_key = os.getenv("LAW_RETRIEVAL_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("LAW_RETRIEVAL_API_KEY 未配置")
    timeout = float(
        os.getenv("LAW_RETRIEVAL_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
    )
    max_retries = int(
        os.getenv("LAW_RETRIEVAL_MAX_RETRIES", str(DEFAULT_MAX_RETRIES))
    )
    if max_retries < 0:
        raise ValueError("LAW_RETRIEVAL_MAX_RETRIES 不能小于 0")
    request = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "X-API-Key": api_key,
        },
        method="POST",
    )
    for attempt in range(max_retries + 1):
        try:
            trace_event(
                "law_api_request",
                provider="legal_citation_validity",
                endpoint=endpoint,
                attempt=attempt + 1,
                payload=payload,
            )
            with urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            if not isinstance(result, dict):
                raise RuntimeError("法条检索 API 返回不是对象")
            trace_event(
                "law_api_response",
                provider="legal_citation_validity",
                endpoint=endpoint,
                attempt=attempt + 1,
                status_code=getattr(response, "status", None),
                response=result,
            )
            return result
        except HTTPError as exc:
            trace_event(
                "law_api_error",
                provider="legal_citation_validity",
                endpoint=endpoint,
                attempt=attempt + 1,
                exception_type=type(exc).__name__,
                message=str(exc),
            )
            if exc.code not in {502, 503} or attempt == max_retries:
                raise RuntimeError(f"法条检索 API 请求失败: {exc}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            trace_event(
                "law_api_error",
                provider="legal_citation_validity",
                endpoint=endpoint,
                attempt=attempt + 1,
                exception_type=type(exc).__name__,
                message=str(exc),
            )
            raise RuntimeError(f"法条检索 API 请求失败: {exc}") from exc
        time.sleep(0.5 * (attempt + 1))
    raise AssertionError("法条检索重试流程异常结束")


def _format_candidates(response: dict[str, Any]) -> list[KnowledgeItem]:
    candidates = response.get("candidates", [])
    if not isinstance(candidates, list):
        return []
    items: list[KnowledgeItem] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        law_name = candidate.get("law_name")
        article = candidate.get("article")
        article_text = candidate.get("article_text")
        if not isinstance(law_name, str) or not isinstance(article_text, str):
            continue
        suffix = candidate.get("article_suffix")
        article_label = f"第{article}条" if isinstance(article, int) else "条文"
        if isinstance(suffix, int):
            article_label += f"之{suffix}"
        focused = candidate.get("focused_text")
        content = (
            "版本化法条检索候选（仅供核对，不作为自动选取违法或处罚依据）：\n"
            f"法规：{law_name}\n"
            f"条文：{article_label}\n"
            f"版本：{candidate.get('version_id')}\n"
            f"生效期间：{candidate.get('effective_from')} 至 "
            f"{candidate.get('effective_to') or '现行'}\n"
            f"相关片段：{focused if isinstance(focused, str) else article_text}\n"
            f"完整正文：{article_text}"
        )
        items.append(KnowledgeItem(content=content))
    return items


@register_knowledge("legal_citation_validity")
async def retrieve_legal_citation_validity(
    context: KnowledgeContext,
) -> list[KnowledgeItem]:
    """用规则配置的上下文查询版本化法条，失败时由服务层降级。"""

    config = _query_config(context)
    query, fact_sources = _query_text(context, config)
    if not query:
        return []
    payload: dict[str, Any] = {
        "text": query,
        "top_k": min(_positive_int(config, "top_k", DEFAULT_TOP_K), 50),
        "version_k": min(
            _positive_int(config, "version_k", DEFAULT_VERSION_K), 30
        ),
        "with_text": True,
        "with_focus": True,
    }
    as_of = _as_of(context.metadata, config)
    if as_of:
        payload["as_of"] = as_of
    trace_event(
        "law_api_query_context",
        provider="legal_citation_validity",
        case_facts_sources=fact_sources,
        as_of=as_of,
    )
    cache_key = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    with _cache_lock:
        cached = _cache.get(cache_key)
    if cached is not None:
        trace_event(
            "law_api_cache_hit",
            provider="legal_citation_validity",
            payload=payload,
            candidate_count=len(cached),
        )
        return cached
    items = _format_candidates(await asyncio.to_thread(_request, payload))
    with _cache_lock:
        _cache[cache_key] = items
    return items
