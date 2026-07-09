from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

KNOWLEDGE_PATH = Path(__file__).resolve().parent / "knowledge" / "rule_example_pack.json"


def normalize_rule_code(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def parent_rule_code(value: Any) -> str:
    code = normalize_rule_code(value)
    if "-" in code:
        return code.split("-", 1)[0]
    return code


def compact_text(value: Any, limit: int = 220) -> str:
    text = str(value or "").replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


@lru_cache(maxsize=1)
def load_rule_knowledge() -> Dict[str, Any]:
    if not KNOWLEDGE_PATH.exists():
        return {"rules": {}, "meta": {"loaded": False, "path": str(KNOWLEDGE_PATH)}}
    try:
        payload = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"rules": {}, "meta": {"loaded": False, "path": str(KNOWLEDGE_PATH), "error": str(exc)}}
    rules = payload.get("rules")
    if not isinstance(rules, dict):
        payload["rules"] = {}
    payload.setdefault("meta", {})
    payload["meta"]["loaded"] = True
    payload["meta"]["path"] = str(KNOWLEDGE_PATH)
    return payload


def get_rule_knowledge(rule_code: Any) -> Optional[Dict[str, Any]]:
    payload = load_rule_knowledge()
    rules = payload.get("rules", {})
    code = normalize_rule_code(rule_code)
    return rules.get(code) or rules.get(parent_rule_code(code))


def list_rule_knowledge() -> Dict[str, Any]:
    payload = load_rule_knowledge()
    rules = payload.get("rules", {})
    summaries = []
    for code, item in sorted(rules.items(), key=lambda pair: pair[0]):
        summaries.append(
            {
                "rule_code": code,
                "rule_name": item.get("rule_name") or item.get("rule_content") or "",
                "review_point_count": len(item.get("review_points") or []),
                "required_document_count": len(item.get("required_documents") or []),
                "positive_example_count": len(item.get("positive_examples") or []),
                "negative_example_count": len(item.get("negative_examples") or []),
                "problem_pattern_count": len(item.get("problem_patterns") or []),
            }
        )
    return {"meta": payload.get("meta", {}), "rules": summaries}


def build_llm_context(rule_code: Any, max_examples_per_label: int = 2) -> Optional[Dict[str, Any]]:
    knowledge = get_rule_knowledge(rule_code)
    if not knowledge:
        return None

    positive_examples = _summarize_examples(knowledge.get("positive_examples") or [], max_examples_per_label)
    negative_examples = _summarize_examples(knowledge.get("negative_examples") or [], max_examples_per_label)
    problem_patterns = [
        {
            "problem": compact_text(item.get("problem"), 180),
            "confirmed_response": compact_text(item.get("confirmed_response"), 180),
        }
        for item in (knowledge.get("problem_patterns") or [])[:3]
    ]

    return {
        "source": "案卷评查示例库",
        "rule_code": normalize_rule_code(rule_code),
        "parent_rule_code": parent_rule_code(rule_code),
        "rule_content": compact_text(knowledge.get("rule_content"), 360),
        "review_points": [compact_text(item, 220) for item in (knowledge.get("review_points") or [])[:5]],
        "required_documents": [compact_text(item, 80) for item in (knowledge.get("required_documents") or [])[:10]],
        "positive_examples": positive_examples,
        "negative_examples": negative_examples,
        "common_problem_patterns": problem_patterns,
        "usage_instruction": (
            "这些内容是人工评查示例库提供的参考材料。审查时应优先判断当前案卷事实，"
            "不得照抄示例结论；可用正例理解合格材料形态，用反例识别常见缺陷，"
            "最终意见必须引用当前案卷目录中的 section_id 和实际提取到的事实。"
        ),
    }


def enrich_rule(rule: Dict[str, Any]) -> Dict[str, Any]:
    code = (
        rule.get("序号")
        or rule.get("编号")
        or rule.get("rule_code")
        or rule.get("原评分点编号")
        or rule.get("子评分点编号")
    )
    context = build_llm_context(code)
    if not context:
        return rule
    enriched = copy.deepcopy(rule)
    enriched["llm_rule_knowledge"] = context
    return enriched


def enrich_rule_list(rule_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [enrich_rule(item) if isinstance(item, dict) else item for item in rule_list]


def _summarize_examples(examples: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    summarized = []
    for item in examples[:limit]:
        summarized.append(
            {
                "example_id": item.get("example_id"),
                "case_short_name": item.get("case_short_name"),
                "documents": item.get("documents") or [],
                "pages": item.get("pages") or [],
                "key_excerpt": compact_text(item.get("key_excerpt"), 260),
                "review_reason": compact_text(item.get("review_reason"), 260),
            }
        )
    return summarized
