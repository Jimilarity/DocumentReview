import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from constants import RULE_ALIASES_CONFIG_PATH
from utils import load_yaml


RuleAliases = Dict[str, Dict[str, Any]]


def load_rule_aliases(
    path: str | Path = RULE_ALIASES_CONFIG_PATH,
) -> RuleAliases:
    configured = load_yaml(path) or {}
    document_type_aliases = configured.get("document_type_aliases") or {}
    field_aliases = configured.get("field_aliases") or {}
    if not isinstance(document_type_aliases, dict):
        raise TypeError("document_type_aliases 必须是对象")
    if not isinstance(field_aliases, dict):
        raise TypeError("field_aliases 必须是对象")
    return {
        "document_type_aliases": document_type_aliases,
        "field_aliases": field_aliases,
    }


def _resolve_alias(value: str, aliases: Mapping[str, str]) -> str:
    resolved = value
    visited = set()
    while resolved in aliases:
        if resolved in visited:
            raise ValueError(f"规则别名配置存在循环: {value}")
        visited.add(resolved)
        target = aliases[resolved]
        if not isinstance(target, str) or not target.strip():
            raise TypeError(f"规则别名 {resolved} 的目标必须是非空字符串")
        resolved = target.strip()
    return resolved


def canonical_document_type(
    document_type: str,
    aliases: RuleAliases,
) -> str:
    return _resolve_alias(
        document_type,
        aliases["document_type_aliases"],
    )


def canonical_field_name(
    document_type: str,
    field_name: str,
    aliases: RuleAliases,
) -> str:
    document_aliases = aliases["field_aliases"].get(document_type) or {}
    if not isinstance(document_aliases, dict):
        raise TypeError(f"field_aliases.{document_type} 必须是对象")
    return _resolve_alias(field_name, document_aliases)


def _merge_field_items(
    existing: List[Dict[str, Any]],
    incoming: Iterable[Dict[str, Any]],
) -> None:
    positions = {
        item.get("field"): index
        for index, item in enumerate(existing)
        if isinstance(item, dict) and isinstance(item.get("field"), str)
    }
    for field_item in incoming:
        field_name = field_item.get("field")
        if field_name not in positions:
            positions[field_name] = len(existing)
            existing.append(field_item)
            continue
        current = existing[positions[field_name]]
        if isinstance(current.get("required"), bool) and isinstance(
            field_item.get("required"), bool
        ):
            current["required"] = (
                current["required"] or field_item["required"]
            )


def _normalize_context_free_documents(
    documents: Dict[str, Any],
    aliases: RuleAliases,
) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for source_name, review_item in documents.items():
        document_type = canonical_document_type(source_name, aliases)
        if document_type in normalized and normalized[document_type] != review_item:
            raise ValueError(
                f"上下文无关审查事项中的文书别名合并后内容冲突: "
                f"{source_name} -> {document_type}"
            )
        normalized[document_type] = review_item
    return normalized


def _normalize_context_fields(
    documents: Dict[str, Any],
    aliases: RuleAliases,
) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for source_name, raw_field_items in documents.items():
        document_type = canonical_document_type(source_name, aliases)
        if not isinstance(raw_field_items, list):
            normalized.setdefault(document_type, raw_field_items)
            continue
        field_items = []
        for raw_field_item in raw_field_items:
            field_item = copy.deepcopy(raw_field_item)
            if isinstance(field_item, dict) and isinstance(
                field_item.get("field"), str
            ):
                field_item["field"] = canonical_field_name(
                    document_type,
                    field_item["field"],
                    aliases,
                )
            field_items.append(field_item)
        target = normalized.setdefault(document_type, [])
        if not isinstance(target, list):
            raise TypeError(f"上下文相关审查字段 {document_type} 必须是数组")
        _merge_field_items(target, field_items)
    return normalized


def normalize_rules(
    rules: Iterable[Dict[str, Any]],
    aliases: RuleAliases | None = None,
) -> List[Dict[str, Any]]:
    configured_aliases = aliases if aliases is not None else load_rule_aliases()
    normalized_rules = copy.deepcopy(list(rules))
    for rule in normalized_rules:
        context_free = rule.get("上下文无关审查事项")
        if isinstance(context_free, dict):
            rule["上下文无关审查事项"] = (
                _normalize_context_free_documents(
                    context_free,
                    configured_aliases,
                )
            )
        for context_item in rule.get("上下文相关审查事项") or []:
            context_fields = context_item.get("字段")
            if isinstance(context_fields, dict):
                context_item["字段"] = _normalize_context_fields(
                    context_fields,
                    configured_aliases,
                )
    return normalized_rules
