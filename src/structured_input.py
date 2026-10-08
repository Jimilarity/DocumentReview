import ast
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from cache_paths import CachePaths, get_cache_paths
from constants import (
    RULES_PATH,
    STRUCTURED_FIELD_CACHE_SCHEMA_VERSION,
)
from review_config import load_context_sensitive_settings
from rules.normalization import (
    canonical_document_type,
    canonical_field_name,
    load_rule_aliases,
)
from rules.rule_set import RuleSetBuilder
from structured_field_cache import (
    StructuredFieldCache,
    build_structured_source_fingerprint,
)
from reviewers.document_mapping import (
    deterministic_document_section_map,
    normalize_document_section_map_lenient,
)
from utils import atomic_write_json, read_json


STRUCTURED_SOURCE_TYPE = "structured_json"
EMPTY_STRING_MARKERS = {"", "空", "null", "none"}
PARENTHETICAL_FIELD_SUFFIX = re.compile(
    r"^(?P<name>.+?)[（(][A-Za-z_][A-Za-z0-9_]*[）)]$"
)


def _normalize_empty_markers(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize_empty_markers(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_empty_markers(item) for item in value]
    if isinstance(value, str) and value.strip().casefold() in EMPTY_STRING_MARKERS:
        return None
    return value


def _document_records(value: Any) -> Iterator[Dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        return
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                raise TypeError("文书数组中的每一项都必须是对象")
            yield item
        return
    if value is not None:
        raise TypeError("顶层文书值必须是对象、对象数组或 null")


def _review_requirements(
    rule_type: int,
    rules_path: str | Path = RULES_PATH,
) -> tuple[List[str], Dict[str, List[str]]]:
    rules = (
        RuleSetBuilder(read_json(rules_path))
        .for_rule_type(rule_type)
        .build()
        .rules
    )
    names: List[str] = []
    fields: Dict[str, List[str]] = defaultdict(list)
    for rule in rules:
        names.extend((rule.get("上下文无关审查事项") or {}).keys())
        for context_item in rule.get("上下文相关审查事项") or []:
            for document_type, field_items in (
                context_item.get("字段") or {}
            ).items():
                names.append(document_type)
                for field_item in field_items:
                    field_name = field_item.get("field")
                    if field_name and field_name not in fields[document_type]:
                        fields[document_type].append(field_name)
    prewarm_fields = load_context_sensitive_settings()["prewarm_fields"]
    names.extend(prewarm_fields)
    for document_type, field_names in prewarm_fields.items():
        for field_name in field_names:
            if field_name not in fields[document_type]:
                fields[document_type].append(field_name)
    return list(dict.fromkeys(names)), dict(fields)


def _field_name_from_spec(
    document_type: str,
    source_name: str,
    field_specs: Dict[str, Dict[str, Dict[str, Any]]],
    aliases: Dict[str, Dict[str, Any]],
) -> str:
    canonical_name = canonical_field_name(
        document_type,
        source_name,
        aliases,
    )
    document_specs = field_specs.get(document_type, {})
    if canonical_name in document_specs:
        return canonical_name

    suffix_match = PARENTHETICAL_FIELD_SUFFIX.fullmatch(canonical_name)
    # 结构化平台常把英文键附在中文字段后，例如“案件编号（case_no）”。
    # 即使该字段未进入当前规则字段表，也应去掉纯技术后缀并保留中文业务名，
    # 避免同一字段因不同数据来源形成两个键。
    if suffix_match:
        return suffix_match.group("name")

    for field_name, spec in document_specs.items():
        if spec.get("eng_name") == source_name:
            return field_name
    return canonical_name


def _literal_values(field_type: str) -> List[Any]:
    body = field_type[len("literal[") : -1]
    return list(ast.literal_eval(f"[{body}]"))


def _coerce_configured_value(value: Any, field_type: str) -> tuple[bool, Any]:
    if value is None:
        return True, None
    if field_type in {"str", "datetime"}:
        return (True, value) if isinstance(value, str) else (False, value)
    if field_type == "bool":
        if isinstance(value, bool):
            return True, value
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized in {"是", "有", "true", "1", "yes"}:
                return True, True
            if normalized in {"否", "无", "false", "0", "no"}:
                return True, False
        return False, value
    if field_type == "int":
        if isinstance(value, int) and not isinstance(value, bool):
            return True, value
        if isinstance(value, str) and re.fullmatch(r"[-+]?\d+", value.strip()):
            return True, int(value)
        return False, value
    if field_type == "float":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return True, value
        if isinstance(value, str):
            try:
                return True, float(value.replace(",", "").strip())
            except ValueError:
                pass
        return False, value
    if field_type == "dict":
        return (True, value) if isinstance(value, dict) else (False, value)
    if field_type == "list[dict]":
        if isinstance(value, dict):
            return True, [value]
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return True, value
        return False, value
    if field_type == "list[str]":
        if isinstance(value, str):
            return True, [value]
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return True, value
        return False, value
    if field_type.startswith("literal[") and field_type.endswith("]"):
        return (True, value) if value in _literal_values(field_type) else (False, value)
    return False, value


def _normalize_section_fields(
    document_type: str,
    record: Dict[str, Any],
    field_specs: Dict[str, Dict[str, Dict[str, Any]]],
    aliases: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    document_specs = field_specs.get(document_type, {})
    for source_name, raw_value in record.items():
        field_name = _field_name_from_spec(
            document_type,
            source_name,
            field_specs,
            aliases,
        )
        value = _normalize_empty_markers(raw_value)
        spec = document_specs.get(field_name)
        if spec is not None:
            valid, coerced_value = _coerce_configured_value(
                value,
                spec["type"],
            )
            if valid:
                value = coerced_value
        elif field_name.startswith("是否"):
            # 结构化来源可能携带规则当前未使用、但后续关联仍会读取的布尔
            # 字段。对明确的“是否”字段做同样的安全转换，避免“否”作为
            # 非空字符串在布尔判断中被误认为 True。
            valid, coerced_value = _coerce_configured_value(value, "bool")
            if valid:
                value = coerced_value
        normalized[field_name] = value
    return normalized


def _first_value(
    records: Iterable[Dict[str, Any]],
    field_names: Iterable[str],
) -> Any:
    materialized_records = list(records)
    for field_name in field_names:
        for record in materialized_records:
            value = record.get(field_name)
            if value not in (None, "", "空"):
                return value
    return None


def _build_metadata(
    source_path: Path,
    source_sha256: str,
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "source_type": STRUCTURED_SOURCE_TYPE,
        "source_path": str(source_path.resolve(strict=False)),
        "source_sha256": source_sha256,
        "案号": _first_value(
            records,
            ("行政处罚决定书文号", "案件编号", "案件案号", "案号"),
        ),
        "执法单位": _first_value(records, ("执法主体名称", "执法机构名称")),
        "案由": _first_value(records, ("案由",)),
        "案情": _first_value(records, ("违法事实", "违法事实/行为", "案件简介")),
        "违法依据": _first_value(records, ("违法依据",)),
        "处罚依据": _first_value(records, ("处罚依据", "拟处罚依据")),
        "立案日期": _first_value(records, ("立案时间", "立案日期")),
        "结案日期": _first_value(records, ("结案日期",)),
        "处理结果": _first_value(
            records,
            ("行政处罚内容", "处罚决定种类", "处罚类别"),
        ),
    }


def _related_section_id(
    record: Dict[str, Any],
    document_section_map: Dict[str, List[int]],
    source_document_section_map: Dict[str, List[int]],
    aliases: Dict[str, Dict[str, Any]],
) -> int | None:
    related_name = record.get("文书名称")
    if not isinstance(related_name, str) or not related_name.strip():
        return None
    source_name = related_name.strip()
    source_section_ids = source_document_section_map.get(source_name, [])
    if len(source_section_ids) == 1:
        return source_section_ids[0]
    canonical_name = canonical_document_type(source_name, aliases)
    section_ids = document_section_map.get(canonical_name, [])
    return section_ids[0] if len(section_ids) == 1 else None


def prepare_structured_json(
    file_path: str | Path,
    rule_type: int,
    *,
    cache_root: str | Path | None = None,
    rules_path: str | Path = RULES_PATH,
) -> Dict[str, Any]:
    source_path = Path(file_path)
    source_bytes = source_path.read_bytes()
    try:
        source_data = json.loads(source_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"结构化 JSON 无法解析: {source_path}") from exc
    if not isinstance(source_data, dict):
        raise TypeError("结构化 JSON 根节点必须是对象")

    aliases = load_rule_aliases()
    context_settings = load_context_sensitive_settings()
    field_specs = context_settings["field_specs"]
    service_receipt_type = context_settings["service_receipt_document_type"]

    section_records: List[Dict[str, Any]] = []
    direct_document_section_map: Dict[str, List[int]] = defaultdict(list)
    source_document_section_map: Dict[str, List[int]] = defaultdict(list)
    for source_document_name, raw_value in source_data.items():
        if source_document_name.startswith("__"):
            continue
        document_type = canonical_document_type(
            source_document_name,
            aliases,
        )
        for record in _document_records(raw_value):
            section_id = len(section_records) + 1
            normalized_record = _normalize_empty_markers(record)
            section_records.append(
                {
                    "section_id": section_id,
                    "source_document_name": source_document_name,
                    "document_type": document_type,
                    "record": normalized_record,
                }
            )
            direct_document_section_map[document_type].append(section_id)
            source_document_section_map[source_document_name].append(
                section_id
            )

    if not section_records:
        raise ValueError("结构化 JSON 中没有可审查的文书对象")

    dir_info = [
        {
            "section_id": item["section_id"],
            "section_name": item["source_document_name"],
            "source_document_name": item["source_document_name"],
            "normalized_document_type": item["document_type"],
            "catalog_source": STRUCTURED_SOURCE_TYPE,
            "section_page": index,
            "section_end_page": index + 1,
        }
        for index, item in enumerate(section_records)
    ]
    ocr_results = [
        {
            "image_index": index,
            "document_content": (
                "[structured_document]\n"
                f"source_document_name: {item['source_document_name']}\n"
                f"document_type: {item['document_type']}\n"
                f"{json.dumps(item['record'], ensure_ascii=False, indent=2)}\n"
                "[/structured_document]"
            ),
        }
        for index, item in enumerate(section_records)
    ]
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    metadata = _build_metadata(
        source_path,
        source_sha256,
        [item["record"] for item in section_records],
    )

    cache_paths: CachePaths = get_cache_paths(
        source_path,
        cache_root=cache_root,
    )
    cache = StructuredFieldCache(
        cache_paths.structured_fields,
        schema_version=STRUCTURED_FIELD_CACHE_SCHEMA_VERSION,
        source_fingerprint=build_structured_source_fingerprint(
            metadata,
            dir_info,
            ocr_results,
        ),
    )
    required_names, required_fields = _review_requirements(
        rule_type,
        rules_path,
    )
    deterministic_mapping = deterministic_document_section_map(
        required_names,
        dir_info,
    )
    required_name_set = set(required_names)
    combined_mapping: Dict[str, List[int]] = {
        name: list(section_ids)
        for name, section_ids in direct_document_section_map.items()
        if name in required_name_set
    }
    for document_type, section_ids in deterministic_mapping.items():
        combined_mapping.setdefault(document_type, []).extend(section_ids)
    document_section_map, _ = normalize_document_section_map_lenient(
        combined_mapping,
        dir_info,
        preferred_document_types=[
            name
            for name in direct_document_section_map
            if name in required_name_set
        ],
    )
    presence = {
        name: bool(document_section_map.get(name))
        for name in required_names
    }
    presence.update({name: True for name in document_section_map})
    cache.initialize_document_presence(presence)
    cache.initialize_document_section_map(dict(document_section_map))

    next_event_id = 1
    for item in section_records:
        section_id = item["section_id"]
        document_type = item["document_type"]
        record = item["record"]
        cache.register_section(section_id, document_type)
        fields = _normalize_section_fields(
            document_type,
            record,
            field_specs,
            aliases,
        )
        mapped_document_types = [
            mapped_type
            for mapped_type, section_ids in document_section_map.items()
            if section_id in section_ids
        ]
        for mapped_document_type in mapped_document_types:
            if mapped_document_type != document_type:
                fields.update(
                    _normalize_section_fields(
                        mapped_document_type,
                        record,
                        field_specs,
                        aliases,
                    )
                )
            for field_name in required_fields.get(mapped_document_type, []):
                fields.setdefault(field_name, None)
        if document_type == service_receipt_type:
            event_text = json.dumps(record, ensure_ascii=False, indent=2)
            cache.set_delivery_events(
                section_id,
                [
                    {
                        "event_id": next_event_id,
                        "source_order": 1,
                        "event_text": event_text,
                        "related_section_id": _related_section_id(
                            record,
                            document_section_map,
                            source_document_section_map,
                            aliases,
                        ),
                        **fields,
                    }
                ],
            )
            next_event_id += 1
        else:
            cache.merge_fields(section_id, fields)

    cache.mark_preparation_completed()

    atomic_write_json(cache_paths.image_list, [])
    atomic_write_json(cache_paths.ocr_results, ocr_results)
    atomic_write_json(cache_paths.metadata, metadata)
    atomic_write_json(cache_paths.directory, dir_info)
    cache.save()

    return {
        "completed": True,
        "source_type": STRUCTURED_SOURCE_TYPE,
        "source_sha256": source_sha256,
        "document_count": len(section_records),
        "cache_files": {
            name: str(path)
            for name, path in cache_paths.pre_review_files.items()
        },
        "structured_fields": str(cache_paths.structured_fields),
    }
