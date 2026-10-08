from functools import cache
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List

from constants import DOCUMENT_MAPPING_CONFIG_PATH, RULES_PATH
from rules.normalization import canonical_document_type, load_rule_aliases
from utils import load_yaml, read_json


RECEIPT_DOCUMENT_TYPE = "送达回证"
# 目录标题只要出现以下任一关键词即视为送达回证/送达回执；其方括号内的被送达
# 文书名称（如“送达回执[责令改正违法行为通知书]”）仅用于标识回证所服务的
# 文书，不应使该 section 被映射到被送达文书本身。
RECEIPT_TITLE_KEYWORDS = ("送达回证", "送达回执")
MAPPING_OCR_HINT_MAX_CHARS = 800


@cache
def load_document_mapping_config(
    config_path: str | Path = DOCUMENT_MAPPING_CONFIG_PATH,
) -> Dict[str, object]:
    return load_yaml(config_path)


@cache
def load_compatible_document_type_groups(
    config_path: str | Path = DOCUMENT_MAPPING_CONFIG_PATH,
) -> Dict[str, List[str]]:
    return load_document_mapping_config(config_path)[
        "compatible_document_type_groups"
    ]


@cache
def load_deterministic_document_matchers(
    config_path: str | Path = DOCUMENT_MAPPING_CONFIG_PATH,
) -> Dict[str, Dict[str, List[str]]]:
    configured = load_document_mapping_config(config_path).get(
        "deterministic_document_matchers",
        {},
    )
    if not isinstance(configured, dict):
        raise TypeError("deterministic_document_matchers 必须是对象")
    normalized: Dict[str, Dict[str, List[str]]] = {}
    for document_type, matcher in configured.items():
        if not isinstance(document_type, str) or not document_type:
            raise TypeError("确定性文书匹配器的文书类型必须是非空字符串")
        if not isinstance(matcher, dict):
            raise TypeError(
                f"deterministic_document_matchers.{document_type} 必须是对象"
            )
        normalized[document_type] = {}
        for key, values in matcher.items():
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value.strip()
                for value in values
            ):
                raise TypeError(
                    f"deterministic_document_matchers.{document_type}.{key} "
                    "必须是非空字符串数组"
                )
            normalized[document_type][key] = list(values)
    return normalized


@cache
def load_aggregate_document_types(
    config_path: str | Path = DOCUMENT_MAPPING_CONFIG_PATH,
) -> List[str]:
    configured = load_document_mapping_config(config_path).get(
        "aggregate_document_types",
        [],
    )
    if not isinstance(configured, list) or any(
        not isinstance(value, str) or not value.strip()
        for value in configured
    ):
        raise TypeError("aggregate_document_types 必须是非空字符串数组")
    return list(dict.fromkeys(configured))


def _iter_rule_records(value: Any) -> Iterable[Dict[str, Any]]:
    """递归遍历规则文件，不依赖顶层类别和子类别的具体组织方式。"""

    if isinstance(value, list):
        for item in value:
            yield from _iter_rule_records(item)
        return
    if not isinstance(value, dict):
        return
    if (
        "上下文无关审查事项" in value
        or "上下文相关审查事项" in value
    ):
        yield value
        return
    for item in value.values():
        yield from _iter_rule_records(item)


def _append_unique_text(
    target: List[str],
    value: Any,
    *,
    max_chars: int = 260,
) -> None:
    if not isinstance(value, str):
        return
    compact = " ".join(value.split())
    if not compact:
        return
    compact = compact[:max_chars]
    if compact not in target:
        target.append(compact)


@cache
def load_document_type_definition_catalog(
    rules_path: str | Path = RULES_PATH,
) -> Dict[str, Dict[str, List[str]]]:
    """从实际规则生成文书类别语义，避免用标题枚举定义类别边界。

    文书名称只是一条线索。审查目标说明该类别在规则里为何被使用，字段则说明
    它通常承载什么信息；二者共同供模型识别任意地方简称、旧称及无目录分段。
    """

    rules_data = read_json(rules_path)
    aliases = load_rule_aliases()
    catalog: Dict[str, Dict[str, List[str]]] = {}

    def definition(source_name: str) -> Dict[str, List[str]]:
        canonical_name = canonical_document_type(source_name, aliases)
        item = catalog.setdefault(
            canonical_name,
            {
                "rule_source_labels": [],
                "review_targets": [],
                "expected_fields": [],
            },
        )
        _append_unique_text(item["rule_source_labels"], source_name)
        _append_unique_text(item["rule_source_labels"], canonical_name)
        return item

    for rule in _iter_rule_records(rules_data):
        context_free = rule.get("上下文无关审查事项")
        if isinstance(context_free, dict):
            for source_name, review_item in context_free.items():
                if not isinstance(source_name, str) or not source_name.strip():
                    continue
                item = definition(source_name.strip())
                if isinstance(review_item, dict):
                    _append_unique_text(
                        item["review_targets"],
                        review_item.get("审查事项"),
                    )
                    _append_unique_text(
                        item["review_targets"],
                        review_item.get("评查说明"),
                    )

        context_related = rule.get("上下文相关审查事项")
        if not isinstance(context_related, list):
            continue
        for review_item in context_related:
            if not isinstance(review_item, dict):
                continue
            fields_by_document = review_item.get("字段")
            if not isinstance(fields_by_document, dict):
                continue
            for source_name, field_items in fields_by_document.items():
                if not isinstance(source_name, str) or not source_name.strip():
                    continue
                item = definition(source_name.strip())
                for key in ("任务", "审查事项", "评查说明", "评查类别"):
                    _append_unique_text(
                        item["review_targets"],
                        review_item.get(key),
                    )
                if isinstance(field_items, list):
                    for field_item in field_items:
                        if isinstance(field_item, dict):
                            _append_unique_text(
                                item["expected_fields"],
                                field_item.get("field"),
                                max_chars=100,
                            )

    # 限制单类提示长度，保留多来源语义而不把整份规则复制进模型上下文。
    for item in catalog.values():
        item["rule_source_labels"] = item["rule_source_labels"][:16]
        item["review_targets"] = item["review_targets"][:10]
        item["expected_fields"] = item["expected_fields"][:40]
    return catalog


def document_type_definitions(
    document_types: Iterable[str],
    rules_path: str | Path = RULES_PATH,
) -> Dict[str, Dict[str, List[str]]]:
    """返回本轮所需类别的规则语义；未知类别仍保留其规范名称。"""

    catalog = load_document_type_definition_catalog(rules_path)
    definitions: Dict[str, Dict[str, List[str]]] = {}
    for document_type in dict.fromkeys(document_types):
        definitions[document_type] = catalog.get(
            document_type,
            {
                "rule_source_labels": [document_type],
                "review_targets": [],
                "expected_fields": [],
            },
        )
    return definitions


def _compact_document_text(value: Any) -> str:
    return (
        "".join(str(value or "").split())
        .replace("(", "（")
        .replace(")", "）")
    )


_BRACKET_OPENERS = ("[", "【", "［")


def _strip_bracket_suffix(text: str) -> str:
    """去掉标题中方括号及其后的注释（如“送达回执[协助调查通知书]”→“送达回执”）。

    案卷目录常用方括号补充说明材料来源或被送达文书，方括号前的部分才是
    判断文书类型的核心名称，映射时只取方括号前的部分。
    """

    positions = [text.find(opener) for opener in _BRACKET_OPENERS]
    positions = [position for position in positions if position != -1]
    if not positions:
        return text
    return text[: min(positions)]


def _contains_any(text: str, values: Iterable[str]) -> bool:
    return any(_compact_document_text(value) in text for value in values)


def _is_receipt_section_name(section_name: str) -> bool:
    return _contains_any(
        _compact_document_text(section_name),
        RECEIPT_TITLE_KEYWORDS,
    )


def _section_matches_document_type(
    section: dict,
    document_type: str,
    matcher: Dict[str, List[str]],
    *,
    previous_section: dict | None = None,
) -> bool:
    section_name = _strip_bracket_suffix(str(section.get("section_name") or ""))
    title = _compact_document_text(
        " ".join(
            [
                section_name,
                str(section.get("normalized_document_type") or ""),
                str(section.get("material_type") or ""),
            ]
        )
    )
    if (
        document_type != RECEIPT_DOCUMENT_TYPE
        and _contains_any(title, RECEIPT_TITLE_KEYWORDS)
    ):
        return False
    material_type = _compact_document_text(section.get("material_type"))
    section_kind = str(section.get("section_kind") or "")
    required_kinds = matcher.get("section_kinds", [])
    if required_kinds and section_kind not in required_kinds:
        return False
    subject_roles = matcher.get("subject_roles", [])
    if subject_roles and str(section.get("subject_role") or "") not in subject_roles:
        return False
    contains_all = matcher.get("title_contains_all", [])
    if contains_all and not all(
        _compact_document_text(value) in title for value in contains_all
    ):
        return False
    contains_any = matcher.get("title_contains_any", [])
    if contains_any and not _contains_any(title, contains_any):
        return False
    if _contains_any(title, matcher.get("title_excludes_any", [])):
        return False
    material_contains = matcher.get("material_contains_any", [])
    if material_contains and not _contains_any(material_type, material_contains):
        return False
    if _contains_any(
        material_type,
        matcher.get("material_excludes_any", []),
    ):
        return False
    previous_title = _compact_document_text(
        (previous_section or {}).get("section_name")
    )
    previous_contains = matcher.get(
        "previous_section_title_contains_any",
        [],
    )
    if previous_contains and not _contains_any(
        previous_title,
        previous_contains,
    ):
        return False
    previous_excludes = matcher.get(
        "previous_section_title_excludes_any",
        [],
    )
    if previous_excludes and _contains_any(
        previous_title,
        previous_excludes,
    ):
        return False
    return bool(
        required_kinds
        or contains_all
        or contains_any
        or material_contains
        or previous_contains
    )


def deterministic_document_section_map(
    document_types: Iterable[str],
    dir_info: Iterable[dict],
) -> Dict[str, List[int]]:
    """Resolve high-confidence document aliases without a model call."""

    matchers = load_deterministic_document_matchers()
    directory = list(dir_info)
    resolved: Dict[str, List[int]] = {}
    for document_type in dict.fromkeys(document_types):
        matcher = matchers.get(document_type)
        if not matcher:
            continue
        section_ids = []
        for index, section in enumerate(directory):
            previous_section = directory[index - 1] if index > 0 else None
            if _section_matches_document_type(
                section,
                document_type,
                matcher,
                previous_section=previous_section,
            ):
                section_ids.append(int(section["section_id"]))
        if section_ids:
            resolved[document_type] = section_ids
    return resolved


def _ocr_text(item: Any) -> str:
    if isinstance(item, dict):
        for key in ("document_content", "text", "content"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(item, str):
        return item.strip()
    return ""


def directory_info_for_mapping(
    dir_info: Iterable[dict],
    ocr_results: Iterable[dict] | None = None,
) -> List[dict]:
    """返回用于文书映射的目录副本，section_name 去掉方括号后缀。

    模型映射时不应依据方括号内的注释（如“视听资料[取证记录]”里的“取证记录”）
    猜测文书类型，因此送入模型前统一改为方括号前的核心名称。
    """

    directory = list(dir_info)
    ocr_pages = list(ocr_results or [])
    stripped: List[dict] = []
    for index, item in enumerate(directory):
        copy = dict(item)
        name = str(copy.get("section_name") or "")
        mapping_name = _strip_bracket_suffix(name)
        if mapping_name and mapping_name != name:
            copy["section_name"] = mapping_name
        if ocr_pages:
            start_page = max(0, int(copy.get("section_page", 0)))
            if "section_end_page" in copy:
                end_page = min(
                    len(ocr_pages),
                    max(start_page + 1, int(copy["section_end_page"])),
                )
            elif index + 1 < len(directory):
                end_page = min(
                    len(ocr_pages),
                    max(
                        start_page + 1,
                        int(directory[index + 1].get("section_page", start_page + 1)),
                    ),
                )
            else:
                end_page = len(ocr_pages)
            hint = "\n".join(
                text
                for text in (
                    _ocr_text(ocr_pages[page_index])
                    for page_index in range(start_page, end_page)
                )
                if text
            )
            hint = " ".join(hint.split())
            if hint:
                copy["ocr_text_hint"] = hint[:MAPPING_OCR_HINT_MAX_CHARS]
        stripped.append(copy)
    return stripped


def document_types_may_share_section(
    document_types: Iterable[str],
) -> bool:
    requested_types = set(document_types)
    if len(requested_types) <= 1:
        return True
    compatible_groups = [
        set(compatible_types)
        for compatible_types in (
            load_compatible_document_type_groups().values()
        )
    ]
    # 一个 section 可能同时具有“具体文书 + 来源属性 + 证据形态”等
    # 多层标签。这些标签不必全部写进同一个超大组，但任意两项都必须在
    # 至少一个相容组中共同出现，才能组合共享，避免无关类型被连带放宽。
    return all(
        any({left, right} <= group for group in compatible_groups)
        for left, right in combinations(requested_types, 2)
    )


def normalize_document_section_map(
    document_section_map: Dict[str, List[int]],
    dir_info: Iterable[dict],
) -> Dict[str, List[int]]:
    """校验并规范化公共文书类型到 section_id 的映射。"""

    directory = list(dir_info)
    valid_section_ids = {int(item["section_id"]) for item in directory}
    section_order = {
        int(item["section_id"]): index
        for index, item in enumerate(directory)
    }
    section_names = {
        int(item["section_id"]): str(item.get("section_name") or "")
        for item in directory
    }
    normalized: Dict[str, List[int]] = {}
    section_owners: Dict[int, List[str]] = {}
    for document_type, section_ids in document_section_map.items():
        if not isinstance(document_type, str) or not document_type:
            raise TypeError("文书类型必须是非空字符串")
        if not isinstance(section_ids, list) or not section_ids:
            raise ValueError(
                f"已确认存在的文书未映射到章节: {document_type}"
            )
        if any(
            isinstance(section_id, bool)
            or not isinstance(section_id, int)
            or section_id <= 0
            for section_id in section_ids
        ):
            raise TypeError(
                f"文书映射必须使用正整数 section_id: {document_type}"
            )
        if not set(section_ids) <= valid_section_ids:
            raise ValueError(
                f"文书映射包含无效 section_id: {document_type}"
            )

        allowed_ids = [
            section_id
            for section_id in section_ids
            if (
                document_type == RECEIPT_DOCUMENT_TYPE
                or not _is_receipt_section_name(section_names[section_id])
            )
        ]
        if not allowed_ids:
            raise ValueError(
                f"已确认存在的文书未映射到有效章节: {document_type}"
            )
        ordered_ids = sorted(
            set(allowed_ids),
            key=section_order.__getitem__,
        )
        for section_id in ordered_ids:
            existing_owners = section_owners.setdefault(section_id, [])
            if (
                existing_owners
                and not document_types_may_share_section(
                    [*existing_owners, document_type]
                )
            ):
                raise ValueError(
                    f"section_id={section_id} 不能同时映射到文书类型 "
                    f"{existing_owners} 和 {document_type}；"
                    "只有同一相容类型组内的文书类型可以共享 section"
                )
            if document_type not in existing_owners:
                existing_owners.append(document_type)
        normalized[document_type] = ordered_ids
    return normalized


def normalize_document_section_map_lenient(
    document_section_map: Dict[str, List[int]],
    dir_info: Iterable[dict],
    *,
    preferred_document_types: Iterable[str] = (),
) -> tuple[Dict[str, List[int]], Dict[str, str]]:
    """Keep valid mappings and report invalid document types instead of failing."""

    preferred = list(dict.fromkeys(preferred_document_types))
    aggregate_types = set(load_aggregate_document_types())
    remaining_types = [
        name for name in document_section_map if name not in preferred
    ]
    ordered_types = [
        *[name for name in preferred if name in document_section_map],
        *[name for name in remaining_types if name not in aggregate_types],
        *[name for name in remaining_types if name in aggregate_types],
    ]
    normalized: Dict[str, List[int]] = {}
    dropped: Dict[str, str] = {}
    for document_type in ordered_types:
        raw_section_ids = document_section_map[document_type]
        if not isinstance(raw_section_ids, list) or not raw_section_ids:
            dropped[str(document_type)] = (
                f"ValueError: 已确认存在的文书未映射到章节: {document_type}"
            )
            continue

        accepted_ids: List[int] = []
        rejected_reasons: List[str] = []
        for section_id in dict.fromkeys(raw_section_ids):
            candidate = {
                **normalized,
                document_type: [*accepted_ids, section_id],
            }
            try:
                validated = normalize_document_section_map(
                    candidate,
                    dir_info,
                )
            except Exception as exc:
                rejected_reasons.append(
                    f"section_id={section_id}: {type(exc).__name__}: {exc}"
                )
                continue
            accepted_ids = validated[document_type]

        if accepted_ids:
            normalized = normalize_document_section_map(
                {
                    **normalized,
                    document_type: accepted_ids,
                },
                dir_info,
            )
            continue

        reason = "; ".join(rejected_reasons) or "没有可用 section_id"
        dropped[str(document_type)] = reason
    return normalized, dropped
