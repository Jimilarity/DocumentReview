from functools import cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

from constants import DOCUMENT_MAPPING_CONFIG_PATH
from utils import load_yaml


RECEIPT_DOCUMENT_TYPE = "送达回证"
# 目录标题只要出现以下任一关键词即视为送达回证/送达回执；其方括号内的被送达
# 文书名称（如“送达回执[责令改正违法行为通知书]”）仅用于标识回证所服务的
# 文书，不应使该 section 被映射到被送达文书本身。
RECEIPT_TITLE_KEYWORDS = ("送达回证", "送达回执")


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
def load_excluded_section_title_keywords(
    config_path: str | Path = DOCUMENT_MAPPING_CONFIG_PATH,
) -> Dict[str, List[str]]:
    configured = load_document_mapping_config(config_path).get(
        "excluded_section_title_keywords",
        {},
    )
    if not isinstance(configured, dict):
        raise TypeError("excluded_section_title_keywords 必须是对象")
    return configured


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
    return bool(required_kinds or contains_all or contains_any or material_contains)


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
        section_ids = [
            int(section["section_id"])
            for section in directory
            if _section_matches_document_type(section, document_type, matcher)
        ]
        if section_ids:
            resolved[document_type] = section_ids
    return resolved


def directory_info_for_mapping(dir_info: Iterable[dict]) -> List[dict]:
    """返回用于文书映射的目录副本，section_name 去掉方括号后缀。

    模型映射时不应依据方括号内的注释（如“视听资料[取证记录]”里的“取证记录”）
    猜测文书类型，因此送入模型前统一改为方括号前的核心名称。
    """

    stripped: List[dict] = []
    for item in dir_info:
        copy = dict(item)
        name = str(copy.get("section_name") or "")
        mapping_name = _strip_bracket_suffix(name)
        if mapping_name and mapping_name != name:
            copy["section_name"] = mapping_name
        stripped.append(copy)
    return stripped


def document_types_may_share_section(
    document_types: Iterable[str],
) -> bool:
    requested_types = set(document_types)
    if len(requested_types) <= 1:
        return True
    return any(
        requested_types <= set(compatible_types)
        for compatible_types in (
            load_compatible_document_type_groups().values()
        )
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
    excluded_keywords = load_excluded_section_title_keywords()
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
            if not any(
                keyword in section_names[section_id]
                for keyword in excluded_keywords.get(document_type, [])
            )
            and (
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
    ordered_types = [
        *[name for name in preferred if name in document_section_map],
        *[name for name in document_section_map if name not in preferred],
    ]
    normalized: Dict[str, List[int]] = {}
    dropped: Dict[str, str] = {}
    for document_type in ordered_types:
        candidate = {
            **normalized,
            document_type: document_section_map[document_type],
        }
        try:
            normalized = normalize_document_section_map(candidate, dir_info)
        except Exception as exc:
            dropped[str(document_type)] = f"{type(exc).__name__}: {exc}"
    return normalized, dropped
