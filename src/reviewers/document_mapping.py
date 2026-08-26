from functools import cache
from pathlib import Path
from typing import Dict, Iterable, List

from constants import DOCUMENT_MAPPING_CONFIG_PATH
from utils import load_yaml


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
