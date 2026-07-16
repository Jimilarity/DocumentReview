from pathlib import Path
from typing import Any, Dict, List, TypedDict

from constants import (
    CASE_LEVEL_CONFIG_PATH,
    CONTEXT_SENSITIVE_CONFIG_PATH,
    REVIEW_PIPELINE_CONFIG_PATH,
    SECTION_FIELDS_PATH,
    ReviewExecutorType,
    SupportExecutorType,
)
from utils import load_yaml


class ContextSensitiveSettings(TypedDict):
    prewarm_fields: Dict[str, List[str]]
    field_specs: Dict[str, Dict[str, Dict[str, Any]]]
    service_receipt_document_type: str


def load_enabled_executor_types(
    path: str | Path = REVIEW_PIPELINE_CONFIG_PATH,
) -> List[ReviewExecutorType]:
    configured = load_yaml(path)["executors"]
    return list(
        dict.fromkeys(ReviewExecutorType(value) for value in configured)
    )


def load_enabled_support_executor_types(
    path: str | Path = REVIEW_PIPELINE_CONFIG_PATH,
) -> List[SupportExecutorType]:
    configured = load_yaml(path).get("support_executors") or []
    return list(
        dict.fromkeys(SupportExecutorType(value) for value in configured)
    )


def load_case_level_review_items(
    path: str | Path = CASE_LEVEL_CONFIG_PATH,
) -> List[Dict[str, Any]]:
    return list(load_yaml(path)["案件级审查事项"])


def load_context_sensitive_settings(
    config_path: str | Path = CONTEXT_SENSITIVE_CONFIG_PATH,
    section_fields_path: str | Path = SECTION_FIELDS_PATH,
) -> ContextSensitiveSettings:
    configured = load_yaml(config_path)
    field_specs = load_yaml(section_fields_path)
    prewarm_fields = {
        document_type: (
            list(field_specs[document_type])
            if field_names == "*"
            else list(field_names)
        )
        for document_type, field_names in configured["prewarm_fields"].items()
    }
    return {
        "prewarm_fields": prewarm_fields,
        "field_specs": field_specs,
        "service_receipt_document_type": configured["service_receipt"][
            "document_type"
        ],
    }
