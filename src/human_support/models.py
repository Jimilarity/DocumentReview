from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class RetrievalStatus(str, Enum):
    MATCHED = "matched"
    NO_MATCH = "no_match"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass(frozen=True)
class HumanSupportContext:
    """人工辅助模块的统一逐文书输入。"""

    metadata: Dict[str, Any]
    dir_info: List[Dict[str, Any]]
    rule: Dict[str, Any]
    review_item: Dict[str, Any]
    document_name: str
    section_id: int
    section_ocr: str


@dataclass(frozen=True)
class HumanSupportRetrieval:
    """稳定的薄信封；payload 由各知识模块自行定义。"""

    knowledge_name: str
    status: RetrievalStatus
    payload: Dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "knowledge_name": self.knowledge_name,
            "status": self.status.value,
            "payload": self.payload,
        }
        if self.warnings:
            result["warnings"] = list(self.warnings)
        return result


@dataclass(frozen=True)
class FactCollection:
    values: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
