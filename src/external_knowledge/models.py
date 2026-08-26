from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class KnowledgeItem:
    """可注入一次审查事务的最小知识单元。"""

    content: str


@dataclass(frozen=True)
class KnowledgeContext:
    """知识函数的统一输入；每个函数自行决定使用其中哪些信息。"""

    metadata: Dict[str, Any]
    dir_info: List[Dict[str, Any]]
    rule: Dict[str, Any]
    review_item: Dict[str, Any]
    document_name: str
    section_id: int
    section_ocr: str
    structured_fields: List[Dict[str, Any]] = field(default_factory=list)
