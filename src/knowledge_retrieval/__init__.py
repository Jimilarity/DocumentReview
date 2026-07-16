"""独立于提示词注入和结果展示方式的知识检索核心。"""

from .case_routing import PenaltyCaseKind, classify_penalty_case
from .common import (
    CitationRecord,
    CitationScope,
    ExactLegalCitationLookup,
    ImposedPenalty,
    LegalCitation,
    LegalCitationExtraction,
    extract_legal_citations,
)

__all__ = [
    "CitationRecord",
    "CitationScope",
    "ExactLegalCitationLookup",
    "ImposedPenalty",
    "LegalCitation",
    "LegalCitationExtraction",
    "PenaltyCaseKind",
    "classify_penalty_case",
    "extract_legal_citations",
]
