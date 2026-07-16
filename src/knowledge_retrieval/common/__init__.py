"""中立的结构化事实提取与精确检索协议。"""

from .legal_citations import (
    CitationRecord,
    CitationScope,
    ExactLegalCitationLookup,
    ImposedPenalty,
    LegalCitation,
    LegalCitationExtraction,
    NonEmptyText,
    StrictModel,
    citation_matches,
    clear_legal_citation_extraction_cache,
    extract_legal_citations,
)

__all__ = [
    "CitationRecord",
    "CitationScope",
    "ExactLegalCitationLookup",
    "ImposedPenalty",
    "LegalCitation",
    "LegalCitationExtraction",
    "NonEmptyText",
    "StrictModel",
    "citation_matches",
    "clear_legal_citation_extraction_cache",
    "extract_legal_citations",
]
