"""深圳市城市管理行政处罚裁量标准的中立检索核心。"""

from .catalog import (
    CityManagementDiscretionCatalog,
    CityManagementDiscretionCitationResult,
    CityManagementDiscretionRecord,
    compute_violation_text_fingerprint,
    get_default_catalog,
)
from .models import (
    DiscretionRetrievalMetadata,
    GeneratedItemCitations,
)
from .paths import (
    CATALOG_PATH,
    RERANK_DIRECTORY,
    RERANK_EMBEDDINGS_PATH,
    RERANK_MANIFEST_PATH,
)
from .reranker import (
    CityManagementDiscretionReranker,
    DiscretionRerankResult,
    ScoredDiscretionRecord,
    get_default_reranker,
    rank_scored_records,
)
from .routing import is_applicable_subdistrict_penalty_case

__all__ = [
    "CATALOG_PATH",
    "CityManagementDiscretionCatalog",
    "CityManagementDiscretionCitationResult",
    "CityManagementDiscretionRecord",
    "CityManagementDiscretionReranker",
    "DiscretionRerankResult",
    "DiscretionRetrievalMetadata",
    "GeneratedItemCitations",
    "RERANK_DIRECTORY",
    "RERANK_EMBEDDINGS_PATH",
    "RERANK_MANIFEST_PATH",
    "ScoredDiscretionRecord",
    "compute_violation_text_fingerprint",
    "get_default_catalog",
    "get_default_reranker",
    "is_applicable_subdistrict_penalty_case",
    "rank_scored_records",
]
