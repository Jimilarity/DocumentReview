import json
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from constants import PROJECT_ROOT

from .catalog import CityManagementDiscretionRecord
from .paths import RERANK_EMBEDDINGS_PATH, RERANK_MANIFEST_PATH


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoredDiscretionRecord:
    record: CityManagementDiscretionRecord
    similarity: float


@dataclass(frozen=True)
class DiscretionRerankResult:
    ranked: tuple[ScoredDiscretionRecord, ...]


def rank_scored_records(
    scored_records: list[ScoredDiscretionRecord],
) -> DiscretionRerankResult:
    return DiscretionRerankResult(
        ranked=tuple(
            sorted(
                scored_records,
                key=lambda item: item.similarity,
                reverse=True,
            )
        )
    )


class CityManagementDiscretionReranker:
    """在精确法条候选内，按案由与违法行为的相似度排序。"""

    def __init__(
        self,
        embeddings_path: Path = RERANK_EMBEDDINGS_PATH,
        manifest_path: Path = RERANK_MANIFEST_PATH,
    ) -> None:
        self.embeddings_path = Path(embeddings_path)
        self.manifest_path = Path(manifest_path)
        self._loaded = False
        self._load_lock = Lock()
        self._embeddings: Any = None
        self._row_by_item_id: dict[str, int] = {}
        self._model: Any = None
        self._query_prefix = ""
        self._violation_text_fingerprint = ""

    def _load(self) -> bool:
        if self._loaded:
            return True
        with self._load_lock:
            if self._loaded:
                return True
            try:
                import numpy as np
                from sentence_transformers import SentenceTransformer

                manifest = json.loads(
                    self.manifest_path.read_text(encoding="utf-8")
                )
                item_ids = manifest.get("item_ids") or []
                self._embeddings = np.load(self.embeddings_path)
                if not item_ids or len(self._embeddings) != len(item_ids):
                    logger.warning(
                        "city management discretion vector data mismatch; "
                        "keeping exact candidates"
                    )
                    return False

                model_reference = manifest.get("model_reference") or {}
                reference_value = model_reference.get("value")
                reference_kind = model_reference.get("kind")
                if reference_kind == "project_relative" and reference_value:
                    model_source: str | Path = (
                        PROJECT_ROOT / reference_value
                    )
                elif reference_kind == "model_id" and reference_value:
                    model_source = reference_value
                else:
                    logger.warning(
                        "city management discretion vector model is "
                        "unavailable; keeping exact candidates"
                    )
                    return False

                self._row_by_item_id = {
                    item_id: row
                    for row, item_id in enumerate(item_ids)
                }
                self._model = SentenceTransformer(str(model_source))
                self._query_prefix = str(
                    manifest.get("query_prefix") or ""
                )
                self._violation_text_fingerprint = str(
                    manifest.get("violation_text_fingerprint") or ""
                )
                self._loaded = True
                return True
            except Exception as exc:
                logger.warning(
                    "city management discretion rerank unavailable; "
                    "keeping exact candidates error=%s: %s",
                    type(exc).__name__,
                    exc,
                )
                return False

    def rank(
        self,
        case_reason: str,
        candidates: list[CityManagementDiscretionRecord],
        *,
        violation_text_fingerprint: str,
    ) -> DiscretionRerankResult:
        unranked = tuple(
            ScoredDiscretionRecord(record=record, similarity=0.0)
            for record in candidates
        )
        fallback = DiscretionRerankResult(
            ranked=unranked,
        )
        try:
            if not self._load():
                return fallback
            if (
                self._violation_text_fingerprint
                != violation_text_fingerprint
            ):
                logger.warning(
                    "city management discretion vectors are stale; "
                    "keeping exact candidates"
                )
                return fallback
            if any(
                record.item_id not in self._row_by_item_id
                for record in candidates
            ):
                logger.warning(
                    "city management discretion vectors miss candidates; "
                    "keeping exact candidates"
                )
                return fallback
            query_embedding = self._model.encode(
                [f"{self._query_prefix}{case_reason}"],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )[0]
            scored = [
                ScoredDiscretionRecord(
                    record=record,
                    similarity=float(
                        self._embeddings[
                            self._row_by_item_id[record.item_id]
                        ]
                        @ query_embedding
                    ),
                )
                for record in candidates
            ]
            return rank_scored_records(scored)
        except Exception as exc:
            logger.warning(
                "city management discretion rerank unavailable; "
                "keeping exact candidates error=%s: %s",
                type(exc).__name__,
                exc,
            )
            return fallback


_default_reranker: CityManagementDiscretionReranker | None = None
_default_reranker_lock = Lock()


def get_default_reranker() -> CityManagementDiscretionReranker:
    global _default_reranker
    if _default_reranker is None:
        with _default_reranker_lock:
            if _default_reranker is None:
                _default_reranker = CityManagementDiscretionReranker()
    return _default_reranker
