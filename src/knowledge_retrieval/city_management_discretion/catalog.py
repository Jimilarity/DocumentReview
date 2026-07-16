import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from ..common import (
    CitationRecord,
    ExactLegalCitationLookup,
    LegalCitation,
)
from .paths import CATALOG_PATH


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CityManagementDiscretionRecord:
    item_id: str
    sequence: int
    citations: tuple[LegalCitation, ...]
    source_item: dict[str, Any]


@dataclass(frozen=True)
class CityManagementDiscretionCitationResult:
    citation: LegalCitation
    matches: tuple[CityManagementDiscretionRecord, ...]
    law_covered: bool
    article_covered: bool


def compute_violation_text_fingerprint(
    source_items: list[dict[str, Any]],
) -> str:
    payload = [
        {
            "item_id": item.get("id"),
            "违法行为": item.get("违法行为"),
        }
        for item in source_items
    ]
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class CityManagementDiscretionCatalog:
    """加载完整裁量标准，并提供逐法条精确检索。"""

    def __init__(self, catalog_path: Path = CATALOG_PATH) -> None:
        self.catalog_path = Path(catalog_path)
        self._loaded = False
        self._load_lock = Lock()
        self._lookup: (
            ExactLegalCitationLookup[CityManagementDiscretionRecord] | None
        ) = None
        self._violation_text_fingerprint: str | None = None
        self._load_error: str | None = None

    def _load(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            records: list[CityManagementDiscretionRecord] = []
            source_items: list[dict[str, Any]] = []
            try:
                payload = json.loads(
                    self.catalog_path.read_text(encoding="utf-8")
                )
                raw_items = payload.get("items") or []
                source_items = [
                    item for item in raw_items if isinstance(item, dict)
                ]
                for position, source_item in enumerate(
                    source_items,
                    start=1,
                ):
                    metadata = source_item.get("检索元数据") or {}
                    raw_citations = metadata.get("处罚依据") or []
                    citations = tuple(
                        LegalCitation.model_construct(
                            law_name=citation.get("law_name"),
                            article=citation.get("article"),
                            paragraph=citation.get("paragraph"),
                            item=citation.get("item"),
                            subitem=citation.get("subitem"),
                            content=citation.get("content"),
                        )
                        for citation in raw_citations
                        if isinstance(citation, dict)
                        and citation.get("law_name")
                        and citation.get("article")
                    )
                    if not citations:
                        continue
                    records.append(
                        CityManagementDiscretionRecord(
                            item_id=str(
                                source_item.get("id")
                                or f"catalog-item-{position}"
                            ),
                            sequence=(
                                source_item.get("序号")
                                if isinstance(
                                    source_item.get("序号"),
                                    int,
                                )
                                else position
                            ),
                            citations=citations,
                            source_item=source_item,
                        )
                    )
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "city management discretion catalog unavailable; "
                    "skipped error=%s: %s",
                    type(exc).__name__,
                    exc,
                )

            self._lookup = ExactLegalCitationLookup(
                CitationRecord(
                    record_id=record.item_id,
                    citations=record.citations,
                    value=record,
                )
                for record in records
            )
            self._violation_text_fingerprint = (
                compute_violation_text_fingerprint(source_items)
            )
            self._loaded = True

    def search(
        self,
        citations: list[LegalCitation],
    ) -> list[CityManagementDiscretionRecord]:
        self._load()
        if self._lookup is None:
            return []
        return self._lookup.search(citations)

    def lookup_citations(
        self,
        citations: list[LegalCitation],
    ) -> list[CityManagementDiscretionCitationResult]:
        """逐法条返回候选及法规、条号的目录覆盖状态。"""

        self._load()
        if self._lookup is None:
            return []

        results: list[CityManagementDiscretionCitationResult] = []
        seen: set[
            tuple[str, str, str | None, str | None, str | None]
        ] = set()
        for citation in citations:
            key = (
                citation.law_name,
                citation.article,
                citation.paragraph,
                citation.item,
                citation.subitem,
            )
            if key in seen:
                continue
            seen.add(key)
            results.append(
                CityManagementDiscretionCitationResult(
                    citation=citation,
                    matches=tuple(self._lookup.search([citation])),
                    law_covered=self._lookup.contains_law(
                        citation.law_name
                    ),
                    article_covered=self._lookup.contains_article(
                        citation.law_name,
                        citation.article,
                    ),
                )
            )
        return results

    @property
    def violation_text_fingerprint(self) -> str:
        self._load()
        return self._violation_text_fingerprint or ""

    @property
    def load_error(self) -> str | None:
        self._load()
        return self._load_error


_default_catalog: CityManagementDiscretionCatalog | None = None
_default_catalog_lock = Lock()


def get_default_catalog() -> CityManagementDiscretionCatalog:
    global _default_catalog
    if _default_catalog is None:
        with _default_catalog_lock:
            if _default_catalog is None:
                _default_catalog = CityManagementDiscretionCatalog()
    return _default_catalog
