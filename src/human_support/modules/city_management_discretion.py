import asyncio
from typing import Any

from knowledge_retrieval.city_management_discretion import (
    CityManagementDiscretionRecord,
    get_default_catalog,
    get_default_reranker,
    is_applicable_subdistrict_penalty_case,
)
from knowledge_retrieval.common import LegalCitation

from ..facts import (
    CASE_REASON_FACT,
    IMPOSED_PENALTIES_FACT,
    PENALTY_LEGAL_CITATIONS_FACT,
)
from ..models import (
    HumanSupportContext,
    HumanSupportRetrieval,
    RetrievalStatus,
)
from ..registry import human_support_module_registry


MAX_RETURNED_CANDIDATES = 5


def _citation_payload(citation: LegalCitation) -> dict[str, Any]:
    return citation.model_dump()


def _candidate_payload(
    record: CityManagementDiscretionRecord,
    *,
    similarity: float | None,
) -> dict[str, Any]:
    source = record.source_item
    return {
        "candidate_id": record.item_id,
        "序号": record.sequence,
        "相似度": similarity,
        "处罚依据": [
            _citation_payload(citation) for citation in record.citations
        ],
        "违法行为": source.get("违法行为"),
        "设定依据": source.get("设定依据"),
        "裁量分组": source.get("裁量分组") or [],
    }


async def _rank_candidates(
    candidates: tuple[CityManagementDiscretionRecord, ...],
    case_reason: str | None,
) -> list[tuple[CityManagementDiscretionRecord, float | None]]:
    if not candidates or not case_reason:
        return [(record, None) for record in candidates]

    catalog = get_default_catalog()
    result = await asyncio.to_thread(
        get_default_reranker().rank,
        case_reason,
        list(candidates),
        violation_text_fingerprint=catalog.violation_text_fingerprint,
    )
    ranked = [
        (item.record, item.similarity)
        for item in result.ranked
    ]
    return ranked


class CityManagementDiscretionCandidatesModule:
    """检索裁量标准候选，只提供证据，不作裁量档次判断。"""

    name = "city_management_discretion_candidates"

    def required_facts(
        self,
        context: HumanSupportContext,
    ) -> set[str]:
        if not is_applicable_subdistrict_penalty_case(
            context.metadata.get("案号")
        ):
            return set()
        return {
            PENALTY_LEGAL_CITATIONS_FACT,
            IMPOSED_PENALTIES_FACT,
            CASE_REASON_FACT,
        }

    async def retrieve(
        self,
        context: HumanSupportContext,
        facts: dict[str, object],
    ) -> HumanSupportRetrieval:
        if not is_applicable_subdistrict_penalty_case(
            context.metadata.get("案号")
        ):
            return HumanSupportRetrieval(
                knowledge_name=self.name,
                status=RetrievalStatus.SKIPPED,
                payload={
                    "reason": (
                        "not_subdistrict_comprehensive_enforcement_case"
                    )
                },
            )

        citations = facts.get(PENALTY_LEGAL_CITATIONS_FACT) or []
        if not citations:
            return HumanSupportRetrieval(
                knowledge_name=self.name,
                status=RetrievalStatus.SKIPPED,
                payload={"reason": "no_penalty_legal_citation_extracted"},
            )
        typed_citations = [
            item for item in citations if isinstance(item, LegalCitation)
        ]
        if not typed_citations:
            return HumanSupportRetrieval(
                knowledge_name=self.name,
                status=RetrievalStatus.ERROR,
                payload={"reason": "invalid_penalty_legal_citation_facts"},
            )

        case_reason_value = facts.get(CASE_REASON_FACT)
        case_reason = (
            case_reason_value
            if isinstance(case_reason_value, str) and case_reason_value
            else None
        )
        catalog = get_default_catalog()
        lookup_results = await asyncio.to_thread(
            catalog.lookup_citations,
            typed_citations,
        )
        catalog_error = getattr(catalog, "load_error", None)
        if catalog_error:
            return HumanSupportRetrieval(
                knowledge_name=self.name,
                status=RetrievalStatus.ERROR,
                payload={
                    "reason": "discretion_catalog_unavailable",
                    "error": catalog_error,
                },
            )
        citation_payloads: list[dict[str, Any]] = []
        has_match = False
        for lookup in lookup_results:
            ranked = await _rank_candidates(
                lookup.matches,
                case_reason,
            )
            limited = ranked[
                :MAX_RETURNED_CANDIDATES
            ]
            candidates = [
                _candidate_payload(record, similarity=similarity)
                for record, similarity in limited
            ]
            has_match = has_match or bool(candidates)
            if candidates:
                coverage_status = "matched"
            elif lookup.law_covered and not lookup.article_covered:
                coverage_status = "law_covered_article_not_listed"
            else:
                coverage_status = "law_not_covered"
            citation_payloads.append(
                {
                    "文书处罚依据": _citation_payload(lookup.citation),
                    "检索状态": coverage_status,
                    "候选总数": len(lookup.matches),
                    "候选": candidates,
                }
            )

        return HumanSupportRetrieval(
            knowledge_name=self.name,
            status=(
                RetrievalStatus.MATCHED
                if has_match
                else RetrievalStatus.NO_MATCH
            ),
            payload={
                "知识来源": (
                    "《深圳市城市管理行政处罚裁量权实施标准"
                    "（2023年版）》"
                ),
                "逐法条检索": citation_payloads,
            },
        )


human_support_module_registry.register(
    CityManagementDiscretionCandidatesModule()
)
