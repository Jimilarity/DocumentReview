import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from knowledge_retrieval.city_management_discretion import (
    CityManagementDiscretionCitationResult,
    CityManagementDiscretionRecord,
    DiscretionRerankResult,
    ScoredDiscretionRecord,
)
from knowledge_retrieval.common import ImposedPenalty, LegalCitation
from human_support.executor import HumanSupportExecutor
from human_support.facts import (
    FactExtractionService,
    FactProviderRegistry,
)
from human_support.models import (
    FactCollection,
    HumanSupportContext,
    HumanSupportRetrieval,
    RetrievalStatus,
)
from human_support.modules.city_management_discretion import (
    CityManagementDiscretionCandidatesModule,
)
from human_support.registry import HumanSupportModuleRegistry
from human_support.result_coordinator import HumanSupportResultCoordinator
from human_support.service import HumanSupportService
from reviewers.base import ReviewSettings
from rules.rule_set import RuleSet


def citation(article: str = "66") -> LegalCitation:
    return LegalCitation(
        law_name="《深圳市生活垃圾分类管理条例》",
        article=article,
        paragraph=None,
        item=None,
        subitem=None,
        content=None,
    )


def record(item_id: str, sequence: int, behavior: str):
    source = {
        "id": item_id,
        "序号": sequence,
        "违法行为": behavior,
        "设定依据": "第六十六条……",
        "裁量分组": [
            {
                "分组序号": 1,
                "处罚种类及幅度": "处五十元罚款",
                "裁量规则": [],
            }
        ],
    }
    return CityManagementDiscretionRecord(
        item_id=item_id,
        sequence=sequence,
        citations=(citation(),),
        source_item=source,
    )


class FactExtractionServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_one_provider_is_called_once_for_multiple_facts(self) -> None:
        class Provider:
            provided_facts = frozenset({"fact_a", "fact_b"})

            def __init__(self):
                self.calls = 0

            async def extract(self, context):
                self.calls += 1
                return {"fact_a": "A", "fact_b": "B"}

        provider = Provider()
        registry = FactProviderRegistry()
        registry.register(provider)
        context = HumanSupportContext(
            metadata={},
            dir_info=[],
            rule={},
            review_item={},
            document_name="行政处罚决定书",
            section_id=1,
            section_ocr="OCR",
        )

        result = await FactExtractionService(registry).collect(
            {"fact_a", "fact_b"},
            context,
        )

        self.assertEqual(provider.calls, 1)
        self.assertEqual(result.values, {"fact_a": "A", "fact_b": "B"})
        self.assertEqual(result.errors, {})


class CityManagementDiscretionSupportTest(
    unittest.IsolatedAsyncioTestCase
):
    async def test_multiple_exact_candidates_are_returned_in_vector_order(
        self,
    ) -> None:
        first = record("first", 1, "行为一")
        second = record("second", 2, "行为二")
        lookup = CityManagementDiscretionCitationResult(
            citation=citation(),
            matches=(first, second),
            law_covered=True,
            article_covered=True,
        )

        class Catalog:
            violation_text_fingerprint = "fingerprint"

            def lookup_citations(self, citations):
                return [lookup]

        class Reranker:
            def rank(self, *args, **kwargs):
                return DiscretionRerankResult(
                    ranked=(
                        ScoredDiscretionRecord(
                            record=second,
                            similarity=0.91,
                        ),
                        ScoredDiscretionRecord(
                            record=first,
                            similarity=0.72,
                        ),
                    ),
                )

        context = HumanSupportContext(
            metadata={
                "案号": "深龙华龙华综行简罚决字第0100号",
                "案由": "未按规定分类投放生活垃圾案",
            },
            dir_info=[],
            rule={"序号": 360},
            review_item={},
            document_name="行政处罚决定书",
            section_id=1,
            section_ocr="OCR",
        )
        penalty = ImposedPenalty(
            penalty_type="罚款",
            target="个人",
            amount_yuan=50,
            content="决定处五十元罚款。",
        )
        facts = {
            "penalty_legal_citations": [citation()],
            "imposed_penalties": [penalty],
            "case_reason": "未按规定分类投放生活垃圾案",
        }

        with patch(
            "human_support.modules.city_management_discretion."
            "get_default_catalog",
            return_value=Catalog(),
        ), patch(
            "human_support.modules.city_management_discretion."
            "get_default_reranker",
            return_value=Reranker(),
        ):
            result = await CityManagementDiscretionCandidatesModule().retrieve(
                context,
                facts,
            )

        self.assertEqual(result.status, RetrievalStatus.MATCHED)
        search = result.payload["逐法条检索"][0]
        self.assertEqual(
            [item["candidate_id"] for item in search["候选"]],
            ["second", "first"],
        )
        self.assertEqual(search["候选总数"], 2)
        self.assertIn("裁量分组", search["候选"][0])
        self.assertNotIn("文书实际处罚", result.payload)

    async def test_single_exact_candidate_also_gets_similarity(
        self,
    ) -> None:
        only = record("only", 1, "未按规定分类投放生活垃圾")
        lookup = CityManagementDiscretionCitationResult(
            citation=citation(),
            matches=(only,),
            law_covered=True,
            article_covered=True,
        )

        class Catalog:
            violation_text_fingerprint = "fingerprint"

            def lookup_citations(self, citations):
                return [lookup]

        class Reranker:
            def __init__(self):
                self.calls = 0

            def rank(self, *args, **kwargs):
                self.calls += 1
                scored = ScoredDiscretionRecord(
                    record=only,
                    similarity=0.88,
                )
                return DiscretionRerankResult(
                    ranked=(scored,),
                )

        reranker = Reranker()
        context = HumanSupportContext(
            metadata={
                "案号": "深龙华龙华综行简罚决字第0100号",
                "案由": "未按规定分类投放生活垃圾案",
            },
            dir_info=[],
            rule={"序号": 360},
            review_item={},
            document_name="行政处罚决定书",
            section_id=1,
            section_ocr="OCR",
        )
        with patch(
            "human_support.modules.city_management_discretion."
            "get_default_catalog",
            return_value=Catalog(),
        ), patch(
            "human_support.modules.city_management_discretion."
            "get_default_reranker",
            return_value=reranker,
        ):
            result = await CityManagementDiscretionCandidatesModule().retrieve(
                context,
                {
                    "penalty_legal_citations": [citation()],
                    "imposed_penalties": [],
                    "case_reason": "未按规定分类投放生活垃圾案",
                },
            )

        self.assertEqual(reranker.calls, 1)
        candidate = result.payload["逐法条检索"][0]["候选"][0]
        self.assertEqual(candidate["相似度"], 0.88)

    async def test_non_subdistrict_case_is_skipped_before_fact_extraction(
        self,
    ) -> None:
        module = CityManagementDiscretionCandidatesModule()
        context = HumanSupportContext(
            metadata={"案号": "深龙华应急罚字第1号"},
            dir_info=[],
            rule={"序号": 360},
            review_item={},
            document_name="行政处罚决定书",
            section_id=1,
            section_ocr="OCR",
        )

        self.assertEqual(module.required_facts(context), set())
        result = await module.retrieve(context, {})

        self.assertEqual(result.status, RetrievalStatus.SKIPPED)


class HumanSupportExecutorTest(unittest.IsolatedAsyncioTestCase):
    async def test_same_section_extracts_shared_facts_once(self) -> None:
        class Module:
            def __init__(self, name):
                self.name = name

            def required_facts(self, context):
                return {"shared_fact"}

            async def retrieve(self, context, facts):
                return HumanSupportRetrieval(
                    knowledge_name=self.name,
                    status=RetrievalStatus.MATCHED,
                    payload={"value": facts["shared_fact"]},
                )

        registry = HumanSupportModuleRegistry()
        registry.register(Module("module_a"))
        registry.register(Module("module_b"))
        support_service = HumanSupportService(registry)
        fact_service = AsyncMock()
        fact_service.collect.return_value = FactCollection(
            values={"shared_fact": "一次提取"},
        )
        rules = RuleSet(
            rules=[
                {
                    "序号": 360,
                    "上下文无关审查事项": {
                        "行政处罚决定书": {"审查事项": "A"}
                    },
                    "检索增强": ["module_a"],
                },
                {
                    "序号": 361,
                    "上下文无关审查事项": {
                        "行政处罚决定书": {"审查事项": "B"}
                    },
                    "检索增强": ["module_b"],
                },
            ]
        )
        executor = HumanSupportExecutor(
            "case.pdf",
            rules,
            document_section_map={"行政处罚决定书": [1]},
            fact_service=fact_service,
            support_service=support_service,
        )
        executor._load_context = AsyncMock(
            return_value=(
                {},
                [
                    {
                        "section_id": 1,
                        "section_page": 0,
                        "section_name": "行政处罚决定书",
                    }
                ],
                [{"image_index": 0, "document_content": "OCR"}],
            )
        )

        result = await executor.execute()

        fact_service.collect.assert_awaited_once()
        self.assertEqual([item["rule_index"] for item in result], [360, 361])
        self.assertEqual(
            set(result[0]),
            {"rule_index", "documents"},
        )
        self.assertEqual(
            result[0]["documents"][0]["extracted_facts"],
            {"shared_fact": "一次提取"},
        )
        self.assertNotIn("warnings", result[0]["documents"][0])

    def test_special_results_are_written_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = ReviewSettings(result_root=Path(directory))
            coordinator = HumanSupportResultCoordinator(
                "case.pdf",
                settings=settings,
            )
            status = coordinator.finalize(
                [
                    {
                        "rule_index": 360,
                        "documents": [],
                    }
                ]
            )
            result_path = Path(status["result_path"])
            self.assertTrue(result_path.is_file())
            self.assertEqual(
                result_path.name,
                "retrieval_enhancement_results.json",
            )


if __name__ == "__main__":
    unittest.main()
