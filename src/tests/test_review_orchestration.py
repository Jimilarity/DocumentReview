import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import review
from constants import ReviewExecutorType, SupportExecutorType
from review import run_review
from reviewers.base import BaseReviewExecutor, DocumentReviewExecutor
from reviewers.case_level import CaseLevelReviewExecutor
from reviewers.context_free import ContextFreeReviewExecutor
from reviewers.context_sensitive import ContextSensitiveReviewExecutor
from review_config import (
    load_enabled_executor_types,
    load_enabled_support_executor_types,
)
from rules.rule_set import RuleSet
from structured_field_cache import StructuredFieldCache


class ReviewOrchestrationTest(unittest.IsolatedAsyncioTestCase):
    def test_context_free_receipt_uses_only_sections_linked_to_rule_document(self) -> None:
        executor = ContextFreeReviewExecutor.__new__(ContextFreeReviewExecutor)
        executor.document_section_map = {
            "行政处罚决定书": [3],
            "送达回证": [10, 11],
        }
        executor.structured_field_cache = SimpleNamespace(
            delivery_events=lambda section_id: {
                10: [{"related_section_id": 3}],
                11: [{"related_section_id": 8}],
            }[section_id]
        )
        available_documents = {
            "行政处罚决定书": {"审查事项": "决定书"},
            "送达回证": {"审查事项": "回证"},
        }

        section_ids = executor._receipt_section_ids_for_rule(
            {"上下文无关审查事项": available_documents},
            available_documents,
        )

        self.assertEqual(section_ids, [10])

    def test_context_free_receipt_skips_unmapped_target_document(self) -> None:
        executor = ContextFreeReviewExecutor.__new__(ContextFreeReviewExecutor)
        executor.document_section_map = {
            "行政处罚决定书": [3, 8],
            "送达回证": [10, 11],
        }
        executor.structured_field_cache = SimpleNamespace(
            delivery_events=lambda section_id: {
                10: [{"related_section_id": 3}],
                11: [{"related_section_id": None}],
            }[section_id]
        )
        available_documents = {
            "行政处罚决定书": {"审查事项": "决定书"},
            "送达回证": {"审查事项": "回证"},
        }

        section_ids = executor._receipt_section_ids_for_rule(
            {"上下文无关审查事项": available_documents},
            available_documents,
        )

        self.assertEqual(section_ids, [10])

    def test_executor_hierarchy_is_symmetric(self) -> None:
        self.assertTrue(issubclass(ContextFreeReviewExecutor, DocumentReviewExecutor))
        self.assertTrue(
            issubclass(ContextSensitiveReviewExecutor, DocumentReviewExecutor)
        )
        self.assertFalse(
            issubclass(ContextSensitiveReviewExecutor, ContextFreeReviewExecutor)
        )
        self.assertTrue(issubclass(CaseLevelReviewExecutor, BaseReviewExecutor))

    def test_document_prompt_omits_irrelevant_rule_metadata(self) -> None:
        effective = ContextFreeReviewExecutor.build_effective_rule(
            {"备注": ""},
            {"审查事项": "审查决定书", "评查说明": ""},
        )
        self.assertEqual(effective, {"审查事项": "审查决定书", "评查说明": ""})

    def test_presence_is_shared_but_rule_sets_use_distinct_documents(self) -> None:
        hybrid = {
            "序号": 101,
            "备注": "",
            "上下文无关审查事项": {
                "行政处罚决定书": {"审查事项": "检查决定书"}
            },
            "上下文相关审查事项": [
                {
                    "任务": "一致性核查",
                    "字段": {
                        "立案审批表": [
                            {"field": "执法主体名称", "required": True}
                        ]
                    },
                }
            ],
        }
        rules_data = {
            "合法性标准": {
                "通用": [hybrid],
                "行政处罚": [],
            },
            "规范性标准": {},
            "附加项": {"通用": []},
        }
        paths = SimpleNamespace(
            metadata="metadata",
            directory="directory",
            ocr_results="ocr",
        )

        def read_json(path):
            if path == "directory":
                return []
            if path == "ocr":
                return []
            if path == "metadata":
                return {}
            return rules_data

        with patch("review.get_cache_paths", return_value=paths), patch(
            "review.read_json", side_effect=read_json
        ), patch(
            "review.normalize_directory_info", return_value=[]
        ), patch(
            "review._load_structured_cache", return_value=object()
        ), patch(
            "review._resolve_document_presence",
            return_value={
                "行政处罚决定书": True,
                "立案审批表": True,
            },
        ) as resolve_presence, patch(
            "review.load_context_sensitive_settings",
            return_value={"prewarm_fields": {}},
        ):
            rule_sets, _ = review._build_document_rule_sets(
                "case.pdf",
                0b10010000,
                [
                    ReviewExecutorType.CONTEXT_FREE,
                    ReviewExecutorType.CONTEXT_SENSITIVE,
                ],
            )

        self.assertEqual(
            resolve_presence.call_args.args[1],
            ["行政处罚决定书", "立案审批表"],
        )
        self.assertEqual(
            set(
                rule_sets[ReviewExecutorType.CONTEXT_FREE]
                .rules[0]["上下文无关审查事项"]
            ),
            {"行政处罚决定书"},
        )
        self.assertEqual(
            set(
                rule_sets[ReviewExecutorType.CONTEXT_SENSITIVE]
                .rules[0]["上下文相关审查事项"][0]["字段"]
            ),
            {"立案审批表"},
        )

    def test_expanded_document_scope_rebuilds_the_whole_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "structured_fields.json"
            cache = StructuredFieldCache.load(
                path,
                schema_version=5,
                source_fingerprint="source-a",
            )
            cache.initialize_document_presence({"文书A": True})
            cache.initialize_document_section_map({"文书A": [1]})
            cache.register_section(1, "文书A")
            cache.merge_fields(1, {"字段": "旧值"})
            cache.save()

            with patch("review.DocumentMappingAgents") as agents_class:
                check_presence = (
                    agents_class.return_value.classify_document_presence
                )
                check_presence.return_value = {
                    "文书A": True,
                    "文书B": True,
                }
                presence = review._resolve_document_presence(
                    cache,
                    ["文书A", "文书B"],
                    [],
                )

            check_presence.assert_called_once_with(
                ["文书A", "文书B"],
                [],
            )

            rebuilt = StructuredFieldCache.load(
                path,
                schema_version=5,
                source_fingerprint="source-a",
            )

        self.assertEqual(presence, {"文书A": True, "文书B": True})
        self.assertEqual(rebuilt.document_presence, presence)
        self.assertEqual(rebuilt.document_section_map, {})
        self.assertEqual(rebuilt.sections, {})

    async def test_shared_mapping_results_merge_and_finalize_once(self) -> None:
        events = []
        context_free_rule_set = RuleSet(
            rules=[{"序号": 101}, {"序号": 201}],
        )
        context_sensitive_rule_set = RuleSet(
            rules=[{"序号": 101}],
        )
        context_free_results = [
                    {"rule_index": 101, "issues": [{"section_ids": [1], "content": "A"}]},
                    {"rule_index": 201, "issues": []},
        ]
        context_sensitive_results = [
                    {"rule_index": 101, "issues": [{"section_ids": [2], "content": "B"}]}
        ]

        class FakeContextFree(DocumentReviewExecutor):
            required_document_names = staticmethod(
                lambda rules: ["立案审批表"]
            )

            def __init__(self, **kwargs):
                self.rule_set = kwargs["rule_set"]

            def executable_rule_indexes(self):
                return [0, 1]

            async def execute_raw(self):
                return context_free_results

        class FakeContextSensitive(DocumentReviewExecutor):
            required_document_names = staticmethod(
                lambda rules: ["立案审批表"]
            )

            def __init__(self, **kwargs):
                self.rule_set = kwargs["rule_set"]
                self.last_preparation_result = {
                    "preparation_completed": True
                }

            def executable_rule_indexes(self):
                return [0]

            async def execute_raw(self):
                return context_sensitive_results

        class FakeCaseLevel:
            def __init__(self, *args, **kwargs):
                pass

            async def execute_raw(self):
                return []

            def status(self):
                return {
                    "implemented": False,
                    "skipped": True,
                    "reason": "not_implemented",
                }

        preparation = SimpleNamespace(
            prepare=AsyncMock(return_value={"立案审批表": [1]})
        )

        async def finalize(raw_results, *, rule_count):
            events.append("finalize")
            return {"rule_results": raw_results, "rule_count": rule_count}

        coordinator = SimpleNamespace(finalize=AsyncMock(side_effect=finalize))
        registry = {
            ReviewExecutorType.CONTEXT_FREE: FakeContextFree,
            ReviewExecutorType.CONTEXT_SENSITIVE: FakeContextSensitive,
            ReviewExecutorType.CASE_LEVEL: FakeCaseLevel,
        }
        with patch(
            "review.load_enabled_executor_types",
            return_value=list(registry),
        ), patch(
            "review._build_document_rule_sets",
            return_value=(
                {
                    ReviewExecutorType.CONTEXT_FREE: context_free_rule_set,
                    ReviewExecutorType.CONTEXT_SENSITIVE: (
                        context_sensitive_rule_set
                    ),
                },
                {"立案审批表": True},
            ),
        ) as build_rule_sets, patch.dict(
            review.EXECUTOR_REGISTRY,
            registry,
            clear=True,
        ), patch(
            "review.load_context_sensitive_settings",
            return_value={"prewarm_fields": {}},
        ), patch(
            "review.DocumentReviewPreparationService", return_value=preparation
        ) as preparation_class, patch(
            "review.ReviewResultCoordinator", return_value=coordinator
        ):
            result = await run_review("case.pdf", 0b11010000)

        build_rule_sets.assert_called_once_with(
            "case.pdf",
            0b11010000,
            [
                ReviewExecutorType.CONTEXT_FREE,
                ReviewExecutorType.CONTEXT_SENSITIVE,
            ],
        )
        preparation_class.assert_called_once()
        preparation.prepare.assert_awaited_once_with()
        coordinator.finalize.assert_awaited_once()
        finalized = coordinator.finalize.await_args.args[0]
        self.assertEqual([item["rule_index"] for item in finalized], [101, 201])
        self.assertEqual(len(finalized[0]["issues"]), 2)
        self.assertEqual(coordinator.finalize.await_args.kwargs["rule_count"], 2)
        self.assertEqual(result["context_sensitive_rule_count"], 1)
        self.assertEqual(result["case_level_review"]["reason"], "not_implemented")

    async def test_disabled_executor_is_not_constructed(self) -> None:
        class FakeContextFree(DocumentReviewExecutor):
            def __init__(self, **kwargs):
                self.rule_set = kwargs["rule_set"]

            def executable_rule_indexes(self):
                return []

        class ForbiddenContextSensitive:
            def __init__(self, *args, **kwargs):
                raise AssertionError("disabled executor was constructed")

        class FakeCaseLevel:
            pass

        registry = {
            ReviewExecutorType.CONTEXT_FREE: FakeContextFree,
            ReviewExecutorType.CONTEXT_SENSITIVE: ForbiddenContextSensitive,
            ReviewExecutorType.CASE_LEVEL: FakeCaseLevel,
        }
        with patch(
            "review.load_enabled_executor_types",
            return_value=[ReviewExecutorType.CONTEXT_FREE],
        ), patch(
            "review._build_document_rule_sets",
            return_value=(
                {ReviewExecutorType.CONTEXT_FREE: RuleSet(rules=[])},
                {},
            ),
        ), patch.dict(
            review.EXECUTOR_REGISTRY,
            registry,
            clear=True,
        ):
            result = await run_review("case.pdf", 0b11010000)

        self.assertEqual(
            result["context_sensitive_preparation"]["reason"],
            "disabled",
        )
        self.assertEqual(result["case_level_review"]["reason"], "disabled")

    async def test_human_support_is_persisted_without_merging_into_review(
        self,
    ) -> None:
        ordinary_rule = {
            "序号": 360,
            "上下文无关审查事项": {
                "行政处罚决定书": {"审查事项": "普通审查"}
            },
        }
        support_rule = {
            **ordinary_rule,
            "检索增强": [
                "city_management_discretion_candidates"
            ],
        }

        class FakeContextFree(DocumentReviewExecutor):
            required_document_names = staticmethod(
                lambda rules: ["行政处罚决定书"]
            )

            def __init__(self, **kwargs):
                self.rule_set = kwargs["rule_set"]

            def executable_rule_indexes(self):
                return [0]

            async def execute_raw(self):
                return [{"rule_index": 360, "issues": []}]

        class FakeHumanSupport:
            def __init__(self, **kwargs):
                self.rule_set = kwargs["rule_set"]

            def executable_rule_indexes(self):
                return [0] if self.rule_set.rules else []

            async def execute(self):
                return [
                    {
                        "rule_index": 360,
                        "documents": [],
                    }
                ]

        class FakeCaseLevel:
            pass

        preparation = SimpleNamespace(
            prepare=AsyncMock(return_value={"行政处罚决定书": [1]})
        )
        review_coordinator = SimpleNamespace(
            finalize=AsyncMock(
                return_value={
                    "rule_results": [
                        {"rule_index": 360, "issues": []}
                    ]
                }
            )
        )
        support_coordinator = SimpleNamespace(
            finalize=Mock(
                return_value={
                    "retrieval_enhancement_results": [
                        {
                            "rule_index": 360,
                            "documents": [],
                        }
                    ],
                    "rule_count": 1,
                    "result_path": (
                        "retrieval_enhancement_results.json"
                    ),
                    "skipped": False,
                    "reason": None,
                }
            )
        )
        with patch(
            "review.load_enabled_executor_types",
            return_value=[ReviewExecutorType.CONTEXT_FREE],
        ), patch(
            "review.load_enabled_support_executor_types",
            return_value=[SupportExecutorType.HUMAN_SUPPORT],
        ), patch(
            "review._build_document_rule_sets",
            return_value=(
                {
                    ReviewExecutorType.CONTEXT_FREE: RuleSet(
                        rules=[ordinary_rule]
                    )
                },
                {"行政处罚决定书": True},
            ),
        ), patch(
            "review._candidate_human_support_rules",
            return_value=[support_rule],
        ), patch(
            "review._ensure_human_support_document_presence",
            return_value={"行政处罚决定书": True},
        ), patch.dict(
            review.EXECUTOR_REGISTRY,
            {
                ReviewExecutorType.CONTEXT_FREE: FakeContextFree,
                ReviewExecutorType.CASE_LEVEL: FakeCaseLevel,
            },
            clear=True,
        ), patch.dict(
            review.SUPPORT_EXECUTOR_REGISTRY,
            {SupportExecutorType.HUMAN_SUPPORT: FakeHumanSupport},
            clear=True,
        ), patch(
            "review.DocumentReviewPreparationService",
            return_value=preparation,
        ), patch(
            "review.ReviewResultCoordinator",
            return_value=review_coordinator,
        ), patch(
            "review.HumanSupportResultCoordinator",
            return_value=support_coordinator,
        ):
            result = await run_review("case.pdf", 0b01010000)

        review_coordinator.finalize.assert_awaited_once()
        ordinary_results = review_coordinator.finalize.await_args.args[0]
        self.assertEqual(ordinary_results, [{"rule_index": 360, "issues": []}])
        support_coordinator.finalize.assert_called_once()
        self.assertEqual(
            result["retrieval_enhancement"]["rule_count"],
            1,
        )
        self.assertEqual(
            result["retrieval_enhancement"][
                "retrieval_enhancement_results"
            ][0],
            {"rule_index": 360, "documents": []},
        )

    async def test_context_sensitive_execute_raw_only_runs_its_own_items(self) -> None:
        executor = ContextSensitiveReviewExecutor.__new__(
            ContextSensitiveReviewExecutor
        )
        executor.last_preparation_result = None
        executor.run_preparation = AsyncMock(
            return_value={"preparation_completed": True}
        )
        executor.run_consistency_reviews = AsyncMock(
            return_value=[{"rule_index": 101, "issues": []}]
        )

        result = await executor.execute_raw()

        executor.run_preparation.assert_awaited_once_with()
        executor.run_consistency_reviews.assert_awaited_once_with()
        self.assertEqual(result, [{"rule_index": 101, "issues": []}])


class ReviewPipelineConfigTest(unittest.TestCase):
    def test_enabled_executors_are_loaded_in_configured_order(self) -> None:
        with patch(
            "review_config.load_yaml",
            return_value={
                "executors": ["context_sensitive", "context_free"]
            },
        ):
            result = load_enabled_executor_types("pipeline.yaml")

        self.assertEqual(
            result,
            [
                ReviewExecutorType.CONTEXT_SENSITIVE,
                ReviewExecutorType.CONTEXT_FREE,
            ],
        )

    def test_duplicate_executor_configuration_is_collapsed(self) -> None:
        with patch(
            "review_config.load_yaml",
            return_value={"executors": ["context_free", "context_free"]},
        ):
            result = load_enabled_executor_types("pipeline.yaml")

        self.assertEqual(result, [ReviewExecutorType.CONTEXT_FREE])

    def test_human_support_is_loaded_from_its_own_configuration(self) -> None:
        with patch(
            "review_config.load_yaml",
            return_value={
                "executors": ["context_free"],
                "support_executors": ["human_support"],
            },
        ):
            result = load_enabled_support_executor_types("pipeline.yaml")

        self.assertEqual(result, [SupportExecutorType.HUMAN_SUPPORT])


if __name__ == "__main__":
    unittest.main()
