import asyncio
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from types import SimpleNamespace


SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from review_config import (
    load_case_level_review_items,
    load_context_sensitive_settings,
)
from constants import RULES_PATH, STRUCTURED_FIELD_CACHE_SCHEMA_VERSION
from errors.handler import CURRENT_PDF_PATH
from structured_field_cache import (
    TECHNICAL_MAPPING_FAILURE,
    StructuredFieldCache,
    build_structured_source_fingerprint,
)
from reviewers.context_free import ContextFreeReviewContext
from reviewers.context_sensitive import (
    ContextSensitiveReviewExecutor,
    normalize_delivery_extracted_fields,
    validate_extracted_fields,
)
from reviewers.consistency import CONSISTENCY_TASK, ConsistencySource
from reviewers.document_mapping import normalize_document_section_map
from rules.filtering import filter_context_sensitive_rules
from rules.rule_set import RuleSetBuilder


class ReviewConfigurationTest(unittest.TestCase):
    def test_completeness_configuration_is_not_rule_traceable(self) -> None:
        settings = load_context_sensitive_settings()
        case_level_items = load_case_level_review_items()

        self.assertEqual(STRUCTURED_FIELD_CACHE_SCHEMA_VERSION, 6)
        self.assertEqual(
            settings["service_receipt_document_type"],
            "送达回证",
        )
        self.assertEqual(case_level_items, [])

    def test_delivery_receipt_prewarms_all_configured_fields(self) -> None:
        settings = load_context_sensitive_settings()

        self.assertEqual(
            settings["prewarm_fields"]["送达回证"],
            list(settings["field_specs"]["送达回证"]),
        )

    def test_prompt_field_specs_omit_internal_english_names(self) -> None:
        settings = load_context_sensitive_settings()

        specs = ContextSensitiveReviewExecutor._minimal_prompt_specs(
            {
                field_name: settings["field_specs"]["送达回证"][field_name]
                for field_name in ["送达日期", "送达方式"]
            }
        )

        self.assertEqual(
            specs,
            [
                {"field": "送达日期", "type": "datetime"},
                {"field": "送达方式", "type": "str"},
            ],
        )

    def test_penalty_decision_fields_are_merged_under_one_type(self) -> None:
        settings = load_context_sensitive_settings()

        self.assertNotIn("当场行政处罚决定书", settings["field_specs"])
        penalty_fields = settings["field_specs"]["行政处罚决定书"]
        self.assertIn("违法时间", penalty_fields)
        self.assertIn("处罚具体内容", penalty_fields)
        self.assertIn("处罚决定种类", penalty_fields)
        self.assertIn("执法人员信息", penalty_fields)

    def test_ordinary_penalty_context_fields_are_all_configured(self) -> None:
        settings = load_context_sensitive_settings()
        rules = (
            RuleSetBuilder.from_json(RULES_PATH)
            .for_rule_type(0b11110100)
            .build()
            .rules
        )

        missing = []
        for rule in rules:
            for context_item in rule["上下文相关审查事项"]:
                for document_type, field_items in context_item["字段"].items():
                    document_specs = settings["field_specs"].get(
                        document_type,
                        {},
                    )
                    for field_item in field_items:
                        field_name = field_item["field"]
                        if field_name not in document_specs:
                            missing.append(
                                (rule["序号"], document_type, field_name)
                            )

        self.assertEqual(missing, [])

    def test_null_is_a_valid_extracted_value(self) -> None:
        validate_extracted_fields(
            {"是否拒收": None, "送达日期": None},
            {
                "是否拒收": {"type": "bool"},
                "送达日期": {"type": "datetime"},
            },
        )

    def test_delivery_fields_discard_unknown_keys_and_fill_missing_with_null(self) -> None:
        specs = {
            "代收人与受送人关系": {"type": "str"},
            "送达日期": {"type": "datetime"},
        }

        normalized = normalize_delivery_extracted_fields(
            {
                "代收人与受送达人关系": "受委托人",
                "送达日期": "2025-03-01",
                "模型自行增加字段": "无关值",
            },
            specs,
        )

        self.assertEqual(
            normalized,
            {
                "代收人与受送人关系": None,
                "送达日期": "2025-03-01",
            },
        )
        validate_extracted_fields(normalized, specs)

    def test_literal_field_accepts_configured_value_and_null(self) -> None:
        specs = {
            "案件来源": {
                "type": "literal['投诉', '举报', '巡查', '检查', '其他']"
            }
        }

        validate_extracted_fields({"案件来源": "检查"}, specs)
        validate_extracted_fields({"案件来源": None}, specs)

    def test_literal_field_rejects_unconfigured_value(self) -> None:
        with self.assertRaisesRegex(TypeError, "字段 案件来源"):
            validate_extracted_fields(
                {"案件来源": "移送"},
                {"案件来源": {"type": "literal['投诉', '举报']"}},
            )

    def test_literal_field_rejects_invalid_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "literal 字段类型配置无效"):
            validate_extracted_fields(
                {"案件来源": "投诉"},
                {"案件来源": {"type": "literal[]"}},
            )


class StructuredFieldCacheTest(unittest.TestCase):
    def test_metadata_is_part_of_the_source_fingerprint(self) -> None:
        first = build_structured_source_fingerprint(
            {"案号": "A"},
            [],
            [],
        )
        second = build_structured_source_fingerprint(
            {"案号": "B"},
            [],
            [],
        )

        self.assertNotEqual(first, second)

    def test_document_presence_mapping_and_completion_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "structured_fields.json"
            cache = StructuredFieldCache.load(
                path,
                schema_version=2,
                source_fingerprint="source-a",
            )
            cache.initialize_document_presence({"立案审批表": True})
            cache.initialize_document_section_map({"立案审批表": [3]})
            cache.mark_preparation_completed()
            cache.save()

            restored = StructuredFieldCache.load(
                path,
                schema_version=2,
                source_fingerprint="source-a",
            )

            self.assertTrue(restored.preparation_completed)
            self.assertEqual(restored.document_presence, {"立案审批表": True})
            self.assertEqual(
                restored.document_section_map,
                {"立案审批表": [3]},
            )

    def test_one_section_can_only_have_one_document_type(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = StructuredFieldCache.load(
                Path(directory) / "structured_fields.json",
                schema_version=2,
                source_fingerprint="source-a",
            )
            cache.register_section(3, "立案审批表")

            self.assertEqual(cache.document_type(3), "立案审批表")
            with self.assertRaisesRegex(ValueError, "不能再次登记"):
                cache.register_section(3, "行政处罚决定书")

    def test_compatible_evidence_mapping_can_share_section(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = StructuredFieldCache.load(
                Path(directory) / "structured_fields.json",
                schema_version=2,
                source_fingerprint="source-a",
            )
            expected = {
                "当事人提交的复制件等证据材料": [3],
                "行政执法人员采集制作的证据材料": [3],
                "计算机数据、录音、录像、图片等证据材料": [3],
            }

            cache.initialize_document_section_map(expected)

            self.assertEqual(cache.document_section_map, expected)

    def test_context_sensitive_metadata_registers_only_needed_types(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = StructuredFieldCache.load(
                Path(directory) / "structured_fields.json",
                schema_version=2,
                source_fingerprint="source-a",
            )
            executor = ContextSensitiveReviewExecutor.__new__(
                ContextSensitiveReviewExecutor
            )
            executor.document_section_map = {
                "行政执法人员采集制作的证据材料": [3],
                "计算机数据、录音、录像、图片等证据材料": [3],
                "立案审批表": [4],
            }

            executor._register_mapped_sections(cache, ["立案审批表"])

            self.assertEqual(cache.document_type(4), "立案审批表")
            self.assertNotIn("3", cache.section_metadata)

    def test_initialized_mapping_cannot_be_replaced_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = StructuredFieldCache.load(
                Path(directory) / "structured_fields.json",
                schema_version=2,
                source_fingerprint="source-a",
            )
            cache.initialize_document_section_map({"立案审批表": [3]})
            cache.register_section(3, "立案审批表")
            cache.merge_fields(3, {"案件来源": "检查发现"})
            cache.mark_preparation_completed()

            with self.assertRaisesRegex(RuntimeError, "必须重建整份缓存"):
                cache.initialize_document_section_map(
                    {"行政处罚决定书": [3]}
                )

            self.assertEqual(cache.peek_field(3, "案件来源"), "检查发现")
            self.assertEqual(
                cache.document_section_map,
                {"立案审批表": [3]},
            )

    def test_missing_field_distinguishes_unextracted_from_null(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = StructuredFieldCache.load(
                Path(directory) / "structured_fields.json",
                schema_version=2,
                source_fingerprint="source-a",
            )
            cache.register_section(3, "立案审批表")
            self.assertEqual(
                cache.missing_fields(3, ["案件来源"]),
                ["案件来源"],
            )

            cache.merge_fields(3, {"案件来源": None})

            self.assertTrue(cache.has_field(3, "案件来源"))
            self.assertIsNone(cache.peek_field(3, "案件来源"))
            self.assertEqual(cache.missing_fields(3, ["案件来源"]), [])

    def test_delivery_records_use_related_section_or_event_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = StructuredFieldCache.load(
                Path(directory) / "structured_fields.json",
                schema_version=2,
                source_fingerprint="source-a",
            )
            cache.register_section(25, "送达回证")
            cache.set_delivery_events(
                25,
                [
                    {
                        "event_id": 1,
                        "source_order": 1,
                        "event_text": "决定书送达记录",
                        "related_section_id": 20,
                        "送达日期": "2025-03-01",
                    },
                    {
                        "event_id": 2,
                        "source_order": 2,
                        "event_text": "找不到文书的送达记录",
                        "related_section_id": None,
                        "送达日期": None,
                    },
                    {
                        "event_id": 3,
                        "source_order": 3,
                        "event_text": "映射调用失败的送达记录",
                        "related_section_id": TECHNICAL_MAPPING_FAILURE,
                        "送达日期": None,
                    },
                ],
            )

            receipt = cache.sections["25"]
            self.assertEqual(receipt["20"]["event_id"], 1)
            self.assertIsNone(receipt["event:2"]["related_section_id"])
            self.assertEqual(
                receipt["event:3"]["related_section_id"],
                TECHNICAL_MAPPING_FAILURE,
            )
            self.assertEqual(
                cache.delivery_event_for_document(20)["送达日期"],
                "2025-03-01",
            )
            self.assertEqual(cache.next_event_id(), 4)

    def test_two_receipts_may_target_one_compound_document_section(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = StructuredFieldCache.load(
                Path(directory) / "structured_fields.json",
                schema_version=2,
                source_fingerprint="source-a",
            )
            cache.register_section(2, "送达回证")
            cache.register_section(3, "送达回证")
            cache.set_delivery_events(
                2,
                [
                    {
                        "event_id": 1,
                        "source_order": 1,
                        "event_text": "行政处罚决定书送达记录",
                        "related_section_id": 1,
                    }
                ],
            )

            cache.set_delivery_events(
                3,
                [
                    {
                        "event_id": 2,
                        "source_order": 1,
                        "event_text": "决定书附件送达记录",
                        "related_section_id": 1,
                    }
                ],
            )

            self.assertEqual(cache.delivery_events(2)[0]["event_id"], 1)
            self.assertEqual(cache.delivery_events(3)[0]["event_id"], 2)

    def test_source_change_invalidates_derived_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "structured_fields.json"
            cache = StructuredFieldCache.load(
                path,
                schema_version=2,
                source_fingerprint="source-a",
            )
            cache.register_section(3, "立案审批表")
            cache.merge_fields(3, {"案件来源": "检查发现"})
            cache.save()

            invalidated = StructuredFieldCache.load(
                path,
                schema_version=2,
                source_fingerprint="source-b",
            )

            self.assertEqual(invalidated.sections, {})
            self.assertEqual(invalidated.document_section_map, {})
            self.assertFalse(invalidated.preparation_completed)

    def test_schema_change_invalidates_preparation_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "structured_fields.json"
            cache = StructuredFieldCache.load(
                path,
                schema_version=2,
                source_fingerprint="source-a",
            )
            cache.initialize_document_presence({"立案审批表": True})
            cache.initialize_document_section_map({"立案审批表": [3]})
            cache.mark_preparation_completed()
            cache.save()

            invalidated = StructuredFieldCache.load(
                path,
                schema_version=3,
                source_fingerprint="source-a",
            )

            self.assertEqual(invalidated.document_presence, {})
            self.assertEqual(invalidated.document_section_map, {})
            self.assertFalse(invalidated.preparation_completed)

    def test_empty_receipt_is_remembered_as_extracted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = StructuredFieldCache.load(
                Path(directory) / "structured_fields.json",
                schema_version=2,
                source_fingerprint="source-a",
            )
            cache.register_section(25, "送达回证")
            cache.set_delivery_events(25, [])

            self.assertEqual(cache.delivery_events(25), [])
            self.assertTrue(cache.delivery_extraction_completed(25))

    def test_unmapped_pending_state_cannot_be_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = StructuredFieldCache.load(
                Path(directory) / "structured_fields.json",
                schema_version=2,
                source_fingerprint="source-a",
            )
            cache.register_section(25, "送达回证")

            with self.assertRaisesRegex(ValueError, "必须完成映射"):
                cache.set_delivery_events(
                    25,
                    [
                        {
                            "event_id": 1,
                            "source_order": 1,
                            "event_text": "送达原文",
                        }
                    ],
                )


class StructuredFieldReadThroughTest(unittest.IsolatedAsyncioTestCase):
    async def test_same_field_concurrency_uses_one_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = StructuredFieldCache.load(
                Path(directory) / "structured_fields.json",
                schema_version=2,
                source_fingerprint="source-a",
            )
            cache.register_section(3, "立案审批表")
            calls = []

            async def extractor(section_id, field_names):
                calls.append((section_id, tuple(field_names)))
                await asyncio.sleep(0)
                return {field_name: None for field_name in field_names}

            cache.bind_field_extractor(extractor)
            first, second = await asyncio.gather(
                cache.get_field(3, "案件来源"),
                cache.get_field(3, "案件来源"),
            )

            self.assertIsNone(first)
            self.assertIsNone(second)
            self.assertEqual(calls, [(3, ("案件来源",))])
            self.assertTrue(cache.path.is_file())

    async def test_fields_in_one_request_are_extracted_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = StructuredFieldCache.load(
                Path(directory) / "structured_fields.json",
                schema_version=2,
                source_fingerprint="source-a",
            )
            cache.register_section(3, "立案审批表")
            calls = []

            async def extractor(section_id, field_names):
                calls.append(tuple(field_names))
                return {field_name: "值" for field_name in field_names}

            cache.bind_field_extractor(extractor)
            values = await cache.get_fields(3, ["案件来源", "案由"])

            self.assertEqual(values, {"案件来源": "值", "案由": "值"})
            self.assertEqual(calls, [("案件来源", "案由")])

    async def test_delivery_field_read_through_is_transparent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = StructuredFieldCache.load(
                Path(directory) / "structured_fields.json",
                schema_version=2,
                source_fingerprint="source-a",
            )
            cache.register_section(25, "送达回证")
            cache.set_delivery_events(
                25,
                [
                    {
                        "event_id": 1,
                        "source_order": 1,
                        "event_text": "送达原文",
                        "related_section_id": 20,
                    }
                ],
            )

            async def extractor(receipt_id, relation_key, field_names):
                self.assertEqual((receipt_id, relation_key), (25, "20"))
                return {"送达日期": "2025-03-01"}

            cache.bind_delivery_field_extractor(extractor)
            value = await cache.get_delivery_field(25, 20, "送达日期")

            self.assertEqual(value, "2025-03-01")
            self.assertEqual(
                cache.sections["25"]["20"]["送达日期"],
                "2025-03-01",
            )


class DeliveryMappingInputTest(unittest.IsolatedAsyncioTestCase):
    async def test_duplicate_existing_mappings_are_repaired_before_cache_write(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = StructuredFieldCache.load(
                Path(directory) / "structured_fields.json",
                schema_version=2,
                source_fingerprint="source-a",
            )
            cache.register_section(3, "送达回证")
            cache.sections["3"] = {
                "event:1": {
                    "event_id": 1,
                    "source_order": 1,
                    "event_text": "第一次送达",
                    "related_section_id": 12,
                },
                "event:2": {
                    "event_id": 2,
                    "source_order": 2,
                    "event_text": "第二次送达",
                    "related_section_id": 12,
                },
            }
            cache.section_metadata["3"]["delivery_extraction_completed"] = True
            executor = ContextSensitiveReviewExecutor.__new__(
                ContextSensitiveReviewExecutor
            )
            executor.logger = MagicMock()

            changed = await executor._prewarm_delivery_receipt(cache, 3, [])

            self.assertTrue(changed)
            events = cache.delivery_events(3)
            self.assertEqual(events[0]["related_section_id"], 12)
            self.assertIsNone(events[1]["related_section_id"])

    async def test_mapping_prefers_nearest_preceding_document_when_name_missing(
        self,
    ) -> None:
        executor = ContextSensitiveReviewExecutor.__new__(
            ContextSensitiveReviewExecutor
        )
        executor.context_settings = {
            "service_receipt_document_type": "送达回证",
        }
        executor.document_section_map = {
            "行政处罚决定书": [8],
            "查封（扣押）决定书": [3],
            "送达回证": [10],
        }
        executor.context = ContextFreeReviewContext(
            meta_info={},
            dir_info=[
                {
                    "section_id": 3,
                    "section_name": "查封（扣押）决定书",
                    "section_page": 1,
                },
                {
                    "section_id": 8,
                    "section_name": "行政处罚决定书",
                    "section_page": 2,
                },
                {
                    "section_id": 10,
                    "section_name": "送达回证",
                    "section_page": 3,
                },
            ],
            ocr_results=[],
        )

        related_section_id = await executor._map_delivery_event(
            10,
            {"event_id": 1, "event_text": "收件人签名栏"},
        )

        self.assertEqual(related_section_id, 8)

    async def test_existing_mapping_is_repaired_when_nearer_document_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = StructuredFieldCache.load(
                Path(directory) / "structured_fields.json",
                schema_version=2,
                source_fingerprint="source-a",
            )
            cache.register_section(13, "送达回证")
            cache.sections["13"] = {
                "12": {
                    "event_id": 1,
                    "source_order": 1,
                    "event_text": "送达记录",
                    "related_section_id": 12,
                }
            }
            cache.section_metadata["13"]["delivery_extraction_completed"] = True
            executor = ContextSensitiveReviewExecutor.__new__(
                ContextSensitiveReviewExecutor
            )
            executor.context_settings = {
                "service_receipt_document_type": "送达回证",
            }
            executor.document_section_map = {
                "查封（扣押）决定书": [3],
                "行政处罚决定书": [8],
                "送达回证": [13],
            }
            executor.context = ContextFreeReviewContext(
                meta_info={},
                dir_info=[
                    {
                        "section_id": 3,
                        "section_name": "查封（扣押）决定书",
                        "section_page": 1,
                    },
                    {
                        "section_id": 8,
                        "section_name": "行政处罚决定书",
                        "section_page": 2,
                    },
                    {
                        "section_id": 12,
                        "section_name": "其他文书",
                        "section_page": 3,
                    },
                    {
                        "section_id": 13,
                        "section_name": "送达回证",
                        "section_page": 4,
                    },
                ],
                ocr_results=[],
            )
            executor.logger = MagicMock()

            changed = await executor._prewarm_delivery_receipt(cache, 13, [])

            self.assertTrue(changed)
            self.assertEqual(
                cache.delivery_events(13)[0]["related_section_id"],
                8,
            )

    async def test_retry_may_target_section_used_by_another_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = StructuredFieldCache.load(
                Path(directory) / "structured_fields.json",
                schema_version=2,
                source_fingerprint="source-a",
            )
            cache.register_section(2, "送达回证")
            cache.register_section(3, "送达回证")
            cache.set_delivery_events(
                2,
                [
                    {
                        "event_id": 1,
                        "source_order": 1,
                        "event_text": "行政处罚决定书送达记录",
                        "related_section_id": 1,
                    }
                ],
            )
            cache.set_delivery_events(
                3,
                [
                    {
                        "event_id": 2,
                        "source_order": 1,
                        "event_text": "决定书附件送达记录",
                        "related_section_id": TECHNICAL_MAPPING_FAILURE,
                    }
                ],
            )
            executor = ContextSensitiveReviewExecutor.__new__(
                ContextSensitiveReviewExecutor
            )

            async def map_event(section_id, event):
                self.assertEqual(section_id, 3)
                self.assertEqual(event["event_id"], 2)
                return 1

            executor._map_delivery_event = map_event

            changed = await executor._prewarm_delivery_receipt(
                cache,
                3,
                [],
            )

            self.assertTrue(changed)
            self.assertEqual(
                cache.delivery_events(3)[0]["related_section_id"],
                1,
            )

    async def test_duplicate_mapping_within_receipt_keeps_later_event_unassociated(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = StructuredFieldCache.load(
                Path(directory) / "structured_fields.json",
                schema_version=2,
                source_fingerprint="source-a",
            )
            cache.register_section(3, "送达回证")
            cache.set_delivery_events(
                3,
                [
                    {
                        "event_id": 1,
                        "source_order": 1,
                        "event_text": "决定书送达记录",
                        "related_section_id": TECHNICAL_MAPPING_FAILURE,
                    },
                    {
                        "event_id": 2,
                        "source_order": 2,
                        "event_text": "决定书再次送达记录",
                        "related_section_id": TECHNICAL_MAPPING_FAILURE,
                    },
                ],
            )
            executor = ContextSensitiveReviewExecutor.__new__(
                ContextSensitiveReviewExecutor
            )
            executor._map_delivery_event = AsyncMock(return_value=1)
            executor.logger = MagicMock()

            changed = await executor._prewarm_delivery_receipt(cache, 3, [])

            self.assertTrue(changed)
            events = cache.delivery_events(3)
            self.assertEqual(events[0]["related_section_id"], 1)
            self.assertIsNone(events[1]["related_section_id"])
            executor.logger.warning.assert_called_once()

    async def test_document_after_receipt_remains_in_full_directory(self) -> None:
        captured = {}

        class FakeAgents:
            def build_task_prompt(self, task_name, **kwargs):
                captured["task_name"] = task_name
                captured.update(kwargs)
                return "mapping prompt"

            async def ainvoke_delivery_receipt_mapper(self, prompt, limit):
                return SimpleNamespace(related_section_id=12)

        executor = ContextSensitiveReviewExecutor.__new__(
            ContextSensitiveReviewExecutor
        )
        executor.context_settings = {
            "service_receipt_document_type": "送达回证",
            "prewarm_fields": {},
            "field_specs": {},
        }
        executor.document_section_map = {"送达回证": [10]}
        executor.context = ContextFreeReviewContext(
            meta_info={"案号": "A"},
            dir_info=[
                {"section_id": 10, "section_name": "送达回证", "section_page": 1},
                {"section_id": 12, "section_name": "行政处罚决定书", "section_page": 3},
            ],
            ocr_results=[],
        )
        executor.settings = SimpleNamespace(agent_recursion_limit=20)
        executor.require_context_sensitive_agents = lambda: FakeAgents()

        related_section_id = await executor._map_delivery_event(
            10,
            {
                "event_id": 1,
                "event_text": "行政处罚决定书送达记录",
            },
        )

        self.assertEqual(related_section_id, 12)
        self.assertEqual(captured["task_name"], "delivery_receipt_mapping")
        self.assertEqual(captured["receipt_section_id"], 10)
        self.assertIn('"section_id": 12', captured["dir_info"])
        self.assertNotIn("priority", captured["dir_info"])
        self.assertNotIn("relative_position", captured["dir_info"])


class DocumentSectionMappingCacheTest(unittest.IsolatedAsyncioTestCase):
    async def test_one_document_type_may_use_multiple_sections(self) -> None:
        mapping = normalize_document_section_map(
            {"询问笔录": [4, 3]},
            [
                {"section_id": 3, "section_name": "询问笔录一"},
                {"section_id": 4, "section_name": "询问笔录二"},
            ],
        )

        self.assertEqual(mapping, {"询问笔录": [3, 4]})

    async def test_one_section_cannot_map_to_two_document_types(self) -> None:
        with self.assertRaisesRegex(ValueError, "不能同时映射"):
            normalize_document_section_map(
                {
                    "立案审批表": [3],
                    "行政处罚决定书": [3],
                },
                [{"section_id": 3, "section_name": "文书"}],
            )


class ContextSensitiveRuleFilterTest(unittest.TestCase):
    def test_sensitive_filter_uses_only_its_presence_subset(self) -> None:
        rule = {
            "序号": 101,
            "上下文无关审查事项": {},
            "上下文相关审查事项": [
                {
                    "字段": {
                        "立案审批表": [
                            {"field": "执法主体名称", "required": True}
                        ]
                    }
                }
            ],
        }
        result = filter_context_sensitive_rules(
            [rule],
            {"立案审批表": True},
        )

        self.assertEqual(result, [rule])

    def test_sensitive_filter_removes_only_missing_document_fields(self) -> None:
        rule = {
            "序号": 101,
            "上下文无关审查事项": {},
            "上下文相关审查事项": [
                {
                    "字段": {
                        "立案审批表": [
                            {"field": "执法主体名称", "required": True}
                        ],
                        "结案审批表": [
                            {"field": "执法主体名称", "required": True}
                        ],
                    }
                }
            ],
        }
        result = filter_context_sensitive_rules(
            [rule],
            {"立案审批表": True, "结案审批表": False},
        )

        self.assertEqual(
            set(result[0]["上下文相关审查事项"][0]["字段"]),
            {"立案审批表"},
        )


class PrewarmPlanTest(unittest.TestCase):
    def test_missing_field_spec_raises_readable_configuration_error(self) -> None:
        executor = ContextSensitiveReviewExecutor.__new__(
            ContextSensitiveReviewExecutor
        )
        executor.context_settings = {
            "service_receipt_document_type": "送达回证",
            "prewarm_fields": {},
            "field_specs": {"询问笔录": {"被询问人信息": {"type": "str"}}},
        }

        with self.assertRaisesRegex(
            ValueError,
            "询问笔录.*缺少字段配置.*当事人信息",
        ):
            executor._specs_for_fields("询问笔录", ["当事人信息"])

    def test_section_resolves_fields_from_its_document_type(self) -> None:
        executor = ContextSensitiveReviewExecutor.__new__(
            ContextSensitiveReviewExecutor
        )
        executor.structured_field_cache = SimpleNamespace(
            document_type=lambda section_id: "检查笔录"
        )
        executor.context_settings = {
            "service_receipt_document_type": "送达回证",
            "prewarm_fields": {},
            "field_specs": {
                "检查笔录": {
                    "当事人": {"type": "str", "notes": "检查笔录"}
                }
            },
        }

        specs = executor._specs_for_section(3, ["当事人"])

        self.assertEqual(specs["当事人"]["type"], "str")
        self.assertEqual(specs["当事人"]["notes"], "检查笔录")

    def test_active_context_fields_are_merged_with_common_fields(self) -> None:
        executor = ContextSensitiveReviewExecutor.__new__(
            ContextSensitiveReviewExecutor
        )
        executor.context_settings = {
            "service_receipt_document_type": "送达回证",
            "prewarm_fields": {"立案审批表": ["执法主体名称"]},
            "field_specs": {
                "立案审批表": {
                    "执法主体名称": {"type": "str"},
                    "案件来源": {"type": "str"},
                },
                "结案审批表": {
                    "结案日期": {"type": "datetime"},
                },
            },
        }
        executor.rule_set = SimpleNamespace(
            rules=[
                {
                    "序号": 101,
                    "上下文相关审查事项": [
                        {
                            "字段": {
                                "立案审批表": [
                                    {"field": "案件来源", "required": True},
                                    {"field": "执法主体名称", "required": True},
                                ],
                                "结案审批表": [
                                    {"field": "结案日期", "required": False}
                                ],
                            }
                        }
                    ],
                }
            ],
        )

        plan = executor._prewarm_fields_by_document()

        self.assertEqual(
            plan,
            {
                "立案审批表": ["执法主体名称", "案件来源"],
                "结案审批表": ["结案日期"],
            },
        )


class ConsistencyExecutionTest(unittest.IsolatedAsyncioTestCase):
    async def test_receipt_group_keeps_only_its_corresponding_document_section(
        self,
    ) -> None:
        executor = ContextSensitiveReviewExecutor.__new__(
            ContextSensitiveReviewExecutor
        )
        executor.context_settings = {
            "service_receipt_document_type": "送达回证"
        }
        executor.document_section_map = {
            "行政处罚决定书": [3],
            "查封（扣押）决定书": [8],
            "送达回证": [10, 11],
        }
        executor._regular_consistency_sources = AsyncMock(
            side_effect=lambda document_type, field_items, section_ids=None: [
                ConsistencySource(
                    document_type=document_type,
                    section_id=section_id,
                    field_name=field_items[0]["field"],
                    required=False,
                    value=f"document-{section_id}",
                )
                for section_id in (
                    section_ids
                    or executor.document_section_map[document_type]
                )
            ]
        )
        executor._delivery_consistency_sources = AsyncMock()
        rule = {
            "上下文相关审查事项": [
                {
                    "任务": CONSISTENCY_TASK,
                    "字段": {
                        "行政处罚决定书": [
                            {"field": "文号", "required": False}
                        ],
                        "查封（扣押）决定书": [
                            {"field": "文号", "required": False}
                        ],
                        "送达回证": [
                            {"field": "送达文书文号", "required": False}
                        ],
                    },
                }
            ]
        }
        executor._delivery_consistency_sources.side_effect = [
            [
                ConsistencySource(
                    document_type="送达回证",
                    section_id=10,
                    related_section_id=3,
                    field_name="送达文书文号",
                    required=False,
                    value="receipt-3",
                )
            ],
            [
                ConsistencySource(
                    document_type="送达回证",
                    section_id=11,
                    related_section_id=8,
                    field_name="送达文书文号",
                    required=False,
                    value="receipt-8",
                )
            ],
        ]

        groups = await executor.collect_consistency_source_groups(rule, 0)

        self.assertEqual(
            [
                [source.section_id for source in group if source.document_type != "送达回证"]
                for group in groups
            ],
            [[3], [8]],
        )

    async def test_receipt_sources_are_grouped_by_the_only_regular_document(self) -> None:
        executor = ContextSensitiveReviewExecutor.__new__(
            ContextSensitiveReviewExecutor
        )
        executor.context_settings = {
            "service_receipt_document_type": "送达回证"
        }
        executor.document_section_map = {
            "行政处罚决定书": [3, 8],
            "送达回证": [10, 11],
        }
        executor._regular_consistency_sources = AsyncMock(
            side_effect=lambda document_type, field_items, section_ids=None: [
                ConsistencySource(
                    document_type=document_type,
                    section_id=section_id,
                    field_name=field_items[0]["field"],
                    required=False,
                    value=f"document-{section_id}",
                )
                for section_id in (section_ids or [3, 8])
            ]
        )
        executor._delivery_consistency_sources = AsyncMock(
            side_effect=lambda document_type, field_items, related_section_ids=None: [
                ConsistencySource(
                    document_type=document_type,
                    section_id=10 + next(iter(related_section_ids)),
                    related_section_id=next(iter(related_section_ids)),
                    field_name=field_items[0]["field"],
                    required=False,
                    value=f"receipt-{next(iter(related_section_ids))}",
                )
            ]
        )
        rule = {
            "上下文相关审查事项": [
                {
                    "任务": CONSISTENCY_TASK,
                    "字段": {
                        "行政处罚决定书": [
                            {"field": "文号", "required": False}
                        ],
                        "送达回证": [
                            {"field": "送达文书文号", "required": False}
                        ],
                    },
                }
            ]
        }

        groups = await executor.collect_consistency_source_groups(rule, 0)

        self.assertEqual(
            [[source.related_section_id for source in group] for group in groups],
            [[None, 3], [None, 8]],
        )

    def test_field_category_accepts_two_values_and_rejects_others(self) -> None:
        field_items = [
            {"field": "当事人信息", "required": True, "字段类别": category}
            for category in ["审查对象", "辅助支撑"]
        ]

        self.assertEqual(
            ContextSensitiveReviewExecutor._validate_field_items(
                "行政处罚决定书",
                field_items,
            ),
            field_items,
        )
        with self.assertRaisesRegex(ValueError, "字段类别"):
            ContextSensitiveReviewExecutor._validate_field_items(
                "行政处罚决定书",
                [
                    {
                        "field": "当事人信息",
                        "required": True,
                        "字段类别": "其他",
                    }
                ],
            )

    async def test_field_category_is_included_in_model_sources(self) -> None:
        executor = ContextSensitiveReviewExecutor.__new__(
            ContextSensitiveReviewExecutor
        )
        agent = SimpleNamespace(
            build_task_prompt=MagicMock(return_value="prompt"),
            ainvoke_consistency_review=AsyncMock(
                return_value=SimpleNamespace(
                    consistent=True,
                    reason="一致",
                )
            ),
        )
        executor.require_context_sensitive_agents = lambda: agent
        executor.settings = SimpleNamespace(
            agent_recursion_limit=10,
            task_timeout_seconds=5,
        )

        await executor._consistency_judgement(
            {"审查事项": "核对当事人"},
            [
                ConsistencySource(
                    document_type="案件调查报告",
                    section_id=3,
                    field_name="违法事实",
                    required=True,
                    value="某违法行为",
                    field_category="辅助支撑",
                )
            ],
            [],
        )

        sources = agent.build_task_prompt.call_args.kwargs["sources"]
        self.assertIn('"field_category":"辅助支撑"', sources)

    async def test_generic_context_task_uses_categorized_prompt(self) -> None:
        executor = ContextSensitiveReviewExecutor.__new__(
            ContextSensitiveReviewExecutor
        )
        source = ConsistencySource(
            document_type="行政处罚决定书",
            section_id=8,
            field_name="当事人信息",
            required=False,
            value=[{"当事人名称": "甲公司"}],
            field_category="辅助支撑",
        )
        executor.collect_consistency_sources = AsyncMock(
            return_value=[source]
        )
        executor.collect_external_knowledge = AsyncMock(return_value=[])
        agent = SimpleNamespace(
            build_task_prompt=MagicMock(return_value="prompt"),
            ainvoke_contextual_legality_review=AsyncMock(
                return_value=SimpleNamespace(issues=[])
            ),
        )
        executor.require_context_sensitive_agents = lambda: agent
        executor.settings = SimpleNamespace(agent_recursion_limit=10)
        rule = {
            "上下文相关审查事项": [
                {
                    "任务": "事实清楚、证据充分",
                    "字段": {},
                    "审查事项": "审查当事人是否适格",
                }
            ]
        }

        result = await executor.run_contextual_legality_item(rule, 0)

        self.assertEqual(result, {"issues": []})
        self.assertEqual(
            agent.build_task_prompt.call_args.args[0],
            "categorized_contextual_review",
        )
        self.assertIn(
            '"field_category": "辅助支撑"',
            agent.build_task_prompt.call_args.kwargs["sources"],
        )

    async def test_generic_context_task_reports_only_missing_required_fields(self) -> None:
        executor = ContextSensitiveReviewExecutor.__new__(
            ContextSensitiveReviewExecutor
        )
        executor.collect_consistency_sources = AsyncMock(
            return_value=[
                ConsistencySource(
                    document_type="案件调查报告",
                    section_id=3,
                    field_name="违法事实",
                    required=True,
                    value=None,
                    field_category="辅助支撑",
                ),
                ConsistencySource(
                    document_type="行政处罚决定书",
                    section_id=8,
                    field_name="处罚具体内容",
                    required=False,
                    value=None,
                    field_category="辅助支撑",
                ),
            ]
        )
        executor.collect_external_knowledge = AsyncMock(return_value=[])
        agent = SimpleNamespace(
            build_task_prompt=MagicMock(return_value="prompt"),
            ainvoke_contextual_legality_review=AsyncMock(
                return_value=SimpleNamespace(issues=[])
            ),
        )
        executor.require_context_sensitive_agents = lambda: agent
        executor.settings = SimpleNamespace(agent_recursion_limit=10)

        result = await executor.run_contextual_legality_item(
            {
                "上下文相关审查事项": [
                    {
                        "任务": "上下文合法性审查",
                        "字段": {},
                        "审查事项": "审查当事人是否适格",
                    }
                ]
            },
            0,
        )

        self.assertEqual(len(result["issues"]), 1)
        self.assertIn("违法事实", result["issues"][0]["content"])

    async def test_different_field_names_use_one_agent_judgement(self) -> None:
        executor = ContextSensitiveReviewExecutor.__new__(
            ContextSensitiveReviewExecutor
        )
        sources = [
            ConsistencySource(
                document_type="现场检查（勘验）笔录",
                section_id=3,
                field_name="嫌疑人",
                required=True,
                value="张三",
            ),
            ConsistencySource(
                document_type="行政处罚决定书",
                section_id=8,
                field_name="被执行人",
                required=True,
                value="张三",
            ),
        ]
        executor.collect_consistency_sources = AsyncMock(
            return_value=sources
        )
        executor._consistency_judgement = AsyncMock(
            return_value=SimpleNamespace(
                consistent=True,
                reason="两个角色字段指向同一人。",
            )
        )
        executor.collect_external_knowledge = AsyncMock(return_value=[])
        rule = {
            "上下文相关审查事项": [
                {
                    "任务": "一致性核查",
                    "审查事项": "核对当事人",
                }
            ]
        }

        result = await executor.run_consistency_item(rule, 0)

        executor._consistency_judgement.assert_awaited_once_with(
            rule["上下文相关审查事项"][0],
            sources,
            [],
        )
        self.assertEqual(result["issues"], [])
        self.assertEqual(set(result), {"issues"})


class ContextSensitiveTraceRoutingTest(unittest.IsolatedAsyncioTestCase):
    async def test_execute_raw_routes_trace_events_to_current_document(self) -> None:
        executor = ContextSensitiveReviewExecutor.__new__(
            ContextSensitiveReviewExecutor
        )
        executor.file_path = Path("C:/cases/context-sensitive.pdf")
        observed_paths = []

        async def run_preparation():
            observed_paths.append(CURRENT_PDF_PATH.get())
            return {"preparation_completed": True}

        async def run_consistency_reviews():
            observed_paths.append(CURRENT_PDF_PATH.get())
            return [{"rule_index": 113, "issues": []}]

        executor.run_preparation = run_preparation
        executor.run_consistency_reviews = run_consistency_reviews

        outer_token = CURRENT_PDF_PATH.set("C:/cases/outer.pdf")
        try:
            result = await executor.execute_raw()
            self.assertEqual(
                observed_paths,
                [
                    "C:/cases/context-sensitive.pdf",
                    "C:/cases/context-sensitive.pdf",
                ],
            )
            self.assertEqual(result, [{"rule_index": 113, "issues": []}])
            self.assertEqual(CURRENT_PDF_PATH.get(), "C:/cases/outer.pdf")
        finally:
            CURRENT_PDF_PATH.reset(outer_token)


if __name__ == "__main__":
    unittest.main()
