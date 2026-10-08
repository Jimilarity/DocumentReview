import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import main as main_module
from cache_paths import get_cache_paths
from constants import STRUCTURED_FIELD_CACHE_SCHEMA_VERSION
from reviewers.base import DocumentReviewContext
from reviewers.context_free import ContextFreeReviewExecutor
from reviewers.context_sensitive import ContextSensitiveReviewExecutor
from rules.rule_set import RuleSet
from structured_field_cache import StructuredFieldCache
from structured_input import prepare_structured_json
from utils import read_json


class StructuredInputTest(unittest.TestCase):
    def test_json_is_adapted_without_modifying_the_source(self) -> None:
        source_data = {
            "立案审批信息": {
                "案件编号": "A-001",
                "执法主体名称": "测试执法单位",
                "案件来源": "空",
                "当事人信息": {"当事人名称": "测试当事人"},
            },
            "行政处罚决定书": {
                "案件编号": "A-001",
                "行政处罚决定书文号": "测试罚决字001号",
                "违法事实": "测试违法事实",
                "处罚依据": "空",
            },
            "当场行政处罚决定书": {
                "案件编号": "A-001",
                "违法事实/行为": "测试违法事实",
                "罚款金额小写": "1000",
            },
            "送达回证": {
                "案件编号（case_no）": "A-001",
                "文书名称": "行政处罚决定书",
                "送达日期": "2026-08-15",
                "是否拒收": "否",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "case.json"
            source_text = json.dumps(
                source_data,
                ensure_ascii=False,
                indent=2,
            )
            source_path.write_text(source_text, encoding="utf-8")
            original_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()

            result = prepare_structured_json(
                source_path,
                0b11110000,
                cache_root=root / "derived-cache",
            )

            self.assertEqual(
                hashlib.sha256(source_path.read_bytes()).hexdigest(),
                original_hash,
            )
            self.assertEqual(source_path.read_text(encoding="utf-8"), source_text)
            paths = get_cache_paths(
                source_path,
                cache_root=root / "derived-cache",
            )
            metadata = read_json(paths.metadata)
            directory_info = read_json(paths.directory)
            ocr_results = read_json(paths.ocr_results)
            self.assertEqual(metadata["source_type"], "structured_json")
            self.assertEqual(metadata["案号"], "测试罚决字001号")
            self.assertEqual(len(directory_info), 4)
            self.assertEqual(len(ocr_results), 4)
            self.assertEqual(read_json(paths.image_list), [])
            self.assertTrue(result["completed"])

            cache = StructuredFieldCache.load(
                paths.structured_fields,
                schema_version=STRUCTURED_FIELD_CACHE_SCHEMA_VERSION,
                source_fingerprint=read_json(paths.structured_fields)[
                    "source_fingerprint"
                ],
            )
            self.assertEqual(
                cache.document_section_map["行政处罚决定书"],
                [2, 3],
            )
            self.assertEqual(
                directory_info[2]["section_name"],
                "当场行政处罚决定书",
            )
            self.assertEqual(
                directory_info[2]["normalized_document_type"],
                "行政处罚决定书",
            )
            self.assertIsNone(cache.sections["2"]["处罚依据"])
            self.assertEqual(
                cache.sections["3"]["罚款金额小写"],
                "1000",
            )
            event = cache.delivery_events(4)[0]
            self.assertEqual(event["案件编号"], "A-001")
            self.assertFalse(event["是否拒收"])
            self.assertEqual(event["related_section_id"], 2)

    def test_context_free_review_disables_images_for_json(self) -> None:
        executor = ContextFreeReviewExecutor(
            file_path="case.json",
            rule_set=RuleSet(rules=[]),
        )
        executor.context = DocumentReviewContext(
            meta_info={"source_type": "structured_json"},
            dir_info=[],
            ocr_results=[],
        )

        self.assertEqual(executor.get_local_tools(), [])

    def test_unconfigured_context_fields_use_structured_cache(self) -> None:
        executor = ContextSensitiveReviewExecutor.__new__(
            ContextSensitiveReviewExecutor
        )
        executor.context_settings = {
            "service_receipt_document_type": "送达回证",
            "prewarm_fields": {},
            "field_specs": {},
        }
        executor.rule_set = type(
            "RuleSetStub",
            (),
            {
                "rules": [
                    {
                        "上下文相关审查事项": [
                            {
                                "字段": {
                                    "委托书": [
                                        {
                                            "field": "委托开始日期",
                                            "required": True,
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ]
            },
        )()
        executor.context = DocumentReviewContext(
            meta_info={"source_type": "structured_json"},
            dir_info=[],
            ocr_results=[],
        )

        self.assertEqual(
            executor._prewarm_fields_by_document(),
            {"委托书": ["委托开始日期"]},
        )

    def test_validate_input_accepts_pdf_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "case.pdf"
            json_path = root / "case.json"
            pdf_path.touch()
            json_path.write_text("{}", encoding="utf-8")

            self.assertEqual(main_module.validate_input(str(pdf_path)), pdf_path)
            self.assertEqual(main_module.validate_input(str(json_path)), json_path)

            text_path = root / "case.txt"
            text_path.touch()
            with self.assertRaisesRegex(ValueError, "PDF 或 JSON"):
                main_module.validate_input(str(text_path))


class StructuredMainFlowTest(unittest.IsolatedAsyncioTestCase):
    async def test_json_uses_adapter_and_skips_pdf_pre_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "case.json"
            source_path.write_text('{"行政处罚决定书": {}}', encoding="utf-8")
            review_result = {
                "review_results": [],
                "retrieval_enhancement": {},
            }
            structured_result = {
                "completed": True,
                "source_type": "structured_json",
            }

            with patch.object(
                main_module,
                "prepare_structured_json",
                return_value=structured_result,
            ) as prepare_json, patch.object(
                main_module,
                "run_pre_review",
                new=AsyncMock(),
            ) as run_pdf_pre_review, patch.object(
                main_module,
                "run_review",
                new=AsyncMock(return_value=review_result),
            ) as run_review, patch.object(
                main_module,
                "run_post_review",
                new=AsyncMock(return_value={"completed": True}),
            ):
                result = await main_module.main(
                    str(source_path),
                    0b11110000,
                )

            self.assertTrue(result["success"])
            prepare_json.assert_called_once_with(
                source_path,
                0b11110000,
                rules_path=main_module.RULES_PATH,
            )
            run_pdf_pre_review.assert_not_awaited()
            run_review.assert_awaited_once_with(
                str(source_path),
                0b11110000,
                rules_path=main_module.RULES_PATH,
            )


if __name__ == "__main__":
    unittest.main()
