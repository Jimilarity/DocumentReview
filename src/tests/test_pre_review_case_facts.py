import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


os.environ.setdefault("AGENT_TRACE_ENABLED", "false")
SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from pre_review import _extract_case_facts


class _CaseFactsAgents:
    def __init__(self, responses: list[dict[str, str | None]]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def build_task_prompt(self, task_name: str) -> str:
        self.task_name = task_name
        return "提取案情"

    async def ainvoke_case_facts(self, prompt_text: str) -> dict[str, str | None]:
        self.prompts.append(prompt_text)
        return self.responses.pop(0)


class PreReviewCaseFactsTest(unittest.IsolatedAsyncioTestCase):
    def _state(self) -> dict:
        return {
            "pdf_path": "case.pdf",
            "case_metadata": {"案由": "未张贴操作规程"},
            "document_ocr_results": [
                {
                    "document_content": "王某作为负责人，未张贴安全操作规程。",
                    "image_path": "page_0.jpeg",
                    "image_index": 0,
                }
            ],
            "document_catalog": [
                {"section_name": "行政处罚决定书", "section_page": 0},
            ],
        }

    async def test_retries_until_case_facts_are_extracted(self) -> None:
        agents = _CaseFactsAgents(
            [
                {"案情": None},
                {"案情": "王某作为负责人，未张贴安全操作规程。"},
            ]
        )

        with patch.dict(
            os.environ,
            {"CASE_FACTS_EXTRACTION_MAX_ATTEMPTS": "3"},
        ), patch("pre_review.atomic_write_json") as write_json:
            result = await _extract_case_facts(self._state(), agents)

        self.assertEqual(len(agents.prompts), 2)
        self.assertIn("上一次未能提取出案情", agents.prompts[1])
        self.assertEqual(
            result["case_metadata"]["案情"],
            "王某作为负责人，未张贴安全操作规程。",
        )
        write_json.assert_called_once()

    async def test_keeps_case_facts_empty_when_all_attempts_are_empty(self) -> None:
        agents = _CaseFactsAgents([{"案情": None}] * 3)

        with patch.dict(
            os.environ,
            {"CASE_FACTS_EXTRACTION_MAX_ATTEMPTS": "3"},
        ), patch("pre_review.atomic_write_json") as write_json:
            result = await _extract_case_facts(self._state(), agents)

        self.assertEqual(len(agents.prompts), 3)
        self.assertIsNone(result["case_metadata"]["案情"])
        write_json.assert_called_once()

    async def test_missing_allowed_documents_does_not_send_full_ocr(self) -> None:
        state = self._state()
        state["document_catalog"] = [
            {"section_name": "询问笔录", "section_page": 0},
        ]
        agents = _CaseFactsAgents([])

        with patch("pre_review.atomic_write_json") as write_json:
            result = await _extract_case_facts(state, agents)

        self.assertEqual(agents.prompts, [])
        self.assertIsNone(result["case_metadata"]["案情"])
        write_json.assert_called_once()

    async def test_falls_back_only_to_closing_or_filing_form(self) -> None:
        state = self._state()
        state["document_ocr_results"].append(
            {
                "document_content": "无关文书",
                "image_path": "page_1.jpeg",
                "image_index": 1,
            }
        )
        state["document_catalog"] = [
            {"section_name": "立案审批表", "section_page": 0},
            {"section_name": "询问笔录", "section_page": 1},
        ]
        agents = _CaseFactsAgents([{"案情": "王某作为负责人，未张贴安全操作规程。"}])

        with patch("pre_review.atomic_write_json"):
            await _extract_case_facts(state, agents)

        self.assertIn("王某作为负责人", agents.prompts[0])
        self.assertNotIn("无关文书", agents.prompts[0])

    async def test_no_catalog_incrementally_fills_metadata_without_overwriting(self) -> None:
        state = self._state()
        state["catalog_source"] = "ocr_segmented"
        state["document_ocr_results"].append(
            {
                "document_content": "结案审批表正文",
                "image_path": "page_1.jpeg",
                "image_index": 1,
            }
        )
        state["document_catalog"] = [
            {
                "section_name": "行政处罚决定书",
                "normalized_document_type": "行政处罚决定书",
                "section_page": 0,
            },
            {
                "section_name": "结案审批表",
                "normalized_document_type": "结案审批表",
                "section_page": 1,
            },
        ]
        first = {
            "案情": "当事人实施了违法行为。",
            "执法单位": "执法单位甲",
            "案卷名称": "案卷甲",
            "案号": "案号甲",
            "案由": "案由甲",
            "当事人": "当事人甲",
            "立案日期": "2022-01-10",
            "结案日期": None,
            "处理结果": "罚款。",
            "案件承办人员及执法证件号": "承办人甲",
        }
        second = {
            "案情": "不应覆盖的案情。",
            "案号": "不应覆盖的案号",
            "结案日期": "2022-01-14",
        }
        agents = _CaseFactsAgents([first, second])

        with patch("pre_review.atomic_write_json") as write_json:
            result = await _extract_case_facts(state, agents)

        metadata = result["case_metadata"]
        self.assertEqual(len(agents.prompts), 2)
        self.assertEqual(metadata["案情"], first["案情"])
        self.assertEqual(metadata["案号"], first["案号"])
        self.assertEqual(metadata["结案日期"], second["结案日期"])
        self.assertEqual(metadata["执法单位"], first["执法单位"])
        self.assertIsNone(metadata.get("额外字段"))
        write_json.assert_called_once()

    async def test_no_catalog_without_key_documents_sets_required_fields_to_null(self) -> None:
        state = self._state()
        state["catalog_source"] = "ocr_segmented"
        state["document_catalog"] = [
            {
                "section_name": "询问笔录",
                "normalized_document_type": "询问笔录",
                "section_page": 0,
            }
        ]
        agents = _CaseFactsAgents([])

        with patch("pre_review.atomic_write_json") as write_json:
            result = await _extract_case_facts(state, agents)

        metadata = result["case_metadata"]
        self.assertEqual(agents.prompts, [])
        self.assertIsNone(metadata["案情"])
        self.assertIsNone(metadata["案号"])
        self.assertIsNone(metadata["结案日期"])
        write_json.assert_called_once()


if __name__ == "__main__":
    unittest.main()
