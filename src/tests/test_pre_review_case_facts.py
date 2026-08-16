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


if __name__ == "__main__":
    unittest.main()
