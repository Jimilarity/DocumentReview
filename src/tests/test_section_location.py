import os
import sys
import unittest
from pathlib import Path


os.environ.setdefault("AGENT_TRACE_ENABLED", "false")
SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from pre_review import (
    _estimate_page_number_offset,
    _locate_document_sections,
    _section_candidate_pages,
)


class _SectionAgents:
    def __init__(self, results: list[str]) -> None:
        self.results = results
        self.calls: list[dict] = []

    def build_task_prompt(self, task_name: str, **kwargs) -> str:
        self.calls.append({"task_name": task_name, **kwargs})
        return "单页章节判断"

    async def ainvoke_section_classifier(
        self,
        prompt_text: str,
        page_index: int,
    ) -> dict[str, str]:
        return {"result": self.results.pop(0)}


class SectionLocationTest(unittest.IsolatedAsyncioTestCase):
    def _state(self) -> dict:
        page_numbers = [None, None, 1, 2, 99, 4, 5, 6]
        ocr_results = []
        for page_index, page_number in enumerate(page_numbers):
            page_number_text = (
                f'<div type="page-number">{page_number}</div>\n'
                if page_number is not None
                else ""
            )
            ocr_results.append(
                {
                    "document_content": (
                        f"{page_number_text}第 {page_index} 页正文"
                    ),
                    "image_path": f"page_{page_index}.jpeg",
                    "image_index": page_index,
                }
            )
        return {
            "pdf_path": "case.pdf",
            "page_image_paths": [
                f"page_{page_index}.jpeg"
                for page_index in range(len(page_numbers))
            ],
            "document_ocr_results": ocr_results,
            "document_catalog": [
                {
                    "section_id": 1,
                    "section_name": "视听资料/电子数据",
                    "section_page": -1,
                    "catalog_page": 3,
                },
                {
                    "section_id": 2,
                    "section_name": "非税收入罚款通知书",
                    "section_page": -1,
                    "catalog_page": 5,
                },
            ],
            "catalog_page_index": 2,
            "body_start_page_index": 2,
            "current_document_id": 1,
            "section_candidate_pages": [],
            "section_candidate_cursor": 0,
            "section_predicted_page": 0,
        }

    def test_page_offset_uses_majority_and_ignores_outlier(self) -> None:
        state = self._state()

        self.assertEqual(_estimate_page_number_offset(state), 1)
        candidates, predicted_page, uses_catalog_page = (
            _section_candidate_pages(state, 1)
        )

        self.assertEqual(predicted_page, 4)
        self.assertEqual(candidates, [4, 3, 5, 2, 6])
        self.assertTrue(uses_catalog_page)

    async def test_matching_predicted_page_is_accepted(self) -> None:
        state = self._state()
        agents = _SectionAgents(["match"])

        result = await _locate_document_sections(state, agents)

        section = result["document_catalog"][0]
        self.assertEqual(section["section_page"], 4)
        self.assertEqual(section["location_source"], "page_and_semantic")
        self.assertFalse(section["needs_review"])
        self.assertEqual(agents.calls[0]["page_index"], 4)
        self.assertEqual(agents.calls[0]["previous_section_name"], "（无）")
        self.assertEqual(
            agents.calls[0]["next_section_name"],
            "非税收入罚款通知书",
        )

    async def test_conflict_checks_nearby_pages_one_at_a_time(self) -> None:
        state = self._state()
        agents = _SectionAgents(["conflict", "match"])

        first_result = await _locate_document_sections(state, agents)
        state.update(first_result)
        second_result = await _locate_document_sections(state, agents)

        self.assertEqual(
            [call["page_index"] for call in agents.calls],
            [4, 3],
        )
        section = second_result["document_catalog"][0]
        self.assertEqual(section["section_page"], 3)
        self.assertEqual(section["location_source"], "nearby_semantic")

    async def test_unknown_uses_catalog_prediction_without_more_calls(self) -> None:
        state = self._state()
        agents = _SectionAgents(["unknown"])

        result = await _locate_document_sections(state, agents)

        section = result["document_catalog"][0]
        self.assertEqual(section["section_page"], 4)
        self.assertEqual(section["location_source"], "catalog_page")
        self.assertEqual(len(agents.calls), 1)

    async def test_unknown_neighbor_does_not_override_conflicted_prediction(self) -> None:
        state = self._state()
        agents = _SectionAgents(["conflict", "unknown", "match"])

        for _ in range(3):
            result = await _locate_document_sections(state, agents)
            state.update(result)

        self.assertEqual(
            [call["page_index"] for call in agents.calls],
            [4, 3, 5],
        )
        self.assertEqual(state["document_catalog"][0]["section_page"], 5)


if __name__ == "__main__":
    unittest.main()
