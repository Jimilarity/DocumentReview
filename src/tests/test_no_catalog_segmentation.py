import os
import sys
import unittest
from pathlib import Path


os.environ.setdefault("AGENT_TRACE_ENABLED", "false")
SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from pre_review import (
    _build_page_profile,
    _build_page_profiles,
    _objective_boundary_relation,
    _route_directory_scan,
    _segment_documents_without_catalog,
    _validate_segment_catalog,
)


class _SegmentationAgents:
    def __init__(self) -> None:
        self.boundaries = iter(["same_document", "new_document"])
        self.names = iter([
            {
                "section_kind": "case_document",
                "normalized_document_type": "询问笔录",
                "material_type": None,
                "subject_role": "不适用",
            },
            {
                "section_kind": "evidence_material",
                "normalized_document_type": "身份证信息",
                "material_type": "身份证信息",
                "subject_role": "当事人",
            },
        ])

    def build_task_prompt(self, task_name: str, **kwargs) -> str:
        return task_name

    async def ainvoke_page_boundary_classifier(self, prompt: str) -> dict:
        return {
            "relation": next(self.boundaries),
            "reason_codes": ["test"],
        }

    async def ainvoke_segment_start_classifier(self, prompt: str) -> dict:
        return next(self.names)


class NoCatalogSegmentationTest(unittest.IsolatedAsyncioTestCase):
    def _state(self) -> dict:
        texts = [
            '<div type="page-number">1</div>询问笔录\n第1页/共2页\n正文开始',
            '<div type="page-number">2</div>接上页\n正文结束\n签名 日期',
            '<div type="img" subtype="true-copy">居民身份证</div>',
        ]
        return {
            "pdf_path": "case.pdf",
            "page_image_paths": [f"page_{i}.jpg" for i in range(3)],
            "document_ocr_results": [
                {
                    "image_index": index,
                    "image_path": f"page_{index}.jpg",
                    "document_content": text,
                }
                for index, text in enumerate(texts)
            ],
            "document_catalog": [],
            "catalog_page_index": 3,
            "case_metadata": {"当事人": "张三"},
        }

    def test_profile_is_bounded_and_extracts_page_fields(self) -> None:
        profile = _build_page_profile(
            0,
            '<div type="page-number">9</div>询问笔录 第1页/共2页'
            + "正文" * 2000,
        )

        self.assertLessEqual(len(profile["head_text"]), 1000)
        self.assertLessEqual(len(profile["tail_text"]), 700)
        self.assertEqual(profile["printed_page_number"], 9)
        self.assertEqual(profile["page_in_document"], 1)
        self.assertEqual(profile["total_pages"], 2)

    def test_page_total_inside_page_number_tag_is_preserved(self) -> None:
        profile = _build_page_profile(
            0,
            '<div type="page-number">第 3 页 共 4 页</div>正文',
        )

        self.assertEqual(profile["page_in_document"], 3)
        self.assertEqual(profile["total_pages"], 4)

    def test_objective_guard_keeps_numbered_continuation(self) -> None:
        start = _build_page_profile(
            0,
            '<div type="page-number">第 1 页 共 3 页</div>询问笔录',
        )
        previous = _build_page_profile(
            1,
            '<div type="page-number">第 2 页 共 3 页</div>正文',
        )
        current = _build_page_profile(
            2,
            '<div type="page-number">第 3 页 共 3 页</div>本页正文',
        )

        relation, reasons = _objective_boundary_relation(
            start,
            previous,
            current,
            "new_document",
        )

        self.assertEqual(relation, "same_document")
        self.assertIn("document_page_sequence", reasons)

    def test_objective_guard_keeps_near_duplicate_page(self) -> None:
        profile = _build_page_profile(
            0,
            '<div type="page-number">第 4 页 共 4 页</div>'
            "处理意见 建议不予行政处罚 出席人员签名",
        )

        relation, reasons = _objective_boundary_relation(
            profile,
            profile,
            profile,
            "new_document",
        )

        self.assertEqual(relation, "same_document")
        self.assertIn("duplicate_or_near_duplicate_page", reasons)

    def test_new_document_without_start_signal_is_merged(self) -> None:
        start = _build_page_profile(0, "不予行政处罚决定书 正文")
        previous = _build_page_profile(1, "上一页未结束的内容")
        current = _build_page_profile(2, "续接前句并在本页落款")

        relation, reasons = _objective_boundary_relation(
            start,
            previous,
            current,
            "new_document",
        )

        self.assertEqual(relation, "same_document")
        self.assertIn("new_document_lacks_objective_start_signal", reasons)

    def test_directory_scan_routes_empty_catalog_to_profiles(self) -> None:
        state = self._state()

        self.assertEqual(_route_directory_scan(state), "build_page_profiles")

    def test_directory_scan_keeps_original_catalog_path(self) -> None:
        state = self._state()
        state["document_catalog"] = [{"section_id": 1}]

        self.assertEqual(
            _route_directory_scan(state),
            "locate_document_sections",
        )

    async def test_segmentation_covers_every_page_and_uses_generic_name(self) -> None:
        state = self._state()
        state.update(_build_page_profiles(state))

        result = await _segment_documents_without_catalog(
            state,
            _SegmentationAgents(),
        )

        self.assertNotIn("error_code", result)
        self.assertEqual(
            [item["section_page"] for item in result["document_catalog"]],
            [0, 2],
        )
        self.assertEqual(
            [item["section_end_page"] for item in result["document_catalog"]],
            [2, 3],
        )
        self.assertEqual(
            result["document_catalog"][1]["section_name"],
            "当事人身份证信息",
        )
        self.assertNotIn("张三", result["document_catalog"][1]["section_name"])
        _validate_segment_catalog(result["document_catalog"], 3)


if __name__ == "__main__":
    unittest.main()
