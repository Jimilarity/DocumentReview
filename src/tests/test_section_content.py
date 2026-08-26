import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from reviewers.section_content import (
    extract_section_ocr_text,
    extract_sections_ocr_text,
    section_page_range,
)
from directory_info import normalize_directory_info


class SectionContentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir_info = [
            {
                "section_id": 1,
                "section_name": "目录",
                "section_page": 0,
            },
            {
                "section_id": 2,
                "section_name": "调查询问笔录",
                "section_page": 2,
            },
            {
                "section_id": 3,
                "section_name": "行政处罚决定书",
                "section_page": 4,
            },
        ]
        self.ocr_results = [
            {"image_index": index, "document_content": f"第{index}页"}
            for index in range(6)
        ]

    def test_section_page_range_uses_next_section_boundary(self) -> None:
        self.assertEqual(
            section_page_range(2, self.dir_info, len(self.ocr_results)),
            (2, 4),
        )
        self.assertEqual(
            section_page_range(3, self.dir_info, len(self.ocr_results)),
            (4, 6),
        )

    def test_extracts_multiple_sections_in_directory_order(self) -> None:
        text = extract_sections_ocr_text(
            [3, 2, 3],
            self.dir_info,
            self.ocr_results,
        )

        self.assertLess(text.index("section_id=2"), text.index("section_id=3"))
        self.assertIn("directory_metadata", text)
        self.assertIn("source=case_directory", text)
        self.assertIn("is_page_title=false", text)
        self.assertIn("[page_index=2]\n第2页", text)
        self.assertIn("[page_index=5]\n第5页", text)
        self.assertEqual(text.count("section_id=3"), 1)

    def test_extracts_only_the_requested_section(self) -> None:
        text = extract_section_ocr_text(
            2,
            self.dir_info,
            self.ocr_results,
        )

        self.assertIn("section_id=2", text)
        self.assertIn("section_name=调查询问笔录", text)
        self.assertIn("is_page_title=false", text)
        self.assertNotIn("section_id=3", text)
        self.assertIn("[page_index=2]\n第2页", text)
        self.assertNotIn("[page_index=4]", text)

    def test_ocr_segmented_metadata_reports_its_real_source(self) -> None:
        dir_info = [
            {
                "section_id": 1,
                "section_name": "当事人身份证信息",
                "section_page": 0,
                "catalog_source": "ocr_segmented",
            }
        ]
        ocr_results = [
            {"image_index": 0, "document_content": "身份证信息"}
        ]

        text = extract_section_ocr_text(1, dir_info, ocr_results)

        self.assertIn("source=ocr_segmentation", text)
        self.assertNotIn("source=case_directory", text)

    def test_normalizes_string_section_fields(self) -> None:
        string_dir_info = [
            {
                **item,
                "section_id": str(item["section_id"]),
                "section_page": str(item["section_page"]),
            }
            for item in self.dir_info
        ]

        normalized = normalize_directory_info(string_dir_info)

        self.assertEqual(
            [item["section_id"] for item in normalized],
            [1, 2, 3],
        )
        self.assertEqual(
            [item["section_page"] for item in normalized],
            [0, 2, 4],
        )
        self.assertEqual(
            section_page_range(2, string_dir_info, len(self.ocr_results)),
            (2, 4),
        )


if __name__ == "__main__":
    unittest.main()
