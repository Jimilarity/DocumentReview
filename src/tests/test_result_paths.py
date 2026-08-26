import sys
import tempfile
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from cache_paths import (
    build_document_key,
    get_cache_paths,
    get_result_directory,
)
from constants import NO_CATALOG_SEGMENTATION_SCHEMA_VERSION
from main import _pre_review_cache_is_reusable
from utils import atomic_write_json


class ResultPathTest(unittest.TestCase):
    def test_same_named_pdfs_in_different_directories_do_not_collide(
        self,
    ) -> None:
        first_pdf = Path("tenant-a") / "case.pdf"
        second_pdf = Path("tenant-b") / "case.pdf"

        first_key = build_document_key(first_pdf)
        second_key = build_document_key(second_pdf)

        self.assertNotEqual(first_key, second_key)
        self.assertTrue(first_key.startswith("case-"))
        self.assertTrue(second_key.startswith("case-"))

    def test_result_directory_uses_the_document_key(self) -> None:
        pdf_path = Path("tenant-a") / "case.pdf"
        result_root = Path("custom-results")

        result_directory = get_result_directory(
            pdf_path,
            result_root=result_root,
        )

        self.assertEqual(result_directory.parent, result_root)
        self.assertEqual(
            result_directory.name,
            build_document_key(pdf_path),
        )

    def test_review_audit_files_use_the_document_cache(self) -> None:
        paths = get_cache_paths(
            Path("tenant-a") / "case.pdf",
            cache_root=Path("custom-cache"),
        )

        self.assertEqual(
            paths.raw_review_results.parent,
            paths.cache_directory,
        )
        self.assertEqual(
            paths.raw_review_results.name,
            "raw_review_results.json",
        )
        self.assertEqual(
            paths.scored_raw_review_results.name,
            "raw_review_results_scored.json",
        )
        self.assertEqual(
            paths.review_result_processing.name,
            "review_result_processing.json",
        )
        self.assertEqual(
            paths.structured_fields.name,
            "structured_fields.json",
        )

    def test_old_no_catalog_segmentation_cache_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = get_cache_paths(
                Path("tenant-a") / "case.pdf",
                cache_root=directory,
            )
            paths.cache_directory.mkdir(parents=True)
            for path in (
                paths.image_list,
                paths.ocr_results,
                paths.metadata,
            ):
                atomic_write_json(path, [])
            atomic_write_json(paths.directory, [{
                "section_id": 1,
                "catalog_source": "ocr_segmented",
            }])

            self.assertFalse(_pre_review_cache_is_reusable(paths))

            atomic_write_json(paths.directory, [{
                "section_id": 1,
                "catalog_source": "ocr_segmented",
                "segmentation_schema_version": (
                    NO_CATALOG_SEGMENTATION_SCHEMA_VERSION
                ),
            }])
            self.assertTrue(_pre_review_cache_is_reusable(paths))


if __name__ == "__main__":
    unittest.main()
