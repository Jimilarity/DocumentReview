import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from pydantic import ValidationError

from knowledge_retrieval.city_management_discretion import (
    CityManagementDiscretionCatalog,
    CityManagementDiscretionRecord,
    GeneratedItemCitations,
    ScoredDiscretionRecord,
    rank_scored_records,
)
from knowledge_retrieval.common import (
    CitationRecord,
    ExactLegalCitationLookup,
    LegalCitation,
    LegalCitationExtraction,
    citation_matches,
    clear_legal_citation_extraction_cache,
    extract_legal_citations,
)
from tools.build_city_management_discretion_catalog import (
    _enrich_catalog,
    _prompt_for_item,
    _validate_generated,
)


def citation(
    *,
    law_name="《深圳市生活垃圾分类管理条例》",
    article="66",
    paragraph=None,
    item=None,
) -> LegalCitation:
    return LegalCitation(
        law_name=law_name,
        article=article,
        paragraph=paragraph,
        item=item,
        subitem=None,
        content=None,
    )


def source_item() -> dict:
    return {
        "id": "sz_cg_2023_041",
        "序号": 41,
        "法规规章": "《深圳市生活垃圾分类管理条例》",
        "违法行为": "未按规定分类投放生活垃圾且拒不改正",
        "设定依据": (
            "第六十六条：拒不改正的，对个人处五十元罚款；"
            "情节严重的，对个人处二百元罚款。"
        ),
        "裁量分组": [
            {
                "分组序号": 1,
                "处罚种类及幅度": "对个人处二百元罚款",
                "裁量规则": [],
            }
        ],
        "检索元数据": {
            "schema_version": 1,
            "处罚依据": [citation().model_dump(mode="json")],
        },
    }


def catalog_record(
    item_id="sz_cg_2023_041",
    sequence=41,
) -> CityManagementDiscretionRecord:
    item = source_item()
    item["id"] = item_id
    item["序号"] = sequence
    return CityManagementDiscretionRecord(
        item_id=item_id,
        sequence=sequence,
        citations=(citation(),),
        source_item=item,
    )


class LegalCitationProtocolTest(unittest.TestCase):
    def test_accepts_only_canonical_model_output(self) -> None:
        self.assertEqual(citation().article, "66")
        self.assertEqual(citation(article="66-1").article, "66-1")

        invalid_values = [
            {"law_name": "深圳市生活垃圾分类管理条例"},
            {"article": "第六十六条"},
            {"article": "六十六"},
            {"article": 66},
            {"paragraph": "第一款"},
            {"paragraph": "01"},
        ]
        for changes in invalid_values:
            values = {
                "law_name": "《深圳市生活垃圾分类管理条例》",
                "article": "66",
                "paragraph": None,
                "item": None,
                "subitem": None,
                "content": None,
            }
            values.update(changes)
            with self.subTest(changes=changes), self.assertRaises(
                ValidationError
            ):
                LegalCitation.model_validate(values)

    def test_optional_lower_levels_use_direct_exact_comparison(self) -> None:
        self.assertTrue(
            citation_matches(
                citation(paragraph=None),
                citation(paragraph="2"),
            )
        )
        self.assertFalse(
            citation_matches(
                citation(paragraph="1"),
                citation(paragraph="2"),
            )
        )

    def test_generic_lookup_is_independent_of_record_type(self) -> None:
        lookup = ExactLegalCitationLookup(
            [
                CitationRecord(
                    record_id="record-1",
                    citations=(citation(),),
                    value={"arbitrary": "value"},
                )
            ]
        )
        self.assertEqual(
            lookup.search([citation()]),
            [{"arbitrary": "value"}],
        )

    def test_extraction_defaults_missing_result_arrays(self) -> None:
        empty = LegalCitationExtraction.model_validate(
            {"citations": []}
        )
        self.assertEqual(empty.imposed_penalties, [])


class LegalCitationExtractionCacheTest(
    unittest.IsolatedAsyncioTestCase
):
    async def asyncSetUp(self) -> None:
        clear_legal_citation_extraction_cache()

    async def test_same_document_ocr_is_only_extracted_once(self) -> None:
        response = Mock(
            content=json.dumps(
                {
                    "citations": [
                        citation().model_dump(mode="json")
                    ],
                    "imposed_penalties": [],
                },
                ensure_ascii=False,
            )
        )
        bound_model = Mock()
        bound_model.ainvoke = AsyncMock(return_value=response)
        model = Mock()
        model.bind.return_value = bound_model
        with patch(
            "knowledge_retrieval.common.legal_citations.build_text_model",
            return_value=model,
        ):
            first = await extract_legal_citations("相同OCR")
            second = await extract_legal_citations("相同OCR")

        self.assertEqual(first, second)
        bound_model.ainvoke.assert_awaited_once()

    async def test_invalid_output_is_retried_not_normalized(self) -> None:
        invalid = Mock(
            content=json.dumps(
                {
                    "citations": [
                        {
                            **citation().model_dump(mode="json"),
                            "article": "第六十六条",
                        }
                    ],
                    "imposed_penalties": [],
                },
                ensure_ascii=False,
            )
        )
        valid = Mock(
            content=json.dumps(
                {
                    "citations": [
                        citation().model_dump(mode="json")
                    ],
                    "imposed_penalties": [],
                },
                ensure_ascii=False,
            )
        )
        bound_model = Mock()
        bound_model.ainvoke = AsyncMock(side_effect=[invalid, valid])
        model = Mock()
        model.bind.return_value = bound_model
        with patch(
            "knowledge_retrieval.common.legal_citations.build_text_model",
            return_value=model,
        ):
            result = await extract_legal_citations("需要重试的OCR")

        self.assertEqual(result.citations[0].article, "66")
        self.assertEqual(bound_model.ainvoke.await_count, 2)


class DiscretionCatalogTest(unittest.TestCase):
    def write_catalog(self, path: Path, item=None) -> None:
        path.write_text(
            json.dumps(
                {"metadata": {}, "items": [item or source_item()]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def test_catalog_reads_complete_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            self.write_catalog(path)
            matches = CityManagementDiscretionCatalog(path).search(
                [citation()]
            )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].source_item["序号"], 41)
        self.assertIn("裁量分组", matches[0].source_item)

    def test_lookup_distinguishes_law_and_article_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            self.write_catalog(path)
            catalog = CityManagementDiscretionCatalog(path)
            missing_article = catalog.lookup_citations(
                [citation(article="65")]
            )[0]
            uncovered_law = catalog.lookup_citations(
                [
                    citation(
                        law_name="《中华人民共和国安全生产法》",
                        article="1",
                    )
                ]
            )[0]

        self.assertTrue(missing_article.law_covered)
        self.assertFalse(missing_article.article_covered)
        self.assertFalse(uncovered_law.law_covered)

    def test_missing_metadata_skips_unusable_record(self) -> None:
        item = source_item()
        item.pop("检索元数据")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            self.write_catalog(path, item)
            result = CityManagementDiscretionCatalog(path).search(
                [citation()]
            )
        self.assertEqual(result, [])


class DiscretionRerankTest(unittest.TestCase):
    def scored(self, first: float, second: float):
        return [
            ScoredDiscretionRecord(
                catalog_record(),
                first,
            ),
            ScoredDiscretionRecord(
                catalog_record("sz_cg_2023_042", 42),
                second,
            ),
        ]

    def test_scored_records_are_sorted_without_auto_selection(self) -> None:
        result = rank_scored_records(self.scored(0.61, 0.80))
        self.assertEqual(
            [item.record.sequence for item in result.ranked],
            [42, 41],
        )
        self.assertFalse(hasattr(result, "selected"))


class DiscretionPreprocessingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source_item = source_item()
        self.source_item.pop("检索元数据")

    def generated(self, **changes) -> GeneratedItemCitations:
        values = {
            "item_id": "sz_cg_2023_041",
            "sequence": 41,
            "citations": [citation()],
        }
        values.update(changes)
        return GeneratedItemCitations(**values)

    def test_preprocessing_only_sends_source_basis(self) -> None:
        prompt = _prompt_for_item(self.source_item)
        source_payload = prompt.split("<source_item>", 1)[1]
        self.assertIn('"设定依据"', source_payload)
        self.assertIn('"法规规章"', source_payload)
        self.assertNotIn('"裁量分组"', source_payload)

    def test_validates_exact_law_name(self) -> None:
        generated = self.generated()
        self.assertIs(
            _validate_generated(
                self.source_item,
                generated,
            ),
            generated,
        )
        with self.assertRaisesRegex(ValueError, "逐字等于"):
            _validate_generated(
                self.source_item,
                self.generated(
                    citations=[
                        citation(law_name="《其他条例》")
                    ]
                ),
            )

    def test_enrichment_adds_metadata_without_mutating_source(self) -> None:
        source = {"metadata": {}, "items": [self.source_item]}
        enriched = _enrich_catalog(
            source,
            {self.source_item["id"]: self.generated()},
        )
        self.assertNotIn("检索元数据", source["items"][0])
        self.assertEqual(
            enriched["items"][0]["检索元数据"]["处罚依据"][0][
                "article"
            ],
            "66",
        )


if __name__ == "__main__":
    unittest.main()
