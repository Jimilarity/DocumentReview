import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from external_knowledge.longhua_subdistrict_penalty import (
    CatalogMatch,
    LonghuaPenaltyCatalogIndex,
    retrieve_longhua_subdistrict_penalty_items,
)
from external_knowledge.models import KnowledgeContext
from knowledge_retrieval.case_routing import (
    PenaltyCaseKind,
    classify_penalty_case,
)


def build_context(
    *,
    case_number="深龙华民治综行罚决字〔2026〕1号",
    case_reason="未按规定分类投放生活垃圾且拒不改正案",
):
    return KnowledgeContext(
        metadata={"案号": case_number, "案由": case_reason},
        dir_info=[],
        rule={"序号": 101},
        review_item={"审查事项": "核查执法主体是否合法"},
        document_name="行政处罚决定书",
        section_id=10,
        section_ocr="行政处罚决定书OCR",
    )


class PenaltyCaseClassificationTest(unittest.TestCase):
    def test_classifies_supported_and_skipped_case_numbers(self) -> None:
        self.assertEqual(
            classify_penalty_case("深龙华民治综行罚决字〔2026〕1号"),
            PenaltyCaseKind.SUBDISTRICT,
        )
        self.assertEqual(
            classify_penalty_case("深龙华大浪综行简罚决字第1号"),
            PenaltyCaseKind.SUBDISTRICT,
        )
        self.assertEqual(
            classify_penalty_case("深龙华应急罚〔2026〕1号"),
            PenaltyCaseKind.EMERGENCY,
        )
        self.assertEqual(
            classify_penalty_case("深龙华市监罚〔2026〕1号"),
            PenaltyCaseKind.MARKET_REGULATION,
        )


class LonghuaPowerKnowledgeFunctionTest(unittest.IsolatedAsyncioTestCase):
    async def test_non_subdistrict_case_skips_index(self) -> None:
        with patch(
            "external_knowledge.longhua_subdistrict_penalty.knowledge.get_default_index"
        ) as get_index:
            result = await retrieve_longhua_subdistrict_penalty_items(
                build_context(case_number="深龙华应急罚〔2026〕1号")
            )

        self.assertEqual(result, [])
        get_index.assert_not_called()

    async def test_uses_metadata_case_reason_and_filters_internal_fields(
        self,
    ) -> None:
        index = Mock()
        index.search.return_value = [
            CatalogMatch(
                item_name="对未按规定分类投放生活垃圾且拒不改正的处罚",
                related_district_authority=(
                    "深圳市龙华区城市管理和综合执法局"
                ),
                implementation_scope=["民治街道办事处", "大浪街道办事处"],
                remark=None,
                score=0.91,
            )
        ]
        with patch(
            "external_knowledge.longhua_subdistrict_penalty.knowledge.get_default_index",
            return_value=index,
        ), patch(
            "external_knowledge.longhua_subdistrict_penalty.knowledge.extract_case_reason",
            new=AsyncMock(),
        ) as extractor:
            result = await retrieve_longhua_subdistrict_penalty_items(
                build_context()
            )

        extractor.assert_not_awaited()
        index.search.assert_called_once()
        self.assertEqual(len(result), 1)
        self.assertIn("承接事项", result[0].content)
        self.assertNotIn("向量", result[0].content)
        self.assertNotIn("相似度", result[0].content)
        self.assertIn("事项名称", result[0].content)
        self.assertIn("实施范围", result[0].content)
        self.assertNotIn("basic_code", result[0].content)
        self.assertNotIn("id", result[0].content)

    async def test_missing_metadata_case_reason_uses_decision_ocr(self) -> None:
        index = Mock()
        index.search.return_value = []
        with patch(
            "external_knowledge.longhua_subdistrict_penalty.knowledge.get_default_index",
            return_value=index,
        ), patch(
            "external_knowledge.longhua_subdistrict_penalty.knowledge.extract_case_reason",
            new=AsyncMock(return_value="张三未分类投放生活垃圾案"),
        ) as extractor:
            result = await retrieve_longhua_subdistrict_penalty_items(
                build_context(case_reason=None)
            )

        self.assertEqual(result, [])
        extractor.assert_awaited_once_with("行政处罚决定书OCR")
        self.assertEqual(
            index.search.call_args.args[0],
            "张三未分类投放生活垃圾案",
        )


class LonghuaPowerIndexLoadingTest(unittest.TestCase):
    def test_missing_local_model_uses_manifest_model_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index_directory = Path(directory)
            (index_directory / "manifest.json").write_text(
                json.dumps(
                    {
                        "model": "BAAI/bge-small-zh-v1.5",
                        "query_prefix": "",
                    }
                ),
                encoding="utf-8",
            )
            (index_directory / "items.json").write_text(
                json.dumps([{"item_name": "测试事项"}]),
                encoding="utf-8",
            )
            np.save(
                index_directory / "embeddings.npy",
                np.zeros((1, 2), dtype=np.float32),
            )
            with patch(
                "sentence_transformers.SentenceTransformer",
                return_value=Mock(),
            ) as model_class:
                loaded = LonghuaPenaltyCatalogIndex(
                    index_directory
                )._load()

        self.assertTrue(loaded)
        model_class.assert_called_once_with("BAAI/bge-small-zh-v1.5")


if __name__ == "__main__":
    unittest.main()
