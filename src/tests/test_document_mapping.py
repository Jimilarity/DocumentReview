import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents import DocumentSectionMappingResult
from reviewers.document_mapping import (
    deterministic_document_section_map,
    document_type_definitions,
    directory_info_for_mapping,
    load_compatible_document_type_groups,
    normalize_document_section_map,
    normalize_document_section_map_lenient,
)
from reviewers.document_preparation import DocumentReviewPreparationService


class DocumentMappingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = [
            {"section_id": 8, "section_name": "行政处罚决定书"},
            {"section_id": 3, "section_name": "询问笔录"},
            {"section_id": 4, "section_name": "询问笔录"},
            {"section_id": 21, "section_name": "李中正身份证复印件"},
            {"section_id": 38, "section_name": "执法证复印件"},
            {"section_id": 39, "section_name": "法律职业资格证书复印件"},
        ]

    def test_one_type_may_map_to_multiple_sections_in_directory_order(
        self,
    ) -> None:
        mapping = normalize_document_section_map(
            {"询问笔录": [4, 3, 4]},
            self.directory,
        )

        self.assertEqual(mapping, {"询问笔录": [3, 4]})

    def test_one_section_cannot_map_to_multiple_rule_types(self) -> None:
        with self.assertRaisesRegex(ValueError, "不能同时映射"):
            normalize_document_section_map(
                {
                    "行政处罚决定书": [8],
                    "处罚决定文书": [8],
                },
                self.directory,
            )

    def test_config_contains_all_407_to_414_evidence_types(self) -> None:
        groups = load_compatible_document_type_groups()

        self.assertTrue(
            {
                "当事人提交的复制件等证据材料",
                "行政执法人员采集制作的证据材料",
                "证件信息",
                "取证记录",
                "热敏纸质传真件等难以长期保存的书证",
                "代替书证原件的复印件、照片、节录本",
                "有关部门保管的书证复制件、影印件、抄录件",
                "计算机数据、录音、录像、图片等证据材料",
                "域外及港澳台形成的证据",
            } <= set(groups["evidence_materials"]),
        )
        self.assertTrue(
            {"证件信息", "法定代表人身份证明书", "营业执照", "身份证"}
            <= set(groups["evidence_materials"])
        )

    def test_compatible_evidence_attributes_may_share_section(self) -> None:
        document_types = load_compatible_document_type_groups()[
            "evidence_materials"
        ]
        expected = {
            document_type: [3] for document_type in document_types
        }

        mapping = normalize_document_section_map(
            expected,
            [
                {
                    "section_id": 3,
                    "section_name": "当事人身份证照片电子数据",
                }
            ],
        )

        self.assertEqual(mapping, expected)

    def test_compatible_type_cannot_share_with_outside_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "不能同时映射"):
            normalize_document_section_map(
                {
                    "当事人提交的复制件等证据材料": [3],
                    "调查询问笔录": [3],
                },
                self.directory,
            )

    def test_lenient_mapping_keeps_valid_types_and_drops_conflicts(self) -> None:
        mapping, dropped = normalize_document_section_map_lenient(
            {
                "行政处罚决定书": [8],
                "处罚决定文书": [8],
                "询问笔录": [3],
            },
            self.directory,
            preferred_document_types=["行政处罚决定书"],
        )

        self.assertEqual(
            mapping,
            {"行政处罚决定书": [8], "询问笔录": [3]},
        )
        self.assertIn("处罚决定文书", dropped)

    def test_lenient_mapping_drops_only_conflicting_section_from_aggregate(
        self,
    ) -> None:
        directory = [
            {"section_id": 15, "section_name": "营业执照"},
            {"section_id": 18, "section_name": "授权委托书"},
        ]

        mapping, dropped = normalize_document_section_map_lenient(
            {
                "营业执照": [15],
                "授权委托书": [18],
                "证件信息": [15, 18],
            },
            directory,
            preferred_document_types=["营业执照", "授权委托书"],
        )

        self.assertEqual(mapping["证件信息"], [15])
        self.assertNotIn("证件信息", dropped)

    def test_overlapping_compatibility_groups_form_valid_tag_combination(
        self,
    ) -> None:
        directory = [
            {"section_id": 18, "section_name": "授权委托书"},
        ]

        mapping = normalize_document_section_map(
            {
                "授权委托书": [18],
                "当事人提交的复制件等证据材料": [18],
                "代替书证原件的复印件、照片、节录本": [18],
            },
            directory,
        )

        self.assertEqual(len(mapping), 3)

    def test_logical_legal_summary_view_may_share_base_document(self) -> None:
        directory = [
            {"section_id": 9, "section_name": "责令改正违法行为通知书"},
        ]

        mapping = normalize_document_section_map(
            {
                "责令改正通知书": [9],
                "法律适用告知（责令改正通知书）": [9],
            },
            directory,
        )

        self.assertEqual(len(mapping), 2)

    def test_mapping_validation_does_not_require_known_title_keywords(self) -> None:
        directory = [
            {
                "section_id": 1,
                "section_name": "本地市场主体设立档案摘录（甲式）",
            },
        ]

        mapping = normalize_document_section_map(
            {"证件信息": [1]},
            directory,
        )

        self.assertEqual(mapping["证件信息"], [1])

    def test_mapping_validation_does_not_use_document_title_exclusion_list(
        self,
    ) -> None:
        directory = [
            {
                "section_id": 1,
                "section_name": "行政相对人身份核验附件（地方系统导出）",
            },
        ]

        mapping, dropped = normalize_document_section_map_lenient(
            {"证件信息": [1]},
            directory,
        )

        self.assertEqual(mapping["证件信息"], [1])
        self.assertEqual(dropped, {})

    def test_document_definitions_are_generated_from_rule_semantics(self) -> None:
        definitions = document_type_definitions(["证件信息"])

        self.assertIn("证件信息", definitions)
        self.assertIn(
            "行政相对人的身份证明或主体资格证明材料",
            definitions["证件信息"]["rule_source_labels"],
        )
        self.assertTrue(definitions["证件信息"]["review_targets"])

    def test_unknown_section_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "无效 section_id"):
            normalize_document_section_map(
                {"行政处罚决定书": [99]},
                self.directory,
            )

    def test_empty_mapping_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "未映射到章节"):
            normalize_document_section_map(
                {"结案审批表": []},
                self.directory,
            )

    def test_deterministic_mapping_handles_local_document_aliases(self) -> None:
        directory = [
            {
                "section_id": 1,
                "section_name": "现场检查记录",
                "normalized_document_type": "现场检查记录",
                "section_kind": "case_document",
            },
            {
                "section_id": 2,
                "section_name": "责令限期整改指令书",
                "normalized_document_type": "责令限期整改指令书",
                "section_kind": "case_document",
            },
            {
                "section_id": 3,
                "section_name": "文书送达回执",
                "normalized_document_type": "文书送达回执",
                "section_kind": "case_document",
            },
        ]

        mapping = deterministic_document_section_map(
            ["现场检查（勘验）笔录", "责令改正通知书", "送达回证"],
            directory,
        )

        self.assertEqual(
            mapping,
            {
                "现场检查（勘验）笔录": [1],
                "责令改正通知书": [2],
                "送达回证": [3],
            },
        )

    def test_deterministic_mapping_does_not_guess_aggregate_types(self) -> None:
        directory = [
            {
                "section_id": 1,
                "section_name": "当事人现场照片证据材料",
                "normalized_document_type": "现场照片证据材料",
                "section_kind": "evidence_material",
                "material_type": "现场照片证据材料",
            },
            {
                "section_id": 2,
                "section_name": "当事人身份证信息",
                "normalized_document_type": "身份证信息",
                "section_kind": "evidence_material",
                "material_type": "身份证信息",
                "subject_role": "当事人",
            },
            {
                "section_id": 3,
                "section_name": "执法人员行政执法证复印件",
                "normalized_document_type": "行政执法证复印件",
                "section_kind": "evidence_material",
                "material_type": "行政执法证复印件",
            },
        ]

        mapping = deterministic_document_section_map(
            ["取证记录", "证件信息"],
            directory,
        )

        self.assertEqual(mapping, {})

    def test_deterministic_mapping_covers_extended_rule_documents(self) -> None:
        directory = [
            {"section_id": 1, "section_name": "履行行政决定催告书"},
            {"section_id": 2, "section_name": "行政强制措施处理决定书"},
            {"section_id": 3, "section_name": "放弃陈述、申辩声明"},
            {"section_id": 4, "section_name": "整改复查意见书"},
        ]

        mapping = deterministic_document_section_map(
            [
                "催告书",
                "行政强制措施决定书",
                "行政强制措施处理决定书",
                "放弃陈述、申辩声明",
                "整改复查意见书",
            ],
            directory,
        )

        self.assertNotIn("行政强制措施决定书", mapping)
        self.assertEqual(mapping["催告书"], [1])
        self.assertEqual(mapping["行政强制措施处理决定书"], [2])
        self.assertEqual(mapping["放弃陈述、申辩声明"], [3])
        self.assertEqual(mapping["整改复查意见书"], [4])

    def test_specific_identity_documents_may_share_with_aggregate_type(
        self,
    ) -> None:
        directory = [
            {"section_id": 15, "section_name": "营业执照"},
            {"section_id": 16, "section_name": "法定代表人身份证明书"},
            {"section_id": 17, "section_name": "中华人民共和国居民身份证"},
        ]
        mapping = deterministic_document_section_map(
            ["营业执照", "法定代表人身份证明书", "身份证"],
            directory,
        )
        mapping["证件信息"] = [15, 16, 17]

        self.assertEqual(
            normalize_document_section_map(mapping, directory),
            {
                "营业执照": [15],
                "法定代表人身份证明书": [16],
                "身份证": [17],
                "证件信息": [15, 16, 17],
            },
        )

    def test_mapping_directory_includes_bounded_ocr_hint(self) -> None:
        mapped = directory_info_for_mapping(
            [
                {"section_id": 1, "section_name": "材料A", "section_page": 0},
                {"section_id": 2, "section_name": "材料B", "section_page": 1},
            ],
            [
                {"document_content": "第一页行政处罚决定书正文"},
                {"document_content": "第二页询问笔录正文"},
            ],
        )

        self.assertEqual(mapped[0]["ocr_text_hint"], "第一页行政处罚决定书正文")
        self.assertEqual(mapped[1]["ocr_text_hint"], "第二页询问笔录正文")

    def test_maps_derived_legal_summary_by_previous_document(self) -> None:
        directory = [
            {"section_id": 1, "section_name": "行政处罚决定书"},
            {"section_id": 2, "section_name": "法律法规摘要"},
            {"section_id": 3, "section_name": "行政处罚（听证）告知书"},
            {"section_id": 4, "section_name": "法律法规摘要"},
        ]

        mapping = deterministic_document_section_map(
            [
                "法律适用告知（行政处罚决定书）",
                "法律适用告知（行政处罚（听证）告知书）",
            ],
            directory,
        )

        self.assertEqual(
            mapping,
            {
                "法律适用告知（行政处罚决定书）": [2],
                "法律适用告知（行政处罚（听证）告知书）": [4],
            },
        )

    def test_one_certificate_section_can_supply_two_officer_slots(self) -> None:
        directory = [
            {"section_id": 6, "section_name": "中华人民共和国行政执法证"}
        ]

        mapping = deterministic_document_section_map(
            ["执法证1", "执法证2"],
            directory,
        )

        self.assertEqual(mapping, {"执法证1": [6], "执法证2": [6]})
        self.assertEqual(
            normalize_document_section_map(mapping, directory),
            mapping,
        )


class DocumentPreparationConflictRecoveryTest(
    unittest.IsolatedAsyncioTestCase
):
    async def test_specific_and_aggregate_identity_mapping_needs_no_retry(
        self,
    ) -> None:
        class FakeAgents:
            def __init__(self) -> None:
                self.calls = 0

            def build_task_prompt(self, task_name: str, **kwargs) -> str:
                return json.dumps(kwargs, ensure_ascii=False)

            async def ainvoke_document_section_mapper(
                self,
                prompt: str,
                recursion_limit: int,
            ) -> DocumentSectionMappingResult:
                self.calls += 1
                return DocumentSectionMappingResult.model_validate(
                    {
                        "mappings": [
                            {
                                "document_name": "证件信息",
                                "section_ids": [16, 17],
                            },
                            {
                                "document_name": "法定代表人身份证明书",
                                "section_ids": [16],
                            },
                            {
                                "document_name": "未请求的说明材料",
                                "section_ids": [16],
                            },
                        ]
                    }
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = SimpleNamespace(
                metadata=root / "meta_info.json",
                directory=root / "dir_info.json",
                ocr_results=root / "ocr_results.json",
                structured_fields=root / "structured_fields.json",
            )
            paths.metadata.write_text("{}", encoding="utf-8")
            paths.directory.write_text(
                json.dumps(
                    [
                        {
                            "section_id": 16,
                            "section_name": "法定代表人身份证明书",
                            "section_page": 0,
                            "section_end_page": 1,
                        },
                        {
                            "section_id": 17,
                            "section_name": "本地市场主体设立档案摘录（甲式）",
                            "section_page": 1,
                            "section_end_page": 2,
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            paths.ocr_results.write_text(
                json.dumps(
                    [
                        {"image_index": 0, "document_content": "身份证明正文"},
                        {"image_index": 1, "document_content": "主体登记正文"},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            agents = FakeAgents()
            with patch(
                "reviewers.document_preparation.get_cache_paths",
                return_value=paths,
            ), patch(
                "reviewers.document_preparation.DocumentMappingAgents",
                return_value=agents,
            ):
                mapping = await DocumentReviewPreparationService(
                    root / "case.pdf",
                    ["证件信息", "法定代表人身份证明书"],
                    settings=SimpleNamespace(agent_recursion_limit=20),
                ).prepare()

        self.assertEqual(agents.calls, 2)
        self.assertEqual(
            mapping,
            {
                "法定代表人身份证明书": [16],
                "证件信息": [16, 17],
            },
        )


if __name__ == "__main__":
    unittest.main()
