import unittest

from reviewers.document_mapping import (
    deterministic_document_section_map,
    load_compatible_document_type_groups,
    normalize_document_section_map,
    normalize_document_section_map_lenient,
)


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

        self.assertEqual(
            set(groups["evidence_materials"]),
            {
                "当事人提交的复制件等证据材料",
                "行政执法人员采集制作的证据材料",
                "行政相对人的身份证明或主体资格证明材料",
                "证件信息",
                "取证记录信息",
                "热敏纸质传真件等难以长期保存的书证",
                "代替书证原件的复印件、照片、节录本",
                "有关部门保管的书证复制件、影印件、抄录件",
                "计算机数据、录音、录像、图片等证据材料",
                "域外及港澳台形成的证据",
            },
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
            self.directory,
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

    def test_identity_mapping_excludes_officer_certificates(self) -> None:
        mapping = normalize_document_section_map(
            {"证件信息": [21, 38, 39], "法律职业资格证书": [39]},
            self.directory,
        )

        self.assertEqual(mapping["证件信息"], [21])
        self.assertEqual(mapping["法律职业资格证书"], [39])

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

    def test_evidence_mapping_excludes_identity_and_officer_materials(self) -> None:
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
            ["取证记录信息", "证件信息", "执法证信息"],
            directory,
        )

        self.assertEqual(mapping["取证记录信息"], [1])
        self.assertEqual(mapping["证件信息"], [2])
        self.assertEqual(mapping["执法证信息"], [3])

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
                "放弃陈述、申辩申明",
                "整改复查意见书",
            ],
            directory,
        )

        self.assertNotIn("行政强制措施决定书", mapping)
        self.assertEqual(mapping["催告书"], [1])
        self.assertEqual(mapping["行政强制措施处理决定书"], [2])
        self.assertEqual(mapping["放弃陈述、申辩申明"], [3])
        self.assertEqual(mapping["整改复查意见书"], [4])


if __name__ == "__main__":
    unittest.main()
