import json
import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))

from constants import DocumentType
from rules.filtering import (
    _select_rules,
    context_free_document_names,
    context_sensitive_document_names,
    filter_human_support_rules,
    filter_context_free_rules,
    filter_context_sensitive_rules,
    human_support_document_names,
    retrieval_enhancement_module_names,
)
from rules.rule_set import RuleSetBuilder
from rules.normalization import (
    canonical_document_type,
    load_rule_aliases,
    normalize_rules,
)
from rules.rule_type import decode_rule_type, parse_rule_type_value


class RuleFilteringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.all_rules = json.loads(
            (PROJECT_ROOT / "data" / "all_rules.json").read_text(
                encoding="utf-8"
            )
        )

    def all_rule_items(self):
        return [
            rule
            for sections in self.all_rules.values()
            for rules in sections.values()
            for rule in rules
        ]

    def test_rule_data_uses_symmetric_executor_independent_schema(self) -> None:
        required_top_level_keys = {
            "序号",
            "上下文无关审查事项",
            "上下文相关审查事项",
            "备注",
            "检索增强",
        }
        optional_top_level_keys = {"评分细则"}
        context_item_keys = {
            "任务",
            "字段",
            "评查类别",
            "审查事项",
            "评查说明",
            "外部知识",
            "法条检索查询",
            "审查提示",
        }
        self.assertEqual(
            set(self.all_rules),
            {"合法性标准", "规范性标准", "加减分项", "附加项"},
        )
        for rule in self.all_rule_items():
            self.assertTrue(required_top_level_keys.issubset(rule))
            self.assertTrue(
                set(rule).issubset(
                    required_top_level_keys | optional_top_level_keys
                )
            )
            self.assertNotIn("executor", rule)
            self.assertIsInstance(rule["上下文无关审查事项"], dict)
            self.assertIsInstance(rule["上下文相关审查事项"], list)
            for name, document_rule in rule[
                "上下文无关审查事项"
            ].items():
                self.assertIsInstance(name, str)
                self.assertTrue(name)
                self.assertTrue(
                    {"审查事项", "评查说明"}.issubset(document_rule)
                )
                self.assertTrue(
                    set(document_rule).issubset(
                        {
                            "审查事项",
                            "评查说明",
                            "外部知识",
                            "法条检索查询",
                        }
                    )
                )
                if "外部知识" in document_rule:
                    self.assertIsInstance(document_rule["外部知识"], list)
                    self.assertTrue(
                        all(
                            isinstance(name, str) and name.strip()
                            for name in document_rule["外部知识"]
                        )
                    )
            for item in rule["上下文相关审查事项"]:
                self.assertTrue(set(item).issubset(context_item_keys))
                self.assertIsInstance(item["字段"], dict)
                if "外部知识" in item:
                    self.assertIsInstance(item["外部知识"], list)
                    self.assertTrue(
                        all(
                            isinstance(name, str) and name.strip()
                            for name in item["外部知识"]
                        )
                    )
            retrieval_modules = rule["检索增强"]
            self.assertIsInstance(retrieval_modules, list)
            self.assertTrue(
                all(
                    isinstance(name, str) and name.strip()
                    for name in retrieval_modules
                )
            )

    def test_configured_context_tasks_are_supported(self) -> None:
        configured_tasks = {
            item["任务"]
            for rule in self.all_rule_items()
            for item in rule["上下文相关审查事项"]
            if item["任务"]
        }
        self.assertEqual(
            configured_tasks,
            {
                "一致性核查",
                "事实清楚、证据充分",
                "适用法律准确",
                "程序合法",
            },
        )

    def test_rule_type_parser_accepts_binary_forms(self) -> None:
        self.assertEqual(parse_rule_type_value("01010100"), 0b01010100)
        self.assertEqual(parse_rule_type_value("0b01010100"), 0b01010100)

    def test_penalty_procedures_are_selected_exactly(self) -> None:
        rules = self.all_rules["合法性标准"]
        simple = _select_rules(rules, decode_rule_type(0b10010000))
        ordinary = _select_rules(rules, decode_rule_type(0b10010100))
        self.assertTrue(all(rule["备注"] in ("", "简易程序") for rule in simple))
        self.assertTrue(all(rule["备注"] in ("", "普通程序") for rule in ordinary))

    def test_two_executors_collect_different_document_dependencies(self) -> None:
        hybrid = {
            "序号": 1,
            "上下文无关审查事项": {
                "行政处罚决定书": {"审查事项": "审查决定书"}
            },
            "上下文相关审查事项": [
                {
                    "任务": "一致性核查",
                    "字段": {
                        "立案审批表": [
                            {"field": "执法主体名称", "required": True}
                        ]
                    },
                }
            ],
        }
        self.assertEqual(
            context_free_document_names([hybrid]),
            ["行政处罚决定书"],
        )
        self.assertEqual(
            context_sensitive_document_names([hybrid]),
            ["立案审批表"],
        )

    def test_missing_document_only_removes_that_executor_source(self) -> None:
        rule = {
            "序号": 1,
            "上下文无关审查事项": {
                "行政处罚决定书": {"审查事项": "审查决定书"}
            },
            "上下文相关审查事项": [
                {
                    "任务": "一致性核查",
                    "字段": {
                        "立案审批表": [
                            {"field": "执法主体名称", "required": True}
                        ]
                    },
                }
            ],
        }
        presence = {"行政处罚决定书": False, "立案审批表": True}
        self.assertEqual(filter_context_free_rules([rule], presence), [])
        sensitive_rules = filter_context_sensitive_rules([rule], presence)
        self.assertEqual(len(sensitive_rules), 1)
        self.assertEqual(
            set(sensitive_rules[0]["上下文相关审查事项"][0]["字段"]),
            {"立案审批表"},
        )

    def test_context_free_receipt_does_not_keep_rule_without_ordinary_document(
        self,
    ) -> None:
        rule = {
            "序号": 1,
            "上下文无关审查事项": {
                "行政处罚决定书": {"审查事项": "决定书"},
                "送达回证": {"审查事项": "回证"},
            },
            "上下文相关审查事项": [],
        }

        result = filter_context_free_rules(
            [rule],
            {"行政处罚决定书": False, "送达回证": True},
        )

        self.assertEqual(result, [])

    def test_context_free_receipt_is_retained_when_ordinary_document_exists(
        self,
    ) -> None:
        rule = {
            "序号": 1,
            "上下文无关审查事项": {
                "行政处罚决定书": {"审查事项": "决定书"},
                "送达回证": {"审查事项": "回证"},
            },
            "上下文相关审查事项": [],
        }

        result = filter_context_free_rules(
            [rule],
            {"行政处罚决定书": True, "送达回证": True},
        )

        self.assertEqual(
            set(result[0]["上下文无关审查事项"]),
            {"行政处罚决定书", "送达回证"},
        )

    def test_builder_injects_executor_rule_filter(self) -> None:
        candidate_rules = (
            RuleSetBuilder(self.all_rules)
            .for_rule_type(0b01010100)
            .build()
            .rules
        )
        document_presence = {
            name: True
            for name in context_free_document_names(candidate_rules)
        }
        rule_set = (
            RuleSetBuilder(self.all_rules)
            .for_executor(
                0b01010100,
                rule_filter=filter_context_free_rules,
                document_presence=document_presence,
            )
            .build()
        )

        self.assertTrue(rule_set.rules)
        self.assertTrue(
            all(rule["上下文无关审查事项"] for rule in rule_set.rules)
        )
        self.assertEqual(set(vars(rule_set)), {"rules"})

    def test_additional_review_uses_additional_group(self) -> None:
        rule_set = (
            RuleSetBuilder(self.all_rules)
            .for_rule_type(0b11110100)
            .build()
        )

        rule_ids = [rule["序号"] for rule in rule_set.rules]
        self.assertEqual(len(rule_ids), len(set(rule_ids)))
        self.assertEqual(rule_ids.count(601), 1)

    def test_adjustment_rules_retain_entries_without_legacy_fields(self) -> None:
        adjustment_rules = self.all_rules["加减分项"]
        self.assertEqual(
            [rule["序号"] for rule in adjustment_rules["通用"]],
            [501, 502, 503, 504, 505, 506, 601],
        )
        self.assertEqual(
            [rule["序号"] for rule in adjustment_rules["特殊"]],
            [507, 508, 509],
        )

        forbidden_fields = {"总分值", "分项分值", "评查方式"}
        for rule in [
            *adjustment_rules["通用"],
            *adjustment_rules["特殊"],
        ]:
            self.assertTrue(forbidden_fields.isdisjoint(rule))
            for item in rule["上下文无关审查事项"].values():
                self.assertTrue(forbidden_fields.isdisjoint(item))
            for item in rule["上下文相关审查事项"]:
                self.assertTrue(forbidden_fields.isdisjoint(item))

    def test_standardization_rules_are_selected_before_executor_filter(self) -> None:
        selected = _select_rules(
            self.all_rules["规范性标准"],
            decode_rule_type(0b01010000),
        )
        self.assertTrue(selected)

    def test_rules_use_canonical_document_types(self) -> None:
        normalized_rules = normalize_rules(
            self.all_rule_items(),
            load_rule_aliases(),
        )
        document_names = {
            name
            for rule in normalized_rules
            for name in rule["上下文无关审查事项"]
        }
        field_document_names = {
            name
            for rule in normalized_rules
            for item in rule["上下文相关审查事项"]
            for name in item["字段"]
        }
        all_document_names = document_names | field_document_names
        aliases = {
            "当场行政处罚决定书",
            "（当场）行政处罚决定书",
            "结案表",
            "结案（审批）表",
            "责令改正违法行为决定书(通知书)",
            "责令改正违法行为决定书（通知书）",
            "现场检查笔录",
        }
        self.assertTrue(aliases.isdisjoint(all_document_names))
        self.assertIn("行政处罚决定书", all_document_names)
        self.assertIn("结案审批表", all_document_names)
        self.assertIn("责令改正通知书", all_document_names)
        self.assertIn("现场检查（勘验）笔录", all_document_names)

    def test_model_document_names_match_standardization_rule_names(self) -> None:
        aliases = load_rule_aliases()
        expected_mappings = {
            "现场勘验笔录": "现场检查（勘验）笔录",
            "立案审批信息": "立案审批表",
            "行政处罚事先(听证)告知书": "行政处罚事先（听证）告知书",
            "行政处罚听证通知书": "听证通知书",
            "行政机关负责人集体讨论笔录": "集体讨论笔录",
            "调查询问笔录": "询问笔录",
            "重大行政处罚决定法制审核意见书": "法制审核意见",
            "行政相对人的身份证明或主体资格证明材料": "证件信息",
            "抽样取证记录": "取证记录信息",
            "当事人提交的复制件等证据材料": "取证记录信息",
            "行政执法人员采集制作的证据材料": "取证记录信息",
            "计算机数据、录音、录像、图片等证据材料": "取证记录信息",
        }
        self.assertEqual(
            {
                name: canonical_document_type(name, aliases)
                for name in expected_mappings
            },
            expected_mappings,
        )

    def test_rule_aliases_normalize_without_mutating_source(self) -> None:
        source = {
            "序号": 1,
            "上下文无关审查事项": {},
            "上下文相关审查事项": [
                {
                    "字段": {
                        "现场勘验笔录": [
                            {"field": "当事人信息", "required": False}
                        ],
                        "现场检查（勘验）笔录": [
                            {"field": "当事人信息", "required": True},
                            {"field": "执法主体名称", "required": True},
                        ],
                    }
                }
            ],
        }

        normalized = normalize_rules([source])[0]

        self.assertIn(
            "现场勘验笔录",
            source["上下文相关审查事项"][0]["字段"],
        )
        fields = normalized["上下文相关审查事项"][0]["字段"]
        self.assertEqual(set(fields), {"现场检查（勘验）笔录"})
        self.assertEqual(
            fields["现场检查（勘验）笔录"],
            [
                {
                    "field": "勘验/检查/询问对象信息",
                    "required": True,
                },
                {"field": "执法主体名称", "required": True},
            ],
        )

    def test_normalization_preserves_multiple_categories_for_one_field(
        self,
    ) -> None:
        source = {
            "序号": 1,
            "上下文无关审查事项": {},
            "上下文相关审查事项": [
                {
                    "字段": {
                        "当场行政处罚决定书": [
                            {
                                "field": "行政处罚决定书文号",
                                "required": True,
                                "字段类别": "审查对象",
                            }
                        ],
                        "行政处罚决定书": [
                            {
                                "field": "行政处罚决定书文号",
                                "required": False,
                                "字段类别": "结果核对",
                            }
                        ],
                    }
                }
            ],
        }

        normalized = normalize_rules([source])[0]
        field_items = normalized["上下文相关审查事项"][0]["字段"][
            "行政处罚决定书"
        ]

        self.assertEqual(len(field_items), 2)
        self.assertEqual(
            {item["字段类别"] for item in field_items},
            {"审查对象", "结果核对"},
        )

    def test_normalization_canonicalizes_legal_query_document_and_field(self):
        source = {
            "序号": 1,
            "上下文无关审查事项": {},
            "上下文相关审查事项": [
                {
                    "字段": {},
                    "法条检索查询": {
                        "字段": [
                            {
                                "文书": "当场行政处罚决定书",
                                "字段": "违法事实/行为",
                            }
                        ]
                    },
                }
            ],
        }

        normalized = normalize_rules([source])[0]

        self.assertEqual(
            normalized["上下文相关审查事项"][0]["法条检索查询"]["字段"],
            [{"文书": "行政处罚决定书", "字段": "违法事实/行为"}],
        )

    def test_builder_normalizes_new_rule_document_and_field_aliases(self) -> None:
        rules = (
            RuleSetBuilder(self.all_rules)
            .for_rule_type(0b10010000)
            .build()
            .rules
        )
        rule = next(item for item in rules if item["序号"] == 104)
        self.assertEqual(
            rule["上下文相关审查事项"][0]["任务"],
            "事实清楚、证据充分",
        )
        fields = rule["上下文相关审查事项"][0]["字段"]

        self.assertIn("立案审批表", fields)
        self.assertNotIn("立案审批信息", fields)
        self.assertIn("检查/勘验对象信息", fields)
        self.assertNotIn("当场行政处罚决定书", fields)
        self.assertEqual(
            fields["询问笔录"][0]["field"],
            "被询问人信息",
        )
        self.assertNotIn("送达回证", fields)
        for field_items in fields.values():
            for field_item in field_items:
                self.assertEqual(
                    field_item["required"],
                    False,
                )

    def test_whole_model_expansion_fields_are_optional(self) -> None:
        rule = next(
            item for item in self.all_rule_items() if item["序号"] == 115
        )
        fields = rule["上下文相关审查事项"][0]["字段"][
            "取证记录信息"
        ]
        expanded_field = next(
            item
            for item in fields
            if item["field"] == "案号"
            and item["字段类别"] == "审查对象"
        )

        self.assertFalse(expanded_field["required"])
        self.assertNotIn("字段信息来源", {item["field"] for item in fields})

    def test_rule_360_uses_city_management_human_support(self) -> None:
        rule = next(
            rule
            for rule in self.all_rule_items()
            if rule["序号"] == 360
        )
        decision_item = rule["上下文无关审查事项"]["行政处罚决定书"]
        self.assertEqual(decision_item["外部知识"], [])
        self.assertEqual(
            rule["检索增强"],
            ["city_management_discretion_candidates"],
        )
        self.assertEqual(
            retrieval_enhancement_module_names(rule),
            ["city_management_discretion_candidates"],
        )
        self.assertEqual(
            human_support_document_names([rule]),
            ["行政处罚决定书"],
        )
        self.assertEqual(
            len(
                filter_human_support_rules(
                    [rule],
                    {"行政处罚决定书": True},
                )
            ),
            1,
        )
        self.assertEqual(
            filter_human_support_rules(
                [rule],
                {"行政处罚决定书": False},
            ),
            [],
        )
        rule_without_retrieval = dict(rule)
        rule_without_retrieval["检索增强"] = []
        self.assertEqual(
            retrieval_enhancement_module_names(
                rule_without_retrieval
            ),
            [],
        )
        self.assertEqual(
            filter_human_support_rules(
                [rule_without_retrieval],
                {"行政处罚决定书": True},
            ),
            [],
        )

    def test_enforcement_subtypes_are_decoded_for_enforcement(self) -> None:
        config = decode_rule_type(0b01011111)
        self.assertEqual(config["document_type"], DocumentType.ADMIN_ENFORCEMENT)
        self.assertTrue(config["use_coercive_measure"])
        self.assertTrue(config["use_admin_enforcement"])
        self.assertTrue(config["use_court_enforcement"])


if __name__ == "__main__":
    unittest.main()
