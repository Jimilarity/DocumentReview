import unittest

from reviewers.consistency import (
    CONSISTENCY_TASK,
    ConsistencySource,
    aggregate_consistency_results,
    comparable_sources,
    format_source_values,
    is_executable_consistency_rule,
    required_field_issues,
)


def source(
    value,
    *,
    document_type="立案审批表",
    field_name="嫌疑人",
    required=True,
    section_id=1,
):
    return ConsistencySource(
        document_type=document_type,
        section_id=section_id,
        field_name=field_name,
        required=required,
        value=value,
    )


class ConsistencySourceTest(unittest.TestCase):
    def test_required_null_is_an_issue_and_optional_null_is_not(self) -> None:
        issues = required_field_issues(
            [source(None, required=True), source(None, required=False)]
        )

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["section_ids"], [1])
        self.assertIn("必填字段", issues[0]["content"])

    def test_different_field_names_stay_in_one_comparison_input(self) -> None:
        sources = [
            source("张三", field_name="嫌疑人", section_id=3),
            source(
                "张三",
                document_type="行政处罚决定书",
                field_name="被执行人",
                section_id=8,
            ),
        ]

        self.assertEqual(comparable_sources(sources), sources)
        rendered = format_source_values(sources)
        self.assertIn("嫌疑人", rendered)
        self.assertIn("被执行人", rendered)


class ConsistencyRuleTest(unittest.TestCase):
    def test_only_fully_supported_context_rules_are_executable(self) -> None:
        supported_rule = {
            "上下文无关审查事项": {},
            "上下文相关审查事项": [{"任务": CONSISTENCY_TASK}],
        }
        combined_rule = {
            "上下文无关审查事项": {"立案审批表": {}},
            "上下文相关审查事项": [{"任务": CONSISTENCY_TASK}],
        }
        unknown_rule = {
            "上下文无关审查事项": {},
            "上下文相关审查事项": [{"任务": "尚未实现"}],
        }

        self.assertTrue(is_executable_consistency_rule(supported_rule))
        self.assertTrue(is_executable_consistency_rule(combined_rule))
        self.assertFalse(is_executable_consistency_rule(unknown_rule))

    def test_multiple_consistency_items_only_merge_issues(
        self,
    ) -> None:
        rule = {
            "序号": 101,
            "上下文相关审查事项": [
                {"任务": CONSISTENCY_TASK},
                {"任务": CONSISTENCY_TASK},
            ],
        }
        result = aggregate_consistency_results(
            rule,
            [
                {"issues": []},
                {
                    "issues": [{"section_ids": [2], "content": "不一致"}],
                },
            ],
        )

        self.assertEqual(set(result), {"rule_index", "issues"})
        self.assertEqual(len(result["issues"]), 1)


if __name__ == "__main__":
    unittest.main()
