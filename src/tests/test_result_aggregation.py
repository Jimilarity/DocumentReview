import sys
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from reviewers.result_aggregation import (
    aggregate_component_results,
    aggregate_section_results,
    merge_rule_results,
)


class ResultAggregationTest(unittest.TestCase):
    def test_section_results_merge_issues(self) -> None:
        result = aggregate_section_results(
            301,
            [
                {"issues": []},
                {
                    "issues": [
                        {
                            "section_ids": [8],
                            "content": "缺少签名",
                        }
                    ],
                },
            ],
        )

        self.assertEqual(
            result,
            {
                "rule_index": 301,
                "issues": [
                    {"section_ids": [8], "content": "缺少签名"}
                ],
            },
        )

    def test_preserves_section_failure(self) -> None:
        result = aggregate_section_results(
            301,
            [
                {
                    "issues": [
                        {
                            "section_ids": [3],
                            "content": "审查失败",
                        }
                    ],
                    "error": "rule_execution_failed",
                    "error_details": {"stage": "single_section"},
                }
            ],
        )

        self.assertEqual(result["error"], "section_review_failed")
        self.assertEqual(
            result["error_details"],
            [{"stage": "single_section"}],
        )

    def test_component_results_merge_issues(self) -> None:
        result = aggregate_component_results(
            363,
            [
                {"issues": []},
                {
                    "issues": [
                        {"section_ids": [9], "content": "日期不一致"}
                    ],
                },
            ],
        )

        self.assertEqual(
            result,
            {
                "rule_index": 363,
                "issues": [
                    {"section_ids": [9], "content": "日期不一致"}
                ],
            },
        )

    def test_executor_results_for_same_rule_are_merged_once(self) -> None:
        result = merge_rule_results(
            [{"rule_index": 101, "issues": [{"content": "A"}]}],
            [{"rule_index": 101, "issues": [{"content": "B"}]}],
        )

        self.assertEqual(
            result,
            [
                {
                    "rule_index": 101,
                    "issues": [{"content": "A"}, {"content": "B"}],
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
