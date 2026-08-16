import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock


SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from agents import RuleScoreDraft, RuleScoringOutput
from reviewers.rule_scoring import RuleScoreProcessor
from utils import load_yaml


class FakeRuleScoringAgents:
    def __init__(self, output) -> None:
        self.tasks_config = load_yaml(SRC_ROOT / "config" / "tasks.yaml")
        self.ainvoke_rule_scoring = AsyncMock(return_value=output)

    def build_task_prompt(self, task_name: str, **kwargs):
        config = self.tasks_config[task_name]
        return (
            config["description"].format(**kwargs)
            + "\n"
            + config["expected_output"]
        )


class RuleScoringTest(unittest.IsolatedAsyncioTestCase):
    def build_rules_file(self, directory: Path) -> Path:
        path = directory / "rules.json"
        path.write_text(
            json.dumps(
                {
                    "规范性标准": {
                        "行政处罚": [
                            {
                                "序号": 304,
                                "评分细则": {
                                    "分值": 2,
                                    "评查方式": "评分",
                                    "评查说明": "每缺一项扣0.5分。",
                                },
                            },
                            {
                                "序号": 201,
                                "评分细则": {
                                    "分值": "/",
                                    "评查方式": "考查",
                                    "评查说明": "",
                                },
                            },
                            {
                                "序号": 601,
                            },
                        ]
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path

    async def test_scores_only_numeric_scoring_rules(self) -> None:
        output = RuleScoringOutput(
            scores=[
                RuleScoreDraft(
                    rule_index=304,
                    score=1.5,
                    explanation="满分2分，扣0.5分，剩余1.5分。",
                )
            ]
        )
        agents = FakeRuleScoringAgents(output)
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = RuleScoreProcessor(
                rules_path=self.build_rules_file(Path(temp_dir)),
                agents=agents,
                max_tries=1,
            )
            results = await processor.process(
                [
                    {
                        "rule_index": 304,
                        "issues": [
                            {"section_ids": [1], "content": "缺少地点。"}
                        ],
                    },
                    {"rule_index": 201, "issues": []},
                    {"rule_index": 601, "issues": []},
                ]
            )

        self.assertEqual(results[0]["分数"], 1.5)
        self.assertIn("扣0.5分", results[0]["扣分/加分说明"])
        self.assertIsNone(results[1]["分数"])
        self.assertIn("考查", results[1]["扣分/加分说明"])
        self.assertIsNone(results[2]["分数"])
        self.assertIn("无对应规则", results[2]["扣分/加分说明"])

    async def test_no_issue_numeric_rule_gets_full_score_without_model(self) -> None:
        agents = FakeRuleScoringAgents(RuleScoringOutput(scores=[]))
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = RuleScoreProcessor(
                rules_path=self.build_rules_file(Path(temp_dir)),
                agents=agents,
            )
            results = await processor.process(
                [{"rule_index": 304, "issues": []}]
            )

        self.assertEqual(results[0]["分数"], 2)
        agents.ainvoke_rule_scoring.assert_not_awaited()

    async def test_invalid_model_score_uses_local_fallback(self) -> None:
        agents = FakeRuleScoringAgents(
            RuleScoringOutput(
                scores=[
                    RuleScoreDraft(
                        rule_index=304,
                        score=3,
                        explanation="错误分数",
                    )
                ]
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = RuleScoreProcessor(
                rules_path=self.build_rules_file(Path(temp_dir)),
                agents=agents,
                max_tries=1,
            )
            results = await processor.process(
                [
                    {
                        "rule_index": 304,
                        "issues": [
                            {"section_ids": [1], "content": "问题。"}
                        ],
                    }
                ]
            )

        self.assertEqual(results[0]["分数"], 1.5)
        self.assertIn("本地兜底估算", results[0]["扣分/加分说明"])
        self.assertIn("仅供参考", results[0]["扣分/加分说明"])


if __name__ == "__main__":
    unittest.main()
