import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch


SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from agents import (
    BaseAgents,
    PostReviewAgents,
    ProcessedFindingDraft,
    ReviewResultProcessingOutput,
)
from langchain.agents.structured_output import ToolStrategy
from reviewers.result_processing import ReviewResultProcessor
from reviewers.base import ReviewSettings
from reviewers.result_coordinator import ReviewResultCoordinator
from utils import load_yaml


class FakePostReviewAgents:
    def __init__(self, results) -> None:
        self.results = results if isinstance(results, list) else [results]
        self.tasks_config = load_yaml(SRC_ROOT / "config" / "tasks.yaml")
        self.ainvoke_result_processing = AsyncMock(
            side_effect=self.results
        )

    def build_task_prompt(self, task_name: str, **kwargs):
        config = self.tasks_config[task_name]
        return (
            config["description"].format(**kwargs)
            + "\n"
            + config["expected_output"]
        )


class ResultProcessingTest(unittest.IsolatedAsyncioTestCase):
    async def test_merges_candidates_and_builds_result_fields(self) -> None:
        structured_result = ReviewResultProcessingOutput(
            findings=[
                ProcessedFindingDraft(
                    source_candidate_ids=["C0001", "C0002"],
                    content="同一处法律依据表述不完整。",
                )
            ],
        )
        agents = FakePostReviewAgents(structured_result)
        processor = ReviewResultProcessor(agents=agents)
        rule_results = [
            {
                "rule_index": 359,
                "issues": [
                    {"section_ids": [1], "content": "依据表述遗漏。"}
                ],
            },
            {
                "rule_index": 360,
                "issues": [
                    {"section_ids": [1], "content": "法律依据不完整。"}
                ],
            },
            {
                "rule_index": 601,
                "issues": [
                    {
                        "section_ids": [6],
                        "content": "可能有误，但最终未发现明显错误。",
                    }
                ],
            },
        ]
        output = await processor.process(rule_results)

        self.assertEqual(len(output["findings"]), 1)
        self.assertEqual(output["findings"][0]["finding_id"], "F0001")
        self.assertEqual(
            output["findings"][0]["rule_indexes"],
            [359, 360],
        )
        self.assertEqual(output["findings"][0]["section_ids"], [1])
        self.assertEqual(
            output["discarded_candidate_ids"],
            ["C0003"],
        )
        prompt = agents.ainvoke_result_processing.call_args.args[0]
        self.assertIn('"candidate_id": "C0001"', prompt)
        self.assertNotIn("<rules>", prompt)
        self.assertIn("JSON", prompt)

    async def test_one_candidate_can_support_multiple_findings(self) -> None:
        structured_result = ReviewResultProcessingOutput(
            findings=[
                ProcessedFindingDraft(
                    source_candidate_ids=["C0001"],
                    content="问题。",
                ),
                ProcessedFindingDraft(
                    source_candidate_ids=["C0001"],
                    content="重复问题。",
                )
            ],
        )
        processor = ReviewResultProcessor(
            agents=FakePostReviewAgents(structured_result),
            max_tries=1,
        )

        output = await processor.process(
            [
                {
                    "rule_index": 359,
                    "issues": [
                        {"section_ids": [1], "content": "复合问题。"}
                    ],
                }
            ],
        )

        self.assertEqual(len(output["findings"]), 2)
        self.assertEqual(
            output["findings"][0]["source_candidate_ids"],
            ["C0001"],
        )
        self.assertEqual(
            output["findings"][1]["source_candidate_ids"],
            ["C0001"],
        )
        self.assertEqual(output["discarded_candidate_ids"], [])

    async def test_skips_model_when_there_are_no_candidates(self) -> None:
        processor = ReviewResultProcessor()

        output = await processor.process(
            [{"rule_index": 359, "issues": []}],
        )

        self.assertEqual(
            output,
            {"findings": [], "discarded_candidate_ids": []},
        )

    async def test_retries_validation_failure_and_then_succeeds(self) -> None:
        invalid_result = ReviewResultProcessingOutput(
            findings=[
                ProcessedFindingDraft(
                    source_candidate_ids=["C9999"],
                    content="问题。",
                )
            ]
        )
        valid_result = ReviewResultProcessingOutput(
            findings=[
                ProcessedFindingDraft(
                    source_candidate_ids=["C0001"],
                    content="问题。",
                )
            ]
        )
        agents = FakePostReviewAgents([invalid_result, valid_result])
        processor = ReviewResultProcessor(
            agents=agents,
            max_tries=2,
        )

        output = await processor.process(
            [
                {
                    "rule_index": 359,
                    "issues": [
                        {"section_ids": [1], "content": "问题。"}
                    ],
                }
            ],
        )

        self.assertEqual(output["findings"][0]["rule_indexes"], [359])
        self.assertEqual(
            agents.ainvoke_result_processing.await_count,
            2,
        )
        retry_prompt = (
            agents.ainvoke_result_processing.await_args_list[1].args[0]
        )
        self.assertIn("上一次结构化结果未通过程序校验", retry_prompt)


class PostReviewAgentsConfigurationTest(unittest.TestCase):
    def test_uses_tool_strategy_and_disables_thinking(self) -> None:
        with patch.object(BaseAgents, "__init__", return_value=None) as init:
            with patch.object(
                PostReviewAgents,
                "create_text_agent",
                return_value=object(),
            ) as create_agent:
                PostReviewAgents()

        init.assert_called_once_with(
            text_parallel_tool_calls=False,
            text_enable_thinking=False,
            vision_enable_thinking=False,
        )
        response_format = create_agent.call_args.kwargs["response_format"]
        self.assertIsInstance(response_format, ToolStrategy)

    def test_max_tries_is_loaded_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {"REVIEW_RESULT_PROCESS_MAX_TRIES": "4"},
        ):
            settings = ReviewSettings.from_env()

        self.assertEqual(settings.result_process_max_tries, 4)


class ResultProcessingDisabledTest(unittest.IsolatedAsyncioTestCase):
    async def test_zero_max_tries_writes_raw_results_as_final(self) -> None:
        raw_results = [
            {
                "rule_index": 359,
                "issues": [
                    {"section_ids": [1], "content": "原始问题。"}
                ],
            }
        ]
        coordinator = object.__new__(ReviewResultCoordinator)
        coordinator.file_path = Path("case.pdf")
        coordinator.settings = ReviewSettings(
            result_process_max_tries=0,
        )
        coordinator.logger = Mock()
        coordinator.cache_paths = SimpleNamespace(
            raw_review_results=Path("raw_review_results.json"),
            scored_raw_review_results=Path(
                "raw_review_results_scored.json"
            ),
        )
        coordinator.save_raw_results = Mock()
        coordinator.save_processing_result = Mock()
        coordinator.save_results = Mock()
        coordinator.process_results = AsyncMock()
        coordinator.score_raw_results = AsyncMock(return_value=raw_results)
        coordinator.save_scored_raw_results = Mock()

        output = await coordinator.finalize(raw_results, rule_count=1)

        coordinator.process_results.assert_not_awaited()
        coordinator.score_raw_results.assert_awaited_once_with(raw_results)
        coordinator.save_scored_raw_results.assert_called_once_with(raw_results)
        coordinator.save_results.assert_called_once_with(raw_results)
        self.assertFalse(output["result_processing_enabled"])
        self.assertEqual(output["review_results"], raw_results)
        self.assertEqual(output["findings"], [])


if __name__ == "__main__":
    unittest.main()
