import json
import os
import sys
import tempfile
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
            overall_revision_advice="统一补充并核正文书中的法律依据。",
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
        self.assertEqual(
            output["overall_revision_advice"],
            "统一补充并核正文书中的法律依据。",
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
            overall_revision_advice="分别修改两项问题。",
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
            {
                "findings": [],
                "overall_revision_advice": "本次评查未发现需要修改的问题。",
                "discarded_candidate_ids": [],
            },
        )

    async def test_retries_validation_failure_and_then_succeeds(self) -> None:
        invalid_result = ReviewResultProcessingOutput(
            findings=[
                ProcessedFindingDraft(
                    source_candidate_ids=["C9999"],
                    content="问题。",
                )
            ],
            overall_revision_advice="修改问题。",
        )
        valid_result = ReviewResultProcessingOutput(
            findings=[
                ProcessedFindingDraft(
                    source_candidate_ids=["C0001"],
                    content="问题。",
                )
            ],
            overall_revision_advice="修改问题。",
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

    def test_location_must_survive_finding_and_overall_advice(self) -> None:
        candidates = [
            {
                "candidate_id": "C0001",
                "rule_index": 337,
                "section_ids": [8],
                "content": (
                    "【PDF第3页《行政处罚事先（听证）告知书》第1页】"
                    "未告知听证申请权。"
                ),
            }
        ]
        location = "【PDF第3页《行政处罚事先（听证）告知书》第1页】"
        valid_result = ReviewResultProcessingOutput(
            findings=[
                ProcessedFindingDraft(
                    source_candidate_ids=["C0001"],
                    content=f"{location}未告知听证申请权。",
                )
            ],
            overall_revision_advice=(
                f"{location}补充与本案听证条件相符的听证权告知。"
            ),
        )

        ReviewResultProcessor.validate_result(valid_result, candidates)

        missing_location_result = ReviewResultProcessingOutput(
            findings=[
                ProcessedFindingDraft(
                    source_candidate_ids=["C0001"],
                    content="《行政处罚事先（听证）告知书》未告知听证申请权。",
                )
            ],
            overall_revision_advice="补充听证权告知。",
        )
        with self.assertRaisesRegex(ValueError, "删除或改写"):
            ReviewResultProcessor.validate_result(
                missing_location_result,
                candidates,
            )

    async def test_repairs_missing_locations_without_retry(self) -> None:
        location = "【PDF第42页《行政处罚告知书》第2页】"
        result = ReviewResultProcessingOutput(
            findings=[
                ProcessedFindingDraft(
                    source_candidate_ids=["C0001"],
                    content="未告知陈述申辩权。",
                )
            ],
            overall_revision_advice="补充权利告知。",
        )
        agents = FakePostReviewAgents(result)
        processor = ReviewResultProcessor(agents=agents, max_tries=3)

        output = await processor.process(
            [
                {
                    "rule_index": 335,
                    "issues": [
                        {
                            "section_ids": [8],
                            "content": f"{location}未告知陈述申辩权。",
                        }
                    ],
                }
            ]
        )

        self.assertIn(location, output["findings"][0]["content"])
        self.assertIn(location, output["overall_revision_advice"])
        self.assertEqual(agents.ainvoke_result_processing.await_count, 1)

    async def test_uses_lossless_fallback_after_all_attempts_fail(self) -> None:
        agents = FakePostReviewAgents(
            [RuntimeError("service unavailable"), RuntimeError("still down")]
        )
        processor = ReviewResultProcessor(agents=agents, max_tries=2)
        location = "【PDF第42页《行政处罚告知书》第2页】"

        output = await processor.process(
            [
                {
                    "rule_index": 335,
                    "issues": [
                        {
                            "section_ids": [8],
                            "content": f"{location}未告知陈述申辩权。",
                        }
                    ],
                }
            ]
        )

        self.assertTrue(output["fallback"])
        self.assertEqual(len(output["findings"]), 1)
        self.assertEqual(
            output["findings"][0]["content"],
            f"{location}未告知陈述申辩权。",
        )
        self.assertIn(location, output["overall_revision_advice"])


class PostReviewAgentsConfigurationTest(unittest.IsolatedAsyncioTestCase):
    def test_uses_tool_strategy_and_disables_thinking(self) -> None:
        text_model = Mock()

        with patch.dict(
            os.environ,
            {
                "REVIEW_RESPONSE_FORMAT_MODE": "tool_strategy",
                "REVIEW_ENABLE_THINKING": "false",
            },
        ):
            with patch.object(
                BaseAgents,
                "__init__",
                return_value=None,
            ) as init:
                with patch.object(
                    PostReviewAgents,
                    "text_model",
                    text_model,
                    create=True,
                ):
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

    def test_uses_json_object_from_environment(self) -> None:
        text_model = Mock()

        with patch.dict(
            os.environ,
            {
                "REVIEW_RESPONSE_FORMAT_MODE": "json_object",
                "REVIEW_ENABLE_THINKING": "false",
            },
        ):
            with patch.object(
                BaseAgents,
                "__init__",
                return_value=None,
            ):
                with patch.object(
                    PostReviewAgents,
                    "text_model",
                    text_model,
                    create=True,
                ):
                    PostReviewAgents()

        text_model.bind.assert_called_once_with(
            response_format={"type": "json_object"},
        )

    async def test_tool_strategy_falls_back_to_json_object(self) -> None:
        agents = object.__new__(PostReviewAgents)
        agents.response_format_mode = agents.TOOL_STRATEGY_MODE
        agents.result_processor_agent = object()
        agents.result_processor_json_agent = object()
        expected = ReviewResultProcessingOutput(
            findings=[],
            overall_revision_advice="本次评查未发现需要修改的问题。",
        )
        agents._ainvoke_structured_agent = AsyncMock(
            side_effect=[TypeError("invalid tool arguments"), expected]
        )

        result = await agents.ainvoke_result_processing("prompt", 20)

        self.assertIs(result, expected)
        self.assertEqual(
            agents._ainvoke_structured_agent.await_args_list[1].kwargs[
                "force_json_object"
            ],
            True,
        )

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
        scored_payload = coordinator.save_scored_raw_results.call_args.args[0]
        self.assertEqual(scored_payload["规则评分"], raw_results)
        self.assertEqual(
            scored_payload["评分汇总"]["整体修改建议"],
            "审查结果整理未完成，未生成整体修改建议。",
        )
        coordinator.save_results.assert_called_once_with(raw_results)
        self.assertFalse(output["result_processing_enabled"])
        self.assertEqual(output["review_results"], raw_results)
        self.assertEqual(output["findings"], [])

    async def test_unexpected_processing_error_preserves_case_result(self) -> None:
        raw_results = [
            {
                "rule_index": 335,
                "issues": [
                    {
                        "section_ids": [8],
                        "content": "【PDF第42页《行政处罚告知书》第2页】未告知权利。",
                    }
                ],
            }
        ]
        coordinator = object.__new__(ReviewResultCoordinator)
        coordinator.file_path = Path("case.pdf")
        coordinator.settings = ReviewSettings(result_process_max_tries=3)
        coordinator.logger = Mock()
        coordinator.cache_paths = SimpleNamespace(
            raw_review_results=Path("raw_review_results.json"),
            scored_raw_review_results=Path("raw_review_results_scored.json"),
        )
        coordinator.save_raw_results = Mock()
        coordinator.save_processing_result = Mock()
        coordinator.save_results = Mock()
        coordinator.process_results = AsyncMock(
            side_effect=RuntimeError("unexpected post-processing error")
        )
        coordinator.score_raw_results = AsyncMock(return_value=raw_results)
        coordinator.save_scored_raw_results = Mock()

        output = await coordinator.finalize(raw_results, rule_count=1)

        saved = coordinator.save_processing_result.call_args.args[0]
        self.assertTrue(saved["fallback"])
        self.assertEqual(len(saved["findings"]), 1)
        self.assertEqual(len(output["findings"]), 1)
        coordinator.score_raw_results.assert_awaited_once_with(raw_results)


class ScoredResultSummaryTest(unittest.TestCase):
    def test_enriches_single_page_issue_location_without_changing_schema(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            directory_path = temp_path / "directory.json"
            ocr_path = temp_path / "ocr.json"
            directory_path.write_text(
                json.dumps(
                    [
                        {
                            "section_id": 1,
                            "section_name": "行政处罚决定书",
                            "section_page": 2,
                        },
                        {
                            "section_id": 2,
                            "section_name": "送达回证",
                            "section_page": 3,
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            ocr_path.write_text(
                json.dumps(
                    [
                        {"image_index": index, "document_content": "正文"}
                        for index in range(4)
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            coordinator = object.__new__(ReviewResultCoordinator)
            coordinator.cache_paths = SimpleNamespace(
                directory=directory_path,
                ocr_results=ocr_path,
            )

            enriched = coordinator.enrich_issue_locations(
                [
                    {
                        "rule_index": 366,
                        "issues": [
                            {
                                "section_ids": [1],
                                "content": "《行政处罚决定书》未载明复议机关。",
                            }
                        ],
                    }
                ]
            )

        issue = enriched[0]["issues"][0]
        self.assertEqual(set(issue), {"section_ids", "content"})
        self.assertEqual(
            issue["content"],
            "【PDF第3页《行政处罚决定书》第1页】未载明复议机关。",
        )

    def test_recomputes_document_page_from_pdf_and_section_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            directory_path = temp_path / "directory.json"
            ocr_path = temp_path / "ocr.json"
            directory_path.write_text(
                json.dumps(
                    [
                        {
                            "section_id": 8,
                            "section_name": "行政处罚告知书",
                            "section_page": 40,
                        },
                        {
                            "section_id": 9,
                            "section_name": "行政处罚决定书",
                            "section_page": 43,
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            ocr_path.write_text(
                json.dumps(
                    [
                        {"image_index": index, "document_content": "正文"}
                        for index in range(44)
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            coordinator = object.__new__(ReviewResultCoordinator)
            coordinator.cache_paths = SimpleNamespace(
                directory=directory_path,
                ocr_results=ocr_path,
            )

            enriched = coordinator.enrich_issue_locations(
                [
                    {
                        "rule_index": 335,
                        "issues": [
                            {
                                "section_ids": [8],
                                "content": (
                                    "【《行政处罚告知书》PDF第42页（文书第39页）】"
                                    "未告知陈述申辩权。"
                                ),
                            }
                        ],
                    }
                ]
            )

        self.assertEqual(
            enriched[0]["issues"][0]["content"],
            "【PDF第42页《行政处罚告知书》第2页】未告知陈述申辩权。",
        )

    def test_calculates_category_and_total_scores_with_zero_floor(self) -> None:
        payload = ReviewResultCoordinator.build_scored_result_payload(
            [
                {"规则类别": "合法性", "扣分": 100, "issues": [{}]},
                {"规则类别": "合法性", "扣分": 20, "issues": [{}]},
                {
                    "规则类别": "规范性",
                    "规则满分": 5,
                    "分数": 2.5,
                    "扣分": 2.5,
                    "issues": [{}],
                },
                {
                    "规则类别": "规范性",
                    "规则满分": 5,
                    "分数": 5,
                    "扣分": 0,
                    "issues": [],
                },
                {
                    "规则类别": "规范性",
                    "规则满分": 2,
                    "分数": None,
                    "扣分": None,
                    "issues": [],
                },
            ],
            {"overall_revision_advice": "优先修正合法性问题。"},
        )

        self.assertEqual(payload["评分汇总"]["合法性得分"], 0)
        self.assertEqual(payload["评分汇总"]["合规性得分"], 75)
        self.assertEqual(payload["评分汇总"]["总得分"], 0)
        self.assertEqual(
            payload["评分汇总"]["整体修改建议"],
            "优先修正合法性问题。",
        )

    def test_compliance_score_is_full_when_no_normative_rule_participates(
        self,
    ) -> None:
        payload = ReviewResultCoordinator.build_scored_result_payload(
            [
                {"规则类别": "合法性", "扣分": 0, "issues": []},
                {
                    "规则类别": "规范性",
                    "规则满分": 2,
                    "分数": None,
                    "扣分": None,
                    "issues": [],
                },
            ],
            {"overall_revision_advice": "无需修改。"},
        )

        self.assertEqual(payload["评分汇总"]["合规性得分"], 100)

    def test_total_score_combines_legality_and_normalized_compliance(self) -> None:
        payload = ReviewResultCoordinator.build_scored_result_payload(
            [
                {"规则类别": "合法性", "扣分": 10, "issues": [{}]},
                {
                    "规则类别": "规范性",
                    "规则满分": 10,
                    "分数": 8,
                    "扣分": 2,
                    "issues": [{}],
                },
            ],
            {"overall_revision_advice": "按问题修改。"},
        )

        self.assertEqual(payload["评分汇总"]["合法性得分"], 90)
        self.assertEqual(payload["评分汇总"]["合规性得分"], 80)
        self.assertEqual(payload["评分汇总"]["总得分"], 70)


if __name__ == "__main__":
    unittest.main()
