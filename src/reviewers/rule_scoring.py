import copy
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from agents import RuleScoreDraft, RuleScoringAgents, RuleScoringOutput
from constants import RULES_PATH
from utils import read_json


class RuleScoreProcessor:
    """从 raw review results 生成带规则级参考分数的独立结果。"""

    def __init__(
        self,
        *,
        rules_path: str | Path = RULES_PATH,
        agents: RuleScoringAgents | None = None,
        recursion_limit: int = 20,
        max_tries: int = 3,
        logger: logging.Logger | None = None,
    ) -> None:
        self.rules_path = Path(rules_path)
        self.agents = agents
        self.recursion_limit = recursion_limit
        self.max_tries = max_tries
        self.logger = logger or logging.getLogger(__name__)
        self.rules_by_index = self._load_rules_by_index()

    def _load_rules_by_index(self) -> Dict[int, Dict[str, Any]]:
        rule_data = read_json(self.rules_path)
        rules_by_index: Dict[int, Dict[str, Any]] = {}
        for category in rule_data.values():
            if not isinstance(category, dict):
                continue
            for rules in category.values():
                if not isinstance(rules, list):
                    continue
                for rule in rules:
                    rule_index = rule.get("序号")
                    if isinstance(rule_index, int):
                        rules_by_index.setdefault(rule_index, rule)
        return rules_by_index

    @staticmethod
    def _set_score(
        result: Dict[str, Any],
        score: float | int | None,
        explanation: str,
    ) -> None:
        result["分数"] = score
        result["扣分/加分说明"] = explanation

    def _prepare_results(
        self,
        raw_results: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        scored_results = copy.deepcopy(raw_results)
        candidates: List[Dict[str, Any]] = []
        for result in scored_results:
            rule_index = result.get("rule_index")
            rule = self.rules_by_index.get(rule_index)
            scoring = rule.get("评分细则") if rule else None
            if not scoring:
                self._set_score(
                    result,
                    None,
                    "评分细则原表无对应规则，分数为空。",
                )
                continue

            method = scoring.get("评查方式")
            maximum_score = scoring.get("分值")
            if method != "评分":
                self._set_score(
                    result,
                    None,
                    f"该规则的评查方式为{method or '未设置'}，不计分。",
                )
                continue
            if not isinstance(maximum_score, (int, float)) or isinstance(
                maximum_score, bool
            ):
                self._set_score(
                    result,
                    None,
                    "该规则没有数字分值，暂不计算规则级分数。",
                )
                continue
            if result.get("error"):
                self._set_score(
                    result,
                    None,
                    "该规则审查执行异常，无法计算分数。",
                )
                continue

            issues = result.get("issues") or []
            if not issues:
                self._set_score(
                    result,
                    maximum_score,
                    (
                        f"该项满分{maximum_score}分，未发现问题，"
                        f"参考得分{maximum_score}分。"
                    ),
                )
                continue

            candidates.append(
                {
                    "rule_index": rule_index,
                    "maximum_score": maximum_score,
                    "scoring_explanation": scoring.get("评查说明") or "",
                    "issues": issues,
                }
            )
        return scored_results, candidates

    @staticmethod
    def _validate_model_result(
        output: RuleScoringOutput,
        candidates: List[Dict[str, Any]],
    ) -> None:
        candidates_by_index = {
            candidate["rule_index"]: candidate for candidate in candidates
        }
        scores_by_index = {item.rule_index: item for item in output.scores}
        if len(scores_by_index) != len(output.scores):
            raise ValueError("评分结果包含重复 rule_index")
        if set(scores_by_index) != set(candidates_by_index):
            raise ValueError("评分结果未完整覆盖候选规则或包含未知规则")
        for rule_index, score_item in scores_by_index.items():
            maximum_score = candidates_by_index[rule_index]["maximum_score"]
            if not 0 <= score_item.score <= maximum_score:
                raise ValueError(
                    f"规则 {rule_index} 的分数 {score_item.score} "
                    f"超出 0 至 {maximum_score} 的范围"
                )

    async def _score_candidates(
        self,
        candidates: List[Dict[str, Any]],
    ) -> RuleScoringOutput:
        agents = self.agents or RuleScoringAgents()
        base_prompt = agents.build_task_prompt(
            "rule_scoring",
            candidates=json.dumps(candidates, ensure_ascii=False, indent=2),
        )
        last_error: Exception | None = None
        for attempt in range(self.max_tries):
            prompt = base_prompt
            if last_error is not None:
                prompt += (
                    "\n\n上一次评分结果未通过程序校验：\n"
                    f"{type(last_error).__name__}: {last_error}\n"
                    "请重新对全部候选规则评分并提交完整结果。"
                )
            try:
                output = await agents.ainvoke_rule_scoring(
                    prompt,
                    self.recursion_limit,
                )
                self._validate_model_result(output, candidates)
                return output
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_tries:
                    self.logger.warning(
                        "rule scoring failed; retrying attempt=%s/%s error=%s: %s",
                        attempt + 1,
                        self.max_tries,
                        type(exc).__name__,
                        exc,
                    )
        raise RuntimeError(
            f"规则级评分连续失败，已达到最大尝试次数 {self.max_tries}"
        ) from last_error

    @staticmethod
    def _fallback_score(candidate: Dict[str, Any]) -> RuleScoreDraft:
        """模型不可用时，依据评查说明生成可解释的保守估算。"""

        maximum_score = float(candidate["maximum_score"])
        issue_count = len(candidate.get("issues") or [])
        explanation = str(candidate.get("scoring_explanation") or "")

        if "全扣" in explanation:
            deduction = maximum_score
            basis = "评查说明包含“全扣”要求"
        else:
            explicit_deductions = [
                float(value)
                for value in re.findall(
                    r"扣(?:减)?\s*(\d+(?:\.\d+)?)\s*分",
                    explanation,
                )
            ]
            occupied_scores = [
                float(value)
                for value in re.findall(
                    r"(?:各占|每项(?:占)?)\s*(\d+(?:\.\d+)?)\s*分",
                    explanation,
                )
            ]
            per_issue_deduction = max(
                [*explicit_deductions, *occupied_scores],
                default=0.0,
            )
            if per_issue_deduction > 0:
                deduction = per_issue_deduction * issue_count
                basis = (
                    f"按说明中可识别的每项最高扣分 "
                    f"{per_issue_deduction:g} 分估算"
                )
            else:
                deduction = maximum_score * min(issue_count * 0.1, 1.0)
                basis = "说明无可直接计算的扣分幅度，按每个问题扣满分10%估算"

        score = round(max(0.0, maximum_score - deduction), 4)
        actual_deduction = round(maximum_score - score, 4)
        return RuleScoreDraft(
            rule_index=candidate["rule_index"],
            score=score,
            explanation=(
                f"模型评分不可用，采用本地兜底估算：满分"
                f"{maximum_score:g}分，{basis}，参考扣分"
                f"{actual_deduction:g}分，参考得分{score:g}分。仅供参考。"
            ),
        )

    def _fallback_scores(
        self,
        candidates: List[Dict[str, Any]],
    ) -> RuleScoringOutput:
        return RuleScoringOutput(
            scores=[self._fallback_score(item) for item in candidates]
        )

    async def process(
        self,
        raw_results: List[Dict[str, Any]],
        *,
        use_model: bool = True,
    ) -> List[Dict[str, Any]]:
        scored_results, candidates = self._prepare_results(raw_results)
        if not candidates:
            return scored_results

        if not use_model:
            output = self._fallback_scores(candidates)
        else:
            try:
                output = await self._score_candidates(candidates)
            except Exception:
                self.logger.exception(
                    "rule scoring failed; using deterministic fallback"
                )
                output = self._fallback_scores(candidates)

        scores_by_index = {
            item.rule_index: item for item in output.scores
        }
        for result in scored_results:
            score_item = scores_by_index.get(result.get("rule_index"))
            if score_item is None:
                continue
            self._set_score(
                result,
                score_item.score,
                score_item.explanation,
            )
        return scored_results
