import copy
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from agents import RuleScoreDraft, RuleScoringAgents, RuleScoringOutput
from constants import RULES_PATH
from utils import read_json


RULE_CATEGORY_LABELS = {
    "合法性标准": "合法性",
    "规范性标准": "规范性",
    "加减分项": "加减分",
    "附加项": "附加项",
}

LOCATION_TAG_PATTERN = re.compile(
    r"【(?:PDF第\d+页《[^》]+》第\d+页"
    r"|《[^》]+》PDF第\d+页(?:（文书第\d+页）)?"
    r"|《[^》]+》文书第\d+页)】"
)


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
        (
            self.rules_by_index,
            self.rule_categories_by_index,
        ) = self._load_rules_by_index()

    def _load_rules_by_index(
        self,
    ) -> tuple[Dict[int, Dict[str, Any]], Dict[int, str]]:
        rule_data = read_json(self.rules_path)
        rules_by_index: Dict[int, Dict[str, Any]] = {}
        categories_by_index: Dict[int, str] = {}
        for category_name, category in rule_data.items():
            if not isinstance(category, dict):
                continue
            for rules in category.values():
                if not isinstance(rules, list):
                    continue
                for rule in rules:
                    rule_index = rule.get("序号")
                    if isinstance(rule_index, int):
                        rules_by_index.setdefault(rule_index, rule)
                        categories_by_index.setdefault(
                            rule_index,
                            RULE_CATEGORY_LABELS.get(
                                category_name,
                                category_name,
                            ),
                        )
        return rules_by_index, categories_by_index

    @staticmethod
    def _normalize_number(value: float | int) -> float | int:
        normalized = round(float(value), 4)
        return int(normalized) if normalized.is_integer() else normalized

    @classmethod
    def _set_score(
        cls,
        result: Dict[str, Any],
        category: str | None,
        maximum_score: float | int | None,
        score: float | int | None,
        explanation: str,
        ai_revision_advice: str,
    ) -> None:
        result["规则类别"] = category
        result["规则满分"] = maximum_score
        result["分数"] = score
        result["扣分"] = (
            None
            if maximum_score is None or score is None
            else cls._normalize_number(
                max(0.0, float(maximum_score) - float(score))
            )
        )
        result["扣分/加分说明"] = explanation
        result["AI修改建议"] = ai_revision_advice

    @staticmethod
    def _default_ai_revision_advice(result: Dict[str, Any]) -> str:
        if result.get("error"):
            return "该规则审查执行异常，请人工复核后再确定修改内容。"
        if not result.get("issues"):
            return "无需修改。"
        return (
            "请依据本规则列出的各条问题逐项修正文书，并核对修改后的内容"
            "与案卷事实、原始材料一致。"
        )

    @staticmethod
    def _location_safe_revision_advice(
        issues: List[Dict[str, Any]],
        model_advice: str,
    ) -> str:
        """拒绝模型编造或遗漏 issue 已有定位，必要时按 issue 确定性回退。"""

        allowed_locations = {
            location
            for issue in issues
            for location in LOCATION_TAG_PATTERN.findall(
                str(issue.get("content") or "")
            )
        }
        advice_locations = set(LOCATION_TAG_PATTERN.findall(model_advice))
        if (
            not model_advice.strip()
            or advice_locations - allowed_locations
            or not allowed_locations.issubset(advice_locations)
        ):
            items = []
            for index, issue in enumerate(issues, start=1):
                content = str(issue.get("content") or "").strip()
                if not content:
                    continue
                items.append(
                    f"{index}. {content}；请针对上述问题核对原始文书并修正，"
                    "修改后复核与案卷事实一致。"
                )
            if items:
                return "\n".join(items)
        return model_advice

    def _prepare_results(
        self,
        raw_results: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        scored_results = copy.deepcopy(raw_results)
        candidates: List[Dict[str, Any]] = []
        for result in scored_results:
            rule_index = result.get("rule_index")
            rule = self.rules_by_index.get(rule_index)
            category = self.rule_categories_by_index.get(rule_index)
            scoring = rule.get("评分细则") if rule else None
            if not scoring:
                self._set_score(
                    result,
                    category,
                    None,
                    None,
                    "评分细则原表无对应规则，分数为空。",
                    self._default_ai_revision_advice(result),
                )
                continue

            method = scoring.get("评查方式")
            maximum_score = scoring.get("分值")
            scoring_explanation = str(scoring.get("评查说明") or "").strip()
            if method != "评分":
                self._set_score(
                    result,
                    category,
                    None,
                    None,
                    f"该规则的评查方式为{method or '未设置'}，不计分。",
                    self._default_ai_revision_advice(result),
                )
                continue
            if not isinstance(maximum_score, (int, float)) or isinstance(
                maximum_score, bool
            ):
                self._set_score(
                    result,
                    category,
                    None,
                    None,
                    "该规则没有数字分值，暂不计算规则级分数。",
                    self._default_ai_revision_advice(result),
                )
                continue
            if result.get("error"):
                self._set_score(
                    result,
                    category,
                    maximum_score,
                    None,
                    "该规则审查执行异常，无法计算分数。",
                    self._default_ai_revision_advice(result),
                )
                continue

            issues = result.get("issues") or []
            if not issues:
                self._set_score(
                    result,
                    category,
                    maximum_score,
                    maximum_score,
                    (
                        (
                            f"该项满分{maximum_score}分，没有具体评分细则；"
                            "本次未发现问题，无需扣分，"
                            f"剩余{maximum_score}分。仅供参考。"
                        )
                        if not scoring_explanation
                        else (
                            f"该项满分{maximum_score}分，未发现问题，"
                            f"扣0分，剩余{maximum_score}分。仅供参考。"
                        )
                    ),
                    "无需修改。",
                )
                continue

            candidates.append(
                {
                    "rule_index": rule_index,
                    "maximum_score": maximum_score,
                    "scoring_explanation": scoring_explanation,
                    "scoring_explanation_is_empty": not bool(scoring_explanation),
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
            candidate = candidates_by_index[rule_index]
            maximum_score = candidate["maximum_score"]
            if not 0 <= score_item.score <= maximum_score:
                raise ValueError(
                    f"规则 {rule_index} 的分数 {score_item.score} "
                    f"超出 0 至 {maximum_score} 的范围"
                )
            explanation = str(score_item.explanation or "")
            missing_parts = [
                part
                for part in ("满分", "扣", "剩余")
                if part not in explanation
            ]
            if missing_parts:
                raise ValueError(
                    f"规则 {rule_index} 的扣分说明缺少必要内容: "
                    f"{missing_parts}"
                )
            if (
                candidate.get("scoring_explanation_is_empty")
                and "没有具体评分细则" not in explanation
            ):
                raise ValueError(
                    f"规则 {rule_index} 的评查说明为空，扣分说明必须明确写出"
                    "“没有具体评分细则”"
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
                basis = (
                    "没有具体评分细则，根据已发现的问题数量，"
                    "按每个问题扣满分10%估算"
                    if not explanation.strip()
                    else "说明无可直接计算的扣分幅度，按每个问题扣满分10%估算"
                )

        score = round(max(0.0, maximum_score - deduction), 4)
        actual_deduction = round(maximum_score - score, 4)
        return RuleScoreDraft(
            rule_index=candidate["rule_index"],
            score=score,
            explanation=(
                f"模型评分不可用，采用本地兜底估算：该项满分"
                f"{maximum_score:g}分，{basis}，参考扣分"
                f"{actual_deduction:g}分，参考得分{score:g}分。仅供参考。"
            ),
            ai_revision_advice=(
                "请依据本规则列出的各条问题逐项修正文书，并核对修改后的内容"
                "与案卷事实、原始材料一致。"
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
            maximum_score = self.rules_by_index[result["rule_index"]][
                "评分细则"
            ]["分值"]
            ai_revision_advice = str(score_item.ai_revision_advice).strip()
            if float(score_item.score) == float(maximum_score):
                # 满分、零扣分与“仍存在问题”在结果语义上互相矛盾。这里以
                # 已通过程序校验的评分结论为准统一收口，不依赖模型是否刚好
                # 输出某一句固定措辞，避免任何规则把合规结论留在 issues。
                result["issues"] = []
                ai_revision_advice = "无需修改。"
            else:
                ai_revision_advice = self._location_safe_revision_advice(
                    result.get("issues") or [],
                    ai_revision_advice,
                )
            self._set_score(
                result,
                self.rule_categories_by_index.get(result.get("rule_index")),
                maximum_score,
                score_item.score,
                score_item.explanation,
                ai_revision_advice,
            )
        return scored_results
