import json
import logging
import re
from typing import Any, Dict, List

from agents import (
    PostReviewAgents,
    ProcessedFindingDraft,
    ReviewResultProcessingOutput,
)


NO_REVISION_NEEDED = "本次评查未发现需要修改的问题。"
LOCATION_PATTERN = re.compile(
    r"【[^】]*(?:PDF第|文书第)[^】]*】"
)


class ReviewResultProcessor:
    """将逐规则 issue 处理为去重后的 finding。"""

    def __init__(
        self,
        *,
        agents: PostReviewAgents | None = None,
        recursion_limit: int = 20,
        max_tries: int = 3,
        logger: logging.Logger | None = None,
    ) -> None:
        self.agents = agents
        self.recursion_limit = recursion_limit
        self.max_tries = max_tries
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def build_candidates(
        rule_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        candidates = []
        for result in rule_results:
            if result.get("error"):
                continue
            for issue in result.get("issues", []):
                content = issue.get("content", "").strip()
                if not content:
                    continue
                candidates.append(
                    {
                        "candidate_id": f"C{len(candidates) + 1:04d}",
                        "rule_index": result["rule_index"],
                        "section_ids": issue["section_ids"],
                        "content": content,
                    }
                )
        return candidates

    @staticmethod
    def validate_result(
        result: ReviewResultProcessingOutput,
        candidates: List[Dict[str, Any]],
    ) -> None:
        candidates_by_id = {
            candidate["candidate_id"]: candidate
            for candidate in candidates
        }

        for finding in result.findings:
            for candidate_id in finding.source_candidate_ids:
                if candidate_id not in candidates_by_id:
                    raise ValueError(
                        f"finding 引用了未知 candidate_id: {candidate_id}"
                    )
            source_locations = {
                location
                for candidate_id in finding.source_candidate_ids
                for location in LOCATION_PATTERN.findall(
                    candidates_by_id[candidate_id]["content"]
                )
            }
            missing_locations = sorted(
                location
                for location in source_locations
                if location not in finding.content
            )
            if missing_locations:
                raise ValueError(
                    "finding 删除或改写了候选中的问题位置: "
                    f"{missing_locations}"
                )

        finding_locations = {
            location
            for finding in result.findings
            for location in LOCATION_PATTERN.findall(finding.content)
        }
        missing_advice_locations = sorted(
            location
            for location in finding_locations
            if location not in result.overall_revision_advice
        )
        if missing_advice_locations:
            raise ValueError(
                "overall_revision_advice 未保留 finding 中的问题位置: "
                f"{missing_advice_locations}"
            )
        invented_advice_locations = sorted(
            location
            for location in LOCATION_PATTERN.findall(
                result.overall_revision_advice
            )
            if location not in finding_locations
        )
        if invented_advice_locations:
            raise ValueError(
                "overall_revision_advice 引入了 finding 中不存在的位置: "
                f"{invented_advice_locations}"
            )

    @staticmethod
    def restore_locations(
        result: ReviewResultProcessingOutput,
        candidates: List[Dict[str, Any]],
    ) -> ReviewResultProcessingOutput:
        """用候选中的原始定位标签修复模型删改，避免为格式问题重跑整卷。"""

        candidates_by_id = {
            candidate["candidate_id"]: candidate
            for candidate in candidates
        }
        findings: list[ProcessedFindingDraft] = []
        finding_locations: list[str] = []
        for finding in result.findings:
            expected_locations: list[str] = []
            for candidate_id in finding.source_candidate_ids:
                candidate = candidates_by_id.get(candidate_id)
                if candidate is None:
                    continue
                for location in LOCATION_PATTERN.findall(candidate["content"]):
                    if location not in expected_locations:
                        expected_locations.append(location)
            content = str(finding.content or "").strip()
            missing = [
                location
                for location in expected_locations
                if location not in content
            ]
            if missing:
                content = "".join(missing) + content
            findings.append(
                ProcessedFindingDraft(
                    source_candidate_ids=finding.source_candidate_ids,
                    content=content,
                )
            )
            for location in LOCATION_PATTERN.findall(content):
                if location not in finding_locations:
                    finding_locations.append(location)

        advice = str(result.overall_revision_advice or "").strip()
        advice = LOCATION_PATTERN.sub(
            lambda match: (
                match.group(0)
                if match.group(0) in finding_locations
                else ""
            ),
            advice,
        ).strip()
        missing_advice_locations = [
            location
            for location in finding_locations
            if location not in advice
        ]
        if missing_advice_locations:
            advice = "".join(missing_advice_locations) + advice
        return ReviewResultProcessingOutput(
            findings=findings,
            overall_revision_advice=advice,
        )

    @classmethod
    def build_fallback_output(
        cls,
        candidates: List[Dict[str, Any]],
        error: Exception | None = None,
    ) -> Dict[str, Any]:
        """模型整理持续失败时保留全部原始问题，不能让整卷评查失败。"""

        findings = [
            ProcessedFindingDraft(
                source_candidate_ids=[candidate["candidate_id"]],
                content=candidate["content"],
            )
            for candidate in candidates
        ]
        locations = list(
            dict.fromkeys(
                location
                for candidate in candidates
                for location in LOCATION_PATTERN.findall(candidate["content"])
            )
        )
        advice = (
            "".join(locations)
            + "结果自动归并未完成，请按上述逐项问题修改并复核。"
        )
        output = cls.build_output(
            ReviewResultProcessingOutput(
                findings=findings,
                overall_revision_advice=advice,
            ),
            candidates,
        )
        output["fallback"] = True
        if error is not None:
            output["fallback_reason"] = (
                f"{type(error).__name__}: {error}"
            )
        return output

    @staticmethod
    def build_output(
        result: ReviewResultProcessingOutput,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        candidates_by_id = {
            candidate["candidate_id"]: candidate
            for candidate in candidates
        }
        used_candidate_ids = set()
        findings = []
        for index, finding in enumerate(result.findings, start=1):
            source_candidates = [
                candidates_by_id[candidate_id]
                for candidate_id in finding.source_candidate_ids
            ]
            used_candidate_ids.update(finding.source_candidate_ids)
            findings.append(
                {
                    "finding_id": f"F{index:04d}",
                    "source_candidate_ids": finding.source_candidate_ids,
                    "rule_indexes": sorted(
                        {
                            candidate["rule_index"]
                            for candidate in source_candidates
                        }
                    ),
                    "section_ids": sorted(
                        {
                            section_id
                            for candidate in source_candidates
                            for section_id in candidate["section_ids"]
                        }
                    ),
                    "content": finding.content,
                }
            )
        return {
            "findings": findings,
            "overall_revision_advice": result.overall_revision_advice,
            "discarded_candidate_ids": [
                candidate["candidate_id"]
                for candidate in candidates
                if candidate["candidate_id"] not in used_candidate_ids
            ],
        }

    async def process(
        self,
        rule_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        candidates = self.build_candidates(rule_results)
        if not candidates:
            return {
                "findings": [],
                "overall_revision_advice": NO_REVISION_NEEDED,
                "discarded_candidate_ids": [],
            }

        agents = self.agents or PostReviewAgents()
        base_prompt = agents.build_task_prompt(
            "review_result_processing",
            candidates=json.dumps(
                candidates,
                ensure_ascii=False,
                indent=2,
            ),
        )

        last_error: Exception | None = None
        for attempt in range(self.max_tries):
            prompt_text = base_prompt
            if last_error is not None:
                prompt_text += (
                    "\n\n上一次结构化结果未通过程序校验：\n"
                    f"{type(last_error).__name__}: {last_error}\n"
                    "请重新处理全部候选并提交完整结果。"
                )
            try:
                result = await agents.ainvoke_result_processing(
                    prompt_text,
                    self.recursion_limit,
                )
                result = self.restore_locations(result, candidates)
                self.validate_result(result, candidates)
                return self.build_output(result, candidates)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_tries:
                    self.logger.warning(
                        "review result processing failed; retrying "
                        "attempt=%s/%s error=%s: %s",
                        attempt + 1,
                        self.max_tries,
                        type(exc).__name__,
                        exc,
                    )

        self.logger.error(
            "review result processing exhausted retries; using fallback "
            "attempts=%s error=%s: %s",
            self.max_tries,
            type(last_error).__name__ if last_error else "UnknownError",
            last_error,
        )
        return self.build_fallback_output(candidates, last_error)
