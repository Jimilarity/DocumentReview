import json
import logging
from typing import Any, Dict, List

from agents import PostReviewAgents, ReviewResultProcessingOutput


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
            return {"findings": [], "discarded_candidate_ids": []}

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

        raise RuntimeError(
            "审查结果处理连续失败，已达到最大尝试次数 "
            f"{self.max_tries}"
        ) from last_error
