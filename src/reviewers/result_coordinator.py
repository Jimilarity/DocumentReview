import copy
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, NoReturn

from cache_paths import get_cache_paths, get_result_directory
from constants import ErrorCode, RULES_PATH
from errors.exceptions import ReviewError
from errors.handler import (
    CURRENT_NODE,
    CURRENT_PDF_PATH,
    CURRENT_RULE_INDEX,
    LOGGER,
    details_from_exception,
    write_error_report,
)
from directory_info import normalize_directory_info
from utils import atomic_write_json, read_json
from .base import ReviewSettings
from .result_processing import NO_REVISION_NEEDED, ReviewResultProcessor
from .rule_scoring import RuleScoreProcessor
from .section_content import section_page_range


REVISION_ADVICE_UNAVAILABLE = "审查结果整理未完成，未生成整体修改建议。"
LEADING_LOCATION_PATTERN = re.compile(
    r"^【(?:"
    r"PDF第(?P<canonical_pdf>\d+)页《(?P<canonical_document>[^》]+)》第(?P<canonical_document_page>\d+)页"
    r"|《(?P<legacy_document>[^》]+)》PDF第(?P<legacy_pdf>\d+)页(?:（文书第(?P<legacy_document_page>\d+)页）)?"
    r"|《(?P<document_only>[^》]+)》文书第(?P<document_only_page>\d+)页"
    r")】"
)
DOCUMENT_NAME_PREFIX_PATTERN = re.compile(r"^《(?P<document>[^》]+)》")


class ReviewResultCoordinator:
    """合并审查范式结果，并统一执行后处理和持久化。"""

    def __init__(
        self,
        file_path: str | Path,
        *,
        settings: ReviewSettings | None = None,
        logger: logging.Logger | None = None,
        rules_path: str | Path = RULES_PATH,
    ) -> None:
        self.file_path = Path(file_path)
        self.settings = settings or ReviewSettings.from_env()
        self.logger = logger or LOGGER
        self.rules_path = Path(rules_path)
        self.cache_paths = get_cache_paths(self.file_path)

    @property
    def result_path(self) -> Path:
        return (
            get_result_directory(
                self.file_path,
                result_root=self.settings.result_root,
            )
            / self.settings.result_filename
        )

    @property
    def error_report_path(self) -> Path:
        return (
            get_result_directory(
                self.file_path,
                result_root=self.settings.result_root,
            )
            / self.settings.error_report_filename
        )

    def save_results(self, results: List[Dict[str, Any]]) -> None:
        atomic_write_json(self.result_path, results)

    def save_raw_results(self, results: List[Dict[str, Any]]) -> None:
        atomic_write_json(self.cache_paths.raw_review_results, results)

    def save_scored_raw_results(
        self,
        results: Dict[str, Any],
    ) -> None:
        atomic_write_json(self.cache_paths.scored_raw_review_results, results)

    def enrich_issue_locations(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """将 issue 定位统一为“PDF真实页 + 该文书内页码”。"""

        enriched = copy.deepcopy(results)
        directory_path = getattr(self.cache_paths, "directory", None)
        ocr_path = getattr(self.cache_paths, "ocr_results", None)
        if (
            not isinstance(directory_path, Path)
            or not directory_path.is_file()
            or not isinstance(ocr_path, Path)
            or not ocr_path.is_file()
        ):
            return enriched

        dir_info = normalize_directory_info(read_json(directory_path))
        ocr_results = read_json(ocr_path)
        directory_by_id = {
            int(item["section_id"]): item for item in dir_info
        }

        def section_location(section_id: int) -> Dict[str, Any] | None:
            section = directory_by_id.get(section_id)
            if section is None:
                return None
            try:
                start_page, end_page = section_page_range(
                    section_id,
                    dir_info,
                    len(ocr_results),
                )
            except (KeyError, StopIteration, TypeError, ValueError):
                return None
            document_name = str(
                section.get("section_name") or "案卷文书"
            ).strip()
            return {
                "section_id": section_id,
                "document": document_name,
                "start": start_page,
                "end": end_page,
            }

        def parse_leading_location(content: str) -> Dict[str, Any] | None:
            match = LEADING_LOCATION_PATTERN.match(content)
            if match is None:
                return None
            groups = match.groupdict()
            pdf_page = groups["canonical_pdf"] or groups["legacy_pdf"]
            document_page = (
                groups["canonical_document_page"]
                or groups["legacy_document_page"]
                or groups["document_only_page"]
            )
            document = (
                groups["canonical_document"]
                or groups["legacy_document"]
                or groups["document_only"]
            )
            return {
                "match": match,
                "pdf_page": int(pdf_page) if pdf_page else None,
                "document_page": (
                    int(document_page) if document_page else None
                ),
                "document": document,
            }

        def canonical_location(
            document: str,
            pdf_page: int,
            document_page: int,
        ) -> str:
            return f"【PDF第{pdf_page}页《{document}》第{document_page}页】"

        for result in enriched:
            for issue in result.get("issues") or []:
                content = str(issue.get("content") or "").strip()
                if not content:
                    continue
                sections = []
                for raw_section_id in issue.get("section_ids") or []:
                    if isinstance(raw_section_id, bool):
                        continue
                    try:
                        section_id = int(raw_section_id)
                    except (TypeError, ValueError):
                        continue
                    location = section_location(section_id)
                    if location is not None and location not in sections:
                        sections.append(location)
                if not sections:
                    continue

                parsed = parse_leading_location(content)
                if parsed is not None:
                    matching_sections = sections
                    if parsed["pdf_page"] is not None:
                        page_index = parsed["pdf_page"] - 1
                        matching_sections = [
                            section
                            for section in matching_sections
                            if section["start"] <= page_index < section["end"]
                        ]
                    name_matches = [
                        section
                        for section in matching_sections
                        if section["document"] == parsed["document"]
                    ]
                    if name_matches:
                        matching_sections = name_matches
                    if len(matching_sections) != 1:
                        continue
                    section = matching_sections[0]
                    pdf_page = parsed["pdf_page"]
                    if pdf_page is None:
                        document_page = parsed["document_page"]
                        if document_page is None:
                            continue
                        pdf_page = section["start"] + document_page
                    document_page = pdf_page - section["start"]
                    if not 1 <= document_page <= section["end"] - section["start"]:
                        continue
                    remainder = content[parsed["match"].end():].lstrip()
                    issue["content"] = canonical_location(
                        section["document"],
                        pdf_page,
                        document_page,
                    ) + remainder
                    content = issue["content"]

                exact_sections = [
                    section
                    for section in sections
                    if section["end"] - section["start"] == 1
                ]
                if not exact_sections:
                    continue
                leading_name = DOCUMENT_NAME_PREFIX_PATTERN.match(content)
                if leading_name and len(exact_sections) == 1:
                    content = content[leading_name.end():].lstrip()
                prefixes = [
                    canonical_location(
                        section["document"],
                        section["start"] + 1,
                        1,
                    )
                    for section in exact_sections
                ]
                missing_prefixes = [
                    prefix for prefix in prefixes if prefix not in content
                ]
                if missing_prefixes:
                    issue["content"] = "".join(missing_prefixes) + content
        return enriched

    @staticmethod
    def _overall_advice_from_scored_results(
        scored_results: List[Dict[str, Any]],
    ) -> str | None:
        """以每条规则已校验的建议为准生成完整、连续编号的总建议。"""

        advice_items: List[str] = []
        numbered_item_pattern = re.compile(
            r"(?:^|\n|(?<=。))\s*\d+[.、．]\s*(?=【|《|请|补|更|删|修|核|将|在)"
        )
        for result in scored_results:
            if result.get("error") or not result.get("issues"):
                continue
            advice = str(result.get("AI修改建议") or "").strip()
            if not advice or advice == NO_REVISION_NEEDED:
                continue
            starts = list(numbered_item_pattern.finditer(advice))
            if not starts:
                advice_items.append(advice)
                continue
            for index, match in enumerate(starts):
                start = match.end()
                end = (
                    starts[index + 1].start()
                    if index + 1 < len(starts)
                    else len(advice)
                )
                item = advice[start:end].strip()
                if item:
                    advice_items.append(item)
        if not advice_items:
            return None
        return "\n".join(
            f"{index}. {item}"
            for index, item in enumerate(advice_items, start=1)
        )

    @staticmethod
    def _normalize_number(value: float | int) -> float | int:
        normalized = round(float(value), 4)
        return int(normalized) if normalized.is_integer() else normalized

    @staticmethod
    def remove_discarded_issues(
        results: List[Dict[str, Any]],
        processing_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """从评分缓存中移除后处理已确认不是最终问题的候选。"""

        discarded = {
            candidate_id
            for candidate_id in (
                processing_result.get("discarded_candidate_ids") or []
            )
            if isinstance(candidate_id, str)
        }
        cleaned = copy.deepcopy(results)
        if not discarded:
            return cleaned

        candidate_number = 0
        for result in cleaned:
            if result.get("error"):
                continue
            retained_issues = []
            for issue in result.get("issues") or []:
                content = str(issue.get("content") or "").strip()
                if not content:
                    retained_issues.append(issue)
                    continue
                candidate_number += 1
                candidate_id = f"C{candidate_number:04d}"
                if candidate_id not in discarded:
                    retained_issues.append(issue)
            result["issues"] = retained_issues
        return cleaned

    @classmethod
    def build_scored_result_payload(
        cls,
        scored_results: List[Dict[str, Any]],
        processing_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        legality_deductions = [
            float(result["扣分"])
            for result in scored_results
            if result.get("规则类别") == "合法性"
            and isinstance(result.get("扣分"), (int, float))
            and not isinstance(result.get("扣分"), bool)
        ]
        compliance_scores = [
            (
                float(result["分数"]),
                float(result["规则满分"]),
            )
            for result in scored_results
            if result.get("规则类别") == "规范性"
            and isinstance(result.get("分数"), (int, float))
            and not isinstance(result.get("分数"), bool)
            and isinstance(result.get("规则满分"), (int, float))
            and not isinstance(result.get("规则满分"), bool)
            and float(result["规则满分"]) > 0
        ]
        problematic_legality_results = [
            result
            for result in scored_results
            if result.get("规则类别") == "合法性"
            and not result.get("error")
            and bool(result.get("issues"))
        ]
        problematic_legality_rule_indexes = sorted(
            {
                int(result["rule_index"])
                for result in problematic_legality_results
                if isinstance(result.get("rule_index"), int)
            }
        )

        advice = cls._overall_advice_from_scored_results(scored_results)
        if advice is None:
            advice = processing_result.get("overall_revision_advice")
        if not isinstance(advice, str) or not advice.strip():
            advice = (
                REVISION_ADVICE_UNAVAILABLE
                if any(result.get("issues") for result in scored_results)
                else NO_REVISION_NEEDED
            )

        def remaining_score(values: List[float]) -> float | int:
            return cls._normalize_number(max(0.0, 100.0 - sum(values)))

        def compliance_score() -> float | int:
            if not compliance_scores:
                return 100
            actual_score = sum(score for score, _ in compliance_scores)
            maximum_score = sum(maximum for _, maximum in compliance_scores)
            return cls._normalize_number(
                max(0.0, min(100.0, 100.0 * actual_score / maximum_score))
            )

        legality_score = (
            0
            if problematic_legality_results
            else remaining_score(legality_deductions)
        )
        normalized_compliance_score = compliance_score()
        total_score = cls._normalize_number(
            max(
                0.0,
                100.0
                - (100.0 - float(legality_score))
                - (100.0 - float(normalized_compliance_score)),
            )
        )

        if compliance_scores:
            compliance_actual_score = cls._normalize_number(
                sum(score for score, _ in compliance_scores)
            )
            compliance_maximum_score = cls._normalize_number(
                sum(maximum for _, maximum in compliance_scores)
            )
            score_explanation = (
                "合规性得分=规范性规则实际得分"
                f"{compliance_actual_score}÷规范性规则满分"
                f"{compliance_maximum_score}×100="
                f"{normalized_compliance_score}。"
            )
        else:
            score_explanation = (
                "本次没有可参与计分的规范性规则，合规性得分按100分计算。"
            )

        if problematic_legality_results:
            rule_indexes = "、".join(
                str(rule_index)
                for rule_index in problematic_legality_rule_indexes
            )
            legality_problem_text = (
                f"规则{rule_indexes}属于合法性规则，且被检查出有问题，"
                if rule_indexes
                else "存在合法性规则被检查出有问题，"
            )
            score_explanation += (
                f" {legality_problem_text}"
                "根据“合法性规则出现问题，此案卷不合格”，"
                f"最终合法性规则得分为0，总得分为{total_score}。"
            )
        else:
            legality_deduction_total = cls._normalize_number(
                sum(legality_deductions)
            )
            score_explanation += (
                " 合法性得分=100-合法性规则扣分合计"
                f"{legality_deduction_total}={legality_score}，"
                f"总得分为{total_score}。"
            )

        return {
            "规则评分": scored_results,
            "评分汇总": {
                "合法性得分": legality_score,
                "合规性得分": normalized_compliance_score,
                "总得分": total_score,
                "评分计算说明": score_explanation,
                "整体修改建议": advice.strip(),
            },
        }

    def save_processing_result(
        self,
        processing_result: Dict[str, Any],
    ) -> None:
        atomic_write_json(
            self.cache_paths.review_result_processing,
            processing_result,
        )

    def create_result_processor(self) -> ReviewResultProcessor:
        return ReviewResultProcessor(
            recursion_limit=self.settings.agent_recursion_limit,
            max_tries=self.settings.result_process_max_tries,
            logger=self.logger,
        )

    async def process_results(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return await self.create_result_processor().process(results)

    def create_rule_score_processor(self) -> RuleScoreProcessor:
        return RuleScoreProcessor(
            rules_path=self.rules_path,
            recursion_limit=self.settings.agent_recursion_limit,
            logger=self.logger,
        )

    async def score_raw_results(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return await self.create_rule_score_processor().process(results)

    def _raise_result_error(
        self,
        exc: BaseException,
        stage: str,
        *,
        message: str,
        result_count: int,
    ) -> NoReturn:
        details = details_from_exception(
            exc,
            stage,
            result_count=result_count,
        )
        write_error_report(self.error_report_path, details)
        raise ReviewError(
            ErrorCode.UNEXPECTED_ERROR,
            (
                f"[{details['error_id']}] {message}: "
                f"{details['exception_type']}: {details['message']}"
            ),
            details=details,
        ) from exc

    async def finalize(
        self,
        raw_results: List[Dict[str, Any]],
        *,
        rule_count: int,
    ) -> Dict[str, Any]:
        pdf_token = CURRENT_PDF_PATH.set(str(self.file_path))
        rule_token = CURRENT_RULE_INDEX.set(None)
        node_token = CURRENT_NODE.set("review_result_coordinator.finalize")
        try:
            raw_results = self.enrich_issue_locations(raw_results)
            try:
                self.save_raw_results(raw_results)
            except Exception as exc:
                self._raise_result_error(
                    exc,
                    "review.write_raw_results",
                    message="审查完成，但原始结果文件写入失败",
                    result_count=len(raw_results),
                )

            processing_enabled = self.settings.result_process_max_tries > 0
            if processing_enabled:
                try:
                    processing_result = await self.process_results(raw_results)
                except Exception as exc:
                    self.logger.exception(
                        "review result processing failed unexpectedly; "
                        "preserving unmerged findings"
                    )
                    processing_result = ReviewResultProcessor.build_fallback_output(
                        ReviewResultProcessor.build_candidates(raw_results),
                        exc,
                    )
                try:
                    self.save_processing_result(processing_result)
                    findings = processing_result["findings"]
                    review_results = findings
                    self.save_results(review_results)
                except Exception as exc:
                    self._raise_result_error(
                        exc,
                        "review.write_results",
                        message="逐规则审查完成，但结果文件写入失败",
                        result_count=len(raw_results),
                    )
            else:
                findings = []
                review_results = raw_results
                processing_result = {
                    "skipped": True,
                    "reason": "max_tries_is_zero",
                }
                try:
                    self.save_processing_result(processing_result)
                    self.save_results(review_results)
                except Exception as exc:
                    self._raise_result_error(
                        exc,
                        "review.write_results",
                        message="审查完成，但结果文件写入失败",
                        result_count=len(raw_results),
                    )
                self.logger.info(
                    "review result processing skipped because max_tries is zero"
                )

            try:
                scoring_input = self.remove_discarded_issues(
                    raw_results,
                    processing_result,
                )
                scored_raw_results = await self.score_raw_results(scoring_input)
                scored_result_payload = self.build_scored_result_payload(
                    scored_raw_results,
                    processing_result,
                )
                self.save_scored_raw_results(scored_result_payload)
            except Exception as exc:
                self._raise_result_error(
                    exc,
                    "review.score_raw_results",
                    message="原始审查结果已生成，但规则级评分流程失败",
                    result_count=len(raw_results),
                )

            self.logger.info(
                "review completed. rule_count=%s finding_count=%s "
                "failed_rule_count=%s result_path=%s",
                rule_count,
                len(findings),
                sum(1 for item in raw_results if item.get("error")),
                self.result_path,
            )
            return {
                "rule_results": raw_results,
                "findings": findings,
                "review_results": review_results,
                "result_processing_enabled": processing_enabled,
                "rule_count": rule_count,
                "result_path": str(self.result_path),
                "raw_result_path": str(
                    self.cache_paths.raw_review_results
                ),
                "scored_raw_result_path": str(
                    self.cache_paths.scored_raw_review_results
                ),
            }
        finally:
            CURRENT_NODE.reset(node_token)
            CURRENT_RULE_INDEX.reset(rule_token)
            CURRENT_PDF_PATH.reset(pdf_token)
