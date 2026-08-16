import logging
from pathlib import Path
from typing import Any, Dict, List, NoReturn

from cache_paths import get_cache_paths, get_result_directory
from constants import ErrorCode
from errors.exceptions import ReviewError
from errors.handler import (
    CURRENT_NODE,
    CURRENT_PDF_PATH,
    CURRENT_RULE_INDEX,
    LOGGER,
    details_from_exception,
    write_error_report,
)
from utils import atomic_write_json
from .base import ReviewSettings
from .result_processing import ReviewResultProcessor
from .rule_scoring import RuleScoreProcessor


class ReviewResultCoordinator:
    """合并审查范式结果，并统一执行后处理和持久化。"""

    def __init__(
        self,
        file_path: str | Path,
        *,
        settings: ReviewSettings | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.settings = settings or ReviewSettings.from_env()
        self.logger = logger or LOGGER
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
        results: List[Dict[str, Any]],
    ) -> None:
        atomic_write_json(self.cache_paths.scored_raw_review_results, results)

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
                    self.save_processing_result(processing_result)
                    findings = processing_result["findings"]
                    review_results = findings
                    self.save_results(review_results)
                except Exception as exc:
                    self._raise_result_error(
                        exc,
                        "review.process_results",
                        message="逐规则审查完成，但结果处理失败",
                        result_count=len(raw_results),
                    )
            else:
                findings = []
                review_results = raw_results
                try:
                    self.save_processing_result(
                        {"skipped": True, "reason": "max_tries_is_zero"}
                    )
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
                scored_raw_results = await self.score_raw_results(raw_results)
                self.save_scored_raw_results(scored_raw_results)
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
