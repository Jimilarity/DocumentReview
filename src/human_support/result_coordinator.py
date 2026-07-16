from pathlib import Path
from typing import Any

from cache_paths import get_result_directory
from constants import (
    RETRIEVAL_ENHANCEMENT_RESULT_FILENAME,
    RESULT_ROOT,
)
from reviewers.base import ReviewSettings
from utils import atomic_write_json


class HumanSupportResultCoordinator:
    """人工辅助结果独立持久化，不进入审查结果后处理。"""

    def __init__(
        self,
        file_path: str | Path,
        *,
        settings: ReviewSettings | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.settings = settings or ReviewSettings.from_env()

    @property
    def result_path(self) -> Path:
        result_root = (
            self.settings.result_root
            if self.settings is not None
            else RESULT_ROOT
        )
        return (
            get_result_directory(
                self.file_path,
                result_root=result_root,
            )
            / RETRIEVAL_ENHANCEMENT_RESULT_FILENAME
        )

    def finalize(
        self,
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        atomic_write_json(self.result_path, results)
        return {
            "retrieval_enhancement_results": results,
            "rule_count": len(results),
            "result_path": str(self.result_path),
            "skipped": not bool(results),
            "reason": (
                None
                if results
                else "no_applicable_retrieval_enhancement_rules"
            ),
        }
