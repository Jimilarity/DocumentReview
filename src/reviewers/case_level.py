from pathlib import Path
from typing import Any, Dict, List

from review_config import load_case_level_review_items

from .base import BaseReviewExecutor, ReviewSettings


class CaseLevelReviewExecutor(BaseReviewExecutor):
    """案件级审查入口；完整性审查将在此实现。"""

    def __init__(
        self,
        file_path: str | Path,
        *,
        settings: ReviewSettings | None = None,
        review_items: List[Dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(file_path, settings=settings)
        self.review_items = (
            review_items
            if review_items is not None
            else load_case_level_review_items()
        )

    async def execute_raw(self) -> List[Dict[str, Any]]:
        # 当前配置明确为空，不伪造“完整性通过”结果。
        return []

    def status(self) -> Dict[str, Any]:
        return {
            "executor": type(self).__name__,
            "implemented": False,
            "skipped": True,
            "reason": "not_implemented",
            "item_count": len(self.review_items),
        }
