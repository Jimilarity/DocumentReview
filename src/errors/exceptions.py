from typing import Any, Dict, Optional


class DocumentReviewError(RuntimeError):
    """文书审查系统对外抛出的业务异常基类。"""

    def __init__(
        self,
        error_code: int,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.error_code = int(error_code)
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "error_message": str(self),
            "error_details": self.details,
        }


class PreReviewError(DocumentReviewError):
    """预处理流程整体失败。"""


class ReviewError(DocumentReviewError):
    """规则审查流程整体失败。"""


class PostReviewError(DocumentReviewError):
    """后处理流程整体失败。"""


class ReviewExecutionError(RuntimeError):
    """节点、工具或单规则执行期间的内部异常。"""

    def __init__(self, details: Dict[str, Any]):
        self.details = details

        error_id = details.get("error_id", "unknown")
        stage = details.get("stage", "unknown")
        exception_type = details.get("exception_type", "Exception")
        message = details.get("message", "")
        location = details.get("location") or "unknown location"

        super().__init__(
            f"[{error_id}] {stage} failed at {location}: "
            f"{exception_type}: {message}"
        )
