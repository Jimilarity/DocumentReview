import logging
import os
import traceback
import uuid
from collections import OrderedDict
from contextvars import ContextVar, Token
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, Union

from constants import (
    PROJECT_ROOT,
    RESULT_ROOT,
    REVIEW_ERROR_REPORT_FILENAME,
)
from cache_paths import build_document_key, get_result_directory
from .exceptions import ReviewExecutionError
from utils import atomic_write_json


CURRENT_PDF_PATH: ContextVar[Optional[str]] = ContextVar(
    "CURRENT_PDF_PATH", default=None
)
CURRENT_RULE_INDEX: ContextVar[Optional[Any]] = ContextVar(
    "CURRENT_RULE_INDEX", default=None
)
CURRENT_PAGE_INDEX: ContextVar[Optional[int]] = ContextVar(
    "CURRENT_PAGE_INDEX", default=None
)
CURRENT_SECTION_ID: ContextVar[Optional[int]] = ContextVar(
    "CURRENT_SECTION_ID", default=None
)
CURRENT_NODE: ContextVar[str] = ContextVar("CURRENT_NODE", default="-")


def _resolve_project_path(path: str | Path) -> Path:
    resolved_path = Path(path)
    if resolved_path.is_absolute():
        return resolved_path
    return PROJECT_ROOT / resolved_path


def build_log_task_key(pdf_path: str | Path | None) -> str:
    """为一次以 PDF 路径标识的审查生成稳定、可读的日志目录名。"""

    if not pdf_path:
        return "_system"
    return build_document_key(pdf_path)


class TaskRotatingFileHandler(logging.Handler):
    """按照 CURRENT_PDF_PATH 将日志路由到独立任务目录。"""

    def __init__(
        self,
        base_log_path: str | Path,
        *,
        max_bytes: int,
        backup_count: int,
        encoding: str = "utf-8",
        max_open_tasks: int | None = None,
    ) -> None:
        super().__init__()
        self.base_log_path = _resolve_project_path(base_log_path)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.encoding = encoding
        self.max_open_tasks = max_open_tasks or int(
            os.getenv("LOG_MAX_OPEN_TASK_HANDLERS", "64")
        )
        self._task_handlers: OrderedDict[
            str, RotatingFileHandler
        ] = OrderedDict()

    def task_log_path(self, pdf_path: str | Path | None) -> Path:
        task_key = build_log_task_key(pdf_path)
        return (
            self.base_log_path.parent
            / task_key
            / self.base_log_path.name
        )

    def _get_task_handler(self, pdf_path: str | Path | None) -> RotatingFileHandler:
        task_log_path = self.task_log_path(pdf_path)
        cache_key = str(task_log_path)
        handler = self._task_handlers.pop(cache_key, None)
        if handler is None:
            task_log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(
                task_log_path,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
                encoding=self.encoding,
                delay=True,
            )
        handler.setFormatter(self.formatter)
        self._task_handlers[cache_key] = handler

        while len(self._task_handlers) > self.max_open_tasks:
            _, oldest_handler = self._task_handlers.popitem(last=False)
            oldest_handler.close()
        return handler

    def emit(self, record: logging.LogRecord) -> None:
        try:
            handler = self._get_task_handler(CURRENT_PDF_PATH.get())
            handler.emit(record)
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        for handler in self._task_handlers.values():
            handler.close()
        self._task_handlers.clear()
        super().close()


class _LogContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.pdf_path = CURRENT_PDF_PATH.get() or "-"
        rule_index = CURRENT_RULE_INDEX.get()
        record.rule_index = "-" if rule_index is None else str(rule_index)
        record.node = CURRENT_NODE.get()
        return True


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("document_review")
    if logger.handlers:
        return logger

    log_level = getattr(
        logging,
        os.getenv("REVIEW_LOG_LEVEL", "INFO").upper(),
        logging.INFO,
    )
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | pdf=%(pdf_path)s | "
        "rule=%(rule_index)s | node=%(node)s | %(message)s"
    )
    context_filter = _LogContextFilter()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(context_filter)

    file_handler = TaskRotatingFileHandler(
        os.getenv("REVIEW_LOG_PATH", "logs/review.log"),
        max_bytes=int(
            os.getenv("REVIEW_LOG_MAX_BYTES", str(10 * 1024 * 1024))
        ),
        backup_count=int(os.getenv("REVIEW_LOG_BACKUP_COUNT", "5")),
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(context_filter)

    logger.setLevel(log_level)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


LOGGER = _build_logger()


def build_error_details(
    exc: BaseException,
    stage: str,
    **context: Any,
) -> Dict[str, Any]:
    extracted = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ else []
    last_frame = extracted[-1] if extracted else None
    location = None
    if last_frame is not None:
        location = f"{last_frame.filename}:{last_frame.lineno} in {last_frame.name}"

    return {
        "error_id": uuid.uuid4().hex[:12],
        "stage": stage,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "location": location,
        "pdf_path": CURRENT_PDF_PATH.get(),
        "rule_index": CURRENT_RULE_INDEX.get(),
        "section_id": CURRENT_SECTION_ID.get(),
        "node": CURRENT_NODE.get(),
        "context": context,
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def log_exception(
    exc: BaseException,
    stage: str,
    **context: Any,
) -> Dict[str, Any]:
    details = build_error_details(exc, stage, **context)
    LOGGER.error(
        "error_id=%s stage=%s exception=%s message=%s location=%s context=%s",
        details["error_id"],
        stage,
        details["exception_type"],
        details["message"],
        details["location"],
        context,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return details


def _as_exception(exc: Union[BaseException, str]) -> BaseException:
    if isinstance(exc, BaseException):
        return exc
    return RuntimeError(str(exc))


def _set_temporary_context(
    *,
    pdf_path: Optional[str] = None,
    rule_index: Optional[Any] = None,
    node: Optional[str] = None,
) -> Dict[str, Token[Any]]:
    tokens: Dict[str, Token[Any]] = {}
    if pdf_path is not None:
        tokens["pdf_path"] = CURRENT_PDF_PATH.set(str(pdf_path))
    if rule_index is not None:
        tokens["rule_index"] = CURRENT_RULE_INDEX.set(rule_index)
    if node is not None:
        tokens["node"] = CURRENT_NODE.set(node)
    return tokens


def _reset_temporary_context(tokens: Dict[str, Token[Any]]) -> None:
    if "node" in tokens:
        CURRENT_NODE.reset(tokens["node"])
    if "rule_index" in tokens:
        CURRENT_RULE_INDEX.reset(tokens["rule_index"])
    if "pdf_path" in tokens:
        CURRENT_PDF_PATH.reset(tokens["pdf_path"])


def capture_error(
    stage: str,
    exc: Union[BaseException, str],
    *,
    pdf_path: Optional[str] = None,
    rule_index: Optional[Any] = None,
    node: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """记录异常并返回统一的结构化错误详情。"""
    actual_exc = _as_exception(exc)
    tokens = _set_temporary_context(
        pdf_path=pdf_path,
        rule_index=rule_index,
        node=node,
    )
    try:
        return log_exception(actual_exc, stage, **(extra or {}))
    finally:
        _reset_temporary_context(tokens)


def build_error_state(
    error_code: int,
    stage: str,
    exc: Union[BaseException, str],
    *,
    pdf_path: Optional[str] = None,
    rule_index: Optional[Any] = None,
    node: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """供 LangGraph 节点使用：把异常转换为可写入 State 的字段。"""
    details = capture_error(
        stage,
        exc,
        pdf_path=pdf_path,
        rule_index=rule_index,
        node=node,
        extra=extra,
    )
    return {
        "error_code": int(error_code),
        "error_message": str(exc),
        "error_details": details,
    }


def raise_with_context(
    exc: BaseException,
    stage: str,
    **context: Any,
) -> None:
    details = log_exception(exc, stage, **context)
    raise ReviewExecutionError(details) from exc


def details_from_exception(
    exc: BaseException,
    stage: str,
    **context: Any,
) -> Dict[str, Any]:
    if isinstance(exc, ReviewExecutionError):
        return exc.details
    return log_exception(exc, stage, **context)


def write_error_report(path: Path, details: Dict[str, Any]) -> None:
    """将结构化错误写入指定路径；写入失败只记日志，不覆盖原异常。"""
    try:
        atomic_write_json(path, details)
    except Exception as write_exc:
        log_exception(
            write_exc,
            "write_error_report",
            report_path=str(path),
            original_error_id=details.get("error_id"),
        )


def safe_write_error_report(
    pdf_path: str | Path,
    details: Dict[str, Any],
    filename: str = REVIEW_ERROR_REPORT_FILENAME,
) -> None:
    write_error_report(
        get_result_directory(
            pdf_path,
            result_root=RESULT_ROOT,
        ) / filename,
        details,
    )
