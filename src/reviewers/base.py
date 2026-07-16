import logging
import os
from dataclasses import dataclass
from operator import add
from pathlib import Path
from typing import Annotated, Any, Dict, List, NoReturn, Optional, TypedDict

from langgraph.graph import END
from langgraph.types import RunnableConfig

from cache_paths import CachePaths, get_cache_paths, get_result_directory
from constants import (
    ErrorCode,
    RESULT_ROOT,
    REVIEW_ERROR_REPORT_FILENAME,
    REVIEW_RESULT_FILENAME,
)
from directory_info import normalize_directory_info
from errors.exceptions import ReviewError
from errors.handler import (
    CURRENT_NODE,
    CURRENT_PDF_PATH,
    CURRENT_RULE_INDEX,
    LOGGER,
    details_from_exception,
    write_error_report,
)
from rules.rule_set import RuleSet
from tools.registry import dedupe_tools
from utils import async_read_json


class ReviewState(TypedDict, total=False):
    rule_id: int
    rule_results: Annotated[List[Dict[str, Any]], add]
    review_completed: bool
    error_code: Optional[int]
    error_message: Optional[str]
    error_details: Optional[Dict[str, Any]]


class SingleRuleState(TypedDict, total=False):
    rule: Dict[str, Any]
    meta_info: Dict[str, Any]
    result: Optional[Dict[str, Any]]


@dataclass(frozen=True)
class ReviewSettings:
    """审查执行参数；未来派生类可以接收不同配置实例。"""

    model_max_concurrency: int = 5
    task_timeout_seconds: int = 120
    agent_recursion_limit: int = 20
    result_process_max_tries: int = 3
    result_root: Path = RESULT_ROOT
    result_filename: str = REVIEW_RESULT_FILENAME
    error_report_filename: str = REVIEW_ERROR_REPORT_FILENAME

    @classmethod
    def from_env(cls) -> "ReviewSettings":
        return cls(
            model_max_concurrency=int(
                os.getenv("REVIEW_MODEL_MAX_CONCURRENCY", "5")
            ),
            task_timeout_seconds=int(
                os.getenv("REVIEW_TASK_TIMEOUT_SECONDS", "120")
            ),
            agent_recursion_limit=int(
                os.getenv("REVIEW_AGENT_RECURSION_LIMIT", "20")
            ),
            result_process_max_tries=int(
                os.getenv("REVIEW_RESULT_PROCESS_MAX_TRIES", "3")
            ),
        )


@dataclass(frozen=True)
class ReviewContext:
    meta_info: Dict[str, Any]
    dir_info: List[Dict[str, Any]]


class BaseReviewExecutor:
    """所有审查执行器共用的生命周期和错误处理基类。"""

    # 派生类必须将其重写为对应审查范式使用的 Agent 集合类型。
    agents_class: type[Any] | None = None

    def __init__(
        self,
        file_path: str | Path,
        rule_set: RuleSet | None = None,
        *,
        settings: ReviewSettings | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.rule_set = rule_set
        self.settings = settings or ReviewSettings.from_env()
        self.logger = logger or LOGGER
        self.cache_paths: CachePaths = get_cache_paths(self.file_path)

        self.context: ReviewContext | None = None
        self.agents: Any | None = None
        self.single_rule_graph: Any = None
        self.review_graph: Any = None
        self._initialized = False

    @property
    def rules(self) -> List[Dict[str, Any]]:
        if self.rule_set is None:
            return []
        return self.rule_set.rules

    @property
    def result_directory(self) -> Path:
        return get_result_directory(
            self.file_path,
            result_root=self.settings.result_root,
        )

    @property
    def error_report_path(self) -> Path:
        return self.result_directory / self.settings.error_report_filename

    def get_local_tools(self) -> List[Any]:
        """返回审查范式允许使用的本地工具；派生类必须重写。"""

        raise NotImplementedError("派生类必须实现 get_local_tools()")

    def create_agents(self, tools: List[Any]) -> Any:
        """创建审查范式使用的 Agent 集合；派生类必须重写。"""

        raise NotImplementedError("派生类必须实现 create_agents()")

    async def load_context(self) -> ReviewContext:
        """读取公共上下文；派生类可重写并通过 super() 扩展专用字段。"""

        meta_info = await async_read_json(self.cache_paths.metadata)
        dir_info = normalize_directory_info(
            await async_read_json(self.cache_paths.directory)
        )

        return ReviewContext(meta_info=meta_info, dir_info=dir_info)

    def validate_pre_review_cache(self) -> None:
        missing_files = [
            str(path)
            for path in self.cache_paths.pre_review_files.values()
            if not path.is_file()
        ]
        if missing_files:
            raise FileNotFoundError(f"预处理缓存缺失: {missing_files}")

    async def initialize(self) -> None:
        """初始化规则、上下文、工具、Agent 和两层审查图。"""

        if self._initialized:
            return

        node_token = CURRENT_NODE.set("review.initialize")
        try:
            self.validate_pre_review_cache()
            self.context = await self.load_context()

            tools = dedupe_tools(self.get_local_tools())
            self.agents = self.create_agents(tools)
            self.single_rule_graph = self.build_single_rule_graph()
            self.review_graph = self.build_review_graph()
            self._initialized = True

            self.logger.info(
                "review initialized. reviewer=%s rule_count=%s "
                "model_max_concurrency=%s "
                "timeout=%s recursion_limit=%s",
                type(self).__name__,
                len(self.rules),
                self.settings.model_max_concurrency,
                self.settings.task_timeout_seconds,
                self.settings.agent_recursion_limit,
            )
        except ReviewError:
            raise
        except Exception as exc:
            self._raise_review_error(exc, "review.initialize")
        finally:
            CURRENT_NODE.reset(node_token)

    def build_rule_prompt(self, state: SingleRuleState) -> str:
        """构建单规则提示词；派生类必须重写。"""

        raise NotImplementedError("派生类必须实现 build_rule_prompt()")

    def build_single_rule_graph(self):
        """构建单规则执行图；派生类必须重写。"""

        raise NotImplementedError("派生类必须实现 build_single_rule_graph()")

    def rule_identifier(self, rule: Dict[str, Any], fallback: Any = None) -> Any:
        return rule.get("序号", fallback)

    def build_rule_error_result(
        self,
        index: int,
        exc: BaseException,
        *,
        stage: str = "single_rule",
    ) -> Dict[str, Any]:
        rule = self.rules[index]
        details = details_from_exception(
            exc,
            stage,
            rule_position=index,
            rule=rule,
        )
        self.logger.error(
            "rule failed. error_id=%s rule_index=%s",
            details["error_id"],
            self.rule_identifier(rule),
        )
        return {
            "rule_index": self.rule_identifier(rule),
            "error": "rule_review_exception",
            "error_details": details,
            "raw": (
                f"[{details['error_id']}] {details['exception_type']}: "
                f"{details['message']}"
            ),
            "issues": [
                {
                    "section_ids": [],
                    "content": (
                        "该规则审查执行异常，已保留为待人工复核。"
                        f"错误编号：{details['error_id']}；"
                        f"阶段：{details['stage']}；"
                        f"位置：{details.get('location') or '未知'}；"
                        f"原因：{details['exception_type']}: {details['message']}"
                    ),
                }
            ],
        }

    def build_single_rule_input(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> SingleRuleState:
        """构建单规则输入；派生类必须按其审查粒度重写。"""

        raise NotImplementedError("派生类必须实现 build_single_rule_input()")

    def normalize_rule_result(
        self,
        rule: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized = dict(result)
        normalized.setdefault("rule_index", self.rule_identifier(rule))
        normalized.setdefault("issues", [])
        return normalized

    async def distribute_rules(
        self,
        state: ReviewState,
        config: RunnableConfig,
    ) -> Dict[str, Any]:
        """按审查范式分发规则；派生类必须重写。"""

        raise NotImplementedError("派生类必须实现 distribute_rules()")

    def route_review(self, state: ReviewState):
        if state.get("error_code") is not None:
            return "error_node"
        if state.get("review_completed", False):
            return END
        if state["rule_id"] < len(self.rules):
            return "distribute_rules"
        return END

    def handle_review_error(self, state: ReviewState) -> Dict[str, Any]:
        node_token = CURRENT_NODE.set("error_node")
        try:
            details = state.get("error_details") or {}
            self.logger.error(
                "review stopped. error_code=%s error_id=%s message=%s",
                state.get("error_code", ErrorCode.UNEXPECTED_ERROR),
                details.get("error_id", "-"),
                state.get("error_message", ""),
            )
            return {}
        finally:
            CURRENT_NODE.reset(node_token)

    def build_review_graph(self):
        """构建完整审查图；派生类必须重写。"""

        raise NotImplementedError("派生类必须实现 build_review_graph()")

    def build_initial_state(self) -> ReviewState:
        """构建完整审查图初始状态；派生类必须重写。"""

        raise NotImplementedError("派生类必须实现 build_initial_state()")

    def _write_error_report(self, details: Dict[str, Any]) -> None:
        write_error_report(self.error_report_path, details)

    def _raise_review_error(
        self,
        exc: BaseException,
        stage: str,
        *,
        message: str | None = None,
        **context: Any,
    ) -> NoReturn:
        details = details_from_exception(exc, stage, **context)
        self._write_error_report(details)
        error_summary = f"{details['exception_type']}: {details['message']}"
        error_message = (
            f"[{details['error_id']}] {message}: {error_summary}"
            if message
            else f"[{details['error_id']}] {error_summary}"
        )
        raise ReviewError(
            ErrorCode.UNEXPECTED_ERROR,
            error_message,
            details=details,
        ) from exc

    def raise_for_final_state(self, final_state: ReviewState) -> None:
        if final_state.get("error_code") is None:
            return

        details = final_state.get("error_details")
        if not details:
            final_error = RuntimeError(
                final_state.get("error_message", "审查失败")
            )
            details = details_from_exception(
                final_error,
                "review.final_state",
                error_code=final_state.get("error_code"),
            )
        assert isinstance(details, dict)
        self._write_error_report(details)
        raise ReviewError(
            final_state["error_code"],
            final_state.get("error_message", "审查失败"),
            details=details,
        )

    async def execute_raw(self) -> List[Dict[str, Any]]:
        """执行审查图并返回原始逐规则结果，不写共享结果文件。"""

        pdf_path_token = CURRENT_PDF_PATH.set(str(self.file_path))
        rule_index_token = CURRENT_RULE_INDEX.set(None)
        node_token = CURRENT_NODE.set("review.execute_raw")
        try:
            await self.initialize()
            try:
                final_state = await self.review_graph.ainvoke(
                    self.build_initial_state()
                )
            except ReviewError:
                raise
            except Exception as exc:
                self._raise_review_error(exc, "review.graph")

            self.raise_for_final_state(final_state)
            return final_state.get("rule_results", [])
        finally:
            CURRENT_NODE.reset(node_token)
            CURRENT_RULE_INDEX.reset(rule_index_token)
            CURRENT_PDF_PATH.reset(pdf_path_token)


@dataclass(frozen=True)
class DocumentReviewContext(ReviewContext):
    ocr_results: List[Dict[str, Any]]


class DocumentReviewExecutor(BaseReviewExecutor):
    """上下文无关与上下文相关执行器共用的文书审查中间层。"""

    def __init__(
        self,
        file_path: str | Path,
        rule_set: RuleSet,
        *,
        document_section_map: Dict[str, List[int]] | None = None,
        settings: ReviewSettings | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            file_path,
            rule_set,
            settings=settings,
            logger=logger,
        )
        self.document_section_map = {
            name: list(section_ids)
            for name, section_ids in (document_section_map or {}).items()
        }

    async def load_context(self) -> DocumentReviewContext:
        context = await super().load_context()
        ocr_results = await async_read_json(self.cache_paths.ocr_results)
        return DocumentReviewContext(
            meta_info=context.meta_info,
            dir_info=context.dir_info,
            ocr_results=ocr_results,
        )

    def require_document_context(self) -> DocumentReviewContext:
        if not isinstance(self.context, DocumentReviewContext):
            raise RuntimeError("文书审查上下文尚未初始化")
        return self.context
