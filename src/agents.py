import json
import os
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Sequence, TypeVar

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from agent_trace import (
    AgentToolTraceMiddleware,
    agent_trace_config,
    trace_event,
)
from constants import RULES_PATH
from errors.handler import CURRENT_PAGE_INDEX
from model_config import build_text_model, build_vision_model
from tools import inspect_current_section_images, inspect_page_image
from utils import extract_json, load_yaml, read_env_bool, strip_thinking_content
from reviewers.document_mapping import (
    document_type_definitions,
    directory_info_for_mapping,
    load_compatible_document_type_groups,
)


# ============================================================
# Pydantic models
# ============================================================

class SchemaModel(BaseModel):

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        str_strip_whitespace=True,
    )


SchemaT = TypeVar("SchemaT", bound=SchemaModel)


NonEmptyText = Annotated[str, StringConstraints(min_length=1)]


class EnforcementOfficer(SchemaModel):
    name: NonEmptyText = Field(alias="执法人")
    certificate_number: NonEmptyText = Field(alias="执法证号")



class CaseMetadata(SchemaModel):
    law_enforcement_unit: NonEmptyText | None = Field(alias="执法单位")
    case_name: NonEmptyText | None = Field(alias="案卷名称")
    case_number: NonEmptyText | None = Field(alias="案号")
    cause_of_action: NonEmptyText | None = Field(alias="案由")
    party: NonEmptyText | None = Field(alias="当事人")
    filing_date: NonEmptyText | None = Field(alias="立案日期")
    closing_date: NonEmptyText | None = Field(alias="结案日期")
    disposition: NonEmptyText | None = Field(alias="处理结果")
    case_officers: (
        list[EnforcementOfficer]
        | NonEmptyText
        | None
    ) = Field(alias="案件承办人员及执法证件号")
    extra_fields: dict[str, str | list[str] | None] = Field(
        alias="额外字段",
    )


class CaseFacts(SchemaModel):
    case_facts: NonEmptyText | None = Field(alias="案情")
    # 无目录案卷会在同一轮关键文书抽取中增量补齐这些案件元数据字段。
    law_enforcement_unit: NonEmptyText | None = Field(
        default=None,
        alias="执法单位",
    )
    case_name: NonEmptyText | None = Field(default=None, alias="案卷名称")
    case_number: NonEmptyText | None = Field(default=None, alias="案号")
    cause_of_action: NonEmptyText | None = Field(default=None, alias="案由")
    party: NonEmptyText | None = Field(default=None, alias="当事人")
    filing_date: NonEmptyText | None = Field(default=None, alias="立案日期")
    closing_date: NonEmptyText | None = Field(default=None, alias="结案日期")
    disposition: NonEmptyText | None = Field(default=None, alias="处理结果")
    case_officers: (
        list[EnforcementOfficer] | NonEmptyText | None
    ) = Field(default=None, alias="案件承办人员及执法证件号")
    extra_fields: dict[str, Any] = Field(
        default_factory=dict,
        alias="额外字段",
    )


class DirectoryItem(SchemaModel):
    section_id: int = Field(gt=0)
    section_name: str = Field(min_length=1)
    section_page: Literal[-1] = -1
    catalog_page: str | int | None = None
    extra_fields: dict[
        str,
        str | int | float | bool | None,
    ] = Field(default_factory=dict)


class DirectoryIdentificationResult(SchemaModel):
    is_directory: bool = Field(alias="is_dir")
    directory_items: list[DirectoryItem] = Field(
        alias="dir_info",
    )

    @model_validator(mode="after")
    def validate_directory_items(self):
        if self.is_directory and not self.directory_items:
            raise ValueError("目录页的 dir_info 不能为空")
        if not self.is_directory and self.directory_items:
            raise ValueError("非目录页的 dir_info 必须为空")
        return self


class SectionIdentificationResult(SchemaModel):
    result: Literal["match", "conflict", "unknown"]


class PageBoundaryResult(SchemaModel):
    relation: Literal["same_document", "new_document"]
    reason_codes: list[str] = Field(default_factory=list)


class SegmentStartClassificationResult(SchemaModel):
    section_kind: Literal[
        "case_document",
        "evidence_material",
        "other_material",
    ]
    normalized_document_type: str | None = None
    material_type: str | None = None
    subject_role: Literal[
        "当事人",
        "法定代表人",
        "委托代理人",
        "执法人员",
        "证人",
        "鉴定检测人员",
        "其他相关人员",
        "无法确定",
        "不适用",
    ] = "无法确定"


class ReviewIssue(SchemaModel):
    section_ids: list[int] = Field(
        default_factory=list,
        description="与该问题相关的案卷章节ID",
    )
    content: str = Field(
        description="只描述实际发现的问题及必要修改建议，不记录通过项"
    )


class ReviewResult(SchemaModel):
    issues: list[ReviewIssue] = Field(
        default_factory=list,
        description=(
            "实际发现的问题列表；不得包含符合要求或审查通过的事项；"
            "没有问题时必须为空列表"
        ),
    )


def _normalize_review_result_payload(payload: Any) -> Any:
    """在校验 ReviewResult 前丢弃模型附带的非协议字段。

    审查结果的 issue 协议只允许 section_ids 和 content。模型偶尔会附带
    detail、reason 等解释字段；这些字段不参与审查结果，应在 Pydantic 校验
    前移除，避免单个额外字段导致整条规则失败。其他结构错误仍交由模型校验
    报错，不做兜底猜测。
    """

    if not isinstance(payload, dict):
        return payload
    issues = payload.get("issues")
    if not isinstance(issues, list):
        return payload
    normalized = dict(payload)
    normalized["issues"] = [
        (
            {key: issue[key] for key in ("section_ids", "content") if key in issue}
            if isinstance(issue, dict)
            else issue
        )
        for issue in issues
    ]
    return normalized


class ProcessedFindingDraft(SchemaModel):
    source_candidate_ids: list[NonEmptyText] = Field(min_length=1)
    content: NonEmptyText

    @model_validator(mode="after")
    def validate_unique_candidate_ids(self):
        if len(self.source_candidate_ids) != len(
            set(self.source_candidate_ids)
        ):
            raise ValueError("source_candidate_ids 不得重复")
        return self


class ReviewResultProcessingOutput(SchemaModel):
    findings: list[ProcessedFindingDraft] = Field(default_factory=list)
    overall_revision_advice: NonEmptyText


class RuleScoreDraft(SchemaModel):
    rule_index: int
    score: float
    explanation: NonEmptyText
    ai_revision_advice: NonEmptyText


class RuleScoringOutput(SchemaModel):
    scores: list[RuleScoreDraft] = Field(default_factory=list)


class DocumentSectionMapping(SchemaModel):
    document_name: str = Field(min_length=1)
    section_ids: list[int] = Field(min_length=1)


class DocumentPresenceResult(SchemaModel):
    document_presence: dict[str, bool]


class DocumentSectionMappingResult(SchemaModel):
    mappings: list[DocumentSectionMapping]

    @model_validator(mode="after")
    def validate_section_ownership(self):
        document_names = [mapping.document_name for mapping in self.mappings]
        if len(document_names) != len(set(document_names)):
            raise ValueError("同一文书类型不得重复出现在映射结果中")
        return self


class SectionFieldExtractionResult(SchemaModel):
    fields: dict[str, Any]


class DeliveryReceiptEventExtraction(SchemaModel):
    source_order: int = Field(ge=1)
    event_text: NonEmptyText
    fields: dict[str, Any]


class DeliveryReceiptExtractionResult(SchemaModel):
    events: list[DeliveryReceiptEventExtraction] = Field(default_factory=list)


class DeliveryReceiptMappingResult(SchemaModel):
    related_section_id: int | None = Field(default=None, gt=0)


class ConsistencyReviewResult(SchemaModel):
    consistent: bool
    reason: NonEmptyText


class SingleUseToolMiddleware(AgentMiddleware):
    """工具一旦出现在当前对话历史中，后续模型轮次不再暴露它。"""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name

    def _was_called(self, messages: Sequence[Any]) -> bool:
        return any(
            tool_call.get("name") == self.tool_name
            for message in messages
            for tool_call in (
                getattr(message, "tool_calls", None) or []
            )
        )

    @staticmethod
    def _tool_name(tool: Any) -> str | None:
        if isinstance(tool, dict):
            function = tool.get("function")
            if isinstance(function, dict):
                return function.get("name")
            return tool.get("name")
        return getattr(tool, "name", None)

    def _prepare_request(self, request: Any) -> Any:
        if not self._was_called(request.messages):
            return request
        remaining_tools = [
            tool
            for tool in request.tools
            if self._tool_name(tool) != self.tool_name
        ]
        if len(remaining_tools) != len(request.tools):
            trace_event(
                "tool_hidden_after_first_call",
                tool_name=self.tool_name,
            )
        return request.override(tools=remaining_tools)

    def wrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        return handler(self._prepare_request(request))

    async def awrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        return await handler(self._prepare_request(request))


# ============================================================
# Base runtime agents
# ============================================================

class BaseAgents:
    """各阶段运行时智能体集合的公共基类。"""

    config_dir = Path(__file__).parent / "config"
    json_object_system_instruction = """\
当前请求使用 json_object 响应格式。最终响应必须是一个合法的 json 对象，
不得在 json 对象前后添加解释、标题、Markdown 或代码块。
"""

    def __init__(
        self,
        *,
        agents_config_name: str = "agents.yaml",
        tasks_config_name: str = "tasks.yaml",
        text_parallel_tool_calls: bool | None = None,
        text_enable_thinking: bool | None = None,
        vision_enable_thinking: bool | None = None,
        text_only: bool | None = None,
    ) -> None:
        self.agents_config = load_yaml(
            self.config_dir / agents_config_name
        )
        self.tasks_config = load_yaml(
            self.config_dir / tasks_config_name
        )

        self.text_model = build_text_model(
            parallel_tool_calls=text_parallel_tool_calls,
            enable_thinking=text_enable_thinking,
        )
        if text_only is None:
            text_only = os.getenv("REVIEW_TEXT_ONLY", "0").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        self.text_only = text_only
        self.vision_model = (
            self.text_model
            if text_only
            else build_vision_model(enable_thinking=vision_enable_thinking)
        )

    def build_agent_prompt(self, agent_name: str) -> str:
        common_prompt = self.agents_config["common"]["system_prompt"]
        config = self.agents_config[agent_name]

        return (
            f"{common_prompt}\n"
            f"角色：\n{config['role']}\n\n"
            f"目标：\n{config['goal']}\n\n"
            f"背景：\n{config['backstory']}"
        )

    def build_task_prompt(self, task_name: str, **kwargs: Any) -> str:
        config = self.tasks_config[task_name]

        description = config["description"].format(**kwargs)
        expected_output = config["expected_output"]

        return (
            f"任务描述：\n{description}\n\n"
            f"输出格式：\n{expected_output}"
        )

    def build_json_object_system_prompt(self, agent_name: str) -> str:
        return (
            f"{self.build_agent_prompt(agent_name)}\n\n"
            f"{self.json_object_system_instruction}"
        )

    @staticmethod
    def message_text(message: Any) -> str:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return strip_thinking_content(content)
        if isinstance(content, list):
            text_parts = [
                item["text"]
                for item in content
                if isinstance(item, dict)
                and isinstance(item.get("text"), str)
            ]
            if text_parts:
                return strip_thinking_content("\n".join(text_parts))
        raise RuntimeError(
            f"模型返回了无法解析的文本内容: {content!r}"
        )

    def create_text_agent(
        self,
        config_name: str,
        tools: Sequence[Any] | None = None,
        response_format: Any = None,
        middleware: Sequence[Any] | None = None,
    ):
        agent = create_agent(
            model=self.text_model,
            tools=list(tools or []),
            system_prompt=self.build_agent_prompt(config_name),
            response_format=response_format,
            middleware=[
                AgentToolTraceMiddleware(config_name),
                *(middleware or []),
            ],
            name=config_name,
        )
        return agent.with_config(agent_trace_config(config_name))

    def create_vision_agent(
        self,
        config_name: str,
        tools: Sequence[Any] | None = None,
        response_format: Any = None,
        middleware: Sequence[Any] | None = None,
        model: Any = None,
    ):
        agent = create_agent(
            model=model or self.vision_model,
            tools=list(tools or []),
            system_prompt=self.build_agent_prompt(config_name),
            response_format=response_format,
            middleware=[
                AgentToolTraceMiddleware(config_name),
                *(middleware or []),
            ],
            name=config_name,
        )
        return agent.with_config(agent_trace_config(config_name))

    def bind_text_tools_with_json_object(
        self,
        tools: Sequence[Any],
    ) -> Any:
        """分别绑定工具和 JSON Object 响应格式，兼容 OpenAI 客户端。"""

        return self.text_model.bind_tools(
            list(tools),
            parallel_tool_calls=False,
            strict=True,
        ).bind(response_format={"type": "json_object"})


# ============================================================
# Pre-review agents
# ============================================================

class PreReviewAgents(BaseAgents):
    """预审图运行时使用的智能体集合。"""

    TOOL_STRATEGY_MODE = "tool_strategy"
    JSON_OBJECT_MODE = "json_object"

    def __init__(self, **kwargs: Any) -> None:
        enable_thinking = read_env_bool(
            "PRE_REVIEW_ENABLE_THINKING"
        )
        kwargs.setdefault("text_parallel_tool_calls", False)
        kwargs.setdefault("text_enable_thinking", enable_thinking)
        kwargs.setdefault("vision_enable_thinking", enable_thinking)
        kwargs.setdefault(
            "text_only",
            os.getenv("REVIEW_TEXT_ONLY", "0").strip().lower()
            in {"1", "true", "yes", "on"},
        )
        self.response_format_mode = self._get_response_format_mode()
        super().__init__(**kwargs)
        self.document_ocr_agent = self.create_vision_agent(
            "document_ocr_agent"
        )
        if self.uses_tool_strategy:
            self._initialize_tool_strategy_agents()
        else:
            self._initialize_json_object_models()
        trace_event(
            "pre_review_response_format_configured",
            mode=self.response_format_mode,
        )

    @property
    def uses_tool_strategy(self) -> bool:
        return self.response_format_mode == self.TOOL_STRATEGY_MODE

    @classmethod
    def _get_response_format_mode(cls) -> str:
        variable_name = "PRE_REVIEW_RESPONSE_FORMAT_MODE"
        raw_value = os.getenv(variable_name, cls.JSON_OBJECT_MODE)
        normalized_value = raw_value.strip().lower().replace("-", "_")
        aliases = {
            "tool": cls.TOOL_STRATEGY_MODE,
            "toolstrategy": cls.TOOL_STRATEGY_MODE,
            "tool_strategy": cls.TOOL_STRATEGY_MODE,
            "json": cls.JSON_OBJECT_MODE,
            "jsonobject": cls.JSON_OBJECT_MODE,
            "json_object": cls.JSON_OBJECT_MODE,
            "native_json": cls.JSON_OBJECT_MODE,
        }
        try:
            return aliases[normalized_value]
        except KeyError as exc:
            raise ValueError(
                f"{variable_name} 仅支持 tool_strategy 或 json_object，"
                f"实际为 {raw_value!r}"
            ) from exc

    def _initialize_tool_strategy_agents(self) -> None:
        self.case_metadata_extractor_agent = self.create_vision_agent(
            "case_metadata_extractor_agent",
            response_format=ToolStrategy(
                CaseMetadata,
                tool_message_content="案卷元数据结构化输出已接收。",
                handle_errors=True,
            ),
        )
        self.case_facts_extractor_agent = self.create_text_agent(
            "case_facts_extractor_agent",
            response_format=ToolStrategy(
                CaseFacts,
                tool_message_content="案件事实结构化输出已接收。",
                handle_errors=True,
            ),
        )
        self.directory_classifier_agent = self.create_text_agent(
            "directory_classifier_agent",
            tools=[] if self.text_only else [inspect_page_image],
            response_format=ToolStrategy(
                DirectoryIdentificationResult,
                tool_message_content="目录页判断结构化输出已接收。",
                handle_errors=True,
            ),
            middleware=[] if self.text_only else [
                SingleUseToolMiddleware(inspect_page_image.name)
            ],
        )
        self.section_classifier_agent = self.create_text_agent(
            "section_classifier_agent",
            tools=[] if self.text_only else [inspect_page_image],
            response_format=ToolStrategy(
                SectionIdentificationResult,
                tool_message_content="文书章节判断结构化输出已接收。",
                handle_errors=True,
            ),
            middleware=[] if self.text_only else [
                SingleUseToolMiddleware(inspect_page_image.name)
            ],
        )
        self.page_boundary_classifier_agent = self.create_text_agent(
            "page_boundary_classifier_agent",
            response_format=ToolStrategy(
                PageBoundaryResult,
                tool_message_content="页间文书边界判断结构化输出已接收。",
                handle_errors=True,
            ),
        )
        self.segment_start_classifier_agent = self.create_text_agent(
            "segment_start_classifier_agent",
            response_format=ToolStrategy(
                SegmentStartClassificationResult,
                tool_message_content="分段起始页分类结构化输出已接收。",
                handle_errors=True,
            ),
        )

    def _initialize_json_object_models(self) -> None:
        json_response_format = {"type": "json_object"}
        self.case_metadata_extractor_agent = self.vision_model.bind(
            response_format=json_response_format,
        ).with_config(
            agent_trace_config("case_metadata_extractor_agent")
        )
        self.case_facts_extractor_agent = self.text_model.bind(
            response_format=json_response_format,
        ).with_config(
            agent_trace_config("case_facts_extractor_agent")
        )
        self.directory_classifier_agent = self.bind_text_tools_with_json_object(
            [] if self.text_only else [inspect_page_image],
        ).with_config(
            agent_trace_config("directory_classifier_agent")
        )
        self.section_classifier_agent = self.bind_text_tools_with_json_object(
            [] if self.text_only else [inspect_page_image],
        ).with_config(
            agent_trace_config("section_classifier_agent")
        )
        self.page_boundary_classifier_agent = self.text_model.bind(
            response_format=json_response_format,
        ).with_config(
            agent_trace_config("page_boundary_classifier_agent")
        )
        self.segment_start_classifier_agent = self.text_model.bind(
            response_format=json_response_format,
        ).with_config(
            agent_trace_config("segment_start_classifier_agent")
        )
        self._classifier_json_result_model = self.text_model.bind(
            response_format=json_response_format,
        ).with_config(
            agent_trace_config("pre_review_classifier_json_result")
        )

    @staticmethod
    def _recursion_limit() -> int:
        variable_name = "PRE_REVIEW_AGENT_RECURSION_LIMIT"
        raw_value = os.getenv(variable_name, "10")
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{variable_name} 必须是正整数，实际为 {raw_value!r}"
            ) from exc
        if value < 3:
            raise ValueError(
                f"{variable_name} 必须大于等于 3，实际为 {value}"
            )
        return value

    @staticmethod
    def _structured_response(
        response: Any,
        response_model: type[SchemaT],
        agent_name: str,
    ) -> SchemaT:
        if not isinstance(response, dict):
            raise TypeError(
                f"{agent_name} 返回值应为字典，"
                f"实际为 {type(response).__name__}"
            )
        if "structured_response" not in response:
            raise RuntimeError(
                f"{agent_name} 返回结果缺少 structured_response"
            )
        structured_response = response["structured_response"]
        if not isinstance(structured_response, response_model):
            raise TypeError(
                f"{agent_name} 的 structured_response 应为 "
                f"{response_model.__name__}，实际为 "
                f"{type(structured_response).__name__}"
            )
        trace_event(
            "agent_structured_response",
            agent_name=agent_name,
            response=structured_response,
        )
        return structured_response

    @classmethod
    def _json_object_response(
        cls,
        response: Any,
        agent_name: str,
    ) -> dict[str, Any]:
        parsed_response = extract_json(cls.message_text(response))
        if not isinstance(parsed_response, dict):
            raise TypeError(
                f"{agent_name} 的 JSON Object 响应应为字典，"
                f"实际为 {type(parsed_response).__name__}"
            )
        trace_event(
            "agent_json_object_response",
            agent_name=agent_name,
            response=parsed_response,
        )
        return parsed_response

    def invoke_case_metadata(
        self,
        message: HumanMessage,
    ) -> CaseMetadata | dict[str, Any]:
        if self.uses_tool_strategy:
            response = self.case_metadata_extractor_agent.invoke(
                {"messages": [message]},
                config={"recursion_limit": self._recursion_limit()},
            )
            return self._structured_response(
                response,
                CaseMetadata,
                "case_metadata_extractor_agent",
            )

        system_message = SystemMessage(
            content=self.build_json_object_system_prompt(
                "case_metadata_extractor_agent"
            )
        )
        response = self.case_metadata_extractor_agent.invoke(
            [system_message, message]
        )
        return self._json_object_response(
            response,
            "case_metadata_extractor_agent",
        )

    async def ainvoke_case_facts(self, prompt_text: str) -> CaseFacts | dict[str, Any]:
        if self.uses_tool_strategy:
            response = await self.case_facts_extractor_agent.ainvoke(
                {"messages": [{"role": "user", "content": prompt_text}]},
                config={"recursion_limit": self._recursion_limit()},
            )
            return self._structured_response(
                response,
                CaseFacts,
                "case_facts_extractor_agent",
            )
        system_message = SystemMessage(
            content=self.build_json_object_system_prompt(
                "case_facts_extractor_agent"
            )
        )
        response = await self.case_facts_extractor_agent.ainvoke(
            [system_message, HumanMessage(content=prompt_text)]
        )
        return self._json_object_response(
            response,
            "case_facts_extractor_agent",
        )

    async def ainvoke_directory_classifier(
        self,
        prompt_text: str,
        page_index: int,
    ) -> DirectoryIdentificationResult | dict[str, Any]:
        if self.uses_tool_strategy:
            return await self._ainvoke_tool_strategy_classifier(
                agent_name="directory_classifier_agent",
                agent=self.directory_classifier_agent,
                response_model=DirectoryIdentificationResult,
                prompt_text=prompt_text,
                page_index=page_index,
            )
        return await self._ainvoke_json_object_classifier(
            agent_name="directory_classifier_agent",
            model=self.directory_classifier_agent,
            prompt_text=prompt_text,
            page_index=page_index,
        )

    async def ainvoke_section_classifier(
        self,
        prompt_text: str,
        page_index: int,
    ) -> SectionIdentificationResult | dict[str, Any]:
        if self.uses_tool_strategy:
            return await self._ainvoke_tool_strategy_classifier(
                agent_name="section_classifier_agent",
                agent=self.section_classifier_agent,
                response_model=SectionIdentificationResult,
                prompt_text=prompt_text,
                page_index=page_index,
            )
        return await self._ainvoke_json_object_classifier(
            agent_name="section_classifier_agent",
            model=self.section_classifier_agent,
            prompt_text=prompt_text,
            page_index=page_index,
        )

    async def ainvoke_page_boundary_classifier(
        self,
        prompt_text: str,
    ) -> PageBoundaryResult:
        if self.uses_tool_strategy:
            response = await self.page_boundary_classifier_agent.ainvoke(
                {"messages": [HumanMessage(content=prompt_text)]},
                config={"recursion_limit": self._recursion_limit()},
            )
            return self._structured_response(
                response,
                PageBoundaryResult,
                "page_boundary_classifier_agent",
            )
        response = await self.page_boundary_classifier_agent.ainvoke(
            [
                SystemMessage(
                    content=self.build_json_object_system_prompt(
                        "page_boundary_classifier_agent"
                    )
                ),
                HumanMessage(content=prompt_text),
            ]
        )
        return PageBoundaryResult.model_validate(
            self._json_object_response(
                response,
                "page_boundary_classifier_agent",
            )
        )

    async def ainvoke_segment_start_classifier(
        self,
        prompt_text: str,
    ) -> SegmentStartClassificationResult:
        if self.uses_tool_strategy:
            response = await self.segment_start_classifier_agent.ainvoke(
                {"messages": [HumanMessage(content=prompt_text)]},
                config={"recursion_limit": self._recursion_limit()},
            )
            return self._structured_response(
                response,
                SegmentStartClassificationResult,
                "segment_start_classifier_agent",
            )
        response = await self.segment_start_classifier_agent.ainvoke(
            [
                SystemMessage(
                    content=self.build_json_object_system_prompt(
                        "segment_start_classifier_agent"
                    )
                ),
                HumanMessage(content=prompt_text),
            ]
        )
        return SegmentStartClassificationResult.model_validate(
            self._json_object_response(
                response,
                "segment_start_classifier_agent",
            )
        )

    async def _ainvoke_tool_strategy_classifier(
        self,
        *,
        agent_name: str,
        agent: Any,
        response_model: type[SchemaT],
        prompt_text: str,
        page_index: int,
    ) -> SchemaT:
        human_message = HumanMessage(content=prompt_text)
        page_token = CURRENT_PAGE_INDEX.set(page_index)
        try:
            response = await agent.ainvoke(
                {"messages": [human_message]},
                config={"recursion_limit": self._recursion_limit()},
            )
            return self._structured_response(
                agent_name=agent_name,
                response=response,
                response_model=response_model,
            )
        finally:
            CURRENT_PAGE_INDEX.reset(page_token)

    async def _ainvoke_json_object_classifier(
        self,
        *,
        agent_name: str,
        model: Any,
        prompt_text: str,
        page_index: int,
    ) -> dict[str, Any]:
        system_message = SystemMessage(
            content=self.build_json_object_system_prompt(agent_name)
        )
        human_message = HumanMessage(content=prompt_text)
        page_token = CURRENT_PAGE_INDEX.set(page_index)
        try:
            response = await model.ainvoke(
                [system_message, human_message]
            )
            tool_calls = list(
                getattr(response, "tool_calls", None) or []
            )

            if not tool_calls:
                return self._json_object_response(response, agent_name)
            if len(tool_calls) != 1:
                raise RuntimeError(
                    "单页分类最多允许一次图片工具调用，"
                    f"实际请求 {len(tool_calls)} 次"
                )

            tool_call = tool_calls[0]
            if tool_call.get("name") != inspect_page_image.name:
                raise RuntimeError(
                    f"不允许的工具调用: {tool_call.get('name')!r}"
                )
            requested_args = tool_call.get("args") or {}
            requested_page_index = requested_args.get("page_index")
            if requested_page_index != page_index:
                raise ValueError(
                    "单页分类只能查看当前页："
                    f"当前页={page_index}，"
                    f"请求页={requested_page_index}"
                )
            task = requested_args.get("task")
            if not isinstance(task, str) or not task.strip():
                raise ValueError("inspect_page_image 的 task 不能为空")

            trace_event(
                "tool_start",
                agent_name=agent_name,
                tool_call=tool_call,
            )
            try:
                tool_result = await inspect_page_image.ainvoke(
                    {
                        "page_index": page_index,
                        "task": task.strip(),
                    }
                )
            except Exception as exc:
                trace_event(
                    "tool_error",
                    agent_name=agent_name,
                    tool_call=tool_call,
                    exception_type=type(exc).__name__,
                    message=str(exc),
                )
                raise
            trace_event(
                "tool_end",
                agent_name=agent_name,
                tool_call=tool_call,
                result=tool_result,
            )
            tool_message = ToolMessage(
                content=str(tool_result),
                tool_call_id=str(
                    tool_call.get("id") or inspect_page_image.name
                ),
                name=inspect_page_image.name,
            )
            final_response = await self._classifier_json_result_model.ainvoke(
                [
                    system_message,
                    human_message,
                    response,
                    tool_message,
                ]
            )
            return self._json_object_response(
                final_response,
                agent_name,
            )
        finally:
            CURRENT_PAGE_INDEX.reset(page_token)


# ============================================================
# Review agents
# ============================================================

class StructuredReviewAgents(BaseAgents):
    """文书审查智能体共用的结构化响应协议。"""

    TOOL_STRATEGY_MODE = "tool_strategy"
    JSON_OBJECT_MODE = "json_object"

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("text_parallel_tool_calls", False)
        kwargs.setdefault(
            "text_enable_thinking",
            read_env_bool(
                "REVIEW_ENABLE_THINKING"
            ),
        )
        self.response_format_mode = self._get_response_format_mode()
        super().__init__(**kwargs)

    @property
    def uses_tool_strategy(self) -> bool:
        return self.response_format_mode == self.TOOL_STRATEGY_MODE

    @classmethod
    def _get_response_format_mode(cls) -> str:
        mode = os.getenv(
            "REVIEW_RESPONSE_FORMAT_MODE",
            cls.JSON_OBJECT_MODE,
        ).strip().lower().replace("-", "_")
        aliases = {
            "tool": cls.TOOL_STRATEGY_MODE,
            "toolstrategy": cls.TOOL_STRATEGY_MODE,
            "tool_strategy": cls.TOOL_STRATEGY_MODE,
            "json": cls.JSON_OBJECT_MODE,
            "jsonobject": cls.JSON_OBJECT_MODE,
            "json_object": cls.JSON_OBJECT_MODE,
        }
        if mode not in aliases:
            raise ValueError(
                "REVIEW_RESPONSE_FORMAT_MODE 仅支持 "
                "tool_strategy 或 json_object"
            )
        return aliases[mode]

    @staticmethod
    def _structured_response(
        response: Any,
        response_model: type[SchemaT],
        agent_name: str,
    ) -> SchemaT:
        structured_response = response.get("structured_response")
        if not isinstance(structured_response, response_model):
            raise TypeError(
                f"{agent_name} 缺少 {response_model.__name__} "
                "类型的 structured_response"
            )
        return structured_response

    @staticmethod
    def _structured_output_max_attempts() -> int:
        variable_name = "REVIEW_STRUCTURED_OUTPUT_MAX_ATTEMPTS"
        raw_value = os.getenv(variable_name, "3")
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{variable_name} 必须是正整数，实际为 {raw_value!r}"
            ) from exc
        if value < 1:
            raise ValueError(
                f"{variable_name} 必须大于等于 1，实际为 {value}"
            )
        return value

    @staticmethod
    def _structured_retry_prompt(
        prompt_text: str,
        response_model: type[SchemaT],
        exc: Exception,
    ) -> str:
        schema = json.dumps(
            response_model.model_json_schema(),
            ensure_ascii=False,
        )
        return (
            f"{prompt_text}\n\n<previous_output_error>\n"
            "上一轮输出未通过程序结构校验。"
            f"错误：{type(exc).__name__}: {exc}\n"
            f"必须返回符合以下 JSON Schema 的对象：{schema}\n"
            "不得返回数字、字符串、数组、解释或 Markdown。"
            "请重新读取原任务并输出完整对象。\n"
            "</previous_output_error>"
        )

    async def _ainvoke_structured_agent(
        self,
        *,
        agent: Any,
        agent_name: str,
        response_model: type[SchemaT],
        prompt_text: str,
        recursion_limit: int,
        force_json_object: bool = False,
    ) -> SchemaT:
        max_attempts = self._structured_output_max_attempts()
        current_prompt = prompt_text
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                if self.uses_tool_strategy and not force_json_object:
                    response = await agent.ainvoke(
                        {"messages": [HumanMessage(content=current_prompt)]},
                        config={"recursion_limit": recursion_limit},
                    )
                    return self._structured_response(
                        response,
                        response_model,
                        agent_name,
                    )

                response = await agent.ainvoke(
                    [
                        SystemMessage(
                            content=self.build_json_object_system_prompt(
                                agent_name
                            )
                        ),
                        HumanMessage(content=current_prompt),
                    ]
                )
                payload = extract_json(self.message_text(response))
                if response_model is ReviewResult:
                    payload = _normalize_review_result_payload(payload)
                return response_model.model_validate(payload)
            except (json.JSONDecodeError, ValidationError, TypeError) as exc:
                last_error = exc
                trace_event(
                    "structured_output_validation_retry",
                    agent_name=agent_name,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    exception_type=type(exc).__name__,
                    message=str(exc),
                )
                if attempt == max_attempts:
                    raise
                current_prompt = self._structured_retry_prompt(
                    prompt_text,
                    response_model,
                    exc,
                )
        assert last_error is not None
        raise last_error


class DocumentMappingAgents(StructuredReviewAgents):
    """共享文书准备服务使用的 section 映射智能体。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.document_presence_classifier_agent = self.text_model.bind(
            response_format={"type": "json_object"},
        ).with_config(
            agent_trace_config("document_presence_classifier_agent")
        )
        if self.uses_tool_strategy:
            self.document_section_mapper_agent = self.create_text_agent(
                "document_section_mapper_agent",
                response_format=ToolStrategy(
                    DocumentSectionMappingResult,
                    tool_message_content="文书章节映射结构化输出已接收。",
                    handle_errors=True,
                ),
            )
        else:
            self.document_section_mapper_agent = self.text_model.bind(
                response_format={"type": "json_object"},
            ).with_config(
                agent_trace_config("document_section_mapper_agent")
            )

    def invoke_document_presence_classifier(
        self,
        prompt_text: str,
    ) -> DocumentPresenceResult:
        response = self.document_presence_classifier_agent.invoke(
            [
                SystemMessage(
                    content=self.build_json_object_system_prompt(
                        "document_presence_classifier_agent"
                    )
                ),
                HumanMessage(content=prompt_text),
            ]
        )
        return DocumentPresenceResult.model_validate(
            extract_json(self.message_text(response))
        )

    def classify_document_presence(
        self,
        document_names: Sequence[str],
        dir_info: Sequence[dict[str, Any]],
        ocr_results: Sequence[dict[str, Any]] | None = None,
        *,
        rules_path: str | Path = RULES_PATH,
    ) -> dict[str, bool]:
        expected_names = list(dict.fromkeys(document_names))
        prompt = self.build_task_prompt(
            "document_presence_check",
            document_names=json.dumps(
                expected_names,
                ensure_ascii=False,
                indent=2,
            ),
            dir_info=json.dumps(
                directory_info_for_mapping(dir_info, ocr_results),
                ensure_ascii=False,
                indent=2,
            ),
            compatible_document_type_groups=json.dumps(
                load_compatible_document_type_groups(),
                ensure_ascii=False,
                indent=2,
            ),
            document_definitions=json.dumps(
                document_type_definitions(expected_names, rules_path),
                ensure_ascii=False,
                indent=2,
            ),
        )
        result = self.invoke_document_presence_classifier(prompt)
        actual_names = set(result.document_presence)
        expected_name_set = set(expected_names)
        if actual_names != expected_name_set:
            missing = sorted(expected_name_set - actual_names)
            unexpected = sorted(actual_names - expected_name_set)
            raise ValueError(
                "文书存在性判断没有严格覆盖输入范围: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return {
            name: result.document_presence[name]
            for name in expected_names
        }

    async def ainvoke_document_section_mapper(
        self,
        prompt_text: str,
        recursion_limit: int,
    ) -> DocumentSectionMappingResult:
        return await self._ainvoke_structured_agent(
            agent=self.document_section_mapper_agent,
            response_model=DocumentSectionMappingResult,
            agent_name="document_section_mapper_agent",
            prompt_text=prompt_text,
            recursion_limit=recursion_limit,
        )


class ContextFreeReviewAgents(StructuredReviewAgents):
    """上下文无关审查使用的逐文书规则审查智能体。"""

    def __init__(
        self,
        review_tools: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.review_tools = list(review_tools or [])
        self.review_tools_by_name = {
            tool.name: tool for tool in self.review_tools
        }
        if self.uses_tool_strategy:
            self.review_agent = self.create_text_agent(
                "context_free_review_agent",
                tools=self.review_tools,
                response_format=ToolStrategy(
                    ReviewResult,
                    tool_message_content="上下文无关审查结构化输出已接收。",
                    handle_errors=True,
                ),
                middleware=[
                    SingleUseToolMiddleware(
                        inspect_current_section_images.name
                    )
                ],
            )
        elif self.review_tools:
            self.review_agent = self.bind_text_tools_with_json_object(
                self.review_tools,
            ).with_config(
                agent_trace_config("context_free_review_agent")
            )
        else:
            self.review_agent = self.text_model.bind(
                response_format={"type": "json_object"},
            ).with_config(
                agent_trace_config("context_free_review_agent")
            )

    async def ainvoke_review(
        self,
        prompt_text: str,
        recursion_limit: int,
    ) -> ReviewResult:
        if self.uses_tool_strategy:
            response = await self.review_agent.ainvoke(
                {"messages": [HumanMessage(content=prompt_text)]},
                config={"recursion_limit": recursion_limit},
            )
            return self._structured_response(
                response,
                ReviewResult,
                "context_free_review_agent",
            )

        messages: list[Any] = [
            SystemMessage(
                content=self.build_json_object_system_prompt(
                    "context_free_review_agent"
                )
            ),
            HumanMessage(content=prompt_text),
        ]
        current_section_image_calls = 0
        for _ in range(recursion_limit):
            response = await self.review_agent.ainvoke(messages)
            tool_calls = list(
                getattr(response, "tool_calls", None) or []
            )
            if not tool_calls:
                return ReviewResult.model_validate(
                    _normalize_review_result_payload(
                        extract_json(self.message_text(response))
                    )
                )

            messages.append(response)
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                if tool_name == inspect_current_section_images.name:
                    current_section_image_calls += 1
                    if current_section_image_calls > 1:
                        raise RuntimeError(
                            "单个 section 审查最多调用一次图片工具"
                        )
                tool = self.review_tools_by_name[tool_name]
                tool_result = await tool.ainvoke(tool_call.get("args") or {})
                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=str(tool_call.get("id") or tool_name),
                        name=tool_name,
                    )
                )

        raise RuntimeError("上下文无关审查智能体超过递归次数限制")


class ContextSensitiveReviewAgents(StructuredReviewAgents):
    """上下文相关审查使用的结构化提取和跨文书判断智能体。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.uses_tool_strategy:
            self.section_field_extractor_agent = self.create_text_agent(
                "section_field_extractor_agent",
                response_format=ToolStrategy(
                    SectionFieldExtractionResult,
                    tool_message_content="文书字段结构化输出已接收。",
                    handle_errors=True,
                ),
            )
            self.delivery_receipt_extractor_agent = self.create_text_agent(
                "delivery_receipt_extractor_agent",
                response_format=ToolStrategy(
                    DeliveryReceiptExtractionResult,
                    tool_message_content="送达事件结构化输出已接收。",
                    handle_errors=True,
                ),
            )
            self.delivery_receipt_mapper_agent = self.create_text_agent(
                "delivery_receipt_mapper_agent",
                response_format=ToolStrategy(
                    DeliveryReceiptMappingResult,
                    tool_message_content="送达事件关联结果已接收。",
                    handle_errors=True,
                ),
            )
            self.consistency_review_agent = self.create_text_agent(
                "consistency_review_agent",
                response_format=ToolStrategy(
                    ConsistencyReviewResult,
                    tool_message_content="一致性判断结果已接收。",
                    handle_errors=True,
                ),
            )
            self.contextual_legality_review_agent = self.create_text_agent(
                "contextual_legality_review_agent",
                response_format=ToolStrategy(
                    ReviewResult,
                    tool_message_content="上下文相关合法性审查结果已接收。",
                    handle_errors=True,
                ),
            )
        else:
            json_response_format = {"type": "json_object"}
            self.section_field_extractor_agent = self.text_model.bind(
                response_format=json_response_format,
            ).with_config(
                agent_trace_config("section_field_extractor_agent")
            )
            self.delivery_receipt_extractor_agent = self.text_model.bind(
                response_format=json_response_format,
            ).with_config(
                agent_trace_config("delivery_receipt_extractor_agent")
            )
            self.delivery_receipt_mapper_agent = self.text_model.bind(
                response_format=json_response_format,
            ).with_config(
                agent_trace_config("delivery_receipt_mapper_agent")
            )
            self.consistency_review_agent = self.text_model.bind(
                response_format=json_response_format,
            ).with_config(
                agent_trace_config("consistency_review_agent")
            )
            self.contextual_legality_review_agent = self.text_model.bind(
                response_format=json_response_format,
            ).with_config(
                agent_trace_config("contextual_legality_review_agent")
            )

    async def ainvoke_section_field_extractor(
        self,
        prompt_text: str,
        recursion_limit: int,
    ) -> SectionFieldExtractionResult:
        return await self._ainvoke_structured_agent(
            agent=self.section_field_extractor_agent,
            agent_name="section_field_extractor_agent",
            response_model=SectionFieldExtractionResult,
            prompt_text=prompt_text,
            recursion_limit=recursion_limit,
        )

    async def ainvoke_delivery_receipt_extractor(
        self,
        prompt_text: str,
        recursion_limit: int,
    ) -> DeliveryReceiptExtractionResult:
        return await self._ainvoke_structured_agent(
            agent=self.delivery_receipt_extractor_agent,
            agent_name="delivery_receipt_extractor_agent",
            response_model=DeliveryReceiptExtractionResult,
            prompt_text=prompt_text,
            recursion_limit=recursion_limit,
        )

    async def ainvoke_delivery_receipt_mapper(
        self,
        prompt_text: str,
        recursion_limit: int,
    ) -> DeliveryReceiptMappingResult:
        return await self._ainvoke_structured_agent(
            agent=self.delivery_receipt_mapper_agent,
            agent_name="delivery_receipt_mapper_agent",
            response_model=DeliveryReceiptMappingResult,
            prompt_text=prompt_text,
            recursion_limit=recursion_limit,
        )

    async def ainvoke_consistency_review(
        self,
        prompt_text: str,
        recursion_limit: int,
    ) -> ConsistencyReviewResult:
        return await self._ainvoke_structured_agent(
            agent=self.consistency_review_agent,
            agent_name="consistency_review_agent",
            response_model=ConsistencyReviewResult,
            prompt_text=prompt_text,
            recursion_limit=recursion_limit,
        )

    async def ainvoke_contextual_legality_review(
        self,
        prompt_text: str,
        recursion_limit: int,
    ) -> ReviewResult:
        return await self._ainvoke_structured_agent(
            agent=self.contextual_legality_review_agent,
            agent_name="contextual_legality_review_agent",
            response_model=ReviewResult,
            prompt_text=prompt_text,
            recursion_limit=recursion_limit,
        )


class PostReviewAgents(StructuredReviewAgents):
    """审查结果协调阶段使用的固定结构化输出智能体。"""

    def __init__(self, **kwargs: Any) -> None:
        kwargs["text_parallel_tool_calls"] = False
        kwargs["text_enable_thinking"] = False
        kwargs["vision_enable_thinking"] = False
        super().__init__(**kwargs)
        if not self.uses_tool_strategy:
            self.result_processor_agent = self.text_model.bind(
                response_format={"type": "json_object"},
            ).with_config(
                agent_trace_config("review_result_processor_agent")
            )
            self.result_processor_json_agent = None
            return
        self.result_processor_agent = self.create_text_agent(
            "review_result_processor_agent",
            response_format=ToolStrategy(
                ReviewResultProcessingOutput,
                tool_message_content="审查结果处理结构化输出已接收。",
                handle_errors=True,
            ),
        )
        if self.uses_tool_strategy:
            self.result_processor_json_agent = self.text_model.bind(
                response_format={"type": "json_object"},
            ).with_config(
                agent_trace_config("review_result_processor_agent_json")
            )

    async def ainvoke_result_processing(
        self,
        prompt_text: str,
        recursion_limit: int,
    ) -> ReviewResultProcessingOutput:
        try:
            return await self._ainvoke_structured_agent(
                agent=self.result_processor_agent,
                agent_name="review_result_processor_agent",
                response_model=ReviewResultProcessingOutput,
                prompt_text=prompt_text,
                recursion_limit=recursion_limit,
            )
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            if self.result_processor_json_agent is None:
                raise
            trace_event(
                "structured_output_format_fallback",
                agent_name="review_result_processor_agent",
                exception_type=type(exc).__name__,
                message=str(exc),
            )
            return await self._ainvoke_structured_agent(
                agent=self.result_processor_json_agent,
                agent_name="review_result_processor_agent",
                response_model=ReviewResultProcessingOutput,
                prompt_text=prompt_text,
                recursion_limit=recursion_limit,
                force_json_object=True,
            )

class RuleScoringAgents(StructuredReviewAgents):
    """根据逐规则原始审查结果计算规则级参考得分。"""

    def __init__(self, **kwargs: Any) -> None:
        kwargs["text_parallel_tool_calls"] = False
        kwargs["text_enable_thinking"] = False
        kwargs["vision_enable_thinking"] = False
        super().__init__(**kwargs)
        if not self.uses_tool_strategy:
            self.rule_scoring_agent = self.text_model.bind(
                response_format={"type": "json_object"},
            ).with_config(
                agent_trace_config("rule_scoring_agent")
            )
            self.rule_scoring_json_agent = None
            return
        self.rule_scoring_agent = self.create_text_agent(
            "rule_scoring_agent",
            response_format=ToolStrategy(
                RuleScoringOutput,
                tool_message_content="规则级评分结构化输出已接收。",
                handle_errors=True,
            ),
        )
        if self.uses_tool_strategy:
            self.rule_scoring_json_agent = self.text_model.bind(
                response_format={"type": "json_object"},
            ).with_config(
                agent_trace_config("rule_scoring_agent_json")
            )

    async def ainvoke_rule_scoring(
        self,
        prompt_text: str,
        recursion_limit: int,
    ) -> RuleScoringOutput:
        try:
            return await self._ainvoke_structured_agent(
                agent=self.rule_scoring_agent,
                agent_name="rule_scoring_agent",
                response_model=RuleScoringOutput,
                prompt_text=prompt_text,
                recursion_limit=recursion_limit,
            )
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            if self.rule_scoring_json_agent is None:
                raise
            trace_event(
                "structured_output_format_fallback",
                agent_name="rule_scoring_agent",
                exception_type=type(exc).__name__,
                message=str(exc),
            )
            return await self._ainvoke_structured_agent(
                agent=self.rule_scoring_json_agent,
                agent_name="rule_scoring_agent",
                response_model=RuleScoringOutput,
                prompt_text=prompt_text,
                recursion_limit=recursion_limit,
                force_json_object=True,
            )

        response = await self.rule_scoring_agent.ainvoke(
            {"messages": [HumanMessage(content=prompt_text)]},
            config={"recursion_limit": recursion_limit},
        )
        structured_response = response.get("structured_response")
        if not isinstance(structured_response, RuleScoringOutput):
            raise TypeError(
                "rule_scoring_agent 缺少 RuleScoringOutput 类型的 "
                "structured_response"
            )
        return structured_response
