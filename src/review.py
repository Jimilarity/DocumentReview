#review.py
import asyncio
import json
import os
from dataclasses import dataclass
from operator import add
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import RunnableConfig
from pydantic import ValidationError

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

from agent import Documentreview, ReviewResult
from constants import ErrorCode
from error import ReviewError
from error_handler import (
    CURRENT_NODE,
    CURRENT_PDF_PATH,
    CURRENT_RULE_INDEX,
    LOGGER,
    details_from_exception as _details_from_exception,
    log_exception as _log_exception,
    raise_with_context as _raise_with_context,
    safe_write_error_report as _safe_write_error_report,
)
from model_config import build_vision_model
from rule_utils import load_rule_list
from tool_utils import dedupe_tools
from utils import (
    async_read_json,
    atomic_write_json,
    combine_images,
    image_to_base64,
    sanitize_filename,
)


MCP_CONFIG = {
    "law_retrieval_semantic": {
        "transport": "http",
        "url": os.environ.get("LAW_RETRIEVAL_SEMANTIC_URL"),
        "headers": {
            "Authorization": f"Bearer {os.environ.get('LAW_RETRIEVAL_SEMANTIC_KEY')}"
        },
    },
    "law_retrieval_keyword": {
        "transport": "http",
        "url": os.environ.get("LAW_RETRIEVAL_KEYWORD_URL"),
        "headers": {
            "Authorization": f"Bearer {os.environ.get('LAW_RETRIEVAL_KEYWORD_KEY')}"
        },
    },
}


@dataclass
class ReviewRuntime:
    dr_instance: Documentreview
    tool_node: ToolNode
    tools: List[Any]


class ReviewState(TypedDict, total=False):
    pdf_path: str
    rule_list: List[Dict[str, Any]]
    rule_id: int
    rule_results: Annotated[List[Dict[str, Any]], add]
    review_completed: bool
    error_code: Optional[int]
    error_message: Optional[str]
    error_details: Optional[Dict[str, Any]]


class SingleRuleState(TypedDict, total=False):
    rule: Dict[str, Any]
    dir_info: List[Dict[str, Any]]
    meta_info: Dict[str, Any]
    messages: Annotated[List[Any], add_messages]
    result: Optional[Dict[str, Any]]
    error: Optional[str]


@tool
async def call_vlm(section_id: int, task: str) -> str:
    """调用多模态模型，根据指定文书编号和任务描述执行审查辅助任务。"""
    node_token = CURRENT_NODE.set("tool.call_vlm")
    try:
        pdf_path = CURRENT_PDF_PATH.get()
        if not pdf_path:
            raise RuntimeError("CURRENT_PDF_PATH 未设置")

        pdf_name = Path(pdf_path).stem
        dir_path = Path("dir_cache") / pdf_name / "dir_info.json"
        list_path = Path("pdf_cache") / pdf_name / "image_list.json"

        dir_info = await async_read_json(dir_path)
        image_list = await async_read_json(list_path)

        if not isinstance(dir_info, list):
            raise TypeError(f"{dir_path} 应为列表，实际为 {type(dir_info).__name__}")
        if not isinstance(image_list, list):
            raise TypeError(f"{list_path} 应为列表，实际为 {type(image_list).__name__}")
        if not 1 <= section_id <= len(dir_info):
            raise IndexError(
                f"section_id={section_id} 超出有效范围 1..{len(dir_info)}"
            )

        current_section = dir_info[section_id - 1]
        if "section_page" not in current_section:
            raise KeyError(f"目录项 {section_id} 缺少 section_page")
        if "section_name" not in current_section:
            raise KeyError(f"目录项 {section_id} 缺少 section_name")

        start_page = int(current_section["section_page"])
        if section_id == len(dir_info):
            end_page = len(image_list)
        else:
            next_section = dir_info[section_id]
            if "section_page" not in next_section:
                raise KeyError(f"目录项 {section_id + 1} 缺少 section_page")
            end_page = int(next_section["section_page"])
            if end_page == start_page:
                end_page = start_page + 1

        section_images = image_list[start_page:end_page]
        if not section_images:
            raise FileNotFoundError(
                f"目录项 {section_id} 没有对应页面，页面范围为 [{start_page}, {end_page})"
            )

        raw_name = current_section["section_name"]
        section_name = sanitize_filename(raw_name)
        section_path = await asyncio.to_thread(
            combine_images,
            section_images,
            f"{section_name}_{section_id}.jpeg",
        )
        image_base64 = await asyncio.to_thread(image_to_base64, section_path)

        message = HumanMessage(
            content=[
                {"type": "text", "text": task},
                {"type": "text", "text": f"以下是{section_name}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    },
                },
            ]
        )
        response = await build_vision_model().ainvoke([message])
        if not getattr(response, "content", None):
            raise RuntimeError("多模态模型返回了空 content")
        return response.content
    except Exception as exc:
        _raise_with_context(
            exc,
            "call_vlm",
            section_id=section_id,
            task_preview=task[:200],
        )
    finally:
        CURRENT_NODE.reset(node_token)


async def build_review_runtime(file_path: str) -> ReviewRuntime:
    CURRENT_PDF_PATH.set(str(file_path))
    node_token = CURRENT_NODE.set("build_review_runtime")
    tools: List[Any] = [call_vlm]

    try:
        try:
            mcp_client = MultiServerMCPClient(MCP_CONFIG)
            mcp_tools = await mcp_client.get_tools()
            tools.extend(mcp_tools)
            LOGGER.info("MCP tools loaded: %s", [tool.name for tool in mcp_tools])
        except Exception as exc:
            # MCP 检索工具属于可选依赖。失败时继续使用 VLM，但完整 traceback 会写入日志。
            details = _log_exception(exc, "load_mcp_tools")
            LOGGER.warning(
                "MCP tools unavailable; continuing with local tools only. error_id=%s",
                details["error_id"],
            )

        tools = dedupe_tools(tools)
        tool_node = ToolNode(tools, handle_tool_errors=True)
        dr_instance = Documentreview(review_tools=tools)
        return ReviewRuntime(
            dr_instance=dr_instance,
            tool_node=tool_node,
            tools=tools,
        )
    except Exception as exc:
        _raise_with_context(exc, "build_review_runtime")
    finally:
        CURRENT_NODE.reset(node_token)


def _build_single_rule_subgraph(runtime: ReviewRuntime):
    dr_instance = runtime.dr_instance
    tool_node = runtime.tool_node

    async def initialize(state: SingleRuleState, config: RunnableConfig):
        node_token = CURRENT_NODE.set("single_rule.initialize")
        try:
            prompt = dr_instance.build_task_prompt(
                "document_review",
                meta_info=state["meta_info"],
                dir_info=state["dir_info"],
                rule_info=state["rule"],
            )
            return {"messages": [HumanMessage(content=prompt)]}
        except Exception as exc:
            _raise_with_context(exc, "single_rule.initialize")
        finally:
            CURRENT_NODE.reset(node_token)

    async def call_model(state: SingleRuleState, config: RunnableConfig):
        node_token = CURRENT_NODE.set("single_rule.call_model")
        try:
            response = await dr_instance.review_agent.ainvoke(
                {"messages": state["messages"]},
                config=config,
            )
            messages = response.get("messages") if isinstance(response, dict) else None
            if not messages:
                raise RuntimeError(f"review_agent 返回结果缺少 messages: {response!r}")
            return {"messages": [messages[-1]]}
        except Exception as exc:
            _raise_with_context(
                exc,
                "single_rule.call_model",
                message_count=len(state.get("messages", [])),
            )
        finally:
            CURRENT_NODE.reset(node_token)

    def should_continue(state: SingleRuleState):
        node_token = CURRENT_NODE.set("single_rule.route")
        try:
            messages = state.get("messages", [])
            if not messages:
                raise RuntimeError("路由时 messages 为空")
            last_message = messages[-1]
            if getattr(last_message, "tool_calls", None):
                return "tools"
            return "finalize"
        except Exception as exc:
            _raise_with_context(exc, "single_rule.route")
        finally:
            CURRENT_NODE.reset(node_token)

    async def execute_tools(state: SingleRuleState, config: RunnableConfig):
        node_token = CURRENT_NODE.set("single_rule.tools")
        try:
            tool_results = await tool_node.ainvoke(
                state["messages"],
                config=config,
            )
            if tool_results is None:
                raise RuntimeError("ToolNode 返回 None")
            return {"messages": tool_results}
        except Exception as exc:
            _raise_with_context(exc, "single_rule.tools")
        finally:
            CURRENT_NODE.reset(node_token)

    async def finalize(state: SingleRuleState, config: RunnableConfig):
        node_token = CURRENT_NODE.set("single_rule.finalize")
        output_text: Any = None
        json_output_text = ""
        try:
            messages = state.get("messages", [])
            if not messages:
                raise RuntimeError("finalize 时 messages 为空")

            output_text = messages[-1].content
            prompt = dr_instance.build_task_prompt(
                "result_write",
                last_response=output_text,
            )
            response = await dr_instance.write_agent.ainvoke(
                [HumanMessage(content=prompt)],
                config=config,
            )
            json_output_text = response.content.strip()
            data = json.loads(json_output_text)
            result = ReviewResult.model_validate(data).model_dump()
        except (ValidationError, json.JSONDecodeError) as exc:
            details = _log_exception(
                exc,
                "single_rule.finalize.validation",
                writer_output=json_output_text[:4000],
                review_output=str(output_text)[:4000],
            )
            result = {
                "error": "invalid_result_json",
                "error_details": details,
                "raw": output_text,
                "writer_output": json_output_text,
                "comment": [],
                "score": 0,
                "confidence": "low",
            }
        except Exception as exc:
            _raise_with_context(
                exc,
                "single_rule.finalize",
                writer_output=json_output_text[:4000],
                review_output=str(output_text)[:4000],
            )
        finally:
            CURRENT_NODE.reset(node_token)

        result["rule_index"] = state["rule"].get("序号")
        return {"result": result}

    graph = StateGraph(SingleRuleState)
    graph.add_node("initialize", initialize)
    graph.add_node("call_model", call_model)
    graph.add_node("tools", execute_tools)
    graph.add_node("finalize", finalize)
    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "call_model")
    graph.add_conditional_edges(
        "call_model",
        should_continue,
        {"tools": "tools", "finalize": "finalize"},
    )
    graph.add_edge("tools", "call_model")
    graph.add_edge("finalize", END)
    return graph.compile()


async def _distribute_rules(
    state: ReviewState,
    config: RunnableConfig,
    runtime: ReviewRuntime,
) -> Dict[str, Any]:
    node_token = CURRENT_NODE.set("distribute_rules")
    try:
        rule_list = state["rule_list"]
        start_index = state["rule_id"]
        if start_index >= len(rule_list):
            return {"review_completed": True}

        max_concurrency = max(
            1, int(os.getenv("REVIEW_RULE_MAX_CONCURRENCY", "30"))
        )
        timeout_seconds = max(
            1, int(os.getenv("REVIEW_RULE_TIMEOUT_SECONDS", "120"))
        )
        recursion_limit = max(
            1, int(os.getenv("REVIEW_RULE_RECURSION_LIMIT", "20"))
        )

        dir_path = (
            Path("dir_cache") / Path(state["pdf_path"]).stem / "dir_info.json"
        )
        meta_path = (
            Path("meta_cache") / Path(state["pdf_path"]).stem / "meta_info.json"
        )
        meta_info = await async_read_json(meta_path)
        dir_info = await async_read_json(dir_path)
        
        if not isinstance(meta_info, dict):
            raise TypeError(f"{meta_path} 应为字典，实际为 {type(meta_info).__name__}")
        if not isinstance(dir_info, list):
            raise TypeError(f"{dir_path} 应为列表，实际为 {type(dir_info).__name__}")

        subgraph = _build_single_rule_subgraph(runtime)
        queue: asyncio.Queue[int] = asyncio.Queue()
        for index in range(start_index, len(rule_list)):
            queue.put_nowait(index)

        result_slots: List[Optional[Dict[str, Any]]] = [
            None
        ] * (len(rule_list) - start_index)

        def build_error_result(
            index: int,
            exc: BaseException,
            stage: str = "single_rule",
        ) -> Dict[str, Any]:
            rule = rule_list[index]
            details = _details_from_exception(
                exc,
                stage,
                rule_position=index,
                rule=rule,
            )
            LOGGER.error(
                "rule failed. error_id=%s rule_index=%s",
                details["error_id"],
                rule.get("序号"),
            )
            return {
                "rule_index": rule.get("序号"),
                "error": "rule_review_exception",
                "error_details": details,
                "raw": (
                    f"[{details['error_id']}] {details['exception_type']}: "
                    f"{details['message']}"
                ),
                "comment": [
                    {
                        "section_id": [],
                        "content": (
                            "该规则审查执行异常，已保留为待人工复核。"
                            f"错误编号：{details['error_id']}；"
                            f"阶段：{details['stage']}；"
                            f"位置：{details.get('location') or '未知'}；"
                            f"原因：{details['exception_type']}: {details['message']}"
                        ),
                    }
                ],
                "score": 0,
                "confidence": "low",
            }

        async def run_one_rule(index: int) -> Dict[str, Any]:
            rule = rule_list[index]
            rule_token = CURRENT_RULE_INDEX.set(rule.get("序号", index))
            try:
                rule_config = dict(config or {})
                rule_config["recursion_limit"] = recursion_limit
                response = await asyncio.wait_for(
                    subgraph.ainvoke(
                        {
                            "rule": rule,
                            "meta_info": meta_info,
                            "dir_info": dir_info,
                            "messages": [],
                        },
                        config=rule_config,
                    ),
                    timeout=timeout_seconds,
                )
                if not isinstance(response, dict) or "result" not in response:
                    raise RuntimeError(f"单规则子图返回结果缺少 result: {response!r}")

                result = response["result"]
                if not isinstance(result, dict):
                    raise TypeError(
                        f"单规则 result 应为 dict，实际为 {type(result).__name__}"
                    )
                result.setdefault("rule_index", rule.get("序号"))
                result.setdefault("comment", [])
                result.setdefault("score", 0)
                result.setdefault("confidence", "low")
                return result
            except asyncio.TimeoutError as exc:
                return build_error_result(
                    index,
                    exc,
                    stage="single_rule.timeout",
                )
            except Exception as exc:
                return build_error_result(index, exc)
            finally:
                CURRENT_RULE_INDEX.reset(rule_token)

        async def worker(worker_id: int):
            worker_token = CURRENT_NODE.set(f"worker.{worker_id}")
            try:
                while True:
                    try:
                        index = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return

                    try:
                        result_slots[index - start_index] = await run_one_rule(index)
                        LOGGER.info(
                            "worker=%s finished rule=%s",
                            worker_id,
                            rule_list[index].get("序号"),
                        )
                    except Exception as exc:
                        # 防止 worker 自身异常造成 queue.join() 永久等待。
                        result_slots[index - start_index] = build_error_result(
                            index,
                            exc,
                            stage="worker",
                        )
                    finally:
                        queue.task_done()
            finally:
                CURRENT_NODE.reset(worker_token)

        worker_count = min(max_concurrency, len(rule_list) - start_index)
        workers = [
            asyncio.create_task(worker(worker_id), name=f"review-worker-{worker_id}")
            for worker_id in range(worker_count)
        ]
        await queue.join()
        worker_outcomes = await asyncio.gather(*workers, return_exceptions=True)
        for worker_id, outcome in enumerate(worker_outcomes):
            if isinstance(outcome, BaseException):
                _log_exception(
                    outcome,
                    "worker.gather",
                    worker_id=worker_id,
                )

        missing_positions = [
            start_index + offset
            for offset, item in enumerate(result_slots)
            if item is None
        ]
        if missing_positions:
            raise RuntimeError(
                f"以下规则没有生成任何结果: {missing_positions}"
            )

        return {
            "rule_results": [item for item in result_slots if item is not None],
            "rule_id": len(rule_list),
            "review_completed": True,
        }
    except Exception as exc:
        details = _details_from_exception(exc, "distribute_rules")
        return {
            "error_code": int(ErrorCode.UNEXPECTED_ERROR),
            "error_message": (
                f"[{details['error_id']}] {details['exception_type']}: "
                f"{details['message']}"
            ),
            "error_details": details,
        }
    finally:
        CURRENT_NODE.reset(node_token)


def _review_router(state: ReviewState):
    if state.get("error_code") is not None:
        return "error_node"
    if state.get("review_completed", False):
        return END
    if state["rule_id"] < len(state["rule_list"]):
        return "distribute_rules"
    return END


def _error_node(state: ReviewState) -> Dict[str, Any]:
    node_token = CURRENT_NODE.set("error_node")
    try:
        details = state.get("error_details") or {}
        LOGGER.error(
            "review stopped. error_code=%s error_id=%s message=%s",
            state.get("error_code", ErrorCode.UNEXPECTED_ERROR),
            details.get("error_id", "-"),
            state.get("error_message", ""),
        )
        return {}
    finally:
        CURRENT_NODE.reset(node_token)


def _build_review_graph(runtime: ReviewRuntime):
    async def distribute_rules_node(
        state: ReviewState,
        config: RunnableConfig,
    ) -> Dict[str, Any]:
        return await _distribute_rules(state, config, runtime)

    graph = StateGraph(ReviewState)
    graph.add_node("distribute_rules", distribute_rules_node)
    graph.add_node("error_node", _error_node)
    graph.set_entry_point("distribute_rules")
    graph.add_conditional_edges("distribute_rules", _review_router)
    graph.add_edge("error_node", END)
    return graph.compile()


async def run_review(
    file_path: str,
    rule_type: int,
    runtime: ReviewRuntime,
) -> Dict[str, Any]:
    CURRENT_PDF_PATH.set(str(file_path))
    CURRENT_RULE_INDEX.set(None)
    node_token = CURRENT_NODE.set("run_review")
    pdf_name = Path(file_path).stem

    try:
        required_cache_files = [
            Path("pdf_cache") / pdf_name / "image_list.json",
            Path("dir_cache") / pdf_name / "dir_info.json",
            Path("meta_cache") / pdf_name / "meta_info.json",
        ]
        missing_files = [
            str(path) for path in required_cache_files if not path.exists()
        ]
        if missing_files:
            exc = FileNotFoundError(f"预处理缓存缺失: {missing_files}")
            details = _log_exception(
                exc,
                "run_review.precheck",
                missing_files=missing_files,
            )
            _safe_write_error_report(pdf_name, details)
            raise ReviewError(
                ErrorCode.UNEXPECTED_ERROR,
                f"[{details['error_id']}] {exc}",
                details=details,
            ) from exc

        try:
            rule_list, config = load_rule_list(
                rule_type,
                required_cache_files[1],
            )
            LOGGER.info(
                "review started. rule_count=%s rule_type=%s rule_config=%s",
                len(rule_list),
                rule_type,
                config,
            )
            final_state = await _build_review_graph(runtime).ainvoke(
                {
                    "pdf_path": str(file_path),
                    "rule_list": rule_list,
                    "rule_id": 0,
                    "rule_results": [],
                    "review_completed": False,
                }
            )
        except ReviewError:
            raise
        except Exception as exc:
            details = _details_from_exception(exc, "run_review.graph")
            _safe_write_error_report(pdf_name, details)
            raise ReviewError(
                ErrorCode.UNEXPECTED_ERROR,
                (
                    f"[{details['error_id']}] {details['exception_type']}: "
                    f"{details['message']}"
                ),
                details=details,
            ) from exc

        if final_state.get("error_code") is not None:
            details = final_state.get("error_details")
            if not details:
                final_error = RuntimeError(
                    final_state.get("error_message", "审查失败")
                )
                details = _details_from_exception(
                    final_error,
                    "run_review.final_state",
                    error_code=final_state.get("error_code"),
                )
            _safe_write_error_report(pdf_name, details)
            raise ReviewError(
                final_state["error_code"],
                final_state.get("error_message", "审查失败"),
                details=details,
            )

        results = final_state.get("rule_results", [])
        try:
            atomic_write_json(
                Path("results") / pdf_name / "review_results.json",
                results,
            )
        except Exception as exc:
            details = _details_from_exception(
                exc,
                "run_review.write_results",
                result_count=len(results),
            )
            _safe_write_error_report(pdf_name, details)
            raise ReviewError(
                ErrorCode.UNEXPECTED_ERROR,
                (
                    f"[{details['error_id']}] 审查完成，但结果文件写入失败："
                    f"{details['exception_type']}: {details['message']}"
                ),
                details=details,
            ) from exc

        LOGGER.info(
            "review completed. rule_count=%s failed_rule_count=%s",
            len(rule_list),
            sum(1 for item in results if item.get("error")),
        )
        return {
            "rule_results": results,
            "rule_count": len(rule_list),
        }
    finally:
        CURRENT_NODE.reset(node_token)
