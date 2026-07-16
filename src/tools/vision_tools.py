import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from agent_trace import agent_trace_config
from cache_paths import get_cache_paths
from errors.handler import (
    CURRENT_NODE,
    CURRENT_PAGE_INDEX,
    CURRENT_PDF_PATH,
    CURRENT_SECTION_ID,
    raise_with_context,
)
from message_utils import build_vision_message
from model_config import build_vision_model
from utils import async_read_json


def _require_pdf_path() -> str:
    pdf_path = CURRENT_PDF_PATH.get()
    if not pdf_path:
        raise RuntimeError("当前运行上下文中没有 PDF 路径")
    return pdf_path


def _validate_task(task: str) -> str:
    if not isinstance(task, str):
        raise TypeError(f"task 应为字符串，实际为 {type(task).__name__}")
    normalized_task = task.strip()
    if not normalized_task:
        raise ValueError("task 不能为空")
    return normalized_task


async def _read_list(path: Path, description: str) -> list[Any]:
    value = await async_read_json(path)
    if not isinstance(value, list):
        raise TypeError(
            f"{description}应为列表，实际为 {type(value).__name__}: {path}"
        )
    return value


def _resolve_image_path(raw_path: Any) -> Path:
    if not isinstance(raw_path, (str, Path)):
        raise TypeError(
            "图片路径应为字符串或 Path，"
            f"实际为 {type(raw_path).__name__}"
        )

    image_path = Path(raw_path)
    if not image_path.is_absolute():
        raise ValueError(f"缓存中的图片路径必须为绝对路径: {raw_path}")

    if not image_path.is_file():
        raise FileNotFoundError(f"缓存中的图片文件不存在: {image_path}")

    return image_path


def _extract_response_text(content: Any) -> str:
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
        text = "\n".join(text_parts).strip()
    else:
        text = ""

    if not text:
        raise RuntimeError("多模态模型返回了空文本")
    return text


async def _inspect_images(
    *,
    image_paths: list[Path],
    task: str,
    context_text: str,
) -> str:
    message = await asyncio.to_thread(
        build_vision_message,
        task,
        image_paths,
        additional_texts=[context_text],
    )
    response = await _get_vision_model().ainvoke([message])
    return _extract_response_text(getattr(response, "content", None))


@lru_cache(maxsize=1)
def _get_vision_model():
    return build_vision_model().with_config(
        agent_trace_config("inspect_image_reader")
    )


@tool
async def inspect_page_image(page_index: int, task: str) -> str:
    """查看 PDF 的单页原图。

    page_index 是 image_list.json 中从 0 开始的页面索引；task 应明确说明
    需要从原图核实、识别或描述的元素。返回多模态模型的文本结果。
    """

    node_token = CURRENT_NODE.set("tool.inspect_page_image")
    try:
        task = _validate_task(task)
        pdf_path = _require_pdf_path()
        cache_paths = get_cache_paths(pdf_path)
        image_list = await _read_list(
            cache_paths.image_list,
            "页面图片清单",
        )

        if isinstance(page_index, bool) or not isinstance(page_index, int):
            raise TypeError(
                f"page_index 应为整数，实际为 {type(page_index).__name__}"
            )
        expected_page_index = CURRENT_PAGE_INDEX.get()
        if (
            expected_page_index is not None
            and page_index != expected_page_index
        ):
            raise ValueError(
                "当前预审节点只允许查看正在判断的页面："
                f"当前页={expected_page_index}，请求页={page_index}"
            )
        if not 0 <= page_index < len(image_list):
            raise IndexError(
                f"page_index={page_index} 超出有效范围 "
                f"0..{len(image_list) - 1}"
            )

        image_path = _resolve_image_path(image_list[page_index])
        return await _inspect_images(
            image_paths=[image_path],
            task=task,
            context_text=(
                f"这是 PDF 的原始页面图片，image_list 索引为 {page_index}。"
            ),
        )
    except Exception as exc:
        raise_with_context(
            exc,
            "inspect_page_image",
            page_index=page_index,
            task_preview=str(task)[:200],
        )
    finally:
        CURRENT_NODE.reset(node_token)


@tool
async def inspect_section_images(section_id: int, task: str) -> str:
    """查看目录中指定文书对应的全部原始页面。

    section_id 是目录列表中从 1 开始的文书编号；task 应明确说明需要从
    原图核实、识别或描述的元素。返回多模态模型的文本结果。
    """

    node_token = CURRENT_NODE.set("tool.inspect_section_images")
    try:
        task = _validate_task(task)
        pdf_path = _require_pdf_path()
        cache_paths = get_cache_paths(pdf_path)
        directory = await _read_list(
            cache_paths.directory,
            "目录信息",
        )
        image_list = await _read_list(
            cache_paths.image_list,
            "页面图片清单",
        )

        if isinstance(section_id, bool) or not isinstance(section_id, int):
            raise TypeError(
                f"section_id 应为整数，实际为 {type(section_id).__name__}"
            )
        if not 1 <= section_id <= len(directory):
            raise IndexError(
                f"section_id={section_id} 超出有效范围 1..{len(directory)}"
            )

        section = directory[section_id - 1]
        if not isinstance(section, dict):
            raise TypeError(
                f"目录项 {section_id} 应为对象，"
                f"实际为 {type(section).__name__}"
            )
        if "section_page" not in section:
            raise KeyError(f"目录项 {section_id} 缺少 section_page")

        start_page = int(section["section_page"])
        if not 0 <= start_page < len(image_list):
            raise IndexError(
                f"目录项 {section_id} 的 section_page={start_page} "
                f"超出页面范围 0..{len(image_list) - 1}"
            )

        if section_id == len(directory):
            end_page = len(image_list)
        else:
            next_section = directory[section_id]
            if not isinstance(next_section, dict):
                raise TypeError(
                    f"目录项 {section_id + 1} 应为对象，"
                    f"实际为 {type(next_section).__name__}"
                )
            if "section_page" not in next_section:
                raise KeyError(
                    f"目录项 {section_id + 1} 缺少 section_page"
                )
            next_start_page = int(next_section["section_page"])
            if next_start_page < start_page:
                raise ValueError(
                    f"目录页面范围倒置: {start_page} -> {next_start_page}"
                )
            end_page = min(
                len(image_list),
                max(start_page + 1, next_start_page),
            )

        raw_image_paths = image_list[start_page:end_page]
        if not raw_image_paths:
            raise FileNotFoundError(
                f"目录项 {section_id} 没有对应页面，"
                f"页面范围为 [{start_page}, {end_page})"
            )

        image_paths = [
            _resolve_image_path(raw_path)
            for raw_path in raw_image_paths
        ]
        section_name = str(
            section.get("section_name") or f"目录项 {section_id}"
        )
        return await _inspect_images(
            image_paths=image_paths,
            task=task,
            context_text=(
                f"以下是文书“{section_name}”的原始页面图片；"
                f"section_id={section_id}，"
                f"页面索引范围为 [{start_page}, {end_page})。"
            ),
        )
    except Exception as exc:
        raise_with_context(
            exc,
            "inspect_section_images",
            section_id=section_id,
            task_preview=str(task)[:200],
        )
    finally:
        CURRENT_NODE.reset(node_token)


@tool
async def inspect_current_section_images(task: str) -> str:
    """查看当前单文书审查任务绑定的 section 原始页面。"""

    section_id = CURRENT_SECTION_ID.get()
    if section_id is None:
        raise RuntimeError("当前审查任务没有绑定 section_id")
    return await inspect_section_images.ainvoke(
        {
            "section_id": section_id,
            "task": task,
        }
    )
