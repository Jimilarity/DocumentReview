#pre_preview.py
import copy
import json
import re
import time
from operator import add
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict

import fitz
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from tqdm.auto import tqdm

from constants import ErrorCode
from utils import combine_images, image_to_base64, atomic_write_json, extract_json
from error import PreReviewError
from error_handler import build_error_state

class PreReviewState(TypedDict, total=False):
    pdf_path: str
    image_list: List[str]
    meta_info: Dict[str, Any]
    dir_info: List[Dict[str, Any]]
    dir_ident_id: int
    dir_completed: bool
    section_id: int
    
    error_code: Optional[int]
    error_message: Optional[str]
    error_details: Optional[Dict[str, Any]]


def _error(
    code: ErrorCode,
    stage: str,
    exc: Exception | str,
    state: Optional[PreReviewState] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return build_error_state(
        error_code=int(code),
        stage=stage,
        exc=exc,
        pdf_path=state.get("pdf_path") if state else None,
        node=stage,
        extra=extra,
    )

def _directory_not_found(state: PreReviewState) -> Dict[str, Any]:
    exc = RuntimeError("未能在 PDF 中识别出目录")

    return _error(
        ErrorCode.DIR_IDENTIFICATION_ERROR,
        "pre_review.directory_not_found",
        exc,
        state,
    )
    
def _section_not_completed(state: PreReviewState) -> Dict[str, Any]:
    exc = RuntimeError(
        f"已到达 PDF 末页，尚未定位目录项 {state['section_id']}"
    )

    return _error(
        ErrorCode.SECTION_IDENTIFICATION_ERROR,
        "pre_review.section_not_completed",
        exc,
        state,
        extra={
            "page_index": state.get("dir_ident_id"),
            "section_id": state.get("section_id"),
        },
    )


def _pdf_to_image(state: PreReviewState) -> Dict[str, Any]:
    doc = None
    try:
        dpi = 200
        pdf_path = Path(state["pdf_path"])
        output_folder = Path("pdf_cache") / pdf_path.stem
        output_folder.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_path)
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        page_number_width = max(3, len(str(max(0, len(doc) - 1))))
        image_list: List[str] = []

        for page_index in tqdm(range(len(doc)), desc="converting pdf to images..."):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
            image_path = output_folder / f"page_{page_index:0{page_number_width}d}.jpeg"
            pix.save(str(image_path))
            image_list.append(str(image_path))

        if not image_list:
            raise ValueError("PDF 中没有可处理的页面")

        atomic_write_json(output_folder / "image_list.json", image_list)
        return {"image_list": image_list}
    except Exception as exc:
        return _error(
        ErrorCode.PDF_TO_IMAGE_ERROR,
        "pre_review.pdf_to_image",
        exc,
        state,
    )
    finally:
        if doc is not None:
            doc.close()


def _meta_data_extraction(state: PreReviewState, dr_instance: Any) -> Dict[str, Any]:
    try:
        image_list = state["image_list"]
        if not image_list:
            raise ValueError("image_list 为空，无法提取卷宗元数据")

        cover_path = combine_images(
            [image_list[0], image_list[-1]],
            "cover.jpeg",
        )
        image_url = f"data:image/jpeg;base64,{image_to_base64(cover_path)}"
        prompt_text = dr_instance.build_task_prompt("meta_data_extraction")
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        )
        response = dr_instance.meta_agent.invoke({"messages": [message]})

        if "structured_response" in response:
            meta_info = response["structured_response"].model_dump(by_alias=True)
        else:
            meta_info = extract_json(response["messages"][-1].content)

        return {"meta_info": meta_info}
    except Exception as exc:
        return _error(
        ErrorCode.META_DATA_EXTRACTION_ERROR,
        "pre_review.meta_data_extraction",
        exc,
        state,
    )


def _dir_identification(state: PreReviewState, dr_instance: Any) -> Dict[str, Any]:
    try:
        page_index = state["dir_ident_id"]
        image_list = state["image_list"]
        if page_index >= len(image_list):
            raise IndexError("已到达 PDF 末页，但尚未完成目录识别")

        prompt_text = dr_instance.build_task_prompt("dir_identification")
        image_url = f"data:image/jpeg;base64,{image_to_base64(image_list[page_index])}"
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        )

        start_time = time.time()
        response = dr_instance.dir_agent.invoke({"messages": [message]})
        print(f"目录识别响应时间: {time.time() - start_time:.4f} 秒")

        if "structured_response" in response:
            parsed = response["structured_response"]
            dir_info = parsed.dir_info
            is_dir = parsed.is_dir
        else:
            parsed = extract_json(response["messages"][-1].content)
            dir_info = parsed["dir_info"]
            is_dir = parsed["is_dir"]

        existing_dir_info = state["dir_info"]
        if not is_dir and existing_dir_info:
            return {"dir_completed": True}

        return {
            "dir_info": existing_dir_info + dir_info,
            "dir_ident_id": page_index + 1,
        }
    except Exception as exc:
        return _error(
        ErrorCode.DIR_IDENTIFICATION_ERROR,
        "pre_review.dir_identification",
        exc,
        state,
        extra={
            "page_index": state.get("dir_ident_id"),
        },
    )


def _section_identification(state: PreReviewState, dr_instance: Any) -> Dict[str, Any]:
    try:
        page_index = state["dir_ident_id"]
        section_id = state["section_id"]
        image_list = state["image_list"]
        dir_info = state["dir_info"]

        if not dir_info:
            raise ValueError("目录信息为空，无法定位文书章节")
        if page_index >= len(image_list):
            raise IndexError(f"已到达 PDF 末页，尚未定位目录项 {section_id}")

        if section_id == len(dir_info):
            section_info = dir_info[section_id - 1]
        else:
            section_info = dir_info[section_id - 1 : section_id + 1]

        prompt_text = dr_instance.build_task_prompt(
            "section_identification",
            section_info=json.dumps(section_info, ensure_ascii=False),
        )

        content: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt_text},
            {"type": "text", "text": "以下为 picture1"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_to_base64(image_list[page_index])}"
                },
            },
        ]
        if page_index + 1 < len(image_list):
            content.extend(
                [
                    {"type": "text", "text": "以下为 picture2"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_to_base64(image_list[page_index + 1])}"
                        },
                    },
                ]
            )

        response = dr_instance.section_agent.invoke(
            {"messages": [HumanMessage(content=content)]}
        )
        if "structured_response" in response:
            is_belong = response["structured_response"].is_belong
        else:
            is_belong = extract_json(response["messages"][-1].content)

        new_dir_info = copy.deepcopy(dir_info)
        current_section_id = section_id
        picture_count = 0

        for picture_count, (_, value) in enumerate(is_belong.items(), start=1):
            value = int(value)
            current_page = page_index + picture_count - 1
            if current_page >= len(image_list):
                break
            if value == -1:
                continue

            relative_target_id = section_id + value - 1
            absolute_target_id = value
            if relative_target_id == current_section_id:
                target_id = relative_target_id
            elif 1 <= absolute_target_id <= len(dir_info):
                target_id = absolute_target_id
            else:
                continue

            target = new_dir_info[target_id - 1]
            if target.get("section_page", -1) == -1:
                target["section_page"] = current_page
            if target_id >= current_section_id:
                current_section_id = target_id + 1

        next_page_index = min(
            len(image_list), page_index + max(1, picture_count)
        )

        if next_page_index >= len(image_list) and current_section_id <= len(new_dir_info):
            fallback_page = max(0, len(image_list) - 1)
            for index in range(current_section_id - 1, len(new_dir_info)):
                if new_dir_info[index].get("section_page", -1) == -1:
                    new_dir_info[index]["section_page"] = fallback_page
            current_section_id = len(new_dir_info) + 1

        return {
            "dir_info": new_dir_info,
            "section_id": current_section_id,
            "dir_ident_id": next_page_index,
        }
    except Exception as exc:
        return _error(
        ErrorCode.SECTION_IDENTIFICATION_ERROR,
        "pre_review.section_identification",
        exc,
        state,
        extra={
            "page_index": state.get("dir_ident_id"),
            "section_id": state.get("section_id"),
        },
    )


def _save_pre_review_data(state: PreReviewState) -> Dict[str, Any]:
    try:
        pdf_name = Path(state["pdf_path"]).stem
        atomic_write_json(
            Path("dir_cache") / pdf_name / "dir_info.json",
            state["dir_info"],
        )
        atomic_write_json(
            Path("meta_cache") / pdf_name / "meta_info.json",
            state["meta_info"],
        )
        return {}
    except Exception as exc:
        return _error(
        ErrorCode.UNEXPECTED_ERROR,
        "pre_review.save_pre_review_data",
        exc,
        state,
    )


def _route_after_node(success_node: str):
    def router(state: PreReviewState):
        if state.get("error_code") is not None:
            return "error_node"
        return success_node

    return router


def _route_dir_identification(state: PreReviewState):
    if state.get("error_code") is not None:
        return "error_node"
    if state.get("dir_completed", False):
        return "section_identification"
    if state["dir_ident_id"] < len(state["image_list"]):
        return "dir_identification"
    return "directory_not_found"




def _route_section_identification(state: PreReviewState):
    if state.get("error_code") is not None:
        return "error_node"
    if state["section_id"] > len(state["dir_info"]):
        return "save_pre_review_data"
    if state["dir_ident_id"] < len(state["image_list"]):
        return "section_identification"
    return "section_not_completed"




def _error_node(state: PreReviewState) -> Dict[str, Any]:
    print(
        f"PRE_REVIEW STOPPED, ERROR CODE: {state.get('error_code', ErrorCode.UNEXPECTED_ERROR)}"
    )
    print(state.get("error_message", ""))
    return {}


def build_pre_review_graph(dr_instance: Any):
    workflow = StateGraph(PreReviewState)
    workflow.add_node("pdf_to_image", _pdf_to_image)
    workflow.add_node(
        "meta_data_extraction",
        lambda state: _meta_data_extraction(state, dr_instance),
    )
    workflow.add_node(
        "dir_identification",
        lambda state: _dir_identification(state, dr_instance),
    )
    workflow.add_node(
        "section_identification",
        lambda state: _section_identification(state, dr_instance),
    )
    workflow.add_node("save_pre_review_data", _save_pre_review_data)
    workflow.add_node("directory_not_found", _directory_not_found)
    workflow.add_node("section_not_completed", _section_not_completed)
    workflow.add_node("error_node", _error_node)

    workflow.set_entry_point("pdf_to_image")
    workflow.add_conditional_edges(
        "pdf_to_image",
        _route_after_node("meta_data_extraction"),
    )
    workflow.add_conditional_edges(
        "meta_data_extraction",
        _route_after_node("dir_identification"),
    )
    workflow.add_conditional_edges("dir_identification", _route_dir_identification)
    workflow.add_conditional_edges(
        "section_identification",
        _route_section_identification,
    )
    workflow.add_conditional_edges(
        "save_pre_review_data",
        _route_after_node(END),
    )
    workflow.add_edge("directory_not_found", "error_node")
    workflow.add_edge("section_not_completed", "error_node")
    workflow.add_edge("error_node", END)
    return workflow.compile()


async def run_pre_review(
    file_path: str,
    dr_instance: Any,
) -> Dict[str, Any]:
    initial_state: PreReviewState = {
        "pdf_path": str(file_path),
        "image_list": [],
        "meta_info": {},
        "dir_info": [],
        "dir_ident_id": 0,
        "dir_completed": False,
        "section_id": 1,
    }

    final_state = await build_pre_review_graph(dr_instance).ainvoke(
        initial_state
    )

    if final_state.get("error_code") is not None:
        raise PreReviewError(
            final_state["error_code"],
            final_state.get("error_message", "预处理失败"),
            details=final_state.get("error_details"),
        )

    return final_state
