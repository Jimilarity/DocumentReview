import asyncio
import copy
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional, TypedDict

import fitz
from langgraph.graph import END, StateGraph
from tqdm.auto import tqdm

from agents import PreReviewAgents
from agent_trace import trace_event
from cache_paths import get_cache_paths
from directory_info import normalize_directory_info
from constants import ErrorCode
from errors.exceptions import PreReviewError
from errors.handler import (
    CURRENT_PAGE_INDEX,
    CURRENT_PDF_PATH,
    build_error_state,
)
from utils import atomic_write_json
from message_utils import build_vision_message


_OCR_DIV_PATTERN = re.compile(
    r'<div\b(?P<attributes>[^>]*)>(?P<content>.*?)</div>',
    re.IGNORECASE | re.DOTALL,
)
_OCR_TAG_PATTERN = re.compile(r'<[^>]+>')
_OCR_TYPE_PATTERN = re.compile(
    r'\btype=["\'](?P<type>[^"\']+)["\']',
    re.IGNORECASE,
)
_OCR_SUBSTANTIVE_THRESHOLD = 80
_PAGE_NUMBER_PATTERN = re.compile(
    r'<div\b(?P<attributes>[^>]*)>\s*(?P<content>[^<]*)</div>',
    re.IGNORECASE | re.DOTALL,
)
_SECTION_PROBE_OFFSETS = (0, -1, 1, -2, 2)
_SECTION_OCR_MAX_CHARS = 4000


def _ocr_substantive_character_count(content: str) -> int:
    """计算 OCR 中非印章、非页码的可读内容长度。"""

    fragments: list[str] = []
    cursor = 0
    for match in _OCR_DIV_PATTERN.finditer(content):
        fragments.append(content[cursor:match.start()])
        cursor = match.end()
        type_match = _OCR_TYPE_PATTERN.search(match.group("attributes"))
        element_type = type_match.group("type").lower() if type_match else ""
        if element_type not in {"imprint", "page-number"}:
            fragments.append(match.group("content"))
    fragments.append(content[cursor:])
    readable_text = _OCR_TAG_PATTERN.sub("", "".join(fragments))
    return len(re.sub(r"\s+", "", readable_text))


def _needs_ocr_fallback(content: str) -> bool:
    """识别疑似只输出印章或页码、遗漏正文的 OCR 结果。"""

    imprint_count = len(re.findall(r'\btype=["\']imprint["\']', content))
    return (
        imprint_count >= 3
        and _ocr_substantive_character_count(content)
        < _OCR_SUBSTANTIVE_THRESHOLD
    )


def _prefer_retry_ocr(primary: str, retry: str) -> str:
    """仅在重读结果包含更多有效正文时替换首次 OCR。"""

    if _ocr_substantive_character_count(retry) > _ocr_substantive_character_count(
        primary
    ):
        return retry
    return primary


class DocumentOcrResult(TypedDict):
    document_content: str
    image_path: str
    image_index: int


class PreReviewState(TypedDict, total=False):
    pdf_path: str
    page_image_paths: list[str]
    case_metadata: dict[str, Any]

    ocr_page_index: int
    ocr_max_concurrency: int
    document_ocr_results: list[DocumentOcrResult]

    document_catalog: list[dict[str, Any]]
    catalog_page_index: int
    catalog_scan_completed: bool
    body_start_page_index: int
    current_document_id: int
    section_candidate_pages: list[int]
    section_candidate_cursor: int
    section_predicted_page: int

    error_code: Optional[int]
    error_message: Optional[str]
    error_details: Optional[dict[str, Any]]


class PreReviewProgress:
    """预审终端进度展示，不写入 LangGraph State。"""

    STAGE_COUNT = 6

    def __init__(self) -> None:
        self._completed_stages: set[str] = set()
        self._page_bar: Any = None
        self._page_stage: Optional[str] = None
        self._stage_bar = tqdm(
            total=self.STAGE_COUNT,
            desc="预审总进度",
            unit="阶段",
            position=0,
        )

    def update_pages(
        self,
        stage: str,
        completed: int,
        total: int,
        *,
        status: Optional[str] = None,
    ) -> None:
        total = max(1, total)
        completed = min(max(0, completed), total)
        if self._page_stage != stage:
            self._close_page_bar()
            self._page_stage = stage
            self._page_bar = tqdm(
                total=total,
                desc=stage,
                unit="页",
                position=1,
                leave=True,
            )

        if status:
            self._page_bar.set_postfix_str(status)
        delta = completed - self._page_bar.n
        if delta > 0:
            self._page_bar.update(delta)

    def start_stage(self, stage: str) -> None:
        if stage not in self._completed_stages:
            self._stage_bar.set_postfix_str(f"正在：{stage}")
            self._stage_bar.refresh()

    def complete_stage(self, stage: str) -> None:
        if stage in self._completed_stages:
            return
        if self._page_stage == stage:
            self._close_page_bar()
        self._completed_stages.add(stage)
        self._stage_bar.set_postfix_str(stage)
        self._stage_bar.update(1)

    def close(self, *, success: bool) -> None:
        self._close_page_bar()
        self._stage_bar.set_postfix_str("完成" if success else "已停止")
        self._stage_bar.close()

    def _close_page_bar(self) -> None:
        if self._page_bar is not None:
            self._page_bar.close()
        self._page_bar = None
        self._page_stage = None


def _get_page_ocr_text(
    state: PreReviewState,
    page_index: int,
) -> str:
    ocr_results = state["document_ocr_results"]
    if not 0 <= page_index < len(ocr_results):
        raise IndexError(
            f"page_index={page_index} 没有对应的 OCR 结果"
        )

    ocr_result = ocr_results[page_index]
    if ocr_result.get("image_index") != page_index:
        raise ValueError(
            "OCR 结果顺序与页面索引不一致："
            f"期望 {page_index}，实际 {ocr_result.get('image_index')}"
        )

    page_text = ocr_result.get("document_content")
    if page_text is None:
        return "[OCR_EMPTY]"
    if not isinstance(page_text, str):
        raise TypeError(
            f"第 {page_index} 页 OCR 结果应为字符串，"
            f"实际为 {type(page_text).__name__}"
        )
    return page_text.strip() or "[OCR_EMPTY]"


def _extract_page_number(page_text: str) -> int | None:
    """Extract a tagged printed page number from one OCR page."""

    for match in _PAGE_NUMBER_PATTERN.finditer(page_text):
        attributes = match.group("attributes")
        if not re.search(
            r'\btype=["\']page-number["\']',
            attributes,
            re.IGNORECASE,
        ):
            continue
        number_match = re.search(r"\d{1,4}", match.group("content"))
        if number_match:
            return int(number_match.group(0))
    return None


def _estimate_page_number_offset(
    state: PreReviewState,
) -> int | None:
    """Estimate PDF-index minus printed-page offset by sequence consensus."""

    body_start = int(
        state.get("body_start_page_index", state.get("catalog_page_index", 0))
    )
    offsets: list[int] = []
    for page_index in range(body_start, len(state["document_ocr_results"])):
        page_number = _extract_page_number(
            _get_page_ocr_text(state, page_index)
        )
        if page_number is not None and page_number > 0:
            offsets.append(page_index - page_number)
    if not offsets:
        return None
    return Counter(offsets).most_common(1)[0][0]


def _parse_catalog_page(item: dict[str, Any]) -> int | None:
    """Read the first page number while tolerating values such as ``20/21``."""

    raw_value = item.get("catalog_page")
    if raw_value is None:
        extra_fields = item.get("extra_fields")
        if isinstance(extra_fields, dict):
            raw_value = (
                extra_fields.get("页号")
                or extra_fields.get("页码")
                or extra_fields.get("目录页号")
            )
    if raw_value is None:
        return None
    match = re.search(r"\d{1,4}", str(raw_value))
    return int(match.group(0)) if match else None


def _compact_section_ocr(page_text: str) -> str:
    if len(page_text) <= _SECTION_OCR_MAX_CHARS:
        return page_text
    head_size = _SECTION_OCR_MAX_CHARS * 3 // 4
    return (
        page_text[:head_size]
        + "\n[OCR中间内容已省略]\n"
        + page_text[-(_SECTION_OCR_MAX_CHARS - head_size):]
    )


def _section_candidate_pages(
    state: PreReviewState,
    document_id: int,
) -> tuple[list[int], int, bool]:
    catalog = state["document_catalog"]
    item = catalog[document_id - 1]
    page_count = len(state["page_image_paths"])
    body_start = int(
        state.get("body_start_page_index", state.get("catalog_page_index", 0))
    )
    previous_pages = [
        int(section["section_page"])
        for section in catalog[: document_id - 1]
        if int(section.get("section_page", -1)) >= 0
    ]
    lower_bound = max(body_start, max(previous_pages, default=body_start - 1) + 1)
    logical_page = _parse_catalog_page(item)
    offset = _estimate_page_number_offset(state) if logical_page is not None else None
    uses_catalog_page = logical_page is not None and offset is not None

    if uses_catalog_page:
        predicted_page = logical_page + offset
        candidates = [
            predicted_page + delta
            for delta in _SECTION_PROBE_OFFSETS
        ]
    else:
        predicted_page = lower_bound
        candidates = list(range(lower_bound, page_count))

    candidates = list(dict.fromkeys(
        page
        for page in candidates
        if lower_bound <= page < page_count
    ))
    if not candidates:
        candidates = [min(max(lower_bound, 0), max(page_count - 1, 0))]
    predicted_page = min(
        max(predicted_page, lower_bound),
        max(page_count - 1, 0),
    )
    return candidates, predicted_page, uses_catalog_page


def _section_result(parsed_result: Any) -> str:
    """Normalize new tri-state and legacy boolean classifier responses."""

    if isinstance(parsed_result, dict):
        result = parsed_result.get("result")
        if result in {"match", "conflict", "unknown"}:
            return result
        return "match" if parsed_result.get("is_belong") else "conflict"
    result = getattr(parsed_result, "result", None)
    if result in {"match", "conflict", "unknown"}:
        return result
    return "match" if getattr(parsed_result, "is_belong", False) else "conflict"


def _get_ocr_max_concurrency() -> int:
    return int(os.getenv("REVIEW_MODEL_MAX_CONCURRENCY", "5"))


def _get_case_facts_max_attempts() -> int:
    variable_name = "CASE_FACTS_EXTRACTION_MAX_ATTEMPTS"
    raw_value = os.getenv(variable_name, "3")
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{variable_name} 必须是正整数，实际为 {raw_value!r}"
        ) from exc
    if value < 1:
        raise ValueError(
            f"{variable_name} 必须是正整数，实际为 {value}"
        )
    return value


def _build_error(
    code: ErrorCode,
    stage: str,
    exc: Exception | str,
    state: PreReviewState | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_error_state(
        error_code=int(code),
        stage=stage,
        exc=exc,
        pdf_path=state.get("pdf_path") if state else None,
        node=stage,
        extra=extra,
    )


def _directory_not_found(state: PreReviewState) -> dict[str, Any]:
    return _build_error(
        ErrorCode.DIR_IDENTIFICATION_ERROR,
        "pre_review.directory_not_found",
        RuntimeError("未能在 PDF 中识别出目录"),
        state,
    )


def _document_section_not_found(state: PreReviewState) -> dict[str, Any]:
    document_id = state["current_document_id"]
    return _build_error(
        ErrorCode.SECTION_IDENTIFICATION_ERROR,
        "pre_review.document_section_not_found",
        RuntimeError(f"已到达 PDF 末页，尚未定位目录项 {document_id}"),
        state,
        extra={
            "page_index": state.get("catalog_page_index"),
            "document_id": document_id,
        },
    )


def _convert_pdf_to_page_images(state: PreReviewState) -> dict[str, Any]:
    pdf_document = None
    try:
        dpi = 200
        pdf_path = Path(state["pdf_path"])
        cache_paths = get_cache_paths(pdf_path)
        output_directory = cache_paths.page_directory
        output_directory.mkdir(parents=True, exist_ok=True)

        pdf_document = fitz.open(pdf_path)
        render_matrix = fitz.Matrix(dpi / 72, dpi / 72)
        page_number_width = max(
            3,
            len(str(max(0, len(pdf_document) - 1))),
        )
        page_image_paths: list[str] = []

        for page_index in tqdm(
            range(len(pdf_document)),
            desc="converting pdf to images...",
            unit="页",
            position=1,
            leave=True,
        ):
            page = pdf_document.load_page(page_index)
            pixmap = page.get_pixmap(
                matrix=render_matrix,
                colorspace=fitz.csRGB,
            )
            image_path = (
                output_directory
                / f"page_{page_index:0{page_number_width}d}.jpeg"
            )
            pixmap.save(str(image_path))
            page_image_paths.append(str(image_path))

        if not page_image_paths:
            raise ValueError("PDF 中没有可处理的页面")

        atomic_write_json(
            cache_paths.image_list,
            page_image_paths,
        )
        return {"page_image_paths": page_image_paths}
    except Exception as exc:
        return _build_error(
            ErrorCode.PDF_TO_IMAGE_ERROR,
            "pre_review.convert_pdf_to_page_images",
            exc,
            state,
        )
    finally:
        if pdf_document is not None:
            pdf_document.close()


def _extract_case_metadata(
    state: PreReviewState,
    agents: PreReviewAgents,
) -> dict[str, Any]:
    try:
        page_image_paths = state["page_image_paths"]
        if not page_image_paths:
            raise ValueError("page_image_paths 为空，无法提取案卷元信息")

        cover_image_path = page_image_paths[0]
        prompt_text = agents.build_task_prompt(
            "case_metadata_extractor"
        )
        message = build_vision_message(prompt_text, cover_image_path)
        parsed_metadata = agents.invoke_case_metadata(message)
        if isinstance(parsed_metadata, dict):
            metadata = parsed_metadata
        else:
            metadata = parsed_metadata.model_dump(by_alias=True)

        return {"case_metadata": metadata}
    except Exception as exc:
        return _build_error(
            ErrorCode.META_DATA_EXTRACTION_ERROR,
            "pre_review.extract_case_metadata",
            exc,
            state,
        )


def _save_case_metadata(
    state: PreReviewState,
) -> dict[str, Any]:
    try:
        cache_paths = get_cache_paths(state["pdf_path"])
        atomic_write_json(
            cache_paths.metadata,
            state["case_metadata"],
        )
        return {}
    except Exception as exc:
        return _build_error(
            ErrorCode.META_DATA_EXTRACTION_ERROR,
            "pre_review.save_case_metadata",
            exc,
            state,
        )


async def _extract_case_facts(
    state: PreReviewState,
    agents: PreReviewAgents,
) -> dict[str, Any]:
    try:
        # 案情只允许来自行政处罚决定书，缺失时依次使用结案审批表、立案审批表。
        # 不使用全文 OCR，避免把无关证据材料送入模型。
        preferred_names = ["行政处罚决定书", "结案审批表", "立案审批表"]
        catalog = state.get("document_catalog", [])
        selected_document = next(
            (
                item
                for name in preferred_names
                for item in catalog
                if str(item.get("section_name", "")).strip() == name
                and item.get("section_page", -1) >= 0
            ),
            None,
        )

        metadata = dict(state.get("case_metadata", {}))
        if selected_document is None:
            metadata["案情"] = None
            cache_paths = get_cache_paths(state["pdf_path"])
            atomic_write_json(cache_paths.metadata, metadata)
            return {"case_metadata": metadata}

        start_page = int(selected_document["section_page"])
        later_pages = [
            int(item["section_page"])
            for item in catalog
            if item.get("section_page", -1) > start_page
        ]
        end_page = min(later_pages, default=len(state["document_ocr_results"]))
        ocr_parts = [
            _get_page_ocr_text(state, index)
            for index in range(start_page, end_page)
        ]
        ocr_text = "\n".join(
            part for part in ocr_parts if part and part != "[OCR_EMPTY]"
        ).strip()
        if not ocr_text:
            metadata["案情"] = None
            cache_paths = get_cache_paths(state["pdf_path"])
            atomic_write_json(cache_paths.metadata, metadata)
            return {"case_metadata": metadata}

        prompt_text = agents.build_task_prompt(
            "case_facts_extractor",
        ) + (
            f"\n<case_metadata>\n{json.dumps(metadata, ensure_ascii=False)}"
            f"\n</case_metadata>\n<document_ocr>\n{ocr_text}\n</document_ocr>"
        )
        max_attempts = _get_case_facts_max_attempts()
        case_facts: str | None = None

        for attempt in range(1, max_attempts + 1):
            attempt_prompt = prompt_text
            if attempt > 1:
                attempt_prompt += (
                    "\n\n上一次未能提取出案情。请重新核对全部 OCR 原文，"
                    "只要存在有原文支持的当事人及核心行为或事实，就必须提炼为非空案情。"
                    "不得因信息不完整而遗漏可确认的事实。"
                )
            result = await agents.ainvoke_case_facts(attempt_prompt)
            facts = (
                result
                if isinstance(result, dict)
                else result.model_dump(by_alias=True)
            )
            extracted_facts = facts.get("案情")
            if extracted_facts is not None and not isinstance(extracted_facts, str):
                raise TypeError("案情必须是字符串或 null")
            if isinstance(extracted_facts, str) and extracted_facts.strip():
                case_facts = extracted_facts.strip()
                trace_event(
                    "case_facts_extraction_succeeded",
                    attempt=attempt,
                    max_attempts=max_attempts,
                )
                break

            trace_event(
                "case_facts_extraction_retry",
                attempt=attempt,
                max_attempts=max_attempts,
            )

        if case_facts is None:
            # 文书存在但没有可确认案情时，保留空值并继续后续预审。
            metadata["案情"] = None
            cache_paths = get_cache_paths(state["pdf_path"])
            atomic_write_json(cache_paths.metadata, metadata)
            return {"case_metadata": metadata}

        metadata["案情"] = case_facts
        cache_paths = get_cache_paths(state["pdf_path"])
        atomic_write_json(cache_paths.metadata, metadata)
        return {"case_metadata": metadata}
    except Exception as exc:
        return _build_error(
            ErrorCode.META_DATA_EXTRACTION_ERROR,
            "pre_review.extract_case_facts",
            exc,
            state,
        )


async def _extract_document_text(
    state: PreReviewState,
    agents: PreReviewAgents,
) -> dict[str, Any]:
    try:
        page_image_paths = state["page_image_paths"]
        start_page_index = state["ocr_page_index"]

        if start_page_index >= len(page_image_paths):
            raise IndexError("已到达图片列表末尾，无法继续进行 OCR")

        max_concurrency = _get_ocr_max_concurrency()
        end_page_index = min(
            len(page_image_paths),
            start_page_index + max_concurrency,
        )

        meta_info = json.dumps(
            state["case_metadata"],
            ensure_ascii=False,
        )
        prompt_text = agents.build_task_prompt(
            "document_ocr",
            meta_info=meta_info,
        )
        fallback_prompt_text = (
            f"{prompt_text}\n\n"
            "上一轮 OCR 结果疑似只识别了印章、日期或页码，遗漏了正文。"
            "请重新逐行核对当前原图：必须优先完整识别页面标题、文书名称、"
            "文号和正文；即使存在表格线、脱敏遮挡、公章、勾选框或空白字段，"
            "也不得只输出局部印章或日期。输出前确认：若图片中有清晰标题，"
            "识别结果必须包含该标题。"
        )

        async def extract_one_page(
            page_index: int,
        ) -> DocumentOcrResult:
            image_path = page_image_paths[page_index]

            async def recognize(prompt: str) -> str:
                message = build_vision_message(prompt, image_path)
                response = await agents.document_ocr_agent.ainvoke(
                    {"messages": [message]}
                )
                messages = response.get("messages", [])
                if not messages:
                    raise RuntimeError("OCR Agent 返回结果缺少 messages")
                content = messages[-1].content
                if not isinstance(content, str):
                    raise TypeError(
                        "OCR Agent 返回内容应为字符串，"
                        f"实际为 {type(content).__name__}"
                    )
                return content

            page_token = CURRENT_PAGE_INDEX.set(page_index)
            try:
                content = await recognize(prompt_text)
                if _needs_ocr_fallback(content):
                    primary_score = _ocr_substantive_character_count(content)
                    trace_event(
                        "document_ocr_fallback_start",
                        page_index=page_index,
                        primary_substantive_character_count=primary_score,
                    )
                    retry_content = await recognize(fallback_prompt_text)
                    content = _prefer_retry_ocr(content, retry_content)
                    trace_event(
                        "document_ocr_fallback_end",
                        page_index=page_index,
                        primary_substantive_character_count=primary_score,
                        retry_substantive_character_count=(
                            _ocr_substantive_character_count(retry_content)
                        ),
                        retry_selected=content == retry_content,
                    )
                return {
                    "document_content": content,
                    "image_path": image_path,
                    "image_index": page_index,
                }
            except Exception as exc:
                raise RuntimeError(
                    f"第 {page_index} 页 OCR 失败: {exc}"
                ) from exc
            finally:
                CURRENT_PAGE_INDEX.reset(page_token)

        batch_results = await asyncio.gather(
            *(
                extract_one_page(page_index)
                for page_index in range(
                    start_page_index,
                    end_page_index,
                )
            )
        )
        return {
            "ocr_page_index": end_page_index,
            "ocr_max_concurrency": max_concurrency,
            "document_ocr_results": [
                *state["document_ocr_results"],
                *batch_results,
            ],
        }
    except Exception as exc:
        return _build_error(
            ErrorCode.UNEXPECTED_ERROR,
            "pre_review.extract_document_text",
            exc,
            state,
            extra={
                "start_page_index": state.get("ocr_page_index"),
                "max_concurrency": _get_ocr_max_concurrency(),
            },
        )


def _save_document_ocr_results(
    state: PreReviewState,
) -> dict[str, Any]:
    try:
        cache_paths = get_cache_paths(state["pdf_path"])
        atomic_write_json(
            cache_paths.ocr_results,
            state["document_ocr_results"],
        )
        return {}
    except Exception as exc:
        return _build_error(
            ErrorCode.UNEXPECTED_ERROR,
            "pre_review.save_document_ocr_results",
            exc,
            state,
        )


async def _classify_directory_page(
    state: PreReviewState,
    agents: PreReviewAgents,
) -> dict[str, Any]:
    try:
        page_index = state["catalog_page_index"]
        if page_index >= len(state["page_image_paths"]):
            raise IndexError("已到达 PDF 末页，但尚未完成目录识别")

        page_text = _get_page_ocr_text(state, page_index)
        prompt_text = agents.build_task_prompt(
            "directory_classifier",
            page_index=page_index,
            page_text=page_text,
        )

        start_time = time.time()
        parsed_result = await agents.ainvoke_directory_classifier(
            prompt_text=prompt_text,
            page_index=page_index,
        )
        print(f"目录识别响应时间: {time.time() - start_time:.4f} 秒")

        directory_items: list[dict[str, Any]] = []
        if isinstance(parsed_result, dict):
            is_directory = parsed_result["is_dir"]
            for item in parsed_result["dir_info"]:
                item_data = dict(item)
                extra_fields = item_data.pop("extra_fields", {})
                if isinstance(extra_fields, dict):
                    item_data.update(extra_fields)
                directory_items.append(item_data)
        else:
            is_directory = parsed_result.is_directory
            for item in parsed_result.directory_items:
                item_data = item.model_dump(exclude={"extra_fields"})
                item_data.update(item.extra_fields)
                directory_items.append(item_data)

        directory_items = normalize_directory_info(directory_items)

        existing_catalog = state["document_catalog"]
        if not is_directory and existing_catalog:
            return {
                "catalog_scan_completed": True,
                "body_start_page_index": page_index,
            }

        return {
            "document_catalog": existing_catalog + directory_items,
            "catalog_page_index": page_index + 1,
        }
    except Exception as exc:
        return _build_error(
            ErrorCode.DIR_IDENTIFICATION_ERROR,
            "pre_review.classify_directory_page",
            exc,
            state,
            extra={"page_index": state.get("catalog_page_index")},
        )


async def _locate_document_sections(
    state: PreReviewState,
    agents: PreReviewAgents,
) -> dict[str, Any]:
    try:
        document_id = state["current_document_id"]
        document_catalog = state["document_catalog"]

        if not document_catalog:
            raise ValueError("目录信息为空，无法定位文书章节")
        if document_id > len(document_catalog):
            return {}

        section_context = document_catalog[document_id - 1]
        candidate_pages = list(state.get("section_candidate_pages", []))
        candidate_cursor = int(state.get("section_candidate_cursor", 0))
        predicted_page = state.get("section_predicted_page")
        if not candidate_pages or not 0 <= candidate_cursor < len(candidate_pages):
            (
                candidate_pages,
                predicted_page,
                uses_catalog_page,
            ) = _section_candidate_pages(state, document_id)
            candidate_cursor = 0
        else:
            logical_page = _parse_catalog_page(section_context)
            uses_catalog_page = (
                logical_page is not None
                and _estimate_page_number_offset(state) is not None
            )
        page_index = candidate_pages[candidate_cursor]

        previous_section_name = (
            str(document_catalog[document_id - 2].get("section_name", ""))
            if document_id > 1
            else "（无）"
        )
        next_section_name = (
            str(document_catalog[document_id].get("section_name", ""))
            if document_id < len(document_catalog)
            else "（无）"
        )
        page_text = _compact_section_ocr(
            _get_page_ocr_text(state, page_index)
        )
        prompt_text = agents.build_task_prompt(
            "section_classifier",
            previous_section_name=previous_section_name,
            section_name=str(section_context.get("section_name", "")),
            next_section_name=next_section_name,
            page_index=page_index,
            page_text=page_text,
        )
        parsed_result = await agents.ainvoke_section_classifier(
            prompt_text=prompt_text,
            page_index=page_index,
        )
        classification = _section_result(parsed_result)

        updated_catalog = copy.deepcopy(document_catalog)
        should_accept = classification == "match" or (
            classification == "unknown"
            and uses_catalog_page
            and page_index == predicted_page
        )
        if should_accept:
            accepted_page = (
                page_index if classification == "match" else int(predicted_page)
            )
            target_document = updated_catalog[document_id - 1]
            target_document["section_page"] = accepted_page
            if classification == "match" and accepted_page == predicted_page:
                target_document["location_source"] = "page_and_semantic"
                target_document["location_confidence"] = 0.95
            elif classification == "match":
                target_document["location_source"] = "nearby_semantic"
                target_document["location_confidence"] = 0.85
            else:
                target_document["location_source"] = "catalog_page"
                target_document["location_confidence"] = 0.65
            target_document["needs_review"] = False
            return {
                "document_catalog": updated_catalog,
                "current_document_id": document_id + 1,
                "catalog_page_index": accepted_page + 1,
                "section_candidate_pages": [],
                "section_candidate_cursor": 0,
                "section_predicted_page": accepted_page,
            }

        next_cursor = candidate_cursor + 1
        if next_cursor < len(candidate_pages):
            return {
                "section_candidate_pages": candidate_pages,
                "section_candidate_cursor": next_cursor,
                "section_predicted_page": int(predicted_page),
                "catalog_page_index": candidate_pages[next_cursor],
            }

        fallback_page = int(predicted_page)
        target_document = updated_catalog[document_id - 1]
        target_document["section_page"] = fallback_page
        target_document["location_source"] = "location_fallback"
        target_document["location_confidence"] = 0.3
        target_document["needs_review"] = True
        return {
            "document_catalog": updated_catalog,
            "current_document_id": document_id + 1,
            "catalog_page_index": fallback_page + 1,
            "section_candidate_pages": [],
            "section_candidate_cursor": 0,
            "section_predicted_page": fallback_page,
        }
    except Exception as exc:
        return _build_error(
            ErrorCode.SECTION_IDENTIFICATION_ERROR,
            "pre_review.locate_document_sections",
            exc,
            state,
            extra={
                "page_index": state.get("catalog_page_index"),
                "document_id": state.get("current_document_id"),
            },
        )


def _save_pre_review_results(state: PreReviewState) -> dict[str, Any]:
    try:
        cache_paths = get_cache_paths(state["pdf_path"])
        atomic_write_json(
            cache_paths.directory,
            normalize_directory_info(state["document_catalog"]),
        )
        return {}
    except Exception as exc:
        return _build_error(
            ErrorCode.UNEXPECTED_ERROR,
            "pre_review.save_pre_review_results",
            exc,
            state,
        )


def _route_to(success_node: str):
    def router(state: PreReviewState):
        if state.get("error_code") is not None:
            return "handle_error"
        return success_node

    return router


def _route_document_ocr(state: PreReviewState):
    if state.get("error_code") is not None:
        return "handle_error"
    if state["ocr_page_index"] < len(state["page_image_paths"]):
        return "extract_document_text"
    return "save_document_ocr_results"


def _route_directory_scan(state: PreReviewState):
    if state.get("error_code") is not None:
        return "handle_error"
    if state.get("catalog_scan_completed", False):
        return "locate_document_sections"
    if state["catalog_page_index"] < len(state["page_image_paths"]):
        return "classify_directory_page"
    return "directory_not_found"


def _route_section_location(state: PreReviewState):
    if state.get("error_code") is not None:
        return "handle_error"
    if state["current_document_id"] > len(state["document_catalog"]):
        return "save_pre_review_results"
    return "locate_document_sections"


def _handle_error(state: PreReviewState) -> dict[str, Any]:
    error_code = state.get("error_code", ErrorCode.UNEXPECTED_ERROR)
    print(f"PRE_REVIEW STOPPED, ERROR CODE: {error_code}")
    print(state.get("error_message", ""))
    return {}


def build_pre_review_graph(
    agents: PreReviewAgents | None = None,
    progress: PreReviewProgress | None = None,
):
    agents = agents or PreReviewAgents()

    def convert_pdf_to_page_images_node(
        state: PreReviewState,
    ) -> dict[str, Any]:
        if progress:
            progress.start_stage("PDF 转图片")
        result = _convert_pdf_to_page_images(state)
        if progress and result.get("error_code") is None:
            progress.complete_stage("PDF 转图片")
        return result

    def extract_case_metadata_node(
        state: PreReviewState,
    ) -> dict[str, Any]:
        if progress:
            progress.start_stage("案卷元数据")
        result = _extract_case_metadata(state, agents)
        return result

    def save_case_metadata_node(
        state: PreReviewState,
    ) -> dict[str, Any]:
        if progress:
            progress.start_stage("案卷元数据")
        result = _save_case_metadata(state)
        if progress and result.get("error_code") is None:
            progress.complete_stage("案卷元数据")
        return result

    async def extract_document_text_node(
        state: PreReviewState,
    ) -> dict[str, Any]:
        if progress:
            progress.start_stage("全文 OCR")
            progress.update_pages(
                "全文 OCR",
                state["ocr_page_index"],
                len(state["page_image_paths"]),
                status=(
                    f"从第 {state['ocr_page_index'] + 1} 页开始并发处理；"
                    "并发上限 "
                    f"{_get_ocr_max_concurrency()}"
                ),
            )
        result = await _extract_document_text(state, agents)
        if progress and result.get("error_code") is None:
            progress.update_pages(
                "全文 OCR",
                result["ocr_page_index"],
                len(state["page_image_paths"]),
                status=(
                    f"已完成 {result['ocr_page_index']} 页；"
                    f"并发上限 {result['ocr_max_concurrency']}"
                ),
            )
        return result

    def save_document_ocr_results_node(
        state: PreReviewState,
    ) -> dict[str, Any]:
        if progress:
            progress.start_stage("全文 OCR")
        result = _save_document_ocr_results(state)
        if progress and result.get("error_code") is None:
            progress.complete_stage("全文 OCR")
        return result

    async def classify_directory_page_node(
        state: PreReviewState,
    ) -> dict[str, Any]:
        if progress:
            progress.start_stage("目录识别")
            progress.update_pages(
                "目录识别",
                state["catalog_page_index"],
                len(state["page_image_paths"]),
                status=f"正在扫描第 {state['catalog_page_index'] + 1} 页",
            )
        result = await _classify_directory_page(state, agents)
        if progress and result.get("error_code") is None:
            catalog_size = len(
                result.get("document_catalog", state["document_catalog"])
            )
            progress.update_pages(
                "目录识别",
                state["catalog_page_index"] + 1,
                len(state["page_image_paths"]),
                status=f"已提取 {catalog_size} 项",
            )
            if result.get("catalog_scan_completed", False):
                progress.complete_stage("目录识别")
        return result

    async def extract_case_facts_node(
        state: PreReviewState,
    ) -> dict[str, Any]:
        return await _extract_case_facts(state, agents)

    async def locate_document_sections_node(
        state: PreReviewState,
    ) -> dict[str, Any]:
        if progress:
            progress.start_stage("章节定位")
            progress.update_pages(
                "章节定位",
                state["catalog_page_index"],
                len(state["page_image_paths"]),
                status=(
                    f"正在定位第 {state['current_document_id']}/"
                    f"{len(state['document_catalog'])} 项"
                ),
            )
        result = await _locate_document_sections(state, agents)
        if progress and result.get("error_code") is None:
            document_count = len(state["document_catalog"])
            next_document_id = result.get(
                "current_document_id",
                state["current_document_id"],
            )
            located_count = min(
                document_count,
                next_document_id - 1,
            )
            progress.update_pages(
                "章节定位",
                result.get(
                    "catalog_page_index",
                    state["catalog_page_index"],
                ),
                len(state["page_image_paths"]),
                status=f"已定位 {located_count}/{document_count} 项",
            )
            if next_document_id > document_count:
                progress.complete_stage("章节定位")
        return result

    def save_pre_review_results_node(
        state: PreReviewState,
    ) -> dict[str, Any]:
        if progress:
            progress.start_stage("保存预审结果")
        result = _save_pre_review_results(state)
        if progress and result.get("error_code") is None:
            progress.complete_stage("保存预审结果")
        return result

    workflow = StateGraph(PreReviewState)

    workflow.add_node(
        "convert_pdf_to_page_images",
        convert_pdf_to_page_images_node,
    )
    workflow.add_node(
        "extract_case_metadata",
        extract_case_metadata_node,
    )
    workflow.add_node(
        "save_case_metadata",
        save_case_metadata_node,
    )
    workflow.add_node(
        "extract_document_text",
        extract_document_text_node,
    )
    workflow.add_node(
        "save_document_ocr_results",
        save_document_ocr_results_node,
    )
    workflow.add_node("extract_case_facts", extract_case_facts_node)
    workflow.add_node(
        "classify_directory_page",
        classify_directory_page_node,
    )
    workflow.add_node(
        "locate_document_sections",
        locate_document_sections_node,
    )
    workflow.add_node(
        "save_pre_review_results",
        save_pre_review_results_node,
    )
    workflow.add_node("directory_not_found", _directory_not_found)
    workflow.add_node(
        "document_section_not_found",
        _document_section_not_found,
    )
    workflow.add_node("handle_error", _handle_error)

    workflow.set_entry_point("convert_pdf_to_page_images")
    workflow.add_conditional_edges(
        "convert_pdf_to_page_images",
        _route_to("extract_case_metadata"),
    )
    workflow.add_conditional_edges(
        "extract_case_metadata",
        _route_to("save_case_metadata"),
    )
    workflow.add_conditional_edges(
        "save_case_metadata",
        _route_to("extract_document_text"),
    )
    workflow.add_conditional_edges(
        "extract_document_text",
        _route_document_ocr,
    )
    workflow.add_conditional_edges(
        "save_document_ocr_results",
        _route_to("classify_directory_page"),
    )
    workflow.add_conditional_edges(
        "classify_directory_page",
        _route_directory_scan,
    )
    workflow.add_conditional_edges(
        "locate_document_sections",
        lambda state: (
            "extract_case_facts"
            if state.get("error_code") is None
            and state["current_document_id"] > len(state["document_catalog"])
            else _route_section_location(state)
        ),
    )
    workflow.add_conditional_edges(
        "extract_case_facts",
        _route_to("save_pre_review_results"),
    )
    workflow.add_conditional_edges(
        "save_pre_review_results",
        _route_to(END),
    )
    workflow.add_edge("directory_not_found", "handle_error")
    workflow.add_edge("document_section_not_found", "handle_error")
    workflow.add_edge("handle_error", END)

    return workflow.compile()


async def run_pre_review(
    file_path: str,
    agents: PreReviewAgents | None = None,
) -> PreReviewState:
    initial_state: PreReviewState = {
        "pdf_path": str(file_path),
        "page_image_paths": [],
        "case_metadata": {},
        "ocr_page_index": 0,
        "document_ocr_results": [],
        "document_catalog": [],
        "catalog_page_index": 0,
        "catalog_scan_completed": False,
        "body_start_page_index": 0,
        "current_document_id": 1,
        "section_candidate_pages": [],
        "section_candidate_cursor": 0,
        "section_predicted_page": 0,
    }

    progress = PreReviewProgress()
    final_state: PreReviewState | None = None
    pdf_path_token = CURRENT_PDF_PATH.set(str(file_path))
    try:
        final_state = await build_pre_review_graph(
            agents,
            progress,
        ).ainvoke(initial_state)
    finally:
        CURRENT_PDF_PATH.reset(pdf_path_token)
        progress.close(
            success=(
                final_state is not None
                and final_state.get("error_code") is None
            )
        )

    if final_state is None:
        raise RuntimeError("预处理图未返回最终状态")

    if final_state.get("error_code") is not None:
        raise PreReviewError(
            final_state["error_code"],
            final_state.get("error_message", "预处理失败"),
            details=final_state.get("error_details"),
        )

    return final_state
