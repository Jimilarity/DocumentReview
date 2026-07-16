import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from constants import (
    CACHE_ROOT,
    PROJECT_ROOT,
    RAW_REVIEW_RESULT_FILENAME,
    RESULT_ROOT,
    REVIEW_RESULT_PROCESSING_FILENAME,
    STRUCTURED_FIELD_CACHE_FILENAME,
)


@dataclass(frozen=True)
class CachePaths:
    """一个 PDF 在当前缓存目录结构下对应的全部路径。"""

    document_key: str
    cache_directory: Path
    page_directory: Path
    image_list: Path
    ocr_results: Path
    metadata: Path
    directory: Path
    raw_review_results: Path
    review_result_processing: Path
    structured_fields: Path

    @property
    def pre_review_files(self) -> dict[str, Path]:
        """判断预审是否可跳过所需的缓存文件。"""

        return {
            "image_list": self.image_list,
            "ocr_results": self.ocr_results,
            "directory": self.directory,
            "metadata": self.metadata,
        }


def build_document_key(pdf_path: str | Path | None) -> str:
    """生成安全、稳定且能区分同名路径的文档目录名。"""

    if not pdf_path:
        raise ValueError("pdf_path 不能为空")

    raw_path = str(pdf_path)
    path = Path(raw_path)
    document_name = path.stem.strip() or "document"
    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", document_name)
    safe_name = safe_name.strip(" ._")[:80] or "document"
    try:
        normalized_path = str(path.resolve(strict=False)).casefold()
    except (OSError, RuntimeError):
        normalized_path = raw_path.casefold()

    path_digest = hashlib.sha256(
        normalized_path.encode("utf-8", errors="replace")
    ).hexdigest()[:12]
    return f"{safe_name}-{path_digest}"


def _resolve_cache_root(cache_root: str | Path | None) -> Path:
    if cache_root is None:
        return CACHE_ROOT

    configured_root = Path(cache_root)
    if configured_root.is_absolute():
        return configured_root
    return PROJECT_ROOT / configured_root


def get_cache_paths(
    pdf_path: str | Path,
    *,
    cache_root: str | Path | None = None,
) -> CachePaths:
    """返回 PDF 的统一缓存路径；目录在首次写入时按需创建。"""

    document_key = build_document_key(pdf_path)
    cache_directory = _resolve_cache_root(cache_root) / document_key
    page_directory = cache_directory / "pages"

    return CachePaths(
        document_key=document_key,
        cache_directory=cache_directory,
        page_directory=page_directory,
        image_list=cache_directory / "image_list.json",
        ocr_results=cache_directory / "ocr_results.json",
        metadata=cache_directory / "meta_info.json",
        directory=cache_directory / "dir_info.json",
        raw_review_results=(
            cache_directory / RAW_REVIEW_RESULT_FILENAME
        ),
        review_result_processing=(
            cache_directory / REVIEW_RESULT_PROCESSING_FILENAME
        ),
        structured_fields=(
            cache_directory / STRUCTURED_FIELD_CACHE_FILENAME
        ),
    )


def get_result_directory(
    pdf_path: str | Path,
    *,
    result_root: str | Path | None = None,
) -> Path:
    """Return the collision-safe result directory for a PDF."""

    root = RESULT_ROOT if result_root is None else Path(result_root)
    return root / build_document_key(pdf_path)
