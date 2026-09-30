import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List

from agents import DocumentMappingAgents
from cache_paths import get_cache_paths
from constants import STRUCTURED_FIELD_CACHE_SCHEMA_VERSION
from directory_info import normalize_directory_info
from structured_field_cache import (
    StructuredFieldCache,
    build_structured_source_fingerprint,
)
from utils import async_read_json

from .base import ReviewSettings
from .document_mapping import (
    deterministic_document_section_map,
    directory_info_for_mapping,
    load_compatible_document_type_groups,
    normalize_document_section_map_lenient,
)


LOGGER = logging.getLogger(__name__)


class DocumentReviewPreparationService:
    """为同一次案件审查统一生成且只生成一份文书 section 映射。"""

    MAPPING_MAX_ATTEMPTS = 3

    def __init__(
        self,
        file_path: str | Path,
        document_names: Iterable[str],
        *,
        settings: ReviewSettings | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self._document_names = list(dict.fromkeys(document_names))
        self.settings = settings or ReviewSettings.from_env()
        self.cache_paths = get_cache_paths(self.file_path)

    def document_names(self) -> List[str]:
        return list(self._document_names)

    async def prepare(self) -> Dict[str, List[int]]:
        document_names = self.document_names()
        if not document_names:
            return {}

        meta_info, raw_dir_info, ocr_results = await asyncio.gather(
            async_read_json(self.cache_paths.metadata),
            async_read_json(self.cache_paths.directory),
            async_read_json(self.cache_paths.ocr_results),
        )
        dir_info = normalize_directory_info(raw_dir_info)
        cache = StructuredFieldCache.load(
            self.cache_paths.structured_fields,
            schema_version=STRUCTURED_FIELD_CACHE_SCHEMA_VERSION,
            source_fingerprint=build_structured_source_fingerprint(
                meta_info,
                dir_info,
                ocr_results,
            ),
        )
        cached_mapping = cache.payload.get("document_section_map", {})
        if cached_mapping:
            try:
                normalized, dropped = normalize_document_section_map_lenient(
                    {
                        name: cached_mapping[name]
                        for name in document_names
                        if name in cached_mapping
                    },
                    dir_info,
                    preferred_document_types=document_names,
                )
                self._log_dropped_mappings("cached", dropped)
                return normalized
            except Exception as exc:
                LOGGER.warning(
                    "cached document mapping ignored. file=%s error=%s: %s",
                    self.file_path,
                    type(exc).__name__,
                    exc,
                )
                return {}

        try:
            deterministic_mapping = deterministic_document_section_map(
                document_names,
                dir_info,
            )
        except Exception as exc:
            LOGGER.warning(
                "deterministic document mapping failed. file=%s error=%s: %s",
                self.file_path,
                type(exc).__name__,
                exc,
            )
            deterministic_mapping = {}
        unresolved_names = [
            name for name in document_names if name not in deterministic_mapping
        ]
        best_mapping, dropped = normalize_document_section_map_lenient(
            deterministic_mapping,
            dir_info,
            preferred_document_types=deterministic_mapping,
        )
        self._log_dropped_mappings("deterministic", dropped)
        if not unresolved_names:
            self._persist_mapping(cache, best_mapping)
            return best_mapping

        try:
            agents = DocumentMappingAgents()
            base_prompt = agents.build_task_prompt(
                "document_section_mapping",
                meta_info=json.dumps(meta_info, ensure_ascii=False, indent=2),
                document_names=json.dumps(
                    unresolved_names,
                    ensure_ascii=False,
                    indent=2,
                ),
                dir_info=json.dumps(
                    directory_info_for_mapping(dir_info),
                    ensure_ascii=False,
                    indent=2,
                ),
                compatible_document_type_groups=json.dumps(
                    load_compatible_document_type_groups(),
                    ensure_ascii=False,
                    indent=2,
                ),
                deterministic_mapping=json.dumps(
                    deterministic_mapping,
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        except Exception as exc:
            LOGGER.warning(
                "document mapping model setup failed; using deterministic "
                "mapping. file=%s error=%s: %s",
                self.file_path,
                type(exc).__name__,
                exc,
            )
            self._persist_mapping(cache, best_mapping)
            return best_mapping

        validation_error: Exception | None = None
        for attempt in range(1, self.MAPPING_MAX_ATTEMPTS + 1):
            prompt = base_prompt
            if validation_error is not None:
                prompt += (
                    "\n\n<previous_mapping_error>\n"
                    f"{validation_error}\n"
                    "</previous_mapping_error>\n"
                    "上一次完整映射未通过程序校验。请根据目录标题逐项重新判断并"
                    "返回完整 mappings；目录标题明确为送达回证/送达回执时才能映射"
                    "为送达回证，其他文书不得因与回证相邻而映射为送达回证。"
                    "除允许相容的类型组外，同一个 section_id 只能属于一个文书类型。"
                )
            try:
                result = await agents.ainvoke_document_section_mapper(
                    prompt,
                    self.settings.agent_recursion_limit,
                )
                model_mapping = {
                    item.document_name: item.section_ids
                    for item in result.mappings
                }
                mapping = {**deterministic_mapping, **model_mapping}
                candidate, dropped = normalize_document_section_map_lenient(
                    mapping,
                    dir_info,
                    preferred_document_types=deterministic_mapping,
                )
                self._log_dropped_mappings("model", dropped)
                if len(candidate) > len(best_mapping):
                    best_mapping = candidate
                missing_names = [
                    name for name in document_names if name not in candidate
                ]
                if not missing_names:
                    best_mapping = candidate
                    break
                raise ValueError(
                    "文书章节映射未覆盖: " + ", ".join(missing_names)
                )
            except Exception as exc:
                validation_error = exc
                if attempt == self.MAPPING_MAX_ATTEMPTS:
                    LOGGER.warning(
                        "document mapping degraded after %s attempts. file=%s "
                        "error=%s: %s",
                        attempt,
                        self.file_path,
                        type(exc).__name__,
                        exc,
                    )

        self._persist_mapping(cache, best_mapping)
        return best_mapping

    def _persist_mapping(
        self,
        cache: StructuredFieldCache,
        mapping: Dict[str, List[int]],
    ) -> None:
        if not mapping:
            return
        try:
            cache.initialize_document_section_map(mapping)
            cache.save()
        except Exception as exc:
            LOGGER.warning(
                "document mapping cache write skipped. file=%s error=%s: %s",
                self.file_path,
                type(exc).__name__,
                exc,
            )

    def _log_dropped_mappings(
        self,
        source: str,
        dropped: Dict[str, str],
    ) -> None:
        for document_type, reason in dropped.items():
            LOGGER.warning(
                "document mapping skipped. file=%s source=%s "
                "document_type=%s reason=%s",
                self.file_path,
                source,
                document_type,
                reason,
            )
