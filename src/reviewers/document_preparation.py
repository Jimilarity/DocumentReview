import asyncio
import json
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
    load_compatible_document_type_groups,
    normalize_document_section_map,
)


class DocumentReviewPreparationService:
    """为同一次案件审查统一生成且只生成一份文书 section 映射。"""

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
        cached_mapping = cache.document_section_map
        if cached_mapping and all(
            cached_mapping.get(name) for name in document_names
        ):
            return normalize_document_section_map(
                {
                    name: cached_mapping[name]
                    for name in document_names
                },
                dir_info,
            )
        if cached_mapping:
            raise RuntimeError(
                "结构化缓存的文书映射范围与本次审查不一致，"
                "必须重建整份缓存快照"
            )

        agents = DocumentMappingAgents()
        prompt = agents.build_task_prompt(
            "document_section_mapping",
            meta_info=json.dumps(meta_info, ensure_ascii=False, indent=2),
            document_names=json.dumps(
                document_names,
                ensure_ascii=False,
                indent=2,
            ),
            dir_info=json.dumps(dir_info, ensure_ascii=False, indent=2),
            compatible_document_type_groups=json.dumps(
                load_compatible_document_type_groups(),
                ensure_ascii=False,
                indent=2,
            ),
        )
        result = await agents.ainvoke_document_section_mapper(
            prompt,
            self.settings.agent_recursion_limit,
        )
        mapping = {
            item.document_name: item.section_ids
            for item in result.mappings
        }
        if len(result.mappings) != len(document_names) or set(mapping) != set(
            document_names
        ):
            raise ValueError("文书章节映射没有覆盖全部已确认存在的文书")

        normalized = normalize_document_section_map(mapping, dir_info)
        cache.initialize_document_section_map(normalized)
        cache.save()
        return normalized
