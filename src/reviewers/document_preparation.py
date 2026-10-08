import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List

from agents import DocumentMappingAgents
from cache_paths import get_cache_paths
from constants import RULES_PATH, STRUCTURED_FIELD_CACHE_SCHEMA_VERSION
from directory_info import normalize_directory_info
from structured_field_cache import (
    StructuredFieldCache,
    build_structured_source_fingerprint,
)
from utils import async_read_json

from .base import ReviewSettings
from .document_mapping import (
    deterministic_document_section_map,
    document_type_definitions,
    directory_info_for_mapping,
    load_aggregate_document_types,
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
        rules_path: str | Path = RULES_PATH,
    ) -> None:
        self.file_path = Path(file_path)
        self._document_names = list(dict.fromkeys(document_names))
        self.settings = settings or ReviewSettings.from_env()
        self.rules_path = Path(rules_path)
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
                if all(
                    bool(cached_mapping.get(name))
                    for name in document_names
                ):
                    normalized, dropped = normalize_document_section_map_lenient(
                        {
                            name: cached_mapping[name]
                            for name in document_names
                        },
                        dir_info,
                        preferred_document_types=document_names,
                    )
                    self._log_dropped_mappings("cached", dropped)
                    if all(name in normalized for name in document_names):
                        return normalized
            except Exception as exc:
                LOGGER.warning(
                    "cached document mapping will be rebuilt. file=%s "
                    "error=%s: %s",
                    self.file_path,
                    type(exc).__name__,
                    exc,
                )
            cache = StructuredFieldCache(
                cache.path,
                schema_version=cache.schema_version,
                source_fingerprint=cache.source_fingerprint,
            )

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
        best_mapping, dropped = normalize_document_section_map_lenient(
            deterministic_mapping,
            dir_info,
            preferred_document_types=deterministic_mapping,
        )
        self._log_dropped_mappings("deterministic", dropped)

        try:
            agents = DocumentMappingAgents()
            base_prompt = agents.build_task_prompt(
                "document_section_mapping",
                meta_info=json.dumps(meta_info, ensure_ascii=False, indent=2),
                document_names=json.dumps(
                    document_names,
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
                deterministic_mapping=json.dumps(
                    deterministic_mapping,
                    ensure_ascii=False,
                    indent=2,
                ),
                document_definitions=json.dumps(
                    document_type_definitions(
                        document_names,
                        self.rules_path,
                    ),
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
        last_model_dropped: Dict[str, str] = {}
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
                raw_model_mapping = {
                    item.document_name: item.section_ids
                    for item in result.mappings
                }
                unexpected_names = sorted(
                    set(raw_model_mapping) - set(document_names)
                )
                if unexpected_names:
                    LOGGER.warning(
                        "document mapping ignored unrequested types. file=%s "
                        "document_types=%s",
                        self.file_path,
                        unexpected_names,
                    )
                # 确定性映射是已确认的最小集合，而不是完整集合。模型可以按
                # 规则语义为同一类别补充任意地方名称、旧称或无目录 OCR 分段，
                # 但不能删除确定性结果。
                model_mapping = {
                    name: section_ids
                    for name, section_ids in raw_model_mapping.items()
                    if name in document_names
                }
                mapping = {
                    name: list(
                        dict.fromkeys(
                            [
                                *deterministic_mapping.get(name, []),
                                *model_mapping.get(name, []),
                            ]
                        )
                    )
                    for name in document_names
                    if deterministic_mapping.get(name)
                    or model_mapping.get(name)
                }
                candidate, dropped = normalize_document_section_map_lenient(
                    mapping,
                    dir_info,
                    preferred_document_types=deterministic_mapping,
                )
                last_model_dropped = dropped
                candidate_quality = (
                    len(candidate),
                    sum(len(section_ids) for section_ids in candidate.values()),
                )
                best_quality = (
                    len(best_mapping),
                    sum(len(section_ids) for section_ids in best_mapping.values()),
                )
                if candidate_quality > best_quality:
                    best_mapping = candidate
                missing_names = [
                    name for name in document_names if name not in candidate
                ]
                if not missing_names:
                    try:
                        candidate = await self._verify_aggregate_mappings(
                            agents=agents,
                            meta_info=meta_info,
                            dir_info=dir_info,
                            ocr_results=ocr_results,
                            candidate=candidate,
                            deterministic_mapping=deterministic_mapping,
                        )
                    except Exception as exc:
                        LOGGER.warning(
                            "aggregate document mapping verification failed; "
                            "using primary mapping. file=%s error=%s: %s",
                            self.file_path,
                            type(exc).__name__,
                            exc,
                        )
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

        if last_model_dropped:
            self._log_dropped_mappings("model", last_model_dropped)

        self._persist_mapping(cache, best_mapping)
        return best_mapping

    async def _verify_aggregate_mappings(
        self,
        *,
        agents: DocumentMappingAgents,
        meta_info: Dict[str, object],
        dir_info: List[dict],
        ocr_results: List[dict],
        candidate: Dict[str, List[int]],
        deterministic_mapping: Dict[str, List[int]],
    ) -> Dict[str, List[int]]:
        """用规则语义复核聚合类别，避免以材料标题字样代替类别判断。"""

        aggregate_types = set(load_aggregate_document_types())
        aggregate_names = [
            name for name in candidate if name in aggregate_types
        ]
        if not aggregate_names:
            return candidate

        candidate_ids = {
            section_id
            for name in aggregate_names
            for section_id in candidate[name]
        }
        focused_directory = [
            item
            for item in directory_info_for_mapping(dir_info, ocr_results)
            if int(item["section_id"]) in candidate_ids
        ]
        prompt = agents.build_task_prompt(
            "document_section_mapping_verification",
            meta_info=json.dumps(meta_info, ensure_ascii=False, indent=2),
            document_definitions=json.dumps(
                document_type_definitions(
                    aggregate_names,
                    self.rules_path,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            candidate_mapping=json.dumps(
                {name: candidate[name] for name in aggregate_names},
                ensure_ascii=False,
                indent=2,
            ),
            candidate_sections=json.dumps(
                focused_directory,
                ensure_ascii=False,
                indent=2,
            ),
        )
        result = await agents.ainvoke_document_section_mapper(
            prompt,
            self.settings.agent_recursion_limit,
        )
        returned = {
            item.document_name: item.section_ids
            for item in result.mappings
            if item.document_name in aggregate_names
        }

        verified: Dict[str, List[int]] = {
            name: section_ids
            for name, section_ids in candidate.items()
            if name not in aggregate_types
        }
        for name in aggregate_names:
            allowed_ids = set(candidate[name])
            confirmed_ids = [
                section_id
                for section_id in returned.get(name, [])
                if section_id in allowed_ids
            ]
            seed_ids = deterministic_mapping.get(name, [])
            final_ids = list(dict.fromkeys([*seed_ids, *confirmed_ids]))
            if final_ids:
                verified[name] = final_ids

        normalized, dropped = normalize_document_section_map_lenient(
            verified,
            dir_info,
            preferred_document_types=deterministic_mapping,
        )
        self._log_dropped_mappings("semantic_verification", dropped)
        return normalized

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
