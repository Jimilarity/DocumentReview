import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Type

from agents import DocumentMappingAgents
from cache_paths import get_cache_paths
from constants import (
    RULES_PATH,
    STRUCTURED_FIELD_CACHE_SCHEMA_VERSION,
    ReviewExecutorType,
    SupportExecutorType,
)
from directory_info import normalize_directory_info
from review_config import (
    load_context_sensitive_settings,
    load_enabled_executor_types,
    load_enabled_support_executor_types,
)
from human_support import HumanSupportExecutor
from human_support.result_coordinator import HumanSupportResultCoordinator
from reviewers.base import (
    BaseReviewExecutor,
    DocumentReviewExecutor,
    ReviewSettings,
)
from reviewers.case_level import CaseLevelReviewExecutor
from reviewers.context_free import ContextFreeReviewExecutor
from reviewers.context_sensitive import ContextSensitiveReviewExecutor
from reviewers.document_preparation import DocumentReviewPreparationService
from reviewers.document_mapping import deterministic_document_section_map
from reviewers.result_aggregation import merge_rule_results
from reviewers.result_coordinator import ReviewResultCoordinator
from rules.rule_set import RuleSet, RuleSetBuilder
from rules.filtering import (
    filter_context_free_rules,
    filter_context_sensitive_rules,
    filter_human_support_rules,
    human_support_document_names,
    retrieval_enhancement_module_names,
)
from rules.rule_type import decode_rule_type
from structured_field_cache import (
    StructuredFieldCache,
    build_structured_source_fingerprint,
)
from utils import read_json


ExecutorClass = Type[BaseReviewExecutor]

EXECUTOR_REGISTRY: Dict[ReviewExecutorType, ExecutorClass] = {
    ReviewExecutorType.CONTEXT_FREE: ContextFreeReviewExecutor,
    ReviewExecutorType.CONTEXT_SENSITIVE: ContextSensitiveReviewExecutor,
    ReviewExecutorType.CASE_LEVEL: CaseLevelReviewExecutor,
}
DOCUMENT_EXECUTOR_TYPES = (
    ReviewExecutorType.CONTEXT_FREE,
    ReviewExecutorType.CONTEXT_SENSITIVE,
)
SUPPORT_EXECUTOR_REGISTRY = {
    SupportExecutorType.HUMAN_SUPPORT: HumanSupportExecutor,
}
LOGGER = logging.getLogger(__name__)


def select_executor_class(
    rule_type: int,
    executor: ReviewExecutorType = ReviewExecutorType.CONTEXT_FREE,
) -> ExecutorClass:
    """解析规则类型并返回配置标识对应的叶子执行器。"""

    decode_rule_type(rule_type)
    return EXECUTOR_REGISTRY[executor]


def _executor_type_for_class(
    executor_class: ExecutorClass,
) -> ReviewExecutorType:
    for executor_type, registered_class in EXECUTOR_REGISTRY.items():
        if issubclass(executor_class, registered_class):
            return executor_type
    raise ValueError(f"未注册的审查执行器: {executor_class.__name__}")


def _ordered_union(groups: Iterable[Iterable[str]]) -> List[str]:
    return list(dict.fromkeys(name for group in groups for name in group))


def _required_document_names(
    executor_type: ReviewExecutorType,
    candidate_rules: List[Dict[str, Any]],
) -> List[str]:
    executor_class = EXECUTOR_REGISTRY[executor_type]
    if not issubclass(executor_class, DocumentReviewExecutor):
        return []
    names = list(executor_class.required_document_names(candidate_rules))
    if executor_type is ReviewExecutorType.CONTEXT_SENSITIVE:
        names.extend(load_context_sensitive_settings()["prewarm_fields"])
    return list(dict.fromkeys(names))


def _load_structured_cache(
    file_path: str | Path,
    meta_info: Dict[str, Any],
    dir_info: List[Dict[str, Any]],
    ocr_results: List[Dict[str, Any]],
) -> StructuredFieldCache:
    return StructuredFieldCache.load(
        get_cache_paths(file_path).structured_fields,
        schema_version=STRUCTURED_FIELD_CACHE_SCHEMA_VERSION,
        source_fingerprint=build_structured_source_fingerprint(
            meta_info,
            dir_info,
            ocr_results,
        ),
    )


def _resolve_document_presence(
    cache: StructuredFieldCache,
    document_names: List[str],
    dir_info: List[Dict[str, Any]],
) -> Dict[str, bool]:
    """共享案件事实；范围扩展时废弃整份旧结构化缓存快照。"""

    if not document_names:
        return {}
    cached_presence = cache.document_presence
    cached_mapping = cache.payload.get("document_section_map", {})
    presence_complete = all(
        name in cached_presence for name in document_names
    )
    mapping_covers_scope = not cached_mapping or all(
        cached_presence[name] is not True or bool(cached_mapping.get(name))
        for name in document_names
        if name in cached_presence
    )
    if presence_complete and mapping_covers_scope:
        return {name: cached_presence[name] for name in document_names}

    try:
        deterministic_mapping = deterministic_document_section_map(
            document_names,
            dir_info,
        )
    except Exception as exc:
        LOGGER.warning(
            "deterministic document presence mapping failed; falling back "
            "to the classifier. error=%s: %s",
            type(exc).__name__,
            exc,
        )
        deterministic_mapping = {}
    unresolved_names = [
        name for name in document_names if name not in deterministic_mapping
    ]
    document_presence = {name: True for name in deterministic_mapping}
    if unresolved_names:
        try:
            document_presence.update(
                DocumentMappingAgents().classify_document_presence(
                    unresolved_names,
                    dir_info,
                )
            )
        except Exception as exc:
            LOGGER.warning(
                "document presence classification failed; unresolved "
                "documents will be skipped. error=%s: %s",
                type(exc).__name__,
                exc,
            )
            document_presence.update(
                {name: False for name in unresolved_names}
            )
    document_presence = {
        name: document_presence[name] for name in document_names
    }
    fresh_cache = StructuredFieldCache(
        cache.path,
        schema_version=cache.schema_version,
        source_fingerprint=cache.source_fingerprint,
    )
    fresh_cache.initialize_document_presence(document_presence)
    fresh_cache.save()
    return document_presence


def _build_document_rule_sets(
    file_path: str | Path,
    rule_type: int,
    executor_types: Iterable[ReviewExecutorType],
    rules_path: str | Path = RULES_PATH,
) -> tuple[
    Dict[ReviewExecutorType, RuleSet],
    Dict[str, bool],
]:
    """一次判断共享 presence，再为每个执行器物化独立 RuleSet。"""

    requested_types = list(dict.fromkeys(executor_types))
    if any(item not in DOCUMENT_EXECUTOR_TYPES for item in requested_types):
        raise ValueError("只有文书级执行器能够通过规则文件构建 RuleSet")

    cache_paths = get_cache_paths(file_path)
    meta_info = read_json(cache_paths.metadata)
    dir_info = normalize_directory_info(read_json(cache_paths.directory))
    ocr_results = read_json(cache_paths.ocr_results)
    candidate_rules_data = read_json(rules_path)
    candidate_rules = (
        RuleSetBuilder(candidate_rules_data)
        .for_rule_type(rule_type)
        .build()
        .rules
    )
    requirements = {
        executor_type: _required_document_names(
            executor_type,
            candidate_rules,
        )
        for executor_type in requested_types
    }
    required_names = _ordered_union(requirements.values())
    cache = _load_structured_cache(
        file_path,
        meta_info,
        dir_info,
        ocr_results,
    )
    document_presence = _resolve_document_presence(
        cache,
        required_names,
        dir_info,
    )

    rule_sets = {
        executor_type: (
            RuleSetBuilder(candidate_rules_data)
            .for_executor(
                rule_type,
                rule_filter=EXECUTOR_REGISTRY[executor_type].filter_rules,
                document_presence={
                    name: document_presence[name]
                    for name in requirements[executor_type]
                },
            )
            .build()
        )
        for executor_type in requested_types
    }
    return rule_sets, document_presence


def _candidate_human_support_rules(
    rule_type: int,
    rules_path: str | Path = RULES_PATH,
) -> List[Dict[str, Any]]:
    candidate_rules_data = read_json(rules_path)
    candidate_rules = (
        RuleSetBuilder(candidate_rules_data)
        .for_rule_type(rule_type)
        .build()
        .rules
    )
    return [
        rule
        for rule in candidate_rules
        if retrieval_enhancement_module_names(rule)
    ]


def _ensure_human_support_document_presence(
    file_path: str | Path,
    required_names: List[str],
    document_presence: Dict[str, bool],
) -> Dict[str, bool]:
    """补齐人工辅助独有文书，使其不依赖某个审查执行器是否启用。"""

    if all(name in document_presence for name in required_names):
        return document_presence
    cache_paths = get_cache_paths(file_path)
    meta_info = read_json(cache_paths.metadata)
    dir_info = normalize_directory_info(read_json(cache_paths.directory))
    ocr_results = read_json(cache_paths.ocr_results)
    cache = _load_structured_cache(
        file_path,
        meta_info,
        dir_info,
        ocr_results,
    )
    return _resolve_document_presence(
        cache,
        _ordered_union([document_presence, required_names]),
        dir_info,
    )


def _build_human_support_rule_set(
    candidate_rules: List[Dict[str, Any]],
    document_presence: Dict[str, bool],
) -> RuleSet:
    """人工辅助独立筛选规则，但复用业务规则选择和文书存在性。"""

    return RuleSet(
        rules=filter_human_support_rules(
            candidate_rules,
            document_presence,
        )
    )


def build_rule_set(
    file_path: str | Path,
    rule_type: int,
    *,
    executor: ReviewExecutorType = ReviewExecutorType.CONTEXT_FREE,
    rules_path: str | Path = RULES_PATH,
) -> RuleSet:
    """按照原 Builder 范式构建一个执行器专属的 RuleSet。"""

    rule_sets, _ = _build_document_rule_sets(
        file_path,
        rule_type,
        [executor],
        rules_path,
    )
    return rule_sets[executor]


def _present_document_names(
    document_presence: Dict[str, bool],
) -> List[str]:
    return [
        name
        for name, exists in document_presence.items()
        if exists
    ]


def _mapped_document_presence(
    document_section_map: Dict[str, List[int]],
) -> Dict[str, bool]:
    return {
        name: bool(section_ids)
        for name, section_ids in document_section_map.items()
    }


def _filter_rule_set_by_mapping(
    executor_type: ReviewExecutorType,
    rule_set: RuleSet,
    document_section_map: Dict[str, List[int]],
) -> RuleSet:
    configured_key = {
        ReviewExecutorType.CONTEXT_FREE: "上下文无关审查事项",
        ReviewExecutorType.CONTEXT_SENSITIVE: "上下文相关审查事项",
    }[executor_type]
    if any(configured_key not in rule for rule in rule_set.rules):
        return rule_set
    rule_filter = {
        ReviewExecutorType.CONTEXT_FREE: filter_context_free_rules,
        ReviewExecutorType.CONTEXT_SENSITIVE: filter_context_sensitive_rules,
    }[executor_type]
    return RuleSet(
        rules=rule_filter(
            rule_set.rules,
            _mapped_document_presence(document_section_map),
        )
    )


async def prepare_context_sensitive_review(
    file_path: str,
    rule_type: int,
    *,
    settings: ReviewSettings | None = None,
    rules_path: str | Path = RULES_PATH,
) -> Dict[str, Any]:
    """单独准备上下文相关审查规则所需的结构化缓存。"""

    rule_sets, document_presence = _build_document_rule_sets(
        file_path,
        rule_type,
        [ReviewExecutorType.CONTEXT_SENSITIVE],
        rules_path,
    )
    rule_set = rule_sets[ReviewExecutorType.CONTEXT_SENSITIVE]
    probe = ContextSensitiveReviewExecutor(
        file_path=file_path,
        rule_set=rule_set,
        document_section_map={},
        settings=settings,
    )
    if not probe.executable_rule_indexes():
        return {
            "preparation_completed": False,
            "rule_count": 0,
            "skipped": True,
            "reason": "no_applicable_context_sensitive_rules",
        }
    document_section_map = await DocumentReviewPreparationService(
        file_path,
        _present_document_names(document_presence),
        settings=settings,
    ).prepare()
    rule_set = _filter_rule_set_by_mapping(
        ReviewExecutorType.CONTEXT_SENSITIVE,
        rule_set,
        document_section_map,
    )
    if not rule_set.rules:
        return {
            "preparation_completed": False,
            "rule_count": 0,
            "skipped": True,
            "reason": "no_mapped_context_sensitive_rules",
        }
    executor = ContextSensitiveReviewExecutor(
        file_path=file_path,
        rule_set=rule_set,
        document_section_map=document_section_map,
        settings=settings,
    )
    return await executor.run_preparation()


def _presence_for_custom_rule_set(
    file_path: str,
    executor_type: ReviewExecutorType,
    rule_set: RuleSet,
) -> tuple[RuleSet, Dict[str, bool]]:
    cache_paths = get_cache_paths(file_path)
    meta_info = read_json(cache_paths.metadata)
    dir_info = normalize_directory_info(read_json(cache_paths.directory))
    ocr_results = read_json(cache_paths.ocr_results)
    required_names = _required_document_names(
        executor_type,
        rule_set.rules,
    )
    cache = _load_structured_cache(
        file_path,
        meta_info,
        dir_info,
        ocr_results,
    )
    document_presence = _resolve_document_presence(
        cache,
        required_names,
        dir_info,
    )
    executor_class = EXECUTOR_REGISTRY[executor_type]
    return (
        RuleSet(
            rules=executor_class.filter_rules(
                rule_set.rules,
                document_presence,
            )
        ),
        document_presence,
    )


async def _run_custom_executor(
    file_path: str,
    rule_type: int,
    executor_class: ExecutorClass | None,
    rule_set: RuleSet | None,
    settings: ReviewSettings | None,
    rules_path: str | Path,
) -> Dict[str, Any]:
    selected_class = executor_class or select_executor_class(rule_type)
    executor_type = _executor_type_for_class(selected_class)

    if executor_type is ReviewExecutorType.CASE_LEVEL:
        if rule_set is not None:
            raise ValueError("CaseLevelReviewExecutor 不接受 RuleSet")
        executor = selected_class(file_path=file_path, settings=settings)
        raw_results = await executor.execute_raw()
    else:
        if rule_set is None:
            rule_sets, document_presence = _build_document_rule_sets(
                file_path,
                rule_type,
                [executor_type],
                rules_path,
            )
            selected_rule_set = rule_sets[executor_type]
        else:
            selected_rule_set, document_presence = (
                _presence_for_custom_rule_set(
                    file_path,
                    executor_type,
                    rule_set,
                )
            )
        if not selected_rule_set.rules:
            return _empty_review_result("no_applicable_rules")

        kwargs: Dict[str, Any] = {
            "file_path": file_path,
            "rule_set": selected_rule_set,
            "settings": settings,
        }
        if issubclass(selected_class, DocumentReviewExecutor):
            document_section_map = await DocumentReviewPreparationService(
                file_path,
                _present_document_names(document_presence),
                settings=settings,
            ).prepare()
            selected_rule_set = _filter_rule_set_by_mapping(
                executor_type,
                selected_rule_set,
                document_section_map,
            )
            if not selected_rule_set.rules:
                return _empty_review_result("no_mapped_applicable_rules")
            kwargs["rule_set"] = selected_rule_set
            kwargs["document_section_map"] = document_section_map
        executor = selected_class(**kwargs)
        raw_results = await executor.execute_raw()

    coordinator = ReviewResultCoordinator(
        file_path,
        settings=executor.settings,
        rules_path=rules_path,
    )
    return await coordinator.finalize(
        raw_results,
        rule_count=len({item["rule_index"] for item in raw_results}),
    )


def _empty_review_result(reason: str) -> Dict[str, Any]:
    return {
        "rule_results": [],
        "findings": [],
        "review_results": [],
        "result_processing_enabled": False,
        "rule_count": 0,
        "result_path": None,
        "raw_result_path": None,
        "skipped": True,
        "reason": reason,
    }


def _disabled_executor_status(executor_type: ReviewExecutorType) -> Dict[str, Any]:
    return {
        "executor": EXECUTOR_REGISTRY[executor_type].__name__,
        "implemented": False,
        "skipped": True,
        "reason": "disabled",
    }


async def run_review(
    file_path: str,
    rule_type: int,
    *,
    executor_class: ExecutorClass | None = None,
    rule_set: RuleSet | None = None,
    settings: ReviewSettings | None = None,
    rules_path: str | Path = RULES_PATH,
) -> Dict[str, Any]:
    """按配置运行叶子审查器，共享案件事实与一次性文书映射。"""

    if executor_class is not None or rule_set is not None:
        return await _run_custom_executor(
            file_path,
            rule_type,
            executor_class,
            rule_set,
            settings,
            rules_path,
        )

    enabled_types = load_enabled_executor_types()
    enabled_support_types = load_enabled_support_executor_types()
    document_types = [
        executor_type
        for executor_type in enabled_types
        if executor_type in DOCUMENT_EXECUTOR_TYPES
    ]
    if document_types:
        rule_sets, document_presence = _build_document_rule_sets(
            file_path,
            rule_type,
            document_types,
            rules_path,
        )
    else:
        rule_sets, document_presence = {}, {}

    human_support_enabled = (
        SupportExecutorType.HUMAN_SUPPORT in enabled_support_types
    )
    human_support_setup_error: str | None = None
    if human_support_enabled:
        try:
            candidate_human_support_rules = (
                _candidate_human_support_rules(rule_type, rules_path)
            )
            support_document_names = human_support_document_names(
                candidate_human_support_rules
            )
            document_presence = _ensure_human_support_document_presence(
                file_path,
                support_document_names,
                document_presence,
            )
            human_support_rule_set = _build_human_support_rule_set(
                candidate_human_support_rules,
                document_presence,
            )
        except Exception as exc:
            human_support_setup_error = (
                f"{type(exc).__name__}: {exc}"
            )
            human_support_rule_set = RuleSet(rules=[])
    else:
        human_support_rule_set = RuleSet(rules=[])
    human_support_executor_class = SUPPORT_EXECUTOR_REGISTRY[
        SupportExecutorType.HUMAN_SUPPORT
    ]
    human_support_probe = human_support_executor_class(
        file_path=file_path,
        rule_set=human_support_rule_set,
        document_section_map={},
        settings=settings,
    )
    human_support_indexes = (
        human_support_probe.executable_rule_indexes()
        if human_support_enabled
        else []
    )

    executable_indexes: Dict[ReviewExecutorType, List[int]] = {}
    executable_rule_ids: set[int] = set()
    for executor_type in document_types:
        executor_class_for_type = EXECUTOR_REGISTRY[executor_type]
        probe = executor_class_for_type(
            file_path=file_path,
            rule_set=rule_sets[executor_type],
            document_section_map={},
            settings=settings,
        )
        indexes = probe.executable_rule_indexes()
        executable_indexes[executor_type] = indexes
        executable_rule_ids.update(
            rule_sets[executor_type].rules[index]["序号"]
            for index in indexes
        )

    if ReviewExecutorType.CASE_LEVEL in enabled_types:
        case_level_executor = EXECUTOR_REGISTRY[ReviewExecutorType.CASE_LEVEL](
            file_path,
            settings=settings,
        )
        try:
            case_level_results = await case_level_executor.execute_raw()
            case_level_status = case_level_executor.status()
        except Exception as exc:
            case_level_results = []
            case_level_status = {
                "executor": type(case_level_executor).__name__,
                "implemented": True,
                "skipped": False,
                "reason": "case_level_execution_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
    else:
        case_level_results = []
        case_level_status = _disabled_executor_status(
            ReviewExecutorType.CASE_LEVEL
        )

    if (
        not executable_rule_ids
        and not case_level_results
        and not human_support_indexes
    ):
        result = _empty_review_result("no_implemented_applicable_rules")
        result.update(
            {
                "context_sensitive_preparation": {
                    "preparation_completed": False,
                    "rule_count": 0,
                    "skipped": True,
                    "reason": (
                        "no_applicable_context_sensitive_rules"
                        if ReviewExecutorType.CONTEXT_SENSITIVE in enabled_types
                        else "disabled"
                    ),
                },
                "context_free_rule_count": 0,
                "context_sensitive_rule_count": 0,
                "case_level_review": case_level_status,
                "retrieval_enhancement": {
                    "retrieval_enhancement_results": [],
                    "rule_count": 0,
                    "result_path": None,
                    "skipped": True,
                    "reason": (
                        "retrieval_enhancement_setup_failed"
                        if human_support_setup_error
                        else (
                            "no_applicable_retrieval_enhancement_rules"
                            if human_support_enabled
                            else "disabled"
                        )
                    ),
                    **(
                        {"error": human_support_setup_error}
                        if human_support_setup_error
                        else {}
                    ),
                },
            }
        )
        return result

    mapping_requirements = []
    for executor_type, indexes in executable_indexes.items():
        if not indexes:
            continue
        mapping_requirements.append(
            _required_document_names(
                executor_type,
                rule_sets[executor_type].rules,
            )
        )
    if human_support_indexes:
        mapping_requirements.append(
            human_support_document_names(
                [
                    human_support_rule_set.rules[index]
                    for index in human_support_indexes
                ]
            )
        )
    mapping_names = [
        name
        for name in _ordered_union(mapping_requirements)
        if document_presence.get(name) is True
    ]
    document_section_map = (
        await DocumentReviewPreparationService(
            file_path,
            mapping_names,
            settings=settings,
        ).prepare()
        if mapping_names
        else {}
    )

    mapped_presence = _mapped_document_presence(document_section_map)
    for executor_type in document_types:
        rule_sets[executor_type] = _filter_rule_set_by_mapping(
            executor_type,
            rule_sets[executor_type],
            document_section_map,
        )
    if human_support_enabled:
        human_support_rule_set = RuleSet(
            rules=filter_human_support_rules(
                human_support_rule_set.rules,
                mapped_presence,
            )
        )

    executable_indexes = {}
    executable_rule_ids = set()
    for executor_type in document_types:
        executor_class_for_type = EXECUTOR_REGISTRY[executor_type]
        probe = executor_class_for_type(
            file_path=file_path,
            rule_set=rule_sets[executor_type],
            document_section_map=document_section_map,
            settings=settings,
        )
        indexes = probe.executable_rule_indexes()
        executable_indexes[executor_type] = indexes
        executable_rule_ids.update(
            rule_sets[executor_type].rules[index]["序号"]
            for index in indexes
        )
    human_support_probe = human_support_executor_class(
        file_path=file_path,
        rule_set=human_support_rule_set,
        document_section_map=document_section_map,
        settings=settings,
    )
    human_support_indexes = (
        human_support_probe.executable_rule_indexes()
        if human_support_enabled
        else []
    )

    result_groups: List[List[Dict[str, Any]]] = []
    executor_errors: List[Dict[str, str]] = []
    context_sensitive_preparation = {
        "preparation_completed": False,
        "rule_count": 0,
        "skipped": True,
        "reason": (
            "no_applicable_context_sensitive_rules"
            if ReviewExecutorType.CONTEXT_SENSITIVE in enabled_types
            else "disabled"
        ),
    }
    receipt_document_type = load_context_sensitive_settings().get(
        "service_receipt_document_type",
        "送达回证",
    )
    context_free_receipt_rules = (
        rule_sets.get(ReviewExecutorType.CONTEXT_FREE, RuleSet(rules=[])).rules
        if ReviewExecutorType.CONTEXT_FREE in rule_sets
        else []
    )
    needs_delivery_preparation = any(
        receipt_document_type in (rule.get("上下文无关审查事项") or {})
        and len(rule.get("上下文无关审查事项") or {}) > 1
        for rule in context_free_receipt_rules
    )
    if needs_delivery_preparation:
        delivery_preparer = ContextSensitiveReviewExecutor(
            file_path=file_path,
            rule_set=RuleSet(rules=[]),
            document_section_map=document_section_map,
            settings=settings,
        )
        try:
            await delivery_preparer.run_preparation()
        except Exception as exc:
            executor_errors.append(
                {
                    "executor": "delivery_preparation",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    for executor_type in document_types:
        if not executable_indexes[executor_type]:
            continue
        executor = EXECUTOR_REGISTRY[executor_type](
            file_path=file_path,
            rule_set=rule_sets[executor_type],
            document_section_map=document_section_map,
            settings=settings,
        )
        try:
            result_groups.append(await executor.execute_raw())
            if executor_type is ReviewExecutorType.CONTEXT_SENSITIVE:
                context_sensitive_preparation = executor.last_preparation_result
        except Exception as exc:
            executor_errors.append(
                {
                    "executor": executor_type.value,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if human_support_indexes:
        try:
            human_support_results = await human_support_executor_class(
                file_path=file_path,
                rule_set=human_support_rule_set,
                document_section_map=document_section_map,
                settings=settings,
            ).execute()
            human_support_status = HumanSupportResultCoordinator(
                file_path,
                settings=settings,
            ).finalize(human_support_results)
        except Exception as exc:
            human_support_status = {
                "retrieval_enhancement_results": [],
                "rule_count": 0,
                "result_path": None,
                "skipped": False,
                "reason": "retrieval_enhancement_execution_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
    else:
        human_support_status = {
            "retrieval_enhancement_results": [],
            "rule_count": 0,
            "result_path": None,
            "skipped": True,
            "reason": (
                "retrieval_enhancement_setup_failed"
                if human_support_setup_error
                else (
                    "no_applicable_retrieval_enhancement_rules"
                    if human_support_enabled
                    else "disabled"
                )
            ),
            **(
                {"error": human_support_setup_error}
                if human_support_setup_error
                else {}
            ),
        }

    result_groups.append(case_level_results)
    raw_results = merge_rule_results(*result_groups)
    if executable_rule_ids or case_level_results:
        coordinator = ReviewResultCoordinator(
            file_path,
            settings=settings,
            rules_path=rules_path,
        )
        result = await coordinator.finalize(
            raw_results,
            rule_count=len(executable_rule_ids),
        )
    else:
        result = _empty_review_result("no_implemented_applicable_rules")
    result["context_sensitive_preparation"] = context_sensitive_preparation
    result["context_free_rule_count"] = len(
        {
            rule_sets[ReviewExecutorType.CONTEXT_FREE].rules[index]["序号"]
            for index in executable_indexes.get(
                ReviewExecutorType.CONTEXT_FREE,
                [],
            )
        }
    )
    result["context_sensitive_rule_count"] = len(
        {
            rule_sets[ReviewExecutorType.CONTEXT_SENSITIVE].rules[index]["序号"]
            for index in executable_indexes.get(
                ReviewExecutorType.CONTEXT_SENSITIVE,
                [],
            )
        }
    )
    result["case_level_review"] = case_level_status
    result["retrieval_enhancement"] = human_support_status
    result["executor_errors"] = executor_errors
    return result
