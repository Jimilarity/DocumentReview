import ast
import asyncio
import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List

from langgraph.graph import END, StateGraph
from langgraph.types import RunnableConfig

from agents import ContextSensitiveReviewAgents
from external_knowledge import KnowledgeContext, KnowledgeItem, KnowledgeService
from constants import STRUCTURED_FIELD_CACHE_SCHEMA_VERSION
from review_config import (
    ContextSensitiveSettings,
    load_context_sensitive_settings,
)
from errors.handler import CURRENT_RULE_INDEX
from structured_field_cache import (
    TECHNICAL_MAPPING_FAILURE,
    StructuredFieldCache,
    build_structured_source_fingerprint,
)
from .base import DocumentReviewExecutor, ReviewState
from .consistency import (
    CONSISTENCY_TASK,
    ConsistencySource,
    aggregate_consistency_results,
    comparable_sources,
    format_source_values,
    is_executable_consistency_rule,
    required_field_issues,
)
from .section_content import extract_section_ocr_text
from rules.filtering import (
    context_sensitive_document_names,
    filter_context_sensitive_rules,
)


class ContextSensitiveReviewState(ReviewState, total=False):
    field_cache_ready: bool
    structured_field_cache_path: str
    prewarmed_section_ids: List[int]
    delivery_event_count: int


def _valid_field_value(field_type: str, value: Any) -> bool:
    if value is None:
        return True
    if field_type in {"str", "datetime"}:
        return isinstance(value, str)
    if field_type == "bool":
        return isinstance(value, bool)
    if field_type == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if field_type == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if field_type == "dict":
        return isinstance(value, dict)
    if field_type == "list[str]":
        return isinstance(value, list) and all(
            isinstance(item, str) for item in value
        )
    if field_type == "list[dict]":
        return isinstance(value, list) and all(
            isinstance(item, dict) for item in value
        )
    if field_type.startswith("literal["):
        if not field_type.endswith("]"):
            raise ValueError(f"literal 字段类型配置无效: {field_type}")
        literal_body = field_type[len("literal[") : -1].strip()
        if not literal_body:
            raise ValueError(f"literal 字段类型配置无效: {field_type}")
        try:
            allowed_values = ast.literal_eval(f"[{literal_body}]")
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"literal 字段类型配置无效: {field_type}"
            ) from exc
        if not all(
            isinstance(item, (str, int, float, bool))
            for item in allowed_values
        ):
            raise ValueError(
                f"literal 字段类型仅支持字符串、数字和布尔值: {field_type}"
            )
        return any(
            type(value) is type(item) and value == item
            for item in allowed_values
        )
    raise ValueError(f"不支持的字段类型: {field_type}")


def validate_extracted_fields(
    extracted_fields: Dict[str, Any],
    requested_specs: Dict[str, Dict[str, Any]],
) -> None:
    if set(extracted_fields) != set(requested_specs):
        missing = sorted(set(requested_specs) - set(extracted_fields))
        unexpected = sorted(set(extracted_fields) - set(requested_specs))
        raise ValueError(
            f"字段提取结果不完整，缺少={missing}，多余={unexpected}"
        )

    for field_name, value in extracted_fields.items():
        field_type = requested_specs[field_name]["type"]
        if not _valid_field_value(field_type, value):
            raise TypeError(
                f"字段 {field_name} 应为 {field_type} 或 null，"
                f"实际为 {type(value).__name__}"
            )


class ContextSensitiveReviewExecutor(DocumentReviewExecutor):
    """准备结构化缓存并执行“上下文相关审查事项”。"""

    agents_class = ContextSensitiveReviewAgents

    required_document_names = staticmethod(context_sensitive_document_names)
    filter_rules = staticmethod(filter_context_sensitive_rules)

    def __init__(
        self,
        *args: Any,
        context_settings: ContextSensitiveSettings | None = None,
        knowledge_service: KnowledgeService | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.context_settings = (
            context_settings or load_context_sensitive_settings()
        )
        self.structured_field_cache: StructuredFieldCache | None = None
        self.last_preparation_result: Dict[str, Any] | None = None
        self.knowledge_service = knowledge_service or KnowledgeService(
            logger=self.logger
        )

    def get_local_tools(self) -> List[Any]:
        return []

    def create_agents(
        self,
        tools: List[Any],
    ) -> ContextSensitiveReviewAgents:
        del tools
        return self.agents_class()

    def require_context_sensitive_agents(self) -> ContextSensitiveReviewAgents:
        if not isinstance(self.agents, ContextSensitiveReviewAgents):
            raise RuntimeError("上下文相关审查智能体尚未初始化")
        return self.agents

    def require_structured_field_cache(self) -> StructuredFieldCache:
        if self.structured_field_cache is None:
            raise RuntimeError("结构化字段缓存尚未初始化")
        return self.structured_field_cache

    def _load_structured_field_cache(self) -> StructuredFieldCache:
        if self.structured_field_cache is not None:
            return self.structured_field_cache
        context = self.require_document_context()
        fingerprint = build_structured_source_fingerprint(
            context.meta_info,
            context.dir_info,
            context.ocr_results,
        )
        cache = StructuredFieldCache.load(
            self.cache_paths.structured_fields,
            schema_version=STRUCTURED_FIELD_CACHE_SCHEMA_VERSION,
            source_fingerprint=fingerprint,
        )
        self.structured_field_cache = cache
        cache.bind_field_extractor(self._extract_regular_fields)
        cache.bind_delivery_field_extractor(
            self._extract_delivery_event_fields
        )
        return cache

    @staticmethod
    def _minimal_prompt_specs(
        specs: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        prompt_specs = []
        for field_name, spec in specs.items():
            item = {"field": field_name, "type": spec["type"]}
            notes = spec.get("notes")
            if isinstance(notes, str) and notes.strip():
                item["notes"] = notes
            prompt_specs.append(item)
        return prompt_specs

    def _specs_for_section(
        self,
        section_id: int,
        field_names: Iterable[str],
    ) -> Dict[str, Dict[str, Any]]:
        cache = self.require_structured_field_cache()
        document_type = cache.document_type(section_id)
        return self._specs_for_fields(
            document_type,
            field_names,
        )

    def _specs_for_fields(
        self,
        document_type: str,
        field_names: Iterable[str],
    ) -> Dict[str, Dict[str, Any]]:
        field_specs = self.context_settings["field_specs"]
        if document_type not in field_specs:
            raise ValueError(
                f"文书类型“{document_type}”未在 section_fields.yaml 中配置"
            )
        document_specs = field_specs[document_type]
        missing_fields = [
            field_name
            for field_name in field_names
            if field_name not in document_specs
        ]
        if missing_fields:
            raise ValueError(
                f"文书类型“{document_type}”缺少字段配置: {missing_fields}"
            )
        return {
            field_name: document_specs[field_name]
            for field_name in field_names
        }

    async def _extract_regular_fields(
        self,
        section_id: int,
        field_names: List[str],
    ) -> Dict[str, Any]:
        cache = self.require_structured_field_cache()
        context = self.require_document_context()
        specs = self._specs_for_section(section_id, field_names)
        document_type = cache.document_type(section_id)
        prompt = self.require_context_sensitive_agents().build_task_prompt(
            "section_field_extraction",
            document_type=document_type,
            field_specs=json.dumps(
                self._minimal_prompt_specs(specs),
                ensure_ascii=False,
                indent=2,
            ),
            ocr_text=extract_section_ocr_text(
                section_id,
                context.dir_info,
                context.ocr_results,
            ),
        )
        result = await self.require_context_sensitive_agents().ainvoke_section_field_extractor(
            prompt,
            self.settings.agent_recursion_limit,
        )
        validate_extracted_fields(result.fields, specs)
        return result.fields

    async def _extract_delivery_event_fields(
        self,
        receipt_section_id: int,
        relation_key: str,
        field_names: List[str],
    ) -> Dict[str, Any]:
        cache = self.require_structured_field_cache()
        document_type = self.context_settings[
            "service_receipt_document_type"
        ]
        specs = self._specs_for_fields(
            document_type,
            field_names,
        )
        event = cache.delivery_event(receipt_section_id, relation_key)
        prompt = self.require_context_sensitive_agents().build_task_prompt(
            "section_field_extraction",
            document_type=f"{document_type}（单个送达记录）",
            field_specs=json.dumps(
                self._minimal_prompt_specs(specs),
                ensure_ascii=False,
                indent=2,
            ),
            ocr_text=event["event_text"],
        )
        result = await self.require_context_sensitive_agents().ainvoke_section_field_extractor(
            prompt,
            self.settings.agent_recursion_limit,
        )
        validate_extracted_fields(result.fields, specs)
        return result.fields

    def _prewarm_fields_by_document(self) -> Dict[str, List[str]]:
        plan = {
            document_type: list(field_names)
            for document_type, field_names in (
                self.context_settings["prewarm_fields"].items()
            )
        }
        for rule in self.rules:
            for context_item in rule["上下文相关审查事项"]:
                context_fields = context_item["字段"]
                for document_type, field_items in context_fields.items():
                    target = plan.setdefault(document_type, [])
                    for field_item in field_items:
                        if not isinstance(field_item, dict):
                            raise TypeError("上下文相关审查字段项必须是对象")
                        field_name = field_item.get("field")
                        if not isinstance(field_name, str) or not field_name:
                            raise TypeError("上下文相关审查字段名必须是非空字符串")
                        required = field_item.get("required")
                        if not isinstance(required, bool):
                            raise TypeError(
                                f"{document_type}.{field_name}.required "
                                "必须是 bool"
                            )
                        if field_name not in target:
                            target.append(field_name)

        context = getattr(self, "context", None)
        structured_input = (
            getattr(context, "meta_info", {}).get("source_type")
            == "structured_json"
        )
        configured_document_types = self.context_settings["field_specs"]
        for document_type, field_names in plan.items():
            if structured_input and document_type not in configured_document_types:
                continue
            self._specs_for_fields(
                document_type,
                field_names,
            )
        return plan

    def _register_mapped_sections(
        self,
        cache: StructuredFieldCache,
        document_types: Iterable[str],
    ) -> None:
        for document_type in document_types:
            section_ids = self.document_section_map.get(document_type, [])
            for section_id in section_ids:
                cache.register_section(section_id, document_type)

    @staticmethod
    def _compact_directory(
        dir_info: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        useful_keys = (
            "section_id",
            "section_name",
            "section_page",
            "section_end_page",
        )
        return [
            {key: item[key] for key in useful_keys if key in item}
            for item in dir_info
        ]

    async def _extract_delivery_receipt(
        self,
        section_id: int,
        field_names: List[str],
    ) -> List[Dict[str, Any]]:
        context = self.require_document_context()
        document_type = self.context_settings[
            "service_receipt_document_type"
        ]
        specs = self._specs_for_fields(
            document_type,
            field_names,
        )
        prompt = self.require_context_sensitive_agents().build_task_prompt(
            "delivery_receipt_extraction",
            field_specs=json.dumps(
                self._minimal_prompt_specs(specs),
                ensure_ascii=False,
                indent=2,
            ),
            ocr_text=extract_section_ocr_text(
                section_id,
                context.dir_info,
                context.ocr_results,
            ),
        )
        result = await self.require_context_sensitive_agents().ainvoke_delivery_receipt_extractor(
            prompt,
            self.settings.agent_recursion_limit,
        )
        source_orders = [event.source_order for event in result.events]
        if len(source_orders) != len(set(source_orders)):
            raise ValueError(
                f"送达回证 section_id={section_id} 的事件顺序重复"
            )

        events = []
        for event in sorted(result.events, key=lambda item: item.source_order):
            validate_extracted_fields(event.fields, specs)
            events.append(
                {
                    "source_order": event.source_order,
                    "event_text": event.event_text,
                    **event.fields,
                }
            )
        return events

    async def _map_delivery_event(
        self,
        receipt_section_id: int,
        event: Dict[str, Any],
    ) -> int | None:
        context = self.require_document_context()
        prompt = self.require_context_sensitive_agents().build_task_prompt(
            "delivery_receipt_mapping",
            meta_info=json.dumps(
                context.meta_info,
                ensure_ascii=False,
            ),
            receipt_section_id=receipt_section_id,
            event_id=event["event_id"],
            dir_info=json.dumps(
                self._compact_directory(context.dir_info),
                ensure_ascii=False,
                indent=2,
            ),
            event_text=event["event_text"],
        )
        try:
            result = await self.require_context_sensitive_agents().ainvoke_delivery_receipt_mapper(
                prompt,
                self.settings.agent_recursion_limit,
            )
        except Exception:
            return TECHNICAL_MAPPING_FAILURE

        related_section_id = result.related_section_id
        if related_section_id is None:
            return None
        valid_section_ids = {
            int(item["section_id"]) for item in context.dir_info
        }
        receipt_section_ids = set(
            self.document_section_map.get(
                self.context_settings["service_receipt_document_type"],
                [],
            )
        )
        if (
            related_section_id not in valid_section_ids
            or related_section_id in receipt_section_ids
        ):
            return TECHNICAL_MAPPING_FAILURE
        return related_section_id

    async def _prewarm_delivery_receipt(
        self,
        cache: StructuredFieldCache,
        section_id: int,
        field_names: List[str],
    ) -> bool:
        changed = False
        if cache.delivery_extraction_completed(section_id):
            events = [dict(event) for event in cache.delivery_events(section_id)]
        else:
            events = await self._extract_delivery_receipt(
                section_id,
                field_names,
            )
            next_event_id = cache.next_event_id()
            for offset, event in enumerate(events):
                event["event_id"] = next_event_id + offset
                event["related_section_id"] = TECHNICAL_MAPPING_FAILURE
            changed = True

        technical_failures = []
        for event in events:
            stored_relation_key = cache.delivery_event_storage_key(event)
            if event.get("related_section_id") == TECHNICAL_MAPPING_FAILURE:
                related_section_id = await self._map_delivery_event(
                    section_id,
                    event,
                )
                event["related_section_id"] = related_section_id
                changed = True

            missing_fields = [
                field_name
                for field_name in field_names
                if field_name not in event
            ]
            if missing_fields:
                if not cache.delivery_extraction_completed(section_id):
                    # 新事件尚未写入缓存，直接使用其原文完成补提取。
                    specs = self._specs_for_fields(
                        self.context_settings[
                            "service_receipt_document_type"
                        ],
                        missing_fields,
                    )
                    prompt = self.require_context_sensitive_agents().build_task_prompt(
                        "section_field_extraction",
                        document_type="送达回证（单个送达记录）",
                        field_specs=json.dumps(
                            self._minimal_prompt_specs(specs),
                            ensure_ascii=False,
                            indent=2,
                        ),
                        ocr_text=event["event_text"],
                    )
                    result = await self.require_context_sensitive_agents().ainvoke_section_field_extractor(
                        prompt,
                        self.settings.agent_recursion_limit,
                    )
                    validate_extracted_fields(result.fields, specs)
                    event.update(result.fields)
                else:
                    event.update(
                        await self._extract_delivery_event_fields(
                            section_id,
                            stored_relation_key,
                            missing_fields,
                        )
                    )
                changed = True

            if event.get("related_section_id") == TECHNICAL_MAPPING_FAILURE:
                technical_failures.append(event["event_id"])

        if changed or not cache.delivery_extraction_completed(section_id):
            cache.set_delivery_events(section_id, events)
            cache.save()
        if technical_failures:
            raise RuntimeError(
                f"送达回证 section_id={section_id} 的事件映射发生技术失败: "
                f"{technical_failures}"
            )
        return changed

    async def prewarm_structured_fields(
        self,
        state: ContextSensitiveReviewState,
        config: RunnableConfig,
    ) -> Dict[str, Any]:
        del state, config
        cache = self._load_structured_field_cache()
        prewarm_by_document = self._prewarm_fields_by_document()
        # section_metadata 只服务于上下文相关字段提取，不登记其他执行器
        # 使用的映射标签。这样证据属性类型的相容映射不会改变字段缓存仍为
        # 单一 document_type 的约束。
        self._register_mapped_sections(cache, prewarm_by_document)
        receipt_document_type = self.context_settings[
            "service_receipt_document_type"
        ]

        section_fields: Dict[int, List[str]] = defaultdict(list)
        for document_type, field_names in prewarm_by_document.items():
            if document_type == receipt_document_type:
                continue
            for section_id in self.document_section_map.get(document_type, []):
                for field_name in field_names:
                    if field_name not in section_fields[section_id]:
                        section_fields[section_id].append(field_name)

        prewarmed_section_ids = []
        semaphore = asyncio.Semaphore(
            max(1, self.settings.model_max_concurrency)
        )

        async def prewarm_regular(section_id: int, fields: List[str]) -> None:
            missing = cache.missing_fields(section_id, fields)
            if not missing:
                return
            async with semaphore:
                await cache.get_fields(section_id, missing)
            prewarmed_section_ids.append(section_id)

        await asyncio.gather(
            *(
                prewarm_regular(section_id, fields)
                for section_id, fields in section_fields.items()
            )
        )

        context = self.require_document_context()
        section_order = {
            int(item["section_id"]): index
            for index, item in enumerate(context.dir_info)
        }
        receipt_fields = prewarm_by_document.get(
            receipt_document_type,
            [],
        )
        receipt_section_ids = sorted(
            self.document_section_map.get(receipt_document_type, []),
            key=section_order.__getitem__,
        )
        for section_id in receipt_section_ids:
            if await self._prewarm_delivery_receipt(
                cache,
                section_id,
                receipt_fields,
            ):
                prewarmed_section_ids.append(section_id)

        cache.mark_preparation_completed()
        cache.save()
        return {
            "field_cache_ready": True,
            "structured_field_cache_path": str(cache.path),
            "prewarmed_section_ids": sorted(set(prewarmed_section_ids)),
            "delivery_event_count": cache.delivery_event_count(),
            "review_completed": True,
        }

    def build_review_graph(self):
        graph = StateGraph(ContextSensitiveReviewState)
        graph.add_node(
            "prewarm_structured_fields",
            self.prewarm_structured_fields,
        )
        graph.set_entry_point("prewarm_structured_fields")
        graph.add_edge("prewarm_structured_fields", END)
        return graph.compile()

    def build_single_rule_graph(self):
        # 上下文相关审查以跨文书事项为执行粒度，不使用单规则子图。
        return None

    def build_initial_state(self) -> ContextSensitiveReviewState:
        return {
            "rule_id": 0,
            "rule_results": [],
            "review_completed": False,
            "field_cache_ready": False,
            "prewarmed_section_ids": [],
            "delivery_event_count": 0,
        }

    async def run_preparation(self) -> Dict[str, Any]:
        """只执行上下文相关正式审查之前的准备流程。"""

        await self.initialize()
        cache = self._load_structured_field_cache()
        cache.mark_preparation_started()
        cache.save()
        try:
            final_state = await self.review_graph.ainvoke(
                self.build_initial_state()
            )
        except Exception as exc:
            self._raise_review_error(
                exc,
                "context_sensitive.preparation",
            )
        self.raise_for_final_state(final_state)
        return {
            "preparation_completed": final_state.get(
                "field_cache_ready",
                False,
            ),
            "rule_count": len(self.executable_rule_indexes()),
            "document_section_map": self.document_section_map,
            "structured_field_cache_path": final_state.get(
                "structured_field_cache_path"
            ),
            "prewarmed_section_ids": final_state.get(
                "prewarmed_section_ids",
                [],
            ),
            "delivery_event_count": final_state.get(
                "delivery_event_count",
                0,
            ),
        }

    def executable_consistency_rule_indexes(self) -> List[int]:
        return [
            index
            for index, rule in enumerate(self.rules)
            if is_executable_consistency_rule(rule)
            and self.executable_item_indexes(rule)
        ]

    def executable_contextual_legality_rule_indexes(self) -> List[int]:
        return [
            index
            for index, rule in enumerate(self.rules)
            if any(
                item.get("任务") == "不予处罚合法性审查"
                and any(item["字段"].values())
                for item in rule["上下文相关审查事项"]
            )
        ]

    def executable_item_indexes(
        self,
        rule: Dict[str, Any],
    ) -> List[int]:
        return [
            index
            for index, item in enumerate(rule["上下文相关审查事项"])
            if item.get("任务")
            and any(item["字段"].values())
        ]

    def validate_supported_context_tasks(self) -> None:
        unsupported = [
            (self.rule_identifier(rule), item.get("任务"))
            for rule in self.rules
            for item in rule["上下文相关审查事项"]
            if item.get("任务")
            and item.get("任务")
            not in {CONSISTENCY_TASK, "不予处罚合法性审查"}
        ]
        if unsupported:
            raise NotImplementedError(
                f"存在尚未实现的上下文相关审查任务: {unsupported}"
            )

    def executable_rule_indexes(self) -> List[int]:
        self.validate_supported_context_tasks()
        return [
            *self.executable_consistency_rule_indexes(),
            *self.executable_contextual_legality_rule_indexes(),
        ]

    @staticmethod
    def _knowledge_function_names(context_item: Dict[str, Any]) -> List[str]:
        configured = context_item.get("外部知识", [])
        if not isinstance(configured, list) or any(
            not isinstance(name, str) or not name.strip()
            for name in configured
        ):
            raise TypeError("上下文相关外部知识必须是非空名称数组")
        return list(dict.fromkeys(name.strip() for name in configured))

    @staticmethod
    def _validate_field_items(
        document_type: str,
        field_items: Any,
    ) -> List[Dict[str, Any]]:
        if not isinstance(field_items, list) or not field_items:
            raise TypeError(
                f"一致性核查字段 {document_type} 必须是非空对象数组"
            )
        for field_item in field_items:
            if not isinstance(field_item, dict):
                raise TypeError("一致性核查字段项必须是对象")
            field_name = field_item.get("field")
            if not isinstance(field_name, str) or not field_name:
                raise TypeError("一致性核查字段名必须是非空字符串")
            if not isinstance(field_item.get("required"), bool):
                raise TypeError(
                    f"{document_type}.{field_name}.required 必须是 bool"
                )
        return field_items

    async def _regular_consistency_sources(
        self,
        document_type: str,
        field_items: List[Dict[str, Any]],
    ) -> List[ConsistencySource]:
        cache = self.require_structured_field_cache()
        field_names = [item["field"] for item in field_items]

        async def collect_section(section_id: int) -> List[ConsistencySource]:
            values = await cache.get_fields(section_id, field_names)
            return [
                ConsistencySource(
                    document_type=document_type,
                    section_id=section_id,
                    field_name=item["field"],
                    required=item["required"],
                    value=values[item["field"]],
                )
                for item in field_items
            ]

        section_sources = await asyncio.gather(
            *(
                collect_section(section_id)
                for section_id in self.document_section_map[document_type]
            )
        )
        return [source for sources in section_sources for source in sources]

    async def _delivery_consistency_sources(
        self,
        document_type: str,
        field_items: List[Dict[str, Any]],
    ) -> List[ConsistencySource]:
        cache = self.require_structured_field_cache()
        field_names = [item["field"] for item in field_items]
        sources: List[ConsistencySource] = []
        for receipt_section_id in self.document_section_map[document_type]:
            events = cache.delivery_events(receipt_section_id)
            if not events:
                sources.extend(
                    ConsistencySource(
                        document_type=document_type,
                        section_id=receipt_section_id,
                        field_name=item["field"],
                        required=item["required"],
                        value=None,
                    )
                    for item in field_items
                )
                continue

            for event in events:
                related_section_id = event.get("related_section_id")
                if related_section_id == TECHNICAL_MAPPING_FAILURE:
                    raise RuntimeError(
                        f"送达回证 section_id={receipt_section_id} 的"
                        f" event_id={event['event_id']} 映射处于技术失败状态"
                    )
                relation_key = cache.delivery_event_storage_key(event)
                values = await cache.get_delivery_fields(
                    receipt_section_id,
                    relation_key,
                    field_names,
                )
                sources.extend(
                    ConsistencySource(
                        document_type=document_type,
                        section_id=receipt_section_id,
                        related_section_id=(
                            related_section_id
                            if isinstance(related_section_id, int)
                            and related_section_id > 0
                            else None
                        ),
                        field_name=item["field"],
                        required=item["required"],
                        value=values[item["field"]],
                    )
                    for item in field_items
                )
        return sources

    async def collect_consistency_sources(
        self,
        rule: Dict[str, Any],
        context_item_index: int,
    ) -> List[ConsistencySource]:
        available_fields = rule["上下文相关审查事项"][
            context_item_index
        ]["字段"]

        receipt_document_type = self.context_settings[
            "service_receipt_document_type"
        ]
        sources: List[ConsistencySource] = []
        for document_type, raw_field_items in available_fields.items():
            field_items = self._validate_field_items(
                document_type,
                raw_field_items,
            )
            if document_type == receipt_document_type:
                document_sources = await self._delivery_consistency_sources(
                    document_type,
                    field_items,
                )
            else:
                document_sources = await self._regular_consistency_sources(
                    document_type,
                    field_items,
                )
            sources.extend(document_sources)
        return sources

    async def _consistency_judgement(
        self,
        context_item: Dict[str, Any],
        sources: List[ConsistencySource],
        knowledge_items: List[KnowledgeItem],
    ) -> Any:
        compact_sources = [
            {
                "document_type": source.document_type,
                "section_id": source.section_id,
                **(
                    {"related_section_id": source.related_section_id}
                    if source.related_section_id is not None
                    else {}
                ),
                "field": source.field_name,
                "value": source.value,
            }
            for source in sources
        ]
        prompt = self.require_context_sensitive_agents().build_task_prompt(
            "consistency_review",
            review_item=context_item.get("审查事项", ""),
            sources=json.dumps(
                compact_sources,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            external_knowledge=json.dumps(
                [item.content for item in knowledge_items],
                ensure_ascii=False,
            ),
        )
        return await asyncio.wait_for(
            self.require_context_sensitive_agents().ainvoke_consistency_review(
                prompt,
                self.settings.agent_recursion_limit,
            ),
            timeout=self.settings.task_timeout_seconds,
        )

    async def collect_external_knowledge(
        self,
        rule: Dict[str, Any],
        context_item: Dict[str, Any],
        sources: List[ConsistencySource],
    ) -> List[KnowledgeItem]:
        """为任一上下文相关审查项汇总声明的外部知识。"""

        context = self.require_document_context()
        field_values = {
            source.field_name: source.value
            for source in sources
            if source.value is not None
        }
        return await self.knowledge_service.collect(
            self._knowledge_function_names(context_item),
            KnowledgeContext(
                metadata={**context.meta_info, **field_values},
                dir_info=context.dir_info,
                rule=rule,
                review_item=context_item,
                document_name="",
                section_id=0,
                section_ocr="",
            ),
        )

    async def run_consistency_item(
        self,
        rule: Dict[str, Any],
        context_item_index: int,
    ) -> Dict[str, Any]:
        context_item = rule["上下文相关审查事项"][context_item_index]
        task_name = context_item["任务"]
        sources = await self.collect_consistency_sources(
            rule,
            context_item_index,
        )
        knowledge_items = await self.collect_external_knowledge(
            rule,
            context_item,
            sources,
        )
        issues = required_field_issues(sources)
        if task_name != CONSISTENCY_TASK:
            raise ValueError(f"不支持的上下文相关审查任务: {task_name}")

        comparable = comparable_sources(sources)
        judgement = None
        if len(comparable) >= 2:
            judgement = await self._consistency_judgement(
                context_item,
                comparable,
                knowledge_items,
            )
            if not judgement.consistent:
                issues.append(
                    {
                        "section_ids": sorted(
                            {source.section_id for source in comparable}
                        ),
                        "content": (
                            "上下文相关字段未通过一致性核查："
                            f"{judgement.reason}；来源值："
                            f"{format_source_values(comparable)}。"
                        ),
                    }
                )

        return {"issues": issues}

    async def run_contextual_legality_item(
        self,
        rule: Dict[str, Any],
        context_item_index: int,
    ) -> Dict[str, Any]:
        context_item = rule["上下文相关审查事项"][context_item_index]
        sources = await self.collect_consistency_sources(
            rule,
            context_item_index,
        )
        knowledge_items = await self.collect_external_knowledge(
            rule,
            context_item,
            sources,
        )
        compact_sources = [
            {
                "document_type": source.document_type,
                "section_id": source.section_id,
                "field": source.field_name,
                "value": source.value,
            }
            for source in sources
        ]
        prompt = self.require_context_sensitive_agents().build_task_prompt(
            "contextual_legality_review",
            review_item=json.dumps(context_item, ensure_ascii=False),
            sources=json.dumps(compact_sources, ensure_ascii=False),
            external_knowledge=json.dumps(
                [item.content for item in knowledge_items],
                ensure_ascii=False,
            ),
        )
        agents = self.require_context_sensitive_agents()
        result = await agents.ainvoke_contextual_legality_review(
            prompt,
            self.settings.agent_recursion_limit,
        )
        return {"issues": [issue.model_dump() for issue in result.issues]}

    async def run_one_consistency_rule(self, index: int) -> Dict[str, Any]:
        rule = self.rules[index]
        rule_token = CURRENT_RULE_INDEX.set(
            self.rule_identifier(rule, index)
        )
        try:
            configured_item_indexes = self.executable_item_indexes(rule)
            item_results = await asyncio.gather(
                *(
                    self.run_consistency_item(rule, item_index)
                    if rule["上下文相关审查事项"][item_index]["任务"]
                    == CONSISTENCY_TASK
                    else self.run_contextual_legality_item(rule, item_index)
                    for item_index in configured_item_indexes
                )
            )
            return aggregate_consistency_results(rule, item_results)
        except Exception as exc:
            return self.build_rule_error_result(
                index,
                exc,
                stage="context_sensitive.review",
            )
        finally:
            CURRENT_RULE_INDEX.reset(rule_token)

    async def run_consistency_reviews(self) -> List[Dict[str, Any]]:
        """执行准备完成且当前实现能够完整覆盖的一致性规则。"""

        self.validate_supported_context_tasks()
        await self.initialize()
        cache = self._load_structured_field_cache()
        if not cache.preparation_completed:
            raise RuntimeError("上下文相关准备尚未完成，不能开始正式审查")

        indexes = self.executable_rule_indexes()
        semaphore = asyncio.Semaphore(
            max(1, self.settings.model_max_concurrency)
        )

        async def run(index: int) -> Dict[str, Any]:
            async with semaphore:
                return await self.run_one_consistency_rule(index)

        return await asyncio.gather(*(run(index) for index in indexes))

    async def execute_raw(self) -> List[Dict[str, Any]]:
        """按结构化准备、上下文相关审查顺序执行。"""

        self.last_preparation_result = await self.run_preparation()
        return await self.run_consistency_reviews()
