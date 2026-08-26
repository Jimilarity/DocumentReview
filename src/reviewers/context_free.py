import asyncio
import json
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import RunnableConfig

from agents import ContextFreeReviewAgents, ContextSensitiveReviewAgents
from constants import ErrorCode
from external_knowledge import (
    KnowledgeContext,
    KnowledgeItem,
    KnowledgeService,
)
from .base import (
    DocumentReviewContext,
    DocumentReviewExecutor,
    ReviewState,
    SingleRuleState,
)
from errors.handler import (
    CURRENT_NODE,
    CURRENT_RULE_INDEX,
    CURRENT_SECTION_ID,
    details_from_exception,
)
from .section_content import extract_section_ocr_text
from .result_aggregation import (
    aggregate_component_results,
    aggregate_section_results,
)
from tools import inspect_current_section_images
from constants import STRUCTURED_FIELD_CACHE_SCHEMA_VERSION
from review_config import load_context_sensitive_settings
from structured_field_cache import (
    StructuredFieldCache,
    build_structured_source_fingerprint,
)
from rules.filtering import (
    context_free_document_names,
    filter_context_free_rules,
)


class ContextFreeReviewState(ReviewState, total=False):
    pass


class ContextFreeSingleRuleState(SingleRuleState, total=False):
    dir_info: List[Dict[str, Any]]
    document_name: str
    document_review_rule: Dict[str, Any]
    section_id: int
    ocr_text: str
    delivery_scope: str
    external_knowledge: List[KnowledgeItem]


ContextFreeReviewContext = DocumentReviewContext


class ContextFreeReviewExecutor(DocumentReviewExecutor):
    """执行不依赖案卷其他 section 的单文书审查。"""

    agents_class = ContextFreeReviewAgents

    required_document_names = staticmethod(context_free_document_names)
    filter_rules = staticmethod(filter_context_free_rules)

    def __init__(
        self,
        *args: Any,
        knowledge_service: KnowledgeService | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.structured_field_cache: StructuredFieldCache | None = None
        self._query_field_cache: Dict[tuple[int, str], Any] = {}
        self._query_field_agents: ContextSensitiveReviewAgents | None = None
        self.knowledge_service = knowledge_service or KnowledgeService(
            logger=self.logger
        )

    def get_local_tools(self) -> List[Any]:
        context = self.require_context_free_context()
        if context.meta_info.get("source_type") == "structured_json":
            return []
        return [inspect_current_section_images]

    def create_agents(
        self,
        tools: List[Any],
    ) -> ContextFreeReviewAgents:
        return self.agents_class(review_tools=tools)

    def require_context_free_agents(self) -> ContextFreeReviewAgents:
        if not isinstance(self.agents, ContextFreeReviewAgents):
            raise RuntimeError("上下文无关审查智能体尚未初始化")
        return self.agents

    def require_context_free_context(self) -> ContextFreeReviewContext:
        return self.require_document_context()

    def _load_structured_field_cache(self) -> StructuredFieldCache | None:
        if self.structured_field_cache is not None:
            return self.structured_field_cache
        context = self.require_context_free_context()
        try:
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
        except (FileNotFoundError, ValueError, TypeError):
            return None
        self.structured_field_cache = cache
        return cache

    def _receipt_section_ids_for_rule(
        self,
        rule: Dict[str, Any],
        available_documents: Dict[str, Dict[str, Any]],
    ) -> List[int]:
        receipt_document_type = "送达回证"
        if receipt_document_type not in available_documents:
            return self.document_section_map.get(
                receipt_document_type,
                [],
            )
        document_review_rule = available_documents[receipt_document_type]
        anchor_document_type = document_review_rule.get(
            "送达回证关联文书"
        )
        ordinary_document_types = [
            document_name
            for document_name in available_documents
            if document_name != receipt_document_type
        ]
        if not anchor_document_type and len(ordinary_document_types) == 1:
            anchor_document_type = ordinary_document_types[0]
        if not isinstance(anchor_document_type, str):
            return []
        target_section_ids = set(
            self.document_section_map.get(anchor_document_type, [])
        )
        cache = self._load_structured_field_cache()
        receipt_section_ids = list(
            self.document_section_map.get(receipt_document_type, [])
        )
        if not target_section_ids or not receipt_section_ids:
            return []
        matching_receipt_sections = []
        if cache is not None:
            for receipt_section_id in receipt_section_ids:
                matching_events = [
                    event
                    for event in cache.delivery_events(receipt_section_id)
                    if event.get("related_section_id") in target_section_ids
                ]
                if matching_events:
                    matching_receipt_sections.append(receipt_section_id)

        return sorted(set(matching_receipt_sections))

    def _delivery_scope_for_section(
        self,
        available_documents: Dict[str, Dict[str, Any]],
        receipt_section_id: int,
    ) -> str:
        receipt_document_type = "送达回证"
        receipt_rule = available_documents.get(receipt_document_type) or {}
        anchor_document_type = receipt_rule.get("送达回证关联文书")
        ordinary_document_types = [
            document_name
            for document_name in available_documents
            if document_name != receipt_document_type
        ]
        if not isinstance(anchor_document_type, str):
            if len(ordinary_document_types) != 1:
                return ""
            anchor_document_type = ordinary_document_types[0]
        target_section_ids = set(
            self.document_section_map.get(anchor_document_type, [])
        )
        cache = self._load_structured_field_cache()
        if cache is None:
            return ""
        events = [
            event
            for event in cache.delivery_events(receipt_section_id)
            if event.get("related_section_id") in target_section_ids
        ]
        if not events:
            return ""
        return json.dumps(
            {
                "目标文书": anchor_document_type,
                "目标文书section_ids": sorted(target_section_ids),
                "本回证中仅审查的送达事件": [
                    {
                        "event_id": event["event_id"],
                        "source_order": event["source_order"],
                        "related_section_id": event["related_section_id"],
                    }
                    for event in events
                ],
                "处理要求": (
                    "送达回证可能包含多个送达事件；只审查上述事件对应的目标文书，"
                    "忽略同一回证中发送给其他文书的事件。"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )

    def document_names(self) -> List[str]:
        return list(
            dict.fromkeys(
                document_name
                for rule in self.rules
                for document_name in self.review_documents_for_rule(rule)
            )
        )

    def review_documents_for_rule(
        self,
        rule: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        return rule["上下文无关审查事项"]

    @staticmethod
    def build_effective_rule(
        rule: Dict[str, Any],
        document_review_rule: Dict[str, Any],
    ) -> Dict[str, Any]:
        effective_rule = {
            key: document_review_rule[key]
            for key in ("审查事项", "评查说明")
            if key in document_review_rule
        }
        if rule["备注"]:
            effective_rule["适用范围"] = rule["备注"]
        return effective_rule

    def build_single_rule_input(
        self,
        rule: Dict[str, Any],
        document_name: str,
        document_review_rule: Dict[str, Any],
        section_id: int,
        delivery_scope: str = "",
    ) -> ContextFreeSingleRuleState:
        context = self.require_context_free_context()
        ocr_text = extract_section_ocr_text(
            section_id,
            context.dir_info,
            context.ocr_results,
        )
        return {
            "rule": rule,
            "meta_info": context.meta_info,
            "dir_info": context.dir_info,
            "document_name": document_name,
            "document_review_rule": document_review_rule,
            "section_id": section_id,
            "ocr_text": ocr_text,
            "delivery_scope": delivery_scope,
            "external_knowledge": [],
        }

    @staticmethod
    def knowledge_function_names(
        document_review_rule: Dict[str, Any],
    ) -> List[str]:
        configured = document_review_rule.get("外部知识", [])
        if configured is None:
            return []
        if not isinstance(configured, list) or any(
            not isinstance(name, str) or not name.strip()
            for name in configured
        ):
            raise TypeError("外部知识必须是由非空知识函数名称组成的数组")
        return list(dict.fromkeys(name.strip() for name in configured))

    @staticmethod
    def _query_field_references(
        document_review_rule: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        config = document_review_rule.get("法条检索查询") or {}
        if not isinstance(config, dict):
            raise TypeError("法条检索查询必须是对象")
        fields = config.get("字段", [])
        if not isinstance(fields, list):
            raise TypeError("法条检索查询.字段必须是数组")
        references = []
        for item in fields:
            if isinstance(item, str) and item.strip():
                references.append({"字段": item.strip()})
            elif isinstance(item, dict):
                field_name = item.get("字段")
                document_name = item.get("文书")
                if not isinstance(field_name, str) or not field_name.strip():
                    raise TypeError("法条检索查询.字段对象必须包含非空字段")
                if document_name is not None and not isinstance(
                    document_name, str
                ):
                    raise TypeError("法条检索查询.字段对象的文书必须是字符串")
                references.append(
                    {
                        "字段": field_name.strip(),
                        **(
                            {"文书": document_name.strip()}
                            if isinstance(document_name, str)
                            and document_name.strip()
                            else {}
                        ),
                    }
                )
            else:
                raise TypeError("法条检索查询.字段必须由字符串或对象组成")
        return references

    async def _extract_query_fields(
        self,
        document_review_rule: Dict[str, Any],
        current_document_name: str,
        current_section_id: int,
    ) -> List[Dict[str, Any]]:
        references = self._query_field_references(document_review_rule)
        if not references:
            return []
        context = self.require_context_free_context()
        settings = load_context_sensitive_settings()
        targets: Dict[str, set[int]] = {}
        for reference in references:
            document_name = reference.get("文书", current_document_name)
            section_ids = (
                [current_section_id]
                if document_name == current_document_name
                else self.document_section_map.get(document_name, [])
            )
            for section_id in section_ids:
                targets.setdefault(document_name, set()).add(section_id)
        if self._query_field_agents is None:
            self._query_field_agents = ContextSensitiveReviewAgents()
        results: List[Dict[str, Any]] = []
        for document_name, section_ids in targets.items():
            field_names = [
                reference["字段"]
                for reference in references
                if reference.get("文书", current_document_name)
                == document_name
            ]
            document_specs = settings["field_specs"].get(document_name, {})
            missing = [
                field_name
                for field_name in field_names
                if field_name not in document_specs
            ]
            if missing:
                raise ValueError(
                    f"文书类型 {document_name} 缺少字段配置: {missing}"
                )
            field_specs = {
                field_name: document_specs[field_name]
                for field_name in dict.fromkeys(field_names)
            }
            prompt_specs = [
                {"field": field_name, "type": spec["type"]}
                for field_name, spec in field_specs.items()
            ]
            for section_id in sorted(section_ids):
                values = {}
                missing_fields = [
                    field_name
                    for field_name in field_specs
                    if (section_id, field_name) not in self._query_field_cache
                ]
                if missing_fields:
                    result = await self._query_field_agents.ainvoke_section_field_extractor(
                        self._query_field_agents.build_task_prompt(
                            "section_field_extraction",
                            document_type=document_name,
                            field_specs=json.dumps(
                                [
                                    {
                                        "field": field_name,
                                        "type": field_specs[field_name]["type"],
                                    }
                                    for field_name in missing_fields
                                ],
                                ensure_ascii=False,
                            ),
                            ocr_text=extract_section_ocr_text(
                                section_id,
                                context.dir_info,
                                context.ocr_results,
                            ),
                        ),
                        self.settings.agent_recursion_limit,
                    )
                    for field_name in missing_fields:
                        self._query_field_cache[(section_id, field_name)] = (
                            result.fields.get(field_name)
                        )
                for field_name in field_specs:
                    values[field_name] = self._query_field_cache.get(
                        (section_id, field_name)
                    )
                results.extend(
                    {
                        "document_type": document_name,
                        "section_id": section_id,
                        "field": field_name,
                        "value": value,
                    }
                    for field_name, value in values.items()
                    if value not in (None, "")
                )
        return results

    async def collect_external_knowledge(
        self,
        state: ContextFreeSingleRuleState,
    ) -> List[KnowledgeItem]:
        function_names = self.knowledge_function_names(
            state["document_review_rule"]
        )
        if not function_names:
            return []
        structured_fields = await self._extract_query_fields(
            state["document_review_rule"],
            state["document_name"],
            state["section_id"],
        )
        return await self.knowledge_service.collect(
            function_names,
            KnowledgeContext(
                metadata=state["meta_info"],
                dir_info=state["dir_info"],
                rule=state["rule"],
                review_item=state["document_review_rule"],
                document_name=state["document_name"],
                section_id=state["section_id"],
                section_ocr=state["ocr_text"],
                structured_fields=structured_fields,
            ),
        )

    def build_rule_prompt(
        self,
        state: ContextFreeSingleRuleState,
    ) -> str:
        effective_rule = self.build_effective_rule(
            state["rule"],
            state["document_review_rule"],
        )
        prompt = self.require_context_free_agents().build_task_prompt(
            "context_free_document_review",
            meta_info=json.dumps(
                state["meta_info"],
                ensure_ascii=False,
            ),
            rule_info=json.dumps(
                effective_rule,
                ensure_ascii=False,
            ),
            document_name=state["document_name"],
            section_id=state["section_id"],
            ocr_text=state["ocr_text"],
        )
        delivery_scope = state.get("delivery_scope", "")
        if delivery_scope:
            prompt = (
                f"{prompt}\n\n"
                "<delivery_scope>\n"
                f"{delivery_scope}\n"
                "</delivery_scope>"
            )
        knowledge_items = state.get("external_knowledge", [])
        if not knowledge_items:
            return prompt

        knowledge_json = json.dumps(
            [{"content": item.content} for item in knowledge_items],
            ensure_ascii=False,
            indent=2,
        )
        return (
            f"{prompt}\n\n"
            "<external_knowledge>\n"
            f"{knowledge_json}\n"
            "</external_knowledge>"
        )

    async def execute_context_free_rule(
        self,
        state: ContextFreeSingleRuleState,
        config: RunnableConfig,
    ) -> Dict[str, Any]:
        node_token = CURRENT_NODE.set("context_free.execute_rule")
        try:
            result = await self.require_context_free_agents().ainvoke_review(
                self.build_rule_prompt(state),
                self.settings.agent_recursion_limit,
            )
            result_data = result.model_dump()
            for issue in result_data["issues"]:
                issue["section_ids"] = [state["section_id"]]
            result_data["rule_index"] = self.rule_identifier(state["rule"])
            result_data["document_name"] = state["document_name"]
            result_data["section_id"] = state["section_id"]
            return {"result": result_data}
        finally:
            CURRENT_NODE.reset(node_token)

    def build_single_rule_graph(self):
        graph = StateGraph(ContextFreeSingleRuleState)
        graph.add_node("execute_rule", self.execute_context_free_rule)
        graph.set_entry_point("execute_rule")
        graph.add_edge("execute_rule", END)
        return graph.compile()

    def build_section_error_result(
        self,
        index: int,
        document_name: str,
        section_id: int,
        exc: BaseException,
        stage: str,
    ) -> Dict[str, Any]:
        result = self.build_rule_error_result(
            index,
            exc,
            stage=stage,
        )
        for issue in result["issues"]:
            issue["section_ids"] = [section_id]
        result["document_name"] = document_name
        result["section_id"] = section_id
        return result

    async def run_one_section(
        self,
        index: int,
        document_name: str,
        document_review_rule: Dict[str, Any],
        section_id: int,
        config: RunnableConfig,
        delivery_scope: str = "",
    ) -> Dict[str, Any]:
        rule = self.rules[index]
        rule_token = CURRENT_RULE_INDEX.set(self.rule_identifier(rule, index))
        section_token = CURRENT_SECTION_ID.set(section_id)
        try:
            rule_config = dict(config or {})
            rule_config["recursion_limit"] = (
                self.settings.agent_recursion_limit
            )
            async def execute_section_transaction() -> Dict[str, Any]:
                single_rule_input = self.build_single_rule_input(
                    rule,
                    document_name,
                    document_review_rule,
                    section_id,
                    delivery_scope,
                )
                single_rule_input["external_knowledge"] = (
                    await self.collect_external_knowledge(single_rule_input)
                )
                return await self.single_rule_graph.ainvoke(
                    single_rule_input,
                    config=rule_config,
                )

            response = await asyncio.wait_for(
                execute_section_transaction(),
                timeout=self.settings.task_timeout_seconds,
            )
            if not isinstance(response, dict) or "result" not in response:
                raise RuntimeError("单 section 审查返回结果缺少 result")
            return self.normalize_rule_result(rule, response["result"])
        except asyncio.TimeoutError as exc:
            return self.build_section_error_result(
                index,
                document_name,
                section_id,
                exc,
                "single_section.timeout",
            )
        except Exception as exc:
            return self.build_section_error_result(
                index,
                document_name,
                section_id,
                exc,
                "single_section",
            )
        finally:
            CURRENT_SECTION_ID.reset(section_token)
            CURRENT_RULE_INDEX.reset(rule_token)

    async def run_document_rule(
        self,
        rule_index: int,
        config: RunnableConfig,
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        rule = self.rules[rule_index]
        available_documents = self.review_documents_for_rule(rule)
        if not available_documents:
            raise ValueError(
                f"规则 {self.rule_identifier(rule)} 没有可执行的单文书分项"
            )

        missing_mappings = [
            document_name
            for document_name in available_documents
            if not self.document_section_map.get(document_name)
        ]
        if missing_mappings:
            raise RuntimeError(
                f"规则 {self.rule_identifier(rule)} 缺少文书 section 映射: "
                f"{missing_mappings}"
            )

        async def run_section(
            document_name: str,
            document_rule: Dict[str, Any],
            section_id: int,
            delivery_scope: str = "",
        ) -> Dict[str, Any]:
            async with semaphore:
                return await self.run_one_section(
                    rule_index,
                    document_name,
                    document_rule,
                    section_id,
                    config,
                    delivery_scope,
                )

        component_results = []
        for document_name, document_rule in available_documents.items():
            section_ids = (
                self._receipt_section_ids_for_rule(
                    rule,
                    available_documents,
                )
                if document_name == "送达回证"
                else self.document_section_map[document_name]
            )
            if not section_ids:
                continue
            section_results = await asyncio.gather(
                *(
                    run_section(
                        document_name,
                        document_rule,
                        section_id,
                        (
                            self._delivery_scope_for_section(
                                available_documents,
                                section_id,
                            )
                            if document_name == "送达回证"
                            else ""
                        ),
                    )
                    for section_id in section_ids
                )
            )
            component_results.append(
                aggregate_section_results(
                    self.rule_identifier(rule),
                    section_results,
                )
            )
        return aggregate_component_results(
            self.rule_identifier(rule),
            component_results,
        )

    async def run_document_rules(
        self,
        rule_indexes: List[int],
        config: RunnableConfig | None = None,
    ) -> List[Dict[str, Any]]:
        semaphore = asyncio.Semaphore(
            max(1, self.settings.model_max_concurrency)
        )
        return await asyncio.gather(
            *(
                self.run_document_rule(
                    rule_index,
                    config or {},
                    semaphore,
                )
                for rule_index in rule_indexes
            )
        )

    async def distribute_rules(
        self,
        state: ContextFreeReviewState,
        config: RunnableConfig,
    ) -> Dict[str, Any]:
        node_token = CURRENT_NODE.set("distribute_rule_sections")
        try:
            start_index = state["rule_id"]
            indexes = self.executable_rule_indexes()
            pending_indexes = [
                index for index in indexes if index >= start_index
            ]
            if not pending_indexes:
                return {"review_completed": True}
            rule_results = await self.run_document_rules(
                pending_indexes,
                config,
            )
            return {
                "rule_results": rule_results,
                "rule_id": len(self.rules),
                "review_completed": True,
            }
        except Exception as exc:
            details = details_from_exception(
                exc,
                "distribute_rule_sections",
            )
            return {
                "error_code": int(ErrorCode.UNEXPECTED_ERROR),
                "error_message": (
                    f"[{details['error_id']}] "
                    f"{details['exception_type']}: {details['message']}"
                ),
                "error_details": details,
            }
        finally:
            CURRENT_NODE.reset(node_token)

    def build_review_graph(self):
        graph = StateGraph(ContextFreeReviewState)
        graph.add_node("distribute_rules", self.distribute_rules)
        graph.add_node("error_node", self.handle_review_error)

        graph.set_entry_point("distribute_rules")
        graph.add_conditional_edges("distribute_rules", self.route_review)
        graph.add_edge("error_node", END)
        return graph.compile()

    def build_initial_state(self) -> ContextFreeReviewState:
        return {
            "rule_id": 0,
            "rule_results": [],
            "review_completed": False,
        }

    def executable_rule_indexes(self) -> List[int]:
        """独立 RuleSet 中的规则都已筛选出至少一个可用文书。"""

        return [
            index
            for index, rule in enumerate(self.rules)
            if rule["上下文无关审查事项"]
        ]
