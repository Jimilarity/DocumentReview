import ast
import asyncio
import json
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List

from langgraph.graph import END, StateGraph
from langgraph.types import RunnableConfig

from agents import ContextSensitiveReviewAgents
from external_knowledge import (
    KNOWLEDGE_UNAVAILABLE_NOTE,
    KnowledgeContext,
    KnowledgeItem,
    KnowledgeService,
)
from constants import STRUCTURED_FIELD_CACHE_SCHEMA_VERSION
from review_config import (
    ContextSensitiveSettings,
    load_context_sensitive_settings,
)
from errors.handler import CURRENT_PDF_PATH, CURRENT_RULE_INDEX
from structured_field_cache import (
    TECHNICAL_MAPPING_FAILURE,
    StructuredFieldCache,
    build_structured_source_fingerprint,
)
from .base import DocumentReviewExecutor, ReviewState
from .consistency import (
    CONSISTENCY_TASK,
    ConsistencySource,
    DEFAULT_FIELD_CATEGORY,
    FIELD_CATEGORIES,
    aggregate_consistency_results,
    comparable_sources,
    format_source_values,
    is_executable_consistency_rule,
    required_field_issues,
)
from rules.normalization import canonical_document_type, load_rule_aliases
from .section_content import extract_section_ocr_text, section_page_range
from rules.filtering import (
    context_sensitive_document_names,
    filter_context_sensitive_rules,
)


class ContextSensitiveReviewState(ReviewState, total=False):
    field_cache_ready: bool
    structured_field_cache_path: str
    prewarmed_section_ids: List[int]
    delivery_event_count: int


CONTEXTUAL_LEGALITY_TASK = "主体合法"
NO_PENALTY_LEGALITY_TASK = "不予处罚合法性审查"

# 字段说明 notes 中预存的转义串（\n、\" 等）被 YAML 普通标量按字面保留，
# 再经 json.dumps 二次转义成 \\n、\\\"，形成噪声。这里只还原常见转义。
_NOTES_ESCAPE_PAIRS = (
    ("\\n", "\n"),
    ("\\t", "\t"),
    ("\\r", "\r"),
    ('\\"', '"'),
    ("\\'", "'"),
    ("\\\\", "\\"),
)

_HEARING_NATURAL_PERSON_FINE_THRESHOLD = Decimal("5000")
_HEARING_ORGANIZATION_FINE_THRESHOLD = Decimal("100000")
_MONEY_PATTERN = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)")


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


def normalize_delivery_extracted_fields(
    extracted_fields: Dict[str, Any],
    requested_specs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Keep configured delivery keys, fill missing values, discard unknown keys."""

    return {
        field_name: extracted_fields.get(field_name)
        for field_name in requested_specs
    }


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

    @staticmethod
    def _iter_structured_records(value: Any) -> Iterable[Dict[str, Any]]:
        """遍历结构化缓存记录，同时兼容普通 section 和送达事件嵌套结构。"""

        if not isinstance(value, dict):
            return
        if any(not isinstance(item, dict) for item in value.values()):
            yield value
        for item in value.values():
            if isinstance(item, dict):
                yield from ContextSensitiveReviewExecutor._iter_structured_records(
                    item
                )

    def _case_party_baseline(self) -> Dict[str, Any]:
        """从全案已提取字段形成当事人基线，防止把代理人等角色当作当事人。"""

        cache = getattr(self, "structured_field_cache", None)
        records = (
            list(self._iter_structured_records(cache.sections))
            if isinstance(cache, StructuredFieldCache)
            else []
        )

        def first_value(*field_names: str) -> Any:
            for record in records:
                for field_name in field_names:
                    value = record.get(field_name)
                    if value not in (None, "", [], {}):
                        return value
            return None

        explicit_type = first_value("当事人类型", "行政相对人类型")
        organization_name = first_value(
            "法人名称", "非法人组织名称", "个体工商户字号名称"
        )
        credit_code = first_value(
            "法人统一社会信用代码", "统一社会信用代码"
        )
        natural_name = first_value("自然人姓名")
        natural_id = first_value("自然人证件号码")

        party_type = "未知"
        explicit_text = str(explicit_type or "")
        if any(label in explicit_text for label in ("法人", "组织", "单位")):
            party_type = "法人或其他组织"
        elif "自然人" in explicit_text or "公民" in explicit_text:
            party_type = "自然人"
        elif organization_name or credit_code:
            party_type = "法人或其他组织"
        elif natural_name or natural_id:
            party_type = "自然人"

        return {
            "当事人类型": party_type,
            "法人或其他组织名称": organization_name,
            "统一社会信用代码": credit_code,
            "自然人姓名": natural_name if party_type == "自然人" else None,
            "自然人证件号码": natural_id if party_type == "自然人" else None,
            "角色区分要求": (
                "驾驶员、送达人、签收人、代收人、代理人、受委托人、被询问人、"
                "法定代表人等人员不是当然的受处罚当事人；只有材料明确将其认定为"
                "责任承担主体时，才能按当事人核对。"
            ),
        }

    @staticmethod
    def _parse_money(value: Any) -> Decimal | None:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, (int, float, Decimal)):
            try:
                return Decimal(str(value))
            except InvalidOperation:
                return None
        text = str(value).replace(",", "").replace("，", "")
        matches = _MONEY_PATTERN.findall(text)
        if not matches:
            return None
        try:
            return max(Decimal(item) for item in matches)
        except InvalidOperation:
            return None

    def _hearing_applicability(self) -> Dict[str, Any]:
        """按已确认的通用金额门槛计算听证适用状态，未知时禁止直接判错。"""

        cache = self.require_structured_field_cache()
        records = list(self._iter_structured_records(cache.sections))
        party = self._case_party_baseline()
        preferred_fields = (
            "罚款金额（小写）",
            "罚款金额",
            "拟处罚款金额",
        )
        amounts: List[Decimal] = []
        for field_name in preferred_fields:
            for record in records:
                amount = self._parse_money(record.get(field_name))
                if amount is not None:
                    amounts.append(amount)
            if amounts:
                break

        amount = max(amounts) if amounts else None
        party_type = party["当事人类型"]
        threshold = None
        if party_type == "自然人":
            threshold = _HEARING_NATURAL_PERSON_FINE_THRESHOLD
        elif party_type == "法人或其他组织":
            threshold = _HEARING_ORGANIZATION_FINE_THRESHOLD

        if threshold is None or amount is None:
            status = "待人工复核"
            applicable = None
            reason = "当事人类型或罚款金额无法从知识库可靠核验，不得直接判错。"
        else:
            applicable = amount >= threshold
            status = "达到通用金额门槛" if applicable else "未达到通用金额门槛"
            reason = (
                f"本案当事人类型为{party_type}，可核验罚款金额为{amount:g}元；"
                f"通用听证门槛为{threshold:g}元，以上含本数。"
            )

        waived_statement = any(
            record.get("是否放弃陈述、申辩权利") is True
            or str(record.get("是否放弃陈述、申辩权利") or "").strip()
            in {"是", "已放弃", "同意", "true", "True", "1"}
            for record in records
        )
        return {
            "核验状态": status,
            "是否达到通用听证金额门槛": applicable,
            "当事人类型": party_type,
            "可核验罚款金额": float(amount) if amount is not None else None,
            "适用通用门槛": float(threshold) if threshold is not None else None,
            "门槛说明": "自然人罚款5000元以上；法人或其他组织罚款100000元以上；以上含本数。",
            "书面放弃陈述申辩权利": waived_statement,
            "处理要求": (
                f"{reason} 未达到通用门槛时，不得以未告知听证权、未等待听证期限"
                "为由输出问题；但规则明确提供特别法更低门槛时从其规定。"
                "特别法适用、主体类型或金额不能可靠确定时，结论为待人工复核，"
                "不得直接输出违规 issue。陈述申辩期限应与听证期限分别判断。"
            ),
        }

    def _issue_conflicts_with_case_baseline(
        self,
        content: str,
        party_baseline: Dict[str, Any],
        hearing: Dict[str, Any] | None,
        issue_payload: Dict[str, Any] | None = None,
    ) -> bool:
        """剔除与程序已确认基线直接冲突的模型误判。"""

        if hearing is not None:
            below_threshold = (
                hearing.get("是否达到通用听证金额门槛") is False
            )
            hearing_terms = ("听证权", "听证期限", "听证申请期限", "申请听证")
            statement_terms = ("陈述", "申辩")
            if (
                below_threshold
                and any(term in content for term in hearing_terms)
                and not any(term in content for term in statement_terms)
            ):
                return True
            if (
                hearing.get("书面放弃陈述申辩权利") is True
                and below_threshold
                and ("提前" in content or "期限" in content)
                and any(term in content for term in (*hearing_terms, *statement_terms))
            ):
                return True

        if party_baseline.get("当事人类型") == "法人或其他组织":
            organization_name = str(
                party_baseline.get("法人或其他组织名称") or ""
            )
            alleges_person_name_mismatch = (
                "当事人姓名" in content
                or "姓名不一致" in content
                or "无法确认是否为同一当事人" in content
                or "将自然人错误认定为处罚对象" in content
            )
            if alleges_person_name_mismatch:
                cache = getattr(self, "structured_field_cache", None)
                section_ids = (
                    issue_payload.get("section_ids") or []
                    if isinstance(issue_payload, dict)
                    else []
                )
                explicit_party_fields = {
                    "当事人类型",
                    "行政相对人类型",
                    "法人名称",
                    "非法人组织名称",
                    "个体工商户字号名称",
                    "法人统一社会信用代码",
                    "统一社会信用代码",
                    "自然人姓名",
                    "自然人证件号码",
                }
                has_explicit_party_evidence = False
                if isinstance(cache, StructuredFieldCache):
                    for raw_section_id in section_ids:
                        record = cache.sections.get(str(raw_section_id), {})
                        if not isinstance(record, dict):
                            continue
                        if any(
                            record.get(field_name) not in (None, "", [], {})
                            for field_name in explicit_party_fields
                        ):
                            has_explicit_party_evidence = True
                            break
                refers_to_org_identity = (
                    (organization_name and organization_name in content)
                    or "法人名称" in content
                    or "组织名称" in content
                    or "统一社会信用代码" in content
                )
                # 仅从“签名/姓名”这类角色不明字段不能推出处罚对象变成自然人。
                # 必须有主体类型、证件号或明确的自然人/法人身份字段支撑。
                if not has_explicit_party_evidence:
                    return True
                if not refers_to_org_identity:
                    return True
        return False

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
    def _unescape_notes(notes: str) -> str:
        for escaped, real in _NOTES_ESCAPE_PAIRS:
            notes = notes.replace(escaped, real)
        return notes

    @staticmethod
    def _minimal_prompt_specs(
        specs: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        prompt_specs = []
        for field_name, spec in specs.items():
            item = {"field": field_name, "type": spec["type"]}
            notes = spec.get("notes")
            if isinstance(notes, str) and notes.strip():
                item["notes"] = ContextSensitiveReviewExecutor._unescape_notes(
                    notes
                )
            prompt_specs.append(item)
        return prompt_specs

    def _specs_for_section(
        self,
        section_id: int,
        field_names: Iterable[str],
        document_type: str | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        cache = self.require_structured_field_cache()
        document_type = cache.document_type(section_id, document_type)
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
        document_specs = field_specs.setdefault(document_type, {})
        missing_fields = [
            field_name
            for field_name in field_names
            if field_name not in document_specs
        ]
        for field_name in missing_fields:
            document_specs[field_name] = {
                "type": "str",
                "notes": "数据库字段未单独配置类型，按文本字段提取",
            }
        return {
            field_name: document_specs[field_name]
            for field_name in field_names
        }

    async def _extract_regular_fields(
        self,
        section_id: int,
        document_type: str,
        field_names: List[str],
    ) -> Dict[str, Any]:
        cache = self.require_structured_field_cache()
        context = self.require_document_context()
        specs = self._specs_for_section(
            section_id,
            field_names,
            document_type,
        )
        ocr_text = extract_section_ocr_text(
            section_id,
            context.dir_info,
            context.ocr_results,
        )

        def build_prompt(target_specs: Dict[str, Dict[str, Any]]) -> str:
            return self.require_context_sensitive_agents().build_task_prompt(
                "section_field_extraction",
                document_type=document_type,
                field_specs=json.dumps(
                    self._minimal_prompt_specs(target_specs),
                    ensure_ascii=False,
                    indent=2,
                ),
                ocr_text=ocr_text,
            )

        fields = await self._run_field_extraction(build_prompt(specs), specs)

        # 第一次提取后，对仍为 null（未提取到）的字段做一次严格重试。
        null_fields = [
            field_name
            for field_name in field_names
            if fields.get(field_name) is None
        ]
        if null_fields:
            retry_specs = {
                field_name: specs[field_name]
                for field_name in null_fields
            }
            retry_prompt = build_prompt(retry_specs)
            retry_prompt += self._retry_extraction_hint(null_fields)
            retry_fields = await self._run_field_extraction(
                retry_prompt,
                retry_specs,
            )
            fields.update(retry_fields)

        return fields

    @staticmethod
    def _retry_extraction_hint(null_fields: List[str]) -> str:
        names = "、".join(null_fields)
        return (
            "\n\n<retry_extraction>\n"
            f"上一轮提取时，以下字段返回了 null（未找到明确对应值）：{names}。\n"
            "现在只重新提取这些字段。请逐字重新阅读当前文书 OCR："
            "只有找到与字段含义严格对应、且能在原文中明确指认的值时才填写；"
            "仍未载明、被遮挡、未填写、无法可靠辨认，或只能找到近义/其他字段的值时，"
            "必须继续返回 null。严禁用当事人住所或注册地址、近似时间、"
            "其他字段的值或任何推测来填充。\n"
            "</retry_extraction>"
        )

    async def _run_field_extraction(
        self,
        prompt: str,
        specs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        agents = self.require_context_sensitive_agents()
        max_attempts = agents._structured_output_max_attempts()
        validation_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            current_prompt = prompt
            if validation_error is not None:
                current_prompt += (
                    "\n\n<previous_field_validation_error>\n"
                    "上一轮虽然返回了 fields 对象，但字段名或字段值类型未通过校验："
                    f"{type(validation_error).__name__}: {validation_error}\n"
                    "请严格按照 field_specs 重新提取；字段必须完整，类型必须匹配，"
                    "找不到时返回 null。\n"
                    "</previous_field_validation_error>"
                )
            result = await agents.ainvoke_section_field_extractor(
                current_prompt,
                self.settings.agent_recursion_limit,
            )
            try:
                validate_extracted_fields(result.fields, specs)
                return result.fields
            except (TypeError, ValueError) as exc:
                validation_error = exc
                if attempt == max_attempts:
                    raise
        assert validation_error is not None
        raise validation_error

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
        fields = normalize_delivery_extracted_fields(result.fields, specs)
        validate_extracted_fields(fields, specs)
        return fields

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

    def _exact_pdf_location(self, section_id: int) -> Dict[str, int]:
        """仅在 section 恰好对应一页时提供可靠的 PDF 页码。"""

        # 构建提示词等轻量流程可能尚未初始化文档上下文；缺少页码不应
        # 中断字段提取或规则审查，只需省略位置字段。
        context = getattr(self, "context", None)
        if context is None:
            return {}
        try:
            start_page, end_page = section_page_range(
                section_id,
                context.dir_info,
                len(context.ocr_results),
            )
        except (KeyError, StopIteration, TypeError, ValueError):
            return {}
        if end_page - start_page != 1:
            return {}
        return {"pdf_page_number": start_page + 1}

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
            fields = normalize_delivery_extracted_fields(event.fields, specs)
            validate_extracted_fields(fields, specs)
            events.append(
                {
                    "source_order": event.source_order,
                    "event_text": event.event_text,
                    **fields,
                }
            )
        return events

    async def _map_delivery_event(
        self,
        receipt_section_id: int,
        event: Dict[str, Any],
    ) -> int | None:
        context = self.require_document_context()
        preferred_section_id = self._preferred_delivery_section(
            receipt_section_id,
            event,
        )
        if preferred_section_id is not None:
            return preferred_section_id
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

    def _preferred_delivery_section(
        self,
        receipt_section_id: int,
        event: Dict[str, Any],
    ) -> int | None:
        context = self.require_document_context()
        receipt_type = self.context_settings["service_receipt_document_type"]
        section_by_id = {
            int(item["section_id"]): item for item in context.dir_info
        }
        document_ids = [
            section_id
            for section_id, item in section_by_id.items()
            if section_id != receipt_section_id
            and str(item.get("section_name") or "") != receipt_type
            and "送达回证" not in str(item.get("section_name") or "")
            and "送达回执" not in str(item.get("section_name") or "")
        ]
        mapped_document_ids = {
            section_id
            for document_type, section_ids in getattr(
                self, "document_section_map", {}
            ).items()
            if document_type != receipt_type
            for section_id in section_ids
        }
        if mapped_document_ids:
            document_ids = [
                section_id
                for section_id in document_ids
                if section_id in mapped_document_ids
            ]
        if not document_ids:
            return None

        event_document_name = event.get("文书名称")
        if isinstance(event_document_name, str) and event_document_name.strip():
            normalized_event_name = self._normalize_delivery_name(
                event_document_name
            )
            matching_ids = [
                section_id
                for section_id in document_ids
                if self._delivery_names_match(
                    normalized_event_name,
                    self._normalize_delivery_name(
                        str(section_by_id[section_id].get("section_name") or "")
                    ),
                )
            ]
            if matching_ids:
                return self._nearest_preceding_section(
                    receipt_section_id,
                    matching_ids,
                    section_by_id,
                )

        preceding_ids = [
            section_id for section_id in document_ids if section_id < receipt_section_id
        ]
        if not preceding_ids:
            return None
        return max(
            preceding_ids,
            key=lambda section_id: (
                int(section_by_id[section_id].get("section_page", section_id)),
                section_id,
            ),
        )

    @staticmethod
    def _normalize_delivery_name(value: str) -> str:
        normalized = "".join(value.split()).replace("（", "(").replace("）", ")")
        try:
            return canonical_document_type(normalized, load_rule_aliases())
        except (TypeError, ValueError):
            return normalized

    @staticmethod
    def _delivery_names_match(event_name: str, section_name: str) -> bool:
        if not event_name or not section_name:
            return False
        return (
            event_name == section_name
            or event_name in section_name
            or section_name in event_name
        )

    @staticmethod
    def _nearest_preceding_section(
        receipt_section_id: int,
        section_ids: Iterable[int],
        section_by_id: Dict[int, Dict[str, Any]],
    ) -> int:
        preceding = [section_id for section_id in section_ids if section_id < receipt_section_id]
        candidates = preceding or list(section_ids)
        return max(
            candidates,
            key=lambda section_id: (
                int(section_by_id[section_id].get("section_page", section_id)),
                section_id,
            ),
        )

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
            current_relation_id = event.get("related_section_id")
            if not (
                isinstance(current_relation_id, int)
                and current_relation_id > 0
            ):
                continue
            preferred_relation_id = None
            if getattr(self, "context", None) is not None:
                preferred_relation_id = self._preferred_delivery_section(
                    section_id,
                    event,
                )
            if (
                isinstance(preferred_relation_id, int)
                and preferred_relation_id > 0
                and preferred_relation_id != current_relation_id
            ):
                self.logger.warning(
                    "校正送达事件文书映射: receipt_section_id=%s event_id=%s "
                    "旧映射=%s 新映射=%s",
                    section_id,
                    event["event_id"],
                    current_relation_id,
                    preferred_relation_id,
                )
                event["related_section_id"] = preferred_relation_id
                changed = True

        mapped_relation_ids: set[int] = set()
        for event in events:
            related_section_id = event.get("related_section_id")
            if (
                isinstance(related_section_id, int)
                and related_section_id > 0
            ):
                if related_section_id in mapped_relation_ids:
                    event["related_section_id"] = None
                    changed = True
                    self.logger.warning(
                        "同一送达回证存在重复文书映射，保留首个事件并将后续事件标记为未关联: "
                        "receipt_section_id=%s event_id=%s related_section_id=%s",
                        section_id,
                        event["event_id"],
                        related_section_id,
                    )
                    continue
                mapped_relation_ids.add(related_section_id)
        for event in events:
            stored_relation_key = cache.delivery_event_storage_key(event)
            if event.get("related_section_id") == TECHNICAL_MAPPING_FAILURE:
                related_section_id = await self._map_delivery_event(
                    section_id,
                    event,
                )
                if (
                    isinstance(related_section_id, int)
                    and related_section_id > 0
                    and related_section_id in mapped_relation_ids
                ):
                    self.logger.warning(
                        "送达事件映射冲突，保留首个事件并将后续事件标记为未关联: "
                        "receipt_section_id=%s event_id=%s related_section_id=%s",
                        section_id,
                        event["event_id"],
                        related_section_id,
                    )
                    related_section_id = None
                event["related_section_id"] = related_section_id
                changed = True
                if (
                    isinstance(related_section_id, int)
                    and related_section_id > 0
                ):
                    mapped_relation_ids.add(related_section_id)

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
                    fields = normalize_delivery_extracted_fields(
                        result.fields,
                        specs,
                    )
                    validate_extracted_fields(fields, specs)
                    event.update(fields)
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
        # 只登记上下文相关字段实际使用的映射标签；同一 section 可以登记
        # 多个相容类型，字段提取时显式携带当前规则使用的文书类型。
        self._register_mapped_sections(cache, prewarm_by_document)
        context_settings = getattr(self, "context_settings", None)
        if context_settings is None:
            context_settings = load_context_sensitive_settings()
        receipt_document_type = context_settings[
            "service_receipt_document_type"
        ]

        section_fields: Dict[tuple[int, str], List[str]] = defaultdict(list)
        for document_type, field_names in prewarm_by_document.items():
            if document_type == receipt_document_type:
                continue
            for section_id in self.document_section_map.get(document_type, []):
                for field_name in field_names:
                    key = (section_id, document_type)
                    if field_name not in section_fields[key]:
                        section_fields[key].append(field_name)

        prewarmed_section_ids = []
        semaphore = asyncio.Semaphore(
            max(1, self.settings.model_max_concurrency)
        )

        async def prewarm_regular(
            section_id: int,
            document_type: str,
            fields: List[str],
        ) -> None:
            missing = cache.missing_fields(section_id, fields)
            if not missing:
                return
            async with semaphore:
                await cache.get_fields(
                    section_id,
                    missing,
                    document_type=document_type,
                )
            prewarmed_section_ids.append(section_id)

        await asyncio.gather(
            *(
                prewarm_regular(section_id, document_type, fields)
                for (section_id, document_type), fields in section_fields.items()
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
                isinstance(item.get("任务"), str)
                and bool(item["任务"].strip())
                and item["任务"] != CONSISTENCY_TASK
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
            and not isinstance(item.get("任务"), str)
        ]
        if unsupported:
            raise TypeError(
                f"上下文相关审查任务必须是非空字符串: {unsupported}"
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
            field_category = field_item.get(
                "字段类别",
                DEFAULT_FIELD_CATEGORY,
            )
            if field_category not in FIELD_CATEGORIES:
                raise ValueError(
                    f"{document_type}.{field_name}.字段类别 必须是"
                    f" {sorted(FIELD_CATEGORIES)} 之一"
                )
        return field_items

    async def _regular_consistency_sources(
        self,
        document_type: str,
        field_items: List[Dict[str, Any]],
        section_ids: Iterable[int] | None = None,
    ) -> List[ConsistencySource]:
        cache = self.require_structured_field_cache()
        field_names = [item["field"] for item in field_items]

        async def collect_section(section_id: int) -> List[ConsistencySource]:
            values = await cache.get_fields(
                section_id,
                field_names,
                document_type=document_type,
            )
            return [
                ConsistencySource(
                    document_type=document_type,
                    section_id=section_id,
                    field_name=item["field"],
                    required=item["required"],
                    value=values[item["field"]],
                    field_category=item.get(
                        "字段类别",
                        DEFAULT_FIELD_CATEGORY,
                    ),
                )
                for item in field_items
            ]

        section_sources = await asyncio.gather(
            *(
                collect_section(section_id)
                for section_id in (
                    list(section_ids)
                    if section_ids is not None
                    else self.document_section_map.get(document_type, [])
                )
            )
        )
        return [source for sources in section_sources for source in sources]

    async def _delivery_consistency_sources(
        self,
        document_type: str,
        field_items: List[Dict[str, Any]],
        related_section_ids: set[int] | None = None,
    ) -> List[ConsistencySource]:
        cache = self.require_structured_field_cache()
        field_names = [item["field"] for item in field_items]
        sources: List[ConsistencySource] = []
        for receipt_section_id in self.document_section_map.get(document_type, []):
            events = cache.delivery_events(receipt_section_id)
            if related_section_ids is not None:
                events = [
                    event
                    for event in events
                    if event.get("related_section_id") in related_section_ids
                    or event.get("related_section_id") == TECHNICAL_MAPPING_FAILURE
                ]
            if not events:
                if related_section_ids is not None:
                    continue
                sources.extend(
                    ConsistencySource(
                        document_type=document_type,
                        section_id=receipt_section_id,
                        field_name=item["field"],
                        required=item["required"],
                        value=None,
                        field_category=item.get(
                            "字段类别",
                            DEFAULT_FIELD_CATEGORY,
                        ),
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
                        field_category=item.get(
                            "字段类别",
                            DEFAULT_FIELD_CATEGORY,
                        ),
                    )
                    for item in field_items
                )
        return sources

    async def collect_consistency_source_groups(
        self,
        rule: Dict[str, Any],
        context_item_index: int,
    ) -> List[List[ConsistencySource]]:
        context_item = rule["上下文相关审查事项"][context_item_index]
        available_fields = context_item.get("字段")
        if not isinstance(available_fields, dict):
            return [
                await self.collect_consistency_sources(
                    rule,
                    context_item_index,
                )
            ]

        context_settings = getattr(self, "context_settings", None)
        if context_settings is None:
            context_settings = load_context_sensitive_settings()
        receipt_document_type = context_settings[
            "service_receipt_document_type"
        ]
        if receipt_document_type not in available_fields:
            return [
                await self.collect_consistency_sources(
                    rule,
                    context_item_index,
                )
            ]

        # 送达回证按关联文书拆组，是“一致性核查”用于逐份比对文号、日期的
        # 专用机制。合法性、期限、事实证据等上下文审查通常需要同时看到多个
        # 文书；若规则没有显式指定关联锚点，拆组会把成立结论所需的证据割裂。
        if (
            context_item.get("任务") != CONSISTENCY_TASK
            and "送达回证关联文书" not in context_item
        ):
            return [
                await self.collect_consistency_sources(
                    rule,
                    context_item_index,
                )
            ]

        ordinary_document_types = [
            document_type
            for document_type, field_items in available_fields.items()
            if document_type != receipt_document_type and field_items
        ]
        if not ordinary_document_types:
            return [
                await self.collect_consistency_sources(
                    rule,
                    context_item_index,
                )
            ]
        configured_anchor = context_item.get("送达回证关联文书")
        if configured_anchor is not None:
            if not isinstance(configured_anchor, str):
                raise TypeError("送达回证关联文书必须是字符串")
            if configured_anchor not in ordinary_document_types:
                raise ValueError(
                    "送达回证关联文书未配置在当前上下文事项中: "
                    f"{configured_anchor}"
                )
            anchor_document_types = [configured_anchor]
        else:
            anchor_document_types = ordinary_document_types

        regular_sources: List[ConsistencySource] = []
        for document_type in ordinary_document_types:
            field_items = self._validate_field_items(
                document_type,
                available_fields[document_type],
            )
            regular_sources.extend(
                await self._regular_consistency_sources(
                    document_type,
                    field_items,
                )
            )
        receipt_field_items = self._validate_field_items(
            receipt_document_type,
            available_fields[receipt_document_type],
        )
        groups: List[List[ConsistencySource]] = []
        anchor_section_ids = [
            section_id
            for document_type in anchor_document_types
            for section_id in self.document_section_map.get(document_type, [])
        ]
        for anchor_section_id in anchor_section_ids:
            delivery_sources = await self._delivery_consistency_sources(
                receipt_document_type,
                receipt_field_items,
                {anchor_section_id},
            )
            if not delivery_sources:
                continue
            group_regular_sources = [
                source
                for source in regular_sources
                if source.section_id == anchor_section_id
            ]
            groups.append([*group_regular_sources, *delivery_sources])
        return groups or [regular_sources]

    async def collect_consistency_sources(
        self,
        rule: Dict[str, Any],
        context_item_index: int,
    ) -> List[ConsistencySource]:
        available_fields = rule["上下文相关审查事项"][
            context_item_index
        ]["字段"]

        context_settings = getattr(self, "context_settings", None)
        if context_settings is None:
            context_settings = load_context_sensitive_settings()
        receipt_document_type = context_settings[
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
                **self._exact_pdf_location(source.section_id),
                **(
                    {"related_section_id": source.related_section_id}
                    if source.related_section_id is not None
                    else {}
                ),
                "field": source.field_name,
                "field_category": source.field_category,
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
        structured_fields = [
            {
                "document_type": source.document_type,
                "section_id": source.section_id,
                "related_section_id": source.related_section_id,
                "field": source.field_name,
                "value": source.value,
            }
            for source in sources
            if source.value is not None
        ]
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
                structured_fields=structured_fields,
            ),
        )

    async def run_consistency_item(
        self,
        rule: Dict[str, Any],
        context_item_index: int,
    ) -> Dict[str, Any]:
        context_item = rule["上下文相关审查事项"][context_item_index]
        task_name = context_item["任务"]
        source_groups = await self.collect_consistency_source_groups(
            rule,
            context_item_index,
        )
        sources = [source for group in source_groups for source in group]
        knowledge_items = await self.collect_external_knowledge(
            rule,
            context_item,
            sources,
        )
        issues = []
        if task_name != CONSISTENCY_TASK:
            raise ValueError(f"不支持的上下文相关审查任务: {task_name}")

        for group in source_groups:
            issues.extend(required_field_issues(group))
            comparable = comparable_sources(group)
            if len(comparable) < 2:
                continue
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

    @staticmethod
    def _trimmed_review_item(context_item: Dict[str, Any]) -> Dict[str, Any]:
        """只把审查目标所需的规则内容交给模型，去掉整块字段配置。"""

        keys = ("任务", "评查类别", "审查事项", "评查说明")
        return {key: context_item.get(key, "") for key in keys}

    @staticmethod
    def _needs_hearing_applicability(
        context_item: Dict[str, Any],
    ) -> bool:
        """仅为真正审查听证权或听证程序的事项注入门槛结论。

        “办案期限扣除听证期间”等文字只是把听证作为时间事件提及，若也注入
        听证门槛，会给模型增加与本事项无关的强提示并稀释真正的审查目标。
        """

        text = "\n".join(
            str(context_item.get(key) or "")
            for key in ("任务", "审查事项", "评查说明")
        )
        hearing_targets = (
            "听证权",
            "听证申请",
            "申请听证",
            "告知听证",
            "听证告知",
            "听证条件",
            "听证门槛",
            "组织听证",
            "举行听证",
            "听证程序",
            "听证通知",
            "听证笔录",
            "听证报告",
        )
        return any(target in text for target in hearing_targets)

    @staticmethod
    def _compact_source_ocr(
        ocr_text: str,
        field_names: Iterable[str],
        max_chars: int,
    ) -> str:
        """在提示词预算内保留页首、页尾和字段名附近的原始 OCR。"""

        if max_chars <= 0 or not ocr_text:
            return ""
        if len(ocr_text) <= max_chars:
            return ocr_text

        window_radius = 550
        ranges = [(0, min(2200, len(ocr_text)))]
        tail_start = max(0, len(ocr_text) - 2200)
        ranges.append((tail_start, len(ocr_text)))
        for field_name in field_names:
            field = str(field_name or "").strip()
            if not field:
                continue
            candidates = [field]
            simplified = re.sub(r"[（(].*?[）)]", "", field).strip()
            if simplified and simplified != field:
                candidates.append(simplified)
            for candidate in candidates:
                start = 0
                while True:
                    index = ocr_text.find(candidate, start)
                    if index < 0:
                        break
                    ranges.append(
                        (
                            max(0, index - window_radius),
                            min(
                                len(ocr_text),
                                index + len(candidate) + window_radius,
                            ),
                        )
                    )
                    start = index + len(candidate)
                    if len(ranges) >= 24:
                        break
                if len(ranges) >= 24:
                    break

        merged: List[tuple[int, int]] = []
        for start, end in sorted(ranges):
            if merged and start <= merged[-1][1] + 80:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        chunks: List[str] = []
        used = 0
        for start, end in merged:
            remaining = max_chars - used
            if remaining <= 0:
                break
            chunk = ocr_text[start:end]
            if len(chunk) > remaining:
                chunk = chunk[:remaining]
            chunks.append(chunk)
            used += len(chunk)
        return "\n...[中间非相关 OCR 已压缩]...\n".join(chunks)

    def _source_documents_for_group(
        self,
        group: List[ConsistencySource],
        *,
        max_total_chars: int = 24000,
        max_section_chars: int = 10000,
    ) -> List[Dict[str, Any]]:
        """提供结构化提取失败时可回查的映射文书 OCR。"""

        context = getattr(self, "context", None)
        if (
            context is None
            or not hasattr(context, "dir_info")
            or not hasattr(context, "ocr_results")
        ):
            return []

        sections: Dict[int, Dict[str, Any]] = {}
        order: List[int] = []
        for source in group:
            section_id = source.section_id
            if section_id not in sections:
                sections[section_id] = {
                    "document_types": [],
                    "field_names": [],
                    "null_review_fields": 0,
                    "null_fields": 0,
                }
                order.append(section_id)
            entry = sections[section_id]
            if source.document_type not in entry["document_types"]:
                entry["document_types"].append(source.document_type)
            if source.field_name not in entry["field_names"]:
                entry["field_names"].append(source.field_name)
            if source.value is None:
                entry["null_fields"] += 1
                if source.field_category == "审查对象":
                    entry["null_review_fields"] += 1

        order_index = {section_id: index for index, section_id in enumerate(order)}
        ranked_section_ids = sorted(
            order,
            key=lambda section_id: (
                -sections[section_id]["null_review_fields"],
                -sections[section_id]["null_fields"],
                order_index[section_id],
            ),
        )

        documents: List[Dict[str, Any]] = []
        remaining = max_total_chars
        for section_id in ranked_section_ids:
            if remaining <= 0:
                break
            try:
                ocr_text = extract_section_ocr_text(
                    section_id,
                    context.dir_info,
                    context.ocr_results,
                )
            except (KeyError, StopIteration, TypeError, ValueError):
                continue
            limit = min(max_section_chars, remaining)
            compact_ocr = self._compact_source_ocr(
                ocr_text,
                sections[section_id]["field_names"],
                limit,
            )
            if not compact_ocr:
                continue
            documents.append(
                {
                    "document_types": sections[section_id]["document_types"],
                    "section_id": section_id,
                    **self._exact_pdf_location(section_id),
                    "ocr_text": compact_ocr,
                }
            )
            remaining -= len(compact_ocr)
        return documents

    async def run_contextual_legality_item(
        self,
        rule: Dict[str, Any],
        context_item_index: int,
    ) -> Dict[str, Any]:
        context_item = rule["上下文相关审查事项"][context_item_index]
        source_groups = await self.collect_consistency_source_groups(
            rule,
            context_item_index,
        )
        sources = [source for group in source_groups for source in group]
        knowledge_items = await self.collect_external_knowledge(
            rule,
            context_item,
            sources,
        )
        prompt_name = (
            "contextual_legality_review"
            if context_item["任务"] == NO_PENALTY_LEGALITY_TASK
            else "categorized_contextual_review"
        )
        agents = self.require_context_sensitive_agents()
        review_item_payload = self._trimmed_review_item(context_item)
        party_baseline = self._case_party_baseline()
        review_item_payload["全案当事人基线"] = party_baseline
        hearing = None
        if self._needs_hearing_applicability(context_item):
            hearing = self._hearing_applicability()
            review_item_payload["听证门槛与权利期限核验"] = hearing
        if knowledge_items:
            external_knowledge_payload = json.dumps(
                [item.content for item in knowledge_items],
                ensure_ascii=False,
            )
        elif self._knowledge_function_names(context_item):
            external_knowledge_payload = json.dumps(
                KNOWLEDGE_UNAVAILABLE_NOTE,
                ensure_ascii=False,
            )
        else:
            external_knowledge_payload = json.dumps([], ensure_ascii=False)
        issues = []
        for group in source_groups:
            compact_sources = [
                {
                    "document_type": source.document_type,
                    "section_id": source.section_id,
                    **self._exact_pdf_location(source.section_id),
                    **(
                        {"related_section_id": source.related_section_id}
                        if source.related_section_id is not None
                        else {}
                    ),
                    "field": source.field_name,
                    "field_category": source.field_category,
                    "value": source.value,
                }
                for source in group
            ]
            source_documents = self._source_documents_for_group(group)
            prompt = self.require_context_sensitive_agents().build_task_prompt(
                prompt_name,
                review_item=json.dumps(
                    review_item_payload,
                    ensure_ascii=False,
                ),
                sources=json.dumps(compact_sources, ensure_ascii=False),
                source_documents=json.dumps(
                    source_documents,
                    ensure_ascii=False,
                ),
                external_knowledge=external_knowledge_payload,
            )
            result = await agents.ainvoke_contextual_legality_review(
                prompt,
                self.settings.agent_recursion_limit,
            )
            if context_item["任务"] != NO_PENALTY_LEGALITY_TASK:
                issues.extend(required_field_issues(group))
            for issue in result.issues:
                issue_payload = issue.model_dump()
                content = str(issue_payload.get("content") or "")
                if self._issue_conflicts_with_case_baseline(
                    content,
                    party_baseline,
                    hearing,
                    issue_payload,
                ):
                    self.logger.warning(
                        "context issue discarded because it conflicts with "
                        "verified case baseline. rule=%s content=%s",
                        rule.get("序号"),
                        content,
                    )
                    continue
                issues.append(issue_payload)
        return {
            "issues": issues
        }

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
        pdf_path_token = CURRENT_PDF_PATH.set(str(self.file_path))
        try:
            self.last_preparation_result = await self.run_preparation()
            return await self.run_consistency_reviews()
        finally:
            CURRENT_PDF_PATH.reset(pdf_path_token)
