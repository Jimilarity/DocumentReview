import copy
from typing import Any, Dict, Iterable, List

from constants import (
    DocumentType,
    PenaltyProcedure,
)


def _unique_document_names(names: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(names))


def context_free_document_names(
    rules: Iterable[Dict[str, Any]],
) -> List[str]:
    """返回上下文无关分支实际依赖的文书类型。"""

    return _unique_document_names(
        document_name
        for rule in rules
        for document_name in rule["上下文无关审查事项"]
    )


def context_sensitive_document_names(
    rules: Iterable[Dict[str, Any]],
) -> List[str]:
    """返回上下文相关分支实际依赖的文书类型。"""

    return _unique_document_names(
        document_name
        for rule in rules
        for context_item in rule["上下文相关审查事项"]
        for document_name, field_items in context_item["字段"].items()
        if field_items
    )


def human_support_document_names(
    rules: Iterable[Dict[str, Any]],
) -> List[str]:
    """返回规则级检索增强需要逐文书处理的文书类型。"""

    return _unique_document_names(
        document_name
        for rule in rules
        if retrieval_enhancement_module_names(rule)
        for document_name in rule["上下文无关审查事项"]
    )


def retrieval_enhancement_module_names(
    rule: Dict[str, Any],
) -> List[str]:
    """读取规则级检索增强模块；未配置或列表为空时不执行。"""

    configured = rule.get("检索增强") or []
    if not isinstance(configured, list):
        return []
    return list(
        dict.fromkeys(
            name.strip()
            for name in configured
            if isinstance(name, str) and name.strip()
        )
    )


def filter_context_free_rules(
    rules: List[Dict[str, Any]],
    document_presence: Dict[str, bool],
) -> List[Dict[str, Any]]:
    """为 ContextFree 物化只含可用单文书事项的独立规则集。"""

    filtered_rules: List[Dict[str, Any]] = []
    for rule in rules:
        available_documents = {
            document_name: copy.deepcopy(review_item)
            for document_name, review_item in rule[
                "上下文无关审查事项"
            ].items()
            if document_presence.get(document_name) is True
        }
        if not available_documents:
            continue
        filtered_rule = copy.deepcopy(rule)
        filtered_rule["上下文无关审查事项"] = available_documents
        filtered_rules.append(filtered_rule)
    return filtered_rules


def filter_context_sensitive_rules(
    rules: List[Dict[str, Any]],
    document_presence: Dict[str, bool],
) -> List[Dict[str, Any]]:
    """为 ContextSensitive 物化只含可用跨文书事项的独立规则集。"""

    filtered_rules: List[Dict[str, Any]] = []
    for rule in rules:
        available_items = []
        for context_item in rule["上下文相关审查事项"]:
            available_fields = {
                document_name: copy.deepcopy(field_items)
                for document_name, field_items in context_item["字段"].items()
                if document_presence.get(document_name) is True
                and field_items
            }
            if not available_fields:
                continue
            filtered_item = copy.deepcopy(context_item)
            filtered_item["字段"] = available_fields
            available_items.append(filtered_item)
        if not available_items:
            continue
        filtered_rule = copy.deepcopy(rule)
        filtered_rule["上下文相关审查事项"] = available_items
        filtered_rules.append(filtered_rule)
    return filtered_rules


def filter_human_support_rules(
    rules: List[Dict[str, Any]],
    document_presence: Dict[str, bool],
) -> List[Dict[str, Any]]:
    """独立筛选配置了检索增强且至少有一份可用文书的规则。"""

    filtered_rules: List[Dict[str, Any]] = []
    for rule in rules:
        if not retrieval_enhancement_module_names(rule):
            continue
        available_documents = {
            document_name: copy.deepcopy(review_item)
            for document_name, review_item in rule[
                "上下文无关审查事项"
            ].items()
            if document_presence.get(document_name) is True
        }
        if not available_documents:
            continue
        filtered_rule = copy.deepcopy(rule)
        filtered_rule["上下文无关审查事项"] = available_documents
        filtered_rules.append(filtered_rule)
    return filtered_rules


def _select_rules(
    rules: Dict[str, List[Dict[str, Any]]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    doc_type = config["document_type"]
    selected_rules = []

    match doc_type:
        case DocumentType.ADMIN_INSPECTION:
            selected_rules = rules["行政检查"]

        case DocumentType.ADMIN_PENALTY:
            penalty_rules = rules["行政处罚"]
            sub_type = config["penalty_procedure"]
            if sub_type == PenaltyProcedure.SIMPLE:
                procedure = "简易程序"
            elif sub_type == PenaltyProcedure.ORDINARY:
                procedure = "普通程序"
            selected_rules = [
                item
                for item in penalty_rules
                if item["备注"] in ("", procedure)
            ]

        case DocumentType.ADMIN_ENFORCEMENT:
            enforcement_rules = rules["行政强制"]
            applicable_types = []
            if config["use_coercive_measure"]:
                applicable_types.append("行政强制措施")
            if config["use_admin_enforcement"]:
                applicable_types.append("行政机关强制执行")
            if config["use_court_enforcement"]:
                applicable_types.append("申请人民法院强制执行")

            selected_rules = [
                item
                for item in enforcement_rules
                if item["备注"] == "" or item["备注"] in applicable_types
            ]

    return selected_rules
