from typing import Any, Dict

from constants import (
    CATEGORY_MASK,
    DOCUMENT_TYPE_MASK,
    SUBTYPE_MASK,
    DocumentType,
    EnforcementRuleType,
    PenaltyProcedure,
    RuleCategory,
)


def parse_rule_type_value(value: str) -> int:
    value = str(value).strip().lower()

    if value.startswith("0b"):
        rule_type = int(value, 2)
    elif len(value) == 8 and set(value) <= {"0", "1"}:
        rule_type = int(value, 2)
    else:
        rule_type = int(value)

    validate_rule_type(rule_type)
    return rule_type


def validate_rule_type(rule_type: int) -> None:
    if not isinstance(rule_type, int):
        raise ValueError("rule_type 必须是整数")

    if not 0 <= rule_type <= 255:
        raise ValueError("rule_type 必须是 0~255 的八位无符号整数")

    categories = rule_type & CATEGORY_MASK
    if categories == 0:
        raise ValueError(
            "rule_type 前三位至少有一位为 1，"
            "必须选择合法性、规范性或附加审查"
        )

    document_type = (rule_type & DOCUMENT_TYPE_MASK) >> 3
    if document_type == 0:
        raise ValueError("rule_type 的文书类型不能为 00")


def decode_rule_type(rule_type: int) -> Dict[str, Any]:
    validate_rule_type(rule_type)

    categories = RuleCategory(rule_type & CATEGORY_MASK)
    document_type = DocumentType(
        (rule_type & DOCUMENT_TYPE_MASK) >> 3
    )
    subtype = rule_type & SUBTYPE_MASK

    result = {
        "rule_type": rule_type,
        "binary": f"{rule_type:08b}",
        "use_legality": bool(categories & RuleCategory.LEGALITY),
        "use_standardization": bool(
            categories & RuleCategory.STANDARDIZATION
        ),
        "use_additional_review": bool(
            categories & RuleCategory.ADDITIONAL_REVIEW
        ),
        "document_type": document_type,
    }

    if document_type == DocumentType.ADMIN_ENFORCEMENT:
        enforcement_types = EnforcementRuleType(subtype)

        result["enforcement_types"] = enforcement_types
        result["use_coercive_measure"] = bool(
            enforcement_types & EnforcementRuleType.COERCIVE_MEASURE
        )
        result["use_admin_enforcement"] = bool(
            enforcement_types & EnforcementRuleType.ADMIN_ENFORCEMENT
        )
        result["use_court_enforcement"] = bool(
            enforcement_types & EnforcementRuleType.COURT_ENFORCEMENT
        )

    elif document_type == DocumentType.ADMIN_PENALTY:
        procedure_bit = (subtype >> 2) & 1
        result["penalty_procedure"] = PenaltyProcedure(procedure_bit)

    return result
