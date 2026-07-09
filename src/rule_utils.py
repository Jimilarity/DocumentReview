import json
import re

from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
env_path = Path(__file__).parents[1]
load_dotenv(env_path)

from constants import DocumentType, PenaltyProcedure, OPTIONAL_STANDARDIZATION_PENALTY_ORDINARY
from model_config import build_text_model
from utils import read_json
from constants import RuleCategory, DocumentType, InspectionRuleType, PenaltyProcedure, RULES_PATH, ErrorCode
from error import ReviewError
from constants import (
    CATEGORY_MASK,
    DOCUMENT_TYPE_MASK,
    SUBTYPE_MASK,
    RuleCategory,
    DocumentType,
    InspectionRuleType,
    PenaltyProcedure,
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
            "必须选择合法性、规范性或加减分项"
        )

    document_type = (rule_type & DOCUMENT_TYPE_MASK) >> 3
    if document_type == 0:
        raise ValueError(
            "rule_type 的文书类型不能为 00"
        )


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
        "use_legality": bool(
            categories & RuleCategory.LEGALITY
        ),
        "use_standardization": bool(
            categories & RuleCategory.STANDARDIZATION
        ),
        "use_score_adjustment": bool(
            categories & RuleCategory.SCORE_ADJUSTMENT
        ),
        "document_type": document_type,
    }

    if document_type == DocumentType.ADMIN_INSPECTION:
        inspection_types = InspectionRuleType(subtype)

        result["inspection_types"] = inspection_types
        result["use_coercive_measure"] = bool(
            inspection_types
            & InspectionRuleType.COERCIVE_MEASURE
        )
        result["use_admin_enforcement"] = bool(
            inspection_types
            & InspectionRuleType.ADMIN_ENFORCEMENT
        )
        result["use_court_enforcement"] = bool(
            inspection_types
            & InspectionRuleType.COURT_ENFORCEMENT
        )

    elif document_type == DocumentType.ADMIN_PENALTY:
        procedure_bit = (subtype >> 2) & 1

        result["penalty_procedure"] = (
            PenaltyProcedure(procedure_bit)
        )

    return result

def check_optional_docs(
    optional_docs: List[str],
    dir_info: List[Dict[str, Any]],
) -> Dict[str, bool]:
    llm = build_text_model()

    json_llm = llm.bind(
        response_format={"type": "json_object"},
        extra_body={
            "enable_thinking": False
        }
    )

    system_prompt = """

判断案卷目录中是否包含指定文书。

判断规则：
1. 不要求目录名称与标准文书名称完全一致，应根据语义判断。
2. 文书名称中的全角或半角括号、空格、换行和轻微表述差异不影响判断。
3. 合并名称也可以视为包含。例如：
   - “现场检查笔录”可以匹配“现场检查（勘验）笔录”；
   - “勘验笔录”可以匹配“现场检查（勘验）笔录”；
   - “先行登记保存证据决定书”可以匹配
     “先行登记保存证据通知书（决定书）”；
   - “责令改正通知书”可以匹配“责令改正违法行为通知书”。
4. 每个待检查文书都必须出现在输出中。
5. 输出的键必须与输入的标准文书名称完全一致。
6. 只能输出一个 JSON 对象，不要输出解释、Markdown 或其他内容。
7. JSON 的值只能是 true 或 false。

输出示例：
{
  "现场检查（勘验）笔录": true,
  "调查询问笔录": false
}
""".strip()

    human_prompt = f"""
待检查文书：

{json.dumps(optional_docs, ensure_ascii=False, indent=2)}

案卷目录：

{json.dumps(dir_info, ensure_ascii=False, indent=2)}

请判断案卷目录中是否包含每一项待检查文书。
""".strip()

    response = json_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
    )

    llm_result = json.loads(response.content)

    return {
        doc_name: llm_result.get(doc_name) is True
        for doc_name in optional_docs
    }


def _filter_rules_by_exist(
    rules: List[Dict[str, Any]], 
    config: Dict[str, Any],
    rule_type: RuleCategory,
    dir_info: List[Dict[str, Any]]
    ):
    doc_type = config['document_type']
    filtered_rules = rules
    if rule_type == RuleCategory.STANDARDIZATION and doc_type ==DocumentType.ADMIN_PENALTY:
        sub_type = config['penalty_procedure']
        if sub_type == PenaltyProcedure.ORDINARY:
            files_exist = check_optional_docs(OPTIONAL_STANDARDIZATION_PENALTY_ORDINARY, dir_info)
            if set(files_exist.keys()) != set(OPTIONAL_STANDARDIZATION_PENALTY_ORDINARY):
                raise ReviewError(
                    ErrorCode.UNEXPECTED_ERROR,
                    f'''
                        Source: rule_utils / _filter_rules_by_exist,
                        Reason: files_exist.keys are not equal to OPTIONAL_STANDARDIZATION_PENALTY_ORDINARY
                        files_exist.keys: {files_exist} 
                        OPTIONAL_STANDARDIZATION_PENALTY_ORDINARY: {OPTIONAL_STANDARDIZATION_PENALTY_ORDINARY}
                    ''',
                )
            filtered_rules = list(filter(
                lambda item: files_exist.get(
                    re.sub(r'[（(]\s*\d+(?:\.\d+)?\s*分\s*[）)]', '', item['评查类别']).strip(),
                    False
                ),
                rules
            ))
    return filtered_rules

def filter_rules(
    rules: Dict[str, List[Dict[str, Any]]],
    config: Dict[str, Any],
    rule_type: RuleCategory,
    dir_info: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
    doc_type = config['document_type']
    rule_list = []
    match doc_type:
        case DocumentType.ADMIN_INSPECTION:
            rule_list += rules['行政检查']
        case DocumentType.ADMIN_PENALTY:
            penalty_rules = rules['行政处罚']
            sub_type = config['penalty_procedure']
            if sub_type == PenaltyProcedure.SIMPLE:
                rule_list += [item for item in penalty_rules if item['备注'] == '简易程序']
            elif sub_type == PenaltyProcedure.ORDINARY:
                ordinary_rules = [item for item in penalty_rules if item['备注'] == '普通程序']
                if rule_type == RuleCategory.STANDARDIZATION:
                    rule_list += _filter_rules_by_exist(
                        ordinary_rules,
                        config=config,
                        rule_type=rule_type,
                        dir_info=dir_info,
                    )
            rule_list += [item for item in penalty_rules if item['备注'] == '']
        case DocumentType.ADMIN_ENFORCEMENT:
            enforce_rules = rules['行政强制']
            if config['use_coercive_measure']:
                rule_list += [item for item in enforce_rules if item['备注'] == '行政强制措施']
            if config['use_admin_enforcement']:
                rule_list += [item for item in enforce_rules if item['备注'] == '行政机关强制执行']
            if config['use_court_enforcement']:
                rule_list += [item for item in enforce_rules if item['备注'] == '申请人民法院强制执行']
            rule_list += [item for item in enforce_rules if item['备注'] == '']
    return rule_list

def load_rule_list(rule_type: int, dir_path: str | Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    all_rules = read_json(RULES_PATH)
    config = decode_rule_type(rule_type)
    dir_info = read_json(dir_path)
    rule_list = []
    
    if config['use_legality']:
        rules = all_rules['合法性标准']
        rule_list += rules.get('通用', [])
        rule_list += filter_rules(
            rules, 
            config=config, 
            rule_type=RuleCategory.LEGALITY, 
            dir_info=dir_info
        )
        
    if config['use_standardization']:
        rules = all_rules['规范性标准']
        rule_list += rules.get('通用', [])
        rule_list += filter_rules(
            rules, 
            config=config, 
            rule_type=RuleCategory.STANDARDIZATION, 
            dir_info=dir_info
        )
    
    if config['use_score_adjustment']:
        rule_list += all_rules['加减分项']['通用']
    return rule_list, config

if __name__ == '__main__':
    
    dir_path = "D:\\DocumentReview\\review_workspace\\documentReview\\dir_cache\\[2025]深龙华龙华综行城罚字第0002号 当事人裴大润在非指定场地堆放建筑废弃物案\\dir_info.json"
    dir_info = read_json(dir_path)
    results = check_optional_docs(OPTIONAL_STANDARDIZATION_PENALTY_ORDINARY, dir_info)
    print(results)