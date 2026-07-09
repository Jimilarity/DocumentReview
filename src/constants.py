from enum import IntEnum, IntFlag
from pathlib import Path

class ErrorCode(IntEnum):
    PDF_TO_IMAGE_ERROR = 401
    META_DATA_EXTRACTION_ERROR = 402
    DIR_IDENTIFICATION_ERROR = 403
    SECTION_IDENTIFICATION_ERROR = 404
    UNEXPECTED_ERROR = 500
    
class RuleCategory(IntFlag):
    LEGALITY = 0b10000000       # 合法性标准
    STANDARDIZATION = 0b01000000  # 规范性标准
    SCORE_ADJUSTMENT = 0b00100000  # 加减分项


class DocumentType(IntEnum):
    ADMIN_INSPECTION = 0b01     # 行政检查
    ADMIN_PENALTY = 0b10        # 行政处罚
    ADMIN_ENFORCEMENT = 0b11    # 行政强制


class InspectionRuleType(IntFlag):
    COERCIVE_MEASURE = 0b100           # 行政强制措施
    ADMIN_ENFORCEMENT = 0b010          # 行政机关强制执行
    COURT_ENFORCEMENT = 0b001          # 申请人民法院强制执行


class PenaltyProcedure(IntEnum):
    SIMPLE = 0      # 简易程序，0XX
    ORDINARY = 1    # 普通程序，1XX


CATEGORY_MASK = 0b11100000
DOCUMENT_TYPE_MASK = 0b00011000
SUBTYPE_MASK = 0b00000111

RULES_PATH = Path(__file__).resolve().parents[1] / 'data' / 'all_rules.json'

STANDARDIZATION_PENALTY_DOCS = {
    '立案审批表',
    '现场检查',
    '（勘验）笔录',
    '调查询问笔录',
    '证据',
    '先行登记保存证据通知书(决定书)',
    '查封（扣押）决定书',
    '责令改正违法行为决定书(通知书)',
    '行政处罚事先（听证）',
    '告知书',
    '行政处罚听证通知书',
    '听证笔录',
    '听证报告',
    '重大行政处罚决定法制审核意见书',
    '行政机关负责人集体讨论笔录',
    '行政处罚决定书',
    '结案表',
}

OPTIONAL_STANDARDIZATION_PENALTY_ORDINARY = [
    '现场检查（勘验）笔录',
    '调查询问笔录',
    '先行登记保存证据通知书（决定书）',
    '查封（扣押）决定书',
    '责令改正违法行为通知书',
    '行政处罚听证通知书',
    '听证笔录',
    '听证报告',
    '重大行政处罚决定法制审核意见书',
    '行政机关负责人集体讨论笔录',
    '行政处罚决定书',
]

