from enum import Enum, IntEnum, IntFlag
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
    ADDITIONAL_REVIEW = 0b00100000  # 附加审查标准


class ReviewExecutorType(str, Enum):
    """可由审查流水线配置启用的叶子执行器。"""

    CONTEXT_FREE = "context_free"
    CONTEXT_SENSITIVE = "context_sensitive"
    CASE_LEVEL = "case_level"


class SupportExecutorType(str, Enum):
    """不作审查结论、只为人工复核准备信息的流水线执行器。"""

    HUMAN_SUPPORT = "human_support"


class DocumentType(IntEnum):
    ADMIN_INSPECTION = 0b01     # 行政检查
    ADMIN_PENALTY = 0b10        # 行政处罚
    ADMIN_ENFORCEMENT = 0b11    # 行政强制


class EnforcementRuleType(IntFlag):
    COERCIVE_MEASURE = 0b100           # 行政强制措施
    ADMIN_ENFORCEMENT = 0b010          # 行政机关强制执行
    COURT_ENFORCEMENT = 0b001          # 申请人民法院强制执行


class PenaltyProcedure(IntEnum):
    SIMPLE = 0      # 简易程序，0XX
    ORDINARY = 1    # 普通程序，1XX


CATEGORY_MASK = 0b11100000
DOCUMENT_TYPE_MASK = 0b00011000
SUBTYPE_MASK = 0b00000111

# File paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RULES_PATH = PROJECT_ROOT / "data" / "all_rules.json"
CACHE_ROOT = PROJECT_ROOT / "cache"
RESULT_ROOT = PROJECT_ROOT / "results"
REVIEW_RESULT_FILENAME = "review_results.json"
RETRIEVAL_ENHANCEMENT_RESULT_FILENAME = (
    "retrieval_enhancement_results.json"
)
REVIEW_ERROR_REPORT_FILENAME = "review_error.json"
RAW_REVIEW_RESULT_FILENAME = "raw_review_results.json"
SCORED_RAW_REVIEW_RESULT_FILENAME = "raw_review_results_scored.json"
REVIEW_RESULT_PROCESSING_FILENAME = "review_result_processing.json"
STRUCTURED_FIELD_CACHE_FILENAME = "structured_fields.json"
STRUCTURED_FIELD_CACHE_SCHEMA_VERSION = 6
SECTION_FIELDS_PATH = PROJECT_ROOT / "src" / "config" / "section_fields.yaml"
CONTEXT_SENSITIVE_CONFIG_PATH = (
    PROJECT_ROOT / "src" / "config" / "context_sensitive.yaml"
)
CASE_LEVEL_CONFIG_PATH = (
    PROJECT_ROOT / "src" / "config" / "case_level_review.yaml"
)
REVIEW_PIPELINE_CONFIG_PATH = (
    PROJECT_ROOT / "src" / "config" / "review_pipeline.yaml"
)
DOCUMENT_MAPPING_CONFIG_PATH = (
    PROJECT_ROOT / "src" / "config" / "document_mapping.yaml"
)
RULE_ALIASES_CONFIG_PATH = (
    PROJECT_ROOT / "src" / "config" / "rule_aliases.yaml"
)

SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
}
