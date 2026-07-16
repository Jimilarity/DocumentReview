import re
from enum import Enum
from typing import Any


class PenaltyCaseKind(str, Enum):
    SUBDISTRICT = "subdistrict"
    EMERGENCY = "emergency"
    FIRE_RESCUE = "fire_rescue"
    MARKET_REGULATION = "market_regulation"
    OTHER = "other"


def classify_penalty_case(case_number: Any) -> PenaltyCaseKind:
    """依据案号用语做最低限度的案件来源分流。"""

    if not isinstance(case_number, str):
        return PenaltyCaseKind.OTHER
    compact = re.sub(r"\s+", "", case_number)
    if re.search(r"综行[^号字]{0,4}罚", compact):
        return PenaltyCaseKind.SUBDISTRICT
    if "应急罚" in compact:
        return PenaltyCaseKind.EMERGENCY
    if "消防罚" in compact:
        return PenaltyCaseKind.FIRE_RESCUE
    if re.search(r"市监[^号字]{0,4}罚", compact):
        return PenaltyCaseKind.MARKET_REGULATION
    return PenaltyCaseKind.OTHER
