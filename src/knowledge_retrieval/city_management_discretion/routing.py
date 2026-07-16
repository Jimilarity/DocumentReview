from typing import Any

from ..case_routing import PenaltyCaseKind, classify_penalty_case


def is_applicable_subdistrict_penalty_case(
    case_number: Any,
) -> bool:
    """当前裁量目录仅用于可识别的街道综合行政执法案卷。"""

    return classify_penalty_case(case_number) is PenaltyCaseKind.SUBDISTRICT
