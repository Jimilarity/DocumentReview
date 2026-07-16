from collections import defaultdict
from typing import Any, Dict, List


def aggregate_section_results(
    rule_index: Any,
    section_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not section_results:
        raise ValueError(f"规则 {rule_index} 没有可汇总的 section 结果")

    aggregated = {
        "rule_index": rule_index,
        "issues": [
            issue
            for result in section_results
            for issue in result["issues"]
        ],
    }
    failed_results = [
        result for result in section_results if result.get("error")
    ]
    if failed_results:
        aggregated["error"] = "section_review_failed"
        aggregated["error_details"] = [
            result.get("error_details")
            for result in failed_results
        ]
    return aggregated


def aggregate_component_results(
    rule_index: Any,
    component_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """汇总同一规则的多个审查分项结果。"""

    if not component_results:
        raise ValueError(f"规则 {rule_index} 没有可汇总的审查分项")
    aggregated = {
        "rule_index": rule_index,
        "issues": [
            issue
            for result in component_results
            for issue in result["issues"]
        ],
    }
    failed_results = [
        result for result in component_results if result.get("error")
    ]
    if failed_results:
        aggregated["error"] = "component_review_failed"
        aggregated["error_details"] = [
            result.get("error_details") for result in failed_results
        ]
    return aggregated


def merge_rule_results(
    *result_groups: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """按 rule_index 合并不同审查器对同一规则产生的正式结果。"""

    by_rule: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for group in result_groups:
        for result in group:
            by_rule[result["rule_index"]].append(result)
    merged = [
        results[0]
        if len(results) == 1
        else aggregate_component_results(rule_index, results)
        for rule_index, results in by_rule.items()
    ]
    merged.sort(key=lambda item: item.get("rule_index", 0))
    return merged
