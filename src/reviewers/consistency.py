import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


CONSISTENCY_TASK = "一致性核查"


@dataclass(frozen=True)
class ConsistencySource:
    """一个参与同一上下文相关事项等价判断的结构化字段来源。"""

    document_type: str
    section_id: int
    field_name: str
    required: bool
    value: Any
    related_section_id: int | None = None

    @property
    def source_label(self) -> str:
        label = f"{self.document_type}(section_id={self.section_id}"
        if self.related_section_id is not None:
            label += f", related_section_id={self.related_section_id}"
        return f"{label}).{self.field_name}"


def is_executable_consistency_rule(rule: Dict[str, Any]) -> bool:
    """判断规则中是否存在当前智能体可完整覆盖的上下文相关事项。"""

    context_items = rule.get("上下文相关审查事项")
    configured_items = [
        item
        for item in context_items or []
        if isinstance(item, dict) and item.get("任务")
    ]
    return (
        isinstance(context_items, list)
        and bool(configured_items)
        and all(
            item.get("任务") == CONSISTENCY_TASK
            for item in configured_items
        )
    )


def required_field_issues(
    sources: Iterable[ConsistencySource],
) -> List[Dict[str, Any]]:
    """required 只约束已经存在并映射成功的文书 section。"""

    return [
        {
            "section_ids": [source.section_id],
            "content": f"{source.source_label} 未提取到必填字段。",
        }
        for source in sources
        if source.required and source.value is None
    ]


def comparable_sources(
    sources: Iterable[ConsistencySource],
) -> List[ConsistencySource]:
    """null 表示文书没有可比较值，不参与值的等价判断。"""

    return [source for source in sources if source.value is not None]


def format_source_values(sources: Iterable[ConsistencySource]) -> str:
    return "；".join(
        f"{source.source_label}="
        f"{json.dumps(source.value, ensure_ascii=False)}"
        for source in sources
    )


def aggregate_consistency_results(
    rule: Dict[str, Any],
    item_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not item_results:
        raise ValueError("一致性规则至少需要一个上下文相关审查结果")

    return {
        "rule_index": rule["序号"],
        "issues": [
            issue
            for result in item_results
            for issue in result["issues"]
        ],
    }
