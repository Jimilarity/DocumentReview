import json

from ..models import KnowledgeContext, KnowledgeItem
from ..registry import register_knowledge


@register_knowledge("case_directory_info")
async def provide_case_directory_info(
    context: KnowledgeContext,
) -> list[KnowledgeItem]:
    """把完整案卷目录提供给显式配置该知识的审查事项。"""

    if not context.dir_info:
        return []
    directory_json = json.dumps(
        context.dir_info,
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    return [
        KnowledgeItem(
            content=f"本案案卷目录结构：\n{directory_json}"
        )
    ]
