"""深圳行政诉讼管辖知识函数占位实现。"""

from ..models import KnowledgeContext, KnowledgeItem
from ..registry import register_knowledge


@register_knowledge("shenzhen_administrative_litigation_jurisdiction")
async def retrieve_administrative_litigation_jurisdiction(
    context: KnowledgeContext,
) -> list[KnowledgeItem]:
    """后续填充经核验的深圳行政诉讼管辖知识。"""

    return []
