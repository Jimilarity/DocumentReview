"""法条有效性知识函数占位实现。"""

from ..models import KnowledgeContext, KnowledgeItem
from ..registry import register_knowledge


@register_knowledge("legal_citation_validity")
async def retrieve_legal_citation_validity(
    context: KnowledgeContext,
) -> list[KnowledgeItem]:
    """检索当前 section 所引法律规范及其有效性；后续接入法条 API。"""

    return []
