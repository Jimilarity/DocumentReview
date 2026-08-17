import asyncio
import logging
from collections.abc import Sequence

from .models import KnowledgeContext, KnowledgeItem
from .registry import KnowledgeRegistry, knowledge_registry
from .tracing import trace_event


class KnowledgeService:
    """调用审查事项声明的知识函数并汇总非空知识。"""

    def __init__(
        self,
        registry: KnowledgeRegistry | None = None,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.registry = registry or knowledge_registry
        self.logger = logger or logging.getLogger(__name__)

    def _normalize_items(
        self,
        function_name: str,
        result: Sequence[KnowledgeItem],
    ) -> list[KnowledgeItem]:
        if isinstance(result, (str, bytes)) or not isinstance(
            result,
            Sequence,
        ):
            self.logger.warning(
                "knowledge function returned a non-sequence; skipped "
                "name=%s",
                function_name,
            )
            return []

        items: list[KnowledgeItem] = []
        for item in result:
            if not isinstance(item, KnowledgeItem):
                self.logger.warning(
                    "knowledge function returned an invalid item; skipped "
                    "name=%s",
                    function_name,
                )
                continue
            content = item.content.strip()
            if content:
                items.append(KnowledgeItem(content=content))
        return items

    async def _invoke(
        self,
        function_name: str,
        context: KnowledgeContext,
    ) -> list[KnowledgeItem]:
        function = self.registry.get(function_name)
        trace_event(
            "external_knowledge_start",
            function_name=function_name,
        )
        try:
            result = await function(context)
            items = self._normalize_items(function_name, result)
            trace_event(
                "external_knowledge_end",
                function_name=function_name,
                item_count=len(items),
            )
            return items
        except Exception as exc:
            trace_event(
                "external_knowledge_error",
                function_name=function_name,
                exception_type=type(exc).__name__,
                message=str(exc),
            )
            self.logger.warning(
                "knowledge function failed; skipped name=%s error=%s: %s",
                function_name,
                type(exc).__name__,
                exc,
            )
            return []

    async def collect(
        self,
        function_names: Sequence[str],
        context: KnowledgeContext,
    ) -> list[KnowledgeItem]:
        names = list(dict.fromkeys(function_names))
        if not names:
            return []

        # 先统一解析名称，使配置拼写错误能够立即暴露，而不是被当成知识未命中。
        for name in names:
            self.registry.get(name)

        groups = await asyncio.gather(
            *(self._invoke(name, context) for name in names)
        )
        unique_items: list[KnowledgeItem] = []
        seen_contents: set[str] = set()
        for group in groups:
            for item in group:
                if item.content in seen_contents:
                    continue
                seen_contents.add(item.content)
                unique_items.append(item)
        return unique_items
