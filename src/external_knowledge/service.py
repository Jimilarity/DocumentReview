import asyncio
import logging
from collections.abc import Sequence

from .models import KnowledgeContext, KnowledgeItem
from .registry import KnowledgeRegistry, knowledge_registry
from .tracing import trace_event


# 外部知识已配置但本次未能获取到内容时，注入给审查模型的保守性说明。
# 目的是：仍然依据规则自身的明确要求完成审查，仅停止依赖缺失的补充知识
# 才能完成的判断，并禁止模型用记忆/常识补造法条或标准。
KNOWLEDGE_UNAVAILABLE_NOTE = (
    "注意：本规则声明了外部知识（如法条检索、裁量基准等），但本次审查未能成功获取到"
    "外部知识内容（知识函数调用失败或未返回可用知识）。当前规则自身已经明确写明的期限、"
    "金额、条件、字段和判断标准仍是有效且应当直接适用的审查依据，不得因为外部知识失败而"
    "跳过；案卷字段和原始文书能够直接证明的问题也应正常输出。只有必须依赖本次缺失的补充"
    "法条、裁量幅度、例外或特别规定才能成立的结论，才不得作出。不得使用模型自身记忆、"
    "法律常识或自行检索的信息补造法条名称、条款内容、金额标准或其他规范性要求。"
)


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
