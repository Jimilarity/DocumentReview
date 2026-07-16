from collections.abc import Awaitable, Callable, Sequence
from typing import Dict

from .models import KnowledgeContext, KnowledgeItem


KnowledgeFunction = Callable[
    [KnowledgeContext],
    Awaitable[Sequence[KnowledgeItem]],
]


class KnowledgeRegistry:
    """以稳定名称注册可插拔知识函数。"""

    def __init__(self) -> None:
        self._functions: Dict[str, KnowledgeFunction] = {}

    def register(
        self,
        name: str,
    ) -> Callable[[KnowledgeFunction], KnowledgeFunction]:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("知识函数名称不能为空")

        def decorator(function: KnowledgeFunction) -> KnowledgeFunction:
            if normalized_name in self._functions:
                raise ValueError(f"知识函数重复注册: {normalized_name}")
            self._functions[normalized_name] = function
            return function

        return decorator

    def get(self, name: str) -> KnowledgeFunction:
        try:
            return self._functions[name]
        except KeyError as exc:
            raise KeyError(f"知识函数未注册: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(self._functions)


knowledge_registry = KnowledgeRegistry()


def register_knowledge(
    name: str,
) -> Callable[[KnowledgeFunction], KnowledgeFunction]:
    """注册到默认知识注册表的便捷装饰器。"""

    return knowledge_registry.register(name)
