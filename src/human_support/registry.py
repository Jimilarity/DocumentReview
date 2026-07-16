from typing import Protocol

from .models import (
    HumanSupportContext,
    HumanSupportRetrieval,
)


class HumanSupportModule(Protocol):
    name: str

    def required_facts(
        self,
        context: HumanSupportContext,
    ) -> set[str]:
        ...

    async def retrieve(
        self,
        context: HumanSupportContext,
        facts: dict[str, object],
    ) -> HumanSupportRetrieval:
        ...


class HumanSupportModuleRegistry:
    """以规则文件中的稳定名称管理人工辅助模块。"""

    def __init__(self) -> None:
        self._modules: dict[str, HumanSupportModule] = {}

    def register(self, module: HumanSupportModule) -> None:
        name = module.name.strip()
        if not name:
            raise ValueError("人工辅助模块名称不能为空")
        if name in self._modules:
            raise ValueError(f"人工辅助模块重复注册: {name}")
        self._modules[name] = module

    def get(self, name: str) -> HumanSupportModule:
        try:
            return self._modules[name]
        except KeyError as exc:
            raise KeyError(f"人工辅助模块未注册: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(self._modules)


human_support_module_registry = HumanSupportModuleRegistry()
