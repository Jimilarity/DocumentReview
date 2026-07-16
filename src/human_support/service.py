import logging

from .models import (
    HumanSupportContext,
    HumanSupportRetrieval,
    RetrievalStatus,
)
from .registry import (
    HumanSupportModule,
    HumanSupportModuleRegistry,
    human_support_module_registry,
)

# 导入即完成内置模块注册。
from . import modules as _modules  # noqa: F401


class HumanSupportService:
    def __init__(
        self,
        registry: HumanSupportModuleRegistry | None = None,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.registry = registry or human_support_module_registry
        self.logger = logger or logging.getLogger(__name__)

    def resolve_modules(
        self,
        names: list[str],
    ) -> tuple[list[HumanSupportModule], list[HumanSupportRetrieval]]:
        modules: list[HumanSupportModule] = []
        errors: list[HumanSupportRetrieval] = []
        for name in dict.fromkeys(names):
            try:
                modules.append(self.registry.get(name))
            except KeyError as exc:
                errors.append(
                    HumanSupportRetrieval(
                        knowledge_name=name,
                        status=RetrievalStatus.ERROR,
                        payload={"reason": str(exc)},
                    )
                )
        return modules, errors
