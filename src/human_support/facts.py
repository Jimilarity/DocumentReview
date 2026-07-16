import asyncio
import logging
from typing import Any, Protocol

from knowledge_retrieval.common import extract_legal_citations

from .models import FactCollection, HumanSupportContext


PENALTY_LEGAL_CITATIONS_FACT = "penalty_legal_citations"
IMPOSED_PENALTIES_FACT = "imposed_penalties"
CASE_REASON_FACT = "case_reason"


class FactProvider(Protocol):
    provided_facts: frozenset[str]

    async def extract(
        self,
        context: HumanSupportContext,
    ) -> dict[str, Any]:
        ...


class FactProviderRegistry:
    """按事实名称映射提供器，同一提供器在一个 section 中只调用一次。"""

    def __init__(self) -> None:
        self._provider_by_fact: dict[str, FactProvider] = {}

    def register(self, provider: FactProvider) -> None:
        for fact_name in provider.provided_facts:
            if fact_name in self._provider_by_fact:
                raise ValueError(f"人工辅助事实重复注册: {fact_name}")
            self._provider_by_fact[fact_name] = provider

    def provider_for(self, fact_name: str) -> FactProvider:
        try:
            return self._provider_by_fact[fact_name]
        except KeyError as exc:
            raise KeyError(f"人工辅助事实未注册: {fact_name}") from exc


class PenaltyLegalFactsProvider:
    provided_facts = frozenset(
        {
            PENALTY_LEGAL_CITATIONS_FACT,
            IMPOSED_PENALTIES_FACT,
        }
    )

    async def extract(
        self,
        context: HumanSupportContext,
    ) -> dict[str, Any]:
        extraction = await extract_legal_citations(
            context.section_ocr,
            raise_on_error=True,
        )
        return {
            PENALTY_LEGAL_CITATIONS_FACT: list(extraction.citations),
            IMPOSED_PENALTIES_FACT: list(extraction.imposed_penalties),
        }


class CaseReasonFactProvider:
    provided_facts = frozenset({CASE_REASON_FACT})

    async def extract(
        self,
        context: HumanSupportContext,
    ) -> dict[str, Any]:
        value = context.metadata.get("案由")
        case_reason = (
            value.strip()
            if isinstance(value, str) and value.strip()
            else None
        )
        return {CASE_REASON_FACT: case_reason}


fact_provider_registry = FactProviderRegistry()
fact_provider_registry.register(PenaltyLegalFactsProvider())
fact_provider_registry.register(CaseReasonFactProvider())


class FactExtractionService:
    def __init__(
        self,
        registry: FactProviderRegistry | None = None,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.registry = registry or fact_provider_registry
        self.logger = logger or logging.getLogger(__name__)

    async def collect(
        self,
        required_facts: set[str],
        context: HumanSupportContext,
    ) -> FactCollection:
        providers: list[FactProvider] = []
        provider_fact_names: dict[int, set[str]] = {}
        errors: dict[str, str] = {}
        for fact_name in sorted(required_facts):
            try:
                provider = self.registry.provider_for(fact_name)
            except KeyError as exc:
                errors[fact_name] = str(exc)
                continue
            provider_key = id(provider)
            provider_fact_names.setdefault(provider_key, set()).add(fact_name)
            if all(id(existing) != provider_key for existing in providers):
                providers.append(provider)

        async def invoke(
            provider: FactProvider,
        ) -> tuple[FactProvider, dict[str, Any] | None, Exception | None]:
            try:
                return provider, await provider.extract(context), None
            except Exception as exc:
                return provider, None, exc

        values: dict[str, Any] = {}
        groups = await asyncio.gather(*(invoke(item) for item in providers))
        for provider, extracted, exc in groups:
            requested = provider_fact_names[id(provider)]
            if exc is not None:
                message = f"{type(exc).__name__}: {exc}"
                for fact_name in requested:
                    errors[fact_name] = message
                self.logger.warning(
                    "human support fact extraction failed; facts=%s "
                    "error=%s",
                    sorted(requested),
                    message,
                )
                continue
            assert extracted is not None
            for fact_name in requested:
                if fact_name in extracted:
                    values[fact_name] = extracted[fact_name]
                else:
                    errors[fact_name] = "事实提供器未返回声明的字段"
        return FactCollection(values=values, errors=errors)
