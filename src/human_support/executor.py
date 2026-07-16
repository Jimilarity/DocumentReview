import asyncio
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from cache_paths import get_cache_paths
from directory_info import normalize_directory_info
from errors.handler import LOGGER
from reviewers.base import ReviewSettings
from reviewers.section_content import extract_section_ocr_text
from rules.filtering import retrieval_enhancement_module_names
from rules.rule_set import RuleSet
from utils import async_read_json

from .facts import FactExtractionService
from .models import (
    HumanSupportContext,
    HumanSupportRetrieval,
    RetrievalStatus,
)
from .service import HumanSupportService


def _serializable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {
            str(key): _serializable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


class HumanSupportExecutor:
    """逐 section 准备人工复核信息，不生成审查问题或结论。"""

    def __init__(
        self,
        file_path: str | Path,
        rule_set: RuleSet,
        *,
        document_section_map: dict[str, list[int]] | None = None,
        settings: ReviewSettings | None = None,
        fact_service: FactExtractionService | None = None,
        support_service: HumanSupportService | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.rule_set = rule_set
        self.document_section_map = {
            name: list(section_ids)
            for name, section_ids in (document_section_map or {}).items()
        }
        self.settings = settings or ReviewSettings.from_env()
        self.logger = logger or LOGGER
        self.cache_paths = get_cache_paths(self.file_path)
        self.fact_service = fact_service or FactExtractionService(
            logger=self.logger
        )
        self.support_service = support_service or HumanSupportService(
            logger=self.logger
        )

    @property
    def rules(self) -> list[dict[str, Any]]:
        return self.rule_set.rules

    def executable_rule_indexes(self) -> list[int]:
        return [
            index
            for index, rule in enumerate(self.rules)
            if retrieval_enhancement_module_names(rule)
            and rule["上下文无关审查事项"]
        ]

    async def _load_context(
        self,
    ) -> tuple[
        dict[str, Any],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        meta_info, raw_dir_info, ocr_results = await asyncio.gather(
            async_read_json(self.cache_paths.metadata),
            async_read_json(self.cache_paths.directory),
            async_read_json(self.cache_paths.ocr_results),
        )
        return (
            meta_info,
            normalize_directory_info(raw_dir_info),
            ocr_results,
        )

    def _build_jobs(self) -> dict[
        tuple[str, int],
        list[tuple[dict[str, Any], dict[str, Any]]],
    ]:
        jobs: dict[
            tuple[str, int],
            list[tuple[dict[str, Any], dict[str, Any]]],
        ] = defaultdict(list)
        for index in self.executable_rule_indexes():
            rule = self.rules[index]
            for document_name, review_item in rule[
                "上下文无关审查事项"
            ].items():
                for section_id in self.document_section_map.get(
                    document_name,
                    [],
                ):
                    jobs[(document_name, section_id)].append(
                        (rule, review_item)
                    )
        return dict(jobs)

    @staticmethod
    def _retrieval_error(
        module_name: str,
        reason: str,
    ) -> HumanSupportRetrieval:
        return HumanSupportRetrieval(
            knowledge_name=module_name,
            status=RetrievalStatus.ERROR,
            payload={"reason": reason},
        )

    async def _process_section(
        self,
        *,
        document_name: str,
        section_id: int,
        assignments: list[tuple[dict[str, Any], dict[str, Any]]],
        metadata: dict[str, Any],
        dir_info: list[dict[str, Any]],
        ocr_results: list[dict[str, Any]],
    ) -> dict[int, dict[str, Any]]:
        section_ocr = extract_section_ocr_text(
            section_id,
            dir_info,
            ocr_results,
        )
        resolved: list[
            tuple[
                dict[str, Any],
                HumanSupportContext,
                list[Any],
                list[HumanSupportRetrieval],
            ]
        ] = []
        all_required_facts: set[str] = set()
        required_by_rule: dict[int, set[str]] = defaultdict(set)
        for rule, review_item in assignments:
            rule_index = int(rule["序号"])
            context = HumanSupportContext(
                metadata=metadata,
                dir_info=dir_info,
                rule=rule,
                review_item=review_item,
                document_name=document_name,
                section_id=section_id,
                section_ocr=section_ocr,
            )
            module_names = retrieval_enhancement_module_names(rule)
            modules, configuration_errors = (
                self.support_service.resolve_modules(module_names)
            )
            for module in modules:
                required = module.required_facts(context)
                required_by_rule[rule_index].update(required)
                all_required_facts.update(required)
            resolved.append(
                (rule, context, modules, configuration_errors)
            )

        fact_context = resolved[0][1]
        fact_collection = await self.fact_service.collect(
            all_required_facts,
            fact_context,
        )
        output: dict[int, dict[str, Any]] = {}
        for rule, context, modules, configuration_errors in resolved:
            rule_index = int(rule["序号"])
            retrievals = list(configuration_errors)
            for module in modules:
                required = module.required_facts(context)
                failed_facts = {
                    name: fact_collection.errors[name]
                    for name in required
                    if name in fact_collection.errors
                }
                if failed_facts:
                    retrievals.append(
                        self._retrieval_error(
                            module.name,
                            (
                                "required_fact_extraction_failed: "
                                f"{failed_facts}"
                            ),
                        )
                    )
                    continue
                try:
                    retrievals.append(
                        await module.retrieve(
                            context,
                            fact_collection.values,
                        )
                    )
                except Exception as exc:
                    self.logger.warning(
                        "human support retrieval failed; module=%s "
                        "rule=%s section=%s error=%s: %s",
                        module.name,
                        rule_index,
                        section_id,
                        type(exc).__name__,
                        exc,
                    )
                    retrievals.append(
                        self._retrieval_error(
                            module.name,
                            f"{type(exc).__name__}: {exc}",
                        )
                    )

            relevant_facts = {
                name: _serializable(fact_collection.values[name])
                for name in sorted(required_by_rule[rule_index])
                if name in fact_collection.values
            }
            warnings = [
                f"{name}: {message}"
                for name, message in fact_collection.errors.items()
                if name in required_by_rule[rule_index]
            ]
            document_result = {
                "document_name": document_name,
                "section_id": section_id,
                "extracted_facts": relevant_facts,
                "retrievals": [item.to_dict() for item in retrievals],
            }
            if warnings:
                document_result["warnings"] = warnings
            output[rule_index] = document_result
        return output

    async def execute(self) -> list[dict[str, Any]]:
        indexes = self.executable_rule_indexes()
        if not indexes:
            return []
        metadata, dir_info, ocr_results = await self._load_context()
        jobs = self._build_jobs()
        semaphore = asyncio.Semaphore(
            max(1, self.settings.model_max_concurrency)
        )

        async def run_job(
            key: tuple[str, int],
            assignments: list[tuple[dict[str, Any], dict[str, Any]]],
        ) -> dict[int, dict[str, Any]]:
            async with semaphore:
                try:
                    return await asyncio.wait_for(
                        self._process_section(
                            document_name=key[0],
                            section_id=key[1],
                            assignments=assignments,
                            metadata=metadata,
                            dir_info=dir_info,
                            ocr_results=ocr_results,
                        ),
                        timeout=self.settings.task_timeout_seconds,
                    )
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    return {
                        int(rule["序号"]): {
                            "document_name": key[0],
                            "section_id": key[1],
                            "extracted_facts": {},
                            "retrievals": [
                                self._retrieval_error(
                                    module_name,
                                    message,
                                ).to_dict()
                                for module_name in (
                                    retrieval_enhancement_module_names(rule)
                                )
                            ],
                            "warnings": [message],
                        }
                        for rule, _ in assignments
                    }

        section_groups = await asyncio.gather(
            *(run_job(key, assignments) for key, assignments in jobs.items())
        )
        documents_by_rule: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for section_group in section_groups:
            for rule_index, result in section_group.items():
                documents_by_rule[rule_index].append(result)

        results: list[dict[str, Any]] = []
        for index in indexes:
            rule = self.rules[index]
            rule_index = int(rule["序号"])
            documents = documents_by_rule.get(rule_index, [])
            results.append(
                {
                    "rule_index": rule_index,
                    "documents": documents,
                }
            )
        return results
