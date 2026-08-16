"""只验证规则 113 的法条检索请求与候选结果，不运行大模型审查。"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from cache_paths import get_cache_paths
from constants import PROJECT_ROOT, RULES_PATH
from external_knowledge import KnowledgeContext
from external_knowledge.legal_citation_validity import knowledge
from utils import read_json


RULE_ID = 113


def _rule_113() -> dict[str, Any]:
    rules = read_json(RULES_PATH)
    for category in rules.values():
        for group in category.values():
            for rule in group:
                if rule.get("序号") == RULE_ID:
                    return rule
    raise LookupError("all_rules.json 中找不到规则 113")


def _context_item(rule: dict[str, Any]) -> dict[str, Any]:
    items = rule.get("上下文相关审查事项", [])
    for item in items:
        if item.get("任务") == "不予处罚合法性审查":
            return item
    raise LookupError("规则 113 中找不到不予处罚合法性审查事项")


def _field_values(
    cache: dict[str, Any],
    context_item: dict[str, Any],
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    sections = cache.get("sections", {})
    section_map = cache.get("document_section_map", {})
    for document_name, field_items in context_item["字段"].items():
        for section_id in section_map.get(document_name, []):
            section = sections.get(str(section_id), {})
            for item in field_items:
                field_name = item["field"]
                value = section.get(field_name)
                if value is not None:
                    values[field_name] = value
    return values


async def inspect(pdf_path: str) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    cache_paths = get_cache_paths(pdf_path)
    metadata = read_json(cache_paths.metadata)
    directory = read_json(cache_paths.directory)
    field_cache = read_json(cache_paths.structured_fields)
    rule = _rule_113()
    context_item = _context_item(rule)
    context = KnowledgeContext(
        metadata={**metadata, **_field_values(field_cache, context_item)},
        dir_info=directory,
        rule=rule,
        review_item=context_item,
        document_name="",
        section_id=0,
        section_ocr="",
    )
    config = knowledge._query_config(context)
    payload: dict[str, Any] = {
        "text": knowledge._query_text(context, config),
        "top_k": min(knowledge._positive_int(config, "top_k", 3), 50),
        "version_k": min(knowledge._positive_int(config, "version_k", 7), 30),
        "with_text": True,
        "with_focus": True,
    }
    as_of = knowledge._as_of(context.metadata, config)
    if as_of:
        payload["as_of"] = as_of
    items = await knowledge.retrieve_legal_citation_validity(context)
    return {
        "rule_id": RULE_ID,
        "request": payload,
        "candidate_count": len(items),
        "candidates": [item.content for item in items],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-path", required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(inspect(args.file_path)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
