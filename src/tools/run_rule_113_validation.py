"""只运行规则 113，并记录其实际发出的法条检索请求。"""

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
from external_knowledge.legal_citation_validity import knowledge
from reviewers.context_sensitive import ContextSensitiveReviewExecutor
from rules.filtering import filter_context_sensitive_rules
from rules.rule_set import RuleSet
from utils import read_json


RULE_ID = 113
RULE_TYPE = 0b11110000


def _load_rule() -> dict[str, Any]:
    for category in read_json(RULES_PATH).values():
        for group in category.values():
            for rule in group:
                if rule.get("序号") == RULE_ID:
                    return rule
    raise LookupError(f"{RULES_PATH.name} 中找不到规则 113")


async def validate(pdf_path: str) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    requests: list[dict[str, Any]] = []
    original_request = knowledge._request

    def record_request(payload: dict[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {"request": payload}
        requests.append(record)
        response = original_request(payload)
        candidates = response.get("candidates", [])
        record["response_count"] = len(candidates) if isinstance(candidates, list) else None
        record["laws"] = [
            candidate.get("law_name")
            for candidate in candidates
            if isinstance(candidate, dict) and isinstance(candidate.get("law_name"), str)
        ]
        return response

    knowledge._request = record_request
    try:
        cache_paths = get_cache_paths(pdf_path)
        field_cache = read_json(cache_paths.structured_fields)
        source_rule = _load_rule()
        rule_set = RuleSet(
            rules=filter_context_sensitive_rules(
                [source_rule],
                field_cache["document_presence"],
            )
        )
        if not rule_set.rules:
            raise RuntimeError("规则 113 所需文书均未映射，无法验证")
        document_section_map = {
            name: field_cache["document_section_map"][name]
            for name in rule_set.rules[0]["上下文相关审查事项"][0]["字段"]
        }
        executor = ContextSensitiveReviewExecutor(
            file_path=pdf_path,
            rule_set=rule_set,
            document_section_map=document_section_map,
        )
        raw_results = await executor.execute_raw()
    finally:
        knowledge._request = original_request
    if len(requests) != 1 or "response_count" not in requests[0]:
        raise RuntimeError(
            "规则 113 未成功完成一次法条 API 调用；请检查 .env 中的 "
            "LAW_RETRIEVAL_API_KEY 和网络连接。"
        )
    return {
        "rule_id": RULE_ID,
        "law_api_calls": requests,
        "review": raw_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-path", required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(validate(args.file_path)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
