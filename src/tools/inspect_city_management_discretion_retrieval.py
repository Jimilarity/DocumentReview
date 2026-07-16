"""查看裁量标准检索增强候选，不运行完整审查。"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
sys.path.insert(0, str(SRC_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from knowledge_retrieval.common import ImposedPenalty, LegalCitation
from human_support.models import HumanSupportContext
from human_support.modules.city_management_discretion import (
    CityManagementDiscretionCandidatesModule,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-number",
        default="深龙华龙华综行简罚决字第0100号",
    )
    parser.add_argument(
        "--case-reason",
        default="未按规定分类投放生活垃圾且拒不改正案",
    )
    parser.add_argument(
        "--law-name",
        default="《深圳市生活垃圾分类管理条例》",
    )
    parser.add_argument("--article", default="66")
    parser.add_argument("--paragraph")
    parser.add_argument("--item")
    parser.add_argument("--subitem")
    parser.add_argument("--penalty-type", default="罚款")
    parser.add_argument("--penalty-target", default="个人")
    parser.add_argument("--penalty-amount", type=int)
    parser.add_argument("--penalty-text")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    citation = LegalCitation(
        law_name=args.law_name,
        article=args.article,
        paragraph=args.paragraph,
        item=args.item,
        subitem=args.subitem,
        content=None,
    )
    penalties = []
    if args.penalty_text:
        penalties.append(
            ImposedPenalty(
                penalty_type=args.penalty_type,
                target=args.penalty_target,
                amount_yuan=args.penalty_amount,
                content=args.penalty_text,
            )
        )
    context = HumanSupportContext(
        metadata={
            "案号": args.case_number,
            "案由": args.case_reason,
        },
        dir_info=[],
        rule={"序号": 360},
        review_item={},
        document_name="行政处罚决定书",
        section_id=1,
        section_ocr="",
    )
    facts = {
        "penalty_legal_citations": [citation],
        "imposed_penalties": penalties,
        "case_reason": args.case_reason,
    }
    result = await CityManagementDiscretionCandidatesModule().retrieve(
        context,
        facts,
    )
    print(
        json.dumps(
            {
                "extracted_facts": {
                    "penalty_legal_citations": [
                        item.model_dump()
                        for item in facts["penalty_legal_citations"]
                    ],
                    "imposed_penalties": [
                        item.model_dump()
                        for item in facts["imposed_penalties"]
                    ],
                    "case_reason": facts["case_reason"],
                },
                "retrieval": result.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
