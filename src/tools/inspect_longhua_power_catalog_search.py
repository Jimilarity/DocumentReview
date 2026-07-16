"""查看龙华区街道承接行政处罚事项的 Top K 检索结果。

可直接修改 DEFAULT_CASE_REASONS 后运行：

    conda run -n myenv python src/tools/inspect_longhua_power_catalog_search.py

也可以用 --query 临时传入一个或多个案由，而不修改本文件。
"""

import argparse
import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from external_knowledge.longhua_subdistrict_penalty import (
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    LonghuaPenaltyCatalogIndex,
)


# 未通过 --query 指定案由时使用的示例。
DEFAULT_CASE_REASONS = [
    "当事人张三未按照规定要求投放生活垃圾案",
    "将厨房、卫生间、阳台等非居住空间单独出租用于居住案",
    "深圳市豪邦物流有限公司生产经营单位主要负责人每年再培训时间少于12学时案",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="打印案由在龙华区街道承接行政处罚事项目录中的 Top K 结果"
    )
    parser.add_argument(
        "--query",
        action="append",
        help="临时指定案由；可重复传入。提供后不再使用文件内置案由。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"每个案由返回的最大候选数，默认 {DEFAULT_TOP_K}",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help=(
            "最低相似度。手工观察排名时默认0.0，正式审查默认值为 "
            f"{DEFAULT_MIN_SCORE}"
        ),
    )
    return parser.parse_args()


def print_match(rank: int, match) -> None:
    print(f"  Top {rank} | 相似度: {match.score:.4f}")
    print(f"  事项名称: {match.item_name}")
    print(f"  相关区行政主管部门: {match.related_district_authority}")
    print(f"  实施范围: {'、'.join(match.implementation_scope)}")
    print(f"  备注: {match.remark or '无'}")


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k 必须是正整数")

    case_reasons = args.query or DEFAULT_CASE_REASONS
    if not case_reasons:
        raise ValueError("没有配置需要测试的案由")

    index = LonghuaPenaltyCatalogIndex()
    for query_index, case_reason in enumerate(case_reasons, start=1):
        print("=" * 88)
        print(f"案由 {query_index}: {case_reason}")
        print(
            f"参数: top_k={args.top_k}, min_score={args.min_score:.4f}"
        )
        matches = index.search(
            case_reason,
            top_k=args.top_k,
            min_score=args.min_score,
        )
        if not matches:
            print("  未检索到达到最低相似度的候选事项")
            continue
        for rank, match in enumerate(matches, start=1):
            print_match(rank, match)
            if rank != len(matches):
                print("-" * 88)


if __name__ == "__main__":
    main()
