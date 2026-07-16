from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CSV_FIELDS = ["rule_index", "comment"]


def convert_json_to_csv(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as file:
        data: Any = json.load(file)

    if not isinstance(data, list):
        raise ValueError("JSON 顶层必须是列表")

    row_count = 0

    # utf-8-sig 便于 Excel/WPS 正确识别中文。
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for item in data:
            if not isinstance(item, dict):
                raise ValueError("JSON 列表中的每一项都必须是对象")

            comments = item.get("comment", [])
            if not isinstance(comments, list):
                continue

            for comment_item in comments:
                if not isinstance(comment_item, dict):
                    continue

                # 删除 section_id，只保留实际评论文本。
                content = comment_item.get("content")
                if not isinstance(content, str) or not content.strip():
                    continue

                writer.writerow(
                    {
                        "rule_index": item.get("rule_index"),
                        "comment": content.strip(),
                    }
                )
                row_count += 1

    print(f"已输出非空评论：{row_count} 行")
    print(f"输出文件：{output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 review_results.json 的每条非空评论拆分为单独一行 CSV"
    )
    parser.add_argument("input", type=Path, help="输入 JSON 文件")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("review_comments.csv"),
        help="输出 CSV 文件",
    )
    args = parser.parse_args()

    convert_json_to_csv(args.input, args.output)


if __name__ == "__main__":
    main()
