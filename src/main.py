#main.py
import argparse
import asyncio
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
env_path = Path(__file__).resolve().parents[1] / '.env'
load_dotenv(env_path)

from constants import ErrorCode, RESULT_ROOT
from cache_paths import get_cache_paths, get_result_directory
from post_review import run_post_review
from pre_review import run_pre_review
from review import run_review
from structured_input import prepare_structured_json
from rules.rule_type import parse_rule_type_value, validate_rule_type
from errors.exceptions import (
    DocumentReviewError,
    PostReviewError,
    PreReviewError,
    ReviewError,
)
from errors.handler import capture_error, write_error_report
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="absolute path of PDF or structured JSON file",
    )
    parser.add_argument(
        "--rule_type",
        type=parse_rule_type_value,
        required=True,
        help=(
        "八位规则类型，例如 11101000；"
        "前三位为评查类别，中间两位为文书类型，"
        "后三位为文书子类型"
    )
    )
    return parser


def validate_input(file_path: str) -> Path:
    path = Path(file_path)
    if path.suffix.lower() not in {".pdf", ".json"}:
        raise ValueError(f"文件必须是 PDF 或 JSON: {file_path}")
    if not path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {file_path}")
    return path


async def main(
    file_path: Optional[str] = None,
    rule_type: int = None,
) -> Dict[str, Any]:
    if file_path is None or rule_type is None:
        args = build_parser().parse_args()
        file_path = args.file_path
        rule_type = args.rule_type

    validate_rule_type(rule_type)

    try:
        input_path = validate_input(file_path)

        start_time = time.time()

        cache_paths = get_cache_paths(input_path)
        cache_files = cache_paths.pre_review_files

        if input_path.suffix.lower() == ".json":
            print("检测到结构化 JSON，开始生成独立审查缓存")
            pre_review_result = prepare_structured_json(
                input_path,
                rule_type,
            )
        elif all(path.is_file() for path in cache_files.values()):
            print("预处理缓存完整，跳过预处理步骤")
            pre_review_result = {
                "skipped": True,
                "reason": "pre_review_cache_exists",
                "cache_files": {
                    name: str(path)
                    for name, path in cache_files.items()
                },
            }
        else:
            missing_cache_files = [
                str(path)
                for path in cache_files.values()
                if not path.is_file()
            ]

            print(
                "预处理缓存不完整，开始执行预处理。"
                f"缺失文件: {missing_cache_files}"
            )

            pre_review_result = await run_pre_review(
                str(input_path),
            )

        review_result = await run_review(
            str(input_path),
            rule_type,
        )
        context_sensitive_preparation = review_result.get(
            "context_sensitive_preparation",
            {
                "preparation_completed": False,
                "skipped": True,
                "reason": "no_applicable_context_sensitive_rules",
            },
        )

        post_review_result = await run_post_review(
            str(input_path),
            rule_type,
        )

        elapsed_time = time.time() - start_time
        print(f"全部流程执行时间: {elapsed_time:.4f} 秒")
        print(f"审查结果数量: {len(review_result['review_results'])}")
        retrieval_result = (
            review_result.get("retrieval_enhancement") or {}
        )
        print(
            "检索增强规则数量: "
            f"{retrieval_result.get('rule_count', 0)}"
        )
        if retrieval_result.get("result_path"):
            print(
                "检索增强结果: "
                f"{retrieval_result['result_path']}"
            )

        return {
            "success": True,
            "pre_review": pre_review_result,
            "context_sensitive_preparation": (
                context_sensitive_preparation
            ),
            "review": review_result,
            "post_review": post_review_result,
            "elapsed_seconds": elapsed_time,
        }
    except DocumentReviewError as exc:
        details = exc.details

        print(
            f"FLOW STOPPED, ERROR CODE: {exc.error_code}, "
            f"ERROR ID: {details.get('error_id', 'unknown')}"
        )
        print(str(exc))

        return {
            "success": False,
            "error_code": exc.error_code,
            "error_message": str(exc),
            "error_details": details,
        }
    except Exception as exc:
        details = capture_error(
            "main.unhandled_exception",
            exc,
            pdf_path=file_path,
            extra={
                "rule_type": rule_type,
            },
        )

        write_error_report(
            get_result_directory(
                file_path or "unknown",
                result_root=RESULT_ROOT,
            ) / "flow_error.json",
            details,
        )

        print(
            f"FLOW STOPPED, "
            f"ERROR CODE: {int(ErrorCode.UNEXPECTED_ERROR)}, "
            f"ERROR ID: {details['error_id']}"
        )
        print(str(exc))

        return {
            "success": False,
            "error_code": int(ErrorCode.UNEXPECTED_ERROR),
            "error_message": str(exc),
            "error_details": details,
        }

if __name__ == "__main__":
    result = asyncio.run(main())
    raise SystemExit(0 if result["success"] else 1)
