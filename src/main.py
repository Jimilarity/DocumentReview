#main.py
import argparse
import asyncio
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

from constants import ErrorCode
from post_review import run_post_review
from pre_review import run_pre_review
from review import run_review, build_review_runtime
from rule_utils import validate_rule_type, parse_rule_type_value
from error import DocumentReviewError, PreReviewError, ReviewError, PostReviewError
from error_handler import capture_error, write_error_report
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="absolute path of PDF file",
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
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"文件不是 PDF: {file_path}")
    if not path.is_file():
        raise FileNotFoundError(f"PDF 文件不存在: {file_path}")
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
        pdf_path = validate_input(file_path)
        
        def get_pre_review_cache_files(pdf_path: Path) -> Dict[str, Path]:
            pdf_name = pdf_path.stem

            return {
                "image_list": Path("pdf_cache") / pdf_name / "image_list.json",
                "dir_info": Path("dir_cache") / pdf_name / "dir_info.json",
                "meta_info": Path("meta_cache") / pdf_name / "meta_info.json",
            }

        def pre_review_cache_exists(pdf_path: Path) -> bool:
            cache_files = get_pre_review_cache_files(pdf_path)
            return all(path.is_file() for path in cache_files.values())
        
        start_time = time.time()

        runtime = await build_review_runtime(str(pdf_path))

        cache_files = get_pre_review_cache_files(pdf_path)

        if pre_review_cache_exists(pdf_path):
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
                str(pdf_path),
                runtime.dr_instance,
            )

        review_result = await run_review(
            str(pdf_path),
            rule_type,
            runtime,
        )

        post_review_result = await run_post_review(
            str(pdf_path),
            rule_type,
        )

        elapsed_time = time.time() - start_time
        print(f"全部流程执行时间: {elapsed_time:.4f} 秒")
        print(f"审查结果数量: {len(review_result['rule_results'])}")

        return {
            "success": True,
            "pre_review": pre_review_result,
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

        pdf_name = Path(file_path).stem if file_path else "unknown"

        write_error_report(
            Path("results") / pdf_name / "flow_error.json",
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
