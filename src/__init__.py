import asyncio
from pathlib import Path
import shutil


async def run_review(task_id: str, input_pdf: str, file_type: int, result_dir: str) -> dict:
    """在当前进程运行审查流水线，并复制生成的结果文件。"""
    out_dir = Path(result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from main import main as run_review_pipeline
    log_dir = Path("backend") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        pipeline_result = await asyncio.wait_for(
            run_review_pipeline(str(input_pdf), int(file_type)),
            timeout=60 * 60,
        )
        if not pipeline_result.get("success", False):
            return {
                "task_id": task_id,
                "status": "failed",
                "error": pipeline_result.get("error_message", "review_failed"),
                "error_details": pipeline_result.get("error_details", {}),
            }
    except asyncio.TimeoutError:
        return {"task_id": task_id, "status": "failed", "error": "timeout"}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        with open(log_dir / f"{task_id}.exception.log", "w", encoding="utf-8") as f:
            f.write(tb)
        return {"task_id": task_id, "status": "failed", "error": str(e)}

    review_result = pipeline_result.get("review", {})
    review_result_path = review_result.get("result_path")
    if review_result_path:
        produced_dir = Path(review_result_path).parent
    else:
        from cache_paths import get_result_directory

        produced_dir = get_result_directory(input_pdf)
    if not produced_dir.exists():
        return {"task_id": task_id, "status": "failed", "error": "no_results_dir"}

    for item in produced_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, out_dir / item.name)

    files = [p.name for p in out_dir.iterdir() if p.is_file()]

    return {"task_id": task_id, "status": "success", "result_dir": str(out_dir), "files": files}
