import asyncio
import json
from pathlib import Path
import sys
import shutil


async def run_review(task_id: str, input_pdf: str, file_type: int, result_dir: str) -> dict:
    """Run the real review pipeline by invoking `review/main.py` as a subprocess.

    This keeps the existing `review/main.py` implementation unchanged and
    calls it as a separate process. After it finishes, copy the produced
    files from `results/{stem}` to `result_dir` and return a summary dict.
    """
    out_dir = Path(result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # call the review logic directly as a coroutine in-process
    from review import main as review_main
    log_dir = Path("backend") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        await asyncio.wait_for(review_main.main(str(input_pdf), int(file_type)), timeout=60*60)
    except asyncio.TimeoutError:
        return {"task_id": task_id, "status": "failed", "error": "timeout"}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        with open(log_dir / f"{task_id}.exception.log", "w", encoding="utf-8") as f:
            f.write(tb)
        return {"task_id": task_id, "status": "failed", "error": str(e)}

    # copy produced results from review's results/<stem>
    produced_dir = Path("results") / Path(input_pdf).stem
    if not produced_dir.exists():
        return {"task_id": task_id, "status": "failed", "error": "no_results_dir"}

    for item in produced_dir.iterdir():
        if item.is_file():
            shutil.copy2(item, out_dir / item.name)

    files = [p.name for p in out_dir.iterdir() if p.is_file()]

    return {"task_id": task_id, "status": "success", "result_dir": str(out_dir), "files": files}
