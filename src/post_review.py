from typing import Any, Dict


async def run_post_review(
    file_path: str,
    rule_type: int,
) -> Dict[str, Any]:
    """后处理预留入口，目前不执行任何业务逻辑。"""
    return {
        "completed": True,
        "message": "post_review is currently a no-op",
    }
