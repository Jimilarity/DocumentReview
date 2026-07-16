import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from knowledge_retrieval.city_management_discretion import (
    CATALOG_PATH,
    RERANK_DIRECTORY,
    RERANK_EMBEDDINGS_PATH,
    RERANK_MANIFEST_PATH,
    compute_violation_text_fingerprint,
)


DEFAULT_LOCAL_MODEL = (
    PROJECT_ROOT
    / "database"
    / "longhua_subdistrict_penalty_items"
    / "model"
)
DEFAULT_MODEL_ID = "BAAI/bge-small-zh-v1.5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建城管裁量标准违法行为语义重排向量"
    )
    parser.add_argument("--source", type=Path, default=CATALOG_PATH)
    parser.add_argument("--output", type=Path, default=RERANK_DIRECTORY)
    parser.add_argument(
        "--model",
        default=(
            str(DEFAULT_LOCAL_MODEL)
            if DEFAULT_LOCAL_MODEL.is_dir()
            else DEFAULT_MODEL_ID
        ),
    )
    return parser.parse_args()


def model_reference(value: str) -> dict[str, str]:
    path = Path(value)
    if path.exists():
        resolved = path.resolve()
        if resolved == DEFAULT_LOCAL_MODEL.resolve():
            return {
                "kind": "model_id",
                "value": DEFAULT_MODEL_ID,
            }
        try:
            relative = resolved.relative_to(PROJECT_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(
                "本地向量模型必须位于项目目录内，确保manifest可移植"
            ) from exc
        return {
            "kind": "project_relative",
            "value": relative.as_posix(),
        }
    return {"kind": "model_id", "value": value}


def main() -> None:
    args = parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    items = source.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("裁量知识文件没有items")

    item_ids: list[str] = []
    violations: list[str] = []
    for item in items:
        item_id = item.get("id")
        violation = item.get("违法行为")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("裁量知识记录缺少id")
        if not isinstance(violation, str) or not violation.strip():
            raise ValueError(f"裁量知识记录缺少违法行为: {item_id}")
        item_ids.append(item_id)
        violations.append(violation.strip())
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("裁量知识记录id重复")

    model = SentenceTransformer(args.model)
    embeddings = model.encode(
        violations,
        batch_size=32,
        # 仅有 75 条记录；关闭 tqdm 也避免 conda run 在中文 Windows
        # 终端转发进度字符时触发 GBK 编码错误。
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    embeddings_path = (
        RERANK_EMBEDDINGS_PATH
        if args.output.resolve() == RERANK_DIRECTORY.resolve()
        else args.output / RERANK_EMBEDDINGS_PATH.name
    )
    manifest_path = (
        RERANK_MANIFEST_PATH
        if args.output.resolve() == RERANK_DIRECTORY.resolve()
        else args.output / RERANK_MANIFEST_PATH.name
    )
    np.save(embeddings_path, embeddings)
    manifest = {
        "schema_version": 1,
        "index_type": "sentence_transformers_cosine_rerank",
        "model_reference": model_reference(args.model),
        "query_prefix": QUERY_PREFIX,
        "indexed_field": "违法行为",
        "item_ids": item_ids,
        "item_count": len(item_ids),
        "embedding_dimension": int(embeddings.shape[1]),
        "violation_text_fingerprint": (
            compute_violation_text_fingerprint(items)
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"已生成{len(item_ids)}条违法行为向量，维度"
        f"{embeddings.shape[1]}: {embeddings_path}"
    )


if __name__ == "__main__":
    main()
