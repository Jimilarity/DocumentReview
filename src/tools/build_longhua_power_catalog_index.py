import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "knowledge"
    / "深圳市龙华区街道承接区政府部门行政处罚事项"
    / "深圳市龙华区街道承接区政府部门行政处罚事项目录_结构化.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "database" / "longhua_subdistrict_penalty_items"
)
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建龙华区街道承接行政处罚事项语义向量索引"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="同时把模型保存到索引目录，供离线环境使用",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_bytes = args.source.read_bytes()
    source = json.loads(source_bytes.decode("utf-8"))
    items = [
        {
            "item_name": item["item_name"],
            "related_district_authority": item["related_district_authority"],
            "implementation_scope": list(item["implementation_scope"]),
            "remark": item.get("remark"),
        }
        for item in source["items"]
    ]
    if not items:
        raise ValueError("源数据没有行政处罚事项")

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "model"
    if model_path.is_dir():
        model = SentenceTransformer(str(model_path))
    else:
        model = SentenceTransformer(args.model)
    if args.save_model and not model_path.is_dir():
        model.save(str(model_path))

    embeddings = model.encode(
        [item["item_name"] for item in items],
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    np.save(args.output / "embeddings.npy", embeddings)
    (args.output / "items.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "index_type": "sentence_transformers_cosine",
        "model": args.model,
        "query_prefix": QUERY_PREFIX,
        "item_count": len(items),
        "embedding_dimension": int(embeddings.shape[1]),
        "source": str(args.source),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "indexed_field": "item_name",
        "retained_fields": [
            "item_name",
            "related_district_authority",
            "implementation_scope",
            "remark",
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"built {len(items)} vectors with dimension {embeddings.shape[1]} "
        f"at {args.output}"
    )


if __name__ == "__main__":
    main()
