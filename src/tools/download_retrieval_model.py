"""预下载检索使用的向量模型，供无网络运行环境打包。"""

import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ID = "BAAI/bge-small-zh-v1.5"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "database"
    / "longhua_subdistrict_penalty_items"
    / "model"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = SentenceTransformer(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    print(f"retrieval model saved to {args.output}")


if __name__ == "__main__":
    main()
