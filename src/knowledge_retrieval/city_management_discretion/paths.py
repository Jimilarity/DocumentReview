from constants import PROJECT_ROOT


CATALOG_PATH = (
    PROJECT_ROOT
    / "knowledge"
    / "深圳市城市管理行政处罚裁量权实施标准"
    / "shenzhen_city_management_discretion_2023.json"
)
RERANK_DIRECTORY = (
    PROJECT_ROOT / "database" / "shenzhen_city_management_discretion"
)
RERANK_EMBEDDINGS_PATH = RERANK_DIRECTORY / "violation_embeddings.npy"
RERANK_MANIFEST_PATH = (
    RERANK_DIRECTORY / "violation_embeddings_manifest.json"
)
