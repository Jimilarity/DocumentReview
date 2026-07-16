import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_ROOT = SRC_ROOT / "knowledge_retrieval"


class KnowledgeRetrievalArchitectureTest(unittest.TestCase):
    def test_neutral_core_has_no_consumer_layer_imports(self) -> None:
        forbidden = ("external_knowledge", "human_support")
        for path in RETRIEVAL_ROOT.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for dependency in forbidden:
                with self.subTest(path=path, dependency=dependency):
                    self.assertNotIn(
                        dependency,
                        source,
                        (
                            "中立检索核心不得反向依赖知识注入或"
                            "检索增强消费层"
                        ),
                    )

    def test_obsolete_discretion_external_adapter_is_removed(self) -> None:
        obsolete = (
            SRC_ROOT
            / "external_knowledge"
            / "city_management_discretion"
        )
        self.assertFalse(obsolete.exists())


if __name__ == "__main__":
    unittest.main()
