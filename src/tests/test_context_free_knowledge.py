import unittest
from types import SimpleNamespace

from external_knowledge import KnowledgeItem
from reviewers.context_free import ContextFreeReviewExecutor


class ContextFreeKnowledgePromptTest(unittest.TestCase):
    @staticmethod
    def build_executor() -> ContextFreeReviewExecutor:
        executor = ContextFreeReviewExecutor.__new__(
            ContextFreeReviewExecutor
        )
        executor.require_context_free_agents = lambda: SimpleNamespace(
            build_task_prompt=lambda *args, **kwargs: "base prompt"
        )
        return executor

    @staticmethod
    def build_state(items):
        return {
            "rule": {"序号": 101, "备注": ""},
            "meta_info": {},
            "document_name": "行政处罚决定书",
            "document_review_rule": {
                "审查事项": "核查执法权限",
                "评查说明": "",
            },
            "section_id": 8,
            "ocr_text": "正文",
            "external_knowledge": items,
        }

    def test_empty_knowledge_does_not_add_prompt_block(self) -> None:
        prompt = self.build_executor().build_rule_prompt(
            self.build_state([])
        )

        self.assertEqual(prompt, "base prompt")
        self.assertNotIn("external_knowledge", prompt)

    def test_non_empty_knowledge_adds_prompt_block(self) -> None:
        prompt = self.build_executor().build_rule_prompt(
            self.build_state([KnowledgeItem(content="目录条目")])
        )

        self.assertIn("<external_knowledge>", prompt)
        self.assertIn('"content": "目录条目"', prompt)
        self.assertIn("</external_knowledge>", prompt)
        self.assertNotIn("知识来源", prompt)
        self.assertNotIn("检索方式", prompt)
        self.assertNotIn("与本案不符时应当忽略", prompt)
        self.assertNotIn("不得据此扩大", prompt)
        self.assertNotIn("可能来自向量相似度召回", prompt)
        self.assertNotIn("仅表示可能相关", prompt)

    def test_review_item_knowledge_names_are_optional_and_deduplicated(
        self,
    ) -> None:
        self.assertEqual(
            ContextFreeReviewExecutor.knowledge_function_names({}),
            [],
        )
        self.assertEqual(
            ContextFreeReviewExecutor.knowledge_function_names(
                {"外部知识": ["catalog", "catalog", " law_api "]}
            ),
            ["catalog", "law_api"],
        )


if __name__ == "__main__":
    unittest.main()
