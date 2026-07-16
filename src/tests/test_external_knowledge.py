import logging
import unittest

from external_knowledge import (
    KnowledgeContext,
    KnowledgeItem,
    KnowledgeRegistry,
    KnowledgeService,
    knowledge_registry,
)


def build_context() -> KnowledgeContext:
    return KnowledgeContext(
        metadata={"案由": "生活垃圾分类"},
        dir_info=[
            {
                "section_id": 1,
                "section_name": "行政处罚决定书",
                "section_page": 2,
            }
        ],
        rule={"序号": 101, "备注": ""},
        review_item={"审查事项": "核查执法权限"},
        document_name="行政处罚决定书",
        section_id=8,
        section_ocr="行政处罚决定书正文",
    )


class KnowledgeServiceTest(unittest.IsolatedAsyncioTestCase):
    def test_default_registry_contains_documented_knowledge_functions(
        self,
    ) -> None:
        self.assertEqual(
            set(knowledge_registry.names()),
            {
                "case_directory_info",
                "legal_citation_validity",
                "longhua_subdistrict_penalty_items_catalog",
                "shenzhen_administrative_litigation_jurisdiction",
            },
        )

    async def test_function_receives_uniform_context_but_may_ignore_it(
        self,
    ) -> None:
        registry = KnowledgeRegistry()
        received = []

        @registry.register("static_knowledge")
        async def static_knowledge(context):
            received.append(context)
            return [KnowledgeItem(content="  固定知识  ")]

        service = KnowledgeService(registry)
        context = build_context()
        result = await service.collect(["static_knowledge"], context)

        self.assertEqual(received, [context])
        self.assertEqual(result, [KnowledgeItem(content="固定知识")])

    async def test_empty_blank_and_failed_results_are_skipped(self) -> None:
        registry = KnowledgeRegistry()

        @registry.register("empty")
        async def empty(context):
            return []

        @registry.register("blank")
        async def blank(context):
            return [KnowledgeItem(content="  ")]

        @registry.register("failed")
        async def failed(context):
            raise RuntimeError("knowledge backend unavailable")

        service = KnowledgeService(
            registry,
            logger=logging.getLogger("test.external_knowledge"),
        )
        with self.assertLogs("test.external_knowledge", level="WARNING"):
            result = await service.collect(
                ["empty", "blank", "failed"],
                build_context(),
            )

        self.assertEqual(result, [])

    async def test_unknown_function_is_a_configuration_error(self) -> None:
        service = KnowledgeService(KnowledgeRegistry())

        with self.assertRaisesRegex(KeyError, "知识函数未注册"):
            await service.collect(["misspelled_name"], build_context())

    async def test_case_directory_info_is_available_as_configured_knowledge(
        self,
    ) -> None:
        service = KnowledgeService(knowledge_registry)

        result = await service.collect(
            ["case_directory_info"],
            build_context(),
        )

        self.assertEqual(len(result), 1)
        self.assertIn('"section_name": "行政处罚决定书"', result[0].content)
        self.assertIn("本案案卷目录结构", result[0].content)
        self.assertNotIn("来源于", result[0].content)

    async def test_empty_directory_does_not_produce_knowledge(self) -> None:
        context = build_context()
        empty_context = KnowledgeContext(
            metadata=context.metadata,
            dir_info=[],
            rule=context.rule,
            review_item=context.review_item,
            document_name=context.document_name,
            section_id=context.section_id,
            section_ocr=context.section_ocr,
        )

        result = await KnowledgeService(knowledge_registry).collect(
            ["case_directory_info"],
            empty_context,
        )

        self.assertEqual(result, [])

if __name__ == "__main__":
    unittest.main()
