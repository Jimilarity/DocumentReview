import json
import unittest
from unittest.mock import patch

from external_knowledge import KnowledgeContext
from external_knowledge.legal_citation_validity import knowledge


def build_context(**review_item):
    return KnowledgeContext(
        metadata={
            "案由": "擅自占道设摊经营",
            "案情": "当事人在城市道路擅自占道设摊经营。",
            "案发日期": "2025-06-01",
        },
        dir_info=[],
        rule={"序号": 108, "备注": ""},
        review_item={
            "审查事项": "适用的法律、法规或者规章准确、具体。",
            "评查说明": "核对法律依据。",
            **review_item,
        },
        document_name="行政处罚决定书",
        section_id=1,
        section_ocr="当事人在城市道路擅自占道设摊经营。依据某条例第18条处理。",
    )


class LegalCitationValidityTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        with knowledge._cache_lock:
            knowledge._cache.clear()

    async def test_retrieval_uses_configured_metadata_and_as_of(self):
        response = {
            "candidates": [
                {
                    "law_name": "深圳经济特区市容和环境卫生管理条例",
                    "version_id": "version-1",
                    "article": 18,
                    "article_suffix": None,
                    "effective_from": "2023-09-01",
                    "effective_to": None,
                    "article_text": "第18条完整正文",
                    "focused_text": "第18条相关片段",
                }
            ]
        }
        context = build_context(
            法条检索查询={"字段": [], "日期字段": ["案发日期"]}
        )
        with patch.object(knowledge, "_request", return_value=response) as request:
            items = await knowledge.retrieve_legal_citation_validity(context)

        payload = request.call_args.args[0]
        self.assertEqual(payload["as_of"], "2025-06-01")
        self.assertIn("案由：擅自占道设摊经营", payload["text"])
        self.assertIn("案情：当事人在城市道路", payload["text"])
        self.assertNotIn("审查事项", payload["text"])
        self.assertEqual(len(items), 1)
        self.assertIn("version-1", items[0].content)
        self.assertIn("第18条完整正文", items[0].content)

    async def test_default_query_is_limited_and_uses_no_optional_fields(self):
        context = build_context()
        context = KnowledgeContext(**{**context.__dict__, "metadata": {**context.metadata, "案情": "甲" * 400}})
        with patch.object(knowledge, "_request", return_value={"candidates": []}) as request:
            await knowledge.retrieve_legal_citation_validity(context)

        payload = request.call_args.args[0]
        self.assertLessEqual(len(payload["text"]), 300)
        self.assertTrue(payload["with_text"])
        self.assertTrue(payload["with_focus"])
        self.assertNotIn("as_of", payload)

    async def test_identical_request_is_cached(self):
        with patch.object(knowledge, "_request", return_value={"candidates": []}) as request:
            await knowledge.retrieve_legal_citation_validity(build_context())
            await knowledge.retrieve_legal_citation_validity(build_context())

        self.assertEqual(request.call_count, 1)

    async def test_invalid_query_config_is_reported_to_service_layer(self):
        context = build_context(法条检索查询={"top_k": 0})
        with self.assertRaisesRegex(TypeError, "top_k"):
            await knowledge.retrieve_legal_citation_validity(context)

    def test_query_falls_back_to_case_reason_when_facts_are_missing(self):
        context = build_context()
        context = KnowledgeContext(**{**context.__dict__, "metadata": {"案由": "案由"}})
        query, sources = knowledge._query_text(context, {})

        self.assertIn("案情：案由", query)
        self.assertEqual(sources, ["案由"])

    async def test_retrieval_falls_back_to_violation_facts_and_filing_date(self):
        context = build_context()
        context = KnowledgeContext(
            **{
                **context.__dict__,
                "metadata": {
                    "案由": "未履行安全生产管理职责",
                    "违法事实": "危化品储存柜未张贴安全操作规程。",
                    "立案日期": "2021-12-19",
                },
            }
        )
        with patch.object(
            knowledge,
            "_request",
            return_value={"candidates": []},
        ) as request, patch.object(knowledge, "trace_event") as trace_event:
            await knowledge.retrieve_legal_citation_validity(context)

        payload = request.call_args.args[0]
        self.assertIn("案情：违法事实：危化品储存柜", payload["text"])
        self.assertEqual(payload["as_of"], "2021-12-19")
        trace_event.assert_any_call(
            "law_api_query_context",
            provider="legal_citation_validity",
            case_facts_sources=["违法事实", "立案日期"],
            as_of="2021-12-19",
        )

    def test_query_requires_case_reason(self):
        context = build_context()
        context = KnowledgeContext(**{**context.__dict__, "metadata": {}})
        with self.assertRaisesRegex(ValueError, "案由"):
            knowledge._query_text(context, {})

    def test_request_uses_api_key_and_public_url(self):
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"candidates": []}'

        with patch.dict(
            "os.environ",
            {"LAW_RETRIEVAL_API_KEY": "secret-key"},
            clear=True,
        ), patch.object(knowledge, "urlopen", return_value=FakeResponse()) as open_request:
            self.assertEqual(knowledge._request({"text": "案情"}), {"candidates": []})

        request = open_request.call_args.args[0]
        self.assertEqual(request.full_url, "https://review.zfqp.fun/law-api/retrieve")
        self.assertEqual(request.get_header("X-api-key"), "secret-key")

    def test_request_traces_payload_and_response_without_api_key(self):
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"candidates": [{"law_name": "Test Law"}]}'

        payload = {"text": "case facts", "top_k": 3}
        with patch.dict(
            "os.environ",
            {"LAW_RETRIEVAL_API_KEY": "secret-key"},
            clear=True,
        ), patch.object(
            knowledge,
            "urlopen",
            return_value=FakeResponse(),
        ), patch.object(knowledge, "trace_event") as trace_event:
            result = knowledge._request(payload)

        self.assertEqual(result["candidates"][0]["law_name"], "Test Law")
        request_event = trace_event.call_args_list[0]
        response_event = trace_event.call_args_list[1]
        self.assertEqual(request_event.args[0], "law_api_request")
        self.assertEqual(request_event.kwargs["payload"], payload)
        self.assertNotIn("secret-key", str(request_event))
        self.assertEqual(response_event.args[0], "law_api_response")
        self.assertEqual(response_event.kwargs["status_code"], 200)
        self.assertEqual(response_event.kwargs["response"], result)

    def test_request_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "LAW_RETRIEVAL_API_KEY"):
                knowledge._request({"text": "案情"})
