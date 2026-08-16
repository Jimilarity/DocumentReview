import os
import sys
import unittest
from pathlib import Path


os.environ.setdefault("AGENT_TRACE_ENABLED", "false")
SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from pre_review import (
    _needs_ocr_fallback,
    _ocr_substantive_character_count,
    _prefer_retry_ocr,
)


class PreReviewOcrFallbackTest(unittest.TestCase):
    def test_imprint_only_ocr_triggers_a_retry(self) -> None:
        content = "\n".join(
            [
                '<div type="page-number">17</div>',
                '<div type="imprint" subtype="seal">某街道办事处</div>',
                '<div type="imprint" subtype="stamp">2022年1月10日</div>',
                '<div type="imprint" subtype="seal">某街道办事处</div>',
            ]
        )

        self.assertEqual(_ocr_substantive_character_count(content), 0)
        self.assertTrue(_needs_ocr_fallback(content))

    def test_readable_document_ocr_does_not_trigger_a_retry(self) -> None:
        content = (
            "深圳市某区某街道办事处\n责令改正违法行为通知书\n"
            "经查，你单位存在违法行为，现责令立即改正。"
        )

        self.assertFalse(_needs_ocr_fallback(content))

    def test_retry_replaces_primary_only_when_it_has_more_body_text(self) -> None:
        primary = '<div type="imprint">某街道办事处</div>'
        retry = "责令改正违法行为通知书\n经查，你单位存在违法行为。"

        self.assertEqual(_prefer_retry_ocr(primary, retry), retry)
        self.assertEqual(_prefer_retry_ocr(retry, primary), retry)


if __name__ == "__main__":
    unittest.main()
