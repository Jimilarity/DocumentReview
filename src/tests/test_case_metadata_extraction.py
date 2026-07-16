import os
import sys
import unittest
from pathlib import Path


os.environ.setdefault("AGENT_TRACE_ENABLED", "false")
SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from agents import CaseMetadata
from utils import load_yaml


class CaseMetadataExtractionTest(unittest.TestCase):
    def test_extractor_prompt_explicitly_excludes_archive_fields(self) -> None:
        tasks = load_yaml(SRC_ROOT / "config" / "tasks.yaml")
        prompt = tasks["case_metadata_extractor"]["description"]

        self.assertIn("【一律不得提取】", prompt)
        self.assertIn("归档人、归档人员、归档日期", prompt)
        self.assertIn("保管期限、保管年限、保存期限", prompt)
        self.assertIn("也不得作为固定字段或额外字段输出", prompt)
        self.assertNotIn("case_metadata_refiner", tasks)

    def test_case_metadata_structures_enforcement_officers(self) -> None:
        metadata = CaseMetadata.model_validate(
            {
                "执法单位": "深圳市某局",
                "案卷名称": "测试案卷",
                "案号": "A-001",
                "案由": None,
                "当事人": "张三",
                "立案日期": None,
                "结案日期": None,
                "处理结果": None,
                "案件承办人员及执法证件号": [
                    {
                        "执法人": "袁鹏龙",
                        "执法证号": "19020996204",
                    }
                ],
                "额外字段": {"承办机构": "执法一科"},
            }
        ).model_dump(by_alias=True)

        self.assertEqual(
            metadata["案件承办人员及执法证件号"][0],
            {"执法人": "袁鹏龙", "执法证号": "19020996204"},
        )

if __name__ == "__main__":
    unittest.main()
