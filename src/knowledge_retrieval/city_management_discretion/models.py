from typing import Literal

from pydantic import Field

from ..common import LegalCitation, StrictModel


class GeneratedItemCitations(StrictModel):
    """离线 LLM 为一个裁量标准序号生成的处罚依据检索元数据。"""

    item_id: str = Field(min_length=1)
    sequence: int = Field(gt=0)
    citations: list[LegalCitation] = Field(min_length=1)


class DiscretionRetrievalMetadata(StrictModel):
    schema_version: Literal[1] = 1
    citations: list[LegalCitation] = Field(
        min_length=1,
        alias="处罚依据",
    )
