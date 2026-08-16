import asyncio
import copy
import hashlib
import json
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any, Dict, List

from reviewers.document_mapping import (
    document_types_may_share_section,
    load_compatible_document_type_groups,
)
from utils import atomic_write_json, read_json


TECHNICAL_MAPPING_FAILURE = -1
EVENT_KEY_PREFIX = "event:"

FieldExtractor = Callable[
    [int, List[str]],
    Awaitable[Dict[str, Any]],
]
DeliveryFieldExtractor = Callable[
    [int, str, List[str]],
    Awaitable[Dict[str, Any]],
]


def build_structured_source_fingerprint(
    meta_info: Dict[str, Any],
    dir_info: List[Dict[str, Any]],
    ocr_results: List[Dict[str, Any]],
) -> str:
    payload = json.dumps(
        {
            "meta_info": meta_info,
            "dir_info": dir_info,
            "ocr_results": ocr_results,
            "compatible_document_type_groups": (
                load_compatible_document_type_groups()
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class StructuredFieldCache:
    """案卷结构化字段的内存镜像和持久化缓存。

    普通文书使用 ``sections[section_id][field]``。送达回证使用
    ``sections[receipt_section_id][related_section_id][field]``；没有可绑定
    文书或发生技术错误的事件使用唯一的 ``event:<event_id>`` 占位键。

    同一缓存还保存文书存在性和文书到 section 的一次性映射。源数据或提取
    协议变化时整份快照失效；映射初始化后只允许幂等地再次提交相同结果。
    """

    def __init__(
        self,
        path: str | Path,
        *,
        schema_version: int,
        source_fingerprint: str,
        payload: Dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.schema_version = schema_version
        self.source_fingerprint = source_fingerprint
        self.payload = payload if payload is not None else self._empty_payload()
        self._field_extractor: FieldExtractor | None = None
        self._delivery_field_extractor: DeliveryFieldExtractor | None = None
        self._state_lock = asyncio.Lock()
        self._field_inflight: Dict[tuple[int, str], asyncio.Task[Any]] = {}
        self._delivery_field_inflight: Dict[
            tuple[int, str, str], asyncio.Task[Any]
        ] = {}

    def _empty_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_fingerprint": self.source_fingerprint,
            "preparation_completed": False,
            "document_presence": {},
            "document_section_map": {},
            "sections": {},
            "section_metadata": {},
        }

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        schema_version: int,
        source_fingerprint: str,
    ) -> "StructuredFieldCache":
        path = Path(path)
        if not path.is_file():
            return cls(
                path,
                schema_version=schema_version,
                source_fingerprint=source_fingerprint,
            )

        payload = read_json(path)
        if not isinstance(payload, dict):
            raise TypeError("结构化字段缓存根节点必须是对象")
        if (
            payload.get("schema_version") != schema_version
            or payload.get("source_fingerprint") != source_fingerprint
        ):
            return cls(
                path,
                schema_version=schema_version,
                source_fingerprint=source_fingerprint,
            )
        if not isinstance(payload.get("sections"), dict):
            raise TypeError("结构化字段缓存 sections 必须是对象")
        if not isinstance(payload.get("section_metadata"), dict):
            raise TypeError("结构化字段缓存 section_metadata 必须是对象")
        if not isinstance(payload.get("preparation_completed"), bool):
            raise TypeError("结构化字段缓存 preparation_completed 必须是布尔值")
        if not isinstance(payload.get("document_presence"), dict):
            raise TypeError("结构化字段缓存 document_presence 必须是对象")
        if not isinstance(payload.get("document_section_map"), dict):
            raise TypeError("结构化字段缓存 document_section_map 必须是对象")
        return cls(
            path,
            schema_version=schema_version,
            source_fingerprint=source_fingerprint,
            payload=payload,
        )

    @property
    def sections(self) -> Dict[str, Dict[str, Any]]:
        return self.payload["sections"]

    @property
    def section_metadata(self) -> Dict[str, Dict[str, Any]]:
        return self.payload["section_metadata"]

    @property
    def preparation_completed(self) -> bool:
        return self.payload["preparation_completed"] is True

    @property
    def document_presence(self) -> Dict[str, bool]:
        presence = self.payload["document_presence"]
        if not all(
            isinstance(name, str)
            and name
            and isinstance(exists, bool)
            for name, exists in presence.items()
        ):
            raise TypeError("document_presence 必须是非空字符串到布尔值的映射")
        return dict(presence)

    @property
    def document_section_map(self) -> Dict[str, List[int]]:
        raw_mapping = self.payload["document_section_map"]
        mapping: Dict[str, List[int]] = {}
        section_owners: Dict[int, List[str]] = {}
        for document_type, section_ids in raw_mapping.items():
            if not isinstance(document_type, str) or not document_type:
                raise TypeError("document_section_map 的文书类型必须是非空字符串")
            if not isinstance(section_ids, list) or not all(
                isinstance(section_id, int)
                and not isinstance(section_id, bool)
                and section_id > 0
                for section_id in section_ids
            ):
                raise TypeError(
                    f"document_section_map.{document_type} 必须是正整数数组"
                )
            normalized_ids = list(dict.fromkeys(section_ids))
            for section_id in normalized_ids:
                existing_owners = section_owners.setdefault(section_id, [])
                if (
                    existing_owners
                    and not document_types_may_share_section(
                        [*existing_owners, document_type]
                    )
                ):
                    raise ValueError(
                        f"section_id={section_id} 不能同时映射到文书类型 "
                        f"{existing_owners} 和 {document_type}；"
                        "只有同一相容类型组内的文书类型可以共享 section"
                    )
                if document_type not in existing_owners:
                    existing_owners.append(document_type)
            mapping[document_type] = normalized_ids
        return mapping

    def initialize_document_presence(self, values: Dict[str, bool]) -> bool:
        if not all(
            isinstance(name, str)
            and name
            and isinstance(exists, bool)
            for name, exists in values.items()
        ):
            raise TypeError("document_presence 必须是非空字符串到布尔值的映射")
        current = self.document_presence
        normalized = dict(values)
        if current == normalized:
            return False
        if current:
            raise RuntimeError("document_presence 已初始化，不能局部合并或替换")
        self.payload["document_presence"] = normalized
        self.payload["preparation_completed"] = False
        return True

    def initialize_document_section_map(
        self,
        values: Dict[str, List[int]],
    ) -> bool:
        normalized: Dict[str, List[int]] = {}
        for document_type, section_ids in values.items():
            if not isinstance(document_type, str) or not document_type:
                raise TypeError("文书类型必须是非空字符串")
            if not isinstance(section_ids, list) or not section_ids:
                raise ValueError(
                    f"document_section_map.{document_type} 必须是非空数组"
                )
            if any(
                isinstance(section_id, bool)
                or not isinstance(section_id, int)
                or section_id <= 0
                for section_id in section_ids
            ):
                raise TypeError(
                    f"document_section_map.{document_type} 必须是正整数数组"
                )
            normalized[document_type] = list(dict.fromkeys(section_ids))

        section_owners: Dict[int, List[str]] = {}
        for document_type, section_ids in normalized.items():
            for section_id in section_ids:
                existing_owners = section_owners.setdefault(section_id, [])
                if (
                    existing_owners
                    and not document_types_may_share_section(
                        [*existing_owners, document_type]
                    )
                ):
                    raise ValueError(
                        f"section_id={section_id} 不能同时映射到文书类型 "
                        f"{existing_owners} 和 {document_type}；"
                        "只有同一相容类型组内的文书类型可以共享 section"
                    )
                if document_type not in existing_owners:
                    existing_owners.append(document_type)
        current = self.document_section_map
        if current == normalized:
            return False
        if current:
            raise RuntimeError(
                "document_section_map 已初始化；映射变化必须重建整份缓存"
            )
        if self.sections or self.section_metadata:
            raise RuntimeError("存在派生字段时不能初始化 document_section_map")
        self.payload["document_section_map"] = normalized
        self.payload["preparation_completed"] = False
        return True

    def mark_preparation_started(self) -> None:
        self.payload["preparation_completed"] = False

    def mark_preparation_completed(self) -> None:
        self.payload["preparation_completed"] = True

    @staticmethod
    def _section_key(section_id: int) -> str:
        return str(int(section_id))

    def _section(self, section_id: int) -> Dict[str, Any]:
        key = self._section_key(section_id)
        section = self.sections.setdefault(key, {})
        if not isinstance(section, dict):
            raise TypeError(f"section_id={section_id} 的缓存必须是对象")
        return section

    def _metadata(self, section_id: int) -> Dict[str, Any]:
        key = self._section_key(section_id)
        metadata = self.section_metadata.setdefault(
            key,
            {"document_type": None},
        )
        if not isinstance(metadata, dict):
            raise TypeError(f"section_id={section_id} 的元数据必须是对象")
        document_type = metadata.setdefault("document_type", None)
        if document_type is not None and (
            not isinstance(document_type, str) or not document_type
        ):
            raise TypeError(
                f"section_id={section_id} 的 document_type "
                "必须是非空字符串或 null"
            )
        return metadata

    def register_section(self, section_id: int, document_type: str) -> None:
        if not isinstance(document_type, str) or not document_type:
            raise TypeError("文书类型必须是非空字符串")
        self._section(section_id)
        metadata = self._metadata(section_id)
        existing_type = metadata["document_type"]
        if existing_type is not None and existing_type != document_type:
            raise ValueError(
                f"section_id={section_id} 已登记为文书类型 {existing_type}，"
                f"不能再次登记为 {document_type}"
            )
        metadata["document_type"] = document_type

    def document_type(self, section_id: int) -> str:
        document_type = self._metadata(section_id)["document_type"]
        if document_type is None:
            raise KeyError(f"section_id={section_id} 尚未登记文书类型")
        return document_type

    def bind_field_extractor(self, extractor: FieldExtractor) -> None:
        self._field_extractor = extractor

    def bind_delivery_field_extractor(
        self,
        extractor: DeliveryFieldExtractor,
    ) -> None:
        self._delivery_field_extractor = extractor

    def missing_fields(
        self,
        section_id: int,
        field_names: Iterable[str],
    ) -> List[str]:
        section = self._section(section_id)
        return [name for name in field_names if name not in section]

    def has_field(self, section_id: int, field_name: str) -> bool:
        return field_name in self._section(section_id)

    def peek_field(self, section_id: int, field_name: str) -> Any:
        return self._section(section_id)[field_name]

    def merge_fields(
        self,
        section_id: int,
        fields: Dict[str, Any],
    ) -> None:
        self._section(section_id).update(fields)

    async def get_field(self, section_id: int, field_name: str) -> Any:
        values = await self.get_fields(section_id, [field_name])
        return values[field_name]

    async def get_fields(
        self,
        section_id: int,
        field_names: Iterable[str],
    ) -> Dict[str, Any]:
        """透明读取普通字段；未命中时立即提取并同步写回。"""

        requested = list(dict.fromkeys(field_names))
        tasks: set[asyncio.Task[Any]] = set()
        async with self._state_lock:
            section = self._section(section_id)
            new_fields = []
            for field_name in requested:
                if field_name in section:
                    continue
                key = (int(section_id), field_name)
                task = self._field_inflight.get(key)
                if task is None:
                    new_fields.append(field_name)
                else:
                    tasks.add(task)

            if new_fields:
                if self._field_extractor is None:
                    raise RuntimeError("普通字段提取器尚未绑定")
                task = asyncio.create_task(
                    self._extract_and_store_fields(section_id, new_fields)
                )
                for field_name in new_fields:
                    self._field_inflight[(int(section_id), field_name)] = task
                tasks.add(task)

        if tasks:
            await asyncio.gather(*tasks)
        return {
            field_name: self._section(section_id)[field_name]
            for field_name in requested
        }

    async def _extract_and_store_fields(
        self,
        section_id: int,
        field_names: List[str],
    ) -> None:
        task = asyncio.current_task()
        try:
            if self._field_extractor is None:
                raise RuntimeError("普通字段提取器尚未绑定")
            extracted = await self._field_extractor(section_id, field_names)
            if set(extracted) != set(field_names):
                raise ValueError("普通字段提取结果与请求字段不一致")
            async with self._state_lock:
                payload = copy.deepcopy(self.payload)
                payload["sections"].setdefault(
                    self._section_key(section_id), {}
                ).update(extracted)
                atomic_write_json(self.path, payload)
                self.payload = payload
        finally:
            async with self._state_lock:
                for field_name in field_names:
                    key = (int(section_id), field_name)
                    if self._field_inflight.get(key) is task:
                        self._field_inflight.pop(key, None)

    @staticmethod
    def delivery_event_storage_key(event: Dict[str, Any]) -> str:
        related_section_id = event.get("related_section_id")
        if (
            isinstance(related_section_id, int)
            and not isinstance(related_section_id, bool)
            and related_section_id > 0
        ):
            return str(related_section_id)
        return f"{EVENT_KEY_PREFIX}{int(event['event_id'])}"

    @staticmethod
    def _is_delivery_event(value: Any) -> bool:
        return (
            isinstance(value, dict)
            and isinstance(value.get("event_id"), int)
            and not isinstance(value.get("event_id"), bool)
        )

    def delivery_events(self, receipt_section_id: int) -> List[Dict[str, Any]]:
        events = [
            value
            for value in self._section(receipt_section_id).values()
            if self._is_delivery_event(value)
        ]
        return sorted(events, key=lambda event: event["source_order"])

    def delivery_event_count(self) -> int:
        return sum(
            len(self.delivery_events(int(section_id)))
            for section_id, metadata in self.section_metadata.items()
            if metadata.get("delivery_extraction_completed") is True
        )

    def next_event_id(self) -> int:
        event_ids = [
            event["event_id"]
            for section_id, metadata in self.section_metadata.items()
            if metadata.get("delivery_extraction_completed") is True
            for event in self.delivery_events(int(section_id))
        ]
        return max(event_ids, default=0) + 1

    def delivery_extraction_completed(self, section_id: int) -> bool:
        return self._metadata(section_id).get(
            "delivery_extraction_completed"
        ) is True

    def set_delivery_events(
        self,
        receipt_section_id: int,
        events: List[Dict[str, Any]],
    ) -> None:
        event_ids = [event["event_id"] for event in events]
        source_orders = [event["source_order"] for event in events]
        if any(
            isinstance(event_id, bool)
            or not isinstance(event_id, int)
            or event_id <= 0
            for event_id in event_ids
        ):
            raise ValueError("送达事件 event_id 必须是正整数")
        if any(
            isinstance(source_order, bool)
            or not isinstance(source_order, int)
            or source_order <= 0
            for source_order in source_orders
        ):
            raise ValueError("送达事件 source_order 必须是正整数")
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("送达事件 event_id 不得重复")
        if len(source_orders) != len(set(source_orders)):
            raise ValueError("同一送达回证的事件顺序不得重复")

        related_ids = []
        for event in events:
            if "related_section_id" not in event:
                raise ValueError(
                    "送达事件写入缓存前必须完成映射，不允许缺少 "
                    "related_section_id"
                )
            event_text = event.get("event_text")
            if not isinstance(event_text, str) or not event_text.strip():
                raise ValueError("送达事件 event_text 必须是非空字符串")
            related_section_id = event.get("related_section_id")
            if related_section_id == TECHNICAL_MAPPING_FAILURE:
                continue
            if related_section_id is None:
                continue
            if (
                isinstance(related_section_id, bool)
                or not isinstance(related_section_id, int)
                or related_section_id <= 0
            ):
                raise ValueError("related_section_id 必须为正整数、null 或 -1")
            related_ids.append(related_section_id)
        if len(related_ids) != len(set(related_ids)):
            raise ValueError("同一文书不能对应当前回证中的多个送达事件")

        section = self._section(receipt_section_id)
        for key in [
            key
            for key, value in section.items()
            if self._is_delivery_event(value)
        ]:
            del section[key]
        for event in sorted(events, key=lambda item: item["source_order"]):
            key = self.delivery_event_storage_key(event)
            if key in section:
                raise ValueError(f"送达事件缓存键冲突: {key}")
            section[key] = dict(event)
        self._metadata(receipt_section_id)[
            "delivery_extraction_completed"
        ] = True

    def delivery_event_for_document(
        self,
        document_section_id: int,
    ) -> Dict[str, Any] | None:
        for receipt_section_id, metadata in self.section_metadata.items():
            if metadata.get("delivery_extraction_completed") is not True:
                continue
            event = self._section(int(receipt_section_id)).get(
                self._section_key(document_section_id)
            )
            if self._is_delivery_event(event):
                return event
        return None

    def delivery_event(
        self,
        receipt_section_id: int,
        relation_key: int | str,
    ) -> Dict[str, Any]:
        event = self._section(receipt_section_id)[str(relation_key)]
        if not self._is_delivery_event(event):
            raise KeyError(
                f"section_id={receipt_section_id} 中不存在送达事件 {relation_key}"
            )
        return event

    async def get_delivery_field(
        self,
        receipt_section_id: int,
        relation_key: int | str,
        field_name: str,
    ) -> Any:
        values = await self.get_delivery_fields(
            receipt_section_id,
            relation_key,
            [field_name],
        )
        return values[field_name]

    async def get_delivery_fields(
        self,
        receipt_section_id: int,
        relation_key: int | str,
        field_names: Iterable[str],
    ) -> Dict[str, Any]:
        """透明读取单个送达事件字段；未命中时立即补提取。"""

        relation_key = str(relation_key)
        requested = list(dict.fromkeys(field_names))
        tasks: set[asyncio.Task[Any]] = set()
        async with self._state_lock:
            event = self.delivery_event(receipt_section_id, relation_key)
            new_fields = []
            for field_name in requested:
                if field_name in event:
                    continue
                key = (int(receipt_section_id), relation_key, field_name)
                task = self._delivery_field_inflight.get(key)
                if task is None:
                    new_fields.append(field_name)
                else:
                    tasks.add(task)

            if new_fields:
                if self._delivery_field_extractor is None:
                    raise RuntimeError("送达事件字段提取器尚未绑定")
                task = asyncio.create_task(
                    self._extract_and_store_delivery_fields(
                        receipt_section_id,
                        relation_key,
                        new_fields,
                    )
                )
                for field_name in new_fields:
                    key = (
                        int(receipt_section_id),
                        relation_key,
                        field_name,
                    )
                    self._delivery_field_inflight[key] = task
                tasks.add(task)

        if tasks:
            await asyncio.gather(*tasks)
        event = self.delivery_event(receipt_section_id, relation_key)
        return {field_name: event[field_name] for field_name in requested}

    async def _extract_and_store_delivery_fields(
        self,
        receipt_section_id: int,
        relation_key: str,
        field_names: List[str],
    ) -> None:
        task = asyncio.current_task()
        try:
            if self._delivery_field_extractor is None:
                raise RuntimeError("送达事件字段提取器尚未绑定")
            extracted = await self._delivery_field_extractor(
                receipt_section_id,
                relation_key,
                field_names,
            )
            if set(extracted) != set(field_names):
                raise ValueError("送达事件字段提取结果与请求字段不一致")
            async with self._state_lock:
                payload = copy.deepcopy(self.payload)
                event = payload["sections"][
                    self._section_key(receipt_section_id)
                ][relation_key]
                event.update(extracted)
                atomic_write_json(self.path, payload)
                self.payload = payload
        finally:
            async with self._state_lock:
                for field_name in field_names:
                    key = (
                        int(receipt_section_id),
                        relation_key,
                        field_name,
                    )
                    if self._delivery_field_inflight.get(key) is task:
                        self._delivery_field_inflight.pop(key, None)

    def save(self) -> None:
        atomic_write_json(self.path, self.payload)
