"""为规则 113 写入可恢复的“不予处罚依据不足”测试缓存。"""

import argparse
import shutil
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_ROOT))

from cache_paths import get_cache_paths
from structured_field_cache import build_structured_source_fingerprint
from utils import atomic_write_json, read_json


BACKUP_META_SUFFIX = ".rule-113-before-test.json"
BACKUP_FIELDS_SUFFIX = ".rule-113-before-test.json"
BACKUP_OCR_SUFFIX = ".rule-113-before-test.json"


def _backup_once(path: Path, suffix: str) -> Path:
    backup = path.with_name(path.stem + suffix)
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def _replace_decision_ocr(ocr_results: list[dict]) -> None:
    for page in ocr_results:
        if page.get("image_index") != 2:
            continue
        content = page.get("document_content")
        if not isinstance(content, str):
            raise TypeError("行政处罚决定书 OCR 必须是字符串")
        replacements = {
            "决定对你（单位）作出如下行政处罚：": (
                "本机关决定对你（单位）不予行政处罚。"
            ),
            "☑3.罚款： 罚款人民币肆仟元整（￥4000.00）": (
                "☑13.法律、行政法规规定的其他行政处罚：不予行政处罚"
            ),
            "☑你（单位）应当自收到本决定书之日起15日内将罚（没）款缴至指定银行（详见相关通知书）。□到期不缴纳罚款的，依据《中华人民共和国行政处罚法》第七十二条第一款第（一）项的规定，每日按罚款数额的3%加处罚款，加处罚款的数额不超过罚款的数额。": (
                "本决定不涉及罚款缴纳。"
            ),
        }
        for source, target in replacements.items():
            if source not in content:
                raise ValueError(f"未在行政处罚决定书 OCR 中找到测试替换文本: {source}")
            content = content.replace(source, target)
        page["document_content"] = content
        return
    raise LookupError("未找到行政处罚决定书对应的 OCR 页")


def apply_test_data(pdf_path: str) -> tuple[Path, Path, Path]:
    cache_paths = get_cache_paths(pdf_path)
    required_paths = (
        cache_paths.metadata,
        cache_paths.ocr_results,
        cache_paths.structured_fields,
    )
    if not all(path.is_file() for path in required_paths):
        raise FileNotFoundError("需要先完成预处理和结构化字段准备")

    metadata_backup = _backup_once(
        cache_paths.metadata,
        BACKUP_META_SUFFIX,
    )
    fields_backup = _backup_once(
        cache_paths.structured_fields,
        BACKUP_FIELDS_SUFFIX,
    )
    ocr_backup = _backup_once(cache_paths.ocr_results, BACKUP_OCR_SUFFIX)
    metadata = read_json(cache_paths.metadata)
    fields = read_json(cache_paths.structured_fields)
    directory = read_json(cache_paths.directory)
    ocr_results = read_json(cache_paths.ocr_results)

    metadata["案发日期"] = "2022-01-10"
    metadata["处理结果"] = "不予行政处罚。"
    _replace_decision_ocr(ocr_results)
    sections = fields["sections"]
    decision = sections["1"]
    decision.update(
        {
            "处罚决定种类": "不予行政处罚",
            "处罚具体内容": [
                {"种类": "不予行政处罚", "内容": "决定不予行政处罚。"}
            ],
            "违法事实": (
                "当事人未安排专人对施工现场进出路口及出场车辆进行"
                "冲洗和清理。"
            ),
            "陈述、申辩处理结果": "未提交陈述、申辩材料。",
        }
    )
    closing = sections["32"]
    closing.update(
        {
            "结案类型": "不予行政处罚结案",
            "案件简要情况": "当事人存在未冲洗、清理施工车辆的行为。",
            "承办人意见": "建议不予行政处罚。",
            "审批负责人审批意见": "同意。",
        }
    )
    fields["document_presence"]["行政机关负责人集体讨论笔录"] = False
    fields["source_fingerprint"] = build_structured_source_fingerprint(
        metadata,
        directory,
        ocr_results,
    )
    fields["preparation_completed"] = True
    atomic_write_json(cache_paths.metadata, metadata)
    atomic_write_json(cache_paths.ocr_results, ocr_results)
    atomic_write_json(cache_paths.structured_fields, fields)
    return metadata_backup, ocr_backup, fields_backup


def restore_test_data(pdf_path: str) -> None:
    cache_paths = get_cache_paths(pdf_path)
    metadata_backup = cache_paths.metadata.with_name(
        cache_paths.metadata.stem + BACKUP_META_SUFFIX
    )
    fields_backup = cache_paths.structured_fields.with_name(
        cache_paths.structured_fields.stem + BACKUP_FIELDS_SUFFIX
    )
    ocr_backup = cache_paths.ocr_results.with_name(
        cache_paths.ocr_results.stem + BACKUP_OCR_SUFFIX
    )
    if not all(
        path.is_file() for path in (metadata_backup, ocr_backup, fields_backup)
    ):
        raise FileNotFoundError("找不到规则 113 测试缓存备份")
    shutil.copy2(metadata_backup, cache_paths.metadata)
    shutil.copy2(ocr_backup, cache_paths.ocr_results)
    shutil.copy2(fields_backup, cache_paths.structured_fields)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-path", required=True)
    parser.add_argument("--restore", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.restore:
        restore_test_data(args.file_path)
        print("已恢复规则 113 测试前的 meta_info 与 structured_fields 缓存。")
        return
    metadata_backup, ocr_backup, fields_backup = apply_test_data(args.file_path)
    print("已写入规则 113 测试缓存：不予行政处罚但未体现法定情形或集体讨论依据。")
    print(f"元数据备份：{metadata_backup}")
    print(f"OCR 备份：{ocr_backup}")
    print(f"结构化字段备份：{fields_backup}")


if __name__ == "__main__":
    main()
