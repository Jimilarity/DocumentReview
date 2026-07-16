from typing import Any, Dict, List


def normalize_directory_info(
    dir_info: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return directory entries with numeric identifiers and page indexes."""

    normalized_items = []
    for item in dir_info:
        normalized_item = dict(item)
        normalized_item["section_id"] = int(normalized_item["section_id"])
        normalized_item["section_page"] = int(
            normalized_item["section_page"]
        )
        normalized_items.append(normalized_item)
    return normalized_items
