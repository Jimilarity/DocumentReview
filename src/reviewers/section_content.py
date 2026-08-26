from typing import Any, Dict, List


def section_page_range(
    section_id: int,
    dir_info: List[Dict[str, Any]],
    page_count: int,
) -> tuple[int, int]:
    section_index = next(
        index
        for index, item in enumerate(dir_info)
        if int(item["section_id"]) == int(section_id)
    )
    start_page = int(dir_info[section_index]["section_page"])
    end_page = (
        int(dir_info[section_index + 1]["section_page"])
        if section_index + 1 < len(dir_info)
        else page_count
    )
    return start_page, end_page


def extract_sections_ocr_text(
    section_ids: List[int],
    dir_info: List[Dict[str, Any]],
    ocr_results: List[Dict[str, Any]],
) -> str:
    ocr_by_page = {
        int(item["image_index"]): str(item["document_content"])
        for item in ocr_results
    }
    directory_order = {
        int(item["section_id"]): index
        for index, item in enumerate(dir_info)
    }
    ordered_section_ids = sorted(
        set(section_ids),
        key=directory_order.__getitem__,
    )

    section_texts = []
    for section_id in ordered_section_ids:
        section = dir_info[directory_order[section_id]]
        start_page, end_page = section_page_range(
            section_id,
            dir_info,
            len(ocr_results),
        )
        page_texts = [
            f"[page_index={page_index}]\n{ocr_by_page[page_index]}"
            for page_index in range(start_page, end_page)
        ]
        catalog_source = str(
            section.get("catalog_source") or "original"
        )
        metadata_source = (
            "ocr_segmentation"
            if catalog_source == "ocr_segmented"
            else "case_directory"
        )
        section_texts.append(
            "\n".join(
                [
                    (
                        "[directory_metadata: "
                        f"section_id={section_id}, "
                        f"section_name={section['section_name']}, "
                        f"source={metadata_source}, is_page_title=false]"
                    ),
                    *page_texts,
                ]
            )
        )

    return "\n\n".join(section_texts)


def extract_section_ocr_text(
    section_id: int,
    dir_info: List[Dict[str, Any]],
    ocr_results: List[Dict[str, Any]],
) -> str:
    return extract_sections_ocr_text(
        [section_id],
        dir_info,
        ocr_results,
    )
