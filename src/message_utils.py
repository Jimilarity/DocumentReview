from pathlib import Path
from typing import Any, Literal, Sequence

from langchain_core.messages import HumanMessage

from utils import image_to_base64


ImagePath = str | Path
ImagePosition = Literal["before_text", "after_text"]


def build_image_content(image_path: ImagePath) -> dict[str, Any]:
    suffix = Path(image_path).suffix.lower()
    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")

    image_url = (
        f"data:{mime_type};base64,"
        f"{image_to_base64(image_path)}"
    )

    return {
        "type": "image_url",
        "image_url": {"url": image_url},
    }


def build_vision_message(
    prompt_text: str,
    image_paths: ImagePath | Sequence[ImagePath],
    *,
    additional_texts: Sequence[str] = (),
    image_position: ImagePosition = "before_text",
) -> HumanMessage:
    """构造多模态消息，默认先放图片，再放上下文和任务提示词。"""

    if isinstance(image_paths, (str, Path)):
        image_paths = [image_paths]

    image_contents = [
        build_image_content(image_path)
        for image_path in image_paths
    ]
    additional_text_contents = [
        {"type": "text", "text": text}
        for text in additional_texts
    ]
    prompt_content = {"type": "text", "text": prompt_text}

    if image_position == "before_text":
        content = [
            *image_contents,
            *additional_text_contents,
            prompt_content,
        ]
    elif image_position == "after_text":
        content = [
            prompt_content,
            *additional_text_contents,
            *image_contents,
        ]
    else:
        raise ValueError(f"不支持的 image_position: {image_position}")

    return HumanMessage(content=content)
