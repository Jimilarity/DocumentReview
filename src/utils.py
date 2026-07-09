import base64
import yaml
import json
import re
import aiofiles

from typing import Annotated, Any, Dict, List, Optional, TypedDict
from PIL import Image
from pathlib import Path


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def image_to_base64(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode(encoding='utf-8')
        
def combine_images(image_list, canvas_name,
                   max_width=1200,
                   max_height=8000,
                   jpeg_quality=85):

    output_folder = Path(image_list[0]).parent
    canvas_path = output_folder / canvas_name

    images = []
    for p in image_list:
        img = Image.open(p).convert("RGB")
        if img.width > max_width:
            ratio = max_width / img.width
            new_size = (max_width, int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        images.append(img)

    max_w = max(img.width for img in images)
    total_h = sum(img.height for img in images)

    if total_h > max_height:
        ratio = max_height / total_h
        resized = []
        for img in images:
            new_size = (int(img.width * ratio), int(img.height * ratio))
            resized.append(img.resize(new_size, Image.LANCZOS))
        images = resized
        max_w = max(img.width for img in images)
        total_h = sum(img.height for img in images)

    long_canvas = Image.new("RGB", (max_w, total_h), (255, 255, 255))

    current_height = 0
    for img in images:
        long_canvas.paste(img, (0, current_height))
        current_height += img.height

    long_canvas.save(canvas_path, "JPEG", quality=jpeg_quality, optimize=True)

    return canvas_path

def atomic_write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        
    tmp_path.replace(path)
    
def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


async def async_read_json(path: str | Path) -> Any:
    async with aiofiles.open(path, "r", encoding="utf-8") as file:
        return json.loads(await file.read())


def extract_json(text: str) -> Any:
    text = text.strip()

    match = re.search(
        r"```(?:json)?\s*([\s\S]*?)\s*```",
        text,
        re.IGNORECASE,
    )
    if match:
        text = match.group(1).strip()

    return json.loads(text)


def model_to_dict(
    model: Any,
    *,
    by_alias: bool = False,
) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(by_alias=by_alias)

    if hasattr(model, "dict"):
        return model.dict(by_alias=by_alias)

    raise TypeError(
        f"无法将对象转换为字典: {type(model).__name__}"
    )


def sanitize_filename(
    filename: str,
    replacement: str = "_",
) -> str:
    filename = re.sub(
        r'[\\/*?:"<>|]',
        replacement,
        filename,
    )
    return filename.strip().rstrip(".")




 