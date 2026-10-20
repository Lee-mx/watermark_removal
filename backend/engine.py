"""Local image validation and masked repair. Originals are never modified."""

from __future__ import annotations

import base64
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import cv2
import numpy as np
from .repair import repair_pixels


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
BASE_SIZE = (960, 1280)
DEFAULT_TEMPLATE = "account-11578066187"
TEMPLATES = {
    DEFAULT_TEMPLATE: "watermark_mask_11578066187.png",
    "account-267777923": "watermark_mask_267777923.png",
    "legacy": "watermark_mask_960x1280.png",
}


def asset_path(name: str) -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "assets" / name  # type: ignore[attr-defined]
    if name == "inpainting_lama_2025jan.onnx":
        return Path(__file__).resolve().parents[1] / "models" / name
    return Path(__file__).resolve().parents[2] / "tools" / name


def read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("仅支持 PNG、JPG 和 WebP 图片")
    if not path.is_file():
        raise ValueError("文件不存在或已被移动")
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8:
        raise ValueError("图片无法解码，或不是 8 位图片")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
        raise ValueError("图片通道格式不受支持")
    return image


def can_use_template(width: int, height: int) -> bool:
    return abs(width / height - 0.75) <= 0.018


def build_mask(width: int, height: int, selection: dict[str, Any]) -> np.ndarray:
    mode = selection.get("mode", "template")
    if mode == "template":
        if not can_use_template(width, height):
            raise ValueError("这张图片不是适配内置掩膜的 3:4 竖图，请手动圈定水印区域")
        template_id = selection.get("templateId", DEFAULT_TEMPLATE)
        if template_id not in TEMPLATES:
            raise ValueError("未知的水印模板")
        template = cv2.imdecode(
            np.fromfile(str(asset_path(TEMPLATES[template_id])), dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE,
        )
        if template is None or template.shape != (BASE_SIZE[1], BASE_SIZE[0]):
            raise ValueError("内置掩膜缺失或尺寸错误")
        mask = cv2.resize(template, (width, height), interpolation=cv2.INTER_NEAREST)
        if template_id == DEFAULT_TEMPLATE:
            # Include faint antialiased glyph outlines, calibrated on matching originals.
            padding = max(1, round(width / 640))
            mask = cv2.dilate(mask, np.ones((padding * 2 + 1, padding * 2 + 1), np.uint8))
        elif template_id == "account-267777923":
            # The visible anti-aliased halo extends beyond the template's glyph core.
            padding = max(1, round(width * 6 / 1280))
            mask = cv2.dilate(mask, np.ones((padding * 2 + 1, padding * 2 + 1), np.uint8))
        return (mask > 0).astype(np.uint8) * 255
    if mode != "regions":
        raise ValueError("未知的区域选择方式")
    regions = selection.get("regions", [])
    if not isinstance(regions, list) or not 1 <= len(regions) <= 12:
        raise ValueError("请在图片上圈定 1～12 个区域")
    mask = np.zeros((height, width), dtype=np.uint8)
    for rect in regions:
        if not isinstance(rect, dict):
            raise ValueError("区域格式错误")
        try:
            x, y, w, h = (float(rect[k]) for k in ("x", "y", "w", "h"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("区域坐标错误") from exc
        if not all(np.isfinite([x, y, w, h])) or not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 - x and 0 < h <= 1 - y):
            raise ValueError("区域坐标超出图片范围")
        left = max(0, round(x * width))
        top = max(0, round(y * height))
        right = min(width, round((x + w) * width))
        bottom = min(height, round((y + h) * height))
        if right <= left or bottom <= top:
            raise ValueError("所选区域太小")
        mask[top:bottom, left:right] = 255
    return mask


def suggested_template(image: np.ndarray) -> str | None:
    """Only suggest a known account when its footer glyphs align clearly."""
    height, width = image.shape[:2]
    if not can_use_template(width, height):
        return None
    color = image[:, :, :3] if image.ndim == 3 and image.shape[2] == 4 else image
    gray = color if color.ndim == 2 else cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    detail = gray - cv2.GaussianBlur(gray, (0, 0), 5)
    row = np.zeros((height, width), dtype=np.uint8)
    row[round(height * .961):round(height * .986), round(width * .71):] = 1
    scores: dict[str, float] = {}
    for template_id in (DEFAULT_TEMPLATE, "account-267777923"):
        core = cv2.imdecode(np.fromfile(str(asset_path(TEMPLATES[template_id])), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if core is None:
            return None
        mask = cv2.resize(core, (width, height), interpolation=cv2.INTER_NEAREST)
        sample = (mask > 0) & (row > 0)
        scores[template_id] = float(detail[sample].mean()) if np.any(sample) else 0.0
    if scores["account-267777923"] >= 5.0 and scores["account-267777923"] > scores[DEFAULT_TEMPLATE] * 1.4:
        return "account-267777923"
    if scores[DEFAULT_TEMPLATE] >= 5.0 and scores[DEFAULT_TEMPLATE] > scores["account-267777923"] * 1.4:
        return DEFAULT_TEMPLATE
    return None


def data_url(image: np.ndarray, max_side: int = 360) -> str:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1:
        image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    if image.ndim == 3 and image.shape[2] == 4:
        ext = ".png"
        mime = "image/png"
        options: list[int] = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    else:
        ext = ".jpg"
        mime = "image/jpeg"
        options = [cv2.IMWRITE_JPEG_QUALITY, 80]
    ok, encoded = cv2.imencode(ext, image, options)
    if not ok:
        raise ValueError("无法生成缩略图")
    return f"data:{mime};base64,{base64.b64encode(encoded).decode('ascii')}"


def mask_preview(width: int, height: int, template_id: str = DEFAULT_TEMPLATE) -> str | None:
    if not can_use_template(width, height):
        return None
    mask = build_mask(width, height, {"mode": "template", "templateId": template_id})
    scale = min(1.0, 720 / max(width, height))
    small = cv2.resize(mask, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_NEAREST)
    overlay = np.zeros((*small.shape, 4), dtype=np.uint8)
    overlay[:, :, 2] = 240
    overlay[:, :, 3] = np.where(small > 0, 125, 0).astype(np.uint8)
    ok, encoded = cv2.imencode(".png", overlay)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")


def inspect(path: Path) -> dict[str, Any]:
    image = read_image(path)
    height, width = image.shape[:2]
    suggested = suggested_template(image)
    template_id = suggested or DEFAULT_TEMPLATE
    return {
        "name": path.name,
        "path": str(path),
        "width": width,
        "height": height,
        "bytes": path.stat().st_size,
        "preview": data_url(image),
        "maskPreview": mask_preview(width, height, template_id),
        "defaultTemplate": template_id,
        "defaultMode": "template" if suggested else "regions",
    }


def unique_target(directory: Path, stem: str) -> Path:
    candidate = directory / f"{stem}_clean.png"
    index = 2
    while candidate.exists():
        candidate = directory / f"{stem}_clean_{index}.png"
        index += 1
    return candidate


def repair_to_file(source: Path, directory: Path, selection: dict[str, Any], output_stem: str | None = None) -> dict[str, Any]:
    image = read_image(source)
    height, width = image.shape[:2]
    mask = build_mask(width, height, selection)
    if not np.any(mask):
        raise ValueError("去除区域为空")
    is_lama_template = selection.get("mode") == "template" and selection.get("templateId") == "account-267777923"
    method = "lama" if is_lama_template else "adaptive" if selection.get("mode") == "regions" else "edge-aware"
    result = repair_pixels(image, mask, method=method, model_path=asset_path("inpainting_lama_2025jan.onnx") if is_lama_template else None)
    directory.mkdir(parents=True, exist_ok=True)
    target = unique_target(directory, output_stem or source.stem)
    if target.resolve() == source.resolve():
        raise ValueError("输出路径与原图相同")
    ok, encoded = cv2.imencode(".png", result, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ValueError("PNG 编码失败")
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".tmp", prefix=".corner-clean-", dir=directory, delete=False) as temporary:
            temp_name = temporary.name
            temporary.write(encoded.tobytes())
        check = cv2.imdecode(np.fromfile(temp_name, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if check is None or check.shape != image.shape or np.count_nonzero(check[mask == 0] != image[mask == 0]):
            raise ValueError("保存后校验失败：尺寸或掩膜外像素发生变化")
        os.replace(temp_name, target)
        temp_name = None
    finally:
        if temp_name and Path(temp_name).exists():
            Path(temp_name).unlink()
    return {"path": str(target), "name": target.name, "preview": data_url(result), "width": width, "height": height}
