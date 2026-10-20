"""Mask-limited repair with directional interpolation across strong edges."""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def _edge_direction(color: np.ndarray, excluded: np.ndarray, bounds: tuple[int, int, int, int]):
    x, y, width, height = bounds
    margin = max(20, min(90, max(width, height)))
    left, top = max(0, x - margin), max(0, y - margin)
    right, bottom = min(color.shape[1], x + width + margin), min(color.shape[0], y + height + margin)
    crop = color[top:bottom, left:right]
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0.8)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    valid = excluded[top:bottom, left:right] == 0
    valid[:3] = False
    valid[-3:] = False
    valid[:, :3] = False
    valid[:, -3:] = False
    if np.count_nonzero(valid) < 50:
        return None
    threshold = max(35.0, float(np.percentile(magnitude[valid], 90)))
    strong = valid & (magnitude > threshold)
    if np.count_nonzero(strong) < 30:
        return None
    sx, sy = gx[strong], gy[strong]
    tensor = np.array([[np.dot(sx, sx), np.dot(sx, sy)], [np.dot(sx, sy), np.dot(sy, sy)]])
    values, vectors = np.linalg.eigh(tensor)
    coherence = float((values[1] - values[0]) / max(1.0, values.sum()))
    if coherence < 0.88:
        return None
    normal = vectors[:, 1]
    direction = np.array([-normal[1], normal[0]], dtype=np.float32)
    ys, xs = np.where(strong)
    points = np.column_stack((xs + left, ys + top))
    projected = points @ direction
    center = np.array([x + width / 2, y + height / 2]) @ direction
    # An edge must be observed on both ends, not just somewhere in the vicinity.
    span = abs(direction[0]) * width / 2 + abs(direction[1]) * height / 2
    if not (np.any(projected < center - span * .65) and np.any(projected > center + span * .65)):
        return None
    return direction


def _directional_values(color: np.ndarray, excluded: np.ndarray, ys: np.ndarray, xs: np.ndarray, direction: np.ndarray, limit: int):
    samples = []
    distances = []
    image_height, image_width = color.shape[:2]
    for sign in (-1, 1):
        distance = np.zeros(len(xs), np.float32)
        donor_x = xs.astype(np.float32).copy()
        donor_y = ys.astype(np.float32).copy()
        for step in range(1, limit + 1):
            unresolved = np.flatnonzero(distance == 0)
            if not len(unresolved):
                break
            px = xs[unresolved] + direction[0] * step * sign
            py = ys[unresolved] + direction[1] * step * sign
            ix, iy = np.rint(px).astype(int), np.rint(py).astype(int)
            inside = (ix >= 1) & (ix < image_width - 1) & (iy >= 1) & (iy < image_height - 1)
            good = inside.copy()
            good[inside] &= excluded[iy[inside], ix[inside]] == 0
            accepted = unresolved[good]
            distance[accepted] = step
            donor_x[accepted] = px[good]
            donor_y[accepted] = py[good]
        # OpenCV remap limits either destination dimension to SHRT_MAX.
        # Interpolate the sparse donor coordinates directly, including large selections.
        x0, y0 = np.floor(donor_x).astype(int), np.floor(donor_y).astype(int)
        x1, y1 = np.minimum(x0 + 1, image_width - 1), np.minimum(y0 + 1, image_height - 1)
        wx, wy = donor_x - x0, donor_y - y0
        if color.ndim == 3:
            wx, wy = wx[:, None], wy[:, None]
        values = (color[y0, x0] * (1 - wx) * (1 - wy)
                  + color[y0, x1] * wx * (1 - wy)
                  + color[y1, x0] * (1 - wx) * wy
                  + color[y1, x1] * wx * wy)
        samples.append(values)
        distances.append(distance)
    valid = (distances[0] > 0) & (distances[1] > 0)
    weight = distances[1] / np.maximum(distances[0] + distances[1], 1)
    if color.ndim == 3:
        weight = weight[:, None]
    value = samples[0] * weight + samples[1] * (1 - weight)
    return valid, value


def _lama_repair(color: np.ndarray, mask: np.ndarray, model_path: Path) -> np.ndarray:
    """Run the offline model on a footer crop, then copy back masked pixels only."""
    if not model_path.is_file():
        raise ValueError("离线修补模型缺失，请重新安装完整应用文件夹")
    ys, xs = np.nonzero(mask)
    margin = 90
    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(color.shape[1], int(xs.max()) + margin + 1)
    y1 = min(color.shape[0], int(ys.max()) + margin + 1)
    crop = color[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]
    try:
        net = cv2.dnn.readNetFromONNX(str(model_path))
        image_blob = cv2.dnn.blobFromImage(crop, 1 / 255, (512, 512), (0, 0, 0), False, False)
        mask_blob = cv2.dnn.blobFromImage(crop_mask, 1, (512, 512), (0,), False, False)
        net.setInput(image_blob, "image")
        net.setInput((mask_blob > 0).astype(np.float32), "mask")
        output = net.forward()[0]
    except cv2.error as exc:
        raise ValueError("离线修补模型运行失败，请确认安装包完整") from exc
    if output.shape != (3, 512, 512):
        raise ValueError("离线修补模型返回了意外的图片尺寸")
    prediction = np.transpose(output, (1, 2, 0))
    prediction = np.clip(prediction, 0, 255).astype(np.uint8)
    prediction = cv2.resize(prediction, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
    repaired = color.copy()
    repaired_crop = repaired[y0:y1, x0:x1]
    repaired_crop[crop_mask > 0] = prediction[crop_mask > 0]
    return repaired


def _smooth_background_repair(color: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Interpolate a smooth rectangle from its four sides, without flattening it to one color."""
    if color.ndim != 3 or color.shape[2] != 3:
        return None
    count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    if count != 2:
        return None
    x, y, width, height, area = map(int, stats[1])
    if min(width, height) < 18 or area < 1000 or area != width * height:
        return None
    if x < 1 or y < 1 or x + width >= color.shape[1] or y + height >= color.shape[0]:
        return None

    # Use this only where the surrounding image is low-texture. A photograph or
    # structural edge should continue through the existing edge-aware path.
    margin = 24
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(color.shape[1], x + width + margin), min(color.shape[0], y + height + margin)
    collar = color[y0:y1, x0:x1].astype(np.float32)
    residual = np.max(np.abs(collar - cv2.GaussianBlur(collar, (0, 0), 3)), axis=2)
    outside = mask[y0:y1, x0:x1] == 0
    if np.count_nonzero(outside) < 100 or np.percentile(residual[outside], 90) > 5:
        return None

    top = color[y - 1, x:x + width].astype(np.float32)
    bottom = color[y + height, x:x + width].astype(np.float32)
    left = color[y:y + height, x - 1].astype(np.float32)
    right = color[y:y + height, x + width].astype(np.float32)
    top = cv2.GaussianBlur(top[None], (17, 1), 3)[0]
    bottom = cv2.GaussianBlur(bottom[None], (17, 1), 3)[0]
    left = cv2.GaussianBlur(left[:, None], (1, 17), 3)[:, 0]
    right = cv2.GaussianBlur(right[:, None], (1, 17), 3)[:, 0]
    sx = (np.arange(width, dtype=np.float32) + 1)[None, :, None] / (width + 1)
    sy = (np.arange(height, dtype=np.float32) + 1)[:, None, None] / (height + 1)
    vertical = (1 - sy) * top[None] + sy * bottom[None]
    horizontal = (1 - sx) * left[:, None] + sx * right[:, None]
    tl = color[y - 1, x - 1].astype(np.float32)
    tr = color[y - 1, x + width].astype(np.float32)
    bl = color[y + height, x - 1].astype(np.float32)
    br = color[y + height, x + width].astype(np.float32)
    corners = (1 - sx) * ((1 - sy) * tl + sy * bl) + sx * ((1 - sy) * tr + sy * br)
    repaired = color.copy()
    repaired[y:y + height, x:x + width] = np.clip(vertical + horizontal - corners, 0, 255).astype(np.uint8)
    return repaired


def repair_pixels(image: np.ndarray, mask: np.ndarray, radius: float = 4.0, method: str = "edge-aware", model_path: Path | None = None) -> np.ndarray:
    """Keep small glyph repair local; continue coherent edges through larger holes."""
    if method not in {"edge-aware", "telea", "lama", "adaptive"}:
        raise ValueError("未知修复算法")
    color = image[:, :, :3] if image.ndim == 3 and image.shape[2] == 4 else image
    if method == "adaptive":
        smooth = _smooth_background_repair(color, mask)
        if smooth is not None:
            result = smooth
        else:
            method = "edge-aware"
    if method == "lama":
        if model_path is None:
            raise ValueError("未配置离线修补模型")
        result = _lama_repair(color, mask, model_path)
    elif method != "adaptive":
        result = cv2.inpaint(color, mask, radius, cv2.INPAINT_TELEA)
    if method == "edge-aware":
        excluded = cv2.dilate(mask, np.ones((7, 7), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
        for label in range(1, count):
            x, y, width, height, area = map(int, stats[label])
            if min(width, height) < 12 or area < 180:
                continue
            direction = _edge_direction(color, excluded, (x, y, width, height))
            if direction is None:
                continue
            ys, xs = np.where(labels == label)
            valid, filled = _directional_values(color, excluded, ys, xs, direction, min(600, max(width, height) * 4))
            if np.count_nonzero(valid) < len(xs) * .90:
                continue
            result[ys[valid], xs[valid]] = np.clip(np.rint(filled[valid]), 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 4:
        result = np.dstack((result, image[:, :, 3]))
    result[mask == 0] = image[mask == 0]
    return result
