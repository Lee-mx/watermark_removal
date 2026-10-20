"""Compare repairs for a bright footer watermark on a smooth dark background."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.deps'), str(ROOT / 'desktop')]

import cv2
import numpy as np

from backend.engine import read_image
from backend.repair import repair_pixels

source = read_image(Path(sys.argv[1]))[:, :, :3]
existing = read_image(Path(sys.argv[2]))[:, :, :3]
output = ROOT / 'desktop/qa/manual-gradient'
output.mkdir(parents=True, exist_ok=True)

mask = np.zeros(source.shape[:2], np.uint8)
mask[945:1011, 565:1018] = 255
gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
roi = gray[945:1011, 565:1018]

variants = {'original': source, 'existing': existing}
for method in ('telea', 'lama'):
    variants[f'full-{method}'] = repair_pixels(
        source, mask, method=method,
        model_path=ROOT / 'desktop/models/inpainting_lama_2025jan.onnx' if method == 'lama' else None,
    )

def coons_patch(image: np.ndarray, blur: bool) -> np.ndarray:
    x0, x1, y0, y1 = 565, 1018, 945, 1011
    top = image[y0 - 1, x0:x1].astype(np.float32)
    bottom = image[y1, x0:x1].astype(np.float32)
    left = image[y0:y1, x0 - 1].astype(np.float32)
    right = image[y0:y1, x1].astype(np.float32)
    if blur:
        top = cv2.GaussianBlur(top[None], (17, 1), 3)[0]
        bottom = cv2.GaussianBlur(bottom[None], (17, 1), 3)[0]
        left = cv2.GaussianBlur(left[:, None], (1, 17), 3)[:, 0]
        right = cv2.GaussianBlur(right[:, None], (1, 17), 3)[:, 0]
    h, w = y1 - y0, x1 - x0
    s = (np.arange(w, dtype=np.float32) + 1)[None, :, None] / (w + 1)
    t = (np.arange(h, dtype=np.float32) + 1)[:, None, None] / (h + 1)
    vertical = (1 - t) * top[None] + t * bottom[None]
    horizontal = (1 - s) * left[:, None] + s * right[:, None]
    tl = image[y0 - 1, x0 - 1].astype(np.float32)
    tr = image[y0 - 1, x1].astype(np.float32)
    bl = image[y1, x0 - 1].astype(np.float32)
    br = image[y1, x1].astype(np.float32)
    corners = (1 - s) * ((1 - t) * tl + t * bl) + s * ((1 - t) * tr + t * br)
    fill = np.clip(vertical + horizontal - corners, 0, 255).astype(np.uint8)
    result = image.copy()
    result[y0:y1, x0:x1] = fill
    return result

variants['coons-raw'] = coons_patch(source, False)
variants['coons-smooth'] = coons_patch(source, True)

for threshold in (70, 90, 110):
    foreground = np.zeros_like(mask)
    foreground[945:1011, 565:1018] = (roi > threshold).astype(np.uint8) * 255
    for padding in (2, 4, 6):
        expanded = cv2.dilate(foreground, np.ones((padding * 2 + 1, padding * 2 + 1), np.uint8))
        expanded &= mask
        variants[f'letters-t{threshold}-p{padding}'] = repair_pixels(source, expanded, method='telea')

tiles = []
for name, result in variants.items():
    cv2.imencode('.png', result)[1].tofile(str(output / f'{name}.png'))
    tile = result[915:1024, 520:1024].copy()
    cv2.putText(tile, name, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 2)
    tiles.append(tile)
while len(tiles) % 3:
    tiles.append(np.zeros_like(tiles[0]))
contact = np.concatenate([np.concatenate(tiles[i:i + 3], axis=1) for i in range(0, len(tiles), 3)], axis=0)
cv2.imencode('.png', contact)[1].tofile(str(output / 'comparison.png'))
print('saved', len(variants), 'variants')
