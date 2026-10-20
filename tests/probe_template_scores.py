"""Measure how well known account glyph masks align with their originals."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.deps'), str(ROOT / 'desktop')]
import cv2
import numpy as np
from backend.engine import read_image, asset_path, TEMPLATES

for raw in sys.argv[1:]:
    path = Path(raw)
    image = read_image(path)
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    detail = gray - cv2.GaussianBlur(gray, (0, 0), 5)
    row = np.zeros((h, w), bool)
    row[round(h*.961):round(h*.986), round(w*.71):] = True
    scores = {}
    for name in ('account-11578066187', 'account-267777923', 'legacy'):
        mask = cv2.resize(cv2.imdecode(np.fromfile(str(asset_path(TEMPLATES[name])), np.uint8), cv2.IMREAD_GRAYSCALE), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        mask &= row
        scores[name] = (round(float(detail[mask].mean()), 2), int(mask.sum()))
    print(path.name, scores)
