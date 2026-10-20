"""Generate a before/after diagnostic for the user-reported plate-rim case."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / ".deps"), str(ROOT / "desktop")]
import cv2
import numpy as np
from backend.engine import read_image, build_mask
from backend.repair import repair_pixels

source = Path(sys.argv[1])
out = ROOT / "desktop" / "qa" / "edge-repair"
out.mkdir(parents=True, exist_ok=True)
image = read_image(source)
h, w = image.shape[:2]
if len(sys.argv) > 2:
    mask = cv2.imdecode(np.fromfile(sys.argv[2], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
else:
    mask = build_mask(w, h, {"mode": "template"})
tiles = []
records = []
for title, im in [("Original", image), ("Telea", repair_pixels(image, mask, method="telea")), ("Edge-aware", repair_pixels(image, mask))]:
    if title != "Original":
        target = out / f"{title.lower()}_full.png"
        ok, data = cv2.imencode(".png", im)
        data.tofile(str(target))
        check = read_image(target)
        records.append({"method": title, "dimensions": list(check.shape[:2]), "changed_outside_mask": int(np.count_nonzero(check[mask == 0] != image[mask == 0]))})
    crop = im[round(h * .90):, round(w * .69):, :3]
    crop = cv2.resize(crop, (600, 260))
    tile = np.full((300, 600, 3), 245, np.uint8)
    tile[40:] = crop
    cv2.putText(tile, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, .75, (30, 30, 30), 2)
    tiles.append(tile)
sheet = np.concatenate(tiles, axis=0)
cv2.imencode(".png", sheet)[1].tofile(str(out / "comparison.png"))
(out / "verification.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
print(json.dumps(records))
