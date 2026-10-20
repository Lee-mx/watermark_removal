"""Inspect footer-mask coverage on the user-reported original, not its screenshot."""
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.deps'), str(ROOT / 'desktop')]
import cv2
import numpy as np
from backend.engine import read_image, build_mask, repair_to_file
from backend.repair import repair_pixels

source = Path(sys.argv[1])
image = read_image(source)
h, w = image.shape[:2]
old_mask = build_mask(w, h, {'mode': 'template', 'templateId': 'legacy'})
new_mask = build_mask(w, h, {'mode': 'template'})
out = ROOT / 'desktop/qa/footer-coverage'
out.mkdir(parents=True, exist_ok=True)
desktop_output = repair_to_file(source, out, {'mode': 'template'}, 'desktop_verified')
tiles = []
for title, mask in [('Original', None), ('Previous', old_mask), ('Matched mask + edge repair', new_mask)]:
    result = image if mask is None else repair_pixels(image, mask)
    crop = result[round(h * .90):, round(w * .69):, :3]
    tile = np.full((300, 600, 3), 245, np.uint8)
    tile[40:] = cv2.resize(crop, (600, 260))
    cv2.putText(tile, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (30,30,30), 2)
    tiles.append(tile)
    if title == 'Matched mask + edge repair':
        np.testing.assert_array_equal(read_image(Path(desktop_output['path'])), result)
        cv2.imencode('.png', result)[1].tofile(str(out / 'clean.png'))
        cv2.imencode('.png', new_mask)[1].tofile(str(out / 'mask.png'))
        check = read_image(out / 'clean.png')
        report = {'dimensions': list(check.shape), 'changed_outside_mask': int(np.count_nonzero(check[new_mask == 0] != image[new_mask == 0]))}
        (out / 'verification.json').write_text(json.dumps(report, indent=2))
        print(report)
cv2.imencode('.png', np.concatenate(tiles))[1].tofile(str(out / 'comparison.png'))
