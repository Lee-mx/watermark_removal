"""Run the desktop engine on the seven supplied originals and inspect footers."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.deps'), str(ROOT / 'desktop')]
import cv2
import numpy as np
from backend.engine import read_image, build_mask, suggested_template, repair_to_file

source_dir = Path(sys.argv[1])
output = ROOT / 'desktop/qa/account267/desktop'
output.mkdir(parents=True, exist_ok=True)
records = []
tiles = []
for source in sorted(source_dir.glob('*_720.png')):
    image = read_image(source)
    h, w = image.shape[:2]
    suggested = suggested_template(image)
    assert suggested == 'account-267777923', source
    selection = {'mode': 'template', 'templateId': suggested}
    target = repair_to_file(source, output, selection)
    clean = read_image(Path(target['path']))
    mask = build_mask(w, h, selection)
    records.append({'source': source.name, 'template': suggested, 'output': target['path'], 'changedOutsideMask': int(np.count_nonzero(clean[mask == 0] != image[mask == 0]))})
    crop = clean[round(h*.92):, round(w*.71):, :3]
    tile = cv2.resize(crop, (400, 160))
    cv2.putText(tile, source.name[:8], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2)
    tiles.append(tile)
cv2.imencode('.png', np.concatenate(tiles))[1].tofile(str(output / 'footers.png'))
(output / 'verification.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
print(json.dumps(records, ensure_ascii=False))
