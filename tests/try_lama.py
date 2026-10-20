"""Compare local LaMa repair with the current OpenCV repair on two hard samples."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.deps'), str(ROOT / 'desktop')]

import cv2
import numpy as np

from backend.engine import build_mask, read_image
from backend.repair import repair_pixels


MODEL = ROOT / 'desktop/models/inpainting_lama_2025jan.onnx'
SOURCE = Path(sys.argv[1])
OUTPUT = ROOT / 'desktop/qa/account267/lama'
OUTPUT.mkdir(parents=True, exist_ok=True)

net = cv2.dnn.readNetFromONNX(str(MODEL))
for stem in ('13957f5325fc70a1f2c44cef544c52f9', '5b90abfed82f8a5f3ac6c70879b762e1'):
    src = SOURCE / f'{stem}_720.png'
    image = read_image(src)
    mask = build_mask(image.shape[1], image.shape[0], {'mode': 'template', 'templateId': 'account-267777923'})
    ys, xs = np.nonzero(mask)
    x0 = max(0, int(xs.min()) - 90)
    y0 = max(0, int(ys.min()) - 90)
    x1 = min(image.shape[1], int(xs.max()) + 91)
    y1 = min(image.shape[0], int(ys.max()) + 91)
    crop = image[y0:y1, x0:x1, :3]
    image_blob = cv2.dnn.blobFromImage(crop, 1 / 255, (512, 512), (0, 0, 0), False, False)
    old = repair_pixels(image, mask, method='telea')
    variants = [image, old]
    for padding in (0, 2, 4, 6):
        test_mask = cv2.dilate(mask, np.ones((padding * 2 + 1, padding * 2 + 1), np.uint8)) if padding else mask
        crop_mask = test_mask[y0:y1, x0:x1]
        mask_blob = cv2.dnn.blobFromImage(crop_mask, 1, (512, 512), (0,), False, False)
        mask_blob = (mask_blob > 0).astype(np.float32)
        net.setInput(image_blob, 'image')
        net.setInput(mask_blob, 'mask')
        output = net.forward()[0]
        output = np.transpose(output, (1, 2, 0))
        output = np.clip(output, 0, 255).astype(np.uint8)
        output = cv2.resize(output, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
        lama = image.copy()
        lama_crop = lama[y0:y1, x0:x1]
        lama_crop[crop_mask > 0] = output[crop_mask > 0]
        variants.append(lama)
        cv2.imencode('.png', lama)[1].tofile(str(OUTPUT / f'{stem}_lama_pad{padding}.png'))
    cx0 = max(0, x0 - 30)
    cy0 = max(0, y0 - 30)
    comparison = np.concatenate([variant[cy0:, cx0:] for variant in variants], axis=1)
    cv2.imencode('.png', comparison)[1].tofile(str(OUTPUT / f'{stem}_comparison_paddings.png'))
    print(src.name, 'roi', (x0, y0, x1, y1))
