"""Visual diagnostics for the two reported hard scenes."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'.deps'), str(ROOT/'desktop')]
import cv2
import numpy as np
from backend.engine import read_image, build_mask

source_dir = Path(sys.argv[1])
out = ROOT/'desktop/qa/account267/methods'
out.mkdir(parents=True, exist_ok=True)
names = ['13957f5325fc70a1f2c44cef544c52f9_720.png', '5b90abfed82f8a5f3ac6c70879b762e1_720.png']
tiles = []
for name in names:
    im = read_image(source_dir/name)
    h,w = im.shape[:2]
    mask = build_mask(w,h,{'mode':'template','templateId':'account-267777923'})
    for label,method,radius in [('Telea3',cv2.INPAINT_TELEA,3),('Telea4',cv2.INPAINT_TELEA,4),('NS3',cv2.INPAINT_NS,3),('NS5',cv2.INPAINT_NS,5)]:
        fixed = cv2.inpaint(im,mask,radius,method)
        fixed[mask==0]=im[mask==0]
        crop=fixed[round(h*.92):,round(w*.71):,:3]
        tile=cv2.resize(crop,(450,180))
        cv2.putText(tile,name[:4]+' '+label,(5,24),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,240,0),2)
        tiles.append(tile)
cv2.imencode('.png',np.concatenate(tiles))[1].tofile(str(out/'comparison.png'))
