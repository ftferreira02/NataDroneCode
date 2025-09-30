# scripts/03a_visualize_yolo_labels.py
import cv2, random
from pathlib import Path

ROOT = Path("data_work/yolo")
samples = list((ROOT/"images/val").glob("*.png"))
random.shuffle(samples)

def draw(png):
    img = cv2.imread(str(png))
    h, w = img.shape[:2]
    lbl = (ROOT/"labels/val"/(png.stem + ".txt"))
    if not lbl.exists(): return img
    for line in lbl.read_text().strip().splitlines():
        cls, xc, yc, ww, hh = map(float, line.split())
        x = int((xc - ww/2) * w)
        y = int((yc - hh/2) * h)
        bw = int(ww * w)
        bh = int(hh * h)
        cv2.rectangle(img, (x,y), (x+bw,y+bh), (0,0,255), 2)
        cv2.putText(img, str(int(cls)), (x, max(y-3,10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    return img

outdir = Path("data_work/yolo_label_viz"); outdir.mkdir(parents=True, exist_ok=True)
for i, p in enumerate(samples[:50]):
    vis = draw(p)
    cv2.imwrite(str(outdir/f"{p.stem}.png"), vis)
print("Wrote previews to", outdir)
