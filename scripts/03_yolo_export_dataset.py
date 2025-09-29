# scripts/03_yolo_export_dataset.py
import os, cv2
from pathlib import Path
from sklearn.model_selection import train_test_split

SRC = Path("data_work/spectrograms")
DST = Path("data_work/yolo")
(DST/"images/train").mkdir(parents=True, exist_ok=True)
(DST/"images/val").mkdir(parents=True, exist_ok=True)
(DST/"labels/train").mkdir(parents=True, exist_ok=True)
(DST/"labels/val").mkdir(parents=True, exist_ok=True)

images = list(SRC.glob("*.png"))
train, val = train_test_split(images, test_size=0.2, random_state=42)

def convert_and_copy(img_path, split):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    # copy image
    out_img = DST/f"images/{split}/{img_path.name}"
    cv2.imwrite(str(out_img), img)
    # read rois
    rois_path = img_path.with_suffix(".rois.txt")
    out_lbl = DST/f"labels/{split}/{img_path.stem}.txt"
    lines=[]
    if rois_path.exists():
        with open(rois_path) as f:
            for line in f:
                x,y,bw,bh = map(int, line.strip().split(","))
                xc = (x + bw/2)/w
                yc = (y + bh/2)/h
                ww = bw/w
                hh = bh/h
                # class 0 = drone_signal
                lines.append(f"0 {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
    with open(out_lbl, "w") as f:
        f.write("\n".join(lines))

for p in train: convert_and_copy(p, "train")
for p in val:   convert_and_copy(p, "val")

# write data.yaml
with open(DST/"data.yaml","w") as f:
    f.write(
f"""path: {DST.resolve()}
train: images/train
val: images/val
nc: 1
names: [drone_signal]
"""
)
