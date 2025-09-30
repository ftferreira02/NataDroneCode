import os
from pathlib import Path
import cv2

# Root dirs
IMG_DIR = Path("data_work/s3r_png")         # where your class PNGs live
PREVIEW_DIR = Path("data_work/s3r_png_preview")  # where you ran ROI auto-labeler
YOLO_OUT = Path("data_work/yolo")           # output dataset for detection

# Class mapping: folder name (00..23) -> int class id
def class_from_path(p: Path) -> int:
    # parent of parent is class folder (train/val/XX/)
    return int(p.parent.name)

def normalize_box(x, y, w, h, img_w, img_h):
    xc = (x + w/2) / img_w
    yc = (y + h/2) / img_h
    nw = w / img_w
    nh = h / img_h
    return xc, yc, nw, nh

def main():
    YOLO_OUT.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val"]:
        img_split = IMG_DIR / split
        for img_path in img_split.rglob("*.png"):
            cls = class_from_path(img_path)

            # Find matching ROI file (same stem but .rois.txt next to original PNG)
            roi_file = img_path.with_suffix(".rois.txt")
            if not roi_file.exists():
                continue

            # Read image shape
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            h, w = img.shape

            # Read ROI boxes
            labels = []
            with open(roi_file, "r") as f:
                for line in f:
                    try:
                        x,y,bw,bh = map(int, line.strip().split(","))
                        xc, yc, nw, nh = normalize_box(x,y,bw,bh,w,h)
                        labels.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                    except:
                        pass

            if not labels:
                continue

            # Save image and YOLO label
            out_img = YOLO_OUT / "images" / split / f"{img_path.stem}.png"
            out_lbl = YOLO_OUT / "labels" / split / f"{img_path.stem}.txt"
            out_img.parent.mkdir(parents=True, exist_ok=True)
            out_lbl.parent.mkdir(parents=True, exist_ok=True)

            # Copy/mirror image
            cv2.imwrite(str(out_img), img)

            with open(out_lbl, "w") as f:
                f.write("\n".join(labels))

    # Write data.yaml
    with open(YOLO_OUT/"data.yaml","w") as f:
        f.write(f"path: {YOLO_OUT.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 24\n")  # number of classes
        names = [f"'{i}'" for i in range(24)]
        f.write(f"names: [{', '.join(names)}]\n")

if __name__ == "__main__":
    main()
