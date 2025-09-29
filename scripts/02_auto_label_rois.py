# scripts/02_auto_label_rois.py
import cv2, os
import numpy as np
from pathlib import Path
from tqdm import tqdm

def find_rois(img):
    # img: grayscale 0..255
    blur = cv2.GaussianBlur(img, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 35, -5)
    # we want bright blobs: invert if needed
    mask = thresh
    # open-close to clean
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 50:  # drop tiny junk
            continue
        boxes.append((x,y,w,h))
    return boxes

def main(img_dir, out_dir_preview):
    os.makedirs(out_dir_preview, exist_ok=True)
    pngs = list(Path(img_dir).rglob("*.png"))  # recursive
    for p in tqdm(pngs):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        boxes = find_rois(img)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for (x,y,w,h) in boxes:
            cv2.rectangle(vis, (x,y), (x+w,y+h), (0,0,255), 2)
        # mirror subfolder structure in preview dir
        rel = p.relative_to(img_dir)
        out_vis = Path(out_dir_preview) / rel
        out_vis.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_vis), vis)
        with open(p.with_suffix(".rois.txt"), "w") as f:
            for (x,y,w,h) in boxes:
                f.write(f"{x},{y},{w},{h}\n")

if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "data_work/s3r_png/val"
    dst = sys.argv[2] if len(sys.argv) > 2 else "data_work/s3r_png_preview"
    main(src, dst)
