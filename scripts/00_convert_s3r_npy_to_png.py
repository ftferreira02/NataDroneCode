import os, re
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

SRC = Path("data_raw/DroneRFb-Spectra/Data")
DST = Path("data_work/s3r_png")  # classification folder layout: train/val/<class>/*.png
IMG_SIZE = 512  # dataset is 512x512
VAL_SPLIT = 0.2
RANDOM_STATE = 42

# Map from folder or filename to class index
# If you have per-class folders like "01_DJI_Phantom_3", we'll parse the leading digits.
# If flat files, create a dict mapping stem -> class index and use it here.
def infer_class_from_path(p: Path) -> int:
    # Prefer folder name if it starts with digits
    m = re.match(r"^(\d+)", p.parent.name)
    if m:
        return int(m.group(1))
    # Else: try digits at file start
    m = re.match(r"^(\d+)", p.stem)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot infer class for file: {p}")

def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    # arr is np.float16 spectrogram, shape 512x512; normalize to 0..255
    arr = arr.astype(np.float32)
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-9)
    img = (arr * 255.0).clip(0,255).astype(np.uint8)
    # ensure 512x512
    if img.shape != (IMG_SIZE, IMG_SIZE):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img

def main():
    files = list(SRC.rglob("*.npy"))
    if not files:
        print(f"No .npy files found under {SRC.resolve()}")
        return
    # build list of (path, class_idx)
    items = []
    for f in files:
        try:
            cls = infer_class_from_path(f)
            items.append((f, cls))
        except Exception as e:
            print(f"[WARN] {e}")

    train_items, val_items = train_test_split(items, test_size=VAL_SPLIT, random_state=RANDOM_STATE, stratify=[c for _,c in items])

    for split, group in [("train", train_items), ("val", val_items)]:
        for f, cls in tqdm(group, desc=f"Converting {split}"):
            try:
                arr = np.load(f)  # float16, 512x512
                img = normalize_to_uint8(arr)
                out_dir = DST / split / f"{cls:02d}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / (f.stem + ".png")
                cv2.imwrite(str(out_path), img)
            except Exception as e:
                print(f"[ERROR] {f}: {e}")

if __name__ == "__main__":
    main()
