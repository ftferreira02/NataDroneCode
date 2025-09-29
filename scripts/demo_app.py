# scripts/demo_cls_osr.py
import numpy as np, cv2
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_distances

MODEL_PATH = "models/yolo_cls_s3r/weights/best.pt"
protos = np.load("models/osr_cls_prototypes.npy")     # shape [K, D]
classes = np.load("models/osr_cls_classes.npy")       # shape [K]
THRESH = float(np.load("models/osr_cls_thresh.npy")[0])

def image_to_embedding(img_gray):
    img = cv2.resize(img_gray, (128,128), interpolation=cv2.INTER_AREA)
    return (img.astype(np.float32).flatten()/255.0)

def classify_or_reject_png(png_path: Path):
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    emb = image_to_embedding(img)[None, :]
    # compute distance to each class prototype
    D = cosine_distances(emb, protos)[0]  # shape [K]
    best_idx = int(np.argmin(D))
    best_dist = float(D[best_idx])
    if best_dist > THRESH:
        return "UNKNOWN", best_dist, None
    return int(classes[best_idx]), best_dist, float(np.exp(-best_dist))  # fake confidence from distance

if __name__ == "__main__":
    test_img = Path("data_work/s3r_png/val/01")  # change to a sample
    sample = next(test_img.glob("*.png"))
    label, dist, conf = classify_or_reject_png(sample)
    print(sample, "->", label, "dist:", round(dist,3), "conf:", conf)
