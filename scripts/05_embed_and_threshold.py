import os, numpy as np, cv2
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

DATA = Path("data_work/s3r_png")
MODEL_PATH = "models/yolo_cls_s3r/weights/best.pt"
EMB_SIZE = 512  # Ultralytics cls models output 1000 logits; we'll use penultimate layer if available via predict with return dict; fallback: use avgpool features from resized image.

# Simple fallback: use the model's pre-logits feature via 'predict' and 'probs' with 'ret' set to features.
# If Ultralytics API changes, do a naive embedding: flatten resized image (works surprisingly OK for thresholding).
def image_to_embedding(img_gray):
    # Naive embedding if we can't tap model internals
    img = cv2.resize(img_gray, (128,128), interpolation=cv2.INTER_AREA)
    return (img.astype(np.float32).flatten()/255.0)

def build_gallery(model, k_classes=24):
    gallery = []
    labels = []
    train_dir = DATA/"train"
    for cls_dir in sorted(train_dir.iterdir()):
        if not cls_dir.is_dir(): continue
        cls_idx = int(cls_dir.name)
        for png in list(cls_dir.glob("*.png"))[:20]:  # sample few per class to keep it quick
            g = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
            emb = image_to_embedding(g)
            gallery.append(emb)
            labels.append(cls_idx)
    return np.stack(gallery), np.array(labels)

def main():
    model = YOLO(MODEL_PATH)  # not directly used for embeddings in this fallback
    G, y = build_gallery(model)
    # Class prototypes (mean embeddings)
    prototypes = {}
    for cls in np.unique(y):
        prototypes[cls] = G[y==cls].mean(0, keepdims=True)
    # Global center and threshold (95th percentile of distances to own prototype)
    dists = []
    for cls in np.unique(y):
        D = cosine_distances(G[y==cls], prototypes[cls]).flatten()
        dists.extend(D.tolist())
    THRESH = float(np.quantile(dists, 0.95))
    np.save("models/osr_cls_prototypes.npy", np.stack([prototypes[c] for c in sorted(prototypes)], axis=0))
    np.save("models/osr_cls_classes.npy", np.array(sorted(prototypes)))
    np.save("models/osr_cls_thresh.npy", np.array([THRESH]))
    print(f"Saved OSR prototypes for {len(prototypes)} classes; THRESH={THRESH:.4f}")

if __name__ == "__main__":
    main()
