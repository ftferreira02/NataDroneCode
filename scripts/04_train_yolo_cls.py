from ultralytics import YOLO
from pathlib import Path

DATA_DIR = str(Path("data_work/s3r_png").resolve())  # folder-per-class

model = YOLO("yolov8n-cls.pt")  # classification model
model.train(
    data=DATA_DIR,      # classification uses the folder path (NOT a yaml)
    epochs=30,
    imgsz=224,
    batch=32,
    device="cpu",       # or "0" for GPU
    project="models",
    name="yolo_cls_s3r"
)
