from ultralytics import YOLO
from pathlib import Path

# Use detection model (not classification)
model = YOLO("yolov8n.pt")

# Absolute path to your data.yaml
data_path = str(Path("data_work/yolo/data.yaml").resolve())

model.train(
    data=data_path,
    epochs=20,
    imgsz=640,
    batch=16,
    project="models",
    name="yolo_det_s3r"
)
