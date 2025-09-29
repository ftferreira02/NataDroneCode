from ultralytics import YOLO

# Ultralytics classification expects:
# data = path to a folder with 'train' and 'val' subfolders, each containing class subfolders.
DATA = "data_work/s3r_png"
MODEL = "yolov8n-cls.pt"   # nano classification backbone (fast)

model = YOLO(MODEL)
model.train(
    data=DATA,
    epochs=30,         # bump to ~50 if you have time
    imgsz=224,         # 224 or 320; 224 is faster
    batch=32,          # adjust to RAM/CPU
    lr0=0.001,
    device='cpu',      # or '0' if you have a CUDA GPU
    project='models',
    name='yolo_cls_s3r'
)
