# DroneRFb-Spectra: Automatic Labeling + Classification + Open-Set Recognition

This project is a hackathon prototype for **RF-based drone recognition**.  
We combine **automatic spectrogram labeling**, **YOLO classification**, and **open-set rejection** on the [DroneRFb-Spectra (S3R)](https://github.com/DaftJun/S3R) dataset.

---

## 🚀 Quick Start

### 1. Create environment
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# or
source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```
2. Prepare dataset

Download DroneRFb-Spectra (S3R). The structure looks like:

data_raw/DroneRFb-Spectra/Data/
  0/
    xxx.npy
  1/
    xxx.npy
  ...
  23/
    xxx.npy


Each .npy is a 512x512 float16 spectrogram with a class label given by its parent folder.
3. Convert .npy to .png
python scripts/00_convert_s3r_npy_to_png.py


Creates:

data_work/s3r_png/
  train/00/*.png
  val/00/*.png
  ...

4. Train classifier
python scripts/04_train_yolo_cls.py


Trains a YOLOv8n-cls model on spectrogram images.

5. Open-set recognition (optional but recommended)

Hold out a few classes (e.g. 14, 20, 23) by moving their folders to data_work/s3r_png/unknown/ before training.

Then build OSR prototypes:

python scripts/05_osr_cls_threshold.py


Test inference:

python scripts/demo_cls_osr.py


Outputs either a known class or UNKNOWN.

6. Automatic ROI labeling (for rubric points)
python scripts/02_auto_label_rois.py data_work/s3r_png/val data_work/s3r_png_preview


Creates previews with red rectangles marking bright regions.

Saves sidecar .rois.txt files with bounding boxes.

Delete bad lines = human-in-the-loop correction.

7. Convert ROIs → YOLO detection dataset
python scripts/03_convert_rois_to_yolo.py


Creates a full YOLO detection dataset:

data_work/yolo/
  images/train/*.png
  labels/train/*.txt
  images/val/*.png
  labels/val/*.txt
  data.yaml


Each .txt contains normalized YOLO bounding boxes from the .rois.txt.

8. Train YOLO detection
python scripts/04_train_yolo.py


Trains a YOLOv8n detection model on ROI bounding boxes.

| Script                           | Purpose                                                                                                                        |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **00_convert_s3r_npy_to_png.py** | Converts S3R `.npy` → `.png` images, builds train/val split.                                                                   |
| **02_auto_label_rois.py**        | Auto-labels time–frequency regions (ROIs) in spectrograms and saves bounding boxes + previews. Demonstrates auto-label + HITL. |
| **03_convert_rois_to_yolo.py**   | Converts `.rois.txt` ROI boxes into YOLO detection dataset (images + labels + `data.yaml`).                                    |
| **04_train_yolo_cls.py**         | Trains a YOLOv8 classification model on spectrogram PNGs.                                                                      |
| **04_train_yolo.py**             | Trains a YOLOv8 detection model using ROI-based YOLO dataset.                                                                  |
| **05_osr_cls_threshold.py**      | Builds open-set recognition (OSR) prototypes + distance threshold from embeddings.                                             |
| **demo_cls_osr.py**              | Inference script: classifies a spectrogram, or rejects as `UNKNOWN` if too far from known prototypes.                          |
| **01_make_spectrograms.py**      | (Unused for S3R) Converts raw IQ/CSV/MAT signals to spectrograms. Useful if new raw RF data arrives.                           |
