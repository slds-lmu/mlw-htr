"""MLW-OCR Projekt.

Entry point to clean lemmata corpus and train tokenizer.
"""

from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
model.train(
    data="/home/ubuntu/mlw-consulting-project/data/processed/yolo_ds/dataset.yaml",
    epochs=100,
    imgsz=640,
)
model.val()
model.export()
