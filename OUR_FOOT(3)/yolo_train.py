from ultralytics import YOLO


model = YOLO(model="yolo11n-seg.pt")

model.train(
  data="dataset/data.yaml",
  epochs=80,
  imgsz=1024,
  batch=16,
  # RoboFlow 이미 augment 되어 있으므로 최소화된 YOLO augment
  mosaic=0.0,
  copy_paste=0.0,
  mixup=0.0,
  degrees=0,
  scale=0.3,
  hsv_h=0.005,
  hsv_s=0.2,
  hsv_v=0.2,
  fliplr=0.5,
)