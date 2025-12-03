from ultralytics import YOLO
import cv2
import os

"""
import numpy as np

def center_crop(img, crop_size=1080):
    h, w = img.shape[:2]
    ch = cw = crop_size

    start_x = max(0, w//2 - cw//2)
    start_y = max(0, h//2 - ch//2)
    end_x = start_x + cw
    end_y = start_y + ch

    cropped = img[start_y:end_y, start_x:end_x]

    if cropped.shape[0] != crop_size or cropped.shape[1] != crop_size:
        padded = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        h2, w2 = cropped.shape[:2]
        padded[:h2, :w2] = cropped
        return padded

    return cropped


def preprocess(img_path):
    img = cv2.imread(img_path)

    crop1080 = center_crop(img, 1080)
    resized640 = cv2.resize(crop1080, (640, 640), interpolation=cv2.INTER_LINEAR)

    return resized640, crop1080
"""

# --------------------------
# YOLO 모델 로드
# --------------------------
model = YOLO("runs/segment/train/weights/best.pt") 

INPUT_NAME = "frame_0464"
FOLDER_HEAD_NAME = "foot"
INPUT_IMAGE = f"{FOLDER_HEAD_NAME}_frames/{INPUT_NAME}.jpg"

result = model.predict(INPUT_IMAGE)[0]
img = cv2.imread(INPUT_IMAGE)

# --------------------------
# YOLO Segmentation mask
# --------------------------
mask = result.masks.data[0].cpu().numpy()
mask = (mask * 255).astype("uint8")
mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

clean = cv2.bitwise_and(img, img, mask=mask)

# --------------------------
# YOLO BBOX 그리기
# --------------------------
annotated = img.copy()

boxes = result.boxes.xyxy.cpu().numpy()
classes = result.boxes.cls.cpu().numpy().astype(int)

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)

    # bbox 사각형
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # 클래스 이름 텍스트
    label = f"class {classes[i]}"
    cv2.putText(
        annotated, label, (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
    )
# --------------------------
# YOLO BBOX 그리기 2
# --------------------------
annotated2 = clean.copy()

boxes = result.boxes.xyxy.cpu().numpy()
classes = result.boxes.cls.cpu().numpy().astype(int)

for i, box in enumerate(boxes):
    x1, y1, x2, y2 = map(int, box)

    # bbox 사각형
    cv2.rectangle(annotated2, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # 클래스 이름 텍스트
    label = f"class {classes[i]}"
    cv2.putText(
        annotated2, label, (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
    )

# --------------------------
# 저장
# --------------------------
os.makedirs("foot_clean", exist_ok=True)
os.makedirs("foot_mask", exist_ok=True)
os.makedirs("foot_bbox", exist_ok=True)
os.makedirs("foot_clean_bbox", exist_ok=True)

cv2.imwrite(f"foot_clean/{INPUT_NAME}.jpg", clean)
cv2.imwrite(f"foot_mask/{INPUT_NAME}_mask.png", mask)
cv2.imwrite(f"foot_bbox/{INPUT_NAME}_bbox.jpg", annotated)
cv2.imwrite(f"foot_clean_bbox/{INPUT_NAME}_bbox.jpg", annotated2)

print("saved:")
print(f" - foot_clean/{INPUT_NAME}.jpg")
print(f" - foot_mask/{INPUT_NAME}_mask.png")
print(f" - foot_bbox/{INPUT_NAME}_bbox.jpg")
print(f" - foot_clean_bbox/{INPUT_NAME}_bbox.jpg")
