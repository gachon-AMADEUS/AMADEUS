from ultralytics import YOLO
import cv2
import os
import numpy as np

def center_crop(img, crop_size=1080):
    h, w = img.shape[:2]
    ch = cw = crop_size

    # 중앙 좌표 계산
    start_x = max(0, w//2 - cw//2)
    start_y = max(0, h//2 - ch//2)
    end_x = start_x + cw
    end_y = start_y + ch

    # Crop 수행 (원본이 더 작다면 패딩도 가능)
    cropped = img[start_y:end_y, start_x:end_x]

    # 만약 이미지가 1080보다 작으면 패딩하기
    if cropped.shape[0] != crop_size or cropped.shape[1] != crop_size:
        padded = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        h2, w2 = cropped.shape[:2]
        padded[:h2, :w2] = cropped
        return padded

    return cropped

def preprocess(img_path):
    img = cv2.imread(img_path)

    # 1) 1080×1080 crop
    crop1080 = center_crop(img, 1080)

    # 2) YOLO 입력 크기 640×640 리사이즈
    resized640 = cv2.resize(crop1080, (640, 640), interpolation=cv2.INTER_LINEAR)

    return resized640, crop1080

model = YOLO("runs/segment/train/weights/best.pt")  # YOLOv11n-seg 컵 전용 가중치

imgname = "frame_0327_jpg.rf.09f151b65a07a291078b24a1cedacfbc.jpg"
imgpath = f"test/images/{imgname}"


results = model(imgpath)
img = cv2.imread(imgpath)
#img_resized640, _ = preprocess(f"cup_frames/{imgname}.jpg")

mask = results[0].masks.data[0].cpu().numpy()
mask = (mask * 255).astype("uint8")
print(mask.shape)
# ⚠️ mask 크기를 원본 이미지와 동일하게 맞추기

mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

# kernel = np.ones((5,5), np.uint8)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

clean = cv2.bitwise_and(img, img, mask=mask)

os.makedirs("cup_clean", exist_ok=True)
os.makedirs("cup_mask", exist_ok=True)

cv2.imwrite(f"cup_clean/{imgname}.jpg", clean)
cv2.imwrite(f"cup_mask/{imgname}_mask.png", mask)
