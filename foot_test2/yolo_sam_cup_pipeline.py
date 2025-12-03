import cv2
import numpy as np
import torch
from ultralytics import YOLO

from segment_anything import sam_model_registry, SamPredictor


# =========================
# 설정
# =========================
YOLO_WEIGHTS = "runs/segment/train/weights/best.pt"  # 컵 검출 YOLO 가중치
SAM_CHECKPOINT = "./models/sam_vit_h_4b8939.pth"  # SAM 체크포인트 경로
SAM_MODEL_TYPE = "vit_h"  # 'vit_h', 'vit_l', 'vit_b' 중 하나

INPUT_NAME = "frame_0464"
FOLDER_HEAD_NAME = "foot"
INPUT_IMAGE = f"{FOLDER_HEAD_NAME}_frames/{INPUT_NAME}.jpg"
OUT_MASK_PATH = f"{FOLDER_HEAD_NAME}_mask/{INPUT_NAME}_sam_mask.png"
OUT_CLEAN_PATH = f"{FOLDER_HEAD_NAME}_clean/{INPUT_NAME}_sam_clean.png"

CUP_CLASS_ID = 0  # YOLO에서 'cup' 클래스 인덱스 (컵만 학습했다면 0일 가능성 높음)


# =========================
# 1. 모델 로드 (YOLO + SAM)
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("[using cuda]")
else:
    print("[using cpu]")

yolo_model = YOLO(YOLO_WEIGHTS)

sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=device)
predictor = SamPredictor(sam)


# =========================
# 2. 이미지 로드
# =========================
image_bgr = cv2.imread(INPUT_IMAGE)
if image_bgr is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {INPUT_IMAGE}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
h, w = image_rgb.shape[:2]

# =========================
# 3. YOLO로 컵 bbox 검출
# =========================
results = yolo_model.predict(INPUT_IMAGE)[0]

if results.boxes is None or len(results.boxes) == 0:
    raise RuntimeError("YOLO에서 아무 객체도 검출되지 않았습니다.")

boxes_xyxy = results.boxes.xyxy.cpu().numpy()      # (N, 4)
classes = results.boxes.cls.cpu().numpy().astype(int)  # (N,)

# 컵 클래스만 필터링
cup_boxes = boxes_xyxy[classes == CUP_CLASS_ID]

if len(cup_boxes) == 0:
    raise RuntimeError("YOLO에서 'cup' 클래스를 검출하지 못했습니다.")

# torch tensor로 변환
cup_boxes_torch = torch.tensor(cup_boxes, device=device)


# =========================
# 4. SAM에 이미지 설정
# =========================
predictor.set_image(image_rgb)

# SAM의 내부 좌표계에 맞게 bbox 변환
transformed_boxes = predictor.transform.apply_boxes_torch(
    cup_boxes_torch, image_rgb.shape[:2]
)

# =========================
# 5. SAM으로 각 컵 bbox에 대한 정밀 마스크 추출
# =========================
with torch.no_grad():
    masks, scores, logits = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=True,  # 각 bbox마다 여러 후보 마스크
    )

# masks shape: (N, 3, H, W)  (N: bbox 개수, 3: 후보 마스크 수)
# 각 bbox마다 점수가 가장 높은 마스크 하나씩 선택
best_mask_per_box = []
for i in range(masks.shape[0]):
    box_masks = masks[i]           # (3, H, W)
    box_scores = scores[i]         # (3,)
    best_idx = torch.argmax(box_scores)
    best_mask = box_masks[best_idx]  # (H, W)
    best_mask_per_box.append(best_mask)

# (N, H, W) 텐서로 쌓기
best_masks = torch.stack(best_mask_per_box, dim=0)  # (N, H, W)

# 여러 컵이 있으면 OR로 합쳐서 하나의 전체 마스크 생성
combined_mask = torch.any(best_masks > 0.5, dim=0)  # (H, W), bool

# numpy uint8로 변환 (0 또는 255)
final_mask = (combined_mask.cpu().numpy().astype(np.uint8)) * 255

# =========================
# 6. 마스크를 이용해 컵만 남긴 깨끗한 이미지 생성
# =========================
clean = cv2.bitwise_and(image_bgr, image_bgr, mask=final_mask)

# =========================
# 7. 결과 저장
# =========================
import os
os.makedirs(f"{FOLDER_HEAD_NAME}_mask", exist_ok=True)
os.makedirs(f"{FOLDER_HEAD_NAME}_clean", exist_ok=True)

cv2.imwrite(OUT_MASK_PATH, final_mask)
cv2.imwrite(OUT_CLEAN_PATH, clean)

print(f"SAM 마스크 저장: {OUT_MASK_PATH}")
print(f"컵만 남긴 이미지 저장: {OUT_CLEAN_PATH}")
