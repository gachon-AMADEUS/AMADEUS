from ultralytics import YOLO
import cv2
import torch
import numpy as np
from glob import glob
import os

from segment_anything import sam_model_registry, SamPredictor

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("[using cuda]")
else:
    print("[using cpu]")

# 모델 로드

model = YOLO("best.pt")

print(f"클래스: {model.names}")

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to(device=device)
predictor = SamPredictor(sam)

# 이미지 로드d

INPUT_DIR = "태량"
input_images_path = glob(os.path.join(INPUT_DIR, "*.jpg"))
print(input_images_path)

#일단 이미지 하나
for ith, input_image_path in enumerate(input_images_path):
    image_bgr = cv2.imread(input_image_path)
    if image_bgr is None:
        raise FileNotFoundError("이미지없음")

    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    print(h, w)

    FOOT_CLASS_ID = 1
    result = model.predict(input_image_path)[0]

    # for result in results:
    if result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError("YOLO 박스검출실패")

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()      # (N, 4)
    classes = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

    foot_boxes = boxes_xyxy[classes == FOOT_CLASS_ID]

    if len(foot_boxes) == 0:
        raise RuntimeError("YOLO에서 FOOT 클래스 검출 x")

    foot_boxes_torch = torch.tensor(foot_boxes, device=device)

    predictor.set_image(img_rgb)

    transformed_boxes = predictor.transform.apply_boxes_torch(
        foot_boxes_torch, img_rgb.shape[:2]
    )

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

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10,10))
    # plt.imshow(clean)
    # plt.axis("off")
    # plt.show()
    OUTPUT_FOLDER = "output2"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"output_{ith}.png"), clean)
    print(f"저장: output_{ith}.png")
