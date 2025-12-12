import argparse
import os
import cv2
import torch
import numpy as np
from glob import glob
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

def run_segmentation(input_dir, output_dir, yolo_path, sam_path):
    # 1. 장치 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Device: {device}")

    # 2. 모델 로드
    print(f"⏳ Loading YOLO model: {yolo_path}")
    if not os.path.exists(yolo_path):
        raise FileNotFoundError(f"❌ YOLO 모델을 찾을 수 없습니다: {yolo_path}")
    model = YOLO(yolo_path)

    print(f"⏳ Loading SAM model: {sam_path}")
    if not os.path.exists(sam_path):
        raise FileNotFoundError(f"❌ SAM 체크포인트를 찾을 수 없습니다: {sam_path}")
    
    sam = sam_model_registry["vit_h"](checkpoint=sam_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # 3. 이미지 로드
    input_images_path = glob(os.path.join(input_dir, "*.jpg"))
    if not input_images_path:
        print(f"⚠️ 경고: '{input_dir}' 폴더에 .jpg 이미지가 없습니다.")
        return

    print(f"📂 Found {len(input_images_path)} images in {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # 4. 이미지 처리 루프
    
    # ★★★ 수정된 부분: 타겟 클래스 ID 설정 ★★★
    # 주의: 모델 학습 설정에 따라 ID가 다를 수 있습니다.
    # 0과 1이 가장 흔한 클래스 ID입니다.
    FOOT_CLASS_ID = 1      
    CHECKERBOARD_CLASS_ID = 0 
    TARGET_CLASS_IDS = [FOOT_CLASS_ID, CHECKERBOARD_CLASS_ID] 
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

    for ith, input_image_path in enumerate(input_images_path):
        filename = os.path.basename(input_image_path)
        print(f"[{ith+1}/{len(input_images_path)}] Processing: {filename}...")

        image_bgr = cv2.imread(input_image_path)
        if image_bgr is None:
            print(f"❌ 읽기 실패: {input_image_path}")
            continue

        original_h, original_w = image_bgr.shape[:2]

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # YOLO 예측
        results = model.predict(input_image_path, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            print(f"   -> ⚠️ No objects detected.")
            continue

        boxes_xyxy = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        # ★★★ 수정된 부분: 타겟 클래스 모두 필터링 ★★★
        # 발과 체커보드 영역의 박스를 모두 선택
        target_boxes_mask = np.isin(classes, TARGET_CLASS_IDS)
        target_boxes = boxes_xyxy[target_boxes_mask]
        
        if len(target_boxes) == 0:
            # 폴백 로직: 타겟 클래스가 없으면 감지된 모든 객체를 사용
            if len(boxes_xyxy) > 0:
                print(f"   -> ⚠️ Target classes not found. Using all {len(boxes_xyxy)} detected objects for robust mask generation.")
                target_boxes = boxes_xyxy
            else:
                print(f"   -> ⚠️ No objects detected, skipping.")
                continue

        # SAM 예측
        target_boxes_torch = torch.tensor(target_boxes, device=device)
        predictor.set_image(img_rgb)
        transformed_boxes = predictor.transform.apply_boxes_torch(
            target_boxes_torch, img_rgb.shape[:2]
        )

        with torch.no_grad():
            masks, scores, logits = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=True,
            )

        # 최고 점수 마스크 선택 및 합치기 (발 + 체커보드 모두 포함)
        best_mask_per_box = []
        for i in range(masks.shape[0]):
            box_masks = masks[i]
            box_scores = scores[i]
            best_idx = torch.argmax(box_scores)
            best_mask_per_box.append(box_masks[best_idx])

        best_masks = torch.stack(best_mask_per_box, dim=0)
        combined_mask = torch.any(best_masks > 0.5, dim=0) # 모든 마스크 합치기
        mask_np = combined_mask.cpu().numpy().astype(np.uint8)
        
        # cv2.resize를 사용하여 원본 크기로 확대
        final_mask_resized = cv2.resize(
            mask_np, 
            (original_w, original_h), 
            interpolation=cv2.INTER_NEAREST # 마스크는 Nearest Neighbor가 적합
        )
        
        final_mask = final_mask_resized * 255 # 0/1 -> 0/255
        # 배경 제거 (Clean Image 생성)
        clean = cv2.bitwise_and(image_bgr, image_bgr, mask=final_mask)

        # 저장
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, clean)
        print(f" -> Saved: {save_path}")

    print("🎉 All processing done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO + SAM Foot Segmentation Pipeline")
    
    # 인자(Argument) 설정
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input images directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save processed images")
    parser.add_argument("--yolo_path", type=str, default="models/best.pt", help="Path to YOLO weights")
    parser.add_argument("--sam_path", type=str, default="models/sam_vit_h_4b8939.pth", help="Path to SAM checkpoint")

    args = parser.parse_args()

    run_segmentation(args.input_dir, args.output_dir, args.yolo_path, args.sam_path)