from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Union

import cv2
import numpy as np

try:
    import torch
    from segment_anything import SamPredictor, sam_model_registry
    from ultralytics import YOLO
except ImportError as exc:
    torch = None
    SamPredictor = None
    sam_model_registry = None
    YOLO = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _require_segmentation_deps() -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "YOLO/SAM dependencies are not installed. Install the optional video "
            "dependencies first, then run this stage again.\n"
            "Example: pip install -r requirements-video.txt\n"
            f"Original import error: {_IMPORT_ERROR}"
        )


def _expand_mask(mask: np.ndarray, expand_pixels: int) -> np.ndarray:
    if expand_pixels <= 0:
        return mask
    kernel_size = expand_pixels * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def run_segmentation(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    target_mode: Literal["foot", "checkerboard", "both"] = "both",
    yolo_model_path: Union[str, Path] = "best.pt",
    sam_checkpoint_path: Union[str, Path] = "sam_vit_h_4b8939.pth",
    device: str | None = None,
    mask_expand_pixels: int = 4,
) -> dict[str, Any]:
    """Run the teammate YOLO + SAM image segmentation stage.

    Notion note: brightness correction did not help much, while expanding the
    SAM mask by roughly 3-5 px helped reduce black gaps around boundaries.
    """
    _require_segmentation_deps()

    input_path = Path(input_dir).expanduser()
    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input image directory not found: {input_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Info] Device: {device}")
    print(f"[Info] Input: {input_path}")
    print(f"[Info] Output: {output_path}")
    print(f"[Info] Mode: {target_mode}")

    print(">> 모델 로딩 중...")
    yolo_model = YOLO(str(yolo_model_path))
    sam = sam_model_registry["vit_h"](checkpoint=str(sam_checkpoint_path))
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    reversed_names = {v: k for k, v in yolo_model.names.items()}

    target_class_ids: list[int] = []
    if target_mode in ["foot", "both"]:
        foot_id = reversed_names.get("foot")
        if foot_id is not None:
            target_class_ids.append(foot_id)

    if target_mode in ["checkerboard", "both"]:
        checkerboard_id = reversed_names.get("checkerboard")
        if checkerboard_id is not None:
            target_class_ids.append(checkerboard_id)

    if not target_class_ids:
        raise RuntimeError("'foot' 또는 'checkerboard' 클래스를 YOLO 모델에서 찾을 수 없습니다.")

    image_files = sorted(input_path.glob("*.jpg"))
    print(f">> 총 {len(image_files)}장의 이미지 처리를 시작합니다.")

    success_count = 0
    skipped_count = 0
    for index, img_file in enumerate(image_files):
        filename = img_file.name
        image_bgr = cv2.imread(str(img_file))
        if image_bgr is None:
            skipped_count += 1
            continue

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        results = yolo_model.predict(str(img_file), verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            print(f"  [Skip] 검출된 객체 없음: {filename}")
            skipped_count += 1
            continue

        result = results[0]
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        mask_indices = np.isin(classes, target_class_ids)
        target_boxes = boxes_xyxy[mask_indices]

        if len(target_boxes) == 0:
            print(f"  [Skip] 타겟({target_mode}) 없음: {filename}")
            skipped_count += 1
            continue

        target_boxes_torch = torch.tensor(target_boxes, device=device)
        sam_predictor.set_image(img_rgb)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(
            target_boxes_torch,
            img_rgb.shape[:2],
        )

        with torch.no_grad():
            masks, scores, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=True,
            )

        best_mask_per_box = []
        for box_index in range(masks.shape[0]):
            best_idx = torch.argmax(scores[box_index])
            best_mask_per_box.append(masks[box_index][best_idx])

        best_masks = torch.stack(best_mask_per_box, dim=0)
        combined_mask = torch.any(best_masks > 0.5, dim=0)

        final_mask = (combined_mask.cpu().numpy().astype(np.uint8)) * 255
        final_mask = _expand_mask(final_mask, mask_expand_pixels)

        clean_image = cv2.bitwise_and(image_bgr, image_bgr, mask=final_mask)
        save_path = output_path / filename
        cv2.imwrite(str(save_path), clean_image)
        success_count += 1

        if (index + 1) % 10 == 0:
            print(f"  ... {index + 1}/{len(image_files)} 완료")

    print(f">> 작업 완료. {success_count}/{len(image_files)}장 저장됨.")
    return {
        "input_dir": str(input_path),
        "output_dir": str(output_path),
        "target_mode": target_mode,
        "success_count": success_count,
        "skipped_count": skipped_count,
        "total_count": len(image_files),
        "mask_expand_pixels": mask_expand_pixels,
    }


if __name__ == "__main__":
    run_segmentation(
        input_dir="태량_original",
        output_dir="태량_segmented",
        target_mode="both",
    )
