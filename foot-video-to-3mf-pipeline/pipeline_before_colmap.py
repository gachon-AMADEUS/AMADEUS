from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from frame_preprocessing import extract_frames_to
import yolo_sam_process


def run_video_pre_colmap(
    video_path: str | Path,
    output_root: str | Path = "output/video_segmentation",
    assets_dir: str | Path = "assets",
    yolo_model_path: str | Path | None = None,
    sam_checkpoint_path: str | Path | None = None,
    device: str | None = None,
    motion_threshold: float = 12,
    min_interval: int = 3,
    blur_threshold: float | None = 200,
    sim_threshold: float | None = 0.92,
    mask_expand_pixels: int = 4,
    overwrite_frames: bool = False,
) -> dict[str, Any]:
    """Video -> selected frames -> YOLO/SAM segmented image folders.

    This is the teammate's `pipeline_before_colmap.py` flow made callable from
    the final Python pipeline. It intentionally stops before SfM/COLMAP/2DGS,
    because those are external reconstruction stages in the current project.
    """
    vid_path = Path(video_path).expanduser()
    if not vid_path.exists():
        raise FileNotFoundError(f"Video not found: {vid_path}")

    root = Path(output_root).expanduser() / vid_path.stem
    original_image_dir = root / "original_image"
    segmentation_image_dir = root / "segmentation"
    asset_path = Path(assets_dir).expanduser()

    yolo_model = Path(yolo_model_path).expanduser() if yolo_model_path else asset_path / "best.pt"
    sam_checkpoint = (
        Path(sam_checkpoint_path).expanduser()
        if sam_checkpoint_path
        else asset_path / "sam_vit_h_4b8939.pth"
    )

    print(f"저장할 경로: {original_image_dir.absolute()}, {segmentation_image_dir.absolute()}")
    os.makedirs(original_image_dir, exist_ok=True)
    os.makedirs(segmentation_image_dir, exist_ok=True)

    extraction_result: dict[str, Any] | None = None
    has_existing_frames = any(file.is_file() for file in original_image_dir.iterdir())
    if overwrite_frames or not has_existing_frames:
        extraction_result = extract_frames_to(
            video_path=vid_path,
            output_dir=original_image_dir,
            motion_threshold=motion_threshold,
            min_interval=min_interval,
            blur_threshold=blur_threshold,
            sim_threshold=sim_threshold,
        )
    else:
        extraction_result = {
            "video_path": str(vid_path),
            "output_dir": str(original_image_dir),
            "saved": len(list(original_image_dir.glob("*.jpg"))),
            "skipped_extract": True,
        }

    foot_dir = segmentation_image_dir / "foot"
    checkerboard_dir = segmentation_image_dir / "checkerboard"
    both_dir = segmentation_image_dir / "both"
    os.makedirs(foot_dir, exist_ok=True)
    os.makedirs(checkerboard_dir, exist_ok=True)
    os.makedirs(both_dir, exist_ok=True)

    foot_result = yolo_sam_process.run_segmentation(
        original_image_dir,
        foot_dir,
        "foot",
        yolo_model,
        sam_checkpoint,
        device,
        mask_expand_pixels=mask_expand_pixels,
    )
    checkerboard_result = yolo_sam_process.run_segmentation(
        original_image_dir,
        checkerboard_dir,
        "checkerboard",
        yolo_model,
        sam_checkpoint,
        device,
        mask_expand_pixels=mask_expand_pixels,
    )
    both_result = yolo_sam_process.run_segmentation(
        original_image_dir,
        both_dir,
        "both",
        yolo_model,
        sam_checkpoint,
        device,
        mask_expand_pixels=mask_expand_pixels,
    )

    return {
        "video_path": str(vid_path),
        "root": str(root),
        "original_image_dir": str(original_image_dir),
        "segmentation_dir": str(segmentation_image_dir),
        "foot_images_dir": str(foot_dir),
        "checkerboard_images_dir": str(checkerboard_dir),
        "both_images_dir": str(both_dir),
        "frame_extraction": extraction_result,
        "foot_segmentation": foot_result,
        "checkerboard_segmentation": checkerboard_result,
        "both_segmentation": both_result,
        "yolo_model_path": str(yolo_model),
        "sam_checkpoint_path": str(sam_checkpoint),
    }


def do_pipeline() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description=(
            "영상 하나를 입력 받아 프레임 추출 후 YOLO/SAM으로 "
            "foot/checkerboard/both segmentation 이미지를 저장합니다."
        )
    )
    parser.add_argument("-p", "--path", required=True, help=".mp4 상대경로 혹은 절대경로")
    parser.add_argument("--output-root", default="output/video_segmentation")
    parser.add_argument("--assets-dir", default="assets")
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite-frames", action="store_true")
    args = parser.parse_args()

    return run_video_pre_colmap(
        video_path=args.path,
        output_root=args.output_root,
        assets_dir=args.assets_dir,
        device=args.device,
        overwrite_frames=args.overwrite_frames,
    )


if __name__ == "__main__":
    do_pipeline()
