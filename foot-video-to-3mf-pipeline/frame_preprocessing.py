from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _frame_similarity(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    resized_a = cv2.resize(gray_a, (160, 90), interpolation=cv2.INTER_AREA)
    resized_b = cv2.resize(gray_b, (160, 90), interpolation=cv2.INTER_AREA)
    mean_diff = np.mean(cv2.absdiff(resized_a, resized_b)) / 255.0
    return float(1.0 - mean_diff)


def extract_frames_to(
    video_path: str | Path,
    output_dir: str | Path,
    motion_threshold: float = 12,
    min_interval: int = 3,
    blur_threshold: float | None = None,
    sim_threshold: float | None = None,
    filename_prefix: str = "frame",
) -> dict[str, Any]:
    """Extract useful frames from a video.

    This keeps the teammate's original motion-based idea, while accepting a
    direct video path/output folder so the final pipeline can call it.
    """
    video_file = Path(video_path).expanduser()
    if not video_file.exists():
        raise FileNotFoundError(f"Video not found: {video_file}")

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_file))
    ret, prev = cap.read()
    if not ret or prev is None:
        cap.release()
        raise RuntimeError(f"Could not read first frame from video: {video_file}")

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    last_saved_gray: np.ndarray | None = None
    saved = 0
    frame_count = 1
    frame_count_after_save = min_interval

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_count_after_save += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_value = float(np.mean(cv2.absdiff(gray, prev_gray)))

        if motion_value <= motion_threshold:
            continue
        if frame_count_after_save < min_interval:
            continue

        blur_value = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_threshold is not None and blur_value < blur_threshold:
            continue

        similarity = None
        if sim_threshold is not None and last_saved_gray is not None:
            similarity = _frame_similarity(gray, last_saved_gray)
            if similarity >= sim_threshold:
                continue

        output_file = output_path / f"{filename_prefix}_{saved:04d}.jpg"
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb_frame).save(output_file)

        prev_gray = gray
        last_saved_gray = gray
        saved += 1
        frame_count_after_save = 0

    cap.release()
    return {
        "video_path": str(video_file),
        "output_dir": str(output_path),
        "saved": saved,
        "frame_count": frame_count,
        "motion_threshold": motion_threshold,
        "min_interval": min_interval,
        "blur_threshold": blur_threshold,
        "sim_threshold": sim_threshold,
    }


def pipeline_extrace_frame(target_name: str = "현동", target_num: str = 1) -> dict[str, Any]:
    """Compatibility wrapper for the teammate's original folder convention."""
    base_path = os.path.join(os.path.abspath("."), target_name)
    video_path = os.path.join(base_path, f"{target_name}{target_num}.mp4")
    output_dir = os.path.join(base_path, f"{target_name}{target_num}")
    total_output_dir = os.path.join(base_path, f"{target_name}_all_frames")
    os.makedirs(total_output_dir, exist_ok=True)

    result = extract_frames_to(video_path, output_dir, motion_threshold=12, min_interval=3)

    for frame in sorted(Path(output_dir).glob("*.jpg")):
        total_output_path = Path(total_output_dir) / frame.name.replace("frame_", f"frame{target_num}_")
        if not total_output_path.exists():
            total_output_path.write_bytes(frame.read_bytes())

    output_meta_path = os.path.join(base_path, f"{target_name}_meta.txt")
    with open(output_meta_path, "at", encoding="utf-8") as meta_file:
        meta_file.write(
            f"{target_name}{target_num}.mp4 -> "
            f"{result['saved']} / {result['frame_count']} frame\n"
        )
    return result


if __name__ == "__main__":
    name_list = ["태량", "현동", "찬우", "호준"]
    num_list = list(range(1, 9))
    for name in name_list:
        for num in num_list:
            pipeline_extrace_frame(name, str(num))
