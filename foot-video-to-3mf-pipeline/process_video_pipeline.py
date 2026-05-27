from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_INPUT_VIDEO = Path("input/foot_capture.mp4")
DEFAULT_FALLBACK_STL = Path("input/test.stl")
DEFAULT_FALLBACK_PLY = Path("input/foot_for_scale_2.ply")
OUTPUT_DIR = Path("output")
REPORTS_DIR = Path("reports")


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _tail(text: str, max_chars: int = 4000) -> str:
    return text[-max_chars:] if len(text) > max_chars else text


def _format_txt_report(result: dict[str, Any]) -> str:
    segmentation = result.get("segmentation", {})
    scaled = result.get("scaled_pipeline", {})
    outputs = result.get("outputs", {})
    lines = [
        "Video To 3MF Pipeline Report",
        "============================",
        "",
        f"Input video:       {result.get('input_video')}",
        f"Segmentation:      {segmentation.get('method', 'n/a')}",
        f"Foot STL:          {segmentation.get('foot_stl', 'n/a')}",
        f"Checkerboard PLY:  {segmentation.get('checker_ply', 'n/a')}",
        f"Status:            {result.get('status', 'n/a')}",
        "",
        "Outputs",
        f"  scaled_stl:      {outputs.get('scaled_stl', 'n/a')}",
        f"  sliced_3mf:      {outputs.get('sliced_3mf', 'n/a')}",
        f"  contains_gcode:  {outputs.get('contains_gcode', False)}",
        f"  report_json:     {outputs.get('scaled_report_json', 'n/a')}",
        f"  report_txt:      {outputs.get('scaled_report_txt', 'n/a')}",
        "",
        "Scaled Pipeline",
        f"  returncode:      {scaled.get('returncode', 'n/a')}",
        f"  command:         {' '.join(scaled.get('command', []))}",
    ]
    if result.get("error"):
        lines.extend(["", "Error", f"  {result['error']}"])
    return "\n".join(lines).rstrip() + "\n"


def _latest_report_for_stem(stem: str, suffix: str) -> Path | None:
    matches = sorted(REPORTS_DIR.glob(f"{stem}_scaled_*_report.{suffix}"))
    return matches[-1] if matches else None


def _read_scaled_report(report_path: Path | None) -> dict[str, Any]:
    if not report_path or not report_path.exists():
        return {}
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run video segmentation, PLY scale detection, STL processing, and 3MF slicing."
    )
    parser.add_argument("input_video", nargs="?", default=str(DEFAULT_INPUT_VIDEO))
    parser.add_argument("--team-module", default=None, help="Python module containing the teammate segment_video function.")
    parser.add_argument("--team-function", default="segment_video", help="Function name inside --team-module.")
    parser.add_argument("--team-script", default=None, help="Optional teammate segmentation script path.")
    parser.add_argument("--segmentation-result-json", default=None, help="Existing JSON containing foot_stl and checker_ply paths.")
    parser.add_argument(
        "--use-notion-team-code",
        action="store_true",
        help="Run the Notion-documented frame extraction + YOLO/SAM + foot PLY postprocessing flow.",
    )
    parser.add_argument("--assets-dir", default="assets", help="Directory containing best.pt and sam_vit_h_4b8939.pth.")
    parser.add_argument("--yolo-model", default=None, help="Path to YOLO segmentation model. Defaults to assets/best.pt.")
    parser.add_argument("--sam-checkpoint", default=None, help="Path to SAM checkpoint. Defaults to assets/sam_vit_h_4b8939.pth.")
    parser.add_argument("--device", default=None, help="Segmentation device: cuda, mps, or cpu. Defaults to auto inside the model code.")
    parser.add_argument(
        "--reconstruction-ply",
        default=None,
        help="External SfM/2DGS foot mesh PLY, usually unbounded_default_post.ply.",
    )
    parser.add_argument(
        "--run-reconstruction",
        action="store_true",
        help="Run COLMAP + 2DGS Docker instead of requiring --reconstruction-ply.",
    )
    parser.add_argument(
        "--reconstruction-image-set",
        choices=["foot", "checkerboard", "both"],
        default="foot",
        help="Which segmented image folder to feed into COLMAP/2DGS.",
    )
    parser.add_argument("--reconstruction-dataset-root", default="output/2dgs_dataset")
    parser.add_argument("--reconstruction-output-root", default="output/2dgs_output")
    parser.add_argument("--scene-name", default="foot_scene")
    parser.add_argument("--colmap-bin", default="colmap")
    parser.add_argument("--skip-colmap", action="store_true", help="Use an existing dataset/<scene>/sparse/0 COLMAP result.")
    parser.add_argument("--colmap-matcher", choices=["sequential", "exhaustive"], default="sequential")
    parser.add_argument("--docker-bin", default="docker")
    parser.add_argument("--2dgs-image", default="2dgs:cu118")
    parser.add_argument("--2dgs-dockerfile", default=None)
    parser.add_argument("--build-2dgs-image", action="store_true")
    parser.add_argument("--train-2dgs-args", default="--depth_ratio 0")
    parser.add_argument("--reconstruction-timeout-seconds", type=int, default=14400)
    parser.add_argument(
        "--scale-ply",
        default=None,
        help=(
            "Checkerboard PLY used to calculate scale factor. If omitted with "
            "--run-reconstruction, the checkerboard image set is reconstructed too."
        ),
    )
    parser.add_argument("--motion-threshold", type=float, default=12)
    parser.add_argument("--min-frame-interval", type=int, default=3)
    parser.add_argument("--blur-threshold", type=float, default=200)
    parser.add_argument("--sim-threshold", type=float, default=0.92)
    parser.add_argument("--mask-expand-pixels", type=int, default=4)
    parser.add_argument("--overwrite-frames", action="store_true")
    parser.add_argument("--postprocess-z-min", type=float, default=0.0)
    parser.add_argument("--postprocess-z-max", type=float, default=0.8)
    parser.add_argument("--postprocess-z-tol", type=float, default=1e-3)
    parser.add_argument("--postprocess-voxel-pitch", type=float, default=0.002)
    parser.add_argument(
        "--use-existing-assets",
        action="store_true",
        help="Skip real video segmentation and use --fallback-stl/--fallback-ply. Useful until teammate code is attached.",
    )
    parser.add_argument("--fallback-stl", default=str(DEFAULT_FALLBACK_STL))
    parser.add_argument("--fallback-ply", default=str(DEFAULT_FALLBACK_PLY))
    parser.add_argument("--checker-square-mm", type=float, default=30.0)
    parser.add_argument("--scale-resolution", type=int, default=1000)
    parser.add_argument("--simplify-threshold", type=int, default=None)
    parser.add_argument("--target-triangles", type=int, default=None)
    parser.add_argument("--no-slice", action="store_true")
    parser.add_argument("--slicer-engine", choices=["orca-legacy", "bambu", "orca"], default="orca-legacy")
    parser.add_argument("--slicer-bin", default=None)
    parser.add_argument("--force-support", action="store_true")
    parser.add_argument("--no-auto-support", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    input_video = _as_path(args.input_video)
    segmentation_dir = OUTPUT_DIR / "video_segmentation" / input_video.stem
    result: dict[str, Any] = {
        "input_video": str(input_video),
        "status": "pending",
        "segmentation": {},
        "scaled_pipeline": {},
        "outputs": {},
    }

    try:
        print("--- [1] Video segmentation / asset preparation ---")
        from video_segmentation import prepare_assets_from_video

        segmentation = prepare_assets_from_video(
            input_video,
            segmentation_dir,
            team_module=args.team_module,
            team_function=args.team_function,
            team_script=args.team_script,
            segmentation_result_json=args.segmentation_result_json,
            use_existing_assets=args.use_existing_assets,
            fallback_stl=args.fallback_stl,
            fallback_ply=args.scale_ply or args.fallback_ply,
            use_notion_team_code=args.use_notion_team_code,
            assets_dir=args.assets_dir,
            yolo_model_path=args.yolo_model,
            sam_checkpoint_path=args.sam_checkpoint,
            device=args.device,
            reconstruction_ply=args.reconstruction_ply,
            scale_ply=args.scale_ply,
            run_reconstruction=args.run_reconstruction,
            reconstruction_image_set=args.reconstruction_image_set,
            reconstruction_dataset_root=args.reconstruction_dataset_root,
            reconstruction_output_root=args.reconstruction_output_root,
            scene_name=args.scene_name,
            colmap_bin=args.colmap_bin,
            skip_colmap=args.skip_colmap,
            colmap_matcher=args.colmap_matcher,
            docker_bin=args.docker_bin,
            docker_image=args.__dict__["2dgs_image"],
            dockerfile_path=args.__dict__["2dgs_dockerfile"],
            build_2dgs_image=args.build_2dgs_image,
            train_2dgs_args=args.train_2dgs_args,
            reconstruction_timeout_seconds=args.reconstruction_timeout_seconds,
            motion_threshold=args.motion_threshold,
            min_interval=args.min_frame_interval,
            blur_threshold=args.blur_threshold,
            sim_threshold=args.sim_threshold,
            mask_expand_pixels=args.mask_expand_pixels,
            overwrite_frames=args.overwrite_frames,
            postprocess_z_min=args.postprocess_z_min,
            postprocess_z_max=args.postprocess_z_max,
            postprocess_z_tol=args.postprocess_z_tol,
            postprocess_voxel_pitch=args.postprocess_voxel_pitch,
            timeout_seconds=args.timeout_seconds,
        )
        result["segmentation"] = segmentation

        print("--- [2] Scale STL with checkerboard PLY and slice 3MF ---")
        sliced_output = OUTPUT_DIR / f"{input_video.stem}_scaled_sliced.3mf"
        command = [
            sys.executable,
            "process_scaled_stl.py",
            segmentation["foot_stl"],
            "--scale-ply",
            segmentation["checker_ply"],
            "--checker-square-mm",
            str(args.checker_square_mm),
            "--scale-resolution",
            str(args.scale_resolution),
            "--sliced-output",
            str(sliced_output),
            "--slicer-engine",
            args.slicer_engine,
        ]
        if args.simplify_threshold is not None:
            command.extend(["--simplify-threshold", str(args.simplify_threshold)])
        if args.target_triangles is not None:
            command.extend(["--target-triangles", str(args.target_triangles)])
        if args.no_slice:
            command.append("--no-slice")
        if args.slicer_bin:
            command.extend(["--slicer-bin", args.slicer_bin])
        if args.force_support:
            command.append("--force-support")
        if args.no_auto_support:
            command.append("--no-auto-support")

        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout_seconds,
        )
        result["scaled_pipeline"] = {
            "command": command,
            "returncode": completed.returncode,
            "stdout_tail": _tail(completed.stdout),
            "stderr_tail": _tail(completed.stderr),
        }
        if completed.returncode != 0:
            raise RuntimeError(
                "Scaled STL pipeline failed.\n"
                f"Return code: {completed.returncode}\n"
                f"STDOUT:\n{completed.stdout}\n"
                f"STDERR:\n{completed.stderr}"
            )

        scaled_stem = Path(segmentation["foot_stl"]).stem
        scaled_report_json = _latest_report_for_stem(scaled_stem, "json")
        scaled_report_txt = _latest_report_for_stem(scaled_stem, "txt")
        scaled_report = _read_scaled_report(scaled_report_json)
        slicing = scaled_report.get("slicing", {})
        result["outputs"] = {
            "scaled_stl": scaled_report.get("scaled_stl"),
            "sliced_3mf": slicing.get("output_file"),
            "contains_gcode": slicing.get("contains_gcode", False),
            "scaled_report_json": str(scaled_report_json) if scaled_report_json else None,
            "scaled_report_txt": str(scaled_report_txt) if scaled_report_txt else None,
        }
        result["status"] = "success"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_json = REPORTS_DIR / f"{input_video.stem}_video_pipeline_{timestamp}_report.json"
    report_txt = REPORTS_DIR / f"{input_video.stem}_video_pipeline_{timestamp}_report.txt"
    report_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report_txt.write_text(_format_txt_report(result), encoding="utf-8")

    print("Video pipeline complete" if result["status"] == "success" else "Video pipeline failed")
    print(f"- Input video:          {input_video}")
    print(f"- Segmentation method:  {result.get('segmentation', {}).get('method', 'n/a')}")
    print(f"- Foot STL:             {result.get('segmentation', {}).get('foot_stl', 'n/a')}")
    print(f"- Checkerboard PLY:     {result.get('segmentation', {}).get('checker_ply', 'n/a')}")
    print(f"- Scaled STL:           {result.get('outputs', {}).get('scaled_stl', 'n/a')}")
    print(f"- Sliced 3MF:           {result.get('outputs', {}).get('sliced_3mf', 'n/a')}")
    print(f"- Contains G-code:      {result.get('outputs', {}).get('contains_gcode', False)}")
    print(f"- JSON report:          {report_json}")
    print(f"- TXT report:           {report_txt}")
    if result["status"] != "success":
        print(f"- Error:                {result.get('error')}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
