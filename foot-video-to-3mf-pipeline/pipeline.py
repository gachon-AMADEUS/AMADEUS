from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
REPORTS_DIR = Path("reports")
ASSETS_DIR = Path("assets")
DEFAULT_DOCKERFILE = Path("docker/2dgs/Dockerfile")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


def _as_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path).expanduser()


def _require_file(path: str | Path, label: str) -> Path:
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"{label} not found: {file_path}")
    if not file_path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {file_path}")
    return file_path


def _find_input_video(input_dir: Path = INPUT_DIR) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(
        [
            item
            for item in input_dir.iterdir()
            if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS
        ],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not videos:
        raise FileNotFoundError(
            "No video file found in input/. Put one .mp4/.mov/.avi/.mkv file in input/, "
            "or pass --input-video."
        )
    if len(videos) > 1:
        print(f"[WARN] Multiple videos found in input/. Using newest file: {videos[0]}")
    return videos[0]


def _json_safe(data: Any) -> Any:
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, dict):
        return {str(key): _json_safe(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_json_safe(value) for value in data]
    if isinstance(data, tuple):
        return [_json_safe(value) for value in data]
    return data


def _scale_stl(input_stl: str | Path, output_stl: str | Path, scale_factor: float) -> dict[str, Any]:
    from mesh_pipeline import load_mesh

    input_file = _require_file(input_stl, "Processed STL")
    output_file = Path(output_stl).expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(input_file)
    before_bounds = mesh.bounds.tolist()
    mesh.apply_scale(float(scale_factor))
    after_bounds = mesh.bounds.tolist()
    mesh.export(output_file)

    return {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "scale_factor": float(scale_factor),
        "before_bounds": before_bounds,
        "after_bounds": after_bounds,
    }


def _slice_stl(
    input_stl: Path,
    output_3mf: Path,
    engine: str,
    slicer_bin: str | None,
    enable_support: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    if engine == "orca-legacy":
        from slicer import slice_with_orca_legacy_cli_safe

        return slice_with_orca_legacy_cli_safe(
            input_stl,
            output_3mf,
            slicer_bin=slicer_bin,
            enable_support=enable_support,
            timeout_seconds=timeout_seconds,
        )

    if engine == "orca":
        from slicer import slice_with_orca_slicer

        return slice_with_orca_slicer(
            input_stl,
            output_3mf,
            slicer_bin=slicer_bin,
            enable_support=enable_support,
            timeout_seconds=timeout_seconds,
        )

    if engine == "bambu":
        from slicer import slice_with_bambu_studio

        return slice_with_bambu_studio(
            input_stl,
            output_3mf,
            bambu_studio_bin=slicer_bin,
            enable_support=enable_support,
            timeout_seconds=timeout_seconds,
        )

    raise ValueError(f"Unknown slicer engine: {engine}")


def _format_txt_report(result: dict[str, Any]) -> str:
    outputs = result.get("outputs", {})
    final = result.get("final_report", {})
    scale = result.get("scale", {})
    slicing = result.get("slicing", {})
    lines = [
        "Foot Video To 3MF Pipeline Report",
        "=================================",
        "",
        f"Status:              {result.get('status')}",
        f"Input video:         {result.get('input_video')}",
        f"Foot STL:            {result.get('segmentation', {}).get('foot_stl')}",
        f"Checkerboard PLY:    {result.get('segmentation', {}).get('checker_ply')}",
        "",
        "Scale",
        f"  scale_factor:      {scale.get('scale_factor', 'n/a')}",
        f"  pixel_distance:    {scale.get('pixel_distance', 'n/a')}",
        "",
        "Final Mesh",
        f"  scaled_stl:        {outputs.get('scaled_stl')}",
        f"  triangle_count:    {final.get('triangle_count', 'n/a')}",
        f"  boundary_edges:    {final.get('boundary_edges', 'n/a')}",
        f"  non_manifold:      {final.get('non_manifold_edges', 'n/a')}",
        f"  watertight:        {final.get('watertight', 'n/a')}",
        "",
        "Slicing",
        f"  enabled:           {slicing.get('enabled')}",
        f"  status:            {slicing.get('status')}",
        f"  engine:            {slicing.get('engine')}",
        f"  output_3mf:        {outputs.get('sliced_3mf')}",
        f"  contains_gcode:    {slicing.get('contains_gcode', False)}",
        "",
        "Reports",
        f"  json:              {outputs.get('report_json')}",
        f"  txt:               {outputs.get('report_txt')}",
    ]
    if result.get("error"):
        lines.extend(["", "Error", str(result["error"])])
    return "\n".join(lines).rstrip() + "\n"


def run_pipeline(
    input_video: str | Path | None = None,
    assets_dir: str | Path = ASSETS_DIR,
    run_reconstruction: bool = True,
    build_2dgs_image: bool = True,
    reconstruction_ply: str | Path | None = None,
    scale_ply: str | Path | None = None,
    slicer_engine: str = "orca",
    slicer_bin: str | None = None,
    no_slice: bool = False,
    checker_square_mm: float = 30.0,
    scale_resolution: int = 1000,
    simplify_threshold: int | None = None,
    target_triangles: int | None = None,
    scene_name: str = "foot_scene",
    reconstruction_image_set: str = "foot",
    colmap_matcher: str = "sequential",
    docker_image: str = "2dgs:cu118",
    dockerfile_path: str | Path | None = DEFAULT_DOCKERFILE,
    colmap_bin: str = "colmap",
    docker_bin: str = "docker",
    device: str | None = None,
    motion_threshold: float = 12,
    min_interval: int = 3,
    blur_threshold: float | None = 200,
    sim_threshold: float | None = 0.92,
    mask_expand_pixels: int = 4,
    overwrite_frames: bool = False,
    timeout_seconds: int = 3600,
    reconstruction_timeout_seconds: int = 14400,
) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    video_file = _require_file(input_video, "Input video") if input_video else _find_input_video()
    asset_path = Path(assets_dir).expanduser()
    yolo_model = asset_path / "best.pt"
    sam_checkpoint = asset_path / "sam_vit_h_4b8939.pth"
    _require_file(yolo_model, "YOLO model assets/best.pt")
    _require_file(sam_checkpoint, "SAM checkpoint assets/sam_vit_h_4b8939.pth")
    if not run_reconstruction and (reconstruction_ply is None or scale_ply is None):
        raise ValueError(
            "--skip-reconstruction requires both --reconstruction-ply and --scale-ply."
        )

    result: dict[str, Any] = {
        "status": "pending",
        "input_video": str(video_file),
        "segmentation": {},
        "outputs": {},
    }

    try:
        print("--- [1] Video segmentation / 3D reconstruction assets ---")
        from video_segmentation import prepare_assets_from_video

        segmentation = prepare_assets_from_video(
            video_file,
            OUTPUT_DIR / "video_segmentation" / video_file.stem,
            use_notion_team_code=True,
            assets_dir=asset_path,
            yolo_model_path=yolo_model,
            sam_checkpoint_path=sam_checkpoint,
            device=device,
            reconstruction_ply=_as_path(reconstruction_ply),
            scale_ply=_as_path(scale_ply),
            run_reconstruction=run_reconstruction,
            reconstruction_image_set=reconstruction_image_set,
            reconstruction_dataset_root=OUTPUT_DIR / "2dgs_dataset",
            reconstruction_output_root=OUTPUT_DIR / "2dgs_output",
            scene_name=scene_name,
            colmap_bin=colmap_bin,
            colmap_matcher=colmap_matcher,
            docker_bin=docker_bin,
            docker_image=docker_image,
            dockerfile_path=_as_path(dockerfile_path),
            build_2dgs_image=build_2dgs_image,
            train_2dgs_args="--depth_ratio 0",
            motion_threshold=motion_threshold,
            min_interval=min_interval,
            blur_threshold=blur_threshold,
            sim_threshold=sim_threshold,
            mask_expand_pixels=mask_expand_pixels,
            overwrite_frames=overwrite_frames,
            reconstruction_timeout_seconds=reconstruction_timeout_seconds,
            timeout_seconds=timeout_seconds,
        )
        result["segmentation"] = segmentation

        print("--- [2] Analyze / repair / simplify STL ---")
        from mesh_pipeline import inspect_stl, process_stl, resolve_floating_regions

        process_kwargs: dict[str, Any] = {}
        if simplify_threshold is not None:
            process_kwargs["simplify_threshold"] = simplify_threshold
        if target_triangles is not None:
            process_kwargs["target_triangles"] = target_triangles

        stl_result = process_stl(segmentation["foot_stl"], OUTPUT_DIR, **process_kwargs)
        result["stl_processing"] = stl_result

        print("--- [3] Checkerboard PLY scale factor ---")
        from scale_from_ply import compute_scale_factor_from_ply

        scale_report = compute_scale_factor_from_ply(
            segmentation["checker_ply"],
            output_dir=OUTPUT_DIR / "scale_debug",
            resolution=scale_resolution,
            square_real_size_mm=checker_square_mm,
        )
        result["scale"] = scale_report

        print("--- [4] Apply real-size scale to STL ---")
        scaled_raw = OUTPUT_DIR / f"{video_file.stem}_scaled_mm_raw.stl"
        scaled_final = OUTPUT_DIR / f"{video_file.stem}_scaled_mm.stl"
        result["scale_stl"] = _scale_stl(
            stl_result["final_file"],
            scaled_raw,
            float(scale_report["scale_factor"]),
        )

        print("--- [5] Floating regions check / cleanup ---")
        floating = resolve_floating_regions(scaled_raw, scaled_final, bed_tolerance=0.05)
        result["floating_regions"] = floating
        final_report = inspect_stl(scaled_final)
        result["final_report"] = final_report

        support_recommended = bool(
            floating.get("support_recommended", False)
            or stl_result.get("support_recommended", False)
        )
        sliced_output = OUTPUT_DIR / f"{video_file.stem}_scaled_sliced.3mf"
        result["slicing"] = {
            "enabled": not no_slice,
            "status": "not_run" if no_slice else "pending",
            "engine": slicer_engine if not no_slice else None,
            "support_recommended": support_recommended,
            "enable_support": support_recommended,
            "output_file": str(sliced_output) if not no_slice else None,
        }

        if not no_slice:
            print("--- [6] Slice scaled STL to 3MF ---")
            slice_result = _slice_stl(
                scaled_final,
                sliced_output,
                slicer_engine,
                slicer_bin,
                enable_support=support_recommended,
                timeout_seconds=timeout_seconds,
            )
            result["slicing"].update(
                {
                    "status": "success",
                    "success": True,
                    "contains_gcode": slice_result.get("contains_gcode", False),
                    "gcode_files": slice_result.get("gcode_files", []),
                    "command": slice_result.get("command"),
                    "stdout_tail": (slice_result.get("stdout") or "")[-4000:],
                    "stderr_tail": (slice_result.get("stderr") or "")[-4000:],
                }
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_json = REPORTS_DIR / f"{video_file.stem}_pipeline_{timestamp}.json"
        report_txt = REPORTS_DIR / f"{video_file.stem}_pipeline_{timestamp}.txt"
        result["outputs"] = {
            "scaled_stl": str(scaled_final),
            "sliced_3mf": str(sliced_output) if not no_slice else None,
            "report_json": str(report_json),
            "report_txt": str(report_txt),
        }
        result["status"] = "success"
        report_json.write_text(json.dumps(_json_safe(result), indent=2, ensure_ascii=False), encoding="utf-8")
        report_txt.write_text(_format_txt_report(result), encoding="utf-8")
        return result
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_json = REPORTS_DIR / f"{video_file.stem}_pipeline_{timestamp}.json"
        report_txt = REPORTS_DIR / f"{video_file.stem}_pipeline_{timestamp}.txt"
        result["outputs"] = {"report_json": str(report_json), "report_txt": str(report_txt)}
        report_json.write_text(json.dumps(_json_safe(result), indent=2, ensure_ascii=False), encoding="utf-8")
        report_txt.write_text(_format_txt_report(result), encoding="utf-8")
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command pipeline: input video -> segmentation -> reconstruction -> STL repair/scale -> sliced 3MF."
    )
    parser.add_argument("--input-video", default=None, help="Video path. If omitted, the newest video in input/ is used.")
    parser.add_argument("--assets-dir", default=str(ASSETS_DIR))
    parser.add_argument("--reconstruction-ply", default=None, help="Use an existing foot PLY instead of running foot reconstruction.")
    parser.add_argument("--scale-ply", default=None, help="Use an existing checkerboard PLY. If omitted, checkerboard reconstruction is run.")
    parser.add_argument("--skip-reconstruction", action="store_true", help="Only valid when --reconstruction-ply and --scale-ply are provided.")
    parser.add_argument("--skip-docker-build", action="store_true", help="Use an already-built 2dgs:cu118 Docker image.")
    parser.add_argument("--slicer-engine", choices=["orca", "bambu", "orca-legacy"], default="orca")
    parser.add_argument("--slicer-bin", default=None)
    parser.add_argument("--no-slice", action="store_true")
    parser.add_argument("--checker-square-mm", type=float, default=30.0)
    parser.add_argument("--scale-resolution", type=int, default=1000)
    parser.add_argument("--simplify-threshold", type=int, default=None)
    parser.add_argument("--target-triangles", type=int, default=None)
    parser.add_argument("--scene-name", default="foot_scene")
    parser.add_argument("--reconstruction-image-set", choices=["foot", "checkerboard", "both"], default="foot")
    parser.add_argument("--colmap-matcher", choices=["sequential", "exhaustive"], default="sequential")
    parser.add_argument("--docker-image", default="2dgs:cu118")
    parser.add_argument("--2dgs-dockerfile", default=str(DEFAULT_DOCKERFILE))
    parser.add_argument("--colmap-bin", default="colmap")
    parser.add_argument("--docker-bin", default="docker")
    parser.add_argument("--device", default=None, help="YOLO/SAM device: cuda, cpu, or mps.")
    parser.add_argument("--motion-threshold", type=float, default=12)
    parser.add_argument("--min-interval", type=int, default=3)
    parser.add_argument("--blur-threshold", type=float, default=200)
    parser.add_argument("--sim-threshold", type=float, default=0.92)
    parser.add_argument("--mask-expand-pixels", type=int, default=4)
    parser.add_argument("--overwrite-frames", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--reconstruction-timeout-seconds", type=int, default=14400)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = run_pipeline(
            input_video=args.input_video,
            assets_dir=args.assets_dir,
            run_reconstruction=not args.skip_reconstruction,
            build_2dgs_image=not args.skip_docker_build,
            reconstruction_ply=args.reconstruction_ply,
            scale_ply=args.scale_ply,
            slicer_engine=args.slicer_engine,
            slicer_bin=args.slicer_bin,
            no_slice=args.no_slice,
            checker_square_mm=args.checker_square_mm,
            scale_resolution=args.scale_resolution,
            simplify_threshold=args.simplify_threshold,
            target_triangles=args.target_triangles,
            scene_name=args.scene_name,
            reconstruction_image_set=args.reconstruction_image_set,
            colmap_matcher=args.colmap_matcher,
            docker_image=args.docker_image,
            dockerfile_path=args.__dict__["2dgs_dockerfile"],
            colmap_bin=args.colmap_bin,
            docker_bin=args.docker_bin,
            device=args.device,
            motion_threshold=args.motion_threshold,
            min_interval=args.min_interval,
            blur_threshold=args.blur_threshold,
            sim_threshold=args.sim_threshold,
            mask_expand_pixels=args.mask_expand_pixels,
            overwrite_frames=args.overwrite_frames,
            timeout_seconds=args.timeout_seconds,
            reconstruction_timeout_seconds=args.reconstruction_timeout_seconds,
        )
    except Exception as exc:
        print("Pipeline failed")
        print(f"- Error: {exc}")
        return 1

    outputs = result.get("outputs", {})
    slicing = result.get("slicing", {})
    print("Pipeline complete")
    print(f"- Input video:       {result.get('input_video')}")
    print(f"- Scaled STL:        {outputs.get('scaled_stl')}")
    print(f"- Sliced 3MF:        {outputs.get('sliced_3mf')}")
    print(f"- Contains G-code:   {slicing.get('contains_gcode', False)}")
    print(f"- JSON report:       {outputs.get('report_json')}")
    print(f"- TXT report:        {outputs.get('report_txt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
