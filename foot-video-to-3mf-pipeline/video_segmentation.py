from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


FOOT_STL_KEYS = (
    "foot_stl",
    "foot_stl_path",
    "stl_path",
    "mesh_stl",
    "mesh_path",
    "foot_mesh",
    "object_stl",
)
CHECKER_PLY_KEYS = (
    "checker_ply",
    "checkerboard_ply",
    "scale_ply",
    "ply_path",
    "checker_path",
    "checkerboard_path",
)
FOOT_PLY_KEYS = (
    "foot_ply",
    "foot_ply_path",
    "reconstruction_ply",
    "mesh_ply",
    "unbounded_default_post_ply",
)


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _require_file(path: str | Path, label: str) -> Path:
    file_path = _as_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{label} not found: {file_path}")
    if not file_path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {file_path}")
    return file_path


def _copy_asset(source: Path, output_dir: Path, name: str) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / name
    shutil.copy2(source, target)
    return str(target)


def _find_key(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in data and data[key]:
            return data[key]
    return None


def _resolve_relative(path: str | Path, base_dir: Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    return resolved


def _normalize_result(raw_result: Any, base_dir: Path) -> dict[str, Any]:
    if raw_result is None:
        raise ValueError("Segmentation returned None.")

    if isinstance(raw_result, (str, Path)):
        result_path = _require_file(raw_result, "Segmentation result JSON")
        data = json.loads(result_path.read_text(encoding="utf-8"))
    elif isinstance(raw_result, dict):
        data = dict(raw_result)
    elif isinstance(raw_result, (list, tuple)) and len(raw_result) >= 2:
        first = Path(raw_result[0])
        second = Path(raw_result[1])
        if first.suffix.lower() == ".ply" and second.suffix.lower() == ".stl":
            data = {"checker_ply": str(first), "foot_stl": str(second)}
        else:
            data = {"foot_stl": str(first), "checker_ply": str(second)}
    else:
        raise TypeError(
            "Segmentation result must be a dict, JSON path, or tuple/list "
            "containing foot STL and checkerboard PLY paths."
        )

    foot_stl = _find_key(data, FOOT_STL_KEYS)
    foot_ply = _find_key(data, FOOT_PLY_KEYS)
    checker_ply = _find_key(data, CHECKER_PLY_KEYS)

    if not foot_stl and foot_ply:
        foot_ply_path = _resolve_relative(foot_ply, base_dir)
        postprocessed = _postprocess_foot_ply(
            _require_file(foot_ply_path, "Segmented foot PLY"),
            base_dir,
            foot_ply_path.stem,
            z_min=0.0,
            z_max=0.8,
            z_tol=1e-3,
            voxel_pitch=0.002,
        )
        foot_stl = postprocessed["foot_stl"]
        data["foot_postprocessing"] = postprocessed["postprocess_report"]

    if not foot_stl or not checker_ply:
        raise ValueError(
            "Segmentation result must include foot STL and checkerboard PLY. "
            f"Known STL keys: {FOOT_STL_KEYS}; known foot PLY keys: {FOOT_PLY_KEYS}; "
            f"known checker PLY keys: {CHECKER_PLY_KEYS}; "
            f"received keys: {sorted(data.keys())}"
        )

    foot_path = _resolve_relative(foot_stl, base_dir)
    checker_path = _resolve_relative(checker_ply, base_dir)

    return {
        "foot_stl": str(_require_file(foot_path, "Segmented foot STL")),
        "checker_ply": str(_require_file(checker_path, "Segmented checkerboard PLY")),
        "raw_result": data,
    }


def _find_first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists() and path.is_file():
            return path
    return None


def _postprocess_foot_ply(
    reconstruction_ply: Path,
    output_dir: Path,
    video_stem: str,
    z_min: float,
    z_max: float,
    z_tol: float,
    voxel_pitch: float,
) -> dict[str, Any]:
    from foot_postprocessing import process_foot_with_manual_caps

    output_stl = output_dir / f"{video_stem}_foot_from_2dgs.stl"
    postprocess_report = process_foot_with_manual_caps(
        reconstruction_ply,
        output_stl,
        z_min=z_min,
        z_max=z_max,
        z_tol=z_tol,
        voxel_pitch=voxel_pitch,
    )
    return {
        "foot_stl": str(_require_file(output_stl, "Postprocessed foot STL")),
        "postprocess_report": postprocess_report,
    }


def _run_notion_team_pipeline(
    video_path: Path,
    output_dir: Path,
    assets_dir: Path,
    yolo_model_path: Path | None,
    sam_checkpoint_path: Path | None,
    device: str | None,
    reconstruction_ply: Path | None,
    checker_ply: Path | None,
    run_reconstruction: bool,
    reconstruction_image_set: str,
    dataset_root: Path,
    reconstruction_output_root: Path,
    scene_name: str,
    colmap_bin: str,
    skip_colmap: bool,
    colmap_matcher: str,
    colmap_vocab_tree_path: Path | None,
    docker_bin: str,
    docker_image: str,
    dockerfile_path: Path | None,
    build_2dgs_image: bool,
    train_2dgs_args: str,
    two_dgs_mesh_res_list: str,
    two_dgs_mesh_depth_ratio: str,
    two_dgs_mesh_num_cluster: int,
    two_dgs_wait_gpu_min_free_mb: int,
    two_dgs_wait_gpu_timeout_sec: int,
    reconstruction_timeout_seconds: int,
    motion_threshold: float,
    min_interval: int,
    blur_threshold: float | None,
    sim_threshold: float | None,
    mask_expand_pixels: int,
    overwrite_frames: bool,
    max_reconstruction_frames: int | None,
    postprocess_z_min: float,
    postprocess_z_max: float,
    postprocess_z_tol: float,
    postprocess_voxel_pitch: float,
) -> dict[str, Any]:
    from pipeline_before_colmap import run_video_pre_colmap

    if run_reconstruction:
        from reconstruction_pipeline import explain_local_gpu_blockers

        blockers = explain_local_gpu_blockers(docker_bin)
        if blockers:
            raise RuntimeError(
                "This machine cannot run the COLMAP + 2DGS Docker reconstruction stage yet.\n"
                + "\n".join(f"- {blocker}" for blocker in blockers)
            )

    pre_colmap_report = run_video_pre_colmap(
        video_path=video_path,
        output_root=output_dir.parent,
        assets_dir=assets_dir,
        yolo_model_path=yolo_model_path,
        sam_checkpoint_path=sam_checkpoint_path,
        device=device,
        motion_threshold=motion_threshold,
        min_interval=min_interval,
        blur_threshold=blur_threshold,
        sim_threshold=sim_threshold,
        mask_expand_pixels=mask_expand_pixels,
        overwrite_frames=overwrite_frames,
        max_reconstruction_frames=max_reconstruction_frames,
    )

    if reconstruction_ply is not None:
        reconstruction_file = _require_file(reconstruction_ply, "2DGS reconstruction PLY")
    else:
        reconstruction_file = _find_first_existing(
            [
                output_dir / "unbounded_default_post.ply",
                output_dir / "2dgs" / "unbounded_default_post.ply",
                output_dir / "mesh" / "unbounded_default_post.ply",
                output_dir / "unbounded_default.ply",
            ]
        )

    if reconstruction_file is None and run_reconstruction:
        from reconstruction_pipeline import run_reconstruction_from_segmented_images

        image_dir_key = {
            "foot": "foot_images_dir",
            "checkerboard": "checkerboard_images_dir",
            "both": "both_images_dir",
        }.get(reconstruction_image_set)
        if image_dir_key is None:
            raise ValueError("--reconstruction-image-set must be one of: foot, checkerboard, both")

        reconstruction_report = run_reconstruction_from_segmented_images(
            images_dir=pre_colmap_report[image_dir_key],
            dataset_root=dataset_root,
            output_root=reconstruction_output_root,
            scene_name=scene_name,
            colmap_bin=colmap_bin,
            docker_bin=docker_bin,
            docker_image=docker_image,
            dockerfile_path=dockerfile_path,
            build_image=build_2dgs_image,
            skip_colmap=skip_colmap,
            matcher=colmap_matcher,
            vocab_tree_path=colmap_vocab_tree_path,
            train_args=train_2dgs_args,
            mesh_res_list=two_dgs_mesh_res_list,
            mesh_depth_ratio=two_dgs_mesh_depth_ratio,
            mesh_num_cluster=two_dgs_mesh_num_cluster,
            wait_gpu_min_free_mb=two_dgs_wait_gpu_min_free_mb,
            wait_gpu_timeout_sec=two_dgs_wait_gpu_timeout_sec,
            timeout_seconds=reconstruction_timeout_seconds,
        )
        reconstruction_file = _require_file(
            reconstruction_report["reconstruction_ply"],
            "Generated 2DGS reconstruction PLY",
        )
        pre_colmap_report["reconstruction"] = reconstruction_report

    if reconstruction_file is None:
        raise FileNotFoundError(
            "YOLO/SAM segmentation finished, but the external SfM/2DGS PLY was not found. "
            "Run with --run-reconstruction on a Docker/CUDA-capable machine, or provide "
            "--reconstruction-ply, usually the unbounded_default_post.ply file."
        )

    if checker_ply is not None:
        checker_file = _require_file(checker_ply, "Checkerboard scale PLY")
    else:
        checker_file = _find_first_existing(
            [
                output_dir / "checkerboard.ply",
                output_dir / "scale.ply",
                output_dir / "checkerboard_scale.ply",
                output_dir / "scale" / "checkerboard.ply",
            ]
        )

    if checker_file is None and run_reconstruction:
        from reconstruction_pipeline import run_reconstruction_from_segmented_images

        checker_scene_name = f"{scene_name}_checkerboard"
        checker_reconstruction_report = run_reconstruction_from_segmented_images(
            images_dir=pre_colmap_report["checkerboard_images_dir"],
            dataset_root=dataset_root,
            output_root=reconstruction_output_root,
            scene_name=checker_scene_name,
            colmap_bin=colmap_bin,
            docker_bin=docker_bin,
            docker_image=docker_image,
            dockerfile_path=dockerfile_path,
            build_image=False,
            skip_colmap=skip_colmap,
            matcher=colmap_matcher,
            vocab_tree_path=colmap_vocab_tree_path,
            train_args=train_2dgs_args,
            mesh_res_list=two_dgs_mesh_res_list,
            mesh_depth_ratio=two_dgs_mesh_depth_ratio,
            mesh_num_cluster=two_dgs_mesh_num_cluster,
            wait_gpu_min_free_mb=two_dgs_wait_gpu_min_free_mb,
            wait_gpu_timeout_sec=two_dgs_wait_gpu_timeout_sec,
            timeout_seconds=reconstruction_timeout_seconds,
        )
        checker_file = _require_file(
            checker_reconstruction_report["reconstruction_ply"],
            "Generated checkerboard scale PLY",
        )
        pre_colmap_report["checkerboard_reconstruction"] = checker_reconstruction_report

    if checker_file is None:
        raise FileNotFoundError(
            "Checkerboard scale PLY was not found. Provide --scale-ply with the "
            "PLY used for checkerboard scale-factor calculation, or run with "
            "--run-reconstruction so the checkerboard segmentation can be reconstructed too."
        )

    postprocessed = _postprocess_foot_ply(
        reconstruction_file,
        output_dir,
        video_path.stem,
        z_min=postprocess_z_min,
        z_max=postprocess_z_max,
        z_tol=postprocess_z_tol,
        voxel_pitch=postprocess_voxel_pitch,
    )

    return {
        "foot_stl": postprocessed["foot_stl"],
        "checker_ply": str(checker_file),
        "raw_result": {
            "pre_colmap": pre_colmap_report,
            "reconstruction_ply": str(reconstruction_file),
            "checker_ply": str(checker_file),
            "foot_postprocessing": postprocessed["postprocess_report"],
        },
    }


def _call_team_module(
    video_path: Path,
    output_dir: Path,
    module_name: str,
    function_name: str,
) -> dict[str, Any]:
    module = importlib.import_module(module_name)
    segment_func = getattr(module, function_name)

    attempts = (
        lambda: segment_func(str(video_path), str(output_dir)),
        lambda: segment_func(video_path=str(video_path), output_dir=str(output_dir)),
        lambda: segment_func(input_video=str(video_path), output_dir=str(output_dir)),
        lambda: segment_func(str(video_path)),
    )
    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return _normalize_result(attempt(), output_dir)
        except TypeError as exc:
            last_error = exc
            continue

    if last_error:
        raise last_error
    raise RuntimeError(f"Failed to call {module_name}.{function_name}.")


def _call_team_script(
    video_path: Path,
    output_dir: Path,
    script_path: Path,
    result_json: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    _require_file(script_path, "Team segmentation script")
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "VIDEO_INPUT_PATH": str(video_path),
            "SEGMENTATION_OUTPUT_DIR": str(output_dir),
            "SEGMENTATION_RESULT_JSON": str(result_json),
        }
    )
    command = [
        sys.executable,
        str(script_path),
        str(video_path),
        str(output_dir),
        str(result_json),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Team segmentation script failed.\n"
            f"Return code: {completed.returncode}\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )
    return _normalize_result(result_json, output_dir)


def prepare_assets_from_video(
    video_path: str | Path,
    output_dir: str | Path,
    team_module: str | None = None,
    team_function: str = "segment_video",
    team_script: str | Path | None = None,
    segmentation_result_json: str | Path | None = None,
    use_existing_assets: bool = False,
    fallback_stl: str | Path | None = None,
    fallback_ply: str | Path | None = None,
    use_notion_team_code: bool = False,
    assets_dir: str | Path = "assets",
    yolo_model_path: str | Path | None = None,
    sam_checkpoint_path: str | Path | None = None,
    device: str | None = None,
    reconstruction_ply: str | Path | None = None,
    scale_ply: str | Path | None = None,
    run_reconstruction: bool = False,
    reconstruction_image_set: str = "foot",
    reconstruction_dataset_root: str | Path = "output/2dgs_dataset",
    reconstruction_output_root: str | Path = "output/2dgs_output",
    scene_name: str = "foot_scene",
    colmap_bin: str = "colmap",
    skip_colmap: bool = False,
    colmap_matcher: str = "sequential",
    colmap_vocab_tree_path: str | Path | None = None,
    docker_bin: str = "docker",
    docker_image: str = "2dgs:cu118",
    dockerfile_path: str | Path | None = None,
    build_2dgs_image: bool = False,
    train_2dgs_args: str = "--depth_ratio 0",
    two_dgs_mesh_res_list: str = "512 384 256 192",
    two_dgs_mesh_depth_ratio: str = "0",
    two_dgs_mesh_num_cluster: int = 30,
    two_dgs_wait_gpu_min_free_mb: int = 2048,
    two_dgs_wait_gpu_timeout_sec: int = 600,
    reconstruction_timeout_seconds: int = 14400,
    motion_threshold: float = 12,
    min_interval: int = 3,
    blur_threshold: float | None = 200,
    sim_threshold: float | None = 0.92,
    mask_expand_pixels: int = 4,
    overwrite_frames: bool = False,
    max_reconstruction_frames: int | None = 120,
    postprocess_z_min: float = 0.0,
    postprocess_z_max: float = 0.8,
    postprocess_z_tol: float = 1e-3,
    postprocess_voxel_pitch: float = 0.002,
    timeout_seconds: int = 3600,
) -> dict[str, Any]:
    """Prepare foot STL and checkerboard PLY from a video segmentation stage.

    This wrapper intentionally does not rewrite teammate segmentation code. It
    only calls a provided module/script or reads its JSON result, then normalizes
    the output paths for the existing STL/PLY pipeline.
    """
    output_path = _as_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    video_file = _as_path(video_path)
    if not use_existing_assets:
        _require_file(video_file, "Input video")

    if segmentation_result_json:
        result_file = _require_file(segmentation_result_json, "Segmentation result JSON")
        normalized = _normalize_result(result_file, result_file.parent)
        method = "result_json"
    elif use_notion_team_code:
        normalized = _run_notion_team_pipeline(
            video_file,
            output_path,
            assets_dir=_as_path(assets_dir),
            yolo_model_path=_as_path(yolo_model_path) if yolo_model_path else None,
            sam_checkpoint_path=_as_path(sam_checkpoint_path) if sam_checkpoint_path else None,
            device=device,
            reconstruction_ply=_as_path(reconstruction_ply) if reconstruction_ply else None,
            checker_ply=_as_path(scale_ply) if scale_ply else None,
            run_reconstruction=run_reconstruction,
            reconstruction_image_set=reconstruction_image_set,
            dataset_root=_as_path(reconstruction_dataset_root),
            reconstruction_output_root=_as_path(reconstruction_output_root),
            scene_name=scene_name,
            colmap_bin=colmap_bin,
            skip_colmap=skip_colmap,
            colmap_matcher=colmap_matcher,
            colmap_vocab_tree_path=_as_path(colmap_vocab_tree_path) if colmap_vocab_tree_path else None,
            docker_bin=docker_bin,
            docker_image=docker_image,
            dockerfile_path=_as_path(dockerfile_path) if dockerfile_path else None,
            build_2dgs_image=build_2dgs_image,
            train_2dgs_args=train_2dgs_args,
            two_dgs_mesh_res_list=two_dgs_mesh_res_list,
            two_dgs_mesh_depth_ratio=two_dgs_mesh_depth_ratio,
            two_dgs_mesh_num_cluster=two_dgs_mesh_num_cluster,
            two_dgs_wait_gpu_min_free_mb=two_dgs_wait_gpu_min_free_mb,
            two_dgs_wait_gpu_timeout_sec=two_dgs_wait_gpu_timeout_sec,
            reconstruction_timeout_seconds=reconstruction_timeout_seconds,
            motion_threshold=motion_threshold,
            min_interval=min_interval,
            blur_threshold=blur_threshold,
            sim_threshold=sim_threshold,
            mask_expand_pixels=mask_expand_pixels,
            overwrite_frames=overwrite_frames,
            max_reconstruction_frames=max_reconstruction_frames,
            postprocess_z_min=postprocess_z_min,
            postprocess_z_max=postprocess_z_max,
            postprocess_z_tol=postprocess_z_tol,
            postprocess_voxel_pitch=postprocess_voxel_pitch,
        )
        method = "notion_team_code:frame_extract+yolo_sam+foot_postprocessing"
    elif team_module:
        normalized = _call_team_module(
            video_file,
            output_path,
            team_module=team_module,
            function_name=team_function,
        )
        method = f"module:{team_module}.{team_function}"
    elif team_script:
        result_file = output_path / "segmentation_result.json"
        normalized = _call_team_script(
            video_file,
            output_path,
            script_path=_as_path(team_script),
            result_json=result_file,
            timeout_seconds=timeout_seconds,
        )
        method = f"script:{team_script}"
    elif use_existing_assets and fallback_stl and fallback_ply:
        fallback_stl_file = _require_file(fallback_stl, "Fallback foot STL")
        fallback_ply_file = _require_file(fallback_ply, "Fallback checkerboard PLY")
        normalized = {
            "foot_stl": _copy_asset(fallback_stl_file, output_path, f"{video_file.stem}_foot.stl"),
            "checker_ply": _copy_asset(fallback_ply_file, output_path, f"{video_file.stem}_checkerboard.ply"),
            "raw_result": {
                "fallback_stl": str(fallback_stl_file),
                "fallback_ply": str(fallback_ply_file),
            },
        }
        method = "existing_assets_fallback"
    else:
        raise RuntimeError(
            "No video segmentation source was configured. Provide one of: "
            "--team-module, --team-script, --segmentation-result-json, or "
            "--use-existing-assets with --fallback-stl and --fallback-ply."
        )

    return {
        "method": method,
        "input_video": str(video_file),
        "output_dir": str(output_path),
        "foot_stl": normalized["foot_stl"],
        "checker_ply": normalized["checker_ply"],
        "raw_result": normalized.get("raw_result"),
    }
