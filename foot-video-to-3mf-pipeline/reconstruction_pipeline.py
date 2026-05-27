from __future__ import annotations

import json
import platform
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_2DGS_IMAGE = "2dgs:cu118"
DEFAULT_SCENE_NAME = "foot_scene"


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _require_dir(path: str | Path, label: str) -> Path:
    dir_path = _as_path(path)
    if not dir_path.exists():
        raise FileNotFoundError(f"{label} not found: {dir_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {dir_path}")
    return dir_path


def _require_file(path: str | Path, label: str) -> Path:
    file_path = _as_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{label} not found: {file_path}")
    if not file_path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {file_path}")
    return file_path


def _check_executable(binary: str, label: str) -> str:
    resolved = shutil.which(binary)
    if not resolved:
        raise RuntimeError(
            f"{label} executable not found: {binary}. "
            "Install it or pass the correct path."
        )
    return resolved


def _run_command(
    command: list[str],
    cwd: Path | None = None,
    timeout_seconds: int = 3600,
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    result = {
        "command": command,
        "cwd": str(cwd) if cwd else None,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Return code: {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )
    return result


def _command_help(colmap: str, command_name: str) -> str:
    try:
        completed = subprocess.run(
            [colmap, command_name, "-h"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return ""
    return f"{completed.stdout}\n{completed.stderr}"


def _add_if_supported(
    command: list[str],
    help_text: str,
    option: str,
    value: str | int | float,
) -> None:
    if not help_text or option in help_text:
        command.extend([option, str(value)])


def _is_global_ba_image_count_failure(exc: Exception) -> bool:
    message = str(exc)
    return (
        "ba_config.NumImages() >= 2" in message
        or "At least two images must be registered for global bundle-adjustment" in message
    )


def explain_local_gpu_blockers(docker_bin: str = "docker") -> list[str]:
    blockers: list[str] = []
    if shutil.which(docker_bin) is None:
        blockers.append(f"Docker is not installed or not on PATH: {docker_bin}")

    system = platform.system()
    machine = platform.machine()
    if system == "Darwin":
        blockers.append(
            "This is macOS. The Notion 2DGS Dockerfile uses nvidia/cuda:11.8 "
            "and docker run --gpus all, which needs an NVIDIA CUDA runtime. "
            "Docker Desktop on this Mac cannot expose an NVIDIA CUDA GPU."
        )
    if machine not in {"x86_64", "AMD64"}:
        blockers.append(
            f"Current CPU architecture is {machine}. The Notion Dockerfile is "
            "written for Linux x86_64 CUDA images."
        )
    return blockers


def prepare_colmap_dataset(
    images_dir: str | Path,
    dataset_root: str | Path,
    scene_name: str = DEFAULT_SCENE_NAME,
    overwrite_images: bool = True,
) -> dict[str, Any]:
    """Create dataset/<scene>/images expected by the Notion 2DGS Dockerfile."""
    source_dir = _require_dir(images_dir, "Segmented image directory")
    dataset_root_path = _as_path(dataset_root)
    scene_dir = dataset_root_path / scene_name
    image_target_dir = scene_dir / "images"
    sparse_dir = scene_dir / "sparse"
    image_target_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        [
            *source_dir.glob("*.jpg"),
            *source_dir.glob("*.jpeg"),
            *source_dir.glob("*.png"),
        ]
    )
    if not image_files:
        raise FileNotFoundError(f"No images found in segmented image directory: {source_dir}")

    if overwrite_images:
        for existing in image_target_dir.glob("*"):
            if existing.is_file():
                existing.unlink()

    copied = []
    for image_file in image_files:
        target = image_target_dir / image_file.name
        if overwrite_images or not target.exists():
            shutil.copy2(image_file, target)
        copied.append(str(target))

    return {
        "source_images_dir": str(source_dir),
        "dataset_root": str(dataset_root_path),
        "scene_name": scene_name,
        "scene_dir": str(scene_dir),
        "images_dir": str(image_target_dir),
        "sparse_dir": str(sparse_dir),
        "image_count": len(copied),
    }


def run_colmap_sfm(
    scene_dir: str | Path,
    colmap_bin: str = "colmap",
    matcher: str = "sequential",
    camera_model: str = "SIMPLE_RADIAL",
    single_camera: bool = True,
    clean_existing: bool = True,
    timeout_seconds: int = 7200,
) -> dict[str, Any]:
    """Run COLMAP so 2DGS can read sparse/0/*.bin."""
    colmap = _check_executable(colmap_bin, "COLMAP")
    scene_path = _require_dir(scene_dir, "COLMAP scene directory")
    images_dir = _require_dir(scene_path / "images", "COLMAP images directory")
    database_path = scene_path / "database.db"
    sparse_dir = scene_path / "sparse"

    if clean_existing:
        if database_path.exists():
            database_path.unlink()
        if sparse_dir.exists():
            shutil.rmtree(sparse_dir)

    sparse_dir.mkdir(parents=True, exist_ok=True)

    commands = []
    feature_command = [
        colmap,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(images_dir),
        "--ImageReader.camera_model",
        camera_model,
    ]
    if single_camera:
        feature_command.extend(["--ImageReader.single_camera", "1"])
    feature_help = _command_help(colmap, "feature_extractor")
    _add_if_supported(feature_command, feature_help, "--SiftExtraction.estimate_affine_shape", 1)
    _add_if_supported(feature_command, feature_help, "--SiftExtraction.domain_size_pooling", 1)
    commands.append(_run_command(feature_command, timeout_seconds=timeout_seconds))

    matcher_command = [
        colmap,
        "sequential_matcher" if matcher == "sequential" else "exhaustive_matcher",
        "--database_path",
        str(database_path),
    ]
    matcher_help = _command_help(colmap, matcher_command[1])
    if matcher == "sequential":
        _add_if_supported(matcher_command, matcher_help, "--SequentialMatching.overlap", 15)
    _add_if_supported(matcher_command, matcher_help, "--SiftMatching.guided_matching", 1)
    commands.append(_run_command(matcher_command, timeout_seconds=timeout_seconds))

    mapper_help = _command_help(colmap, "mapper")
    mapper_command = [
        colmap,
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(images_dir),
        "--output_path",
        str(sparse_dir),
    ]
    _add_if_supported(mapper_command, mapper_help, "--Mapper.tri_ignore_two_view_tracks", 0)
    _add_if_supported(mapper_command, mapper_help, "--Mapper.min_num_matches", 8)
    _add_if_supported(mapper_command, mapper_help, "--Mapper.abs_pose_min_num_inliers", 15)
    _add_if_supported(mapper_command, mapper_help, "--Mapper.init_min_num_inliers", 50)

    try:
        commands.append(_run_command(mapper_command, timeout_seconds=timeout_seconds))
    except RuntimeError as exc:
        if not _is_global_ba_image_count_failure(exc):
            raise
        if sparse_dir.exists():
            shutil.rmtree(sparse_dir)
        sparse_dir.mkdir(parents=True, exist_ok=True)

        retry_command = list(mapper_command)
        _add_if_supported(retry_command, mapper_help, "--Mapper.ba_global_max_refinements", 0)
        _add_if_supported(retry_command, mapper_help, "--Mapper.ba_global_images_freq", 1000000)
        _add_if_supported(retry_command, mapper_help, "--Mapper.ba_global_frames_freq", 1000000)
        _add_if_supported(retry_command, mapper_help, "--Mapper.ba_global_points_freq", 100000000)
        _add_if_supported(retry_command, mapper_help, "--Mapper.ba_global_images_ratio", 100.0)
        _add_if_supported(retry_command, mapper_help, "--Mapper.ba_global_frames_ratio", 100.0)
        _add_if_supported(retry_command, mapper_help, "--Mapper.ba_global_points_ratio", 100.0)
        retry_result = _run_command(retry_command, timeout_seconds=timeout_seconds)
        retry_result["retry_reason"] = "COLMAP global bundle-adjustment image-count failure"
        commands.append(retry_result)

    sparse0 = scene_path / "sparse" / "0"
    for required in ("cameras.bin", "images.bin", "points3D.bin"):
        _require_file(sparse0 / required, f"COLMAP {required}")

    from colmap_alignment import align_colmap_sparse_model

    alignment_report = align_colmap_sparse_model(
        sparse0,
        restore_existing_backup=False,
        create_backup=True,
        clip_z_threshold=0.0,
    )

    return {
        "scene_dir": str(scene_path),
        "database_path": str(database_path),
        "sparse0": str(sparse0),
        "matcher": matcher,
        "alignment": alignment_report,
        "commands": commands,
    }


def build_2dgs_docker_image(
    dockerfile_path: str | Path,
    image_name: str = DEFAULT_2DGS_IMAGE,
    docker_bin: str = "docker",
    timeout_seconds: int = 7200,
) -> dict[str, Any]:
    docker = _check_executable(docker_bin, "Docker")
    dockerfile = _require_file(dockerfile_path, "2DGS Dockerfile")
    return _run_command(
        [
            docker,
            "build",
            "-t",
            image_name,
            "-f",
            str(dockerfile),
            str(dockerfile.parent),
        ],
        timeout_seconds=timeout_seconds,
    )


def run_2dgs_docker(
    dataset_root: str | Path,
    output_root: str | Path,
    scene_name: str = DEFAULT_SCENE_NAME,
    image_name: str = DEFAULT_2DGS_IMAGE,
    docker_bin: str = "docker",
    train_args: str = "--depth_ratio 0",
    extract_command: str = "extract_mesh_quick.sh",
    timeout_seconds: int = 14400,
) -> dict[str, Any]:
    """Run the Notion 2DGS Docker train + quick mesh extraction stage."""
    docker = _check_executable(docker_bin, "Docker")
    dataset_root_path = _require_dir(dataset_root, "2DGS dataset root")
    output_root_path = _as_path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    command_inside_container = (
        f"export SCENE={scene_name} && "
        f"train_2dgs.sh {train_args} && "
        f"{extract_command}"
    )
    command = [
        docker,
        "run",
        "--gpus",
        "all",
        "--rm",
        "-v",
        f"{dataset_root_path.resolve()}:/app/dataset",
        "-v",
        f"{output_root_path.resolve()}:/app/output",
        image_name,
        "bash",
        "-lc",
        command_inside_container,
    ]
    run_result = _run_command(command, timeout_seconds=timeout_seconds)

    output_ply = output_root_path / scene_name / "mesh_quick" / "unbounded_default_post.ply"
    _require_file(output_ply, "2DGS unbounded_default_post.ply")
    return {
        "scene_name": scene_name,
        "dataset_root": str(dataset_root_path),
        "output_root": str(output_root_path),
        "reconstruction_ply": str(output_ply),
        "command": command,
        "run": run_result,
    }


def run_reconstruction_from_segmented_images(
    images_dir: str | Path,
    dataset_root: str | Path = "output/2dgs_dataset",
    output_root: str | Path = "output/2dgs_output",
    scene_name: str = DEFAULT_SCENE_NAME,
    colmap_bin: str = "colmap",
    docker_bin: str = "docker",
    docker_image: str = DEFAULT_2DGS_IMAGE,
    dockerfile_path: str | Path | None = None,
    build_image: bool = False,
    skip_colmap: bool = False,
    matcher: str = "sequential",
    train_args: str = "--depth_ratio 0",
    timeout_seconds: int = 14400,
) -> dict[str, Any]:
    """Segmented images -> COLMAP sparse dataset -> 2DGS mesh PLY."""
    local_blockers = explain_local_gpu_blockers(docker_bin)
    report: dict[str, Any] = {
        "images_dir": str(images_dir),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "scene_name": scene_name,
        "status": "pending",
        "local_blockers": local_blockers,
    }

    if local_blockers:
        raise RuntimeError(
            "This machine cannot run the COLMAP + 2DGS Docker reconstruction stage yet.\n"
            + "\n".join(f"- {blocker}" for blocker in local_blockers)
        )

    dataset_report = prepare_colmap_dataset(images_dir, dataset_root, scene_name)
    report["dataset"] = dataset_report

    if build_image:
        if dockerfile_path is None:
            raise RuntimeError("--build-2dgs-image requires --2dgs-dockerfile.")
        report["docker_build"] = build_2dgs_docker_image(
            dockerfile_path,
            image_name=docker_image,
            docker_bin=docker_bin,
            timeout_seconds=timeout_seconds,
        )

    if not skip_colmap:
        report["colmap"] = run_colmap_sfm(
            dataset_report["scene_dir"],
            colmap_bin=colmap_bin,
            matcher=matcher,
            timeout_seconds=timeout_seconds,
        )
    else:
        sparse0 = Path(dataset_report["scene_dir"]) / "sparse" / "0"
        for required in ("cameras.bin", "images.bin", "points3D.bin"):
            _require_file(sparse0 / required, f"Existing COLMAP {required}")
        report["colmap"] = {"skipped": True, "sparse0": str(sparse0)}

    report["two_dgs"] = run_2dgs_docker(
        dataset_root=dataset_root,
        output_root=output_root,
        scene_name=scene_name,
        image_name=docker_image,
        docker_bin=docker_bin,
        train_args=train_args,
        timeout_seconds=timeout_seconds,
    )
    report["reconstruction_ply"] = report["two_dgs"]["reconstruction_ply"]
    report["status"] = "success"
    return report


def write_reconstruction_report(report: dict[str, Any], reports_dir: str | Path = "reports") -> Path:
    reports_path = _as_path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports_path / f"{report.get('scene_name', DEFAULT_SCENE_NAME)}_reconstruction_{timestamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
