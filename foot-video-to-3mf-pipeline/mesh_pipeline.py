from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


# Simplify defaults.
# If a mesh has more than DEFAULT_SIMPLIFY_THRESHOLD triangles, Open3D decimates
# it down toward DEFAULT_TARGET_TRIANGLES triangles.
#
# Example:
# - 250,000 triangles -> no simplify, because it is below 300,000
# - 500,000 triangles -> simplify toward 180,000
#
# Raise these values for maximum detail. Lower them for faster slicing and
# smaller files. Avoid setting target higher than threshold.
DEFAULT_SIMPLIFY_THRESHOLD = 300_000
DEFAULT_TARGET_TRIANGLES = 180_000

# Floating-region defaults.
# These values are intentionally conservative. The pipeline removes only very
# small disconnected floating fragments, then asks the slicer to enable supports
# when larger floating/overhang regions remain.
#
# Units follow the STL units. In most Bambu/Orca workflows this means mm.
DEFAULT_FLOATING_BED_TOLERANCE = 0.05
DEFAULT_FLOATING_REMOVE_TRIANGLE_RATIO = 0.001
DEFAULT_FLOATING_MIN_REMOVE_TRIANGLES = 30
DEFAULT_OVERHANG_NORMAL_Z = -0.5
DEFAULT_OVERHANG_AREA_RATIO_FOR_SUPPORT = 0.005


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _require_file(path: str | Path) -> Path:
    file_path = _as_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"STL file not found: {file_path}")
    if not file_path.is_file():
        raise FileNotFoundError(f"Path is not a file: {file_path}")
    return file_path


def _ensure_parent(path: str | Path) -> Path:
    file_path = _as_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def _safe_error(stage: str, exc: Exception) -> dict[str, str]:
    return {"stage": stage, "error": f"{type(exc).__name__}: {exc}"}


def _valid_report(report: Any) -> bool:
    return isinstance(report, dict) and "error" not in report


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load an STL file with trimesh and collapse scenes into one mesh."""
    file_path = _require_file(path)

    try:
        loaded = trimesh.load(file_path, process=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load STL with trimesh: {file_path}. Reason: {exc}") from exc

    if isinstance(loaded, trimesh.Scene):
        geometries = [
            geom
            for geom in loaded.geometry.values()
            if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0
        ]
        if not geometries:
            raise ValueError(f"No mesh geometry found in STL scene: {file_path}")
        mesh = trimesh.util.concatenate(geometries)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported mesh object loaded from {file_path}: {type(loaded)!r}")

    if mesh.vertices is None or mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Loaded STL has no triangles: {file_path}")

    mesh = mesh.copy()
    try:
        mesh.merge_vertices()
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    return mesh


def count_edge_problems(mesh: trimesh.Trimesh) -> tuple[int, int]:
    """Return boundary edge count and non-manifold edge count."""
    faces = np.asarray(mesh.faces)
    if faces.size == 0:
        return 0, 0

    edges = np.vstack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    )
    edges = np.sort(edges, axis=1)
    edge_counts = Counter(map(tuple, edges))

    boundary_edges = sum(1 for count in edge_counts.values() if count == 1)
    non_manifold_edges = sum(1 for count in edge_counts.values() if count >= 3)
    return int(boundary_edges), int(non_manifold_edges)


def inspect_stl(path: str | Path) -> dict[str, Any]:
    mesh = load_mesh(path)
    boundary_edges, non_manifold_edges = count_edge_problems(mesh)

    volume_available = False
    volume = None
    try:
        raw_volume = float(mesh.volume)
        if mesh.is_watertight and np.isfinite(raw_volume):
            volume_available = True
            volume = raw_volume
    except Exception:
        volume_available = False
        volume = None

    return {
        "triangle_count": int(len(mesh.faces)),
        "vertex_count": int(len(mesh.vertices)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "boundary_edges": boundary_edges,
        "non_manifold_edges": non_manifold_edges,
        "volume_available": volume_available,
        "volume": volume,
    }


def is_printable_enough(report: dict[str, Any]) -> bool:
    return (
        _valid_report(report)
        and int(report.get("triangle_count", 0)) > 0
        and int(report.get("boundary_edges", -1)) == 0
        and int(report.get("non_manifold_edges", -1)) == 0
        and bool(report.get("watertight", False))
    )


def inspect_floating_regions(
    path: str | Path,
    bed_tolerance: float = DEFAULT_FLOATING_BED_TOLERANCE,
    overhang_normal_z: float = DEFAULT_OVERHANG_NORMAL_Z,
    overhang_area_ratio_for_support: float = DEFAULT_OVERHANG_AREA_RATIO_FOR_SUPPORT,
) -> dict[str, Any]:
    """Inspect disconnected floating shells and downward overhang regions.

    Bambu Studio's "floating regions" warning can mean either disconnected
    islands that are above the build plate, or printable overhang islands that
    need support. This inspection reports both. It does not delete anything.
    """
    mesh = load_mesh(path)
    total_triangles = int(len(mesh.faces))
    total_area = float(mesh.area) if np.isfinite(mesh.area) else 0.0
    bed_z = float(mesh.bounds[0][2])

    components = list(mesh.split(only_watertight=False))
    component_infos: list[dict[str, Any]] = []
    floating_components: list[dict[str, Any]] = []

    for index, component in enumerate(components):
        if len(component.faces) == 0:
            continue
        bounds = component.bounds
        min_z = float(bounds[0][2])
        max_z = float(bounds[1][2])
        triangle_count = int(len(component.faces))
        area = float(component.area) if np.isfinite(component.area) else 0.0
        volume = None
        try:
            raw_volume = float(component.volume)
            if component.is_watertight and np.isfinite(raw_volume):
                volume = raw_volume
        except Exception:
            volume = None

        is_floating = min_z > bed_z + float(bed_tolerance)
        info = {
            "index": index,
            "triangle_count": triangle_count,
            "area": area,
            "volume": volume,
            "min_z": min_z,
            "max_z": max_z,
            "is_floating": bool(is_floating),
        }
        component_infos.append(info)
        if is_floating:
            floating_components.append(info)

    face_normals = np.asarray(mesh.face_normals)
    face_vertices = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
    face_min_z = face_vertices[:, :, 2].min(axis=1)
    downward_mask = face_normals[:, 2] < float(overhang_normal_z)
    above_bed_mask = face_min_z > bed_z + float(bed_tolerance)
    unsupported_downward_mask = downward_mask & above_bed_mask
    area_faces = np.asarray(mesh.area_faces)
    unsupported_downward_area = float(area_faces[unsupported_downward_mask].sum())
    unsupported_downward_face_count = int(unsupported_downward_mask.sum())
    unsupported_area_ratio = (
        unsupported_downward_area / total_area
        if total_area > 0
        else 0.0
    )

    support_recommended = bool(
        floating_components
        or unsupported_area_ratio >= float(overhang_area_ratio_for_support)
    )

    return {
        "component_count": int(len(component_infos)),
        "floating_component_count": int(len(floating_components)),
        "floating_components": floating_components[:20],
        "bed_z": bed_z,
        "bed_tolerance": float(bed_tolerance),
        "total_triangles": total_triangles,
        "unsupported_downward_face_count": unsupported_downward_face_count,
        "unsupported_downward_area": unsupported_downward_area,
        "unsupported_downward_area_ratio": unsupported_area_ratio,
        "overhang_normal_z": float(overhang_normal_z),
        "support_recommended": support_recommended,
    }


def resolve_floating_regions(
    input_path: str | Path,
    output_path: str | Path,
    bed_tolerance: float = DEFAULT_FLOATING_BED_TOLERANCE,
    remove_triangle_ratio: float = DEFAULT_FLOATING_REMOVE_TRIANGLE_RATIO,
    min_remove_triangles: int = DEFAULT_FLOATING_MIN_REMOVE_TRIANGLES,
    overhang_normal_z: float = DEFAULT_OVERHANG_NORMAL_Z,
    overhang_area_ratio_for_support: float = DEFAULT_OVERHANG_AREA_RATIO_FOR_SUPPORT,
) -> dict[str, Any]:
    """Remove tiny floating fragments and report whether slicer supports are needed."""
    input_file = _require_file(input_path)
    output_file = _ensure_parent(output_path)
    before = inspect_floating_regions(
        input_file,
        bed_tolerance=bed_tolerance,
        overhang_normal_z=overhang_normal_z,
        overhang_area_ratio_for_support=overhang_area_ratio_for_support,
    )

    mesh = load_mesh(input_file)
    components = list(mesh.split(only_watertight=False))
    total_triangles = int(len(mesh.faces))
    bed_z = float(mesh.bounds[0][2])
    remove_limit = max(
        int(min_remove_triangles),
        int(total_triangles * float(remove_triangle_ratio)),
    )

    kept_components: list[trimesh.Trimesh] = []
    removed_components: list[dict[str, Any]] = []

    for index, component in enumerate(components):
        if len(component.faces) == 0:
            continue
        min_z = float(component.bounds[0][2])
        triangle_count = int(len(component.faces))
        is_floating = min_z > bed_z + float(bed_tolerance)
        should_remove = bool(is_floating and triangle_count <= remove_limit)

        if should_remove:
            removed_components.append(
                {
                    "index": index,
                    "triangle_count": triangle_count,
                    "min_z": min_z,
                    "max_z": float(component.bounds[1][2]),
                    "reason": "small disconnected floating component",
                }
            )
            continue
        kept_components.append(component)

    if removed_components:
        if not kept_components:
            raise RuntimeError("Floating-region cleanup would remove every component.")
        cleaned = trimesh.util.concatenate(kept_components)
        cleaned.export(output_file)
        action = "removed_small_floating_components"
    else:
        shutil.copy2(input_file, output_file)
        action = "none"

    after = inspect_floating_regions(
        output_file,
        bed_tolerance=bed_tolerance,
        overhang_normal_z=overhang_normal_z,
        overhang_area_ratio_for_support=overhang_area_ratio_for_support,
    )

    return {
        "output_file": str(output_file),
        "action": action,
        "remove_limit_triangles": remove_limit,
        "removed_component_count": int(len(removed_components)),
        "removed_components": removed_components[:20],
        "before": before,
        "after": after,
        "support_recommended": bool(after.get("support_recommended", False)),
    }


def repair_with_pymeshlab(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    input_file = _require_file(input_path)
    output_file = _ensure_parent(output_path)
    try:
        pre_repair_report = inspect_stl(input_file)
        should_close_holes = int(pre_repair_report.get("boundary_edges", 0)) > 0
    except Exception:
        should_close_holes = True

    try:
        import pymeshlab
    except Exception as exc:
        raise RuntimeError(
            "PyMeshLab is not installed or could not be imported. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    mesh_set = pymeshlab.MeshSet()
    try:
        mesh_set.load_new_mesh(str(input_file))
    except Exception as exc:
        raise RuntimeError(f"PyMeshLab failed to load STL: {input_file}. Reason: {exc}") from exc

    applied: list[str] = []
    skipped: list[dict[str, str]] = []

    def safe_apply(label: str, filter_names: list[str], **kwargs: Any) -> None:
        last_error = None
        for filter_name in filter_names:
            attempts = [kwargs] if kwargs else [{}]
            if kwargs:
                attempts.append({})
            for params in attempts:
                try:
                    mesh_set.apply_filter(filter_name, **params)
                    applied.append(filter_name)
                    return
                except Exception as exc:
                    last_error = exc
        skipped.append(
            {
                "step": label,
                "tried": ", ".join(filter_names),
                "reason": str(last_error) if last_error else "unknown error",
            }
        )

    safe_apply(
        "remove duplicate faces",
        ["meshing_remove_duplicate_faces", "remove_duplicate_faces"],
    )
    safe_apply(
        "remove duplicate vertices",
        ["meshing_remove_duplicate_vertices", "remove_duplicate_vertices"],
    )
    safe_apply(
        "remove unreferenced vertices",
        ["meshing_remove_unreferenced_vertices", "remove_unreferenced_vertices"],
    )
    safe_apply(
        "remove null or degenerate faces",
        ["meshing_remove_null_faces", "remove_null_faces"],
    )
    safe_apply(
        "repair non-manifold edges",
        ["meshing_repair_non_manifold_edges", "repair_non_manifold_edges"],
    )
    safe_apply(
        "repair non-manifold vertices",
        ["meshing_repair_non_manifold_vertices", "repair_non_manifold_vertices"],
    )
    if should_close_holes:
        safe_apply(
            "close holes",
            ["meshing_close_holes", "close_holes"],
            maxholesize=100,
        )
    else:
        skipped.append(
            {
                "step": "close holes",
                "tried": "meshing_close_holes, close_holes",
                "reason": "skipped because no boundary edges were detected before repair",
            }
        )
    safe_apply(
        "re-orient faces coherently",
        ["meshing_re_orient_faces_coherently", "re_orient_faces_coherently"],
    )
    safe_apply(
        "remove unreferenced vertices after repair",
        ["meshing_remove_unreferenced_vertices", "remove_unreferenced_vertices"],
    )

    try:
        mesh_set.save_current_mesh(str(output_file))
    except Exception as exc:
        raise RuntimeError(f"PyMeshLab failed to save repaired STL: {output_file}. Reason: {exc}") from exc

    return {
        "output_file": str(output_file),
        "applied_filters": applied,
        "skipped_filters": skipped,
    }


def repair_with_meshfix(input_path: str | Path, output_path: str | Path) -> dict[str, str]:
    input_file = _require_file(input_path)
    output_file = _ensure_parent(output_path)

    try:
        import pymeshfix
    except Exception as exc:
        raise RuntimeError(
            "pymeshfix is not installed or could not be imported. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    try:
        pymeshfix.clean_from_file(str(input_file), str(output_file))
    except Exception as exc:
        raise RuntimeError(f"MeshFix failed to repair STL: {input_file}. Reason: {exc}") from exc

    return {"output_file": str(output_file)}


def simplify_with_open3d(
    input_path: str | Path,
    output_path: str | Path,
    target_triangles: int,
) -> dict[str, Any]:
    input_file = _require_file(input_path)
    output_file = _ensure_parent(output_path)

    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError(
            "Open3D is not installed or could not be imported. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    mesh = o3d.io.read_triangle_mesh(str(input_file))
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise ValueError(f"Open3D loaded an empty mesh: {input_file}")

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.remove_non_manifold_edges()

    before_triangles = int(len(mesh.triangles))
    after_triangles = before_triangles

    # Decimation only happens when the current triangle count is above the
    # target. The caller decides whether this function should run by comparing
    # the mesh against DEFAULT_SIMPLIFY_THRESHOLD.
    if before_triangles > int(target_triangles):
        mesh = mesh.simplify_quadric_decimation(int(target_triangles))
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        mesh.remove_non_manifold_edges()
        after_triangles = int(len(mesh.triangles))

    mesh.compute_vertex_normals()
    ok = o3d.io.write_triangle_mesh(str(output_file), mesh, write_ascii=False)
    if not ok:
        raise RuntimeError(f"Open3D failed to write simplified STL: {output_file}")

    return {
        "output_file": str(output_file),
        "before_triangles": before_triangles,
        "after_triangles": after_triangles,
        "target_triangles": int(target_triangles),
    }


def process_stl(
    input_path: str | Path,
    output_dir: str | Path,
    simplify_threshold: int = DEFAULT_SIMPLIFY_THRESHOLD,
    target_triangles: int = DEFAULT_TARGET_TRIANGLES,
    floating_bed_tolerance: float = DEFAULT_FLOATING_BED_TOLERANCE,
    floating_remove_triangle_ratio: float = DEFAULT_FLOATING_REMOVE_TRIANGLE_RATIO,
    floating_min_remove_triangles: int = DEFAULT_FLOATING_MIN_REMOVE_TRIANGLES,
) -> dict[str, Any]:
    input_file = _require_file(input_path)
    output_path = _as_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stem = input_file.stem
    pymeshlab_file = output_path / f"{stem}_pymeshlab_repaired.stl"
    meshfix_file = output_path / f"{stem}_meshfix_repaired.stl"
    simplified_file = output_path / f"{stem}_simplified.stl"
    floating_fixed_file = output_path / f"{stem}_floating_fixed.stl"
    final_file = output_path / f"{stem}_processed.stl"

    before = inspect_stl(input_file)

    current_file = input_file
    repair_used = "none"
    after_pymeshlab: dict[str, Any] | None = None
    after_meshfix: dict[str, Any] | None = None

    if not is_printable_enough(before):
        try:
            repair_with_pymeshlab(input_file, pymeshlab_file)
            after_pymeshlab = inspect_stl(pymeshlab_file)
            current_file = pymeshlab_file
            repair_used = "pymeshlab"
        except Exception as exc:
            after_pymeshlab = _safe_error("pymeshlab", exc)

        needs_meshfix = not (_valid_report(after_pymeshlab) and is_printable_enough(after_pymeshlab))
        if needs_meshfix:
            meshfix_input = current_file if current_file.exists() else input_file
            try:
                repair_with_meshfix(meshfix_input, meshfix_file)
                after_meshfix = inspect_stl(meshfix_file)
                current_file = meshfix_file
                repair_used = "pymeshfix" if repair_used == "none" else "pymeshlab+pymeshfix"
            except Exception as exc:
                after_meshfix = _safe_error("meshfix", exc)
                if current_file == input_file:
                    raise RuntimeError(
                        "Both PyMeshLab repair and MeshFix fallback failed. "
                        f"PyMeshLab error: {after_pymeshlab.get('error')}; "
                        f"MeshFix error: {after_meshfix.get('error')}"
                    ) from exc

    current_report = inspect_stl(current_file)
    simplified = False
    if int(current_report["triangle_count"]) > int(simplify_threshold):
        simplify_with_open3d(current_file, simplified_file, int(target_triangles))
        current_file = simplified_file
        simplified = True

    try:
        floating_regions = resolve_floating_regions(
            current_file,
            floating_fixed_file,
            bed_tolerance=float(floating_bed_tolerance),
            remove_triangle_ratio=float(floating_remove_triangle_ratio),
            min_remove_triangles=int(floating_min_remove_triangles),
        )
        current_file = floating_fixed_file
    except Exception as exc:
        floating_regions = _safe_error("floating_regions", exc)

    if current_file.resolve() != final_file.resolve():
        shutil.copy2(current_file, final_file)

    final = inspect_stl(final_file)

    return {
        "input_file": str(input_file),
        "final_file": str(final_file),
        "repair_used": repair_used,
        "simplified": simplified,
        "before": before,
        "after_pymeshlab": after_pymeshlab,
        "after_meshfix": after_meshfix,
        "floating_regions": floating_regions,
        "support_recommended": bool(
            _valid_report(floating_regions)
            and floating_regions.get("support_recommended", False)
        ),
        "final": final,
        "printable_enough": is_printable_enough(final),
    }
