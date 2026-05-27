from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.remove_unreferenced_vertices()

    try:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        mesh.merge_vertices(digits_vertex=6)
    except Exception:
        try:
            mesh.merge_vertices()
        except Exception:
            pass

    try:
        unique = mesh.unique_faces()
        mesh.update_faces(unique)
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        mesh.fix_normals()
    except Exception:
        pass

    return mesh


def keep_largest_component(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    parts = mesh.split(only_watertight=False)
    if len(parts) == 0:
        return mesh
    return max(parts, key=lambda part: len(part.vertices))


def safe_slice(
    mesh: trimesh.Trimesh,
    plane_normal: list[float],
    plane_origin: list[float],
) -> trimesh.Trimesh:
    sliced = trimesh.intersections.slice_mesh_plane(
        mesh,
        plane_normal=plane_normal,
        plane_origin=plane_origin,
        cap=False,
    )
    if sliced is None or sliced.is_empty:
        raise ValueError("절단 후 메시가 비었습니다.")
    return sliced


def boundary_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    edges = mesh.edges_sorted
    edges_unique, counts = np.unique(edges, axis=0, return_counts=True)
    return edges_unique[counts == 1]


def edges_to_loops(edges: np.ndarray) -> list[list[int]]:
    from collections import defaultdict

    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    visited_edges: set[tuple[int, int]] = set()
    loops: list[list[int]] = []

    def edge_key(u: int, v: int) -> tuple[int, int]:
        return tuple(sorted((int(u), int(v))))

    for a, b in edges:
        a = int(a)
        b = int(b)
        edge = edge_key(a, b)
        if edge in visited_edges:
            continue

        loop = [a, b]
        visited_edges.add(edge)

        current = b
        prev = a
        while True:
            candidates = [vertex for vertex in adj[current] if vertex != prev]
            next_v = None
            for candidate in candidates:
                candidate_edge = edge_key(current, candidate)
                if candidate_edge not in visited_edges:
                    next_v = candidate
                    break

            if next_v is None:
                if loop[0] in adj[current]:
                    break
                loop = []
                break

            loop.append(next_v)
            visited_edges.add(edge_key(current, next_v))
            prev, current = current, next_v

            if next_v == loop[0]:
                break

        if loop:
            if loop[-1] == loop[0]:
                loop = loop[:-1]
            if len(loop) >= 3:
                loops.append(loop)

    return loops


def polygon_area_2d(points2d: np.ndarray) -> float:
    x = points2d[:, 0]
    y = points2d[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def triangulate_loop_xy(loop_points_3d: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    import mapbox_earcut as earcut

    pts2d = loop_points_3d[:, :2]
    cleaned = [pts2d[0]]
    for point in pts2d[1:]:
        if np.linalg.norm(point - cleaned[-1]) > 1e-8:
            cleaned.append(point)

    pts2d = np.asarray(cleaned, dtype=np.float64)
    if len(pts2d) >= 2 and np.linalg.norm(pts2d[0] - pts2d[-1]) < 1e-8:
        pts2d = pts2d[:-1]

    if len(pts2d) < 3:
        return None, None

    ring_end_indices = np.array([len(pts2d)], dtype=np.uint32)
    try:
        tri_idx = earcut.triangulate_float64(pts2d, ring_end_indices)
    except Exception as exc:
        print("earcut triangulation 실패:", exc)
        return None, None

    tri_idx = np.asarray(tri_idx, dtype=np.int64)
    if len(tri_idx) == 0:
        return None, None

    faces = tri_idx.reshape(-1, 3)
    z_val = float(np.mean(loop_points_3d[:, 2]))
    verts3d = np.column_stack(
        [
            pts2d[:, 0],
            pts2d[:, 1],
            np.full(len(pts2d), z_val, dtype=np.float64),
        ]
    )
    return verts3d, faces


def add_cap_for_plane(
    mesh: trimesh.Trimesh,
    target_z: float,
    z_tol: float = 1e-4,
    area_min_ratio: float = 0.01,
) -> trimesh.Trimesh:
    b_edges = boundary_edges(mesh)
    loops = edges_to_loops(b_edges)

    if len(loops) == 0:
        print(f"[z={target_z}] boundary loop를 찾지 못했습니다.")
        return mesh

    verts = mesh.vertices
    loop_infos = []

    for loop in loops:
        pts = verts[np.array(loop)]
        mean_z = float(np.mean(pts[:, 2]))
        z_span = float(np.max(pts[:, 2]) - np.min(pts[:, 2]))
        area_xy = abs(polygon_area_2d(pts[:, :2]))
        loop_infos.append(
            {
                "loop": loop,
                "mean_z": mean_z,
                "z_span": z_span,
                "area_xy": area_xy,
            }
        )

    candidates = [info for info in loop_infos if abs(info["mean_z"] - target_z) <= z_tol]
    if len(candidates) == 0:
        print(f"[z={target_z}] target_z 근처 루프가 없습니다. z_tol을 늘려보세요.")
        for index, info in enumerate(sorted(loop_infos, key=lambda data: abs(data["mean_z"] - target_z))[:10]):
            print(
                f"  후보{index}: mean_z={info['mean_z']:.6f}, "
                f"z_span={info['z_span']:.6f}, area={info['area_xy']:.6f}"
            )
        return mesh

    max_area = max(info["area_xy"] for info in candidates)
    candidates = [info for info in candidates if info["area_xy"] >= max_area * area_min_ratio]

    chosen = max(candidates, key=lambda data: data["area_xy"])
    loop = chosen["loop"]
    loop_pts = verts[np.array(loop)]

    print(
        f"[z={target_z}] loop 선택: "
        f"points={len(loop)}, mean_z={chosen['mean_z']:.6f}, "
        f"z_span={chosen['z_span']:.6f}, area={chosen['area_xy']:.6f}"
    )

    cap_verts, cap_faces = triangulate_loop_xy(loop_pts)
    if cap_verts is None or cap_faces is None or len(cap_faces) == 0:
        print(f"[z={target_z}] 삼각분할 실패")
        return mesh

    offset = len(mesh.vertices)
    new_vertices = np.vstack([mesh.vertices.copy(), cap_verts])
    new_faces = np.vstack([mesh.faces.copy(), cap_faces + offset])
    capped = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    return clean_mesh(capped)


def make_watertight_voxel(mesh: trimesh.Trimesh, pitch: float = 0.002) -> trimesh.Trimesh:
    print("Voxel 기반 watertight 생성 중...")
    voxel = mesh.voxelized(pitch)
    voxel = voxel.fill()
    watertight_mesh = voxel.marching_cubes
    watertight_mesh.remove_unreferenced_vertices()
    watertight_mesh.merge_vertices()
    return watertight_mesh


def process_foot_with_manual_caps(
    input_path: str | Path,
    output_path: str | Path,
    z_min: float = 0.0,
    z_max: float = 0.8,
    z_tol: float = 1e-3,
    voxel_pitch: float = 0.002,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    input_file = Path(input_path).expanduser()
    output_file = Path(output_path).expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("1. 메시 로드")
    mesh = trimesh.load(input_file, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError("비어 있는 Scene입니다.")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    if mesh.is_empty:
        raise ValueError("메시가 비어 있습니다.")

    mesh = clean_mesh(mesh)

    print("2. 가장 큰 연결 성분만 유지")
    mesh = keep_largest_component(mesh)
    mesh = clean_mesh(mesh)

    print("3. z_min으로 절단")
    mesh = safe_slice(mesh, [0, 0, 1], [0, 0, z_min])
    mesh = clean_mesh(mesh)

    print("4. z_max로 절단")
    mesh = safe_slice(mesh, [0, 0, -1], [0, 0, z_max])
    mesh = clean_mesh(mesh)

    print("5. 절단 후 가장 큰 연결 성분만 유지")
    mesh = keep_largest_component(mesh)
    mesh = clean_mesh(mesh)

    print("6. 아래 면 cap 추가")
    mesh = add_cap_for_plane(mesh, z_min, z_tol=z_tol)

    print("7. 위 면 cap 추가")
    mesh = add_cap_for_plane(mesh, z_max, z_tol=z_tol)
    mesh = clean_mesh(mesh)

    before_voxel = {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "bounds": mesh.bounds.tolist(),
    }

    print("8. 최종 상태")
    print("vertices:", before_voxel["vertices"])
    print("faces:", before_voxel["faces"])
    print("watertight:", before_voxel["watertight"])
    print("bounds:", before_voxel["bounds"])

    mesh = make_watertight_voxel(mesh, pitch=voxel_pitch)
    mesh.export(output_file)
    print("저장 완료:", output_file)

    execution_time = time.perf_counter() - start_time
    return {
        "input_path": str(input_file),
        "output_path": str(output_file),
        "z_min": z_min,
        "z_max": z_max,
        "z_tol": z_tol,
        "voxel_pitch": voxel_pitch,
        "before_voxel": before_voxel,
        "after_voxel": {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "watertight": bool(mesh.is_watertight),
            "bounds": mesh.bounds.tolist(),
        },
        "execution_time_seconds": execution_time,
    }


if __name__ == "__main__":
    process_foot_with_manual_caps(
        input_path="unbounded_default_post.ply",
        output_path="final_foot_manual_caps_watertight.stl",
        z_min=0.0,
        z_max=0.8,
        z_tol=1e-3,
    )
