from __future__ import annotations

import argparse
import collections
import shutil
import struct
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R_scipy


CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    return R_scipy.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()


def rotmat2qvec(rotmat: np.ndarray) -> np.ndarray:
    quat = R_scipy.from_matrix(rotmat).as_quat()
    return np.array([quat[3], quat[0], quat[1], quat[2]])


def read_next_bytes(fid, num_bytes: int, format_char_sequence: str, endian_character: str = "<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_images_binary(path_to_model_file: str | Path) -> dict[int, Image]:
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            xys = read_next_bytes(fid, 16 * num_points2D, "dd" * num_points2D)
            xys = np.array(xys).reshape((num_points2D, 2))
            point3D_ids = read_next_bytes(fid, 8 * num_points2D, "q" * num_points2D)
            point3D_ids = np.array(point3D_ids, dtype=np.int64)
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def read_points3D_binary(path_to_model_file: str | Path) -> dict[int, Point3D]:
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_properties = read_next_bytes(fid, 43, "QdddBBBd")
            point3D_id = binary_point_properties[0]
            xyz = np.array(binary_point_properties[1:4], dtype=np.float64)
            rgb = np.array(binary_point_properties[4:7], dtype=np.uint8)
            error = float(binary_point_properties[7])
            track_length = read_next_bytes(fid, 8, "Q")[0]
            track_elems = read_next_bytes(fid, 8 * track_length, "ii" * track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])), dtype=np.int32)
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])), dtype=np.int32)
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D


def write_images_binary(images: dict[int, Image], path_to_model_file: str | Path) -> None:
    with open(path_to_model_file, "wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for _, img in images.items():
            image_header = struct.pack("<idddddddi", img.id, *img.qvec, *img.tvec, img.camera_id)
            fid.write(image_header)
            fid.write(img.name.encode("utf-8") + b"\x00")
            num_points = len(img.point3D_ids)
            fid.write(struct.pack("<Q", num_points))
            for i in range(num_points):
                fid.write(struct.pack("<dd", float(img.xys[i][0]), float(img.xys[i][1])))
            for i in range(num_points):
                fid.write(struct.pack("<q", int(img.point3D_ids[i])))


def write_points3D_binary(points3D: dict[int, Point3D], path_to_model_file: str | Path) -> None:
    with open(path_to_model_file, "wb") as fid:
        fid.write(struct.pack("<Q", len(points3D)))
        for _, pt in points3D.items():
            fid.write(
                struct.pack(
                    "<QdddBBBd",
                    int(pt.id),
                    float(pt.xyz[0]),
                    float(pt.xyz[1]),
                    float(pt.xyz[2]),
                    int(pt.rgb[0]),
                    int(pt.rgb[1]),
                    int(pt.rgb[2]),
                    float(pt.error),
                )
            )
            track_len = len(pt.image_ids)
            fid.write(struct.pack("<Q", track_len))
            for i in range(track_len):
                fid.write(struct.pack("<ii", int(pt.image_ids[i]), int(pt.point2D_idxs[i])))


def compute_alignment_matrix(
    points3D_dict: dict[int, Point3D],
    images_dict: dict[int, Image],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    print(">>> Detecting Floor Plane using RANSAC...")
    pts = [p.xyz for p in points3D_dict.values()]
    if not pts:
        print("No 3D points found.")
        return np.eye(3), np.zeros(3), {"status": "skipped", "reason": "no 3D points"}

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))

    bbox = pcd.get_axis_aligned_bounding_box()
    extent = float(np.linalg.norm(bbox.get_extent()))
    threshold = extent * 0.001

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=threshold,
        ransac_n=3,
        num_iterations=5000,
    )
    a, b, c, d = plane_model

    print(f"    Plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
    normal = np.array([a, b, c], dtype=np.float64)
    normal /= np.linalg.norm(normal) + 1e-12
    target = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    rot_axis = np.cross(normal, target)
    dot_val = float(np.dot(normal, target))

    if np.linalg.norm(rot_axis) < 1e-6:
        r_align = np.eye(3) if dot_val > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        rot_axis /= np.linalg.norm(rot_axis)
        rot_angle = np.arccos(np.clip(dot_val, -1.0, 1.0))
        r_align = R_scipy.from_rotvec(rot_axis * rot_angle).as_matrix()

    inlier_cloud = pcd.select_by_index(inliers)
    center = inlier_cloud.get_center()
    rotated_center = r_align @ center
    t_align = np.array([0.0, 0.0, -rotated_center[2]], dtype=np.float64)

    cam_z_sum = 0.0
    cam_count = 0
    for img in images_dict.values():
        r_cam = qvec2rotmat(img.qvec)
        c_world = -r_cam.T @ img.tvec
        c_aligned = r_align @ c_world + t_align
        cam_z_sum += float(c_aligned[2])
        cam_count += 1

    avg_cam_z = cam_z_sum / cam_count if cam_count > 0 else 0.0
    flipped = False
    if avg_cam_z < 0:
        print("    [!] World is Upside Down. Flipping 180 degrees...")
        r_flip = np.diag([1.0, -1.0, -1.0])
        r_align = r_flip @ r_align
        t_align = r_flip @ t_align
        flipped = True

    details = {
        "status": "success",
        "plane_model": [float(a), float(b), float(c), float(d)],
        "extent": extent,
        "ransac_distance_threshold": float(threshold),
        "inlier_count": int(len(inliers)),
        "point_count": int(len(points3D_dict)),
        "camera_count": int(cam_count),
        "average_camera_z_before_flip_check": float(avg_cam_z),
        "flipped": bool(flipped),
    }
    return r_align, t_align, details


def apply_alignment(input_path: str | Path, clip_z_threshold: float = 0.0) -> dict[str, Any]:
    input_dir = Path(input_path).expanduser()
    images_bin = input_dir / "images.bin"
    points_bin = input_dir / "points3D.bin"

    if not images_bin.exists() or not points_bin.exists():
        raise FileNotFoundError(f"COLMAP binaries not found in {input_dir}")

    print(f">>> Loading COLMAP model from {input_dir}...")
    images = read_images_binary(images_bin)
    points3D = read_points3D_binary(points_bin)

    r_align, t_align, alignment_details = compute_alignment_matrix(points3D, images)

    print(">>> Transforming 3D Points & Clipping Negative Z...")
    new_points3D = {}
    valid_point_ids = set()
    removed_count = 0

    for pid, point in points3D.items():
        new_xyz = r_align @ point.xyz + t_align
        if new_xyz[2] >= clip_z_threshold:
            new_points3D[pid] = point._replace(xyz=new_xyz)
            valid_point_ids.add(pid)
        else:
            removed_count += 1

    print(f"    - Removed {removed_count} points below Z={clip_z_threshold}")
    points3D = new_points3D

    print(">>> Transforming Camera Poses...")
    for iid, img in images.items():
        r_old = qvec2rotmat(img.qvec)
        t_old = img.tvec
        r_new = r_old @ r_align.T
        t_new = t_old - r_new @ t_align
        new_qvec = rotmat2qvec(r_new)

        new_point3D_ids = []
        for pid in img.point3D_ids:
            new_point3D_ids.append(int(pid) if int(pid) in valid_point_ids else -1)

        images[iid] = img._replace(
            qvec=new_qvec,
            tvec=t_new,
            point3D_ids=np.array(new_point3D_ids, dtype=np.int64),
        )

    print(">>> Overwriting binary files...")
    write_images_binary(images, images_bin)
    write_points3D_binary(points3D, points_bin)
    print(">>> Alignment & Clipping Complete.")

    return {
        "input_path": str(input_dir),
        "clip_z_threshold": float(clip_z_threshold),
        "original_point_count": int(alignment_details.get("point_count", len(valid_point_ids) + removed_count)),
        "remaining_point_count": int(len(points3D)),
        "removed_point_count": int(removed_count),
        "rotation_matrix": r_align.tolist(),
        "translation": t_align.tolist(),
        **alignment_details,
    }


def align_colmap_sparse_model(
    input_path: str | Path,
    restore_existing_backup: bool = False,
    create_backup: bool = True,
    clip_z_threshold: float = 0.0,
) -> dict[str, Any]:
    target_dir = Path(input_path).expanduser()
    if not target_dir.exists():
        raise FileNotFoundError(f"COLMAP sparse folder not found: {target_dir}")

    backup_dir = Path(str(target_dir).rstrip("/\\")).with_name(target_dir.name + "_backup")
    if restore_existing_backup and backup_dir.exists():
        print(">>> Restoring from backup for fresh start...")
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(backup_dir, target_dir)
    elif create_backup and not backup_dir.exists():
        print(">>> Creating backup...")
        shutil.copytree(target_dir, backup_dir)

    report = apply_alignment(target_dir, clip_z_threshold=clip_z_threshold)
    report["backup_dir"] = str(backup_dir) if backup_dir.exists() else None
    report["restored_existing_backup"] = bool(restore_existing_backup and backup_dir.exists())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto align COLMAP model to floor plane")
    parser.add_argument("--input_path", type=str, required=True, help="Path to COLMAP sparse folder containing .bin files")
    parser.add_argument("--clip-z-threshold", type=float, default=0.0)
    parser.add_argument("--restore-existing-backup", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    align_colmap_sparse_model(
        args.input_path,
        restore_existing_backup=args.restore_existing_backup,
        create_backup=not args.no_backup,
        clip_z_threshold=args.clip_z_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
