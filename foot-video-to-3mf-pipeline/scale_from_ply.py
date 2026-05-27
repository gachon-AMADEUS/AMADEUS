from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


DEFAULT_PLY_INPUT = Path("input/foot_for_scale_2.ply")
DEFAULT_SCALE_OUTPUT_DIR = Path("output/scale_debug")
DEFAULT_RESOLUTION = 1000
DEFAULT_SQUARE_REAL_SIZE_MM = 30.0


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _require_file(path: str | Path, label: str) -> Path:
    file_path = _as_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{label} not found: {file_path}")
    if not file_path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {file_path}")
    return file_path


def _require_cv2():
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(
            "OpenCV(cv2) is required for checkerboard scale detection. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return cv2


def refine_normal_with_pca(inlier_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a stable plane normal and centroid from point cloud coordinates."""
    centroid = np.mean(inlier_points, axis=0)
    centered_points = inlier_points - centroid
    cov_matrix = np.cov(centered_points, rowvar=False)
    _eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    pca_normal = eigenvectors[:, 0]
    if pca_normal[2] < 0:
        pca_normal = -pca_normal
    return pca_normal, centroid


def get_rotation_matrix(normal_vector: np.ndarray) -> np.ndarray:
    """Return a matrix that rotates the plane normal onto the +Z axis."""
    source_vec = np.asarray(normal_vector, dtype=float)
    norm = np.linalg.norm(source_vec)
    if norm <= 1e-12:
        raise ValueError("Normal vector is too small to build a rotation matrix.")

    source_vec = source_vec / norm
    target_vec = np.array([0.0, 0.0, 1.0])
    cross_vec = np.cross(source_vec, target_vec)
    cos_angle = float(np.dot(source_vec, target_vec))
    sin_angle = float(np.linalg.norm(cross_vec))

    if sin_angle < 1e-6:
        return np.eye(3)

    skew_symmetric = np.array(
        [
            [0, -cross_vec[2], cross_vec[1]],
            [cross_vec[2], 0, -cross_vec[0]],
            [-cross_vec[1], cross_vec[0], 0],
        ]
    )
    return (
        np.eye(3)
        + skew_symmetric
        + np.dot(skew_symmetric, skew_symmetric) * ((1 - cos_angle) / (sin_angle**2))
    )


def project_to_2d(pcd: o3d.geometry.PointCloud, resolution: int) -> np.ndarray:
    """Orthographically project an aligned point cloud to a checkerboard PNG image."""
    cv2 = _require_cv2()
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    if len(points) == 0:
        raise ValueError("Point cloud is empty.")
    if len(colors) != len(points):
        colors = np.ones((len(points), 3), dtype=float)

    x_min, y_min = np.min(points[:, 0]), np.min(points[:, 1])
    x_max, y_max = np.max(points[:, 0]), np.max(points[:, 1])
    width = int((x_max - x_min) * resolution) + 1
    height = int((y_max - y_min) * resolution) + 1

    if width <= 1 or height <= 1:
        raise ValueError(f"Projected image is too small: {width}x{height}")

    proj_image = np.zeros((height, width, 3), dtype=np.uint8)
    pixel_x = ((points[:, 0] - x_min) * resolution).astype(int)
    pixel_y = ((points[:, 1] - y_min) * resolution).astype(int)
    pixel_x = np.clip(pixel_x, 0, width - 1)
    pixel_y = np.clip(pixel_y, 0, height - 1)

    colors_cv = np.clip(colors * 255, 0, 255).astype(np.uint8)
    proj_image[pixel_y, pixel_x] = colors_cv[:, [2, 1, 0]]

    kernel = np.ones((5, 5), np.uint8)
    proj_image = cv2.dilate(proj_image, kernel, iterations=2)
    return cv2.GaussianBlur(proj_image, (3, 3), 0)


def _process_axis_intervals(
    rho_values: list[float],
    merge_threshold: float = 15,
    min_interval: float = 150,
    max_interval: float = 350,
) -> tuple[float, float, int]:
    """Merge near-duplicate Hough lines and return the median checker interval."""
    if len(rho_values) < 2:
        return 0, float("inf"), 0

    sorted_rhos = np.sort(rho_values)
    merged_rhos = []
    current_group = [sorted_rhos[0]]

    for rho in sorted_rhos[1:]:
        if rho - current_group[-1] <= merge_threshold:
            current_group.append(rho)
        else:
            merged_rhos.append(float(np.mean(current_group)))
            current_group = [rho]
    merged_rhos.append(float(np.mean(current_group)))

    if len(merged_rhos) < 2:
        return 0, float("inf"), 0

    intervals = np.diff(merged_rhos)
    base_intervals = intervals[(intervals >= min_interval) & (intervals <= max_interval)]
    if len(base_intervals) < 2:
        return 0, float("inf"), 0

    q1 = np.percentile(base_intervals, 25)
    q3 = np.percentile(base_intervals, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    valid_intervals = base_intervals[
        (base_intervals >= lower_bound) & (base_intervals <= upper_bound)
    ]
    if len(valid_intervals) == 0:
        return 0, float("inf"), 0

    final_interval = float(np.median(valid_intervals))
    cv = float(np.std(valid_intervals) / final_interval) if final_interval > 0 else float("inf")
    return final_interval, cv, int(len(valid_intervals))


def calculate_pixel_distance(
    image_path: str | Path,
    debug_output_path: str | Path | None = None,
) -> tuple[float | None, dict[str, Any]]:
    """Detect checkerboard grid line spacing from a projected image."""
    cv2 = _require_cv2()
    image_file = _require_file(image_path, "Projected checkerboard image")
    img = cv2.imread(str(image_file))
    if img is None:
        raise RuntimeError(f"OpenCV failed to read image: {image_file}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    debug_img = img.copy()
    blurred = cv2.GaussianBlur(gray, (9, 9), 1.5)
    edges = cv2.Canny(blurred, 80, 120)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=150)

    if lines is None:
        return None, {"line_count": 0, "reason": "no Hough lines detected"}

    angles_deg = np.rad2deg(lines[:, 0, 1]) % 90
    valid_angles = angles_deg[(angles_deg > 1.0) & (angles_deg < 89.0)]
    if len(valid_angles) == 0:
        return None, {"line_count": int(len(lines)), "reason": "no valid line angles"}

    hist, bin_edges = np.histogram(valid_angles, bins=86, range=(2, 88))
    dominant_angle = float(bin_edges[np.argmax(hist)])
    rho_groups: list[list[float]] = [[], []]
    strict_tolerance = 1.0

    for line in lines:
        rho, theta = line[0]
        angle = float(np.rad2deg(theta))

        if abs((angle % 90) - dominant_angle) <= strict_tolerance:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 10000 * (-b)), int(y0 + 10000 * a))
            pt2 = (int(x0 - 10000 * (-b)), int(y0 - 10000 * a))

            if (angle % 180) < 45 or (angle % 180) > 135:
                rho_groups[0].append(float(rho))
                cv2.line(debug_img, pt1, pt2, (0, 0, 255), 2)
            else:
                rho_groups[1].append(float(rho))
                cv2.line(debug_img, pt1, pt2, (0, 255, 0), 2)

    if debug_output_path:
        debug_file = _as_path(debug_output_path)
        debug_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_file), debug_img)

    med_0, cv_0, count_0 = _process_axis_intervals(rho_groups[0])
    med_1, cv_1, count_1 = _process_axis_intervals(rho_groups[1])
    details = {
        "line_count": int(len(lines)),
        "dominant_angle": dominant_angle,
        "axis_0_interval": med_0,
        "axis_0_cv": cv_0,
        "axis_0_count": count_0,
        "axis_1_interval": med_1,
        "axis_1_cv": cv_1,
        "axis_1_count": count_1,
    }

    if count_0 == 0 and count_1 == 0:
        details["reason"] = "no valid checker intervals"
        return None, details
    if count_0 == 0:
        return med_1, details
    if count_1 == 0:
        return med_0, details
    return (med_0 if cv_0 <= cv_1 else med_1), details


def compute_scale_factor_from_ply(
    ply_path: str | Path = DEFAULT_PLY_INPUT,
    output_dir: str | Path = DEFAULT_SCALE_OUTPUT_DIR,
    resolution: int = DEFAULT_RESOLUTION,
    square_real_size_mm: float = DEFAULT_SQUARE_REAL_SIZE_MM,
) -> dict[str, Any]:
    """Compute the STL scale factor from a checkerboard point cloud PLY."""
    input_file = _require_file(ply_path, "Scale PLY")
    output_path = _as_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pcd = o3d.io.read_point_cloud(str(input_file))
    points = np.asarray(pcd.points)
    if len(points) == 0:
        raise ValueError(f"Point cloud is empty: {input_file}")

    pca_normal, centroid = refine_normal_with_pca(points)
    rotation_matrix = get_rotation_matrix(pca_normal)

    aligned = o3d.geometry.PointCloud(pcd)
    aligned.translate(-centroid)
    aligned.rotate(rotation_matrix, center=(0, 0, 0))

    projected_image_path = output_path / f"{input_file.stem}_projected_checkerboard.png"
    debug_lines_path = output_path / f"{input_file.stem}_debug_lines.png"
    projection = project_to_2d(aligned, resolution=resolution)

    cv2 = _require_cv2()
    cv2.imwrite(str(projected_image_path), projection)
    pixel_distance, detection_details = calculate_pixel_distance(
        projected_image_path,
        debug_output_path=debug_lines_path,
    )
    if pixel_distance is None or pixel_distance <= 0:
        raise RuntimeError(
            "Checkerboard grid detection failed. "
            f"Details: {json.dumps(detection_details, ensure_ascii=False)}"
        )

    unit_per_pixel = 1.0 / float(resolution)
    square_unit_size = float(pixel_distance) * unit_per_pixel
    scale_factor = float(square_real_size_mm) / square_unit_size

    report = {
        "input_file": str(input_file),
        "point_count": int(len(points)),
        "has_colors": bool(pcd.has_colors()),
        "resolution": int(resolution),
        "square_real_size_mm": float(square_real_size_mm),
        "pixel_distance": float(pixel_distance),
        "unit_per_pixel": unit_per_pixel,
        "square_unit_size": square_unit_size,
        "scale_factor": scale_factor,
        "pca_normal": pca_normal.tolist(),
        "centroid": centroid.tolist(),
        "projected_image": str(projected_image_path),
        "debug_lines_image": str(debug_lines_path),
        "detection": detection_details,
    }

    report_path = output_path / f"{input_file.stem}_scale_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["report_file"] = str(report_path)
    return report


if __name__ == "__main__":
    result = compute_scale_factor_from_ply()
    print(json.dumps(result, indent=2, ensure_ascii=False))
