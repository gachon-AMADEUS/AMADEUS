import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree
import argparse

def clean_3dgs_ply(
    input_ply,
    output_ply,
    nb_neighbors=20,
    std_ratio=2.0,
    use_radius=False,
    radius_ratio=0.01,   # bbox diagonal 대비 비율
    min_radius_neighbors=16
):
    print(f"[+] Loading 3DGS PLY: {input_ply}")
    ply = PlyData.read(input_ply)
    v = ply["vertex"].data

    # xyz 추출
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)
    N = xyz.shape[0]
    print(f"[+] Original Points: {N}")

    # -----------------------------
    # 1) Statistical Outlier Removal (Open3D 방식과 유사하게 구현)
    # -----------------------------
    tree = KDTree(xyz, leaf_size=40)
    dists, idx = tree.query(xyz, k=nb_neighbors + 1)  # self 포함
    mean_d = dists[:, 1:].mean(axis=1)               # self 제외 평균거리

    mu = mean_d.mean()
    sigma = mean_d.std()
    thresh = mu + std_ratio * sigma

    keep_mask = mean_d <= thresh
    print(f"[+] SOR keep: {int(keep_mask.sum())} / {N} (thresh={thresh:.6f})")

    # -----------------------------
    # 2) Radius Outlier Removal (선택)
    # -----------------------------
    if use_radius:
        # bbox diagonal 기반으로 radius 자동 설정
        xyz_min = xyz.min(axis=0)
        xyz_max = xyz.max(axis=0)
        diag = np.linalg.norm(xyz_max - xyz_min)
        radius = radius_ratio * diag

        # SOR 통과한 점들 기준으로 radius neighbor count 계산
        xyz_sor = xyz[keep_mask]
        tree2 = KDTree(xyz_sor, leaf_size=40)

        # query_radius는 indices list 반환
        neighbors_list = tree2.query_radius(xyz_sor, r=radius, count_only=False)

        counts = np.array([len(nbs) for nbs in neighbors_list])
        keep_mask_sor = counts >= min_radius_neighbors

        # 원래 인덱스 마스크로 복원
        sor_indices = np.where(keep_mask)[0]
        final_keep = np.zeros(N, dtype=bool)
        final_keep[sor_indices[keep_mask_sor]] = True

        keep_mask = final_keep
        print(f"[+] ROR keep: {int(keep_mask.sum())} / {N} "
              f"(radius={radius:.6f}, min_n={min_radius_neighbors})")

    # -----------------------------
    # 3) 3DGS 속성 보존 저장
    # -----------------------------
    new_v = v[keep_mask]
    new_el = PlyElement.describe(new_v, "vertex")
    new_ply = PlyData([new_el], text=ply.text)
    new_ply.comments = ply.comments
    new_ply.write(output_ply)

    print(f"[+] Cleaned Points: {len(new_v)}")
    print(f"[+] Saved (3DGS-compatible) to: {output_ply}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--std_ratio", type=float, default=2.0)
    parser.add_argument("--use_radius", action="store_true")
    parser.add_argument("--radius_ratio", type=float, default=0.01)
    parser.add_argument("--min_radius_neighbors", type=int, default=16)
    args = parser.parse_args()

    clean_3dgs_ply(
        args.input, args.output,
        nb_neighbors=args.k,
        std_ratio=args.std_ratio,
        use_radius=args.use_radius,
        radius_ratio=args.radius_ratio,
        min_radius_neighbors=args.min_radius_neighbors
    )

