#!/usr/bin/env python
import argparse
import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_pcd", required=True,
                        help="발 모양 point cloud ply (마스크용)")
    parser.add_argument("--input_mesh", required=True,
                        help="SuGaR에서 나온 mesh (obj/ply 등)")
    parser.add_argument("--output_mesh", required=True,
                        help="잘라낸 결과 mesh 저장 경로")
    parser.add_argument("--dist_thresh", type=float, default=None,
                        help="vertex-포인트 거리 threshold (단위: scene 스케일)")
    parser.add_argument("--voxel_size", type=float, default=None,
                        help="ref pcd downsample 크기 (옵션)")
    args = parser.parse_args()

    print(f"[+] Load ref PCD: {args.ref_pcd}")
    ref_pcd = o3d.io.read_point_cloud(args.ref_pcd)
    if len(ref_pcd.points) == 0:
        raise RuntimeError("ref_pcd 에 포인트가 없습니다.")

    # 필요하면 downsample 해서 속도 개선
    if args.voxel_size is not None and args.voxel_size > 0:
        print(f"[i] Voxel downsample (size={args.voxel_size})")
        ref_pcd = ref_pcd.voxel_down_sample(args.voxel_size)

    ref_pts = np.asarray(ref_pcd.points)
    print(f"[i] ref points: {ref_pts.shape}")

    print(f"[+] Load SuGaR mesh: {args.input_mesh}")
    mesh = o3d.io.read_triangle_mesh(args.input_mesh)
    if len(mesh.vertices) == 0:
        raise RuntimeError("input_mesh 에 vertices 가 없습니다.")
    mesh.compute_vertex_normals()
    verts = np.asarray(mesh.vertices)
    print(f"[i] mesh vertices: {verts.shape}, triangles: {np.asarray(mesh.triangles).shape}")

    # 거리 threshold 자동 설정 (ref_pcd bounding box 기준)
    if args.dist_thresh is None:
        bbox = ref_pcd.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        # 대략 전체 대각선 길이의 1~2% 정도
        dist_thresh = diag * 0.02
        print(f"[i] dist_thresh 자동 설정: {dist_thresh:.6f}")
    else:
        dist_thresh = args.dist_thresh
        print(f"[i] dist_thresh 수동 설정: {dist_thresh:.6f}")

    # KDTree 만들기
    print("[+] Build KDTree from ref_pcd...")
    kdtree = o3d.geometry.KDTreeFlann(ref_pcd)

    print("[+] Compute distance from each mesh vertex to nearest ref point...")
    dists = np.zeros(len(verts), dtype=np.float32)

    for i, v in enumerate(verts):
        # k=1 NN 검색
        k, idx, dist2 = kdtree.search_knn_vector_3d(v, 1)
        if k > 0:
            dists[i] = np.sqrt(dist2[0])
        else:
            dists[i] = np.inf
        # 너무 오래 걸리면 중간중간 진행상황 출력
        if (i + 1) % 50000 == 0:
            print(f"  processed {i + 1}/{len(verts)} vertices")

    print(f"[i] dist min/max/mean: {dists.min():.6f}, {dists.max():.6f}, {dists.mean():.6f}")

    # threshold 이하인 vertex만 유지
    keep_mask = dists < dist_thresh
    keep_indices = np.where(keep_mask)[0]
    print(f"[i] keep vertices: {len(keep_indices)} / {len(verts)}")

    if len(keep_indices) == 0:
        raise RuntimeError("남는 vertex 가 없습니다. dist_thresh 를 키워보세요.")

    # vertex 인덱스로 mesh 잘라내기
    print("[+] Cut mesh with ref_pcd mask...")
    mesh_cut = mesh.select_by_index(keep_indices, cleanup=True)

    print(f"[i] after cut: vertices={len(mesh_cut.vertices)}, "
          f"triangles={len(mesh_cut.triangles)}")

    # 가장 큰 connected component 만 남기기 (성게 조각 제거)
    print("[+] Keep largest connected component...")
    triangle_clusters, cluster_n_triangles, _ = \
        mesh_cut.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    if len(cluster_n_triangles) > 0:
        largest_cluster_idx = int(cluster_n_triangles.argmax())
        keep_triangles = np.where(triangle_clusters == largest_cluster_idx)[0]
        mesh_final = mesh_cut.select_by_index(keep_triangles, cleanup=True)
        print(f"[i] largest component: vertices={len(mesh_final.vertices)}, "
              f"triangles={len(mesh_final.triangles)}")
    else:
        print("[!] connected component 분석 실패. cut 결과 그대로 사용.")
        mesh_final = mesh_cut

    # 필요하면 살짝 smoothing
    mesh_final = mesh_final.filter_smooth_taubin(number_of_iterations=10)

    print(f"[+] Save output mesh: {args.output_mesh}")
    o3d.io.write_triangle_mesh(args.output_mesh, mesh_final)
    print("[✓] Done.")


if __name__ == "__main__":
    main()