import open3d as o3d
import argparse
import numpy as np

def keep_largest_cluster(input_path, output_path, min_points=100):
    """
    가장 큰 포인트 클러스터만 남기고 나머지 노이즈를 제거합니다.
    """
    try:
        pcd = o3d.io.read_point_cloud(input_path)
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return

    print(f"[+] Original Points: {len(pcd.points)}")

    # DBSCAN 클러스터링을 사용하여 클러스터 찾기
    # eps와 min_points는 데이터셋에 따라 조정될 수 있지만, 여기서는 표준값을 사용합니다.
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=min_points))

    if len(labels) == 0:
        print("[-] No clusters found.")
        o3d.io.write_point_cloud(output_path, pcd)
        return
        
    # 각 클러스터의 크기 계산
    label_set = np.unique(labels)
    largest_label = -1
    largest_count = 0

    for label in label_set:
        if label == -1: # 노이즈 레이블은 무시
            continue
        count = np.sum(labels == label)
        if count > largest_count:
            largest_count = count
            largest_label = label

    if largest_label == -1:
        print("[-] Only noise points found (label -1).")
        o3d.io.write_point_cloud(output_path, pcd)
        return
        
    # 가장 큰 클러스터의 인덱스만 추출
    indices = np.where(labels == largest_label)[0]
    
    # 새로운 포인트 클라우드 생성
    points = np.asarray(pcd.points)[indices]
    colors = np.asarray(pcd.colors)[indices] if pcd.has_colors() else None
    
    cleaned_pcd = o3d.geometry.PointCloud()
    cleaned_pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        cleaned_pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"[+] Largest Cluster Found (Label {largest_label}): {len(cleaned_pcd.points)} points")
    
    # 결과 저장
    o3d.io.write_point_cloud(output_path, cleaned_pcd)
    print(f"[+] Saved largest cluster to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keep the largest point cluster in a PCD/PLY file.")
    parser.add_argument("--input", required=True, help="Input PLY or PCD file path.")
    parser.add_argument("--output", required=True, help="Output PLY or PCD file path.")
    parser.add_argument("--min_points", type=int, default=100, help="Minimum number of points to form a cluster.")
    
    args = parser.parse_args()
    
    print(f"Running largest cluster extraction: {args.input}")
    keep_largest_cluster(args.input, args.output, args.min_points)