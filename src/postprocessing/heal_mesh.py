import pymeshlab
import os
import argparse

def heal_mesh(input_mesh, output_mesh):
    print(f"[+] Loading Mesh: {input_mesh}")
    ms = pymeshlab.MeshSet()
    try:
        ms.load_new_mesh(input_mesh)
    except Exception as e:
        print(f"[!] Error loading mesh: {e}")
        return

    m = ms.current_mesh()
    print(f"    - Initial Vertices: {m.vertex_number()}")
    print(f"    - Initial Faces: {m.face_number()}")

    # 1. Clean Debris (부스러기 제거 - 면 개수 기준)
    print("[+] Removing small components (Face Count < 2000)...")
    try:
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=2000)
    except Exception:
        # 혹시 구버전이라 함수명이 다를 경우를 대비해 diameter 방식으로 fallback
        print("    [!] Face count filter failed, trying diameter filter...")
        # 절대값 직경 0.5 (Scene scale에 따라 다름, 보통 안전한 값)
        ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=0.5) 

    # 2. Close Holes (구멍 메우기)
    print("[+] Closing holes...")
    ms.meshing_close_holes(maxholesize=2000)

    # 3. Smoothing (피부 다림질)
    print("[+] Smoothing surface (Taubin)...")
    ms.apply_coord_taubin_smoothing(lambda_=0.5, mu=-0.53, stepsmoothnum=10)

    # 4. Final Cleanup
    print("[+] Final Cleanup...")
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_unreferenced_vertices()
    
    # [삭제됨] 에러 유발 코드: meshing_remove_zero_area_faces
    # 대신 기본적인 repair 수행
    ms.meshing_repair_non_manifold_edges()

    m_final = ms.current_mesh()
    print(f"    - Final Vertices: {m_final.vertex_number()}")
    print(f"    - Final Faces: {m_final.face_number()}")

    print(f"[+] Saving Healed Mesh to: {output_mesh}")
    ms.save_current_mesh(output_mesh)
    print("[+] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input Mesh (.obj)")
    parser.add_argument("--output", required=True, help="Output Healed Mesh (.obj)")
    args = parser.parse_args()

    heal_mesh(args.input, args.output)