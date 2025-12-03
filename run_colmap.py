import os
import subprocess
import sys

# 본인의 pc에서의 colmap.exe경로
COLMAP_BIN = r"C:\Users\hojun\bin\colmap.exe"

# 경로 안전장치
if not os.path.exists(COLMAP_BIN):
    print(f"\n❌ [오류] COLMAP 파일이 없습니다: {COLMAP_BIN}")
    sys.exit(1)

def run_command(cmd):
    print(f"🚀 Running: {cmd[0]} {cmd[1]} ...") 
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error occurred: {e}")
        sys.exit(1)

def run_colmap_pipeline(project_path):
    database_path = os.path.join(project_path, "database.db")
    image_path = os.path.join(project_path, "images")
    output_path = os.path.join(project_path, "sparse")
    
    os.makedirs(output_path, exist_ok=True)

    print("\n[1/3] Feature Extraction (Robust Mode)...")
    feature_extractor_cmd = [
        COLMAP_BIN, "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_path,
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--ImageReader.single_camera", "1" 
    ]
    run_command(feature_extractor_cmd)


    print("\n[2/3] Feature Matching (Exhaustive Mode)...")
    matcher_cmd = [
        COLMAP_BIN, "exhaustive_matcher",
        "--database_path", database_path
    ]
    run_command(matcher_cmd)

    print("\n[3/3] Reconstruction (Mapper)...")
    mapper_cmd = [
        COLMAP_BIN, "mapper",
        "--database_path", database_path,
        "--image_path", image_path,
        "--output_path", output_path
    ]
    run_command(mapper_cmd)

    print(f"\n 완료! 결과물 위치: {output_path}/0")
    print("터미널에서 'colmap gui' 입력 후, File > Import Model로 확인해보세요.")

if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    
    # 기존 DB 삭제 (설정이 바뀌었으니 깨끗하게 다시 시작)
    db_file = os.path.join(current_path, "database.db")
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
            print("🔄 기존 database.db 삭제 완료 (초기화)")
        except:
            pass

    if not os.path.exists(os.path.join(current_path, "images")):
        print(f"❌ '{current_path}' 폴더 안에 'images' 폴더가 없습니다.")
    else:
        run_colmap_pipeline(current_path)