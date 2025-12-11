#!/bin/bash

# ==============================================================================
# AMADEUS Pipeline Runner (Standard Version)
# ------------------------------------------------------------------------------
# [정석 파이프라인 구조]
# 1. Preprocessing: 원본에서 발만 남긴 마스크 이미지 생성
# 2. COLMAP SfM: '원본'을 사용하여 정확한 카메라 위치 추적 (Feature Matching)
# 3. Undistortion: '마스크 이미지'를 보정하여 3DGS 학습 데이터 준비
# 4. Training: 발 영역만 학습 진행
# ==============================================================================

set -e
echo "🚀 [AMADEUS] Pipeline Started..."

# ------------------------------------------------------------------------------
# 1. 경로 설정
# ------------------------------------------------------------------------------
BASE_DIR=$(pwd)
RAW_DATA_DIR="$BASE_DIR/data/raw_images"       # 원본 (위치 추적용)
MASKED_DATA_DIR="$BASE_DIR/data/masked_images" # 마스크 (학습용)
WORK_DIR="$BASE_DIR/colmap_work"
OUTPUT_DIR="$BASE_DIR/output"

YOLO_MODEL="$BASE_DIR/models/yolo_best.pt"
SAM_MODEL="$BASE_DIR/models/sam_vit_h.pth"

# ------------------------------------------------------------------------------
# 2. [Step 1] 전처리 (Masking)
# ------------------------------------------------------------------------------
echo "📸 [Step 1] Running Data Preprocessing..."
if [ -d "$MASKED_DATA_DIR" ]; then rm -rf "$MASKED_DATA_DIR"; fi
mkdir -p "$MASKED_DATA_DIR"

# [Check Point] 나중에 실행할 때, 이 스크립트가 리사이징을 하지 않도록 주의해야 함.
# (masked_images 해상도 == raw_images 해상도여야 Step 3 에러가 안 남)
python src/preprocessing/segment_foot.py \
    --input_dir "$RAW_DATA_DIR" \
    --output_dir "$MASKED_DATA_DIR" \
    --yolo_path "$YOLO_MODEL" \
    --sam_path "$SAM_MODEL"

# ------------------------------------------------------------------------------
# 3. [Step 2] COLMAP SfM (위치 추적)
# ------------------------------------------------------------------------------
echo "📐 [Step 2] Running COLMAP SfM..."
# 특징점 매칭을 위해 '원본 이미지'를 사용합니다.

mkdir -p "$WORK_DIR/sparse"

colmap feature_extractor \
    --database_path "$WORK_DIR/database.db" \
    --image_path "$RAW_DATA_DIR" \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --ImageReader.single_camera 1

# 300장 데이터 연결을 위해 전수 조사(Exhaustive) 사용
colmap exhaustive_matcher \
    --database_path "$WORK_DIR/database.db" 

colmap mapper \
    --database_path "$WORK_DIR/database.db" \
    --image_path "$RAW_DATA_DIR" \
    --output_path "$WORK_DIR/sparse" \
    --Mapper.min_num_matches 4 \
    --Mapper.init_min_tri_angle 2 \
    --Mapper.multiple_models 0 \
    --Mapper.extract_colors 1

# ------------------------------------------------------------------------------
# 4. [Step 3] 데이터 준비 (Undistort)
# ------------------------------------------------------------------------------
echo "🔧 [Step 3] Pre-processing for 3DGS..."
mkdir -p "$WORK_DIR/sugar_ready"

# ★ 중요: 학습용 데이터 생성을 위해 '마스크 이미지'를 사용합니다.
# (전제조건: MASKED 이미지와 RAW 이미지의 해상도가 동일해야 함)
colmap image_undistorter \
    --image_path "$MASKED_DATA_DIR" \
    --input_path "$WORK_DIR/sparse/0" \
    --output_path "$WORK_DIR/sugar_ready" \
    --output_type COLMAP \
    --max_image_size 2000 

# [Patch] Undistorter가 0번 폴더를 안 만드는 문제 해결
mkdir -p "$WORK_DIR/sugar_ready/sparse/0"
mv "$WORK_DIR/sugar_ready/sparse/"*.bin "$WORK_DIR/sugar_ready/sparse/0/" 2>/dev/null || true
mv "$WORK_DIR/sugar_ready/sparse/"*.txt "$WORK_DIR/sugar_ready/sparse/0/" 2>/dev/null || true

# 정렬 (Auto Align) - 필요 시 주석 해제
# python src/postprocessing/auto_align_colmap.py \
#     --input_path "$WORK_DIR/sugar_ready/sparse/0"

# ------------------------------------------------------------------------------
# 5. [Step 4] 3DGS 학습
# ------------------------------------------------------------------------------
echo "🔥 [Step 4] Training 3D Gaussian Splatting..."
cd gaussian-splatting
python train.py \
    -s "$WORK_DIR/sugar_ready" \
    -m "$OUTPUT_DIR/vanilla_3dgs" \
    --iterations 7000 \
    --sh_degree 0
cd ..

# ------------------------------------------------------------------------------
# 6. [Step 5~7] 후처리 (Cleaning & Meshing)
# ------------------------------------------------------------------------------
echo "🧹 [Step 5] Cleaning Point Cloud..."
python src/postprocessing/clean_3dgs_ply_2.py \
    --input "$OUTPUT_DIR/vanilla_3dgs/point_cloud/iteration_7000/point_cloud.ply" \
    --output "$OUTPUT_DIR/vanilla_3dgs/cleaned.ply"

python src/postprocessing/keep_largest_cluster_pcd.py \
    --input "$OUTPUT_DIR/vanilla_3dgs/cleaned.ply" \
    --output "$OUTPUT_DIR/vanilla_3dgs/largest.ply"

echo "🍬 [Step 6] Extracting Mesh..."
cd SuGaR
python train_full_pipeline.py \
    -s "$WORK_DIR/sugar_ready" \
    -c "$OUTPUT_DIR/vanilla_3dgs" \
    -r "sdf" \
    --low_poly True \
    --export_obj True
cd ..

echo "✨ [Step 7] Final Mesh Processing..."
# 파일명 자동 탐색
SUGAR_MESH=$(find "$OUTPUT_DIR/sugar/refined_mesh/sugar_ready" -name "sugarfine_*.obj" | head -n 1)

if [ -z "$SUGAR_MESH" ]; then
    SUGAR_MESH="$OUTPUT_DIR/sugar/refined_mesh/sugar_ready/sugarfine_mesh.obj"
    echo "⚠️ Warning: Could not auto-find mesh, trying default: $SUGAR_MESH"
fi

python src/postprocessing/cut_sugar_with_pcd_mask.py \
    --ref_pcd "$OUTPUT_DIR/vanilla_3dgs/largest.ply" \
    --input_mesh "$SUGAR_MESH" \
    --output_mesh "$OUTPUT_DIR/sugar/cut_mesh.obj"

python src/postprocessing/heal_mesh.py \
    --input "$OUTPUT_DIR/sugar/cut_mesh.obj" \
    --output "$OUTPUT_DIR/final_foot_mesh.obj"

echo "🎉 [AMADEUS] Pipeline Completed Successfully!"