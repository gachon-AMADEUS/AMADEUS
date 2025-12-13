#!/bin/bash

# ==============================================================================
# AMADEUS Pipeline Runner (Golden Master Version)
# ------------------------------------------------------------------------------
# [작동 순서]
# Step 1: 마스킹 (발+체커보드 남기기)
# Step 2: COLMAP (카메라 위치 추적)
# Step 3: 데이터 정규화 (Undistortion & 정렬)
# Step 4: Vanilla 3DGS 학습 (기본 형상 복원)
# Step 5: 점군 클리닝 (노이즈 제거 & 덩어리 추출)
# Step 6: SuGaR 메쉬 추출 (더블 확장자 버그 패치 포함)
# Step 7: 최종 후처리 (자르기 & 힐링)
# ==============================================================================

set -e
echo "🚀 [AMADEUS] Full Pipeline Started..."

# ------------------------------------------------------------------------------
# 0. 경로 및 환경 설정
# ------------------------------------------------------------------------------
BASE_DIR=$(pwd)
RAW_DATA_DIR="$BASE_DIR/data/raw_images"       # 원본 이미지 (jpg/png)
MASKED_DATA_DIR="$BASE_DIR/data/masked_images" # 전처리된 이미지 저장소
WORK_DIR="$BASE_DIR/colmap_work"               # 중간 작업 결과물
OUTPUT_DIR="$BASE_DIR/output"                  # 최종 결과물

# 모델 파일 경로
YOLO_MODEL="$BASE_DIR/models/yolo_best.pt"
SAM_MODEL="$BASE_DIR/models/sam_vit_h.pth"

# ------------------------------------------------------------------------------
# 1. [Step 1] 전처리 (Masking)
# ------------------------------------------------------------------------------
echo "📸 [Step 1] Running Data Preprocessing..."

# 기존 데이터 삭제 후 재생성 (섞임 방지)
if [ -d "$MASKED_DATA_DIR" ]; then rm -rf "$MASKED_DATA_DIR"; fi
mkdir -p "$MASKED_DATA_DIR"

# 발과 체커보드를 포함하여 마스킹 수행
python src/preprocessing/segment_foot.py \
    --input_dir "$RAW_DATA_DIR" \
    --output_dir "$MASKED_DATA_DIR" \
    --yolo_path "$YOLO_MODEL" \
    --sam_path "$SAM_MODEL"

# ------------------------------------------------------------------------------
# 2. [Step 2] COLMAP SfM
# ------------------------------------------------------------------------------
echo "📐 [Step 2] Running COLMAP SfM..."

mkdir -p "$WORK_DIR/sparse"

# Feature Extractor (특징점 8192개로 상향)
xvfb-run -a colmap feature_extractor \
    --database_path "$WORK_DIR/database.db" \
    --image_path "$MASKED_DATA_DIR" \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 1 \
    --SiftExtraction.max_num_features 8192

# Matcher
xvfb-run -a colmap exhaustive_matcher \
    --database_path "$WORK_DIR/database.db" \
    --SiftMatching.use_gpu 1

# Mapper
xvfb-run -a colmap mapper \
    --database_path "$WORK_DIR/database.db" \
    --image_path "$MASKED_DATA_DIR" \
    --output_path "$WORK_DIR/sparse" \
    --Mapper.init_min_tri_angle 0.1 --Mapper.multiple_models 1

# ------------------------------------------------------------------------------
# 3. [Step 3] 데이터 준비 (Undistort)
# ------------------------------------------------------------------------------
echo "🔧 [Step 3] Pre-processing for 3DGS (Undistortion)..."

# 기존 폴더 청소 (필수)
if [ -d "$WORK_DIR/sugar_ready" ]; then rm -rf "$WORK_DIR/sugar_ready"; fi
mkdir -p "$WORK_DIR/sugar_ready"

# 이미지 왜곡 보정
xvfb-run -a colmap image_undistorter \
    --image_path "$MASKED_DATA_DIR" \
    --input_path "$WORK_DIR/sparse/0" \
    --output_path "$WORK_DIR/sugar_ready" \
    --output_type COLMAP \
    --max_image_size 2000 

# 폴더 구조 재배치 (sparse/0 생성)
echo "📂 Reorganizing folder structure..."
mkdir -p "$WORK_DIR/sugar_ready/sparse/0"
mv "$WORK_DIR/sugar_ready/sparse/"*.bin "$WORK_DIR/sugar_ready/sparse/0/" 2>/dev/null || true
mv "$WORK_DIR/sugar_ready/sparse/"*.txt "$WORK_DIR/sugar_ready/sparse/0/" 2>/dev/null || true
mv "$WORK_DIR/sugar_ready/sparse/"*.ply "$WORK_DIR/sugar_ready/sparse/0/" 2>/dev/null || true

# 자동 정렬 (Auto Align)
if [ -f "src/postprocessing/auto_align_colmap.py" ]; then
    echo "📐 Auto-aligning COLMAP result..."
    python src/postprocessing/auto_align_colmap.py \
        --input_path "$WORK_DIR/sugar_ready/sparse/0"
fi

# ------------------------------------------------------------------------------
# 4. [Step 4] 3DGS 학습
# ------------------------------------------------------------------------------
echo "🔥 [Step 4] Training 3D Gaussian Splatting..."

# 기존 결과 삭제 후 재생성
if [ -d "$OUTPUT_DIR/vanilla_3dgs" ]; then rm -rf "$OUTPUT_DIR/vanilla_3dgs"; fi
mkdir -p "$OUTPUT_DIR/vanilla_3dgs"

cd gaussian-splatting
python train.py \
    -s "$WORK_DIR/sugar_ready" \
    -m "$OUTPUT_DIR/vanilla_3dgs" \
    --iterations 7000 \
    --sh_degree 0
cd ..

# ------------------------------------------------------------------------------
# 5. [Step 5] 후처리 (Cleaning)
# ------------------------------------------------------------------------------
echo "🧹 [Step 5] Cleaning Point Cloud..."

INPUT_PLY="$OUTPUT_DIR/vanilla_3dgs/point_cloud/iteration_7000/point_cloud.ply"

if [ ! -f "$INPUT_PLY" ]; then
    echo "❌ Error: Training output not found!"
    exit 1
fi

# 노이즈 제거
python src/postprocessing/clean_3dgs_ply_2.py \
    --input "$INPUT_PLY" \
    --output "$OUTPUT_DIR/vanilla_3dgs/cleaned.ply"

# 가장 큰 덩어리(발) 추출
python src/postprocessing/keep_largest_cluster_pcd.py \
    --input "$OUTPUT_DIR/vanilla_3dgs/cleaned.ply" \
    --output "$OUTPUT_DIR/vanilla_3dgs/largest.ply"

# ------------------------------------------------------------------------------
# 6. [Step 6] SuGaR 메쉬 추출
# ------------------------------------------------------------------------------
echo "🍬 [Step 6] Extracting Mesh (SuGaR)..."

# ★★★ [패치] Double Extension (.jpg.jpg) 버그 해결 ★★★
IMG_DIR="$WORK_DIR/sugar_ready/images"
if [ -d "$IMG_DIR" ]; then
    echo "🩹 Applying 'Double Extension' Patch..."
    cd "$IMG_DIR"
    for file in *.jpg; do
        if [ -f "$file" ] && [ ! -f "${file}.jpg" ]; then
            cp "$file" "${file}.jpg"
        fi
    done
    cd "$BASE_DIR"
fi

cd SuGaR
python train_full_pipeline.py \
    -s "$WORK_DIR/sugar_ready" \
    --gs_output_dir "$OUTPUT_DIR/vanilla_3dgs" \
    -r dn_consistency \
    --high_poly True \
    --export_obj True \
    --export_ply True \
    --square_size 32 \
    --refinement_time short
cd ..

# ------------------------------------------------------------------------------
# 7. [Step 7] 최종 메쉬 후처리
# ------------------------------------------------------------------------------
echo "✨ [Step 7] Final Mesh Processing..."

# SuGaR 결과물 자동 탐색
SUGAR_OUTPUT_DIR="$BASE_DIR/SuGaR/output/refined_mesh/sugar_ready"
SUGAR_MESH=$(find "$SUGAR_OUTPUT_DIR" -name "sugarfine_*.obj" | head -n 1)

if [ -z "$SUGAR_MESH" ]; then
    # 혹시 몰라 메인 output 폴더도 검색
    SUGAR_MESH=$(find "$OUTPUT_DIR" -name "sugarfine_*.obj" | head -n 1)
fi

if [ -z "$SUGAR_MESH" ]; then
    echo "❌ Error: SuGaR mesh file not found!"
    exit 1
fi

echo "   -> Found Mesh: $SUGAR_MESH"
mkdir -p "$OUTPUT_DIR/sugar"

# 자르기 (Cut)
python src/postprocessing/cut_sugar_with_pcd_mask.py \
    --ref_pcd "$OUTPUT_DIR/vanilla_3dgs/largest.ply" \
    --input_mesh "$SUGAR_MESH" \
    --output_mesh "$OUTPUT_DIR/sugar/cut_mesh.obj"

# 힐링 (Heal)
python src/postprocessing/heal_mesh.py \
    --input "$OUTPUT_DIR/sugar/cut_mesh.obj" \
    --output "$OUTPUT_DIR/final_foot_mesh.obj"

echo "🎉 [AMADEUS] All Steps Completed Successfully!"
echo "   -> Final Output: $OUTPUT_DIR/final_foot_mesh.obj"
!/bin/bash
set -e
echo "🚀 [AMADEUS] Step 6 & 7 (Absolute Path Version) Started..."

# ==============================================================================
# 7. [Step 6] SuGaR 메쉬 추출
# ==============================================================================
echo "🍬 [Step 6] Extracting Mesh..."

# 1. Double Extension Patch (.jpg.jpg) - 안전장치
# 경로: /app/colmap_work/sugar_ready/images
if [ -d "/app/colmap_work/sugar_ready/images" ]; then
    echo "🩹 Verifying image extensions..."
    cd "/app/colmap_work/sugar_ready/images"
    for file in *.jpg; do
        if [ -f "$file" ] && [ ! -f "${file}.jpg" ]; then
            cp "$file" "${file}.jpg"
        fi
    done
    cd "/app"
fi

# 2. SuGaR 실행 (경로를 절대경로로 박아넣음)
cd /app/SuGaR

# -s: 소스 데이터 (변수 대신 실제 경로 입력)
# --gs_output_dir: 학습 모델 (변수 대신 실제 경로 입력)
python train_full_pipeline.py \
    -s "/app/colmap_work/sugar_ready" \
    --gs_output_dir "/app/output/vanilla_3dgs" \
    -r dn_consistency \
    --high_poly True \
    --export_obj True \
    --export_ply True \
    --square_size 32 \
    --refinement_time short \
    # 상황에 맞게 변경해주기(필요 없으면 아래 두 줄 삭제)
    -f 50 \
    --postprocess_iterations 50

cd /app

# ==============================================================================
# 8. [Step 7] 최종 메쉬 후처리
# ==============================================================================
echo "✨ [Step 7] Final Mesh Processing..."

# 1. SuGaR 결과물 찾기 (절대 경로 사용)
# SuGaR 출력 예상 경로
SUGAR_MESH=$(find "/app/SuGaR/output/refined_mesh/sugar_ready" -name "sugarfine_*.obj" | head -n 1)

if [ -z "$SUGAR_MESH" ]; then
    echo "⚠️ Warning: Mesh not found in SuGaR folder. Checking fallback..."
    SUGAR_MESH=$(find "/app/output" -name "sugarfine_*.obj" | head -n 1)
fi

if [ -z "$SUGAR_MESH" ]; then
    echo "❌ Error: Mesh file not found!"
    exit 1
fi

echo "   -> Found Mesh: $SUGAR_MESH"

# 폴더 생성
mkdir -p "/app/output/sugar"

# 2. 자르기 (Cut)
python src/postprocessing/cut_sugar_with_pcd_mask.py \
    --ref_pcd "/app/output/vanilla_3dgs/largest.ply" \
    --input_mesh "$SUGAR_MESH" \
    --output_mesh "/app/output/sugar/cut_mesh.obj"

# 3. 힐링 (Heal)
python src/postprocessing/heal_mesh.py \
    --input "/app/output/sugar/cut_mesh.obj" \
    --output "/app/output/final_foot_mesh.obj"

echo "🎉 [AMADEUS] Completed!"
echo "   -> Result: /app/output/final_foot_mesh.obj"