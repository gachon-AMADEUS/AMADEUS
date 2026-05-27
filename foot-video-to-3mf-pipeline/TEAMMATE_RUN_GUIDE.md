# Teammate Run Guide

이 폴더는 촬영 영상에서 Bambu/OrcaSlicer용 `.3mf`까지 만드는 최종 파이프라인입니다.

## 0. GitHub에서 받기

GitHub에서 repository를 clone한 뒤 이 폴더로 들어갑니다.

```bash
git clone <TEAM_PROJECT_REPOSITORY_URL>
cd <repository-name>/foot-video-to-3mf-pipeline
```

GitHub 웹에서 ZIP으로 받는 경우:

1. GitHub repository 페이지에서 `Code > Download ZIP`을 누릅니다.
2. ZIP 압축을 풉니다.
3. 압축을 푼 폴더 안의 `foot-video-to-3mf-pipeline` 폴더를 VS Code로 엽니다.

## 1. 먼저 준비할 파일

아래 파일을 넣어주세요.

```text
assets/best.pt
assets/sam_vit_h_4b8939.pth
input/foot_capture.mp4
```

- `best.pt`: foot/checkerboard를 검출하도록 학습된 YOLO segmentation 모델
- `sam_vit_h_4b8939.pth`: SAM checkpoint
- `foot_capture.mp4`: 처리할 촬영 영상

이미 2DGS 결과 PLY가 있다면 아래 파일도 넣을 수 있습니다.

```text
input/unbounded_default_post.ply
input/foot_for_scale_2.ply
```

## 2. Python 환경 만들기

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements-video.txt
```

STL/PLY 처리만 테스트할 때는 `requirements.txt`만 설치해도 됩니다.

```bash
pip install -r requirements.txt
```

## 3. 기존 STL/PLY로 빠른 테스트

영상, YOLO, SAM, Docker 없이 파이프라인 연결만 확인합니다.

```bash
python process_video_pipeline.py input/demo_video.mp4 \
  --use-existing-assets \
  --fallback-stl input/test.stl \
  --fallback-ply input/foot_for_scale_2.ply \
  --no-slice
```

## 4. 이미 만들어진 2DGS PLY를 사용해서 실행

외부에서 만든 `unbounded_default_post.ply`가 이미 있으면 이 방식이 가장 단순합니다.

```bash
python process_video_pipeline.py input/foot_capture.mp4 \
  --use-notion-team-code \
  --assets-dir assets \
  --reconstruction-ply input/unbounded_default_post.ply \
  --scale-ply input/foot_for_scale_2.ply
```

결과:

```text
output/foot_capture_scaled_sliced.3mf
reports/foot_capture_video_pipeline_YYYYMMDD_HHMMSS_report.json
reports/foot_capture_video_pipeline_YYYYMMDD_HHMMSS_report.txt
```

## 5. COLMAP + 2DGS Docker까지 자동 실행

이 단계는 macOS가 아니라 NVIDIA CUDA GPU가 잡히는 Linux/Docker 환경에서 실행하세요.

필요한 것:

- Docker
- NVIDIA Container Toolkit
- COLMAP
- NVIDIA CUDA GPU

먼저 2DGS Docker image를 빌드합니다.

```bash
docker build -t 2dgs:cu118 docker/2dgs
```

그 다음 전체 파이프라인을 실행합니다.

```bash
python process_video_pipeline.py input/foot_capture.mp4 \
  --use-notion-team-code \
  --run-reconstruction \
  --assets-dir assets \
  --scene-name foot_scene \
  --reconstruction-image-set foot
```

`--scale-ply`를 주지 않으면 checkerboard segmentation 이미지도 별도로 COLMAP/2DGS reconstruction을 실행해서 scale factor 계산에 사용합니다.

## 6. VS Code에서 실행

1. VS Code에서 이 폴더를 엽니다.
2. Python 확장을 설치합니다.
3. `Terminal > Run Task... > Setup Python Environment`를 실행합니다.
4. 왼쪽 `Run and Debug`에서 원하는 실행 항목을 선택합니다.

자주 쓰는 항목:

- `Process STL`
- `Process Scaled STL from PLY`
- `Process Video With Notion Team Code`
- `Process Video With COLMAP + 2DGS Docker`

## 7. 출력 위치

```text
output/
reports/
```

최종 sliced 3MF는 보통 아래 이름으로 생성됩니다.

```text
output/<video_name>_scaled_sliced.3mf
```

## 8. 현재 구현하지 않은 것

- Bambu Cloud 자동 업로드 API 호출
- 웹 UI
- YOLO 데이터셋 라벨링/학습 자동화
- SAM 모델 학습
