# Foot Video To 3MF Pipeline

촬영된 발 영상을 입력하면 segmentation, COLMAP/2DGS reconstruction, 실제 크기 scaling, STL repair/simplify, slicing을 거쳐 Bambu/OrcaSlicer용 `.3mf`를 생성하는 파이프라인입니다.

실행해야 하는 파일은 하나입니다.

```bash
python pipeline.py
```

## 최종 흐름

```text
input/ 영상
-> 프레임 추출
-> YOLO/SAM foot/checkerboard segmentation
-> COLMAP SfM
-> 2DGS Docker mesh reconstruction
-> foot PLY를 STL로 후처리
-> checkerboard PLY로 scale factor 계산
-> STL 실제 크기 scaling
-> analyze / repair / simplify / floating check
-> OrcaSlicer 또는 Bambu Studio CLI slicing
-> output/<video_name>_scaled_sliced.3mf
```

## 폴더 구조

```text
foot-video-to-3mf-pipeline/
  pipeline.py                  # 최종 실행 파일
  input/                       # 영상 또는 테스트 STL/PLY 입력
  output/                      # STL/3MF 결과
  reports/                     # JSON/TXT 리포트
  assets/                      # best.pt, sam_vit_h_4b8939.pth 위치
  docker/2dgs/Dockerfile       # 2DGS CUDA Docker 환경
  mesh_pipeline.py             # STL analyze/repair/simplify helper
  scale_from_ply.py            # checkerboard PLY scale factor helper
  video_segmentation.py        # 영상 segmentation/reconstruction adapter
  frame_preprocessing.py       # 프레임 추출 helper
  yolo_sam_process.py          # YOLO/SAM segmentation helper
  reconstruction_pipeline.py   # COLMAP/2DGS Docker helper
  foot_postprocessing.py       # foot PLY -> STL helper
  slicer.py                    # Orca/Bambu CLI slicing helper
  requirements.txt
  requirements-video.txt
```

`pipeline.py`만 직접 실행하면 됩니다. 나머지 `.py` 파일들은 내부 helper입니다.

## 팀원이 준비해야 하는 것

`assets/`에 모델 파일을 넣습니다.

```text
assets/best.pt
assets/sam_vit_h_4b8939.pth
```

`input/`에 처리할 영상을 하나 넣습니다.

```text
input/foot_capture.mp4
```

전체 자동화를 위해 컴퓨터에 아래 프로그램도 필요합니다.

- Python 3.10 또는 3.11 권장
- Docker
- NVIDIA Container Toolkit
- NVIDIA CUDA GPU
- COLMAP
- OrcaSlicer 또는 Bambu Studio CLI

Windows + NVIDIA GPU 환경에서는 Docker Desktop의 WSL2/NVIDIA GPU 사용이 켜져 있어야 합니다.

## 설치

```bash
cd foot-video-to-3mf-pipeline
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

패키지 설치:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements-video.txt
```

## 2DGS Docker image

처음 한 번은 Docker image를 빌드해야 합니다.

```bash
docker build -t 2dgs:cu118 docker/2dgs
```

`pipeline.py`는 기본적으로 Docker build도 시도합니다. 이미 image가 있으면 Docker cache를 사용합니다.

매번 build 시도를 건너뛰려면:

```bash
python pipeline.py --skip-docker-build
```

## 실행

가장 단순한 실행:

```bash
python pipeline.py
```

이 명령은 `input/` 폴더에서 가장 최근 영상 파일을 자동으로 찾습니다.

특정 영상을 지정하려면:

```bash
python pipeline.py --input-video input/foot_capture.mp4
```

최종 결과:

```text
output/<video_name>_scaled_sliced.3mf
reports/<video_name>_pipeline_YYYYMMDD_HHMMSS.json
reports/<video_name>_pipeline_YYYYMMDD_HHMMSS.txt
```

## 이미 reconstruction PLY가 있을 때

COLMAP/2DGS를 다시 돌리지 않고 기존 PLY를 쓰려면:

```bash
python pipeline.py \
  --input-video input/foot_capture.mp4 \
  --reconstruction-ply input/unbounded_default_post.ply \
  --scale-ply input/foot_for_scale_2.ply \
  --skip-reconstruction
```

## COLMAP reconstruction이 실패할 때

발 표면은 텍스처가 약하고 segmentation 이미지의 배경이 지워져 COLMAP이 불안정할 수 있습니다. 먼저 기존 중간 결과를 지우거나 `--overwrite-frames`를 붙여 프레임을 다시 뽑은 뒤 재시도합니다.

```bash
python pipeline.py \
  --input-video input/foot_capture.mp4 \
  --skip-docker-build \
  --overwrite-frames \
  --reconstruction-image-set both \
  --min-interval 2 \
  --sim-threshold 0.96
```

`assets/vocab_tree_flickr100K_words32K.bin` 파일이 있으면 sequential matcher에서 loop detection을 자동으로 사용합니다. 다른 위치에 있다면 직접 넘깁니다.

```bash
python pipeline.py \
  --input-video input/foot_capture.mp4 \
  --skip-docker-build \
  --colmap-vocab-tree-path assets/vocab_tree_flickr100K_words32K.bin
```

그래도 실패하면 영상 자체를 다시 촬영하는 편이 빠릅니다. 발과 체커보드가 선명하게 보이도록 천천히 360도 돌고, 프레임 간 겹침이 많게 촬영하세요.

## Slicer 설정

기본 slicer engine은 `orca`입니다.

```bash
python pipeline.py --slicer-engine orca
```

Bambu Studio CLI를 쓰려면:

```bash
python pipeline.py --slicer-engine bambu
```

자동 탐색이 실패하면 slicer 실행 파일 경로를 직접 넘깁니다.

Windows 예시:

```bash
python pipeline.py --slicer-bin "C:\Program Files\OrcaSlicer\orca-slicer.exe"
```

macOS 예시:

```bash
python pipeline.py --slicer-bin "/Applications/OrcaSlicer.app/Contents/MacOS/OrcaSlicer"
```

## VS Code에서 실행

1. VS Code에서 `foot-video-to-3mf-pipeline` 폴더를 엽니다.
2. Python 확장을 설치합니다.
3. `Terminal > Run Task... > Setup Python Environment`를 실행합니다.
4. `assets/`에 모델 파일을 넣습니다.
5. `input/`에 영상 파일을 넣습니다.
6. Run and Debug에서 `Run Full Video To 3MF Pipeline`을 실행합니다.

## 정말 영상 하나만 넣으면 바로 되나?

아래 조건이 한 번 준비되어 있으면 가능합니다.

- `assets/best.pt` 존재
- `assets/sam_vit_h_4b8939.pth` 존재
- Docker/NVIDIA/COLMAP 사용 가능
- OrcaSlicer 또는 Bambu Studio CLI 사용 가능
- `input/`에 영상이 하나 이상 있음

그 상태에서는:

```bash
python pipeline.py
```

만 실행하면 `.3mf` 생성까지 이어집니다.

## 현재 구현하지 않은 것

- Bambu Cloud 자동 업로드 API 호출
- 웹 UI
- YOLO 데이터셋 라벨링/학습 자동화
- SAM 모델 학습
