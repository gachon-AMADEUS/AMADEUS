# Bambu STL Pipeline MVP

로컬 STL 파일을 자동으로 분석, repair, simplify하고, 필요하면 OrcaSlicer/Bambu Studio CLI로 slicing까지 실행하는 최소 Python 파이프라인입니다.

현재 MVP는 로컬 파일 처리와 3MF slicing까지만 담당합니다. 웹 서버, 프론트엔드, Bambu 프린터 업로드는 아직 구현하지 않았습니다.

현재 이 Mac에서는 Bambu Studio `02.06.00.51` CLI와 최신 Orca CLI가 slicing 중 내부 crash를 냈습니다. 그래서 기본 slicing 엔진은 성공 확인된 구버전 Orca CLI와 CLI-safe profile을 사용하도록 설정했습니다.

## 폴더 구조

```text
bambu-stl-pipeline/
  input/
  output/
  reports/
  mesh_pipeline.py
  analyze_stl.py
  repair_stl.py
  process_stl.py
  process_scaled_stl.py
  scale_from_ply.py
  frame_preprocessing.py
  yolo_sam_process.py
  pipeline_before_colmap.py
  foot_postprocessing.py
  reconstruction_pipeline.py
  video_segmentation.py
  process_video_pipeline.py
  slicer.py
  requirements.txt
  requirements-video.txt
  docker/2dgs/Dockerfile
  docker/2dgs/README.md
  README.md
```

- `input/`: 처리할 STL 파일을 넣는 폴더
- `output/`: repair/simplify된 STL과 slicing 결과가 저장되는 폴더
- `reports/`: 전체 처리 결과 JSON/TXT report가 저장되는 폴더
- `mesh_pipeline.py`: 분석, repair, simplify 핵심 함수
- `analyze_stl.py`: STL 분석만 실행
- `repair_stl.py`: PyMeshLab repair만 실행
- `process_stl.py`: 전체 파이프라인 실행
- `process_scaled_stl.py`: STL 처리 후 PLY로 계산한 scale factor를 적용하고 slicing
- `scale_from_ply.py`: PLY 체커보드에서 scale factor 계산
- `frame_preprocessing.py`: 팀원 코드 기반 영상 프레임 추출
- `yolo_sam_process.py`: 팀원 코드 기반 YOLO + SAM segmentation
- `pipeline_before_colmap.py`: 영상에서 foot/checkerboard/both 이미지 폴더 생성
- `foot_postprocessing.py`: 2DGS 결과 PLY를 발 STL로 후처리
- `reconstruction_pipeline.py`: segmentation 이미지에서 COLMAP dataset을 만들고 2DGS Docker를 호출
- `video_segmentation.py`: 영상 segmentation/후처리 코드를 최종 파이프라인에 연결하는 어댑터
- `process_video_pipeline.py`: 영상 입력부터 STL/PLY 스케일링 및 3MF slicing까지 최종 통합 실행
- `slicer.py`: OrcaSlicer/Bambu Studio CLI slicing 실행
- `docker/2dgs/Dockerfile`: CUDA 11.8 기반 2DGS 학습/mesh 추출 Docker image

## 설치

프로젝트 폴더로 이동합니다.

```bash
cd bambu-stl-pipeline
```

아래 예시는 `python` 명령을 사용합니다. macOS에서 `python: command not found`가 나오면 같은 자리에 `python3`를 사용하세요.

가상환경을 만듭니다.

```bash
python -m venv .venv
```

가상환경을 켭니다.

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

필요한 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

## VS Code에서 실행하기

터미널 명령을 직접 치지 않고 VS Code에서 실행하려면 아래 순서로 하면 됩니다.

1. VS Code를 엽니다.
2. `File > Open Folder...`를 누릅니다.
3. 이 폴더를 선택합니다: `bambu-stl-pipeline`
4. 왼쪽 Extensions에서 `Python` 확장을 설치합니다. Microsoft에서 만든 `Python` 확장입니다.
5. 메뉴에서 `Terminal > Run Task...`를 누릅니다.
6. `Setup Python Environment`를 선택합니다. 이 작업이 `.venv`를 만들고 필요한 패키지를 설치합니다.
7. `input/` 폴더에 STL 파일을 넣고 이름을 `test.stl`로 맞춥니다.
8. 왼쪽 `Run and Debug` 아이콘을 누릅니다.
9. 위쪽 드롭다운에서 원하는 실행을 고릅니다.

실행 항목:

- `Analyze STL`: 분석만 실행
- `Repair STL`: PyMeshLab repair만 실행
- `Process STL`: repair, simplify, slicing까지 전체 실행
- `Process Scaled STL from PLY`: `test.stl`을 처리하고 `foot_for_scale_2.ply`로 실제 크기 스케일 적용 후 slicing
- `Process Scaled STL from PLY (No Slice)`: slicing 없이 스케일된 STL까지만 생성
- `Process Video With Notion Team Code`: Notion 팀원 코드 흐름으로 영상 입력부터 3MF까지 실행
- `Process Video With COLMAP + 2DGS Docker`: COLMAP/2DGS Docker 단계까지 실행
- `Process Video Pipeline Demo (Existing STL/PLY)`: 영상 segmentation 코드 없이 기존 STL/PLY로 최종 연결 테스트
- `Process STL Only (No Slice)`: slicing 없이 STL 처리만 실행
- `Slice Processed STL`: `output/test_processed.stl`을 Orca legacy CLI로 slicing
- `Slice Processed STL Dry Run`: slicing 명령어만 미리 확인

VS Code가 Python 인터프리터를 물어보면 프로젝트 안의 `.venv/bin/python`을 선택하세요.

설치되는 주요 패키지:

- `numpy`
- `trimesh`
- `open3d`
- `pymeshlab`
- `pymeshfix`
- `opencv-python-headless`
- `Pillow`
- `mapbox-earcut`

영상 segmentation까지 직접 실행하려면 추가 패키지가 필요합니다.

```bash
pip install -r requirements-video.txt
```

추가 설치 항목:

- `torch`
- `ultralytics`
- `segment-anything`

## STL 파일 넣기

처리할 STL 파일을 `input/` 폴더에 넣습니다.

예:

```text
bambu-stl-pipeline/input/test.stl
```

## 분석만 실행

```bash
python analyze_stl.py input/test.stl
```

인자를 생략하면 기본으로 `input/test.stl`을 분석합니다.

```bash
python analyze_stl.py
```

출력 예시:

```text
STL analysis: input/test.stl
{
  "triangle_count": 12000,
  "vertex_count": 6002,
  "watertight": true,
  "winding_consistent": true,
  "boundary_edges": 0,
  "non_manifold_edges": 0,
  "volume_available": true,
  "volume": 12345.67
}
```

## PyMeshLab repair만 실행

```bash
python repair_stl.py input/test.stl
```

결과 파일:

```text
output/test_repaired.stl
```

repair 전/후 report가 터미널에 출력됩니다.

## 전체 처리 실행

```bash
python process_stl.py input/test.stl
```

이 명령은 기본으로 repair, simplify 후 Orca legacy CLI slicing까지 시도합니다.

전체 순서:

1. `trimesh`로 처리 전 분석
2. `PyMeshLab`으로 기본 cleaning/repair
3. 다시 분석
4. 아직 출력에 충분하지 않으면 `MeshFix` fallback
5. 다시 분석
6. triangle 수가 너무 많으면 `Open3D` simplify
7. floating regions 검사 및 작은 떠 있는 조각 정리
8. 최종 STL 저장
9. Orca legacy CLI로 slicing 시도
10. JSON report와 TXT report 저장

slicing 없이 STL 처리만 하고 싶으면:

```bash
python process_stl.py input/test.stl --no-slice
```

기본 simplify 기준:

- `300000` triangles 이하: simplify 안 함
- `300000` triangles 초과: 약 `180000` triangles로 줄이기 시도

코드에서 기본값을 바꾸려면 `mesh_pipeline.py` 상단의 아래 값을 수정하세요.

```python
DEFAULT_SIMPLIFY_THRESHOLD = 300_000
DEFAULT_TARGET_TRIANGLES = 180_000
```

실행할 때만 바꾸려면:

```bash
python process_stl.py input/test.stl --simplify-threshold 500000 --target-triangles 250000
```

floating regions 단계는 다음을 확인합니다.

- 본체와 분리되어 공중에 떠 있는 작은 조각
- bed 위쪽에 있는 아래 방향 면/오버행 영역
- support가 필요해 보이는지 여부

작은 떠 있는 분리 조각은 자동으로 제거합니다. 큰 오버행이나 모델의 중요한 일부로 보이는 영역은 형상을 억지로 바꾸지 않고 report에 `support_recommended`로 표시합니다.

floating regions 기준을 실행할 때만 바꾸려면:

```bash
python process_stl.py input/test.stl --floating-bed-tolerance 0.1 --floating-remove-ratio 0.002
```

결과 STL:

```text
output/test_processed.stl
```

Slicing이 성공하면 실제 G-code가 들어 있는 3MF도 생성됩니다.

```text
output/test_sliced.3mf
```

중간 결과가 생길 수도 있습니다.

```text
output/test_pymeshlab_repaired.stl
output/test_meshfix_repaired.stl
output/test_simplified.stl
output/test_floating_fixed.stl
```

Report 파일:

```text
reports/test_YYYYMMDD_HHMMSS_report.json
reports/test_YYYYMMDD_HHMMSS_report.txt
```

## PLY 기준 실제 크기 스케일링

`input/test.stl`을 실제 출력할 오브젝트로 사용하고, `input/foot_for_scale_2.ply`를 체커보드 스케일 기준으로 사용하려면 아래 명령을 실행합니다.

```bash
python process_scaled_stl.py input/test.stl --scale-ply input/foot_for_scale_2.ply
```

전체 순서:

1. `input/test.stl` 분석
2. PyMeshLab/MeshFix repair
3. 필요하면 Open3D simplify
4. `input/foot_for_scale_2.ply`를 PCA로 수평 정렬
5. 체커보드 이미지를 정사 투영 PNG로 저장
6. Hough line으로 격자 간격을 찾아 scale factor 계산
7. 처리된 STL에 scale factor 적용
8. 스케일된 STL을 다시 검사
9. support가 필요하면 support profile 자동 적용
10. Bambu Cloud 업로드에 사용할 sliced `.3mf` 생성

기본 체커보드 한 칸 크기는 `30mm`입니다. 바꾸려면:

```bash
python process_scaled_stl.py input/test.stl --scale-ply input/foot_for_scale_2.ply --checker-square-mm 25
```

스케일 계산만 확인하려면:

```bash
python scale_from_ply.py
```

생성 파일:

```text
output/test_scaled_mm.stl
output/test_scaled_sliced.3mf
output/scale_debug/foot_for_scale_2_projected_checkerboard.png
output/scale_debug/foot_for_scale_2_debug_lines.png
output/scale_debug/foot_for_scale_2_scale_report.json
reports/test_scaled_YYYYMMDD_HHMMSS_report.json
reports/test_scaled_YYYYMMDD_HHMMSS_report.txt
```

현재 `input/foot_for_scale_2.ply` 기준 검출값:

```text
pixel distance: 245 px
scale factor: 122.44897959
```

## 영상 입력 최종 파이프라인

최종 목표 흐름은 아래와 같습니다.

```text
촬영 영상
-> 팀원 코드로 프레임 추출
-> YOLO + SAM으로 foot/checkerboard/both segmentation 이미지 생성
-> 외부 SfM/COLMAP/2DGS에서 unbounded_default_post.ply 생성
-> foot_postprocessing.py로 foot STL 생성
-> checkerboard PLY로 scale factor 계산
-> STL에 실제 크기 스케일 적용
-> analyze / repair / simplify / floating check
-> Orca legacy CLI slicing
-> Bambu Cloud 업로드용 .3mf 생성
```

Notion에 정리된 팀원 코드 흐름을 사용하는 기본 명령은 아래와 같습니다.

```bash
python process_video_pipeline.py input/foot_capture.mp4 \
  --use-notion-team-code \
  --assets-dir assets \
  --reconstruction-ply input/unbounded_default_post.ply \
  --scale-ply input/foot_for_scale_2.ply
```

필요한 입력:

- `input/foot_capture.mp4`: 촬영 영상
- `assets/best.pt`: finetuning된 YOLO segmentation 모델
- `assets/sam_vit_h_4b8939.pth`: SAM checkpoint
- `input/unbounded_default_post.ply`: 외부 SfM/COLMAP/2DGS에서 나온 발 mesh PLY
- `input/foot_for_scale_2.ply`: scale factor 계산에 사용할 checkerboard PLY

`unbounded_default_post.ply`를 이미 가지고 있으면 `--reconstruction-ply`로 넘기면 됩니다. 아직 PLY가 없다면 `--run-reconstruction`으로 COLMAP/2DGS Docker 단계까지 자동 실행할 수 있습니다.

COLMAP과 2DGS Docker까지 자동으로 붙이고 싶으면 아래처럼 실행합니다.

```bash
python process_video_pipeline.py input/foot_capture.mp4 \
  --use-notion-team-code \
  --run-reconstruction \
  --assets-dir assets \
  --scale-ply input/foot_for_scale_2.ply \
  --scene-name foot_scene \
  --reconstruction-image-set foot
```

이 명령의 추가 흐름:

1. `output/video_segmentation/<video_name>/segmentation/foot` 이미지를 모읍니다.
2. `output/2dgs_dataset/foot_scene/images`로 복사합니다.
3. `colmap feature_extractor`, `sequential_matcher`, `mapper`를 실행해 `sparse/0/*.bin`을 만듭니다.
4. Docker로 `2dgs:cu118` 이미지를 실행합니다.
5. 컨테이너 안에서 `train_2dgs.sh --depth_ratio 0`을 실행합니다.
6. 이어서 `extract_mesh_quick.sh`를 실행합니다.
7. `output/2dgs_output/foot_scene/mesh_quick/unbounded_default_post.ply`를 찾아 발 STL 후처리로 넘깁니다.

`--scale-ply`를 생략하면 checkerboard segmentation 이미지도 별도 scene으로 COLMAP/2DGS 재구성한 뒤, 그 PLY를 scale factor 계산에 사용합니다. 이 방식은 더 자동화되어 있지만 2DGS를 발용/체커보드용으로 두 번 실행하므로 시간이 더 걸립니다.

```bash
python process_video_pipeline.py input/foot_capture.mp4 \
  --use-notion-team-code \
  --run-reconstruction \
  --assets-dir assets \
  --scene-name foot_scene \
  --reconstruction-image-set foot
```

필요한 실행 환경:

- Docker
- COLMAP
- NVIDIA CUDA GPU가 잡히는 Linux 환경
- Docker image `2dgs:cu118`

이 프로젝트에는 실행용 Dockerfile이 포함되어 있습니다. Linux + NVIDIA CUDA 환경에서 먼저 이미지를 빌드하세요.

```bash
docker build -t 2dgs:cu118 docker/2dgs
```

또는 파이프라인에서 빌드까지 하려면:

```bash
python process_video_pipeline.py input/foot_capture.mp4 \
  --use-notion-team-code \
  --run-reconstruction \
  --build-2dgs-image \
  --2dgs-dockerfile docker/2dgs/Dockerfile \
  --scale-ply input/foot_for_scale_2.ply
```

현재 이 Mac에서 바로 막히는 이유:

- 이 Mac에는 `docker` 명령이 없었습니다.
- Notion Dockerfile은 `nvidia/cuda:11.8` 기반이고 실행 명령도 `docker run --gpus all`입니다.
- macOS Docker Desktop은 이 CUDA GPU runtime을 그대로 제공하지 못합니다.
- 따라서 2DGS 학습/mesh 추출은 NVIDIA GPU가 있는 Linux PC나 클라우드/원격 서버에서 실행하는 것이 현실적입니다.

영상 segmentation dependency 설치:

```bash
pip install -r requirements-video.txt
```

segmentation 단계만 끝내고 2DGS 결과 PLY가 없으면, 프로그램은 어디에 `unbounded_default_post.ply`를 넣어야 하는지 알려주고 멈춥니다.

이전처럼 이미 foot STL과 checkerboard PLY가 있는 경우에도 실행할 수 있습니다.

```bash
python process_video_pipeline.py input/foot_capture.mp4 \
  --team-module team_segmentation \
  --team-function segment_video
```

팀원 함수는 원본 코드를 크게 바꾸지 않고 아래 중 하나만 반환하면 됩니다.

```python
{"foot_stl": "path/to/foot.stl", "checker_ply": "path/to/checkerboard.ply"}
```

또는:

```python
("path/to/foot.stl", "path/to/checkerboard.ply")
```

팀원 코드가 별도 스크립트라면:

```bash
python process_video_pipeline.py input/foot_capture.mp4 \
  --team-script path/to/team_script.py
```

이 경우 어댑터는 스크립트를 다음 인자와 환경변수로 호출합니다.

```text
python team_script.py <video_path> <output_dir> <result_json>

VIDEO_INPUT_PATH
SEGMENTATION_OUTPUT_DIR
SEGMENTATION_RESULT_JSON
```

이미 segmentation 결과 JSON이 있으면:

```bash
python process_video_pipeline.py input/foot_capture.mp4 \
  --segmentation-result-json output/video_segmentation/result.json
```

현재처럼 영상 segmentation 모델 파일이나 2DGS 결과 PLY가 없을 때 전체 연결만 테스트하려면:

```bash
python process_video_pipeline.py input/demo_video.mp4 \
  --use-existing-assets \
  --fallback-stl output/test_processed.stl \
  --fallback-ply input/foot_for_scale_2.ply
```

최종 출력:

```text
output/<video_name>_scaled_sliced.3mf
reports/<video_name>_video_pipeline_YYYYMMDD_HHMMSS_report.json
reports/<video_name>_video_pipeline_YYYYMMDD_HHMMSS_report.txt
```

## Slicing 실행

`process_stl.py`는 기본으로 repair/simplify 후 slicing까지 실행합니다. 기본 엔진은 Bambu Studio가 아니라 `orca-legacy`입니다.

```bash
python process_stl.py input/test.stl
```

STL 처리만 하고 slicing을 끄려면:

```bash
python process_stl.py input/test.stl --no-slice
```

`slicer.py`로 slicing만 따로 실행할 수도 있습니다.

단독 slicing 실행:

```bash
python slicer.py output/test_processed.stl --output output/test_sliced.3mf
```

명령어만 미리 확인:

```bash
python slicer.py output/test_processed.stl --output output/test_sliced.3mf --dry-run
```

현재 기본 slicing 방식:

- 실행 파일: `~/Applications/OrcaSlicer-CLI-1.9.5.app/Contents/MacOS/OrcaSlicer`
- 엔진: `orca-legacy`
- 프로필: `slicer.py`가 `output/cli_safe_profiles_orca_legacy/`에 CLI-safe profile을 자동 생성
- Floating/overhang 위험 감지 시: support를 자동으로 켠 process profile 생성
- 완료 검증: 생성된 `.3mf` 안에 `Metadata/plate_1.gcode`가 있는지 확인

Bambu Studio CLI를 직접 테스트하고 싶으면 아래처럼 실행할 수 있습니다. 다만 이 Mac에서는 Bambu Studio `02.06.00.51` CLI가 crash를 냈기 때문에 기본값으로 쓰지 않습니다.

```bash
python slicer.py output/test_processed.stl --engine bambu --output output/test_sliced_bambu.3mf
```

기본 프로필은 아래 조합입니다.

- Machine: `Bambu Lab X1 Carbon 0.4 nozzle`
- Process: `0.20mm Standard @BBL X1C`
- Filament: `Bambu PLA Basic @BBL X1C`

코드에서 기본 slicing 프로필을 바꾸려면 `slicer.py` 상단의 아래 값을 수정하세요.

```python
DEFAULT_MACHINE_NAME = "Bambu Lab X1 Carbon 0.4 nozzle"
DEFAULT_PROCESS_NAME = "0.20mm Standard @BBL X1C"
DEFAULT_FILAMENT_NAME = "Bambu PLA Basic @BBL X1C"
```

다른 프린터를 쓰면 프로필 이름을 바꿔서 실행하세요.

```bash
python slicer.py output/test_processed.stl \
  --machine-name "Bambu Lab P1S 0.4 nozzle" \
  --process-name "0.20mm Standard @BBL P1S" \
  --filament-name "Generic PLA @BBL P1S" \
  --output output/test_sliced.3mf
```

또는 정확한 JSON 프로필 경로를 직접 줄 수 있습니다. 세 가지를 모두 주는 것이 가장 안전합니다.

```bash
python slicer.py output/test_processed.stl \
  --machine-json "/path/to/machine.json" \
  --process-json "/path/to/process.json" \
  --filament-json "/path/to/filament.json" \
  --output output/test_sliced.3mf
```

support가 필요하면 기본 `orca-legacy` 엔진은 자동으로 support-enabled process profile을 만들어 사용합니다. 수동으로 켜려면:

```bash
python slicer.py output/test_processed.stl --enable-support --output output/test_sliced.3mf
```

## Report 확인

report는 두 가지 형식으로 저장됩니다.

- `.json`: 프로그램에서 다시 읽기 좋은 상세 데이터
- `.txt`: 사람이 바로 읽기 좋은 요약 리포트

터미널에서 확인할 수 있습니다.

```bash
cat reports/test_YYYYMMDD_HHMMSS_report.json
cat reports/test_YYYYMMDD_HHMMSS_report.txt
```

또는 VS Code, PyCharm, 메모장 같은 편집기로 열어도 됩니다.

report에는 다음 정보가 들어 있습니다.

- 입력 파일 경로
- 최종 파일 경로
- 사용된 repair 방식
- simplify 적용 여부
- floating regions 정리 결과
- support 필요 여부(`support_recommended`)
- 처리 전 분석 결과
- PyMeshLab 후 분석 결과
- MeshFix 후 분석 결과
- 최종 분석 결과
- slicing 실행 여부와 성공/실패 상태
- 3MF 내부 G-code 포함 여부(`contains_gcode`)
- 출력 가능한 상태로 보이는지 여부

## 설치 실패 시 참고

`open3d`, `pymeshlab`, `pymeshfix`는 운영체제와 Python 버전에 따라 설치가 실패할 수 있습니다.

문제가 생기면 먼저 Python 3.10 또는 3.11 가상환경에서 다시 시도해 보세요.

```bash
python --version
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

그래도 실패하면 패키지를 하나씩 설치해서 어느 패키지에서 실패하는지 확인하세요.

```bash
pip install open3d
pip install pymeshlab
pip install pymeshfix
pip install opencv-python-headless
```

## 현재 구현하지 않은 것

현재 MVP에서는 다음 기능을 구현하지 않았습니다.

- G-code 파일 직접 export
- Bambu 프린터 업로드
- 웹 업로드 화면
- 프론트엔드
- YOLO 데이터셋 라벨링/학습 자동화
- SAM 모델 학습

현재 slicing 출력은 Orca/Bambu CLI의 `--export-3mf`를 사용한 `.3mf`입니다. 이 `.3mf`는 Bambu Cloud 업로드에 넘길 파일로 사용할 수 있지만, 클라우드 업로드 API 호출 자체는 아직 구현하지 않았습니다. 프린터 업로드 자동화를 붙이려면 `slicer.py` 다음 단계에 업로드 모듈을 추가하면 됩니다.

관련 함수:

- `slice_with_orca_legacy_cli_safe(...)`
- `slice_with_orca_slicer(...)`
- `slice_with_bambu_studio(...)`

## 바로 실행하는 순서

```bash
cd bambu-stl-pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp /path/to/your/model.stl input/test.stl
python analyze_stl.py input/test.stl
python repair_stl.py input/test.stl
python process_stl.py input/test.stl
python process_scaled_stl.py input/test.stl --scale-ply input/foot_for_scale_2.ply
python process_video_pipeline.py input/foot_capture.mp4 --use-notion-team-code --assets-dir assets --reconstruction-ply input/unbounded_default_post.ply --scale-ply input/foot_for_scale_2.ply
```

영상에서 segmentation까지 새로 실행하려면 추가로:

```bash
pip install -r requirements-video.txt
```

Linux + NVIDIA CUDA 환경에서 COLMAP/2DGS까지 한 번에 실행하려면:

```bash
docker build -t 2dgs:cu118 docker/2dgs
python process_video_pipeline.py input/foot_capture.mp4 --use-notion-team-code --run-reconstruction --assets-dir assets --scene-name foot_scene --reconstruction-image-set foot
```
