# Teammate Run Guide

팀원이 실행할 파일은 하나입니다.

```bash
python pipeline.py
```

## 준비

```text
assets/best.pt
assets/sam_vit_h_4b8939.pth
input/<촬영영상>.mp4
```

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-video.txt
```

Windows PowerShell에서는:

```powershell
.venv\Scripts\Activate.ps1
```

## 실행

```bash
python pipeline.py
```

결과:

```text
output/<video_name>_scaled_sliced.3mf
reports/<video_name>_pipeline_YYYYMMDD_HHMMSS.json
reports/<video_name>_pipeline_YYYYMMDD_HHMMSS.txt
```

## 필요한 외부 프로그램

- Docker
- NVIDIA Container Toolkit
- NVIDIA CUDA GPU
- COLMAP
- OrcaSlicer 또는 Bambu Studio CLI

위 환경이 준비되어 있고 `assets/`, `input/` 파일만 맞으면 `pipeline.py` 하나로 끝까지 실행됩니다.

