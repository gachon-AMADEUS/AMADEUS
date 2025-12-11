# 1. Python 3.10이 기본 탑재된 Ubuntu 22.04 + CUDA 11.8 이미지 사용
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# 2. 기본 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV TORCH_CUDA_ARCH_LIST="8.6"

# 3. 시스템 패키지 (OpenCV 등 필수)
RUN apt-get update && apt-get install -y \
    git wget curl build-essential cmake ffmpeg colmap \
    libgl1-mesa-glx libglib2.0-0 unzip xvfb libxcb-xinerama0 \
    python3 python3-pip python3-dev \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# 4. PyTorch 설치 (2.0.1 + CUDA 11.8)
# 이 버전이 PyTorch3D와 가장 호환성이 좋습니다.
RUN pip install --upgrade pip setuptools wheel \
 && pip install \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118

# 5. ★ PyTorch3D 설치 (여기가 핵심) ★
# (1) 필수 의존성을 먼저 설치합니다. (이게 없으면 Wheel 설치가 실패함)
RUN pip install --no-cache-dir fvcore iopath xformers

# (2) "Python 3.10 + CUDA 11.8 + PyTorch 2.0.1" 전용 공식 설치 파일 주소 지정
# 컴파일 안 하고 다운로드만 하므로 빠르고 메모리 안 터집니다.
RUN pip install --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html

# 6. 나머지 라이브러리 설치
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# 7. 마무리
COPY . /app
WORKDIR /app
CMD ["/bin/bash"]