# 1. Base Image: SuGaR와 PyTorch 2.x 호환성이 가장 좋은 CUDA 11.8 Devel 사용
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# 2. 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# [중요] GPU 아키텍처 설정 (빌드 속도 최적화 및 오류 방지)
# RTX 30시리즈=8.6, RTX 40시리즈=8.9, V100=7.0 등. 
# 범용적으로 쓰려면 아래와 같이 설정 (시간이 좀 더 걸림)
ENV TORCH_CUDA_ARCH_LIST="8.6"

# 3. 시스템 패키지 및 COLMAP 설치
# Ubuntu 22.04의 colmap은 버전이 적당하여 SuGaR와 호환됩니다.
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    curl \
    wget \
    unzip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    colmap \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Miniconda 설치
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# [핵심 해결책]
# 1. 설정 명령어가 아닌 파일 직접 생성으로 'defaults' 채널을 원천 차단합니다.
# 2. 채널 우선순위를 strict로 설정하여 conda-forge만 바라보게 합니다.
RUN echo "channels:\n  - conda-forge\nchannel_priority: strict" > /root/.condarc

# 5. Conda 환경 생성
# --override-channels: 시스템 기본 채널 설정을 무시 (ToS 에러 방지 핵심)
# -c conda-forge: 오직 conda-forge에서만 패키지를 가져옴
RUN conda create -n amadeus python=3.9 -c conda-forge --override-channels -y

# [중요] 이후 모든 RUN 명령어는 'amadeus' 콘다 환경 내부에서 실행됨
SHELL ["conda", "run", "-n", "amadeus", "/bin/bash", "-c"]

# PyTorch 및 핵심 라이브러리 설치 (CUDA 11.8 매칭)
# 1. -c conda-forge: PyTorch가 필요로 하는 기본 라이브러리를 conda-forge에서 찾도록 추가
# 2. --override-channels: defaults 채널 접근을 완벽하게 차단 (ToS 에러 해결)
RUN conda install -y pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 \
    -c pytorch -c nvidia -c conda-forge \
    --override-channels

# 6. 공통 의존성 및 프로젝트 라이브러리 설치 (pip)
# plyfile, tqdm (3dgs) + opencv, matplotlib (공통) + ultralytics (YOLO) + segment-anything (SAM) 등
# 작업 경로를 메인 디렉토리(/app)로 설정
WORKDIR /app

# [중요] 호스트(내 컴퓨터)에 있는 requirements.txt를 컨테이너(/app)로 복사
COPY requirements.txt .

# requirements.txt 설치
# (여기에 yolo, sam, numpy<2 등)
RUN pip install -r requirements.txt

# 7. PyTorch3D 설치 (SuGaR 필수 의존성)
# 소스 빌드는 오래 걸리므로, CUDA 11.8 / PyTorch 2.1.2에 맞는 pre-built wheel 사용 권장 (없으면 소스빌드)
# 아래는 소스 빌드 방식 (시간은 걸리지만 확실함)
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"


# -----------------------------------------------------------------------------
# 8. 3D Gaussian Splatting 설치 및 패치
# -----------------------------------------------------------------------------
WORKDIR /app

# 1) 레포지토리 클론
RUN git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git

# 2) [사용자 요청 반영] 소스코드 패치: <cstdint> 헤더 추가
# sed를 이용해 rasterizer_impl.h 파일의 1번째 줄(1i)에 include 구문을 삽입합니다.
WORKDIR /app/gaussian-splatting
RUN sed -i '1i #include <cstdint>' submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h

# 3) [사용자 요청 반영] Submodules 설치 (옵션 변경)
# --no-build-isolation: 현재 환경의 PyTorch를 사용하여 빌드
# -e: Editable 모드 (컨테이너 내에서 코드를 수정하며 실험할 때 유용)

# (1) Diff-Gaussian-Rasterization
WORKDIR /app/gaussian-splatting/submodules/diff-gaussian-rasterization
RUN pip install . --no-build-isolation

# (2) Simple-KNN
WORKDIR /app/gaussian-splatting/submodules/simple-knn
RUN pip install . --no-build-isolation

# (3) Fused-SSIM (일부 3DGS 변형이나 SuGaR 등에서 요구할 경우)
# *주의: 기본 3DGS 레포에는 submodule 폴더에 fused-ssim이 없을 수 있습니다.
# 만약 에러가 난다면, 별도로 clone 하거나 해당 라인을 주석 처리해야 합니다.
# 여기서는 폴더가 존재한다고 가정하고 실행합니다.
# RUN git clone https://github.com/... (필요시 클론 추가)
WORKDIR /app/gaussian-splatting/submodules/fused-ssim
RUN pip install . --no-build-isolation

# 9. SuGaR 설치 (옵션)
# SuGaR도 비슷한 rasterizer를 쓰지만 별도 설치가 필요할 수 있음
WORKDIR /app
RUN git clone --recursive https://github.com/Anttwo/SuGaR.git /app/SuGaR
# SuGaR의 요구사항 설치

# (필요시 SuGaR 내부의 submodule도 pip install . 수행)

# 10. 최종 작업 디렉토리 설정
# WORKDIR /app

# 컨테이너 실행 시 기본적으로 conda 환경 활성화
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
RUN echo "conda activate amadeus" >> ~/.bashrc
CMD ["/bin/bash"]