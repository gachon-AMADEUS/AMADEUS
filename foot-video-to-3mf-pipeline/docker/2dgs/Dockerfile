FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:/opt/conda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_HOME="/usr/local/cuda"
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"

RUN apt-get update && apt-get install -y \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    coreutils \
    curl \
    ffmpeg \
    findutils \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libssl-dev \
    nano \
    ninja-build \
    pkg-config \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda create -n surfel_splatting python=3.8 pip -y

SHELL ["conda", "run", "--no-capture-output", "-n", "surfel_splatting", "/bin/bash", "-c"]

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install \
    torch==2.1.2+cu118 \
    torchvision==0.16.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

WORKDIR /app
RUN git clone --recursive https://github.com/hbb1/2d-gaussian-splatting.git /app/2d-gaussian-splatting
WORKDIR /app/2d-gaussian-splatting
RUN git submodule update --init --recursive

RUN python -m pip install \
    mediapy==1.1.2 \
    lpips==0.1.4 \
    scikit-image==0.21.0 \
    tqdm==4.66.2 \
    trimesh==4.3.2 \
    plyfile \
    opencv-python \
    open3d==0.18.0

RUN python -m pip install ./submodules/simple-knn --no-build-isolation
RUN python -m pip install ./submodules/diff-surfel-rasterization --no-build-isolation

RUN python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"
RUN python -c "import diff_surfel_rasterization, simple_knn; print('2DGS extensions OK')"

WORKDIR /app
RUN mkdir -p /app/dataset /app/output /app/scripts

RUN cat > /usr/local/bin/check_2dgs_dataset.sh <<'EOF' && chmod +x /usr/local/bin/check_2dgs_dataset.sh
#!/bin/bash
set -euo pipefail
: "${SCENE:?SCENE environment variable is required}"

DATASET_DIR="/app/dataset/${SCENE}"

echo "[INFO] checking dataset: ${DATASET_DIR}"
[ -d "${DATASET_DIR}" ] || { echo "[ERROR] dataset dir not found: ${DATASET_DIR}"; exit 1; }
[ -d "${DATASET_DIR}/images" ] || { echo "[ERROR] images dir not found: ${DATASET_DIR}/images"; exit 1; }
[ -d "${DATASET_DIR}/sparse/0" ] || { echo "[ERROR] sparse/0 dir not found: ${DATASET_DIR}/sparse/0"; exit 1; }
[ -f "${DATASET_DIR}/sparse/0/cameras.bin" ] || { echo "[ERROR] cameras.bin missing"; exit 1; }
[ -f "${DATASET_DIR}/sparse/0/images.bin" ] || { echo "[ERROR] images.bin missing"; exit 1; }
[ -f "${DATASET_DIR}/sparse/0/points3D.bin" ] || { echo "[ERROR] points3D.bin missing"; exit 1; }

IMG_COUNT=$(find "${DATASET_DIR}/images" -maxdepth 1 \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
if [ "${IMG_COUNT}" -eq 0 ]; then
  echo "[ERROR] no images found in ${DATASET_DIR}/images"
  exit 1
fi

echo "[OK] dataset structure looks valid"
echo "[OK] image count: ${IMG_COUNT}"
EOF

RUN cat > /usr/local/bin/train_2dgs.sh <<'EOF' && chmod +x /usr/local/bin/train_2dgs.sh
#!/bin/bash
set -euo pipefail
: "${SCENE:?SCENE environment variable is required}"

source /opt/conda/etc/profile.d/conda.sh
conda activate surfel_splatting
check_2dgs_dataset.sh

cd /app/2d-gaussian-splatting
python train.py \
  -s /app/dataset/${SCENE} \
  -m /app/output/${SCENE} \
  "${@}"
EOF

RUN cat > /usr/local/bin/extract_mesh_quick.sh <<'EOF' && chmod +x /usr/local/bin/extract_mesh_quick.sh
#!/bin/bash
set -euo pipefail
: "${SCENE:?SCENE environment variable is required}"

source /opt/conda/etc/profile.d/conda.sh
conda activate surfel_splatting
cd /app/2d-gaussian-splatting

MODEL_PATH="/app/output/${SCENE}"
SAVE_ROOT="${MODEL_PATH}/mesh_quick"
mkdir -p "${SAVE_ROOT}"

python render.py \
  -s /app/dataset/${SCENE} \
  -m ${MODEL_PATH} \
  --skip_train \
  --skip_test \
  --unbounded \
  --mesh_res 1024 \
  --depth_ratio 0 \
  --num_cluster 30

LATEST_DIR=$(find "${MODEL_PATH}/train" -maxdepth 1 -type d -name 'ours_*' | sort | tail -n 1)
[ -f "${LATEST_DIR}/fuse_unbounded.ply" ] && cp "${LATEST_DIR}/fuse_unbounded.ply" "${SAVE_ROOT}/unbounded_default.ply" || true
[ -f "${LATEST_DIR}/fuse_unbounded_post.ply" ] && cp "${LATEST_DIR}/fuse_unbounded_post.ply" "${SAVE_ROOT}/unbounded_default_post.ply" || true

echo "[DONE] quick mesh extraction saved under ${SAVE_ROOT}"
EOF

RUN cat > /usr/local/bin/extract_mesh_variants.sh <<'EOF' && chmod +x /usr/local/bin/extract_mesh_variants.sh
#!/bin/bash
set -euo pipefail
: "${SCENE:?SCENE environment variable is required}"

source /opt/conda/etc/profile.d/conda.sh
conda activate surfel_splatting
cd /app/2d-gaussian-splatting

DATASET_PATH="/app/dataset/${SCENE}"
MODEL_PATH="/app/output/${SCENE}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SAVE_ROOT="${MODEL_PATH}/mesh_sweeps/${TIMESTAMP}"
mkdir -p "${SAVE_ROOT}"

get_latest_train_dir() {
  find "${MODEL_PATH}/train" -maxdepth 1 -type d -name 'ours_*' | sort | tail -n 1
}

for depth_ratio in 0 1; do
  for mesh_res in 512 1024 1536; do
    for num_cluster in 10 30 50; do
      tag="unbounded_dr${depth_ratio}_res${mesh_res}_nc${num_cluster}"
      echo "[RUN] ${tag}"
      python render.py \
        -s "${DATASET_PATH}" \
        -m "${MODEL_PATH}" \
        --skip_train \
        --skip_test \
        --unbounded \
        --depth_ratio "${depth_ratio}" \
        --mesh_res "${mesh_res}" \
        --num_cluster "${num_cluster}"

      SRC_DIR="$(get_latest_train_dir || true)"
      [ -f "${SRC_DIR}/fuse_unbounded.ply" ] && cp "${SRC_DIR}/fuse_unbounded.ply" "${SAVE_ROOT}/${tag}.ply" || true
      [ -f "${SRC_DIR}/fuse_unbounded_post.ply" ] && cp "${SRC_DIR}/fuse_unbounded_post.ply" "${SAVE_ROOT}/${tag}_post.ply" || true
    done
  done
done

echo "[DONE] mesh sweep results saved under: ${SAVE_ROOT}"
EOF

RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc \
    && echo "conda activate surfel_splatting" >> /root/.bashrc

CMD ["/bin/bash"]
