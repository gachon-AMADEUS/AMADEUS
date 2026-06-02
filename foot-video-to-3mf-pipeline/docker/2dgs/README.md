# 2DGS CUDA Docker

This Dockerfile builds the CUDA 11.8 2D Gaussian Splatting environment used by
the final pipeline.

Build on a Linux machine with Docker and NVIDIA Container Toolkit:

```bash
docker build -t 2dgs:cu118 docker/2dgs
```

The main Python pipeline runs the container with:

```bash
docker run --gpus all --rm \
  -v <dataset_root>:/app/dataset \
  -v <output_root>:/app/output \
  -e TWO_DGS_MESH_RES_LIST="512 384 256 192" \
  2dgs:cu118 \
  bash -lc "export SCENE=foot_scene && train_2dgs.sh --depth_ratio 0 && extract_mesh_quick.sh"
```

For low-memory GPUs, use lower mesh extraction candidates:

```bash
python pipeline.py --two-dgs-mesh-res-list "256 192 128" --max-reconstruction-frames 60
```

Expected dataset layout:

```text
dataset/
  foot_scene/
    images/
      frame_0000.jpg
      frame_0001.jpg
    sparse/
      0/
        cameras.bin
        images.bin
        points3D.bin
```

Expected output:

```text
output/
  foot_scene/
    mesh_quick/
      unbounded_default_post.ply
```
