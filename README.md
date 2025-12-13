## 🦶 AMADEUS: Customized Shoe Last Generation
The AMADEUS project is a 3D scanning and modeling pipeline that uses smartphone video/photos of a user's foot to create a precise, customized Shoe Last.

The ultimate goal of this project is to generate highly accurate 3D models, scale them to real-world dimensions (Real-world Scaling), and then use the models for 3D printing a physical shoe last

## 📊 Current Progress
| Pipeline | Status | Note |
| --- | --- | --- |
| Data preprocessing (YOLO + SAM) |✅|Auto-masking foot region|
| 3D Reconstruction (COLMAP SfM) |✅|Sparse point cloud generation|
| 3DGS Training (Gaussian Splatting) |✅|sh_degree=0, 7k iters|
| Mesh Extraction (SuGaR) |✅|Pipeline Integration Complete|
| Mesh Cleaning & Healing |✅|Automated Cutting & Hole Filling|
|Real-world Scaling (mm conversion)|⬜|Next Step|
|Shoe Last Engineering (Modeling)|⬜|To-do|
|3D Printing & Validation|⬜|To-do|






 

## 📂 Overall Layout
```
AMADEUS/
├── data/
│   ├── raw_images/       # (Input) Original foot photos
│   └── masked_images/    # (Generated) Background-removed data
├── models/               # YOLOv11, SAM ViT-H models
├── src/
│   ├── preprocessing/    # Segmentation scripts
│   └── postprocessing/   # Mesh cutting & Healing scripts
├── SuGaR/                # (Submodule) Patched SuGaR implementation
├── gaussian-splatting/   # (Submodule) Patched Vanilla 3DGS
├── colmap_work/          # (Work) Intermediate COLMAP files
├── output/               # (Output) Final .obj Mesh & Checkpoints
├── Dockerfile            # Optimized Environment
└── run_pipeline.sh       # ★ Main Execution Script (Full Pipeline)
```

## ⚙️ Prerequisites 
The use of a Docker environment is mandatory to avoid complex library dependency issues.

- OS: Windows 10/11 (WSL2 required) or Linux
Hardware: Hardware: NVIDIA GPU (RTX 4060 8GB or higher recommended)
- Software:
  - NVIDIA GPU Driver (Latest)
  - Docker Desktop (In Windows, Activate WSL2 Backend)
- Data Input:
  - photos/frames captured from a 360-degree perspective.


## 🚀 Install & Start
This repository **includes pre-patched source codes** for `SuGaR` and `3DGS`.
You do **NOT** need to download submodules or fix the code manually.

#### 1. Clone Repository
Just clone this repository. It contains all the necessary custom fixes.

```bash
# Clone the repository (No --recursive needed)
git clone [https://github.com/gachon-AMADEUS/Jjajangmyeon.git](https://github.com/gachon-AMADEUS/Jjajangmyeon.git) AMADEUS

# Move to project directory
cd AMADEUS
```
#### 2. Build Docker Image
```bash
# Build the image named 'amadeus'
docker build -t amadeus .
```
#### 3. Run Container
```bash
# Run with GPU support
docker run --gpus all -it --rm -v ${PWD}:/app amadeus
```
#### 4. Run Pipeline
```bash
# 1. Grant execution permission
chmod +x run_pipeline.sh

# 2. Run the full pipeline
xvfb-run -a ./run_pipeline.sh
```

## 🔄 Pipeline (Current)
#### 1. Preprocessing (Step 1)
This stage is the most crucial initial task, determining the accuracy of 3D reconstruction and the efficiency of training.
- **Goal**: To enhance the quality of the training data by isolating the foot region and removing surrounding background noise.
- **Key Technologies**:
  - YOLOv11: Detects the bounding box of the 'foot' region in the image.
  - SAM (Segment Anything Model): Precisely segments (Masking) the detected foot region at the pixel level.
- **Operation**: Each frame from raw_images is processed by YOLO and SAM, converted into a masked image with a blacked-out background, and saved in the masked_images folder.
- **Output**: `data/masked_images` (Set of images with the background removed)

#### 2. COLMAP SfM (Step 2)
This stage lays the foundation for 3D reconstruction by estimating the camera's pose and the foot's 3D spatial coordinates.
- **Goal**: To calculate the relative camera poses and internal parameters (Intrinsic) for all captured images and generate a Sparse Point Cloud of the foot.
- **Key Technologies**:
  - Feature Extraction: Extracts key feature points from the images.
  - Sequential Matching: Matching by comparing only adjacent frames (e.g., frame_001 and frame_002) according to the order in which images are shot
  - Bundle Adjustment: Optimizes the positions of all cameras and feature points.
- **Operation**:
  1. raw_images are used as input, as background information is needed for reliable pose estimation.
  2. After matching and adjustment, a sparse/0 model containing camera information and initial 3D point data is created.
- **Output**: `colmap_work/sparse/0` (Camera files (.bin) and initial 3D Point Cloud)

#### 3. Undistortion & Alignment (Step 3)
This stage prepares the data for 3DGS training and improves the quality of the resulting model.
- **Goal**: To align the 3D model calculated by COLMAP with the images without distortion, and to orient the 3D model's coordinate system.
- **Key Technologies**:
  - Image Undistorter: Generates images with lens distortion removed based on the COLMAP model.
  - Auto Alignment: Uses the `auto_align_colmap.py` script to align the model's bottom surface horizontally (leveling) and adjust the axes.
- **Operation**: Based on the 3D model from Step 2, the masked images from Step 1 are corrected for distortion and placed into the sugar_ready folder as the training dataset.
- **Output**: `colmap_work/sugar_ready` (Camera and image files formatted for 3DGS training)

#### 4. 3D Gaussian Splatting Training (Step 4)
This is a critical stage that renders the foot surface in high fidelity and prepares the foundational data for the next meshing step.
- **Goal**: To place and train numerous Gaussian Balls in 3D space, enabling real-time rendering of the foot's 3D view that appears realistic from any angle.
- **Key Technologies**: 3D Gaussian Splatting (3DGS) algorithm.
- **Operation**: The sugar_ready dataset is fed into the system for thousands of iterations of training. Upon completion, the final Point Cloud containing density and color information in 3D space is saved.
- **Output**: `output/vanilla_3dgs/point_cloud/` (High-density 3D Gaussian Point Cloud in .ply format)

#### 5. Point Cloud Cleaning (Step 5)
Refines the Step 4 result to create a clean "reference shape" for meshing.
- **Goal:** Remove floaters and background artifacts.
- **Key Technologies:**
  - Statistical Outlier Removal: Removes noise points.
  - Largest Cluster Extraction: Keeps only the main foot cluster, discarding disconnected debris.
- **Output:** `output/vanilla_3dgs/largest.ply`

#### 6. SuGaR Mesh Extraction (Step 6)
Extracts a polygonal mesh from the Gaussian Splatting scene.
- **Goal:** Convert the volumetric Gaussian representation into a surface mesh.
- **Key Technologies:**
  - SuGaR: Regularizes Gaussians to align with the surface and extracts a high-poly mesh.
  - Double Extension Patch: Automatically fixes file extension errors (`.jpg.jpg`) during loading.
- **Output:** `SuGaR/output/.../sugarfine_*.obj`

#### 7. Final Mesh Post-processing (Step 7)
The final engineering step to make the mesh 3D-printable.
- **Goal:** Create a clean, watertight mesh.
- **Key Technologies:**
  - Mesh Cutting: Uses the clean Point Cloud (from Step 5) as a 3D mask to trim excess parts of the SuGaR mesh.
  - Healing: Fills holes and performs Laplacian smoothing for a high-quality finish.
- **Output:** `output/final_foot_mesh.obj`

## ⚠️ Troubleshooting (Known Issues)
#### 1. Installation & Environment
- `ModuleNotFoundError: No module named 'diff_gaussian_rasterization'`
  - **Cause**: The submodule was not compiled correctly during Docker build, or pip install -e . failed to link the C++ binaries.
  - **Solution**:
    1. Ensure you used the provided `Dockerfile` (It uses `--no-build-isolation` flag)
    2. If running manually, go to `gaussian-splatting/submodules/diff-gaussian-rasterization` and run:
       ```bash
       pip install . --no-build-isolation
        ```
- `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"`
  - **Cause**: `open3d` or `pymeshlab` tries to open a window on a server/container without a monitor.
  - **Solution**: Always use `xvfb-run` when executing the pipeline.
    ``` bash
    xvfb-run -a ./run_pipeline.sh
    ```
#### 2. Runtime Errors (SuGaR + 3DGS)
- `AssertionError: At the beginning of SuGaR training, sh_degree is 0...`
   -  **Cause**: Vanilla 3DGS was trained with `sh_degree=0` (for speed), but default SuGaR expects `sh_degree=3`.
  - **Fix (Applied)**: We patched `SuGaR/sugar_scene/gs_model.py` to force `GaussianModel(0)`. Do not revert this code.

- `ValueError: too many values to unpack (expected 2)`
  - **Cause**: Mismatch between the latest `diff-gaussian-rasterization` (returns 3+ values) and old SuGaR code (expects 2).
  -  **Fix (Applied)**: Code updated to `rendered_image, radii, _ = rasterizer(...)`.

- `RuntimeError: min(): Expected reduction dim to be specified...`
  - **Cause**: Occurs when `sh_degree=0`. The `_sh_coordinates_rest` tensor is empty, but the logger tries to calculate its min/max stats.
  - **Fix (Applied)**: Added a safety check if `tensor.numel() > 0`: in `coarse_density_and_dn_consistency.py`.

## 📚 References & Acknowledgments

#### 1. Core Algorithms
- **3D Gaussian Splatting (Vanilla 3DGS)**
    - *Paper:* [3D Gaussian Splatting for Real-Time Radiance Field Rendering (SIGGRAPH 2023)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
    - *Repository:* [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
    - *License:* Gaussian-Splatting License

- **SuGaR (Surface-Aligned Gaussian Splatting)**
    - *Paper:* [SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering (CVPR 2024)](https://antwo.github.io/sugar/)
    - *Repository:* [Anttwo/SuGaR](https://github.com/Anttwo/SuGaR)
    -  *License:* Apache License 2.0

- **COLMAP (Structure-from-Motion)**
    - *Project:* [COLMAP](https://colmap.github.io/)
    - *Authors:* Johannes L. Schönberger et al.

#### 2. Preprocessing Models
- **YOLOv11 (Object Detection)**
    - *Repository:* [Ultralytics](https://github.com/ultralytics/ultralytics)
    - *License:* AGPL-3.0

- **Segment Anything Model (SAM)**
    - *Paper:* [Segment Anything (ICCV 2023)](https://segment-anything.com/)
    - *Repository:* [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
    - *License:* Apache License 2.0

#### 3. Libraries
- **PyTorch3D / Open3D / PyMeshLab** used for mesh processing and visualization.