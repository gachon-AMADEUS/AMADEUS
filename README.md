## 🦶 AMADEUS: Customized Shoe Last Generation
The AMADEUS project is a 3D scanning and modeling pipeline that uses smartphone video/photos of a user's foot to create a precise, customized Shoe Last.

The ultimate goal of this project is to generate highly accurate 3D models, scale them to real-world dimensions (Real-world Scaling), and then use the models for 3D printing a physical shoe last

## 📊 Current Progress
| Pipeline | Status |
| --- | --- |
| Data preprocessing (YOLO + SAM) |✅| 
| 3D Reconstruction (COLMAP SfM) |✅| 
| 3DGS Training (Gaussian Splatting) |✅| 
| Mesh Extraction (SuGaR) |✅|
| Mesh Cleaning & Healing |✅|
|Resolution Mismatch Fix|🚧|
|Real-world Scaling (mm conversion)|⬜|
|Shoe Last Engineering (Modeling)|⬜|
|3D Printing & Validation|⬜|






 

## 📂 Overall Layout
```
AMADEUS/
├── data/
│   ├── raw_images/       # (Input) Original foot photos (360-degree capture)
│   └── masked_images/    # (Generated]) Background-removed data for training
├── models/               # AI model files (YOLOv11, SAM ViT-H)
├── src/
│   ├── preprocessing/    # Preprocessing scripts (Auto Segmentation)
│   ├── postprocessing/   # Post-processing scripts (Mesh Cutting, Healing)
│   └── utils/            # Utilities
├── colmap_work/          # (Work) COLMAP intermediate files (Sparse Model)
├── output/               # (Output) Final .obj Mesh file
├── Dockerfile            # Optimized Docker environment setup
├── requirements.txt      # Python dependencies
└── run_pipeline.sh       # Full pipeline execution script
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
#### 1. Docker Build
```bash
# run in Dockerfile path
docker build -t amadeus .
```

#### 2. Docker Run
```Powershell
docker run --gpus all -it --rm -v ${PWD}:/app amadeus
```
#### 3. Run Pipeline
```bash
# Grant Execution Permissions
chmod +x run_pipeline.sh

# Pipeline Operation with Virtual Display (xvfb)
xvfb-run -a ./run_pipeline.sh
```
## 🔄 Pipeline (Current)
#### 1. Preprocessing (Step 1)
This stage is the most crucial initial task, determining the accuracy of 3D reconstruction and the efficiency of training.
- Goal: To enhance the quality of the training data by isolating the foot region and removing surrounding background noise.
- Key Technologies:
  - YOLOv11: Detects the bounding box of the 'foot' region in the image.
  - SAM (Segment Anything Model): Precisely segments (Masking) the detected foot region at the pixel level.
- Operation: Each frame from raw_images is processed by YOLO and SAM, converted into a masked image with a blacked-out background, and saved in the masked_images folder.
- Output: data/masked_images (Set of images with the background removed)

#### 2. COLMAP SfM (Step 2)
This stage lays the foundation for 3D reconstruction by estimating the camera's pose and the foot's 3D spatial coordinates.
- Goal: To calculate the relative camera poses and internal parameters (Intrinsic) for all captured images and generate a Sparse Point Cloud of the foot.
- Key Technologies:
  - Feature Extraction: Extracts key feature points from the images.
  - Sequential Matching: Matching by comparing only adjacent frames (e.g., frame_001 and frame_002) according to the order in which images are shot
  - Bundle Adjustment: Optimizes the positions of all cameras and feature points.
- Operation:
  1. raw_images are used as input, as background information is needed for reliable pose estimation.
  2. After matching and adjustment, a sparse/0 model containing camera information and initial 3D point data is created.
- Output: colmap_work/sparse/0 (Camera files (.bin) and initial 3D Point Cloud)

#### 3. Undistortion & Alignment (Step 3)
This stage prepares the data for 3DGS training and improves the quality of the resulting model.
- Goal: To align the 3D model calculated by COLMAP with the images without distortion, and to orient the 3D model's coordinate system.
- Key Technologies:
  - Image Undistorter: Generates images with lens distortion removed based on the COLMAP model.
  - Auto Alignment: Uses the auto_align_colmap.py script to align the model's bottom surface horizontally (leveling) and adjust the axes.
- Operation: Based on the 3D model from Step 2, the masked images from Step 1 are corrected for distortion and placed into the sugar_ready folder as the training dataset.
- Output: colmap_work/sugar_ready (Camera and image files formatted for 3DGS training)

#### 4. 3D Gaussian Splatting Training (Step 4)
This is a critical stage that renders the foot surface in high fidelity and prepares the foundational data for the next meshing step.
- Goal: To place and train numerous Gaussian Balls in 3D space, enabling real-time rendering of the foot's 3D view that appears realistic from any angle.
- Key Technology: 3D Gaussian Splatting (3DGS) algorithm.
- Operation: The sugar_ready dataset is fed into the system for thousands of iterations of training. Upon completion, the final Point Cloud containing density and color information in 3D space is saved.
- Output: output/vanilla_3dgs/point_cloud/ (High-density 3D Gaussian Point Cloud in .ply format)

#### 5. Meshing & Healing (3D Printable Output) (Step 5)
This stage transforms and refines the 3DGS results into a solid form that is ready for actual 3D printing.
- Goal: To convert the high-quality 3DGS result into a Watertight (hole-free) Mesh suitable for printing.
- Key Technologies:
  - SuGaR (Surface-Aligned Gaussian Splatting): Extracts a precise surface (Mesh) from the 3DGS Point Cloud.
  - Cleaning: Uses the keep_largest_cluster algorithm to remove noise and artifacts floating in the air, retaining only the main foot body.
  - Healing: Essential for 3D printing, this process involves closing small holes in the mesh (Close Holes) and smoothing the surface for a cleaner finish.
- Output: output/final_foot_mesh.obj (Refined mesh ready for 3D printing)

## ⚠️ Troubleshooting (Known Issues)
#### 1. Resolution Mismatch Error
- Symptom: Error occurs in Step 3 due to coordinate system conflict.
- Cause: COLMAP uses the full resolution of the original image, but the input masked image may have been resized during Step 1.
- Solution (To-Do): Modify segment_foot.py to ensure the masked image is saved with the same resolution as the original image.

#### 2. Auto Align Failed
- Symptom: auto_align_colmap.py throws a RANSAC points not enough error.
- Cause: Insufficient density of the initial point cloud to reliably detect the ground plane.
- Mitigation: The pipeline is configured to ignore this error and proceed (|| true), but the mesh may be slightly tilted.
