# LawnForge-NET: Lawn Segmentation From Dense Point Clouds

Production-oriented pipeline for extracting lawn polygons and per-point lawn labels from dense outdoor point clouds.

## Project Goals
- Segment lawn areas robustly in dense street scans.
- Support very large scenes (millions of points).
- Export GIS-ready vector outputs and point cloud predictions with preserved CRS.
- Provide a GUI workflow for non-code operation.

## Core Approach
The current pipeline is a hybrid method:
1. `CSF` (Cloth Simulation Filter) removes non-ground points.
2. A `2.5D U-Net` performs `lawn vs non-lawn` segmentation on ground-only data.
3. Predictions are projected back to points and exported as:
- `point_predictions.laz`
- `lawn_polygons.gpkg`

Why this approach:
- Better robustness against trees/buildings than plain raster segmentation.
- Lower memory footprint than full 3D sparse models.
- Strong GIS compatibility.

## Repository Layout
- `configs/default.yaml`: default pipeline config.
- `scripts/prepare_dataset.py`: tile generation for training.
- `scripts/train_stage2.py`: stage-2 model training.
- `scripts/infer_scene.py`: single-scene inference.
- `scripts/gui_app.py`: full GUI launcher.
- `src/csf_filter.py`: CSF ground filtering wrapper.
- `src/raster.py`: 2.5D feature rasterization and tiling.
- `src/models/unet2d.py`: 2D U-Net segmentation model.
- `src/train_utils.py`: training loop and metrics.
- `src/infer_utils.py`: tiled inference and point projection.
- `src/io_utils.py`: scene reading/writing, LAZ export.
- `src/vectorize_utils.py`: polygonization and GPKG export.
- `run.bat`: one-click setup and GUI start for Windows.

## System Requirements
See full details in `docs/SYSTEM_REQUIREMENTS.md`.

Baseline target:
- Python `3.11`
- CUDA `11.8` (PyTorch cu118 build)
- NVIDIA GPU with 6 GB+ VRAM recommended
- Windows 10/11 x64

## Quick Start (Windows)
1. Place data in expected folders:
- `data/raw/train`
- `data/raw/val`

2. Launch:
```bat
run.bat
```

`run.bat` will:
- create/update temporary venv (`.venv_temp_gui`)
- install dependencies
- install PyTorch CUDA 11.8 build
- start GUI

## Manual CLI Workflow
### 1) Prepare dataset
```powershell
python scripts/prepare_dataset.py --config configs/default.yaml
```

### 2) Train model
```powershell
python scripts/train_stage2.py --config configs/default.yaml --weights-out artifacts/models/stage2_lawn_vs_nonlawn.pt
```

### 3) Inference (minimal outputs)
```powershell
python scripts/infer_scene.py --config configs/default.yaml --scene data/raw/val/scene_001.laz --out-dir outputs/scene_001 --weights artifacts/models/stage2_lawn_vs_nonlawn.pt --minimal-outputs
```

## Outputs
Minimal mode (`--minimal-outputs`):
- `point_predictions.laz`
- `lawn_polygons.gpkg`

Full mode:
- `point_predictions.laz`
- `lawn_polygons.gpkg`
- `lawn_polygons.geojson`
- `lawn_polygons.shp`
- `class_raster.tif`
- `lawn_mask.tif`
- `csf_ground_mask.npz`
- `point_labels.npz`

## Data Requirements
Training scenes require:
- `x, y, z, intensity, label`

Inference scenes require:
- `x, y, z, intensity`
- `label/classification` optional

Supported formats:
- `.laz/.las`
- `.npz`
- `.csv`

## Coordinate Reference System (CRS)
- CRS is read from input LAS/LAZ when available.
- LAZ prediction output preserves original dimensions and CRS metadata.
- GPKG polygons are written with CRS WKT.

## GUI Capabilities
- Full settings editing in UI (self-contained, no manual YAML required).
- Batch inference over multiple files.
- Per-file output subfolders.
- Weights file selection for train/inference.
- Detailed live logs.
- Status checks:
- `Torch`
- `CUDA`
- `Model checkpoint load`
- RU/EN language switch.

## Documentation Index
- `docs/SYSTEM_REQUIREMENTS.md`
- `docs/METHODS.md`
- `docs/INSTALLATION.md`
- `docs/USAGE.md`
- `docs/DATA_FORMAT.md`
- docs/TRAINING_DATA_PREP.md

## Current Scope
Current model predicts lawn-oriented classes only (lawn vs non-lawn ground after CSF filtering). This repository is structured for extension to more vegetation classes.


