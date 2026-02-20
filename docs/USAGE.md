# Usage Guide

## Data Layout
```text
data/
  raw/
    train/
    val/
```

## Training Workflow
1. Prepare tiles:
```powershell
python scripts/prepare_dataset.py --config configs/default.yaml
```
2. Train:
```powershell
python scripts/train_stage2.py --config configs/default.yaml --weights-out artifacts/models/stage2_lawn_vs_nonlawn.pt
```

## Inference Workflow
Single file:
```powershell
python scripts/infer_scene.py --config configs/default.yaml --scene data/raw/val/scene_001.laz --out-dir outputs/scene_001 --weights artifacts/models/stage2_lawn_vs_nonlawn.pt --minimal-outputs
```

Batch mode:
- Use GUI (`python scripts/gui_app.py`)
- Add multiple files
- Set output root
- Start batch inference

## Output Structure
For each input scene in batch mode, a dedicated output folder is created:
```text
outputs/
  scene_name/
    point_predictions.laz
    lawn_polygons.gpkg
```

## GUI Notes
- Settings are editable directly in GUI.
- Settings panel opens separately.
- RU/EN language switch is available.
- Status checks include Torch, CUDA, and checkpoint readability.
