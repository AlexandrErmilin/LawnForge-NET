# Training Data Preparation

## Goal
Prepare scenes so `scripts/prepare_dataset.py` can generate valid tiles for model training.

## Required Fields for Training
Each training scene must contain:
- `x`
- `y`
- `z`
- `intensity`
- `label`

All arrays/columns must have the same length.

## Supported Input Formats
- `.las` / `.laz`
- `.npz`
- `.csv`

## Label Rules
Source annotation convention:
- `1` -> lawn / vegetated ground
- `2` -> roads
- `3` -> curbs
- `4` -> trees / shrubs
- `5` -> anthropogenic objects

Current stage-2 target is binary on ground:
- `lawn` = source label `1`
- `non-lawn ground` = source labels `2` and `3`

`4` and `5` are still important for consistent scene annotation but are not positive target classes for stage-2.

## Format-Specific Requirements
### LAS/LAZ
Use one of:
- extra dimension named `label`
- standard LAS `classification`

If both are missing, training scene loading fails.

### NPZ
Must include arrays:
- `x`, `y`, `z`, `intensity`, `label`

Example:
```python
np.savez_compressed(
    "scene_001.npz",
    x=x, y=y, z=z,
    intensity=intensity,
    label=label
)
```

### CSV
Must include columns:
- `x,y,z,intensity,label`

## Folder Layout
Expected by default config:
```text
data/
  raw/
    train/
      scene_001.laz
      scene_002.npz
    val/
      scene_101.laz
```

## Pre-Flight Validation Checklist
Before running preparation:
1. Labels are in expected range `1..5`.
2. No NaN/Inf in `x,y,z,intensity,label`.
3. `intensity` is numeric and not constant zero.
4. Scene CRS is present in LAS/LAZ header when possible.
5. Train and val folders are both non-empty.

## Quick Validation Script (optional)
```powershell
python - << 'PY'
from pathlib import Path
import numpy as np
from src.io_utils import list_scene_files, read_scene

for split in ["data/raw/train", "data/raw/val"]:
    files = list_scene_files(split)
    print(split, "files:", len(files))
    for p in files:
        s = read_scene(str(p), require_label=True)
        lb = s["label"]
        ok = np.isin(lb, [1,2,3,4,5]).all()
        print(p.name, "n=", len(lb), "labels_ok=", ok)
PY
```

## Common Errors
- `Missing required field: label`:
  add `label` column/array or LAS classification.
- `Inconsistent array lengths`:
  one or more fields have different point count.
- Empty tile sets after preparation:
  check scene extents, config `cell_size`, and input validity.

## Run Preparation
```powershell
python scripts/prepare_dataset.py --config configs/default.yaml
```

After success:
- tiles are written to `data/prepared/train` and `data/prepared/val`.
