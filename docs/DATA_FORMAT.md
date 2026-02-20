# Data Format

Detailed training data preparation guide:
- `docs/TRAINING_DATA_PREP.md`

## Supported Input Types
- LAS/LAZ
- NPZ
- CSV

## Required Fields
Training:
- `x`, `y`, `z`, `intensity`, `label`

Inference:
- `x`, `y`, `z`, `intensity`
- `label` optional

## Label Semantics in Source Data
Original annotation convention:
- `1`: lawn / vegetated ground
- `2`: roads
- `3`: curbs
- `4`: trees/shrubs
- `5`: anthropogenic objects

Current stage-2 model target:
- lawn vs non-lawn ground (binary on CSF-ground points)

## CRS
- LAS/LAZ CRS metadata is used when present.
- Outputs preserve project coordinates and can be overlaid in GIS.
- For OSM overlay, ensure proper CRS assignment/reprojection in QGIS.
