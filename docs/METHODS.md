# Methods and Technical Design

## Problem Definition
Given dense outdoor point clouds, extract:
- per-point lawn predictions
- GIS polygon layer for lawn areas

## Pipeline Overview
1. Ground filtering with CSF.
2. 2.5D feature rasterization from ground points.
3. U-Net semantic segmentation (`lawn` vs `non-lawn` ground).
4. Projection back to points.
5. Polygonization and GIS export.

## Why CSF + 2.5D
- CSF suppresses trees/buildings before segmentation.
- 2.5D segmentation is computationally lighter than full 3D learning.
- Good balance of quality, speed, and deployment complexity.

## Feature Engineering (Raster Channels)
Computed per XY grid cell:
- `count`
- `z_min`, `z_max`, `z_mean`, `z_std`
- `intensity_mean`, `intensity_std`
- `ndhm = z_max - z_min`

These channels are normalized and passed as multi-channel image input to U-Net.

## Model
- Architecture: `UNet2D`
- Input: `C x H x W`
- Output classes:
- class 0 -> lawn
- class 1 -> non-lawn ground

## Training
- Dataset generated as overlapping tiles.
- Loss: cross-entropy with ignore index for invalid cells.
- Validation metric: Dice score.
- Best checkpoint saved by validation Dice.

## Inference
- Runs tiled prediction with overlap blending.
- Reconstructs full scene raster labels.
- Projects labels back to original points.

## Vectorization
- Lawn mask -> contours -> polygons.
- Small polygons filtered by area threshold.
- Export to GPKG (and optional other formats).

## CRS Handling
- CRS is read from source LAS/LAZ where possible.
- Output LAZ and GIS layers preserve or receive CRS metadata/WKT.

## Current Limitations
- Current target is lawn-oriented segmentation only.
- CSF errors can propagate to segmentation stage.
- For highly complex vegetation taxonomy, additional classes/models are needed.
