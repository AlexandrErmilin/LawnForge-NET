import argparse
from pathlib import Path
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.io_utils import read_scene, save_point_predictions, save_laz_predictions, save_laz_predictions_from_source
from src.csf_filter import csf_ground_mask
from src.models.unet2d import UNet2D
from src.raster import rasterize_scene
from src.infer_utils import infer_scene, load_checkpoint, ensure_out_dir
from src.vectorize_utils import save_raster_tif, lawn_polygons_from_mask, save_geojson, save_shapefile, save_gpkg


def subset_points(points: dict, mask: np.ndarray):
    return {k: v[mask] for k, v in points.items() if isinstance(v, np.ndarray) and len(v) == len(mask)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--weights', default=None, help='Path to trained .pt checkpoint')
    ap.add_argument('--minimal-outputs', action='store_true', help='Write only GPKG and LAZ outputs')
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_out_dir(args.out_dir)
    print(f'[infer] scene: {args.scene}')
    print(f'[infer] out: {args.out_dir}')

    points_all = read_scene(args.scene, require_label=False)
    crs_wkt = points_all.get('_crs_wkt', None)
    point_ground_mask = csf_ground_mask(points_all, cfg)
    ground_points = subset_points(points_all, point_ground_mask)
    if len(ground_points['x']) == 0:
        raise RuntimeError('CSF returned zero ground points. Check CSF parameters.')

    features, _, _, _, meta = rasterize_scene(ground_points, cell_size=cfg['raster']['cell_size'])

    in_ch = len(cfg['features']['channels'])
    m2 = UNet2D(in_channels=in_ch, num_classes=2)

    weights_path = args.weights or f"{cfg['paths']['models']}/stage2_lawn_vs_nonlawn.pt"
    print(f'[infer] weights: {weights_path}')
    m2 = load_checkpoint(m2, weights_path)

    class_raster, ground_point_labels = infer_scene(m2, ground_points, features, meta, cfg)
    point_labels = np.full(points_all['x'].shape[0], 5, dtype=np.int16)
    point_labels[point_ground_mask] = ground_point_labels

    out_dir = Path(args.out_dir)
    lawn_mask = (class_raster == 1).astype(np.uint8)

    polys = lawn_polygons_from_mask(
        lawn_mask,
        meta,
        min_area_m2=cfg['inference']['min_lawn_polygon_area_m2'],
    )
    save_gpkg(str(out_dir / 'lawn_polygons.gpkg'), polys, crs_wkt=crs_wkt)

    pred_lawn = (point_labels == 1).astype(np.uint8)
    scene_suffix = Path(args.scene).suffix.lower()
    if scene_suffix in ['.las', '.laz']:
        save_laz_predictions_from_source(
            args.scene,
            str(out_dir / 'point_predictions.laz'),
            pred_lawn,
            point_labels.astype(np.uint8),
        )
    else:
        save_laz_predictions(str(out_dir / 'point_predictions.laz'), points_all, pred_lawn, point_labels.astype(np.uint8))

    if not args.minimal_outputs:
        np.savez_compressed(out_dir / 'csf_ground_mask.npz', point_ground_mask=point_ground_mask.astype(np.uint8))
        save_raster_tif(str(out_dir / 'class_raster.tif'), class_raster.astype(np.int16), meta, crs_wkt=crs_wkt)
        save_raster_tif(str(out_dir / 'lawn_mask.tif'), lawn_mask, meta, crs_wkt=crs_wkt)
        save_geojson(str(out_dir / 'lawn_polygons.geojson'), polys, crs_wkt=crs_wkt)
        save_shapefile(str(out_dir / 'lawn_polygons.shp'), polys, crs_wkt=crs_wkt)
        save_point_predictions(str(out_dir / 'point_labels.npz'), points_all, point_labels, pred_lawn)

    print(f'Done. Outputs in: {out_dir}')


if __name__ == '__main__':
    main()
