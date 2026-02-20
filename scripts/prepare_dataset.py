import argparse
from pathlib import Path
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, ensure_dirs, seed_everything
from src.io_utils import list_scene_files, read_scene
from src.raster import rasterize_scene, normalize_features, make_tiles


def save_tiles(scene_files, out_dir: str, cfg: dict):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cell_size = cfg['raster']['cell_size']
    tile_px = int(round(cfg['raster']['tile_size_m'] / cell_size))
    overlap_px = int(round(cfg['raster']['tile_overlap_m'] / cell_size))

    idx = 0
    for sf in scene_files:
        print(f'Preparing {sf} ...')
        points = read_scene(str(sf))
        feats, _, s2, valid, _ = rasterize_scene(points, cell_size=cell_size)
        feats = normalize_features(feats)
        tiles = make_tiles(feats, np.zeros_like(s2), s2, valid, tile_px, overlap_px)

        for f, y1, y2, vm, y, x in tiles:
            out = Path(out_dir) / f'tile_{idx:08d}.npz'
            np.savez_compressed(out, features=f, stage1=y1, stage2=y2, valid=vm, tile_y=y, tile_x=x)
            idx += 1
    print(f'Total tiles saved to {out_dir}: {idx}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    seed_everything(cfg['seed'])

    tr = list_scene_files(cfg['paths']['raw_train'])
    va = list_scene_files(cfg['paths']['raw_val'])

    if not tr:
        raise RuntimeError(f'No training scenes in {cfg["paths"]["raw_train"]}')
    if not va:
        raise RuntimeError(f'No validation scenes in {cfg["paths"]["raw_val"]}')

    save_tiles(tr, cfg['paths']['prepared_train'], cfg)
    save_tiles(va, cfg['paths']['prepared_val'], cfg)


if __name__ == '__main__':
    main()
