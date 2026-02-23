import argparse
from pathlib import Path
import numpy as np
import sys
import time

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
    scene_total = len(scene_files)
    for sf in scene_files:
        scene_start = time.time()
        print(f'Preparing {sf} ...', flush=True)
        points = read_scene(str(sf))
        n_points = len(points['x'])
        print(f'  points: {n_points}', flush=True)

        t0 = time.time()
        feats, _, s2, valid, _ = rasterize_scene(
            points,
            cell_size=cell_size,
            progress_cb=lambda msg: print(f'  [raster] {msg}', flush=True),
        )
        print(f'  rasterize: {time.time() - t0:.1f}s', flush=True)

        t0 = time.time()
        feats = normalize_features(feats)
        print(f'  normalize: {time.time() - t0:.1f}s', flush=True)

        t0 = time.time()
        tiles = make_tiles(feats, np.zeros_like(s2), s2, valid, tile_px, overlap_px)
        print(f'  tile generation: {time.time() - t0:.1f}s | tiles: {len(tiles)}', flush=True)

        scene_saved = 0
        save_t0 = time.time()
        for f, y1, y2, vm, y, x in tiles:
            out = Path(out_dir) / f'tile_{idx:08d}.npz'
            np.savez_compressed(out, features=f, stage1=y1, stage2=y2, valid=vm, tile_y=y, tile_x=x)
            idx += 1
            scene_saved += 1
            if scene_saved % 100 == 0:
                elapsed = time.time() - save_t0
                rate = scene_saved / max(elapsed, 1e-6)
                print(f'  saved tiles: {scene_saved}/{len(tiles)} ({rate:.1f} tiles/s)', flush=True)

        scene_elapsed = time.time() - scene_start
        print(f'Finished {Path(sf).name}: {scene_saved} tiles in {scene_elapsed:.1f}s', flush=True)

    print(f'Total tiles saved to {out_dir}: {idx}', flush=True)


def resolve_scene_inputs(cli_scenes, folder_from_cfg: str, split_name: str):
    if cli_scenes:
        files = [Path(p) for p in cli_scenes]
        missing = [str(p) for p in files if not p.exists()]
        if missing:
            raise FileNotFoundError(f'Missing {split_name} scenes: {missing}')
        print(f'[{split_name}] using explicit file list ({len(files)} files)', flush=True)
        return files

    files = list_scene_files(folder_from_cfg)
    print(f'[{split_name}] using folder scan: {folder_from_cfg} ({len(files)} files)', flush=True)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--train-scenes', nargs='+', default=None, help='Explicit training scene files (overrides paths.raw_train scan)')
    ap.add_argument('--val-scenes', nargs='+', default=None, help='Explicit validation scene files (overrides paths.raw_val scan)')
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    seed_everything(cfg['seed'])

    tr = resolve_scene_inputs(args.train_scenes, cfg['paths']['raw_train'], 'train')
    va = resolve_scene_inputs(args.val_scenes, cfg['paths']['raw_val'], 'val')

    if not tr:
        raise RuntimeError(f'No training scenes in {cfg["paths"]["raw_train"]}')
    if not va:
        raise RuntimeError(f'No validation scenes in {cfg["paths"]["raw_val"]}')

    save_tiles(tr, cfg['paths']['prepared_train'], cfg)
    save_tiles(va, cfg['paths']['prepared_val'], cfg)


if __name__ == '__main__':
    main()
