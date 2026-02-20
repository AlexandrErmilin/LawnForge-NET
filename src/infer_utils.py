from pathlib import Path
import numpy as np
import torch

from src.raster import make_tiles, normalize_features, points_to_cell_indices


def _predict_tiled(model, features: np.ndarray, tile_px: int, overlap_px: int, device, num_classes: int):
    model.eval()
    c, h, w = features.shape

    logits_sum = np.zeros((num_classes, h, w), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)

    # soft blending on overlaps
    win = np.outer(np.hanning(tile_px), np.hanning(tile_px)).astype(np.float32)
    win = np.maximum(win, 1e-3)

    tiles = make_tiles(features, np.zeros((h, w), dtype=np.int64), np.zeros((h, w), dtype=np.int64), np.ones((h, w), dtype=np.uint8), tile_px, overlap_px)

    with torch.no_grad():
        for f, _, _, _, y, x in tiles:
            inp = torch.from_numpy(f[None]).to(device)
            out = model(inp)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]

            y2 = min(y + tile_px, h)
            x2 = min(x + tile_px, w)
            ph = y2 - y
            pw = x2 - x

            logits_sum[:, y:y2, x:x2] += probs[:, :ph, :pw] * win[:ph, :pw][None]
            weight_sum[y:y2, x:x2] += win[:ph, :pw]

    logits_sum /= np.maximum(weight_sum[None], 1e-6)
    return logits_sum


def infer_scene(model, points: dict, feature_stack: np.ndarray, meta: dict, cfg: dict):
    cell_size = cfg['raster']['cell_size']
    tile_px = int(round(cfg['raster']['tile_size_m'] / cell_size))
    overlap_px = int(round(cfg['raster']['tile_overlap_m'] / cell_size))

    feature_stack = normalize_features(feature_stack)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    p2 = _predict_tiled(model, feature_stack, tile_px, overlap_px, device, num_classes=2)
    stage2_pred = np.argmax(p2, axis=0)  # 0..1 mapped to labels 1,2
    map_back = np.array([1, 2], dtype=np.int16)
    final_raster = map_back[stage2_pred]

    iy, ix = points_to_cell_indices(points, meta)
    point_labels = final_raster[iy, ix].astype(np.int16)
    return final_raster.astype(np.int16), point_labels


def load_checkpoint(model, path: str):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    return model


def ensure_out_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
