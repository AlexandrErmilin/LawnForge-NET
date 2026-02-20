import numpy as np


RASTER_FEATURES = [
    'count', 'z_min', 'z_max', 'z_mean', 'z_std', 'intensity_mean', 'intensity_std', 'ndhm'
]


def _empty_grid(h: int, w: int):
    return {
        'count': np.zeros((h, w), dtype=np.float32),
        'z_min': np.full((h, w), np.inf, dtype=np.float32),
        'z_max': np.full((h, w), -np.inf, dtype=np.float32),
        'z_sum': np.zeros((h, w), dtype=np.float32),
        'z_sq_sum': np.zeros((h, w), dtype=np.float32),
        'i_sum': np.zeros((h, w), dtype=np.float32),
        'i_sq_sum': np.zeros((h, w), dtype=np.float32),
    }


def rasterize_scene(points: dict, cell_size: float):
    x, y, z, intensity = points['x'], points['y'], points['z'], points['intensity']
    label = points.get('label', None)

    x0, y0 = float(x.min()), float(y.min())
    x1, y1 = float(x.max()), float(y.max())

    w = int(np.floor((x1 - x0) / cell_size)) + 1
    h = int(np.floor((y1 - y0) / cell_size)) + 1

    ix = np.floor((x - x0) / cell_size).astype(np.int32)
    iy = np.floor((y - y0) / cell_size).astype(np.int32)

    grid = _empty_grid(h, w)

    for k in range(len(x)):
        r = iy[k]
        c = ix[k]
        zz = z[k]
        ii = intensity[k]

        grid['count'][r, c] += 1.0
        if zz < grid['z_min'][r, c]:
            grid['z_min'][r, c] = zz
        if zz > grid['z_max'][r, c]:
            grid['z_max'][r, c] = zz
        grid['z_sum'][r, c] += zz
        grid['z_sq_sum'][r, c] += zz * zz
        grid['i_sum'][r, c] += ii
        grid['i_sq_sum'][r, c] += ii * ii

    count = grid['count']
    valid = count > 0
    z_mean = np.zeros_like(count)
    i_mean = np.zeros_like(count)
    z_std = np.zeros_like(count)
    i_std = np.zeros_like(count)

    z_mean[valid] = grid['z_sum'][valid] / count[valid]
    i_mean[valid] = grid['i_sum'][valid] / count[valid]

    z_var = np.zeros_like(count)
    i_var = np.zeros_like(count)
    z_var[valid] = grid['z_sq_sum'][valid] / count[valid] - z_mean[valid] ** 2
    i_var[valid] = grid['i_sq_sum'][valid] / count[valid] - i_mean[valid] ** 2

    z_std[valid] = np.sqrt(np.clip(z_var[valid], 0, None))
    i_std[valid] = np.sqrt(np.clip(i_var[valid], 0, None))

    z_min = grid['z_min']
    z_max = grid['z_max']
    z_min[~valid] = 0.0
    z_max[~valid] = 0.0
    ndhm = z_max - z_min

    feature_stack = np.stack([
        count,
        z_min,
        z_max,
        z_mean,
        z_std,
        i_mean,
        i_std,
        ndhm,
    ], axis=0)

    # Cell-level labels by majority vote.
    # stage1 is legacy (not used in CSF-first pipeline).
    # stage2 target: lawn(1) vs non-lawn(2) only for ground cells; 0 for ignored.
    votes = np.zeros((h, w, 6), dtype=np.int32)
    if label is not None:
        for k in range(len(label)):
            r = iy[k]
            c = ix[k]
            lb = int(label[k])
            if 1 <= lb <= 5:
                votes[r, c, lb] += 1

    argmax_label = np.argmax(votes[:, :, 1:6], axis=2) + 1
    has_vote = votes[:, :, 1:6].sum(axis=2) > 0

    stage1 = np.zeros((h, w), dtype=np.int64)
    stage1[has_vote & np.isin(argmax_label, [1, 2, 3])] = 1

    stage2 = np.zeros((h, w), dtype=np.int64)
    stage2[has_vote & (argmax_label == 1)] = 1
    stage2[has_vote & np.isin(argmax_label, [2, 3])] = 2

    meta = {
        'x0': x0,
        'y0': y0,
        'cell_size': cell_size,
        'width': w,
        'height': h,
    }

    return feature_stack.astype(np.float32), stage1, stage2, valid.astype(np.uint8), meta


def normalize_features(features: np.ndarray, eps: float = 1e-6):
    # features: [C,H,W]
    out = features.copy()
    for c in range(out.shape[0]):
        ch = out[c]
        mean = float(ch.mean())
        std = float(ch.std())
        out[c] = (ch - mean) / (std + eps)
    return out


def tile_indices(h: int, w: int, tile_px: int, overlap_px: int):
    step = max(1, tile_px - overlap_px)
    ys = list(range(0, max(1, h - tile_px + 1), step))
    xs = list(range(0, max(1, w - tile_px + 1), step))

    if len(ys) == 0 or ys[-1] != max(0, h - tile_px):
        ys.append(max(0, h - tile_px))
    if len(xs) == 0 or xs[-1] != max(0, w - tile_px):
        xs.append(max(0, w - tile_px))

    for y in ys:
        for x in xs:
            yield y, x


def make_tiles(features: np.ndarray, stage1: np.ndarray, stage2: np.ndarray, valid: np.ndarray, tile_px: int, overlap_px: int):
    c, h, w = features.shape
    tiles = []
    for y, x in tile_indices(h, w, tile_px, overlap_px):
        y2, x2 = y + tile_px, x + tile_px
        f = features[:, y:y2, x:x2]
        s1 = stage1[y:y2, x:x2]
        s2 = stage2[y:y2, x:x2]
        vm = valid[y:y2, x:x2]

        # Pad borders if needed
        pad_h = tile_px - f.shape[1]
        pad_w = tile_px - f.shape[2]
        if pad_h > 0 or pad_w > 0:
            f = np.pad(f, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
            s1 = np.pad(s1, ((0, pad_h), (0, pad_w)), mode='constant')
            s2 = np.pad(s2, ((0, pad_h), (0, pad_w)), mode='constant')
            vm = np.pad(vm, ((0, pad_h), (0, pad_w)), mode='constant')

        tiles.append((f.astype(np.float32), s1.astype(np.int64), s2.astype(np.int64), vm.astype(np.uint8), y, x))
    return tiles


def points_to_cell_indices(points: dict, meta: dict):
    x = points['x']
    y = points['y']
    ix = np.floor((x - meta['x0']) / meta['cell_size']).astype(np.int64)
    iy = np.floor((y - meta['y0']) / meta['cell_size']).astype(np.int64)
    ix = np.clip(ix, 0, meta['width'] - 1)
    iy = np.clip(iy, 0, meta['height'] - 1)
    return iy, ix
