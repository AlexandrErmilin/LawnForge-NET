from pathlib import Path
import numpy as np
import pandas as pd
import laspy


REQUIRED_COLUMNS = ['x', 'y', 'z', 'intensity', 'label']


def _decode_npz_scalar(v):
    if isinstance(v, np.ndarray):
        if v.shape == ():
            return v.item()
        if v.size == 1:
            return v.reshape(-1)[0].item()
    return v


def _validate_arrays(data: dict, require_label: bool = True):
    required = ['x', 'y', 'z', 'intensity'] + (['label'] if require_label else [])
    for k in required:
        if k not in data:
            raise ValueError(f'Missing required field: {k}')
    n = len(data['x'])
    for k in required:
        if len(data[k]) != n:
            raise ValueError(f'Inconsistent array lengths for {k}')


def read_scene(path: str, require_label: bool = True):
    p = Path(path)
    suffix = p.suffix.lower()
    crs_wkt = None
    if suffix == '.npz':
        arr = np.load(p)
        data = {k: arr[k] for k in arr.files}
        if 'crs_wkt' in data:
            crs_wkt = str(_decode_npz_scalar(data['crs_wkt']))
    elif suffix == '.csv':
        df = pd.read_csv(p)
        cols = ['x', 'y', 'z', 'intensity'] + (['label'] if require_label else [])
        data = {k: df[k].to_numpy() for k in cols if k in df.columns}
    elif suffix in ['.las', '.laz']:
        las = laspy.read(p)
        data = {
            'x': np.asarray(las.x),
            'y': np.asarray(las.y),
            'z': np.asarray(las.z),
            'intensity': np.asarray(las.intensity, dtype=np.float32),
        }
        label = None
        if 'label' in las.point_format.dimension_names:
            label = np.asarray(las['label'])
        elif hasattr(las, 'classification'):
            label = np.asarray(las.classification)
        if label is not None:
            data['label'] = label
        elif require_label:
            raise ValueError('No label field found in LAS/LAZ (expected label or classification).')
        try:
            crs = las.header.parse_crs()
            if crs is not None:
                crs_wkt = crs.to_wkt()
        except Exception:
            crs_wkt = None
    else:
        raise ValueError(f'Unsupported scene format: {p}')

    _validate_arrays(data, require_label=require_label)

    x = data['x'].astype(np.float32)
    y = data['y'].astype(np.float32)
    z = data['z'].astype(np.float32)
    intensity = data['intensity'].astype(np.float32)
    out = {
        'x': x,
        'y': y,
        'z': z,
        'intensity': intensity,
    }
    if 'label' in data:
        out['label'] = data['label'].astype(np.int16)
    out['_crs_wkt'] = crs_wkt
    out['_source_path'] = str(p)
    return out


def list_scene_files(folder: str):
    p = Path(folder)
    files = []
    for ext in ['*.npz', '*.csv', '*.las', '*.laz']:
        files.extend(p.glob(ext))
    return sorted(files)


def save_point_predictions(path: str, points: dict, pred_label: np.ndarray, pred_lawn: np.ndarray):
    np.savez_compressed(
        path,
        x=points['x'],
        y=points['y'],
        z=points['z'],
        intensity=points['intensity'],
        pred_label=pred_label.astype(np.int16),
        pred_lawn=pred_lawn.astype(np.uint8),
    )


def save_laz_predictions(path: str, points: dict, pred_lawn: np.ndarray, pred_class: np.ndarray):
    header = laspy.LasHeader(point_format=3, version='1.2')
    header.scales = np.array([0.001, 0.001, 0.001], dtype=np.float64)
    header.offsets = np.array(
        [float(points['x'].min()), float(points['y'].min()), float(points['z'].min())],
        dtype=np.float64,
    )
    las = laspy.LasData(header)
    las.x = points['x'].astype(np.float64)
    las.y = points['y'].astype(np.float64)
    las.z = points['z'].astype(np.float64)

    inten = np.asarray(points['intensity'])
    inten = np.clip(inten, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    las.intensity = inten

    las.add_extra_dim(laspy.ExtraBytesParams(name='pred_lawn', type=np.uint8))
    las.add_extra_dim(laspy.ExtraBytesParams(name='pred_class', type=np.uint8))
    las.pred_lawn = pred_lawn.astype(np.uint8)
    las.pred_class = pred_class.astype(np.uint8)
    crs_wkt = points.get('_crs_wkt', None)
    if crs_wkt:
        try:
            from pyproj import CRS  # type: ignore
            las.header.add_crs(CRS.from_wkt(crs_wkt))
        except Exception:
            pass
    las.write(path)


def save_laz_predictions_from_source(source_las_path: str, out_las_path: str, pred_lawn: np.ndarray, pred_class: np.ndarray):
    """
    Preserve all original LAS/LAZ dimensions and metadata, then append/update prediction fields.
    """
    las = laspy.read(source_las_path)
    n = len(las.x)
    if len(pred_lawn) != n or len(pred_class) != n:
        raise ValueError("Prediction length does not match source LAS/LAZ point count.")

    dim_names = set(las.point_format.dimension_names)
    if 'pred_lawn' not in dim_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name='pred_lawn', type=np.uint8))
    if 'pred_class' not in dim_names:
        las.add_extra_dim(laspy.ExtraBytesParams(name='pred_class', type=np.uint8))

    las.pred_lawn = pred_lawn.astype(np.uint8)
    las.pred_class = pred_class.astype(np.uint8)
    las.write(out_las_path)
