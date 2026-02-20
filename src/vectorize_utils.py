from pathlib import Path
import json
import numpy as np
from skimage import measure
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
import fiona
import rasterio
from rasterio.transform import from_origin


def save_raster_tif(path: str, arr: np.ndarray, meta: dict, crs_wkt: str = None):
    transform = from_origin(meta['x0'], meta['y0'] + meta['height'] * meta['cell_size'], meta['cell_size'], meta['cell_size'])
    profile = {
        'driver': 'GTiff',
        'height': arr.shape[0],
        'width': arr.shape[1],
        'count': 1,
        'dtype': str(arr.dtype),
        'transform': transform,
        'crs': crs_wkt,
    }
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr, 1)


def lawn_polygons_from_mask(mask: np.ndarray, meta: dict, min_area_m2: float = 1.0):
    contours = measure.find_contours(mask.astype(np.uint8), level=0.5)
    polys = []

    for cnt in contours:
        if len(cnt) < 4:
            continue
        coords = []
        for r, c in cnt:
            x = meta['x0'] + c * meta['cell_size']
            y = meta['y0'] + r * meta['cell_size']
            coords.append((float(x), float(y)))
        poly = Polygon(coords)
        if poly.is_valid and poly.area >= min_area_m2:
            polys.append(poly)

    if not polys:
        return []

    merged = unary_union(polys)
    if merged.geom_type == 'Polygon':
        geoms = [merged]
    else:
        geoms = [g for g in merged.geoms if g.area >= min_area_m2]

    return geoms


def save_geojson(path: str, polygons, crs_wkt: str = None):
    feats = []
    for i, p in enumerate(polygons):
        feats.append({
            'type': 'Feature',
            'properties': {'id': i + 1, 'class': 'lawn'},
            'geometry': mapping(p),
        })

    fc = {'type': 'FeatureCollection', 'features': feats}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(fc, f)
    if crs_wkt:
        prj_path = str(Path(path).with_suffix('.prj'))
        with open(prj_path, 'w', encoding='utf-8') as f:
            f.write(crs_wkt)


def save_shapefile(path: str, polygons, crs_wkt: str = None):
    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int', 'class': 'str:16'},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with fiona.open(
        path,
        mode='w',
        driver='ESRI Shapefile',
        schema=schema,
        crs_wkt=crs_wkt,
        encoding='UTF-8',
    ) as dst:
        for i, p in enumerate(polygons):
            dst.write(
                {
                    'geometry': mapping(p),
                    'properties': {'id': i + 1, 'class': 'lawn'},
                }
            )


def save_gpkg(path: str, polygons, crs_wkt: str = None, layer: str = 'lawn_polygons'):
    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int', 'class': 'str:16'},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with fiona.open(
        path,
        mode='w',
        driver='GPKG',
        schema=schema,
        crs_wkt=crs_wkt,
        encoding='UTF-8',
        layer=layer,
    ) as dst:
        for i, p in enumerate(polygons):
            dst.write(
                {
                    'geometry': mapping(p),
                    'properties': {'id': i + 1, 'class': 'lawn'},
                }
            )
