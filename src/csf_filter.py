import numpy as np


def csf_ground_mask(points: dict, cfg: dict) -> np.ndarray:
    """
    Returns boolean mask over points: True for ground points.
    Requires python package `CSF`.
    """
    try:
        import CSF  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "CSF package is not installed. Install it first (pip install cloth-simulation-filter)."
        ) from e

    xyz = np.stack([points["x"], points["y"], points["z"]], axis=1).astype(np.float64)
    csf = CSF.CSF()
    params = cfg["csf"]

    csf.params.bSloopSmooth = bool(params["bSloopSmooth"])
    csf.params.cloth_resolution = float(params["cloth_resolution"])
    csf.params.rigidness = int(params["rigidness"])
    csf.params.class_threshold = float(params["class_threshold"])
    csf.params.time_step = float(params["time_step"])
    csf.params.interations = int(params["interations"])

    csf.setPointCloud(xyz)
    ground = CSF.VecInt()
    non_ground = CSF.VecInt()
    csf.do_filtering(ground, non_ground)

    mask = np.zeros(xyz.shape[0], dtype=bool)
    if len(ground) > 0:
        idx = np.asarray(list(ground), dtype=np.int64)
        mask[idx] = True
    return mask

