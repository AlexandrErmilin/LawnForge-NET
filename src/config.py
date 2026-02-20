from pathlib import Path
import random
import numpy as np
import torch
import yaml


def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dirs(cfg: dict):
    for key in ['prepared_train', 'prepared_val', 'models']:
        p = cfg['paths'][key]
        Path(p).mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
