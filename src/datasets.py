import torch
from torch.utils.data import Dataset
import numpy as np


class TileDataset(Dataset):
    def __init__(self, items, stage: int):
        self.items = items
        self.stage = stage
        # stage2 mapping (ground-only): 1->lawn(class 0), 2->non-lawn(class 1)
        self.stage2_map = {0: 255, 1: 0, 2: 1}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p = self.items[idx]
        arr = np.load(p)
        x = arr['features'].astype(np.float32)
        valid = arr['valid'].astype(np.uint8)

        if self.stage == 1:
            y = arr['stage1'].astype(np.int64)
            y[valid == 0] = 255
        else:
            y_raw = arr['stage2'].astype(np.int64)
            y = np.full_like(y_raw, 255, dtype=np.int64)
            for src, dst in self.stage2_map.items():
                y[y_raw == src] = dst
            y[valid == 0] = 255

        return torch.from_numpy(x), torch.from_numpy(y)
