from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import TileDataset
from src.models.unet2d import dice_score


def collect_tile_files(folder: str):
    return sorted(Path(folder).glob('*.npz'))


def build_loaders(train_dir: str, val_dir: str, stage: int, batch_size: int, num_workers: int):
    tr_files = collect_tile_files(train_dir)
    va_files = collect_tile_files(val_dir)

    if len(tr_files) == 0:
        raise RuntimeError(f'No training tiles found in {train_dir}')
    if len(va_files) == 0:
        raise RuntimeError(f'No validation tiles found in {val_dir}')

    train_ds = TileDataset(tr_files, stage=stage)
    val_ds = TileDataset(va_files, stage=stage)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl


def run_epoch(model, loader, optimizer, scaler, device, num_classes: int, train: bool, amp: bool):
    ce = nn.CrossEntropyLoss(ignore_index=255)
    model.train(train)

    total_loss = 0.0
    total_dice = 0.0
    n = 0

    for x, y in tqdm(loader, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with torch.amp.autocast(device_type=device_type, enabled=amp):
            logits = model(x)
            loss = ce(logits, y)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            d = dice_score(logits, y, num_classes=num_classes).item()

        bs = x.shape[0]
        total_loss += loss.item() * bs
        total_dice += d * bs
        n += bs

    return total_loss / max(1, n), total_dice / max(1, n)


def train_model(model, train_dl, val_dl, device, epochs, lr, weight_decay, amp, out_ckpt: str, num_classes: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    best_val = -1.0
    out_path = Path(out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_dice = run_epoch(model, train_dl, optimizer, scaler, device, num_classes, train=True, amp=amp)
        va_loss, va_dice = run_epoch(model, val_dl, optimizer, scaler, device, num_classes, train=False, amp=amp)
        print(f'Epoch {epoch:03d} | train loss {tr_loss:.4f} dice {tr_dice:.4f} | val loss {va_loss:.4f} dice {va_dice:.4f}')

        if va_dice > best_val:
            best_val = va_dice
            torch.save({'model_state': model.state_dict(), 'val_dice': va_dice, 'epoch': epoch}, out_path)
            print(f'Saved best checkpoint: {out_path} (val dice {va_dice:.4f})')

    print(f'Best val dice: {best_val:.4f}')
