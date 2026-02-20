import argparse
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, seed_everything
from src.models.unet2d import UNet2D
from src.train_utils import build_loaders, train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--weights-out', default=None, help='Path to output checkpoint .pt')
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg['seed'])

    train_dl, val_dl = build_loaders(
        cfg['paths']['prepared_train'],
        cfg['paths']['prepared_val'],
        stage=2,
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['training']['num_workers'],
    )

    in_ch = len(cfg['features']['channels'])
    model = UNet2D(in_channels=in_ch, num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    out_ckpt = args.weights_out or f"{cfg['paths']['models']}/stage2_lawn_vs_nonlawn.pt"
    print(f'[train] checkpoint output: {out_ckpt}')

    train_model(
        model,
        train_dl,
        val_dl,
        device=device,
        epochs=cfg['training']['epochs_stage2'],
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay'],
        amp=bool(cfg['training']['amp'] and torch.cuda.is_available()),
        out_ckpt=out_ckpt,
        num_classes=2,
    )


if __name__ == '__main__':
    main()
