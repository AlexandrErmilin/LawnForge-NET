# System Requirements

## Target Runtime (Recommended)
- OS: Windows 10/11 x64
- Python: 3.11.x
- NVIDIA Driver: compatible with CUDA 11.8 runtime
- CUDA target for PyTorch: cu118

## GPU
- Minimum: 4 GB VRAM (small experiments)
- Recommended: 6-12 GB VRAM
- For larger scenes and faster training: 8 GB+ VRAM

## CPU / RAM
- CPU: 6+ logical cores recommended
- RAM:
- Minimum: 16 GB
- Recommended: 32 GB+

## Disk
- 10+ GB free for environment and dependencies
- Additional free space for prepared tiles and outputs

## Python Packages
Installed from `requirements.txt` plus explicit CUDA build in `run.bat`:
- `torch`, `torchvision`, `torchaudio` from `https://download.pytorch.org/whl/cu118`
- `cloth-simulation-filter` (CSF)
- `laspy`, `rasterio`, `fiona`, `shapely`, `numpy`, `scipy`, `pandas`, etc.

## Verification Commands
```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Expected:
- Torch version with `+cu118`
- `True` for CUDA availability
- device count >= 1

## Notes
- On Windows WDDM, some VRAM is always occupied by desktop applications.
- `nvidia-smi` may show baseline memory usage even with no active training process.
