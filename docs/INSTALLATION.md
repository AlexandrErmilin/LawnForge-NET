# Installation Guide

## Option A: One-click Windows setup (recommended)
Use `run.bat` from repository root:
```bat
run.bat
```

What it does:
1. Creates temporary venv `.venv_temp_gui` (Python 3.11 preferred).
2. Installs dependencies from `requirements.txt`.
3. Installs PyTorch CUDA 11.8 build.
4. Launches GUI.

## Option B: Manual setup

### 1) Create and activate environment
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3) Verify GPU runtime
```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Common Issues
- `cuda_available=False`:
- install CUDA-compatible NVIDIA driver
- ensure cu118 PyTorch wheel is installed
- `ModuleNotFoundError` for project imports:
- run scripts from repo root
- use provided script wrappers in `scripts/`
- LAS/LAZ CRS not recognized:
- check source file CRS metadata
- assign CRS in GIS manually if metadata is absent
