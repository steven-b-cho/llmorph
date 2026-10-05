import torch
from config_data import config_data

class DeviceManager:
    def __init__(self):
        self.use_gpu = config_data.get("use_gpu_for_local_models", False)

        self.device = self._select_device()
        self.backend = self._detect_backend()

    def _select_device(self):
        
        # Slower but allows it to run on any device and avoids needing CUDA/ROCm/XPU setup.
        if not self.use_gpu:
            print("Using CPU for local models")
            return torch.device("cpu")
        
        # NVIDIA CUDA and AMD ROCm, requires CUDA/ROCm build of PyTorch to be installed.
        elif torch.cuda.is_available():
            print(f"Using GPU for local models: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        
        # Intel XPU, requires XPU build of PyTorch to be installed.
        elif torch.xpu.is_available():
            print(f"Using GPU for local models: {torch.xpu.get_device_name(0)}")
            return torch.device("xpu")
        
        # Apple Silicon
        elif torch.backends.mps.is_available():
            print("Using GPU for local models: Apple MPS")
            return torch.device("mps")
        
        else:
            print("GPU requested but unavailable, falling back to CPU for local models")
            print("Make sure you have the correct PyTorch build installed and your GPU driver is up to date")
            print("If you do not wish to use GPU, disable \"use_gpu_for_local_models\" in the config")
            return torch.device("cpu")

    def _detect_backend(self):
        if self.device.type != "cuda":
            return None

        name = torch.cuda.get_device_name(0)

        if "AMD" in name or "Radeon" in name:
            return "ROCm"
        return "CUDA"

    def info(self):
        return {
            "device": str(self.device),
            "backend": self.backend
        }