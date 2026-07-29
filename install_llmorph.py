# Tested on:
# Linux Mint, CPU
# Windows 11, NVIDIA, Pascal
# Windows 11, AMD, RDNA3
import sys
import platform
import subprocess
import re
from pathlib import Path

REQUIRED_PYTHON_VERSION = "3.14.5"
# If PyTorch version is upgraded in future, test if wheels for all GPU's and OS'es are available, especially ROCm on Windows
PYTORCH_VERSION = "2.12.0"

def check_python_version():
    version = sys.version_info
    major, minor, patch = map(int, REQUIRED_PYTHON_VERSION.split("."))

    if(version.major == major and version.minor == minor and version.micro == patch):
        step_complete("Correct Python version detected.")
    else:
        print("=" * 60)
        print("Required Python version not installed!")
        print(f"Please install Python {REQUIRED_PYTHON_VERSION}, then restart the installer.")
        print("=" * 60)
        input("Press Enter to exit...")
        sys.exit(1)

def check_architecture():
    platform_used = platform.machine().lower()
    is_arm = (platform_used.startswith("arm") or platform_used.startswith("aarch"))

    if(is_arm):
        print("=" * 60)
        print(f"Unsupported CPU architecture detected: {platform_used}")
        print()
        print("LLMorph does not support ARM based systems, because a required")
        print("dependency (intel-openmp) is only available for x86_64.")
        print()
        print("Please use an Intel/AMD based x86_64 system.")
        print("=" * 60)

        input("Press Enter to exit...")
        sys.exit(1)

    step_complete(f"Supported architecture detected: {platform_used}")

def venv_python():
    if(IS_WINDOWS):
        return VENV_DIR / "Scripts" / "python.exe"
    else:
        return VENV_DIR / "bin" / "python"

def create_venv():
    python_path = venv_python()

    if not python_path.exists():
        print("Creating virtual environment...")
        
        VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
        return True

    return False

def pip_install(*args):
    subprocess.check_call([str(venv_python()), "-m", "pip", "install", *args])

def get_nvidia_specs():
    compute_capability = float(subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]).decode().strip())

    if 7.5 > compute_capability >= 5.0: # Maxwell and Pascal
        cuda_version = "cu126"
    elif compute_capability >= 7.5: # Turing and up
        cuda_version = "cu130"
    else:
        return None

    gpu_name = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).decode().strip()
    global GPU_NAME 
    GPU_NAME = gpu_name
    return cuda_version

# More tedious to get an AMD GPU ID without having ROCm installed
# In general, getting the correct PyTorch build installed for an AMD GPU, especially on Windows, is a mess
def get_amd_specs():
    if(IS_WINDOWS):  
        device_info = subprocess.check_output(["powershell", "-Command", "Get-WmiObject Win32_VideoController | Select-Object Name, PNPDeviceID"], text=True)

        device_id = re.search(r"VEN_1002&DEV_([0-9A-Fa-f]{4})", device_info)
        if not device_id:
            return None
        gpu_id = int(device_id.group(1), 16)
    
        raw_gpu_name = re.search(r"(AMD\s*.+?)\s*PCI", device_info)
        gpu_name = raw_gpu_name.group(1).strip() if raw_gpu_name else "AMD GPU"
    else:
        device_info = subprocess.check_output(["lspci", "-nn"], text=True)

        device_id = re.search(r"\[1002:([0-9a-fA-F]{4})\]", device_info)
        if not device_id:
            return None

        gpu_id = int(device_id.group(1), 16)
        raw_gpu_name = re.search(r"\[(Radeon[^\]]+)\]", device_info)
        gpu_name = "AMD " + raw_gpu_name.group(1) if raw_gpu_name else "AMD GPU"

    global IS_AMD_GPU
    # PyTorch 2.12.0 ROCm wheels only officially exist for ROCm 7.14+, which only supports RDNA 3+
    # Custom builds for RDNA 2 might work but can't support that here, so LLMorph only supports RDNA3+
    if gpu_id >= 29696: # >= 0x7400, RDNA3+
        rocm_version = "rocm7.14.0"
    else:
        IS_AMD_GPU = False
        return None

    # AMD distributes Pytorch wheels through their own repo, 
    # so if AMD GPU is present set it here to later use different link
    IS_AMD_GPU = True

    global GPU_NAME 
    GPU_NAME = gpu_name
    return rocm_version

# Get the matching gfx number to retroactively install GPU specfic ROCm packages, which without there is only a non functional (for execution) base ROCm build installed
# By far the nicest way of doing it, as you cannot detect it before installing the ROCm base and avoids hardcoded lookup tables
def get_amd_gfx():
    result = subprocess.check_output([str(venv_python()), "-c", "import torch; print(torch.cuda.get_device_properties(0).gcnArchName)"], text=True)
    return result.strip()

# Same tedious process as for AMD
def get_intel_specs():
    if(IS_WINDOWS):
        device_info = subprocess.check_output(["powershell", "-Command", "Get-WmiObject Win32_VideoController | Select-Object Name, PNPDeviceID"], text=True)

        device_id = re.search(r"VEN_8086&DEV_([0-9A-Fa-f]{4})", device_info)
        if not device_id:
            return None
        gpu_id = int(device_id.group(1), 16)
    
        raw_gpu_name = re.search(r"(Intel\s*.+?)\s*PCI", device_info)
        gpu_name = raw_gpu_name.group(1).strip() if raw_gpu_name else "Intel GPU"
    else:
        device_info = subprocess.check_output(["lspci", "-nn"], text=True)

        device_id = re.search(r"\[8086:([0-9a-fA-F]{4})\]", device_info)
        if not device_id:
            return None

        gpu_id = int(device_id.group(1), 16)
        raw_gpu_name = re.search(r"\[(Arc[^\]]+)\]", device_info)
        gpu_name = "Intel " + raw_gpu_name.group(1) if raw_gpu_name else "Intel GPU"
    
    if 22176 <= gpu_id <= 22271: # 0x56A0 - 0x56FF, ARC
        xpu_version = "xpu"
    else:
        return None
    
    global GPU_NAME 
    GPU_NAME = gpu_name
    return xpu_version

def get_gpu():
    for attempt in (get_nvidia_specs, get_amd_specs, get_intel_specs):
        try:
            return attempt()
        except Exception:
            pass
    return None

def install_torch():
    if sys.platform == "darwin":
        step_complete("Apple system detected. Installing Apple MPS PyTorch build.")
        return [f"torch=={PYTORCH_VERSION}", "--index-url", "https://download.pytorch.org/whl/cpu"] # Apple MPS is included in cpu torch build
    else:
        print("-" * 30)
        print("Attempting automatic GPU detection")
        pytorch_wheel = get_gpu()
        if pytorch_wheel is not None:
            print("-" * 60)
            print(f"Supported GPU detected: {GPU_NAME}")
            print("Installing matching PyTorch build.")
            print("\"use_gpu_for_local_models\" config option can be enabled.")
            print("-" * 60)
            # AMD's own repo
            if (IS_AMD_GPU):
                # Make wheel global for AMD specifically so it can be reused in the retroactive patch later
                global PYTORCH_WHEEL
                PYTORCH_WHEEL = pytorch_wheel
                # Install ROCm base first
                step_complete("Installing ROCm base...")
                return [f"torch=={PYTORCH_VERSION}+{PYTORCH_WHEEL}", "--index-url", "https://repo.amd.com/rocm/whl-multi-arch/"]
            # Default PyTorch repo
            else:
                return [f"torch=={PYTORCH_VERSION}", "--index-url", f"https://download.pytorch.org/whl/{pytorch_wheel}"]
        else:
            print("-" * 60)
            print("No supported GPU configuration detected. Installing CPU Only PyTorch build.")
            print("If you do have a supported GPU with updated drivers, please restart the installer with the \"-manual-torch\" flag.")
            print("Older GPUs (e.g. GTX 7xx or RX 6xxx) may work with custom setups but are not officially supported.")
            print("-" * 60)
            return [f"torch=={PYTORCH_VERSION}", "--index-url", "https://download.pytorch.org/whl/cpu"]

# Manual fallback if automatic GPU detection fails despite GPU being present
def choose_torch_build():
    torch_installed = subprocess.run([str(venv_python()), "-m", "pip", "show", "torch"], capture_output=True, text=True)

    # Uninstall the automatically installed fallback cpu build
    if(torch_installed.returncode == 0):
        subprocess.check_call([str(venv_python()), "-m", "pip", "uninstall", "-y", "torch"])
        step_complete("Uninstalled previous PyTorch build.")

    print("-" * 60)
    print("Choose PyTorch build by entering the corresponding number:")
    print("1) CPU Only - Default and recommended if no dedicated GPU is available")
    print("2) CUDA 12.6 - For NVIDIA GPUs, Maxwell and Pascal (900 and 10 series)")
    print("3) CUDA 13.0 - For NVIDIA GPUs, Turing (16 series) and up")
    print("4) ROCm 7.14.0 - For AMD GPUs, RDNA3 (7000 series) and up")
    print("5) XPU - For Intel GPUs, ARC and up")
    print("-" * 60)
    print("Determines whether the \"use_gpu_for_local_models\" config option can be used.")
    print("If you want to use GPU, make sure your GPU driver is up to date.")
    print("If you are on an Apple device, select CPU Only, Apple MPS is included.")
    print("Older GPUs (e.g. GTX 7xx or RX 6xxx) may work with custom setups but are not officially supported.")
    print("-" * 60)

    while True:
        choice = input("> ").strip()
        
        if(choice == "1"):
            return [f"torch=={PYTORCH_VERSION}", "--index-url", "https://download.pytorch.org/whl/cpu"]
        elif(choice == "2"):
            return [f"torch=={PYTORCH_VERSION}", "--index-url", "https://download.pytorch.org/whl/cu126"]
        elif(choice == "3"):
            return [f"torch=={PYTORCH_VERSION}", "--index-url", "https://download.pytorch.org/whl/cu130"]
        elif(choice == "4"):
            global IS_AMD_GPU
            IS_AMD_GPU = True
            global PYTORCH_WHEEL
            PYTORCH_WHEEL = "rocm7.14.0"
            step_complete("Installing ROCm base...")
            return [f"torch=={PYTORCH_VERSION}+rocm7.14.0", "--index-url", "https://repo.amd.com/rocm/whl-multi-arch/"]
        elif(choice == "5"):
            return [f"torch=={PYTORCH_VERSION}", "--index-url", "https://download.pytorch.org/whl/xpu"]
        else:
            print("Invalid choice. Please enter a number from 1 to 5.")

def install_requirements_from_file():
    with open("misc/requirements.txt", "r") as file:
        packages = [
            line.strip()
            for line in file if line.strip() and not line.startswith("#")
        ]

    pip_install(*packages)

def add_openai_key():
    openai_key = input("Enter an OpenAI API Key: ")
    with open("misc/api-key.txt", "w") as file:
        file.write(openai_key)

def create_linux_auto_activating_shell():
    global VENV_DIR
    # Uses raw string as using quotes makes python think the entire thing is a comment
    script_template = r"""#!/bin/bash
    # Auto-detect and launch terminal with venv activated
    
    # Check for macOS terminal
    if [[ "$OSTYPE" == "darwin"* ]]; then
        osascript -e "tell application \"Terminal\" to do script \"source '{venv_path}/bin/activate'\""
        exit 0
    fi

    # Check for known terminals
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal -- bash --init-file "{venv_path}/bin/activate"
    elif command -v konsole &> /dev/null; then
        konsole -e bash --init-file "{venv_path}/bin/activate"
    elif command -v mate-terminal &> /dev/null; then
        mate-terminal -e bash --init-file "{venv_path}/bin/activate"
    elif command -v xfce4-terminal &> /dev/null; then
        xfce4-terminal -e bash --init-file "{venv_path}/bin/activate"
    elif command -v terminator &> /dev/null; then
        terminator -e bash --init-file "{venv_path}/bin/activate"
    elif command -v xterm &> /dev/null; then
        xterm -e bash --init-file "{venv_path}/bin/activate"
    else
        echo "No supported terminal emulator found!"
        echo "Please manually activate with: source {venv_path}/bin/activate"
        read -p "Press Enter to exit..."
    fi
    """
    with open("llmorph.sh", "w") as file:
        file.write(script_template.format(venv_path=VENV_DIR))
    Path("llmorph.sh").chmod(0o755)

def step_complete(message):
    print("-" * 30)
    print(message)
    print("-" * 30)
    
def main():
    with open("pyproject.toml", "r") as file:
        content = file.read()
        llmorph_version = re.search(r"version =\s*(.*)", content).group(1).strip("\"")

    print("=" * 30)    
    print(f"LLMorph {llmorph_version} Installer")
    print("=" * 30)

    check_architecture()
    
    check_python_version()
    global IS_WINDOWS 
    IS_WINDOWS = sys.platform == "win32"

    global VENV_DIR
    venv_path = input("Enter full venv path (existing or new): ").strip()

    if not venv_path:
        raise ValueError("Venv path cannot be empty")
    
    VENV_DIR = Path(venv_path)

    venv_created = create_venv()
    if(venv_created):
        step_complete("Venv created.")
    else:
        step_complete("Using existing venv.")

    print("Updating PIP...")
    pip_install("--upgrade", "pip")

    # Install torch first to avoid other packages installing an undesired torch build as part of their dependencies
    if(len(sys.argv) > 1):
        installer_choice = sys.argv[1]
    else: 
        installer_choice = None

    if(installer_choice == "-manual-torch"):
        torch_args = choose_torch_build()
    else:
        torch_args = install_torch()
    pip_install(*torch_args)

    # Retroactively install gpu specific ROCm on top of base, required as the ROCm base is not functional for execution
    if (IS_AMD_GPU):
        gfx = get_amd_gfx()
        print("-" * 30)
        print(f"Detected gfx: {gfx}")
        print("Installing GPU specific ROCm build on top of base...")
        print("-" * 30)
        pip_install(f"torch[device-{gfx}]=={PYTORCH_VERSION}+{PYTORCH_WHEEL}", "--index-url", "https://repo.amd.com/rocm/whl-multi-arch/")
    
    step_complete("PyTorch installed.")

    # Install all other requirements on top
    install_requirements_from_file()
    step_complete("Dependencies installed.")

    # Install additional nltk packages
    subprocess.check_call([str(venv_python()), "-c", "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"])
    step_complete("Additional NLTK packages installed.")

    add_openai_key()
    step_complete("OpenAI API Key installed.")

    # Install Python project from pyproject.toml
    subprocess.check_call([str(venv_python()), "-m", "pip", "install", "-e", "."])
    step_complete("Project installed.")

    # Create auto activating shell script for convenience
    if(IS_WINDOWS):
        with open("llmorph.bat", "w") as file:
            file.write(f'@echo off\nstart cmd /k "{venv_path}\\Scripts\\activate"')
    else:
        create_linux_auto_activating_shell()
    step_complete("Start script created.")

    print("=" * 30)
    print(f"LLMorph {llmorph_version} successfully installed!")
    if(IS_WINDOWS):
        print("You can run llmorph.bat to start.")
    else:
        print("You can run llmorph.sh to start.")
        print("Note: If the file opens in a text editor run: chmod +x llmorph.sh")

main()
