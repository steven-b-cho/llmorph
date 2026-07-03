# Tested on:
# Linux Mint, CPU
# Windows 11, NVIDIA
# Windows 11, AMD
import sys
import platform
import subprocess
import re
from pathlib import Path

REQUIRED_PYTHON_VERSION = "3.14.5"
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
    
    if 29440 <= gpu_id <= 29695: # 0x7300 - 0x73FF, RDNA2
        rocm_version = "rocm6.4"
    elif gpu_id >= 29696: # >= 0x7400, RDNA3+
        rocm_version = "rocm7.2"
    else:
        return None

    # AMD GPU detection on Windows works, but no ROCm wheel for PyTorch 2.12.0 exists for Windows at this time, 
    # so PIP just errors and stops installer. Falling back to CPU only if on Windows with AMD card for now.
    if(IS_WINDOWS and gpu_id is not None):
        return None
    else:
        global GPU_NAME 
        GPU_NAME = gpu_name
        return rocm_version

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
            return [f"torch=={PYTORCH_VERSION}", "--index-url", f"https://download.pytorch.org/whl/{pytorch_wheel}"]
        else:
            print("-" * 60)
            print("No supported GPU configuration detected. Installing CPU Only PyTorch build.")
            # Explain why ROCm on Windows doesn't work at this time.
            if(IS_WINDOWS):
                print("If you're using an AMD GPU, then you currently cannot use it in LLMorph")
                print(f"as AMD hasn't released a Windows compatible PyTorch {PYTORCH_VERSION} wheel yet.")
            print("If you do have a supported GPU with updated drivers, please restart the installer with the \"-manual-torch\" flag.")
            print("Older GPUs (e.g. GTX 7xx or RX 5xxx) may work with custom setups but are not officially supported.")
            print("-" * 60)
            return [f"torch=={PYTORCH_VERSION}", "--index-url", "https://download.pytorch.org/whl/cpu"]

# Manual fallback if automatic GPU detection fails despite GPU being present
def choose_torch_build():
    torch_installed = subprocess.run([str(venv_python()), "-m", "pip", "show", "torch"], capture_output=True, text=True)

    if(torch_installed.returncode == 0):
        subprocess.check_call([str(venv_python()), "-m", "pip", "uninstall", "-y", "torch"])
        step_complete("Uninstalled previous PyTorch build.")

    print("-" * 60)
    print("Choose PyTorch build by entering the corresponding number:")
    print("1) CPU Only - Default and recommended if no dedicated GPU is available")
    print("2) CUDA 12.6 - For NVIDIA GPUs, Maxwell and Pascal (900 and 10 series)")
    print("3) CUDA 13.0 - For NVIDIA GPUs, Turing (16 series) and up")
    print("4) ROCm 6.4 - For AMD GPUs, RDNA2 (6000 series)")
    print("5) ROCm 7.2 - For AMD GPUs, RDNA3 (7000 series) and up")
    print("6) XPU - For Intel GPUs, ARC and up")
    print("-" * 60)
    # Explain why ROCm on Windows doesn't work at this time.
    if(IS_WINDOWS):
        print("If you're using an AMD GPU, then you currently cannot use it in LLMorph")
        print(f"as AMD hasn't released a Windows compatible PyTorch {PYTORCH_VERSION} wheel yet.")
    print("Determines whether the \"use_gpu_for_local_models\" config option can be used.")
    print("If you want to use GPU, make sure your GPU driver is up to date.")
    print("If you are on an Apple device with an M series chip, select CPU Only, Apple MPS is included.")
    print("Older GPUs (e.g. GTX 7xx or RX 5xxx) may work with custom setups but are not officially supported.")
    print("-" * 60)

    while True:
        choice = input("> ").strip()
        
        if(choice == "1"):
            return [f"torch=={PYTORCH_VERSION}", "--index-url", "https://download.pytorch.org/whl/cpu"]
        elif(choice == "2"):
            return [f"torch=={PYTORCH_VERSION}", "--index-url", "https://download.pytorch.org/whl/cu126"]
        elif(choice == "3"):
            return [f"torch=={PYTORCH_VERSION}", "--index-url", "https://download.pytorch.org/whl/cu130"]
        
        # Prevent manual install attempt of ROCm if on Windows
        elif(IS_WINDOWS and (choice == "4" or choice == "5")):
            print("-" * 60)
            print(f"ROCm is currently unavailable on Windows for PyTorch {PYTORCH_VERSION}.")
            print("Please choose another option.")
            print("-" * 60)

        elif(choice == "4"):
            return [f"torch=={PYTORCH_VERSION}", "--index-url", "https://download.pytorch.org/whl/rocm6.4"]
        elif(choice == "5"):
            return [f"torch=={PYTORCH_VERSION}", "--index-url", "https://download.pytorch.org/whl/rocm7.2"]
        elif(choice == "6"):
            return [f"torch=={PYTORCH_VERSION}", "--index-url", "https://download.pytorch.org/whl/xpu"]
        else:
            print("Invalid choice. Please enter a number from 1 to 6.")

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
