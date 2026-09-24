"""
Amazon ML Worker — Hardware & Environment Detection.

Detects system capabilities and optional package availability.
Reports everything in a structured dict for JSON serialization.
"""

import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Dict


def detect_hardware() -> Dict[str, Any]:
    """Return a structured hardware/environment report."""
    report: Dict[str, Any] = {}

    # ── System ────────────────────────────────────────────────────────────
    report["python_version"] = sys.version
    report["platform"] = platform.platform()
    report["os"] = platform.system()
    report["machine"] = platform.machine()
    report["cpu_count"] = os.cpu_count()

    # RAM
    try:
        import psutil
        mem = psutil.virtual_memory()
        report["ram_total_gb"] = round(mem.total / (1024 ** 3), 2)
        report["ram_available_gb"] = round(mem.available / (1024 ** 3), 2)
    except ImportError:
        # Fallback for systems without psutil
        if platform.system() == "Windows":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ulonglong = ctypes.c_ulonglong

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", c_ulonglong),
                        ("ullAvailPhys", c_ulonglong),
                        ("ullTotalPageFile", c_ulonglong),
                        ("ullAvailPageFile", c_ulonglong),
                        ("ullTotalVirtual", c_ulonglong),
                        ("ullAvailVirtual", c_ulonglong),
                        ("ullAvailExtendedVirtual", c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(stat)
                kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                report["ram_total_gb"] = round(stat.ullTotalPhys / (1024 ** 3), 2)
                report["ram_available_gb"] = round(stat.ullAvailPhys / (1024 ** 3), 2)
            except Exception:
                report["ram_total_gb"] = "unknown"
                report["ram_available_gb"] = "unknown"
        else:
            report["ram_total_gb"] = "unknown"
            report["ram_available_gb"] = "unknown"

    # Disk
    total, used, free = shutil.disk_usage(Path.cwd())
    report["disk_total_gb"] = round(total / (1024 ** 3), 2)
    report["disk_free_gb"] = round(free / (1024 ** 3), 2)

    # ── PyTorch / CUDA ────────────────────────────────────────────────────
    report["pytorch"] = _probe_package("torch", attr="__version__")
    report["cuda_available"] = False
    report["cuda_version"] = None
    report["gpu_count"] = 0
    report["gpus"] = []

    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["cuda_version"] = torch.version.cuda
            report["gpu_count"] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                report["gpus"].append({
                    "index": i,
                    "name": props.name,
                    "vram_gb": round(props.total_mem / (1024 ** 3), 2),
                })
    except ImportError:
        pass

    # ── Key packages ──────────────────────────────────────────────────────
    report["pandas"] = _probe_package("pandas")
    report["numpy"] = _probe_package("numpy")
    report["scipy"] = _probe_package("scipy")
    report["scikit_learn"] = _probe_package("sklearn", attr="__version__")
    report["opencv"] = _probe_package("cv2", attr="__version__")
    report["pillow"] = _probe_package("PIL", attr="__version__")
    report["pyarrow"] = _probe_package("pyarrow")
    report["matplotlib"] = _probe_package("matplotlib")
    report["pyyaml"] = _probe_package("yaml", attr="__version__")
    report["tqdm"] = _probe_package("tqdm")

    # ── Optional packages ────────────────────────────────────────────────
    report["transformers"] = _probe_package("transformers")
    report["peft"] = _probe_package("peft")
    report["bitsandbytes"] = _probe_package("bitsandbytes")
    report["accelerate"] = _probe_package("accelerate")
    report["datasets"] = _probe_package("datasets")
    report["lightgbm"] = _probe_package("lightgbm")
    report["xgboost"] = _probe_package("xgboost")
    report["catboost"] = _probe_package("catboost")

    return report


def _probe_package(module_name: str, attr: str = "__version__") -> str:
    """Try importing *module_name* and return its version string, or 'not installed'."""
    try:
        mod = __import__(module_name)
        return getattr(mod, attr, "installed (version unknown)")
    except ImportError:
        return "not installed"


def detect_environment_type() -> str:
    """Heuristic detection: 'kaggle' | 'colab' | 'local'."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if "google.colab" in sys.modules:
        return "colab"
    return "local"


def print_report(report: Dict[str, Any]) -> None:
    """Pretty-print a hardware report."""
    print("=" * 60)
    print("  Amazon ML Worker — Environment Report")
    print("=" * 60)
    for key, value in report.items():
        if key == "gpus":
            print(f"  {key}:")
            for gpu in value:
                print(f"    [{gpu['index']}] {gpu['name']} — {gpu['vram_gb']} GB VRAM")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)


def save_report(report: Dict[str, Any], path: str) -> None:
    """Save the hardware report as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
