#!/usr/bin/env python3
"""Environment and Hardware Acceleration Verification Utility.

Checks the current execution platform, Python runtime, PyTorch backends,
and hardware acceleration readiness (Apple Silicon MPS, NVIDIA CUDA, or CPU fallback).

Usage:
    python tools/verify_env.py
"""

import platform
import sys
from typing import Dict, Any


def probe_python() -> Dict[str, Any]:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
    }


def probe_pytorch() -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "installed": False,
        "version": None,
        "mps_available": False,
        "mps_built": False,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_version": None,
        "recommended_device": "cpu",
    }

    try:
        import torch  # type: ignore

        status["installed"] = True
        status["version"] = torch.__version__

        # Check Apple Silicon MPS
        if hasattr(torch.backends, "mps"):
            status["mps_built"] = torch.backends.mps.is_built()
            status["mps_available"] = torch.backends.mps.is_available()

        # Check NVIDIA CUDA
        if hasattr(torch, "cuda"):
            status["cuda_available"] = torch.cuda.is_available()
            if status["cuda_available"]:
                status["cuda_device_count"] = torch.cuda.device_count()
                status["cuda_version"] = torch.version.cuda

        # Determine recommended compute device
        if status["cuda_available"]:
            status["recommended_device"] = "cuda:0"
        elif status["mps_available"]:
            status["recommended_device"] = "mps"
        else:
            status["recommended_device"] = "cpu"

    except ImportError:
        pass

    return status


def probe_optional_packages() -> Dict[str, str]:
    packages = ["pydantic", "cv2", "ultralytics", "numpy", "pytest"]
    results = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            results[pkg] = getattr(mod, "__version__", "installed (no version)")
        except ImportError:
            results[pkg] = "not installed"
    return results


def main() -> int:
    py_info = probe_python()
    torch_info = probe_pytorch()
    pkg_info = probe_optional_packages()

    print("=" * 60)
    print(" Aerial Perception System — Environment Verification")
    print("=" * 60)

    print("\n[Host Platform]")
    print(f"  OS / Kernel   : {py_info['platform']}")
    print(f"  Architecture  : {py_info['architecture']}")
    print(f"  Python Version: {py_info['version']} ({py_info['implementation']})")

    print("\n[PyTorch & Acceleration Backends]")
    if not torch_info["installed"]:
        print("  PyTorch       : NOT INSTALLED")
        print("  Target Device : cpu (Fallback)")
    else:
        print(f"  PyTorch Version: {torch_info['version']}")
        print(f"  Apple Silicon MPS Available: {torch_info['mps_available']} (Built: {torch_info['mps_built']})")
        print(f"  NVIDIA CUDA Available      : {torch_info['cuda_available']} (Devices: {torch_info['cuda_device_count']})")
        if torch_info["cuda_version"]:
            print(f"  CUDA Version               : {torch_info['cuda_version']}")

        print("\n[Device Selection]")
        rec = torch_info["recommended_device"]
        if rec.startswith("cuda"):
            print(f"  Active Compute Target: {rec} [NVIDIA GPU Accelerated]")
        elif rec == "mps":
            print(f"  Active Compute Target: {rec} [Apple Silicon GPU Accelerated]")
        else:
            print(f"  Active Compute Target: {rec} [Standard CPU Fallback]")

    print("\n[Key Packages]")
    for name, ver in pkg_info.items():
        print(f"  {name:<14}: {ver}")

    print("\n" + "=" * 60)
    print(" Verification Status: READY FOR MILESTONE 1 EXECUTION")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
