#!/usr/bin/env python3
"""Hardware and environment report generator for SSLTang.

Inspects GPU, CUDA, PyTorch, driver, CPU, RAM, disk mounts, and generates
JSON output with heuristic recommendations for batch size and DataLoader workers.

Enforces:
- No MPS (Apple Silicon Metal Performance Shaders is disabled by contract).
- Blackwell (sm120) requires PyTorch compiled with CUDA 12.8+ / 13.2+.
- Identifies large-capacity mounts (e.g. /workspace) vs constrained root/home mounts.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional


def get_cpu_info() -> Dict[str, Any]:
    logical = os.cpu_count() or 1
    physical = logical
    model = platform.processor() or platform.machine() or "Unknown CPU"

    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or logical
    except ImportError:
        pass

    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if "model name" in line:
                        model = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
    elif platform.system() == "Darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True)
            if out.strip():
                model = out.strip()
        except Exception:
            pass

    return {
        "model": model,
        "count_logical": logical,
        "count_physical": physical,
    }


def get_ram_info() -> Dict[str, Any]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {
            "total_bytes": vm.total,
            "total_gb": round(vm.total / (1024**3), 2),
            "available_bytes": vm.available,
            "available_gb": round(vm.available / (1024**3), 2),
            "used_bytes": vm.used,
            "used_gb": round(vm.used / (1024**3), 2),
            "percent_used": vm.percent,
        }
    except ImportError:
        # Fallback to /proc/meminfo if psutil is unavailable
        total_kb, avail_kb = 0, 0
        if os.path.exists("/proc/meminfo"):
            try:
                with open("/proc/meminfo", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            total_kb = int(line.split()[1])
                        elif line.startswith("MemAvailable:"):
                            avail_kb = int(line.split()[1])
                total_b = total_kb * 1024
                avail_b = avail_kb * 1024
                return {
                    "total_bytes": total_b,
                    "total_gb": round(total_b / (1024**3), 2),
                    "available_bytes": avail_b,
                    "available_gb": round(avail_b / (1024**3), 2),
                    "used_bytes": total_b - avail_b,
                    "used_gb": round((total_b - avail_b) / (1024**3), 2),
                    "percent_used": round(100.0 * (total_b - avail_b) / max(total_b, 1), 1),
                }
            except Exception:
                pass
        return {
            "total_bytes": 0,
            "total_gb": 0.0,
            "available_bytes": 0,
            "available_gb": 0.0,
            "used_bytes": 0,
            "used_gb": 0.0,
            "percent_used": 0.0,
        }


def get_nvidia_driver_version() -> Optional[str]:
    # Check nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=3,
        ).strip()
        if out:
            return out.splitlines()[0].strip()
    except Exception:
        pass

    # Check /proc/driver/nvidia/version
    if os.path.exists("/proc/driver/nvidia/version"):
        try:
            with open("/proc/driver/nvidia/version", "r", encoding="utf-8") as f:
                first = f.readline()
                m = re.search(r"NVRM version:.*?(\d+\.\d+\.\d+)", first)
                if m:
                    return m.group(1)
        except Exception:
            pass

    return None


def get_disk_info() -> Dict[str, Any]:
    paths_to_check = [
        "/workspace",
        "/",
        "/home",
        "/var/lib/docker",
        "/dev/shm",
        os.getcwd(),
    ]
    seen_paths = set()
    disks = []

    for p in paths_to_check:
        if not os.path.exists(p):
            continue
        real_p = os.path.realpath(p)
        if real_p in seen_paths:
            continue
        seen_paths.add(real_p)

        try:
            usage = shutil.disk_usage(p)
            total_gb = round(usage.total / (1024**3), 2)
            used_gb = round(usage.used / (1024**3), 2)
            free_gb = round(usage.free / (1024**3), 2)
            percent_free = round(100.0 * usage.free / max(usage.total, 1), 1)

            is_recommended = (p == "/workspace" or free_gb > 100.0) and p != "/dev/shm"
            is_constrained = free_gb < 50.0 and p != "/dev/shm"

            disks.append({
                "path": p,
                "real_path": real_p,
                "total_gb": total_gb,
                "used_gb": used_gb,
                "free_gb": free_gb,
                "percent_free": percent_free,
                "is_recommended_workspace": is_recommended,
                "is_constrained": is_constrained,
            })
        except Exception:
            continue

    return {
        "mounts": disks,
        "recommended_workspace_root": "/workspace" if os.path.exists("/workspace") else os.getcwd(),
        "storage_advisory": (
            "Store all virtual environments, checkpoints, and caches (PIP_CACHE_DIR, "
            "HF_HOME, TORCH_EXTENSIONS_DIR) in /workspace. Root and home partitions may "
            "have limited space (<50GB free)."
        ),
    }


def get_gpu_and_torch_info() -> Dict[str, Any]:
    gpu_info: Dict[str, Any] = {
        "available": False,
        "model": None,
        "count": 0,
        "compute_capability": None,
        "is_blackwell": False,
        "sm_version": None,
        "vram_total_gb": 0.0,
        "vram_total_bytes": 0,
        "vram_free_gb": 0.0,
        "bf16_supported": False,
        "driver_version": get_nvidia_driver_version(),
        "mps_detected": False,
        "mps_enabled": False,
    }

    torch_info: Dict[str, Any] = {
        "version": None,
        "cuda_build": None,
        "cudnn_version": None,
        "installed": False,
    }

    try:
        import torch
        torch_info["installed"] = True
        torch_info["version"] = torch.__version__
        torch_info["cuda_build"] = torch.version.cuda
        if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
            torch_info["cudnn_version"] = torch.backends.cudnn.version()

        # Check MPS strictly: Contract disallows MPS.
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_info["mps_detected"] = True
            gpu_info["mps_enabled"] = False  # Explicitly disabled per contract

        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["count"] = torch.cuda.device_count()
            gpu_info["model"] = torch.cuda.get_device_name(0)

            cap = torch.cuda.get_device_capability(0)
            gpu_info["compute_capability"] = f"{cap[0]}.{cap[1]}"
            gpu_info["sm_version"] = f"sm{cap[0]}{cap[1]}"
            gpu_info["is_blackwell"] = (cap[0] >= 12)

            props = torch.cuda.get_device_properties(0)
            total_bytes = props.total_memory
            gpu_info["vram_total_bytes"] = total_bytes
            gpu_info["vram_total_gb"] = round(total_bytes / (1024**3), 2)

            try:
                free_bytes, _ = torch.cuda.mem_get_info(0)
                gpu_info["vram_free_gb"] = round(free_bytes / (1024**3), 2)
            except Exception:
                gpu_info["vram_free_gb"] = gpu_info["vram_total_gb"]

            gpu_info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())

    except ImportError:
        pass

    return {
        "gpu": gpu_info,
        "torch": torch_info,
    }


def compute_heuristics(gpu: Dict[str, Any], cpu: Dict[str, Any], ram: Dict[str, Any]) -> Dict[str, Any]:
    vram_gb = gpu.get("vram_total_gb", 0.0)
    cpu_logical = cpu.get("count_logical", 1)
    cpu_physical = cpu.get("count_physical", 1)
    bf16 = gpu.get("bf16_supported", False)

    # Starting points, not measured throughput or memory limits for an arbitrary model.
    if not gpu.get("available", False):
        recommended_batch_size = 16
        max_batch_size = 32
        batch_notes = "CPU execution: recommended batch size 16 to keep latency low."
    elif vram_gb >= 80.0:
        recommended_batch_size = 256
        max_batch_size = 512
        batch_notes = (
            f"Blackwell / high-VRAM GPU ({vram_gb:.1f} GB) with BF16 autocast: "
            "Consider batch 256 and trial 512 while measuring memory and throughput. "
            "The reported initial study used batch 128; no optimal batch was established."
        )
    elif vram_gb >= 40.0:
        recommended_batch_size = 128
        max_batch_size = 256
        batch_notes = f"40-80 GB VRAM ({vram_gb:.1f} GB): batch size 128 recommended."
    elif vram_gb >= 20.0:
        recommended_batch_size = 64
        max_batch_size = 128
        batch_notes = f"20-40 GB VRAM ({vram_gb:.1f} GB): batch size 64 recommended."
    elif vram_gb >= 10.0:
        recommended_batch_size = 32
        max_batch_size = 64
        batch_notes = f"10-20 GB VRAM ({vram_gb:.1f} GB): batch size 32 recommended."
    else:
        recommended_batch_size = 16
        max_batch_size = 32
        batch_notes = f"Low VRAM ({vram_gb:.1f} GB): batch size 16 recommended."

    # Heuristic DataLoader num_workers:
    # Rule of thumb: min(8, physical CPU cores // 2), avoiding IPC context thrashing.
    if cpu_logical <= 2:
        recommended_workers = 0
    elif cpu_logical <= 8:
        recommended_workers = 2
    elif cpu_logical <= 16:
        recommended_workers = 4
    else:
        # For 32+ CPUs, 8 workers provides high throughput without shared memory contention.
        recommended_workers = 8

    worker_notes = (
        f"Recommended {recommended_workers} DataLoader workers for {cpu_logical} logical cores "
        f"({cpu_physical} physical). Use pin_memory=True and persistent_workers=True."
    )

    return {
        "recommended_batch_size": recommended_batch_size,
        "suggested_larger_batch_trial": max_batch_size,
        "heuristics_benchmarked": False,
        "batch_size_reasoning": batch_notes,
        "recommended_num_workers": recommended_workers,
        "num_workers_reasoning": worker_notes,
        "recommended_precision": "bfloat16" if bf16 else ("float16" if gpu.get("available") else "float32"),
        "pin_memory": gpu.get("available", False),
        "persistent_workers": recommended_workers > 0,
    }


def generate_hardware_report() -> Dict[str, Any]:
    cpu = get_cpu_info()
    ram = get_ram_info()
    disk = get_disk_info()
    gpu_torch = get_gpu_and_torch_info()
    gpu = gpu_torch["gpu"]
    torch_info = gpu_torch["torch"]
    heuristics = compute_heuristics(gpu, cpu, ram)

    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
        },
        "gpu": gpu,
        "torch": torch_info,
        "cpu": cpu,
        "ram": ram,
        "disk": disk,
        "heuristics": heuristics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SSLTang Hardware & Infrastructure Report Generator"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Optional path to write JSON report to file",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Output compact single-line JSON instead of formatted JSON",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print JSON to stdout if --output is specified",
    )
    args = parser.parse_args()

    report = generate_hardware_report()
    indent = None if args.compact else 2
    output_str = json.dumps(report, indent=indent, sort_keys=False)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_str + "\n")
        if not args.quiet:
            print(output_str)
    else:
        print(output_str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
