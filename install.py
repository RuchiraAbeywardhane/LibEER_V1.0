"""
LibEER - Dependency Installer
Run this script to install all required libraries.
Usage: python install.py
"""

import subprocess
import sys
import os

def run(cmd):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[WARNING] Command failed: {cmd}")
    else:
        print(f"[OK]")

def main():
    print("=" * 60)
    print("  LibEER - Installing Required Libraries")
    print("=" * 60)

    pip = f"{sys.executable} -m pip"

    # ── Core scientific libraries ──────────────────────────────
    print("\n[1/6] Installing core scientific libraries...")
    run(f"{pip} install numpy==1.24.3")
    run(f"{pip} install scipy==1.9.3")
    run(f"{pip} install scikit-learn==1.4.2")
    run(f"{pip} install pandas")

    # ── PyTorch (CUDA 11.7) ───────────────────────────────────
    print("\n[2/6] Installing PyTorch (CUDA 11.7)...")
    print("      If you don't have a GPU, replace cu117 with cpu.")
    run(
        f"{pip} install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 "
        f"--extra-index-url https://download.pytorch.org/whl/cu117"
    )

    # ── PyTorch Geometric ─────────────────────────────────────
    print("\n[3/6] Installing PyTorch Geometric (for DGCNN, RGNN)...")
    run(f"{pip} install torch_geometric==2.4.0")
    run(
        f"{pip} install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv "
        f"-f https://data.pyg.org/whl/torch-1.13.0+cu117.html"
    )

    # ── EEG / Signal Processing libraries ────────────────────
    print("\n[4/6] Installing EEG & signal processing libraries...")
    run(f"{pip} install mne")           # EEG data loading & preprocessing
    run(f"{pip} install mat73")         # Load MATLAB v7.3 .mat files
    run(f"{pip} install xmltodict")     # XML config parsing
    run(f"{pip} install braindecode")   # For FBSTCNet model
    run(f"{pip} install skorch")        # For FBSTCNet model

    # ── Utilities ─────────────────────────────────────────────
    print("\n[5/6] Installing utility libraries...")
    run(f"{pip} install tqdm==4.66.4")
    run(f"{pip} install PyYAML==6.0.1")

    # ── Verify key imports ────────────────────────────────────
    print("\n[6/6] Verifying key installations...")
    libraries = [
        "numpy", "scipy", "sklearn", "pandas",
        "torch", "torch_geometric",
        "mne", "mat73", "xmltodict", "braindecode", "skorch",
        "tqdm", "yaml"
    ]
    all_ok = True
    for lib in libraries:
        try:
            __import__(lib)
            print(f"  [OK] {lib}")
        except ImportError:
            print(f"  [MISSING] {lib} - please install manually")
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("  All libraries installed successfully!")
    else:
        print("  Some libraries are missing. Please check the warnings above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
