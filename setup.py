#!/usr/bin/env python3
"""
Setup script for Qwen3-TTS + Seed-VC V2 voice server.

Installs all dependencies (with correct version ordering to avoid conflicts),
clones the Seed-VC repo, and downloads model weights.

Usage:
    python setup.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
SEED_VC_REPO_DIR = ROOT_DIR / "seed_vc_repo"

SEED_VC_GIT_URL = "https://github.com/Plachtaa/seed-vc.git"
QWEN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

# Seed-VC V2 weights on HuggingFace
DOWNLOAD_TASKS = [
    # (repo_id, remote_filename, local_filename, target_subdir)
    ("Plachta/Seed-VC", "v2/cfm_small.pth", "cfm_small.pth", "seed_vc_v2"),
    ("Plachta/Seed-VC", "v2/ar_base.pth", "ar_base.pth", "seed_vc_v2"),
    ("funasr/campplus", "campplus_cn_common.bin", "campplus_cn_common.bin", "seed_vc_v2"),
]


def banner(msg):
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60 + "\n")


def run(cmd, error_msg="Command failed", allow_fail=False):
    """Run a shell command, printing it first.  Exit on failure unless allow_fail."""
    print(f"$ {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print(f"\n[ERROR] {error_msg}  (exit code {ret})")
        if not allow_fail:
            sys.exit(1)
    return ret


def install_dependencies():
    banner("Step 1: Installing Python dependencies")

    # 1. PyTorch >= 2.6 with CUDA 12.4 (required by transformers >= 4.52)
    print("[INFO] Installing PyTorch 2.6 with CUDA 12.4 …")
    run(
        "pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 "
        "--index-url https://download.pytorch.org/whl/cu124",
        "Failed to install PyTorch"
    )

    # 2. Core dependencies — install before qwen-tts to avoid version fights
    print("\n[INFO] Installing core dependencies …")
    run(
        "pip install --upgrade "
        "transformers>=4.52 accelerate soundfile librosa numpy "
        "fastapi uvicorn python-multipart huggingface_hub "
        "pydub pyyaml munch einops hydra-core omegaconf scipy",
        "Failed to install core dependencies"
    )

    # 3. Qwen-TTS
    print("\n[INFO] Installing qwen-tts …")
    run(
        "pip install --upgrade qwen-tts",
        "Failed to install qwen-tts"
    )

    # 4. Optional: flash-attn (non-fatal if it fails)
    print("\n[INFO] Trying to install flash-attn (optional, speeds up Qwen) …")
    run(
        "pip install flash-attn --no-build-isolation",
        "flash-attn not installed — Qwen will use manual attention (slower but OK)",
        allow_fail=True,
    )

    print("\n[OK] All dependencies installed.")


def setup_seed_vc_repo():
    banner("Step 2: Setting up Seed-VC repository")

    if SEED_VC_REPO_DIR.exists():
        print("[INFO] Seed-VC repo already exists, pulling latest …")
        run(f"git -C {SEED_VC_REPO_DIR} pull", "Failed to update Seed-VC", allow_fail=True)
    else:
        print("[INFO] Cloning Seed-VC …")
        run(f"git clone {SEED_VC_GIT_URL} {SEED_VC_REPO_DIR}", "Failed to clone Seed-VC")

    print("[OK] Seed-VC repo ready.")


def download_models():
    banner("Step 3: Downloading model weights")
    MODELS_DIR.mkdir(exist_ok=True)

    # 1. Qwen Voice Design
    qwen_dir = MODELS_DIR / "qwen_voice_design"
    if qwen_dir.exists() and (qwen_dir / "model.safetensors").exists():
        print(f"[SKIP] Qwen model already downloaded at {qwen_dir}")
    else:
        print(f"[INFO] Downloading Qwen model {QWEN_MODEL_ID} …")
        run(
            f"huggingface-cli download {QWEN_MODEL_ID} --local-dir {qwen_dir}",
            "Failed to download Qwen model"
        )

    # 2. Seed-VC V2 checkpoints
    for repo_id, remote_path, local_name, target_subdir in DOWNLOAD_TASKS:
        target_dir = MODELS_DIR / target_subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / local_name

        if target_file.exists():
            print(f"[SKIP] {local_name} already exists")
            continue

        print(f"[INFO] Downloading {local_name} from {repo_id} …")
        run(
            f"huggingface-cli download {repo_id} {remote_path} --local-dir {target_dir}",
            f"Failed to download {remote_path} from {repo_id}"
        )

        # hf-cli preserves folder structure (e.g. v2/cfm_small.pth) — flatten
        downloaded_path = target_dir / remote_path
        if downloaded_path.exists() and downloaded_path != target_file:
            print(f"  Moving {downloaded_path} → {target_file}")
            shutil.move(str(downloaded_path), str(target_file))
            parent = downloaded_path.parent
            if parent != target_dir and parent.exists():
                try:
                    parent.rmdir()
                except OSError:
                    pass

    # 3. Copy campplus to seed_vc_repo root (needed for HF fallback path)
    campplus_src = MODELS_DIR / "seed_vc_v2" / "campplus_cn_common.bin"
    campplus_dst = SEED_VC_REPO_DIR / "campplus_cn_common.bin"
    if campplus_src.exists() and not campplus_dst.exists():
        print("[INFO] Copying campplus to seed_vc_repo root …")
        shutil.copy2(str(campplus_src), str(campplus_dst))

    print("\n[OK] All models downloaded.")


def verify():
    banner("Step 4: Verification")

    errors = []

    # Check torch
    try:
        import torch
        print(f"  torch {torch.__version__}  CUDA={torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            print("  [WARN] CUDA not available — inference will be very slow")
    except ImportError:
        errors.append("torch not importable")

    # Check qwen-tts
    try:
        from qwen_tts import Qwen3TTSModel
        print("  qwen-tts OK")
    except Exception as e:
        errors.append(f"qwen-tts import: {e}")

    # Check hydra + omegaconf
    try:
        from hydra.utils import instantiate
        from omegaconf import DictConfig
        print("  hydra + omegaconf OK")
    except Exception as e:
        errors.append(f"hydra/omegaconf: {e}")

    # Check Seed-VC config exists
    cfg = SEED_VC_REPO_DIR / "configs" / "v2" / "vc_wrapper.yaml"
    if cfg.exists():
        print(f"  Seed-VC V2 config found: {cfg}")
    else:
        errors.append(f"Seed-VC V2 config missing: {cfg}")

    # Check model files
    for name, path in [
        ("Qwen model", MODELS_DIR / "qwen_voice_design" / "model.safetensors"),
        ("CFM checkpoint", MODELS_DIR / "seed_vc_v2" / "cfm_small.pth"),
        ("AR checkpoint", MODELS_DIR / "seed_vc_v2" / "ar_base.pth"),
        ("CamPPlus", MODELS_DIR / "seed_vc_v2" / "campplus_cn_common.bin"),
    ]:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  {name}: {path.name} ({size_mb:.0f} MB)")
        else:
            errors.append(f"{name} not found: {path}")

    if errors:
        print("\n[WARN] Issues found:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\n[OK] Everything looks good!")

    print(f"\nTo start the server:\n  cd {ROOT_DIR}\n  python server.py")


if __name__ == "__main__":
    banner("Qwen3-TTS + Seed-VC V2 Setup")
    install_dependencies()
    setup_seed_vc_repo()
    download_models()
    verify()
