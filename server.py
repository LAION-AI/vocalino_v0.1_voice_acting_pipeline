"""
Voice Studio — FastAPI server combining Qwen3-TTS VoiceDesign + Seed-VC V2.

Endpoints:
  POST /tts/generate-voice-design     — Generate speech with a style description
  POST /vc/convert                    — Convert voice (source audio → target timbre)
  POST /pipeline/tts-then-vc          — TTS + VC in one call (style + identity)
  POST /voice-design/batch            — Generate N voice design samples (batched)
  POST /pipeline/ranked               — Generate K candidates, rank by speaker similarity
  POST /pipeline/ranked-stream        — Same as above but streams results via SSE
  GET  /health                        — Model status and configuration info

All audio I/O is base64-encoded WAV/MP3.

Configuration (environment variables — see README.md):
  TTS_DEVICE         GPU for TTS model       (default: cuda:0)
  VC_DEVICE          GPU for VC model         (default: same as TTS)
  SE_DEVICE          GPU for speaker encoder   (default: same as TTS)
  TTS_QUANTIZE       Quantization mode         (default: none, options: none | int8)
  DEFAULT_DIFF_STEPS Default VC diffusion steps (default: 12)
  EMB_CACHE_SIZE     Speaker embedding cache    (default: 32 entries)

Optimization tiers implemented (see OPTIMIZATION_PLAN.txt):
  Tier 1: VC steps 25→12, speaker embedding LRU cache
  Tier 2: SSE streaming — candidates arrive as they complete
  Tier 4: TTS batching — K candidates in one forward pass
  Tier 5: INT8 quantization (opt-in), multi-GPU device split (opt-in)
"""

import sys
import os
import traceback
import tempfile
import uuid
import hashlib
import json as json_module
import asyncio
import time

# ── Resolve paths before any relative imports ──────────────────────────
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SEED_VC_REPO = ROOT_DIR / "seed_vc_repo"
MODELS_DIR = ROOT_DIR / "models"

# Seed-VC modules must be importable (Hydra resolves them by dotted path)
sys.path.insert(0, str(SEED_VC_REPO))

# ── Standard imports ───────────────────────────────────────────────────
import torch
import numpy as np
import soundfile as sf
import base64
import io
import gc
import yaml
import librosa
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── Qwen TTS ───────────────────────────────────────────────────────────
from qwen_tts import Qwen3TTSModel

# ── Seed-VC V2 (loaded via Hydra from the repo's config) ──────────────
from hydra.utils import instantiate
from omegaconf import DictConfig

# ── Speaker encoder for similarity ranking ─────────────────────────────
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSSpeakerEncoder,
    mel_spectrogram,
)

# =====================================================================
# Configuration — all tunable via environment variables
# =====================================================================
#
# Device assignment:
#   TTS_DEVICE   — where to load the Qwen3-TTS model (default: "cuda:0")
#   VC_DEVICE    — where to load Seed-VC V2 (default: same as TTS_DEVICE)
#   SE_DEVICE    — where to load ECAPA-TDNN speaker encoder (default: same as TTS_DEVICE)
#
# When a second GPU is available, set VC_DEVICE=cuda:1 to split models
# across GPUs.  This reduces VRAM pressure and allows concurrent request
# handling without GPU memory contention.
#
# Quantization:
#   TTS_QUANTIZE — "none" (default) or "int8" (bitsandbytes INT8).
#   INT8 reduces TTS VRAM from ~15 GB to ~7 GB with minimal quality loss.
#   Requires: pip install bitsandbytes
#
# Other:
#   DEFAULT_DIFF_STEPS — default VC diffusion steps (default: 12)
#   EMB_CACHE_SIZE     — max speaker embedding cache entries (default: 32)

_CFG_TTS_DEVICE = os.environ.get("TTS_DEVICE", "cuda:0")
_CFG_VC_DEVICE = os.environ.get("VC_DEVICE", "")       # empty = same as TTS
_CFG_SE_DEVICE = os.environ.get("SE_DEVICE", "")       # empty = same as TTS
_CFG_TTS_QUANTIZE = os.environ.get("TTS_QUANTIZE", "none").lower()
_CFG_DEFAULT_DIFF_STEPS = int(os.environ.get("DEFAULT_DIFF_STEPS", "12"))
_CFG_EMB_CACHE_SIZE = int(os.environ.get("EMB_CACHE_SIZE", "32"))

# ── Resolve devices ───────────────────────────────────────────────────
# Falls back to CPU if CUDA is unavailable.

def _resolve_device(cfg_val: str, fallback: str) -> torch.device:
    """Resolve a device string like 'cuda:0' to a torch.device."""
    name = cfg_val or fallback
    if name.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] CUDA unavailable, falling back to CPU (requested {name})")
        return torch.device("cpu")
    return torch.device(name)

_tts_device = _resolve_device(_CFG_TTS_DEVICE, "cuda:0")
_vc_device = _resolve_device(_CFG_VC_DEVICE, str(_tts_device))
_se_device = _resolve_device(_CFG_SE_DEVICE, str(_tts_device))
_vc_dtype = torch.float16   # Seed-VC uses float16 for CFM diffusion

# ── Globals ────────────────────────────────────────────────────────────
_qwen_model: Optional[Qwen3TTSModel] = None
_vc_wrapper = None  # VoiceConversionWrapper instance
_speaker_encoder: Optional[Qwen3TTSSpeakerEncoder] = None
_SPEAKER_ENCODER_SR = 24000

# ── Speaker embedding LRU cache ───────────────────────────────────────
# Caches embeddings keyed by SHA-256 of the audio bytes so repeated
# requests with the same reference audio skip the mel + ECAPA-TDNN pass.
_embedding_cache: dict[str, torch.Tensor] = {}


# =====================================================================
# Model loading
# =====================================================================

def _load_qwen():
    """Load Qwen3-TTS VoiceDesign model onto _tts_device.

    Supports optional INT8 quantization via TTS_QUANTIZE=int8 env var.
    INT8 uses bitsandbytes to quantize linear layers to 8-bit integers,
    reducing VRAM from ~15 GB (bfloat16) to ~7 GB with minimal quality
    impact on codec token generation.
    """
    global _qwen_model
    model_path = MODELS_DIR / "qwen_voice_design"
    if not model_path.exists():
        print(f"[WARN] Qwen model dir not found at {model_path}, will download from HF")
        model_path = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    else:
        model_path = str(model_path)

    # Build kwargs for from_pretrained based on quantization config
    load_kwargs = {}
    if _CFG_TTS_QUANTIZE == "int8":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,  # default outlier threshold
        )
        # bitsandbytes requires device_map="auto" or a specific device string
        load_kwargs["device_map"] = str(_tts_device)
        print(f"[INFO] Loading Qwen3-TTS VoiceDesign (INT8) from {model_path} "
              f"onto {_tts_device} …")
    else:
        load_kwargs["device_map"] = str(_tts_device)
        load_kwargs["dtype"] = torch.bfloat16
        print(f"[INFO] Loading Qwen3-TTS VoiceDesign (bfloat16) from {model_path} "
              f"onto {_tts_device} …")

    _qwen_model = Qwen3TTSModel.from_pretrained(model_path, **load_kwargs)

    quant_label = "INT8" if _CFG_TTS_QUANTIZE == "int8" else "bfloat16"
    print(f"[INFO] Qwen3-TTS VoiceDesign loaded ({quant_label}, {_tts_device}).")


def _load_seed_vc():
    """Instantiate VoiceConversionWrapper via Hydra, then load checkpoints.

    The VC model is loaded onto _vc_device, which can be a different GPU
    than the TTS model (set VC_DEVICE=cuda:1 to split).  This reduces
    VRAM contention and allows future pipeline parallelism.
    """
    global _vc_wrapper

    config_path = SEED_VC_REPO / "configs" / "v2" / "vc_wrapper.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Seed-VC V2 config not found: {config_path}")

    print(f"[INFO] Instantiating Seed-VC V2 from {config_path} …")

    # We must be in seed_vc_repo dir so that relative Hydra targets resolve
    prev_cwd = os.getcwd()
    os.chdir(str(SEED_VC_REPO))
    try:
        cfg = DictConfig(yaml.safe_load(open(str(config_path), "r")))
        _vc_wrapper = instantiate(cfg)
    finally:
        os.chdir(prev_cwd)

    # Checkpoint paths — prefer local downloads, fall back to auto-download
    cfm_path = MODELS_DIR / "seed_vc_v2" / "cfm_small.pth"
    ar_path = MODELS_DIR / "seed_vc_v2" / "ar_base.pth"

    cfm_arg = str(cfm_path) if cfm_path.exists() else None
    ar_arg = str(ar_path) if ar_path.exists() else None

    print(f"[INFO] Loading Seed-VC checkpoints (cfm={cfm_arg}, ar={ar_arg}) …")

    prev_cwd = os.getcwd()
    os.chdir(str(SEED_VC_REPO))
    try:
        _vc_wrapper.load_checkpoints(
            cfm_checkpoint_path=cfm_arg,
            ar_checkpoint_path=ar_arg,
        )
    finally:
        os.chdir(prev_cwd)

    _vc_wrapper.to(_vc_device)
    _vc_wrapper.eval()
    _vc_wrapper.setup_ar_caches(
        max_batch_size=1, max_seq_len=4096,
        dtype=_vc_dtype, device=_vc_device,
    )

    print(f"[INFO] Seed-VC V2 loaded and ready ({_vc_device}).")


def _load_speaker_encoder():
    """Load ECAPA-TDNN speaker encoder from Qwen3-TTS-Base for similarity ranking.

    The speaker encoder is a small CNN (~20 MB) that produces 2048-dim
    embeddings.  We extract its weights from the Base model's safetensors
    file (prefix "speaker_encoder.*") instead of loading the full 1.7B model.
    """
    global _speaker_encoder
    from safetensors import safe_open

    # Try HF hub (or local cache)
    from huggingface_hub import hf_hub_download
    try:
        safetensors_path = hf_hub_download(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "model.safetensors"
        )
    except Exception:
        # Fallback: local fine-tuned checkpoint
        local_path = ROOT_DIR / "voice-pipeline" / "finetune_output_base" / "epoch-2" / "model.safetensors"
        if local_path.exists():
            safetensors_path = str(local_path)
        else:
            raise RuntimeError("Cannot find Base model safetensors for speaker encoder")

    print(f"[INFO] Loading speaker encoder from {safetensors_path} …")

    config = Qwen3TTSSpeakerEncoderConfig(enc_dim=2048, sample_rate=24000)
    _speaker_encoder = Qwen3TTSSpeakerEncoder(config)

    # Extract only the speaker_encoder.* tensors from the full checkpoint.
    # This avoids loading the entire 1.7B model just for the small ECAPA-TDNN.
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
        se_prefix = None
        for prefix in ["speaker_encoder.", "model.speaker_encoder."]:
            if any(k.startswith(prefix) for k in all_keys):
                se_prefix = prefix
                break
        if se_prefix is None:
            raise RuntimeError("No speaker_encoder keys found in safetensors")

        se_keys = [k for k in all_keys if k.startswith(se_prefix)]
        state_dict = {k[len(se_prefix):]: f.get_tensor(k) for k in se_keys}

    _speaker_encoder.load_state_dict(state_dict)
    _speaker_encoder.to(_se_device).to(torch.bfloat16)
    _speaker_encoder.eval()
    print(f"[INFO] Speaker encoder loaded ({len(state_dict)} tensors, "
          f"enc_dim=2048, {_se_device}).")


# =====================================================================
# Lifespan
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Print configuration banner ────────────────────────────────────
    n_gpus = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]

    print("=" * 64)
    print("  Voice Studio — Starting Server")
    print("=" * 64)
    print(f"  GPUs available : {n_gpus} ({', '.join(gpu_names) or 'none'})")
    print(f"  TTS device     : {_tts_device}")
    print(f"  VC device      : {_vc_device}")
    print(f"  SE device      : {_se_device}")
    print(f"  TTS quantize   : {_CFG_TTS_QUANTIZE}")
    print(f"  VC diff steps  : {_CFG_DEFAULT_DIFF_STEPS}")
    print(f"  Embed cache    : {_CFG_EMB_CACHE_SIZE} entries")
    multi_gpu = (_tts_device != _vc_device)
    if multi_gpu:
        print(f"  Multi-GPU      : YES (TTS={_tts_device}, VC={_vc_device})")
    print("=" * 64)

    # ── Load models ───────────────────────────────────────────────────
    try:
        _load_qwen()
    except Exception:
        print("[ERROR] Failed to load Qwen3-TTS VoiceDesign:")
        traceback.print_exc()

    try:
        _load_seed_vc()
    except Exception:
        print("[ERROR] Failed to load Seed-VC V2:")
        traceback.print_exc()

    try:
        _load_speaker_encoder()
    except Exception:
        print("[ERROR] Failed to load speaker encoder:")
        traceback.print_exc()

    # ── Print VRAM usage after loading ────────────────────────────────
    for i in range(n_gpus):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i} VRAM: {alloc:.1f} / {total:.1f} GB")

    print("=" * 64)
    print("  Models loaded. Server ready.")
    print("=" * 64)
    yield

    # ── Shutdown cleanup ──────────────────────────────────────────────
    print("[INFO] Shutting down — releasing GPU memory …")
    global _qwen_model, _vc_wrapper, _speaker_encoder
    _qwen_model = None
    _vc_wrapper = None
    _speaker_encoder = None
    gc.collect()
    torch.cuda.empty_cache()


# =====================================================================
# FastAPI app
# =====================================================================

app = FastAPI(
    title="Qwen3-TTS + Seed-VC Voice Server",
    description=(
        "Generate expressive speech with Qwen3-TTS Voice Design, "
        "then optionally convert it to a target identity with Seed-VC V2."
    ),
    lifespan=lifespan,
)

# CORS — allow the UI to call from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# Helpers
# =====================================================================

def audio_to_base64(audio: np.ndarray, sr: int) -> str:
    """Encode a numpy waveform as a base64 WAV string."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def base64_to_wav_path(b64: str, tmp_dir: str, name: str = "audio") -> str:
    """Decode a base64 audio string (WAV or MP3) and write it as a WAV file."""
    raw = base64.b64decode(b64)
    buf = io.BytesIO(raw)
    try:
        data, sr = sf.read(buf)
    except Exception:
        from pydub import AudioSegment
        buf.seek(0)
        seg = AudioSegment.from_file(buf)
        sr = seg.frame_rate
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        if seg.channels > 1:
            samples = samples.reshape(-1, seg.channels).mean(axis=1)
        data = samples / (2 ** (seg.sample_width * 8 - 1))

    path = os.path.join(tmp_dir, f"{name}_{uuid.uuid4().hex[:8]}.wav")
    sf.write(path, data, sr)
    return path


def base64_to_numpy(b64: str):
    """Decode base64 audio to (mono_numpy_array, sample_rate)."""
    raw = base64.b64decode(b64)
    buf = io.BytesIO(raw)
    try:
        data, sr = sf.read(buf)
    except Exception:
        from pydub import AudioSegment
        buf.seek(0)
        seg = AudioSegment.from_file(buf)
        sr = seg.frame_rate
        samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
        if seg.channels > 1:
            samples = samples.reshape(-1, seg.channels).mean(axis=1)
        data = samples / (2 ** (seg.sample_width * 8 - 1))
    # Ensure mono
    if data.ndim > 1:
        data = data.mean(axis=-1)
    return data.astype(np.float32), sr


def extract_speaker_embedding(audio_np: np.ndarray, sr: int,
                               cache_key: str = None) -> torch.Tensor:
    """Extract 2048-dim speaker embedding using ECAPA-TDNN encoder.

    Args:
        audio_np: Mono or stereo audio as numpy array.
        sr: Sample rate of the input audio.
        cache_key: Optional hash key for LRU caching. If provided and the
                   embedding is already cached, the GPU computation is skipped.

    Returns:
        torch.Tensor of shape (2048,) — the speaker embedding vector.
    """
    if _speaker_encoder is None:
        raise RuntimeError("Speaker encoder not loaded")

    # ── Check cache first ──────────────────────────────────────────────
    if cache_key and cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    # ── Ensure mono float32 ────────────────────────────────────────────
    audio_np = audio_np.astype(np.float32)
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=-1)

    # ── Resample to 24 kHz (speaker encoder's native rate) ─────────────
    if sr != _SPEAKER_ENCODER_SR:
        audio_np = librosa.resample(
            audio_np, orig_sr=sr, target_sr=_SPEAKER_ENCODER_SR
        )

    # ── Compute mel spectrogram → ECAPA-TDNN forward pass ──────────────
    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)  # (1, T)
    mels = mel_spectrogram(
        audio_tensor, n_fft=1024, num_mels=128, sampling_rate=24000,
        hop_size=256, win_size=1024, fmin=0, fmax=12000,
    )
    mels = mels.transpose(1, 2)  # (1, T, 128)

    with torch.no_grad():
        emb = _speaker_encoder(mels.to(_se_device).to(torch.bfloat16))[0]

    # ── Store in cache (evict oldest if full) ──────────────────────────
    if cache_key:
        if len(_embedding_cache) >= _CFG_EMB_CACHE_SIZE:
            # Remove the oldest entry (first inserted key)
            oldest = next(iter(_embedding_cache))
            del _embedding_cache[oldest]
        _embedding_cache[cache_key] = emb

    return emb


def run_voice_conversion(source_audio: np.ndarray, source_sr: int,
                         target_path: str, vc_params: dict):
    """Run Seed-VC voice conversion on _vc_device. Returns (audio_np, sample_rate).

    This is a synchronous helper that writes the source audio to a temp WAV,
    runs the Seed-VC streaming pipeline (which yields chunks), and returns
    the final full-quality output.
    """
    import shutil
    tmp_dir = tempfile.mkdtemp(prefix="vc_helper_")
    try:
        src_path = os.path.join(tmp_dir, "source.wav")
        sf.write(src_path, source_audio, source_sr)

        full_audio = None
        for _mp3_chunk, chunk_full in _vc_wrapper.convert_voice_with_streaming(
            source_audio_path=src_path,
            target_audio_path=target_path,
            diffusion_steps=vc_params.get("diffusion_steps", _CFG_DEFAULT_DIFF_STEPS),
            length_adjust=vc_params.get("length_adjust", 1.0),
            intelligebility_cfg_rate=vc_params.get("intelligibility_cfg_rate", 0.7),
            similarity_cfg_rate=vc_params.get("similarity_cfg_rate", 0.7),
            top_p=vc_params.get("top_p", 0.9),
            temperature=vc_params.get("temperature", 1.0),
            repetition_penalty=vc_params.get("repetition_penalty", 1.0),
            convert_style=vc_params.get("convert_style", False),
            anonymization_only=vc_params.get("anonymization_only", False),
            device=_vc_device,
            dtype=_vc_dtype,
            stream_output=True,
        ):
            if chunk_full is not None:
                full_audio = chunk_full

        if full_audio is None:
            raise RuntimeError("Seed-VC produced no output")

        out_sr, out_wav = full_audio
        return out_wav, out_sr
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# =====================================================================
# Request / Response schemas
# =====================================================================

class TTSRequest(BaseModel):
    """Generate speech with Qwen3-TTS Voice Design."""
    text: str = Field(..., description="Text to synthesize")
    style_prompt: str = Field(..., description="Natural-language description of desired voice/style")
    language: str = Field("Auto", description="Language code (Auto, Chinese, English, Japanese, …)")

    max_new_tokens: Optional[int] = Field(None, description="Max new codec tokens")
    top_p: Optional[float] = Field(None, description="Top-p sampling")
    temperature: Optional[float] = Field(None, description="Sampling temperature")
    repetition_penalty: Optional[float] = Field(None, description="Repetition penalty")


class VCRequest(BaseModel):
    """Voice conversion with Seed-VC V2."""
    source_audio_base64: str = Field(..., description="Base64-encoded source audio (WAV or MP3)")
    target_audio_base64: str = Field(..., description="Base64-encoded target/reference audio")

    diffusion_steps: int = Field(12, ge=1, le=100)
    length_adjust: float = Field(1.0, ge=0.5, le=2.0)
    intelligibility_cfg_rate: float = Field(0.7, ge=0.0, le=1.0)
    similarity_cfg_rate: float = Field(0.7, ge=0.0, le=1.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    temperature: float = Field(1.0, ge=0.1, le=2.0)
    repetition_penalty: float = Field(1.0, ge=1.0, le=2.0)
    convert_style: bool = Field(True)
    anonymization_only: bool = Field(False)


class CombinedRequest(BaseModel):
    """Full pipeline: TTS Voice Design → Seed-VC conversion to target identity."""
    text: str = Field(..., description="Text to synthesize")
    style_prompt: str = Field(..., description="How to say it (emotion, style, etc.)")
    target_audio_base64: str = Field(..., description="Reference audio for the target speaker identity")
    language: str = Field("Auto")

    tts_max_new_tokens: Optional[int] = None
    tts_top_p: Optional[float] = None
    tts_temperature: Optional[float] = None
    tts_repetition_penalty: Optional[float] = None

    diffusion_steps: int = Field(12, ge=1, le=100)
    length_adjust: float = Field(1.0, ge=0.5, le=2.0)
    intelligibility_cfg_rate: float = Field(0.7, ge=0.0, le=1.0)
    similarity_cfg_rate: float = Field(0.7, ge=0.0, le=1.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    temperature: float = Field(1.0, ge=0.1, le=2.0)
    repetition_penalty: float = Field(1.0, ge=1.0, le=2.0)
    convert_style: bool = Field(False)
    anonymization_only: bool = Field(False)
    return_intermediate: bool = Field(False)


class BatchTTSRequest(BaseModel):
    """Generate N voice design samples."""
    text: str = Field(..., description="Text to synthesize")
    style_prompt: str = Field(..., description="Natural-language style description")
    language: str = Field("Auto")
    n_samples: int = Field(3, ge=1, le=10, description="Number of samples to generate")

    max_new_tokens: Optional[int] = None
    top_p: Optional[float] = None
    temperature: Optional[float] = None
    repetition_penalty: Optional[float] = None


class RankedPipelineRequest(BaseModel):
    """Full pipeline with K candidates ranked by speaker similarity."""
    text: str = Field(..., description="Text to synthesize")
    style_prompt: str = Field(..., description="Emotion/style instruction")
    reference_audio_base64: str = Field(..., description="Reference audio for voice consistency")
    language: str = Field("Auto")
    k_candidates: int = Field(3, ge=1, le=10, description="Number of candidates")

    tts_max_new_tokens: Optional[int] = None
    tts_top_p: Optional[float] = None
    tts_temperature: Optional[float] = None
    tts_repetition_penalty: Optional[float] = None

    diffusion_steps: int = Field(12, ge=1, le=100)
    length_adjust: float = Field(1.0, ge=0.5, le=2.0)
    intelligibility_cfg_rate: float = Field(0.7, ge=0.0, le=1.0)
    similarity_cfg_rate: float = Field(0.7, ge=0.0, le=1.0)
    vc_top_p: float = Field(0.9, ge=0.0, le=1.0)
    vc_temperature: float = Field(1.0, ge=0.1, le=2.0)
    vc_repetition_penalty: float = Field(1.0, ge=1.0, le=2.0)


# =====================================================================
# Endpoints
# =====================================================================

@app.post("/tts/generate-voice-design")
async def tts_generate(req: TTSRequest):
    """Generate speech using Qwen3-TTS Voice Design."""
    if _qwen_model is None:
        raise HTTPException(500, "Qwen3-TTS model not loaded")

    print(f"[TTS] text={req.text!r:.60} style={req.style_prompt!r:.60}")

    gen_kwargs = {}
    if req.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = req.max_new_tokens
    if req.top_p is not None:
        gen_kwargs["top_p"] = req.top_p
    if req.temperature is not None:
        gen_kwargs["temperature"] = req.temperature
    if req.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = req.repetition_penalty

    try:
        wavs, sr = _qwen_model.generate_voice_design(
            text=req.text,
            instruct=req.style_prompt,
            language=req.language,
            **gen_kwargs,
        )
        audio = wavs[0]
        return {
            "status": "success",
            "sample_rate": sr,
            "audio_base64": audio_to_base64(audio, sr),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"TTS failed: {e}")


@app.post("/vc/convert")
async def vc_convert(req: VCRequest):
    """Voice conversion with Seed-VC V2."""
    if _vc_wrapper is None:
        raise HTTPException(500, "Seed-VC V2 model not loaded")

    print(f"[VC] diffusion_steps={req.diffusion_steps} convert_style={req.convert_style}")

    tmp_dir = tempfile.mkdtemp(prefix="vc_")
    try:
        src_path = base64_to_wav_path(req.source_audio_base64, tmp_dir, "source")
        tgt_path = base64_to_wav_path(req.target_audio_base64, tmp_dir, "target")

        full_audio = None
        for _mp3_chunk, chunk_full in _vc_wrapper.convert_voice_with_streaming(
            source_audio_path=src_path,
            target_audio_path=tgt_path,
            diffusion_steps=req.diffusion_steps,
            length_adjust=req.length_adjust,
            intelligebility_cfg_rate=req.intelligibility_cfg_rate,
            similarity_cfg_rate=req.similarity_cfg_rate,
            top_p=req.top_p,
            temperature=req.temperature,
            repetition_penalty=req.repetition_penalty,
            convert_style=req.convert_style,
            anonymization_only=req.anonymization_only,
            device=_vc_device,
            dtype=_vc_dtype,
            stream_output=True,
        ):
            if chunk_full is not None:
                full_audio = chunk_full

        if full_audio is None:
            raise RuntimeError("Seed-VC produced no output")

        out_sr, out_wav = full_audio
        return {
            "status": "success",
            "sample_rate": out_sr,
            "audio_base64": audio_to_base64(out_wav, out_sr),
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Voice conversion failed: {e}")
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/pipeline/tts-then-vc")
async def pipeline_tts_then_vc(req: CombinedRequest):
    """Full pipeline: generate styled speech, then convert to target identity."""
    if _qwen_model is None:
        raise HTTPException(500, "Qwen3-TTS model not loaded")
    if _vc_wrapper is None:
        raise HTTPException(500, "Seed-VC V2 model not loaded")

    print(f"[PIPELINE] text={req.text!r:.60} style={req.style_prompt!r:.60}")

    gen_kwargs = {}
    if req.tts_max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = req.tts_max_new_tokens
    if req.tts_top_p is not None:
        gen_kwargs["top_p"] = req.tts_top_p
    if req.tts_temperature is not None:
        gen_kwargs["temperature"] = req.tts_temperature
    if req.tts_repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = req.tts_repetition_penalty

    try:
        wavs, tts_sr = _qwen_model.generate_voice_design(
            text=req.text,
            instruct=req.style_prompt,
            language=req.language,
            **gen_kwargs,
        )
        tts_audio = wavs[0]
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"TTS step failed: {e}")

    intermediate_b64 = audio_to_base64(tts_audio, tts_sr) if req.return_intermediate else None

    tmp_dir = tempfile.mkdtemp(prefix="pipeline_")
    try:
        src_path = os.path.join(tmp_dir, "tts_source.wav")
        sf.write(src_path, tts_audio, tts_sr)
        tgt_path = base64_to_wav_path(req.target_audio_base64, tmp_dir, "target")

        full_audio = None
        for _mp3_chunk, chunk_full in _vc_wrapper.convert_voice_with_streaming(
            source_audio_path=src_path,
            target_audio_path=tgt_path,
            diffusion_steps=req.diffusion_steps,
            length_adjust=req.length_adjust,
            intelligebility_cfg_rate=req.intelligibility_cfg_rate,
            similarity_cfg_rate=req.similarity_cfg_rate,
            top_p=req.top_p,
            temperature=req.temperature,
            repetition_penalty=req.repetition_penalty,
            convert_style=req.convert_style,
            anonymization_only=req.anonymization_only,
            device=_vc_device,
            dtype=_vc_dtype,
            stream_output=True,
        ):
            if chunk_full is not None:
                full_audio = chunk_full

        if full_audio is None:
            raise RuntimeError("Seed-VC produced no output")

        out_sr, out_wav = full_audio

        result = {
            "status": "success",
            "sample_rate": out_sr,
            "audio_base64": audio_to_base64(out_wav, out_sr),
        }
        if intermediate_b64 is not None:
            result["intermediate_tts_audio_base64"] = intermediate_b64
            result["intermediate_tts_sample_rate"] = tts_sr

        return result

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"VC step failed: {e}")
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── New: Batch Voice Design ───────────────────────────────────────────

@app.post("/voice-design/batch")
async def voice_design_batch(req: BatchTTSRequest):
    """Generate N voice design samples in a single batched TTS call.

    Optimization: Instead of N sequential generate_voice_design() calls,
    we pass N copies of the text/instruct/language as lists.  The model
    tokenizes each, pads to equal length, and runs one batched forward
    pass through the autoregressive decoder.  Each output differs because
    sampling is stochastic (do_sample=True, temperature > 0).
    """
    if _qwen_model is None:
        raise HTTPException(500, "Qwen3-TTS model not loaded")

    N = req.n_samples
    print(f"[BATCH-TTS] n={N} text={req.text!r:.60} style={req.style_prompt!r:.60}")

    gen_kwargs = {}
    if req.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = req.max_new_tokens
    if req.top_p is not None:
        gen_kwargs["top_p"] = req.top_p
    if req.temperature is not None:
        gen_kwargs["temperature"] = req.temperature
    if req.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = req.repetition_penalty

    try:
        t0 = time.time()
        # Batched generation: pass N identical prompts as lists.
        # The model processes them in a single forward pass.
        wavs, sr = _qwen_model.generate_voice_design(
            text=[req.text] * N,
            instruct=[req.style_prompt] * N,
            language=[req.language] * N,
            **gen_kwargs,
        )
        batch_time = time.time() - t0
        print(f"  [BATCH-TTS] Generated {len(wavs)} samples in {batch_time:.1f}s")

        samples = []
        for i, audio in enumerate(wavs):
            samples.append({
                "id": uuid.uuid4().hex[:8],
                "index": i,
                "audio_base64": audio_to_base64(audio, sr),
                "sample_rate": sr,
                "duration": round(len(audio) / sr, 1),
            })
            print(f"    [{i+1}/{N}] {len(audio)/sr:.1f}s")

        return {"status": "success", "samples": samples, "batch_time": round(batch_time, 1)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Batch TTS failed: {e}")


# ── Ranked Pipeline (shared logic) ────────────────────────────────────
#
# _run_ranked_pipeline() is the core generator used by both the batch
# endpoint (/pipeline/ranked) and the SSE streaming endpoint
# (/pipeline/ranked-stream).  It yields one dict per candidate as each
# completes, so SSE can push results incrementally.

def _build_gen_kwargs(req):
    """Extract optional TTS generation kwargs from a request object."""
    kw = {}
    if req.tts_max_new_tokens is not None:
        kw["max_new_tokens"] = req.tts_max_new_tokens
    if req.tts_top_p is not None:
        kw["top_p"] = req.tts_top_p
    if req.tts_temperature is not None:
        kw["temperature"] = req.tts_temperature
    if req.tts_repetition_penalty is not None:
        kw["repetition_penalty"] = req.tts_repetition_penalty
    return kw


def _run_ranked_pipeline(req: RankedPipelineRequest):
    """Core generator: batch TTS, sequential VC, yield candidates progressively.

    Optimization strategy (Tier 4 — TTS batching):
      Instead of K sequential TTS calls (~45s each = ~135s for K=3), we pass
      K copies of the same text/instruct/language to generate_voice_design()
      which processes them in a single batched forward pass through the
      transformer.  Each output differs due to stochastic sampling
      (do_sample=True, temperature=0.9).  This reduces the TTS phase from
      ~135s to ~60-80s for K=3.

    After the batch TTS, we run voice conversion sequentially on each output
    (VC can't easily be batched since Seed-VC's streaming API is single-input).
    Each candidate is yielded as it completes for SSE streaming compatibility.

    Yields:
        dict: One of three types:
          - Progress event: {"_progress": "tts_batch_complete", "count": K, ...}
          - Candidate:      {"id": ..., "audio_base64": ..., "similarity": ..., ...}
          - Error:          {"id": ..., "index": ..., "error": "..."}
    """
    t0 = time.time()
    K = req.k_candidates

    # ── Prepare reference audio and speaker embedding ──────────────────
    tmp_dir = tempfile.mkdtemp(prefix="ranked_")
    try:
        ref_path = base64_to_wav_path(
            req.reference_audio_base64, tmp_dir, "reference"
        )
        ref_audio, ref_sr = base64_to_numpy(req.reference_audio_base64)

        # Cache key = SHA-256 of the raw base64 bytes so repeated calls
        # with the same reference skip the ECAPA-TDNN forward pass.
        ref_hash = hashlib.sha256(
            req.reference_audio_base64.encode()
        ).hexdigest()
        ref_embedding = extract_speaker_embedding(
            ref_audio, ref_sr, cache_key=ref_hash
        )

        gen_kwargs = _build_gen_kwargs(req)
        vc_params = {
            "diffusion_steps": req.diffusion_steps,
            "length_adjust": req.length_adjust,
            "intelligibility_cfg_rate": req.intelligibility_cfg_rate,
            "similarity_cfg_rate": req.similarity_cfg_rate,
            "top_p": req.vc_top_p,
            "temperature": req.vc_temperature,
            "repetition_penalty": req.vc_repetition_penalty,
            "convert_style": False,
            "anonymization_only": False,
        }

        # ── Phase 1: Batched TTS Generation ───────────────────────────
        # Pass K copies of the same text to generate_voice_design().
        # The model tokenizes each, pads to equal length, and runs a
        # single batched forward pass through the autoregressive decoder.
        # Each output differs because sampling is stochastic.
        print(f"  [TTS-BATCH] Generating {K} candidates in one batch call...")
        t_tts = time.time()

        tts_wavs, tts_sr = _qwen_model.generate_voice_design(
            text=[req.text] * K,
            instruct=[req.style_prompt] * K,
            language=[req.language] * K,
            **gen_kwargs,
        )

        tts_time = time.time() - t_tts
        durations_str = ", ".join(f"{len(w)/tts_sr:.1f}s" for w in tts_wavs)
        print(f"  [TTS-BATCH] Done: {K} outputs in {tts_time:.1f}s "
              f"({durations_str})")

        # Notify SSE clients that the TTS phase is complete so they can
        # update their UI (e.g. "Converting voices 1/3...")
        yield {
            "_progress": "tts_batch_complete",
            "count": K,
            "tts_time": round(tts_time, 1),
        }

        # ── Phase 2: Sequential VC + Embedding ────────────────────────
        # Voice conversion uses Seed-VC (CFM diffusion + AR) which is a
        # different model than TTS.  Each VC takes ~8s at 12 diffusion steps.
        # We run them one at a time and yield each candidate as it completes.
        for i in range(K):
            cid = uuid.uuid4().hex[:8]
            tc = time.time()
            try:
                # Voice conversion (CFM diffusion, N steps)
                t_vc = time.time()
                vc_audio, vc_sr = run_voice_conversion(
                    tts_wavs[i], tts_sr, ref_path, vc_params
                )
                vc_time = time.time() - t_vc
                print(f"  [{i+1}/{K}] VC: {len(vc_audio)/vc_sr:.1f}s "
                      f"in {vc_time:.1f}s")

                # Speaker similarity (ECAPA-TDNN cosine distance)
                gen_embedding = extract_speaker_embedding(vc_audio, vc_sr)
                sim = torch.nn.functional.cosine_similarity(
                    ref_embedding.unsqueeze(0), gen_embedding.unsqueeze(0)
                ).item()
                print(f"  [{i+1}/{K}] sim={sim:.4f} "
                      f"(vc+emb: {time.time()-tc:.1f}s)")

                yield {
                    "id": cid,
                    "index": i,
                    "audio_base64": audio_to_base64(vc_audio, vc_sr),
                    "intermediate_tts_base64": audio_to_base64(tts_wavs[i], tts_sr),
                    "sample_rate": vc_sr,
                    "tts_sample_rate": tts_sr,
                    "similarity": round(sim, 4),
                    "duration": round(len(vc_audio) / vc_sr, 1),
                }
            except Exception as e:
                traceback.print_exc()
                yield {"id": cid, "index": i, "error": str(e)}

        print(f"[RANKED] Total: {time.time()-t0:.1f}s "
              f"(TTS batch: {tts_time:.1f}s + VC: {time.time()-t0-tts_time:.1f}s)")
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Ranked Pipeline — batch mode (returns all at once) ────────────────

@app.post("/pipeline/ranked")
async def pipeline_ranked(req: RankedPipelineRequest):
    """Generate K candidates via TTS+VC, rank by speaker similarity.

    Returns all candidates sorted by similarity after the last one completes.
    For real-time progressive results, use /pipeline/ranked-stream instead.
    """
    if _qwen_model is None:
        raise HTTPException(500, "Qwen3-TTS model not loaded")
    if _vc_wrapper is None:
        raise HTTPException(500, "Seed-VC V2 model not loaded")
    if _speaker_encoder is None:
        raise HTTPException(500, "Speaker encoder not loaded")

    print(f"[RANKED] k={req.k_candidates} text={req.text!r:.60} "
          f"style={req.style_prompt!r:.60}")

    # Collect all candidates from the shared generator, skipping progress events
    candidates = [c for c in _run_ranked_pipeline(req) if "_progress" not in c]

    # Sort valid candidates by similarity (highest first)
    valid = [c for c in candidates if "error" not in c]
    valid.sort(key=lambda x: x["similarity"], reverse=True)
    errors = [c for c in candidates if "error" in c]

    return {
        "status": "success",
        "candidates": valid + errors,
        "best_id": valid[0]["id"] if valid else None,
    }


# ── Ranked Pipeline — SSE streaming mode ──────────────────────────────
#
# Server-Sent Events (SSE) let the client receive each candidate the
# moment it finishes generating, instead of waiting for all K.
#
# Event types sent:
#   "progress"   — intermediate status (e.g. TTS batch complete)
#   "candidate"  — one candidate dict (audio, similarity, etc.)
#   "done"       — final summary with sorted candidates and best_id
#   "error"      — top-level error (model not loaded, etc.)
#
# The client opens a POST fetch and reads the response body as a stream
# of "data: ...\n\n" lines, parsing each JSON payload.

@app.post("/pipeline/ranked-stream")
async def pipeline_ranked_stream(req: RankedPipelineRequest):
    """Stream K candidates via SSE as each completes (TTS+VC+ranking).

    Returns text/event-stream.  Each event is a JSON payload:
      event: candidate\ndata: {...}\n\n   — one completed candidate
      event: done\ndata: {...}\n\n        — final summary with best_id
    """
    if _qwen_model is None:
        raise HTTPException(500, "Qwen3-TTS model not loaded")
    if _vc_wrapper is None:
        raise HTTPException(500, "Seed-VC V2 model not loaded")
    if _speaker_encoder is None:
        raise HTTPException(500, "Speaker encoder not loaded")

    print(f"[RANKED-STREAM] k={req.k_candidates} text={req.text!r:.60} "
          f"style={req.style_prompt!r:.60}")

    async def event_generator():
        """Async generator that wraps the sync pipeline and yields SSE lines.

        The core pipeline (_run_ranked_pipeline) is a blocking synchronous
        generator that:
          1. Runs batched TTS (one long call producing K outputs)
          2. Runs sequential VC on each TTS output (~8s each)
        It yields progress events and candidate dicts.  We pull one item
        at a time via run_in_executor so the event loop can flush SSE data
        to the client between candidates.
        """
        candidates = []
        loop = asyncio.get_event_loop()
        gen = _run_ranked_pipeline(req)

        def _next(g):
            """Pull one item from the sync generator in a thread."""
            try:
                return next(g)
            except StopIteration:
                return None

        while True:
            # Run the blocking next() call in the default thread pool
            # so the event loop stays free to flush data to the client.
            item = await loop.run_in_executor(None, _next, gen)
            if item is None:
                break

            # Progress events (e.g. "tts_batch_complete") are forwarded
            # to the client so the UI can show intermediate status updates.
            if "_progress" in item:
                yield f"event: progress\ndata: {json_module.dumps(item)}\n\n"
            else:
                candidates.append(item)
                payload = json_module.dumps(item)
                yield f"event: candidate\ndata: {payload}\n\n"

        # ── Final summary event ────────────────────────────────────────
        valid = [c for c in candidates if "error" not in c]
        valid.sort(key=lambda x: x["similarity"], reverse=True)
        best_id = valid[0]["id"] if valid else None

        summary = {
            "status": "success",
            "best_id": best_id,
            "total": len(candidates),
            "valid": len(valid),
        }
        yield f"event: done\ndata: {json_module.dumps(summary)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",       # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


# =====================================================================
# Health check
# =====================================================================

@app.get("/health")
async def health():
    """Return model status and server configuration."""
    return {
        "qwen_loaded": _qwen_model is not None,
        "seed_vc_loaded": _vc_wrapper is not None,
        "speaker_encoder_loaded": _speaker_encoder is not None,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "tts_device": str(_tts_device),
        "vc_device": str(_vc_device),
        "se_device": str(_se_device),
        "tts_quantize": _CFG_TTS_QUANTIZE,
        "default_diff_steps": _CFG_DEFAULT_DIFF_STEPS,
        "emb_cache_size": _CFG_EMB_CACHE_SIZE,
        "emb_cache_used": len(_embedding_cache),
    }


# =====================================================================
# Static files — serve the web UI at /ui
# =====================================================================

UI_DIR = ROOT_DIR / "ui"
UI_DIR.mkdir(exist_ok=True)
app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
