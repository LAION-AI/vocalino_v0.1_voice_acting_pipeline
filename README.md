# Vocalino V 0.1: Voice Acting Pipeline
*By <a href="https://scholar.google.com/citations?user=EvrlaSAAAAAJ">Christoph Schuhmann </a>*

**The first fully open-source voice acting pipeline that combines zero-shot voice cloning with natural language performance direction.** Vocalino allows you to provide a reference voice (or generate one from scratch) and use free-form text instructions to direct *how* the line is performed. It generates speech that maintains strict voice consistency with your reference audio while adhering to your specific emotional and stylistic prompts—giving you total control over the actor and the performance without any model training.

<p align="center">
  <a href="https://www.youtube.com/watch?v=bOA9e5p1Oy0">
    <img src="https://img.youtube.com/vi/bOA9e5p1Oy0/maxresdefault.jpg" width="700">
  </a>
</p>

<p align="center">
  ▶ Click to watch demo video
</p>


## How It Works

### The Concept: "Directing" AI Speech

Vocalino v0.1 was built to solve the limitation of existing open-source audio models. Standard Text-to-Speech (TTS) can generate emotions but usually with random voices. Standard Voice Conversion (VC) can clone a specific person but requires pre-acted source audio.

To our knowledge, Vocalino is the **first open pipeline** that decouples **vocal identity** from **performance style**. By chaining advanced stylistic generation with high-fidelity voice conversion, Vocalino lets you "cast" an actor (via a reference clip) and "direct" them (via text prompts like *"whisper with trembling fear"* or *"shout with overwhelming joy"*). The result is a unified audio file that sounds like your target speaker performing exactly the way you instructed.

### Architecture

```
                     ┌────────────────────┐
    Text + Style ──> │  Qwen3-TTS 1.7B    │ ──> Raw TTS audio
                     │  (VoiceDesign)      │     (12 Hz codec tokens → wav)
                     └────────────────────┘
                              │
                              ▼
                     ┌────────────────────┐
    Reference WAV ─> │  Seed-VC V2        │ ──> Voice-converted audio
                     │  (CFM + AR)        │     (matches reference timbre)
                     └────────────────────┘
                              │
                              ▼
                     ┌────────────────────┐
                     │  ECAPA-TDNN        │ ──> 2048-dim embedding
                     │  (Speaker Encoder) │     → cosine similarity vs ref
                     └────────────────────┘
```

### Two-Stage Pipeline

**Stage 1 — Qwen3-TTS Voice Design** (by Alibaba, 1.7B parameters)

A large language model that generates speech from text + a natural-language style instruction. You describe the desired emotion, pace, energy, and pitch in plain English (e.g. *"speak with trembling fear, whispering, medium-pitched male voice"*), and the model generates audio that matches. The voice identity is random — only the style/emotion is controlled.

**Stage 2 — Seed-VC V2** (by Plachtaa / ByteDance)

A voice conversion model combining Conditional Flow Matching (CFM) diffusion with an autoregressive (AR) style transfer model. Given any source audio and a short reference clip of the target speaker:

1. **Content Extraction**: The source audio's phonetic content and prosody (rhythm, intonation, emotion) are preserved while speaker identity is removed.
2. **Speaker Embedding**: The reference clip is processed to extract a speaker embedding capturing timbre, formant structure, and vocal characteristics.
3. **CFM Diffusion Decoder**: A flow-matching generative model synthesizes a new waveform that matches the source content/prosody while sounding like the target speaker — all from just a few seconds of reference audio.

**Stage 3 — ECAPA-TDNN Ranking** (from Qwen3-TTS-Base)

When generating K candidates, each voice-converted output is scored against the reference audio using cosine similarity of 2048-dimensional speaker embeddings. Candidates are ranked so you always get the most voice-consistent result.

### Features

- **Web UI** — dark-themed browser interface served at `/ui` for interactive voice design and pipeline generation
- **Batched TTS** — generate K candidates in a single forward pass instead of K sequential calls (~2x faster)
- **SSE Streaming** — candidates stream to the UI as they complete, no waiting for all K
- **Speaker Similarity Ranking** — ECAPA-TDNN embeddings rank candidates by voice consistency
- **INT8 Quantization** — optional bitsandbytes INT8 reduces TTS VRAM from ~15 GB to ~7 GB
- **Multi-GPU** — split TTS and VC across GPUs for VRAM isolation and concurrency

---

## Project Structure

```
Vocalino-V0.1-Voice-Acting-Pipeline/
├── README.md                # This file
├── server.py                # FastAPI server (all endpoints + optimizations)
├── ui/
│   └── index.html           # Web UI (served at /ui)
├── setup.py                 # Install dependencies + download model weights
├── generate_samples.py      # Standalone: TTS+VC for multiple emotions
├── convert_voice.py         # Standalone: voice conversion only
├── OPTIMIZATION_PLAN.txt    # Detailed optimization roadmap
├── seed_vc_repo/            # Seed-VC V2 repository (cloned separately)
├── models/                  # (auto-created) local model cache
└── output/                  # Generated at runtime
```

## Requirements

### Hardware

- NVIDIA GPU with >= 24 GB VRAM (RTX 3090, A5000, etc.)
  - With INT8 quantization: >= 16 GB VRAM is sufficient
- Two GPUs recommended for multi-GPU mode (reduces VRAM contention)

### Software

| Package | Version | Purpose |
|---------|---------|---------|
| Python | >= 3.10 | |
| PyTorch | >= 2.6 | with CUDA 12.4 |
| transformers | >= 4.57 | HuggingFace model loading |
| qwen-tts | >= 0.1.1 | Qwen3-TTS model + tokenizer |
| fastapi | >= 0.129 | HTTP server |
| uvicorn | >= 0.40 | ASGI server |
| soundfile | >= 0.12 | WAV I/O |
| librosa | >= 0.10 | Audio resampling |
| safetensors | >= 0.5 | Weight loading |
| huggingface_hub | >= 0.36 | Model downloads |
| hydra-core | >= 1.3 | Seed-VC config loading |
| omegaconf | >= 2.3 | Seed-VC config parsing |
| pydub | >= 0.25 | MP3/format handling |
| bitsandbytes | >= 0.45 | INT8 quantization (optional) |
| pydantic | >= 2.11 | Request validation |
| numpy, scipy | | Numerical ops |

### Models (downloaded automatically on first run)

| Model | Size | Source |
|-------|------|--------|
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | ~3.4 GB (bf16) | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
| Seed-VC V2 (CFM + AR) | ~800 MB | `seed_vc_repo/` checkpoints |
| ECAPA-TDNN speaker encoder | ~20 MB | Extracted from `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |

---

## Installation

```bash
# Clone this repository
git clone https://github.com/LAION-AI/Vocalino-V0.1-Voice-Acting-Pipeline.git
cd Vocalino-V0.1-Voice-Acting-Pipeline

# Run the setup script (installs dependencies + downloads model weights)
python setup.py

# Clone Seed-VC V2 into the expected directory
git clone https://github.com/Plachtaa/seed-vc.git seed_vc_repo

# (Optional) Install flash-attn for faster TTS attention
pip install flash-attn --no-build-isolation
```

---

## Quick Start

```bash
# Basic launch (single GPU, bfloat16)
python server.py

# With INT8 quantization (halves TTS VRAM)
TTS_QUANTIZE=int8 python server.py

# Multi-GPU (TTS on GPU 0, VC on GPU 1)
CUDA_VISIBLE_DEVICES=0,1 VC_DEVICE=cuda:1 python server.py

# Multi-GPU + INT8 quantization
CUDA_VISIBLE_DEVICES=0,1 TTS_QUANTIZE=int8 VC_DEVICE=cuda:1 python server.py
```

The server starts on `http://0.0.0.0:8000`. Open the web UI at
`http://<server-ip>:8000/ui/`.

---

## Configuration

All settings are configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_DEVICE` | `cuda:0` | GPU device for Qwen3-TTS model |
| `VC_DEVICE` | *(same as TTS)* | GPU device for Seed-VC model |
| `SE_DEVICE` | *(same as TTS)* | GPU device for speaker encoder |
| `TTS_QUANTIZE` | `none` | `none` = bfloat16, `int8` = bitsandbytes INT8 |
| `DEFAULT_DIFF_STEPS` | `12` | Default VC diffusion steps (lower = faster) |
| `EMB_CACHE_SIZE` | `32` | Max speaker embedding cache entries |

### Quantization Modes

| Mode | VRAM | Speed | Quality |
|------|------|-------|---------|
| `none` (bfloat16) | ~15 GB | Baseline | Best |
| `int8` | ~7 GB | Similar or slightly slower | Near-identical for TTS |

INT8 quantization uses [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
to quantize the transformer's linear layers to 8-bit integers. The codec token
decoder and speech tokenizer remain in full precision. Quality impact is
minimal for speech synthesis since the output is discrete codec tokens.

### Multi-GPU Setup

When two GPUs are available, splitting TTS and VC across devices:
- Eliminates VRAM contention between the two largest models
- Allows the server to handle concurrent requests more efficiently
- Frees VRAM on each GPU for other processes

```bash
# Example: RTX 3090 x2
CUDA_VISIBLE_DEVICES=0,1 VC_DEVICE=cuda:1 python server.py
```

---

## Web UI

The web UI is served at `/ui` and provides two sections:

### Section 1: Voice Design (Reference Creation)

- Enter text and a natural-language voice/style description
- Generate N samples (batched for speed)
- Listen, download, or select any sample as reference for the pipeline

### Section 2: Full Pipeline (Voice-Consistent Generation)

- Upload or select a reference audio (target speaker identity)
- Enter text and emotion/style instruction
- Generate K candidates — each streamed to the UI as it completes
- Candidates ranked by speaker embedding similarity (green = best match)
- Download final audio or intermediate TTS (before voice conversion)

---

## API Reference

### `POST /tts/generate-voice-design`

Generate speech using Qwen3-TTS Voice Design.

```json
{
  "text": "Hello, how are you today?",
  "style_prompt": "A warm female voice, speaking calmly",
  "language": "English"
}
```

**Response**: `{ "status": "success", "sample_rate": 12000, "audio_base64": "..." }`

### `POST /voice-design/batch`

Generate N voice design samples in a single **batched** forward pass.
Each output differs due to stochastic sampling.

```json
{
  "text": "Hello, how are you today?",
  "style_prompt": "A warm female voice, speaking calmly",
  "language": "English",
  "n_samples": 3
}
```

**Response**: `{ "status": "success", "samples": [...], "batch_time": 65.2 }`

### `POST /vc/convert`

Voice conversion with Seed-VC V2.

```json
{
  "source_audio_base64": "<base64 WAV>",
  "target_audio_base64": "<base64 WAV>",
  "diffusion_steps": 12
}
```

### `POST /pipeline/tts-then-vc`

Combined: generate styled speech then convert to target voice.

```json
{
  "text": "Hello!",
  "style_prompt": "Excited, high energy",
  "target_audio_base64": "<base64 reference WAV>",
  "language": "English",
  "diffusion_steps": 12,
  "return_intermediate": true
}
```

Set `return_intermediate: true` to also receive the raw TTS output (before voice conversion) in `intermediate_tts_audio_base64`.

### `POST /pipeline/ranked`

Generate K candidates, rank by ECAPA-TDNN speaker similarity. Returns all
candidates sorted by similarity after the last one completes.

```json
{
  "text": "Hello!",
  "style_prompt": "Excited, high energy",
  "reference_audio_base64": "<base64 reference WAV>",
  "language": "English",
  "k_candidates": 3,
  "diffusion_steps": 12
}
```

### `POST /pipeline/ranked-stream` (SSE)

Same as `/pipeline/ranked` but streams results via Server-Sent Events.
Each candidate is sent as it completes. The UI uses this endpoint.

**SSE event types**:
- `event: progress` — TTS batch completed, VC phase starting
- `event: candidate` — one completed candidate (audio + similarity)
- `event: done` — final summary with `best_id`

### `GET /health`

Returns model status and server configuration.

```json
{
  "qwen_loaded": true,
  "seed_vc_loaded": true,
  "speaker_encoder_loaded": true,
  "tts_device": "cuda:0",
  "vc_device": "cuda:1",
  "tts_quantize": "int8",
  "emb_cache_used": 3
}
```

---

## Optimization Tiers

See `OPTIMIZATION_PLAN.txt` for the full roadmap. Summary of implemented tiers:

| Tier | Optimization | Status | Impact |
|------|-------------|--------|--------|
| 1 | VC diffusion steps 25 → 12 | Done | ~50% faster VC |
| 1 | Speaker embedding LRU cache | Done | Skip repeated ECAPA-TDNN passes |
| 2 | SSE streaming | Done | First result visible immediately |
| 4 | TTS batching | Done | K samples in ~1.5x single-call time |
| 5a | torch.compile | Skipped | Known issues with Qwen models |
| 5b | Multi-GPU split | Done | Opt-in via VC_DEVICE env var |
| 5c | vLLM-Omni serving | Future | Requires vllm-omni package |
| 5d | INT8 quantization | Done | Opt-in, halves TTS VRAM |

### Typical Timings (RTX 3090, K=3)

| Configuration | TTS Phase | VC Phase | Total |
|--------------|-----------|----------|-------|
| Sequential (baseline) | ~135s | ~24s | ~160s |
| Batched TTS | ~65s | ~24s | ~90s |
| Batched TTS + INT8 | ~65s* | ~24s | ~90s* |

*INT8 timing depends on hardware; may be slightly faster or slower than bf16.

---

## Future Improvements

### vLLM-Omni for TTS Serving

[vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/) provides official
Qwen3-TTS support with continuous batching, PagedAttention, and CUDA graph
acceleration. This is the recommended production path for high-throughput
TTS serving.

```bash
pip install vllm-omni
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
    --omni --port 8091 --trust-remote-code --enforce-eager
```

Note: `--enforce-eager` is required — `torch.compile` has known
compatibility issues with Qwen-family models.

### VC Batching

The Seed-VC CFM (diffusion) model natively supports batched inputs, but
the wrapper API and AR generation loop are single-sample only. Refactoring
these could enable batched voice conversion for additional throughput gains.

---

## Troubleshooting

### "flash-attn is not installed" warning

This is non-fatal. The model falls back to PyTorch SDPA attention.
To install (requires compatible glibc):

```bash
pip install flash-attn --no-build-isolation
```

### Server not responding

Check logs:
```bash
tail -f server.log
```

Model loading takes 1-2 minutes. Wait for "Models loaded. Server ready."

### CUDA out of memory

- Use INT8 quantization: `TTS_QUANTIZE=int8`
- Reduce batch size (K candidates) in the UI
- Use multi-GPU: `CUDA_VISIBLE_DEVICES=0,1 VC_DEVICE=cuda:1`

### VC quality issues

- Increase `diffusion_steps` (e.g., 20-25) for higher quality at the cost of speed
- Adjust `similarity_cfg_rate` (0.5-0.9) to balance intelligibility vs similarity
- Ensure reference audio is clean, 5-30 seconds, single speaker

---

## License

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — Apache 2.0
- [Seed-VC](https://github.com/Plachtaa/seed-vc) — MIT
- This pipeline code — MIT
