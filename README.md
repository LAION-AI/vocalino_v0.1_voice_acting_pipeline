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

### The Problem
Text-to-speech models can generate expressive speech with controlled emotion, but the voice identity is random each time. Voice conversion models can make audio sound like a specific person, but they need source audio to convert. Neither alone gives you full control over *what is said*, *how it's said*, and *who says it*.

### The Solution: Two Stages

**Stage 1 — Qwen3-TTS Voice Design** (by Alibaba, 1.7B parameters)

A large language model that generates speech from text + a natural-language style instruction. You describe the desired emotion, pace, energy, and pitch in plain English (e.g. *"speak with trembling fear, whispering, medium-pitched male voice"*), and the model generates audio that matches. The voice identity is random — only the style/emotion is controlled.

**Stage 2 — Chatterbox VC** (by Resemble AI)

A zero-shot voice conversion model that transplants a target speaker's voice identity onto any audio. Here's how it works internally:

1. **S3 Speech Tokenization**: The source audio is encoded into discrete S3 speech tokens by a neural tokenizer. These tokens capture the phonetic content (what is said) and prosody (rhythm, intonation, emotion) but throw away the speaker identity.

2. **Speaker Embedding**: A short reference clip of the target speaker (up to 10 seconds) is processed to extract a compact speaker embedding that encodes the target's vocal characteristics — timbre, formant structure, breathiness, etc.

3. **Flow-Matching Decoder**: The S3 tokens and speaker embedding are fed into a flow-matching generative model that synthesizes a new waveform at 24 kHz. The decoder learns to generate audio that matches the phonetic content and prosody of the tokens while sounding like the speaker in the embedding.

The result: the target speaker's voice saying the source content with the source emotion/style — all from just a few seconds of reference audio.

### Combined Pipeline

```
Text + Style Instruction ──→ [Qwen3-TTS] ──→ Expressive audio (random voice)
                                                        │
Reference audio of target ──→ [Chatterbox VC] ←────────┘
                                     │
                                     ▼
                          Final audio (target voice,
                          controlled emotion/style)
```

## Project Structure

```
voice-pipeline/
├── README.md              ← You are here
├── setup.py               ← Install dependencies + download model weights
├── server.py              ← FastAPI server (all endpoints)
├── generate_samples.py    ← Standalone: TTS+VC for multiple emotions
├── convert_voice.py       ← Standalone: voice conversion only
└── output/                ← Generated at runtime
```

## Installation

### Requirements
- Python 3.10+
- NVIDIA GPU with ≥8 GB VRAM
- NVIDIA driver ≥525 (for CUDA 12.4)

### Setup

```bash
cd voice-pipeline
python setup.py
```

This installs:
- PyTorch 2.6 with CUDA 12.4
- Qwen3-TTS (`qwen-tts` package)
- Chatterbox TTS+VC (`chatterbox-tts` package)
- transformers, FastAPI, uvicorn, and other dependencies

And downloads:
- Qwen3-TTS-12Hz-1.7B-VoiceDesign weights (~3.6 GB)
- Chatterbox s3gen weights (auto-downloaded on first use)

## Usage

### Option A: FastAPI Server

Start the server:

```bash
python server.py
python server.py --port 9000  # custom port
```

The server loads both models on startup (~5 GB VRAM) and exposes these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tts/generate-voice-design` | POST | Text → speech with emotion control |
| `/vc/convert` | POST | Voice conversion (source → target voice) |
| `/pipeline/tts-then-vc` | POST | Combined: text + emotion + target voice |
| `/health` | GET | Check model status |

Interactive API docs are available at `http://localhost:8000/docs` once the server is running.

**Example: Combined pipeline**

```python
import base64, requests

# Load your reference audio
with open("chris-ref.mp3", "rb") as f:
    ref_b64 = base64.b64encode(f.read()).decode()

resp = requests.post("http://localhost:8000/pipeline/tts-then-vc", json={
    "text": "This is the best day of my life!",
    "style_prompt": "Speak with excited joy, medium-pitched male voice.",
    "target_audio_base64": ref_b64,
    "language": "English",
})

# Save the result
audio_bytes = base64.b64decode(resp.json()["audio_base64"])
with open("output.wav", "wb") as f:
    f.write(audio_bytes)
```

### Option B: Standalone Sample Generation

Generate multiple emotion samples for a target speaker, with an HTML report:

```bash
python generate_samples.py --ref chris-ref.mp3 --name "Chris" --output output_chris/
```

This loads both models locally (no server needed), generates 5 emotion variations, and creates `output_chris/report.html` with embedded audio players.

Edit the `SAMPLES` list and `VOICE_HINT` at the top of the script to customize emotions and voice gender/pitch hints.

### Option C: Voice Conversion Only

Convert any audio to a target voice:

```bash
# Single file
python convert_voice.py --source input.wav --target ref.wav --html

# Multiple files
python convert_voice.py --source a.wav b.wav c.wav --target ref.wav --html

# Custom output path
python convert_voice.py --source input.wav --target ref.wav -o converted.wav
```

The `--html` flag generates a comparison page with source and converted audio side by side.

## API Reference

### POST /tts/generate-voice-design

Generate speech with Qwen3-TTS Voice Design.

**Request body:**
```json
{
  "text": "Hello, how are you?",
  "style_prompt": "Speak warmly with a gentle smile, medium-pitched male voice.",
  "language": "English"
}
```

**Response:**
```json
{
  "status": "success",
  "sample_rate": 24000,
  "audio_base64": "<base64 WAV>"
}
```

### POST /vc/convert

Convert source audio to a target speaker's voice.

**Request body:**
```json
{
  "source_audio_base64": "<base64 WAV or MP3>",
  "target_audio_base64": "<base64 WAV or MP3>"
}
```

**Response:**
```json
{
  "status": "success",
  "sample_rate": 24000,
  "audio_base64": "<base64 WAV>"
}
```

### POST /pipeline/tts-then-vc

Combined: generate styled speech then convert to target voice.

**Request body:**
```json
{
  "text": "This is amazing!",
  "style_prompt": "Excited and breathless with joy.",
  "target_audio_base64": "<base64 WAV or MP3>",
  "language": "English",
  "return_intermediate": false
}
```

Set `return_intermediate: true` to also receive the raw TTS output (before voice conversion) in `intermediate_tts_audio_base64`.

## Tips

- **Voice hint matters**: Add gender and pitch information to your style prompts (e.g. "medium-pitched male voice") so the TTS output roughly matches the target speaker. This makes the voice conversion smoother.

- **Reference audio quality**: Use a clean, clear reference clip of 3-10 seconds. Background noise or music in the reference will degrade conversion quality.

- **Chatterbox is fast**: Voice conversion takes ~1-2 seconds per clip on a modern GPU. The TTS step is the bottleneck (~20-30 seconds per generation).

## Models Used

| Model | Source | Parameters | Purpose |
|-------|--------|------------|---------|
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | [HuggingFace](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) | 1.7B | Text-to-speech with emotion control |
| Chatterbox VC (s3gen) | [HuggingFace](https://huggingface.co/ResembleAI/chatterbox) | ~100M | Zero-shot voice conversion |

## License

- Qwen3-TTS: Apache 2.0
- Chatterbox: MIT
- This pipeline code: MIT





