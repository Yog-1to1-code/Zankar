<![CDATA[<div align="center">

# 📡 Project Zankar

### *The Vibrations of Desire*

**Real-time HAM radio transcription & intelligent logbook generation,  
powered by Apple Silicon GPU acceleration and Google Gemini.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![MLX](https://img.shields.io/badge/Apple_MLX-Whisper-000000?style=for-the-badge&logo=apple&logoColor=white)](https://github.com/ml-explore/mlx)
[![Gemini](https://img.shields.io/badge/Google-Gemini_2.5-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)

---

*"Zankar" (ज़ंकार) — the resonant vibration of a struck string; the lingering hum that carries meaning through the air.*

</div>

---

## 🌐 What is Zankar?

**Zankar** is a two-stage pipeline that captures live amateur (HAM) radio transmissions, transcribes them in real-time using OpenAI's Whisper model (accelerated on Apple Silicon via [MLX](https://github.com/ml-explore/mlx)), and then sends the recording + raw transcript to **Google Gemini 2.5 Flash** — which listens to the audio, cross-references the Whisper transcript, corrects errors, decodes NATO phonetic callsigns, and produces a fully structured **amateur radio logbook** in JSON.

> **In short:** Plug in your radio → Zankar listens → You get a beautiful, corrected QSO log without writing a single line by hand.

---

## 🏗️ Architecture Overview

```
┌──────────────┐     ┌─────────────────────┐     ┌──────────────────────┐
│  Microphone  │────▶│  live_transcribe.py  │────▶│  gemini_parser.py    │
│  / SDR Input │     │                      │     │                      │
└──────────────┘     │  • WebRTC VAD        │     │  • Uploads .wav +    │
                     │  • DSP bandpass      │     │    .log to Gemini    │
                     │  • MLX Whisper GPU   │     │  • Extracts QSOs,    │
                     │  • Text stabilizer   │     │    callsigns, RST    │
                     │  • .wav + .log output │     │  • Structured JSON   │
                     └─────────────────────┘     └──────────────────────┘
                            Stage 1                      Stage 2
```

---

## ✨ Features

### Stage 1 — Live Transcription (`live_transcribe.py`)
- 🎙️ **Real-time audio capture** at 16 kHz mono via `sounddevice`
- 🧠 **WebRTC VAD (Voice Activity Detection)** — only processes frames containing human speech, ignoring dead air and static
- 🔊 **DSP Bandpass Filtering** — 5th-order Butterworth filter (300 Hz – 3000 Hz) isolates the amateur radio voice band, stripping noise
- ⚡ **Apple Silicon GPU acceleration** — transcription runs on-device via `mlx-whisper` with FP16 precision (no cloud latency)
- 🔄 **Streaming text stabilization** — prefix-matching algorithm locks confirmed words, showing unstable suffixes as live previews
- 📝 **Automatic session logging** — timestamps every committed sentence in UTC to a `.log` file
- 🎵 **Clean audio export** — concatenates only VAD-verified, DSP-filtered audio frames into a single `.wav` file for Gemini

### Stage 2 — Gemini Log Parsing (`gemini_parser.py`)
- 📤 **Multimodal analysis** — uploads both the filtered `.wav` audio and the Whisper `.log` transcript to Gemini 2.5 Flash
- 🔤 **NATO phonetic decoding** — converts spoken callsigns ("Whiskey Two Papa Victor Foxtrot") to proper format (`W2PVF`)
- 📋 **Structured QSO extraction** — date, time, callsign, operator name, QTH, frequency, mode, RST reports
- 💬 **Dialogue reconstruction** — corrected, speaker-attributed conversation history
- 📊 **Session summary** — auto-generated paragraph summarizing the entire radio session
- 💾 **JSON export** — saves the complete structured logbook to `logs/`

---

## 📂 Project Structure

```
Whisp/
├── live_transcribe.py      # Stage 1: Real-time VAD + DSP + Whisper transcription
├── gemini_parser.py         # Stage 2: Gemini multimodal log parser
├── test_transcribe.py       # Quick test script for file-based transcription
├── whisper_dowload.py       # Model downloader (Hugging Face → local)
├── fix_config.py            # Config mapper for MLX-compatible Whisper params
├── .env                     # API keys (GEMINI_API_KEY)
├── .gitignore               # Excludes .env from version control
├── recordings/              # Session outputs (.wav audio + .log transcripts)
│   ├── session_YYYYMMDD_HHMMSS.wav
│   └── session_YYYYMMDD_HHMMSS.log
├── logs/                    # Structured JSON logbooks from Gemini
│   └── session_YYYYMMDD_HHMMSS_log.json
└── Test_audio_files/        # Sample audio files for testing
```

---

## 🚀 Getting Started

### Prerequisites

| Requirement | Details |
|---|---|
| **Hardware** | Apple Silicon Mac (M1/M2/M3/M4) — MLX requires Apple GPU |
| **Python** | 3.10 or higher |
| **API Key** | [Google Gemini API key](https://aistudio.google.com/apikey) (free tier works) |
| **Audio Input** | Microphone, SDR receiver, or line-in from radio |

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/zankar.git
cd zankar
```

### 2. Install Dependencies

```bash
pip install mlx-whisper sounddevice numpy scipy webrtcvad google-generativeai python-dotenv huggingface_hub
```

<details>
<summary>📦 Full dependency list</summary>

| Package | Purpose |
|---|---|
| `mlx-whisper` | Whisper transcription on Apple Silicon GPU |
| `sounddevice` | Real-time audio capture |
| `numpy` | Numerical array operations |
| `scipy` | DSP bandpass filter (Butterworth) |
| `webrtcvad` | Voice Activity Detection |
| `google-generativeai` | Gemini API client |
| `python-dotenv` | Environment variable management |
| `huggingface_hub` | Model downloading |

</details>

### 3. Download the Whisper Model

```bash
python3 whisper_dowload.py
```

When prompted, enter a local path such as:
```
~/Downloads/AI_Models/whisper-large-v3-turbo
```

Then run the config fixer to map Hugging Face parameters to MLX format:
```bash
python3 fix_config.py
```

> **Note:** Update the `model_id` path in `live_transcribe.py` (line 215) and `test_transcribe.py` (line 6) to match your download location.

### 4. Configure API Keys

Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Test with a Sample File (Optional)

```bash
python3 test_transcribe.py
```

Place an audio file (e.g., `sample.mp3`) in the project root and update the path in `test_transcribe.py` to verify Whisper is working.

---

## 🎙️ Usage

### Stage 1 — Live Transcription

```bash
python3 live_transcribe.py
```

1. Zankar begins listening on your default microphone
2. The console displays a live, streaming transcript with locked and unstable words
3. Press **Ctrl+C** to stop — the session is saved as:
   - `recordings/session_<timestamp>.wav` — clean, VAD-filtered audio
   - `recordings/session_<timestamp>.log` — timestamped raw transcript

### Stage 2 — Gemini Log Parsing

```bash
# Parse the most recent session automatically
python3 gemini_parser.py

# Or specify a session explicitly
python3 gemini_parser.py recordings/session_20260320_122817
```

Gemini analyzes both the audio and transcript, then outputs:
- A formatted **QSO logbook table** in the terminal
- A complete **structured JSON** logbook saved to `logs/`

---

## 📋 Example Output

### Terminal Output (Gemini Parser)
```
📋 STRUCTURED HAM RADIO LOGBOOK
=================================================================

📝 Summary: This session involved a brief SSB contact with LU1ZE,
   operated by Charlie from the Argentine Research Base in Antarctica...

#    Date         Time    Callsign   Name         QTH                  RST S/R    Mode
--------------------------------------------------------------------------------
1    2026-03-20   12:28   LU1ZE      Charlie      Antarctica (HERO)    ?/59       SSB
     💬 Contact from Argentine Research Base, operating from vessel HERO.
```

### JSON Logbook (`logs/session_*_log.json`)
```json
{
  "qso_entries": [
    {
      "date_utc": "2026-03-20",
      "time_utc": "12:28",
      "callsign": "LU1ZE",
      "operator_name": "Charlie",
      "qth": "Antarctica (Argentine Research Base, vessel HERO)",
      "mode": "SSB",
      "rst_received": "59",
      "remarks": "Contact from Argentine Research Base in Antarctica..."
    }
  ],
  "session_summary": "...",
  "dialogue": [
    { "speaker": "You", "text": "LU1ZE, LU1 Zebra Echo. Here is New Jersey, W2 Papa Victor Foxtrot." },
    { "speaker": "They", "text": "W2 Papa Victor Foxtrot, how copy?" }
  ]
}
```

---

## ⚙️ How It Works — Technical Deep Dive

### Audio Pipeline (Stage 1)

1. **Capture** — Audio streams in at 16 kHz, 32-bit float, mono via `sounddevice`
2. **VAD Gating** — Each 500ms chunk is evaluated by WebRTC's GMM-based VAD; only speech frames pass through
3. **Pre-buffering** — 1 second of audio before detected speech is prepended to catch sentence beginnings
4. **Bandpass Filter** — A 5th-order Butterworth filter (300–3000 Hz) removes noise outside the voice band
5. **Whisper Transcription** — Cleaned audio is fed to `mlx-whisper` for GPU-accelerated inference
6. **Text Stabilization** — A prefix-matching algorithm compares consecutive transcriptions, locking words that appear consistently and displaying uncertain suffixes as a live preview
7. **Commit & Log** — After 2.5 seconds of silence (or 12 seconds max buffer), the sentence is finalized, logged, and the stabilizer resets

### Intelligent Parsing (Stage 2)

1. **File Discovery** — Automatically finds the latest `.wav` + `.log` pair in `recordings/`
2. **Gemini Upload** — The clean `.wav` audio is uploaded via the Gemini File API
3. **Multimodal Prompt** — Gemini receives both the audio and the Whisper transcript, with instructions to decode NATO callsigns, extract RST reports, and attribute speakers
4. **JSON Extraction** — The response is parsed into a structured logbook format
5. **Cleanup** — Uploaded files are removed from Gemini's servers after processing

---

## 🛠️ Configuration

| Parameter | Location | Default | Description |
|---|---|---|---|
| `FS` (Sample Rate) | `live_transcribe.py:212` | `16000` | Audio sample rate in Hz |
| `CHUNK_SEC` | `live_transcribe.py:213` | `0.5` | Audio chunk size in seconds |
| `POST_SPEECH_SEC` | `live_transcribe.py:123` | `2.5` | Silence duration before sentence commit |
| `MAX_BUFFER_SEC` | `live_transcribe.py:124` | `12.0` | Maximum buffer before forced commit |
| VAD Aggressiveness | `live_transcribe.py:115` | `2` | WebRTC VAD mode (0–3, higher = more aggressive) |
| Bandpass Range | `live_transcribe.py:119` | `300–3000 Hz` | Voice band isolation range |
| `model_id` | `live_transcribe.py:215` | Local path | Path to downloaded Whisper model |
| Gemini Model | `gemini_parser.py:125` | `gemini-2.5-flash` | Gemini model for log parsing |
| Temperature | `gemini_parser.py:130` | `0.1` | Low temp for factual extraction |

---

## 🤝 Contributing

Contributions are welcome! Some ideas for future enhancements:

- [ ] Multi-language support beyond English
- [ ] Automatic frequency detection from SDR metadata
- [ ] Web dashboard for browsing session logs
- [ ] ADIF export format for integration with standard logging software
- [ ] Real-time Gemini streaming for live corrected output
- [ ] Speaker diarization for multi-party QSOs

---

## 📜 License

This project is open source. See [LICENSE](LICENSE) for details.

---

<div align="center">

*Built with 🎙️ by a HAM radio enthusiast who got tired of writing logs by hand.*

**73 de Zankar** 📡

</div>
]]>
