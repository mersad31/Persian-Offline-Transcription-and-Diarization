# 🎙️ Persian Offline Transcription & Diarization

A Streamlit app that turns Persian (Farsi) audio into text **and** labels who said what, entirely on your own machine.

---

## 🎯 The Problem

Meetings, interviews and calls often contain sensitive speech. Cloud transcription services send that audio to a third party, and many of them handle Persian and multiple speakers poorly. This tool keeps everything local.

## 💡 The Approach

1. **Transcribe** with Faster-Whisper (`language="fa"`, optional VAD filter).
2. **Embed** the audio in sliding windows with a SpeechBrain speaker-embedding model.
3. **Cluster** the windows (agglomerative clustering) into the number of speakers you choose.
4. **Merge** the transcript segments with the speaker labels and show them as a chat-like conversation.

## ✨ Features

- 🇮🇷 Persian transcription (Faster-Whisper)
- 👥 Speaker diarization (Speaker A, Speaker B, ...), up to 10 speakers
- 🔒 Fully offline: models are loaded from local folders
- 📂 Input: `mp3`, `wav`, `m4a`, `ogg`
- 💾 Export: `.txt` transcript and `.srt` subtitles
- ↔️ Right-to-left display for Persian
- 🎚️ Adjustable window and hop size for diarization

<!-- TODO: add a screenshot of the chat-style result, e.g. ![Demo](docs/demo.png) -->

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| Transcription | Faster-Whisper |
| Speaker embeddings | SpeechBrain |
| Clustering | scikit-learn (agglomerative clustering) |
| Audio handling | PyDub, TorchAudio, FFmpeg |

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- FFmpeg on your `PATH`
  - Ubuntu: `sudo apt install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Windows: download from [ffmpeg.org](https://ffmpeg.org/)

### Install

```bash
git clone https://github.com/mersad31/Persian-Offline-Transcription-and-Diarization.git
cd Persian-Offline-Transcription-and-Diarization
pip install -r requirement.txt
```

### Models

The app loads models from local folders and does **not** download them. Download them once, then place them like this:

```text
models/
├── whisper/        # a Faster-Whisper (CTranslate2) model folder
└── speechbrain/    # a SpeechBrain speaker-embedding model folder
```

### Run

```bash
streamlit run app.py
```

## ⚡ Scaling Up (with a strong GPU)

- Use a larger Whisper model for harder Persian vocabulary
- Add forced alignment (e.g. WhisperX) for word-level timestamps
- Add neural noise reduction for noisy recordings
- Post-correct punctuation with a local LLM

## 📝 License

MIT License. See [LICENSE](LICENSE).

## 🙏 Acknowledgments

OpenAI Whisper, the SpeechBrain team and the Streamlit community.
