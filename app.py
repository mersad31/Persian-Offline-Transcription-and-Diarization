# app.py
# Offline Persian transcription + speaker diarization (CPU-only friendly)
# - No HuggingFace token required at runtime
# - Uses faster-whisper (CTranslate2) local model folder
# - Uses SpeechBrain ECAPA speaker embedding local folder
# - Quality jump: diarize on sliding windows + majority-vote labels per Whisper segment
# - Better merging + RTL display + robust audio decoding

import os
import uuid
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import streamlit as st
import torch
import torchaudio
from pydub import AudioSegment
from faster_whisper import WhisperModel
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


# =========================================================
# CONFIG
# =========================================================
TEMP_DIR = "./temp"
MODELS_DIR = "./models"

# You MUST have these folders already on disk (offline):
#   ./models/whisper/   -> faster-whisper (CTranslate2) model folder
#   ./models/speechbrain/ -> SpeechBrain pretrained folder (hyperparams + checkpoint)
WHISPER_DIR = os.path.join(MODELS_DIR, "whisper")
SPEAKER_DIR = os.path.join(MODELS_DIR, "speechbrain")

os.makedirs(TEMP_DIR, exist_ok=True)

TARGET_SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"  # best for CPU memory/speed

# Diarization (quality-focused but CPU reasonable)
WIN_SEC = 1.5          # window size for speaker embedding extraction
HOP_SEC = 0.75         # hop size (overlap improves stability)
MIN_VOTE_FRAC = 0.55   # majority vote threshold for a label to "win" a segment
SMOOTH_GAP_SEC = 0.35  # merge adjacent same-speaker blocks if gap small

# Whisper settings (CPU-friendly but decent)
WHISPER_BEAM_SIZE = 3
WHISPER_VAD = True

st.set_page_config(page_title="Persian Offline Transcription + Diarization", page_icon="🎙️")


# =========================================================
# UTIL: CHECK FFMPEG (pydub needs it for mp3/m4a/ogg)
# =========================================================
def has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource
def load_whisper_model() -> WhisperModel:
    if not os.path.isdir(WHISPER_DIR):
        raise FileNotFoundError(
            f"Missing Whisper model folder: {WHISPER_DIR}\n"
            "You must place a faster-whisper (CTranslate2) model directory there for offline use."
        )
    return WhisperModel(WHISPER_DIR, device=DEVICE, compute_type=COMPUTE_TYPE)


@st.cache_resource
def load_speaker_model() -> EncoderClassifier:
    if not os.path.isdir(SPEAKER_DIR):
        raise FileNotFoundError(
            f"Missing SpeechBrain model folder: {SPEAKER_DIR}\n"
            "You must place a SpeechBrain pretrained speaker embedding package there for offline use."
        )
    # IMPORTANT: source should point to the same local folder. If those files exist, no download happens.
    return EncoderClassifier.from_hparams(
        source=SPEAKER_DIR,
        savedir=SPEAKER_DIR,
        run_opts={"device": DEVICE},
    )


# =========================================================
# AUDIO PREPROCESSING (robust)
# =========================================================
def preprocess_audio(uploaded_file, uid: str) -> Tuple[str, float]:
    """
    Saves uploaded file with its extension, decodes via pydub/ffmpeg, converts to mono 16k wav.
    """
    name = uploaded_file.name or "audio"
    _, ext = os.path.splitext(name)
    ext = ext.lower() if ext else ".bin"

    raw_path = os.path.join(TEMP_DIR, f"{uid}_raw{ext}")
    wav_path = os.path.join(TEMP_DIR, f"{uid}.wav")

    with open(raw_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    audio = AudioSegment.from_file(raw_path)
    duration_min = (len(audio) / 1000) / 60

    audio = audio.set_frame_rate(TARGET_SR).set_channels(1)
    audio.export(wav_path, format="wav")

    return wav_path, duration_min


def load_mono_16k(path: str) -> torch.Tensor:
    """
    Loads wav, ensures mono 16k float32 tensor shape [1, T].
    """
    wav, sr = torchaudio.load(path)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    wav = wav.to(torch.float32)
    return wav


# =========================================================
# DATA STRUCTURES
# =========================================================
@dataclass
class WSeg:
    start: float
    end: float
    text: str


@dataclass
class Win:
    start: float
    end: float


# =========================================================
# WHISPER TRANSCRIPTION
# =========================================================
def transcribe_persian(whisper: WhisperModel, wav_path: str) -> List[WSeg]:
    seg_gen, _info = whisper.transcribe(
        wav_path,
        language="fa",
        beam_size=WHISPER_BEAM_SIZE,
        vad_filter=WHISPER_VAD,
    )
    segments = []
    for s in seg_gen:
        txt = (s.text or "").strip()
        if not txt:
            continue
        segments.append(WSeg(start=float(s.start), end=float(s.end), text=txt))
    return segments


# =========================================================
# SLIDING WINDOW SPEAKER EMBEDDINGS (quality jump)
# =========================================================
def make_windows(total_sec: float, win_sec: float, hop_sec: float) -> List[Win]:
    if total_sec <= 0:
        return []
    wins: List[Win] = []
    t = 0.0
    while t < total_sec:
        s = t
        e = min(t + win_sec, total_sec)
        if e - s >= 0.6:  # ignore too tiny end window
            wins.append(Win(start=s, end=e))
        t += hop_sec
    return wins


def encode_windows(
    wav: torch.Tensor,
    windows: List[Win],
    classifier: EncoderClassifier,
) -> Tuple[np.ndarray, List[Win]]:
    """
    Returns embeddings [N, D] for windows. Skips low-energy windows.
    """
    if not windows:
        return np.empty((0, 0), dtype=np.float32), []

    embeddings: List[np.ndarray] = []
    kept: List[Win] = []

    prog = st.progress(0)
    for i, w in enumerate(windows):
        prog.progress((i + 1) / len(windows))

        s = int(w.start * TARGET_SR)
        e = int(w.end * TARGET_SR)
        chunk = wav[:, s:e]  # [1, T]

        if chunk.shape[1] < int(0.6 * TARGET_SR):
            continue

        # energy gate (skip near-silence)
        rms = torch.sqrt(torch.mean(chunk ** 2)).item()
        if rms < 0.003:
            continue

        # normalize
        mx = chunk.abs().max()
        if mx > 0:
            chunk = chunk / mx

        with torch.no_grad():
            emb = classifier.encode_batch(chunk.to(DEVICE))  # [1, 1, D] or [1, D] depending version
            emb = emb.squeeze().detach().cpu().numpy().astype(np.float32)
            embeddings.append(emb)
            kept.append(w)

    prog.empty()

    if not embeddings:
        return np.empty((0, 0), dtype=np.float32), []

    X = np.stack(embeddings, axis=0)
    # L2 normalize embeddings (important for clustering stability)
    X = normalize(X, norm="l2")
    return X, kept


# =========================================================
# CLUSTERING (more stable than raw KMeans)
# =========================================================
def cluster_speakers(X: np.ndarray, num_speakers: int) -> np.ndarray:
    """
    Agglomerative clustering on cosine distances works well for embeddings.
    Requires n_clusters <= len(X).
    """
    k = max(1, min(int(num_speakers), int(X.shape[0])))
    if X.shape[0] == 1:
        return np.array([0], dtype=int)

    # AgglomerativeClustering expects distances; metric='cosine' available in newer sklearn,
    # but for compatibility, we can use linkage='average' with metric='cosine' if supported.
    # We'll try modern API first, fallback otherwise.
    try:
        cl = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    except TypeError:
        # older sklearn uses "affinity" instead of "metric"
        cl = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")

    labels = cl.fit_predict(X)
    return labels.astype(int)


# =========================================================
# ASSIGN WHISPER SEGMENTS BY MAJORITY VOTE OVER WINDOWS
# =========================================================
def assign_segment_labels(
    whisper_segments: List[WSeg],
    win_list: List[Win],
    win_labels: np.ndarray,
) -> List[int]:
    """
    For each Whisper segment, find overlapping windows and majority-vote the speaker label.
    If overlap is weak, fall back to nearest window center.
    """
    if not whisper_segments:
        return []
    if not win_list or win_labels.size == 0:
        return [0] * len(whisper_segments)

    # precompute window centers
    win_centers = np.array([(w.start + w.end) / 2 for w in win_list], dtype=np.float32)

    seg_labels: List[int] = []

    for seg in whisper_segments:
        overlaps = []
        seg_len = max(1e-6, seg.end - seg.start)
        for w, lab in zip(win_list, win_labels):
            # overlap duration
            ov = max(0.0, min(seg.end, w.end) - max(seg.start, w.start))
            if ov > 0:
                overlaps.append((lab, ov))

        if overlaps:
            # vote by overlap seconds (weighted)
            by_lab: Dict[int, float] = {}
            for lab, ov in overlaps:
                by_lab[int(lab)] = by_lab.get(int(lab), 0.0) + float(ov)

            # winner
            winner = max(by_lab.items(), key=lambda x: x[1])[0]
            winner_frac = by_lab[winner] / seg_len

            if winner_frac >= MIN_VOTE_FRAC:
                seg_labels.append(int(winner))
            else:
                # weak majority: choose nearest window center to seg center
                seg_center = (seg.start + seg.end) / 2
                idx = int(np.argmin(np.abs(win_centers - seg_center)))
                seg_labels.append(int(win_labels[idx]))
        else:
            # no overlap (rare): nearest window
            seg_center = (seg.start + seg.end) / 2
            idx = int(np.argmin(np.abs(win_centers - seg_center)))
            seg_labels.append(int(win_labels[idx]))

    return seg_labels


# =========================================================
# MERGE SEGMENTS INTO SPEAKER TURNS (clean readable output)
# =========================================================
def merge_turns(segments: List[WSeg], labels: List[int]) -> List[Tuple[int, float, float, str]]:
    if not segments:
        return []
    assert len(segments) == len(labels)

    merged: List[Tuple[int, float, float, str]] = []

    cur_lab = labels[0]
    cur_start = segments[0].start
    cur_end = segments[0].end
    cur_text = segments[0].text

    for seg, lab in zip(segments[1:], labels[1:]):
        gap = seg.start - cur_end

        # merge if same speaker and small gap
        if lab == cur_lab and gap <= SMOOTH_GAP_SEC:
            cur_text = (cur_text + " " + seg.text).strip()
            cur_end = seg.end
        else:
            merged.append((int(cur_lab), float(cur_start), float(cur_end), cur_text.strip()))
            cur_lab = lab
            cur_start = seg.start
            cur_end = seg.end
            cur_text = seg.text

    merged.append((int(cur_lab), float(cur_start), float(cur_end), cur_text.strip()))
    return merged


# =========================================================
# SRT GENERATOR
# =========================================================
def format_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def make_srt(merged: List[Tuple[int, float, float, str]], speaker_map: Dict[int, str]) -> str:
    out = []
    for i, (lab, start, end, text) in enumerate(merged, 1):
        spk = speaker_map.get(lab, f"Speaker {lab}")
        out.append(f"{i}\n{format_time(start)} --> {format_time(end)}\n{spk}: {text}\n")
    return "\n".join(out).strip() + "\n"


# =========================================================
# RTL DISPLAY
# =========================================================
def rtl_block(title: str, meta: str, text: str) -> str:
    safe_text = (text or "").replace("\n", "<br/>")
    return f"""
    <div dir="rtl" style="text-align:right; line-height:1.9; padding:10px 12px; border-radius:12px; border:1px solid #2b2b2b33;">
        <div style="font-weight:700; margin-bottom:6px;">{title} <span style="font-weight:400; opacity:0.75;">{meta}</span></div>
        <div style="white-space:normal;">{safe_text}</div>
    </div>
    """


# =========================================================
# CLEANUP
# =========================================================
def cleanup_uid(uid: str):
    try:
        if os.path.exists(TEMP_DIR):
            for fn in os.listdir(TEMP_DIR):
                if uid in fn:
                    try:
                        os.remove(os.path.join(TEMP_DIR, fn))
                    except Exception:
                        pass
    except Exception:
        pass


# =========================================================
# UI
# =========================================================
st.title("🎙️ Persian Offline Transcription + Diarization (CPU-friendly)")

st.caption(
    "You need local model folders for offline use:\n"
    "- ./models/whisper (faster-whisper / CTranslate2)\n"
    "- ./models/speechbrain (SpeechBrain speaker embedding package)\n"
)

if not has_ffmpeg():
    st.warning("ffmpeg is not found. mp3/m4a/ogg decoding may fail. Install ffmpeg and ensure it's in PATH.")

uploaded = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a", "ogg"])
num_speakers = st.number_input("Number of speakers", min_value=1, max_value=10, value=2, step=1)

colA, colB = st.columns(2)
with colA:
    st.write("Diarization windows")
    win_sec = st.slider("Window size (sec)", 0.8, 3.0, float(WIN_SEC), 0.1)
with colB:
    st.write("Overlap / hop")
    hop_sec = st.slider("Hop size (sec)", 0.3, 2.0, float(HOP_SEC), 0.05)

start_btn = st.button("Start Processing", type="primary", disabled=(uploaded is None))

if start_btn and uploaded is not None:
    uid = str(uuid.uuid4())

    try:
        with st.spinner("Preprocessing audio (mono 16k wav)..."):
            wav_path, duration_min = preprocess_audio(uploaded, uid)
            wav = load_mono_16k(wav_path)
            total_sec = wav.shape[1] / TARGET_SR

        st.info(f"Duration: {duration_min:.1f} minutes")

        with st.spinner("Loading models (local/offline)..."):
            whisper = load_whisper_model()
            speaker_model = load_speaker_model()

        with st.spinner("Transcribing (Persian)..."):
            wsegs = transcribe_persian(whisper, wav_path)

        if not wsegs:
            st.warning("No speech detected (or transcription returned empty).")
            cleanup_uid(uid)
            st.stop()

        st.success(f"Transcription segments: {len(wsegs)}")

        with st.spinner("Diarizing speakers (sliding windows + clustering)..."):
            windows = make_windows(total_sec=total_sec, win_sec=win_sec, hop_sec=hop_sec)
            X, kept_windows = encode_windows(wav, windows, speaker_model)

            if X.size == 0 or not kept_windows:
                st.warning("Could not extract speaker embeddings (audio might be too silent/noisy).")
                cleanup_uid(uid)
                st.stop()

            win_labels = cluster_speakers(X, int(num_speakers))
            seg_labels = assign_segment_labels(wsegs, kept_windows, win_labels)
            merged = merge_turns(wsegs, seg_labels)

        # Speaker mapping in stable order of appearance
        order = []
        for lab, _, _, _ in merged:
            if lab not in order:
                order.append(lab)

        speaker_map = {lab: f"Speaker {chr(65 + i)}" for i, lab in enumerate(order)}

        st.success("✅ Processing Complete")

        # Display
        st.divider()
        st.subheader("Transcript (RTL)")

        full_text_lines = []
        for lab, start, end, text in merged:
            name = speaker_map.get(lab, f"Speaker {lab}")
            meta = f"[{start:.2f}s - {end:.2f}s]"
            st.markdown(rtl_block(name, meta, text), unsafe_allow_html=True)
            st.write("")  # spacing

            full_text_lines.append(f"{name} ({start:.1f}-{end:.1f}s): {text}")

        full_text = "\n".join(full_text_lines).strip() + "\n"

        # Downloads
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download TXT",
                full_text,
                file_name=f"transcript_{uid}.txt",
                mime="text/plain",
            )
        with c2:
            st.download_button(
                "Download SRT",
                make_srt(merged, speaker_map),
                file_name=f"subtitles_{uid}.srt",
                mime="text/plain",
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        cleanup_uid(uid)