"""
� extract_text_embedding.py → Whisper + MiniLM
    Whisper (base)
    Вход: аудио 16 кГц
    Выход: транскрипт (текст речи)
        Whisper сам по себе даёт текст, не эмбеддинг.
        
    Текст подаётся в all-MiniLM-L6-v2.
    MiniLM (sentence-transformers/all-MiniLM-L6-v2)
        Вход: текст
        Выход: эмбеддинг [384]
    📘 E_text:
        размерность 384,
        семантический текстовый признак — смысл сказанного, лексика, контекст, значения слов.
"""

import os
from typing import Optional, Tuple

import librosa
import numpy as np
import torch
import tempfile
import ffmpeg
import noisereduce as nr
import whisper
from sentence_transformers import SentenceTransformer

VIDEO_PATH = "data/videos/sample.mp4"
TRANSCRIPT_TXT = "data/embeddings/sample_transcript.txt"
OUT_PATH = "data/embeddings/sample_text_embedding.npy"
SAMPLE_RATE = 16000
NOISE_PROFILE_SECS = 0.5  # сколько секунд берём для оценки шума в начале
WHISPER_MODEL = "base"    # можно "small" / "medium" / "large-v3"
SENT_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_torch_device_str() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_audio_16k_mono(path: str, target_sr: int = 16000) -> np.ndarray:
    """Пытаемся через ffmpeg→wav."""
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "audio_16k.wav")
        (
            ffmpeg
            .input(path)
            .output(
                wav_path,
                format="wav",
                acodec="pcm_s16le",
                ac=1,
                ar=str(target_sr)
            )
            .overwrite_output()
            .run(quiet=True)
        )
        y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
        return y.astype(np.float32)

def denoise_audio(y: np.ndarray, sr: int, profile_secs: float = 0.5) -> np.ndarray:
    """Простое подавление шума: spectral gating с профилем по начальному отрезку."""
    n_profile = int(profile_secs * sr)
    n_profile = max(1, min(n_profile, len(y)))
    noise_profile = y[:n_profile]
    y_denoised = nr.reduce_noise(y=y, y_noise=noise_profile, sr=sr, stationary=False)
    return y_denoised.astype(np.float32)

def extract_text_embedding(
    video_path: str,
    sample_rate: int = SAMPLE_RATE,
    noise_profile_secs: float = NOISE_PROFILE_SECS,
    whisper_model_name: str = WHISPER_MODEL,
    sentence_model_name: str = SENT_EMB_MODEL,
    whisper_model: Optional[whisper.Whisper] = None,
    sentence_model: Optional[SentenceTransformer] = None,
    denoise: bool = True,
) -> Tuple[np.ndarray, str]:
    """Возвращает (эмбеддинг, текст) полученные через Whisper + MiniLM."""

    device_str = get_torch_device_str()
    y = load_audio_16k_mono(video_path, target_sr=sample_rate)

    if denoise:
        y_proc = denoise_audio(y, sample_rate, profile_secs=noise_profile_secs)
    else:
        y_proc = y

    own_whisper = whisper_model is None
    if whisper_model is None:
        whisper_device = "cpu" if device_str == "mps" else device_str
        whisper_model = whisper.load_model(whisper_model_name, device=whisper_device)

    result = whisper_model.transcribe(y_proc, task="transcribe", fp16=False)
    text = (result.get("text") or "").strip()

    own_sentence = sentence_model is None
    if sentence_model is None:
        sbert_device = "cpu" if device_str == "mps" else device_str
        sentence_model = SentenceTransformer(sentence_model_name, device=sbert_device)

    if not text:
        emb = np.zeros((384,), dtype=np.float32)
    else:
        emb_vec = sentence_model.encode(text, normalize_embeddings=True)
        emb = np.asarray(emb_vec, dtype=np.float32)

    if own_sentence:
        del sentence_model
    if own_whisper:
        del whisper_model

    return emb, text


def main():
    device_str = get_torch_device_str()
    print("Using device:", device_str)

    emb, text = extract_text_embedding(
        VIDEO_PATH,
        sample_rate=SAMPLE_RATE,
        noise_profile_secs=NOISE_PROFILE_SECS,
        whisper_model_name=WHISPER_MODEL,
        sentence_model_name=SENT_EMB_MODEL,
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    if text:
        with open(TRANSCRIPT_TXT, "w", encoding="utf-8") as f:
            f.write(text)
        print("📄 Транскрипт сохранён:", TRANSCRIPT_TXT)

    np.save(OUT_PATH, emb)
    print("✅ E_text сохранён:", OUT_PATH, "shape:", emb.shape)

if __name__ == "__main__":
    main()
