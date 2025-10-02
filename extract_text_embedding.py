# extract_text_embedding.py
# Конвейер:
#  - извлекаем аудио 16 кГц
#  - подавляем шум (noisereduce)
#  - распознаём речь Whisper -> текст
#  - получаем текстовый эмбеддинг через sentence-transformers/all-MiniLM-L6-v2
#  - сохраняем E_text и (опционально) транскрипт
#
# Зависимости:
#   pip install torch librosa soundfile ffmpeg-python noisereduce scipy
#   pip install openai-whisper
#   pip install sentence-transformers
#   sudo apt install ffmpeg  # или brew install ffmpeg
#
# Примечания:
#   - Для Apple Silicon можно использовать torch+mps (PyTorch>=2).
#   - Модель Whisper можно сменить на "small", "medium", "large-v3" и т.д.

import os
import numpy as np
import librosa
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

def main():
    device_str = get_torch_device_str()
    print("Using device:", device_str)

    # 1) Аудио 16 кГц
    print("🎧 Извлекаем аудио:", VIDEO_PATH)
    y = load_audio_16k_mono(VIDEO_PATH, target_sr=SAMPLE_RATE)
    print("   Длина:", len(y), "сэмплов (~", f"{len(y)/SAMPLE_RATE:.2f}", "сек)")

    # 2) Подавление шума
    print("🧹 Подавление шума...")
    y_clean = denoise_audio(y, SAMPLE_RATE, profile_secs=NOISE_PROFILE_SECS)
    print("   Готово.")

    # 3) Распознавание Whisper
    print("🗣️ Распознаём Whisper:", WHISPER_MODEL)
    whisper_device = "cpu" if device_str == "mps" else device_str   # <<< ключевая строка
    wmodel = whisper.load_model(WHISPER_MODEL, device=whisper_device)
    result = wmodel.transcribe(y_clean, task="transcribe", fp16=False)  # fp16=False для CPU/MPS
    text = (result.get("text") or "").strip()
    print("   Транскрипт (preview):", text[:120] + ("..." if len(text) > 120 else ""))

    # 4) Текстовый эмбеддинг (all-MiniLM-L6-v2)
    print("🔤 Эмбеддинг текста:", SENT_EMB_MODEL)
    sbert = SentenceTransformer(SENT_EMB_MODEL, device=("cpu" if device_str == "mps" else device_str))
    if not text:
        print("⚠️ Пустой текст от Whisper; сохраняю нулевой вектор.")
        emb = np.zeros((384,), dtype=np.float32)
    else:
        emb_vec = sbert.encode(text, normalize_embeddings=True)
        emb = np.asarray(emb_vec, dtype=np.float32)

    # 5) Сохранение
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    if text:
        with open(TRANSCRIPT_TXT, "w", encoding="utf-8") as f:
            f.write(text)
        print("📄 Транскрипт сохранён:", TRANSCRIPT_TXT)

    np.save(OUT_PATH, emb)
    print("✅ E_text сохранён:", OUT_PATH, "shape:", emb.shape)

if __name__ == "__main__":
    main()
