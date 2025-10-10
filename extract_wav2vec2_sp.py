# extract_wav2vec2_sp.py  (chunked)
# Получаем эмбеддинг речи E_aud_sp без OOM:
#  - извлекаем аудио 16 кГц из видео
#  - режем на куски по CHUNK_SEC (например, 20с)
#  - прогоняем через Wav2Vec2, аккуратно суммируем скрытые состояния
#  - делаем mean-пулинг ПО ВСЕМУ аудио (взвешенно по числу фреймов)
#  - нормализация и сохранение

import os
import numpy as np
import librosa
import torch
from transformers import AutoProcessor, Wav2Vec2Model
import tempfile
import ffmpeg

"""
🗣 extract_wav2vec2_sp.py → Wav2Vec2
    Модель: facebook/wav2vec2-base
    Вход: моно-аудио 16 кГц
    Выход:
        last_hidden_state: [1, T', 768], где T' ≈ длина/20 (фреймы по ~20 мс).
        mean-пулинг по времени →
    📘 E_aud_sp:
        размерность 768,
            признаки речи (акустические): интонации, голос, фонетика, тембр, стиль речи, не смысл слов.
"""

VIDEO_PATH = "data/videos/sample.mp4"
OUT_PATH = "data/embeddings/sample_wav2vec2_sp.npy"
SAMPLE_RATE = 16000
MODEL_ID = "facebook/wav2vec2-base"
CHUNK_SEC = 20.0
OVERLAP_SEC = 0.0

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_audio_16k_mono(path: str, target_sr: int = 16000) -> np.ndarray:
    # Надёжно декодируем через ffmpeg -> wav
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

def main():
    dev = get_device()
    print("Using device:", dev)

    # 1) Аудио 16 кГц
    print("🎧 Извлекаем аудио:", VIDEO_PATH)
    y = load_audio_16k_mono(VIDEO_PATH, target_sr=SAMPLE_RATE)
    n = len(y)
    print("   Длина:", n, f"сэмплов (~{n/SAMPLE_RATE:.2f} сек)")

    # 2) Модель
    print("📥 Загружаем Wav2Vec2:", MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Wav2Vec2Model.from_pretrained(MODEL_ID).to(dev)
    model.eval()

    # 3) Чанкование и аккуратное усреднение по времени
    chunk_size = int(CHUNK_SEC * SAMPLE_RATE)
    hop = int(max(1, chunk_size - int(OVERLAP_SEC * SAMPLE_RATE)))

    total_frames = 0  # суммарное число временных шагов после энкодера
    sum_hidden = None

    with torch.no_grad():
        start = 0
        idx = 0
        while start < n:
            end = min(n, start + chunk_size)
            chunk = y[start:end]

            inputs = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            input_values = inputs["input_values"].to(dev)  # [1, T]

            outputs = model(input_values=input_values)     # last_hidden_state: [1, T', D]
            hidden = outputs.last_hidden_state             # torch.float32
            Tprime = hidden.shape[1]                       # число временных шагов
            # аккуратно суммируем по времени для глобального среднего
            chunk_sum = hidden.sum(dim=1)                  # [1, D]
            if sum_hidden is None:
                sum_hidden = chunk_sum.squeeze(0).cpu()
            else:
                sum_hidden += chunk_sum.squeeze(0).cpu()
            total_frames += Tprime

            idx += 1
            print(f"   ✔️ chunk {idx}: samples [{start}:{end}) -> frames {Tprime}")

            if end == n:
                break
            start += hop

    if total_frames == 0:
        raise RuntimeError("Wav2Vec2 вернул ноль временных кадров.")

    # Глобальное среднее по времени
    mean_hidden = (sum_hidden / float(total_frames))  # [D]
    # L2-нормализация
    emb = torch.nn.functional.normalize(mean_hidden.unsqueeze(0), dim=-1).squeeze(0).numpy()

    # 4) Сохранение
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.save(OUT_PATH, emb.astype(np.float32))
    print("✅ E_aud_sp сохранён:", OUT_PATH, "shape:", emb.shape)

if __name__ == "__main__":
    main()
