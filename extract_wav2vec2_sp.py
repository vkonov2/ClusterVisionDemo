# extract_wav2vec2_sp.py  (chunked)
# Получаем эмбеддинг речи E_aud_sp без OOM:
#  - извлекаем аудио 16 кГц из видео
#  - режем на куски по CHUNK_SEC (например, 20с)
#  - прогоняем через Wav2Vec2, аккуратно суммируем скрытые состояния
#  - делаем mean-пулинг ПО ВСЕМУ аудио (взвешенно по числу фреймов)
#  - нормализация и сохранение

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import librosa
import numpy as np
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

@dataclass
class Wav2Vec2Resources:
    processor: AutoProcessor
    model: Wav2Vec2Model


def extract_wav2vec2_embedding(
    video_path: str,
    sample_rate: int = SAMPLE_RATE,
    chunk_sec: float = CHUNK_SEC,
    overlap_sec: float = OVERLAP_SEC,
    device: Optional[torch.device] = None,
    resources: Optional[Wav2Vec2Resources] = None,
) -> Tuple[np.ndarray, int]:
    """Возвращает mean-пулинг эмбеддинга речи и длину аудио в сэмплах."""

    dev = device or get_device()
    y = load_audio_16k_mono(video_path, target_sr=sample_rate)
    n = len(y)

    if n == 0:
        raise RuntimeError("Аудиодорожка пуста")

    own_resources = resources is None
    if resources is None:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = Wav2Vec2Model.from_pretrained(MODEL_ID).to(dev)
        model.eval()
        resources = Wav2Vec2Resources(processor=processor, model=model)

    processor = resources.processor
    model = resources.model

    chunk_size = int(chunk_sec * sample_rate)
    hop = int(max(1, chunk_size - int(overlap_sec * sample_rate)))

    total_frames = 0
    sum_hidden = None

    with torch.no_grad():
        start = 0
        while start < n:
            end = min(n, start + chunk_size)
            chunk = y[start:end]

            inputs = processor(chunk, sampling_rate=sample_rate, return_tensors="pt")
            input_values = inputs["input_values"].to(dev)

            outputs = model(input_values=input_values)
            hidden = outputs.last_hidden_state
            Tprime = hidden.shape[1]

            chunk_sum = hidden.sum(dim=1)
            if sum_hidden is None:
                sum_hidden = chunk_sum.squeeze(0).cpu()
            else:
                sum_hidden += chunk_sum.squeeze(0).cpu()
            total_frames += Tprime

            if end == n:
                break
            start += hop

    if total_frames == 0:
        raise RuntimeError("Wav2Vec2 вернул ноль временных кадров.")

    mean_hidden = sum_hidden / float(total_frames)
    emb = torch.nn.functional.normalize(mean_hidden.unsqueeze(0), dim=-1).squeeze(0).numpy()

    if own_resources:
        del model
        del processor

    return emb.astype(np.float32), n


def main():
    dev = get_device()
    print("Using device:", dev)

    emb, _ = extract_wav2vec2_embedding(
        VIDEO_PATH,
        sample_rate=SAMPLE_RATE,
        chunk_sec=CHUNK_SEC,
        overlap_sec=OVERLAP_SEC,
        device=dev,
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.save(OUT_PATH, emb.astype(np.float32))
    print("✅ E_aud_sp сохранён:", OUT_PATH, "shape:", emb.shape)

if __name__ == "__main__":
    main()
