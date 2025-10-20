import os
from typing import Optional

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tempfile
import ffmpeg

"""
🎧 extract_yamnet_bg.py → YAMNet
    Модель: https://tfhub.dev/google/yamnet/1
    Вход: моно-аудио 16 кГц
    Выход:
        embeddings: [N, 1024] — окно ~0.48 сек с шагом 0.48 сек,
        scores: [N, 521] — вероятности классов аудио (Speech, Music, Noise, Animal и т.п.).
        mean-пулинг по времени →
    📘 E_aud_bg:
        размерность 1024,
        признаки фонового звука: шумы, музыка, окружение, но не содержание речи.
"""

VIDEO_PATH = "data/videos/sample.mp4"
OUT_PATH = "data/embeddings/sample_yamnet_bg.npy"
SAMPLE_RATE = 16000

def load_audio_16k_mono(path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Декодируем через ffmpeg во временный WAV и снова читаем librosa.
    """
    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "audio_16k.wav")
        (
            ffmpeg
            .input(path)
            .output(
                wav_path,
                format="wav",
                acodec="pcm_s16le",
                ac=1,              # mono
                ar=str(target_sr)  # 16 kHz
            )
            .overwrite_output()
            .run(quiet=True)
        )
        y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
        return y.astype(np.float32)

def extract_yamnet_background_embedding(
    video_path: str,
    sample_rate: int = SAMPLE_RATE,
    yamnet_layer: Optional[hub.KerasLayer] = None,
) -> np.ndarray:
    """Возвращает mean-pooled фоновые аудио-признаки из YAMNet."""

    waveform = load_audio_16k_mono(video_path, target_sr=sample_rate)

    own_layer = yamnet_layer is None
    if yamnet_layer is None:
        yamnet_layer = hub.KerasLayer("https://tfhub.dev/google/yamnet/1")

    waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
    _, embeddings, _ = yamnet_layer(waveform_tf)
    emb_np = embeddings.numpy()
    if emb_np.size == 0:
        raise RuntimeError("YAMNet вернул пустой эмбеддинг")

    pooled = emb_np.mean(axis=0).astype(np.float32)

    if own_layer:
        del yamnet_layer

    return pooled


def main():
    print("🎧 Извлечение аудио из:", VIDEO_PATH)

    E_aud_bg = extract_yamnet_background_embedding(VIDEO_PATH, sample_rate=SAMPLE_RATE)
    print("   Итоговый эмбеддинг E_aud_bg:", E_aud_bg.shape)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.save(OUT_PATH, E_aud_bg.astype(np.float32))
    print("✅ Эмбеддинг сохранён:", OUT_PATH)

if __name__ == "__main__":
    # GPU настройка (необязательно)
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

    main()
