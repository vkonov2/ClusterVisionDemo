import os
import numpy as np
import librosa
import tempfile
import ffmpeg
import tensorflow as tf
import tensorflow_hub as hub

VIDEO_PATH = "data/videos/sample.mp4"
OUT_PATH = "data/embeddings/sample_yamnet_bg.npy"
SAMPLE_RATE = 16000

def load_audio_16k_mono(path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Сначала пытаемся через librosa+soundfile.
    Если soundfile недоступен — декодируем через ffmpeg во временный WAV и снова читаем librosa.
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

def main():
    print("🎧 Извлечение аудио из:", VIDEO_PATH)

    # 1) Загружаем аудио
    waveform = load_audio_16k_mono(VIDEO_PATH, target_sr=SAMPLE_RATE)
    print("   Длина аудио:", len(waveform), "сэмплов,", f"{len(waveform)/SAMPLE_RATE:.2f} сек")

    # 2) Загружаем модель YAMNet
    print("📥 Загрузка YAMNet...")
    yamnet = hub.KerasLayer("https://tfhub.dev/google/yamnet/1")
    print("✅ Модель загружена.")

    # 3) Прогоняем через YAMNet
    waveform_tf = tf.convert_to_tensor(waveform, dtype=tf.float32)
    scores, embeddings, spectrogram = yamnet(waveform_tf)  # embeddings: [num_windows, 1024]
    emb_np = embeddings.numpy()
    print("   Окон эмбеддинга:", emb_np.shape[0], "Размерность:", emb_np.shape[1])

    # 4) Mean pooling
    E_aud_bg = emb_np.mean(axis=0)  # (1024,)
    print("   Итоговый эмбеддинг E_aud_bg:", E_aud_bg.shape)

    # 5) Сохраняем
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
