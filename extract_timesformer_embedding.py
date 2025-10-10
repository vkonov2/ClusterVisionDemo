import os
import numpy as np
import torch
import cv2
from transformers import AutoImageProcessor, TimesformerModel

"""
🎬 extract_timesformer_embedding.py → TimeSformer
    Модель: facebook/timesformer-base-finetuned-k400
    Вход: последовательность кадров (видео)
    Выход: last_hidden_state формы [B, T+1, D], где
        T — число патчей (временных окон),
        D = 768 — размерность признаков.
        CLS-токен outputs.last_hidden_state[:, 0, :] и нормализуем.
    📘 Результат:
        E_video,
        размерность: 768,
        содержит видеопризнаки, захватывающие пространственно-временные паттерны: движения, сцены, объекты, контекст.
"""

VIDEO_PATH = "data/videos/sample.mp4"
MODEL_ID = "facebook/timesformer-base-finetuned-k400"
NUM_FRAMES = 16

def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def sample_frame_indices(n_total: int, n_want: int):
    if n_total <= 0:
        return []
    if n_total <= n_want:
        idxs = list(range(n_total))
        while len(idxs) < n_want:
            idxs.append(idxs[-1])
        return idxs[:n_want]
    lin = np.linspace(0, n_total - 1, n_want)
    return [int(round(x)) for x in lin]

def read_frames_opencv(path: str, num_frames: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError("Cannot read frame count")

    idxs = sample_frame_indices(total, num_frames)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()

    if not frames:
        raise RuntimeError("No frames decoded")

    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames[:num_frames]

def main():
    dev = get_device()
    print("Using device:", dev)

    # 1) Модель и препроцессор
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)  # у TimeSformer нет fast-версии
    model = TimesformerModel.from_pretrained(MODEL_ID).to(dev)
    model.eval()

    # 2) Кадры (список np.ndarray HxWxC, RGB, uint8)
    frames = read_frames_opencv(VIDEO_PATH, NUM_FRAMES)

    # 3) Препроцессинг
    proc_out = processor(images=frames, return_tensors="pt")  # обычно [T, C, H, W]
    pixel_values = proc_out["pixel_values"]

    # Приводим форму к [B, T, C, H, W]:
    # - если [T, C, H, W] -> добавляем batch
    # - если [B, T, C, H, W] -> оставляем как есть
    # - если что-то ещё -> аккуратно выжимаем лишние оси
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(0)
    elif pixel_values.ndim > 5:
        # на всякий случай: уберём оси размера 1 сверх нужных
        while pixel_values.ndim > 5:
            pixel_values = pixel_values.squeeze(0)

    pixel_values = pixel_values.to(dev)

    # 4) Инференс и CLS-эмбеддинг
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)  # last_hidden_state: [B, T+1, D]
        cls = outputs.last_hidden_state[:, 0, :]     # [B, D]
        emb = torch.nn.functional.normalize(cls, dim=-1).squeeze(0).cpu().numpy()

    # 5) Сохранение
    os.makedirs("data/embeddings", exist_ok=True)
    out_path = "data/embeddings/sample_timesformer.npy"
    np.save(out_path, emb)
    print("✅ Эмбеддинг сохранён:", out_path, "shape:", emb.shape)

if __name__ == "__main__":
    main()
