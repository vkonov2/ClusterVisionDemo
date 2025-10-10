import os
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch
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

def sample_frame_indices(n_total: int, n_want: int) -> List[int]:
    if n_total <= 0:
        return []
    if n_total <= n_want:
        idxs = list(range(n_total))
        while len(idxs) < n_want:
            idxs.append(idxs[-1])
        return idxs[:n_want]
    lin = np.linspace(0, n_total - 1, n_want)
    return [int(round(x)) for x in lin]

def read_frames_opencv(path: str, num_frames: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError("Cannot read frame count")

    idxs = sample_frame_indices(total, num_frames)
    if not idxs:
        raise RuntimeError("No frame indices requested")

    frames: List[Optional[np.ndarray]] = [None] * len(idxs)
    next_ptr = 0
    target_idx = idxs[next_ptr]
    frame_no = 0

    while True:
        if not cap.grab():
            break

        if frame_no == target_idx:
            ok, frame_bgr = cap.retrieve()
            if not ok or frame_bgr is None:
                cap.release()
                raise RuntimeError(f"Failed to decode frame {frame_no}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames[next_ptr] = frame_rgb
            next_ptr += 1

            while next_ptr < len(idxs) and idxs[next_ptr] == frame_no:
                frames[next_ptr] = frame_rgb
                next_ptr += 1

            if next_ptr >= len(idxs):
                break
            target_idx = idxs[next_ptr]

        frame_no += 1
        if frame_no > target_idx and next_ptr < len(idxs):
            # Seek forward by reading additional frames until we reach the target.
            # cap.grab advances by one frame; the loop will handle further grabs.
            continue

    cap.release()

    valid_frames = [f for f in frames if f is not None]
    if not valid_frames:
        raise RuntimeError("No frames decoded")

    while len(valid_frames) < num_frames:
        valid_frames.append(valid_frames[-1])

    return valid_frames[:num_frames]

def _ensure_batch_dimension(pixel_values: torch.Tensor) -> torch.Tensor:
    """Приводим форму к [B, T, C, H, W]."""

    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(0)
    elif pixel_values.ndim > 5:
        while pixel_values.ndim > 5:
            pixel_values = pixel_values.squeeze(0)
    return pixel_values


def extract_timesformer_embedding(
    video_path: str,
    num_frames: int = NUM_FRAMES,
    device: Optional[torch.device] = None,
    processor: Optional[AutoImageProcessor] = None,
    model: Optional[TimesformerModel] = None,
) -> np.ndarray:
    """Извлекает CLS-эмбеддинг TimeSformer из видео."""

    dev = device or get_device()

    own_processor = processor is None
    own_model = model is None

    if processor is None:
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    if model is None:
        model = TimesformerModel.from_pretrained(MODEL_ID).to(dev)
        model.eval()

    frames: Sequence[np.ndarray] = read_frames_opencv(video_path, num_frames)
    proc_out = processor(images=list(frames), return_tensors="pt")
    pixel_values = _ensure_batch_dimension(proc_out["pixel_values"]).to(dev)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        cls = outputs.last_hidden_state[:, 0, :]
        emb = torch.nn.functional.normalize(cls, dim=-1).squeeze(0).cpu().numpy()

    if own_model:
        del model
    if own_processor:
        del processor

    return emb.astype(np.float32)


def main():
    dev = get_device()
    print("Using device:", dev)

    emb = extract_timesformer_embedding(VIDEO_PATH, num_frames=NUM_FRAMES, device=dev)

    os.makedirs("data/embeddings", exist_ok=True)
    out_path = "data/embeddings/sample_timesformer.npy"
    np.save(out_path, emb)
    print("✅ Эмбеддинг сохранён:", out_path, "shape:", emb.shape)

if __name__ == "__main__":
    main()
