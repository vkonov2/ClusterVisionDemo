"""
Простейший stub для decord, использующий imageio.

Реализует:
- VideoReader(uri, ctx=cpu(0), num_threads=1, ...)
- len(vr)
- vr.get_batch(indices).asnumpy()
- vr[idx].asnumpy()
- cpu()
- bridge.set_bridge(...)

Этого обычно хватает для Video-LLaMA.
"""

from __future__ import annotations

from typing import Iterable, List
import numpy as np

try:
    import imageio.v2 as imageio
except ImportError as exc:
    raise ImportError(
        "Stub decord требует imageio. Установите: pip install imageio"
    ) from exc


class _BatchArray:
    """Обёртка, чтобы поддержать .asnumpy() как в настоящем decord."""

    def __init__(self, arr: np.ndarray):
        self._arr = np.asarray(arr)

    def asnumpy(self) -> np.ndarray:
        return self._arr


class VideoReader:
    def __init__(self, uri, ctx=None, num_threads: int = 1, **kwargs):
        """
        Минималистичная реализация VideoReader.

        - Загружает все кадры в память (осторожно с длинными видео).
        - Кадры хранятся как список np.ndarray (H, W, C) uint8.
        """
        self.uri = uri
        self._frames: List[np.ndarray] = []

        reader = imageio.get_reader(uri)
        for frame in reader:
            self._frames.append(np.asarray(frame))

    def __len__(self) -> int:
        return len(self._frames)

    def get_batch(self, indices: Iterable[int]) -> _BatchArray:
        frames = [self._frames[int(i)] for i in indices]
        return _BatchArray(np.stack(frames, axis=0))

    def __getitem__(self, idx: int) -> _BatchArray:
        return _BatchArray(self._frames[int(idx)])


def cpu(device_id: int = 0):
    """
    Заглушка под decord.cpu(0). В оригинале возвращает контекст устройства,
    здесь просто None, т.к. мы всё делаем через numpy.
    """
    return None


class _Bridge:
    """Заглушка для decord.bridge.set_bridge(...)."""

    def set_bridge(self, *args, **kwargs):
        # В настоящем decord это настраивает backend (torch/ndarray).
        # Для нашего stub безразлично — просто игнорируем.
        return None


bridge = _Bridge()
