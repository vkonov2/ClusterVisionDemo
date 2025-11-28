"""Generate optical-flow metrics and an interactive HTML report for a video.

This script estimates dense optical flow (TV-L1 if available, otherwise Farnebäck)
for a video, decomposes motion into camera vs. object components with an affine
model, and summarizes motion energy, chaos, and saliency in Russian with Plotly
visualizations.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import plotly.graph_objects as go


@dataclass
class MotionMetrics:
    times: np.ndarray
    mean_motion: np.ndarray
    p95_motion: np.ndarray
    entropy: np.ndarray
    acceleration: np.ndarray
    camera_energy: np.ndarray
    residual_energy: np.ndarray
    residual_ratio: float
    saliency_map: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Построить HTML-отчёт по оптическому потоку и метрикам движения"
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=Path("data/videos/000-youtube.mp4"),
        help="Путь к видеофайлу",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("outputs/optical_flow_report.html"),
        help="Путь для сохранения HTML-отчёта",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=25.0,
        help="FPS для нормализации видео перед расчётом потока",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Ширина кадра для расчёта потока",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=288,
        help="Высота кадра для расчёта потока",
    )
    return parser.parse_args()


def read_and_resample_video(
    video_path: Path, target_fps: float, width: int, height: int
) -> Tuple[List[np.ndarray], float]:
    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    frame_interval = max(1, int(round(src_fps / target_fps)))

    frames: List[np.ndarray] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frame_resized = cv2.resize(frame, (width, height))
            frames.append(frame_resized)
        idx += 1

    cap.release()
    effective_fps = src_fps / frame_interval if frame_interval else target_fps
    return frames, effective_fps


def compute_dense_flow(frames: List[np.ndarray]) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    if len(frames) < 2:
        raise ValueError("Недостаточно кадров для расчёта оптического потока")

    try:
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()  # type: ignore[attr-defined]
    except Exception:
        tvl1 = None

    flows: List[np.ndarray] = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if tvl1 is not None:
            flow = tvl1.calc(prev_gray, gray, None)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
        flows.append(flow)
        prev_gray = gray

    h, w = flows[0].shape[:2]
    return flows, (w, h)


def decompose_camera_motion(flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    pts0 = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32)
    pts1 = (pts0 + flow.reshape(-1, 2)).astype(np.float32)

    try:
        M, inliers = cv2.estimateAffine2D(pts0, pts1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    except Exception:
        M, inliers = None, None

    if M is None:
        return np.zeros_like(flow), flow

    pts0_h = np.concatenate([pts0, np.ones((pts0.shape[0], 1), dtype=np.float32)], axis=1)
    pred = (M @ pts0_h.T).T
    predicted_disp = pred - pts0
    predicted_flow = predicted_disp.reshape(h, w, 2)
    residual_flow = flow - predicted_flow
    return predicted_flow, residual_flow


def entropy_of_flow(flow: np.ndarray) -> float:
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mag = mag.flatten()
    ang = ang.flatten()
    if mag.size == 0:
        return float("nan")

    mag = np.clip(mag, 1e-6, None)
    weights = mag / (np.sum(mag) + 1e-9)
    bins = 36
    hist, _ = np.histogram(ang, bins=bins, range=(0, 2 * np.pi), weights=weights)
    hist = hist / (np.sum(hist) + 1e-9)
    entropy = -np.sum(hist * np.log2(hist + 1e-12))
    return float(entropy)


def aggregate_motion_metrics(
    flows: List[np.ndarray], fps: float, grid_size: Tuple[int, int] = (8, 8)
) -> MotionMetrics:
    fps = fps or 1.0

    mean_motion = []
    p95_motion = []
    entropy_values = []
    camera_energy = []
    residual_energy = []

    for flow in flows:
        mag = np.linalg.norm(flow, axis=2)
        mean_motion.append(float(np.mean(mag)))
        p95_motion.append(float(np.percentile(mag, 95)))
        entropy_values.append(entropy_of_flow(flow))
        cam_flow, residual_flow = decompose_camera_motion(flow)
        camera_energy.append(float(np.mean(np.linalg.norm(cam_flow, axis=2))))
        residual_energy.append(float(np.mean(np.linalg.norm(residual_flow, axis=2))))

    mean_motion = np.array(mean_motion)
    p95_motion = np.array(p95_motion)
    entropy_values = np.array(entropy_values)
    camera_energy = np.array(camera_energy)
    residual_energy = np.array(residual_energy)

    times = np.arange(len(mean_motion)) / fps
    acceleration = np.concatenate([[0.0], np.diff(mean_motion)]) * fps

    residual_ratio = float(np.mean(residual_energy) / (np.mean(camera_energy) + 1e-9))

    # Saliency map: average residual magnitude per grid cell across frames
    h, w = flows[0].shape[:2]
    grid_h, grid_w = grid_size
    saliency = np.zeros((grid_h, grid_w))
    for flow in flows:
        _, residual_flow = decompose_camera_motion(flow)
        mag = np.linalg.norm(residual_flow, axis=2)
        cell_h = h // grid_h
        cell_w = w // grid_w
        for gy in range(grid_h):
            for gx in range(grid_w):
                y0, y1 = gy * cell_h, (gy + 1) * cell_h if gy < grid_h - 1 else h
                x0, x1 = gx * cell_w, (gx + 1) * cell_w if gx < grid_w - 1 else w
                saliency[gy, gx] += np.mean(mag[y0:y1, x0:x1])
    saliency /= len(flows)

    grid_x = np.linspace(0, 1, grid_w)
    grid_y = np.linspace(0, 1, grid_h)

    return MotionMetrics(
        times=times,
        mean_motion=mean_motion,
        p95_motion=p95_motion,
        entropy=entropy_values,
        acceleration=acceleration,
        camera_energy=camera_energy,
        residual_energy=residual_energy,
        residual_ratio=residual_ratio,
        saliency_map=saliency,
        grid_x=grid_x,
        grid_y=grid_y,
    )


def plot_motion_timeline(metrics: MotionMetrics) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics.times, y=metrics.mean_motion, name="Mean |F|", mode="lines"))
    fig.add_trace(go.Scatter(x=metrics.times, y=metrics.p95_motion, name="95-перц. |F|", mode="lines"))
    fig.update_layout(
        title="Профиль движения: средний и 95-й перцентиль модуля потока",
        xaxis_title="Время, с",
        yaxis_title="Пикселей/кадр",
        hovermode="x unified",
    )
    return fig


def plot_entropy(metrics: MotionMetrics) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=metrics.times, y=metrics.entropy, mode="lines", name="Энтропия направлений"))
    fig.update_layout(
        title="Хаотичность движения (энтропия направлений)",
        xaxis_title="Время, с",
        yaxis_title="Энтропия, бит",
        hovermode="x unified",
    )
    return fig


def plot_acceleration(metrics: MotionMetrics) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=metrics.times, y=metrics.acceleration, name="Δ mean |F|"))
    fig.update_layout(
        title="Ускорение/рывки движения (кадр-кадр)",
        xaxis_title="Время, с",
        yaxis_title="Изменение mean |F|",
    )
    return fig


def plot_camera_vs_object(metrics: MotionMetrics) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=metrics.times,
            y=metrics.camera_energy,
            name="Движение камеры",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=metrics.times,
            y=metrics.residual_energy,
            name="Движение объектов",
            mode="lines",
        )
    )
    fig.update_layout(
        title="Декомпозиция: энергия движения камеры vs объектов",
        xaxis_title="Время, с",
        yaxis_title="Средний модуль потока",
        hovermode="x unified",
    )
    return fig


def plot_saliency_map(metrics: MotionMetrics) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=metrics.saliency_map,
            x=metrics.grid_x,
            y=metrics.grid_y,
            colorscale="Turbo",
            colorbar_title="Средний |F|",
        )
    )
    fig.update_layout(
        title="Карта motion-saliency (остаточный поток объектов)",
        xaxis_title="Нормированная ширина",
        yaxis_title="Нормированная высота",
    )
    return fig


def generate_report(metrics: MotionMetrics, video_path: Path, output_html: Path) -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)

    fig_timeline = plot_motion_timeline(metrics)
    fig_entropy = plot_entropy(metrics)
    fig_accel = plot_acceleration(metrics)
    fig_decomp = plot_camera_vs_object(metrics)
    fig_saliency = plot_saliency_map(metrics)

    parts = [
        """<!DOCTYPE html>
<html lang=\"ru\">
<head>
  <meta charset=\"UTF-8\" />
  <title>Оптический поток — аналитический отчёт</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; line-height: 1.55; margin: 32px; }
    h1, h2, h3 { margin-top: 1.2em; }
    .metric { background: #f5f7fb; padding: 12px 16px; border-radius: 8px; margin: 8px 0; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }
    .code { font-family: 'SFMono-Regular', Consolas, monospace; }
  </style>
</head>
<body>
  <h1>Аналитический отчёт по оптическому потоку</h1>
  <p>Видео: <strong>{video}</strong>. Оптический поток описывает, как смещаются пиксели между соседними кадрами: каждому пикселю соответствует вектор (δx, δy). Мы используем плотный поток TV-L1 (при наличии) либо Farnebäck, нормализуем частоту до целевого FPS и декомпозируем движение на вклад камеры и объектов.</p>

  <h2>Краткие определения (простым языком)</h2>
  <ul>
    <li><strong>Оптический поток</strong> — поле векторов смещения между кадрами. Модуль |F| показывает скорость пикселя, угол — направление.</li>
    <li><strong>TV-L1 / Farnebäck</strong> — алгоритмы для оценки потока: TV-L1 устойчив к шуму и резким границам, Farnebäck — быстрый классический метод.</li>
    <li><strong>Декомпозиция камеры</strong> — фитим аффинную модель (смещение+масштаб+поворот) по всему полю. То, что хорошо объясняется моделью, считаем движением камеры; остаток — движение объектов.</li>
    <li><strong>Энтропия направлений</strong> — мера хаотичности: если движение разбросано по всем углам, энтропия выше.</li>
    <li><strong>Motion saliency</strong> — карта, где остаточный поток (объекты) максимален; туда естественно тянется взгляд.</li>
  </ul>

  <h2>Сводные числовые показатели</h2>
  <div class=\"grid\">
    <div class=\"metric\">Средний модуль потока (медиана по роликам): <strong>{mean_motion:.3f}</strong> пикс/кадр</div>
    <div class=\"metric\">95-й перцентиль модуля: <strong>{p95_motion:.3f}</strong> пикс/кадр</div>
    <div class=\"metric\">Энтропия направлений (средняя): <strong>{entropy:.3f}</strong> бит</div>
    <div class=\"metric\">Соотношение объектов/камеры по энергии: <strong>{ratio:.2f}×</strong></div>
  </div>

  <h2>Как читать графики</h2>
  <p><strong>Профиль движения</strong>: средний и 95-й перцентиль |F| показывают базовый уровень и вспышки движения. <strong>Энтропия</strong> — насколько направления хаотичны (хаос = реклама «дергается»). <strong>Ускорение</strong> ловит рывки (внезапные смены). <strong>Декомпозиция</strong> отделяет «камера едет» от «объекты двигаются». <strong>Карта saliency</strong> подсвечивает зоны, где объекты двигаются сильнее всего.</p>

  <h2>Графики</h2>
  <div id=\"timeline\"></div>
  <div id=\"entropy\"></div>
  <div id=\"accel\"></div>
  <div id=\"decomp\"></div>
  <div id=\"saliency\"></div>

  <h2>Практические выводы</h2>
  <ul>
    <li>Если вклад камеры высок и остаток малый — кадр плавный, внимание нужно поддерживать другими каналами (лицо/текст/звук).</li>
    <li>Высокий 95-й перцентиль или ускорение — вспышки движения, которые можно синхронизировать с появлением бренда/CTA.</li>
    <li>Зоны высокой motion-saliency подскажут, где разместить ключевой объект, чтобы совпасть с естественным сдвигом взгляда.</li>
    <li>Избыточная энтропия направлений = хаотичное дергание; это может перегружать зрителя.</li>
  </ul>
</body>
</html>
""".format(
            video=video_path,
            mean_motion=float(np.nanmean(metrics.mean_motion)),
            p95_motion=float(np.nanmean(metrics.p95_motion)),
            entropy=float(np.nanmean(metrics.entropy)),
            ratio=metrics.residual_ratio,
        )
    ]

    parts.append(
        f"<script>Plotly.newPlot('timeline', {fig_timeline.to_json()});"  # type: ignore[str-format]
        f"Plotly.newPlot('entropy', {fig_entropy.to_json()});"
        f"Plotly.newPlot('accel', {fig_accel.to_json()});"
        f"Plotly.newPlot('decomp', {fig_decomp.to_json()});"
        f"Plotly.newPlot('saliency', {fig_saliency.to_json()});"  # noqa: E501
        "</script>"
    )

    output_html.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    frames, fps = read_and_resample_video(args.video_path, args.target_fps, args.width, args.height)
    flows, _ = compute_dense_flow(frames)
    metrics = aggregate_motion_metrics(flows, fps)
    generate_report(metrics, args.video_path, args.output_html)
    print(f"Готово: отчёт сохранён в {args.output_html}")


if __name__ == "__main__":
    main()
