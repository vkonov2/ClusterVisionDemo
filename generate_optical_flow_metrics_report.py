"""Generate optical-flow metrics and an interactive HTML report for a video.

This script estimates dense optical flow (TV-L1 if available, otherwise Farnebäck)
for a video, decomposes motion into camera vs. object components with an affine
model, and summarizes motion energy, chaos, and saliency in Russian with Plotly
visualizations.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.utils import PlotlyJSONEncoder


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
        M, _ = cv2.estimateAffine2D(pts0, pts1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    except Exception:
        M = None

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
    flows: List[np.ndarray], fps: float, grid_size: Tuple[int, int] = (16, 9)
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
    fig.add_trace(
        go.Scatter(
            x=metrics.times.tolist(),
            y=metrics.mean_motion.tolist(),
            name="Mean |F|",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=metrics.times.tolist(),
            y=metrics.p95_motion.tolist(),
            name="95-перц. |F|",
            mode="lines",
        )
    )
    fig.update_layout(
        title="Профиль движения: средний и 95-й перцентиль модуля потока",
        xaxis_title="Время, с",
        yaxis_title="Пикселей/кадр",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def plot_entropy(metrics: MotionMetrics) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=metrics.times.tolist(),
            y=metrics.entropy.tolist(),
            mode="lines",
            name="Энтропия направлений",
        )
    )
    fig.update_layout(
        title="Хаотичность движения (энтропия направлений)",
        xaxis_title="Время, с",
        yaxis_title="Энтропия, бит",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def plot_acceleration(metrics: MotionMetrics) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=metrics.times.tolist(),
            y=metrics.acceleration.tolist(),
            name="Δ mean |F|",
        )
    )
    fig.update_layout(
        title="Ускорение/рывки движения (кадр-кадр)",
        xaxis_title="Время, с",
        yaxis_title="Изменение mean |F|",
        template="plotly_white",
    )
    return fig


def plot_camera_vs_object(metrics: MotionMetrics) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=metrics.times.tolist(),
            y=metrics.camera_energy.tolist(),
            name="Движение камеры",
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=metrics.times.tolist(),
            y=metrics.residual_energy.tolist(),
            name="Движение объектов",
            mode="lines",
        )
    )
    fig.update_layout(
        title="Декомпозиция: энергия движения камеры vs объектов",
        xaxis_title="Время, с",
        yaxis_title="Средний модуль потока",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def plot_saliency_map(metrics: MotionMetrics) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=metrics.saliency_map,
            x=metrics.grid_x.tolist(),
            y=metrics.grid_y.tolist(),
            colorscale="Turbo",
            colorbar_title="Средний |F|",
        )
    )
    fig.update_layout(
        title="Карта motion-saliency (остаточный поток объектов)",
        xaxis_title="Нормированная ширина",
        yaxis_title="Нормированная высота",
        template="plotly_white",
        width=720,
        height=405,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def encode_frame(frame: np.ndarray) -> str:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    success, buffer = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not success:
        raise RuntimeError("Не удалось закодировать кадр в JPEG")
    b64 = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def make_frame_pairs(frames: List[np.ndarray]) -> List[dict]:
    pairs = []
    for idx in range(len(frames) - 1):
        pairs.append({"prev": encode_frame(frames[idx]), "next": encode_frame(frames[idx + 1])})
    return pairs


def generate_report(metrics: MotionMetrics, frames: List[np.ndarray], video_path: Path, output_html: Path) -> None:
    output_html.parent.mkdir(parents=True, exist_ok=True)

    fig_timeline = plot_motion_timeline(metrics)
    fig_entropy = plot_entropy(metrics)
    fig_accel = plot_acceleration(metrics)
    fig_decomp = plot_camera_vs_object(metrics)
    fig_saliency = plot_saliency_map(metrics)

    mean_motion = float(np.nanmean(metrics.mean_motion))
    p95_motion = float(np.nanmean(metrics.p95_motion))
    entropy_mean = float(np.nanmean(metrics.entropy))
    ratio = float(metrics.residual_ratio)

    frame_pairs = make_frame_pairs(frames)

    timeline_json = json.dumps(fig_timeline, cls=PlotlyJSONEncoder)
    entropy_json = json.dumps(fig_entropy, cls=PlotlyJSONEncoder)
    accel_json = json.dumps(fig_accel, cls=PlotlyJSONEncoder)
    decomp_json = json.dumps(fig_decomp, cls=PlotlyJSONEncoder)
    saliency_json = json.dumps(fig_saliency, cls=PlotlyJSONEncoder)
    frames_json = json.dumps(frame_pairs)

    html = f"""<!DOCTYPE html>
<html lang=\"ru\">
<head>
  <meta charset=\"UTF-8\" />
  <title>Аналитический отчёт по оптическому потоку</title>
  <script src=\"https://cdn.plot.ly/plotly-2.29.1.min.js\"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; line-height: 1.55; }}
    h2 {{ margin-top: 1.4em; }}
    h3 {{ margin-top: 1.1em; }}
    .metric {{ background:#f5f7fb; padding:12px 16px; border-radius:8px; margin:6px 0; }}
    .frame-pair {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0 32px; align-items: flex-start; }}
    .frame-pair img {{ width: 240px; max-width: 100%; border-radius: 8px; box-shadow: 0 6px 20px rgba(0,0,0,0.08); background: #111; object-fit: contain; }}
    .frame-caption {{ font-size: 14px; color: #444; margin: 4px 0 0 0; }}
    #fig-saliency {{ max-width: 760px; margin: 0 auto; }}
  </style>
</head>
<body>

<h1>Аналитический отчёт по оптическому потоку</h1>
<p>Видео: <strong>{video_path}</strong>. Оптический поток описывает, как смещаются пиксели между соседними кадрами: каждому пикселю соответствует вектор (δx, δy). Мы нормализуем fps, считаем плотный поток (TV-L1 если доступен, иначе Farnebäck), отделяем движение камеры (аффинная модель) от движения объектов и строим интерактивные графики.</p>

<h2>Сводные показатели</h2>
<div class=\"metric\">Средний модуль потока: <strong>{mean_motion:.3f}</strong> пикс/кадр</div>
<div class=\"metric\">95-й перцентиль: <strong>{p95_motion:.3f}</strong> пикс/кадр — значение, которое превышает только 5% самых быстрых пикселей; фиксирует всплески движения.</div>
<div class=\"metric\">Средняя энтропия направлений: <strong>{entropy_mean:.3f}</strong> бит — 0 бит = все движения в одну сторону, выше → хаос направлений.</div>
<div class=\"metric\">Соотношение объектов/камеры: <strong>{ratio:.2f}×</strong> — насколько энергия движения объектов выше энергии движения камеры (&gt;1× = объекты доминируют).</div>

<h2>Графики и кадры</h2>

<h3>1. Профиль движения</h3>
<p>График показывает, как меняется средний модуль потока и 95-й перцентиль (вспышки). Ниже — пара кадров для выбранной точки на графике: слева кадр t, справа t+1.</p>
<div id=\"fig-timeline\"></div>
<div class=\"frame-pair\">
  <div>
    <img id=\"timeline-prev\" alt=\"Кадр t\" />
    <div class=\"frame-caption\">Кадр t</div>
  </div>
  <div>
    <img id=\"timeline-next\" alt=\"Кадр t+1\" />
    <div class=\"frame-caption\">Кадр t+1</div>
  </div>
</div>

<h3>2. Энтропия направлений</h3>
<p>Отражает хаотичность движения: низко = движение преимущественно в одном направлении; высоко = дергание/разброс по всем углам.</p>
<div id=\"fig-entropy\"></div>
<div class=\"frame-pair\">
  <div>
    <img id=\"entropy-prev\" alt=\"Кадр t\" />
    <div class=\"frame-caption\">Кадр t</div>
  </div>
  <div>
    <img id=\"entropy-next\" alt=\"Кадр t+1\" />
    <div class=\"frame-caption\">Кадр t+1</div>
  </div>
</div>

<h3>3. Ускорение движения</h3>
<p>Δ mean|F| кадр-к-кадру. Пики показывают рывки — смены планов, внезапное появление объектов. Кадры ниже обновляются при наведении/клике на столбец.</p>
<div id=\"fig-accel\"></div>
<div class=\"frame-pair\">
  <div>
    <img id=\"accel-prev\" alt=\"Кадр t\" />
    <div class=\"frame-caption\">Кадр t</div>
  </div>
  <div>
    <img id=\"accel-next\" alt=\"Кадр t+1\" />
    <div class=\"frame-caption\">Кадр t+1</div>
  </div>
</div>

<h3>4. Движение камеры vs объектов</h3>
<p>Аффинная модель описывает движение камеры; остаток — движение объектов. Если объектная кривая выше, внимание зрителя скорее цепляется за движущиеся элементы кадра, а не за панораму/зум камеры.</p>
<div id=\"fig-decomp\"></div>
<div class=\"frame-pair\">
  <div>
    <img id=\"decomp-prev\" alt=\"Кадр t\" />
    <div class=\"frame-caption\">Кадр t</div>
  </div>
  <div>
    <img id=\"decomp-next\" alt=\"Кадр t+1\" />
    <div class=\"frame-caption\">Кадр t+1</div>
  </div>
</div>

<h3>5. Карта motion-saliency</h3>
<p>Более детализированная тепловая карта (16×9 ячеек): где остаточный поток (движение объектов) максимален. Эти зоны естественно притягивают взгляд.</p>
<div id=\"fig-saliency\"></div>

<script>
  const timelineFig = {timeline_json};
  const entropyFig = {entropy_json};
  const accelFig = {accel_json};
  const decompFig = {decomp_json};
  const saliencyFig = {saliency_json};
  const framePairs = {frames_json};

  Plotly.newPlot('fig-timeline', timelineFig.data, timelineFig.layout);
  Plotly.newPlot('fig-entropy', entropyFig.data, entropyFig.layout);
  Plotly.newPlot('fig-accel', accelFig.data, accelFig.layout);
  Plotly.newPlot('fig-decomp', decompFig.data, decompFig.layout);
  Plotly.newPlot('fig-saliency', saliencyFig.data, saliencyFig.layout);

  function attachFramePreview(divId, prevId, nextId) {{
    const div = document.getElementById(divId);
    const prevImg = document.getElementById(prevId);
    const nextImg = document.getElementById(nextId);
    if (!div || !prevImg || !nextImg || framePairs.length === 0) return;

    function setFromIndex(idx) {{
      if (idx == null) return;
      const safeIdx = Math.max(0, Math.min(framePairs.length - 1, idx));
      const pair = framePairs[safeIdx];
      if (!pair) return;
      prevImg.src = pair.prev;
      nextImg.src = pair.next;
    }}

    setFromIndex(0);
    div.on('plotly_hover', (ev) => {{ const idx = ev?.points?.[0]?.pointIndex; setFromIndex(idx); }});
    div.on('plotly_click', (ev) => {{ const idx = ev?.points?.[0]?.pointIndex; setFromIndex(idx); }});
  }}

  attachFramePreview('fig-timeline', 'timeline-prev', 'timeline-next');
  attachFramePreview('fig-entropy', 'entropy-prev', 'entropy-next');
  attachFramePreview('fig-accel', 'accel-prev', 'accel-next');
  attachFramePreview('fig-decomp', 'decomp-prev', 'decomp-next');
</script>

</body>
</html>
"""

    output_html.write_text(html, encoding="utf-8")


def show_interactive_plots(metrics: MotionMetrics) -> None:
    pio.renderers.default = "browser"

    figs = [
        plot_motion_timeline(metrics),
        plot_entropy(metrics),
        plot_acceleration(metrics),
        plot_camera_vs_object(metrics),
        plot_saliency_map(metrics),
    ]

    for fig in figs:
        fig.show()


def main() -> None:
    args = parse_args()
    frames, fps = read_and_resample_video(args.video_path, args.target_fps, args.width, args.height)
    flows, _ = compute_dense_flow(frames)
    metrics = aggregate_motion_metrics(flows, fps, grid_size=(16, 9))
    generate_report(metrics, frames, args.video_path, args.output_html)
    print(f"Готово: отчёт сохранён в {args.output_html}")


if __name__ == "__main__":
    main()
