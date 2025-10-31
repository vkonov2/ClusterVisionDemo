"""Генерация HTML-отчётов с метриками внимания DeepGaze 2e по видеороликам."""
from __future__ import annotations

import argparse
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch
from tqdm import tqdm

from generate_deepgaze_saliency import (
    DeepGazePrediction,
    build_centerbias,
    ensure_centerbias,
    ensure_deepgaze_repo,
    iterate_deepgaze_predictions,
    load_deepgaze_model,
)
from generate_unisal_saliency import read_video_frames, render_heatmap


DEFAULT_SECONDS = 5.0
DEFAULT_OUTPUT_DIR = Path("outputs/deepgaze_metrics")
DEFAULT_VIDEOS_DIR = Path("data/videos")
DEFAULT_CHUNK_SIZE = 4


@dataclass
class FrameSample:
    index: int
    prob_map: np.ndarray
    overlay_full: np.ndarray
    overlay_b64: str
    max_position: tuple[int, int]


@dataclass
class PairSample:
    start_index: int
    end_index: int
    diff_metrics: Dict[str, float]
    maxima_distance: float
    maxima_distance_normalized: float
    diff_image_b64: str
    distance_image_b64: str


def encode_image(image_bgr: np.ndarray, *, fmt: str = ".jpg") -> str:
    """Кодировать изображение в base64 data URI."""

    success, buffer = cv2.imencode(fmt, image_bgr)
    if not success:
        raise RuntimeError("Не удалось закодировать изображение")
    encoded = base64.b64encode(buffer).decode("ascii")
    mime = "image/png" if fmt == ".png" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def resize_preview(image: np.ndarray, *, max_width: int = 420) -> np.ndarray:
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = max_width / float(width)
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def create_overlay(frame_bgr: np.ndarray, prob_map: np.ndarray) -> np.ndarray:
    heatmap = render_heatmap(prob_map)
    return cv2.addWeighted(frame_bgr, 0.55, heatmap, 0.45, 0)


def render_difference_heatmap(diff_map: np.ndarray) -> np.ndarray:
    max_abs = float(np.max(np.abs(diff_map)))
    if max_abs <= 0:
        normalized = np.full_like(diff_map, 128, dtype=np.uint8)
    else:
        scaled = diff_map / (2 * max_abs) + 0.5
        clipped = np.clip(scaled, 0.0, 1.0)
        normalized = (clipped * 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TWILIGHT_SHIFTED)
    return colored


def render_maxima_distance_overlay(
    base_overlay: np.ndarray,
    first_max: tuple[int, int],
    second_max: tuple[int, int],
    distance: float,
    distance_norm: float,
) -> np.ndarray:
    overlay = base_overlay.copy()
    color_first = (0, 255, 255)
    color_second = (255, 128, 0)
    cv2.circle(overlay, (first_max[1], first_max[0]), 10, color_first, thickness=2)
    cv2.circle(overlay, (second_max[1], second_max[0]), 10, color_second, thickness=2)
    cv2.line(
        overlay,
        (first_max[1], first_max[0]),
        (second_max[1], second_max[0]),
        (0, 255, 0),
        thickness=2,
    )
    label = f"{distance:.1f}px · {distance_norm:.2f}"
    cv2.putText(
        overlay,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    return overlay


def compute_concentration(prob_map: np.ndarray, top_fraction: float) -> float:
    flat = prob_map.flatten()
    total = float(flat.sum())
    if total <= 0:
        return 0.0
    count = max(1, int(np.ceil(flat.size * top_fraction)))
    top_indices = np.argpartition(flat, -count)[-count:]
    top_sum = float(flat[top_indices].sum())
    return top_sum / total


def compute_frame_samples(predictions: Sequence[DeepGazePrediction]) -> List[FrameSample]:
    samples: List[FrameSample] = []
    for pred in predictions:
        overlay_full = create_overlay(pred.frame_bgr, pred.prob_map)
        overlay_preview = resize_preview(overlay_full)
        overlay_b64 = encode_image(overlay_preview)
        max_pos = np.unravel_index(np.argmax(pred.prob_map), pred.prob_map.shape)
        samples.append(
            FrameSample(
                index=pred.frame_index,
                prob_map=pred.prob_map,
                overlay_full=overlay_full,
                overlay_b64=overlay_b64,
                max_position=(int(max_pos[0]), int(max_pos[1])),
            ),
        )
    return samples


def compute_pair_samples(frames: Sequence[FrameSample]) -> List[PairSample]:
    pairs: List[PairSample] = []
    if len(frames) < 2:
        return pairs

    prob_height, prob_width = frames[0].prob_map.shape
    diagonal = float(np.hypot(prob_height, prob_width))
    diagonal = max(diagonal, 1.0)

    for current, nxt in zip(frames[:-1], frames[1:]):
        diff_map = nxt.prob_map - current.prob_map
        diff_metrics = {
            "mean_abs": float(np.mean(np.abs(diff_map))),
            "rms": float(np.sqrt(np.mean(np.square(diff_map)))),
            "max_abs": float(np.max(np.abs(diff_map))),
        }
        diff_image = render_difference_heatmap(diff_map)
        diff_small = resize_preview(diff_image)
        diff_b64 = encode_image(diff_small, fmt=".png")

        max_a = np.array(current.max_position, dtype=np.float32)
        max_b = np.array(nxt.max_position, dtype=np.float32)
        distance = float(np.linalg.norm(max_b - max_a))
        distance_norm = float(distance / diagonal)

        distance_overlay_full = render_maxima_distance_overlay(
            nxt.overlay_full,
            current.max_position,
            nxt.max_position,
            distance,
            distance_norm,
        )
        distance_small = resize_preview(distance_overlay_full)
        distance_b64 = encode_image(distance_small)

        pairs.append(
            PairSample(
                start_index=current.index,
                end_index=nxt.index,
                diff_metrics=diff_metrics,
                maxima_distance=distance,
                maxima_distance_normalized=distance_norm,
                diff_image_b64=diff_b64,
                distance_image_b64=distance_b64,
            ),
        )
    return pairs


def build_single_frame_figure(
    *,
    title: str,
    x: Sequence[int],
    y: Sequence[float],
    avg_value: float,
    value_label: str,
    customdata: Sequence[int],
    color: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            marker=dict(size=8, color=color),
            line=dict(color=color),
            customdata=customdata,
            hovertemplate="Кадр %{x}<br>" + value_label + ": %{y:.4f}<extra></extra>",
        ),
    )
    fig.add_hline(
        y=avg_value,
        line_dash="dash",
        line_color="#666",
        annotation_text=f"Среднее: {avg_value:.4f}",
        annotation_position="top right",
        annotation_font=dict(color="#333"),
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Номер кадра",
        yaxis_title=value_label,
        hovermode="x",
    )
    fig.update_yaxes(range=[0, 1], constrain="domain")
    return fig


def build_pair_figure(
    *,
    title: str,
    x_labels: Sequence[str],
    y_values: Sequence[float],
    avg_value: float,
    value_label: str,
    customdata: Sequence[Sequence[object]],
    color: str,
    hovertemplate: str | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=y_values,
            mode="lines+markers",
            marker=dict(size=8, color=color),
            line=dict(color=color),
            customdata=customdata,
            hovertemplate=(
                hovertemplate
                or ("Кадры %{x}<br>" + value_label + ": %{y:.4f}<extra></extra>")
            ),
        ),
    )
    fig.add_hline(
        y=avg_value,
        line_dash="dash",
        line_color="#666",
        annotation_text=f"Среднее: {avg_value:.4f}",
        annotation_position="top right",
        annotation_font=dict(color="#333"),
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Пара кадров",
        yaxis_title=value_label,
        hovermode="x",
        xaxis_tickangle=-90,
    )
    fig.update_xaxes(type="category")
    fig.update_yaxes(range=[0, 1], constrain="domain")
    return fig


def figure_to_html(fig: go.Figure, div_id: str) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id=div_id)


def render_report_html(
    *,
    video_path: Path,
    seconds: float,
    fps: float,
    frame_samples: Sequence[FrameSample],
    pair_samples: Sequence[PairSample],
    figures: Dict[str, str],
) -> str:
    frame_overlays = {str(sample.index): sample.overlay_b64 for sample in frame_samples}
    pair_diff_images: Dict[str, str] = {}
    pair_distance_images: Dict[str, str] = {}
    for pair in pair_samples:
        key = f"{pair.start_index:04d}-{pair.end_index:04d}"
        pair_diff_images[key] = pair.diff_image_b64
        pair_distance_images[key] = pair.distance_image_b64

    frame_json = json.dumps(frame_overlays, ensure_ascii=False)
    diff_json = json.dumps(pair_diff_images, ensure_ascii=False)
    distance_json = json.dumps(pair_distance_images, ensure_ascii=False)

    initial_frame_key = next(iter(frame_overlays.keys()), "")
    initial_pair_key = next(iter(pair_diff_images.keys()), "")
    if pair_samples:
        first_pair = pair_samples[0]
        initial_pair_a = str(first_pair.start_index)
        initial_pair_b = str(first_pair.end_index)
    else:
        initial_pair_a = initial_frame_key
        initial_pair_b = initial_frame_key

    def format_frame_caption(label: str, key: str, *, default: str = "Наведите курсор на график") -> str:
        if not key:
            return default
        try:
            number = int(key) + 1
        except ValueError:
            return default
        normalized_label = label.strip().lower()
        if normalized_label.startswith("кадр"):
            return f"{label} · {number}"
        return f"{label} · кадр {number}"

    def format_pair_label(a_key: str, b_key: str) -> str:
        try:
            a_num = int(a_key) + 1
            b_num = int(b_key) + 1
        except (ValueError, TypeError):
            return ""
        return f"{a_num}-{b_num}"

    initial_pair_label = format_pair_label(initial_pair_a, initial_pair_b)
    initial_pair_distance_value = pair_samples[0].maxima_distance if pair_samples else None

    single_metric_definitions = [
        (
            "mean",
            "Mean",
            "Mean — средняя вероятность по кадру, показывающая общий уровень распределённого внимания.",
        ),
        (
            "peak",
            "Peak",
            "Peak — максимальная вероятность в кадре, отражающая силу наиболее заметной области.",
        ),
        (
            "conc10",
            "Concentration 10%",
            "Concentration 10% — доля вероятности в верхних 10% пикселей, характеризующая компактность внимания.",
        ),
        (
            "conc20",
            "Concentration 20%",
            "Concentration 20% — доля вероятности в верхних 20% пикселей, фиксирующая ширину зоны интереса.",
        ),
    ]

    single_metrics_html = "".join(
        f"""
            <div class=\"metric-block\">
                <div class=\"figure\">{figures[key]}</div>
                <p class=\"metric-note\">{description}</p>
                <div class=\"metric-preview-wrapper\">
                    <div class=\"preview-card\">
                        <h3>{display} · выбранный кадр</h3>
                        <img id=\"{key}-preview\" src=\"{frame_overlays.get(initial_frame_key, '')}\" alt=\"{display} overlay\" />
                        <div class=\"caption\" id=\"{key}-caption\">{format_frame_caption(display, initial_frame_key)}</div>
                    </div>
                </div>
            </div>
        """
        for key, display, description in single_metric_definitions
    )

    volatility1_definitions = [
        (
            "vol1_mean_abs",
            "vol1-mean",
            "Mean |Δ|",
            "Mean |Δ| — среднее абсолютное изменение между соседними картами, показывающее, насколько сильно перераспределяется внимание.",
        ),
        (
            "vol1_rms",
            "vol1-rms",
            "RMS Δ",
            "RMS Δ — среднеквадратичное изменение, чувствительное к единичным резким скачкам внимания.",
        ),
        (
            "vol1_max_abs",
            "vol1-max",
            "Max |Δ|",
            "Max |Δ| — максимальное точечное изменение вероятности, отражающее самый заметный локальный сдвиг.",
        ),
    ]

    if pair_samples:
        initial_pair_caption_a = format_frame_caption("Кадр A", initial_pair_a, default="Наведите курсор")
        initial_pair_caption_b = format_frame_caption("Кадр B", initial_pair_b, default="Наведите курсор")
        if initial_pair_label:
            distance_suffix = (
                f" ({initial_pair_distance_value:.1f} px)"
                if initial_pair_distance_value is not None
                else ""
            )
            initial_distance_caption = f"Смещение максимума · {initial_pair_label}{distance_suffix}"
        else:
            initial_distance_caption = "Наведите курсор"

        volatility1_html = "".join(
            f"""
                <div class=\"metric-block\">
                    <div class=\"figure\">{figures[key]}</div>
                    <p class=\"metric-note\">{description}</p>
                    <div class=\"pair-preview-grid\">
                        <div class=\"preview-card\">
                            <h3>Кадр A</h3>
                            <img id=\"{prefix}-frame-a\" src=\"{frame_overlays.get(initial_pair_a, '')}\" alt=\"frame A\" />
                            <div class=\"caption\" id=\"{prefix}-caption-a\">{initial_pair_caption_a}</div>
                        </div>
                        <div class=\"preview-card\">
                            <h3>Кадр B</h3>
                            <img id=\"{prefix}-frame-b\" src=\"{frame_overlays.get(initial_pair_b, '')}\" alt=\"frame B\" />
                            <div class=\"caption\" id=\"{prefix}-caption-b\">{initial_pair_caption_b}</div>
                        </div>
                        <div class=\"preview-card\">
                            <h3>Разность карт</h3>
                            <img id=\"{prefix}-third\" src=\"{pair_diff_images.get(initial_pair_key, '')}\" alt=\"difference\" />
                        <div class=\"caption\" id=\"{prefix}-caption-third\">{((display + ' · ' + initial_pair_label) if initial_pair_label else 'Наведите курсор')}</div>
                        </div>
                    </div>
                </div>
            """
            for key, prefix, display, description in volatility1_definitions
        )
    else:
        volatility1_html = '<p class="empty-note">Недостаточно кадров для расчёта (нужно ≥2).</p>'

    if pair_samples:
        vol2_prefix = "vol2-distance"
        volatility2_html = f"""
            <div class=\"metric-block\">
                <div class=\"figure\">{figures['vol2_distance']}</div>
                <p class=\"metric-note\">Нормированное расстояние между максимумами показывает, насколько далеко смещается главный пик внимания между соседними кадрами.</p>
                <div class=\"pair-preview-grid\">
                    <div class=\"preview-card\">
                        <h3>Кадр A</h3>
                        <img id=\"{vol2_prefix}-frame-a\" src=\"{frame_overlays.get(initial_pair_a, '')}\" alt=\"frame A\" />
                        <div class=\"caption\" id=\"{vol2_prefix}-caption-a\">{initial_pair_caption_a}</div>
                    </div>
                    <div class=\"preview-card\">
                        <h3>Кадр B</h3>
                        <img id=\"{vol2_prefix}-frame-b\" src=\"{frame_overlays.get(initial_pair_b, '')}\" alt=\"frame B\" />
                        <div class=\"caption\" id=\"{vol2_prefix}-caption-b\">{initial_pair_caption_b}</div>
                    </div>
                    <div class=\"preview-card\">
                        <h3>Максимумы и расстояние</h3>
                        <img id=\"{vol2_prefix}-third\" src=\"{pair_distance_images.get(initial_pair_key, '')}\" alt=\"distance\" />
                        <div class=\"caption\" id=\"{vol2_prefix}-caption-third\">{initial_distance_caption}</div>
                    </div>
                </div>
            </div>
        """
    else:
        volatility2_html = '<p class="empty-note">Недостаточно кадров для расчёта (нужно ≥2).</p>'

    html = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="utf-8" />
    <title>DeepGaze 2e — метрики внимания: {video_path.name}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <script src="https://cdn.plot.ly/plotly-2.29.1.min.js"></script>
    <style>
        body {{
            font-family: "Inter", "Segoe UI", sans-serif;
            margin: 0;
            padding: 0;
            background: #f6f7fb;
            color: #1a1c23;
        }}
        header {{
            background: #14171f;
            color: white;
            padding: 24px 32px;
        }}
        main {{
            padding: 24px 32px 48px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        section {{
            margin-bottom: 48px;
            background: white;
            border-radius: 16px;
            box-shadow: 0 12px 32px rgba(20, 25, 40, 0.12);
            padding: 24px 28px;
        }}
        h1 {{
            margin: 0;
            font-size: 28px;
        }}
        h2 {{
            font-size: 22px;
            margin-bottom: 16px;
        }}
        h3 {{
            font-size: 18px;
            margin-top: 24px;
            margin-bottom: 12px;
        }}
        .figure {{
            margin-bottom: 16px;
        }}
        .preview-card {{
            background: #0f111a;
            color: white;
            border-radius: 14px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            min-height: 300px;
        }}
        .preview-card img {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 8px 18px rgba(0, 0, 0, 0.35);
        }}
        .metric-block {{
            margin-bottom: 32px;
        }}
        .metric-note {{
            margin: 0 0 16px;
            color: #4c5166;
            line-height: 1.5;
        }}
        .metric-preview-wrapper {{
            display: flex;
            justify-content: center;
        }}
        .metric-preview-wrapper .preview-card {{
            width: 320px;
        }}
        .caption {{
            margin-top: 12px;
            font-size: 15px;
            text-align: center;
        }}
        .pair-preview-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-top: 16px;
        }}
        footer {{
            text-align: center;
            padding: 24px;
            color: #5a6073;
        }}
        .metric-description {{
            margin-bottom: 18px;
            color: #3c4153;
        }}
        .empty-note {{
            color: #777d90;
            font-style: italic;
            margin-top: 8px;
        }}
    </style>
</head>
<body>
    <header>
        <h1>DeepGaze 2e · {video_path.name}</h1>
        <p>Первые {seconds:g} секунд, частота кадров ≈ {fps:.2f} FPS</p>
    </header>
    <main>
        <section>
            <h2>Покадровые метрики</h2>
            <p class="metric-description">
                Наведите курсор на точку графика, чтобы увидеть соответствующий кадр с тепловой картой.
            </p>
            {single_metrics_html}
        </section>
        <section>
            <h2>Volatility 1 — динамика изменений</h2>
            <p class="metric-description">
                Для каждой пары соседних кадров показаны три метрики разницы тепловых карт. При наведении отображаются обе карты и их разность.
            </p>
            {volatility1_html}
        </section>
        <section>
            <h2>Volatility 2 — перемещение максимума внимания</h2>
            <p class="metric-description">
                Оценивает расстояние между точками максимума вероятности на соседних кадрах. При наведении отображается линия, соединяющая пики.
            </p>
            {volatility2_html}
        </section>
    </main>
    <footer>Отчёт создан автоматически скриптом DeepGaze metrics.</footer>
    <script>
        const frameOverlays = {frame_json};
        const pairDiffImages = {diff_json};
        const pairDistanceImages = {distance_json};

        const singleMetricConfigs = [
            {{ figureId: 'fig-mean', imgId: 'mean-preview', captionId: 'mean-caption', label: 'Mean' }},
            {{ figureId: 'fig-peak', imgId: 'peak-preview', captionId: 'peak-caption', label: 'Peak' }},
            {{ figureId: 'fig-conc10', imgId: 'conc10-preview', captionId: 'conc10-caption', label: 'Concentration 10%' }},
            {{ figureId: 'fig-conc20', imgId: 'conc20-preview', captionId: 'conc20-caption', label: 'Concentration 20%' }},
        ];

        const pairMetricConfigs = [
            {{
                figureId: 'fig-vol1-mean-abs',
                mode: 'vol1',
                frameAImgId: 'vol1-mean-frame-a',
                frameACaptionId: 'vol1-mean-caption-a',
                frameBImgId: 'vol1-mean-frame-b',
                frameBCaptionId: 'vol1-mean-caption-b',
                thirdImgId: 'vol1-mean-third',
                thirdCaptionId: 'vol1-mean-caption-third',
                metricLabel: 'Mean |Δ|',
                thirdSource: 'diff',
            }},
            {{
                figureId: 'fig-vol1-rms',
                mode: 'vol1',
                frameAImgId: 'vol1-rms-frame-a',
                frameACaptionId: 'vol1-rms-caption-a',
                frameBImgId: 'vol1-rms-frame-b',
                frameBCaptionId: 'vol1-rms-caption-b',
                thirdImgId: 'vol1-rms-third',
                thirdCaptionId: 'vol1-rms-caption-third',
                metricLabel: 'RMS Δ',
                thirdSource: 'diff',
            }},
            {{
                figureId: 'fig-vol1-max-abs',
                mode: 'vol1',
                frameAImgId: 'vol1-max-frame-a',
                frameACaptionId: 'vol1-max-caption-a',
                frameBImgId: 'vol1-max-frame-b',
                frameBCaptionId: 'vol1-max-caption-b',
                thirdImgId: 'vol1-max-third',
                thirdCaptionId: 'vol1-max-caption-third',
                metricLabel: 'Max |Δ|',
                thirdSource: 'diff',
            }},
            {{
                figureId: 'fig-vol2-distance',
                mode: 'vol2',
                frameAImgId: 'vol2-distance-frame-a',
                frameACaptionId: 'vol2-distance-caption-a',
                frameBImgId: 'vol2-distance-frame-b',
                frameBCaptionId: 'vol2-distance-caption-b',
                thirdImgId: 'vol2-distance-third',
                thirdCaptionId: 'vol2-distance-caption-third',
                metricLabel: 'Смещение максимума',
                thirdSource: 'distance',
                distanceIndex: 3,
            }},
        ];

        function updateSingleMetricPreview(config, frameIndex) {{
            const key = String(frameIndex);
            const overlay = frameOverlays[key];
            if (!overlay) return;
            const img = document.getElementById(config.imgId);
            const caption = document.getElementById(config.captionId);
            if (!img || !caption) return;
            img.src = overlay;
            caption.textContent = `${{config.label}} · кадр ${{Number(frameIndex) + 1}}`;
        }}

        function updatePairMetricPreview(config, frameA, frameB, pairKey, data) {{
            const pairLabel = `${{Number(frameA) + 1}}-{{Number(frameB) + 1}}`;
            const frameAKey = String(frameA);
            const frameBKey = String(frameB);
            const imgA = document.getElementById(config.frameAImgId);
            const imgB = document.getElementById(config.frameBImgId);
            const capA = document.getElementById(config.frameACaptionId);
            const capB = document.getElementById(config.frameBCaptionId);
            const thirdImg = document.getElementById(config.thirdImgId);
            const thirdCap = document.getElementById(config.thirdCaptionId);

            if (imgA && frameOverlays[frameAKey]) {{
                imgA.src = frameOverlays[frameAKey];
            }}
            if (capA) {{
                capA.textContent = `Кадр ${{Number(frameA) + 1}}`;
            }}
            if (imgB && frameOverlays[frameBKey]) {{
                imgB.src = frameOverlays[frameBKey];
            }}
            if (capB) {{
                capB.textContent = `Кадр ${{Number(frameB) + 1}}`;
            }}

            let thirdOverlay = null;
            if (config.thirdSource === 'diff') {{
                thirdOverlay = pairDiffImages[pairKey];
            }} else if (config.thirdSource === 'distance') {{
                thirdOverlay = pairDistanceImages[pairKey];
            }}
            if (thirdImg && thirdOverlay) {{
                thirdImg.src = thirdOverlay;
            }}
            if (thirdCap) {{
                let captionText = config.metricLabel ? `${{config.metricLabel}} · ${{pairLabel}}` : pairLabel;
                if (config.mode === 'vol2' && typeof config.distanceIndex === 'number' && data && data.length > config.distanceIndex) {{
                    const actual = Number(data[config.distanceIndex]);
                    if (Number.isNaN(actual) === false) {{
                        captionText += ` (${{actual.toFixed(1)}} px)`;
                    }}
                }}
                thirdCap.textContent = captionText;
            }}
        }}

        function attachSingleFrameHover(configs) {{
            configs.forEach((config) => {{
                const el = document.getElementById(config.figureId);
                if (!el) return;
                el.on('plotly_hover', (event) => {{
                    const frameIndex = event.points?.[0]?.customdata;
                    if (frameIndex === undefined || frameIndex === null) return;
                    updateSingleMetricPreview(config, frameIndex);
                }});
            }});
        }}

        function attachPairHover(configs) {{
            configs.forEach((config) => {{
                const el = document.getElementById(config.figureId);
                if (!el) return;
                el.on('plotly_hover', (event) => {{
                    const data = event.points?.[0]?.customdata;
                    if (!data || data.length < 3) return;
                    const frameA = data[0];
                    const frameB = data[1];
                    const pairKey = data[2];
                    updatePairMetricPreview(config, frameA, frameB, pairKey, data);
                }});
            }});
        }}

        window.addEventListener('load', () => {{
            attachSingleFrameHover(singleMetricConfigs);
            attachPairHover(pairMetricConfigs);
        }});
    </script>
</body>
</html>
"""
    return html


def compute_figures(frame_samples: Sequence[FrameSample], pair_samples: Sequence[PairSample]) -> Dict[str, str]:
    if not frame_samples:
        raise ValueError("Нет данных для построения метрик")

    frame_numbers = [sample.index + 1 for sample in frame_samples]
    customdata = [sample.index for sample in frame_samples]

    mean_values = [float(np.clip(sample.prob_map.mean(), 0.0, 1.0)) for sample in frame_samples]
    peak_values = [float(np.clip(sample.prob_map.max(), 0.0, 1.0)) for sample in frame_samples]
    conc10_values = [float(np.clip(compute_concentration(sample.prob_map, 0.10), 0.0, 1.0)) for sample in frame_samples]
    conc20_values = [float(np.clip(compute_concentration(sample.prob_map, 0.20), 0.0, 1.0)) for sample in frame_samples]

    figures: Dict[str, go.Figure] = {}
    figures['mean'] = build_single_frame_figure(
        title="Mean — средняя вероятность по кадру",
        x=frame_numbers,
        y=mean_values,
        avg_value=float(np.mean(mean_values)),
        value_label="Норм. среднее",
        customdata=customdata,
        color="#2565d0",
    )
    figures['peak'] = build_single_frame_figure(
        title="Peak — максимум вероятности",
        x=frame_numbers,
        y=peak_values,
        avg_value=float(np.mean(peak_values)),
        value_label="Норм. максимум",
        customdata=customdata,
        color="#e4572e",
    )
    figures['conc10'] = build_single_frame_figure(
        title="Concentration 10% — доля вероятности в топ-10% пикселей",
        x=frame_numbers,
        y=conc10_values,
        avg_value=float(np.mean(conc10_values)),
        value_label="Норм. концентрация 10%",
        customdata=customdata,
        color="#3aa17e",
    )
    figures['conc20'] = build_single_frame_figure(
        title="Concentration 20% — доля вероятности в топ-20% пикселей",
        x=frame_numbers,
        y=conc20_values,
        avg_value=float(np.mean(conc20_values)),
        value_label="Норм. концентрация 20%",
        customdata=customdata,
        color="#a64ac9",
    )

    if pair_samples:
        pair_labels = [f"{pair.start_index + 1}-{pair.end_index + 1}" for pair in pair_samples]
        pair_customdata = [
            [pair.start_index, pair.end_index, f"{pair.start_index:04d}-{pair.end_index:04d}"]
            for pair in pair_samples
        ]
        mean_abs_values = [float(np.clip(pair.diff_metrics['mean_abs'], 0.0, 1.0)) for pair in pair_samples]
        rms_values = [float(np.clip(pair.diff_metrics['rms'], 0.0, 1.0)) for pair in pair_samples]
        max_abs_values = [float(np.clip(pair.diff_metrics['max_abs'], 0.0, 1.0)) for pair in pair_samples]
        distance_values = [float(np.clip(pair.maxima_distance_normalized, 0.0, 1.0)) for pair in pair_samples]

        figures['vol1_mean_abs'] = build_pair_figure(
            title="Volatility1 — Mean |Δ| (средняя амплитуда изменений)",
            x_labels=pair_labels,
            y_values=mean_abs_values,
            avg_value=float(np.mean(mean_abs_values)),
            value_label="Норм. Mean |Δ|",
            customdata=pair_customdata,
            color="#4c6ef5",
        )
        figures['vol1_rms'] = build_pair_figure(
            title="Volatility1 — RMS Δ (среднеквадратичное изменение)",
            x_labels=pair_labels,
            y_values=rms_values,
            avg_value=float(np.mean(rms_values)),
            value_label="Норм. RMS Δ",
            customdata=pair_customdata,
            color="#fa8c16",
        )
        figures['vol1_max_abs'] = build_pair_figure(
            title="Volatility1 — Max |Δ| (наибольший скачок)",
            x_labels=pair_labels,
            y_values=max_abs_values,
            avg_value=float(np.mean(max_abs_values)),
            value_label="Норм. Max |Δ|",
            customdata=pair_customdata,
            color="#ff4d4f",
        )
        figures['vol2_distance'] = build_pair_figure(
            title="Volatility2 — смещение максимума внимания",
            x_labels=pair_labels,
            y_values=distance_values,
            avg_value=float(np.mean(distance_values)),
            value_label="Норм. расстояние",
            customdata=[data + [pair.maxima_distance] for data, pair in zip(pair_customdata, pair_samples)],
            color="#13a8a8",
            hovertemplate=(
                "Кадры %{x}<br>Норм. расстояние: %{y:.4f}<br>"
                "Фактическое: %{customdata[3]:.1f} px<extra></extra>"
            ),
        )

    return {name: figure_to_html(fig, f"fig-{name.replace('_', '-')}") for name, fig in figures.items()}


def process_video(
    *,
    video_path: Path,
    seconds: float,
    frame_skip: int,
    chunk_size: int,
    model: torch.nn.Module,
    device: torch.device,
    centerbias_template: np.ndarray,
    output_dir: Path,
) -> None:
    frames_info = read_video_frames(video_path, seconds, frame_skip)
    height, width = frames_info.frame_size
    centerbias_log = build_centerbias(height, width, uniform=False, template=centerbias_template)

    predictions = list(
        iterate_deepgaze_predictions(
            frames_info.frames_bgr,
            model,
            device,
            centerbias_log,
            chunk_size=max(1, chunk_size),
        ),
    )

    frame_samples = compute_frame_samples(predictions)
    pair_samples = compute_pair_samples(frame_samples)
    figures_html = compute_figures(frame_samples, pair_samples)

    report_html = render_report_html(
        video_path=video_path,
        seconds=seconds,
        fps=frames_info.fps,
        frame_samples=frame_samples,
        pair_samples=pair_samples,
        figures=figures_html,
    )

    target_dir = output_dir / video_path.stem
    target_dir.mkdir(parents=True, exist_ok=True)
    report_path = target_dir / "report.html"
    report_path.write_text(report_html, encoding="utf-8")
    print(f"Отчёт сохранён: {report_path}")


def collect_videos(videos_dir: Path) -> List[Path]:
    video_files: List[Path] = []
    for pattern in ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm"):
        video_files.extend(sorted(videos_dir.glob(pattern)))
    return sorted({path.resolve() for path in video_files})


def main() -> int:
    parser = argparse.ArgumentParser(description="Построение HTML-отчётов с метриками DeepGaze 2e")
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS_DIR, help="Каталог с видео")
    parser.add_argument("--video", type=Path, default=None, help="Обработать только конкретный файл")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Каталог для HTML отчётов")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS, help="Длительность фрагмента")
    parser.add_argument("--frame-skip", type=int, default=1, help="Использовать каждый N-й кадр")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Размер батча для инференса")
    parser.add_argument("--device", type=str, default=None, help="Устройство (cpu/cuda)")
    args = parser.parse_args()

    if args.video is not None and not args.video.exists():
        raise FileNotFoundError(f"Видео не найдено: {args.video}")

    repo_dir = ensure_deepgaze_repo()
    centerbias_path = ensure_centerbias()
    centerbias_template = np.load(centerbias_path)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_deepgaze_model(repo_dir, "deepgaze2e", device)

    if args.video is not None:
        videos = [args.video.resolve()]
    else:
        videos = collect_videos(args.videos_dir.resolve())

    if not videos:
        print("Видео не найдены")
        return 0

    for video_path in tqdm(videos, desc="Видео", unit="видео"):
        process_video(
            video_path=video_path,
            seconds=args.seconds,
            frame_skip=max(1, args.frame_skip),
            chunk_size=max(1, args.chunk_size),
            model=model,
            device=device,
            centerbias_template=centerbias_template,
            output_dir=args.output.resolve(),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
