"""Build interactive histograms for shot pacing metrics across videos."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.io import to_html
import numpy as np


METRIC_COLUMNS: list[str] = [
    "scene_count",
    "cut_count",
    "cuts_per_second",
    "median_shot_duration",
    "iqr_shot_duration",
    "cuts_in_first_5s",
    "cuts_in_first_5s_per_second",
]

METRIC_LABELS = {
    "scene_count": "Количество сцен",
    "cut_count": "Количество склеек",
    "cuts_per_second": "Склейки в секунду",
    "median_shot_duration": "Медианная длительность шота (с)",
    "iqr_shot_duration": "IQR длительности шота (с)",
    "cuts_in_first_5s": "Склейки в первые 5 с",
    "cuts_in_first_5s_per_second": "Склейки/с в первые 5 с",
}

METRIC_DESCRIPTIONS = {
    "scene_count": (
        "Сцена в терминологии PySceneDetect — последовательность шотов,"
        " объединённых единой ситуацией. Метрика показывает, сколько таких"
        " сцен выделил детектор во всём ролике."
    ),
    "cut_count": (
        "Склейка (cut) — переход между двумя шотами. Значение показывает,"
        " сколько монтажных переходов нашёл алгоритм."
    ),
    "cuts_per_second": (
        "Относительная интенсивность монтажа: число склеек, делённое на"
        " длительность видео. Помогает сравнивать ролики разной продолжительности."
    ),
    "median_shot_duration": (
        "Шот — отрезок между двумя склейками. Медианная длительность отражает"
        " типичную длину шота и устойчива к выбросам."
    ),
    "iqr_shot_duration": (
        "IQR (interquartile range) — межквартильный размах длительности шотов."
        " Чем он меньше, тем более равномерны по длине шоты."
    ),
    "cuts_in_first_5s": (
        "Количество склеек, которое произошло в первые пять секунд ролика."
        " Позволяет оценить агрессивность вступления."
    ),
    "cuts_in_first_5s_per_second": (
        "Нормированная версия предыдущей метрики: склейки в первые пять секунд"
        " в пересчёте на секунду видео."
    ),
}

METRIC_PEAK_ORDER = {
    "scene_count": "desc",
    "cut_count": "desc",
    "cuts_per_second": "desc",
    "median_shot_duration": "asc",
    "iqr_shot_duration": "asc",
    "cuts_in_first_5s": "desc",
    "cuts_in_first_5s_per_second": "desc",
}


def _ensure_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(
            "Отсутствуют столбцы с метриками: " + ", ".join(sorted(missing))
        )


def _load_metrics(metrics_path: Path, status_column: str) -> pd.DataFrame:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Не найден файл с метриками: {metrics_path}")

    frame = pd.read_csv(metrics_path)
    if status_column in frame.columns:
        frame = frame.loc[frame[status_column] == "ok"].copy()

    _ensure_columns(frame, ["index", *METRIC_COLUMNS])

    frame["index"] = pd.to_numeric(frame["index"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["index"]).copy()
    frame["index"] = frame["index"].astype(int)

    for column in METRIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def _load_dataset(dataset_path: Path, index_column: str, topic_column: str) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Не найден датасет: {dataset_path}")

    dataset = pd.read_excel(dataset_path)
    if index_column not in dataset.columns:
        dataset = dataset.reset_index().rename(columns={"index": index_column})

    dataset[index_column] = pd.to_numeric(dataset[index_column], errors="coerce")
    dataset = dataset.dropna(subset=[index_column])
    dataset[index_column] = dataset[index_column].astype(int)

    if topic_column not in dataset.columns:
        raise ValueError(
            f"В датасете отсутствует столбец '{topic_column}' с топиками"
        )

    return dataset[[index_column, topic_column]].copy()


def _build_hist_grid(
    frame: pd.DataFrame,
    metrics: list[str],
    title: str,
    nbins: int = 30,
    template: str = "plotly_white",
) -> go.Figure:
    total_metrics = len(metrics)
    cols = 2 if total_metrics > 1 else 1
    rows = math.ceil(total_metrics / cols)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[METRIC_LABELS.get(m, m) for m in metrics],
    )

    for idx, metric in enumerate(metrics):
        row = idx // cols + 1
        col = idx % cols + 1
        values = frame[metric].dropna()
        if values.empty:
            x_pos = (col - 0.5) / cols
            y_pos = 1 - (row - 0.5) / rows
            fig.add_annotation(
                text="Нет данных",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=x_pos,
                y=y_pos,
                font={"color": "#666"},
            )
            continue

        fig.add_trace(
            go.Histogram(
                x=values,
                nbinsx=nbins,
                marker_color="#1f77b4",
                opacity=0.85,
                hovertemplate="%{x:.3f}<extra></extra>",
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=title,
        showlegend=False,
        template=template,
        height=max(360 * rows, 320),
        bargap=0.05,
        margin=dict(t=80, l=40, r=20, b=40),
    )

    return fig


def _build_single_histogram(
    values: pd.Series,
    metric: str,
    title: str,
    nbins: int = 30,
    template: str = "plotly_white",
) -> go.Figure:
    fig = go.Figure()
    clean_values = values.dropna()

    if clean_values.empty:
        fig.add_annotation(
            text="Нет данных",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            font={"color": "#666"},
        )
    else:
        fig.add_trace(
            go.Histogram(
                x=clean_values,
                nbinsx=nbins,
                marker_color="#1f77b4",
                opacity=0.85,
                hovertemplate="%{x:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        showlegend=False,
        template=template,
        height=320,
        bargap=0.05,
        margin=dict(t=60, l=40, r=20, b=40),
    )

    return fig


def _compute_topic_peaks(
    frame: pd.DataFrame,
    metric: str,
    topic_column: str,
    nbins: int,
) -> list[dict[str, object]]:
    values = frame[metric].dropna()
    if values.empty:
        return []

    bin_edges = np.histogram_bin_edges(values, bins=nbins)
    peaks: list[dict[str, object]] = []

    for topic, subset in frame.groupby(topic_column):
        topic_values = subset[metric].dropna()
        if topic_values.empty:
            continue

        counts, edges = np.histogram(topic_values, bins=bin_edges)
        if not counts.size:
            continue

        peak_idx = int(counts.argmax())
        peak_count = int(counts[peak_idx])
        left = float(edges[peak_idx])
        right = float(edges[peak_idx + 1])
        midpoint = (left + right) / 2
        peaks.append(
            {
                "topic": topic,
                "peak_value": midpoint,
                "peak_count": peak_count,
                "bin_range": (left, right),
            }
        )

    order = METRIC_PEAK_ORDER.get(metric, "desc")
    reverse = order == "desc"
    peaks.sort(key=lambda item: item["peak_value"], reverse=reverse)
    return peaks


def _render_peak_list(peaks: list[dict[str, object]], metric: str) -> str:
    if not peaks:
        return "<p class=\"metric-peaks-empty\">Нет данных для расчёта.</p>"

    unit = ""
    if "duration" in metric:
        unit = " с"
    elif metric in {"cuts_per_second", "cuts_in_first_5s_per_second"}:
        unit = " скл./с"

    items = []
    for item in peaks:
        left, right = item["bin_range"]
        formatted_range = f"{left:.2f}–{right:.2f}{unit}" if unit else f"{left:.2f}–{right:.2f}"
        items.append(
            "<li>"
            f"<strong>{item['topic']}</strong>: пик {formatted_range},"
            f" {int(item['peak_count'])} видео"
            "</li>"
        )

    return "<ol class=\"metric-peaks\">" + "".join(items) + "</ol>"


def _build_metric_blocks(
    frame: pd.DataFrame,
    metrics: list[str],
    topic_column: str,
    nbins: int,
) -> str:
    blocks: list[str] = []
    include_plotly = True

    for metric in metrics:
        title = METRIC_LABELS.get(metric, metric)
        fig = _build_single_histogram(frame[metric], metric, title, nbins=nbins)
        html = to_html(
            fig,
            include_plotlyjs="cdn" if include_plotly else False,
            full_html=False,
        )
        include_plotly = False

        description = METRIC_DESCRIPTIONS.get(metric, "")
        peaks = _compute_topic_peaks(frame, metric, topic_column, nbins)
        peak_list = _render_peak_list(peaks, metric)

        block_html = (
            "<section class=\"metric-block\">"
            "  <div class=\"metric-chart\">"
            f"    {html}"
            "  </div>"
            "  <div class=\"metric-details\">"
            f"    <h3>{title}</h3>"
            f"    <p>{description}</p>"
            "    <h4>Топики по пику распределения</h4>"
            f"    {peak_list}"
            "  </div>"
            "</section>"
        )
        blocks.append(block_html)

    return "\n".join(blocks)


def _make_summary_table(frame: pd.DataFrame, topic_column: str) -> str:
    counts = (
        frame.groupby(topic_column)
        .size()
        .reset_index(name="videos")
        .sort_values("videos", ascending=False)
    )
    counts["videos"] = counts["videos"].astype(int)

    total = counts["videos"].sum()
    counts.loc[len(counts)] = {topic_column: "Всего", "videos": int(total)}

    html_table = counts.rename(
        columns={topic_column: "Топик", "videos": "Количество видео"}
    ).to_html(classes="data-table", index=False, border=0)

    return html_table


def build_report(
    metrics_path: Path,
    dataset_path: Path,
    output_path: Path,
    *,
    index_column: str = "index",
    topic_column: str = "topic",
    status_column: str = "status",
    nbins: int = 30,
) -> Path:
    metrics_frame = _load_metrics(metrics_path, status_column=status_column)
    dataset_frame = _load_dataset(dataset_path, index_column=index_column, topic_column=topic_column)

    merged = metrics_frame.merge(dataset_frame, on=index_column, how="left")
    merged[topic_column] = merged[topic_column].fillna("Не указано")

    overall_metrics_html = _build_metric_blocks(
        merged,
        METRIC_COLUMNS,
        topic_column,
        nbins,
    )

    topic_sections: list[tuple[str, str]] = []
    for value, subset in merged.groupby(topic_column):
        fig = _build_hist_grid(subset, METRIC_COLUMNS, f"Топик: {value}", nbins=nbins)
        topic_sections.append((value, to_html(fig, include_plotlyjs=False, full_html=False)))

    summary_table = _make_summary_table(merged, topic_column)

    sections = [
        "<!DOCTYPE html>",
        "<html lang=\"ru\">",
        "<head>",
        "  <meta charset=\"utf-8\">",
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "  <title>Аналитика монтажных метрик</title>",
        "  <style>",
        "    body { font-family: 'Inter', 'Segoe UI', Tahoma, sans-serif; margin: 0; padding: 0; background: #f7f7f9; color: #222; }",
        "    main { max-width: 1200px; margin: 0 auto; padding: 32px 24px 48px; }",
        "    h1 { font-size: 28px; margin-bottom: 8px; }",
        "    h2 { margin-top: 40px; font-size: 22px; }",
        "    h3 { font-size: 18px; margin-top: 24px; }",
        "    h4 { font-size: 16px; margin-top: 16px; }",
        "    p.description { max-width: 880px; line-height: 1.6; color: #444; }",
        "    ul.glossary { margin-top: 16px; padding-left: 20px; color: #444; line-height: 1.6; }",
        "    ul.glossary li { margin-bottom: 8px; }",
        "    section { background: #fff; border-radius: 12px; padding: 24px; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06); margin-top: 24px; }",
        "    .data-table { border-collapse: collapse; width: 100%; margin-top: 16px; font-size: 14px; }",
        "    .data-table th, .data-table td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #e5e7eb; }",
        "    details { margin-top: 16px; }",
        "    summary { cursor: pointer; font-weight: 600; font-size: 16px; }",
        "    .metric-block { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr); gap: 24px; margin-top: 24px; background: #fff; border-radius: 12px; padding: 24px; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06); }",
        "    .metric-block:first-of-type { margin-top: 16px; }",
        "    .metric-details { display: flex; flex-direction: column; }",
        "    .metric-details p { margin-top: 8px; line-height: 1.6; color: #444; }",
        "    .metric-peaks { margin-top: 12px; padding-left: 20px; color: #444; }",
        "    .metric-peaks li { margin-bottom: 6px; }",
        "    .metric-peaks-empty { margin-top: 12px; color: #666; }",
        "    @media (max-width: 960px) { .metric-block { grid-template-columns: 1fr; } }",
        "  </style>",
        "</head>",
        "<body>",
        "  <main>",
        "    <h1>Профили монтажных метрик</h1>",
        "    <p class=\"description\">Интерактивный отчёт по семи метрикам темпа монтажа. Он основан на детекторах PySceneDetect и помогает понять, как устроен монтаж на уровне сцен и шотов.</p>",
        "    <ul class=\"glossary\">",
        "      <li><strong>Сцена</strong> — группа последовательных шотов с общей ситуацией или локацией.</li>",
        "      <li><strong>Склейка</strong> — резкий переход между шотами; определяет границы шотов.</li>",
        "      <li><strong>Шот</strong> — непрерывный фрагмент между двумя склейками.</li>",
        "      <li><strong>IQR</strong> — межквартильный размах, диапазон между 25 и 75 процентилями.</li>",
        "    </ul>",
        "    <section>",
        "      <h2>Сводка</h2>",
        summary_table,
        "    </section>",
        "    <section style=\"background:transparent; box-shadow:none; padding:0;\">",
        "      <h2>Все видео</h2>",
        overall_metrics_html,
        "    </section>",
    ]

    if topic_sections:
        sections.append("    <section>")
        sections.append("      <h2>По топикам</h2>")
        for value, html in topic_sections:
            sections.append("      <details open>")
            sections.append(f"        <summary>{value}</summary>")
            sections.append("        <div style=\"margin-top:16px;\">")
            sections.append(html)
            sections.append("        </div>")
            sections.append("      </details>")
        sections.append("    </section>")

    sections.extend([
        "  </main>",
        "</body>",
        "</html>",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections), encoding="utf-8")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Построить HTML-отчёт с гистограммами монтажных метрик"
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("data/analytics/shot_metrics.csv"),
        help="CSV с результатами compute_shot_metrics.py",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("VideoDataset.xlsx"),
        help="Excel с метаданными и претопиками",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/analytics/shot_metrics_report.html"),
        help="Путь к HTML-отчёту",
    )
    parser.add_argument(
        "--index-column",
        default="index",
        help="Название столбца с индексом видео",
    )
    parser.add_argument(
        "--topic-column",
        default="topic",
        help="Название столбца с претопиком",
    )
    parser.add_argument(
        "--status-column",
        default="status",
        help="Столбец в CSV, по которому фильтруются успешные расчёты",
    )
    parser.add_argument(
        "--nbins",
        type=int,
        default=30,
        help="Количество бинов в гистограммах",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = build_report(
        metrics_path=args.metrics,
        dataset_path=args.dataset,
        output_path=args.output,
        index_column=args.index_column,
        topic_column=args.topic_column,
        status_column=args.status_column,
        nbins=args.nbins,
    )
    print(f"HTML-отчёт сохранён в {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
