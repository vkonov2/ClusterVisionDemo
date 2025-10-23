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


def _load_dataset(dataset_path: Path, index_column: str, pretopic_column: str) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Не найден датасет: {dataset_path}")

    dataset = pd.read_excel(dataset_path)
    if index_column not in dataset.columns:
        dataset = dataset.reset_index().rename(columns={"index": index_column})

    dataset[index_column] = pd.to_numeric(dataset[index_column], errors="coerce")
    dataset = dataset.dropna(subset=[index_column])
    dataset[index_column] = dataset[index_column].astype(int)

    if pretopic_column not in dataset.columns:
        raise ValueError(
            f"В датасете отсутствует столбец '{pretopic_column}' с претопиками"
        )

    return dataset[[index_column, pretopic_column]].copy()


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


def _make_summary_table(frame: pd.DataFrame, pretopic_column: str) -> str:
    counts = (
        frame.groupby(pretopic_column)
        .size()
        .reset_index(name="videos")
        .sort_values("videos", ascending=False)
    )
    counts["videos"] = counts["videos"].astype(int)

    total = counts["videos"].sum()
    counts.loc[len(counts)] = {pretopic_column: "Всего", "videos": int(total)}

    html_table = counts.rename(
        columns={pretopic_column: "Претопик", "videos": "Количество видео"}
    ).to_html(classes="data-table", index=False, border=0)

    return html_table


def build_report(
    metrics_path: Path,
    dataset_path: Path,
    output_path: Path,
    *,
    index_column: str = "index",
    pretopic_column: str = "pretopic",
    status_column: str = "status",
    nbins: int = 30,
) -> Path:
    metrics_frame = _load_metrics(metrics_path, status_column=status_column)
    dataset_frame = _load_dataset(dataset_path, index_column=index_column, pretopic_column=pretopic_column)

    merged = metrics_frame.merge(dataset_frame, on=index_column, how="left")
    merged[pretopic_column] = merged[pretopic_column].fillna("Не указано")

    overall_fig = _build_hist_grid(merged, METRIC_COLUMNS, "Распределения по всем видео", nbins=nbins)

    pretopic_sections: list[tuple[str, str]] = []
    for value, subset in merged.groupby(pretopic_column):
        fig = _build_hist_grid(subset, METRIC_COLUMNS, f"Претопик: {value}", nbins=nbins)
        pretopic_sections.append((value, to_html(fig, include_plotlyjs=False, full_html=False)))

    overall_html = to_html(overall_fig, include_plotlyjs="cdn", full_html=False)
    summary_table = _make_summary_table(merged, pretopic_column)

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
        "    h3 { font-size: 18px; margin-top: 32px; }",
        "    p.description { max-width: 880px; line-height: 1.6; color: #444; }",
        "    section { background: #fff; border-radius: 12px; padding: 24px; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06); margin-top: 24px; }",
        "    .data-table { border-collapse: collapse; width: 100%; margin-top: 16px; font-size: 14px; }",
        "    .data-table th, .data-table td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #e5e7eb; }",
        "    details { margin-top: 16px; }",
        "    summary { cursor: pointer; font-weight: 600; font-size: 16px; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <main>",
        "    <h1>Профили монтажных метрик</h1>",
        "    <p class=\"description\">Интерактивный отчёт по семи метрикам темпа монтажа. Вверху показаны распределения для всей выборки, ниже — отдельные гистограммы по каждому претопику.</p>",
        "    <section>",
        "      <h2>Сводка</h2>",
        summary_table,
        "    </section>",
        "    <section>",
        "      <h2>Все видео</h2>",
        overall_html,
        "    </section>",
    ]

    if pretopic_sections:
        sections.append("    <section>")
        sections.append("      <h2>По претопикам</h2>")
        for value, html in pretopic_sections:
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
        "--pretopic-column",
        default="pretopic",
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
        pretopic_column=args.pretopic_column,
        status_column=args.status_column,
        nbins=args.nbins,
    )
    print(f"HTML-отчёт сохранён в {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
