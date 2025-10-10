"""Generate textual and analytical reports to inspect clustering results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

DEFAULT_MANIFEST = Path("data/embeddings/cluster_manifest.json")
DEFAULT_SUMMARY_CSV = Path("data/embeddings/cluster_summary.csv")
DEFAULT_REPORT_HTML = Path("data/embeddings/cluster_report.html")
DEFAULT_DUPLICATES_CSV = Path("data/embeddings/cluster_near_duplicates.csv")
DEFAULT_OUTLIERS_CSV = Path("data/embeddings/cluster_outliers.csv")


def load_manifest(path: Path) -> Tuple[pd.DataFrame, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Cluster manifest not found: {path}")

    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Manifest does not contain an 'entries' list")

    frame = pd.DataFrame(entries)
    if "label" not in frame.columns:
        raise ValueError("Manifest entries must contain a 'label' column")

    frame = frame.reset_index(drop=True)
    frame["_row"] = frame.index
    return frame, manifest


def manifest_duplicate_groups(manifest: dict) -> pd.DataFrame:
    duplicates = manifest.get("duplicates") or []
    if not duplicates:
        return pd.DataFrame(columns=["representative_index", "members", "reason", "label", "cluster_name", "slug", "link"])

    entries = manifest.get("entries", [])
    by_index = {
        int(entry["index"]): entry
        for entry in entries
        if isinstance(entry, dict) and "index" in entry and entry["index"] == entry["index"]
    }

    rows = []
    for group in duplicates:
        rep_idx = int(group.get("representative_index", -1))
        entry = by_index.get(rep_idx, {})
        rows.append(
            {
                "representative_index": rep_idx,
                "members": ", ".join(str(int(idx)) for idx in group.get("members", []) if int(idx) != rep_idx),
                "reason": group.get("reason"),
                "label": entry.get("label"),
                "cluster_name": entry.get("cluster_name"),
                "slug": entry.get("slug"),
                "link": entry.get("link"),
            }
        )
    return pd.DataFrame(rows)


def manifest_noise_entries(manifest: dict) -> pd.DataFrame:
    noise_indices = manifest.get("noise_indices") or []
    if not noise_indices:
        return pd.DataFrame(columns=["index", "label", "cluster_name", "slug", "link"])

    entries = manifest.get("entries", [])
    by_index = {
        int(entry["index"]): entry
        for entry in entries
        if isinstance(entry, dict) and "index" in entry and entry["index"] == entry["index"]
    }

    rows = []
    for idx in noise_indices:
        entry = by_index.get(int(idx), {})
        rows.append(
            {
                "index": int(idx),
                "label": entry.get("label"),
                "cluster_name": entry.get("cluster_name"),
                "slug": entry.get("slug"),
                "link": entry.get("link"),
            }
        )
    return pd.DataFrame(rows)


def compute_summary(df: pd.DataFrame, manifest: dict, min_cluster_size: int) -> pd.DataFrame:
    summary = df.groupby("label").size().rename("count").reset_index()
    total = summary["count"].sum()
    summary["share"] = summary["count"] / total if total else 0.0
    summary["percentage"] = summary["share"] * 100

    cluster_names = manifest.get("cluster_names", {})
    summary["cluster_name"] = summary["label"].map(lambda lbl: cluster_names.get(str(int(lbl))))
    summary["is_small"] = summary["count"] <= min_cluster_size
    return summary.sort_values("label").reset_index(drop=True)


def load_embeddings_from_manifest(manifest: dict) -> np.ndarray | None:
    path_str = manifest.get("embedding_path")
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    data = np.load(path)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {data.shape}")
    return data.astype(np.float32)


def identify_small_clusters(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[summary["is_small"]].copy()


def detect_duplicate_links(df: pd.DataFrame) -> pd.DataFrame:
    if "link" not in df.columns:
        return pd.DataFrame(columns=["link", "label", "cluster_name", "slug", "name"])
    links = df.dropna(subset=["link"]).copy()
    if links.empty:
        return pd.DataFrame(columns=["link", "label", "cluster_name", "slug", "name"])
    links["link_norm"] = links["link"].str.strip().str.lower()
    dup_groups = links.groupby("link_norm").filter(lambda group: len(group) > 1)
    columns = ["link", "label", "cluster_name", "slug", "name"]
    return dup_groups[columns] if not dup_groups.empty else dup_groups[columns]


def detect_embedding_duplicates(
    df: pd.DataFrame,
    embeddings: np.ndarray | None,
    distance_threshold: float,
    max_pairs: int,
) -> pd.DataFrame:
    if embeddings is None or len(embeddings) == 0:
        return pd.DataFrame(columns=["label", "cluster_name", "slug_a", "slug_b", "distance"])

    records = []
    for label, group in df.groupby("label"):
        if len(group) < 2:
            continue
        idxs = group["_row"].to_numpy()
        cluster_embs = embeddings[idxs]
        distances = pairwise_distances(cluster_embs)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                dist = float(distances[i, j])
                if dist <= distance_threshold:
                    records.append(
                        {
                            "label": int(label),
                            "cluster_name": group.iloc[i].get("cluster_name"),
                            "slug_a": group.iloc[i].get("slug"),
                            "slug_b": group.iloc[j].get("slug"),
                            "distance": dist,
                        }
                    )
    if not records:
        return pd.DataFrame(columns=["label", "cluster_name", "slug_a", "slug_b", "distance"])
    dup_df = pd.DataFrame(records)
    dup_df = dup_df.nsmallest(max_pairs, "distance")
    return dup_df


def detect_outliers(
    df: pd.DataFrame,
    embeddings: np.ndarray | None,
    z_threshold: float,
    max_per_cluster: int,
) -> pd.DataFrame:
    if embeddings is None or len(embeddings) == 0:
        return pd.DataFrame(columns=["label", "cluster_name", "slug", "distance", "z_score"])

    rows = []
    for label, group in df.groupby("label"):
        if len(group) < 3:
            continue
        idxs = group["_row"].to_numpy()
        cluster_embs = embeddings[idxs]
        centroid = cluster_embs.mean(axis=0)
        dists = np.linalg.norm(cluster_embs - centroid, axis=1)
        mean = float(dists.mean())
        std = float(dists.std())
        if std <= 1e-6:
            continue
        z_scores = (dists - mean) / std
        order = np.argsort(-z_scores)
        flagged = 0
        for idx in order:
            z_val = float(z_scores[idx])
            if z_val < z_threshold:
                break
            rows.append(
                {
                    "label": int(label),
                    "cluster_name": group.iloc[idx].get("cluster_name"),
                    "slug": group.iloc[idx].get("slug"),
                    "distance": float(dists[idx]),
                    "z_score": z_val,
                }
            )
            flagged += 1
            if flagged >= max_per_cluster:
                break
    if not rows:
        return pd.DataFrame(columns=["label", "cluster_name", "slug", "distance", "z_score"])
    out_df = pd.DataFrame(rows)
    return out_df.sort_values("z_score", ascending=False).reset_index(drop=True)


def render_section(title: str, body: str, description: str | None = None) -> str:
    desc_html = f"<p class='section-description'>{description}</p>" if description else ""
    return f"<section><h2>{title}</h2>{desc_html}{body}</section>"


def dataframe_to_html(df: pd.DataFrame, classes: Iterable[str] | None = None) -> str:
    if df.empty:
        return "<p><em>No entries.</em></p>"
    display = df.copy()
    return display.to_html(index=False, escape=False, border=0, classes=list(classes or ()))


def build_report(
    summary: pd.DataFrame,
    df: pd.DataFrame,
    manifest: dict,
    small_clusters: pd.DataFrame,
    duplicate_links: pd.DataFrame,
    embedding_duplicates: pd.DataFrame,
    outliers: pd.DataFrame,
) -> str:
    parts = [
        "<html><head><meta charset='utf-8'>",
        "<style>body{font-family:Arial,sans-serif;margin:2rem;}",
        "table{border-collapse:collapse;margin-bottom:2rem;}",
        "th,td{border:1px solid #ddd;padding:0.4rem 0.6rem;}",
        "th{background:#f0f0f0;}",
        ".intro{max-width:65em;line-height:1.5;}",
        ".section-description{color:#555;max-width:65em;line-height:1.45;margin:0.3rem 0 0.9rem;}",
        "h1,h2{margin-top:1.2rem;}",
        "section{margin-bottom:2.4rem;}",
        "</style></head><body>",
    ]
    parts.append("<h1>Cluster report</h1>")
    parts.append(
        "<p class='intro'>Этот отчёт подробно описывает результаты кластеризации рекламных роликов. "
        "Ниже можно увидеть использованный алгоритм, параметры подбора k, сжатую статистику по каждому кластеру, "
        "а также подозрительные случаи: дубликаты, выбросы и элементы, помеченные как шум.</p>"
    )
    parts.append(
        "<p><strong>Алгоритм:</strong> {}</p>".format(manifest.get("algorithm", "unknown"))
    )
    parts.append("<p><strong>Количество кластеров (k):</strong> {}</p>".format(manifest.get("n_clusters", "?")))
    if "evaluation" in manifest and manifest["evaluation"]:
        metrics_rows = [
            "k={k} silhouette={silhouette!s} davies_bouldin={davies_bouldin!s} inertia={inertia!s}".format(**row)
            for row in manifest["evaluation"]
        ]
        parts.append("<p><strong>Evaluated k:</strong><br>{}</p>".format("<br>".join(metrics_rows)))
    if "inertia" in manifest:
        parts.append("<p><strong>Inertia:</strong> {:.4f}</p>".format(manifest["inertia"]))
    if manifest.get("metrics_plot"):
        parts.append(
            "<p><a href='{url}' target='_blank'>Interactive metrics plot</a></p>".format(url=manifest["metrics_plot"])
        )

    dedup_df = manifest_duplicate_groups(manifest)
    if not dedup_df.empty:
        parts.append(
            render_section(
                "Схлопнутые дубликаты",
                dataframe_to_html(dedup_df),
                "Группы роликов, которые были объединены в процессе подготовки данных. "
                "В таблице указан представитель группы, остальные элементы приведены в столбце members.",
            )
        )

    noise_df = manifest_noise_entries(manifest)
    if not noise_df.empty:
        parts.append(
            render_section(
                "Объекты, помеченные как шум",
                dataframe_to_html(noise_df),
                "Эти ролики были определены алгоритмом DBSCAN как выбросы и не участвовали в построении центроидов кластеров.",
            )
        )

    summary_display = summary.copy()
    summary_display["share"] = summary_display["share"].map(lambda x: f"{x:.4f}")
    summary_display["percentage"] = summary_display["percentage"].map(lambda x: f"{x:.2f}%")
    column_order = ["label", "cluster_name", "count", "percentage", "share", "is_small"]
    summary_display = summary_display[[col for col in column_order if col in summary_display.columns]]
    parts.append(
        render_section(
            "Сводная статистика кластеров",
            dataframe_to_html(summary_display),
            "Основные метрики по каждому кластеру: размер, доля выборки и признак того, считается ли кластер малочисленным.",
        )
    )

    if not small_clusters.empty:
        parts.append(
            render_section(
                "Малочисленные кластеры",
                dataframe_to_html(small_clusters[[c for c in column_order if c in small_clusters.columns]]),
                "Кластеры, чей размер не превышает заданного порога. Их стоит проверить на предмет шума или редких сценариев.",
            )
        )

    if not duplicate_links.empty:
        parts.append(
            render_section(
                "Совпадающие ссылки",
                dataframe_to_html(duplicate_links),
                "Ролики с идентичными ссылками YouTube — вероятные точные дубликаты в исходном датасете.",
            )
        )

    if not embedding_duplicates.empty:
        parts.append(
            render_section(
                "Близкие по эмбеддингам ролики",
                dataframe_to_html(embedding_duplicates),
                "Пары роликов внутри кластера, у которых косинусное расстояние между эмбеддингами ниже выбранного порога.",
            )
        )

    if not outliers.empty:
        parts.append(
            render_section(
                "Потенциальные выбросы",
                dataframe_to_html(outliers),
                "Участники кластеров, которые заметно удалены от центров (по z-оценке расстояния). Рекомендуется проверить содержимое роликов.",
            )
        )

    for label, group in sorted(df.groupby("label"), key=lambda item: item[0]):
        display_group = group.drop(columns=["_row"], errors="ignore")
        if "index" in display_group.columns:
            group_sorted = display_group.sort_values(by="index")
        else:
            group_sorted = display_group
        cluster_name = manifest.get("cluster_names", {}).get(str(int(label)))
        description = "Полный список роликов, отнесённых к кластеру. Наведите на ссылку или slug, чтобы перейти к исходным данным."
        if cluster_name:
            description = (
                f"Кластер «{cluster_name}». {description}"
            )
        parts.append(
            render_section(
                f"Кластер {label}",
                dataframe_to_html(group_sorted),
                description,
            )
        )

    parts.append("</body></html>")
    return "".join(parts)


def ensure_parent(path: Path) -> None:
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)


def generate_report(
    manifest_path: Path,
    summary_csv: Path,
    report_html: Path,
    duplicates_csv: Path,
    outliers_csv: Path,
    min_cluster_size: int,
    distance_threshold: float,
    z_threshold: float,
    max_duplicate_pairs: int,
    max_outliers_per_cluster: int,
) -> None:
    df, manifest = load_manifest(manifest_path)
    embeddings = load_embeddings_from_manifest(manifest)

    summary = compute_summary(df, manifest, min_cluster_size)
    small_clusters = identify_small_clusters(summary)
    duplicate_links = detect_duplicate_links(df)
    embedding_duplicates = detect_embedding_duplicates(
        df, embeddings, distance_threshold=distance_threshold, max_pairs=max_duplicate_pairs
    )
    outliers = detect_outliers(
        df,
        embeddings,
        z_threshold=z_threshold,
        max_per_cluster=max_outliers_per_cluster,
    )

    ensure_parent(summary_csv)
    summary.to_csv(summary_csv, index=False)

    duplicate_exports = []
    if not duplicate_links.empty:
        link_export = duplicate_links.copy()
        link_export["duplicate_type"] = "link"
        duplicate_exports.append(link_export)
    if not embedding_duplicates.empty:
        emb_export = embedding_duplicates.copy()
        emb_export["duplicate_type"] = "embedding"
        duplicate_exports.append(emb_export)
    if duplicate_exports:
        ensure_parent(duplicates_csv)
        pd.concat(duplicate_exports, ignore_index=True).to_csv(duplicates_csv, index=False)

    if not outliers.empty:
        ensure_parent(outliers_csv)
        outliers.to_csv(outliers_csv, index=False)

    ensure_parent(report_html)
    report_html.write_text(
        build_report(summary, df, manifest, small_clusters, duplicate_links, embedding_duplicates, outliers),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reports for clustering results")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Path to cluster_manifest.json")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV, help="Where to write cluster summary CSV")
    parser.add_argument("--report-html", type=Path, default=DEFAULT_REPORT_HTML, help="Where to write cluster report HTML")
    parser.add_argument(
        "--duplicates-csv", type=Path, default=DEFAULT_DUPLICATES_CSV, help="Where to export duplicate suggestions"
    )
    parser.add_argument(
        "--outliers-csv", type=Path, default=DEFAULT_OUTLIERS_CSV, help="Where to export outlier suggestions"
    )
    parser.add_argument("--min-cluster-size", type=int, default=3, help="Minimum cluster size before flagging as small")
    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=0.18,
        help="Cosine/L2 distance threshold for considering embeddings near-duplicates",
    )
    parser.add_argument(
        "--outlier-z", type=float, default=2.5, help="Z-score above which members are flagged as outliers"
    )
    parser.add_argument(
        "--max-duplicate-pairs",
        type=int,
        default=100,
        help="Maximum number of near-duplicate pairs to keep in the report",
    )
    parser.add_argument(
        "--max-outliers-per-cluster",
        type=int,
        default=5,
        help="Maximum number of outliers to list per cluster",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_report(
        manifest_path=args.manifest,
        summary_csv=args.summary_csv,
        report_html=args.report_html,
        duplicates_csv=args.duplicates_csv,
        outliers_csv=args.outliers_csv,
        min_cluster_size=args.min_cluster_size,
        distance_threshold=args.duplicate_threshold,
        z_threshold=args.outlier_z,
        max_duplicate_pairs=args.max_duplicate_pairs,
        max_outliers_per_cluster=args.max_outliers_per_cluster,
    )


if __name__ == "__main__":
    main()
