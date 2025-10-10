"""Визуализация кластеризации видеороликов на плоскости."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

DEFAULT_REDUCED_PATH = Path("data/embeddings/reduced_embeddings.npy")
DEFAULT_METADATA_CSV = Path("data/embeddings/embeddings_metadata.csv")
DEFAULT_CLUSTER_JSON = Path("data/embeddings/cluster_manifest.json")
DEFAULT_OUTPUT_HTML = Path("data/embeddings/cluster_visualization.html")


class ProjectionError(RuntimeError):
    """Raised when projection cannot be computed."""


def load_reduced_embeddings(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Reduced embeddings file not found: {path}")
    data = np.load(path)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D embeddings matrix, got shape {data.shape}")
    return data.astype(np.float32)


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")
    df = pd.read_csv(path)
    if "index" not in df.columns:
        raise ValueError("Metadata must contain 'index' column")
    return df


def load_clusters(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Cluster manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def compute_projection(
    embeddings: np.ndarray,
    method: str,
    random_state: int | None,
    perplexity: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if len(embeddings) == 0:
        return np.zeros((0, 2), dtype=np.float32), {"method": method, "params": {}}

    if embeddings.shape[1] < 2:
        pad = np.zeros((embeddings.shape[0], 2), dtype=np.float32)
        pad[:, : embeddings.shape[1]] = embeddings
        return pad, {"method": method, "params": {"note": "input_dim<2, zero-padded"}}

    method = method.lower()
    if method == "pca":
        projector = PCA(n_components=2, svd_solver="full", random_state=random_state)
        coords = projector.fit_transform(embeddings)
        meta = {
            "method": "pca",
            "params": {
                "explained_variance_ratio": projector.explained_variance_ratio_.astype(float).tolist(),
            },
        }
        return coords.astype(np.float32), meta
    if method == "tsne":
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, max(5.0, len(embeddings) / 3)),
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
        coords = tsne.fit_transform(embeddings)
        return coords.astype(np.float32), {"method": "tsne", "params": {"perplexity": float(tsne.perplexity)}}

    raise ProjectionError(f"Unknown projection method: {method}")


def prepare_plot_dataframe(
    metadata: pd.DataFrame,
    cluster_manifest: Dict[str, object],
) -> pd.DataFrame:
    if "entries" not in cluster_manifest:
        raise ValueError("Cluster manifest does not contain entries")
    entries = cluster_manifest["entries"]
    label_map = {int(entry["index"]): int(entry["label"]) for entry in entries}
    df = metadata.copy()
    df["cluster"] = df["index"].map(label_map)
    if df["cluster"].isna().any():
        missing = df[df["cluster"].isna()]["index"].tolist()
        raise ValueError(f"Cluster labels missing for indices: {missing[:10]}")
    df["cluster"] = df["cluster"].astype(int)
    return df


def make_visualization(
    embeddings_path: Path,
    metadata_path: Path,
    cluster_path: Path,
    output_html: Path,
    method: str,
    random_state: int | None,
    perplexity: float,
) -> Dict[str, object]:
    embeddings = load_reduced_embeddings(embeddings_path)
    metadata = load_metadata(metadata_path)
    cluster_manifest = load_clusters(cluster_path)

    df = prepare_plot_dataframe(metadata, cluster_manifest)
    if len(df) != len(embeddings):
        raise ValueError(
            "Количество записей метаданных не совпадает с числом эмбеддингов. "
            f"Metadata: {len(df)}, embeddings: {len(embeddings)}"
        )

    coords, projection_meta = compute_projection(embeddings, method=method, random_state=random_state, perplexity=perplexity)
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    df["cluster_label"] = df["cluster"].astype(str)

    hover_columns = [col for col in df.columns if col not in {"x", "y", "cluster", "cluster_label"}]
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster_label",
        hover_data=hover_columns,
        title="Кластеризация рекламных роликов",
        labels={"cluster_label": "Кластер"},
    )
    fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=0)))
    fig.update_layout(legend_title_text="Кластер", legend=dict(itemsizing="constant"))

    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_html, include_plotlyjs="cdn")

    return {
        "projection": projection_meta,
        "output_html": str(output_html),
        "n_items": len(df),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize clustered videos in 2D")
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_REDUCED_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTER_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_HTML)
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0, help="Perplexity for t-SNE projection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_visualization(
        embeddings_path=args.embeddings,
        metadata_path=args.metadata,
        cluster_path=args.clusters,
        output_html=args.output,
        method=args.method,
        random_state=args.random_state,
        perplexity=args.perplexity,
    )


if __name__ == "__main__":
    main()
