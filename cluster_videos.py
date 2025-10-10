"""Кластеризация видеороликов на основе редуцированных эмбеддингов."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DEFAULT_REDUCED_PATH = Path("data/embeddings/reduced_embeddings.npy")
DEFAULT_METADATA_CSV = Path("data/embeddings/embeddings_metadata.csv")
DEFAULT_OUTPUT_JSON = Path("data/embeddings/cluster_manifest.json")
DEFAULT_OUTPUT_LABELS = Path("data/embeddings/cluster_labels.npy")


@dataclass
class ClusterResult:
    labels: np.ndarray
    model_info: Dict[str, object]


def load_embeddings(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Reduced embeddings file not found: {path}")
    embeddings = np.load(path)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings matrix, got shape {embeddings.shape}")
    return embeddings.astype(np.float32)


def load_metadata(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")
    df = pd.read_csv(path)
    if "index" not in df.columns:
        raise ValueError("Metadata must contain 'index' column with dataset indices")
    return df.to_dict(orient="records")


def run_kmeans(embeddings: np.ndarray, n_clusters: int, random_state: int | None) -> ClusterResult:
    if len(embeddings) == 0:
        return ClusterResult(labels=np.zeros((0,), dtype=np.int32), model_info={"algorithm": "kmeans", "n_clusters": n_clusters, "random_state": random_state, "inertia": 0.0})

    if n_clusters <= 0:
        raise ValueError("n_clusters must be a positive integer")
    if n_clusters > len(embeddings):
        raise ValueError("n_clusters cannot exceed number of available embeddings")

    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = model.fit_predict(embeddings)
    info = {
        "algorithm": "kmeans",
        "n_clusters": int(n_clusters),
        "random_state": random_state,
        "inertia": float(model.inertia_),
        "cluster_centers": model.cluster_centers_.astype(np.float32).tolist(),
    }
    return ClusterResult(labels=labels.astype(np.int32), model_info=info)


def build_manifest(
    metadata: Sequence[Dict[str, object]],
    labels: Sequence[int],
    model_info: Dict[str, object],
    embedding_path: Path,
) -> Dict[str, object]:
    if len(metadata) != len(labels):
        raise ValueError("Metadata and labels length mismatch")

    counts = Counter(int(label) for label in labels)
    entries: List[Dict[str, object]] = []
    for meta, label in zip(metadata, labels):
        entry = {**meta}
        entry["label"] = int(label)
        entries.append(entry)

    manifest = {
        "embedding_path": str(embedding_path),
        "n_items": len(labels),
        "label_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "entries": entries,
    }
    manifest.update(model_info)
    return manifest


def ensure_parent(path: Path) -> None:
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)


def cluster_embeddings(
    embeddings_path: Path,
    metadata_path: Path,
    output_json: Path,
    output_labels: Path,
    n_clusters: int,
    random_state: int | None,
) -> Dict[str, object]:
    embeddings = load_embeddings(embeddings_path)
    metadata = load_metadata(metadata_path)

    if len(metadata) != len(embeddings):
        raise ValueError(
            "Количество строк в metadata CSV не совпадает с числом эмбеддингов. "
            f"Metadata: {len(metadata)}, embeddings: {len(embeddings)}"
        )

    result = run_kmeans(embeddings, n_clusters=n_clusters, random_state=random_state)

    ensure_parent(output_labels)
    np.save(output_labels, result.labels)

    manifest = build_manifest(metadata, result.labels.tolist(), result.model_info, embeddings_path)
    ensure_parent(output_json)
    output_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster reduced video embeddings")
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_REDUCED_PATH, help="Path to reduced embeddings .npy file")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_CSV, help="CSV with metadata rows corresponding to embeddings")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON, help="Where to write clustering manifest")
    parser.add_argument("--output-labels", type=Path, default=DEFAULT_OUTPUT_LABELS, help="Where to write raw cluster labels (.npy)")
    parser.add_argument("--n-clusters", type=int, default=10, help="Number of clusters for KMeans")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cluster_embeddings(
        embeddings_path=args.embeddings,
        metadata_path=args.metadata,
        output_json=args.output_json,
        output_labels=args.output_labels,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
