"""Кластеризация видеороликов на основе редуцированных эмбеддингов."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

DEFAULT_REDUCED_PATH = Path("data/embeddings/reduced_embeddings.npy")
DEFAULT_METADATA_CSV = Path("data/embeddings/embeddings_metadata.csv")
DEFAULT_OUTPUT_JSON = Path("data/embeddings/cluster_manifest.json")
DEFAULT_OUTPUT_LABELS = Path("data/embeddings/cluster_labels.npy")
DEFAULT_METRICS_JSON = Path("data/embeddings/cluster_metrics.json")
DEFAULT_PIPELINE_MANIFEST = Path("data/embeddings/pipeline_manifest.json")


@dataclass
class ClusterResult:
    labels: np.ndarray
    model_info: Dict[str, object]
    evaluation: List[Dict[str, object]]


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


def run_kmeans(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int | None,
) -> ClusterResult:
    if len(embeddings) == 0:
        return ClusterResult(
            labels=np.zeros((0,), dtype=np.int32),
            model_info={
                "algorithm": "kmeans",
                "n_clusters": n_clusters,
                "random_state": random_state,
                "inertia": 0.0,
            },
            evaluation=[],
        )

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
    return ClusterResult(labels=labels.astype(np.int32), model_info=info, evaluation=[])


def evaluate_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int,
    sample_size: int | None = None,
    random_state: int | None = None,
) -> Dict[str, object]:
    if k <= 1 or len(np.unique(labels)) <= 1:
        return {"k": int(k), "silhouette": None, "davies_bouldin": None}

    if sample_size is not None and sample_size < len(embeddings):
        seed = random_state if random_state is not None else 42
        rng = np.random.default_rng(seed=seed)
        subset_idx = rng.choice(len(embeddings), size=sample_size, replace=False)
        sample_vectors = embeddings[subset_idx]
        sample_labels = labels[subset_idx]
    else:
        sample_vectors = embeddings
        sample_labels = labels

    unique_labels = np.unique(sample_labels)
    if len(unique_labels) <= 1:
        return {"k": int(k), "silhouette": None, "davies_bouldin": None}

    silhouette = float(silhouette_score(sample_vectors, sample_labels)) if len(sample_labels) > 1 else None
    davies = float(davies_bouldin_score(sample_vectors, sample_labels)) if len(unique_labels) > 1 else None
    return {"k": int(k), "silhouette": silhouette, "davies_bouldin": davies}


def evaluate_k_range(
    embeddings: np.ndarray,
    k_values: Iterable[int],
    random_state: int | None,
    sample_size: int | None,
) -> List[Dict[str, object]]:
    evaluations: List[Dict[str, object]] = []
    for k in k_values:
        result = run_kmeans(embeddings, n_clusters=k, random_state=random_state)
        metrics = evaluate_k(
            embeddings,
            result.labels,
            k=k,
            sample_size=sample_size,
            random_state=random_state,
        )
        metrics.update({"inertia": result.model_info.get("inertia", 0.0)})
        evaluations.append(metrics)
    return evaluations


def choose_best_k(evaluations: List[Dict[str, object]]) -> int:
    valid = [item for item in evaluations if item.get("silhouette") is not None]
    if not valid:
        return int(evaluations[0]["k"])

    def composite_score(item: Dict[str, object]) -> float:
        silhouette = item.get("silhouette") or -1.0
        davies = item.get("davies_bouldin")
        if davies is None or davies <= 0:
            penalty = 0.0
        else:
            penalty = 1.0 / davies
        return silhouette + penalty

    best = max(valid, key=composite_score)
    return int(best["k"])


STOPWORDS = {
    "и",
    "в",
    "на",
    "для",
    "это",
    "как",
    "с",
    "по",
    "от",
    "за",
    "до",
    "о",
    "из",
    "the",
    "of",
    "to",
    "a",
    "наша",
    "наш",
    "ваш",
    "твой",
    "мой",
}

TOKEN_RE = re.compile(r"[\w\-]+", re.UNICODE)


def infer_cluster_names(
    metadata: Sequence[Dict[str, object]],
    labels: Sequence[int],
    top_tokens: int = 3,
) -> Dict[int, str]:
    buckets: Dict[int, Counter] = defaultdict(Counter)
    for record, label in zip(metadata, labels):
        text_parts: List[str] = []
        for key in ("cluster_hint", "name", "brand", "category", "product"):
            value = record.get(key)
            if isinstance(value, str):
                text_parts.append(value.lower())
        if not text_parts:
            continue
        tokens = [tok for part in text_parts for tok in TOKEN_RE.findall(part)]
        filtered = [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 2 and not tok.isdigit()]
        buckets[int(label)].update(filtered)

    names: Dict[int, str] = {}
    for label, counter in buckets.items():
        most_common = [token for token, _ in counter.most_common(top_tokens)]
        if most_common:
            names[label] = " / ".join(most_common)
        else:
            names[label] = f"Cluster {label}"
    for label in set(int(l) for l in labels):
        names.setdefault(label, f"Cluster {label}")
    return names


def build_manifest(
    metadata: Sequence[Dict[str, object]],
    labels: Sequence[int],
    model_info: Dict[str, object],
    embedding_path: Path,
    evaluation: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    if len(metadata) != len(labels):
        raise ValueError("Metadata and labels length mismatch")

    counts = Counter(int(label) for label in labels)
    cluster_names = infer_cluster_names(metadata, labels)
    entries: List[Dict[str, object]] = []
    for meta, label in zip(metadata, labels):
        entry = {**meta}
        entry["label"] = int(label)
        entry["cluster_name"] = cluster_names.get(int(label))
        entries.append(entry)

    manifest = {
        "embedding_path": str(embedding_path),
        "n_items": len(labels),
        "label_counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "cluster_names": {str(k): v for k, v in sorted(cluster_names.items())},
        "entries": entries,
    }
    manifest.update(model_info)
    manifest["evaluation"] = list(evaluation)
    return manifest


def update_pipeline_manifest(
    pipeline_manifest_path: Path,
    cluster_manifest_path: Path,
    cluster_manifest: Dict[str, object],
) -> None:
    if not pipeline_manifest_path.exists():
        return

    data = json.loads(pipeline_manifest_path.read_text(encoding="utf-8"))
    cluster_entries = cluster_manifest.get("entries", [])
    mapping: Dict[int, Dict[str, object]] = {}
    for entry in cluster_entries:
        idx = entry.get("index")
        if idx is None:
            continue
        try:
            idx_int = int(idx)
        except (TypeError, ValueError):
            continue
        mapping[idx_int] = {
            "label": int(entry.get("label", -1)),
            "name": entry.get("cluster_name"),
        }

    for pipeline_entry in data.get("entries", []):
        idx = pipeline_entry.get("index")
        if idx is None:
            continue
        try:
            idx_int = int(idx)
        except (TypeError, ValueError):
            continue
        info = mapping.get(idx_int)
        if not info:
            continue
        cluster_info = pipeline_entry.setdefault("cluster", {})
        cluster_info["label"] = int(info["label"])
        if info.get("name"):
            cluster_info["name"] = info["name"]

    data["clusters"] = {
        "manifest_path": str(cluster_manifest_path),
        "label_counts": cluster_manifest.get("label_counts"),
        "cluster_names": cluster_manifest.get("cluster_names"),
    }

    pipeline_manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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
    auto_k: bool,
    k_values: Iterable[int],
    metrics_path: Path | None,
    metric_sample: int | None,
    pipeline_manifest_path: Path | None,
) -> Dict[str, object]:
    embeddings = load_embeddings(embeddings_path)
    metadata = load_metadata(metadata_path)

    if len(metadata) != len(embeddings):
        raise ValueError(
            "Количество строк в metadata CSV не совпадает с числом эмбеддингов. "
            f"Metadata: {len(metadata)}, embeddings: {len(embeddings)}"
        )

    evaluation: List[Dict[str, object]] = []
    chosen_k = n_clusters
    if auto_k:
        evaluation = evaluate_k_range(
            embeddings,
            k_values=k_values,
            random_state=random_state,
            sample_size=metric_sample,
        )
        chosen_k = choose_best_k(evaluation)

    result = run_kmeans(embeddings, n_clusters=chosen_k, random_state=random_state)
    if evaluation:
        result.evaluation = evaluation
    else:
        result.evaluation = [
            {
                "k": chosen_k,
                "silhouette": None,
                "davies_bouldin": None,
                "inertia": result.model_info.get("inertia", 0.0),
            }
        ]

    ensure_parent(output_labels)
    np.save(output_labels, result.labels)

    manifest = build_manifest(
        metadata,
        result.labels.tolist(),
        result.model_info,
        embeddings_path,
        evaluation=result.evaluation,
    )
    ensure_parent(output_json)
    output_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if metrics_path is not None and result.evaluation:
        ensure_parent(metrics_path)
        metrics_path.write_text(json.dumps(result.evaluation, ensure_ascii=False, indent=2), encoding="utf-8")

    if pipeline_manifest_path is not None:
        update_pipeline_manifest(pipeline_manifest_path, output_json, manifest)

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster reduced video embeddings")
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_REDUCED_PATH, help="Path to reduced embeddings .npy file")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_CSV, help="CSV with metadata rows corresponding to embeddings")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON, help="Where to write clustering manifest")
    parser.add_argument("--output-labels", type=Path, default=DEFAULT_OUTPUT_LABELS, help="Where to write raw cluster labels (.npy)")
    parser.add_argument("--n-clusters", type=int, default=10, help="Number of clusters for KMeans (used if --auto-k is not set)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--auto-k",
        action="store_true",
        help="Automatically pick k by evaluating silhouette/Davies-Bouldin over a range",
    )
    parser.add_argument("--min-k", type=int, default=6, help="Minimum k for automatic search")
    parser.add_argument("--max-k", type=int, default=12, help="Maximum k for automatic search")
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=DEFAULT_METRICS_JSON,
        help="Optional path to dump evaluation metrics for inspected k values",
    )
    parser.add_argument(
        "--metric-sample",
        type=int,
        default=2000,
        help="Sample size for metric computation (None uses full dataset)",
    )
    parser.add_argument(
        "--pipeline-manifest",
        type=Path,
        default=DEFAULT_PIPELINE_MANIFEST,
        help="If provided, annotate the pipeline manifest with cluster labels",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.auto_k:
        if args.min_k < 2 or args.max_k < args.min_k:
            raise ValueError("Invalid k range: ensure min_k >= 2 and max_k >= min_k")
        k_values = range(args.min_k, args.max_k + 1)
    else:
        k_values = [args.n_clusters]
    cluster_embeddings(
        embeddings_path=args.embeddings,
        metadata_path=args.metadata,
        output_json=args.output_json,
        output_labels=args.output_labels,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        auto_k=args.auto_k,
        k_values=k_values,
        metrics_path=args.metrics_json,
        metric_sample=args.metric_sample if args.metric_sample > 0 else None,
        pipeline_manifest_path=args.pipeline_manifest,
    )


if __name__ == "__main__":
    main()
