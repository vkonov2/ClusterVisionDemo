"""Кластеризация видеороликов на основе редуцированных эмбеддингов."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - plotly is optional at runtime
    go = None

DEFAULT_REDUCED_PATH = Path("data/embeddings/reduced_embeddings.npy")
DEFAULT_METADATA_CSV = Path("data/embeddings/embeddings_metadata.csv")
DEFAULT_OUTPUT_JSON = Path("data/embeddings/cluster_manifest.json")
DEFAULT_OUTPUT_LABELS = Path("data/embeddings/cluster_labels.npy")
DEFAULT_METRICS_JSON = Path("data/embeddings/cluster_metrics.json")
DEFAULT_METRICS_PLOT = Path("data/embeddings/cluster_metrics.html")
DEFAULT_PIPELINE_MANIFEST = Path("data/embeddings/pipeline_manifest.json")


@dataclass
class ClusterResult:
    labels: np.ndarray
    model_info: Dict[str, object]
    evaluation: List[Dict[str, object]]


@dataclass
class DeduplicationResult:
    embeddings: np.ndarray
    metadata: List[Dict[str, object]]
    representative_indices: List[int]
    original_to_rep: Dict[int, int]
    groups: List[Dict[str, object]]


def normalize_link(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    return text or None


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


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

    best_silhouette = max(valid, key=lambda item: item.get("silhouette", -1.0))
    silhouette_candidates = [item for item in valid if abs(item.get("silhouette", 0) - best_silhouette.get("silhouette", 0)) < 1e-6]
    if len(silhouette_candidates) == 1:
        return int(best_silhouette["k"])

    def davies_value(item: Dict[str, object]) -> float:
        value = item.get("davies_bouldin")
        return float(value) if value is not None else float("inf")

    best = min(silhouette_candidates, key=davies_value)
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


def build_cluster_documents(record: Dict[str, object]) -> str:
    text_parts: List[str] = []
    for key in ("cluster_hint", "name", "brand", "category", "product", "text"):
        value = record.get(key)
        if isinstance(value, str) and value:
            text_parts.append(value.lower())
    for key in ("tags", "keywords", "topics"):
        value = record.get(key)
        if isinstance(value, str) and value:
            text_parts.append(value.lower())
        elif isinstance(value, (list, tuple)):
            text_parts.extend(str(v).lower() for v in value if isinstance(v, str))
    return " \n ".join(text_parts)


def infer_format_hint(text: str) -> str | None:
    if not text:
        return None
    lowered = text.lower()
    if any(token in lowered for token in ("анимац", "animation", "cartoon", "моушн", "motion")):
        return "animation"
    if any(token in lowered for token in ("3d", "cg", "cgi")):
        return "3d"
    if any(token in lowered for token in ("live", "съем", "съём", "people", "actor")):
        return "live-action"
    if any(token in lowered for token in ("инфограф", "presentation", "slides")):
        return "infographic"
    return None


def infer_cluster_names(
    metadata: Sequence[Dict[str, object]],
    labels: Sequence[int],
    top_tokens: int = 3,
) -> Dict[int, str]:
    docs = [build_cluster_documents(record) for record in metadata]
    labels_array = np.array(labels, dtype=int)
    unique_labels = sorted(set(labels_array.tolist()))
    names: Dict[int, str] = {}

    if docs and any(doc for doc in docs):
        vectorizer = TfidfVectorizer(max_features=512, stop_words=list(STOPWORDS), ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform(docs)
        terms = np.array(vectorizer.get_feature_names_out())
    else:
        tfidf = None
        terms = np.array([])

    for label in unique_labels:
        mask = labels_array == label
        if not mask.any():
            names[label] = f"Cluster {label}"
            continue
        label_docs = [docs[i] for i in np.where(mask)[0]]
        format_hint = infer_format_hint(" \n ".join(label_docs))
        if tfidf is not None and tfidf.shape[0] >= mask.sum():
            cluster_matrix = tfidf[mask]
            scores = np.asarray(cluster_matrix.mean(axis=0)).ravel()
            top_idx = np.argsort(scores)[::-1]
            top_terms = [terms[idx] for idx in top_idx if scores[idx] > 0][:top_tokens]
        else:
            bucket = Counter()
            for idx in np.where(mask)[0]:
                tokens = [tok for tok in TOKEN_RE.findall(docs[idx]) if tok not in STOPWORDS and len(tok) > 2]
                bucket.update(tokens)
            top_terms = [token for token, _ in bucket.most_common(top_tokens)]

        if top_terms:
            base = " / ".join(top_terms)
        else:
            base = f"Cluster {label}"
        if format_hint:
            names[label] = f"{base} — {format_hint}"
        else:
            names[label] = base

    return names


def build_manifest(
    metadata: Sequence[Dict[str, object]],
    labels: Sequence[int],
    model_info: Dict[str, object],
    embedding_path: Path,
    evaluation: Sequence[Dict[str, object]],
    duplicate_groups: Sequence[Dict[str, object]] | None = None,
    noise_indices: Sequence[int] | None = None,
    metrics_plot_path: Path | None = None,
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
    if duplicate_groups:
        manifest["duplicates"] = [
            {
                "representative_index": int(group["representative_index"]),
                "members": [int(idx) for idx in group.get("members", [])],
                "size": int(group.get("size", 0)),
                "reason": group.get("reason"),
            }
            for group in duplicate_groups
        ]
    if noise_indices:
        manifest["noise_indices"] = [int(idx) for idx in noise_indices]
    if metrics_plot_path is not None:
        manifest["metrics_plot"] = str(metrics_plot_path)
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


def select_medoid(indices: Sequence[int], embeddings: np.ndarray) -> int:
    if len(indices) == 1:
        return indices[0]
    subset = embeddings[list(indices)]
    distances = pairwise_distances(subset)
    totals = distances.sum(axis=1)
    best_local = int(np.argmin(totals))
    return indices[best_local]


def deduplicate_embeddings(
    embeddings: np.ndarray,
    metadata: Sequence[Dict[str, object]],
    near_duplicate_quantile: float,
    near_duplicate_multiplier: float,
) -> DeduplicationResult:
    n_items = len(embeddings)
    uf = UnionFind(n_items)
    link_groups: Dict[str, int] = {}
    reasons: Dict[int, set] = defaultdict(set)

    for idx, record in enumerate(metadata):
        link_norm = normalize_link(record.get("link"))
        if not link_norm:
            continue
        existing = link_groups.get(link_norm)
        if existing is None:
            link_groups[link_norm] = idx
            continue
        root_existing = uf.find(existing)
        root_idx = uf.find(idx)
        combined = set()
        combined.update(reasons.get(root_existing, set()))
        combined.update(reasons.get(root_idx, set()))
        combined.add("link")
        uf.union(root_existing, root_idx)
        reasons[uf.find(root_existing)] = combined

    if n_items > 1:
        distance_matrix = pairwise_distances(embeddings, metric="euclidean")
        np.fill_diagonal(distance_matrix, np.inf)
        nearest = distance_matrix.min(axis=1)
        finite_nearest = nearest[np.isfinite(nearest)]
        if finite_nearest.size:
            baseline = float(np.quantile(finite_nearest, np.clip(near_duplicate_quantile, 0.0, 1.0)))
            threshold = baseline * max(near_duplicate_multiplier, 1.0)
            pairs = np.where(distance_matrix <= threshold)
            for i, j in zip(pairs[0], pairs[1]):
                if i >= j:
                    continue
                root_i = uf.find(int(i))
                root_j = uf.find(int(j))
                if root_i == root_j:
                    continue
                combined = set()
                combined.update(reasons.get(root_i, set()))
                combined.update(reasons.get(root_j, set()))
                combined.add("embedding")
                uf.union(root_i, root_j)
                reasons[uf.find(root_i)] = combined

    groups_map: Dict[int, List[int]] = defaultdict(list)
    for idx in range(n_items):
        root = uf.find(idx)
        groups_map[root].append(idx)

    representative_indices: List[int] = []
    dedup_metadata: List[Dict[str, object]] = []
    dedup_embeddings_list: List[np.ndarray] = []
    original_to_rep: Dict[int, int] = {}
    duplicate_groups: List[Dict[str, object]] = []

    for group_members in groups_map.values():
        medoid_idx = select_medoid(group_members, embeddings)
        rep_position = len(representative_indices)
        representative_indices.append(medoid_idx)
        dedup_embeddings_list.append(embeddings[medoid_idx])
        dedup_metadata.append(metadata[medoid_idx])
        for member in group_members:
            original_to_rep[member] = rep_position
        if len(group_members) > 1:
            root = uf.find(group_members[0])
            duplicate_groups.append(
                {
                    "representative_index": int(medoid_idx),
                    "members": [int(idx) for idx in sorted(group_members)],
                    "size": len(group_members),
                    "reason": "/".join(sorted(reasons.get(root, {"embedding"}))),
                }
            )

    dedup_embeddings_array = (
        np.stack(dedup_embeddings_list, axis=0)
        if dedup_embeddings_list
        else np.empty((0, embeddings.shape[1]))
    )
    return DeduplicationResult(
        embeddings=dedup_embeddings_array.astype(np.float32),
        metadata=dedup_metadata,
        representative_indices=representative_indices,
        original_to_rep=original_to_rep,
        groups=duplicate_groups,
    )


def detect_noise_points(
    embeddings: np.ndarray,
    min_samples: int,
    eps_quantile: float,
    eps_multiplier: float,
) -> Tuple[set[int], Dict[str, float]]:
    if embeddings.size == 0 or embeddings.shape[0] < max(min_samples, 3):
        return set(), {"eps": 0.0, "min_samples": float(min_samples)}

    n_neighbors = min(max(min_samples, 2), embeddings.shape[0])
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)
    kth_distances = distances[:, -1]
    eps = float(np.quantile(kth_distances, np.clip(eps_quantile, 0.0, 1.0)))
    eps *= max(eps_multiplier, 0.5)
    if eps <= 0:
        return set(), {"eps": eps, "min_samples": float(min_samples)}

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(embeddings)
    noise = {int(idx) for idx, label in enumerate(labels) if label == -1}
    return noise, {"eps": eps, "min_samples": float(min_samples)}


def render_metrics_plot(
    evaluations: Sequence[Dict[str, object]],
    output_path: Path,
) -> bool:
    if not evaluations or go is None:
        return False

    ks = [item.get("k") for item in evaluations]
    silhouette = [item.get("silhouette") for item in evaluations]
    davies = [item.get("davies_bouldin") for item in evaluations]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=silhouette, mode="lines+markers", name="Silhouette"))
    fig.add_trace(go.Scatter(x=ks, y=davies, mode="lines+markers", name="Davies-Bouldin", yaxis="y2"))
    fig.update_layout(
        title="Clustering metrics across k",
        xaxis_title="k",
        yaxis=dict(title="Silhouette", range=[min((v for v in silhouette if v is not None), default=0), 1.0]),
        yaxis2=dict(
            title="Davies-Bouldin",
            overlaying="y",
            side="right",
            range=[min((v for v in davies if v is not None), default=0), max((v for v in davies if v is not None), default=1)],
        ),
    )
    ensure_parent(output_path)
    fig.write_html(output_path, include_plotlyjs="cdn")
    return True


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
    metrics_plot_path: Path | None,
    near_duplicate_quantile: float,
    near_duplicate_multiplier: float,
    noise_min_samples: int,
    noise_eps_quantile: float,
    noise_eps_multiplier: float,
) -> Dict[str, object]:
    embeddings = load_embeddings(embeddings_path)
    metadata = load_metadata(metadata_path)

    if len(metadata) != len(embeddings):
        raise ValueError(
            "Количество строк в metadata CSV не совпадает с числом эмбеддингов. "
            f"Metadata: {len(metadata)}, embeddings: {len(embeddings)}"
        )

    dedup = deduplicate_embeddings(
        embeddings,
        metadata,
        near_duplicate_quantile=near_duplicate_quantile,
        near_duplicate_multiplier=near_duplicate_multiplier,
    )

    noise_indices, noise_info = detect_noise_points(
        dedup.embeddings,
        min_samples=noise_min_samples,
        eps_quantile=noise_eps_quantile,
        eps_multiplier=noise_eps_multiplier,
    )

    usable_mask = np.ones(len(dedup.embeddings), dtype=bool)
    for idx in noise_indices:
        usable_mask[int(idx)] = False

    clustering_embeddings = dedup.embeddings[usable_mask]

    available_points = clustering_embeddings.shape[0]
    if available_points == 0:
        raise ValueError("No embeddings available for clustering after deduplication and noise filtering")

    candidate_k = sorted({int(k) for k in k_values if 1 < int(k) <= available_points})
    if not candidate_k:
        candidate_k = [min(max(2, n_clusters), available_points)]
    if n_clusters not in candidate_k and 1 < n_clusters <= available_points:
        candidate_k.append(int(n_clusters))
        candidate_k = sorted(candidate_k)

    evaluation: List[Dict[str, object]] = evaluate_k_range(
        clustering_embeddings,
        k_values=candidate_k,
        random_state=random_state,
        sample_size=metric_sample,
    )

    if auto_k:
        chosen_k = choose_best_k(evaluation)
    else:
        chosen_k = n_clusters

    if chosen_k > available_points:
        raise ValueError(
            f"Chosen k={chosen_k} exceeds number of usable embeddings ({available_points}). "
            "Adjust near-duplicate or noise parameters or pick a smaller k."
        )
    if chosen_k < 2:
        raise ValueError("Need at least two clusters for KMeans; adjust parameters or disable noise removal")

    result = run_kmeans(clustering_embeddings, n_clusters=chosen_k, random_state=random_state)
    result.evaluation = evaluation

    full_dedup_labels = np.full(len(dedup.embeddings), -1, dtype=np.int32)
    usable_indices = np.where(usable_mask)[0]
    for idx, label in zip(usable_indices, result.labels):
        full_dedup_labels[int(idx)] = int(label)

    original_labels = np.full(len(metadata), -1, dtype=np.int32)
    for original_idx, rep_position in dedup.original_to_rep.items():
        original_labels[original_idx] = full_dedup_labels[rep_position]

    ensure_parent(output_labels)
    np.save(output_labels, original_labels)

    plot_written = False
    if metrics_plot_path is not None and result.evaluation:
        plot_written = render_metrics_plot(result.evaluation, metrics_plot_path)

    manifest = build_manifest(
        metadata,
        original_labels.tolist(),
        result.model_info,
        embeddings_path,
        evaluation=result.evaluation,
        duplicate_groups=dedup.groups,
        noise_indices=sorted(int(dedup.representative_indices[idx]) for idx in noise_indices),
        metrics_plot_path=metrics_plot_path if plot_written else None,
    )
    noise_meta = manifest.setdefault("noise_detection", {})
    noise_meta.update(noise_info)
    noise_meta["count"] = len(noise_indices)
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
        "--metrics-plot",
        type=Path,
        default=DEFAULT_METRICS_PLOT,
        help="Optional HTML file to visualise metric curves",
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
    parser.add_argument(
        "--near-duplicate-quantile",
        type=float,
        default=0.05,
        help="Quantile of nearest-neighbour distance to treat as baseline for near-duplicate detection",
    )
    parser.add_argument(
        "--near-duplicate-multiplier",
        type=float,
        default=1.2,
        help="Multiplier applied to the baseline distance when merging near duplicates",
    )
    parser.add_argument(
        "--noise-min-samples",
        type=int,
        default=4,
        help="DBSCAN min_samples parameter for noise detection",
    )
    parser.add_argument(
        "--noise-eps-quantile",
        type=float,
        default=0.9,
        help="Quantile of k-distance curve to derive DBSCAN eps",
    )
    parser.add_argument(
        "--noise-eps-multiplier",
        type=float,
        default=1.0,
        help="Scaling factor for automatically derived DBSCAN eps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_k < 2 or args.max_k < args.min_k:
        raise ValueError("Invalid k range: ensure min_k >= 2 and max_k >= min_k")
    k_values = range(args.min_k, args.max_k + 1)
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
        metrics_plot_path=args.metrics_plot,
        near_duplicate_quantile=args.near_duplicate_quantile,
        near_duplicate_multiplier=args.near_duplicate_multiplier,
        noise_min_samples=args.noise_min_samples,
        noise_eps_quantile=args.noise_eps_quantile,
        noise_eps_multiplier=args.noise_eps_multiplier,
    )


if __name__ == "__main__":
    main()
