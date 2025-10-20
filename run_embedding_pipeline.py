"""Полный пайплайн извлечения эмбеддингов из рекламных роликов."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Hugging Face токенизаторы должны знать, что мы запрещаем внутренний параллелизм
# до того, как они будут импортированы. Это убирает повторяющиеся предупреждения
# "The current process just got forked..." при работе SentenceTransformer.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tqdm import tqdm

from extract_text_embedding import SENT_EMB_MODEL, WHISPER_MODEL, extract_text_embedding
from extract_timesformer_embedding import (
    MODEL_ID as TIMESFORMER_ID,
    NUM_FRAMES as TIMESFORMER_FRAMES,
    extract_timesformer_embedding,
    get_device as get_video_device,
)
from extract_wav2vec2_sp import (
    CHUNK_SEC as WAV_CHUNK_SEC,
    MODEL_ID as WAV_MODEL_ID,
    OVERLAP_SEC as WAV_OVERLAP_SEC,
    SAMPLE_RATE as WAV_SAMPLE_RATE,
    Wav2Vec2Resources,
    extract_wav2vec2_embedding,
    get_device as get_audio_device,
)
from extract_yamnet_bg import SAMPLE_RATE as YAMNET_SAMPLE_RATE, extract_yamnet_background_embedding
from pipeline_utils import as_posix, build_video_slug, find_video_file


DATASET_PATH = Path("VideoDataset.xlsx")
MANIFEST_PATH = Path("data/videos/download_manifest.json")
VIDEO_DIR = Path("data/videos")
EMBEDDINGS_DIR = Path("data/embeddings")
PIPELINE_MANIFEST = EMBEDDINGS_DIR / "pipeline_manifest.json"
CONCAT_PATH = EMBEDDINGS_DIR / "concatenated_embeddings.npy"
SCALED_PATH = EMBEDDINGS_DIR / "scaled_embeddings.npy"
REDUCED_PATH = EMBEDDINGS_DIR / "reduced_embeddings.npy"
METADATA_CSV = EMBEDDINGS_DIR / "embeddings_metadata.csv"

VARIANCE_THRESHOLD = 0.95
MAX_COMPONENTS = 256

EXPECTED_DIMENSIONS = {
    "E_video": 768,
    "E_aud_bg": 1024,
    "E_aud_sp": 768,
    "E_text": 384,
}


@dataclass
class ExtractorBundle:
    timesformer_processor: object
    timesformer_model: object
    yamnet_layer: object
    wav_resources: Wav2Vec2Resources
    whisper_model: object
    sentence_model: object
    video_device: object
    audio_device: object


def prepare_extractors() -> ExtractorBundle:
    from transformers import AutoImageProcessor, TimesformerModel, AutoProcessor, Wav2Vec2Model
    import tensorflow_hub as hub
    import whisper
    from sentence_transformers import SentenceTransformer

    video_device = get_video_device()
    timesformer_processor = AutoImageProcessor.from_pretrained(TIMESFORMER_ID)
    timesformer_model = TimesformerModel.from_pretrained(TIMESFORMER_ID).to(video_device)
    timesformer_model.eval()

    yamnet_layer = hub.KerasLayer("https://tfhub.dev/google/yamnet/1")

    audio_device = get_audio_device()
    wav_processor = AutoProcessor.from_pretrained(WAV_MODEL_ID)
    wav_model = Wav2Vec2Model.from_pretrained(WAV_MODEL_ID).to(audio_device)
    wav_model.eval()
    wav_resources = Wav2Vec2Resources(processor=wav_processor, model=wav_model)

    whisper_device = "cpu" if str(video_device) == "mps" else str(audio_device)
    whisper_model = whisper.load_model(WHISPER_MODEL, device=whisper_device)
    sentence_model = SentenceTransformer(SENT_EMB_MODEL, device="cpu" if str(video_device) == "mps" else str(audio_device))

    return ExtractorBundle(
        timesformer_processor=timesformer_processor,
        timesformer_model=timesformer_model,
        yamnet_layer=yamnet_layer,
        wav_resources=wav_resources,
        whisper_model=whisper_model,
        sentence_model=sentence_model,
        video_device=video_device,
        audio_device=audio_device,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_extract(
    name: str,
    expected_dim: int,
    func,
    *args,
    **kwargs,
) -> Tuple[np.ndarray, Dict[str, Optional[str]], Optional[object]]:
    info: Dict[str, Optional[str]] = {"status": "ok", "error": None}
    extra: Optional[object] = None
    try:
        result = func(*args, **kwargs)
        if isinstance(result, tuple):
            embedding = result[0]
            extra = result[1] if len(result) > 1 else None
        else:
            embedding = result
        emb_np = np.asarray(embedding, dtype=np.float32)
        if emb_np.ndim != 1 or emb_np.shape[0] != expected_dim:
            raise ValueError(f"{name}: unexpected shape {emb_np.shape}")
        return emb_np, info, extra
    except Exception as exc:
        info["status"] = "failed"
        info["error"] = str(exc)
        return np.zeros((expected_dim,), dtype=np.float32), info, None


def gather_manifest(manifest_path: Path) -> Dict[int, Dict[str, object]]:
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {int(entry["index"]): entry for entry in manifest.get("entries", [])}
    return {}


def reduce_embeddings(matrix: np.ndarray, source_slices: Dict[str, Tuple[int, int]]) -> Tuple[np.ndarray, Dict[str, object]]:
    if len(matrix) == 0:
        return matrix, {
            "method": "pca",
            "n_components": 0,
            "explained_variance_ratio": [],
            "variance_threshold": VARIANCE_THRESHOLD,
            "max_components": MAX_COMPONENTS,
            "component_source_energy": [],
        }

    max_components = min(matrix.shape[0], matrix.shape[1], MAX_COMPONENTS)
    full_pca = PCA(n_components=max_components, svd_solver="full")
    full_pca.fit(matrix)
    cumsum = np.cumsum(full_pca.explained_variance_ratio_)
    keep = np.searchsorted(cumsum, VARIANCE_THRESHOLD) + 1
    keep = int(max(1, min(keep, max_components)))

    pca = PCA(n_components=keep, svd_solver="full")
    reduced = pca.fit_transform(matrix)

    loadings = pca.components_ ** 2
    component_source_energy: List[Dict[str, float]] = []
    for comp_idx in range(loadings.shape[0]):
        total = float(loadings[comp_idx].sum())
        energy: Dict[str, float] = {}
        for name, (start, end) in source_slices.items():
            portion = float(loadings[comp_idx, start:end].sum())
            energy[name] = portion / total if total > 0 else 0.0
        component_source_energy.append(energy)

    metadata = {
        "method": "pca",
        "n_components": int(keep),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "variance_threshold": VARIANCE_THRESHOLD,
        "max_components": MAX_COMPONENTS,
        "component_source_energy": component_source_energy,
    }
    return reduced.astype(np.float32), metadata


def run_pipeline(
    dataset_path: Path = DATASET_PATH,
    manifest_path: Path = MANIFEST_PATH,
    video_dir: Path = VIDEO_DIR,
    embeddings_dir: Path = EMBEDDINGS_DIR,
) -> Dict[str, object]:
    ensure_dir(video_dir)
    ensure_dir(embeddings_dir)

    df = pd.read_excel(dataset_path)
    manifest_entries = gather_manifest(manifest_path)
    extractors = prepare_extractors()

    concatenated: List[np.ndarray] = []
    metadata_rows: List[Dict[str, object]] = []
    pipeline_entries: List[Dict[str, object]] = []

    source_slices: Dict[str, Tuple[int, int]] = {}
    running_index = 0
    for name, dim in EXPECTED_DIMENSIONS.items():
        source_slices[name] = (running_index, running_index + dim)
        running_index += dim

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embeddings"):
        slug = build_video_slug(idx, row.get("name"), row.get("link"))
        video_path = find_video_file(video_dir, slug)
        entry_meta: Dict[str, object] = {
            "index": int(idx),
            "slug": slug,
            "video_path": as_posix(video_path),
            "name": row.get("name"),
            "link": row.get("link"),
        }

        download_record = manifest_entries.get(int(idx))
        if download_record:
            entry_meta["download"] = download_record

        if video_path is None:
            entry_meta["status"] = "missing_video"
            pipeline_entries.append(entry_meta)
            continue

        embed_dir = embeddings_dir / slug
        ensure_dir(embed_dir)

        results: Dict[str, Dict[str, Optional[str]]] = {}

        video_emb, info, _ = safe_extract(
            "E_video",
            EXPECTED_DIMENSIONS["E_video"],
            extract_timesformer_embedding,
            video_path.as_posix(),
            num_frames=TIMESFORMER_FRAMES,
            device=extractors.video_device,
            processor=extractors.timesformer_processor,
            model=extractors.timesformer_model,
        )
        results["E_video"] = info
        np.save(embed_dir / "E_video.npy", video_emb)

        yam_emb, info, _ = safe_extract(
            "E_aud_bg",
            EXPECTED_DIMENSIONS["E_aud_bg"],
            extract_yamnet_background_embedding,
            video_path.as_posix(),
            sample_rate=YAMNET_SAMPLE_RATE,
            yamnet_layer=extractors.yamnet_layer,
        )
        results["E_aud_bg"] = info
        np.save(embed_dir / "E_aud_bg.npy", yam_emb)

        wav_emb, info, wav_extra = safe_extract(
            "E_aud_sp",
            EXPECTED_DIMENSIONS["E_aud_sp"],
            extract_wav2vec2_embedding,
            video_path.as_posix(),
            sample_rate=WAV_SAMPLE_RATE,
            chunk_sec=WAV_CHUNK_SEC,
            overlap_sec=WAV_OVERLAP_SEC,
            device=extractors.audio_device,
            resources=extractors.wav_resources,
        )
        results["E_aud_sp"] = info
        np.save(embed_dir / "E_aud_sp.npy", wav_emb)
        if wav_extra is not None:
            samples = int(wav_extra)
            entry_meta["audio_samples"] = samples
            entry_meta["audio_seconds"] = round(samples / float(WAV_SAMPLE_RATE), 3)

        text_emb, info, transcript = safe_extract(
            "E_text",
            EXPECTED_DIMENSIONS["E_text"],
            extract_text_embedding,
            video_path.as_posix(),
            sample_rate=WAV_SAMPLE_RATE,
            noise_profile_secs=0.5,
            whisper_model_name=WHISPER_MODEL,
            sentence_model_name=SENT_EMB_MODEL,
            whisper_model=extractors.whisper_model,
            sentence_model=extractors.sentence_model,
        )
        results["E_text"] = info
        np.save(embed_dir / "E_text.npy", text_emb)

        if transcript:
            transcript_path = embed_dir / "transcript.txt"
            transcript_path.write_text(transcript, encoding="utf-8")
            entry_meta["transcript_path"] = as_posix(transcript_path)

        concat_vec = np.concatenate([video_emb, yam_emb, wav_emb, text_emb]).astype(np.float32)
        np.save(embed_dir / "E_concat.npy", concat_vec)
        entry_meta["combined_path"] = as_posix(embed_dir / "E_concat.npy")
        entry_meta["embedding_status"] = results

        concatenated.append(concat_vec)
        metadata_rows.append({"index": idx, "slug": slug, **row.to_dict()})
        pipeline_entries.append(entry_meta)

    matrix = (
        np.stack(concatenated)
        if concatenated
        else np.zeros((0, sum(EXPECTED_DIMENSIONS.values())), dtype=np.float32)
    )
    np.save(CONCAT_PATH, matrix)

    scaling_info: Dict[str, Dict[str, float]] = {}
    if matrix.size:
        modality_norms: Dict[str, float] = {}
        for name, (start, end) in source_slices.items():
            slice_view = matrix[:, start:end]
            norms = np.linalg.norm(slice_view, axis=1)
            mean_norm = float(np.mean(norms)) if len(norms) else 0.0
            modality_norms[name] = mean_norm

        valid_norms = [value for value in modality_norms.values() if value > 0]
        target_norm = float(np.mean(valid_norms)) if valid_norms else 1.0
        target_norm = target_norm or 1.0

        scaled_matrix = matrix.copy()
        for name, (start, end) in source_slices.items():
            mean_norm = modality_norms.get(name, 0.0)
            if mean_norm > 0:
                weight = target_norm / mean_norm
            else:
                weight = 1.0
            scaled_matrix[:, start:end] *= weight
            scaling_info[name] = {"mean_norm": mean_norm, "weight": weight}
    else:
        scaled_matrix = matrix

    if scaled_matrix.size:
        np.save(SCALED_PATH, scaled_matrix)
    else:
        SCALED_PATH.write_text("[]", encoding="utf-8")

    reduced, reduction_meta = reduce_embeddings(scaled_matrix, source_slices)
    if reduced.size:
        np.save(REDUCED_PATH, reduced)
    else:
        REDUCED_PATH.write_text("[]", encoding="utf-8")

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(METADATA_CSV, index=False)

    pipeline_manifest = {
        "dataset": as_posix(dataset_path),
        "video_dir": as_posix(video_dir),
        "embeddings_dir": as_posix(embeddings_dir),
        "source_slices": {k: [int(v[0]), int(v[1])] for k, v in source_slices.items()},
        "entries": pipeline_entries,
        "reduction": reduction_meta,
        "concatenated_path": as_posix(CONCAT_PATH),
        "scaled_path": as_posix(SCALED_PATH) if scaled_matrix.size else None,
        "reduced_path": as_posix(REDUCED_PATH) if reduced.size else None,
        "modality_scaling": scaling_info,
    }
    PIPELINE_MANIFEST.write_text(json.dumps(pipeline_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return pipeline_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multimodal embedding pipeline")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--videos", type=Path, default=VIDEO_DIR)
    parser.add_argument("--embeddings", type=Path, default=EMBEDDINGS_DIR)
    args = parser.parse_args()

    run_pipeline(args.dataset, args.manifest, args.videos, args.embeddings)


if __name__ == "__main__":
    main()

