#!/usr/bin/env python3
"""
Generate rich captions for short videos using the openinterx/UGC-VideoCaptioner model.

The script performs the following steps:
1. Installs required dependencies (torch, transformers, soundfile, qwen_omni_utils).
2. Loads the UGC-VideoCaptioner (Qwen2.5 Omni 3B) model with automatic device placement.
3. Sends the video and a structured marketing-oriented prompt to the model.
4. Prints the generated caption describing storyline, perceived advertising intent, and sentiment cues.

Example:
    python generate_ugc_video_caption.py --video path/to/video.mp4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


REQUIRED_PACKAGES: List[str] = [
    "torch",  # core tensor library with CUDA support when available
    "transformers",  # model and processor
    "soundfile",  # optional audio output dependency from reference code
    "imageio[ffmpeg]",  # lightweight video decoding to avoid backend segfaults
    "qwen_omni_utils",  # utilities for multimodal input formatting
]


def _is_installed(package: str) -> bool:
    """Heuristically check whether a package/module can be imported."""

    import importlib.util

    name = package.split("[")[0].split("==")[0].replace("-", "_")
    return importlib.util.find_spec(name) is not None


def _install_packages(packages: Iterable[str]) -> None:
    env = dict(os.environ)
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    for package in packages:
        command = [sys.executable, "-m", "pip", "install", "-U", "--quiet", package]
        subprocess.check_call(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env)


def install_dependencies() -> None:
    """Install only the missing Python dependencies via pip in the current environment."""

    missing = [pkg for pkg in REQUIRED_PACKAGES if not _is_installed(pkg)]
    if not missing:
        print("[setup] Dependencies already installed; skipping pip installs.")
        return

    print("[setup] Installing missing dependencies: " + " ".join(missing))
    _install_packages(missing)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate marketing-aware captions for a short video.")
    parser.add_argument(
        "--video",
        required=False,
        default=Path("data/videos/000-youtube.mp4"),
        type=Path,
        help="Path to the video file (recommend <= ~1 minute, GPU ~24GB).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=320,
        help="Maximum number of tokens to generate for the caption.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature; lower values yield more deterministic outputs.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling value.",
    )
    return parser.parse_args()


def build_prompt() -> str:
    """Create a structured prompt targeting storyline, audience view, and sentiment."""
    return (
        "You will watch a short-form UGC video with audio. Provide a concise yet rich description that covers: "
        "(1) Storyline — what happens, the setting, key people/objects, and their actions; "
        "(2) Audience perception — how the target viewer would interpret the clip and what product/service or brand is being hinted or promoted; "
        "(3) Emotion — the emotional tone or sentiment conveyed by visuals and audio. "
        "Write in a natural paragraph (no bullet points) that fuses visual and audio cues."
    )


def load_video_frames(video_path: Path, target_fps: float = 2.0, max_frames: int = 256) -> List["Image.Image"]:
    """Decode a video into RGB frames using imageio with simple downsampling."""

    import imageio.v2 as imageio
    from PIL import Image

    reader = imageio.get_reader(str(video_path))
    meta = reader.get_meta_data()
    native_fps = float(meta.get("fps", target_fps) or target_fps)
    fps = target_fps if native_fps <= 0 else native_fps
    step = max(int(round(fps / target_fps)), 1)

    frames: List[Image.Image] = []
    for idx, frame in enumerate(reader):
        if idx % step != 0:
            continue
        frames.append(Image.fromarray(frame).convert("RGB"))
        if len(frames) >= max_frames:
            break
    reader.close()

    if not frames:
        raise ValueError(f"No frames could be read from video: {video_path}")
    return frames


def main() -> None:
    args = parse_args()
    install_dependencies()

    # Imports after dependency installation
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info
    from PIL import Image

    if not args.video.exists():
        raise FileNotFoundError(f"Video file not found: {args.video}")

    prompt_text = build_prompt()

    print("[prepare] Decoding video frames with imageio to avoid native backend issues...")
    frames: List[Image.Image] = load_video_frames(args.video)

    print("[load] Loading model and processor (may take a while on first run)...")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "openinterx/UGC-VideoCaptioner",
        torch_dtype="auto",
        device_map="auto",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained("openinterx/UGC-VideoCaptioner")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames, "fps": 2.0},
                {"type": "audio", "audio": str(args.video)},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    use_audio_in_video = False

    text_prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=use_audio_in_video
    )
    inputs = processor(
        text=text_prompt,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    print("[infer] Generating caption...")
    generated_tokens = model.generate(
        **inputs,
        use_audio_in_video=use_audio_in_video,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    captions = processor.batch_decode(
        generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print("\n=== Generated Caption ===")
    for caption in captions:
        print(caption.strip())


if __name__ == "__main__":
    main()
