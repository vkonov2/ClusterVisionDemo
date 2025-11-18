#!/usr/bin/env python3
"""
Generate rich captions for short videos using the openinterx/UGC-VideoCaptioner model.

The script performs the following steps:
1. Installs required dependencies (torch, transformers, decord, soundfile, qwen_omni_utils).
2. Loads the UGC-VideoCaptioner (Qwen2.5 Omni 3B) model with automatic device placement.
3. Sends the video and a structured marketing-oriented prompt to the model.
4. Prints the generated caption describing storyline, perceived advertising intent, and sentiment cues.

Example:
    python generate_ugc_video_caption.py --video path/to/video.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


REQUIRED_PACKAGES: List[str] = [
    "torch",  # core tensor library with CUDA support when available
    "transformers",  # model and processor
    "decord",  # video decoding backend expected by the processor
    "soundfile",  # optional audio output dependency from reference code
    "qwen_omni_utils",  # utilities for multimodal input formatting
]


def install_dependencies() -> None:
    """Install missing Python dependencies via pip in the current environment."""
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-U",
        *REQUIRED_PACKAGES,
    ]
    print("[setup] Installing/updating dependencies: " + " ".join(REQUIRED_PACKAGES))
    subprocess.check_call(command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate marketing-aware captions for a short video.")
    parser.add_argument(
        "--video",
        required=True,
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


def main() -> None:
    args = parse_args()
    install_dependencies()

    # Imports after dependency installation
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info

    if not args.video.exists():
        raise FileNotFoundError(f"Video file not found: {args.video}")

    prompt_text = build_prompt()

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
                {"type": "video", "video": str(args.video)},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    use_audio_in_video = True

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
