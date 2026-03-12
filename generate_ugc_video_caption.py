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


def install_dependencies() -> None:
    """Install only the missing Python dependencies via pip in the current environment."""

    missing = [pkg for pkg in REQUIRED_PACKAGES if not _is_installed(pkg)]
    if not missing:
        print("[setup] Dependencies already installed; skipping pip installs.")
        return

    print("[setup] Installing missing dependencies: " + " ".join(missing))
    env = dict(os.environ)
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    for package in missing:
        command = [sys.executable, "-m", "pip", "install", "-U", "--quiet", package]
        subprocess.check_call(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate marketing-aware captions for a short video.")
    parser.add_argument(
        "--video",
        required=False,
        # default=Path("data/videos/000-youtube.mp4"),
        default=Path("/Users/konov/Downloads/synapsight/rsa_test.mp4"),
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
    lines = (
        "You are a video-to-text transcription model. Describe the video with maximal detail and complete neutrality.",
        "Do NOT infer or guess intent, emotions, or target audience — only describe what is observable.",
        "Your output must follow this structure:",
        "1. Characters and Demographics: Number of people, their approximate age range, gender presentation, clothing style, notable visual traits. Social context indicators (e.g., family, professionals, students).",
        "2. Setting and Environment: Indoor/outdoor location, style of interior or environment, objects present, socioeconomic cues.",
        "3. Narrative Summary: Step-by-step description of what happens in the video. Include actions, transitions, scene changes, product usage, interactions.",
        "4. Product and Branding: What product or service appears. How it is visually shown: packaging, logos, close-ups, usage demonstrations.",
        "5. Text and Speech: All spoken lines, on-screen text, slogans, captions. Tone of voice, pace, and formality level (without interpretation).",
        "6. Visual and Symbolic Details: Colors, lighting, props, gestures, symbols, cultural references. Any technology, vehicles, food, clothing brands, or recognizable items.",
        "7. Audio and Music: Genre, tempo, instruments, sound effects, voice-over characteristics.",
        "8. Visual Style and Editing: Camera movements, shot duration, transitions, pacing, framing style.",
        "9. Call to Action: Any explicit CTA shown or spoken.",
        "Describe everything fully, factually, and concretely, without adding analysis, assumptions, or opinions.",
    )
    return "\n".join(lines)


def load_video_frames(video_path: Path, target_fps: float = 2.0, max_frames: int = 256):
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
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
    from qwen_omni_utils import process_mm_info
    from PIL import Image
    import soundfile as sf

    if not args.video.exists():
        raise FileNotFoundError(f"Video file not found: {args.video}")

    prompt_text = build_prompt()

    print("[prepare] Decoding video frames with imageio to avoid native backend issues...")
    frames: List[Image.Image] = load_video_frames(args.video)

    print("[load] Loading model and processor (may take a while on first run)...")
    model_local = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "openinterx/UGC-VideoCaptioner",
        dtype="auto",
        device_map={"": "cpu"},
        # attn_implementation="flash_attention_2",
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
    inputs = inputs.to(model_local.device).to(model_local.dtype)

    print("[infer] Generating caption...")
    generated_tokens, audio = model_local.generate(
        **inputs,
        use_audio_in_video=use_audio_in_video,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if audio is not None:
        audio_path = Path("data/audios/010-30.wav")
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_waveform = audio.reshape(-1).detach().cpu().numpy()
        sf.write(audio_path, audio_waveform, samplerate=24000)
        print(f"[audio] Saved synthesized narration to {audio_path}")

    captions = processor.batch_decode(
        generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print("\n=== Generated Caption ===")
    print("\n".join([caption.strip() for caption in captions]))
    output_path = Path("outputs/captions") / f"{args.video.stem}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for caption in captions:
            caption_stripped = caption.strip()
            f.write(caption_stripped + "\n")
    print(f"[output] Captions saved to {output_path}")


if __name__ == "__main__":
    main()
