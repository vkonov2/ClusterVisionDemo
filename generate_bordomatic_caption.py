#!/usr/bin/env python3
"""
Offline boardomatic (image + caption) analyzer using open-source vision-language models.

Default model: Qwen/Qwen2.5-VL-7B-Instruct (self-hosted, no external API).
Input format: folder with images `1.png..N.png` and `caption.json`.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def build_prompt() -> str:
    lines = (
        "You are a video-to-text transcription model. Describe the video with maximal detail and complete neutrality.",
        "Do NOT infer or guess intent, emotions, or target audience — only describe what is observable.",
        "Your output must follow this structure:",
        "1. Characters and Demographics: Number of people, approximate age range, gender presentation, clothing, notable traits.",
        "2. Setting and Environment: Indoor/outdoor, environment style, objects present.",
        "3. Narrative Summary: Step-by-step description of what happens. Include actions, transitions, scene changes, interactions.",
        "4. Product and Branding: What product/service appears; packaging, logos, close-ups, usage demonstrations.",
        "5. Text and Speech: All spoken lines and on-screen text (if any).",
        "6. Visual Details: Colors, lighting, props, symbols, recognizable items.",
        "7. Audio/Music: Mention only if clearly present (but do not infer intent).",
        "8. Visual Style and Editing: Camera moves, shot duration, transitions, pacing, framing.",
        "9. Call to Action: Any explicit CTA shown or spoken.",
        "Describe everything fully, factually, and concretely, without analysis or opinions.",
    )
    return "\n".join(lines)


@dataclass
class SceneItem:
    scene_id: str
    image_path: Path
    scene_caption: str


STRUCTURED_KEYS: list[tuple[str, str]] = [
    ("characters_and_demographics", "Characters and Demographics"),
    ("setting_and_environment", "Setting and Environment"),
    ("narrative_summary", "Narrative Summary"),
    ("product_and_branding", "Product and Branding"),
    ("text_and_speech", "Text and Speech"),
    ("visual_details", "Visual Details"),
    ("audio_music", "Audio/Music"),
    ("visual_style_and_editing", "Visual Style and Editing"),
    ("call_to_action", "Call to Action"),
]


def _scene_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def load_bordomatic(data_dir: Path, caption_file: Path) -> tuple[dict[str, Any], list[SceneItem]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not caption_file.exists():
        raise FileNotFoundError(f"caption.json not found: {caption_file}")

    payload = json.loads(caption_file.read_text(encoding="utf-8"))
    scenes_map = payload.get("сцены", {})
    if not isinstance(scenes_map, dict):
        raise ValueError('Field "сцены" must be an object in caption.json')

    image_paths = sorted(
        [p for p in data_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}],
        key=_scene_sort_key,
    )
    if not image_paths:
        raise ValueError(f"No images found in {data_dir}")

    items: list[SceneItem] = []
    for image_path in image_paths:
        scene_id = image_path.stem
        items.append(
            SceneItem(
                scene_id=scene_id,
                image_path=image_path,
                scene_caption=str(scenes_map.get(scene_id, "")),
            )
        )
    return payload, items


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if args.prompt:
        return args.prompt.strip()
    return build_prompt()


def build_json_schema() -> dict[str, Any]:
    props = {key: {"type": "string"} for key, _ in STRUCTURED_KEYS}
    return {
        "type": "object",
        "properties": props,
        "required": [key for key, _ in STRUCTURED_KEYS],
        "additionalProperties": False,
    }


def render_structured_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    for idx, (key, title) in enumerate(STRUCTURED_KEYS, start=1):
        value = str(payload.get(key, "")).strip() or "Not explicitly visible in provided storyboard."
        lines.append(f"{idx}. {title}: {value}")
    return "\n".join(lines)


def extract_assistant_text(raw_generated: Any) -> str:
    if isinstance(raw_generated, list):
        for turn in reversed(raw_generated):
            if isinstance(turn, dict) and turn.get("role") == "assistant":
                return str(turn.get("content", "")).strip()
        return str(raw_generated).strip()
    return str(raw_generated).strip()


def parse_json_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def build_scene_instruction(
    *,
    base_prompt: str,
    scene_id: str,
    scene_caption: str,
    plot_caption: str,
    voice_over: str,
) -> str:
    return (
        f"{base_prompt}\n\n"
        "Context for this task:\n"
        f"- Story (сюжет): {plot_caption}\n"
        f"- Voice-over (закадровый текст): {voice_over}\n"
        f"- Current scene id: {scene_id}\n"
        f"- Scene caption (сцены[{scene_id}]): {scene_caption}\n\n"
        "Analyze only what is visible in this frame and what is explicitly written in the provided texts."
    )


def analyze_scenes(
    *,
    model_id: str,
    items: list[SceneItem],
    base_prompt: str,
    plot_caption: str,
    voice_over: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> list[dict[str, Any]]:
    from PIL import Image
    from transformers import pipeline
    import torch

    if device == "auto":
        resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this machine.")
        resolved_device = "cuda:0"
    elif device == "cpu":
        resolved_device = "cpu"
    else:
        raise ValueError(f"Unsupported device: {device}")

    print(f"[load] Using device: {resolved_device} (MPS disabled)")

    pipe = pipeline(
        task="image-text-to-text",
        model=model_id,
        dtype="auto",
        device_map={"": resolved_device},
    )

    results: list[dict[str, Any]] = []
    for item in items:
        image = Image.open(item.image_path).convert("RGB")
        instruction = build_scene_instruction(
            base_prompt=base_prompt,
            scene_id=item.scene_id,
            scene_caption=item.scene_caption,
            plot_caption=plot_caption,
            voice_over=voice_over,
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        generate_kwargs: dict[str, Any] = {
            "text": messages,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
        generated = pipe(**generate_kwargs)
        raw = generated[0].get("generated_text", "")
        if isinstance(raw, list):
            # Some chat pipelines return a full conversation list; keep only last assistant turn.
            assistant_text = ""
            for turn in reversed(raw):
                if isinstance(turn, dict) and turn.get("role") == "assistant":
                    assistant_text = str(turn.get("content", "")).strip()
                    break
            model_text = assistant_text or str(raw)
        else:
            model_text = str(raw).strip()

        results.append(
            {
                "scene_id": item.scene_id,
                "image_path": str(item.image_path),
                "scene_caption": item.scene_caption,
                "analysis": model_text,
            }
        )
    return results


def analyze_bordomatic_holistic(
    *,
    model_id: str,
    items: list[SceneItem],
    base_prompt: str,
    plot_caption: str,
    voice_over: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    guided_json: bool,
) -> tuple[str, dict[str, Any] | None, str | None]:
    from PIL import Image
    from transformers import pipeline
    import torch

    if device == "auto":
        resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this machine.")
        resolved_device = "cuda:0"
    elif device == "cpu":
        resolved_device = "cpu"
    else:
        raise ValueError(f"Unsupported device: {device}")

    print(f"[load] Using device: {resolved_device} (MPS disabled)")
    pipe = pipeline(
        task="image-text-to-text",
        model=model_id,
        dtype="auto",
        device_map={"": resolved_device},
    )

    content: list[dict[str, Any]] = []
    content.append(
        {
            "type": "text",
            "text": (
                f"{base_prompt}\n\n"
                "You are given a storyboard (bordomatic) for a future video. "
                "Treat all scenes as a single connected narrative and produce ONE unified final report.\n\n"
                "Global context:\n"
                f"- Story (сюжет): {plot_caption}\n"
                f"- Voice-over (закадровый текст): {voice_over}\n\n"
                "Scene materials follow. Each scene contains its original scene caption and image."
            ),
        }
    )

    for item in items:
        content.append(
            {
                "type": "text",
                "text": f"Scene {item.scene_id} caption: {item.scene_caption}",
            }
        )
        content.append(
            {
                "type": "image",
                "image": Image.open(item.image_path).convert("RGB"),
            }
        )

    content.append(
        {
            "type": "text",
            "text": (
                "Now provide one consolidated answer across all scenes.\n"
                "Do not produce per-scene mini-reports; output one cohesive analysis only.\n"
                "Do not infer unobservable intent or emotions.\n"
                "Return content for all 9 required sections."
            ),
        }
    )

    messages = [{"role": "user", "content": content}]
    generate_kwargs: dict[str, Any] = {
        "text": messages,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    schema = build_json_schema()
    if guided_json:
        try:
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import (
                build_transformers_prefix_allowed_tokens_fn,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Guided JSON mode requires lm-format-enforcer. "
                "Install it in your environment: pip install lm-format-enforcer"
            ) from exc

        parser = JsonSchemaParser(schema)
        prefix_fn = build_transformers_prefix_allowed_tokens_fn(pipe.tokenizer, parser)
        generate_kwargs["prefix_allowed_tokens_fn"] = prefix_fn
        content.append(
            {
                "type": "text",
                "text": (
                    "IMPORTANT: return ONLY valid JSON that strictly matches this schema:\n"
                    f"{json.dumps(schema, ensure_ascii=False)}"
                ),
            }
        )

    generated = pipe(**generate_kwargs)

    raw = generated[0].get("generated_text", "")
    assistant_text = extract_assistant_text(raw)
    if not guided_json:
        return assistant_text, None, None

    try:
        parsed = parse_json_response(assistant_text)
    except json.JSONDecodeError as exc:
        err = (
            "guided_json_parse_failed: model output is not valid JSON "
            f"(likely truncated). Consider increasing --max-new-tokens. Details: {exc}"
        )
        return assistant_text, None, err

    # Ensure every required key exists even if model returned sparse fields.
    normalized = {key: str(parsed.get(key, "")).strip() for key, _ in STRUCTURED_KEYS}
    return render_structured_report(normalized), normalized, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze boardomatic image+caption pairs with a local VLM.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/bordomatics/test"))
    parser.add_argument("--caption-file", type=Path, default=None, help="Path to caption.json")
    parser.add_argument("--output", type=Path, default=Path("outputs/captions/bordomatic_test.json"))
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--prompt", default=None, help="Inline prompt text (overrides default).")
    parser.add_argument("--prompt-file", type=Path, default=None, help="Path to prompt .txt/.md file.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device. 'auto' uses CUDA if available, otherwise CPU. MPS is intentionally not used.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling (can be less stable).")
    parser.add_argument("--temperature", type=float, default=0.2, help="Used only with --do-sample.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Used only with --do-sample.")
    parser.add_argument(
        "--guided-json",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce strict JSON schema during decoding (recommended for stable output format).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    caption_file = args.caption_file or (args.data_dir / "caption.json")
    payload, items = load_bordomatic(args.data_dir, caption_file)
    prompt = resolve_prompt(args)

    final_analysis, structured_json, parse_error = analyze_bordomatic_holistic(
        model_id=args.model_id,
        items=items,
        base_prompt=prompt,
        plot_caption=str(payload.get("сюжет", "")),
        voice_over=str(payload.get("закадровый текст", "")),
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        guided_json=args.guided_json,
    )

    output = {
        "model": args.model_id,
        "data_dir": str(args.data_dir),
        "caption_file": str(caption_file),
        "prompt": prompt,
        "story": payload.get("сюжет", ""),
        "voice_over": payload.get("закадровый текст", ""),
        "scene_count": len(items),
        "scene_inputs": [
            {
                "scene_id": item.scene_id,
                "image_path": str(item.image_path),
                "scene_caption": item.scene_caption,
            }
            for item in items
        ],
        "guided_json": args.guided_json,
        "analysis_parse_error": parse_error,
        "analysis_structured": structured_json,
        "analysis": final_analysis,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[output] Saved boardomatic analysis to {args.output}")


if __name__ == "__main__":
    main()
