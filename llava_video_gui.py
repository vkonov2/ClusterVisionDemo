"""
Interactive Gradio app for LLaVA-NeXT-Video-7B.

The app lets a user upload a video, previews it with a built-in player,
and starts a multimodal chat about the clip. The model is loaded once at
startup and re-used for all requests. See the README in the model card
for more details: https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf.
"""
from __future__ import annotations

import argparse
import os
from functools import lru_cache
from typing import List, Sequence, Tuple

import av
import gradio as gr
import numpy as np
import torch
from transformers import (
    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoProcessor,
)

MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"
DEFAULT_FRAMES = 16


def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_dtype() -> torch.dtype:
    return torch.float16 if torch.cuda.is_available() else torch.float32


@lru_cache(maxsize=1)
def load_model() -> Tuple[LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor]:
    """Load and cache the LLaVA-NeXT-Video model and processor."""

    device = _get_device()
    dtype = _get_dtype()
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model.to(device)

    processor = LlavaNextVideoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def _read_video_pyav(container: av.container.input.InputContainer, indices: Sequence[int]) -> np.ndarray:
    """
    Decode the video with PyAV decoder.

    Args:
        container: PyAV container.
        indices: Frame indices to decode.

    Returns:
        Decoded frames as a numpy array with shape (num_frames, height, width, 3).
    """

    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_video_frames(video_path: str, num_frames: int = DEFAULT_FRAMES) -> Tuple[np.ndarray, int]:
    """Load a video file and sample evenly spaced frames."""

    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    if total_frames == 0:
        raise ValueError("Видео не содержит кадров")

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    clip = _read_video_pyav(container, indices)
    return clip, int(total_frames)


def build_conversation(history: List[Tuple[str, str]], user_message: str) -> List[dict]:
    """Convert chatbot history into the conversation schema expected by the processor."""

    messages: List[dict] = []
    if history:
        first_user, first_assistant = history[0]
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": first_user},
                    {"type": "video"},
                ],
            }
        )
        messages.append({"role": "assistant", "content": [{"type": "text", "text": first_assistant}]})
        for user_text, assistant_text in history[1:]:
            messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})
            messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})
    else:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {"type": "video"},
                ],
            }
        )
        return messages

    messages.append({"role": "user", "content": [{"type": "text", "text": user_message}]})
    return messages


def chat(user_message: str, history: List[Tuple[str, str]], clip: np.ndarray | None):
    """Run a chat turn against the model using the uploaded video clip."""

    if not user_message:
        return history, "Пожалуйста, введите вопрос.", ""

    if clip is None:
        history = history + [(user_message, "Сначала загрузите видео, чтобы начать диалог.")]
        return history, history, ""

    model, processor = load_model()
    device = _get_device()

    messages = build_conversation(history, user_message)
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(
        text=prompt,
        videos=clip,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

    answer = processor.decode(output[0][2:], skip_special_tokens=True)
    updated_history = history + [(user_message, answer)]
    return updated_history, updated_history, ""


def load_video(file_obj) -> Tuple[str, str, np.ndarray, list]:
    """Handle video upload, preview, and frame sampling."""

    if file_obj is None:
        return None, "", None, []

    video_path = file_obj if isinstance(file_obj, str) else file_obj.name
    clip, total_frames = sample_video_frames(video_path)

    info = (
        f"**Загружен файл:** {os.path.basename(video_path)}\n"
        f"• Длина: {total_frames} кадров\n"
        f"• Используется {clip.shape[0]} кадров для контекста"
    )

    return video_path, info, clip, []


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="LLaVA-NeXT-Video Chat") as demo:
        gr.Markdown(
            """
            # Видеочат с LLaVA-NeXT-Video-7B
            1. Слева загрузите видео и дождитесь предпросмотра.\\
            2. Справа задавайте вопросы о ролике — модель поддерживает диалог.
            """
        )

        video_state = gr.State(value=None)
        history_state = gr.State(value=[])

        with gr.Row():
            with gr.Column(scale=2):
                video_input = gr.File(label="Выберите видео", file_types=[".mp4", ".mov", ".mkv"], interactive=True)
                load_btn = gr.Button("Загрузить видео")
                video_player = gr.Video(label="Предпросмотр", interactive=False)
                video_info = gr.Markdown()
            with gr.Column(scale=3):
                chat_box = gr.Chatbot(height=480, label="Диалог")
                user_box = gr.Textbox(label="Ваш вопрос", placeholder="Спросите про содержимое видео…")
                send_btn = gr.Button("Отправить")
                clear_btn = gr.Button("Очистить диалог")

        load_btn.click(
            load_video,
            inputs=[video_input],
            outputs=[video_player, video_info, video_state, history_state],
        )

        send_btn.click(
            chat,
            inputs=[user_box, history_state, video_state],
            outputs=[chat_box, history_state, user_box],
        )

        clear_btn.click(lambda: ([], [], ""), None, [chat_box, history_state, user_box])

    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio GUI for LLaVA-NeXT-Video-7B")
    parser.add_argument("--share", action="store_true", help="Включить публичный Gradio share")
    parser.add_argument("--server-port", type=int, default=7860, help="Порт запуска сервера")
    args = parser.parse_args()

    demo = build_demo()
    demo.queue().launch(share=args.share, server_port=args.server_port)


if __name__ == "__main__":
    main()
