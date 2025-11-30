"""Gradio интерфейс для Video-LLaMA с расширенным логированием.

Скрипт ожидает установленную оригинальную модель Video-LLaMA. Если зависимости
не найдены, загрузка завершается с ошибкой и подробным сообщением, чтобы
пользователь установил модель корректно.

Особенности:
- Левая колонка: загрузка видео, плеер и поток логов.
- Правая колонка: чат с поддержкой диалога по выбранному видео.
- Все сообщения идут через loguru и отображаются также в интерфейсе.
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
from loguru import logger


class UILogSink:
    """Потокобезопасный буфер логов для вывода на страницу.

    Собирает последние сообщения и возвращает их в виде многострочной строки.
    """

    def __init__(self, max_lines: int = 200):
        self._max_lines = max_lines
        self._messages: List[str] = []
        self._lock = threading.Lock()

    def __call__(self, message):
        formatted = (
            f"[{message.record['time']:%H:%M:%S}] "
            f"{message.record['level'].name}: {message.record['message']}"
        )
        with self._lock:
            self._messages.append(formatted)
            # Ограничиваем размер буфера, чтобы не переполнять память.
            if len(self._messages) > self._max_lines:
                self._messages = self._messages[-self._max_lines :]

    def to_text(self) -> str:
        with self._lock:
            return "\n".join(self._messages)


@dataclass
class SessionState:
    """Данные состояния сессии Gradio."""

    video_path: Optional[str] = None
    video_token: Optional[str] = None
    history: List[Tuple[str, str]] = field(default_factory=list)


class VideoLLaMALoader:
    """Менеджер загрузки модели Video-LLaMA.

    Если зависимости недоступны, процесс завершается с подсказкой по установке
    реальной модели, чтобы избежать «тихого» резервного режима.
    """

    def __init__(self, model_root: Optional[str] = None):
        self.model_root = Path(model_root) if model_root else None
        self.model = self._load_model()

    def _load_model(self):
        if importlib.util.find_spec("videollama") is None:
            raise RuntimeError(
                "Пакет videollama не найден. Установите репозиторий Video-LLaMA по инструкции docs/video_llama_setup.md"
            )

        # Импорты выносим без try/except, чтобы отсутствие зависимостей сразу падало с понятной ошибкой.
        load_pretrained_model = importlib.import_module(
            "videollama.model.builder"
        ).load_pretrained_model  # type: ignore[attr-defined]
        eval_model = importlib.import_module("videollama.eval.run_llava").eval_model  # type: ignore[attr-defined]

        try:
            logger.info("Обнаружен пакет Video-LLaMA, начинаем загрузку модели")
            model_path = str(self.model_root) if self.model_root else None
            # Подставьте нужный идентификатор модели из README проекта.
            pretrained = model_path or "DAMO-NLP-SG/VideoLLaMA-7B"
            llm_model, vis_processor, tokenizer = load_pretrained_model(
                pretrained, None, None
            )

            class RealVideoLLaMAWrapper:
                def __init__(self):
                    self.model = llm_model
                    self.vis_processor = vis_processor
                    self.tokenizer = tokenizer

                def prepare_video(self, video_path: Path) -> str:
                    logger.info("Подготовка видео для Video-LLaMA: %s", video_path)
                    # Используем встроенный препроцессор.
                    _ = self.vis_processor(video_path)
                    token = video_path.stem
                    logger.success("Видео подготовлено моделью, token=%s", token)
                    return token

                def chat(self, message: str, history, video_token: str):
                    logger.debug(
                        "Отправка запроса в модель (token=%s, история=%d)",
                        video_token,
                        len(history),
                    )
                    video_path = Path(video_token)
                    conversation = eval_model(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        video=None,
                        image=None,
                        video_path=str(video_path) if video_path.exists() else None,
                        question=message,
                        do_sample=True,
                    )
                    # eval_model может возвращать список сообщений, берём последний текст.
                    if isinstance(conversation, list) and conversation:
                        response = conversation[-1].get("value", "")
                    else:
                        response = str(conversation)
                    logger.success("Ответ модели получен")
                    return response

            logger.success("Модель Video-LLaMA загружена")
            return RealVideoLLaMAWrapper()
        except Exception as exc:  # pragma: no cover - падение загрузки
            hint = (
                "Video-LLaMA не установлена или модель не скачана. "
                "Установите зависимости и веса, затем запустите приложение снова."
            )
            logger.exception("Не удалось загрузить Video-LLaMA: %s", exc)
            raise RuntimeError(hint) from exc

    def prepare_video(self, video_path: Path) -> str:
        return self.model.prepare_video(video_path)

    def chat(self, message: str, history: List[Tuple[str, str]], video_token: str) -> str:
        return self.model.chat(message, history, video_token)


# Настройка глобального логгера: консоль + буфер для UI.
ui_log_sink = UILogSink()
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(ui_log_sink, level="DEBUG")


def _persist_video(temp_video) -> Path:
    """Копирует загруженное видео в рабочую директорию."""
    if temp_video is None:
        raise ValueError("Файл видео не был передан")

    target_dir = Path(tempfile.mkdtemp(prefix="videollama_"))
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / Path(temp_video).name
    shutil.copy(temp_video, target_path)
    logger.info("Видео сохранено во временную папку: %s", target_path)
    return target_path


def handle_video_upload(temp_video, state: SessionState):
    try:
        saved_path = _persist_video(temp_video)
    except Exception as exc:
        logger.exception("Не удалось сохранить видео: %s", exc)
        return gr.update(), ui_log_sink.to_text(), state, state.history

    state.video_path = str(saved_path)
    loader = get_loader()
    token = loader.prepare_video(saved_path)
    state.video_token = token
    state.history = []
    logger.info("Видео %s готово к диалогу", saved_path.name)
    return (
        gr.update(value=str(saved_path)),
        ui_log_sink.to_text(),
        state,
        [],
    )


def handle_chat(message: str, history: List[Tuple[str, str]], state: SessionState):
    if not message:
        logger.debug("Пустое сообщение пользователя, ответ не требуется")
        return "", history, state, ui_log_sink.to_text()

    if not state.video_path or not state.video_token:
        warning = "Сначала загрузите видео слева, чтобы начать диалог."
        logger.warning(warning)
        history = history + [(message, warning)]
        return "", history, state, ui_log_sink.to_text()

    loader = get_loader()
    logger.info("Получен вопрос: %s", message)
    response = loader.chat(message, history, state.video_token)
    history = history + [(message, response)]
    state.history = history
    logger.info("Ответ добавлен в чат")
    return "", history, state, ui_log_sink.to_text()


def reset_chat(state: SessionState):
    logger.info("Чат очищен пользователем")
    state.history = []
    return [], ui_log_sink.to_text()


_loader_instance: Optional[VideoLLaMALoader] = None


def get_loader() -> VideoLLaMALoader:
    global _loader_instance
    if _loader_instance is None:
        logger.info("Инициализация загрузчика Video-LLaMA")
        model_root = os.getenv("VIDEO_LLAMA_MODEL_ROOT")
        if model_root:
            logger.info("Используется путь к модели из VIDEO_LLAMA_MODEL_ROOT: %s", model_root)
        _loader_instance = VideoLLaMALoader(model_root=model_root)
    return _loader_instance


def build_interface():
    logger.debug("Конструирование интерфейса Gradio")
    with gr.Blocks(title="Video-LLaMA чат") as demo:
        app_state = gr.State(SessionState())

        with gr.Row():
            with gr.Column(scale=1, min_width=400):
                gr.Markdown("## Видео")
                video_input = gr.Video(label="Шаг 1. Выберите видео", interactive=True)
                process_btn = gr.Button("Загрузить и обработать", variant="primary")
                video_player = gr.Video(label="Плеер загруженного видео", interactive=False)
                log_output = gr.Textbox(
                    label="Логи", lines=12, interactive=False, value=ui_log_sink.to_text()
                )

            with gr.Column(scale=2, min_width=500):
                gr.Markdown("## Диалог")
                chatbot = gr.Chatbot(label="Ответы Video-LLaMA", height=520)
                user_message = gr.Textbox(label="Ваш вопрос", placeholder="Спросите о содержании видео...")
                with gr.Row():
                    send_btn = gr.Button("Отправить", variant="primary")
                    clear_btn = gr.Button("Очистить чат")

        process_btn.click(
            handle_video_upload,
            inputs=[video_input, app_state],
            outputs=[video_player, log_output, app_state, chatbot],
        )

        send_btn.click(
            handle_chat,
            inputs=[user_message, chatbot, app_state],
            outputs=[user_message, chatbot, app_state, log_output],
        )

        user_message.submit(
            handle_chat,
            inputs=[user_message, chatbot, app_state],
            outputs=[user_message, chatbot, app_state, log_output],
        )

        clear_btn.click(reset_chat, inputs=[app_state], outputs=[chatbot, log_output])

    logger.success("Интерфейс собран")
    return demo


def main():
    logger.info("Запуск интерфейса Video-LLaMA")
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()
