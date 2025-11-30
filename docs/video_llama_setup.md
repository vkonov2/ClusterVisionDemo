# Настройка Video-LLaMA для работы с `video_llama_gradio_app.py`

Приложение больше не использует заглушку: если модель не установлена, загрузка завершится ошибкой. Ниже — пошаговый порядок, как подготовить среду с настоящей Video-LLaMA на macOS или Linux.

## 1. Подготовка окружения
- Установите Python 3.10–3.12 и `ffmpeg` (на macOS: `brew install ffmpeg`).
- Создайте виртуальное окружение и активируйте его.
- Установите зависимости проекта: `pip install -r requirements.txt`.

## 2. Установка зависимостей Video-LLaMA
Video-LLaMA использует CUDA для ускорения. На macOS без GPU расчёты будут очень медленными или невозможными; для полноценных результатов предпочтительна машина с NVIDIA GPU и установленным CUDA.

```bash
# Клонируйте оригинальный репозиторий
git clone https://github.com/DAMO-NLP-SG/Video-LLaMA.git
cd Video-LLaMA

# Установите требования самой модели
pip install -r requirements.txt
```

Если в процессе потребуются дополнительные системные пакеты (например, `ffmpeg` или `xformers`), установите их согласно README репозитория.

## 3. Загрузка весов модели
Скачайте рекомендуемые веса из README проекта (например, `DAMO-NLP-SG/VideoLLaMA-7B`). Часто это делается через `huggingface-cli` или прямую загрузку по ссылкам в разделе «Checkpoints».

```bash
# Пример через huggingface-cli (требуется токен Hugging Face)
huggingface-cli download DAMO-NLP-SG/VideoLLaMA-7B --local-dir ./checkpoints/VideoLLaMA-7B
```

Убедитесь, что внутри каталога с весами есть файлы конфигурации и `.bin`/`.safetensors` весов, как указано в документации Video-LLaMA.

## 4. Подключение весов к приложению
`video_llama_gradio_app.py` автоматически вызывает `load_pretrained_model` без заглушек. Чтобы приложение увидело веса, есть два варианта:

1. **Использовать стандартный идентификатор** — оставить значение по умолчанию `DAMO-NLP-SG/VideoLLaMA-7B`, если веса доступны локально через `huggingface_hub`.
2. **Указать локальный путь** — передать путь к чекпоинту в переменной окружения `VIDEO_LLAMA_MODEL_ROOT` при запуске:

```bash
export VIDEO_LLAMA_MODEL_ROOT=/absolute/path/to/Video-LLaMA/checkpoints/VideoLLaMA-7B
python video_llama_gradio_app.py
```

## 5. Запуск приложения
После выполнения шагов выше запустите интерфейс:

```bash
python video_llama_gradio_app.py
```

Если модель загружена успешно, в логах появится сообщение «Модель Video-LLaMA загружена». При отсутствии зависимостей или весов запуск завершится исключением с подсказкой, что нужно установить модель.
