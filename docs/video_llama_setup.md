# Настройка Video-LLaMA для работы с `video_llama_gradio_app.py`

Приложение больше не использует заглушку: если модель не установлена, загрузка завершится ошибкой. Ниже — пошаговый порядок, как подготовить среду с настоящей Video-LLaMA на macOS или Linux. Все команды выполняйте **внутри вашего виртуального окружения**, чтобы ничего не ставить в глобальный Python.

## 1. Подготовка окружения
- Установите Python 3.10–3.12 и `ffmpeg` (на macOS: `brew install ffmpeg`).
- Создайте виртуальное окружение и активируйте его.
- Установите зависимости проекта: `pip install -r requirements.txt`.
  Если окружение ещё не создано, пример (macOS/Linux):

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

## 2. Установка зависимостей Video-LLaMA
Video-LLaMA использует CUDA для ускорения. На macOS без GPU расчёты будут очень медленными или невозможными; для полноценных результатов предпочтительна машина с NVIDIA GPU и установленным CUDA.

```bash
# Клонируйте оригинальный репозиторий рядом с проектом (не в глобальный Python)
cd /Users/konov/projects  # пример каталога, где лежит ClusterVisionDemo
git clone https://github.com/DAMO-NLP-SG/Video-LLaMA.git

# Перейдите в репозиторий и установите зависимости МОДЕЛИ в том же venv
cd Video-LLaMA
pip install -r requirements.txt

# Поставьте пакет в editable-режиме, чтобы модуль `videollama` был доступен
pip install -e .
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
2. **Указать локальный путь к весам** — передать путь к чекпоинту в переменной окружения `VIDEO_LLAMA_MODEL_ROOT` при запуске:

```bash
export VIDEO_LLAMA_MODEL_ROOT=/absolute/path/to/Video-LLaMA/checkpoints/VideoLLaMA-7B
python video_llama_gradio_app.py
```

3. **Если модуль не находится в PYTHONPATH** — передайте путь к клонированному репозиторию в `VIDEO_LLAMA_REPO`, чтобы загрузчик добавил его в `sys.path`:

 ```bash
 export VIDEO_LLAMA_REPO=/absolute/path/to/Video-LLaMA
 python video_llama_gradio_app.py
 ```

Полный пример запуска в venv на macOS, если проект в `/Users/konov/projects/ClusterVisionDemo` и репозиторий модели в `/Users/konov/projects/Video-LLaMA`:

```bash
cd /Users/konov/projects/ClusterVisionDemo
source .venv/bin/activate
export VIDEO_LLAMA_MODEL_ROOT=/Users/konov/projects/ClusterVisionDemo/checkpoints/VideoLLaMA-7B
export VIDEO_LLAMA_REPO=/Users/konov/projects/Video-LLaMA
python video_llama_gradio_app.py
```

## 5. Запуск приложения
После выполнения шагов выше запустите интерфейс:

```bash
python video_llama_gradio_app.py
```

Если модель загружена успешно, в логах появится сообщение «Модель Video-LLaMA загружена». При отсутствии зависимостей или весов запуск завершится исключением с подсказкой, что нужно установить модель.
