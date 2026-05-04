# Пакетное редактирование изображений с SDXL-InstructPix2Pix
 
Редактирование изображений по текстовому промпту с помощью модели [diffusers/sdxl-instructpix2pix-768](https://huggingface.co/diffusers/sdxl-instructpix2pix-768). Результаты сохраняются в JPEG с JSON-индексом.
 
## Структура проекта
 
```
.
├── Dockerfile
├── requirements.txt
├── run_edit.py              # основной скрипт
├── resolution_changer.ipynb # предобработка: приведение изображений к 768×768
├── photos/                  # входные изображения (jpg/jpeg/png)
├── photos_edited/           # результаты (создаётся автоматически)
└── results.json             # метаданные результатов (создаётся автоматически)
```
 
## Требования
 
- Docker с поддержкой NVIDIA GPU (`nvidia-container-toolkit`)
- Входные изображения в папке `photos/`
> **Примечание:** по умолчанию модель работает на CPU (`pipe.to("cpu")`) из-за ограничений видеопамяти. Для запуска на GPU нужно не менее ~10–12 ГБ VRAM — замените `pipe.to("cpu")` на `pipe.to("cuda")` в `run_edit.py`.
 
## Подготовка изображений (опционально)
 
Если изображения не имеют разрешения 768×768, предварительно запустите `resolution_changer.ipynb`. Скрипт масштабирует и центрирует все `.jpg` файлы в текущей директории до 768×768 с чёрным фоном.
 
## Сборка и запуск
 
```bash
docker build -t img2img .
 
# Linux / macOS
docker run --gpus all -v $(pwd)/models:/app/models -v $(pwd):/app img2img
 
# Windows (cmd)
docker run --gpus all -v %cd%/models:/app/models -v %cd%:/app img2img
```
 
При первом запуске веса модели скачиваются в `./models` (~6 ГБ). При последующих запусках используется кэш.
 
## Результаты
 
Каждое обработанное изображение сохраняется в `photos_edited/` под именем `res_<uuid8>.jpg`.  
По завершении создаётся `results.json` с записью для каждого изображения:
 
```json
[
    {
        "id_исходного_изображения": "z1.jpg",
        "промпт_редактирования": "Increase color saturation and brightness moderately without overexposure. Add a small yellow five-pointed star in the bottom-left corner with a slight margin from the edges.",
        "id_финального_изображения": "res_b9a341fa"
    }
]
```
 
## Основные параметры (`run_edit.py`)
 
| Параметр | Значение | Описание |
|---|---|---|
| `num_inference_steps` | 20 | Количество шагов денойзинга; увеличить для качества |
| `image_guidance_scale` | 1.5 | Близость результата к исходному изображению |
| `guidance_scale` | 7.5 | Близость результата к текстовому промпту |
| `torch.set_num_threads` | 24 | Установить равным числу потоков вашего CPU |