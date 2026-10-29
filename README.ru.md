English: [README.md](README.md) · 中文: [README.zh.md](README.zh.md)

# CMF — Cortiq Model Format

CMF — проверяемый контейнер моделей: один файл может хранить веса, токенизатор,
метаданные чата, маски задач и накладки навыков для инференса без тяжёлого
ML-фреймворка.

[![CI](https://github.com/infosave2007/cmf/actions/workflows/ci.yml/badge.svg)](https://github.com/infosave2007/cmf/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/cortiq-core.svg)](https://crates.io/crates/cortiq-core)
[![docs.rs](https://img.shields.io/docsrs/cortiq-core)](https://docs.rs/cortiq-core)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)


## Решения на Metal и Vulkan

**Cortiq 0.8.0:** один CMF-файл с навыками, локальные решения по ошибке
реконструкции — без генерации токенов. CPU по умолчанию; GPU включается через
`CORTIQ_DECISION_DEVICE=metal` или `vulkan`. Обычная установка из crates.io
включает GPU-поддержку (Rust 1.88+).

![Полный локальный путь CPU / GPU; сравнения внутри одного стенда](docs/decision/hf/figures/gpu.svg)

[Запуск, свои навыки и API](docs/decision/hf/README_RU.md) ·
[Замеры и ограничения](docs/decision/hf/GPU.md)

## Состояние

CMF v2 — текущий формат на диске. Читатели проверяют конверт, границы секций,
метаданные тензоров и хэши; несовместимые изменения требуют бита возможностей
или смены версии. API Rust-крейтов пока до 1.0 и может меняться.

Проект пригоден для локального инференса и экспериментов с форматом. Цифры
качества и скорости зависят от модели и стенда; методика измерений находится в
тематических руководствах.

## Быстрый старт

Установите CLI и сконвертируйте небольшой открытый чекпоинт:

```sh
cargo install cortiq-cli --locked
cortiq convert --model Qwen/Qwen3-0.6B --quant q8 --output qwen.cmf
cortiq run qwen.cmf --prompt "Столица Франции?" --greedy --no-think
```

Для GGUF:

```sh
cortiq import-gguf model.gguf --output model.cmf
cortiq verify model.cmf
```

## Диффузионные модели по блокам (0.8.40)

[Clover-35B-A3B-Preview](https://huggingface.co/infosave/Clover-35B-A3B-Preview-cmf)
пишет ответ блоками по 32 токена и раскрывает по нескольку токенов за проход
денойзинга, а не по одному за шаг:

```sh
hf download infosave/Clover-35B-A3B-Preview-cmf clover-35b-a3b-preview-q4tp-v2.cmf --local-dir .
cortiq run clover-35b-a3b-preview-q4tp-v2.cmf --prompt "Объясни, как работает хеш-таблица." --no-think
```

На Vulkan каждый проход денойзинга — одна отправка на GPU (34 мс на блок из
32 токенов на RTX 5090: ~26 ток/с на прозе, 80–95 на коде при пороге из
релиза модели). Конвертация, параметры генерации и замеры —
[docs/CLOVER.md](docs/CLOVER.md). В этом же релизе графы всех MoE-моделей на
Vulkan занимают меньше видеопамяти: стек 35B-A3B q4tp — 18.8 ГБ вместо 31.8 ГБ,
а опубликованная Qwen3.6-35B-A3B декодирует 156 ток/с на карте 32 ГБ.

## Сжатие модели под задачу одной командой (0.8.39)

Общая модель превращается в специалиста под ваши промпты:

```sh
cortiq compress model.cmf prompts.jsonl model-task.cmf --ratio 0.9
```

Учитель отвечает на промпты: сама модель, другой `.cmf` или любая модель за
OpenAI-совместимым API (`--teacher openrouter:qwen/qwen3.8-27b`,
`openai:MODEL`, `api:MODEL --api-base URL`). Нейроны FFN, которые задача не
использует, уходят из модели: по засветке активаций или по обученной маске
DTG-MA (`--select dtgma`). Их вклад переносится в оставшиеся решением в
замкнутой форме, затем модель долечивается на ответах. Отложенные ответы
оценивают результат, и ворота качества решают, писать ли файл. У MoE-моделей
уходят эксперты, которых задача не выбирает. Итог остаётся в кодеке
исходника, квантование — отдельный шаг. GPU-стадии работают на NVIDIA (Vulkan)
и на Apple Silicon (Metal). Подробности — в [руководстве](docs/COMPRESS.ru.md).

## Компактация MTP draft по своим промптам (0.8.38)

Для CMF с уже поддерживаемым dense MTP draft одна команда создаёт более узкий
draft, откалиброванный по типичным запросам задачи:

```sh
cortiq compact source.cmf prompts.jsonl task-model.cmf
```

Source используется как teacher; команда создаёт или переиспользует
`task-model.tdata`, сохраняет проверяемый plan и публикует только runnable
CMF. Это физическая compaction MTP-draft, а не general fine-tuning: trunk
остается исходным, а greedy output контролируется verifier-ом. JSONL, safety
gates и проверка на Metal/Vulkan описаны в
[руководстве MTP](docs/MTP_COMPACTION.ru.md).

GGUF трансформера Qwen Image можно импортировать как отдельный CMF;
подробности упаковки приведены в руководстве Qwen Image. Для нативного
редактирования используйте готовый однофайловый bundle или сохраните
компоненты отдельно.

## Нативное редактирование Qwen Image

Сначала соберите bundle из `transformer.cmf`, `text_encoder.cmf`,
`vae.cmf` и `scheduler_config.json`:

```sh
cortiq imagine-pack --bundle qwen-image \
  --out qwen-image/qwen-image-edit-2509-q4tp.cmf
```

Bundle встраивает трансформер, текстовый и зрительный энкодер Qwen2.5-VL,
токенизатор, процессор, конфигурации, VAE и планировщик. Для запуска нужны
только bundle и входное изображение:

```sh
cortiq imagine qwen-image/qwen-image-edit-2509-q4tp.cmf \
  --image docs/media/fox-512.png \
  --prompt "Добавьте яркий синий вязаный шарф, сохранив лису, позу и снежный фон." \
  --height 1024 --width 1024 --steps 40 --cfg 4 --seed 7 \
  --reference-size 1024 --out fox-scarf.png
```

Для гибкой раскладки оставьте компоненты в одном каталоге или укажите их
явно:

```sh
cortiq imagine qwen-image/transformer.cmf \
  --text-encoder qwen-image/text_encoder.cmf \
  --vae qwen-image/vae.cmf \
  --scheduler qwen-image/scheduler_config.json \
  --image docs/media/fox-512.png \
  --prompt "Превратите сцену в акварельную иллюстрацию." \
  --height 1024 --width 1024 --steps 40 --cfg 4 --seed 7 \
  --reference-size 1024 --out watercolor.png
```

Нужна хотя бы одна ссылка `--image`; для нескольких изображений повторите
флаг. Профиль `--reference-size 1024` задаёт корень из площади входного VAE
1024² и не меняет размер результата.
На совместимом Vulkan-устройстве `cortiq imagine` автоматически выбирает
резидентный проход трансформера Qwen: скрытое состояние остаётся на устройстве
между блоками, а в конце каждого прохода выполняется одно чтение обратно.
Пути CPU и Metal остаются доступными, а проверка бюджета памяти выбирает
совместимый путь. Правила получения и упаковки с
фиксированными ревизиями приведены в [руководстве Qwen Image](docs/QWEN_IMAGE.md).

CLI также предоставляет `info`, `bench`, `ppl`, `serve`, `skill`, `moe-mask`,
`moe-defrag`, `requant`, `compact`, `sign`, `imagine`, `animate` и `ltx-video`.
Команда `cortiq <команда> --help` показывает флаги и ограничения.

### Транскрибация Whisper

Cortiq преобразует 128-бинные чекпоинты Whisper из Hugging Face (в том числе
large-v3 и large-v3-turbo) в CMF, а затем транскрибирует WAV без Python:

```sh
cortiq convert --model openai/whisper-large-v3-turbo --quant q4tp \
  --output whisper-large-v3-turbo-q4tp.cmf
cortiq verify whisper-large-v3-turbo-q4tp.cmf
cortiq transcribe whisper-large-v3-turbo-q4tp.cmf interview.wav --language ru
```

Смешанный профиль q4tp оставляет внимание и общую матрицу токенов/выхода
декодера в q8_2f, FFN-матрицы — в q4tp, позиционные embeddings — в f16.
Это не гарантия паритета или качества: перед внедрением измерьте WER на своих
языках и аудио. WAV PCM/float сводится к mono 16 кГц. При наличии используется
Metal/wgpu; `CMF_GPU=0` принудительно включает CPU. По умолчанию сохраняется
полный контекст 30 секунд; экспериментальный `CMF_WHISPER_TRIM_ENCODER=1`
может повлиять на качество распознавания.

### Многоязычный TTS Phonon 40M (0.8.36)

Команды `phonon-pack` и `tts` добавляются в **cortiq-cli 0.8.36**. В уже
опубликованных 0.8.35 и более ранних крейтах этих команд нет. До появления
тега `v0.8.36` и публикации крейта 0.8.36 нельзя считать, что обычный
`cargo install cortiq-cli` их установит. После публикации релиза:

```sh
# Обычная установка включает GPU и мост Phonon.
cargo install cortiq-cli --version 0.8.36 --locked

# Либо переносимая CPU-сборка с сохранением поддержки Phonon.
cargo install cortiq-cli --version 0.8.36 --locked \
  --no-default-features --features phonon
```

После принятия условий исходной модели можно упаковать её локальную копию;
готовый CMF также скачивается через API-команду Hugging Face:

```sh
hf download gradium/phonon-40m-multilingual --local-dir phonon-source
cortiq phonon-pack phonon-source --output phonon-40m-multilingual-q8.cmf

hf download infosave/phonon-40m-multilingual-cmf \
  phonon-40m-multilingual-q8.cmf --local-dir .
cortiq verify phonon-40m-multilingual-q8.cmf
cortiq tts phonon-40m-multilingual-q8.cmf "Bonjour, monde." \
  --lang fr --temperature 0.3 --seed 4242424242424242 --out bonjour.wav
```

CMF побайтно хранит исходные Q8 GGUF, конфигурацию, токенизатор и latent
стандартного голоса; после проверки Cortiq материализует эти записи и вызывает
поддерживаемый Gradium рантайм `ptts`. Это мост, а не заявление о нативной
реализации Phonon. Модель принимает **только** `en`, `fr`, `de`, `es`, `pt`:
русский язык и автоопределение языка не поддерживаются; `--voice-file`
принимает только заранее вычисленный совместимый latent, а не запись голоса.
Исходные веса распространяются на условиях CC-BY-NC-4.0. Подробности и
происхождение — в [исходнике карточки](docs/hf/Phonon-40M-Multilingual-cmf.md);
работающий публичный тест доступен в
[Space](https://huggingface.co/spaces/infosave/phonon-40m-multilingual-cmf-demo).

## Что хранит файл CMF

- Фиксированный 128-байтовый конверт, адресующий все секции.
- JSON-заголовок с архитектурой, квантованием, метаданными чата и источником.
- Бинарный каталог тензоров с dtype, формой, смещениями, длинами и `hash64`.
- Выровненный по страницам блоб весов, который можно отобразить через mmap.
- Необязательные маски задач, накладки навыков, байты токенизатора и sparse-индекс.

`cortiq verify` завершается ошибкой при неверных границах или несовпадении хэша;
`python/cmf_reader.py` — небольшой независимый ридер для инспекции. Нормативная
[спецификация CMF v2](docs/CMF_V2_SPEC.ru.md) описывает раскладку.

## Квантование

Квантование выбирается для каждого тензора: чувствительные тензоры можно
оставить в высокой точности, а большие матрицы упаковать компактнее.

| Кодек | Назначение |
|---|---|
| `f16`, `f32` | нормы, эмбеддинги и точные управляющие тензоры |
| `q8`, `q8_2f` | высокая точность; `q8_2f` добавляет масштабы входных каналов |
| `q4`, `q4t`, `q4tp` | обычные плотные и MoE-веса |
| `q2tp`, `vbit`, `vbit_ro` | ограниченный размер или смешанные разрядности |
| `q1`, `q1t`, `q1s` | обученные бинарные или экспериментальные PTQ-пути |

Экспериментальный низкоразрядный путь описан в [Q1T/PTQ](docs/Q1T_PTQ.md), а
матрица поддержки кодеков — в [обзоре покрытия](docs/QUANT_COVERAGE.ru.md).

## Возможности рантайма

- **CPU:** переносимая реализация на Rust с mmap-весами.
- **GPU:** нативный Metal на macOS и wgpu (Vulkan/DX12), если это поддерживают
  сборка и устройство. Для запроса GPU задайте `CMF_GPU=1`.
- **Длинный контекст:** `--o1` использует состояние фиксированного размера и
  меняет рост памяти на измеряемую потерю качества; проверяйте свою модель.
- **Навыки:** один хребет хранит маски задач и накладки замещающих тензоров,
  не требуя второй полной копии модели для неактивного навыка.
- **MoE:** маска задачи и физическая дефрагментация могут уменьшить набор
  активных экспертов. Перед публикацией проверьте PPL на отложенных данных.
- **Спекуляция:** MTP и draft-пути включаются там, где метаданные и измеренная
  доля принятых токенов делают их полезными.
- **Сервер:** `cortiq serve` предоставляет локальный OpenAI-совместимый HTTP API.
- **Медиа:** `imagine`, `animate` и `ltx-video` запускают подходящие пайплайны
  изображений, видео и аудио.

## Платформы и семейства моделей

| Цель | Поддержка |
|---|---|
| Linux/macOS/Windows | CPU; GPU через доступный бэкенд wgpu |
| Apple Silicon | CPU и Metal; память общая с системой |
| NVIDIA/AMD | Vulkan или DX12 через wgpu там, где путь протестирован |
| Android/iOS | См. мобильный проект и руководство split |

Нативная конвертация охватывает Qwen, Llama, Mistral, Gemma, Phi, DeepSeek,
Kimi Linear, MiniCPM и несколько MoE/video-семейств. Ограничения и опубликованные
файлы собраны в [руководствах моделей](docs/hf/README.md). Для неподдержанного
чекпоинта попробуйте `import-gguf`; при отказе создайте воспроизводимый issue.

## Тематические документы

- [Спецификация CMF v2](docs/CMF_V2_SPEC.ru.md) — нормативная раскладка файла.
- [Сравнение форматов](docs/COMPARISON.ru.md) — критерии и источники измерений.
- [Навыки](docs/SKILLS.ru.md) — маски, накладки и маршрутизация.
- [GPU-рецепты](docs/GPU_KERNEL_RECIPES.md) — измерения бэкендов.
- [Несколько GPU](docs/MULTI_GPU.md) и [разделение на мобильном](docs/MOBILE_SPLIT.ru.md).
- [FCD-восстановление](docs/RUST_FCD.md) и [низкоразрядный PTQ](docs/Q1T_PTQ.md).
- [Компактация MTP FFN](docs/MTP_COMPACTION.ru.md) — независимая калибровка и
  физическое уменьшение FFN у MTP-draft.
- [Qwen Image Edit](docs/QWEN_IMAGE.md) — однофайловое редактирование и
  отдельная раскладка компонентов.
- [Карточки моделей и конвертация](docs/hf/README.md).
- [Cortiq Spectra](docs/SPECTRA.ru.md) — детерминированная потоковая CPU-раскраска
  двухэнергетического рентгена, профили сканера, маска отказов и измеренные
  ограничения.

## Сборка из исходников

```sh
cargo build --release --workspace
cargo build --release -p cortiq-cli --no-default-features  # только CPU
cargo test --workspace
```

Рабочая область распространяется по Apache-2.0. Перед распространением или
сообщением об уязвимости прочтите [LICENSE](LICENSE), [PATENTS.md](PATENTS.md),
[CONTRIBUTING.md](CONTRIBUTING.md) и [SECURITY.md](SECURITY.md). Готовые сборки
и контрольные суммы публикуются на [странице релизов GitHub](https://github.com/infosave2007/cmf/releases).
