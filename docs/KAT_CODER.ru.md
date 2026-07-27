# Запуск 35B MoE-кодера из GGUF на GPU — по шагам

*[English version](KAT_CODER.md)*

Этот гайд проводит **KAT-Coder-V2.5-Dev** (Kwaipilot, = Qwen3.6-35B-A3B:
40 слоёв, из них 30 — линейное внимание GatedDeltaNet, 256 роутируемых
экспертов top-8 плюс постоянный shared-эксперт) от публичного GGUF на
Hugging Face до GPU-декода — каждая команда в том виде, в каком реально
выполнялась. Один и тот же `.cmf`-файл и одни и те же команды работают на
обоих бэкендах; различается только шаг 4.

Проверено end-to-end:

| | железо | декод (steady) |
|---|---|---|
| Vulkan | RTX 5090 32 ГБ, Ubuntu 24.04 | **32.8 tok/s** (CPU той же машины: 14.4) |
| CPU | 32-ядерный EPYC-класс | 14.4–16.6 tok/s (llama.cpp на том же файле: 4.7) |
| Metal | Apple M4, 24 ГБ | ~7 tok/s (арбитраж пробы, см. §4b) |

## 0. Что понадобится

- **Диск**: ~41 ГБ свободных на время конвертации — 21.4 ГБ GGUF +
  19.6 ГБ `.cmf` (GGUF после импорта можно удалить).
- **RAM**: 24 ГБ достаточно и для импорта (GGUF мапится в память, целиком
  не читается), и для CPU-инференса (декод стримит ~600 МБ весов экспертов
  на токен из page cache).
- **Путь Vulkan**: дискретный GPU. *Полный* граф модели хочет карту на
  32 ГБ (17.4 ГБ весов экспертов + KV-зеркало + веса внимания ≈ 31 ГБ
  резидентно); на меньших картах VRAM-бюджет откажет тому, что не влезает,
  и эти слои честно посчитает CPU — деградация плавная, не поломка.
- **Путь Metal**: Apple silicon с 24+ ГБ unified memory.

## 1. Установить cortiq

Проще всего — через crates.io (Rust ставится одной командой с
[rustup.rs](https://rustup.rs)):

```sh
# macOS — Metal вкомпилен всегда, ничего включать не нужно
cargo install cortiq-cli

# Linux / Windows — добавьте wgpu-бэкенд (Vulkan / DX12) для шага 4a
cargo install cortiq-cli --features gpu
```

Уже стоит старая версия? Добавьте `--force` для обновления на месте.

Альтернатива — релизные бинарники, они идут сразу с обоими GPU-бэкендами:

```sh
# Linux x86_64
curl -LO https://github.com/infosave2007/cmf/releases/latest/download/cortiq-x86_64-unknown-linux-gnu.tar.gz
tar xzf cortiq-x86_64-unknown-linux-gnu.tar.gz && sudo mv cortiq /usr/local/bin/

# macOS Apple silicon
curl -LO https://github.com/infosave2007/cmf/releases/latest/download/cortiq-aarch64-apple-darwin.tar.gz
tar xzf cortiq-aarch64-apple-darwin.tar.gz && sudo mv cortiq /usr/local/bin/
```

Либо сборка из чекаута исходников (если хочется править сам cortiq):

```sh
git clone https://github.com/infosave2007/cmf && cd cmf
cargo build --release -p cortiq-cli --features gpu   # gpu = wgpu (Vulkan/DX12); Metal на macOS встроен всегда
```

Проверка: `cortiq --version` → 0.5.25 или новее.

## 2. Конвертация: GGUF → .cmf (одинаково на обеих платформах)

Одна команда — cortiq сам скачает GGUF с Hugging Face и импортирует его
на чистом Rust (никакого Python и llama.cpp):

```sh
cortiq import-gguf bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF/Kwaipilot_KAT-Coder-V2.5-Dev-Q4_K_M.gguf \
    --output kat-q4t.cmf --quant q4t
```

Если удобнее качать самому (докачка, зеркала) — возьмите файл
`Kwaipilot_KAT-Coder-V2.5-Dev-Q4_K_M.gguf` (21.4 ГБ) из
[bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF](https://huggingface.co/bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF)
и передайте локальный путь первым аргументом.

Импорт занимает ~7 минут на 32-ядерной машине (в основном переквантование).
Что происходит под капотом: расписание слоёв (GDN vs полное внимание)
выводится из наличия тензоров, 3-мерные тензоры экспертов режутся на
пер-экспертные матрицы, и откатывается каждая конвенция хранения llama.cpp —
вшитая `+1` в весах RMS-норм (кроме GDN gated norm, она хранится как есть),
`ssm_a` в виде −exp(A_log), тайловый порядок V-голов на всех V-индексных
тензорах включая столбцы out_proj, слитая QKV-раскладка. На выходе —
стандартный `.cmf` на 19.6 ГБ со встроенными токенизатором и чат-шаблоном:
один самодостаточный файл.

## 3. Проверка на CPU (одинаково на обеих платформах)

```sh
cortiq run kat-q4t.cmf --prompt "Write a Python function that checks if a number is prime." --max-tokens 120
cortiq bench kat-q4t.cmf
```

Должен получиться связный код с рассуждением. Ожидаемый декод: ~16 tok/s
на 32-ядерном серверном CPU, ~7 tok/s на M4. Если выход — мусор, значит
сломался импорт: повторите шаг 2 и посмотрите его лог.

## 4a. Vulkan (Linux, дискретный GPU)

Headless-образы серверов (PyTorch-контейнеры RunPod/vast и т.п.) —
compute-only, в них нет GL-библиотек вендора, на которые линкуется
NVIDIA-драйвер Vulkan. Чинится один раз:

```sh
apt-get update && apt-get install -y vulkan-tools libglvnd0 libegl1 libgl1 libglx0
vulkaninfo --summary | grep deviceName   # должно напечатать ваш GPU, а не ошибку
```

Запуск — бюджет весов под вашу карту (показана карта на 32 ГБ; дефолт —
консервативные 8 ГБ):

```sh
CMF_GPU=1 CMF_GPU_VRAM_MB=26000 cortiq run kat-q4t.cmf \
    --prompt "Write a Python function that checks if a number is prime." --max-tokens 200

CMF_GPU=1 CMF_GPU_VRAM_MB=26000 cortiq bench kat-q4t.cmf
```

Что происходит: все 40 слоёв декодируются **одним GPU-submit'ом на токен** —
включая GDN-рекуррентность, внимание, MoE-роутер, выбор top-k экспертов и
все выбранные эксперты. Первый токен платит одноразовую заливку экспертов
(~20 с при тёплом page cache; несколько минут, если файл модели на холодном
сетевом диске) — держите процесс живым (`serve`, интерактивный `run`), а не
перезапускайте на каждый промпт.

Ожидание на RTX 5090: **32.8 tok/s steady** против 14.4 CPU-only на той же
машине. `RUST_LOG=cortiq_engine=info` печатает
`wgpu GPU path: on (NVIDIA ... / Vulkan, discrete, weight budget ...)` и
вердикты пробы. `CMF_GPU_WGPU_GRAPH=0` возвращает per-op offload,
`CMF_GPU=0` — чистый CPU.

## 4b. Metal (macOS Apple silicon)

Тот же файл, та же команда — никаких флагов кроме `CMF_GPU=1`:

```sh
CMF_GPU=1 cortiq run kat-q4t.cmf \
    --prompt "Write a Python function that checks if a number is prime." --max-tokens 200
```

Честные ожидания: MoE-граф целого токена — сегодня фича Vulkan/DX12; на
Metal рантайм-проба арбитрирует per-op GPU против CPU и оставляет
победителя. На M4 с 24 ГБ это ≈ скорость CPU (~7 tok/s): модель на 19.6 ГБ
почти насыщает пропускную способность памяти 24-гигабайтной машины, и проба
корректно отказывает проигрывающим оффлоадам. Dense- и q1-модели получают
полный Metal-граф (27B q1 декодит 11–12 tok/s на том же M4); порт MoE-графа
на Metal — в роадмапе.

## 5. Опциональные рычаги (обе платформы)

```sh
CMF_MOE_TAU=0.9  cortiq run kat-q4t.cmf ...   # адаптивный по уверенности роутинг: ~+12% к декоду
                                              # при perplexity не хуже штатного top-8 (CPU-путь)
CMF_MOE_TOPK=4   cortiq run kat-q4t.cmf ...   # фиксированный меньший top-k (быстрее, лёгкая цена качества)
cortiq run kat-q4t.cmf --o1 all ...           # O(1)-внимание по контексту: KV+state на ctx 4K
                                              # падает 238 → 83 МБ; на 32K+ обязательно
```

Диагностика: `CMF_DEBUG_LAYERS=1` — трасса rms/max скрытого состояния по
слоям; `CMF_GRAPH_PROF=1` — тайминги графа на токен (сборка / энкодинг /
submit+readback).

## Если что-то не так

| симптом | причина / фикс |
|---|---|
| `vulkaninfo`: `ERROR_INCOMPATIBLE_DRIVER` | нет GL-библиотек вендора — строка `apt-get install` из §4a |
| GPU-запуск ≈ скорости CPU на Vulkan | бюджет не вместил экспертов — поднимите `CMF_GPU_VRAM_MB`; смотрите `RUST_LOG=cortiq_engine=info` |
| очень медленный первый токен | одноразовая заливка экспертов с холодного диска; с page cache следующий старт ~20 с |
| `unknown quant 'q4t'` | cortiq старше 0.5.24 — обновитесь |
| бессвязный выход | сломанный импорт — повторите шаг 2 с cortiq ≥ 0.5.24 (раньше импортёра qwen35moe не было) |
