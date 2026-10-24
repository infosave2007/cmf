# MiniMax-M3.1-Flash на новом поде — быстрый разворот

Инструкция **не содержит ключей, токенов и паролей** — только имена
переменных окружения, значения которых оператор берёт сам из своих
аккаунтов (GitHub SSH, HF access tokens). Значения нигде не записывайте
в файлы репозитория.

## 1. Требования к поду

- GPU NVIDIA (проверялось на RTX PRO 6000, 97 ГБ видеопамяти; меньше
  ~24 ГБ под q2tp-mix не подойдёт — веса моделей и банк экспертов);
- диск ≥ 450 ГБ свободно (модели + драфт + target сборки);
- Ubuntu 22.04+, драйверы NVIDIA установлены;
- сеть с доступом к `huggingface.co` и `github.com`.

## 2. Инструменты

```bash
# Rust (stable, тот же тулчейн, что в CI)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
rustup toolchain install stable --profile minimal --component clippy

# HF CLI
pip install -U "huggingface_hub[cli]"
```

Авторизация HF: задайте в окружении сессии переменную `HF_TOKEN`,
токен берётся в вебе (HF → Settings → Access tokens, write-права на
организацию `infosave`). Значение не сохраняйте в файлах и не вставляйте
в команды-однострочники, которые потом попадают в историю/логи.

## 3. Код и сборка

```bash
git clone https://github.com/infosave2007/cmf
cd cmf
git checkout v0.8.33              # или более новый тег
cargo build --release
./target/release/cortiq --version # ожидается: cortiq 0.8.33
```

## 4. Модели и DSpark-sidecar'ы

Основа и оба sidecar'а лежат в репозитории HF `infosave/MiniMax-M3.1-Flash-cmf`:

```bash
hf download infosave/MiniMax-M3.1-Flash-cmf --local-dir models \
  --include "*.cmf" "*.sha256" "*.json"
```

Проверка целостности sidecar'ов (sha256 сняты при генерации):

```
65e5e321b1b65ed11cd4a91d7107930a2723a490d7e7ec49fca67874cf724a1f  minimax-m3.1-flash-q2tp-mix.dspark.cmf
0a587b6124d67c0458901a6047a138e16b40757724161e053086d0be6062a682  minimax-m3.1-flash-q4tp.dspark.cmf
```

```bash
cd models && shasum -a 256 -c <<< "65e5e321b1b65ed11cd4a91d7107930a2723a490d7e7ec49fca67874cf724a1f  minimax-m3.1-flash-q2tp-mix.dspark.cmf
0a587b6124d67c0458901a6047a138e16b40757724161e053086d0be6062a682  minimax-m3.1-flash-q4tp.dspark.cmf"
```

## 5. Smoke-тест (корректность)

```bash
CMF_MINIMAX_DYNAMIC_TAIL=1 ./target/release/cortiq run \
  models/minimax-m3.1-flash-q2tp-mix.cmf \
  --prompt "What is the capital of France? Answer in one short sentence." \
  --greedy --max-tokens 30
```

Ожидаемый вывод: `The capital of France is Paris.`

## 6. Бенчмарк скорости

```bash
CMF_THREADS=$(nproc) CMF_MINIMAX_DYNAMIC_TAIL=1 ./target/release/cortiq bench \
  models/minimax-m3.1-flash-q2tp-mix.cmf --tokens 120
```

В JSON-вывод попадают счётчики `m3_*` (пул, cold/fills на токен).
Честные ориентиры с прошлого пода (общий хост, парные замеры):
5.35 tok/s на CPU-пути cold, 7.51 tok/s с cold на устройстве,
87.3 vs 131.9 мс/токен холодной стадии.

## 7. Тесты (опционально, на свежем коде)

```bash
CMF_GPU=wgpu cargo test -p cortiq-engine --features gpu --lib minimax_device
```

**Важно:** без заданного `CMF_GPU` эти тесты *пропускаются* (vacuous pass),
а не проходят — «зелёный» прогон без переменной ничего не доказывает.

## 8. Пересоздание sidecar'ов (если HF-репозиторий недоступен)

`--model` — локальная директория с исходником драфта (config.json +
safetensors), `--output` — **уже существующий** base-CMF (sidecar пишется
рядом как `<stem>.dspark.cmf`, перезапись запрещена). Квант sidecar'а —
`q8_2f` по умолчанию, допустимы также `q4tp` и `f16`.

```bash
hf download MiniMaxAI/MiniMax-M3.1-Flash --include "dspark/*" --local-dir mm-src

./target/release/cortiq convert \
  --model mm-src/dspark \
  --output models/minimax-m3.1-flash-q2tp-mix.cmf \
  --dspark-sidecar

./target/release/cortiq convert \
  --model mm-src/dspark \
  --output models/minimax-m3.1-flash-q4tp.cmf \
  --dspark-sidecar
```

После пересоздания сверить sha256 с разделом 4 — должны совпасть.

## 9. Окружение (сводка)

| Переменная | Назначение |
|---|---|
| `HF_TOKEN` | авторизация HF (значение берётся у оператора, в файлы не пишется) |
| `CMF_THREADS` | число потоков CPU (`$(nproc)`) |
| `CMF_MINIMAX_DYNAMIC_TAIL=1` | пул экспертов в токен-графе (нужен для корректных цифр) |
| `CMF_GPU=wgpu` | включает GPU-тесты в `cargo test` |
| `CMF_M3_PROFILE_SAVE` / `CMF_M3_PROFILE` / `CMF_M3_PREWARM_N` | профиль/прогрев банка (замерено: профит 0, выключено по умолчанию) |

## 10. Чего здесь нет

- Паролей, токенов, SSH-ключей и IP-адресов с доступом — по политике
  релиза их в документацию класть нельзя.
- Производительных обещаний: все цифры выше — измеренные на конкретном
  поде, честные; своя первая прогонка на новом поде пусть тоже будет
  парной (A/B), а не одиночной.