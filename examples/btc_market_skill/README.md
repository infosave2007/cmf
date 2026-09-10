# Qwen3.5-2B: бинарный BTC-навык в CMF

Пример создаёт три артефакта нативного формата CMF:

- `qwen35-2b-q4tp.cmf` — универсальная Qwen3.5-2B в `q4tp`;
- `btc-binary-v2.skill.cmf` — отдельный переносимый skill, привязанный к
  SHA каталога точной базы;
- `qwen35-btc-binary-v2-specialist.cmf` — автономный специалист после
  DTG-MA, validation-отбора и физического defrag; FCD включается только если
  улучшает held-период.

## Полное повторение

Из корня репозитория:

```bash
cargo build --release -p cortiq-cli

cd examples/btc_market_skill
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Фиксированная граница воспроизводит набор статьи.
python btc_skill.py download --end 2026-08-26T00:00:00Z
python btc_skill.py prepare
python btc_skill.py convert
python btc_skill.py bake
python btc_skill.py apply \
  --skill artifacts/btc-binary-v2.skill.cmf \
  --output artifacts/qwen35-btc-binary-v2-applied.cmf
python btc_skill.py evaluate \
  --applied artifacts/qwen35-btc-binary-v2-applied.cmf \
  --baked artifacts/qwen35-btc-binary-v2-specialist.cmf \
  --skill artifacts/btc-binary-v2.skill.cmf \
  --report-dir artifacts/evaluation-binary-v2
python btc_skill.py predict \
  --model artifacts/qwen35-btc-binary-v2-applied.cmf
```

`prepare` строит один строгий target: `UP`, если
`close[t+1] > close[t]`, иначе `DOWN`. Удаления малых движений нет. Каждый
пример видит 48 закрытых часовых свечей, но prompt компактно описывает их
через returns, trend, oscillators, volatility, support/resistance,
candle/volume и последние 12 изменений close.

Фиксированный год даёт:

```text
train       6131  {'DOWN': 3055, 'UP': 3076}
validation  1265  {'DOWN':  664, 'UP':  601}
test        1265  {'DOWN':  623, 'UP':  642}
purge=49 часов
```

Train, validation и test идут строго по времени. `skill bake` получает train
через `--files`, а отдельный validation через `--held-files`. Для выбора
актива или гиперпараметров используется `evaluate --split validation`; test
открывается один раз командой `evaluate --split test` для финального варианта.

## Что делает bake

```bash
cortiq skill bake artifacts/qwen35-2b-q4tp.cmf \
  --files data/market_skill_binary/cmf_corpus.txt \
  --held-files data/market_skill_binary/validation_corpus.txt \
  --focus-tokens DOWN,UP \
  --steps-a 180 --steps-b 0 \
  --lr-b 0.0001 --eval-every 30 \
  --fcd-layers 4 --chunk 256 \
  --held 24 --calib-chunks 1200 \
  --ffn-align 32 \
  --output artifacts/qwen35-btc-binary-v2-specialist.cmf
```

Фаза A обучает DTG-MA-маску FFN и возвращается в лучшее validation-дно.
Короткий FCD-пилот (`steps-b=30`) ухудшил held PPL с 2.058 до 4.459, поэтому
зафиксированный протокол использует `steps-b=0`: сохраняет более сильную
mask-only точку и сразу выполняет defrag. FCD остаётся опциональным, но его
нельзя оставлять только потому, что он звучит эффектно. `--focus-tokens`
оставляет весь prompt контекстом, но считает loss только на однотокенных
ответах `DOWN`/`UP`.

После bake скрипт выполняет:

```bash
cortiq skill export artifacts/qwen35-btc-binary-v2-specialist.cmf \
  --base artifacts/qwen35-2b-q4tp.cmf \
  --id btc-binary-v2 \
  --output artifacts/btc-binary-v2.skill.cmf

cortiq skill apply artifacts/qwen35-2b-q4tp.cmf \
  artifacts/btc-binary-v2.skill.cmf \
  --output artifacts/qwen35-btc-binary-v2-applied.cmf
```

Отдельный `.skill.cmf` не запускается сам: это replacement-тензоры и
архитектурные данные для точной базы. Запечённый specialist запускается без
базы.

## Форматы навыков

Skill overlay больше не ограничен Q8. Экспорт, применение и проверка
replacement-тензоров поддерживают `q8`, `q8_2f`, `q4`, `q4t`, `q4tp`,
`q2tp`, `f16`, `vbit`, `vbit-ro`, `q1`, `q1p`, `q1s` и `q1t`. Кодек каждого
тензора сохраняется в skill-файле; `apply` восстанавливает тот же кодек и
проверяет совместимость с хэшем базы.

## Что публикует evaluate

На одинаковых locked-test примерах считаются:

- majority и знак предыдущего часа;
- чистая q4tp-база и та же база с наложенным `.skill.cmf`;
- совпадение ответов applied-модели и автономного baked-специалиста на 30
  равномерных parity-примерах (основная метрика считается по applied-файлу);
- accuracy, macro-F1, balanced accuracy и confusion matrix;
- paired block-bootstrap 95% CI для разницы accuracy;
- PPL, размер, SHA-256 и время каждого прогона.

Это исследовательский классификатор, не торговая система и не инвестиционная
рекомендация.

## Как выбран актив без подглядывания в test

Файл `experiment_protocol.json` заранее фиксирует два кандидата — BTCUSDT и
ETHUSDT — и одинаковые окно, горизонт, split и bake-параметры. Оба навыка
сравниваются только командой `evaluate --split validation`. Выигрывает актив с
наибольшим приростом accuracy у `q4tp + applied skill` относительно той же
чистой q4tp-базы; macro-F1 и balanced accuracy используются как tie-breakers.
Test выбранного актива открывается один раз после выбора.
