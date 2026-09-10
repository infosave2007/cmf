# Как превратить Qwen3.5-2B в компактного BTC-специалиста: Q4TP + навык CMF

> **Статус: исследовательский черновик, не публиковать.** Locked-test прогон
> от 26 августа 2026 года не подтвердил прирост: BF16 FCD-пилот на 400
> равномерных BTC test-окнах дал `47,75% → 47,25%` (−0,50 п.п.; paired
> block-bootstrap 95% CI пересекает ноль). Отдельный `q8_2f` skill размером
> 37,8 MB корректно экспортируется и применяется к `q4tp`, но этот конкретный
> checkpoint имеет отрицательный quality verdict. Таблица ниже намеренно
> оставлена незаполненной: красивого publishable-результата пока нет.

> Это воспроизводимый технический эксперимент, а не торговая стратегия и не
> инвестиционная рекомендация. Мы измеряем только знак следующей часовой
> свечи. Комиссии, проскальзывание и исполнение ордеров здесь не моделируются.

Универсальная языковая модель умеет рассуждать о рынке, но из коробки не знает,
как именно мы кодируем 48 часовых свечей и какой ответ считаем правильным.
Обычно узкую специализацию представляют как ещё один довесок к модели. В CMF
можно сделать иначе:

1. оставить универсальную Qwen3.5-2B в компактном формате `q4tp`;
2. обучить нативный skill как набор replacement-тензоров;
3. передавать skill отдельным файлом и прикладывать к точной базе;
4. или запечь его в автономный CMF и физически удалить ненужные FFN-нейроны.

В статье мы пройдём весь путь на годе BTCUSDT: от Binance API до текущего
прогноза одной командой. Итог проверим на более позднем периоде, который не
участвовал ни в обучении маски, ни в выборе FCD-checkpoint.

## Что именно мы строим

Задача намеренно короткая:

```text
Вход:  48 закрытых часовых свечей BTCUSDT
Цель:  UP, если close[t+1] > close[t], иначе DOWN
Выход: одно слово — UP или DOWN
```

FLAT-класса нет. Малые движения не удаляются, поэтому accuracy считается на
каждом доступном часе. Это чуть сложнее, зато число нельзя улучшить простым
фильтром «будем отвечать только в удобных режимах».

Основа — [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B),
постобученная модель под Apache-2.0, которую сама карточка рекомендует в том
числе для task-specific fine-tuning и исследовательских задач. На выходе
получаются три разных файла:

```text
qwen35-2b-q4tp.cmf                         универсальная база
          │
          ├── btc-binary-v2.skill.cmf      переносимый навык
          │           + точная база
          │           └── skill apply ──►  применённый специалист
          │
          └── skill bake ───────────────►  автономный defrag-специалист
```

Отдельный `.skill.cmf` не является маленькой моделью. В нём лежат
replacement-тензоры, описание изменившейся архитектуры и хэш каталога базы.
`skill apply` отвергнет чужую или переквантованную базу. Автономный специалист,
наоборот, запускается сам.

## Что такое skill в CMF

Здесь важно не смешивать три сущности.

- **База** — обычный `.cmf` с Qwen3.5-2B.
- **Skill overlay** — тензоры, которые при активации читаются вместо тензоров
  базы. Они не складываются с весами в рантайме.
- **Запечённый специалист** — новый `.cmf`, в котором skill уже применён, а
  мёртвые строки `gate_proj`/`up_proj` и столбцы `down_proj` удалены физически.

Нативный `skill bake` состоит из трёх фаз:

1. **DTG-MA.** Для FFN-нейронов обучается L1-регуляризованная маска. На
   validation движок ищет «дно денойзинга»: точку, где шум уже убран, а
   полезные нейроны ещё не потеряны.
2. **FCD (опционально).** Последние FFN можно полировать под жёсткой маской.
   Сохраняется лучший validation-checkpoint, а не последний шаг. В нашем
   опыте validation отверг эту фазу, поэтому финальный рецепт задаёт
   `steps-b=0`.
3. **Defrag.** Выжившие нейроны собираются в плотные матрицы меньшей формы.
   Они не занимают место и не вычисляются как нули.

Skill overlay поддерживает не только Q8. В текущем CMF replacement-тензоры
можно хранить и применять в `q8`, `q8_2f`, `q4`, `q4t`, `q4tp`, `q2tp`,
`f16`, `vbit`, `vbit-ro`, `q1`, `q1p`, `q1s` и `q1t`. Кодек записан у каждого
тензора и сохраняется при `export`/`apply`.

## Подготовка окружения

Нужны Rust toolchain, Python 3.10+ и свободное место для исходных весов,
кэша и нескольких CMF-файлов. Python используется только для данных и
измерений; DTG-MA, FCD, defrag, export и apply выполняет `cortiq`.

```bash
git clone https://github.com/infosave2007/cmf.git
cd cmf
cargo build --release -p cortiq-cli

cd examples/btc_market_skill
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

В `requirements.txt` пять прямых зависимостей:

```text
numpy>=2.0
pandas>=2.2
safetensors>=0.5
torch>=2.6
transformers>=5.0
```

Проверим CLI:

```bash
../../target/release/cortiq --version
../../target/release/cortiq skill bake --help
```

## Шаг 1. Скачиваем ровно год закрытых свечей

Чтобы получить те же границы, фиксируем конец периода:

```bash
python btc_skill.py download --end 2026-08-26T00:00:00Z
```

Скрипт использует официальный
[`/api/v3/klines`](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market)
Binance Spot через market-data-only endpoint, проходит пагинацию по 1000
свечей и забирает 8760 целевых часов плюс 240 warm-up часов для EMA200 и
rolling-признаков.

Перед сохранением проверяются:

- дубли timestamp;
- шаг ровно один час;
- положительные OHLC;
- `high >= max(open, close)` и `low <= min(open, close)`;
- только свечи, которые закрылись до правой границы.

Получатся два файла:

```text
data/btcusdt_1h.csv
data/btcusdt_1h.csv.manifest.json
```

Manifest содержит источник, временные границы, число строк и SHA-256 CSV. Для
получения последних доступных закрытых свечей параметр `--end` не нужен.

## Шаг 2. Строим причинные признаки

```bash
python btc_skill.py prepare
```

На каждом времени `t` используются только строки `t-47 ... t`. Будущий
`close[t+1]` появляется исключительно в label.

Абсолютная цена почти не нужна. Скрипт переводит рынок в относительные
величины:

- returns за 1, 3, 6, 12, 24 и 48 часов;
- EMA 8/21, EMA 21/55, close/EMA200, MACD и ROC10;
- RSI14 и CCI20;
- ATR14, историческая волатильность и положение в Bollinger bands;
- положение между support/resistance за 48 часов;
- тело, диапазон и тени последней свечи;
- z-score объёма и доля taker-buy;
- известные на момент прогноза UTC hour и weekday.

Prompt получается компактным — одна строка сводки вместо 48 сырых строк
примерно на тысячу токенов. Это идея из старого
`colab_dtgma_trading_skill_1.7b.ipynb`, которая оказалась важнее размера
исходной модели.

Пример:

```text
BTCUSDT 1h causal summary of 48 closed bars. r1=-21 r3=+34 r6=+76
r12=-18 r24=+102 r48=-55 ema8_21=+8 ema21_55=-3 ema200=+112
macd=+6 roc10=+71 rsi14=54 cci20=+71 bb=+0.2 sr=0.6 atr14=42
hv20=180 volz=+0.4 buy=0.5 body=-8 upper=12 lower=4 range=24
hour=13 weekday=2. Next close direction:
```

### Временной split

После подготовки фиксированный год даёт:

```text
train       6131  {'DOWN': 3055, 'UP': 3076}
validation  1265  {'DOWN':  664, 'UP':  601}
test        1265  {'DOWN':  623, 'UP':  642}
```

Разбиение — 70/15/15 строго по времени. Между частями выбрасывается 49 часов:

```text
48 часов входного окна + 1 час target = 49 часов purge
```

Это не случайный split. Соседние скользящие окна почти полностью совпадают;
случайное перемешивание разнесло бы одни и те же свечи между train и test.

### Как выбрать BTC или ETH и не подогнать результат

До генерационной оценки мы фиксируем в `experiment_protocol.json` только два
кандидата: BTCUSDT и ETHUSDT. Для них одинаковы год данных, окно 48, горизонт
один час, признаки, q4tp-база и параметры bake. Сначала выполняется только:

```bash
python btc_skill.py evaluate --split validation ...
```

Критерий выбора тоже задан заранее: максимальный прирост accuracy именно у
`q4tp base + applied .skill.cmf` относительно чистой q4tp-базы. При равенстве
смотрим macro-F1, затем balanced accuracy. Test-период победителя открывается
один раз после выбора. Это не доказывает универсальность навыка для любого
рынка, но не позволяет выбирать актив по красивой test-цифре.

Команда создаёт:

```text
train.jsonl
validation.jsonl
test.jsonl
cmf_corpus.txt
validation_corpus.txt
test_corpus.txt
dataset_manifest.json
```

## Шаг 3. Почему loss считается только на ответе

Даже компактный prompt значительно длиннее одного слова `UP`/`DOWN`. Если
учить обычный language-model loss на всех позициях, основная часть градиента
пойдёт на воспроизведение чисел, тегов ChatML и русского текста.

Поэтому используется:

```text
--focus-tokens DOWN,UP
```

Весь prompt проходит через модель как контекст, но cross-entropy считается
только там, где следующим токеном является одна из двух меток. CLI проверяет,
что каждая метка кодируется ровно одним токеном встроенного токенизатора.

Есть ещё одна важная деталь. Train и validation передаются разными флагами:

```text
--files       train corpus
--held-files  validation corpus
```

Раньше первые chunks train-корпуса одновременно служили quality gate. Для
обычного доменного текста это терпимо, для supervised-классификации — нет.
Теперь mask/FCD checkpoints выбираются на другом временном периоде, а test
вообще не читается во время bake.

## Шаг 4. Конвертируем Qwen3.5-2B в Q4TP

```bash
python btc_skill.py convert
```

Внутри выполняется:

```bash
cortiq convert \
  --model Qwen/Qwen3.5-2B \
  --quant q4tp \
  --output artifacts/qwen35-2b-q4tp.cmf

cortiq verify artifacts/qwen35-2b-q4tp.cmf
cortiq info artifacts/qwen35-2b-q4tp.cmf
```

В нашем прогоне база занимает `1 035 327 278` байт. `verify` пересчитывает
хэши записей контейнера; это стоит делать после любого копирования или
запекания.

## Шаг 5. Запекаем навык

Пользовательская команда:

```bash
python btc_skill.py bake
```

Полная команда, которую запускает скрипт:

```bash
cortiq skill bake artifacts/qwen35-2b-q4tp.cmf \
  --files data/market_skill_binary/cmf_corpus.txt \
  --held-files data/market_skill_binary/validation_corpus.txt \
  --output artifacts/qwen35-btc-binary-v2-specialist.cmf \
  --steps-a 180 \
  --steps-b 0 \
  --lr-a 0.1 \
  --lr-b 0.0001 \
  --eval-every 30 \
  --fcd-layers 4 \
  --chunk 256 \
  --held 24 \
  --calib-chunks 1200 \
  --focus-tokens DOWN,UP \
  --ffn-align 32
```

В старом блокноте последние четыре слоя проходили полноценную эпоху с
learning rate `5e-4`. Для q4tp мы проверили более умеренный FCD с `1e-4`,
однако уже первый checkpoint ухудшил held PPL с 2.058 до 4.459. Измерение
сделано до генерационной оценки и открытия test. Поэтому финальный рецепт
честно оставляет `steps-b=0`: DTG-MA-маска сразу переходит в defrag, а
заведомо отвергнутые FCD-шаги не тратят время читателя. FCD не обязан
улучшать каждый навык — именно для этого и нужна отдельная validation.

`--ffn-align 32` округляет ширину каждого слоя вверх до кратной 32. Это может
вернуть несколько пограничных нейронов, зато групповые кодеки и SIMD-пути не
переходят на медленный хвост.

Скрипт сохраняет полный stdout в `artifacts/bake-binary-v2.log`, а сводку с
командой, PPL, временем, размерами и SHA-256 — в
`artifacts/bake-binary-v2.json`.

## Шаг 6. Экспортируем отдельный skill

После bake скрипт автоматически выполняет:

```bash
cortiq skill export \
  artifacts/qwen35-btc-binary-v2-specialist.cmf \
  --base artifacts/qwen35-2b-q4tp.cmf \
  --id btc-binary \
  --name "BTC hourly UP DOWN classifier" \
  --output artifacts/btc-binary-v2.skill.cmf
```

Посмотреть содержимое:

```bash
cortiq skill list artifacts/btc-binary-v2.skill.cmf
```

Самостоятельный skill удобно публиковать отдельно: читатель скачивает одну
общую q4tp-базу и прикладывает к ней нужных специалистов.

## Шаг 7. Применяем skill к базе

```bash
python btc_skill.py apply \
  --skill artifacts/btc-binary-v2.skill.cmf \
  --output artifacts/qwen35-btc-binary-v2-applied.cmf
```

Эквивалентная команда:

```bash
cortiq skill apply \
  artifacts/qwen35-2b-q4tp.cmf \
  artifacts/btc-binary-v2.skill.cmf \
  --output artifacts/qwen35-btc-binary-v2-applied.cmf
```

После применения запускается `verify`. Для полной проверки можно прогнать
30 одинаковых равномерных test-prompts через baked и applied файлы: их ответы
должны совпадать. Основная accuracy считается именно по applied-файлу на всей
зафиксированной выборке; baked здесь только проверка эквивалентности, поэтому
нет смысла удваивать длинный прогон.

## Шаг 8. Честное сравнение на locked test

```bash
python btc_skill.py evaluate \
  --applied artifacts/qwen35-btc-binary-v2-applied.cmf \
  --baked artifacts/qwen35-btc-binary-v2-specialist.cmf \
  --skill artifacts/btc-binary-v2.skill.cmf \
  --split test \
  --report-dir artifacts/evaluation-binary-v2
```

По умолчанию берутся равномерно расположенные test-примеры. Для каждого
используется один и тот же prompt и greedy decoding. Сравниваются четыре линии:

- majority class из train;
- знак предыдущего часа;
- исходная Qwen3.5-2B q4tp;
- исходная q4tp-база после наложения отдельного `.skill.cmf`;
- автономный baked-специалист на 30 parity-примерах как проверка
  эквивалентности.

Публикуются не только accuracy:

- macro-F1 и balanced accuracy;
- confusion matrix и доля каждого предсказанного класса;
- число невалидных ответов;
- paired block-bootstrap 95% CI разницы skill − base;
- focused/test PPL;
- размер, SHA-256 и wall time.

### Измеренный результат

| Модель | Accuracy | Macro-F1 | Balanced accuracy | DOWN/UP predictions |
|---|---:|---:|---:|---:|
| Majority | {{MAJ_ACC}} | {{MAJ_F1}} | {{MAJ_BAL}} | {{MAJ_DIST}} |
| Previous-hour sign | {{MOM_ACC}} | {{MOM_F1}} | {{MOM_BAL}} | {{MOM_DIST}} |
| Qwen3.5-2B q4tp | {{BASE_ACC}} | {{BASE_F1}} | {{BASE_BAL}} | {{BASE_DIST}} |
| q4tp + CMF skill | {{SKILL_ACC}} | {{SKILL_F1}} | {{SKILL_BAL}} | {{SKILL_DIST}} |

```text
Разница accuracy skill − base: {{ACC_DELTA_PP}} п.п.
Paired block-bootstrap 95% CI: {{ACC_CI}}
Focused validation PPL: {{VAL_PPL}}
Размер автономного CMF: {{SIZE_RESULT}}
```

{{RESULT_INTERPRETATION}}

Здесь принципиально нельзя заменить accuracy величиной PPL. PPL может показать,
что модель лучше объясняет ответы, но бинарный PPL около 2 также получается у
модели, которая всегда даёт вероятности 50/50. Поэтому финальный verdict
опирается на confusion matrix и locked-test predictions.

## Шаг 9. Получаем текущий прогноз

```bash
python btc_skill.py predict \
  --model artifacts/qwen35-btc-binary-v2-applied.cmf
```

Команда скачивает достаточно последних закрытых свечей для EMA200, строит то же
48-часовое окно, поднимает локальный OpenAI-совместимый endpoint CMF и печатает
JSON:

```json
{
  "symbol": "BTCUSDT",
  "last_closed_candle_utc": "...",
  "close": 0.0,
  "model": ".../qwen35-btc-binary-v2-applied.cmf",
  "predictions": [
    {
      "horizon_hours": 1,
      "scenario": "UP",
      "confidence": 0.71,
      "probabilities": {"DOWN": 0.29, "UP": 0.71},
      "raw": "UP"
    }
  ],
  "warning": "Исследовательский сценарий, не инвестиционная рекомендация."
}
```

Для воспроизводимого исторического запроса можно передать `--end` с UTC-датой.

## Почему старые блокноты показывали 66–76%

Перед новым опытом я рекурсивно проверил 115 старых блокнотов Cortiq. Там
оказались три разные причины больших чисел:

1. **Teacher forcing.** В contrastive-блокноте 76% получались после
   подстановки правильного ответа. Его собственная диагностика показывает
   32–36% без знания label.
2. **Мажоритарный NO_TRADE.** В v18 accuracy 66,4% соответствует ответу
   `NO_TRADE` на всех 256 примерах.
3. **Условное покрытие.** У BTC v20 UP-маска имеет 70,2%, но только на 10,7%
   characterization-набора; это per-mask win rate, не full accuracy.

Кроме того, 22 per-asset блокнота v20 имеют одинаковые сохранённые outputs, а
20 verification-блокнотов — другой общий output; например, сохранённый
`verify_BTC` фактически загружал ETH. Это не обесценивает идеи DTG-MA и
режимных масок, но запрещает цитировать outputs как независимые benchmarks.
Подробный разбор сохранён в `docs/DTGMA_NOTEBOOK_AUDIT.ru.md`.

## Что читатель может поменять

После полного повторения безопасно исследовать по одному фактору:

- включить `--steps-b` и подобрать `lr-b` только по новому validation-периоду;
- увеличить `--held` и `--calib-chunks`;
- сравнить компактный prompt с сырыми 48 строками;
- повторить rolling-origin walk-forward на нескольких годовых отрезках;
- добавить комиссию и отдельный backtest только после фиксации классификатора.

Нельзя выбирать гиперпараметры по test и потом тем же test доказывать
улучшение. Если хочется продолжить исследование после просмотра test, нужен
новый более поздний locked period.

## Итог

Вау-эффект CMF не в обещании угадать рынок. Он в проверяемой упаковке узкой
специализации:

- одна Qwen3.5-2B q4tp служит общей базой;
- skill передаётся отдельным файлом с проверкой идентичности базы;
- тот же skill можно запечь в автономную модель;
- defrag действительно меняет формы FFN и уменьшает хранимую сеть;
- train, validation и test физически разделены;
- одна Python-команда повторяет данные, bake, apply, оценку и текущий прогноз.

А качество рынка остаётся числом, которое нужно заслужить на locked test. Если
доверительный интервал включает ноль или модель схлопнулась в один класс,
правильный вывод — продолжить исследование, а не превращать PPL в рекламную
accuracy.
