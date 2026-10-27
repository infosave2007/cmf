# Независимая компактация FFN у MTP-драфта

`cortiq mtp-compact` уменьшает **только** dense-FFN слои MTP-драфта. Основной
backbone не переупаковывается: в плане явно записано `trunk.mode = preserve`.
Это позволяет отдельно подобрать размер draft-слоёв и затем измерить, окупает
ли уменьшение веса падение acceptance. Команда не обещает ни качество, ни
ускорение: оба свойства должны быть подтверждены нижеописанным тестом.

> MTP здесь — draft для speculative decode, а не основной MoE/trunk. Физическая компактация
> изменяет нейроны draft-FFN и потому может изменить число принятых черновых
> токенов, но не должна менять проверенный greedy-результат основного модели.

## Что поддерживается

Поддерживаются два расположения канонического **плотного** MTP:

1. `model.mtp.*` находится в основном CMF: результатом будет новый основной
   `.cmf`.
2. MTP находится в sidecar `model.mtp.cmf` (типичный MiMo): команда находит
   его автоматически и создаёт **готовую пару** кандидата.

Предпочтительно `--output` задаёт путь нового запускаемого **main**-CMF. Для MiMo
команда сама создаёт рядом соответствующий `<output-stem>.mtp.cmf`: unsharded
backbone сначала hard-link-ится от исходника (без лишней копии), а если
filesystem не поддерживает hard link — копируется. Sharded backbone вместо
этого потоково coalesce-ится в один candidate. Hard link — это неизменяемый для
команды alias тех же байтов: `mtp-compact` не записывает ни исходный путь, ни
alias. Не изменяйте source во время job; и после успешного hard-link варианта
не редактируйте candidate main «на месте» — это тот же inode, что source. Если
нужен независимо изменяемый main, сделайте отдельную copy/reflink уже после
валидации. Поэтому после одной команды получается именно рабочая пара, а не
отдельный sidecar:

```text
base/model.cmf              # исходный backbone, не изменяется
base/model.mtp.cmf          # исходный draft, не изменяется

candidate/model.cmf         # новый runnable main (hard link/copy/coalesce)
candidate/model.mtp.cmf     # новый compact draft
```

Для совместимости с ранними скриптами внешний MiMo-draft можно явно указать
как `--output candidate/model.mtp.cmf`. В этом **legacy sidecar-only** режиме
публикуется только новый sidecar и plan; `candidate/model.cmf` не создаётся и
такой результат сам по себе не является запускаемой моделью. Его нужно вручную
положить рядом с совместимым main-CMF. Для нового или автоматизированного
запуска всегда указывайте main-путь без суффикса `.mtp.cmf`.

Не заменяйте исходник. Команда отказывается перезаписывать существующий main,
sidecar или plan, включая dangling symlink. Sharded **main** поддержан: потоковый
writer читает merged shards через mmap и создаёт один новый, самостоятельно
запускаемый unsharded candidate (`candidate/model.cmf`). Для MiMo это также
работает с sharded main и одним обычным `.mtp.cmf` sidecar: main coalesce-ится,
draft компактируется рядом. Sharded внешний `.mtp.cmf` пока fail-closed
отклоняется — MiMo loader намеренно ищет один sidecar с производным именем.

## Калибровка и план

Подайте финализированный и проверяемый `.tdata`-selector. Для выбора каналов
используйте только train/calibration split; validation/test оставьте для
последующей оценки. Команда сама проверяет main CMF, обнаруженный MTP-sidecar
и `.tdata`. До загрузки runtime preflight разбирает **все** `records.jsonl`/
`tokens.idx` строки (включая невыбранные val/test) и аутентифицирует payload
по size/SHA из manifest. Для невыбранных records не выделяется `Vec<u32>`;
legacy v1 без manifest-descriptor проходит тот же read-fence фиксированным
64-KiB буфером. Выбранные train records затем читаются обычным streaming-
проходом для калибровки.

После этого preflight `.tdata` должен оставаться неизменяемым до конца job.
Source CMF получает start/end identity-fence, а повторный полный SHA-проход
датасета после многочасовой калибровки намеренно не делается: это была бы ещё
одна полная цена I/O. Не заменяйте `records.jsonl`, `tokens.idx` или manifest
под работающей командой.

Это O(1) именно по **token payload** невыбранных records. Для нативного
`input.format=jsonl-v1` строгая проверка prompt-ledger намеренно держит
`id → split` и множество увиденных ID, то есть требует память, растущую с
числом prompts. На миллионах prompts не считайте этот ledger бесплатным:
планируйте RAM либо готовьте отдельный finalized calibration selector. Это
сознательная плата за обнаружение дубликата/подмены prompt вне выбранного
split; команда не ослабляет её молча.

Если manifest содержит нативный `tokenizer` contract, команда fail-closed
сверяет `vocab_size`, `vocab_sha256` и `config_sha256` с main-CMF до первого
forward. Это предотвращает тихую калибровку по token IDs от другой модели с
пересекающимся диапазоном ID. Старый generic `.tdata` без contract всё ещё
поддерживается, но выводит warning и помечается в plan как
`tokenizer_contract: "absent-legacy"`; нативный совпавший contract получает
`"verified-v1"`.

Калибровка сейчас канонически **CPU-only**: `probe_mtp_ffn` намеренно исполняет
teacher-forced trunk и pre-down draft activation на CPU, с детерминированным
`f64` накоплением. На время этого probe runtime принудительно выключает
`CMF_FFN_GATE_TOPK` и `CMF_FFN_ORACLE_TOPK`, поэтому plan всегда описывает
полный dense draft, а не случайный serving-эксперимент. `CMF_GPU=1` не ускоряет
этот этап и не должен использоваться как обещание GPU-конвертации;
Metal/Vulkan проверяются уже на готовом кандидате. Это защищает один и тот же
plan от различий FMA/SIMD и порядка reduction между backend-ами.

Физическая запись рассчитана на большие модели: сохранённые trunk tensors
передаются из mmap в `CmfStreamWriter` кусками по 16 MiB, без `Vec<u8>` на всю
модель. В private `mtp-compact`-stage уже проверенный hash неизменённого
payload повторно используется при записи, поэтому source не перечитывается
только ради того же hash; перед publish candidate всё равно полностью
проверяется заново. MTP FFN переписывается projection-за-projection:
одновременно живут только исходная и compacted f32-матрица одной projection
(для `down_proj.t` — ещё её временный transpose); закодированный `down_proj`
явно освобождается до построения transpose, а не все gate/up/down удерживаются
одновременно. Canonical MTP calibration также не хранит весь trunk hidden
history; MiMo replay держит 128 trunk rows внутри одного probe. Главное
ограничение памяти теперь явное: каждая исходная запись разбивается на
непрерывные fresh-causal chunks не длиннее `--max-record-tokens` (по умолчанию
4096), и KV trunk/draft сбрасывается на границе chunk. Поэтому один аномально
длинный `.tdata` record не может разрастить KV до длины всего record; память
для causal-state calibration ограничена cap и геометрией модели, а не размером
контекста record. Сам `.tdata` reader всё ещё materialize-ит выбранный record
как `Vec<u32>`, поэтому не храните один pathological multi-million-token record
вместо нормальных calibration records. Это **не** сохраняет attention через
границу: для quality-critical long-context модели выбирайте cap, сопоставимый
с рабочим контекстом и доступной RAM, а не уменьшайте его ради скорости. На
диске должны одновременно умещаться source и
новый candidate. Не запускайте два rewrite одной и той же модели в один
output-каталог одновременно.

Для исходных `q4tp`/`q2tp` gate/up применяется дополнительный exact fast path:
сохранённый neuron — это целая row при неизменной hidden-оси, поэтому его
weight-plane, row ladder и scale-code копируются byte-for-byte в новом порядке.
Это сохраняет декодированную выбранную row **точно**, не делает второй PTQ-pass
и не выделяет f32 panel gate/up. `down_proj` и `down_proj.t` так копировать
нельзя: compacting выбирает и переставляет их columns, меняя codec groups;
`down_proj` переупаковывается, а transpose строится от него заново. Но у
`q4tp`/`q2tp` writer декодирует для `down_proj` только реально выбранные
source-columns прямо из row/group planes, не materialize-ит полный source f32
panel и затем кодирует компактный matrix обычным способом. Выбранные columns
сначала группируются по исходным 32-wide tile, поэтому ladder/code читается
один раз на группу/row, но порядок ранжирования в результате сохраняется.
Для codec без одного из direct paths ниже gate/up и down остаются корректным
dequant → select → requant path.

У raw `f32`/`bf16`/`f16`, `q8_row` и `q8_2f` fast path также exact, но по
другой причине: gate/up копируют выбранные rows, а `down_proj` выбирает и
переставляет raw values либо q8 codes вместе с их исходными row/column scales.
Поэтому сужение не делает лишний PTQ-pass и не вносит дополнительную ошибку
поверх source-CMF; `bf16` сохраняет свой dtype и не молча понижается до f16.
Если есть `down_proj.t`, лишь уже компактный selected `down_proj` кратко
декодируется для построения transpose; полный source `down_proj` не
materialize-ится. Остальные codecs остаются fail-safe fallback.

Практический RAM-budget считается по **одной** крупнейшей draft-projection, а
не по trunk. Score-pass теперь считает `down_proj_column_l2_squared` прямо по
mmap bytes, удерживая только `Vec<f64>(I)`, для `f32`, `f16`, `bf16`, `q8_row`,
`q8_2f`, `q4tp` и `q2tp`; порядок f32-dequant и f64 суммирования совпадает с
историческим full-panel ranker. Поэтому этот этап не создаёт временный
`[H,I]` f32 buffer для обычных production codecs. Редкий неизвестный/сложный
codec остаётся fail-safe fallback `dequant → score` и может потребовать полный
source panel. Для raw `f32`/`bf16`/`f16`, `q8_row`, `q8_2f`, `q4tp` и `q2tp`
bake также не выделяет полный f32 source `down_proj`; во время `down_proj.t`-стадии
одновременно нужны compacted
`down_proj`, его f32-transpose и закодированный transpose; уже записанный
encoded `down_proj` к этому моменту освобождён. Другие codecs при самом bake
всё ещё могут временно держать исходный и compacted panel.

Перед score-pass каждой draft-layer команда печатает её номер, форму
`down_proj`, codec и elapsed time; это намеренно отдельный progress fence для
очень широких матриц, а не признак зависания. В layer-report плана сохраняется
`down_proj_dtype`, чтобы после job было видно, какой storage-path был оценён.

Это намеренный компромисс: не требует полного `Vec<TensorSpec>`/полной модели
в RAM, но очень широкий отдельный MTP-layer всё равно требует достаточной
оперативной памяти. При сомнении сначала прогоните один слой/малую модель и
следите за RSS, не запускайте параллельные conversion jobs.

Целостность — осознанная цена безопасности: перед калибровкой и после неё
source проверяется полностью; после bake исходный MTP-draft проверяется ещё
раз, а перед publish полностью проверяется candidate. Для 100B+ это несколько
последовательных проходов по диску; это не «зависание» и не обещание мгновенной
конвертации. Каждый full-CMF scan, включая внешний MiMo-sidecar, печатает
начало/конец и wall time; progress калибровки печатается после первого и затем каждого
восьмого bounded chunk. Не изменяйте source до завершения команды: повторная проверка ловит
обычную in-place подмену и команда не публикует смешанный candidate.

На реальном teacher-forced MTP-пути она собирает для каждого канала

```text
sum_teacher_forced((activation(gate, up))^2) * down_proj_column_l2_squared
```

и сохраняет каналы с наибольшим score. Это не trunk-статистика и не
предположение по `intermediate_size` backbone. `activation(gate, up)` — это
ровно функция MLP данного draft из runtime (например, `silu(gate) * up`), а
не всегда SiLU; её контракт также записывается в plan. `--keep-ratio` применяется к
каждому draft-слою, после чего ширина округляется вниз до группы из 32 каналов;
минимум — одна 32-канальная группа. Итоговый JSON-plan фиксирует selector,
число прочитанных selected records и отдельно реально contributing records,
фактических chunks/токенов, cap и policy разбиения,
статус tokenizer contract,
отдельные directory hash и merged `trunk_hash` main/draft, order каналов и
долю сохранённой энергии — храните его рядом с артефактом.
`trunk_hash` покрывает все merged shard tensor entries и их payload hashes.
При `mtp.mode=compact` `tube bake` сверяет обязательный
`source_draft_directory_hash` и, в новых plan, `source_draft_trunk_hash` с
входным draft; калиброванный plan нельзя случайно применить к другому
same-shape sidecar или подменённому позднему shard.

Короткие records/chunks всё ещё участвуют в статистике тех draft-слоёв, до
которых они доходят, но admission-prefix выбирается из **самого длинного**
подходящего bounded chunk (не более 256 token и не короче
`draft_layers + 2`). Это даёт маленькому runtime-gate больше покрытых позиций,
не удерживая record целиком. Если такого chunk нет,
либо у какого-то слоя после агрегации нулевая activation energy, команда
отказывается выбирать каналы по произвольному tie-break, а не выпускает
формально валидный, но непроверяемый compact draft.

## Простая команда для своих промптов

Для обычного случая не нужны ни отдельный teacher, ни ручной каталог данных:

```bash
cortiq compact source.cmf prompts.jsonl task-model.cmf
```

`prompts.jsonl` — обычный JSONL: одна задача на строку. Достаточно простого
`prompt`, а для chat-модели можно передать полный `messages`-массив:

```jsonl
{"id":"sql","prompt":"Напиши безопасный SQL-запрос для поиска заказа по id."}
{"id":"review","messages":[{"role":"system","content":"Отвечай кратко."},{"role":"user","content":"Проверь этот Rust-код на off-by-one ошибки."}]}
```

Эта форма использует `source.cmf` как teacher, создаёт рядом с кандидатом
`task-model.tdata`, генерирует по `prompts.jsonl` deterministic direct-answer
records, финализирует и проверяет набор, затем калибрует/физически сжимает MTP
draft и публикует только проверенный runnable `task-model.cmf`. При повторном
запуске с тем же output уже финализированный `task-model.tdata` **не
перезаписывается**, а переиспользуется; для другого набора промптов выберите
новый output/cache (или осознанно удалите старый cache до запуска).

Это не generic fine-tuning: trunk остаётся byte-identical source, а меняется
только dense-FFN существующего MTP draft. Speculative verifier по-прежнему
является источником emitted token, поэтому greedy output не должен меняться
из-за принятого draft. Модель без поддерживаемого dense MTP fail-closed
отклоняется, а не получает фиктивную «адаптацию». Для управления ratio,
split и лимитами простая форма принимает `--keep-ratio`, `--max-tokens`,
`--samples`, `--seed`, `--split`, `--limit`, `--max-records`,
`--max-record-tokens`, `--keep-truncated` и `--disable-mtp`. Например, быстрый
smoke-run с 25% сужением draft:

```bash
cortiq compact source.cmf prompts.jsonl task-model.cmf \
  --keep-ratio 0.75 --max-tokens 512 --limit 24 --split 100/0/0
```

Для отдельного teacher, независимого `.tdata` пути, split-seed или audit plan
используйте расширенный `mtp-compact` ниже.

## Один запуск: создать или переиспользовать teacher-data

`mtp-compact` может сам подготовить отсутствующий calibration-набор перед
физической компактацией. Это оркестратор **только для dense MTP draft FFN**;
он не является общим `compress` для trunk, MoE или DSpark.

```bash
BIN=target/release/cortiq

# Если calibration.tdata ещё нет, команда создаёт его из teacher/prompts,
# финализирует, затем сразу калибрует и пишет candidate.
"$BIN" mtp-compact base/model.cmf \
  --data calibration.tdata \
  --teacher /models/teacher.cmf \
  --prompts calibration_prompts.jsonl \
  --teacher-max-tokens 512 \
  --teacher-samples 1 \
  --teacher-seed 42 \
  --teacher-split 90/5/5 \
  --keep-ratio 0.90 \
  --max-record-tokens 4096 \
  --output candidate/model.cmf
```

`--data` остаётся явным путём, чтобы generated corpus можно было проверить,
сохранить и повторно использовать. Допустимы `PATH.tdata[:train][=SHARE]`:
compact использует только train split (и при необходимости его детерминированную
долю), оставляя val/test для отдельной оценки. При автоматическом создании
корнем каталога всегда служит часть `PATH.tdata` до selector/suffix. Когда
каталога нет, обязательны одновременно `--teacher` и
`--prompts`; teacher генерирует deterministic greedy direct-answer records
(`--no-think`, `top-k=0`, без `--raw`) и writer сначала финализирует
immutable `.tdata`. Учитель может совпадать с source, однако token IDs и
tokenizer contract набора всё равно сверяются с source до первого calibration
forward. Ещё **до** долгой генерации metadata-preflight сравнивает точные
байты vocabulary и tokenizer config source/teacher: несовместимая пара
отклоняется без траты pod-времени. Для короткого smoke-набора, который будет сразу сжиматься, задайте
`--teacher-split 100/0/0`, чтобы usable records попали в train.

Когда каталог уже существует, команда **не перезаписывает и не генерирует его
заново**: она переиспользует только finalized/readable `.tdata`, печатает это
решение и продолжает обычный полный integrity/tokenizer preflight. Переданные
в таком случае `--teacher`, `--prompts` и teacher-generation flags намеренно
игнорируются. Если их всё же передавать при reuse, CLI требует передать
`--teacher` и `--prompts` вместе. Неполный или повреждённый каталог fail-closed; выберите новый
путь вместо попытки молча продолжить его.

Для генерации доступны `--teacher-limit`, `--teacher-keep-truncated` и
`--teacher-disable-mtp`, а также `--teacher-split-seed`. Параметры
`--max-records` и `--max-record-tokens` относятся уже к compaction, а не к
генерации ответов.

## Ручной режим: готовый teacher-data

Одна команда для встроенного MTP **и** MiMo-sidecar с заранее подготовленным
набором:

```sh
BIN=target/release/cortiq

"$BIN" mtp-compact base/model.cmf \
  --data calibration.tdata:train \
  --keep-ratio 0.90 \
  --max-record-tokens 4096 \
  --output candidate/model.cmf \
  --plan-out candidate/model.mtp-plan.json
```

Для встроенного MTP она запишет `candidate/model.cmf`; если source был
sharded, это будет один valid unsharded CMF. Для MiMo она запишет
`candidate/model.cmf` и `candidate/model.mtp.cmf` без ручного `ln`/`cp`; при
sharded main она потоково coalesce-ит main. Путь
`--output candidate/model.mtp.cmf` остаётся совместимым legacy-вариантом только
для внешнего MiMo-sidecar: он публикует один compact draft и plan, но не создаёт
main-CMF. Это не готовая самостоятельная модель; для новой работы используйте
`--output candidate/model.cmf`, чтобы команда выпустила проверяемую runnable
пару.

До публикации final имён команда использует nonce-private staging-каталоги,
заново открывает и проверяет hashes кандидата, собирает `Pipeline`, подтверждает
загрузку именно MTP-драфта и выполняет на сохранённой calibration-последовательности
teacher-forced draft probe на новой FFN-ширине. Для этого admission хранит
самый длинный подходящий calibration-prefix до 256 token (или
`draft_layers + 2`, если draft глубже); его длина записывается в plan как
`admission_probe_tokens`. Он выполняет
teacher-forced probe compact draft **и** отдельный CPU-only plain greedy gate:
source и staged candidate получают одинаковые первые ≤16 token prefix и должны
выдать одинаковые следующие 2 token при отключённой speculative ветке и GPU.
После bake source main/draft заново проверяются по directory **и** merged
`trunk_hash`; изменившийся source вызывает отказ. У MiMo также сверяется
`trunk_hash` staged backbone с тем, по которому строился plan. Если admission
gate не пройдёт, final candidate не публикуется. Этот короткий CPU gate не
сертифицирует Metal/Vulkan kernel или speculative verifier — для них нужен
отдельный exact-ID тест ниже.

До полного source hash-scan команда проверяет topology явных `--output` /
`--plan-out` (включая alias, parent/child и `.mtp.cmf` sidecar-имя) и убеждается,
что эти final leaves свободны. Поэтому ошибка вроде
`--plan-out candidate.cmf/plan.json` не тратит часы на 100B scan. Реальная
race-safe no-clobber защита всё равно выполняется поздним atomic publish; новые
output parent/stage не создаются до успешных tokenizer и полного `.tdata`
preflight. Поэтому ошибка selector-а или чужого tokenizer не оставляет пустую
вложенную директорию рядом с будущим candidate. После открытия source, но до
`.tdata`, отдельно проверяются canonical dense-MTP geometry, materializable
codec и `--keep-ratio`, чтобы и эти ошибки не запускали калибровку.

До publish используются атомарно созданные nonce-private stage **directories**;
внутренние writer-ы открывают файлы через `create_new`, поэтому collision не
может затереть чужой stage. Payload и родительские directory entries stage/final
синхронизируются (`sync_data`/`fsync`) до сообщения об успехе. Публикация
каждого final имени также no-clobber;
для MiMo sidecar публикуется раньше main, так что не возникает видимого
runnable main без draft. Это не обещание одной файловой транзакции для всей
пары/plan при сбое filesystem: если второй final-link не удался, уже
опубликованный plan/sidecar намеренно **не удаляется**, чтобы race не мог
удалить чужой файл. Осмотрите и удалите такой неполный артефакт вручную перед
повтором; команда откажется его перезаписать. Поэтому сохраняйте source и plan. Gate
доказывает целостность и запускаемость CMF/MTP, но не заменяет hold-out quality
и benchmark целевого backend-а.

`--max-records N` ограничивает число **прочитанных selected records** только
после обязательного full `.tdata` integrity preflight; он делает plan зависимым
от усечённой выборки и не является способом пропустить проверку большого
dataset. В plan `calibration_selected_records` фиксирует этот фактический
лимитный проход, а `calibration_records` — только records, давшие ≥2 token и
внесшие статистику. Для быстрой отладки используйте маленький отдельный
finalized selector.
`--max-record-tokens N` (default 4096) —
production safety cap, а не quality knob: `N` должен быть не меньше
`draft_layers + 2`, и слишком маленький `N` меняет long-context activation
distribution из-за fresh-causal границ. Для кандидата к выпуску используйте
полный заранее зафиксированный calibration selector и заранее выбранный,
задокументированный context cap.

## Независимая ширина draft

Новый заголовок хранит `arch.mtp.ffn_intermediate_sizes` — необязательный
`Vec<usize>` с локальной FFN-шириной каждого MTP-слоя. После компактации она
может быть меньше trunk `arch.intermediate_size`. Для старых файлов поле
отсутствует; загрузчик получает ширину из проверенной тройки tensor-ов
`gate_proj`, `up_proj`, `down_proj`. Если поле есть, оно обязано совпасть с
реальными tensor shapes. MiMo аналогично выводит ширину каждого draft-слоя из
его tensor-ов, а не из geometry trunk.

`eh_proj`, `enorm`, `hnorm`, attention и layer/final norms работают по hidden
width, а не по FFN-axis: при этой операции они побайтно сохраняются; меняются
только согласованные оси `gate/up/down` (и `down_proj.t`, если он был) и
`arch.mtp.ffn_intermediate_sizes`. Поэтому compactor не пытается «уменьшить»
projection/norm по неверной оси.

Следствие: не изменяйте сохранённые directory/trunk hash или width в JSON и не
применяйте план от другой модели. Отсутствующие или несовпадающие hash, а также несовпадение
заголовка, числа слоёв или shapes — причина отказа, а не причина «попробовать
запустить». Несколько draft-слоёв допускаются только для `mimo_v2`, поскольку
именно для него в runtime есть многослойный executor.

## Обязательная проверка корректности

Сначала квалифицируйте **обычный CPU greedy**, затем отдельно — тот же backend
Metal. Во всех прогонах фиксируйте один binary, template, `--raw`/prompt,
`--greedy`, число токенов и environment-переменные. Сравнивайте **token id**,
а не отрендеренный текст. `--trace-json` пишет JSONL в stderr.

CPU является обязательной baseline-проверкой compaction: он не включает MTP
pair, поэтому не может «успешно» принять ошибочный draft. Явно выключайте GPU,
чтобы не унаследовать случайный backend из shell:

```bash
set -euo pipefail

BIN="${BIN:-target/release/cortiq}"
PROMPT="${PROMPT:-Explain in one paragraph why deterministic decoding is useful.}"
mkdir -p verify

extract_trace_ids() {
  local trace_stderr="$1" trace_ids="$2"
  if ! grep '^{"t":' "$trace_stderr" | \
    sed -E 's/.*"token_id":([0-9]+).*/\1/' >"$trace_ids"; then
    echo "cannot extract token IDs from $trace_stderr" >&2
    return 1
  fi
  if ! test -s "$trace_ids"; then
    echo "trace contains no token IDs: $trace_stderr" >&2
    return 1
  fi
}

trace_cpu_plain() {
  model="$1"; out="$2"
  CMF_GPU=0 CMF_MTP=0 CMF_MIMO_MTP=0 "$BIN" run "$model" \
    --raw --prompt "$PROMPT" --greedy -n 128 --trace --trace-json \
    >/dev/null 2>"$out.stderr"
  extract_trace_ids "$out.stderr" "$out.ids"
}

trace_cpu_plain base/model.cmf      verify/base-cpu-plain
trace_cpu_plain candidate/model.cmf verify/candidate-cpu-plain
cmp verify/base-cpu-plain.ids verify/candidate-cpu-plain.ids
```

Для встроенного MTP проверяйте один `.cmf`; для MiMo дополнительно выполните
`"$BIN" verify` для main и `.mtp.cmf` у обеих моделей. Admission-команда уже
делает структурную проверку, но этот trace обязателен на нескольких hold-out
prompt-ах и достаточной длине.

### Host fused-pair — не release-default

`CMF_MTP_PAIR=1` включает экспериментальный CPU fused pair. Он оптимизирует
арифметику двух позиций, но не является verifier. У q4tp Qwen3.5 уже найден
near-tie, где fused pair дал иной greedy ID, чем две серийные target-forward.
Поэтому стандартный CPU `CMF_MTP=1` **не** включает этот путь; не считайте
нулевой `mtp_drafted` доказательством speculative speed.

Если конкретная модель и CPU backend должны использовать host pair, сначала
проведите отдельный long exact-ID gate с `CMF_GPU=0 CMF_MTP=1
CMF_MTP_PAIR=1` для source и candidate против их собственных plain traces. При
первом несовпадении оставьте переменную выключенной. Этот эксперимент не
заменяет Metal qualification ниже.

У GDN-гибридов host pair дополнительно требует `CMF_MTP=1`, потому что его
рекуррентная часть не параллелится и по умолчанию исключена из этого
эксперимента; это не рекомендуемый speed preset.

Trace — также часть контракта проверки: в нём должна быть ровно одна JSON-строка
на каждый выданный token, в том числе на дополнительные draft-token'ы, принятые
одним batched graph-speculative round. Для таких принятых строк runtime не
делает отдельный полный softmax только ради telemetry, поэтому `confidence`
честно равен `0.0`; это не означает нулевую вероятность и не должно
использоваться как quality-метрика.

## Acceptance и скорость на Metal/Vulkan

Сначала убедитесь, что backend виден:

```sh
"$BIN" gpu
```

### Native Metal: безопасный default

Для стандартного канонического MTP native-Metal **batched graph verifier
выключен по умолчанию**. Это намеренный fail-closed default: на реальной модели
одна near-tie строка batched verifier выдала другой *emitted* greedy ID, чем
обычный serial Metal path. Поэтому `CMF_GPU=1 CMF_MTP=1` при неустановленных
`CMF_GRAPH_SPEC` и `CMF_MTP_PAIR` сохраняет обычный serial target decode; это
не ошибка загрузки draft и не причина повторно конвертировать модель. В таком
run `mtp_drafted=0` для canonical MTP — ожидаемый безопасный результат.

`CMF_GRAPH_SPEC=1` оставлен только как явный эксперимент для диагностики
kernel-а. Он **не является production speed preset**: finite набор prompt-ов
не доказывает отсутствие следующей near-tie строки. Не добавляйте эту
переменную в инструкции к модели, пока batch и serial kernels не будут
исправлены и заново широко stress-tested. Отдельный `CMF_MTP_PAIR=1` — тоже
экспериментальный host fused-pair из предыдущего раздела; он не включает и не
исправляет Metal graph verifier. MiMo-speculation управляется
`CMF_MIMO_MTP` отдельно, поэтому это ограничение относится именно к
каноническому MTP graph verifier, а не ко всем MTP-драфтам.

Безопасный Metal smoke/benchmark запускайте с явным удалением опытных флагов:

```sh
env -u CMF_GRAPH_SPEC -u CMF_MTP_PAIR \
  CMF_GPU=1 CMF_MTP=1 CMF_MIMO_MTP=0 \
  "$BIN" bench candidate/model.cmf --tokens 256 --ignore-eos --json \
  > verify/candidate-metal-serial.json
```

Это корректное измерение обычной Metal генерации компактного candidate, но не
доказательство MTP speedup. Сохраняйте из JSON `decode_tok_s_steady`, backend,
device и версию binary; не называйте этот результат speculative benchmark.

### Native Metal: только diagnostic exact-ID gate

Ниже приведён gate **только для расследования** `CMF_GRAPH_SPEC=1`. Различия
CPU/Metal арифметики сами по себе не являются ошибкой, но внутри одного Metal
backend plain и experimental graph ID обязаны совпасть. Используйте несколько
hold-out prompt-ов и достаточную длину. `CMF_METAL_VERIFY_CHECK=1` — именно
native-Metal диагностика: она печатает сравнение batched-vs-serial, но не
меняет уже принятое решение verifier-а и не является fallback.

```bash
set -euo pipefail

BIN="${BIN:-target/release/cortiq}"
PROMPT="${PROMPT:-Explain in one paragraph why deterministic decoding is useful.}"
mkdir -p verify

extract_trace_ids() {
  local trace_stderr="$1" trace_ids="$2"
  if ! grep '^{"t":' "$trace_stderr" | \
    sed -E 's/.*"token_id":([0-9]+).*/\1/' >"$trace_ids"; then
    echo "cannot extract token IDs from $trace_stderr" >&2
    return 1
  fi
  if ! test -s "$trace_ids"; then
    echo "trace contains no token IDs: $trace_stderr" >&2
    return 1
  fi
}

trace_metal_graph_experiment() {
  mode="$1"; model="$2"; out="$3"
  if [ "$mode" = plain ]; then
    CMF_GPU=1 CMF_MTP=0 CMF_MIMO_MTP=0 CMF_METAL_VERIFY_CHECK=1 \
      "$BIN" run "$model" --raw --prompt "$PROMPT" --greedy -n 128 \
      --trace --trace-json >/dev/null 2>"$out.stderr"
  else
    CMF_GPU=1 CMF_MTP=1 CMF_MIMO_MTP=0 CMF_GRAPH_SPEC=1 \
      CMF_METAL_VERIFY_CHECK=1 \
      "$BIN" run "$model" --raw --prompt "$PROMPT" --greedy -n 128 \
      --trace --trace-json >/dev/null 2>"$out.stderr"
  fi
  extract_trace_ids "$out.stderr" "$out.ids"
}

trace_metal_graph_experiment plain base/model.cmf      verify/base-metal-plain
trace_metal_graph_experiment graph base/model.cmf      verify/base-metal-graph
trace_metal_graph_experiment plain candidate/model.cmf verify/candidate-metal-plain
trace_metal_graph_experiment graph candidate/model.cmf verify/candidate-metal-graph

cmp verify/base-metal-plain.ids verify/candidate-metal-plain.ids
cmp verify/base-metal-plain.ids verify/base-metal-graph.ids
cmp verify/candidate-metal-plain.ids verify/candidate-metal-graph.ids

# Любой MISMATCH означает, что batch verifier уже не является exact для
# этого запуска. Он не откатывается автоматически.
if grep -H ' MISMATCH ' verify/*-metal-graph.stderr; then
  echo 'Metal graph verifier mismatch: leave CMF_GRAPH_SPEC unset.' >&2
  exit 1
fi
```

Успешный короткий gate полезен для поиска kernel bugs, но не отменяет
безопасный default выше. Если `mtp_drafted` равен нулю, graph вообще не был
admitted и такой trace не является проверкой speculative route.

### Vulkan/WGPU

Для Vulkan сохранён существующий graph policy, но ускорение всё равно нельзя
объявлять без exact-ID gate на **том же** устройстве. Явно выберите Vulkan,
зафиксируйте `XDG_RUNTIME_DIR` на pod и сравните plain/speculative trace
внутри одного backend. Это пример для canonical MTP; MiMo проверяйте отдельным
run с `CMF_MIMO_MTP=1`, потому что его round не контролируется
`CMF_GRAPH_SPEC`.

```bash
set -euo pipefail

BIN="${BIN:-target/release/cortiq}"
PROMPT="${PROMPT:-Explain in one paragraph why deterministic decoding is useful.}"
mkdir -p verify

extract_trace_ids() {
  local trace_stderr="$1" trace_ids="$2"
  if ! grep '^{"t":' "$trace_stderr" | \
    sed -E 's/.*"token_id":([0-9]+).*/\1/' >"$trace_ids"; then
    echo "cannot extract token IDs from $trace_stderr" >&2
    return 1
  fi
  if ! test -s "$trace_ids"; then
    echo "trace contains no token IDs: $trace_stderr" >&2
    return 1
  fi
}

trace_vulkan() {
  mode="$1"; model="$2"; out="$3"
  if [ "$mode" = plain ]; then
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}" WGPU_BACKEND=vulkan \
      CMF_GPU=1 CMF_MTP=0 CMF_MIMO_MTP=0 \
      "$BIN" run "$model" --raw --prompt "$PROMPT" --greedy -n 128 \
      --trace --trace-json >/dev/null 2>"$out.stderr"
  else
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}" WGPU_BACKEND=vulkan \
      CMF_GPU=1 CMF_MTP=1 CMF_MIMO_MTP=0 \
      "$BIN" run "$model" --raw --prompt "$PROMPT" --greedy -n 128 \
      --trace --trace-json >/dev/null 2>"$out.stderr"
  fi
  extract_trace_ids "$out.stderr" "$out.ids"
}

trace_vulkan plain base/model.cmf      verify/base-vulkan-plain
trace_vulkan spec  base/model.cmf      verify/base-vulkan-spec
trace_vulkan plain candidate/model.cmf verify/candidate-vulkan-plain
trace_vulkan spec  candidate/model.cmf verify/candidate-vulkan-spec
cmp verify/base-vulkan-plain.ids verify/base-vulkan-spec.ids
cmp verify/candidate-vulkan-plain.ids verify/candidate-vulkan-spec.ids
```

На новом Vulkan device сначала выполните этот gate с `CMF_MTP_GRAPH=0`, затем
повторите при обычном graph-настройке. Поздний `Declined` после admission
fail-closed отменяет speculative round, а не обязан прозрачно перейти на host;
смотрите `mtp_drafted` и логи. Только если `mtp_drafted > 0`, считайте
acceptance и сравнивайте несколько чередующихся `decode_tok_s_steady` запусков
base/candidate с одинаковыми binary, context, device и питанием. Меньший draft
может снизить acceptance и сделать candidate медленнее — это валидный результат.

q4tp использует локальную ширину `gate_proj`, а не
`intermediate_size` backbone; 32-channel alignment сохраняет пригодность к
graph path. q8_2f требует собственных alignment/`graph_x` условий. q2tp у
canonical MTP намеренно не принимается native-Metal draft graph и остаётся на
обычном пути. MiMo-sidecar с независимыми draft-width корректно загружается,
но его небольшие draft batches пока не имеют отдельного fused native-Metal MTP
graph. Поэтому `mtp-compact` уменьшает его draft storage/CPU work, но не
обещает Metal speedup без отдельного exact-ID и benchmark evidence.

## Границы и отказы

Команда намеренно fail-closed. Она рассчитана на полную каноническую dense
тройку MTP FFN с shapes `gate=[I,H]`, `up=[I,H]`, `down=[H,I]`, где `I` кратно
32. Не поддерживаются без отдельной реализации/валидации:

* routed/MoE или segmented MTP FFN, неполные tensor triples и несовпадающие
  shapes;
* MTP с width, не допускающей 32-канальную группу, невалидный keep ratio или
  пустая teacher-forced калибровка;
* специальные MTP-форматы, не представленные такой dense-тройкой, и внешние
  DSpark/draft sidecar, не являющиеся MTP FFN;
* frozen genome, если MiMo main нужно физически coalesce-ить из shards:
  full rewrite нарушил бы append-only layout, поэтому такой input отклоняется;
* plan без `selection.source_draft_directory_hash`, с несовпавшим (в новых
  plan) `source_draft_trunk_hash` или с hash, принадлежащим иному draft; изменение
  plan не отменяет проверок source tensor/config.

Сохраните исходные main, sidecar и plan до завершения всех проверок. Компактный
draft можно принять только после целостности файлов, точного greedy-сравнения
и benchmark целевого backend-а по соответствующей exact-ID политике выше.
