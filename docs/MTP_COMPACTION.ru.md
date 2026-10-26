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
   его автоматически и пишет новый sidecar. Основной файл остаётся побайтно
   тем же.

Для MiMo кандидат должен иметь одно и то же имя основы у main и sidecar:

```text
base/model.cmf              # исходный backbone
base/model.mtp.cmf          # исходный draft

candidate/model.cmf         # копия или hard link base/model.cmf
candidate/model.mtp.cmf     # новый compact draft
```

Не заменяйте исходник. `mtp-compact` отказывается писать и в `--model`, и в
обнаруженный исходный draft; `--plan-out` также не может совпадать с CMF.

## Калибровка и план

Подайте финализированный и проверяемый `.tdata`-selector. Для выбора каналов
используйте только train/calibration split; validation/test оставьте для
последующей оценки. Команда сама проверяет main CMF, обнаруженный MTP-sidecar
и `.tdata`.

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
число записей/токенов, отдельные directory hash main и draft, order каналов и
долю сохранённой энергии — храните его рядом с артефактом. При `mtp.mode=compact`
`tube bake` сверяет `source_draft_directory_hash` с входным draft, поэтому
калиброванный план нельзя случайно применить к другому same-shape sidecar.

Пример sidecar-варианта:

```sh
BIN=target/release/cortiq
mkdir -p candidate
# Hard link экономит место, если каталоги на одном filesystem. Иначе скопируйте файл.
ln base/model.cmf candidate/model.cmf

"$BIN" mtp-compact base/model.cmf \
  --data calibration.tdata:train \
  --keep-ratio 0.90 \
  --output candidate/model.mtp.cmf \
  --plan-out candidate/model.mtp-plan.json
```

`--max-records N` пригоден только для быстрой отладки: он делает план
зависимым от усечённой выборки. Для кандидата к выпуску используйте полный
заранее зафиксированный calibration selector. Для MTP, встроенного в main,
путь выхода должен быть новым, например `--output candidate/model.cmf`.

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

Следствие: не изменяйте fingerprint или width в JSON и не применяйте план от
другой модели. Отсутствующий/несовпадающий fingerprint, а также несовпадение
заголовка, числа слоёв или shapes — причина отказа, а не причина «попробовать
запустить». Несколько draft-слоёв допускаются только для `mimo_v2`, поскольку
именно для него в runtime есть многослойный executor.

## Обязательная проверка корректности

Сначала проверяйте CPU/обычный путь, затем тот же набор на Metal. Во всех
четырёх прогонах фиксируйте один и тот же binary, модельный template,
`--raw`/prompt, `--greedy`, число токенов и все environment-переменные,
влияющие на MTP. Не включайте непроверенные экспериментальные graph/chain
переключатели только ради benchmark.

Ниже `base/model.cmf` и `candidate/model.cmf` образуют пару из схемы выше.
`--trace-json` пишет JSONL в stderr; сравнивать следует **token id**, а не
отрендеренный текст.

```sh
BIN=target/release/cortiq
PROMPT='Explain in one paragraph why deterministic decoding is useful.'
mkdir -p verify

trace_ids() {
  mode="$1"; model="$2"; out="$3"
  if [ "$mode" = plain ]; then
    CMF_MTP=0 CMF_MIMO_MTP=0 "$BIN" run "$model" \
      --raw --prompt "$PROMPT" --greedy -n 128 --trace --trace-json \
      >/dev/null 2>"$out.stderr"
  else
    CMF_MTP=1 CMF_MIMO_MTP=1 "$BIN" run "$model" \
      --raw --prompt "$PROMPT" --greedy -n 128 --trace --trace-json \
      >/dev/null 2>"$out.stderr"
  fi
  grep '^{"t":' "$out.stderr" | \
    sed -E 's/.*"token_id":([0-9]+).*/\1/' >"$out.ids"
}

trace_ids plain base/model.cmf       verify/base-plain
trace_ids spec  base/model.cmf       verify/base-spec
trace_ids plain candidate/model.cmf  verify/candidate-plain
trace_ids spec  candidate/model.cmf  verify/candidate-spec

"$BIN" verify base/model.cmf
"$BIN" verify base/model.mtp.cmf
"$BIN" verify candidate/model.cmf
"$BIN" verify candidate/model.mtp.cmf

# Все три условия должны пройти.
cmp verify/base-plain.ids verify/candidate-plain.ids
cmp verify/base-plain.ids verify/base-spec.ids
cmp verify/candidate-plain.ids verify/candidate-spec.ids
```

Проведите этот тест на нескольких prompt-ах и до достаточной длины, чтобы
draft успел использоваться. Для встроенного MTP проверяйте один `.cmf`, без
несуществующего sidecar. Если любой `cmp` не проходит, есть ошибка компактации
или speculative verifier; такой файл не выпускается.

## Acceptance и скорость на Metal

Сначала убедитесь, что backend виден, и используйте GPU-сборку:

```sh
"$BIN" gpu
```

Повторите предыдущие четыре trace-прогона с одинаковым `CMF_GPU=1` перед
каждой командой. Это проверяет именно Metal-путь; обычный CPU-успех его не
доказывает. Для измерения сначала сделайте один прогревочный, неучитываемый
прогон, затем несколько раз поочерёдно запускайте baseline/candidate при
одинаковых устройстве, питании, контексте, binary и env:

```sh
# Сохраните каждый полный JSON отдельно, например base-spec-1.json.
CMF_GPU=1 CMF_MTP=1 CMF_MIMO_MTP=1 \
  "$BIN" bench base/model.cmf --tokens 256 --ignore-eos --json \
  > verify/base-spec-1.json

CMF_GPU=1 CMF_MTP=1 CMF_MIMO_MTP=1 \
  "$BIN" bench candidate/model.cmf --tokens 256 --ignore-eos --json \
  > verify/candidate-spec-1.json
```

Из JSON каждого прогона сохраните как минимум:

* `decode_tok_s_steady` — steady-state скорость decode;
* `mtp_drafted` и `mtp_accepted`; вычислите
  `mtp_accepted / mtp_drafted` только если `mtp_drafted > 0`;
* используемый device/backend и версию binary.

`mtp_drafted == 0` означает, что данный benchmark не доказал работу draft.
Не объявляйте ускорение по одному запуску или по `incl_prefill`; сравнивайте
повторяемые `decode_tok_s_steady` при одинаковом контракте. Меньший draft
может снизить acceptance настолько, что кандидат станет медленнее — это
измеряемый результат, а не повод скрывать счётчики.

Для компактного MTP с `q4tp` fast-path Metal/WGPU использует локальную ширину
`gate_proj`, а не `intermediate_size` backbone; поэтому 32-кратно сжатый draft
не получает устаревшую geometry trunk. У `q2tp`/Prism MTP native-Metal graph
сейчас безопасно отклоняется до выполнения и идёт через обычный fallback; у
`q8_2f` одношаговый путь корректен, но batched warm-up также может перейти в
fallback. Это не ошибка точности, но не основание обещать скорость для этих
codec без отдельного замера. Для выпуска с целью Metal-speed выбирайте `q4tp`
и всё равно подтверждайте его приведённым benchmark.

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
* plan без `selection.source_draft_directory_hash` или с fingerprint не того
  draft; изменение plan не отменяет проверок source tensor/config.

Сохраните исходные main, sidecar и plan до завершения всех проверок. Компактный
draft можно принять только после целостности файлов, точного greedy-сравнения
и честного Metal benchmark с acceptance-счётчиками.
