# MiniCPM-V-4.7-35B-A3B: точный Metal MoE на UMA

Этот документ фиксирует план ускорения `MiniCPM-V-4.7-35B-A3B-q4tp.cmf` на
Apple Silicon. Его цель — не получить красивую цифру ценой другой модели, а
сохранить исходный router/top-k, shared expert и ответ target-модели.

## Отправная точка

На стенде Apple M4 / 24 GiB:

| величина | значение |
|---|---:|
| q4tp CMF | 19 372 423 311 B (18.04 GiB) |
| рекомендованный Metal working set | 19 069 665 280 B (17.76 GiB) |
| routed-expert payload | ~16.86 GB |
| обычный MoE | 40 слоёв, top-8 + shared expert |

Полный native Metal graph корректно отказывается собираться до аллокации:
вся mmap-модель чуть больше рекомендованного working set. Этот отказ нельзя
просто убрать: прежний `WeightArena` создаёт no-copy окна по всей primary
mmap, а Metal учитывает длину таких views при memory accounting, даже когда
один token реально читает только несколько экспертов.

У модели 30 Gated DeltaNet и 10 full-attention слоёв. Поэтому возврат hidden
state на CPU после каждого MoE означает не только стоимость матриц, но и
около сорока последовательных CPU↔GPU границ за token. Это главный предел
первой фазы ниже.

## Фаза P1: selected no-copy views (текущая реализация)

Для oversized q4tp MoE до создания legacy full-file arena runtime:

1. оставляет router на CPU и использует его **точный** top-8 + shared route;
2. проверяет dtype/shape/shard/offset каждой из 27 матриц selected experts;
3. создаёт `newBufferWithBytesNoCopy` только на page-rounded область каждой
   выбранной gate/up/down матрицы, с offset к фактическому payload;
4. для обычного MiniCPM route из девяти jobs (top-8 + shared) исполняет
   gate/up/SiLU и down+упорядоченный weighted mix двумя fused dispatches в
   двух encoder; это один command buffer и один checked wait на слой;
   нестандартное число jobs остаётся на generic selected-view fallback;
5. при любой ошибке admission или Metal completion берёт исходный CPU-path,
   а не пытается создать full-file arena после частично выполненной GPU работы.

Это не копируемый VRAM-bank: Metal видит узкие mmap views, а OS/UMA решает
физическую residency. Views живут до `wait_fast_checked()`. Их нельзя
свободно кэшировать без лимита: за достаточное число tokens cache снова
соберёт почти весь 18-GiB файл в Metal accounting. Также нельзя делать
`madvise(DONTNEED)` при вытеснении — это превратит следующий decode в disk
faults.

Диагностика:

```sh
CMF_GPU=1 CMF_GPU_PROBE=0 CMF_METAL_SPARSE_MOE=1 \
  cortiq run MiniCPM-V-4.7-35B-A3B-q4tp.cmf --greedy --no-think ...
```

`CMF_METAL_SPARSE_MOE=0` полностью выключает эксперимент. Счётчики
`SPARSE_MOE_BLOCKS` и `SPARSE_MOE_SPAN_BYTES`, а также лог
`native Metal selected MoE views active`, отличают реально пройденный pager
от CPU fallback.

### Что P1 гарантирует и чего не гарантирует

- **Гарантирует:** не меняет router, не подменяет cold experts, не требует
  переквантизации и не bind-ит гигабайтные дырки между разнесёнными в файле
  tensor payloads.
- **Не гарантирует:** высокий tok/s. Он всё ещё делает host routing и один
  wait на каждый MoE слой. Если benchmark проигрывает CPU, он не должен
  включаться как «ускорение» без явного opt-in.
- **Ограничение API:** `metal-rs` представляет `newBufferWithBytesNoCopy` как
  non-null `Buffer`, поэтому nil при самом создании view нельзя надёжно
  отличить от успеха и вернуть CPU fallback; admission и ошибки completion
  обработаны, но по этой причине P1 остаётся явным opt-in.
- **Проверено на M4 / 24 GiB (изоляция только MoE):** на raw 64-token
  greedy «numbers 1..100» strict CPU дал 10.3 tok/s decode, selected views
  дали 12.6 tok/s (+22%); на 32-token forced-ID траектории совпали все
  argmax, max |Δlogit| = 2.48e-5, max |Δlayer hidden| = 7.73e-5. Это
  regression-gate одного стенда, а не универсальное обещание скорости.
- **Не покрывает q8_2f 35B:** q8-файл существенно больше UMA memory budget;
  его техническая исполнимость не равна разумной цели скорости на M4/24 GiB.

## Почему не достаточно одного layer span прямо сейчас

Конвертер знает `exec_order_key`, но при multi-shard source сначала
выгружает накопленные tensor-списки shard за shard, а финальная сортировка
происходит слишком поздно для физического layout. В опубликованном q4tp
файле большинство MoE-layer span близки к 0.4 GiB, но есть заметные outlier:
например, у одного слоя span около 2.65 GiB. Более того, gate/up/down одного
эксперта не обязаны лежать рядом.

Поэтому на текущем артефакте один role/layer span может включить огромную
физическую дыру и проиграть узким views. Это причина, по которой P1 не
притворяется финальным graph pager.

## Фаза P2: воспроизводимая compaction в execution order

Нужна отдельная команда вида `compact --execution-order`, а не незаметная
смена семантики обычного `compact`:

- stream-rewrite уже открытого CMF заимствованными mmap slices, без 19-GiB
  RAM копии;
- физически упаковать tensor payloads в порядок layer → expert → gate/up/down;
- выпустить новый SHA/manifest и новую ревизию артефакта только после
  валидации.

Гейты до публикации:

1. `cortiq verify` старого и нового файлов;
2. равенство directory по `(name, dtype, shape, payload hash)`, а не raw file
   SHA (порядок байтов законно меняется);
3. fixed-prompt CPU greedy token IDs и допустимая logit tolerance;
4. layout test: min..max span каждого MiniCPM MoE-layer укладывается в budget,
   selected triple покрывается его layer span;
5. cold/warm benchmark без OOM и с явным счётчиком реально выполненных
   graph/pager blocks.

## Фаза P3: настоящий Metal graph pager

Только после P2 имеет смысл возвращать fast path, который действительно
может убрать 40 host waits:

- GPU-resident hidden state, GDN recurrent state и KV;
- on-device router/top-k;
- no-copy span текущего слоя (или небольшого GDN/attention segment) с
  in-flight lease, удерживаемым до completion command buffer;
- asynchronous commit уже существующего `TokenGraph`, не новый scheduler;
- bounded число in-flight spans и measurement gate по Metal working set/RSS.

Гранулярность надо измерять: один слой, GDN-run и 4–8 слоёв. Малый span
снижает residency, слишком много submit/commit съедают выигрыш, большой span
снова упирается в working set.

Копируемый expert-slot bank из MiMo/Qwen нельзя брать как дефолт на UMA:
он дублирует те же unified-memory страницы, а не создаёт отдельную VRAM.
Его допустимо исследовать только после profiler evidence, с bounded global
budget, pin до completion и exact CPU correction для cold expert.

## Фаза P4: DSpark — отдельный sidecar, не MTP «на глаз»

У базовой 35B нет физических `mtp.*` слоёв. Зато есть официальный
[MiniCPM-V-4.7-35B-A3B-DSpark](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-DSpark):
~389M BF16 sidecar, 5 dense Qwen3 layers, до 15 draft tokens за проход,
с target captures после `[3, 11, 19, 27, 35]`.

Нужен отдельный `<base>.dspark.cmf` converter/runtime, а не reuse текущего
DeepSeek-V4-specific MTP кода. Важно:

- draft делит target embedding/lm_head, но имеет собственные 5 KV-cache;
- verify обязан подтверждать target-модель и транзакционно откатывать **и KV,
  и GDN recurrent state** при reject;
- sampling либо имеет математически корректный residual/rejection, либо
  выключен;
- DSpark включается автоматически только после token-ID parity и warm tok/s
  выигрыша.

Официальная card сообщает mean accepted length 3.50 (general), 6.69 (math),
7.39 (code); это не равно такому же speedup. В документе
[LES / lookahead expert scheduling](https://github.com/tc-mb/llama.cpp-omni/blob/master/docs/les/lookahead-expert-scheduling.md)
показан ориентир порядка +17% от DSpark на iPad M4 после других оптимизаций.
Основной выигрыш для 35B остаётся за pager/graph residency, а не за одним
драфтом.

## Честный benchmark-протокол

Перед любым заявлением скорости нужны:

- CPU, legacy safe Metal fallback, P1 views и P3 graph как отдельные строки;
- prompts 16/128/512 tokens, decode 128, cold и warm повторы;
- `first-token`, steady decode, whole wall time, GPU busy time, number of
  command buffers и bound view bytes;
- greedy token parity с CPU на фиксированном корпусе; отдельная проверка
  logits/argmax около границы;
- forced admission failure и Metal-error test: никакого stale readback,
  только exact CPU fallback;
- memory-pressure/OOM negative test.

Пока эти гейты не пройдены, нельзя называть экспериментальный pager или
DSpark финальным ускорением и нельзя менять уже опубликованный model card
скоростным обещанием.
