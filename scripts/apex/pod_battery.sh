#!/bin/bash
# Greedy --no-think answers of each file to a short fact/arithmetic/format
# battery — a smoke check that the quantized files answer alike.
#
# usage: pod_battery.sh CORTIQ MODEL.cmf [MODEL.cmf ...]
set -u
B=$1; shift
export XDG_RUNTIME_DIR=/tmp
PROMPTS=(
  "Compute 17 * 23. Reply with just the number."
  "What is 1234 + 5678? Reply with just the number."
  "What is the capital of France? Answer in one word."
  "В каком году началась Вторая мировая война? Ответь одним числом."
  "Write one line of Python that prints the sum of 2 and 3."
  "All cats are mammals. Some mammals swim. Does it follow that some cats swim? Answer yes or no."
  "Write exactly two sentences about a robot gardener."
  "Name the chemical symbol for gold. Answer with the symbol only."
)
for M in "$@"; do
  echo "== $(basename "$M")"
  for p in "${PROMPTS[@]}"; do
    a=$("$B" run "$M" --prompt "$p" --greedy --no-think --max-tokens 96 2>/dev/null \
      | sed -n '/^Prompt:/,$p' | sed '1,2d' | grep -v -E "tokens ·|MTP draft" | tr '\n' ' ' | sed 's/  */ /g')
    printf '%s\n  -> %s\n' "$p" "$a"
  done
done
echo BATTERY_DONE
