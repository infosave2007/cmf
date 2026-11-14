#!/usr/bin/env bash
# Hermes core, stage 0: open 135-360M models on the R set ("reasoning over a
# card"), before any training of our own.
#
#   scripts/hermes-core/stage0/run.sh [setup|build|baselines|models|grade|report|all]
#
# Environment:
#   STAGE0  run root, outside the repo  [~/embryo-data/hermes-core/stage0]
#
# Downloads, each into its own directory under $STAGE0 (licences checked
# before the download):
#   venv/                      torch (BSD-3), transformers (Apache-2.0)
#   dl-boris2-think/           opencerebral/Boris-2-Think (Apache-2.0)
#   dl-smollm2-135m-instruct/  HuggingFaceTB/SmolLM2-135M-Instruct (Apache-2.0)
#   dl-smollm2-360m-instruct/  HuggingFaceTB/SmolLM2-360M-Instruct (Apache-2.0)
#   dl-tldr/                   tldr-pages (CC-BY 4.0): the command cards
# The sandbox image `cortiq-r-sandbox:1` is built from sandbox/ (Debian
# stable-slim). Docker must be running.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
B="${STAGE0:-$HOME/embryo-data/hermes-core/stage0}"
PY="$B/venv/bin/python"
R="$B/run/r.jsonl"
TLDR="$B/dl-tldr/tldr-main"
export PYTORCH_ENABLE_MPS_FALLBACK=1

hf() { # repo dir file...
  local repo="$1" dir="$2"; shift 2
  mkdir -p "$dir"
  for f in "$@"; do
    [ -s "$dir/$f" ] || curl -sL -o "$dir/$f" "https://huggingface.co/$repo/resolve/main/$f"
  done
}

setup() {
  mkdir -p "$B/run"
  if [ ! -x "$PY" ]; then
    uv venv --python python3.12 "$B/venv"
    VIRTUAL_ENV="$B/venv" uv pip install --python "$PY" torch transformers safetensors
  fi
  hf opencerebral/Boris-2-Think "$B/dl-boris2-think" config.json generation_config.json \
    tokenizer.json tokenizer_config.json configuration_boris2.py modeling_boris2.py model.safetensors
  for m in SmolLM2-135M-Instruct SmolLM2-360M-Instruct; do
    hf "HuggingFaceTB/$m" "$B/dl-$(echo "$m" | tr 'A-Z' 'a-z')" config.json generation_config.json \
      tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt model.safetensors
  done
  if [ ! -d "$TLDR" ]; then
    mkdir -p "$B/dl-tldr"
    curl -sL https://codeload.github.com/tldr-pages/tldr/tar.gz/refs/heads/main | tar -xz -C "$B/dl-tldr"
  fi
  docker image inspect cortiq-r-sandbox:1 >/dev/null 2>&1 || docker build -q -t cortiq-r-sandbox:1 "$HERE/sandbox"
}

model() { # name dir style
  "$PY" -I "$HERE/run_model.py" --model-dir "$B/$2" --style "$3" --tasks "$R" --tldr "$TLDR" \
    --out "$B/run/$1.outputs.jsonl"
}

case "${1:-all}" in
  setup) setup ;;
  build) setup; "$PY" -I "$HERE/build_r.py" --tldr "$TLDR" --out "$R" ;;
  baselines)
    for b in ref copy_first; do
      echo "== $b"; "$PY" -I "$HERE/grade.py" --tasks "$R" --baseline "$b" --out "$B/run/$b.graded.jsonl"
    done ;;
  models)
    # One model at a time.
    model smollm2-135m dl-smollm2-135m-instruct smollm
    model smollm2-360m dl-smollm2-360m-instruct smollm
    model boris2-think dl-boris2-think boris ;;
  grade)
    for m in smollm2-135m smollm2-360m boris2-think; do
      echo "== $m"; "$PY" -I "$HERE/grade.py" --tasks "$R" --outputs "$B/run/$m.outputs.jsonl" --out "$B/run/$m.graded.jsonl"
    done ;;
  report) "$PY" -I "$HERE/report.py" --run "$B/run" --out "$HERE/results.md" ;;
  all) "$0" build && "$0" baselines && "$0" models && "$0" grade && "$0" report ;;
  *) echo "usage: $0 [setup|build|baselines|models|grade|report|all]" >&2; exit 2 ;;
esac
