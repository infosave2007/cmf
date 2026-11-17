#!/usr/bin/env bash
# Embryo-56M on the KLM-1.6 / Open SLM leaderboard benchmarks.
#
#   scripts/embryo/bench/run.sh [smoke|verify|lmeval|arithmark|chat|table|chart|all]
#
# Environment (defaults in brackets):
#   BENCH   run root, outside the repo        [~/embryo-data/bench-klm]
#   MODEL   CMF file                          [~/embryo-data/o1/chat-sft/final5-release/model.cmf]
#   CORTIQ  engine binary with `loglik`       [<repo>/target/release/cortiq]
#   SVG     the KLM-1.6 card chart            [~/Downloads/benchmarks.svg]
#   SVG_OUT the chart with Embryo-56M added   [<this dir>/benchmarks-embryo.svg, + .png]
#
# Downloads land in their own empty directories under $BENCH:
#   venv/               lm-eval 0.4.12 (MIT), no torch
#   hf-home/            HF datasets cache: allenai/ai2_arc (CC-BY-SA-4.0),
#                       Rowan/hellaswag (MIT), baber/piqa (PIQA, AFL-3.0)
#   dl-arithmark3/      AxiomicLabs/ArithMark-3.0 (Apache-2.0): script + data
#   dl-openslm-space/   index.html of the Open SLM leaderboard Space
#                       (restricted license: read for the formula and a few
#                       reference rows, never copied into the repo)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
BENCH="${BENCH:-$HOME/embryo-data/bench-klm}"
MODEL="${MODEL:-$HOME/embryo-data/o1/chat-sft/final5-release/model.cmf}"
CORTIQ="${CORTIQ:-$REPO/target/release/cortiq}"
PY="$BENCH/venv/bin/python"
export HF_HOME="$BENCH/hf-home"

setup() {
  mkdir -p "$BENCH"
  if [ ! -x "$PY" ]; then
    uv venv --python python3.12 "$BENCH/venv"
    VIRTUAL_ENV="$BENCH/venv" uv pip install --python "$PY" "lm-eval==0.4.12"
  fi
  if [ ! -f "$BENCH/dl-arithmark3/arithmark-3.jsonl" ]; then
    mkdir -p "$BENCH/dl-arithmark3"
    for f in README.md bencharithmark-3.py arithmark-3.jsonl; do
      curl -sL -o "$BENCH/dl-arithmark3/$f" \
        "https://huggingface.co/datasets/AxiomicLabs/ArithMark-3.0/resolve/main/$f"
    done
  fi
  if [ ! -f "$BENCH/dl-openslm-space/index.html" ]; then
    mkdir -p "$BENCH/dl-openslm-space"
    curl -sL -o "$BENCH/dl-openslm-space/index.html" \
      "https://huggingface.co/spaces/AxiomicLabs/Open_SLM_Leaderboard/resolve/main/index.html"
  fi
  [ -x "$CORTIQ" ] || (cd "$REPO" && cargo build --release -p cortiq-cli)
}

case "${1:-all}" in
  smoke)
    setup
    "$PY" -I "$HERE/run_lmeval.py" --cortiq "$CORTIQ" --model "$MODEL" \
      --out-dir "$BENCH/runs/smoke" --tasks arc_easy --limit 10 ;;
  verify)
    "$0" smoke
    W="$BENCH/runs/smoke/work"
    "$PY" -I "$HERE/verify_ll.py" --cortiq "$CORTIQ" --model "$MODEL" \
      --requests "$W/arc_easy.raw.limit10.requests.jsonl" \
      --adapter "$W/arc_easy.raw.limit10.loglik.jsonl" --out-dir "$BENCH/runs/verify" ;;
  lmeval)
    setup
    "$PY" -I "$HERE/run_lmeval.py" --cortiq "$CORTIQ" --model "$MODEL" \
      --out-dir "$BENCH/runs/full" --tasks arc_easy,arc_challenge,hellaswag,piqa ;;
  arithmark)
    setup
    "$PY" -I "$HERE/arithmark3_cortiq.py" --cortiq "$CORTIQ" --model "$MODEL" \
      --script "$BENCH/dl-arithmark3/bencharithmark-3.py" \
      --data "$BENCH/dl-arithmark3/arithmark-3.jsonl" --out-dir "$BENCH/runs/full" ;;
  chat)
    # Side check, not in the index: the same tasks with the context as a
    # cmf-im-v1 user turn and the choice as the assistant answer.
    setup
    "$PY" -I "$HERE/run_lmeval.py" --cortiq "$CORTIQ" --model "$MODEL" \
      --out-dir "$BENCH/runs/full" --tasks arc_easy,arc_challenge,piqa --frame chat ;;
  table)
    setup
    "$PY" -I "$HERE/table.py" --runs "$BENCH/runs/full" \
      --space "$BENCH/dl-openslm-space/index.html" --out "$HERE/results.md" \
      --json "$BENCH/runs/full/table.json" --chart-scores "$BENCH/runs/full/chart-scores.json" ;;
  chart)
    OUT="${SVG_OUT:-$HERE/benchmarks-embryo.svg}"
    "$PY" -I "$HERE/chart.py" --svg "${SVG:-$HOME/Downloads/benchmarks.svg}" \
      --scores "$BENCH/runs/full/chart-scores.json" --out "$OUT"
    # PNG at 2x through headless Chrome (renders the embedded Inter woff).
    # Headless Chrome does not always exit after the screenshot: wait for
    # the file, then stop it.
    CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    PNG="${OUT%.svg}.png"
    PROF="$(mktemp -d)"
    rm -f "$PNG"
    "$CHROME" --headless --disable-gpu --hide-scrollbars --no-first-run \
      --user-data-dir="$PROF" --force-device-scale-factor=2 --window-size=1000,616 \
      --screenshot="$PNG" "file://$OUT" >/dev/null 2>&1 &
    CPID=$!
    for _ in $(seq 60); do [ -s "$PNG" ] && break; sleep 1; done
    sleep 2
    kill "$CPID" 2>/dev/null || true
    wait "$CPID" 2>/dev/null || true
    rm -rf "$PROF"
    [ -s "$PNG" ] || { echo "chart: no PNG from Chrome" >&2; exit 1; } ;;
  all)
    "$0" verify && "$0" lmeval && "$0" arithmark && "$0" table && "$0" chart ;;
  *) echo "usage: $0 [smoke|verify|lmeval|arithmark|chat|table|chart|all]" >&2; exit 2 ;;
esac
