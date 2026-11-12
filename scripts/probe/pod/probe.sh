#!/bin/bash
# Which popular models already run on CUDA, and how fast against Vulkan: per
# line of probe_list.txt ("repo file") the file is downloaded, its arch read,
# 64 tokens benched on CUDA (the decline reasons kept) and over Vulkan, and
# the file deleted. One TSV line a model in logs/probe.tsv.
#   probe.sh [LIST]
set -u
root=${H3I_ROOT:-/root/h3i}
list=${1:-$root/scripts/probe_list.txt}
bin=${BIN:-$root/bin/cortiq}
cd "$root" || exit 1
mkdir -p cmf/probe logs/probe
export XDG_RUNTIME_DIR=/tmp
tps() { grep -o '"decode_tok_s_steady": *[0-9.]*' "$1" | head -1 | awk -F': *' '{printf "%.1f", $2}'; }
while read -r repo file; do
  [ -z "$repo" ] && continue
  tag=$(basename "$file" .cmf)
  venv/bin/hf download "$repo" "$file" --local-dir cmf/probe > logs/probe/$tag.dl 2>&1
  f=cmf/probe/$file
  if [ ! -f "$f" ]; then echo -e "$repo\t$file\tdownload-failed" >> logs/probe.tsv; continue; fi
  size=$(stat -c %s "$f")
  arch=$("$bin" info "$f" 2>/dev/null | awk '/Arch:/ {print $2}')
  # the card is shared with other runs: one at a time under gpu.lock
  CMF_GPU=cuda CMF_CUDA_DEBUG=1 flock "$root/gpu.lock" timeout 900 "$bin" bench "$f" --tokens 64 --json > logs/probe/$tag.cuda 2>&1
  CMF_GPU=1 flock "$root/gpu.lock" timeout 900 "$bin" bench "$f" --tokens 64 --json > logs/probe/$tag.vk 2>&1
  why=$(grep -h -o "declined: [^\"]*" logs/probe/$tag.cuda | sort -u | head -2 | tr '\n' ';' | cut -c1-160)
  echo -e "$repo\t$file\t$size\t$arch\tcuda=$(tps logs/probe/$tag.cuda)\tvulkan=$(tps logs/probe/$tag.vk)\t$why" >> logs/probe.tsv
  rm -f "$f"
done < "$list"
echo PROBE_DONE >> logs/probe.tsv
