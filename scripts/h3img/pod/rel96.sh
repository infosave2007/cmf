#!/bin/bash
# The 0.8.96 release sweep on the A5000 for the HF cards: the release build,
# then each file of rel96_list.txt in turn: download, `bench` on CUDA and on
# Vulkan (64-token decode twice, then a long prompt), the file deleted.
# Embryo-56M runs from cmf/ with the resident graph. rel96.sh [LIST]
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
export XDG_RUNTIME_DIR=/tmp
list=${1:-scripts/rel96_list.txt}
bin=bin/cortiq
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  bash scripts/build.sh release > logs/r96_build.log 2>&1 || { echo BUILD_RELEASE_FAILED; tail -30 logs/r96_build.log; exit 1; }
  rm -rf srcrs/target-fast
  echo "build ok, disk $(df -h /root | awk 'NR==2{print $4}') free"
fi
keys='"(decode_tok_s_steady|prefill_tok_s|ttft_s)": *[0-9.]*'
bench() { # tag backend kind args...
  local tag=$1 be=$2 kind=$3; shift 3
  local log=logs/r96_${tag}_${be}_${kind}.log
  flock gpu.lock env CMF_GPU=$be timeout "${TMO:-900}" $bin bench "$@" --json > "$log" 2>&1
  echo "$tag $be $kind: $(grep -oE "$keys" "$log" | tr '\n' ' ')"
}
mkdir -p cmf/dl
while read -r repo file ctx envs; do
  case "$repo" in ''|\#*) continue;; esac
  tag=$(basename "$file" .cmf)
  f=cmf/dl/$file
  if [ ! -s "$f" ]; then
    timeout 3600 venv/bin/hf download "$repo" "$file" --local-dir cmf/dl > "logs/r96_dl_$tag.log" 2>&1 || { echo "$tag DOWNLOAD_FAILED"; tail -3 "logs/r96_dl_$tag.log"; rm -f "$f"; continue; }
  fi
  for be in cuda 1; do
    bench "$tag" $be d1 "$f" --tokens 64
    bench "$tag" $be d2 "$f" --tokens 64
    TMO=600 bench "$tag" $be p "$f" --tokens 32 --ctx "$ctx"
  done
  rm -f "$f"
done < "$list"
# Embryo-56M (kept in cmf/): the resident graph on both backends, `run`'s own
# prefill / steady-decode line.
for be in cuda 1; do
  for i in 1 2; do
    log=logs/r96_embryo_${be}_$i.log
    flock gpu.lock env CMF_GPU=$be CMF_EMBRYO_RESIDENT=1 timeout 300 $bin run cmf/embryo-56m.cmf \
      --prompt "What is the capital of France?" -n 64 --greedy > "$log" 2>&1
    echo "embryo-56m $be run$i: $(grep -oE '[0-9.]+ tok/s( steady)?' "$log" | tr '\n' ' ')"
  done
done
rm -rf cmf/dl/.cache
echo "disk $(df -h /root | awk 'NR==2{print $4}') free"
echo REL96_DONE
