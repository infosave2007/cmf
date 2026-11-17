#!/bin/bash
# MiniMax-Music-3 on CUDA against Vulkan: where the AR and the denoise
# lose. Two seconds of song (50 AR frames) and two denoise steps per run,
# each arm in its own process; the stage clocks come from `cortiq music`.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p cmf/dl logs/mc out/mc
m=cmf/dl/minimax-music3-q4tp.cmf
[ -s $m ] || timeout 3600 venv/bin/hf download infosave/MiniMax-Music-3-cmf minimax-music3-q4tp.cmf --local-dir cmf/dl > logs/mc/dl.log 2>&1
[ -s $m ] || { echo MC_DL_FAILED; exit 1; }
run() { # tag, then env assignments
  local tag=$1${MC_TAG:-}; shift
  local t0 t1
  t0=$(date +%s.%N)
  flock gpu.lock env CMF_MUSIC3_LOGITS_DUMP=out/mc/$tag.logits "$@" timeout 1500 bin/cortiq music $m \
    --prompt "Warm acoustic folk, gentle guitar, soft female voice." \
    --lyrics "${MC_LYRICS:-[verse] The sea is calm tonight}" --seconds ${MC_SECONDS:-2} --steps ${MC_STEPS:-2} --seed ${MC_SEED:-7} \
    --out out/mc/$tag.wav > logs/mc/$tag.out 2> logs/mc/$tag.err
  local rc=$?
  t1=$(date +%s.%N)
  echo "$tag: exit $rc, $(awk "BEGIN{printf \"%.1f\", $t1 - $t0}") s | $(grep -a -o 'ar done[^)]*)' logs/mc/$tag.err | tail -1) | $(grep -a -o 'stages:.*' logs/mc/$tag.err | tail -1)"
}
for arm in ${MC_ARMS:-cuda cuda_graph vk vk_graph}; do
  case $arm in
    cuda) run cuda CMF_GPU=cuda ;;
    cuda_graph) run cuda_graph CMF_GPU=cuda CMF_MUSIC3_GRAPH=1 ;;
    vk) run vk CMF_GPU=1 ;;
    vk_graph) run vk_graph CMF_GPU=1 CMF_MUSIC3_GRAPH=1 ;;
    cpu) run cpu CMF_GPU=0 ;;
    cuda_host) run cuda_host CMF_GPU=cuda CMF_MUSIC3_GRAPH=0 ;;
    cpu_exact) run cpu_exact CMF_GPU=0 CMF_SDOT=0 ;;
    cuda_cache) run cuda_cache CMF_GPU=cuda CMF_CUDA_MM_CACHE_MB=${MC_CACHE_MB:-16000} CMF_MUSIC3_PROF=1 ;;
    cuda_graph_cache) run cuda_graph_cache CMF_GPU=cuda CMF_MUSIC3_GRAPH=1 CMF_CUDA_MM_CACHE_MB=${MC_CACHE_MB:-16000} CMF_MUSIC3_PROF=1 ;;
    vk_prof) run vk_prof CMF_GPU=1 CMF_MUSIC3_PROF=1 ;;
    cuda_prof) run cuda_prof CMF_GPU=cuda CMF_CUDA_PROF=1 CMF_MUSIC3_PROF=1 ;;
    cuda_prof_fixed) run cuda_prof_fixed CMF_GPU=cuda CMF_CUDA_PROF=1 CMF_MUSIC3_PROF=1 CMF_CUDA_MM_CACHE_MB=1400 ;;
    cuda_graph_check) run cuda_graph_check CMF_GPU=cuda CMF_MUSIC3_GRAPH=1 CMF_MUSIC3_GRAPH_CHECK=1 ;;
    vk_graph_check) run vk_graph_check CMF_GPU=1 CMF_MUSIC3_GRAPH=1 CMF_MUSIC3_GRAPH_CHECK=1 ;;
    cuda_graph_noplane) run cuda_graph_noplane CMF_GPU=cuda CMF_MUSIC3_GRAPH=1 CMF_CUDA_MM_CACHE_MB=0 ;;
    cuda_graph_verify) run cuda_graph_verify CMF_GPU=cuda CMF_MUSIC3_GRAPH=1 CMF_CUDA_MM_VERIFY=1 ;;
    cuda_graph_prof) run cuda_graph_prof CMF_GPU=cuda CMF_MUSIC3_GRAPH=1 CMF_CUDA_PROF=1 CMF_MUSIC3_PROF=1 ;;
  esac
done
venv/bin/python - <<'PY'
import numpy as np, glob, os
L = {os.path.basename(f)[:-7]: np.fromfile(f, dtype='<f4') for f in sorted(glob.glob('out/mc/*.logits'))}
ref = next((L[k] for k in L if k.startswith('cpu')), next((L[k] for k in L if k.startswith('vk') and 'graph' not in k), None))
for k, v in L.items():
    if ref is None or v.shape != ref.shape: print(k, v.shape); continue
    n = len(v) // 2
    rel = np.linalg.norm(v - ref) / max(np.linalg.norm(ref), 1e-30)
    print(f"{k}: rel vs ref {rel:.3e}, top5 cond {np.argsort(-v[:n])[:5].tolist()}, ref {np.argsort(-ref[:n])[:5].tolist()}")
PY
echo MC_DONE
