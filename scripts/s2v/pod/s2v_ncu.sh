#!/bin/bash
# Kernel metrics of one S2V CUDA step under Nsight Compute: the flash
# attention (and with KERN=... another kernel regex) at 512 x 512.
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
export XDG_RUNTIME_DIR=/tmp
f=cmf/kandinsky-s2v-pro-q4tp.cmf
kern=${KERN:-h3_flash}
tag=${TAG:-ncu_flash}
exec 9>"$root/gpu.lock"; flock 9
CMF_GPU=cuda timeout 1500 /usr/local/cuda/bin/ncu --kernel-name "regex:$kern" --launch-skip ${SKIP:-0} --launch-count ${COUNT:-1} \
  --section SpeedOfLight --section Occupancy --section LaunchStats --section MemoryWorkloadAnalysis --section WarpStateStats --section ComputeWorkloadAnalysis \
  bin/cortiq-fast s2v --model $f --image ref/s2v_ref.png --audio ref/s2v_track.wav \
  --height 512 --width 512 --steps ${STEPS:-1} --chunks 1 --out out/$tag.avi > out/$tag.log 2>&1
echo "rc=$?"
grep -vE "^\s*$" out/$tag.log | grep -E "^  [A-Za-z]|h3_flash|hv_|gemm|==PROF==|Section" | grep -v "INFO" | head -150
echo NCU_DONE
