#!/bin/bash
# The whole pod sequence after `pod_setup.sh`: reference dumps and containers side
# by side, then the parity checks and a render on the device.
cd /workspace/s2v
while ! grep -q FETCH_DONE fetch.log 2>/dev/null || ! grep -q BUILD_EXIT build.log 2>/dev/null; do sleep 5; done
grep -q "BUILD_EXIT 0" build.log || exit 1
(
  . venv/bin/activate
  export OMP_NUM_THREADS=13 TOKENIZERS_PARALLELISM=false PYTORCH_ALLOC_CONF=expandable_segments:True
  timeout 900 python oracle.py enc --image assets/reference_square.jpg --audio assets/speech.wav --chunks 2 --out out/enc.safetensors > out/enc.log 2>&1
  echo ENC_EXIT $? >> out/enc.log
  timeout 300 python oracle.py dit --enc out/enc.safetensors --lh 32 --lw 32 --lt 8 --dtype f32 --out out/dit_32_f32.safetensors > out/dit_32.log 2>&1
  echo DIT_EXIT $? >> out/dit_32.log
  timeout 2400 python oracle.py run --enc out/enc.safetensors --chunks 2 --seed 5 --dtype bf16 --out out/run_bf16.safetensors > out/run_bf16.log 2>&1
  echo RUN_EXIT $? >> out/run_bf16.log
  timeout 600 python oracle.py vae --latents out/run_bf16.safetensors --out out/frames.safetensors > out/vae.log 2>&1
  echo VAE_EXIT $? >> out/vae.log
) &
(
  export CMF_THREADS=8
  timeout 1800 target/release/cortiq s2v-pack hf --quant q4tp --out cmf/s2v-q4tp.cmf > out/pack_q4tp.log 2>&1
  echo PACK_EXIT $? >> out/pack_q4tp.log
  timeout 1800 target/release/cortiq s2v-pack hf --quant q8 --out cmf/s2v-q8.cmf > out/pack_q8.log 2>&1
  echo PACK_EXIT $? >> out/pack_q8.log
) &
wait
export XDG_RUNTIME_DIR=/tmp RUST_BACKTRACE=1 CMF_S2V_PROF=1
M=cmf/s2v-q8.cmf
. venv/bin/activate
python - <<'PY'
from safetensors.torch import load_file, save_file
import torch
d = load_file("out/enc.safetensors")
save_file({k: (v.float() if v.dtype == torch.int32 else v) for k, v in d.items()}, "out/enc2.safetensors")
PY
target/release/cortiq s2v-check --model $M --stage sched --oracle out/run_bf16.safetensors > out/check_sched.log 2>&1
target/release/cortiq s2v-check --model $M --stage enc --oracle out/enc2.safetensors --extra assets/reference_square.jpg > out/check_enc.log 2>&1
timeout 300 target/release/cortiq s2v-check --model $M --stage vae --oracle out/frames.safetensors --extra out/run_bf16.safetensors > out/check_vae.log 2>&1
CMF_S2V_NO_HOST=1 timeout 300 target/release/cortiq s2v-check --model $M --stage dit --oracle out/dit_32_f32.safetensors > out/check_dit.log 2>&1
CMF_S2V_CLASS=1 timeout 900 target/release/cortiq s2v --model $M --image assets/reference_square.jpg --audio assets/speech.wav --oracle-noise out/run_bf16.safetensors --chunks 1 --out-latents out/lat_q8.safetensors --out out/q8.avi > out/r_q8.log 2>&1
echo R_EXIT $? >> out/r_q8.log
python src/scripts/s2v/compare.py out/run_bf16.safetensors out/lat_q8.safetensors >> out/r_q8.log 2>&1
CMF_S2V_CLASS=1 timeout 900 target/release/cortiq s2v --model cmf/s2v-q4tp.cmf --image assets/reference_square.jpg --audio assets/speech.wav --oracle-noise out/run_bf16.safetensors --chunks 1 --out-latents out/lat_q4tp.safetensors --out out/q4tp.avi > out/r_q4tp.log 2>&1
echo R_EXIT $? >> out/r_q4tp.log
python src/scripts/s2v/compare.py out/run_bf16.safetensors out/lat_q4tp.safetensors >> out/r_q4tp.log 2>&1
echo CHAIN_DONE > out/chain.done
