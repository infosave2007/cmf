#!/bin/bash
# The numbers for the card: a whole render, the portrait bucket, a custom prompt.
cd /workspace/s2v/src; . ~/.cargo/env
export CARGO_TARGET_DIR=/workspace/s2v/target
timeout 1200 cargo build --release -p cortiq-cli --config profile.release.lto=false --config profile.release.codegen-units=16 > /workspace/s2v/build.log 2>&1
echo BUILD_EXIT $? >> /workspace/s2v/build.log
cd /workspace/s2v
export XDG_RUNTIME_DIR=/tmp RUST_BACKTRACE=1 CMF_S2V_PROF=1
B=target/release/cortiq
M=cmf/s2v-q4tp.cmf
vram() { while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done; }
peak() { sort -n "$1" | tail -1; }
. venv/bin/activate
vram > out/vram_q4tp.txt &
VP=$!
timeout 1500 /usr/bin/time -v $B s2v --model $M --image assets/reference_square.jpg --audio assets/speech.wav --oracle-noise out/run_bf16.safetensors --out-latents out/lat_q4tp2.safetensors --out out/q4tp2.avi > out/f_q4tp.log 2>&1
kill $VP
echo "F_EXIT $? peak MiB $(peak out/vram_q4tp.txt)" >> out/f_q4tp.log
python src/scripts/s2v/compare.py out/run_bf16.safetensors out/lat_q4tp2.safetensors >> out/f_q4tp.log 2>&1
timeout 200 $B s2v-check --model $M --stage vae --oracle out/frames.safetensors --extra out/lat_q4tp2.safetensors > out/f_psnr_q4tp.log 2>&1
timeout 200 $B s2v-check --model $M --stage vae --oracle out/frames.safetensors --extra out/lat_q8.safetensors > out/f_psnr_q8.log 2>&1
vram > out/vram_768.txt &
VP=$!
timeout 600 $B s2v --model $M --image assets/reference.jpg --audio assets/speech.wav --chunks 1 --max-steps 2 --out out/portrait.avi > out/f_768.log 2>&1
kill $VP
echo "F_EXIT $? peak MiB $(peak out/vram_768.txt)" >> out/f_768.log
timeout 300 $B s2v --model $M --image assets/reference_square.jpg --audio assets/speech.wav --chunks 1 --max-steps 1 --prompt "A person talks calmly to the camera in a bright room." --out out/custom.avi > out/f_custom.log 2>&1
echo F_EXIT $? >> out/f_custom.log
vram > out/vram_q8.txt &
VP=$!
timeout 300 $B s2v --model cmf/s2v-q8.cmf --image assets/reference_square.jpg --audio assets/speech.wav --chunks 1 --max-steps 2 --out out/q8b.avi > out/f_q8.log 2>&1
kill $VP
echo "F_EXIT $? peak MiB $(peak out/vram_q8.txt)" >> out/f_q8.log
timeout 1000 $B s2v --model $M --image assets/reference_square.jpg --audio assets/speech.wav --out out/sample.avi > out/f_sample.log 2>&1
echo F_EXIT $? >> out/f_sample.log
ffmpeg -y -loglevel error -i out/sample.avi -vf "select=not(mod(n\,30)),scale=256:256,tile=5x1" -frames:v 1 out/fr/sample.png
echo FINAL_DONE > out/final.done
