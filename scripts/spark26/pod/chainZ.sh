#!/bin/bash
# chainZ.sh: after the zi build (build_to.sh zi → ./zi, renamed cortiq.zi): the GEMM micro-bench +
# variant parity test (cargo test), A/B zi (the default route through zi_mm) vs coop
# (CMF_BATCH_MM=coop) at 4096/16000, the stage profile at 16000, greedy parity on 9 000 characters —
# each GPU step under quiet.sh (the recall chain paused; a 240k trial has 3000 s). Output
# logs/mmbench_zi.log, logs/gab_zi.txt, logs/gZ_prof_16000.log, logs/parity_zi_9k.out, chainZ.done.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=140000
until [ -f logs/build_zi.done ]; do sleep 10; done
[ -x zi ] && mv zi cortiq.zi
[ -x cortiq.zi ] || { echo "no cortiq.zi"; touch logs/chainZ.done; exit 1; }
. "$HOME/.cargo/env"; export CARGO_TARGET_DIR=/root/spark26/target
(cd cmf && CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo test -p cortiq-engine --features gpu --release --test gpu_q4tp_mm_bench --no-run > /root/spark26/logs/mmbench_build.log 2>&1)
bash quiet.sh bash -c "cd cmf && CMF_GPU=wgpu CMF_MM_BENCH_VARIANTS=zi,coop,scalar CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo test -p cortiq-engine --features gpu --release --test gpu_q4tp_mm_bench -- --nocapture --test-threads 1 > /root/spark26/logs/mmbench_zi.log 2>&1"
B=./cortiq.zi; F=hf/cmf/spark-x2.6-4b-q4tp.cmf
BIN=$B VARIANTS="zi:CMF_BATCH_MM=zi coop:CMF_BATCH_MM=coop" TOKENS=16 bash quiet.sh bash graph_ab.sh zi 4096 16000
bash quiet.sh bash -c "CMF_GPU_TS=1 CMF_BATCH_TS=1 CMF_BATCH_KERNEL_TS=1 $B bench $F --core --ignore-eos --tokens 4 --ctx 16000 > logs/gZ_prof_16000.log 2>&1"
BIN=$B bash quiet.sh bash graph_parity.sh $F 9000 zi_9k > logs/parity_zi_9k.out 2>&1
touch logs/chainZ.done
