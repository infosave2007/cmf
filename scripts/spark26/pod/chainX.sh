#!/bin/bash
# chainX.sh: after the zi2 build (build_to.sh zi2 → ./zi2, renamed cortiq.zi2): the cooperative
# probes (which load reads a [key][dim] f16 plane as the B operand), the attend tile parity test
# (tiles 1, 2 and the plane-fed 3), A/B xc2 (default) vs CMF_ATTEND_XC2=0 at 4096/16000, the stage
# profile at 16000, greedy parity on 9 000 characters — each GPU step under quiet.sh (the recall
# chain paused). Output logs/probe_x.log, logs/parity_tiles_x.log, logs/gab_xc2.txt,
# logs/gX_prof_16000.log, logs/parity_xc2_9k.out, chainX.done.
cd /root/spark26; export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=140000
until [ -f logs/build_zi2.done ]; do sleep 10; done
[ -x zi2 ] && mv zi2 cortiq.zi2
[ -x cortiq.zi2 ] || { echo "no cortiq.zi2"; touch logs/chainX.done; exit 1; }
. "$HOME/.cargo/env"; export CARGO_TARGET_DIR=/root/spark26/target
CT="CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo test -p cortiq-engine --features gpu --release"
(cd cmf && eval $CT --test gpu_coop_probe --test gpu_attend_x_parity --no-run > /root/spark26/logs/tilex_build.log 2>&1)
bash quiet.sh bash -c "cd cmf && CMF_GPU=wgpu $CT --test gpu_coop_probe -- --nocapture --test-threads 1 > /root/spark26/logs/probe_x.log 2>&1"
bash quiet.sh bash -c "cd cmf && CMF_GPU=wgpu $CT --test gpu_attend_x_parity attend_tiles_match -- --nocapture --test-threads 1 > /root/spark26/logs/parity_tiles_x.log 2>&1"
B=./cortiq.zi2; F=hf/cmf/spark-x2.6-4b-q4tp.cmf
BIN=$B VARIANTS="xc2:CMF_ATTEND_XC2=1 xc:CMF_ATTEND_XC2=0" TOKENS=16 bash quiet.sh bash graph_ab.sh xc2 4096 16000
bash quiet.sh bash -c "CMF_GPU_TS=1 CMF_BATCH_TS=1 CMF_BATCH_KERNEL_TS=1 $B bench $F --core --ignore-eos --tokens 4 --ctx 16000 > logs/gX_prof_16000.log 2>&1"
BIN=$B bash quiet.sh bash graph_parity.sh $F 9000 xc2_9k > logs/parity_xc2_9k.out 2>&1
touch logs/chainX.done
