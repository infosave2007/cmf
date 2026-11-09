#!/bin/bash
# mac_decode_prof.sh DIR SCRIPTS: where a decoded token's time goes on Metal. Three runs with the
# GPU timestamps of every command buffer (CMF_METAL_GPUPROF + CMF_METAL_CMD_TIMES: kernel time, the
# gap before the GPU starts it) and the host waits: the standard file on p2k and p14k, the turbo file
# on p14k. Then the decode speed with one command buffer per N layers (CMF_METAL_COMMIT_EVERY).
# Waits for the quality suite and the rebuilt binary; runs the llama.cpp rows afterwards.
D=$1; S=$2; cd "$D"
until grep -q SUITE2_DONE suite2.log && [ -f build_ce.done ]; do sleep 15; done
prof() { # tag model prompt env...
  local tag=$1 m=$2 p=$3; shift 3
  env "$@" CMF_METAL_GPUPROF=1 CMF_METAL_CMD_TIMES=1 CMF_METAL_HOSTPROF=1 ./cortiq run "$m" --prompt-file "$p" --no-think --greedy -n 64 > prof_$tag.out 2> prof_$tag.err
  echo "== $tag: $(tail -1 prof_$tag.out)"
  grep -E "gpuprof|waitprof" prof_$tag.err | tail -3
  python3 - prof_$tag.err <<'EOF'
import re, sys
k = g = u = 0.0
n = 0
for line in open(sys.argv[1]):
    m = re.match(r"cmd-times: kernel ([\d.]+) ms \| kernel end to gpu start ([\d.-]+) ms \| gpu ([\d.]+) ms", line)
    if m:
        k += float(m[1]); g += float(m[2]); u += float(m[3]); n += 1
print(f"cmd buffers {n} ({n/64:.1f}/token): kernel {k/64:.2f} ms/token, gaps {g/64:.2f} ms/token, gpu {u/64:.2f} ms/token")
EOF
}
prof std2k hf/spark-x2.6-4b-q4tp.cmf p2k.txt X=1
prof std14k hf/spark-x2.6-4b-q4tp.cmf p14k.txt X=1
prof turbo14k hf/spark-x2.6-4b-q4tp-turbo.cmf p14k.txt X=1
for n in 1 3 6 12 36; do
  CMF_METAL_COMMIT_EVERY=$n ./cortiq run hf/spark-x2.6-4b-q4tp.cmf --prompt-file p2k.txt --no-think --greedy -n 96 2>/dev/null | tail -1 | sed "s/^/commit_every=$n p2k: /"
done
echo PROF_DONE
bash "$S/mac_llama_rows.sh" "$D" > suite2_llama.log 2>&1
