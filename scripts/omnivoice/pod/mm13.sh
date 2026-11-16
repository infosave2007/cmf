#!/bin/bash
# GLM-5.3's prompt on CUDA's prompt path: its layers (mHC beside a routed
# FFN) went the small path 16 rows at a time, every chunk rebuilding its
# picks through 1319 slots within 24 GB (mm12b: first token 34.8 s against
# Vulkan's 15 s, which stages each bank whole for 512 positions). Now the
# grouped experts' mixture goes into the mHC merge, and the DSA indexer's
# caches are written on the prompt path (16 rows at a time through the small
# path's kernels); a chunk where DSA selects (past its top-k) still goes the
# small path. GLM-5.3 CUDA within 24 GB and on the card, its perplexity
# (CPU 2.715) and with the selection cut to 64 positions (CPU 2.728, the
# small path 2.730), a 3072-token prompt (crossing 2048), "2+2", Vulkan
# within 24 GB beside it; Ornith-122B q4tp CUDA within 12 / 24 GB (the
# prompt path's other users; mm11: 10.45 / 40.1). Built now (the binary is
# swapped by a rename: mm12b's MiMo runs take it); the download and the
# runs wait for mm12b.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p logs/mm13 out/mm13
bash scripts/build.sh fast > logs/mm13/build.log 2>&1
grep -q BUILD_FAST_DONE logs/mm13/build.log || { echo MM13_BUILD_FAILED; grep -E '^error|error:' -A8 logs/mm13/build.log | head -40; exit 1; }
B=bin/cortiq-fast
strings $B | grep -q "a DSA layer selecting within the chunk" || { echo MM13_STALE_BINARY; exit 1; }
echo MM13_BUILT
# the disk holds MiMo (164 GB) until mm12b is done
end=$((SECONDS + 43200))
until grep -qE "MM12B_DONE" logs/mm12b.out 2>/dev/null || [ $SECONDS -gt $end ]; do sleep 30; done
g=cmf/big/glm-5.3-flash-q2tp.cmf
[ -s $g ] || timeout 14400 venv/bin/hf download infosave/GLM-5.3-Flash-cmf glm-5.3-flash-q2tp.cmf --local-dir cmf/big > logs/mm13/dl.log 2>&1
ls -la $g || { echo "MM13_DOWNLOAD_FAILED: $(df -h /root | tail -1) $(tail -3 logs/mm13/dl.log)"; exit 1; }
pick() { sed -E 's/\x1b\[[0-9;]*m//g' $1 | grep -hoE "container's usage less its page cache grew .*|GPU-visible system memory [a-z ]+the container's limit.*|RAM tier of .*|RAM tier filled at load: .*|prompt chunks [a-z ]+: .*|expert stream of .*|routed experts: .*|graph declined: .*|hybrid batch.*declined.*|panicked.*|out of memory|Out of Memory|CUDA_ERROR[A-Z_]*" | sort -u | cut -c1-150 | tr '\n' '|'; }
bench() { # tag file ctx env...
  local tag=$1 f=$2 ctx=$3; shift 3
  local l=logs/mm13/$tag.log
  flock gpu.lock env "$@" CMF_STREAM_STATS=1 RUST_LOG=info timeout 3600 $B bench $f --tokens 64 --ctx $ctx --ignore-eos --json > $l 2>&1
  echo "$tag: exit $? | $(grep -hoE '"(decode_tok_s_steady|prefill_tok_s|ttft_s)": [0-9.e+-]+' $l | tr '\n' ' ') | $(pick $l)"
}
ppl() { # tag file env...
  local tag=$1 f=$2; shift 2
  flock gpu.lock env "$@" RUST_LOG=info timeout 3600 $B ppl $f --file prompts/rail.txt --tokens 256 > logs/mm13/$tag.log 2>&1
  echo "$tag: exit $? | $(grep -hiE 'ppl|perplexity' logs/mm13/$tag.log | tail -1 | cut -c1-80) | $(pick logs/mm13/$tag.log)"
}
two() { # tag file env...
  local tag=$1 f=$2; shift 2
  flock gpu.lock env "$@" timeout 3600 $B run $f --prompt "What is 2+2? Answer briefly." --no-think -n 16 --greedy > out/mm13/$tag.txt 2> logs/mm13/$tag.log
  echo "$tag: exit $? | $(grep -v '^\[' out/mm13/$tag.txt | tail -n +5 | tr '\n' ' ' | cut -c1-120)"
}
bench glm_cuda_24g $g 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=24576
bench glm_cuda_card $g 512 CMF_GPU=cuda CMF_X=1
ppl glm_cuda_24g_ppl $g CMF_GPU=cuda CMF_GPU_VRAM_MB=24576
ppl glm_cuda_24g_ppl_top64 $g CMF_GPU=cuda CMF_GPU_VRAM_MB=24576 CMF_GLM_DSA_TOPK=64
bench glm_cuda_card_3k $g 3072 CMF_GPU=cuda CMF_X=1
two glm_cuda_two $g CMF_GPU=cuda
bench glm_vk_24g $g 512 CMF_GPU=1 CMF_GPU_VRAM_MB=24576
o=cmf/big/ornith-1.5-122b-q4tp.cmf
bench orn_cuda_12288 $o 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=12288
bench orn_cuda_24576 $o 512 CMF_GPU=cuda CMF_GPU_VRAM_MB=24576
echo MM13_DONE
