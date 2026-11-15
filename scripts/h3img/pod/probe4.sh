#!/bin/bash
# The models never measured on CUDA (14.11): decode (64 tokens, two runs) and
# the 4k prompt on CUDA and Vulkan with CUDA's decline reasons, a greedy
# answer CUDA against Vulkan, and for the speech files their speech paths
# (LFM3-Audio/Omni ASR + TTS, d1-Omni with a clip, Phonon TTS) on CUDA,
# Vulkan and the CPU. Each file deleted after its turn.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
mkdir -p cmf/dl logs/p4 out/p4
keys='"(decode_tok_s_steady|prefill_tok_s|ttft_s)": *[0-9.]*'
clip=/root/qt/mac_g/g_q8_2f_cpux_en.wav   # "The quick brown fox ... without Python."
now() { date +%s.%N; }
el() { awk -v a=$1 -v b=$(now) 'BEGIN{printf "%.1f s", b-a}'; }
why() { grep -ihE "declin|refused|failed|→ cpu" "$@" | sed -E 's/\x1b\[[0-9;]*m//g; s/.*(INFO|WARN|ERROR)[^ ]* //' | sort | uniq -c | sort -rn | head -5; }
llm() { # tag file
  local tag=$1 f=$2
  for be in cuda 1; do
    for i in 1 2; do
      flock gpu.lock env CMF_GPU=$be CMF_CUDA_DEBUG=1 timeout 1200 bin/cortiq bench $f --tokens 64 --ignore-eos --json > logs/p4/${tag}_d_${be}_$i.log 2>&1
      echo "$tag $be decode run$i: exit $?; $(grep -oE "$keys" logs/p4/${tag}_d_${be}_$i.log | tr '\n' ' ')"
    done
    flock gpu.lock env CMF_GPU=$be CMF_CUDA_DEBUG=1 timeout 1200 bin/cortiq bench $f --tokens 32 --ctx 4096 --json > logs/p4/${tag}_4k_$be.log 2>&1
    echo "$tag $be 4k: exit $?; $(grep -oE "$keys" logs/p4/${tag}_4k_$be.log | tr '\n' ' ')"
    [ $be = cuda ] && why logs/p4/${tag}_d_cuda_2.log logs/p4/${tag}_4k_cuda.log
    flock gpu.lock env CMF_GPU=$be timeout 1200 bin/cortiq run $f --prompt "Explain in two sentences why the sky is blue." -n 48 --greedy 2>/dev/null | grep -v '^\[' > out/p4/${tag}_$be.txt
  done
  cmp -s out/p4/${tag}_cuda.txt out/p4/${tag}_1.txt && echo "$tag: CUDA text == Vulkan" || { echo "$tag: CUDA text differs from Vulkan"; tail -2 out/p4/${tag}_cuda.txt; tail -2 out/p4/${tag}_1.txt; }
}
speech() { # tag file (LFM3-Audio / Omni): ASR of the clip, then TTS of a sentence
  local tag=$1 f=$2
  for be in cuda 1 0; do
    t=$(now)
    flock gpu.lock env CMF_GPU=$be CMF_CUDA_DEBUG=1 timeout 1200 bin/cortiq run $f --audio $clip --system "Perform ASR in English." -n 64 --greedy > logs/p4/${tag}_asr_$be.log 2>&1
    echo "$tag $be asr: exit $?, $(el $t) | $(grep -v -E '^\[|INFO|WARN|cuda ' logs/p4/${tag}_asr_$be.log | tr '\n' ' ' | cut -c1-140)"
    [ $be = cuda ] && why logs/p4/${tag}_asr_cuda.log
    t=$(now)
    flock gpu.lock env CMF_GPU=$be CMF_CUDA_DEBUG=1 timeout 1200 bin/cortiq run $f --prompt "Speech synthesis now runs natively in cortiq, without Python." --system "Perform TTS in American English using a female voice." --speech-out out/p4/${tag}_tts_$be.wav > logs/p4/${tag}_tts_$be.log 2>&1
    echo "$tag $be tts: exit $?, $(el $t) | $(grep -oE '[0-9.]+ ?s(econds)? of (speech|audio)[^|]*|wrote [^ ]+ \([^)]*\)' logs/p4/${tag}_tts_$be.log | head -1)"
    [ $be = cuda ] && why logs/p4/${tag}_tts_cuda.log
    [ -s out/p4/${tag}_tts_$be.wav ] && flock gpu.lock env CMF_GPU=cuda timeout 600 bin/cortiq transcribe cmf/dl/whisper-large-v3-turbo-q8.cmf out/p4/${tag}_tts_$be.wav --language en 2>/dev/null | tail -1 | sed 's/^/    whisper: /'
  done
}
get() { # repo file
  [ -s cmf/dl/$2 ] || timeout 3600 venv/bin/hf download $1 $2 --local-dir cmf/dl > logs/p4/${2%.cmf}_dl.log 2>&1
  [ -s cmf/dl/$2 ]
}
# Phonon TTS
if get infosave/phonon-40m-multilingual-cmf phonon-40m-multilingual-q8.cmf; then
  for be in cuda 1 0; do
    t=$(now)
    flock gpu.lock env CMF_GPU=$be CMF_CUDA_DEBUG=1 timeout 1200 bin/cortiq tts cmf/dl/phonon-40m-multilingual-q8.cmf "Speech synthesis now runs natively in cortiq, without Python." --lang en --out out/p4/phonon_$be.wav > logs/p4/phonon_$be.log 2>&1
    echo "phonon $be: exit $?, $(el $t) | $(grep -oE 'wrote .*' logs/p4/phonon_$be.log | head -1) $(grep -oE '\[.*real time.*\]' logs/p4/phonon_$be.log | head -1)"
    [ $be = cuda ] && why logs/p4/phonon_cuda.log
    [ -s out/p4/phonon_$be.wav ] && flock gpu.lock env CMF_GPU=cuda timeout 600 bin/cortiq transcribe cmf/dl/whisper-large-v3-turbo-q8.cmf out/p4/phonon_$be.wav --language en 2>/dev/null | tail -1 | sed 's/^/    whisper: /'
  done
  rm -f cmf/dl/phonon-40m-multilingual-q8.cmf
else echo "phonon: DOWNLOAD FAILED"; fi
while read -r repo file kind; do
  [ -z "$repo" ] && continue
  tag=${file%.cmf}
  get $repo $file || { echo "$tag: DOWNLOAD FAILED"; continue; }
  llm $tag cmf/dl/$file
  [ "$kind" = speech ] && speech $tag cmf/dl/$file
  if [ "$kind" = d1 ]; then
    for be in cuda 1 0; do
      t=$(now)
      flock gpu.lock env CMF_GPU=$be CMF_CUDA_DEBUG=1 timeout 1200 bin/cortiq system-one cmf/dl/$file --state "A customer voice message." --questions '{"animal": {"type": "noul", "instructions": "Does the clip mention an animal?"}}' --audio $clip > logs/p4/${tag}_s1_$be.log 2>&1
      echo "$tag $be system-one: exit $?, $(el $t) | $(grep -oE '\{"answers".*' logs/p4/${tag}_s1_$be.log | cut -c1-160)"
      [ $be = cuda ] && why logs/p4/${tag}_s1_cuda.log
    done
  fi
  rm -f cmf/dl/$file
done <<'LIST'
infosave/MNN-Pocket-0.3B-cmf MNN-Pocket-0.3B-q4tp.cmf llm
infosave/MNN-Pocket-0.3B-cmf MNN-Pocket-0.3B-q8_2f.cmf llm
infosave/LFM3-Audio-2B-cmf lfm3-audio-2b-q4tp.cmf speech
infosave/LFM3-Omni-2B-cmf lfm3-omni-2b-q4tp.cmf speech
infosave/d1-Omni-2B-cmf d1-omni-2b-q4tp.cmf d1
infosave/Qwen3.8-27B-Code-14B-Experimental-cmf qwen38-27b-code-14b-q4tp.cmf llm
infosave/Qwen3.8-27B-NVG-0.55bpw-Experimental-cmf qwen38-27b-nvg055-10m.cmf llm
infosave/Arete-27B-Turbo-cmf arete-27b-turbo.cmf llm
LIST
rm -rf cmf/dl/.cache
echo P4_DONE
