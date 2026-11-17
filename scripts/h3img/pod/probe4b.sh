#!/bin/bash
# After probe4: the LFM3 speech files' ASR (with the empty prompt the media
# order needs) and in-process clocks of their speech paths and of d1-Omni,
# CUDA / Vulkan / CPU, with the CUDA product profile.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
end=$(( $(date +%s) + 10800 ))
until grep -q P4_DONE logs/probe4.out 2>/dev/null; do
  [ "$(date +%s)" -gt $end ] && { echo P4_WAIT_TIMEOUT; exit 1; }
  sleep 30
done
mkdir -p cmf/dl logs/p4b out/p4b
clip=/root/qt/mac_g/g_q8_2f_cpux_en.wav   # "The quick brown fox ... without Python."
get() { [ -s cmf/dl/$2 ] || timeout 3600 venv/bin/hf download $1 $2 --local-dir cmf/dl > logs/p4b/${2%.cmf}_dl.log 2>&1; [ -s cmf/dl/$2 ]; }
clk() { grep -h -oE '\[[^]]*(tok/s|real time|generated)[^]]*\]|system-one: [0-9]+ ms|[0-9.]+ s of speech[^|]*|cuda h3 product: .*|  (gemm|copy up|rows down|host planes|rows up|expand) .*' "$@" | head -12 | sed 's/^/    /'; }
while read -r repo file kind; do
  [ -z "$repo" ] && continue
  tag=${file%.cmf}
  get $repo $file || { echo "$tag: DOWNLOAD FAILED"; continue; }
  f=cmf/dl/$file
  for be in cuda 1 0; do
    if [ $kind = d1 ]; then
      flock gpu.lock env CMF_GPU=$be CMF_CUDA_PROF=1 timeout 1200 bin/cortiq system-one $f --state "A customer voice message." --questions '{"animal": {"type": "noul", "instructions": "Does the clip mention an animal?"}}' --audio $clip > logs/p4b/${tag}_s1_$be.log 2>&1
      echo "$tag $be system-one: exit $? | $(grep -oE '"noul": [0-9.]+' logs/p4b/${tag}_s1_$be.log)"; clk logs/p4b/${tag}_s1_$be.log
    else
      flock gpu.lock env CMF_GPU=$be CMF_CUDA_PROF=1 timeout 1200 bin/cortiq run $f --system "Perform ASR in English." --audio $clip --prompt "" -n 64 --greedy > logs/p4b/${tag}_asr_$be.log 2>&1
      echo "$tag $be asr: exit $? | $(grep -v -E '^\[|INFO|WARN|cuda |^  ' logs/p4b/${tag}_asr_$be.log | tr '\n' ' ' | cut -c1-140)"; clk logs/p4b/${tag}_asr_$be.log
      flock gpu.lock env CMF_GPU=$be CMF_CUDA_PROF=1 CMF_LFM3_SPEECH_PROF=1 timeout 1200 bin/cortiq run $f --prompt "Speech synthesis now runs natively in cortiq, without Python." --system "Perform TTS in American English using a female voice." --speech-out out/p4b/${tag}_tts_$be.wav > logs/p4b/${tag}_tts_$be.log 2>&1
      echo "$tag $be tts: exit $?"; clk logs/p4b/${tag}_tts_$be.log; grep -h -iE 'speech prof|stage' logs/p4b/${tag}_tts_$be.log | head -6 | sed 's/^/    /'
    fi
  done
  rm -f $f
done <<'LIST'
infosave/LFM3-Audio-2B-cmf lfm3-audio-2b-q4tp.cmf speech
infosave/LFM3-Omni-2B-cmf lfm3-omni-2b-q4tp.cmf speech
infosave/d1-Omni-2B-cmf d1-omni-2b-q4tp.cmf d1
LIST
rm -rf cmf/dl/.cache
echo P4B_DONE
