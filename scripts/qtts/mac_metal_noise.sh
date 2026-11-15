#!/bin/bash
# Qwen3-TTS: is Metal noisier than the CPU on the M4? Greedy codes per
# backend (Metal, CPU, CPU with exact activations) and sampled seeds on Metal
# and CPU; the wavs go to the pod for whisper (qtts_whisper.sh).
# usage: mac_metal_noise.sh <cortiq> <dir with qtts-{q4tp,q8_2f}.cmf> [seeds]
set -u
bin=$1; d=$2; seeds=${3:-8}
out=$d/out; mkdir -p $out
ru="Привет! Это пример синтеза речи на русском языке."
en="The quick brown fox jumps over the lazy dog. Speech synthesis now runs natively in cortiq, without Python."
run() { # tag file backend lang seed temperature(or -)
  local tag=$1 f=$2 be=$3 l=$4 s=$5 t=$6 env=(CMF_GPU_PROBE=0)
  local text=$ru; [ $l = en ] && text=$en
  case $be in
    cpu) env+=(CMF_GPU=0) ;;
    cpux) env+=(CMF_GPU=0 CMF_SDOT=0) ;;
  esac
  local temp=(); [ "$t" = - ] || temp=(--temperature $t)
  env "${env[@]}" CMF_QTTS_CODES_DUMP=$out/$tag.json timeout 900 $bin tts $d/qtts-$f.cmf "$text" --lang $l --voice serena --seed $s ${temp[@]+"${temp[@]}"} --out $out/$tag.wav > $out/$tag.log 2>&1
  echo "$tag: exit $? | $(grep -oE '[0-9]+ frames' $out/$tag.log) | $(grep -oE '\[generated.*' $out/$tag.log)"
}
[ -n "${MN_SKIP_GREEDY:-}" ] || for f in q4tp q8_2f; do
  for l in ru en; do
    for be in metal cpu cpux; do run g_${f}_${be}_$l $f $be $l 7 0; done
  done
done
for l in ru en; do
  for s in $(seq 1 $seeds); do
    for be in metal cpu cpux; do run s_q4tp_${be}_${l}_$s q4tp $be $l $s -; done
  done
done
echo MAC_QTTS_DONE
