#!/bin/bash
# After the gate dump: speech, music and embedding models on CPU, CUDA and
# Vulkan — wall time of one job each (the second run, the first warms the
# file and the kernels), the GPU declines, and the outputs compared.
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
end=$(( $(date +%s) + 14400 ))
until grep -q -E 'GD_DONE|_FAILED|TIMEOUT' logs/gd.out 2>/dev/null; do
  [ "$(date +%s)" -gt $end ] && { echo GD_WAIT_TIMEOUT; exit 1; }
  sleep 30
done
mkdir -p cmf/dl logs/pa out/pa
wav=ref/s2v_track.wav
dl() { [ -s cmf/dl/$2 ] || timeout 3600 venv/bin/hf download $1 $2 --local-dir cmf/dl > logs/pa/dl_$2.log 2>&1; }
job() { # tag backend command...
  local tag=$1 be=$2; shift 2
  local t0 t1 rc
  for i in 1 2; do
    t0=$(date +%s.%N)
    flock gpu.lock env CMF_GPU=$be CMF_CUDA_DEBUG=1 CMF_GPU_DEBUG=1 timeout 1800 "$@" > out/pa/${tag}_$be.txt 2> logs/pa/${tag}_$be.err
    rc=$?
    t1=$(date +%s.%N)
  done
  echo "$tag $be: exit $rc, $(awk "BEGIN{printf \"%.1f\", $t1 - $t0}") s"
  grep -ihE "declin|refused|→ cpu|not mine|fall(s)? back" logs/pa/${tag}_$be.err | sed -E 's/^.*Z[^ ]* +//' | sort | uniq -c | sort -rn | head -3
}
same() { cmp -s out/pa/$1_cuda.txt out/pa/$1_0.txt && echo "$1: CUDA output == CPU" || echo "$1: CUDA output differs from CPU"; }

dl infosave/whisper-large-v3-turbo-cmf whisper-large-v3-turbo-q8.cmf
for be in 0 cuda 1; do job whisper $be bin/cortiq transcribe cmf/dl/whisper-large-v3-turbo-q8.cmf $wav; done
same whisper; head -c 300 out/pa/whisper_cuda.txt; echo

dl infosave/LFM3-Audio-2B-cmf lfm3-audio-2b-q4tp.cmf
for be in 0 cuda 1; do job lfm3asr $be bin/cortiq run cmf/dl/lfm3-audio-2b-q4tp.cmf --system "Perform ASR in English." --audio $wav --prompt "" --greedy; done
same lfm3asr
for be in 0 cuda 1; do job lfm3tts $be bin/cortiq run cmf/dl/lfm3-audio-2b-q4tp.cmf --system "Perform TTS in American English using a female voice." --prompt "The weather is lovely today and the sea is calm." --greedy; done
rm -f cmf/dl/lfm3-audio-2b-q4tp.cmf

dl infosave/EmbeddingGemma-2-cmf embeddinggemma-2-q8_2f.cmf
for be in 0 cuda 1; do job egemma $be bin/cortiq embed --model cmf/dl/embeddinggemma-2-q8_2f.cmf --task SearchQuery "a red fox in the snow" --image ref/portrait.jpg; done
venv/bin/python - <<'PY'
import re, math
def vec(p):
    return [float(x) for x in re.findall(r'-?\d+\.\d+(?:e-?\d+)?', open(p).read())]
a, b = vec('out/pa/egemma_cuda.txt'), vec('out/pa/egemma_0.txt')
if a and b and len(a) == len(b):
    print(f'egemma: CUDA vs CPU rel {math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/max(sum(y*y for y in b),1e-30)):.2e} over {len(a)} values')
else:
    print('egemma: outputs not comparable', len(a), len(b))
PY

dl infosave/phonon-40m-multilingual-cmf phonon-40m-multilingual-q8.cmf
for be in 0 cuda 1; do job phonon $be bin/cortiq tts cmf/dl/phonon-40m-multilingual-q8.cmf "A calm, clear sentence about the sea and the sky." --out out/pa/phonon_$be.wav; done

dl infosave/qwen3-tts-12hz-0.6b-customvoice-cmf qwen3-tts-12hz-0.6b-customvoice.cmf
for be in 0 cuda 1; do job qtts $be bin/cortiq tts cmf/dl/qwen3-tts-12hz-0.6b-customvoice.cmf "A calm, clear sentence about the sea and the sky." --out out/pa/qtts_$be.wav; done
rm -f cmf/dl/qwen3-tts-12hz-0.6b-customvoice.cmf

dl infosave/MiniMax-Music-3-cmf minimax-music3-q4tp.cmf
for be in cuda 1; do job music $be bin/cortiq music cmf/dl/minimax-music3-q4tp.cmf --prompt "Warm acoustic folk, gentle guitar, soft female voice." --lyrics "[verse] The sea is calm tonight" --seconds 10 --steps 8 --seed 7 --out out/pa/music_$be.wav; done
rm -f cmf/dl/minimax-music3-q4tp.cmf

dl infosave/Nemotron-3.5-Transcribe-30B-A4B-cmf nemotron-3.5-transcribe-30b-a4b-q4tp.cmf
for be in cuda 1; do job nemo $be bin/cortiq transcribe cmf/dl/nemotron-3.5-transcribe-30b-a4b-q4tp.cmf $wav --stats; done
cmp -s out/pa/nemo_cuda.txt out/pa/nemo_1.txt && echo "nemo: CUDA == Vulkan" || echo "nemo: CUDA differs from Vulkan"
rm -f cmf/dl/nemotron-3.5-transcribe-30b-a4b-q4tp.cmf cmf/dl/whisper-large-v3-turbo-q8.cmf cmf/dl/embeddinggemma-2-q8_2f.cmf cmf/dl/phonon-40m-multilingual-q8.cmf
rm -rf cmf/dl/.cache
echo PA_DONE
