#!/bin/bash
# whisper (CUDA) over every wav of a directory; the language is the tag's
# `_ru` / `_en` part. Prints "tag<TAB>text" lines.
# usage: qtts_whisper.sh <dir>
set -u
cd /root/h3i || exit 1
export XDG_RUNTIME_DIR=/tmp
w=cmf/dl/whisper-large-v3-turbo-q8.cmf
for f in "$1"/*.wav; do
  tag=$(basename "$f" .wav)
  l=en; case $tag in *_ru*) l=ru ;; esac
  txt=$(flock gpu.lock env CMF_GPU=cuda timeout 600 bin/cortiq transcribe $w "$f" --language $l 2>/dev/null | tail -1)
  printf '%s\t%s\n' "$tag" "$txt"
done
echo WHISPER_DONE
