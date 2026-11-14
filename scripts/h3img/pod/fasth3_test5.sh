#!/bin/bash
# Round 5 (after round 4): the compact FastH3 — the 4B prompt encoder with its
# ClipProj carried from the Turbo container, the FastH3 DiT by reference — then
# peak VRAM of both files at both sizes on CUDA, the card samples, and the
# private backup of both files on HF. The token comes in the environment from
# the caller's stdin and is never written anywhere.
set -u
root=${H3I_ROOT:-/root/h3i}
cd "$root" || exit 1
export XDG_RUNTIME_DIR=/tmp
end=$(( $(date +%s) + 10800 ))
until grep -q -E 'FH_TEST4_DONE|BUILD_RELEASE_FAILED' logs/fh_test4.out 2>/dev/null; do
  [ "$(date +%s)" -gt $end ] && { echo ROUND4_WAIT_TIMEOUT; exit 1; }
  sleep 30
done
grep -q BUILD_RELEASE_FAILED logs/fh_test4.out && { echo ROUND4_BUILD_FAILED; exit 1; }
d=cmf/fh
rm -f $d/mmh3-turbo-q4tp.cmf   # published: infosave/MiniMax-H3-Turbo-cmf
if [ ! -s $d/fasth3-clipproj4b-q4tp.cmf ]; then
  [ -s $d/mmh3-turbo-clipproj4b-q4tp.cmf ] || timeout 3600 venv/bin/hf download infosave/MiniMax-H3-Turbo-cmf \
    mmh3-turbo-clipproj4b-q4tp.cmf --local-dir $d > logs/fh5_dl.log 2>&1 || { echo CLIP_DL_FAILED; tail -3 logs/fh5_dl.log; exit 1; }
  t0=$(date +%s)
  timeout 3600 bin/cortiq animate-pack --in $d/mmh3-turbo-clipproj4b-q4tp.cmf --dit-from $d/fasth3-dit-q4tp.cmf \
    --quant q4tp --out $d/fasth3-clipproj4b-q4tp.cmf > logs/fh5_pack.log 2>&1 || { echo PACK_CLIP_FAILED; tail -20 logs/fh5_pack.log; exit 1; }
  echo "compact packed in $(( $(date +%s) - t0 )) s"
  rm -f $d/mmh3-turbo-clipproj4b-q4tp.cmf
fi
ls -l $d | grep cmf
echo "disk $(df -h /root | awk 'NR==2{print $4}') free"
beach="A corgi runs along a sunny beach, waves rolling in behind it, the camera tracking alongside; the dog barks twice."
chef="A corgi in a chef hat flipping a pancake, sizzling sounds and a cheerful bark."
run() { # tag file width height frames seed prompt
  local tag=$1 f=$2 w=$3 h=$4 n=$5 seed=$6 p=$7
  mkdir -p out/fh5_$tag
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 250 > logs/fh5_vram_$tag.txt 2>/dev/null &
  local smi=$!
  local t0
  t0=$(date +%s)
  flock gpu.lock env CMF_GPU=cuda timeout 3600 bin/cortiq animate $f --prompt "$p" --width $w --height $h --frames $n \
    --steps 4 --seed $seed --out out/fh5_$tag/clip.avi --frames-dir out/fh5_$tag/frames > logs/fh5_$tag.log 2>&1
  local rc=$?
  kill $smi 2>/dev/null
  echo "$tag: exit $rc in $(( $(date +%s) - t0 )) s; peak $(sort -n logs/fh5_vram_$tag.txt | tail -1) MiB; $(grep -oE 'stages: .*' logs/fh5_$tag.log | tail -1)"
  grep -iE "declin|refused|panick|error" logs/fh5_$tag.log | sort -u | head -3
}
run clip_s $d/fasth3-clipproj4b-q4tp.cmf 512 288 39 7 "$beach"
run full_s $d/fasth3-q4tp.cmf 512 288 39 7 "$beach"
run clip_b $d/fasth3-clipproj4b-q4tp.cmf 1344 768 124 7 "$beach"
run full_b $d/fasth3-q4tp.cmf 1344 768 124 7 "$beach"
run chef_full $d/fasth3-q4tp.cmf 512 288 39 42 "$chef"
run chef_clip $d/fasth3-clipproj4b-q4tp.cmf 512 288 39 42 "$chef"
s=out/fh5_samples
mkdir -p $s
mp4() { ffmpeg -y -loglevel error -i "$1" -c:v libx264 -pix_fmt yuv420p -crf 20 -c:a aac -b:a 160k -movflags +faststart "$2"; }
gif() { ffmpeg -y -loglevel error -i "$1" -vf "fps=12,scale=$3:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" "$2"; }
mp4 out/fh5_full_b/clip.avi $s/beach_full.mp4
mp4 out/fh5_clip_b/clip.avi $s/beach_clipproj.mp4
gif out/fh5_full_b/clip.avi $s/beach_full.gif 640
gif out/fh5_clip_b/clip.avi $s/beach_clipproj.gif 640
for t in chef_full chef_clip; do
  mp4 out/fh5_$t/clip.avi $s/ab_${t#chef_}.mp4
  gif out/fh5_$t/clip.avi $s/ab_${t#chef_}.gif 512
done
[ -s out/fh_bigdense/clip.avi ] && mp4 out/fh_bigdense/clip.avi $s/beach_full_attention.mp4
cp -f out/fh_big_sheet.png $s/vsa_vs_full_attention.png 2>/dev/null
ls -l $s
# private backup (the token only in this process's environment)
if [ -n "${HF_TOKEN:-}" ]; then
  mkdir -p /root/hfexp/fasth3
  ln -f $d/fasth3-q4tp.cmf $d/fasth3-clipproj4b-q4tp.cmf /root/hfexp/fasth3/
  for f in /root/hfexp/fasth3/*.cmf; do echo "$(sha256sum "$f" | cut -c1-64) $(stat -c %s "$f") $(basename "$f")"; done
  nice -n 19 ionice -c3 timeout 14400 venv/bin/hf upload infosave/cmf-experiments /root/hfexp/fasth3 fasth3 --repo-type model \
    > logs/fh5_up.log 2>&1
  echo "upload exit $?"
  tail -2 logs/fh5_up.log
else
  echo "no HF_TOKEN: upload skipped"
fi
echo FH_TEST5_DONE
