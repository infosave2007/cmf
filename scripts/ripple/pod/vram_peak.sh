#!/bin/bash
# The peaks of device memory and of anonymous RAM over a whole run, and of a spill directory:
# vram_peak.sh NAME LOG SPILLDIR -- COMMAND...  (the card's use before the run is subtracted)
name=$1; log=$2; dir=$3; shift 4
base=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
"$@" > /tmp/vp.out 2>&1 &
pid=$!
peak=0; anon=0; disk=0
while kill -0 $pid 2>/dev/null; do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$u" -gt "$peak" ] && peak=$u
  a=$(grep RssAnon /proc/$pid/status 2>/dev/null | awk '{print $2}')
  [ -n "$a" ] && [ "$a" -gt "$anon" ] && anon=$a
  if [ -d "$dir" ]; then d=$(du -sm "$dir" 2>/dev/null | cut -f1); [ -n "$d" ] && [ "$d" -gt "$disk" ] && disk=$d; fi
  sleep 0.5
done
r=$(grep -E "Prompt|Decode|panicked" /tmp/vp.out | sed -e 's/\x1b\[[0-9;]*m//g' -e 's/.*→ cpu//' -e 's/  */ /g' | tr '\n' ' ' | cut -c1-165)
echo "$name: VRAM $((peak-base)) MiB, RAM $((anon/1024)) MiB, disk $disk MiB | $r" | tee -a $log
