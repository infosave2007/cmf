#!/bin/bash
# Private backup of the Qwen3-TTS and Haru conversions (infosave/cmf-experiments),
# the token in this process's environment only; sha256 and sizes printed for the
# API check.
set -u
mkdir -p /root/hfexp/qtts /root/hfexp/haru
ln -f /root/qt/qtts-q8_2f.cmf /root/hfexp/qtts/qwen3-tts-12hz-0.6b-customvoice-q8_2f.cmf
ln -f /root/qt/qtts-q4tp.cmf /root/hfexp/qtts/qwen3-tts-12hz-0.6b-customvoice-q4tp.cmf
ln -f /root/haru/haru-q8_2f.cmf /root/hfexp/haru/haru-4b-q8_2f.cmf
for f in /root/hfexp/qtts/*.cmf /root/hfexp/haru/*.cmf; do echo "$(sha256sum "$f" | cut -c1-64) $(stat -c %s "$f") $(basename "$f")"; done
cd /root/h3i
for d in qtts haru; do
  nice -n 19 ionice -c3 timeout 7200 venv/bin/hf upload infosave/cmf-experiments /root/hfexp/$d $d --repo-type model > logs/bk_$d.log 2>&1
  echo "$d upload exit $?"
done
rm -rf /root/hfexp
echo BK_DONE
