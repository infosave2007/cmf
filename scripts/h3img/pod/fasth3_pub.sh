#!/bin/bash
# FastH3 public page: both CMFs from the private experiments repo into the
# (still private) FastH3 repo one at a time, then the card, LICENSE, NOTICE
# and samples. HF_TOKEN comes from the launcher's environment only.
#   fasth3_pub.sh <staging dir with README.md LICENSE NOTICE>
set -u
cd /root/h3i || exit 1
st=$1
repo=infosave/FastH3-4-step-Preview-v1-VSA-cmf
mkdir -p cmf/fh logs/fh
for f in fasth3-clipproj4b-q4tp.cmf fasth3-q4tp.cmf; do
  timeout 7200 venv/bin/hf download infosave/cmf-experiments fasth3/$f --local-dir cmf/fh > logs/fh/dl_$f.log 2>&1 || { echo "$f: DOWNLOAD FAILED"; tail -3 logs/fh/dl_$f.log; exit 1; }
  ls -la cmf/fh/fasth3/$f | awk '{print $5, $9}'
  timeout 7200 venv/bin/hf upload $repo cmf/fh/fasth3/$f $f --commit-message "$f" > logs/fh/up_$f.log 2>&1 || { echo "$f: UPLOAD FAILED"; tail -3 logs/fh/up_$f.log; exit 1; }
  echo "$f uploaded"
  rm -f cmf/fh/fasth3/$f
done
mkdir -p $st/samples
cp out/fh5_samples/* $st/samples/
timeout 1800 venv/bin/hf upload $repo $st . --commit-message "Card, license, notice and samples" > logs/fh/up_card.log 2>&1 || { echo "CARD UPLOAD FAILED"; tail -3 logs/fh/up_card.log; exit 1; }
rm -rf cmf/fh
echo FH_PUB_DONE
