#!/bin/bash
# publish_files.sh: the model files and their checksums into the two (private until opened) HF repos, each
# checked by size and sha256 through the API. HF_TOKEN in the environment only.
set -eu
cd /root/sn/out
for f in sensenova-loop-9b-q4tp.cmf sensenova-loop-9b-q8_2f.cmf sensenova-loop-9b-turbo.cmf; do
  [ -f $f.sha256.pub ] || sha256sum $f > $f.sha256.pub
done
mkdir -p /root/sn/pub/base /root/sn/pub/turbo
for f in sensenova-loop-9b-q4tp.cmf sensenova-loop-9b-q8_2f.cmf; do ln -f $f /root/sn/pub/base/$f; cp $f.sha256.pub /root/sn/pub/base/$f.sha256; done
ln -f sensenova-loop-9b-turbo.cmf /root/sn/pub/turbo/sensenova-loop-9b-turbo.cmf
cp sensenova-loop-9b-turbo.cmf.sha256.pub /root/sn/pub/turbo/sensenova-loop-9b-turbo.cmf.sha256
cp /root/sn/turbo/keep_sn9_long.bin /root/sn/pub/turbo/
cd /root/sn/pub/base
python3 -I /root/sn/src/scripts/snloop/hf_up.py infosave/SenseNova-Loop-9B-cmf . sensenova-loop-9b-q4tp.cmf sensenova-loop-9b-q8_2f.cmf sensenova-loop-9b-q4tp.cmf.sha256 sensenova-loop-9b-q8_2f.cmf.sha256
cd /root/sn/pub/turbo
python3 -I /root/sn/src/scripts/snloop/hf_up.py infosave/SenseNova-Loop-9B-Turbo-cmf . sensenova-loop-9b-turbo.cmf sensenova-loop-9b-turbo.cmf.sha256 keep_sn9_long.bin
echo PUBLISH_FILES_DONE
