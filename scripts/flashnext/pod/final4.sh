#!/bin/bash
# final4.sh: the hot file over the plain q2tp-v2 (no far-field header), sha, ppl, upload to the renamed repo -> logs/final4.txt
# HF_TOKEN comes from the environment of the launching shell (read from stdin there), never from disk.
cd /root/rip && . fnx/env.sh
BIN=/root/rip/cortiq.ft8
V2=/root/rip/fn/qwen38-flash-next-q2tp-v2.cmf
OUT=/root/rip/fn/qwen38-flash-next-q2tp-hot.cmf
REPO=infosave/Qwen3.8-Flash-Next-hot-cmf
{
echo "--- final4 $(date -u)"
t0=$(date +%s)
$BIN hot-bake --experts $V2 Qwen/Qwen3.8-Flash-Next hot/experts.stat $OUT --whole-experts 28 --shared-q8 --scratch /root/rip/src > logs/bake_hot.log 2>&1
echo "bake: exit $? | $(( $(date +%s) - t0 )) s | $(sed -e "s/\x1b\[[0-9;]*m//g" logs/bake_hot.log | grep -a "holding\|->\|Error" | tr "\n" " " | cut -c1-300)"
ls -l $OUT
cp fn/qwen38-flash-next-q2tp-turbo.mtp.cmf fn/qwen38-flash-next-q2tp-hot.mtp.cmf
(cd fn && sha256sum qwen38-flash-next-q2tp-hot.cmf > qwen38-flash-next-q2tp-hot.cmf.sha256 && sha256sum qwen38-flash-next-q2tp-hot.mtp.cmf > qwen38-flash-next-q2tp-hot.mtp.cmf.sha256 && cat qwen38-flash-next-q2tp-hot*.sha256)
t0=$(date +%s); hf upload $REPO $OUT qwen38-flash-next-q2tp-hot.cmf > logs/upload_hotp.log 2>&1; echo "upload: exit $? | $(( $(date +%s) - t0 )) s"
hf upload $REPO fn/qwen38-flash-next-q2tp-hot.mtp.cmf qwen38-flash-next-q2tp-hot.mtp.cmf > logs/upload_hotpm.log 2>&1; echo "upload mtp: exit $?"
hf upload $REPO fn/qwen38-flash-next-q2tp-hot.cmf.sha256 qwen38-flash-next-q2tp-hot.cmf.sha256 > /dev/null 2>&1; echo "upload sha: exit $?"
hf upload $REPO fn/qwen38-flash-next-q2tp-hot.mtp.cmf.sha256 qwen38-flash-next-q2tp-hot.mtp.cmf.sha256 > /dev/null 2>&1; echo "upload sha mtp: exit $?"
echo UPLOAD_DONE
BIN=$BIN FILE=$OUT fnx/ppl.sh hot_plain 1536
echo FINAL4_DONE
} > logs/final4.txt 2>&1
