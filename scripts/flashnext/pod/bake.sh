#!/bin/bash
# bake.sh: after the probe, hot-bake --experts 5 % over the turbo file -> fn/qwen38-flash-next-q2tp-hot-turbo.cmf; logs/bake.txt
cd /root/rip && . fnx/env.sh
until grep -q PROBE_DONE logs/probe.txt 2>/dev/null; do sleep 20; done
{
cat logs/probe.txt
t0=$(date +%s)
$BIN hot-bake --experts fn/qwen38-flash-next-q2tp-turbo.cmf Qwen/Qwen3.8-Flash-Next hot/experts.stat fn/qwen38-flash-next-q2tp-hot-turbo.cmf --frac 0.05 --scratch /root/rip/src > logs/bake.log 2>&1
echo "exit $? | $(( $(date +%s) - t0 )) s"
sed -e "s/\x1b\[[0-9;]*m//g" logs/bake.log | grep -a -v "hot-bake: " | tail -12 | cut -c1-200
ls -la fn/qwen38-flash-next-q2tp-hot-turbo.cmf
echo BAKE_DONE
} > logs/bake.txt 2>&1
