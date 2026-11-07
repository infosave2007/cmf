#!/bin/bash
# rebake.sh: rebuild, then hot-bake 5 % with the cold down -> hot-turbo; ppl; then a 25 % bake -> fn/hot25.cmf; ppl -> logs/rebake.txt
cd /root/rip && . fnx/env.sh
{
./build_ft2.sh
t0=$(date +%s)
$BIN hot-bake --experts fn/qwen38-flash-next-q2tp-turbo.cmf Qwen/Qwen3.8-Flash-Next hot/experts.stat fn/qwen38-flash-next-q2tp-hot-turbo.cmf --frac 0.05 --scratch /root/rip/src > logs/bake5c.log 2>&1
echo "bake 5% cold-down: exit $? | $(( $(date +%s) - t0 )) s | $(sed -e "s/\x1b\[[0-9;]*m//g" logs/bake5c.log | grep -a "holding\|->" | tr "\n" " " | cut -c1-300)"
FILE=/root/rip/fn/qwen38-flash-next-q2tp-hot-turbo.cmf fnx/ppl.sh hot5c 1536
FILE=/root/rip/fn/qwen38-flash-next-q2tp-hot-turbo.cmf fnx/ppl.sh hot5c_128 128
t0=$(date +%s)
$BIN hot-bake --experts fn/qwen38-flash-next-q2tp-turbo.cmf Qwen/Qwen3.8-Flash-Next hot/experts.stat fn/hot25.cmf --frac 0.25 --scratch /root/rip/src > logs/bake25c.log 2>&1
echo "bake 25% cold-down: exit $? | $(( $(date +%s) - t0 )) s | $(sed -e "s/\x1b\[[0-9;]*m//g" logs/bake25c.log | grep -a "holding\|->" | tr "\n" " " | cut -c1-300)"
FILE=/root/rip/fn/hot25.cmf fnx/ppl.sh hot25c 1536
echo REBAKE_DONE
} > logs/rebake.txt 2>&1
