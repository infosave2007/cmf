#!/bin/bash
# turbo.sh IN OUT: the far-turbo header (32,1800,4 from 8192 rows, no exact rows) over IN -> OUT; logs/turbo.log
cd /root/rip && . fnx/env.sh
t0=$(date +%s)
$BIN far-turbo $1 $2 --o1 32,1800,4 --seal 8192 > logs/turbo.log 2>&1
echo "exit $? | $(( $(date +%s) - t0 )) s"
ls -la $2
echo TURBO_DONE
