#!/bin/bash
# kill the bench chain and its running bench
for p in $(ps -eo pid,args | awk "/fnx\/benchchain.sh/ && !/awk/ {print \$1}"); do kill $p; echo "killed chain $p"; done
for p in $(ps -eo pid,args | awk "/cortiq.ft2 bench/ && !/awk/ {print \$1}"); do kill $p; echo "killed bench $p"; done
sleep 2; ps -eo pid,args | grep -c "[c]ortiq.ft2 bench"
