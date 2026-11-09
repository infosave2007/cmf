#!/bin/bash
# restart_f7.sh: stop chain5 (f6 collided two bind-cache steps), build f7, run chain6
cd /root/fc; J=/root/fc/JOURNAL.md
for p in $(pgrep -f "chain5[.]sh") $(pgrep -f "serve_tests[.]sh") $(pgrep -f "speedtest[.]py") $(pgrep -f "humaneval[.]py"); do kill $p 2>/dev/null; done
for p in $(pgrep -f "serve/server[.]py") $(pgrep -f "cortiq[.]f6 serve"); do kill -- -$p 2>/dev/null; kill $p 2>/dev/null; done
sleep 8; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill $p 2>/dev/null; done
echo "$(date +%T) RESTART: f6 bound the split attention under rawk's cache step (513/514); f7 uses 524/525, arena + staging; numbers from f6 above are void" >> $J
./build.sh f7 > logs/build_f7.log 2>&1
grep -q BUILD_DONE logs/build_f7.log || { echo "$(date +%T) BUILD f7 FAILED" >> $J; exit 1; }
exec ./chain6.sh
