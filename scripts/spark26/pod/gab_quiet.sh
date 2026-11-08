#!/bin/bash
# gab_quiet.sh ARGS...: pause the recall chain (its python and the cortiq it runs) while
# graph_ab.sh ARGS runs on a free card, then resume it. The paused trial's first-token
# time stretches by the pause; its recall verdict does not change.
cd /root/spark26
PY=$(pgrep -f passkey_file.py | head -1)
pause() { [ -n "$PY" ] && { kill -STOP $(pgrep -P "$PY") 2>/dev/null; kill -STOP "$PY"; }; }
resume() { [ -n "$PY" ] && { kill -CONT "$PY"; kill -CONT $(pgrep -P "$PY") 2>/dev/null; }; }
trap resume EXIT
pause; sleep 3
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
bash graph_ab.sh "$@"
