#!/bin/bash
cd /root/cherrypit; export XDG_RUNTIME_DIR=/tmp CMF_MAX_SEQ=140000 PYTHONUNBUFFERED=1
S=hfrepo/CherryPit-1.2-2B-q4tp.cmf; T=models/CherryPit-1.2-2B-turbo.cmf
./turbobench.sh $S std 1000 4000 16000 32000 64000
./turbobench.sh $T turbo 1000 4000 16000 32000 64000
python3 scripts/passkey_file.py ./cortiq $T /root/rip/data/wiki.test.raw 300000 4 > logs/facts_turbo_300k.log 2>&1
./turboq.sh
echo CHAIN2_DONE > logs/turbochain2.done
