#!/bin/bash
# fin2.sh: the release cut fcaq256 on cortiq.f26 (this branch on PR #102): server speed twice (fresh session tags,
# so the 1K answer is not the one-token reply of the fcaq run), then the file, its profile and checksum to the PRIVATE
# HF repo under the published names, each checked by the API. HF_TOKEN arrives on stdin, never on disk.
set -uo pipefail
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
read -r HF_TOKEN; export HF_TOKEN
PY=/root/fc/venv/bin/python; B=/root/fc/cortiq.f26; port=8091; n=rel_fcaq256
rm -f out/fcaqhot.cmf
setsid env CMF_QWEN_PROFILE=out/fcaq256.profile $B serve out/fcaq256.cmf --host 127.0.0.1 --port $port > logs/srv_$n.log 2>&1 &
sp=$!
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > logs/srv_$n.mem &
mp=$!
ok=0
for i in $(seq 1 360); do curl -s -m 5 http://127.0.0.1:$port/v1/models > /dev/null && { ok=1; break; }; kill -0 $sp 2>/dev/null || break; sleep 5; done
if [ $ok = 1 ]; then
  for r in 1 2; do
    while read -r l; do log "rel r$r $l"; done < <($PY -I speedtest.py http://127.0.0.1:$port ${n}_r$r res/${n}_r${r}_speed.json 1000,4000,16000,32000 2>&1 | grep -a "ctx ")
  done
else
  log "rel SERVER_FAILED $(tail -3 logs/srv_$n.log | tr '\n' ' ')"
fi
kill -- -$sp 2>/dev/null; sleep 8; kill -9 -- -$sp 2>/dev/null; kill $mp 2>/dev/null
log "rel peak VRAM $(sort -n logs/srv_$n.mem | tail -1) MiB"
REPO=infosave/Qwen3.8-Flash-Next-Fast-Code-cmf
mkdir -p pub
sha=$(sha256sum out/fcaq256.cmf | cut -d' ' -f1)
echo "$sha  qwen38-flash-next-fast-code.cmf" > pub/qwen38-flash-next-fast-code.cmf.sha256
cp out/fcaq256.profile pub/fast-code.profile
up() { # LOCAL PATH_IN_REPO
  local sz sha
  sz=$(stat -c %s "$1"); sha=$(sha256sum "$1" | cut -d' ' -f1)
  nice -n 10 /root/fc/venv/bin/hf upload $REPO "$1" "$2" --repo-type model --private > logs/upr_$(basename "$2").log 2>&1 || { log "UPR_FAILED $2"; return 1; }
  $PY -I - "$2" "$sha" "$sz" <<'PY'
import sys
from huggingface_hub import HfApi
p, sha, sz = sys.argv[1], sys.argv[2], int(sys.argv[3])
i = HfApi().get_paths_info("infosave/Qwen3.8-Flash-Next-Fast-Code-cmf", [p], expand=True)
ok = bool(i) and i[0].size == sz and (i[0].lfs.sha256 == sha if i[0].lfs else True)
print(("UPR_OK " if ok else "UPR_MISMATCH ") + f"{p} {sz} {sha[:16]} remote={(i[0].size, i[0].lfs.sha256[:16] if i[0].lfs else 'git') if i else None}")
PY
}
log "$(up pub/fast-code.profile fast-code.profile)"
log "$(up pub/qwen38-flash-next-fast-code.cmf.sha256 qwen38-flash-next-fast-code.cmf.sha256)"
log "$(up out/fcaq256.cmf qwen38-flash-next-fast-code.cmf)"
log FIN2_DONE
