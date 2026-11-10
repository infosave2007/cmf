#!/bin/bash
# chainF.sh: the empty answer at 1K. The speed client's 1K request (fixed tag) through `cortiq run`, greedy, on fcq2:
# default (resident, MTP on), MTP off, the streamed path, and fcaq256; the first 300 characters of each answer.
set -uo pipefail
cd /root/fc; J=/root/fc/JOURNAL.md; export XDG_RUNTIME_DIR=/tmp
log() { echo "$(date +%T) $*" | tee -a $J; }
for i in $(seq 1 720); do grep -q CHAINB_DONE $J && break; sleep 10; done
python3 - <<'PY'
code = open("/root/fc/data/eval_code.txt").read()
for ctx in (1000, 4000):
    nchar = int(ctx * 3.4)
    body = (code * (1 + nchar // len(code)))[:nchar]
    msg = ("[session 0badc0de]\n" + "Here is a code base:\n\n" + body + "\n\nWrite a new Python module `repo_summary.py`: a function `summarize_repo(files: dict[str, str]) -> str` "
           "that lists every function and class defined in these files with a short docstring, then pytest unit tests for it. Answer with code only.")
    open(f"/root/fc/data/p_speed{ctx}.txt", "w").write(msg)
PY
B=/root/fc/cortiq.s2; [ -x $B ] || B=/root/fc/cortiq.s1
one() { local name=$1 file=$2 p=$3; shift 3
  env CMF_QWEN_PROFILE=$p "$@" timeout 900 $B run $file --prompt-file data/p_speed1000.txt -n 120 --temperature 0 --no-think > logs/diag_$name.txt 2> logs/diag_$name.err
  log "diag $name rc=$? | $(grep -a -v -E '^(Loading|Ready|Prompt|<\||Here is|\[session)' logs/diag_$name.txt | tr '\n' ' ' | tail -c 420)"
}
one def out/fcq2.cmf out/fcq2.profile
one nomtp out/fcq2.cmf out/fcq2.profile CMF_QWEN_MTP=0
one stream out/fcq2.cmf out/fcq2.profile CMF_QWEN_RESIDENT=0
one fcaq out/fcaq256.cmf out/fcaq256.profile
log CHAINF_DONE
