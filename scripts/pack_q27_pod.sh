#!/bin/bash
# Pack what a fresh pod needs for scripts/pod_setup_q27.sh: the teacher data, and — only when the
# checkout differs from the release tag — its changed files (tracked and new), which make the pod
# build cortiq instead of downloading the tag's binary. Usage: pack_q27_pod.sh OUT_DIR TDATA
set -euo pipefail
OUT=${1:?out dir}
TDATA=${2:?teacher data .tdata dir}
TAG=${TAG:-v0.8.42}
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
cd "$REPO"
{ git diff --name-only "$TAG"; git ls-files --others --exclude-standard; } | sort -u | while read -r f; do [ -f "$f" ] && echo "$f"; done > "$OUT/changed.txt"
rm -f "$OUT/cmf-changes.tgz"
[ -s "$OUT/changed.txt" ] && COPYFILE_DISABLE=1 tar czf "$OUT/cmf-changes.tgz" -T "$OUT/changed.txt"
COPYFILE_DISABLE=1 tar czf "$OUT/q27-code-t27.tdata.tgz" -C "$(dirname "$TDATA")" "$(basename "$TDATA")"
cp scripts/pod_setup_q27.sh "$OUT/"
ls -la "$OUT"
echo "files over $TAG:"; cat "$OUT/changed.txt"
echo "then: scp -P PORT $OUT/*.tgz $OUT/pod_setup_q27.sh root@HOST:/root/q27/ && ssh … 'bash /root/q27/pod_setup_q27.sh'"
