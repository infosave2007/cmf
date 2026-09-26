#!/usr/bin/env bash
# Assemble the Hugging Face upload of infosave/cortiq-decision (release 0.7.8):
# a directory with exactly five files —
#
#   cortiq-decision.cmf   the published model, copied unchanged
#   README.md API.md ORACLE.md   from docs/decision/hf/
#   SHA256SUMS            sha256 of the other four (`shasum -a 256 -c SHA256SUMS`)
#
# Usage:
#   tools/decision_hf_bundle.sh OUT_DIR --model PATH [--sha256 HEX]
#   CORTIQ_DECISION_MODEL=PATH tools/decision_hf_bundle.sh OUT_DIR
#
#   OUT_DIR   created if missing; must be empty (nothing is overwritten)
#   --model   the model file (required: --model or $CORTIQ_DECISION_MODEL;
#             there is no default path)
#   --sha256  the model's expected sha256 (default: the published 0.7.8 file's);
#             the bundle is refused when the copy does not match it
#
# The model is only read. The script checks the copy against the expected
# sha256, writes SHA256SUMS, re-verifies it and refuses to finish unless the
# directory holds exactly the five files.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DOCS="$REPO/docs/decision/hf"
MODEL="${CORTIQ_DECISION_MODEL:-}"
EXPECT_SHA=ed9b8ec2bbfe9e9fd30f14a5eaf82314f38bc7e7510a39772baa2de3801d79b1
FILES=(cortiq-decision.cmf README.md API.md ORACLE.md)

die() { echo "decision_hf_bundle: $*" >&2; exit 1; }

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

sums_check() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum -c SHA256SUMS
    else
        shasum -a 256 -c SHA256SUMS
    fi
}

OUT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --model) [ $# -ge 2 ] || die "--model needs a path"; MODEL=$2; shift 2 ;;
        --sha256) [ $# -ge 2 ] || die "--sha256 needs a value"; EXPECT_SHA=$2; shift 2 ;;
        -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
        -*) die "unknown option $1" ;;
        *) [ -z "$OUT" ] || die "one output directory only"; OUT=$1; shift ;;
    esac
done
USAGE="usage: tools/decision_hf_bundle.sh OUT_DIR --model PATH [--sha256 HEX]"
[ -n "$OUT" ] || die "$USAGE"
[ -n "$MODEL" ] || die "no model: pass --model PATH or set CORTIQ_DECISION_MODEL ($USAGE)"
[ -f "$MODEL" ] || die "model not found: $MODEL"
for f in README.md API.md ORACLE.md; do
    [ -s "$DOCS/$f" ] || die "missing $DOCS/$f"
done
if [ -e "$OUT" ]; then
    [ -d "$OUT" ] || die "$OUT exists and is not a directory"
    [ -z "$(ls -A "$OUT")" ] || die "$OUT is not empty; refusing to overwrite"
else
    mkdir -p "$OUT"
fi

echo "model:  $MODEL"
cp "$MODEL" "$OUT/cortiq-decision.cmf"
got=$(sha256_of "$OUT/cortiq-decision.cmf")
[ "$got" = "$EXPECT_SHA" ] || {
    rm -f "$OUT/cortiq-decision.cmf"
    die "model sha256 $got, expected $EXPECT_SHA"
}
for f in README.md API.md ORACLE.md; do
    cp "$DOCS/$f" "$OUT/$f"
done

cd "$OUT"
: > SHA256SUMS.tmp
for f in "${FILES[@]}"; do
    printf '%s  %s\n' "$(sha256_of "$f")" "$f" >> SHA256SUMS.tmp
done
mv SHA256SUMS.tmp SHA256SUMS
sums_check

n=$(ls -A | wc -l | tr -d ' ')
[ "$n" = 5 ] || die "expected 5 files in $OUT, found $n"
echo "bundle: $(pwd) (5 files)"
cat SHA256SUMS
