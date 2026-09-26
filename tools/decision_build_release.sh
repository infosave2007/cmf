#!/usr/bin/env bash
# Build the Cortiq Decision files by the documented commands (spec decision-v4
# §3.8 and §3.9), twice, and publish them only when both builds are
# byte-identical.
#
#   tools/decision_build_release.sh [--out DIR] [--work DIR] [--keep-work]
#                                   [--skip-cargo] [--threads-a N] [--threads-b N]
#
# Every build (run "a" and run "b", each from scratch in its own directory):
#
#   python3 tools/decision_export_encoder.py --onnx $ENC_SRC/encoder.onnx \
#       --tokenizer-dir $ENC_SRC/encoder_tokenizer --out encoder-export
#   cortiq decision init --encoder-dir encoder-export -o enc.cmf
#
#   reproduction model (§3.8: train only, K = 16, dev evaluated):
#   cortiq decision add-skill enc.cmf --skill banking77 --train $B/train.jsonl \
#       --calibration $B/calibration.jsonl --dev $B/dev.jsonl --question $B/question.json -o s1.cmf
#   cortiq decision add-skill s1.cmf  --skill clinc150  ... $C ... -o s2.cmf
#   cortiq decision add-skill s2.cmf  --skill massive   ... $M ... -o cortiq-decision.cmf
#
#   published model (§3.9: topologies on train ∪ dev, K per skill = chosen_K of
#   max-recipe/cv.json, calibration only for the gate, no test anywhere):
#   cortiq decision add-skill enc.cmf --skill banking77 --train $B/train.jsonl --train $B/dev.jsonl \
#       --calibration $B/calibration.jsonl --question $B/question.json --k $K_B --k-source "…" -o s1.cmf
#   … clinc150, massive the same way … -o cortiq-decision.cmf
#
#   cortiq decision verify <each final file>
#
# SOURCE_DATE_EPOCH is fixed (default 1790380800 = 2026-09-26T00:00:00Z), so the
# bytes do not depend on the clock; run "a" uses all cores and run "b" 4 threads,
# so they also show that the bytes do not depend on the thread count. The
# encoder export, the encoder-only file and both final files must have equal
# sha256 in the two runs, otherwise nothing is published and the script fails.
#
# Published (never overwritten; an existing file with another sha256 is an error):
#   $OUT/cortiq-decision-base.cmf          the reproduction model (§3.8, gates §6.2)
#   $OUT/release/cortiq-decision.cmf       the published model (§3.9)
#   $OUT/build/build-release.json          commands, inputs and their sha256, K per
#                                          skill, both runs' sha256, timings
#   $OUT/build/run-{a,b}/*.json            the JSON reports of init/add-skill/verify
#
# The work directory (default: a new directory under $TMPDIR) holds the
# intermediate files (≈1.5 GB per run) and is removed at the end unless
# --keep-work is given. No test split is read by this script. No GPU backend is
# initialised (CMF_GPU=0) and there is no network access.
#
# Environment (defaults):
#   CMFPUBLIC   /Users/oleg/dev/cmfpublic
#   ENC_SRC     /Users/oleg/Documents/cortiq-bot/cortiq-router/registry_bake
#   B, C, M     the BANKING77 / CLINC150 / MASSIVE split directories of spec §3.8
#   CV          $CMFPUBLIC/reports/decision-v4-20260926/max-recipe/cv.json (K = 16
#               for every skill when it is missing; recorded in build-release.json)
#   CORTIQ      <repo>/target/release/cortiq (built with `cargo build --release
#               -p cortiq-cli --offline` unless --skip-cargo)
#   PYTHON      python3 (numpy, onnx, onnxruntime for the encoder export)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CMFPUBLIC="${CMFPUBLIC:-/Users/oleg/dev/cmfpublic}"
ENC_SRC="${ENC_SRC:-/Users/oleg/Documents/cortiq-bot/cortiq-router/registry_bake}"
B="${B:-$CMFPUBLIC/artifacts/decision-v2-20260926/splits/banking77}"
C="${C:-$CMFPUBLIC/artifacts/decision-clinc150-20260925/data}"
M="${M:-$CMFPUBLIC/artifacts/decision-massive-20260926/data}"
CV="${CV:-$CMFPUBLIC/reports/decision-v4-20260926/max-recipe/cv.json}"
OUT="${OUT:-$CMFPUBLIC/artifacts/decision-v4-20260926}"
CORTIQ="${CORTIQ:-$REPO/target/release/cortiq}"
PYTHON="${PYTHON:-python3}"
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1790380800}"
export CMF_GPU=0

WORK=""
KEEP_WORK=0
SKIP_CARGO=0
THREADS_A=0
THREADS_B=4
while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        --work) WORK="$2"; shift 2 ;;
        --keep-work) KEEP_WORK=1; shift ;;
        --skip-cargo) SKIP_CARGO=1; shift ;;
        --threads-a) THREADS_A="$2"; shift 2 ;;
        --threads-b) THREADS_B="$2"; shift 2 ;;
        -h|--help) sed -n '2,62p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

SKILLS=(banking77 clinc150 massive)
skill_dir() {
    case "$1" in
        banking77) echo "$B" ;;
        clinc150) echo "$C" ;;
        massive) echo "$M" ;;
    esac
}

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }
sha256() { shasum -a 256 "$1" | awk '{print $1}'; }
now() { "$PYTHON" -c 'import time; print(f"{time.time():.3f}")'; }

for f in "$ENC_SRC/encoder.onnx" "$ENC_SRC/encoder_tokenizer/tokenizer.json" "$ENC_SRC/encoder_tokenizer/vocab.txt"; do
    [ -f "$f" ] || die "missing encoder source $f"
done
for s in "${SKILLS[@]}"; do
    d="$(skill_dir "$s")"
    for f in train.jsonl calibration.jsonl dev.jsonl question.json; do
        [ -f "$d/$f" ] || die "missing $d/$f"
    done
done

if [ -z "$WORK" ]; then
    WORK="$(mktemp -d "${TMPDIR:-/tmp}/cortiq-decision-build.XXXXXX")"
    CREATED_WORK=1
else
    [ -e "$WORK" ] && die "work directory $WORK exists (give a new one)"
    mkdir -p "$WORK"
    CREATED_WORK=1
fi
cleanup() {
    if [ "$KEEP_WORK" = 0 ] && [ "${CREATED_WORK:-0}" = 1 ] && [ -d "$WORK" ]; then
        rm -rf "$WORK"
    fi
}
trap cleanup EXIT
log "work directory $WORK (SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH)"

# ------------------------------------------------------------------ binary
if [ "$SKIP_CARGO" = 0 ]; then
    log "cargo build --release -p cortiq-cli --offline"
    (cd "$REPO" && cargo build --release -p cortiq-cli --offline) >"$WORK/cargo-build.log" 2>&1 \
        || { tail -30 "$WORK/cargo-build.log" >&2; die "cargo build failed"; }
fi
[ -x "$CORTIQ" ] || die "no cortiq binary at $CORTIQ"

# ------------------------------------------------------------------ K per skill (§3.9)
K_NOTE=""
declare -a RELEASE_K
if [ -f "$CV" ]; then
    CV_SHA="$(sha256 "$CV")"
    for i in "${!SKILLS[@]}"; do
        RELEASE_K[$i]="$("$PYTHON" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["chosen_K"][sys.argv[2]]))' "$CV" "${SKILLS[$i]}")"
    done
    K_SOURCE="reports/decision-v4-20260926/max-recipe/cv.json chosen_K (sha256 $CV_SHA)"
else
    CV_SHA=""
    for i in "${!SKILLS[@]}"; do RELEASE_K[$i]=16; done
    K_SOURCE="default K=16 (max-recipe/cv.json missing)"
    K_NOTE="cv.json missing: K=16 for every skill of the published model"
    log "WARNING: $K_NOTE"
fi

# ------------------------------------------------------------------ one build
build_run() {
    local tag="$1" threads="$2"
    local W="$WORK/run-$tag"
    mkdir -p "$W/logs" "$W/base" "$W/release"
    local t0; t0="$(now)"
    log "run $tag: encoder export"
    "$PYTHON" "$REPO/tools/decision_export_encoder.py" --onnx "$ENC_SRC/encoder.onnx" \
        --tokenizer-dir "$ENC_SRC/encoder_tokenizer" --out "$W/encoder-export" >"$W/logs/export.log" 2>&1 \
        || { tail -30 "$W/logs/export.log" >&2; die "encoder export failed (run $tag)"; }
    cp "$W/encoder-export/encoder.json" "$W/logs/encoder-export.json"
    log "run $tag: cortiq decision init"
    "$CORTIQ" decision init --encoder-dir "$W/encoder-export" -o "$W/enc.cmf" --json >"$W/logs/init.json"

    local prev="$W/enc.cmf" i s d out
    for i in "${!SKILLS[@]}"; do
        s="${SKILLS[$i]}"; d="$(skill_dir "$s")"
        if [ "$i" = 2 ]; then out="$W/base/cortiq-decision.cmf"; else out="$W/base/s$((i + 1)).cmf"; fi
        log "run $tag: reproduction add-skill $s (threads $threads)"
        "$CORTIQ" decision add-skill "$prev" --skill "$s" --train "$d/train.jsonl" \
            --calibration "$d/calibration.jsonl" --dev "$d/dev.jsonl" --question "$d/question.json" \
            --threads "$threads" --json -o "$out" >"$W/logs/base-$s.json"
        prev="$out"
    done
    prev="$W/enc.cmf"
    for i in "${!SKILLS[@]}"; do
        s="${SKILLS[$i]}"; d="$(skill_dir "$s")"
        if [ "$i" = 2 ]; then out="$W/release/cortiq-decision.cmf"; else out="$W/release/s$((i + 1)).cmf"; fi
        log "run $tag: published add-skill $s K=${RELEASE_K[$i]} on train ∪ dev (threads $threads)"
        "$CORTIQ" decision add-skill "$prev" --skill "$s" --train "$d/train.jsonl" --train "$d/dev.jsonl" \
            --calibration "$d/calibration.jsonl" --question "$d/question.json" \
            --k "${RELEASE_K[$i]}" --k-source "$K_SOURCE" \
            --threads "$threads" --json -o "$out" >"$W/logs/release-$s.json"
        prev="$out"
    done
    log "run $tag: cortiq decision verify"
    "$CORTIQ" decision verify "$W/base/cortiq-decision.cmf" --json >"$W/logs/verify-base.json"
    "$CORTIQ" decision verify "$W/release/cortiq-decision.cmf" --json >"$W/logs/verify-release.json"
    "$CORTIQ" decision info "$W/base/cortiq-decision.cmf" --json >"$W/logs/info-base.json"
    "$CORTIQ" decision info "$W/release/cortiq-decision.cmf" --json >"$W/logs/info-release.json"
    local t1; t1="$(now)"
    {
        echo "threads=$threads"
        echo "seconds=$("$PYTHON" -c "print(round($t1 - $t0, 3))")"
        echo "export_encoder_json=$(sha256 "$W/encoder-export/encoder.json")"
        echo "enc=$(sha256 "$W/enc.cmf")"
        echo "base=$(sha256 "$W/base/cortiq-decision.cmf")"
        echo "release=$(sha256 "$W/release/cortiq-decision.cmf")"
    } >"$W/logs/sha256.txt"
    # The intermediate files are not needed any more.
    rm -rf "$W/encoder-export" "$W/base/s1.cmf" "$W/base/s2.cmf" "$W/release/s1.cmf" "$W/release/s2.cmf"
    log "run $tag: done in $(grep '^seconds=' "$W/logs/sha256.txt" | cut -d= -f2) s"
}

build_run a "$THREADS_A"
build_run b "$THREADS_B"

for key in enc base release; do
    ha="$(grep "^$key=" "$WORK/run-a/logs/sha256.txt" | cut -d= -f2)"
    hb="$(grep "^$key=" "$WORK/run-b/logs/sha256.txt" | cut -d= -f2)"
    [ "$ha" = "$hb" ] || die "$key differs between the two builds: $ha (run a) vs $hb (run b)"
    log "$key: sha256 $ha in both runs"
done

# ------------------------------------------------------------------ publish (no clobber)
publish() {
    local src="$1" dst="$2" want
    want="$(sha256 "$src")"
    mkdir -p "$(dirname "$dst")"
    if [ -e "$dst" ]; then
        [ "$(sha256 "$dst")" = "$want" ] || die "$dst exists with another sha256; refusing to overwrite it"
        log "$dst already present (same sha256)"
    else
        cp -n "$src" "$dst.partial.$$"
        [ "$(sha256 "$dst.partial.$$")" = "$want" ] || die "copy of $src is corrupt"
        mv -n "$dst.partial.$$" "$dst"
        if [ -e "$dst.partial.$$" ]; then
            rm -f "$dst.partial.$$"
            die "$dst appeared during the copy; refusing to overwrite it"
        fi
        log "published $dst"
    fi
}
publish "$WORK/run-a/base/cortiq-decision.cmf" "$OUT/cortiq-decision-base.cmf"
publish "$WORK/run-a/release/cortiq-decision.cmf" "$OUT/release/cortiq-decision.cmf"

mkdir -p "$OUT/build"
for tag in a b; do
    [ -e "$OUT/build/run-$tag" ] && rm -rf "$OUT/build/run-$tag"
    mkdir -p "$OUT/build/run-$tag"
    cp "$WORK/run-$tag/logs/"*.json "$WORK/run-$tag/logs/sha256.txt" "$WORK/run-$tag/logs/export.log" "$OUT/build/run-$tag/"
done

"$PYTHON" - "$OUT" "$WORK" "$REPO" "$CORTIQ" "$ENC_SRC" "$B" "$C" "$M" "$CV" "$CV_SHA" "$K_SOURCE" "$K_NOTE" \
    "${RELEASE_K[0]}" "${RELEASE_K[1]}" "${RELEASE_K[2]}" <<'PY'
import datetime, hashlib, json, os, subprocess, sys
(out, work, repo, cortiq, enc_src, b, c, m, cv, cv_sha, k_source, k_note, kb, kc, km) = sys.argv[1:]
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()
def kv(p):
    return dict(l.strip().split('=', 1) for l in open(p) if '=' in l)
def git(*a):
    try:
        return subprocess.run(['git', '-C', repo, *a], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None
dirs = {'banking77': b, 'clinc150': c, 'massive': m}
inputs = {}
for s, d in dirs.items():
    inputs[s] = {f: {'path': os.path.join(d, f), 'sha256': sha(os.path.join(d, f))}
                 for f in ('train.jsonl', 'calibration.jsonl', 'dev.jsonl', 'question.json')}
runs = {}
for tag in ('a', 'b'):
    r = kv(os.path.join(work, 'run-' + tag, 'logs', 'sha256.txt'))
    runs[tag] = {'threads': int(r['threads']), 'seconds': float(r['seconds']),
                 'sha256': {k: r[k] for k in ('export_encoder_json', 'enc', 'base', 'release')}}
def report(tag, name):
    return json.load(open(os.path.join(work, 'run-' + tag, 'logs', name)))
base_file = os.path.join(out, 'cortiq-decision-base.cmf')
rel_file = os.path.join(out, 'release', 'cortiq-decision.cmf')
version = subprocess.run([cortiq, '--version'], capture_output=True, text=True).stdout.strip()
doc = {
    'schema': 'cortiq-decision-v4-build/1',
    'utc': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
    'script': 'tools/decision_build_release.sh',
    'commit': git('rev-parse', 'HEAD'),
    'worktree_clean': git('status', '--porcelain') == '',
    'binary': {'path': cortiq, 'sha256': sha(cortiq), 'version': version},
    'source_date_epoch': int(os.environ['SOURCE_DATE_EPOCH']),
    'encoder_source': {f: {'path': os.path.join(enc_src, f), 'sha256': sha(os.path.join(enc_src, f))}
                       for f in ('encoder.onnx', 'encoder_tokenizer/tokenizer.json', 'encoder_tokenizer/vocab.txt')},
    'inputs': inputs,
    'reproduction': {
        'spec': '§3.8: train only, K = 16, calibration file, dev evaluated',
        'commands': [f'cortiq decision add-skill {"enc.cmf" if i == 0 else f"s{i}.cmf"} --skill {s} --train {d}/train.jsonl '
                     f'--calibration {d}/calibration.jsonl --dev {d}/dev.jsonl --question {d}/question.json '
                     f'-o {"cortiq-decision.cmf" if i == 2 else f"s{i + 1}.cmf"}'
                     for i, (s, d) in enumerate(dirs.items())],
        'file': base_file, 'bytes': os.path.getsize(base_file), 'sha256': sha(base_file),
        'model_sha': report('a', 'base-massive.json')['out']['model_sha'],
        'skills': {s: report('a', f'base-{s}.json') for s in dirs},
    },
    'published': {
        'spec': '§3.9: topologies on train ∪ dev, K per skill from max-recipe/cv.json, calibration for the gate only, test unused',
        'k': {'banking77': int(kb), 'clinc150': int(kc), 'massive': int(km)},
        'k_source': k_source, 'cv_json': {'path': cv, 'sha256': cv_sha or None}, 'note': k_note or None,
        'commands': [f'cortiq decision add-skill {"enc.cmf" if i == 0 else f"s{i}.cmf"} --skill {s} --train {d}/train.jsonl '
                     f'--train {d}/dev.jsonl --calibration {d}/calibration.jsonl --question {d}/question.json '
                     f'--k {k} --k-source "{k_source}" -o {"cortiq-decision.cmf" if i == 2 else f"s{i + 1}.cmf"}'
                     for i, ((s, d), k) in enumerate(zip(dirs.items(), (kb, kc, km)))],
        'file': rel_file, 'bytes': os.path.getsize(rel_file), 'sha256': sha(rel_file),
        'model_sha': report('a', 'release-massive.json')['out']['model_sha'],
        'skills': {s: report('a', f'release-{s}.json') for s in dirs},
    },
    'runs': runs,
    'reproducible': all(runs['a']['sha256'][k] == runs['b']['sha256'][k] for k in ('enc', 'base', 'release')),
    'verify': {'base': report('a', 'verify-base.json'), 'release': report('a', 'verify-release.json')},
}
dst = os.path.join(out, 'build', 'build-release.json')
tmp = dst + '.tmp'
with open(tmp, 'w') as f:
    json.dump(doc, f, indent=1, ensure_ascii=False)
    f.write('\n')
os.replace(tmp, dst)
print(json.dumps({'reproducible': doc['reproducible'], 'base': {'sha256': doc['reproduction']['sha256'], 'bytes': doc['reproduction']['bytes']},
                  'release': {'sha256': doc['published']['sha256'], 'bytes': doc['published']['bytes']}, 'report': dst}))
PY
log "done"
