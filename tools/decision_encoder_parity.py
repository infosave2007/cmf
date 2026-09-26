#!/usr/bin/env python3
"""Reference data for the local encoder gates E1–E5 (spec decision-v4 §1.6), read by
the ignored tests in crates/cortiq-decision/tests/encoder_real.rs.

    python3 tools/decision_encoder_parity.py hf-ids --tokenizer-dir DIR --out OUT
        HF `tokenizers` ids (tokenizer.json of the release encoder, truncation 512) of
        every benchmark text: train/dev/calibration/test of BANKING77, CLINC150 and
        MASSIVE, CLINC150 oos and latency, MASSIVE latency, plus the stress set of
        tools/mk_decision_toy.py. One file per source, `OUT/{ds}.{split}.json`:
        {"source", "sha256", "n", "tokenizers", "ids": [[...], ...]}; `OUT/index.json`
        lists them. Every read of a test, oos or latency file is appended to
        artifacts/decision-v4-20260926/test-access.log first.

    python3 tools/decision_encoder_parity.py unicode-probe --out FILE
        Every Unicode scalar value c through HF `BertNormalizer` (uncased) and
        `BertPreTokenizer` ("a" + c + "a"): one line per scalar whose normalization is
        not c itself or which the pre-tokenizer removes (whitespace) or isolates
        (punctuation): "HEX<TAB>class<TAB>normalized as space-separated HEX". class is
        w (removed), p (isolated) or o (kept inside the word).

The ids of the stored features come from HF `tokenizers` 0.22.2; the tool refuses
another version unless --any-version is given.

The benchmark splits are read from --artifacts-dir (default: $CMFPUBLIC/artifacts,
where CMFPUBLIC defaults to the main checkout of this repository, which holds the
local artifacts/) and every sealed read is logged to --access-log (default:
ARTIFACTS/decision-v4-20260926/test-access.log).
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_cmfpublic():
    """$CMFPUBLIC, else the main checkout of this repository (a worktree's common git dir)."""
    if os.environ.get("CMFPUBLIC"):
        return os.environ["CMFPUBLIC"]
    try:
        common = subprocess.run(["git", "-C", REPO, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True, check=True).stdout.strip()
        return os.path.dirname(common)
    except (OSError, subprocess.CalledProcessError):
        return REPO


# Set from the command line (main).
ART = None
ACCESS_LOG = None
SOURCES = [
    ("banking77", "decision-v2-20260926/splits/banking77", ["train", "dev", "calibration", "test"]),
    ("clinc150", "decision-clinc150-20260925/data", ["train", "dev", "calibration", "test", "oos", "latency"]),
    ("massive", "decision-massive-20260926/data", ["train", "dev", "calibration", "test", "latency"]),
]
SEALED = {"test", "oos", "latency"}
EXPECTED_TOKENIZERS = "0.22.2"


def log_access(path, purpose):
    line = {"utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "file": path, "purpose": purpose}
    with open(ACCESS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def read_texts(path):
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                texts.append(json.loads(line)["text"])
    return texts


def cmd_hf_ids(a):
    import tokenizers
    from tokenizers import Tokenizer

    if tokenizers.__version__ != EXPECTED_TOKENIZERS and not a.any_version:
        sys.exit(f"tokenizers {tokenizers.__version__} (the features were made with {EXPECTED_TOKENIZERS})")
    tok = Tokenizer.from_file(os.path.join(a.tokenizer_dir, "tokenizer.json"))
    tok.no_padding()
    tr = tok.truncation
    if not tr or tr["max_length"] != 512 or tr["direction"] != "right":
        sys.exit(f"unexpected truncation {tr}")
    os.makedirs(a.out, exist_ok=True)
    index = []
    for ds, rel, splits in SOURCES:
        for split in splits:
            path = os.path.join(ART, rel, f"{split}.jsonl")
            if split in SEALED:
                log_access(path, f"decision-v4 WP3 gate E1: HF tokenizers {tokenizers.__version__} "
                                 f"reference ids (tools/decision_encoder_parity.py hf-ids)")
            raw = open(path, "rb").read()
            texts = read_texts(path)
            ids = [e.ids for e in tok.encode_batch(texts, add_special_tokens=True)]
            name = f"{ds}.{split}.json"
            with open(os.path.join(a.out, name), "w", encoding="utf-8") as f:
                json.dump({"source": path, "sha256": hashlib.sha256(raw).hexdigest(), "n": len(texts),
                           "tokenizers": tokenizers.__version__, "ids": ids}, f, separators=(",", ":"))
            index.append({"dataset": ds, "split": split, "file": name, "n": len(texts), "sealed": split in SEALED})
            print(f"{name}: {len(texts)} texts, max {max(len(i) for i in ids)} tokens")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mk_decision_toy as toy
    stress = toy.STRESS + toy.long_texts()
    with open(os.path.join(a.out, "stress.json"), "w", encoding="utf-8") as f:
        json.dump({"source": "tools/mk_decision_toy.py STRESS + long_texts()", "n": len(stress),
                   "tokenizers": tokenizers.__version__, "texts": stress,
                   "ids": [tok.encode(t).ids for t in stress]}, f, ensure_ascii=False)
    index.append({"dataset": "stress", "split": "stress", "file": "stress.json", "n": len(stress), "sealed": False})
    with open(os.path.join(a.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"tokenizers": tokenizers.__version__,
                   "tokenizer_json_sha256": hashlib.sha256(
                       open(os.path.join(a.tokenizer_dir, "tokenizer.json"), "rb").read()).hexdigest(),
                   "files": index}, f, indent=1)
    print(f"wrote {a.out}/index.json ({len(index)} files)")


def cmd_unicode_probe(a):
    import tokenizers
    from tokenizers import normalizers, pre_tokenizers

    if tokenizers.__version__ != EXPECTED_TOKENIZERS and not a.any_version:
        sys.exit(f"tokenizers {tokenizers.__version__} (expected {EXPECTED_TOKENIZERS})")
    norm = normalizers.BertNormalizer(clean_text=True, handle_chinese_chars=True, strip_accents=None,
                                      lowercase=True)
    pre = pre_tokenizers.BertPreTokenizer()
    lines = 0
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(f"# HF tokenizers {tokenizers.__version__}: BertNormalizer(uncased) of c; "
                f"BertPreTokenizer class of 'a'+c+'a' (w removed, p isolated, o kept)\n")
        for cp in range(0x110000):
            if 0xD800 <= cp <= 0xDFFF:
                continue
            c = chr(cp)
            n = norm.normalize_str(c)
            pieces = [p for p, _ in pre.pre_tokenize_str("a" + c + "a")]
            if pieces == ["a", "a"]:
                cls = "w"
            elif pieces == ["a", c, "a"]:
                cls = "p"
            elif pieces == ["a" + c + "a"]:
                cls = "o"
            else:
                sys.exit(f"U+{cp:04X}: unexpected pre-tokenization {pieces!r}")
            if n != c or cls != "o":
                f.write(f"{cp:X}\t{cls}\t{' '.join(f'{ord(x):X}' for x in n)}\n")
                lines += 1
    print(f"wrote {a.out}: {lines} scalars differ from identity/kept")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("hf-ids")
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--any-version", action="store_true")
    p = sub.add_parser("unicode-probe")
    p.add_argument("--out", required=True)
    p.add_argument("--any-version", action="store_true")
    for p in sub.choices.values():
        p.add_argument("--artifacts-dir", default=None,
                       help="the local artifacts/ directory (default: $CMFPUBLIC/artifacts)")
        p.add_argument("--access-log", default=None,
                       help="test-split access log (default: ARTIFACTS/decision-v4-20260926/test-access.log)")
    a = ap.parse_args()
    global ART, ACCESS_LOG
    ART = a.artifacts_dir or os.path.join(default_cmfpublic(), "artifacts")
    ACCESS_LOG = a.access_log or os.path.join(ART, "decision-v4-20260926", "test-access.log")
    {"hf-ids": cmd_hf_ids, "unicode-probe": cmd_unicode_probe}[a.cmd](a)


if __name__ == "__main__":
    main()
