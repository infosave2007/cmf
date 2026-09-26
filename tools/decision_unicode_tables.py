#!/usr/bin/env python3
"""Generate crates/cortiq-decision/src/unicode_tables.rs (spec decision-v4 §1.3).

The WordPiece normalizer and pre-tokenizer of HF `tokenizers` query Unicode
categories through the `unicode_categories` crate and decompose through
`unicode-normalization-alignments`. Both are old (Unicode 9.0 tables), so a
copy of the HF pipeline has to use *those* tables, not the ones of a current
Unicode release: a character added after 9.0 is simply unassigned for HF — it
is no `Mn`, no punctuation, no control character and it has no canonical
decomposition.

This script reads the versions HF `tokenizers` 0.21.4 locks (its Cargo.lock in
~/.cargo/registry) and turns the category tables of that `unicode_categories`
into inclusive code-point ranges:

* ``OTHER``           Cc ∪ Cf ∪ Co — ``UnicodeCategories::is_other`` (no Cn:
                      the crate's ``is_other`` does not include it);
* ``PUNCTUATION``     Pc ∪ Pd ∪ Ps ∪ Pe ∪ Pi ∪ Pf ∪ Po — ``is_punctuation``;
* ``MARK_NONSPACING`` Mn — ``is_mark_nonspacing``;
* ``ASSIGNED``        the union of every category table and range arm: the
                      code points Unicode 9.0 assigns (Cs excluded). The NFD
                      step runs `unicode-normalization` (workspace crate) only
                      on these; every other scalar passes through as a starter,
                      which is what a Unicode 9.0 NFD does with it.

Range arms written as ``match`` arms in the crate's lib.rs (CJK, Hangul,
Tangut, private use) are parsed from the source and merged in.

    python3 tools/decision_unicode_tables.py [--registry DIR] [--out FILE] [--check]

``--check`` exits 1 if the output file differs from what would be generated.
"""

import argparse
import glob
import hashlib
import os
import re
import sys

HF_TOKENIZERS = "tokenizers-0.21.4"
OUT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "crates", "cortiq-decision", "src", "unicode_tables.rs")

OTHER_TABLES = ["OTHER_CONTROL", "OTHER_FORMAT", "OTHER_PRIVATE_USE"]
PUNCT_TABLES = ["PUNCTUATION_CONNECTOR", "PUNCTUATION_DASH", "PUNCTUATION_CLOSE",
                "PUNCTUATION_FINAL_QUOTE", "PUNCTUATION_INITIAL_QUOTE",
                "PUNCTUATION_OTHER", "PUNCTUATION_OPEN"]
MN_TABLES = ["MARK_NONSPACING"]
# The trait methods whose `match` arms add whole ranges on top of the tables.
RANGE_FUNCS = {"is_other_private_use": "OTHER_PRIVATE_USE", "is_letter_other": "LETTER_OTHER"}


def registry_src(registry):
    if registry:
        return registry
    hits = sorted(glob.glob(os.path.expanduser("~/.cargo/registry/src/*/" + HF_TOKENIZERS)))
    if not hits:
        sys.exit(f"{HF_TOKENIZERS} not found in ~/.cargo/registry/src (cargo fetch it first)")
    return os.path.dirname(hits[0])


def locked_version(lock_text, name):
    m = re.search(r'\[\[package\]\]\nname = "' + re.escape(name) + r'"\nversion = "([^"]+)"', lock_text)
    if not m:
        sys.exit(f"{name} is not in the {HF_TOKENIZERS} Cargo.lock")
    return m.group(1)


def parse_tables(text):
    tables = {}
    for m in re.finditer(r"pub static (\w+) : &'static \[char\] = &\[(.*?)\];", text, re.S):
        cps = [int(h, 16) for h in re.findall(r"'\\u\{([0-9A-Fa-f]+)\}'", m.group(2))]
        assert cps == sorted(cps), f"{m.group(1)} is not sorted (the crate binary-searches it)"
        tables[m.group(1)] = cps
    return tables


def parse_range_arms(lib_text):
    """{table name: [(lo, hi)]} from `'\\u{X}'...'\\u{Y}' => true` arms."""
    out = {}
    for fn, table in RANGE_FUNCS.items():
        m = re.search(r"fn " + fn + r"\(self\) -> bool \{(.*?)\n    \}", lib_text, re.S)
        assert m, f"fn {fn} not found in lib.rs"
        arms = re.findall(r"'\\u\{([0-9A-Fa-f]+)\}'\s*\.\.\.\s*'\\u\{([0-9A-Fa-f]+)\}'\s*=>\s*true",
                          m.group(1))
        assert arms, f"no range arms in {fn}"
        out[table] = [(int(a, 16), int(b, 16)) for a, b in arms]
    return out


def to_ranges(cps):
    cps = sorted(set(cps))
    out = []
    for c in cps:
        if out and c == out[-1][1] + 1:
            out[-1][1] = c
        else:
            out.append([c, c])
    return [(a, b) for a, b in out]


def merge_ranges(rs):
    rs = sorted(rs)
    out = []
    for a, b in rs:
        if out and a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def category_ranges(tables, arms, names):
    rs = []
    for n in names:
        rs.extend(to_ranges(tables[n]))
        rs.extend(arms.get(n, []))
    rs = merge_ranges(rs)
    for a, b in rs:
        assert not (a <= 0xDFFF and b >= 0xD800), "surrogates are not scalar values"
    return rs


def fmt_table(name, doc, rs):
    lines = [f"/// {doc}", f"pub static {name}: &[(u32, u32)] = &["]
    for a, b in rs:
        lines.append(f"    (0x{a:04X}, 0x{b:04X}),")
    lines.append("];")
    return "\n".join(lines)


def generate(src):
    tok_dir = os.path.join(src, HF_TOKENIZERS)
    lock = open(os.path.join(tok_dir, "Cargo.lock"), encoding="utf-8").read()
    cat_ver = locked_version(lock, "unicode_categories")
    nfd_ver = locked_version(lock, "unicode-normalization-alignments")
    cat_dir = os.path.join(src, f"unicode_categories-{cat_ver}")
    nfd_dir = os.path.join(src, f"unicode-normalization-alignments-{nfd_ver}")
    tables_path = os.path.join(cat_dir, "src", "tables.rs")
    lib_path = os.path.join(cat_dir, "src", "lib.rs")
    tables_bytes = open(tables_path, "rb").read()
    lib_bytes = open(lib_path, "rb").read()
    tables = parse_tables(tables_bytes.decode("utf-8"))
    arms = parse_range_arms(lib_bytes.decode("utf-8"))
    # The Unicode version of the category tables: Tangut (9.0) is present and the
    # unified CJK block ends at U+9FD5 (9.0; 10.0 extends it to U+9FEA).
    letter_arms = dict((a, b) for a, b in arms["LETTER_OTHER"])
    assert letter_arms.get(0x17000) == 0x187EC and letter_arms.get(0x4E00) == 0x9FD5, \
        "unexpected unicode_categories ranges (not the Unicode 9.0 tables)"
    cat_unicode = "9.0.0"
    nfd_tables = open(os.path.join(nfd_dir, "src", "tables.rs"), encoding="utf-8").read()
    m = re.search(r"UNICODE_VERSION: \(u64, u64, u64\) = \((\d+), (\d+), (\d+)\)", nfd_tables)
    assert m, "UNICODE_VERSION not found in unicode-normalization-alignments"
    nfd_unicode = ".".join(m.groups())
    assert nfd_unicode == cat_unicode, (nfd_unicode, cat_unicode)

    other = category_ranges(tables, arms, OTHER_TABLES)
    punct = category_ranges(tables, arms, PUNCT_TABLES)
    mn = category_ranges(tables, arms, MN_TABLES)
    assigned = category_ranges(tables, arms, sorted(tables))

    source = (f"HF tokenizers 0.21.4 Cargo.lock: unicode_categories {cat_ver} (Unicode {cat_unicode}; "
              f"is_other = Cc|Cf|Co, is_punctuation = P*, Mn) and unicode-normalization-alignments "
              f"{nfd_ver} (NFD, Unicode {nfd_unicode}); NFD here = unicode-normalization "
              f"{workspace_nfd_version()} on Unicode {cat_unicode}-assigned scalars, others pass as "
              f"starters; lowercase, whitespace and ASCII punctuation = Rust std char")
    tables_sha = hashlib.sha256(tables_bytes).hexdigest()
    lib_sha = hashlib.sha256(lib_bytes).hexdigest()

    parts = []
    parts.append(f"""//! Unicode category tables for the WordPiece normalizer and pre-tokenizer
//! (spec decision-v4 §1.3).
//!
//! GENERATED by `tools/decision_unicode_tables.py` — do not edit by hand.
//!
//! HF `tokenizers` (0.21.4 locks `unicode_categories` {cat_ver} and
//! `unicode-normalization-alignments` {nfd_ver}) answers every category question
//! from Unicode {cat_unicode} tables. The native tokenizer has to answer them from
//! the same tables, so they are the crate's own, as inclusive code-point ranges:
//! [`OTHER`] (`is_other` = Cc ∪ Cf ∪ Co; the crate's `is_other` has no Cn),
//! [`PUNCTUATION`] (P*), [`MARK_NONSPACING`] (Mn) and [`ASSIGNED`] (every
//! category, i.e. the scalars Unicode {cat_unicode} assigns). A scalar outside
//! [`ASSIGNED`] is unassigned for HF: no decomposition, combining class 0.
//!
//! Source files: `unicode_categories-{cat_ver}/src/tables.rs` sha256
//! `{tables_sha}`,
//! `src/lib.rs` sha256 `{lib_sha}`.

/// Recorded in the encoder's tokenizer record (`tables`).
pub const SOURCE: &str = "{source}";

/// Unicode version of the category tables (and of HF's NFD tables).
pub const UNICODE_VERSION: (u8, u8, u8) = ({cat_unicode.replace('.', ', ')});

/// sha256 of the `unicode_categories` `src/tables.rs` the ranges come from.
pub const SOURCE_TABLES_SHA256: &str =
    "{tables_sha}";

/// sha256 of the `unicode_categories` `src/lib.rs` (range arms).
pub const SOURCE_LIB_SHA256: &str =
    "{lib_sha}";

#[inline]
fn in_ranges(c: char, table: &[(u32, u32)]) -> bool {{
    let c = c as u32;
    table
        .binary_search_by(|&(lo, hi)| {{
            if hi < c {{
                std::cmp::Ordering::Less
            }} else if lo > c {{
                std::cmp::Ordering::Greater
            }} else {{
                std::cmp::Ordering::Equal
            }}
        }})
        .is_ok()
}}

/// `UnicodeCategories::is_other`: Cc, Cf or Co.
#[inline]
pub fn is_other(c: char) -> bool {{
    in_ranges(c, OTHER)
}}

/// `UnicodeCategories::is_punctuation`: Pc, Pd, Ps, Pe, Pi, Pf or Po.
#[inline]
pub fn is_punctuation(c: char) -> bool {{
    in_ranges(c, PUNCTUATION)
}}

/// `UnicodeCategories::is_mark_nonspacing`: Mn.
#[inline]
pub fn is_mark_nonspacing(c: char) -> bool {{
    in_ranges(c, MARK_NONSPACING)
}}

/// Assigned in Unicode {cat_unicode} (any category of the source tables).
#[inline]
pub fn is_assigned(c: char) -> bool {{
    in_ranges(c, ASSIGNED)
}}""")
    parts.append(fmt_table("OTHER", "Cc ∪ Cf ∪ Co.", other))
    parts.append(fmt_table("PUNCTUATION", "Pc ∪ Pd ∪ Ps ∪ Pe ∪ Pi ∪ Pf ∪ Po.", punct))
    parts.append(fmt_table("MARK_NONSPACING", "Mn.", mn))
    parts.append(fmt_table("ASSIGNED", f"Every category table and range arm: the scalars Unicode {cat_unicode} assigns.",
                           assigned))
    text = "\n\n".join(parts) + "\n"
    stats = dict(other=len(other), punctuation=len(punct), mn=len(mn), assigned=len(assigned),
                 unicode_categories=cat_ver, alignments=nfd_ver, unicode=cat_unicode)
    return text, stats


def workspace_nfd_version():
    lock = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Cargo.lock")
    text = open(lock, encoding="utf-8").read()
    return locked_version(text, "unicode-normalization")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", help="cargo registry src dir holding tokenizers-0.21.4 and its deps")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--check", action="store_true", help="compare with --out instead of writing")
    a = ap.parse_args()
    text, stats = generate(registry_src(a.registry))
    out = os.path.abspath(a.out)
    if a.check:
        cur = open(out, encoding="utf-8").read() if os.path.exists(out) else ""
        if cur != text:
            print(f"{out} differs from the generated tables", file=sys.stderr)
            sys.exit(1)
        print(f"{out} is up to date {stats}")
        return
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {out} {stats} sha256 {hashlib.sha256(text.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
