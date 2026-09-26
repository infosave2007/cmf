//! The `cortiq-hashfeat-v1` contract (spec §1.5, §2.6): the file is the verbatim
//! embryo copy plus one marked contract section, the 32 golden vectors are pinned,
//! and the contract strings are the ones of `contract_v2.rs`.
use cortiq_decision::hashfeat::{self, DIM, GOLDEN_DENSE_SHA256, GOLDEN_TEXTS};
use sha2::{Digest, Sha256};

const SECTION_BEGIN: &str = "\n// ==== cortiq-hashfeat-v1 contract";
const SECTION_END: &str = "// ==== end cortiq-hashfeat-v1 contract ====\n";
const EMBRYO_DIR: &str = "/Users/oleg/dev/cmfpublic/tools/cortiq-decision-embryo/src";

fn sha256_hex(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

/// The source of `hashfeat.rs` with the contract section removed.
fn without_contract_section() -> String {
    let src = include_str!("../src/hashfeat.rs");
    let a = src.find(SECTION_BEGIN).expect("contract section start");
    let e = src.find(SECTION_END).expect("contract section end") + SECTION_END.len();
    assert_eq!(src.matches(SECTION_BEGIN).count(), 1);
    format!("{}{}", &src[..a], &src[e..])
}

#[test]
fn file_is_the_embryo_copy_plus_the_contract_section() {
    let copy = without_contract_section();
    assert_eq!(
        sha256_hex(copy.as_bytes()),
        hashfeat::EMBRYO_COPY_SHA256,
        "hashfeat.rs outside the contract section is not the verbatim embryo file"
    );
    let embryo = std::path::Path::new(EMBRYO_DIR).join("hashfeat.rs");
    if !embryo.exists() {
        eprintln!(
            "embryo source absent at {}; byte comparison skipped",
            embryo.display()
        );
        return;
    }
    assert_eq!(std::fs::read(&embryo).unwrap(), copy.as_bytes());
}

#[test]
fn contract_strings_are_the_contract_v2_ones() {
    assert_eq!(hashfeat::KIND, "cortiq-hashfeat-v1");
    assert_eq!(DIM, 4096);
    assert_eq!(hashfeat::HASH, "fnv1a64");
    assert_eq!(
        hashfeat::NORMALIZATION,
        "l2-f32-index-order-mul-inv;zero-stays-zero"
    );
    assert_eq!(
        (
            hashfeat::WORD_WEIGHT,
            hashfeat::BIGRAM_WEIGHT,
            hashfeat::NGRAM_WEIGHT
        ),
        (1.0, 0.7, 0.5)
    );
    assert_eq!(hashfeat::NGRAMS, [3, 4, 5]);
    assert_eq!((hashfeat::SEED, hashfeat::SIGN_BIT), (0, 33));
    let contract = std::path::Path::new(EMBRYO_DIR).join("contract_v2.rs");
    if !contract.exists() {
        eprintln!(
            "contract_v2.rs absent at {}; string comparison skipped",
            contract.display()
        );
        return;
    }
    let src = std::fs::read_to_string(contract).unwrap();
    assert!(src.contains(&format!(
        "pub const HASHING_DEFINITION: &str = {:?};",
        hashfeat::DEFINITION
    )));
    assert!(src.contains(&format!(
        "pub const HASHING_NORMALIZATION: &str = {:?};",
        hashfeat::NORMALIZATION
    )));
    assert!(src.contains(&format!(
        "pub const HASHING_HASH: &str = {:?};",
        hashfeat::HASH
    )));
}

#[test]
fn golden_dense_vectors_are_pinned() {
    let computed: Vec<String> = GOLDEN_TEXTS
        .iter()
        .map(|t| hashfeat::dense_f32le_sha256(t, DIM))
        .collect();
    if computed
        .iter()
        .zip(GOLDEN_DENSE_SHA256)
        .any(|(a, b)| a != b)
    {
        for (t, h) in GOLDEN_TEXTS.iter().zip(&computed) {
            eprintln!("    \"{h}\", // {t:?}");
        }
    }
    let mismatches = hashfeat::golden_mismatches();
    assert!(
        mismatches.is_empty(),
        "hashing contract mismatch on golden texts {mismatches:?} (unicode {})",
        hashfeat::unicode_version()
    );
    assert_eq!(GOLDEN_TEXTS.len(), 32);
    let distinct: std::collections::BTreeSet<&str> = GOLDEN_TEXTS.iter().copied().collect();
    assert_eq!(distinct.len(), 32, "golden texts must be distinct");
    let distinct: std::collections::BTreeSet<&str> = GOLDEN_DENSE_SHA256.iter().copied().collect();
    assert_eq!(distinct.len(), 32, "golden vectors must be distinct");
    // Coverage the spec asks for: Cyrillic, CJK, Greek sigma, Turkish İ,
    // Devanagari, circled letters, emoji.
    for needle in ["Привет", "日本語", "ς", "İ", "हिन्दी", "Ⓐ", "😀"] {
        assert!(GOLDEN_TEXTS.iter().any(|t| t.contains(needle)), "{needle}");
    }
    for t in GOLDEN_TEXTS {
        let d = hashfeat::dense(t, DIM);
        let n: f32 = d.iter().map(|x| x * x).sum();
        assert!(n == 0.0 || (n - 1.0).abs() < 1e-5, "{t:?}: norm² {n}");
        assert_eq!(n == 0.0, t.is_empty(), "{t:?}");
    }
}

#[test]
fn contract_record_carries_the_goldens() {
    let r = hashfeat::contract_record();
    assert_eq!(r["kind"], "cortiq-hashfeat-v1");
    assert_eq!(r["dim"], 4096);
    assert_eq!(r["definition"], hashfeat::DEFINITION);
    assert_eq!(r["source"]["sha256"], hashfeat::DISTILL_RS_SHA256);
    assert_eq!(r["source"]["file"], "cortiq-router/src/distill.rs");
    assert_eq!(r["tokens"]["byte_ngrams"], serde_json::json!([3, 4, 5]));
    let golden = r["golden"].as_array().unwrap();
    assert_eq!(golden.len(), 32);
    for (g, (t, h)) in golden
        .iter()
        .zip(GOLDEN_TEXTS.iter().zip(GOLDEN_DENSE_SHA256))
    {
        assert_eq!(g["text"], *t);
        assert_eq!(g["dense_f32le_sha256"], h);
    }
    assert_eq!(r["unicode_version"], hashfeat::unicode_version());
}

#[test]
fn dense_is_the_runtime_form() {
    // fnv1a64("a") = 0xaf63dc4c8601ec8c: index 0xc8c, bit 33 clear -> −1.
    let d = hashfeat::dense("a", DIM);
    assert_eq!(d.len(), DIM);
    assert_eq!(d[0xc8c], -1.0);
    assert_eq!(d.iter().filter(|&&x| x != 0.0).count(), 1);
    // Case folding happens before hashing.
    assert_eq!(
        hashfeat::dense("Card DECLINED", DIM),
        hashfeat::dense("card declined", DIM)
    );
    // The bytes are the little-endian f32 values.
    let b = hashfeat::dense_f32le("a", DIM);
    assert_eq!(b.len(), DIM * 4);
    assert_eq!(&b[0xc8c * 4..0xc8c * 4 + 4], &(-1.0f32).to_le_bytes());
}
