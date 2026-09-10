//! Detached model signing — authenticity on top of integrity.
//!
//! The format's hash chain (envelope → header → directory → per-tensor
//! `hash64`) detects corruption but cannot prove WHO produced a file.
//! `cortiq sign` writes a detached `<model>.sig` — Ed25519 over the
//! file's SHA-256 — so a 19 GB container is never rewritten and old
//! tooling is untouched. `cortiq verify` picks the `.sig` up
//! automatically when it sits next to the model.
//!
//! Key handling is deliberately plain: the signing key is a 32-byte
//! hex seed in a file you keep private; `sign` generates one on first
//! use. The `.sig` is JSON so registries and humans can read it.

use anyhow::{Context, bail};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use sha2::{Digest, Sha256};
use std::io::Read;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(s: &str) -> anyhow::Result<Vec<u8>> {
    if s.len() % 2 != 0 {
        bail!("odd-length hex");
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).context("bad hex"))
        .collect()
}

/// Streaming SHA-256 of the whole file (the page cache makes a warm
/// pass memory-bandwidth-bound; a cold 19 GB file is disk-bound).
fn sha256_file(path: &str) -> anyhow::Result<[u8; 32]> {
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {path}"))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().into())
}

pub fn cmd_sign(model_path: &str, key_path: &str) -> anyhow::Result<()> {
    let key = match std::fs::read_to_string(key_path) {
        Ok(s) => {
            let seed = unhex(s.trim()).with_context(|| format!("parsing {key_path}"))?;
            let seed: [u8; 32] = seed
                .try_into()
                .map_err(|_| anyhow::anyhow!("{key_path}: key must be 32 hex bytes"))?;
            SigningKey::from_bytes(&seed)
        }
        Err(_) => {
            let key = SigningKey::generate(&mut rand_core::OsRng);
            std::fs::write(key_path, hex(&key.to_bytes()))?;
            eprintln!("new signing key written to {key_path} — keep it private");
            key
        }
    };
    let digest = sha256_file(model_path)?;
    let sig = key.sign(&digest);
    let out = format!("{model_path}.sig");
    std::fs::write(
        &out,
        serde_json::to_string_pretty(&serde_json::json!({
            "alg": "ed25519-sha256",
            "pubkey": hex(key.verifying_key().as_bytes()),
            "sha256": hex(&digest),
            "sig": hex(&sig.to_bytes()),
        }))?,
    )?;
    println!(
        "signed: {out}\npubkey: {}",
        hex(key.verifying_key().as_bytes())
    );
    Ok(())
}

/// Verify a detached signature; `expect_pubkey` (hex) pins the signer.
/// Returns Ok(true) when a valid signature was checked, Ok(false) when
/// no `.sig` exists (the caller decides whether that is an error).
pub fn verify_detached(model_path: &str, expect_pubkey: Option<&str>) -> anyhow::Result<bool> {
    let sig_path = format!("{model_path}.sig");
    let Ok(text) = std::fs::read_to_string(&sig_path) else {
        return Ok(false);
    };
    let v: serde_json::Value = serde_json::from_str(&text).context("parsing .sig")?;
    let alg = v["alg"].as_str().unwrap_or("");
    if alg != "ed25519-sha256" {
        bail!("{sig_path}: unknown alg '{alg}'");
    }
    let pubkey_hex = v["pubkey"].as_str().context(".sig: no pubkey")?;
    if let Some(expect) = expect_pubkey {
        if !expect.eq_ignore_ascii_case(pubkey_hex) {
            bail!("{sig_path}: pubkey {pubkey_hex} does not match the expected signer");
        }
    }
    let pubkey: [u8; 32] = unhex(pubkey_hex)?
        .try_into()
        .map_err(|_| anyhow::anyhow!(".sig: pubkey must be 32 bytes"))?;
    let sig_bytes: [u8; 64] = unhex(v["sig"].as_str().context(".sig: no sig")?)?
        .try_into()
        .map_err(|_| anyhow::anyhow!(".sig: signature must be 64 bytes"))?;
    let claimed = v["sha256"].as_str().context(".sig: no sha256")?;
    let digest = sha256_file(model_path)?;
    if hex(&digest) != claimed.to_lowercase() {
        bail!("{model_path}: SHA-256 mismatch — file differs from the signed bytes");
    }
    VerifyingKey::from_bytes(&pubkey)?
        .verify(&digest, &Signature::from_bytes(&sig_bytes))
        .map_err(|e| anyhow::anyhow!("{sig_path}: signature INVALID: {e}"))?;
    println!("signature OK (ed25519, signer {pubkey_hex})");
    Ok(true)
}
