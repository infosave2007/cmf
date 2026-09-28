//! Produce a weights/descriptors-only checkpoint for backend bring-up.
//! This intentionally resets optimizer history; it is NOT an exact training resume.
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() == 3, "usage: compact_checkpoint INPUT OUTPUT");
    let out = Path::new(&args[2]);
    anyhow::ensure!(!out.exists(), "refusing to overwrite output");
    anyhow::ensure!(
        !out.with_extension("tmp").exists(),
        "temporary output already exists"
    );
    let ck = cortiq_embryo::train::load_checkpoint(Path::new(&args[1]))?;
    let extras: Vec<_> = ck
        .extras
        .iter()
        .map(|(n, x)| (n.as_str(), x.as_slice()))
        .collect();
    cortiq_embryo::train::save_checkpoint(out, &ck.cfg, ck.step, &ck.params, None, None, &extras)?;
    let compact = cortiq_embryo::train::load_checkpoint(out)?;
    anyhow::ensure!(
        compact.m.is_none() && compact.v.is_none(),
        "optimizer was not stripped"
    );
    anyhow::ensure!(
        compact.step == ck.step
            && compact
                .params
                .iter()
                .map(|x| x.to_bits())
                .eq(ck.params.iter().map(|x| x.to_bits())),
        "parameter round-trip mismatch"
    );
    anyhow::ensure!(
        compact.extras.len() == ck.extras.len()
            && compact
                .extras
                .iter()
                .zip(&ck.extras)
                .all(|((an, a), (bn, b))| an == bn
                    && a.iter()
                        .map(|x| x.to_bits())
                        .eq(b.iter().map(|x| x.to_bits()))),
        "descriptor round-trip mismatch"
    );
    println!(
        "verified {} parameters, {} descriptor blobs; original step {}, optimizer deliberately omitted",
        ck.params.len(),
        ck.extras.len(),
        ck.step
    );
    Ok(())
}
