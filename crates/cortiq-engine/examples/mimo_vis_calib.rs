//! Dev-only: capture the MiMo ViT's input Hessians over calibration images
//! and write them as a `cortiq quantize-gptq --hessians` cache, so the GPTQ
//! q4tp driver can fold the tower without its text calibration pass.
//!
//!     mimo_vis_calib TOWER.cmf OUT.hess MAX_PIXELS IMG_OR_DIR...
//!
//! TOWER must hold dense (F32/F16/BF16) `visual.*` matrices: the tower's own
//! hook folds the inputs of dense projections only. The cache layout is the
//! one `cortiq-cli/src/gptq.rs::save_hessians` writes (magic `CMFHESS1`,
//! then per group: names, cols, count, H length, Σx², upper triangle of H),
//! here with one name per group.

use cortiq_core::CmfModel;
use cortiq_engine::gptq_capture;
use cortiq_engine::media;
use cortiq_engine::mimo_vision::{MimoProcessorConfig, MimoVit, prepare_image};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: mimo_vis_calib TOWER.cmf OUT.hess MAX_PIXELS IMG_OR_DIR...");
        std::process::exit(2);
    }
    let model = Arc::new(CmfModel::open(&args[1]).expect("open tower"));
    let vit = MimoVit::from_model(&model).expect("load tower");
    let max_px: usize = args[3].parse().expect("MAX_PIXELS");
    let mut paths: Vec<PathBuf> = Vec::new();
    for a in &args[4..] {
        let p = PathBuf::from(a);
        if p.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(&p)
                .unwrap()
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|e| matches!(e, "png" | "jpg" | "jpeg" | "webp"))
                })
                .collect();
            v.sort();
            paths.extend(v);
        } else {
            paths.push(p);
        }
    }
    let cfg = MimoProcessorConfig::default();
    let t0 = std::time::Instant::now();
    gptq_capture::begin(true);
    let mut rows = 0usize;
    for p in &paths {
        let frame = match media::read_rgb(p) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("skip {}: {e}", p.display());
                continue;
            }
        };
        let input = match prepare_image(&frame, &cfg, Some(max_px)) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("skip {}: {e}", p.display());
                continue;
            }
        };
        vit.forward(&input).expect("forward");
        rows += input.patches();
        eprintln!(
            "  {} {}x{} → {} patches ({:.0} s)",
            p.display(),
            frame.width,
            frame.height,
            input.patches(),
            t0.elapsed().as_secs_f64()
        );
    }
    let hess = gptq_capture::end();
    let mut names: Vec<&String> = hess.keys().collect();
    names.sort();
    let tmp = format!("{}.tmp", args[2]);
    let mut f = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(&tmp).unwrap());
    f.write_all(b"CMFHESS1").unwrap();
    f.write_all(&(names.len() as u64).to_le_bytes()).unwrap();
    for n in &names {
        let a = &hess[*n];
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&(n.len() as u32).to_le_bytes()).unwrap();
        f.write_all(n.as_bytes()).unwrap();
        f.write_all(&(a.cols as u64).to_le_bytes()).unwrap();
        f.write_all(&(a.count as u64).to_le_bytes()).unwrap();
        f.write_all(&(a.h.len() as u64).to_le_bytes()).unwrap();
        for v in &a.sumsq {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
        let c = a.cols;
        if a.h.len() == c * c {
            for i in 0..c {
                for v in &a.h[i * c + i..i * c + c] {
                    f.write_all(&v.to_le_bytes()).unwrap();
                }
            }
        }
    }
    f.flush().unwrap();
    drop(f);
    std::fs::rename(&tmp, &args[2]).unwrap();
    println!(
        "wrote {}: {} linears, {rows} patches from {} images, {:.0} s",
        args[2],
        names.len(),
        paths.len(),
        t0.elapsed().as_secs_f64()
    );
}
