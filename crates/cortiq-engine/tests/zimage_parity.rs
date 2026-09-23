//! Z-Image parity against the diffusers oracles (`python/zimage_oracle.py`).
//!
//! Env-gated; every test skips when its variables are absent:
//! - `CMF_ZIMAGE_ORACLE=<dir>`: an oracle directory (`…/oracles/v1/turbo`
//!   or `…/base`; the TE/tokenizer oracles live in the turbo dir).
//! - `CMF_ZIMAGE_CMF=<file.cmf>`: a Z-Image container (tokenizer, TE, VAE,
//!   and the DiT unless `CMF_ZIMAGE_DIT_DIR` is set).
//! - `CMF_ZIMAGE_DIT_DIR=<diffusers transformer dir>`: DiT from source.
//! - `CMF_ZIMAGE_CASES=<comma list>`: restrict the DiT cases (file stems
//!   without `dit_`/`_fp32`, e.g. `r512_p0_t8_i0`).
//!
//! Run on the CPU (`CMF_GPU=0`), release build:
//! `cargo test --release -p cortiq-engine --test zimage_parity -- --nocapture --test-threads 1`

use cortiq_engine::zimage::{self, ZImageDit, ZShape};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

enum T {
    F32(Vec<usize>, Vec<f32>),
    I64(Vec<usize>, Vec<i64>),
    U8(Vec<usize>, Vec<u8>),
}

fn read_st(path: &Path) -> (HashMap<String, T>, serde_json::Value) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
    let h: serde_json::Value = serde_json::from_slice(&b[8..8 + n]).unwrap();
    let base = 8 + n;
    let mut out = HashMap::new();
    let mut meta = serde_json::Value::Null;
    for (k, v) in h.as_object().unwrap() {
        if k == "__metadata__" {
            meta = v.clone();
            continue;
        }
        let shape: Vec<usize> = v["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let o = v["data_offsets"].as_array().unwrap();
        let raw = &b[base + o[0].as_u64().unwrap() as usize..base + o[1].as_u64().unwrap() as usize];
        let t = match v["dtype"].as_str().unwrap() {
            "F32" => T::F32(
                shape,
                raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
            ),
            "I64" => T::I64(
                shape,
                raw.chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
            ),
            "U8" => T::U8(shape, raw.to_vec()),
            d => panic!("{k}: dtype {d}"),
        };
        out.insert(k.clone(), t);
    }
    (out, meta)
}

fn f(m: &HashMap<String, T>, k: &str) -> Vec<f32> {
    match m.get(k) {
        Some(T::F32(_, v)) => v.clone(),
        _ => panic!("oracle tensor {k} missing or not f32"),
    }
}

fn meta_usize(meta: &serde_json::Value, k: &str) -> usize {
    let s = meta[k].as_str().unwrap_or_else(|| panic!("meta {k}"));
    s.trim_matches('"').parse().unwrap()
}

/// (rel, cos, maxabs)
fn cmp(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let (mut d2, mut b2, mut a2, mut ab, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        d2 += (x - y) * (x - y);
        b2 += y * y;
        a2 += x * x;
        ab += x * y;
        mx = mx.max((x - y).abs());
    }
    ((d2 / b2.max(1e-300)).sqrt(), ab / (a2.sqrt() * b2.sqrt()).max(1e-300), mx)
}

fn report(name: &str, a: &[f32], b: &[f32], bar: f64) -> bool {
    let (rel, cos, mx) = cmp(a, b);
    let ok = rel <= bar;
    println!(
        "{} {name:<28} rel {rel:.3e}  cos {cos:.9}  maxabs {mx:.3e}  (bar {bar:.0e})",
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.is_empty())
}

fn te_dir(oracle: &Path) -> PathBuf {
    // TE + tokenizer oracles are written once, into the turbo directory.
    if oracle.join("tok.safetensors").exists() {
        oracle.to_path_buf()
    } else {
        oracle.parent().unwrap().join("turbo")
    }
}

#[test]
fn zimage_tokenizer_and_te() {
    let (Some(oracle), Some(cmf)) = (env("CMF_ZIMAGE_ORACLE"), env("CMF_ZIMAGE_CMF")) else {
        eprintln!("skip: CMF_ZIMAGE_ORACLE / CMF_ZIMAGE_CMF not set");
        return;
    };
    let od = te_dir(Path::new(&oracle));
    let model = Arc::new(cortiq_core::CmfModel::open(&cmf).unwrap());
    let tok = cortiq_engine::tokenizer::Tokenizer::from_bytes(model.vocab.as_deref().unwrap()).unwrap();
    let (tk, _) = read_st(&od.join("tok.safetensors"));
    let mut all_ok = true;
    for (k, t) in &tk {
        let key = k.strip_prefix("ids_").unwrap();
        let T::I64(_, want) = t else { panic!() };
        let prompt = match key {
            "long" => (0..700).map(|i| format!("word{i}")).collect::<Vec<_>>().join(" "),
            _ => continue_prompt(&od, key),
        };
        let got = cortiq_engine::zimagegen::prompt_ids(&tok, &prompt, 512);
        let got: Vec<i64> = got.iter().map(|&v| v as i64).collect();
        let ok = &got == want;
        all_ok &= ok;
        println!("{} tok {key}: {} ids", if ok { "PASS" } else { "FAIL" }, got.len());
        if !ok {
            println!("   want {:?}\n   got  {:?}", &want[..want.len().min(40)], &got[..got.len().min(40)]);
        }
    }
    let enc = cortiq_engine::qwen3te::Qwen3Encoder::from_cmf(&model).unwrap();
    for key in ["p0", "p1", "p2", "p3", "n0"] {
        let path = od.join(format!("te_{key}_fp32.safetensors"));
        if !path.exists() {
            continue;
        }
        let (o, _) = read_st(&path);
        let T::I64(_, ids) = &o["ids"] else { panic!() };
        let ids: Vec<u32> = ids.iter().map(|&v| v as u32).collect();
        let t0 = std::time::Instant::now();
        let h = enc.encode(&ids);
        println!("te {key}: {} tokens in {:.2}s", ids.len(), t0.elapsed().as_secs_f64());
        all_ok &= report(&format!("te {key} h_m2"), &h, &f(&o, "h_m2"), 1e-4);
    }
    assert!(all_ok, "tokenizer/TE parity failed");
}

/// The prompt text of an oracle key, from `tok.safetensors` metadata.
fn continue_prompt(od: &Path, key: &str) -> String {
    let (_, meta) = read_st(&od.join("tok.safetensors"));
    let prompts: serde_json::Value =
        serde_json::from_str(meta["prompts"].as_str().unwrap()).unwrap();
    prompts[key]["prompt"].as_str().unwrap().to_string()
}

fn load_dit() -> Option<ZImageDit> {
    if let Some(dir) = env("CMF_ZIMAGE_DIT_DIR") {
        return Some(ZImageDit::load_dir(Path::new(&dir)).unwrap());
    }
    let cmf = env("CMF_ZIMAGE_CMF")?;
    let model = Arc::new(cortiq_core::CmfModel::open(&cmf).unwrap());
    Some(ZImageDit::from_cmf(&model).unwrap())
}

#[test]
fn zimage_dit_forward() {
    let Some(oracle) = env("CMF_ZIMAGE_ORACLE") else {
        eprintln!("skip: CMF_ZIMAGE_ORACLE not set");
        return;
    };
    let od = PathBuf::from(&oracle);
    let mut cases: Vec<String> = std::fs::read_dir(&od)
        .unwrap()
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter_map(|n| {
            n.strip_prefix("dit_")?
                .strip_suffix("_fp32.safetensors")
                .map(String::from)
        })
        .collect();
    cases.sort();
    if let Some(sel) = env("CMF_ZIMAGE_CASES") {
        let sel: Vec<&str> = sel.split(',').collect();
        cases.retain(|c| sel.contains(&c.as_str()));
    }
    if cases.is_empty() {
        eprintln!("skip: no dit_*_fp32 oracles");
        return;
    }
    let t0 = std::time::Instant::now();
    let Some(dit) = load_dit() else {
        eprintln!("skip: CMF_ZIMAGE_DIT_DIR / CMF_ZIMAGE_CMF not set");
        return;
    };
    println!("dit loaded in {:.1}s", t0.elapsed().as_secs_f64());
    let dim = dit.cfg.dim;
    let md = 4 * dim;
    let mut all_ok = true;
    for case in &cases {
        println!("── case {case}");
        let (o, meta) = read_st(&od.join(format!("dit_{case}_fp32.safetensors")));
        let (hh, ww, l) = (meta_usize(&meta, "H"), meta_usize(&meta, "W"), meta_usize(&meta, "L"));
        let t = f(&o, "t_model")[0];
        let cap = f(&o, "cap");
        let x_in = f(&o, "x_in");
        let shape = ZShape::new(hh, ww, l);
        assert_eq!(shape.n_img_p, meta_usize(&meta, "n_img_p"));
        assert_eq!(shape.l_p, meta_usize(&meta, "L_p"));
        all_ok &= report("temb", &dit.temb(t), &f(&o, "temb"), 1e-6);
        let mods = dit.mods_for_steps(&[t]);
        let nb = dit.cfg.n_mod_blocks();
        all_ok &= report("mod_nr0", &mods[..md], &f(&o, "mod_nr0"), 1e-6);
        all_ok &= report("mod_l0", &mods[2 * md..3 * md], &f(&o, "mod_l0"), 1e-6);
        all_ok &= report("mod_l29", &mods[(nb - 1) * md..nb * md], &f(&o, "mod_l29"), 1e-6);
        let fs = dit.final_scale_for_steps(&[t]);
        let fm: Vec<f32> = fs.iter().map(|v| v - 1.0).collect();
        all_ok &= report("final_mod", &fm, &f(&o, "final_mod"), 1e-5);
        let rope = zimage::ids_and_rope(shape.grid, l, dit.cfg.rope_theta, dit.cfg.axes_dims);
        for (nm, (c, s)) in [("rope_img", &rope.img), ("rope_cap", &rope.cap), ("rope_joint", &rope.joint)] {
            let (_, _, mc) = cmp(c, &f(&o, &format!("{nm}_cos")));
            let (_, _, ms) = cmp(s, &f(&o, &format!("{nm}_sin")));
            let ok = mc <= 2e-6 && ms <= 2e-6;
            all_ok &= ok;
            println!("{} {nm:<28} maxabs cos {mc:.3e} sin {ms:.3e}", if ok { "PASS" } else { "FAIL" });
        }
        let emb = dit.embed_caption(&cap, l);
        all_ok &= report("cap_seq", &emb, &f(&o, "cap_seq"), 1e-6);
        let t1 = std::time::Instant::now();
        let prep = dit.prepare(&cap, shape, 1, None).unwrap();
        println!("   prepare {:.2}s", t1.elapsed().as_secs_f64());
        all_ok &= report("cr1_out", &prep.cap, &f(&o, "cr1_out"), 1e-5);
        let (c, lh, lw) = (dit.cfg.in_channels, hh / 8, ww / 8);
        let x_tok = zimage::pad_rows_repeat_last(
            &zimage::patchify(&x_in, c, lh, lw),
            shape.n_img,
            shape.n_img_p,
            dit.geom().patch_dim,
        );
        let mut taps: Vec<(String, Vec<f32>)> = Vec::new();
        let t1 = std::time::Instant::now();
        let v_tok = dit.step_cpu_taps(&prep, &x_tok, &mods, &fs, &mut |n, v| {
            if matches!(n, "x_seq" | "nr0_out" | "nr1_out" | "u_in" | "l0_out" | "l1_out" | "l14_out" | "l29_out" | "final_out") {
                taps.push((n.to_string(), v.to_vec()));
            }
        });
        println!("   step_cpu {:.2}s", t1.elapsed().as_secs_f64());
        for (n, v) in &taps {
            let mut want = f(&o, n);
            if n == "final_out" {
                want.truncate(shape.n_img * dit.geom().patch_dim);
            }
            let bar = match n.as_str() {
                "x_seq" => 1e-6,
                "nr0_out" | "nr1_out" | "u_in" | "l0_out" | "l1_out" => 1e-5,
                _ => 2e-4,
            };
            all_ok &= report(n, v, &want, bar);
        }
        let v = zimage::unpatchify(&v_tok, c, lh, lw);
        all_ok &= report("v", &v, &f(&o, "v"), 2e-4);
        // the bf16 floor: diffusers-bf16's own distance from fp32
        let bf = od.join(format!("dit_{case}_bf16.safetensors"));
        if bf.exists() {
            let (ob, _) = read_st(&bf);
            let (rel, cos, _) = cmp(&f(&ob, "v"), &f(&o, "v"));
            println!("     (diffusers bf16 v vs fp32: rel {rel:.3e} cos {cos:.9})");
        }
    }
    assert!(all_ok, "DiT parity failed");
}

/// M5 handoff gate for the device backends (WP2 wgpu, WP3 Metal): the
/// device `step` (`gpu::zimage_prepare` + `gpu::zimage_step`) against the
/// host `step_cpu` on the SAME container, at the oracle's inputs, plus both
/// against the fp32 oracle `v`. Skips while the backend declines. Run with
/// the GPU enabled (no `CMF_GPU=0`).
#[test]
fn zimage_device_vs_cpu() {
    let (Some(oracle), Some(cmf)) = (env("CMF_ZIMAGE_ORACLE"), env("CMF_ZIMAGE_CMF")) else {
        eprintln!("skip: CMF_ZIMAGE_ORACLE / CMF_ZIMAGE_CMF not set");
        return;
    };
    let od = PathBuf::from(&oracle);
    let case = env("CMF_ZIMAGE_CASES").unwrap_or_else(|| "r512_p0_t8_i0".into());
    let model = Arc::new(cortiq_core::CmfModel::open(&cmf).unwrap());
    let dit = ZImageDit::from_cmf(&model).unwrap();
    let mut all_ok = true;
    for case in case.split(',') {
        let (o, meta) = read_st(&od.join(format!("dit_{case}_fp32.safetensors")));
        let (hh, ww, l) = (meta_usize(&meta, "H"), meta_usize(&meta, "W"), meta_usize(&meta, "L"));
        let t = f(&o, "t_model")[0];
        let cap = f(&o, "cap");
        let shape = ZShape::new(hh, ww, l);
        let mods = dit.mods_for_steps(&[t]);
        let fs = dit.final_scale_for_steps(&[t]);
        let x_tok = dit.tokens(&f(&o, "x_in"), &shape);
        let host = dit.prepare_with(&cap, shape, 7001, None, false).unwrap();
        let dev = dit.prepare_with(&cap, shape, 7002, Some((&mods, &fs)), true).unwrap();
        if !dev.device {
            eprintln!("skip {case}: the device backend declined zimage_prepare");
            continue;
        }
        let (c, lh, lw) = (dit.cfg.in_channels, hh / 8, ww / 8);
        let t0 = std::time::Instant::now();
        let v_dev = dit.step(&dev, 0, &x_tok, &mods, &fs);
        let td = t0.elapsed().as_secs_f64();
        let t0 = std::time::Instant::now();
        let v_cpu = dit.step_cpu(&host, &x_tok, &mods, &fs);
        let tc = t0.elapsed().as_secs_f64();
        println!("── {case}: device step {td:.3}s, host step {tc:.2}s");
        all_ok &= report("device cap vs host cap", &dev.cap, &host.cap, 3e-3);
        all_ok &= report("device v vs step_cpu", &v_dev, &v_cpu, 3e-3);
        let want = f(&o, "v");
        report("step_cpu v vs fp32 oracle", &zimage::unpatchify(&v_cpu, c, lh, lw), &want, 1.0);
        report("device v vs fp32 oracle", &zimage::unpatchify(&v_dev, c, lh, lw), &want, 1.0);
    }
    assert!(all_ok, "device vs host gate failed");
}

#[test]
fn zimage_vae() {
    let (Some(oracle), Some(cmf)) = (env("CMF_ZIMAGE_ORACLE"), env("CMF_ZIMAGE_CMF")) else {
        eprintln!("skip: CMF_ZIMAGE_ORACLE / CMF_ZIMAGE_CMF not set");
        return;
    };
    let od = te_dir(Path::new(&oracle));
    let model = cortiq_core::CmfModel::open(&cmf).unwrap();
    let vae = cortiq_engine::vae::VaeDecoder::from_cmf(&model).unwrap();
    let mut all_ok = true;
    let sel = env("CMF_ZIMAGE_VAE_RES").unwrap_or_else(|| "r512,r400x592".into());
    for res in sel.split(',') {
        let path = od.join(format!("vae_{res}_p0_t8.safetensors"));
        if !path.exists() {
            continue;
        }
        let (o, _) = read_st(&path);
        let T::F32(zs, z) = &o["z"] else { panic!() };
        let t0 = std::time::Instant::now();
        let img = vae.decode(z, zs[1], zs[2]);
        println!("vae {res}: {:.1}s", t0.elapsed().as_secs_f64());
        all_ok &= report(&format!("vae {res} img"), &img, &f(&o, "img"), 1e-4);
    }
    assert!(all_ok, "VAE parity failed");
}
