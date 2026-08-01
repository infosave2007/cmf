//! The whole attention block of a DeepSeek-V4 layer in one submission,
//! against the same eight steps run separately on the CPU.
//!
//!     CMF_GPU=wgpu CMF_Q4TP_PARITY=/root/gtoy.cmf cargo test -p cortiq-engine \
//!         --features gpu --test gpu_dsv4_frame -- --nocapture
//!
//! Each kernel in the chain is already verified alone. What this adds is the
//! wiring: a buffer bound to the wrong stage, a norm applied to the pre-LoRA
//! vector instead of the post-, an output projection reading the queries
//! rather than attention's output — none of which any single-kernel test can
//! see, and all of which produce numbers rather than errors.

#![cfg(feature = "gpu")]

use cortiq_engine::{dsv4, gpu_wgpu, qtensor::QTensor};

#[test]
fn the_fused_attention_block_matches_the_cpu() {
    let Ok(path) = std::env::var("CMF_Q4TP_PARITY") else {
        return;
    };
    let model = std::sync::Arc::new(cortiq_core::CmfModel::open(&path).expect("open"));
    let find = |n: &str| model.tensors.iter().position(|e| e.name == n);
    let p = "model.layers.0.self_attn";
    let (Some(i_qa), Some(i_qb), Some(i_oa), Some(i_ob), Some(i_kv)) = (
        find(&format!("{p}.wq_a.weight")),
        find(&format!("{p}.wq_b.weight")),
        find(&format!("{p}.wo_a.weight")),
        find(&format!("{p}.wo_b.weight")),
        find(&format!("{p}.wkv.weight")),
    ) else {
        eprintln!("{path} — не игрушка DeepSeek-V4, пропуск");
        return;
    };
    let sh = |i: usize| (model.tensors[i].shape[0], model.tensors[i].shape[1]);
    let (q_lora, dim) = sh(i_qa);
    let (qb_rows, _) = sh(i_qb);
    let (hd, _) = sh(i_kv);
    let (oa_rows, per_group) = sh(i_oa);
    let nh = qb_rows / hd;
    let o_groups = qb_rows / per_group;
    let o_lora = oa_rows / o_groups;
    let rd = if hd >= 16 { 16 } else { hd & !1 };
    let eps = 1e-6f32;
    let scale = (hd as f32).powf(-0.5);
    println!("dim={dim} q_lora={q_lora} nh={nh} hd={hd} групп={o_groups} o_lora={o_lora}");

    let hidden: Vec<f32> = (0..dim).map(|i| ((i * 13) as f32 * 0.017).sin()).collect();
    let q_norm: Vec<f32> = (0..q_lora).map(|i| 1.0 + (i as f32) * 0.01).collect();
    let sink: Vec<f32> = (0..nh).map(|h| (h as f32) * 0.7 - 1.0).collect();
    let inv_freq: Vec<f32> = (0..rd / 2).map(|i| 1.0 / 10000f32.powf(2.0 * i as f32 / rd as f32)).collect();
    let pos = 37usize;

    // A cache of 24 positions, only some of them attended — the index list is
    // the whole point of this attention and a frame that ignored it would
    // still look sane against a dense reference.
    let npos = 24usize;
    let cache: Vec<f32> = (0..npos * hd).map(|i| ((i * 5) as f32 * 0.011).cos()).collect();
    let idxs: Vec<u32> = (0..npos as u32).filter(|i| i % 4 != 2).collect();
    let idx_us: Vec<usize> = idxs.iter().map(|&i| i as usize).collect();

    // ── the CPU's eight steps ──
    let t = |i: usize| QTensor::from_model(&model, &model.tensors[i].name).expect("tensor");
    let (tqa, tqb, toa, tob) = (t(i_qa), t(i_qb), t(i_oa), t(i_ob));
    let mut qr_raw = vec![0.0f32; q_lora];
    tqa.matvec(&hidden, &mut qr_raw, None);
    let mut qr = qr_raw.clone();
    dsv4::rms_weighted(&mut qr, &q_norm, eps);
    let mut q = vec![0.0f32; nh * hd];
    tqb.matvec(&qr, &mut q, None);
    for h in 0..nh {
        let head = &mut q[h * hd..(h + 1) * hd];
        dsv4::rms_inplace(head, eps);
        dsv4::rope_tail(head, &inv_freq, pos, rd, false);
    }
    let mut attn = vec![0.0f32; nh * hd];
    for h in 0..nh {
        let mut oh = vec![0.0f32; hd];
        dsv4::sparse_attend(
            &q[h * hd..(h + 1) * hd],
            &cache,
            &idx_us,
            sink[h],
            scale,
            hd,
            &mut oh,
        );
        dsv4::rope_tail(&mut oh, &inv_freq, pos, rd, true);
        attn[h * hd..(h + 1) * hd].copy_from_slice(&oh);
    }
    let mut mid_cpu = vec![0.0f32; o_groups * o_lora];
    {
        let mut sc = vec![0.0f32; per_group];
        for (i, m) in mid_cpu.iter_mut().enumerate() {
            let gi = i / o_lora;
            *m = toa.row_dot(i, &attn[gi * per_group..(gi + 1) * per_group], &mut sc);
        }
    }
    let mut want = vec![0.0f32; dim];
    let mut scratch = vec![0.0f32; per_group];
    dsv4::o_project(
        &attn,
        &|r, x, sc: &mut [f32]| toa.row_dot(r, x, sc),
        per_group,
        &|mid, dst| tob.matvec(mid, dst, None),
        o_groups,
        o_lora,
        None,
        &mut want,
    );
    let _ = &mut scratch;

    // ── the device's one submission ──
    assert!(
        gpu_wgpu::dsv4_cache_write(7, 0, 0, &cache, npos * hd),
        "кеш не сел на карту"
    );
    let w = gpu_wgpu::Dsv4AttnW {
        wq_a: i_qa,
        wq_b: i_qb,
        wo_a: i_oa,
        wo_b: i_ob,
        q_norm: &q_norm,
        sink: &sink,
    };
    let g = gpu_wgpu::Dsv4AttnGeom {
        dim,
        nh,
        hd,
        rd,
        q_lora,
        o_lora,
        o_groups,
        eps,
        scale,
    };
    // Stage by stage, then whole. The first stage that moves is the wiring
    // fault; comparing only the output tells you there is one and no more.
    let stages: Vec<(&str, &[f32])> = vec![
        ("qr", &qr_raw),
        ("qn", &qr),
        ("q", &q),
        ("attn", &attn),
        ("mid", &mid_cpu),
        ("", &want),
    ];
    let mut worst = 0.0f32;
    for (tap, cpu) in stages {
        // One test in its own binary, no threads: nothing else can observe
        // the variable between the set and the call it steers.
        unsafe {
            if tap.is_empty() {
                std::env::remove_var("CMF_DSV4_FRAME_TAP");
            } else {
                std::env::set_var("CMF_DSV4_FRAME_TAP", tap);
            }
        }
        let mut got = vec![0.0f32; nh * hd + dim];
        if !gpu_wgpu::dsv4_attn_frame(
            &model, &w, g, &hidden, 7, 0, &idxs, &inv_freq, pos, &mut got,
        ) {
            eprintln!("кадр отклонён устройством — пропуск");
            return;
        }
        let got = &got[..cpu.len()];
        let num: f32 = cpu.iter().zip(got).map(|(a, b)| (a - b) * (a - b)).sum();
        let den: f32 = cpu.iter().map(|a| a * a).sum::<f32>().max(1e-20);
        let rel = (num / den).sqrt();
        let name = if tap.is_empty() { "выход" } else { tap };
        println!("  {name:>5}: {rel:.3e}");
        worst = worst.max(rel);
        assert!(rel < 5e-3, "ступень {name} разошлась на {rel:.3e}");
    }
    println!("сшитый блок внимания: худшая ступень {worst:.3e}");
    gpu_wgpu::dsv4_cache_clear(7);
}
