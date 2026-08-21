//! Buffer-level localization of the verify-vs-walk drift: one device layer
//! (CMF_DSV4_PACK_MAX_LI=0 keeps layers 1.. on the host), one token, and a
//! stage-by-stage comparison of the frame buffers the two paths leave
//! behind. The first buffer that differs names the divergent op.
#![cfg(feature = "gpu")]

use std::sync::Arc;

#[test]
fn first_divergent_stage() {
    for (k, v) in [
        ("CMF_SDOT", "0"),
        ("CMF_GPU_VRAM_MB", "2000"),
        ("CMF_DSV4_GPU_LAYER", "1"),
        ("CMF_DSV4_GPU_ATTN", "1"),
        ("CMF_DSV4_GPU_MOE2", "1"),
        ("CMF_DSV4_CHAIN", "1"),
        ("CMF_DSV4_PACK_MAX_LI", "0"),
    ] {
        unsafe { std::env::set_var(k, v) };
    }
    match cortiq_engine::gpu_wgpu::selected_and_up() {
        None => {
            cortiq_engine::gpu_wgpu::skip_or_fail(module_path!());
            return;
        }
        Some(false) => panic!("wgpu запрошен, но контекст не поднялся"),
        Some(true) => {}
    }
    let dir = std::env::var("CMF_TOY_DIR").unwrap_or_else(|_| {
        "/private/tmp/claude-501/-Users-oleg-Documents-cortiq-bot-cmfpublic/674db62a-643b-4641-b330-5feed6d40b67/scratchpad".into()
    });
    let path = std::path::Path::new(&dir).join("q4.cmf");
    if !path.exists() {
        eprintln!("стендов нет — пропуск");
        return;
    }
    let model = Arc::new(cortiq_core::CmfModel::open_sharded(path.to_str().unwrap()).unwrap());
    let mk = || {
        cortiq_engine::pipeline::Pipeline::from_model(
            &model,
            cortiq_engine::sampler::SamplerConfig::default(),
        )
        .unwrap()
    };
    let mut walk = mk();
    let mut spec = mk();
    let vocab = 128usize;
    let ids: Vec<u32> = (0..40u32)
        .map(|i| ((i * 37 + 11) % (vocab as u32 - 2)) + 1)
        .collect();
    let warm = 24usize;
    let step = |p: &mut cortiq_engine::pipeline::Pipeline, id: u32| {
        let b = p.dsv4.as_mut().unwrap();
        let (g, layers, cfg) = (&b.0, &b.1, b.2);
        let st = &mut b.3;
        let mut lg = Vec::new();
        cortiq_engine::dsv4::forward_token(
            g,
            layers,
            &cfg,
            st,
            id,
            &g.inv_freq_window.clone(),
            None,
            &mut lg,
        );
        lg
    };
    for t in 0..warm {
        step(&mut walk, ids[t]);
        step(&mut spec, ids[t]);
    }
    let real = ids[warm];
    // Verify FIRST (its frame buffers are strided _bt tags, untouched by the
    // walk), then the walk (its single-token tags 5/6/7/8 stay behind).
    {
        let b = spec.dsv4.as_mut().unwrap();
        let (g, layers, cfg) = (&b.0, &b.1, b.2);
        let st = &mut b.3;
        let pos0 = st.pos;
        let fed: Vec<u32> = (0..5u32)
            .map(|i| {
                if i == 0 {
                    real
                } else {
                    (real + 7 + i * 13) % vocab as u32
                }
            })
            .collect();
        let mut argmax = Vec::new();
        let mut lg_all = Vec::new();
        let mut walked = Vec::new();
        let txn = cortiq_engine::dsv4::dsv4_verify_chunk(
            g,
            layers,
            &cfg,
            st,
            &fed,
            pos0,
            &g.inv_freq_window.clone(),
            None,
            &[],
            &mut argmax,
            &mut lg_all,
            &mut walked,
        )
        .expect("verify отказал");
        assert!(cortiq_engine::dsv4::dsv4_spec_finish(
            g,
            layers,
            &cfg,
            st,
            txn,
            1,
            &fed,
            &g.inv_freq_window.clone(),
            None,
        ));
    }
    let nhhd = {
        let b = spec.dsv4.as_ref().unwrap();
        b.2.n_heads * b.2.head_dim
    };
    let (dim, q_lora) = {
        let b = spec.dsv4.as_ref().unwrap();
        (b.2.dim, b.2.q_lora_rank)
    };
    // Batch buffers, token 0 slices.
    let rd = |tag: u8, n: usize| cortiq_engine::gpu_wgpu::dsv4_dbg_read_tag(tag, 0, n).unwrap();
    let b_qn = rd(183, q_lora);
    let rdb = |tag: u8, n: usize, total: usize| {
        cortiq_engine::gpu_wgpu::dsv4_dbg_read_tag(tag, 0, total).unwrap()[..n].to_vec()
    };
    let b_q = rdb(184, nhhd, 5 * nhhd);
    let b_attn = rdb(185, nhhd, 5 * nhhd);
    let b_mid = cortiq_engine::gpu_wgpu::dsv4_dbg_read_tag(186, 0, 4096).unwrap();
    let b_ao = rdb(187, dim, 5 * dim);
    let b_x2 = rdb(181, dim, 5 * dim);
    // Now the walk of the same token — its frame leaves tags 4/5/6/8/45.
    step(&mut walk, real);
    let w_qn = rd(4, q_lora);
    let w_q = rd(5, nhhd);
    let w_attn = rd(6, nhhd);
    let w_mid = cortiq_engine::gpu_wgpu::dsv4_dbg_read_tag(7, 0, 4096).unwrap();
    let w_ao = rd(8, dim);
    let w_x2 = rd(45, dim);
    let cmp = |name: &str, a: &[f32], b: &[f32]| {
        let n = a.len().min(b.len());
        let mut mx = 0.0f32;
        let mut nd = 0usize;
        for i in 0..n {
            let d = (a[i] - b[i]).abs();
            if d > 0.0 {
                nd += 1;
                mx = mx.max(d);
            }
        }
        println!("{name}: {nd}/{n} различий, max {mx:e}");
    };
    // The walk's seeds for this token overwrite tag 45/4 before its frame
    // runs, so x2/qn here are the NEXT layer's; compare loosely and read the
    // trend, not the letter.
    cmp("qn(вход q)", &w_qn, &b_qn);
    cmp("q после rope", &w_q, &b_q);
    cmp("attn", &w_attn, &b_attn);
    cmp("mid(o_lora)", &w_mid, &b_mid);
    cmp("ao(wo_b)", &w_ao, &b_ao);
    cmp("x2(следующий вход)", &w_x2, &b_x2);
}
