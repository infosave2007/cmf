//! The hyper-connection join, device against CPU.
//!
//!     CMF_GPU=wgpu cargo test -p cortiq-engine --features gpu --test gpu_hc_parity -- --nocapture
//!
//! Step one of the whole-token graph. It saves no time on its own — it costs
//! a submission and replaces arithmetic the CPU did in microseconds — and
//! that is the point: the join is where a transposed mixing matrix or a
//! Sinkhorn off by one iteration hides, and both produce output that reads
//! as perfectly reasonable. Check it before anything is built on top.

#[cfg(feature = "gpu")]
#[test]
fn device_hc_join_matches_the_cpu() {
    use cortiq_engine::dsv4;

    let (hc, dim, iters, eps) = (4usize, 256usize, 20u32, 1e-6f32);
    let mix_hc = (2 + hc) * hc;
    let state: Vec<f32> = (0..hc * dim).map(|i| ((i * 13) as f32 * 0.017).sin()).collect();
    let mixes: Vec<f32> = (0..mix_hc).map(|i| ((i * 7) as f32 * 0.31).cos() * 2.0).collect();
    let base: Vec<f32> = (0..mix_hc).map(|i| ((i * 5) as f32 * 0.11).sin()).collect();
    let scale = [1.0f32, 1.0, 1.0];
    let block_out: Vec<f32> = (0..dim).map(|i| ((i * 3) as f32 * 0.023).cos()).collect();

    // CPU: the same three steps the port runs between every pair of blocks.
    // `mixes` here is already the projection's output, so the rsqrt the
    // device applies has to be applied by hand for the comparison to mean
    // anything.
    let rsq = 1.0 / (state.iter().map(|v| v * v).sum::<f32>() / state.len() as f32 + eps).sqrt();
    let scaled: Vec<f32> = mixes.iter().map(|m| m * rsq).collect();
    let (mut pre, mut post, mut comb) = (vec![0.0; hc], vec![0.0; hc], vec![0.0; hc * hc]);
    dsv4::hc_split_sinkhorn(&scaled, &scale, &base, hc, iters as usize, eps, &mut pre, &mut post, &mut comb);
    let mut want_fold = vec![0.0f32; dim];
    dsv4::hc_fold(&state, &pre, hc, dim, &mut want_fold);
    let mut want_exp = vec![0.0f32; hc * dim];
    dsv4::hc_expand(&block_out, &state, &post, &comb, hc, dim, &mut want_exp);

    let mut got_fold = vec![0.0f32; dim];
    let mut got_exp = vec![0.0f32; hc * dim];
    let ok = cortiq_engine::gpu_wgpu::hc_join_for_test(
        &state, &mixes, &scale, &base, &block_out, hc, dim, iters, eps,
        &mut got_fold, &mut got_exp,
    );
    if !ok {
        panic!("устройство не поднялось — тест не проверил ничего");
    }
    let rel = |a: &[f32], b: &[f32]| {
        let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f32 = b.iter().map(|x| x * x).sum::<f32>().max(1e-20);
        (num / den).sqrt()
    };
    let df = rel(&got_fold, &want_fold);
    let de = rel(&got_exp, &want_exp);
    println!("свёртка: {df:.3e}   развёртка: {de:.3e}");
    assert!(df < 1e-5, "свёртка разошлась: {df:.3e}");
    assert!(de < 1e-5, "развёртка разошлась: {de:.3e}");
}
