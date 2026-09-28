//! Runtime vs trainer on a REAL checkpoint (EMBRYO_CKPT, EMBRYO_TOK env):
//! log-probs at several positions of a held-out window. Skipped otherwise.
#![cfg(any(target_os = "macos", feature = "vulkan"))]

use cortiq_embryo::model::{EmbryoGpu, Layout};

fn parse_runtime_routes(trace: &str, layers: usize, rows: usize) -> Vec<Vec<u32>> {
    let mut routes = vec![Vec::with_capacity(rows); layers];
    for line in trace.lines() {
        let Some((layer, ids)) = line.split_once(':') else {
            continue;
        };
        let Ok(layer) = layer.parse::<usize>() else {
            continue;
        };
        if layer >= layers {
            continue;
        }
        let mut ids = ids.split(',');
        let Some(id) = ids.next().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        assert!(
            ids.next().is_none(),
            "expected top-1 runtime route, got {line}"
        );
        routes[layer].push(id);
    }
    for (layer, ids) in routes.iter().enumerate() {
        assert_eq!(
            ids.len(),
            rows,
            "runtime route trace layer {layer}: {} rows, expected {rows}",
            ids.len()
        );
    }
    routes
}

fn counts(routes: &[u32], experts: usize) -> Vec<u32> {
    let mut out = vec![0u32; experts];
    for &e in routes {
        *out.get_mut(e as usize)
            .unwrap_or_else(|| panic!("route expert {e} >= {experts}")) += 1;
    }
    out
}

#[test]
fn real_checkpoint_runtime_parity() {
    let (Ok(ck_path), Ok(tok_path)) = (std::env::var("EMBRYO_CKPT"), std::env::var("EMBRYO_TOK"))
    else {
        return;
    };
    let Some(_) = cortiq_embryo::metal::ctx() else {
        return;
    };
    let ck = cortiq_embryo::train::load_checkpoint(std::path::Path::new(&ck_path)).unwrap();
    let cfg = ck.cfg.clone();
    let t = 256usize;
    // tokens: a slice of held-out english
    let bpe = cortiq_embryo::tokenizer::Bpe::load(std::path::Path::new(&tok_path)).unwrap();
    let heldout_path = std::env::var("EMBRYO_HELDOUT")
        .unwrap_or_else(|_| "/Users/oleg/embryo-data/heldout-en.txt".to_owned());
    let text = std::fs::read_to_string(heldout_path).unwrap();
    let mut ids = Vec::new();
    let mut cache = std::collections::HashMap::new();
    bpe.encode(&text[..20000], &mut cache, &mut ids);
    let tokens: Vec<u32> = ids[..t].to_vec();
    assert_eq!(cfg.phase_delta_layers_for_export().unwrap(), vec![3]);

    // Compute the trainer reference first, then let this block drop every
    // trainer allocation before the runtime opens the exported model.  The
    // test therefore obeys the real-checkpoint one-heavy-model-at-a-time
    // resource contract instead of keeping a second full copy resident.
    let (expected, trainer_routes, trainer_counts, trainer_drops, moe_cap) = {
        let lay = Layout::new(&cfg);
        let gpu = EmbryoGpu::new_eval_dropless(cfg.clone(), 1, t, &ck.params).unwrap();
        gpu.set_desc(&ck.extras);
        gpu.desc_updates.set(false);
        let xf = gpu.forward_hidden(&tokens);
        let (h, v, ncl) = (cfg.hidden, cfg.vocab, cfg.head_clusters);
        let cs = v / ncl;
        let e = &ck.params[lay.embed..lay.embed + v * h];
        let cm = &ck.params[lay.head_clusters..lay.head_clusters + ncl * h];
        let logprobs = |x: &[f32]| -> Vec<f32> {
            let mut lc: Vec<f32> = (0..ncl)
                .map(|c| (0..h).map(|j| cm[c * h + j] * x[j]).sum())
                .collect();
            let mx = lc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = mx + lc.iter().map(|z| (z - mx).exp()).sum::<f32>().ln();
            for z in &mut lc {
                *z -= lse;
            }
            let mut out = vec![0.0f32; v];
            for c in 0..ncl {
                let lg: Vec<f32> = (0..cs)
                    .map(|s| (0..h).map(|j| e[(c * cs + s) * h + j] * x[j]).sum())
                    .collect();
                let bm = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let bl = bm + lg.iter().map(|z| (z - bm).exp()).sum::<f32>().ln();
                for s in 0..cs {
                    out[c * cs + s] = lc[c] + lg[s] - bl;
                }
            }
            out
        };
        let expected = [
            0usize, 63, 127, 173, 174, 187, 188, 191, 200, 210, 220, 230, 240, 250, 254,
        ]
        .into_iter()
        .map(|pos| {
            let lp = logprobs(&xf[pos * h..(pos + 1) * h]);
            assert!(
                lp.iter().all(|x| x.is_finite()),
                "trainer logprobs not finite at {pos}"
            );
            let nxt = tokens[(pos + 1).min(t - 1)] as usize;
            (pos, nxt, lp)
        })
        .collect::<Vec<_>>();
        let trainer_routes = gpu
            .moe
            .iter()
            .map(|mo| {
                // Snapshot the host mirror before any later GPU submission;
                // Vulkan's readback borrow is explicitly unsafe and must not
                // outlive this immediate copy.
                unsafe { mo.assign.as_u32_slice().to_vec() }
            })
            .collect::<Vec<_>>();
        let trainer_drops = gpu
            .moe
            .iter()
            .map(|mo| {
                let slots = unsafe { mo.slot.as_u32_slice().to_vec() };
                slots
                    .iter()
                    .enumerate()
                    .filter_map(|(pos, &slot)| (slot >= gpu.moe_cap as u32).then_some(pos))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert!(
            trainer_drops.iter().all(|drops| drops.is_empty()),
            "dropless evaluation produced capacity drops: {trainer_drops:?}"
        );
        (
            expected,
            trainer_routes,
            gpu.routing_counts(),
            trainer_drops,
            gpu.moe_cap,
        )
    };
    drop(ck);

    let cmf = std::env::var("EMBRYO_CMF").unwrap();
    let model = std::sync::Arc::new(cortiq_core::format::CmfModel::open(&cmf).unwrap());
    let lc = model.header.arch.linear_core.as_ref().unwrap();
    assert_eq!(lc.kind, "vmf_phase_delta_v1");
    assert_eq!(lc.phase_delta_layers, Some(vec![3]));
    let mut pipe = cortiq_engine::pipeline::Pipeline::from_model(
        &model,
        cortiq_engine::sampler::SamplerConfig::default(),
    )
    .unwrap();
    // The runtime already has a bounded selected-expert trace.  Capture one
    // full 256-token replay before the probe loop; the trainer route buffers
    // above are the matching full-batch assignments.
    let route_trace = std::env::temp_dir().join(format!(
        "embryo-native-parity-routes-{}.log",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&route_trace);
    // This test runs with one test thread; install the process-local trace
    // before the runtime's OnceLock opens it.
    unsafe { std::env::set_var("CMF_MOE_TRACE", &route_trace) };
    let _ = pipe.prefill_next_logits(&tokens, None);
    let runtime_trace = std::fs::read_to_string(&route_trace).unwrap();
    let runtime_routes = parse_runtime_routes(&runtime_trace, cfg.layers, t);
    for layer in 0..cfg.layers {
        let tr = &trainer_routes[layer];
        let rr = &runtime_routes[layer];
        let first = tr.iter().zip(rr).position(|(a, b)| a != b);
        let runtime_counts = counts(rr, cfg.experts);
        eprintln!(
            "route layer {layer}: trainer counts {:?}, runtime counts {:?}, first mismatch {:?}, trainer drops {} first {:?} last {:?} (cap {})",
            trainer_counts[layer],
            runtime_counts,
            first,
            trainer_drops[layer].len(),
            trainer_drops[layer].first(),
            trainer_drops[layer].last(),
            moe_cap,
        );
    }
    let _ = std::fs::remove_file(&route_trace);
    // trainer's own loss on the window vs runtime's
    let mut tl = 0.0f64;
    let mut rl = 0.0f64;
    let nprobes = expected.len() as f64;
    let tolerance = std::env::var("EMBRYO_PARITY_TOL")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(5e-2);
    for (pos, nxt, want) in expected {
        let got = pipe.prefill_next_logits(&tokens[..=pos], None);
        assert!(
            got.iter().all(|x| x.is_finite()),
            "runtime logits not finite at {pos}"
        );
        let d = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            d <= tolerance,
            "real runtime parity at {pos}: max|delta| {d:.6} > tolerance {tolerance:.6}"
        );
        tl += -want[nxt] as f64;
        rl += -got[nxt] as f64;
        eprintln!(
            "pos {pos:>3}: max|Δ| {d:.3e}  trainer lp(next) {:.3}  runtime lp(next) {:.3}",
            want[nxt], got[nxt]
        );
    }
    eprintln!(
        "mean nll over probes: trainer {:.3} runtime {:.3}",
        tl / nprobes,
        rl / nprobes
    );
    assert!(tl.is_finite() && rl.is_finite());
}
