//! A seal from rows by index (`seal_gathered`, the device path's host half)
//! leaves the state the seal over ordered copies does (`prefill_group`): the
//! same landmarks, skeletons, sinks and window, and with the far rows replayed
//! afterwards (`replay_far_idx`) the same far sums, bit for bit.

use cortiq_engine::nystrom::{NystromState, O1DeviceView};

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}

fn same(a: &O1DeviceView<'_>, b: &O1DeviceView<'_>) {
    assert_eq!((a.m_eff, a.w, a.sink_len, a.d, a.dv, a.exact_only), (b.m_eff, b.w, b.sink_len, b.d, b.dv, b.exact_only));
    assert_eq!((a.win_len, a.win_head, a.far_len), (b.win_len, b.win_head, b.far_len));
    assert_eq!(a.scale.to_bits(), b.scale.to_bits());
    for (x, y) in [(a.win_k, b.win_k), (a.win_v, b.win_v), (a.sink_k, b.sink_k), (a.sink_v, b.sink_v), (a.k_tilde, b.k_tilde)] {
        assert_eq!(x.len(), y.len());
        assert!(x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()));
    }
    assert_eq!(a.heads.len(), b.heads.len());
    for (h, g) in a.heads.iter().zip(&b.heads) {
        for (x, y) in [(h.t_hat, g.t_hat), (h.z_hat, g.z_hat), (h.m_max, g.m_max), (h.q_tilde, g.q_tilde), (h.mu, g.mu)] {
            assert_eq!(x.len(), y.len());
            assert!(x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()), "a head's far state differs");
        }
    }
}

#[test]
fn gathered_seal_is_the_seal_over_ordered_copies() {
    let (t, d, m, w, sink) = (300usize, 16usize, 8usize, 24usize, 4usize);
    let (nh, hpk, h0, stride) = (4usize, 2usize, 2usize, 3usize);
    let mut seed = 11u64;
    let ks: Vec<f32> = (0..t * d).map(|_| lcg(&mut seed)).collect();
    let vs: Vec<f32> = (0..t * d).map(|_| lcg(&mut seed)).collect();
    let q_trace: Vec<f32> = (0..t.div_ceil(stride) * nh * d).map(|_| lcg(&mut seed)).collect();
    for keep in [vec![], vec![40usize, 41, 150]] {
        let mut chosen = vec![false; t];
        keep.iter().for_each(|&r| chosen[r] = true);
        let order: Vec<u32> = (0..sink).chain(keep.iter().copied()).chain((sink..t).filter(|&r| !chosen[r])).map(|r| r as u32).collect();
        let st_sink = sink + keep.len();
        // the reference: ordered copies, the queries one per position
        let pick = |src: &[f32]| -> Vec<f32> { order.iter().flat_map(|&r| src[r as usize * d..(r as usize + 1) * d].iter().copied()).collect() };
        let qo: Vec<Vec<f32>> = (0..hpk)
            .map(|h| order.iter().flat_map(|&r| { let at = ((r as usize / stride) * nh + h0 + h) * d; q_trace[at..at + d].iter().copied() }).collect())
            .collect();
        let qr: Vec<&[f32]> = qo.iter().map(|q| &q[..]).collect();
        let mut a = NystromState::new_group(m, w, st_sink, hpk);
        a.prefill_group(&qr, &pick(&ks), &pick(&vs), t, d, d);
        let mut b = NystromState::new_group(m, w, st_sink, hpk);
        let range = b.seal_gathered(&ks, &vs, &q_trace, stride, nh, h0, &order, t, d, d).expect("a seal past the window");
        assert_eq!(range, st_sink..t - w);
        b.replay_far_idx(&ks, &vs, &order[range]);
        same(&a.device_view(), &b.device_view());
    }
}
