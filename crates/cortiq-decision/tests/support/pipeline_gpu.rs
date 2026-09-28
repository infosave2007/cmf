use cortiq_decision::{
    bert::{EncoderDevice, EncoderExport, GOLDEN_MAX_ABS},
    packed::Packed,
    resonance::Topology,
    signal::{ScoredSignal, SignalEncoder},
};
use std::path::PathBuf;

fn compare(a: &ScoredSignal, b: &ScoredSignal) {
    assert_eq!(a.features.phi_h, b.features.phi_h);
    assert_eq!(a.errors.len(), b.errors.len());
    for (a, b) in a.features.phi_p.iter().zip(&b.features.phi_p) {
        assert!((a - b).abs() <= GOLDEN_MAX_ABS, "embedding {a} != {b}");
    }
    for (a, b) in a.errors.iter().zip(&b.errors) {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert!((a - b).abs() <= 2e-5 * (1.0 + a.abs()), "error {a} != {b}");
        }
    }
}

pub fn joint_pipeline_contract(device: EncoderDevice) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/toy/encoder");
    let encoder = EncoderExport::read(path).unwrap().encoder().unwrap();
    let cpu = SignalEncoder::new(encoder.clone());
    let gpu = SignalEncoder::new(encoder.clone().with_device(device).unwrap());
    // A genuinely separate encoder, not a clone sharing the same mutex.
    let gpu2 = SignalEncoder::new(encoder.with_device(device).unwrap());
    let dim = cpu.dim();
    let topologies: Vec<_> = (0..4)
        .map(|t| Topology {
            mean: (0..dim).map(|i| ((i + t) % 17) as f32 * 0.002).collect(),
            // Non-orthogonal bases, different ranks, including rank zero.
            basis: (0..t * dim)
                .map(|i| ((i / dim + i % dim) % 11) as f32 * 0.001)
                .collect(),
        })
        .collect();
    let views: Vec<_> = topologies.iter().map(Topology::view).collect();
    let c1 = Packed::new(&views).unwrap();
    let c2 = Packed::new(&views[1..]).unwrap();
    let g1 = Packed::new(&views).unwrap().with_device(device).unwrap();
    let g2 = Packed::new(&views[1..])
        .unwrap()
        .with_device(device)
        .unwrap();
    let empty = Packed::new(&[]).unwrap().with_device(device).unwrap();
    let long = "hello world ".repeat(100);
    for text in ["", "hello world", "Привет 😀", &long, "short again"] {
        let a = cpu.score_timed(text, &[&c1, &empty, &c2]).unwrap();
        let b = gpu.score_timed(text, &[&g1, &empty, &g2]).unwrap();
        compare(&a, &b);
        assert!(a.timings.gpu.is_zero());
        assert!(b.timings.encode.is_zero());
        assert!(b.resonance.is_zero());
        assert!(!b.timings.gpu.is_zero());
        // Joint and standalone scoring must agree on the same GPU features.
        let errors = g1.errors(&b.features.signal()).unwrap();
        assert_eq!(errors, b.errors[0]);
    }
    assert_eq!(gpu.encoder().gpu_submissions(), 5);
    assert_eq!(g1.gpu_submissions(), 10); // joint + standalone
    assert_eq!(g2.gpu_submissions(), 5);
    assert_eq!(empty.gpu_submissions(), 0);
    let before = gpu.encoder().gpu_submissions();
    assert!(gpu.score_timed("duplicate", &[&g1, &g1]).is_err());
    let wrong = Topology {
        mean: vec![0.; 3],
        basis: vec![],
    };
    let wrong = Packed::new(&[wrong.view()]).unwrap();
    assert!(gpu.score_timed("wrong dimension", &[&wrong]).is_err());
    assert_eq!(gpu.encoder().gpu_submissions(), before);
    let a = cpu.score_timed("hello world", &[&c1]).unwrap();
    for (e, p) in [(&cpu, &g1), (&gpu, &c1)] {
        let mixed = e.score_timed("hello world", &[p]).unwrap();
        compare(&a, &mixed);
        assert!(mixed.timings.gpu.is_zero());
        assert!(!mixed.timings.encode.is_zero());
    }
    let no_skills = gpu.score_timed("hello world", &[]).unwrap();
    assert!(no_skills.errors.is_empty());
    assert_eq!(
        no_skills.features.phi_p,
        gpu.encoder().encode("hello world")
    );
    // Reverse skill orders on independently locked encoders sharing both
    // scorers: address-ordered locking prevents AB/BA deadlock.
    let barrier = std::sync::Barrier::new(2);
    std::thread::scope(|s| {
        for reverse in [false, true] {
            let barrier = &barrier;
            let e = if reverse { &gpu2 } else { &gpu };
            let cs = if reverse { [&c2, &c1] } else { [&c1, &c2] };
            let gs = if reverse { [&g2, &g1] } else { [&g1, &g2] };
            let expected = cpu.score_timed("hello world", &cs).unwrap();
            s.spawn(move || {
                for _ in 0..16 {
                    barrier.wait();
                    compare(&expected, &e.score_timed("hello world", &gs).unwrap());
                }
            });
        }
    });
    assert_eq!(cpu.encoder().gpu_submissions(), 0);
}
