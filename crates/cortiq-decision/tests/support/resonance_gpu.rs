use cortiq_decision::{
    bert::EncoderDevice,
    packed::Packed,
    resonance::{Topology, errors_reference},
};

pub fn reconstruction_parity(device: EncoderDevice) {
    for dim in [7, 35, 4480] {
        let topologies: Vec<_> = (0..9)
            .map(|t| {
                let rank = t % 5;
                // Deliberately correlated rows: a norm-minus-projections shortcut
                // would fail this test, even with an orthogonal trained model.
                Topology {
                    mean: (0..dim).map(|i| ((i + t) % 19) as f32 * 0.001).collect(),
                    basis: (0..rank * dim)
                        .map(|i| ((i % dim + i / dim) % 13) as f32 * 0.004 / (dim as f32).sqrt())
                        .collect(),
                }
            })
            .collect();
        let views: Vec<_> = topologies.iter().map(Topology::view).collect();
        let cpu = Packed::new(&views).unwrap();
        let metal = Packed::new(&views).unwrap().with_device(device).unwrap();
        let x: Vec<_> = (0..dim).map(|i| (i % 31) as f32 * 0.003).collect();
        let expected = errors_reference(&x, &views).unwrap();
        assert_eq!(cpu.errors(&x).unwrap(), expected);
        let check = || {
            let actual = metal.errors(&x).unwrap();
            for (i, (a, b)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (a - b).abs() <= 2e-5 * (1.0 + b.abs()),
                    "dim {dim}, task {i}: {a} != {b}"
                );
            }
        };
        check();
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(check);
            }
        });
        assert_eq!(metal.gpu_submissions(), 5);
        assert_eq!(cpu.gpu_submissions(), 0);
        let mut invalid = x.clone();
        invalid[0] = f32::NAN;
        assert!(metal.errors(&invalid).is_err());
        assert!(metal.errors(&x[1..]).is_err());
        assert!(metal.errors_into(&x, &mut []).is_err());
        assert_eq!(metal.gpu_submissions(), 5);
    }
    let empty = Packed::new(&[]).unwrap().with_device(device).unwrap();
    assert!(empty.errors(&[]).unwrap().is_empty());
}
