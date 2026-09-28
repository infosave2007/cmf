use cortiq_decision::bert::{EncoderDevice, EncoderExport, GOLDEN_MAX_ABS};
use std::path::PathBuf;

pub fn encoder_parity_concurrency(device: EncoderDevice) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/toy/encoder");
    let cpu = EncoderExport::read(path).unwrap().encoder().unwrap();
    let metal = cpu.clone().with_device(device).unwrap();
    let long = "a ".repeat(200);
    let texts = [
        "",
        "hello world",
        "café naïve 😀",
        "Привет, мир",
        "日本語のテキスト",
        &long,
    ];
    let expected: Vec<_> = texts.iter().map(|t| cpu.encode(t)).collect();
    let compare = |i: usize| {
        let got = metal.try_embed_ids(&metal.tokenize(texts[i])).unwrap();
        let delta = got
            .iter()
            .zip(&expected[i])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(delta <= GOLDEN_MAX_ABS, "row {i}: {delta:e}");
    };
    for i in 0..texts.len() {
        compare(i);
    }
    assert_eq!(metal.gpu_submissions(), texts.len() as u64);
    std::thread::scope(|scope| {
        for i in 0..texts.len() {
            let compare = &compare;
            scope.spawn(move || compare(i));
        }
    });
    assert_eq!(metal.gpu_submissions(), 2 * texts.len() as u64);
    assert_eq!(cpu.gpu_submissions(), 0);
    assert!(metal.try_embed_ids(&[]).is_err());
    assert!(metal.try_embed_ids(&[u32::MAX]).is_err());
    assert!(
        metal
            .try_embed_ids(&vec![0; cpu.model().dims().max_position + 1])
            .is_err()
    );
    assert_eq!(metal.gpu_submissions(), 2 * texts.len() as u64);
}

pub fn encoder_tails_heads(device: EncoderDevice) {
    use cortiq_decision::bert::{BertModel, Encoder, Weights};
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/toy/encoder");
    let ex = EncoderExport::read(path).unwrap();
    for (hidden, heads, intermediate) in [(15, 3, 23), (65, 1, 79), (40, 5, 24)] {
        let mut config = ex.config.clone();
        config.hidden = hidden;
        config.heads = heads;
        config.head_dim = hidden / heads;
        config.intermediate = intermediate;
        // The aligned direct kernel must allocate a complete final token tile.
        config.max_position = 25;
        let shapes: std::collections::BTreeMap<_, _> = config.weight_shapes().into_iter().collect();
        let model = BertModel::from_weights(&config, |name| {
            let len = shapes[name].iter().product();
            Ok(Weights::Owned(
                (0..len)
                    .map(|i| {
                        if name.ends_with("LayerNorm.weight") {
                            1.0
                        } else {
                            ((i * 13 + name.len() * 7) % 113) as f32 * 0.0001 - 0.0056
                        }
                    })
                    .collect(),
            ))
        })
        .unwrap();
        let cpu = Encoder::new(ex.wordpiece().unwrap(), model).unwrap();
        let gpu = cpu.clone().with_device(device).unwrap();
        for text in [
            "",
            "hello world",
            "This sequence checks tails and both attention implementations.",
        ] {
            let a = cpu.encode(text);
            let b = gpu.encode(text);
            let max = a
                .iter()
                .zip(&b)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(max <= GOLDEN_MAX_ABS, "hidden={hidden}, max={max:e}");
        }
        let ids = vec![0; config.max_position as usize];
        let a = cpu.try_embed_ids(&ids).unwrap();
        let b = gpu.try_embed_ids(&ids).unwrap();
        assert!(
            a.iter()
                .zip(b)
                .all(|(a, b)| (a - b).abs() <= GOLDEN_MAX_ABS)
        );
        assert_eq!(gpu.gpu_submissions(), 4);
    }
}
