use cortiq_core::format::{CmfModel, TensorSpec};
use cortiq_embryo::model::{EmbryoCfg, Layout, init_params};

#[test]
fn phase_delta_selector_is_serialized_and_parameter_neutral() {
    let base = EmbryoCfg::embryo0();
    let mut selected = base.clone();
    selected.phase_delta_layer = Some(3);

    let base_layout = Layout::new(&base);
    let selected_layout = Layout::new(&selected);
    assert_eq!(base_layout.total, selected_layout.total);
    assert_eq!(base_layout.names, selected_layout.names);
    assert_eq!(base.params(), selected.params());
    assert_eq!(
        init_params(&base, &base_layout, 1),
        init_params(&selected, &selected_layout, 1)
    );

    let encoded = serde_json::to_string(&selected).unwrap();
    assert!(encoded.contains("phase_delta_layer"));
    let decoded: EmbryoCfg = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded.phase_delta_layer, Some(3));
    assert!(decoded.phase_delta_active());
    assert!(decoded.phase_delta_for_layer(3));
    assert!(!decoded.phase_delta_for_layer(2));
    assert!(!decoded.phase_delta_for_layer(7)); // anchor_every=8

    // Checkpoints/configs written before the selector existed still decode.
    let mut legacy_json = serde_json::to_value(&base).unwrap();
    legacy_json
        .as_object_mut()
        .unwrap()
        .remove("phase_delta_layer");
    let legacy: EmbryoCfg = serde_json::from_value(legacy_json).unwrap();
    assert_eq!(legacy.phase_delta_layer, None);
    assert!(!legacy.phase_delta_active());

    let path =
        std::env::temp_dir().join(format!("cmf-phase-delta-layer-{}.ckpt", std::process::id()));
    let params = init_params(&selected, &selected_layout, 9);
    cortiq_embryo::train::save_checkpoint(
        &path,
        &selected,
        23,
        &params,
        Some(&params),
        Some(&params),
        &[],
    )
    .unwrap();
    let loaded = cortiq_embryo::train::load_checkpoint(&path).unwrap();
    assert_eq!(loaded.cfg.phase_delta_layer, Some(3));
    assert_eq!(loaded.params, params);
    assert_eq!(loaded.m.as_deref(), Some(params.as_slice()));
    assert_eq!(loaded.v.as_deref(), Some(params.as_slice()));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn phase_delta_boolean_keeps_all_hybrid_research_mode() {
    let mut cfg = EmbryoCfg::embryo0();
    cfg.phase_delta = true;
    assert!(cfg.phase_delta_active());
    for layer in 0..cfg.layers {
        assert_eq!(cfg.phase_delta_for_layer(layer), !cfg.is_anchor(layer));
    }
}

#[test]
fn selected_layer_only_dispatches_middle_hybrid() {
    let mut cfg = EmbryoCfg::embryo0();
    cfg.phase_delta_layer = Some(3);
    let selected: Vec<usize> = (0..cfg.layers)
        .filter(|&l| cfg.phase_delta_for_layer(l))
        .collect();
    assert_eq!(selected, vec![3]);
    let legacy_hybrids: Vec<usize> = (0..cfg.layers)
        .filter(|&l| !cfg.is_anchor(l) && l != 3)
        .collect();
    assert_eq!(legacy_hybrids, vec![0, 1, 2, 4, 5, 6]);
}

#[test]
fn phase_delta_selector_rejects_out_of_range_and_anchor() {
    let mut out_of_range = EmbryoCfg::embryo0();
    out_of_range.phase_delta_layer = Some(out_of_range.layers);
    assert!(std::panic::catch_unwind(|| Layout::new(&out_of_range)).is_err());

    let mut anchor = EmbryoCfg::embryo0();
    anchor.phase_delta_layer = Some(7);
    assert!(std::panic::catch_unwind(|| Layout::new(&anchor)).is_err());
}

#[test]
fn phase_delta_dual_selector_is_serialized_parameter_neutral_and_precise() {
    let base = EmbryoCfg::embryo0();
    let mut dual = base.clone();
    dual.phase_delta_layers = Some(vec![3, 6]);

    let base_layout = Layout::new(&base);
    let dual_layout = Layout::new(&dual);
    assert_eq!(base_layout.total, dual_layout.total);
    assert_eq!(base_layout.names, dual_layout.names);
    assert_eq!(base.params(), dual.params());
    assert_eq!(
        init_params(&base, &base_layout, 17),
        init_params(&dual, &dual_layout, 17)
    );

    assert!(dual.phase_delta_active());
    let selected: Vec<usize> = (0..dual.layers)
        .filter(|&layer| dual.phase_delta_for_layer(layer))
        .collect();
    assert_eq!(selected, vec![3, 6]);
    assert!(!dual.phase_delta_for_layer(7)); // anchor remains attention
    assert!(!dual.phase_delta_for_layer(2));

    let encoded = serde_json::to_string(&dual).unwrap();
    assert!(encoded.contains("phase_delta_layers"));
    let decoded: EmbryoCfg = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded.phase_delta_layers, Some(vec![3, 6]));

    let path =
        std::env::temp_dir().join(format!("cmf-phase-delta-dual-{}.ckpt", std::process::id()));
    let params = init_params(&dual, &dual_layout, 23);
    cortiq_embryo::train::save_checkpoint(
        &path,
        &dual,
        29,
        &params,
        Some(&params),
        Some(&params),
        &[],
    )
    .unwrap();
    let loaded = cortiq_embryo::train::load_checkpoint(&path).unwrap();
    assert_eq!(loaded.cfg.phase_delta_layers, Some(vec![3, 6]));
    assert_eq!(loaded.params, params);
    assert_eq!(loaded.m.as_deref(), Some(params.as_slice()));
    assert_eq!(loaded.v.as_deref(), Some(params.as_slice()));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn phase_delta_dual_selector_rejects_malformed_sets() {
    for malformed in [vec![], vec![3], vec![3, 6, 5], vec![3, 3], vec![3, 7]] {
        let mut cfg = EmbryoCfg::embryo0();
        cfg.phase_delta_layers = Some(malformed);
        assert!(
            std::panic::catch_unwind(|| Layout::new(&cfg)).is_err(),
            "malformed dual selector unexpectedly accepted"
        );
    }

    let mut conflict = EmbryoCfg::embryo0();
    conflict.phase_delta_layer = Some(3);
    conflict.phase_delta_layers = Some(vec![3, 6]);
    assert!(std::panic::catch_unwind(|| Layout::new(&conflict)).is_err());
}

#[test]
fn export_uses_a_new_kind_and_canonical_per_layer_selector() {
    let mut selected = EmbryoCfg::embryo0();
    selected.phase_delta_layer = Some(3);
    let arch: cortiq_core::ModelArch =
        serde_json::from_value(cortiq_embryo::export::arch_json(&selected)).unwrap();
    let lc = arch.linear_core.unwrap();
    assert_eq!(lc.kind, "vmf_phase_delta_v1");
    assert_eq!(lc.phase_delta_layers, Some(vec![3]));

    let mut dual = EmbryoCfg::embryo0();
    // Input order is deliberately reversed; export records a canonical set.
    dual.phase_delta_layers = Some(vec![6, 3]);
    let arch: cortiq_core::ModelArch =
        serde_json::from_value(cortiq_embryo::export::arch_json(&dual)).unwrap();
    assert_eq!(
        arch.linear_core.unwrap().phase_delta_layers,
        Some(vec![3, 6])
    );

    let legacy: cortiq_core::ModelArch =
        serde_json::from_value(cortiq_embryo::export::arch_json(&EmbryoCfg::embryo0())).unwrap();
    let lc = legacy.linear_core.unwrap();
    assert_eq!(lc.kind, "vmf_phase");
    assert_eq!(lc.phase_delta_layers, None);
}

#[test]
fn export_rejects_malformed_selectors_and_unsupported_experimental_flags_before_output() {
    let base = EmbryoCfg::tiny();
    let params = init_params(&base, &Layout::new(&base), 7);
    let cases = [
        ("empty_selector", {
            let mut c = base.clone();
            c.phase_delta_layers = Some(vec![]);
            c
        }),
        ("duplicate_selector", {
            let mut c = base.clone();
            c.phase_delta_layers = Some(vec![0, 0]);
            c
        }),
        ("anchor_selector", {
            let mut c = base.clone();
            c.phase_delta_layer = Some(1);
            c
        }),
        ("gdn_tail", {
            let mut c = base.clone();
            c.gdn_lane = true;
            c
        }),
        ("gqa_tail", {
            let mut c = base.clone();
            c.gqa_lane = true;
            c
        }),
        ("router_smoothing", {
            let mut c = base.clone();
            c.router_smooth_k4 = true;
            c
        }),
        ("router_top2", {
            let mut c = base.clone();
            c.router_top2_margin = Some(0.0);
            c
        }),
    ];
    for (i, (name, cfg)) in cases.into_iter().enumerate() {
        let path = std::env::temp_dir().join(format!(
            "embryo-export-reject-{}-{i}-{}.cmf",
            std::process::id(),
            name
        ));
        let ck = cortiq_embryo::train::Checkpoint {
            cfg,
            step: 0,
            params: params.clone(),
            m: None,
            v: None,
            extras: Vec::new(),
        };
        let result = cortiq_embryo::export::export(&ck, b"{}", &path);
        assert!(result.is_err(), "unsupported {name} unexpectedly exported");
        assert!(!path.exists(), "unsupported {name} left a partial output");
    }
}

#[test]
fn runtime_rejects_missing_delta_tag_and_legacy_selector_metadata() {
    let mut cfg = EmbryoCfg::tiny();
    cfg.phase_delta_layer = Some(0);
    let params = init_params(&cfg, &Layout::new(&cfg), 31);
    let ck = cortiq_embryo::train::Checkpoint {
        cfg,
        step: 0,
        params,
        m: None,
        v: None,
        extras: Vec::new(),
    };
    let dir = std::env::temp_dir().join(format!("embryo-delta-loader-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("source.cmf");
    cortiq_embryo::export::export(&ck, b"{}", &source).unwrap();
    let source_model = CmfModel::open(&source).unwrap();
    let tensors: Vec<TensorSpec> = source_model
        .tensors
        .iter()
        .map(|entry| TensorSpec {
            name: entry.name.clone(),
            dtype: entry.dtype,
            shape: entry.shape.clone(),
            data: source_model.tensor_bytes(&entry.name).unwrap().to_vec(),
        })
        .collect();

    let rewrite = |name: &str, kind: &str, selector: Option<Vec<usize>>| {
        let mut header = source_model.header.clone();
        let lc = header.arch.linear_core.as_mut().unwrap();
        lc.kind = kind.to_string();
        lc.phase_delta_layers = selector;
        let path = dir.join(name);
        let err = CmfModel::write(&path, &header, &tensors, None, Some(b"{}"))
            .expect_err("malformed linear-core metadata unexpectedly wrote: {name}");
        assert!(
            err.to_string().contains("phase_delta") || err.to_string().contains("linear_core"),
            "unexpected malformed selector error for {name}: {err}"
        );
    };
    rewrite("missing-selector.cmf", "vmf_phase_delta_v1", None);
    rewrite("legacy-selector.cmf", "vmf_phase", Some(vec![0]));
    std::fs::remove_dir_all(dir).unwrap();
}
