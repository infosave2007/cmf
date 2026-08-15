//! Which layout do a model's MoE experts actually use?
//!
//!     CMF_Q4TP_PARITY=model.cmf cargo test -p cortiq-engine --test expert_dtypes -- --nocapture
//!
//! Written because a "q2tp toy" agreed bit for bit with the CPU while the
//! q2tp release did not — the first thing to rule out is that the toy's
//! experts came out q4tp and the test never touched the path it named.

#[test]
fn report_expert_layouts() {
    let Ok(path) = std::env::var("CMF_Q4TP_PARITY") else {
        return;
    };
    let m = cortiq_core::CmfModel::open(&path).expect("open");
    let mut seen = std::collections::BTreeMap::new();
    for e in m.tensors.iter() {
        if !e.name.contains(".experts.") && !e.name.contains("shared_expert") {
            continue;
        }
        let role = if e.name.ends_with("gate_proj.weight") {
            "gate"
        } else if e.name.ends_with("up_proj.weight") {
            "up"
        } else if e.name.ends_with("down_proj.weight") {
            "down"
        } else {
            continue;
        };
        *seen
            .entry((role, format!("{:?}", e.dtype)))
            .or_insert(0usize) += 1;
    }
    for ((role, dt), n) in seen {
        println!("{role:>5}: {dt} × {n}");
    }
}
