//! Host-only deterministic layer-4 donor-anchor graft contract.

use cortiq_embryo::model::{EmbryoCfg, Layout, init_params};
use cortiq_embryo::train::{
    Checkpoint, graft_layer4_anchor_checkpoint, load_checkpoint, save_checkpoint,
};
use std::collections::{HashMap, HashSet};

fn fixture() -> (Checkpoint, Checkpoint) {
    let mut student_cfg = EmbryoCfg::tiny();
    student_cfg.vocab = 64;
    student_cfg.layers = 8;
    student_cfg.anchor_every = 8;
    student_cfg.conv_k = 4;
    student_cfg.experts = 0;
    student_cfg.head_clusters = 0;
    student_cfg.seq = 64;
    let mut donor_cfg = student_cfg.clone();
    donor_cfg.anchor_every = 1;
    let student_lay = Layout::new(&student_cfg);
    let donor_lay = Layout::new(&donor_cfg);
    let student_params = init_params(&student_cfg, &student_lay, 7);
    let donor_params = init_params(&donor_cfg, &donor_lay, 19);
    let student = Checkpoint {
        cfg: student_cfg,
        step: 61_000,
        params: student_params.clone(),
        m: Some(student_params.iter().map(|x| x + 0.1).collect()),
        v: Some(student_params.iter().map(|x| x + 0.2).collect()),
        extras: vec![("desc.mu".into(), vec![0.25; 3])],
    };
    let donor = Checkpoint {
        cfg: donor_cfg,
        step: 61_000,
        params: donor_params,
        m: Some(vec![3.0; donor_lay.total]),
        v: Some(vec![4.0; donor_lay.total]),
        extras: vec![("desc.mu".into(), vec![9.0; 3])],
    };
    (student, donor)
}

#[test]
fn layer4_graft_is_name_exact_and_roundtrips() {
    let (student, donor) = fixture();
    let result = graft_layer4_anchor_checkpoint(&student, &donor).unwrap();
    let candidate = &result.checkpoint;
    assert_eq!(candidate.cfg.anchor_every, 4);
    assert_eq!(candidate.step, student.step);
    assert_eq!(candidate.extras, student.extras);
    assert_eq!(result.donor_tensors.len(), 4);
    assert_eq!(
        result
            .donor_tensors
            .iter()
            .map(|x| x.name.as_str())
            .collect::<Vec<_>>(),
        vec![
            "layers.3.attn.k",
            "layers.3.attn.o",
            "layers.3.attn.q",
            "layers.3.attn.v",
        ]
    );
    assert!(
        result
            .donor_tensors
            .iter()
            .all(|x| x.source == "donor.layer4.gqa")
    );
    assert_eq!(result.donor_tensors[0].shape, vec![64, 64]);

    let sl = Layout::new(&student.cfg);
    let dl = Layout::new(&donor.cfg);
    let cl = Layout::new(&candidate.cfg);
    let sm: HashMap<&str, (usize, usize)> = sl
        .names
        .iter()
        .map(|(n, o, l)| (n.as_str(), (*o, *l)))
        .collect();
    let dm: HashMap<&str, (usize, usize)> = dl
        .names
        .iter()
        .map(|(n, o, l)| (n.as_str(), (*o, *l)))
        .collect();
    let donor_names: HashSet<&str> = result
        .donor_tensors
        .iter()
        .map(|x| x.name.as_str())
        .collect();
    for (name, co, clen) in &cl.names {
        if donor_names.contains(name.as_str()) {
            let (doff, dlen) = dm[name.as_str()];
            assert_eq!(*clen, dlen);
            assert_eq!(
                &candidate.params[*co..*co + *clen],
                &donor.params[doff..doff + dlen]
            );
            assert!(
                candidate.m.as_ref().unwrap()[*co..*co + *clen]
                    .iter()
                    .all(|x| *x == 0.0)
            );
            assert!(
                candidate.v.as_ref().unwrap()[*co..*co + *clen]
                    .iter()
                    .all(|x| *x == 0.0)
            );
        } else {
            let (soff, slen) = sm[name.as_str()];
            assert_eq!(*clen, slen);
            assert_eq!(
                &candidate.params[*co..*co + *clen],
                &student.params[soff..soff + slen]
            );
            assert_eq!(
                &candidate.m.as_ref().unwrap()[*co..*co + *clen],
                &student.m.as_ref().unwrap()[soff..soff + slen]
            );
            assert_eq!(
                &candidate.v.as_ref().unwrap()[*co..*co + *clen],
                &student.v.as_ref().unwrap()[soff..soff + slen]
            );
        }
    }

    // Calling the host transform again is byte-deterministic.
    let result2 = graft_layer4_anchor_checkpoint(&student, &donor).unwrap();
    assert_eq!(candidate.params, result2.checkpoint.params);
    assert_eq!(candidate.m, result2.checkpoint.m);
    assert_eq!(candidate.v, result2.checkpoint.v);

    let path = std::env::temp_dir().join(format!("anchor-graft-{}.ckpt", std::process::id()));
    let ex = candidate
        .extras
        .iter()
        .map(|(n, x)| (n.as_str(), x.as_slice()))
        .collect::<Vec<_>>();
    save_checkpoint(
        &path,
        &candidate.cfg,
        candidate.step,
        &candidate.params,
        candidate.m.as_deref(),
        candidate.v.as_deref(),
        &ex,
    )
    .unwrap();
    let loaded = load_checkpoint(&path).unwrap();
    assert_eq!(loaded.params, candidate.params);
    assert_eq!(loaded.m, candidate.m);
    assert_eq!(loaded.v, candidate.v);
    assert_eq!(loaded.extras, candidate.extras);
    let _ = std::fs::remove_file(path);
}
