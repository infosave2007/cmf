#[test]
fn g4_chat_answer_nll() {
    let p = "/Users/oleg/Documents/cortiq-bot/models/g4moe-q4t.cmf";
    if !std::path::Path::new(p).exists() { return; }
    let m = std::sync::Arc::new(cortiq_core::CmfModel::open(p).unwrap());
    let mut pl = cortiq_engine::Pipeline::from_model(&m, cortiq_engine::SamplerConfig::default()).unwrap();
    let msgs = vec![("user".to_string(),
        "Explain briefly why memory-mapped files make loading large models fast.".to_string())];
    let prompt_ids = pl.tokenizer.apply_chat_template(&msgs);
    let answer = "Memory-mapped files make loading large models fast because the operating system maps the file into the process address space instead of copying it. Pages are loaded lazily on first access, so startup is almost instant and unused weights never occupy RAM.";
    let mut ids = prompt_ids.clone();
    ids.extend(pl.tokenizer.encode(answer));
    let start = prompt_ids.len();
    let (nll, cnt) = pl.nll_ids_from(&ids, start);
    eprintln!("chat-answer ppl = {:.3} over {} tokens (prompt {} ids)",
        (nll / cnt.max(1) as f64).exp(), cnt, start);
    // Self-consistency: score the model's OWN greedy continuation.
    let cfg = cortiq_engine::SamplerConfig {
        temperature: 0.0,
        repetition_penalty: 1.0,
        ..Default::default()
    };
    let mut pl2 = cortiq_engine::Pipeline::from_model(&m, cfg).unwrap();
    let out = pl2
        .generate_from_ids(&prompt_ids, 40, None, None)
        .unwrap();
    // token_ids is the GENERATED slice only (prompt excluded).
    let mut ids2 = prompt_ids.clone();
    ids2.extend(out.token_ids.iter().copied());
    let (nll2, cnt2) = pl.nll_ids_from(&ids2, start);
    eprintln!("self-greedy ppl = {:.3} over {} tokens", (nll2 / cnt2.max(1) as f64).exp(), cnt2);
    // The REAL consistency check: at each scored position the scorer's
    // argmax must equal the token greedy decode produced there.
    let mut ok = 0usize;
    let mut tot = 0usize;
    for pos in start - 1..ids2.len() - 1 {
        let lg = pl.prefill_next_logits(&ids2[..pos + 1], None);
        let am = lg
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        tot += 1;
        if am == ids2[pos + 1] {
            ok += 1;
        } else {
            let mut v: Vec<(usize, f32)> = lg.iter().cloned().enumerate().collect();
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let want = lg[ids2[pos + 1] as usize];
            eprintln!(
                "  mismatch @pos {pos}: scorer top1={} ({:.4}) top2={} ({:.4}) | generated tok {} lg {:.4} (gap {:.5})",
                v[0].0, v[0].1, v[1].0, v[1].1, ids2[pos + 1], want, v[0].1 - want
            );
        }
    }
    eprintln!("argmax match: {ok}/{tot}");
    // Scorer/decoder parity: at most a couple of near-tie flips from the
    // int8-activation fast paths; a real path bug fails by a mile.
    assert!(ok + 2 >= tot, "scorer argmax diverges from greedy decode: {ok}/{tot}");
    let sg = (nll2 / cnt2.max(1) as f64).exp();
    assert!(sg < 3.0, "self-greedy ppl {sg:.2} — scorer disagrees with decode path");
}
