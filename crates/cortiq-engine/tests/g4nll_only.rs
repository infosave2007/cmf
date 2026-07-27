#[test]
fn g4_nll_only() {
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
}
