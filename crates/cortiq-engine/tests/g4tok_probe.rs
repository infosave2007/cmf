#[test]
fn g4_tokenizer_probe() {
    let p = "/Users/oleg/Documents/cortiq-bot/models/g4moe-q4t.cmf";
    if !std::path::Path::new(p).exists() { return; }
    let m = std::sync::Arc::new(cortiq_core::CmfModel::open(p).unwrap());
    let t = cortiq_engine::tokenizer::Tokenizer::from_bytes(m.vocab.as_ref().unwrap()).unwrap();
    for s in ["The capital of France is Paris.", "Memory-mapped files let a program treat"] {
        let ids = t.encode(s);
        let toks: Vec<String> = ids.iter().map(|&i| t.decode(&[i])).collect();
        eprintln!("{s:?} -> {} ids: {:?}", ids.len(), &ids[..ids.len().min(12)]);
        eprintln!("   pieces: {:?}", &toks[..toks.len().min(12)]);
    }
    for id in [100u32, 107, 108, 236772, 236779, 236813] {
        eprintln!("id {id} = {:?}", t.decode(&[id]));
    }
}
