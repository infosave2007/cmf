#[test]
fn g4_bos_probe() {
    let p = "/Users/oleg/Documents/cortiq-bot/models/g4moe-q4t.cmf";
    if !std::path::Path::new(p).exists() {
        return;
    }
    let m = std::sync::Arc::new(cortiq_core::CmfModel::open(p).unwrap());
    let t = cortiq_engine::tokenizer::Tokenizer::from_bytes(m.vocab.as_ref().unwrap()).unwrap();
    eprintln!("add_bos={} bos_id={:?}", t.add_bos, t.bos_token_id);
    let ids = t.encode("Memory-mapped files");
    eprintln!("ids[0..4]={:?}", &ids[..ids.len().min(4)]);
    for id in [1u32, 2, 105, 106, 236820] {
        eprintln!("id {id}={:?}", t.decode(&[id]));
    }
    eprintln!("enc sot len={}", t.encode("<start_of_turn>").len());
}
