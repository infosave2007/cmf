#[test]
fn g3n_tokenizer_roundtrip() {
    let p = "/Users/oleg/Documents/cortiq-bot/models/g3n-e4b-it-q4t.cmf";
    if !std::path::Path::new(p).exists() { return; }
    let m = std::sync::Arc::new(cortiq_core::CmfModel::open(p).unwrap());
    let pl = cortiq_engine::Pipeline::from_model(&m, cortiq_engine::SamplerConfig::default()).unwrap();
    let text = "def f():\n    return 1";
    let ids = pl.tokenizer.encode(text);
    let full = pl.tokenizer.decode(&ids);
    let stream: String = ids.iter().map(|&i| pl.tokenizer.decode_token(i)).collect();
    eprintln!("ids: {ids:?}");
    eprintln!("full  : {full:?}");
    eprintln!("stream: {stream:?}");
    eprintln!("tok debug: {:?}", pl.tokenizer);
    // Raw repo json, bypassing the bundle.
    let raw = "/private/tmp/claude-501/-Users-oleg-Documents-cortiq-bot-cmfpublic/33ea22c3-8c05-487e-8044-e5e6683e17b8/scratchpad/g3n_tok.json";
    if std::path::Path::new(raw).exists() {
        let t2 = cortiq_engine::tokenizer::Tokenizer::from_file(raw).unwrap();
        let ids2 = t2.encode(text);
        eprintln!("raw ids: {ids2:?}");
        eprintln!("raw full: {:?}", t2.decode(&ids2));
    }
}
