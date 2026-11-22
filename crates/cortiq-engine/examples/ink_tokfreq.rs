//! Token frequencies of a corpus under a CMF file's own tokenizer, for the
//! DSpark drafter's table (`CMF_INKLING_DRAFT_TABLE=RANKING:N`): the special
//! tokens first, then every token seen by count, one id a line
//! (`ink_tokfreq FILE.cmf OUT corpus...`).
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let model = cortiq_core::CmfModel::open(&a[1]).expect("cmf");
    let tok = cortiq_engine::tokenizer::Tokenizer::from_bytes(model.vocab.as_ref().expect("embedded vocab")).expect("tokenizer");
    let mut count = vec![0u64; tok.vocab_size().max(1)];
    let mut total = 0u64;
    for f in &a[3..] {
        let text = std::fs::read_to_string(f).unwrap_or_default();
        // pieces of about 16 KB cut after a newline: the newlines and indents stay
        let mut at = 0usize;
        while at < text.len() {
            let mut end = (at + 16384).min(text.len());
            while end < text.len() && !text.is_char_boundary(end) {
                end += 1;
            }
            if let Some(nl) = text[at..end].rfind('\n').filter(|&i| i > 0 && end < text.len()) {
                end = at + nl + 1;
            }
            for id in tok.encode_plain(&text[at..end]) {
                if (id as usize) >= count.len() {
                    count.resize(id as usize + 1, 0);
                }
                count[id as usize] += 1;
                total += 1;
            }
            at = end;
        }
    }
    let mut special: Vec<u32> = tok.special_texts().iter().filter_map(|t| tok.token_to_id(t)).collect();
    special.sort_unstable();
    special.dedup();
    let mut ids: Vec<u32> = (0..count.len() as u32).filter(|&i| count[i as usize] > 0 && !special.contains(&i)).collect();
    ids.sort_by(|&x, &y| count[y as usize].cmp(&count[x as usize]).then(x.cmp(&y)));
    let cover = |n: usize| ids[..n.min(ids.len())].iter().map(|&i| count[i as usize]).sum::<u64>() as f64 / total.max(1) as f64;
    eprintln!("{total} tokens, {} distinct, {} special; top 16K {:.4}, 32K {:.4}", ids.len(), special.len(), cover(16384), cover(32768));
    let out: String = special.iter().chain(&ids).map(|i| format!("{i}\n")).collect();
    std::fs::write(&a[2], out).expect("write");
}
