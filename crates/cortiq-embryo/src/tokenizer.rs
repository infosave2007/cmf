//! Our own byte-level BPE: trainer + encoder + HF `tokenizer.json` writer.
//! Byte-level (GPT-2 byte↔unicode map), pre-tokenised by the same Split
//! regex the runtime's tokenizer defaults to, so the runtime loads the
//! file as-is and encodes identically (tests/tokenizer_parity.rs).

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::Path;

/// The pre-tokenizer split (the runtime's DEFAULT_SPLIT, GPT-2 family).
pub const SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub const EOT: &str = "<|endoftext|>";
pub const SPECIALS: [&str; 8] = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|pad|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|reserved|>",
];

/// GPT-2 byte → printable unicode map (and back).
pub fn bytes_to_unicode() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).collect();
    bs.extend(0xA1..=0xAC);
    bs.extend(0xAE..=0xFF);
    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut enc = ['\0'; 256];
    let mut dec = HashMap::new();
    for (b, c) in bs.iter().zip(&cs) {
        let ch = char::from_u32(*c).unwrap();
        enc[*b as usize] = ch;
        dec.insert(ch, *b as u8);
    }
    (enc, dec)
}

/// A trained (or loaded) BPE: token strings in byte-level unicode form.
pub struct Bpe {
    pub vocab: HashMap<String, u32>,
    pub id_to_token: Vec<String>,
    /// merge ranks over (left, right) token strings
    pub ranks: HashMap<(String, String), u32>,
    pub merges: Vec<(String, String)>,
    pub specials: Vec<(String, u32)>,
    re: fancy_regex::Regex,
    byte_enc: [char; 256],
}

/// Word-frequency table from text: pre-tokenise, count byte-level words.
pub fn count_words(text: &str, re: &fancy_regex::Regex, counts: &mut HashMap<String, u64>) {
    let (enc, _) = bytes_to_unicode();
    for m in re.find_iter(text) {
        let Ok(m) = m else { continue };
        let w: String = m.as_str().bytes().map(|b| enc[b as usize]).collect();
        *counts.entry(w).or_insert(0) += 1;
    }
}

/// Train byte-level BPE to `vocab_size` (including the 256 bytes and the
/// specials) from word counts. Standard incremental algorithm: pair counts
/// weighted by word frequency, merge the most frequent pair, update only
/// the words containing it.
pub fn train(counts: &HashMap<String, u64>, vocab_size: usize, log: bool) -> Bpe {
    let (enc, _) = bytes_to_unicode();
    let n_merges = vocab_size - 256 - SPECIALS.len();
    // symbol table: 0..256 = bytes (as their unicode chars)
    let mut symbols: Vec<String> = (0..256).map(|b| enc[b].to_string()).collect();
    let mut sym_id: HashMap<String, u32> = symbols.iter().enumerate().map(|(i, s)| (s.clone(), i as u32)).collect();
    // words as symbol-id sequences
    let mut words: Vec<Vec<u32>> = Vec::with_capacity(counts.len());
    let mut freqs: Vec<u64> = Vec::with_capacity(counts.len());
    for (w, &f) in counts {
        words.push(w.chars().map(|c| sym_id[&c.to_string()]).collect());
        freqs.push(f);
    }
    // pair → (count, set of word indices)
    let mut pair_count: HashMap<(u32, u32), i64> = HashMap::new();
    let mut pair_words: HashMap<(u32, u32), HashSet<u32>> = HashMap::new();
    for (wi, w) in words.iter().enumerate() {
        for k in 0..w.len().saturating_sub(1) {
            let p = (w[k], w[k + 1]);
            *pair_count.entry(p).or_insert(0) += freqs[wi] as i64;
            pair_words.entry(p).or_default().insert(wi as u32);
        }
    }
    let mut merges: Vec<(String, String)> = Vec::with_capacity(n_merges);
    // max-heap with lazy invalidation: entries (count, pair); an entry is
    // live iff its count equals the pair's current count.
    let mut heap: std::collections::BinaryHeap<(i64, std::cmp::Reverse<(u32, u32)>)> =
        pair_count.iter().map(|(p, c)| (*c, std::cmp::Reverse(*p))).collect();
    let t0 = std::time::Instant::now();
    for mi in 0..n_merges {
        let mut top = None;
        while let Some((c, std::cmp::Reverse(p))) = heap.pop() {
            if pair_count.get(&p).copied().unwrap_or(0) == c && c > 0 {
                top = Some((p, c));
                break;
            }
        }
        let Some((best, bc)) = top else { break };
        if bc <= 1 {
            break;
        }
        let new_sym = format!("{}{}", symbols[best.0 as usize], symbols[best.1 as usize]);
        let new_id = symbols.len() as u32;
        symbols.push(new_sym.clone());
        sym_id.insert(new_sym, new_id);
        merges.push((symbols[best.0 as usize].clone(), symbols[best.1 as usize].clone()));
        // apply to every word containing the pair
        let affected: Vec<u32> = pair_words.remove(&best).map(|s| s.into_iter().collect()).unwrap_or_default();
        pair_count.remove(&best);
        let mut touched: HashSet<(u32, u32)> = HashSet::new();
        for wi in affected {
            let w = &mut words[wi as usize];
            let f = freqs[wi as usize] as i64;
            for k in 0..w.len().saturating_sub(1) {
                let p = (w[k], w[k + 1]);
                if let Some(c) = pair_count.get_mut(&p) {
                    *c -= f;
                    touched.insert(p);
                }
                if let Some(s) = pair_words.get_mut(&p) {
                    s.remove(&wi);
                }
            }
            let mut nw = Vec::with_capacity(w.len());
            let mut k = 0;
            while k < w.len() {
                if k + 1 < w.len() && w[k] == best.0 && w[k + 1] == best.1 {
                    nw.push(new_id);
                    k += 2;
                } else {
                    nw.push(w[k]);
                    k += 1;
                }
            }
            *w = nw;
            for k in 0..w.len().saturating_sub(1) {
                let p = (w[k], w[k + 1]);
                *pair_count.entry(p).or_insert(0) += f;
                pair_words.entry(p).or_default().insert(wi);
                touched.insert(p);
            }
        }
        for p in touched {
            if p == best {
                continue;
            }
            let c = pair_count.get(&p).copied().unwrap_or(0);
            if c > 0 {
                heap.push((c, std::cmp::Reverse(p)));
            }
        }
        if log && (mi % 2000 == 0 || mi + 1 == n_merges) {
            eprintln!("bpe merge {mi}/{n_merges}: {:?} count {bc}  [{:.0} s]", merges.last().unwrap(), t0.elapsed().as_secs_f64());
        }
    }
    // vocab: bytes, merges, then specials at the end
    let mut vocab = HashMap::new();
    let mut id_to_token = Vec::new();
    for s in &symbols {
        vocab.insert(s.clone(), id_to_token.len() as u32);
        id_to_token.push(s.clone());
    }
    let mut specials = Vec::new();
    // pad up to vocab_size − specials with reserved ids so the specials sit
    // at fixed positions at the very end
    while id_to_token.len() < vocab_size - SPECIALS.len() {
        let name = format!("<|unused{}|>", id_to_token.len());
        vocab.insert(name.clone(), id_to_token.len() as u32);
        id_to_token.push(name);
    }
    for s in SPECIALS {
        let id = id_to_token.len() as u32;
        vocab.insert(s.to_string(), id);
        id_to_token.push(s.to_string());
        specials.push((s.to_string(), id));
    }
    let ranks = merges.iter().enumerate().map(|(i, m)| (m.clone(), i as u32)).collect();
    Bpe { vocab, id_to_token, ranks, merges, specials, re: fancy_regex::Regex::new(SPLIT).unwrap(), byte_enc: enc }
}

impl Bpe {
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }
    pub fn special_id(&self, name: &str) -> Option<u32> {
        self.specials.iter().find(|(s, _)| s == name).map(|(_, i)| *i)
    }

    /// BPE-merge one pre-token (byte-level unicode string) into ids.
    fn bpe_word(&self, word: &str, out: &mut Vec<u32>) {
        let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..parts.len().saturating_sub(1) {
                if let Some(&r) = self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())) {
                    if best.is_none_or(|(br, _)| r < br) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", parts[i], parts[i + 1]);
            parts.splice(i..i + 2, [merged]);
        }
        for p in parts {
            match self.vocab.get(&p) {
                Some(&id) => out.push(id),
                None => {
                    // unreachable for byte-level (every char is a byte token)
                    for ch in p.chars() {
                        out.push(self.vocab[&ch.to_string()]);
                    }
                }
            }
        }
    }

    /// Encode text (no specials inside; the caller appends EOT between
    /// documents). `cache` speeds repeated words up.
    pub fn encode(&self, text: &str, cache: &mut HashMap<String, Vec<u32>>, out: &mut Vec<u32>) {
        for m in self.re.find_iter(text) {
            let Ok(m) = m else { continue };
            let w: String = m.as_str().bytes().map(|b| self.byte_enc[b as usize]).collect();
            if let Some(ids) = cache.get(&w) {
                out.extend_from_slice(ids);
                continue;
            }
            let mut ids = Vec::new();
            self.bpe_word(&w, &mut ids);
            out.extend_from_slice(&ids);
            if cache.len() < 2_000_000 {
                cache.insert(w, ids);
            }
        }
    }

    /// Decode ids to text (byte-level inverse; specials emitted raw).
    pub fn decode(&self, ids: &[u32]) -> String {
        let (_, dec) = bytes_to_unicode();
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(tok) = self.id_to_token.get(id as usize) else { continue };
            if self.specials.iter().any(|(s, _)| s == tok) {
                bytes.extend_from_slice(tok.as_bytes());
                continue;
            }
            for ch in tok.chars() {
                match dec.get(&ch) {
                    Some(b) => bytes.push(*b),
                    None => bytes.extend_from_slice(ch.to_string().as_bytes()),
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// HF `tokenizer.json` (byte-level BPE with an explicit Split regex —
    /// exactly what the runtime's `Tokenizer::from_json` reads).
    pub fn to_hf_json(&self) -> String {
        let mut vocab_sorted: Vec<(&String, &u32)> = self.vocab.iter().collect();
        vocab_sorted.sort_by_key(|(_, id)| **id);
        let vocab: serde_json::Map<String, serde_json::Value> =
            vocab_sorted.iter().map(|(t, id)| ((*t).clone(), serde_json::json!(**id))).collect();
        let merges: Vec<String> = self.merges.iter().map(|(a, b)| format!("{a} {b}")).collect();
        let added: Vec<serde_json::Value> = self
            .specials
            .iter()
            .map(|(s, id)| {
                serde_json::json!({"id": id, "content": s, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true})
            })
            .collect();
        let j = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added,
            "normalizer": null,
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": SPLIT}, "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
            ]},
            "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false},
            "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                      "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                      "vocab": vocab, "merges": merges}
        });
        serde_json::to_string_pretty(&j).unwrap()
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let mut f = std::fs::File::create(path)?;
        f.write_all(self.to_hf_json().as_bytes())?;
        Ok(())
    }

    /// Load our own tokenizer.json back (vocab + merges + specials).
    pub fn load(path: &Path) -> anyhow::Result<Bpe> {
        let s = std::fs::read_to_string(path)?;
        let j: serde_json::Value = serde_json::from_str(&s)?;
        let vocab_j = j["model"]["vocab"].as_object().ok_or_else(|| anyhow::anyhow!("no vocab"))?;
        let mut vocab = HashMap::new();
        let mut max_id = 0u32;
        for (k, v) in vocab_j {
            let id = v.as_u64().unwrap() as u32;
            vocab.insert(k.clone(), id);
            max_id = max_id.max(id);
        }
        let mut id_to_token = vec![String::new(); max_id as usize + 1];
        for (k, &id) in &vocab {
            id_to_token[id as usize] = k.clone();
        }
        let mut merges = Vec::new();
        for m in j["model"]["merges"].as_array().unwrap_or(&Vec::new()) {
            if let Some(s) = m.as_str() {
                let mut it = s.splitn(2, ' ');
                if let (Some(a), Some(b)) = (it.next(), it.next()) {
                    merges.push((a.to_string(), b.to_string()));
                }
            } else if let Some(arr) = m.as_array() {
                merges.push((arr[0].as_str().unwrap().to_string(), arr[1].as_str().unwrap().to_string()));
            }
        }
        let mut specials = Vec::new();
        for at in j["added_tokens"].as_array().unwrap_or(&Vec::new()) {
            let content = at["content"].as_str().unwrap().to_string();
            let id = at["id"].as_u64().unwrap() as u32;
            vocab.insert(content.clone(), id);
            if id as usize >= id_to_token.len() {
                id_to_token.resize(id as usize + 1, String::new());
            }
            id_to_token[id as usize] = content.clone();
            specials.push((content, id));
        }
        let ranks = merges.iter().enumerate().map(|(i, m)| (m.clone(), i as u32)).collect();
        let (enc, _) = bytes_to_unicode();
        Ok(Bpe { vocab, id_to_token, ranks, merges, specials, re: fancy_regex::Regex::new(SPLIT).unwrap(), byte_enc: enc })
    }
}
