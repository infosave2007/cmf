//! `StreamSampler::with_retrieval`: planted retrieval tasks stay consistent
//! across the carried windows of one stream. The shard is random text in
//! `1..8192`, so a planted key/value (upper 3/4 of the vocabulary) is
//! recognisable, and a span copy is the only way a token trigram repeats.

use cortiq_embryo::train::{Mix, Retrieval, Shard, StreamSampler};
use std::collections::HashMap;

const NATURAL: u32 = 8192;

#[test]
fn retrieval_tasks_hold_across_windows() {
    let mut z = 0x1234_5678_9abc_def0u64;
    let tokens: Vec<u16> = (0..1usize << 21)
        .map(|_| {
            z = z.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (1 + (z >> 33) % (NATURAL as u64 - 1)) as u16
        })
        .collect();
    let mix = Mix {
        shards: vec![Shard { tokens }],
        weights: vec![1.0],
    };
    let (b, t, windows) = (4, 256, 48);
    let (dmin, dmax, pairs) = (16, 600, 4);
    let mut st = StreamSampler::new(b, t, 7, 0, None).with_retrieval(Retrieval {
        rate: 2.0,
        dmin,
        dmax,
        max_pairs: pairs,
        vocab: 32768,
    });
    // per row: the stream (tokens) and whether the target predicting
    // stream[i] is scored
    let mut stream: Vec<Vec<u32>> = vec![Vec::new(); b];
    let mut scored: Vec<Vec<bool>> = vec![Vec::new(); b];
    let mut tail: Vec<Option<u32>> = vec![None; b];
    for w in 0..windows {
        let (mut tok, mut tar) = (Vec::new(), Vec::new());
        let reset = st.batch_mix(&mix, &mut tok, &mut tar);
        for r in 0..b {
            assert_eq!(reset[r], w == 0, "a long shard and reset_every 0 restart only at the start");
            let (tk, tg) = (&tok[r * t..(r + 1) * t], &tar[r * t..(r + 1) * t]);
            // the last target of the previous window is this window's first token
            if let Some(prev) = tail[r] {
                assert!(prev == u32::MAX || prev == tk[0], "window seam broke in row {r} window {w}");
            }
            for i in 0..t - 1 {
                assert!(tg[i] == u32::MAX || tg[i] == tk[i + 1], "target is not the next token");
            }
            if w == 0 {
                scored[r].push(true);
            }
            for i in 0..t {
                stream[r].push(tk[i]);
                if i + 1 < t {
                    scored[r].push(tg[i] != u32::MAX);
                }
            }
            scored[r].push(tg[t - 1] != u32::MAX);
            tail[r] = Some(tg[t - 1]);
        }
    }
    let (mut values, mut copies) = (0usize, 0usize);
    for r in 0..b {
        let (s, sc) = (&stream[r], &scored[r]);
        let mut bigrams: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
        let mut trigrams: HashMap<(u32, u32, u32), Vec<usize>> = HashMap::new();
        for i in 1..s.len() {
            if s[i] >= NATURAL && sc[i] {
                // a scored planted token is a queried value: its (key, value)
                // pair was written earlier, at the task distance
                let d = bigrams
                    .get(&(s[i - 1], s[i]))
                    .and_then(|v| v.last())
                    .map(|&j| i - j)
                    .unwrap_or_else(|| panic!("row {r}: value at {i} was never planted"));
                assert!(d >= dmin && d <= dmax + 2 * pairs + 1, "row {r}: value distance {d}");
                values += 1;
            }
            if i >= 2 && s[i - 2] < NATURAL && s[i - 1] < NATURAL && s[i] < NATURAL {
                let key = (s[i - 2], s[i - 1], s[i]);
                if let Some(&j) = trigrams.get(&key).and_then(|v| v.last()) {
                    // a repeated trigram of random text is a span copy: scored,
                    // at the task distance (span 8..32 plus D)
                    assert!(sc[i], "row {r}: copied token at {i} is not scored");
                    let d = i - j;
                    assert!(d >= dmin + 8 && d <= dmax + 32, "row {r}: copy distance {d}");
                    copies += 1;
                }
                trigrams.entry(key).or_default().push(i);
            }
            bigrams.entry((s[i - 1], s[i])).or_default().push(i);
        }
    }
    let (tasks, targets) = st.planted;
    assert!(values > 10 && copies > 50, "values {values}, copies {copies}");
    assert!(values + copies <= targets, "served {} > planted {targets}", values + copies);
    assert!(tasks > b * windows, "only {tasks} tasks planted");
    eprintln!("tasks {tasks}, targets {targets}, served values {values}, copies {copies}");
}

#[test]
fn retrieval_off_keeps_the_plain_stream() {
    let tokens: Vec<u16> = (0..100_000u32).map(|k| (k % 5003) as u16).collect();
    let mix = Mix {
        shards: vec![Shard { tokens }],
        weights: vec![1.0],
    };
    let mut a = StreamSampler::new(3, 128, 5, 4, None);
    let mut b = StreamSampler::new(3, 128, 5, 4, None);
    for _ in 0..12 {
        let (mut ta, mut ga, mut tb, mut gb) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        assert_eq!(a.batch_mix(&mix, &mut ta, &mut ga), b.batch_mix(&mix, &mut tb, &mut gb));
        assert!(gb.iter().all(|&x| x != u32::MAX));
        assert_eq!((ta, ga), (tb, gb));
    }
    assert_eq!(a.planted, (0, 0));
}
