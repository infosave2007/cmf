//! Per-shape A/B of the two q4tp matmat arms, measured inside one
//! process (`CMF_MM_AB=1`).
//!
//! It exists because wall-clock A/B on a shared machine lies. Three
//! separate runs of the same change on the same stand read 2.4x, 1.0x
//! and 1.0x, and one 8-second render's denoise drifted 44.9 s -> 52.5 s
//! across six back-to-back repeats — a 25% band that swallows any real
//! effect smaller than itself. Interleaving whole processes does not
//! help: the drift is slower than a process.
//!
//! So both arms run back to back on the same activations inside the
//! same call, and what is reported is their RATIO per shape. Whatever
//! the machine is doing to one, it is doing to the other. The maximum
//! disagreement between the two outputs comes along for free, which is
//! the check that says the faster arm is also the right one.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

#[derive(Default, Clone)]
struct Row {
    calls: u64,
    gpu_ns: u128,
    cpu_ns: u128,
    refused: u64,
    worst: f32,
}

fn table() -> &'static Mutex<BTreeMap<(usize, usize, usize), Row>> {
    static T: OnceLock<Mutex<BTreeMap<(usize, usize, usize), Row>>> = OnceLock::new();
    T.get_or_init(|| Mutex::new(BTreeMap::new()))
}

pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_MM_AB").as_deref() == Ok("1"))
}

/// Calls seen, so a report can say "nothing measured" rather than
/// print an empty table that reads like "no difference".
static SEEN: AtomicU64 = AtomicU64::new(0);

#[allow(clippy::too_many_arguments)]
pub fn record(
    b: usize,
    rows: usize,
    cols: usize,
    gpu_took_it: bool,
    gpu: Duration,
    cpu: Duration,
    g: &[f32],
    c: &[f32],
) {
    SEEN.fetch_add(1, Ordering::Relaxed);
    let mut t = table().lock().unwrap();
    let e = t.entry((b, rows, cols)).or_default();
    e.calls += 1;
    e.cpu_ns += cpu.as_nanos();
    if gpu_took_it {
        e.gpu_ns += gpu.as_nanos();
        // Relative to the CPU arm's own scale, so a quiet row and a loud
        // one are comparable.
        let scale = c.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-6);
        let d = g
            .iter()
            .zip(c)
            .fold(0f32, |m, (&a, &b)| m.max((a - b).abs()))
            / scale;
        e.worst = e.worst.max(d);
    } else {
        e.refused += 1;
    }
}

/// One line per shape, widest first — the shapes that dominate a render
/// are the ones worth reading.
pub fn report() -> String {
    if SEEN.load(Ordering::Relaxed) == 0 {
        return "no q4tp matmat calls were eligible for the device arm".into();
    }
    let t = table().lock().unwrap();
    let mut rows: Vec<_> = t.iter().collect();
    rows.sort_by_key(|(_, r)| std::cmp::Reverse(r.cpu_ns));
    let mut s = String::from(
        "\n  q4tp matmat, both arms per call (CMF_MM_AB=1)\n\
         \x20   b     rows    cols  calls    gpu ms    cpu ms   ratio  worst\n",
    );
    let (mut tg, mut tc) = (0u128, 0u128);
    for ((b, r, c), e) in rows {
        let took = e.calls - e.refused;
        let g = e.gpu_ns as f64 / 1e6;
        let cp = e.cpu_ns as f64 / 1e6;
        tg += e.gpu_ns;
        tc += e.cpu_ns;
        let ratio = if took > 0 && g > 0.0 {
            format!("{:.2}x", cp / g)
        } else {
            "refused".into()
        };
        s.push_str(&format!(
            "  {b:>5} {r:>8} {c:>7} {:>6} {g:>9.1} {cp:>9.1} {ratio:>7} {:>6.4}\n",
            e.calls, e.worst
        ));
    }
    s.push_str(&format!(
        "  total gpu {:.1} ms, cpu {:.1} ms — the device arm is {:.2}x the host's\n",
        tg as f64 / 1e6,
        tc as f64 / 1e6,
        tc as f64 / (tg as f64).max(1.0),
    ));
    s
}
