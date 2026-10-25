//! In-process CPU op timers (`CMF_CPU_PROF=1`).
//!
//! Wall-clock A/B on a shared box is not evidence (see the perf notes):
//! the same change measured 23%, 0% and 22% across whole runs. These
//! counters accumulate nanoseconds inside the process around each CPU
//! stage of the layer loop, so contention inflates every slot alike and
//! the SPLIT survives it. Off by default: one cached bool per call site.
//!
//! `report(tokens)` prints ms per token for every slot that ran.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Slot {
    /// Q/K/V projections of one decode position.
    Qkv,
    /// RoPE + append + attend (the attention core, one position).
    AttnCore,
    /// Output projection.
    AttnO,
    /// Fused gate/up (+SiLU·mul).
    FfnGateUp,
    /// Down projection.
    FfnDown,
    /// RMSNorms + residual adds of the layer loop.
    Norms,
    /// Final norm + lm_head matvec.
    Head,
    /// Sampler (argmax / top-k / penalties).
    Sampler,
    /// Whole layer stack of one forward (decode or prefill chunk).
    Layers,
    /// Batched prefill projections (every matmat).
    Matmat,
    /// Batched prefill attention core (per-position or batched attend).
    PrefillAttend,
    /// Activation quantization (split_act) inside the kernels.
    SplitAct,
}

const N: usize = 12;
const NAMES: [&str; N] = [
    "qkv",
    "attn_core",
    "attn_o",
    "ffn_gate_up",
    "ffn_down",
    "norms",
    "head",
    "sampler",
    "layers(total)",
    "matmat(prefill)",
    "attend(prefill)",
    "split_act",
];

static NS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static CALLS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];

#[inline]
pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_CPU_PROF").is_ok_and(|v| v != "0"))
}

/// Scope timer: records on drop. `None` when profiling is off.
pub struct Timer(Option<(Slot, std::time::Instant)>);

impl Drop for Timer {
    #[inline]
    fn drop(&mut self) {
        if let Some((s, t0)) = self.0.take() {
            NS[s as usize].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            CALLS[s as usize].fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[inline]
pub fn time(s: Slot) -> Timer {
    if on() {
        Timer(Some((s, std::time::Instant::now())))
    } else {
        Timer(None)
    }
}

/// Zero every slot (bench calls it right before the measured window).
pub fn reset() {
    for i in 0..N {
        NS[i].store(0, Ordering::Relaxed);
        CALLS[i].store(0, Ordering::Relaxed);
    }
}

/// Snapshot of (name, total ms, calls) for every slot that ran.
pub fn snapshot() -> Vec<(&'static str, f64, u64)> {
    (0..N)
        .filter(|&i| CALLS[i].load(Ordering::Relaxed) > 0)
        .map(|i| {
            (
                NAMES[i],
                NS[i].load(Ordering::Relaxed) as f64 / 1e6,
                CALLS[i].load(Ordering::Relaxed),
            )
        })
        .collect()
}

/// Labeled stage spans (`SAM3_PROF=1`) — the generic sibling of [`time`]:
/// a process-wide map label → (ms, calls), accumulated by RAII spans and
/// printed (then cleared) by a `dump_on_drop` guard. The accumulator is a
/// shared Mutex so spans taken ON POOL WORKER THREADS inside run_rows
/// closures land in the caller's dump too. Inclusive: a child span's ms
/// are also inside its parent's. Off by default: one cached bool per
/// `span()` call site.
pub mod spans {
    use std::collections::BTreeMap;
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;

    type Acc = Mutex<BTreeMap<&'static str, (f64, u64)>>;

    #[inline]
    fn acc() -> &'static Acc {
        static ACC: OnceLock<Acc> = OnceLock::new();
        ACC.get_or_init(|| Mutex::new(BTreeMap::new()))
    }

    #[inline]
    pub fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("SAM3_PROF").is_ok_and(|v| v != "0"))
    }

    /// Records wall time from creation to drop under `label`.
    pub struct Span(Option<(&'static str, Instant)>);

    impl Drop for Span {
        #[inline]
        fn drop(&mut self) {
            if let Some((label, t0)) = self.0.take() {
                add(label, t0.elapsed());
            }
        }
    }

    /// Manually accumulate `d` under `label` (worker-thread friendly).
    #[inline]
    pub fn add(label: &'static str, d: std::time::Duration) {
        let mut m = match acc().lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        let e = m.entry(label).or_insert((0.0, 0));
        e.0 += d.as_secs_f64() * 1e3;
        e.1 += 1;
    }

    /// Start a span; a no-op when `SAM3_PROF` is off.
    #[inline]
    pub fn span(label: &'static str) -> Span {
        if on() {
            Span(Some((label, Instant::now())))
        } else {
            Span(None)
        }
    }

    /// Print the accumulated spans under `prefix`, then clear.
    pub fn dump(prefix: &str) {
        if !on() {
            return;
        }
        let mut m = match acc().lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        if m.is_empty() {
            return;
        }
        eprintln!("[sam3-prof] {prefix}:");
        let rows: Vec<(&'static str, f64, u64)> =
            m.iter().map(|(k, v)| (*k, v.0, v.1)).collect();
        for (k, ms, n) in rows {
            eprintln!("[sam3-prof]   {k:<18} {ms:>10.1} ms  ({n} calls)");
        }
        m.clear();
    }

    /// Prints (and clears) the accumulator when dropped — one dump per
    /// forward even through its early-return paths.
    pub struct Dump(String);

    impl Drop for Dump {
        fn drop(&mut self) {
            dump(&self.0);
        }
    }

    pub fn dump_on_drop(prefix: impl Into<String>) -> Dump {
        Dump(prefix.into())
    }
}

/// Print ms/token per slot to stderr.
pub fn report(label: &str, tokens: usize) {
    if !on() {
        return;
    }
    let t = tokens.max(1) as f64;
    eprintln!("cpu-prof [{label}] per token over {tokens} tokens:");
    for (name, ms, calls) in snapshot() {
        eprintln!(
            "  {name:<16} {:>8.3} ms/tok  ({calls} calls, {:.1} us/call)",
            ms / t,
            ms * 1e3 / calls.max(1) as f64
        );
    }
}
