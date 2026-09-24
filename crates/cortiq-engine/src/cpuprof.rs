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
