//! MiMo-V2 expert placement on a card that cannot hold every expert.
//!
//! MiMo-V2.6-Flash has 47 MoE layers of 256 experts (top-8, sigmoid +
//! selection-bias routing, no shared expert). At the default q4tp profile
//! one expert is 13.1 MB, a layer 3.36 GB and all experts 157.7 GB, so on
//! every card some experts live only in host RAM. Three placements exist:
//!
//! * **prefix** — whole MoE layers resident from the bottom of the stack up
//!   while the budget lasts (the generic token graph's device prefix, or the
//!   per-op residency arena when the graph declines), the remaining layers
//!   walk the host: attention AND all 8 experts of every tail layer stream
//!   from RAM. Cost per token ≈ one graph submit + (L−P)·(A + 8e)/B_cpu.
//! * **dynamic** — every MoE layer walks the host loop with its projections
//!   on the device (per-op) and its experts in ONE model-wide VRAM bank
//!   keyed (layer, expert) with an LRU ([`Bank`]: the segmented
//!   `dsv4_global_*` device buffers DeepSeek-V4.1 and Qwen3.8 already use,
//!   with its own slot map whose fills upload on a background thread while
//!   the missing token computes the expert on the host). The host
//!   routes exactly (`moe_ffn_route`), resident picks run in
//!   `dsv4_moe_frame` (forced ids, preweighted, no shared expert) while the
//!   cold picks run CONCURRENTLY on the CPU with the same q4tp kernels the
//!   host path uses (`moe_cold_experts_cpu`), and the two parts are summed.
//!   Cost per token ≈ L·(3 fences) + all bytes at device speed + the cold
//!   share 8·(1−h(s))·e/B_cpu per layer, h(s) = LRU hit rate at s slots.
//! * **hybrid** — a whole-layer prefix of P layers plus the bank for the
//!   rest. Only a device graph makes whole-layer residency cheaper than the
//!   bank (one submit for the prefix instead of per-layer fences); without
//!   one, pinning 256 slots of a layer serves fewer hits than giving the same
//!   slots to the LRU.
//!
//! Nothing here approximates: routing is the host's, every chosen expert is
//! computed (on the device or on the host) with its full weight, and the
//! result differs from the host path only in f32 summation order.
//!
//! The mode is chosen once per process from the VRAM budget and the measured
//! costs in [`Costs::measured`] (see [`place`]); `CMF_MIMO_MOE` =
//! `prefix|dynamic|hybrid|auto` overrides it. One `info` line states the
//! choice and why.

use crate::pipeline::{MoeFfn, MoeRoute};
use crate::pool::Pool;
#[cfg(feature = "gpu")]
use cortiq_core::{CmfModel, TensorDtype};
#[cfg(feature = "gpu")]
use std::sync::Arc;

/// Where the experts of the MoE layers execute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeMode {
    Prefix,
    Dynamic,
    Hybrid,
}

impl MoeMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Prefix => "prefix",
            Self::Dynamic => "dynamic",
            Self::Hybrid => "hybrid",
        }
    }

    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    fn parse(s: &str) -> Option<Option<Self>> {
        match s.trim().to_ascii_lowercase().as_str() {
            "prefix" => Some(Some(Self::Prefix)),
            "dynamic" => Some(Some(Self::Dynamic)),
            "hybrid" => Some(Some(Self::Hybrid)),
            "auto" | "" => Some(None),
            _ => None,
        }
    }
}

/// Everything the placement decision reads — plain data, so the policy is
/// testable without a device.
#[derive(Clone, Debug)]
pub struct PlacementInputs {
    /// Device weight budget in bytes (`CMF_GPU_VRAM_MB` or VRAM − reserve).
    pub budget: u64,
    /// Device bytes of every non-expert weight (attention, norms, routers,
    /// dense layers, lm_head) — they must stay resident in every mode.
    pub non_expert: u64,
    /// Bytes of one expert (gate + up + down).
    pub per_expert: u64,
    pub moe_layers: usize,
    pub n_experts: usize,
    pub top_k: usize,
    /// Mean non-expert device bytes of one layer (attention, router, norms).
    pub attn_per_layer: u64,
    /// A whole-token device graph can run whole resident layers of this
    /// model (then a whole-layer prefix costs one submit, not per-layer
    /// fences).
    pub graph_prefix: bool,
}

/// A placement: the mode, how many MoE layers (from the first MoE layer)
/// run on the whole-layer prefix path, and how many bank slots to allocate.
#[derive(Clone, Debug, PartialEq)]
pub struct Placement {
    pub mode: MoeMode,
    pub prefix_layers: usize,
    pub bank_slots: usize,
    /// Predicted seconds per token of the chosen mode (model only).
    pub predicted_s: f64,
    pub reason: String,
}

/// Measured cost constants of the decision.
#[derive(Clone, Debug)]
pub struct Costs {
    /// Effective device bytes/s for a resident weight stream.
    pub dev_bytes_per_s: f64,
    /// Effective host bytes/s for q4tp expert / q8_2f projection streams.
    pub cpu_bytes_per_s: f64,
    /// Fixed cost of one host-walked layer whose projections and resident
    /// experts run per-op on the device: the QKV and O submits, the MoE frame
    /// fence, the host attention core and route.
    pub dyn_layer_s: f64,
    /// Fixed cost of one whole-token graph submit.
    pub graph_submit_s: f64,
    /// Fixed cost of one pure-host layer (pool dispatch barriers).
    pub cpu_layer_s: f64,
    /// Upload bytes/s of a bank fill.
    pub fill_bytes_per_s: f64,
    /// Per-layer LRU hit rate of a decode stream by slots per layer (trace
    /// replay of real routing). Interpolated in log(slots).
    pub hit_curve: Vec<(f64, f64)>,
}

impl Costs {
    /// RTX PRO 6000 Blackwell (96 GB, Vulkan) + EPYC 9655 in a 24-core
    /// cgroup, MiMo-V2.6-Flash q4tp (runs of 2026-09-24):
    /// - `dyn_layer_s`: a bank-served layer measured 0.9–1.0 ms of host
    ///   walk per layer at ~100 % hits with the generic frame (QKV submit
    ///   0.22 ms, O 0.14 ms, frame 0.43 ms, attention core 0.06 ms); the
    ///   dedicated bank kernels take ~0.2 ms off the frame. The device's
    ///   own bytes are charged separately at `dev_bytes_per_s`.
    /// - `hit_curve`: per-layer LRU hit rate after a 64-token warm-up,
    ///   replayed from a 446-token CMF_MOE_TRACE of docs/ppl_nat.txt
    ///   (tools/moe_lru_sim.py).
    /// - `cpu_bytes_per_s`: q4tp/q8_2f streaming of the host walk with the
    ///   22-thread pool.
    pub fn measured() -> Self {
        Self {
            dev_bytes_per_s: 1.2e12,
            cpu_bytes_per_s: 60e9,
            dyn_layer_s: 0.6e-3,
            graph_submit_s: 0.3e-3,
            cpu_layer_s: 0.1e-3,
            fill_bytes_per_s: 20e9,
            hit_curve: vec![
                (8.0, 0.341),
                (16.0, 0.460),
                (32.0, 0.603),
                (64.0, 0.777),
                (96.0, 0.875),
                (128.0, 0.930),
                (160.0, 0.959),
                (192.0, 0.972),
                (224.0, 0.975),
                (256.0, 1.0),
            ],
        }
    }

    /// LRU hit rate at `slots` per layer.
    pub fn hit_rate(&self, slots: f64, n_experts: usize) -> f64 {
        if slots >= n_experts as f64 {
            return 1.0;
        }
        if slots <= 0.0 {
            return 0.0;
        }
        let c = &self.hit_curve;
        if c.is_empty() {
            return 0.0;
        }
        if slots <= c[0].0 {
            return c[0].1 * slots / c[0].0;
        }
        for w in c.windows(2) {
            let ((s0, h0), (s1, h1)) = (w[0], w[1]);
            if slots <= s1 {
                let t = (slots.ln() - s0.ln()) / (s1.ln() - s0.ln());
                return h0 + t * (h1 - h0);
            }
        }
        c[c.len() - 1].1
    }
}

/// Device reserve kept outside the bank: the allocator workspace
/// (budget/10 within 2–4 GiB, the same rule `dsv4_global_moe_create`
/// applies) plus 1 GiB for activations, KV mirrors and driver slack.
pub fn device_reserve(budget: u64) -> u64 {
    let gib = 1u64 << 30;
    (budget / 10).clamp(2 * gib, 4 * gib) + gib
}

/// Choose the placement. `forced` = the `CMF_MIMO_MOE` override.
pub fn place(inp: &PlacementInputs, costs: &Costs, forced: Option<MoeMode>) -> Placement {
    let gb = |b: u64| b as f64 / 1e9;
    let l = inp.moe_layers.max(1);
    let ne = inp.n_experts.max(1);
    let k = inp.top_k as f64;
    let e = inp.per_expert as f64;
    let a = inp.attn_per_layer as f64;
    let total_experts = l * ne;
    let room = inp
        .budget
        .saturating_sub(inp.non_expert)
        .saturating_sub(device_reserve(inp.budget));
    let slots = (room / inp.per_expert.max(1)) as usize;
    let whole = (slots / ne).min(l);
    // Per-token time of the placements (the dense layers and the head are
    // common to all of them and left out).
    let dev_layer = (a + k * e) / costs.dev_bytes_per_s;
    let prefix_cost = |p: usize| -> f64 {
        if costs.dev_bytes_per_s <= 0.0 {
            return f64::INFINITY;
        }
        let dev = if inp.graph_prefix {
            costs.graph_submit_s + p as f64 * dev_layer
        } else {
            p as f64 * (costs.dyn_layer_s + dev_layer)
        };
        dev + (l - p) as f64 * ((a + k * e) / costs.cpu_bytes_per_s + costs.cpu_layer_s)
    };
    let dyn_cost = |p: usize| -> (f64, f64) {
        // p whole layers, the rest share what is left of the bank.
        let rest = l - p;
        if rest == 0 {
            return (prefix_cost(p), 1.0);
        }
        let per = slots.saturating_sub(p * ne) as f64 / rest as f64;
        let h = costs.hit_rate(per, ne);
        let hot = k * h * e / costs.dev_bytes_per_s;
        let cold = k * (1.0 - h) * e / costs.cpu_bytes_per_s;
        let layer = costs.dyn_layer_s + a / costs.dev_bytes_per_s + hot.max(cold);
        let head = if p == 0 {
            0.0
        } else if inp.graph_prefix {
            costs.graph_submit_s + p as f64 * dev_layer
        } else {
            p as f64 * (costs.dyn_layer_s + dev_layer)
        };
        (head + rest as f64 * layer, h)
    };
    let describe = |mode: MoeMode, p: usize, t: f64, h: f64, why: &str| -> String {
        format!(
            "{why}; budget {:.1} GB, non-expert {:.1} GB, experts {:.1} GB ({} × {:.1} MB), \
             room {} slots = {:.0}/layer, whole layers {whole}/{l}{}; predicted {:.1} ms/token \
             ({}{})",
            gb(inp.budget),
            gb(inp.non_expert),
            gb(inp.per_expert * total_experts as u64),
            total_experts,
            e / 1e6,
            slots,
            slots as f64 / l as f64,
            if inp.graph_prefix {
                ", graph prefix"
            } else {
                ", no graph prefix"
            },
            t * 1e3,
            mode.name(),
            match mode {
                MoeMode::Prefix => format!(" P={p}"),
                MoeMode::Dynamic => format!(" hit≈{:.0}%", h * 100.0),
                MoeMode::Hybrid => format!(" P={p} hit≈{:.0}%", h * 100.0),
            },
        )
    };
    let mk = |mode: MoeMode, p: usize, bank: usize, t: f64, why: String| Placement {
        mode,
        prefix_layers: p,
        bank_slots: bank,
        predicted_s: t,
        reason: why,
    };
    if slots >= total_experts && forced.is_none() {
        let t = prefix_cost(l);
        let why = describe(MoeMode::Prefix, l, t, 1.0, "every expert fits");
        return mk(MoeMode::Prefix, l, 0, t, why);
    }
    let (t_prefix, t_dyn) = (prefix_cost(whole), dyn_cost(0));
    // Best hybrid split: at least one whole layer, at least one bank layer.
    let hybrid = (1..whole.min(l.saturating_sub(1)) + 1)
        .map(|p| (p, dyn_cost(p)))
        .min_by(|x, y| {
            x.1.0
                .partial_cmp(&y.1.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    let bank_for = |p: usize| slots.saturating_sub(p * ne);
    match forced {
        Some(MoeMode::Prefix) => {
            let why = describe(MoeMode::Prefix, whole, t_prefix, 0.0, "CMF_MIMO_MOE=prefix");
            mk(MoeMode::Prefix, whole, 0, t_prefix, why)
        }
        Some(MoeMode::Dynamic) => {
            let why = describe(
                MoeMode::Dynamic,
                0,
                t_dyn.0,
                t_dyn.1,
                "CMF_MIMO_MOE=dynamic",
            );
            mk(MoeMode::Dynamic, 0, bank_for(0), t_dyn.0, why)
        }
        Some(MoeMode::Hybrid) => {
            let (p, (t, h)) = hybrid.unwrap_or((0, t_dyn));
            let why = describe(MoeMode::Hybrid, p, t, h, "CMF_MIMO_MOE=hybrid");
            mk(MoeMode::Hybrid, p, bank_for(p), t, why)
        }
        None => {
            let mut best = (MoeMode::Prefix, whole, t_prefix, 0.0);
            if t_dyn.0 < best.2 {
                best = (MoeMode::Dynamic, 0, t_dyn.0, t_dyn.1);
            }
            if let Some((p, (t, h))) = hybrid
                && t < best.2
            {
                best = (MoeMode::Hybrid, p, t, h);
            }
            let (mode, p, t, h) = best;
            let alt = format!(
                "auto: prefix {:.1} ms, dynamic {:.1} ms{}",
                t_prefix * 1e3,
                t_dyn.0 * 1e3,
                hybrid.map_or(String::new(), |(p, (t, _))| format!(
                    ", best hybrid P={p} {:.1} ms",
                    t * 1e3
                )),
            );
            let bank = if mode == MoeMode::Prefix {
                0
            } else {
                bank_for(p)
            };
            let why = describe(mode, p, t, h, &alt);
            mk(mode, p, bank, t, why)
        }
    }
}

/// Per-process counters of the dynamic executor.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    /// Completed attention-only graph calls / rows in the dynamic tail.
    pub attn_graph_calls: u64,
    pub attn_graph_rows: u64,
    pub attn_graph_ns: u64,
    /// Layer calls served by the bank frame.
    pub calls: u64,
    /// Expert picks seen by those calls.
    pub picks: u64,
    /// Picks already resident before the call.
    pub hits: u64,
    /// Picks uploaded into the bank by the call (then run on the device).
    pub fills: u64,
    /// Picks computed on the host.
    pub cold: u64,
    /// Wall time inside the device frame (submit + wait), ns.
    pub frame_ns: u64,
    /// Wall time of the whole layer call, ns.
    pub call_ns: u64,
    /// Calls that fell back to the host path.
    pub fallbacks: u64,
    /// Wall time of the host route (router matvec + top-k) of bank calls, ns.
    pub route_ns: u64,
}

static STATS: std::sync::Mutex<Stats> = std::sync::Mutex::new(Stats {
    attn_graph_calls: 0,
    attn_graph_rows: 0,
    attn_graph_ns: 0,
    calls: 0,
    picks: 0,
    hits: 0,
    fills: 0,
    cold: 0,
    frame_ns: 0,
    call_ns: 0,
    fallbacks: 0,
    route_ns: 0,
});

pub(crate) fn note_attention_graph(rows: usize, ns: u64) {
    let mut s = STATS.lock().unwrap();
    s.attn_graph_calls += 1;
    s.attn_graph_rows += rows as u64;
    s.attn_graph_ns += ns;
}

/// Process-wide executor counters (benches read deltas).
pub fn stats() -> Stats {
    *STATS.lock().unwrap()
}

static LAST_DECISION: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

/// The placement line of the most recent decision (also logged at `info`).
pub fn last_decision() -> String {
    LAST_DECISION.lock().unwrap().clone()
}

/// The model-wide expert bank: which `(layer, expert)` occupies which slot
/// of the segmented device buffers (`dsv4_global_*`), and the policy that
/// admits and evicts.
///
/// Fills never stall a token. An admitted expert is uploaded by a filler
/// thread while the token that missed it computes it on the host; the slot
/// becomes visible to the device only once its bytes are queued, so no
/// frame can read a half-written slot (and a frame already queued finishes
/// on the evicted expert's bytes: queue writes land between submits).
///
/// Admission: while free slots remain every miss is admitted; once the bank
/// is full a miss must recur (`min_seen` sightings within the decay window)
/// and evicts the least recently used slot of a layer holding more than its
/// floor, never a slot the current token already used.
///
/// The threshold follows the bank size ([`default_min_seen`]): a fill is
/// ~13 MB of PCIe traffic queued ahead of the next frame, a cold pick is
/// the same bytes streamed by the host in parallel with the frame, so a
/// fill pays only for an expert that stays long enough to be reused.
#[cfg(feature = "gpu")]
pub(crate) struct Bank {
    pub(crate) segment_slots: usize,
    n_experts: usize,
    /// Per `(layer, expert)` key: its slot, `NONE`, or `PENDING` (a fill in
    /// flight — cold until it lands).
    slot_for: Vec<u32>,
    /// Per slot: its key, or `NONE`.
    owner: Vec<u32>,
    /// Per slot: the token that last used it.
    last: Vec<u64>,
    occupancy: Vec<u32>,
    free: Vec<u32>,
    floor: u32,
    /// Decayed sightings per key and the token they were last decayed at.
    seen: Vec<u16>,
    seen_tok: Vec<u32>,
    tok: u64,
    pending: usize,
    /// Fills queued so far (the profile's "fills").
    admitted: u64,
    max_pending: usize,
    /// Queue bound while priming from a prompt (fills are cheap to queue;
    /// the filler drains them during the rest of the prefill).
    prime_queue: usize,
    min_seen: u16,
    decay_tokens: u64,
    tx: Option<std::sync::mpsc::Sender<(u32, (usize, usize, usize))>>,
    done: Arc<std::sync::Mutex<Vec<(u32, bool)>>>,
    filler: Option<std::thread::JoinHandle<()>>,
}

/// Recurrences a miss needs before a full bank admits it. Replaying the
/// 446-token docs/ppl_nat.txt routing trace through this policy (after a
/// 64-token warm-up): at 142 slots/layer `1` → 96.6 % hits with 12.8
/// fills/token, `2` → 95.9 % with 6.4; at 64 slots `2` → 79.7 % / 17.1
/// against `1` → 77.9 % / 82.9; at 13 slots `3` → 41.8 % / 66.5 against
/// `2` → 39.7 % / 137. Small banks churn: demand more evidence there.
pub fn default_min_seen(slots_per_layer: usize) -> u64 {
    if slots_per_layer >= 48 { 2 } else { 3 }
}

#[cfg(feature = "gpu")]
const NONE: u32 = u32::MAX;
#[cfg(feature = "gpu")]
const PENDING: u32 = u32::MAX - 1;

#[cfg(feature = "gpu")]
impl Bank {
    /// Allocate `slots` device slots (rounded to whole segments) for
    /// `n_layers × n_experts` keys.
    fn create(
        model: &Arc<CmfModel>,
        inter: usize,
        hidden: usize,
        n_layers: usize,
        moe_layers: usize,
        n_experts: usize,
        slots: usize,
    ) -> Option<Self> {
        let keys = n_layers.checked_mul(n_experts)?;
        if keys >= PENDING as usize {
            return None;
        }
        let (capacity, segment_slots) =
            crate::gpu_wgpu::dsv4_global_moe_create_slots(model, slots, inter, hidden, false)?;
        let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<u64>().ok());
        let done = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (tx, rx) = std::sync::mpsc::channel::<(u32, (usize, usize, usize))>();
        let filler = {
            let (model, done) = (model.clone(), done.clone());
            let dev = crate::gpu::current_device();
            std::thread::Builder::new()
                .name("mimo-bank-fill".into())
                .spawn(move || {
                    crate::gpu::set_current_device(dev);
                    while let Ok((slot, triple)) = rx.recv() {
                        let ok =
                            crate::gpu_wgpu::dsv4_global_slot_fill(&model, slot as usize, triple);
                        done.lock().unwrap().push((slot, ok));
                    }
                })
                .ok()?
        };
        Some(Self {
            segment_slots,
            n_experts,
            slot_for: vec![NONE; keys],
            owner: vec![NONE; capacity],
            last: vec![0; capacity],
            occupancy: vec![0; n_layers],
            free: (0..capacity as u32).rev().collect(),
            floor: (capacity / moe_layers.max(1) / 2) as u32,
            seen: vec![0; keys],
            seen_tok: vec![0; keys],
            tok: 1,
            pending: 0,
            admitted: 0,
            max_pending: env("CMF_MIMO_FILL_QUEUE").unwrap_or(256) as usize,
            prime_queue: env("CMF_MIMO_PRIME_QUEUE").unwrap_or(4096) as usize,
            min_seen: env("CMF_MIMO_FETCH_MIN_SEEN")
                .unwrap_or(default_min_seen(capacity / moe_layers.max(1)))
                as u16,
            decay_tokens: env("CMF_MIMO_SEEN_DECAY").unwrap_or(16).max(1),
            tx: Some(tx),
            done,
            filler: Some(filler),
        })
    }

    pub(crate) fn capacity(&self) -> usize {
        self.owner.len()
    }

    pub(crate) fn free_slots(&self) -> usize {
        self.free.len()
    }

    pub(crate) fn pending(&self) -> usize {
        self.pending
    }

    /// A new token starts (called at its first bank layer).
    fn next_token(&mut self) {
        self.tok += 1;
    }

    /// Fold the filler's completions into the slot map.
    fn drain(&mut self) {
        let done = std::mem::take(&mut *self.done.lock().unwrap());
        for (slot, ok) in done {
            self.pending = self.pending.saturating_sub(1);
            let key = self.owner[slot as usize];
            if key == NONE {
                continue;
            }
            if ok {
                self.slot_for[key as usize] = slot;
                self.last[slot as usize] = self.tok;
            } else {
                self.slot_for[key as usize] = NONE;
                self.owner[slot as usize] = NONE;
                let layer = key as usize / self.n_experts;
                self.occupancy[layer] = self.occupancy[layer].saturating_sub(1);
                self.free.push(slot);
            }
        }
    }

    fn victim(&mut self, layer: usize) -> Option<u32> {
        if let Some(s) = self.free.pop() {
            return Some(s);
        }
        let ne = self.n_experts as u32;
        let tok = self.tok;
        let pick = |over_floor: bool, me: &Self| -> Option<u32> {
            let mut best: Option<(u64, u32)> = None;
            for (slot, &key) in me.owner.iter().enumerate() {
                if key == NONE || me.slot_for[key as usize] != slot as u32 {
                    continue; // empty or still filling
                }
                let l = me.last[slot];
                if l >= tok {
                    continue; // this token used it
                }
                let kl = (key / ne) as usize;
                if over_floor && me.occupancy[kl] <= me.floor && kl != layer {
                    continue;
                }
                if best.is_none_or(|(bl, _)| l < bl) {
                    best = Some((l, slot as u32));
                }
            }
            best.map(|(_, s)| s)
        };
        pick(true, self).or_else(|| pick(false, self))
    }

    /// Resolve `picks` of `layer`: the remap (slot or `u32::MAX` per expert)
    /// for this call, and admissions of recurring misses for later tokens.
    fn resolve(
        &mut self,
        layer: usize,
        picks: &[usize],
        triples: &[(usize, usize, usize)],
    ) -> Option<Vec<u32>> {
        self.drain();
        let base = layer.checked_mul(self.n_experts)?;
        let mut remap = vec![NONE; self.n_experts];
        for &e in picks {
            let key = base + *(e < self.n_experts).then_some(&e)?;
            self.see(key, 1);
            let slot = self.slot_for[key];
            if slot < PENDING {
                remap[e] = slot;
                self.last[slot as usize] = self.tok;
            }
        }
        for &e in picks {
            if !self.admit(layer, e, triples[e], self.max_pending)? {
                break;
            }
        }
        Some(remap)
    }

    /// Count `n` sightings of `key`, decaying older ones by half per
    /// `decay_tokens` tokens.
    fn see(&mut self, key: usize, n: u16) {
        let tok32 = (self.tok / self.decay_tokens).min(u32::MAX as u64) as u32;
        let shift = tok32.saturating_sub(self.seen_tok[key]);
        self.seen[key] = if shift >= 16 {
            0
        } else {
            self.seen[key] >> shift
        };
        self.seen_tok[key] = tok32;
        self.seen[key] = self.seen[key].saturating_add(n);
    }

    /// Queue `(layer, e)` for a fill if the policy admits it. `Some(false)`
    /// = no victim left (stop admitting this call), `None` = the filler is
    /// gone.
    fn admit(
        &mut self,
        layer: usize,
        e: usize,
        triple: (usize, usize, usize),
        queue_cap: usize,
    ) -> Option<bool> {
        let key = layer * self.n_experts + e;
        if self.slot_for[key] != NONE || self.pending >= queue_cap {
            return Some(true);
        }
        if self.free.is_empty() && self.seen[key] < self.min_seen {
            return Some(true);
        }
        let Some(slot) = self.victim(layer) else {
            return Some(false);
        };
        let old = self.owner[slot as usize];
        if old != NONE {
            self.slot_for[old as usize] = NONE;
            let ol = old as usize / self.n_experts;
            self.occupancy[ol] = self.occupancy[ol].saturating_sub(1);
        }
        self.owner[slot as usize] = key as u32;
        self.slot_for[key] = PENDING;
        self.occupancy[layer] += 1;
        self.pending += 1;
        self.admitted += 1;
        self.tx
            .as_ref()
            .is_some_and(|tx| tx.send((slot, triple)).is_ok())
            .then_some(true)
    }

    /// A prompt's expert usage for `layer` (the host-computed prefill):
    /// count it as sightings and queue the most used experts, so decode
    /// starts on a bank the prompt already warmed. The fills run in the
    /// background; nothing waits on them.
    fn prime(&mut self, layer: usize, counts: &[u64], triples: &[(usize, usize, usize)]) {
        self.drain();
        let base = layer * self.n_experts;
        let mut used: Vec<(usize, u64)> = counts
            .iter()
            .enumerate()
            .filter(|&(e, &c)| c > 0 && e < self.n_experts && e < triples.len())
            .map(|(e, &c)| (e, c))
            .collect();
        used.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        for &(e, c) in &used {
            self.see(base + e, c.min(u16::MAX as u64) as u16);
        }
        for &(e, _) in &used {
            match self.admit(layer, e, triples[e], self.prime_queue) {
                Some(true) => {}
                _ => break,
            }
        }
    }
}

#[cfg(feature = "gpu")]
impl Drop for Bank {
    fn drop(&mut self) {
        self.tx = None;
        if let Some(h) = self.filler.take() {
            let _ = h.join();
        }
    }
}

/// One bank per model file: pipelines sharing a model share its device
/// slots, so they must share the slot map too.
#[cfg(feature = "gpu")]
static BANKS: std::sync::Mutex<Vec<(u64, Arc<std::sync::Mutex<Bank>>)>> =
    std::sync::Mutex::new(Vec::new());

/// The per-pipeline state: undecided until the first forward, then off or
/// on with its bank.
#[derive(Default)]
pub enum Slot {
    #[default]
    Undecided,
    Off,
    #[cfg(feature = "gpu")]
    On(Box<Dynamic>),
}

/// A decided placement with its bank.
#[cfg(feature = "gpu")]
pub struct Dynamic {
    pub placement: Placement,
    model: Arc<CmfModel>,
    bank: Arc<std::sync::Mutex<Bank>>,
    /// `(gate, up, down)` directory indices of every expert, by absolute
    /// layer; empty on layers without experts.
    ids: Vec<Vec<(usize, usize, usize)>>,
    /// Absolute index of the first layer the bank serves.
    pub dyn_from: usize,
    failed: bool,
    /// The dedicated bank kernels serve this model (decided on first use).
    fast: Option<bool>,
    /// Last layer with experts (`CMF_MIMO_PROF` prints here). A bank
    /// epoch starts at `dyn_from`, not the first graph-resident MoE layer.
    last_moe: usize,
    prof: bool,
    prof_mark: (std::time::Instant, Stats, [u64; 7]),
}

/// Device-side counters the profile line reports as deltas: frame encode,
/// frame wait, frame uploads, frame passes (ns), queue submits, and the
/// card's own time inside the frames with `CMF_GPU_TS=1` (ns, frames).
#[cfg(feature = "gpu")]
fn device_counters() -> [u64; 7] {
    use std::sync::atomic::Ordering::Relaxed;
    [
        crate::gpu_wgpu::MOE_ENC_NS.load(Relaxed),
        crate::gpu_wgpu::MOE_WAIT_NS.load(Relaxed),
        crate::gpu_wgpu::MOE_UP_NS.load(Relaxed),
        crate::gpu_wgpu::MOE_PASS_NS.load(Relaxed),
        crate::gpu_wgpu::SUBMITS.load(Relaxed),
        crate::gpu_wgpu::MOE_GPU_NS[0].load(Relaxed),
        crate::gpu_wgpu::MOE_GPU_N.load(Relaxed),
    ]
}

#[cfg(feature = "gpu")]
fn env_mode() -> Option<MoeMode> {
    let raw = std::env::var("CMF_MIMO_MOE").ok()?;
    match MoeMode::parse(&raw) {
        Some(m) => m,
        None => {
            tracing::warn!("CMF_MIMO_MOE={raw}: not prefix|dynamic|hybrid|auto — using auto");
            None
        }
    }
}

/// Why a layer stack cannot use the bank, if it cannot.
#[cfg(feature = "gpu")]
fn bank_refusal(layers: &[(usize, &MoeFfn)]) -> Option<String> {
    let first = layers.first()?.1;
    let model = first
        .experts
        .first()
        .and_then(|e| e.gate_proj.model_arc())?;
    let inter = first.experts[0].gate_proj.rows();
    let hidden = first.experts[0].gate_proj.cols();
    for &(li, m) in layers {
        if m.shared.is_some()
            || m.per_expert_scale.is_some()
            || m.resonance.is_some()
            || m.route_tau.is_some()
            || m.mask.is_some()
            || m.router_input_norm
        {
            return Some(format!(
                "layer {li}: routing extras the bank frame does not carry"
            ));
        }
        for (ei, d) in m.experts.iter().enumerate() {
            let q4tp = |t: &crate::qtensor::QTensor| {
                t.model_dtype() == Some(TensorDtype::Q4TiledP)
                    && t.model_arc().is_some_and(|a| a.uid() == model.uid())
            };
            if !(q4tp(&d.gate_proj) && q4tp(&d.up_proj) && q4tp(&d.down_proj))
                || d.act != crate::pipeline::Act::Silu
                || d.gate_proj.rows() != inter
                || d.gate_proj.cols() != hidden
                || d.down_proj.rows() != hidden
            {
                return Some(format!(
                    "layer {li} expert {ei}: not a mapped q4tp SiLU expert of the common shape"
                ));
            }
        }
    }
    if hidden % 32 != 0 || inter % 32 != 0 {
        return Some(format!(
            "hidden {hidden} / inter {inter} not multiples of 32"
        ));
    }
    None
}

impl Slot {
    pub fn is_undecided(&self) -> bool {
        matches!(self, Self::Undecided)
    }

    /// Whether the bank is active.
    pub fn is_on(&self) -> bool {
        match self {
            #[cfg(feature = "gpu")]
            Self::On(d) => !d.failed,
            _ => false,
        }
    }

    /// First bank-owned layer. Graph builders must stop before it even
    /// when the generic capacity heuristic would admit more layers.
    pub(crate) fn graph_prefix_end(&self) -> Option<usize> {
        match self {
            #[cfg(feature = "gpu")]
            Self::On(d) => Some(d.dyn_from),
            _ => None,
        }
    }

    /// Does layer `li` run its experts through the bank? `host_tail` = the
    /// walk reached this layer after a device graph prefix handed it over:
    /// with a bank present every such MoE layer takes the bank (a whole-layer
    /// per-op path would stream experts through the residency arena).
    pub fn is_dynamic(&self, li: usize, host_tail: bool) -> bool {
        match self {
            #[cfg(feature = "gpu")]
            Self::On(d) => {
                !d.failed
                    && d.ids.get(li).is_some_and(|v| !v.is_empty())
                    && (li >= d.dyn_from || host_tail)
            }
            _ => {
                let _ = (li, host_tail);
                false
            }
        }
    }

    /// Decide the placement for a MiMo-V2 stack (`layers` = its MoE layers,
    /// by absolute index). Any other model is `Off`.
    #[cfg(not(feature = "gpu"))]
    pub fn decide(layers: &[(usize, &MoeFfn)], n_layers: usize, graph_prefix: bool) -> Self {
        let _ = (layers, n_layers, graph_prefix);
        Self::Off
    }

    /// Decide the placement for a MiMo-V2 stack (`layers` = its MoE layers,
    /// by absolute index). Any other model is `Off`.
    #[cfg(feature = "gpu")]
    pub fn decide(layers: &[(usize, &MoeFfn)], n_layers: usize, graph_prefix: bool) -> Self {
        let exact = std::env::var("CMF_MIMO_EXPERT_SLOTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        Self::decide_with(layers, n_layers, graph_prefix, env_mode(), exact)
    }

    /// `decide` with the operator knobs passed in: `forced` =
    /// `CMF_MIMO_MOE`, `exact` = `CMF_MIMO_EXPERT_SLOTS` (a bank of exactly
    /// that many slots, for tests and A/B runs).
    #[cfg(feature = "gpu")]
    pub fn decide_with(
        layers: &[(usize, &MoeFfn)],
        n_layers: usize,
        graph_prefix: bool,
        forced: Option<MoeMode>,
        exact: Option<usize>,
    ) -> Self {
        let Some(model) = layers
            .first()
            .and_then(|(_, m)| m.experts.first())
            .and_then(|e| e.gate_proj.model_arc())
        else {
            return Self::Off;
        };
        if model.arch().arch_name != "mimo_v2" {
            return Self::Off;
        }
        let say = |msg: &str| {
            tracing::info!("MiMo MoE placement: {msg}");
            *LAST_DECISION.lock().unwrap() = msg.to_string();
        };
        if !crate::gpu::enabled() || !crate::gpu::wgpu_active() {
            say("prefix — no wgpu device (experts on the host path)");
            return Self::Off;
        }
        if let Some(why) = bank_refusal(layers) {
            say(&format!("prefix — {why}"));
            return Self::Off;
        }
        let first = layers[0].1;
        let inter = first.experts[0].gate_proj.rows();
        let hidden = first.experts[0].gate_proj.cols();
        let n_experts = first.experts.len();
        if layers.iter().any(|(_, m)| m.experts.len() != n_experts) {
            say("prefix — MoE layers differ in expert count");
            return Self::Off;
        }
        let Some(budget) = crate::gpu_wgpu::dsv4_vram_budget() else {
            say("prefix — no device budget");
            return Self::Off;
        };
        if budget == u64::MAX {
            say("prefix — unified memory: the host pages experts, nothing to place");
            return Self::Off;
        }
        if !crate::gpu_wgpu::dsv4_global_moe_supported() {
            say("prefix — this adapter has no segmented expert bank (descriptor arrays)");
            return Self::Off;
        }
        let per_expert = {
            let gu = cortiq_core::quant::expected_nbytes(TensorDtype::Q4TiledP, &[inter, hidden]);
            let dn = cortiq_core::quant::expected_nbytes(TensorDtype::Q4TiledP, &[hidden, inter]);
            match (gu, dn) {
                (Some(gu), Some(dn)) => (2 * gu + dn) as u64,
                _ => {
                    say("prefix — expert size unknown");
                    return Self::Off;
                }
            }
        };
        let is_expert = |name: &str| name.contains(".mlp.experts.");
        let non_expert: u64 = model
            .tensors
            .iter()
            .filter(|t| !is_expert(&t.name) && !t.name.starts_with("model.embed_tokens."))
            .map(|t| t.nbytes)
            .sum();
        let layer_non_expert: u64 = model
            .tensors
            .iter()
            .filter(|t| t.name.starts_with("model.layers.") && !is_expert(&t.name))
            .map(|t| t.nbytes)
            .sum();
        let inp = PlacementInputs {
            budget,
            non_expert,
            per_expert,
            moe_layers: layers.len(),
            n_experts,
            top_k: first.top_k,
            attn_per_layer: layer_non_expert / n_layers.max(1) as u64,
            graph_prefix,
        };
        let costs = Costs::measured();
        let mut placement = place(&inp, &costs, forced);
        if placement.mode == MoeMode::Prefix {
            say(&placement.reason);
            return Self::Off;
        }
        if let Some(n) = exact {
            placement.bank_slots = n;
            placement.reason = format!("{} [CMF_MIMO_EXPERT_SLOTS={n}]", placement.reason);
        }
        let bank = {
            let mut reg = BANKS.lock().unwrap();
            match reg.iter().find(|(uid, _)| *uid == model.uid()) {
                Some((_, b)) => Some(b.clone()),
                None => Bank::create(
                    &model,
                    inter,
                    hidden,
                    n_layers,
                    layers.len().saturating_sub(placement.prefix_layers),
                    n_experts,
                    placement.bank_slots,
                )
                .map(|b| {
                    let b = Arc::new(std::sync::Mutex::new(b));
                    reg.push((model.uid(), b.clone()));
                    b
                }),
            }
        };
        let Some(bank) = bank else {
            say(&format!(
                "prefix — the {}-slot expert bank could not be allocated ({})",
                placement.bank_slots, placement.reason
            ));
            return Self::Off;
        };
        let mut ids = vec![Vec::new(); n_layers];
        for &(li, m) in layers {
            let triples: Option<Vec<_>> = m
                .experts
                .iter()
                .map(|d| {
                    Some((
                        d.gate_proj.model_idx()?,
                        d.up_proj.model_idx()?,
                        d.down_proj.model_idx()?,
                    ))
                })
                .collect();
            match (triples, ids.get_mut(li)) {
                (Some(t), Some(slot)) => *slot = t,
                _ => {
                    say("prefix — an expert is not mmap-backed");
                    return Self::Off;
                }
            }
        }
        // The whole-layer prefix counts MoE layers from the first one.
        let dyn_from = layers
            .get(placement.prefix_layers)
            .map_or(n_layers, |&(li, _)| li);
        let cap = bank.lock().unwrap().capacity();
        say(&format!(
            "{} — {}; bank {cap} slots ({:.1} GB), bank layers from {dyn_from}",
            placement.mode.name(),
            placement.reason,
            cap as f64 * per_expert as f64 / 1e9,
        ));
        Self::On(Box::new(Dynamic {
            placement,
            model,
            bank,
            ids,
            dyn_from,
            failed: false,
            fast: None,
            last_moe: layers.last().map_or(0, |&(li, _)| li),
            prof: std::env::var_os("CMF_MIMO_PROF").is_some(),
            prof_mark: (std::time::Instant::now(), stats(), device_counters()),
        }))
    }

    /// After a host-computed prefill of bank layer `li`: `before` is the
    /// layer's selection counters (`MoeFfn::stats`) from before the chunk;
    /// the difference is the prompt's expert usage, which primes the bank.
    pub(crate) fn prime(&mut self, li: usize, m: &MoeFfn, before: &[u64]) {
        #[cfg(feature = "gpu")]
        if let Self::On(d) = self
            && !d.failed
            && let Some(triples) = d.ids.get(li).filter(|t| !t.is_empty())
            && std::env::var("CMF_MIMO_PRIME").as_deref() != Ok("0")
        {
            let now = m.stats.borrow();
            let counts: Vec<u64> = (0..now.len())
                .map(|e| now[e].saturating_sub(before.get(e).copied().unwrap_or(0)))
                .collect();
            drop(now);
            d.bank.lock().unwrap().prime(li, &counts, triples);
        }
        #[cfg(not(feature = "gpu"))]
        let _ = (li, m, before);
    }

    /// Charge a bank call's host route time (the caller routes).
    pub(crate) fn note_route(&self, ns: u64) {
        if self.is_on() {
            STATS.lock().unwrap().route_ns += ns;
        }
    }

    /// Run a routed MoE layer through the bank. `None` = not served (the
    /// caller runs the host path with the SAME route).
    pub(crate) fn forward(
        &mut self,
        li: usize,
        m: &MoeFfn,
        x: &[f32],
        route: &MoeRoute,
        pool: Option<&Pool>,
    ) -> Option<Vec<f32>> {
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (li, m, x, route, pool);
            None
        }
        #[cfg(feature = "gpu")]
        self.forward_bank(li, m, x, route, pool)
    }

    /// Verify a short block against one stable snapshot of the bank. The
    /// union route is pinned before any admission, so filling a later row
    /// cannot overwrite a slot used by an earlier row in the same submit.
    pub(crate) fn forward_rows(
        &mut self,
        li: usize,
        m: &MoeFfn,
        xs: &[f32],
        routes: &[MoeRoute],
        pool: Option<&Pool>,
    ) -> Option<Vec<f32>> {
        #[cfg(feature = "gpu")]
        {
            let Self::On(d) = self else { return None };
            if d.failed || d.fast == Some(false) {
                return None;
            }
            let t0 = std::time::Instant::now();
            let out = d.run_rows(li, m, xs, routes, pool);
            STATS.lock().unwrap().call_ns += t0.elapsed().as_nanos() as u64;
            out
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (li, m, xs, routes, pool);
            None
        }
    }

    #[cfg(feature = "gpu")]
    fn forward_bank(
        &mut self,
        li: usize,
        m: &MoeFfn,
        x: &[f32],
        route: &MoeRoute,
        pool: Option<&Pool>,
    ) -> Option<Vec<f32>> {
        let Self::On(d) = self else { return None };
        if d.failed {
            return None;
        }
        let t0 = std::time::Instant::now();
        let out = d.run(li, m, x, route, pool);
        {
            let mut st = STATS.lock().unwrap();
            st.call_ns += t0.elapsed().as_nanos() as u64;
            if out.is_none() {
                st.fallbacks += 1;
            }
        }
        if d.prof && li == d.last_moe {
            let (t, s0, c0) = d.prof_mark;
            let s1 = stats();
            let c1 = device_counters();
            let ms = |i: usize| (c1[i] - c0[i]) as f64 / 1e6;
            let picks = (s1.picks - s0.picks).max(1);
            eprintln!(
                "mimo-moe token: {:.1} ms wall, bank calls {} | hits {:.1}% fills {} cold {} of {} \
                 picks | frame {:.2} ms (encode {:.2} wait {:.2} upload {:.2} pass {:.2}), bank \
                 calls {:.2} ms, route {:.2} ms | submits {} | card {:.2} ms over {} frames | bank \
                 free {} of {}",
                t.elapsed().as_secs_f64() * 1e3,
                s1.calls - s0.calls,
                (s1.hits - s0.hits) as f64 / picks as f64 * 100.0,
                s1.fills - s0.fills,
                s1.cold - s0.cold,
                picks,
                (s1.frame_ns - s0.frame_ns) as f64 / 1e6,
                ms(0),
                ms(1),
                ms(2),
                ms(3),
                (s1.call_ns - s0.call_ns) as f64 / 1e6,
                (s1.route_ns - s0.route_ns) as f64 / 1e6,
                c1[4] - c0[4],
                ms(5),
                c1[6] - c0[6],
                {
                    let b = d.bank.lock().unwrap();
                    format!("{} (filling {})", b.free_slots(), b.pending())
                },
                d.bank.lock().unwrap().capacity(),
            );
            d.prof_mark = (std::time::Instant::now(), s1, c1);
        }
        out
    }
}

#[cfg(feature = "gpu")]
impl Dynamic {
    fn run_rows(
        &mut self,
        li: usize,
        m: &MoeFfn,
        xs: &[f32],
        routes: &[MoeRoute],
        pool: Option<&Pool>,
    ) -> Option<Vec<f32>> {
        let rows = routes.len();
        let top_k = routes.first()?.idx.len();
        let hidden = m.experts.first()?.gate_proj.cols();
        let inter = m.experts[0].gate_proj.rows();
        if rows > 4
            || top_k == 0
            || xs.len() != rows * hidden
            || routes
                .iter()
                .any(|r| r.idx.len() != top_k || r.logits.len() != m.experts.len())
            || std::env::var("CMF_MIMO_BANK_KERNEL").as_deref() == Ok("generic")
        {
            return None;
        }
        let triples = self.ids.get(li).filter(|t| t.len() == m.experts.len())?;
        let mut union = Vec::new();
        for r in routes {
            for &e in &r.idx {
                if e >= triples.len() {
                    return None;
                }
                if !union.contains(&e) {
                    union.push(e);
                }
            }
        }
        let mut bank = self.bank.lock().unwrap();
        if li == self.dyn_from {
            bank.next_token();
        }
        let admitted0 = bank.admitted;
        let remap = bank.resolve(li, &union, triples)?;
        let mut sel = Vec::with_capacity(rows * top_k);
        let mut wt = Vec::with_capacity(rows * top_k);
        let mut cold_jobs = Vec::with_capacity(rows);
        for r in routes {
            let mut jobs = Vec::new();
            for &e in &r.idx {
                let w = r.p[e] / r.wsum;
                sel.push(remap[e]);
                wt.push(w);
                if remap[e] == u32::MAX {
                    jobs.push((&m.experts[e], w));
                }
            }
            cold_jobs.push(jobs);
        }
        let cold = sel.iter().filter(|&&s| s == u32::MAX).count();
        let host_rows = || {
            crate::gpu::cpu_scope(|| {
                crate::qtensor::float_activations_scope(|| {
                    crate::pipeline::moe_cold_experts_rows_cpu(&cold_jobs, xs, hidden, pool)
                })
            })
        };
        let t_frame = std::time::Instant::now();
        let mut out = vec![0.0; xs.len()];
        if cold == sel.len() {
            out = host_rows();
        } else {
            let (ok, host) = std::thread::scope(|scope| {
                let host = (cold > 0).then(|| scope.spawn(host_rows));
                let ok = crate::gpu_wgpu::mimo_bank::mimo_bank_rows(
                    &self.model,
                    xs,
                    &sel,
                    &wt,
                    inter,
                    rows,
                    &mut out,
                );
                (
                    ok,
                    host.map(|h| h.join().expect("MiMo cold-expert worker panicked")),
                )
            });
            if !ok {
                return None;
            }
            if let Some(host) = host {
                for (o, h) in out.iter_mut().zip(host) {
                    *o += h;
                }
            }
        }
        let mut st = STATS.lock().unwrap();
        st.calls += 1;
        st.picks += sel.len() as u64;
        st.hits += (sel.len() - cold) as u64;
        st.cold += cold as u64;
        st.fills += bank.admitted - admitted0;
        st.frame_ns += t_frame.elapsed().as_nanos() as u64;
        Some(out)
    }

    fn run(
        &mut self,
        li: usize,
        m: &MoeFfn,
        x: &[f32],
        route: &MoeRoute,
        pool: Option<&Pool>,
    ) -> Option<Vec<f32>> {
        let triples = self.ids.get(li).filter(|t| t.len() == m.experts.len())?;
        let picks = &route.idx;
        if picks.is_empty() || route.logits.len() != m.experts.len() {
            return None;
        }
        let hidden = x.len();
        // The host route's final weights, exactly as the host path applies
        // them (`moe_ffn_cpu`: p[e] / wsum).
        let mut mix = vec![0.0f32; m.experts.len()];
        for &e in picks {
            mix[e] = route.p[e] / route.wsum;
        }
        // The slot map stays locked for the whole call: another pipeline on
        // the same model must not evict a slot between this resolve and the
        // frame that reads it.
        let bank_arc = self.bank.clone();
        let mut bank = bank_arc.lock().unwrap();
        if li == self.dyn_from {
            bank.next_token();
        }
        let admitted0 = bank.admitted;
        let Some(remap) = bank.resolve(li, picks, triples) else {
            tracing::warn!("MiMo MoE bank: slot map refused layer {li} — host path from now on");
            self.failed = true;
            return None;
        };
        let cold_ids: Vec<usize> = picks
            .iter()
            .copied()
            .filter(|&e| remap[e] == u32::MAX)
            .collect();
        let cold_jobs: Vec<(&crate::pipeline::DenseFfn, f32)> =
            cold_ids.iter().map(|&e| (&m.experts[e], mix[e])).collect();
        let weights = crate::gpu_wgpu::Dsv4MoeW {
            router: &[],
            experts: triples,
            logits: &route.logits,
            // Forced + preweighted: this is the final route table, the
            // shader does no scoring or normalization of its own.
            bias: Some(&mix),
            mask: None,
            forced: Some(picks),
            remap: Some(&remap),
            global: Some(crate::gpu_wgpu::Dsv4GlobalMoe {
                pool_uid: self.model.uid(),
                shared_slot: 0,
                segment_slots: bank.segment_slots as u32,
            }),
            has_shared: false,
            shared_weight: 1.0,
            preweighted: true,
            qwen_softmax: false,
        };
        let geom = crate::gpu_wgpu::Dsv4MoeGeom {
            hidden,
            inter: m.experts[0].gate_proj.rows(),
            top_k: picks.len(),
            route_scale: 1.0,
            swiglu_limit: 0.0,
            gu_q2: false,
            bf16: false,
        };
        let mut out = vec![0.0f32; hidden];
        let mut cold_seen = Vec::new();
        let mut cold_x = Vec::new();
        let model = self.model.clone();
        let t_frame = std::time::Instant::now();
        if cold_ids.len() == picks.len() {
            // Nothing resident: the frame would run eight zero-weight slots.
            let fills = bank.admitted - admitted0;
            drop(bank);
            let out = crate::gpu::cpu_scope(|| {
                crate::qtensor::float_activations_scope(|| {
                    crate::pipeline::moe_cold_experts_cpu(&cold_jobs, x, pool)
                })
            });
            let mut st = STATS.lock().unwrap();
            st.calls += 1;
            st.picks += picks.len() as u64;
            st.cold += picks.len() as u64;
            st.fills += fills;
            return Some(out);
        }
        // The dedicated bank kernels (`gpu_wgpu::mimo_bank`) when this
        // adapter builds them; the generic DSV4 bank frame otherwise, or
        // with `CMF_MIMO_BANK_KERNEL=generic`.
        let inter = geom.inter;
        let fast = *self.fast.get_or_insert_with(|| {
            std::env::var("CMF_MIMO_BANK_KERNEL").as_deref() != Ok("generic")
                && crate::gpu_wgpu::mimo_bank::mimo_bank_ready(&model, hidden, inter, picks.len())
        });
        let sel: Vec<u32> = picks.iter().map(|&e| remap[e]).collect();
        let wt: Vec<f32> = picks.iter().map(|&e| mix[e]).collect();
        let (ok, cold_out) = std::thread::scope(|s| {
            // Cold picks run on the host WHILE the device runs the resident
            // ones; the scope keeps their matvecs off the per-op GPU hooks.
            let host = (!cold_jobs.is_empty()).then(|| {
                s.spawn(|| {
                    crate::gpu::cpu_scope(|| {
                        crate::qtensor::float_activations_scope(|| {
                            crate::pipeline::moe_cold_experts_cpu(&cold_jobs, x, pool)
                        })
                    })
                })
            });
            let ok = if fast {
                crate::gpu_wgpu::mimo_bank::mimo_bank_frame(&model, x, &sel, &wt, inter, &mut out)
            } else {
                crate::gpu_wgpu::dsv4_moe_frame(
                    &model,
                    &weights,
                    geom,
                    x,
                    &mut cold_seen,
                    &mut cold_x,
                    None,
                    None,
                    &mut out,
                ) && cold_seen
                    .iter()
                    .map(|&(e, _)| e)
                    .eq(cold_ids.iter().copied())
            };
            let cold = host.and_then(|h| h.join().ok());
            (ok, cold)
        });
        let frame_ns = t_frame.elapsed().as_nanos() as u64;
        let fills = bank.admitted - admitted0;
        drop(bank);
        // The device hands back the picks it could not serve; they must be
        // exactly the ones the host computed.
        if !ok || (!cold_ids.is_empty() && cold_out.is_none()) {
            tracing::warn!(
                "MiMo MoE bank: frame refused at layer {li} (ok={ok}, {} kernels, {} cold) — host \
                 path from now on",
                if fast { "bank" } else { "generic" },
                cold_ids.len()
            );
            self.failed = true;
            return None;
        }
        if let Some(mut c) = cold_out {
            for (o, v) in out.iter_mut().zip(&c) {
                *o += v;
            }
            crate::attention::recycle_buf(&mut c);
        }
        let mut st = STATS.lock().unwrap();
        st.calls += 1;
        st.picks += picks.len() as u64;
        st.hits += (picks.len() - cold_ids.len()) as u64;
        st.cold += cold_ids.len() as u64;
        st.fills += fills;
        st.frame_ns += frame_ns;
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    fn mimo(budget: u64, graph: bool) -> PlacementInputs {
        PlacementInputs {
            budget,
            non_expert: 5_500_000_000,
            per_expert: 13_107_200,
            moe_layers: 47,
            n_experts: 256,
            top_k: 8,
            attn_per_layer: 93_000_000,
            graph_prefix: graph,
        }
    }

    #[test]
    fn everything_fits_means_prefix() {
        let p = place(&mimo(400 * GB, true), &Costs::measured(), None);
        assert_eq!(p.mode, MoeMode::Prefix);
        assert_eq!(p.bank_slots, 0);
        assert_eq!(p.prefix_layers, 47);
    }

    #[test]
    fn forced_modes_are_honoured_and_sized_from_the_budget() {
        let c = Costs::measured();
        let inp = mimo(98 * GB, false);
        let room = inp.budget - inp.non_expert - device_reserve(inp.budget);
        let slots = (room / inp.per_expert) as usize;
        let d = place(&inp, &c, Some(MoeMode::Dynamic));
        assert_eq!(
            (d.mode, d.prefix_layers, d.bank_slots),
            (MoeMode::Dynamic, 0, slots)
        );
        let p = place(&inp, &c, Some(MoeMode::Prefix));
        assert_eq!((p.mode, p.bank_slots), (MoeMode::Prefix, 0));
        assert_eq!(p.prefix_layers, slots / 256);
        let h = place(&inp, &c, Some(MoeMode::Hybrid));
        assert_eq!(h.mode, MoeMode::Hybrid);
        assert!(h.prefix_layers >= 1 && h.prefix_layers <= slots / 256);
        assert_eq!(h.bank_slots, slots - h.prefix_layers * 256);
    }

    #[test]
    fn auto_prefers_the_cheapest_prediction() {
        let c = Costs::measured();
        for budget in [16 * GB, 24 * GB, 48 * GB, 80 * GB, 98 * GB] {
            for graph in [false, true] {
                let inp = mimo(budget, graph);
                let auto = place(&inp, &c, None);
                for m in [MoeMode::Prefix, MoeMode::Dynamic, MoeMode::Hybrid] {
                    let f = place(&inp, &c, Some(m));
                    assert!(
                        auto.predicted_s <= f.predicted_s + 1e-12,
                        "budget {budget} graph {graph}: auto {:?} {} > {m:?} {}",
                        auto.mode,
                        auto.predicted_s,
                        f.predicted_s
                    );
                }
            }
        }
    }

    /// The measured costs on MiMo-V2.6-Flash q4tp: without a device graph
    /// for its layers every ladder budget places the experts in the bank
    /// (a per-op whole-layer prefix pays the same fences as a bank layer and
    /// streams the rest from RAM); `cargo test -- --nocapture` prints the
    /// predictions with and without a graph prefix.
    #[test]
    fn ladder_choices_for_mimo() {
        let c = Costs::measured();
        for mb in [16_000u64, 24_000, 48_000, 80_000, 93_791] {
            let budget = mb * 1024 * 1024;
            let no_graph = place(&mimo(budget, false), &c, None);
            let graph = place(&mimo(budget, true), &c, None);
            println!(
                "{mb} MB: no graph → {:?} P={} bank {} ({:.1} ms); graph → {:?} P={} bank {} ({:.1} ms)",
                no_graph.mode,
                no_graph.prefix_layers,
                no_graph.bank_slots,
                no_graph.predicted_s * 1e3,
                graph.mode,
                graph.prefix_layers,
                graph.bank_slots,
                graph.predicted_s * 1e3,
            );
            assert_eq!(
                no_graph.mode,
                MoeMode::Dynamic,
                "{mb} MB: {}",
                no_graph.reason
            );
            assert!(graph.predicted_s <= no_graph.predicted_s + 1e-12);
        }
    }

    #[test]
    fn default_min_seen_follows_bank_size() {
        assert_eq!(default_min_seen(142), 2);
        assert_eq!(default_min_seen(48), 2);
        assert_eq!(default_min_seen(47), 3);
        assert_eq!(default_min_seen(13), 3);
    }

    #[test]
    fn a_budget_below_the_non_expert_weights_leaves_no_bank() {
        let inp = mimo(4 * GB, false);
        let d = place(&inp, &Costs::measured(), Some(MoeMode::Dynamic));
        assert_eq!(d.bank_slots, 0);
    }

    #[test]
    fn hit_curve_interpolates_monotonically() {
        let c = Costs::measured();
        let mut last = 0.0;
        for s in [
            1.0, 4.0, 8.0, 20.0, 32.0, 50.0, 64.0, 100.0, 128.0, 160.0, 192.0, 255.0,
        ] {
            let h = c.hit_rate(s, 256);
            assert!(h >= last && (0.0..=1.0).contains(&h), "{s}: {h}");
            last = h;
        }
        assert_eq!(c.hit_rate(256.0, 256), 1.0);
        assert_eq!(c.hit_rate(0.0, 256), 0.0);
    }

    #[test]
    fn env_mode_names_parse() {
        assert_eq!(MoeMode::parse("dynamic"), Some(Some(MoeMode::Dynamic)));
        assert_eq!(MoeMode::parse("HYBRID"), Some(Some(MoeMode::Hybrid)));
        assert_eq!(MoeMode::parse("prefix"), Some(Some(MoeMode::Prefix)));
        assert_eq!(MoeMode::parse("auto"), Some(None));
        assert_eq!(MoeMode::parse("fast"), None);
    }
}

/// The bank against the host path on a MiMo-shaped synthetic file whose
/// bank is far smaller than its expert count, so every run mixes resident
/// hits, fills and cold host picks. Needs a wgpu adapter with descriptor
/// arrays; skips otherwise. Run the CPU arm exact for the tight bounds:
///
///     CMF_SDOT=0 cargo test --release -p cortiq-engine --features gpu --lib bank_tests
#[cfg(all(test, feature = "gpu"))]
mod bank_tests {
    use super::*;
    use crate::pipeline::{FfnKind, Pipeline};
    use crate::sampler::SamplerConfig;
    use cortiq_core::CMF_VERSION;
    use cortiq_core::format::{CmfHeader, TensorSpec};
    use cortiq_core::quant::{
        GROUP_SIZE, dequant_q4tp, f32_to_f16, q4tp_code_stride, q4tp_put_code, q4tp_sections,
    };
    use cortiq_core::types::{ModelArch, QuantType};
    use std::collections::HashMap;

    const HS: usize = 256;
    const INTER: usize = 64;
    const NE: usize = 16;
    const TOPK: usize = 4;
    const NH: usize = 4;
    const HD: usize = 32;
    const VD: usize = 16;
    const VOCAB: usize = 64;
    const DENSE_INTER: usize = 96;
    /// KV heads per layer: full, sliding, sliding, full (MiMo's pattern).
    const KVH: [usize; 4] = [1, 2, 2, 1];
    /// The bank: 8 slots for 3 × 16 experts.
    const SLOTS: usize = 8;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }
        /// Uniform in [-0.5, 0.5).
        fn f(&mut self) -> f32 {
            (self.next() & 0xFF_FFFF) as f32 / (1u32 << 24) as f32 - 0.5
        }
    }

    fn f32_spec(name: String, shape: &[usize], rng: &mut Rng, scale: f32, bias: f32) -> TensorSpec {
        let n: usize = shape.iter().product();
        TensorSpec {
            name,
            dtype: TensorDtype::F32,
            shape: shape.to_vec(),
            data: (0..n)
                .flat_map(|_| (bias + scale * rng.f()).to_le_bytes())
                .collect(),
        }
    }

    /// A valid q4tp payload: random nibbles and rung codes, a per-row ladder
    /// around 2^-7 so the dequantized weights sit near ±0.1.
    fn q4tp_bytes(rows: usize, cols: usize, rng: &mut Rng) -> Vec<u8> {
        let gpr = cols / GROUP_SIZE;
        let stride = q4tp_code_stride(gpr);
        let (params_off, codes_off, _) = q4tp_sections(rows, cols);
        let mut b = vec![0u8; codes_off + rows * stride];
        for byte in b[..params_off].iter_mut() {
            *byte = rng.next() as u8;
        }
        for r in 0..rows {
            let p = params_off + r * 4;
            let lo = -7.0 + 0.5 * rng.f();
            let st = 0.06 + 0.02 * rng.f();
            b[p..p + 2].copy_from_slice(&f32_to_f16(lo).to_le_bytes());
            b[p + 2..p + 4].copy_from_slice(&f32_to_f16(st).to_le_bytes());
            let crow = &mut b[codes_off + r * stride..codes_off + (r + 1) * stride];
            for g in 0..gpr {
                q4tp_put_code(crow, g, (rng.next() % 32) as usize);
            }
        }
        b
    }

    fn arch() -> ModelArch {
        serde_json::from_value(serde_json::json!({
            "arch_name": "mimo_v2",
            "hidden_size": HS,
            "intermediate_size": DENSE_INTER,
            "num_layers": 4,
            "num_attention_heads": NH,
            "num_kv_heads": KVH[0],
            "head_dim": HD,
            "vocab_size": VOCAB,
            "layer_types": ["FullAttention", "SlidingAttention", "SlidingAttention", "FullAttention"],
            "rms_norm_eps": 1e-6,
            "rope_theta": 10_000_000.0,
            "rope_local_base_freq": 10_000.0,
            "partial_rotary_factor": 0.5,
            "sliding_window": 3,
            "tie_word_embeddings": true,
            "max_position_embeddings": 256,
            "linear_conv_kernel_dim": 0,
            "linear_num_key_heads": 0,
            "linear_num_value_heads": 0,
            "kv_heads_per_layer": KVH,
            "v_head_dim": VD,
            "moe": {
                "num_experts": NE,
                "top_k": TOPK,
                "moe_intermediate_size": INTER,
                "norm_topk_prob": true,
                "router_sigmoid": true
            }
        }))
        .expect("arch")
    }

    /// Write the file; returns the expert payloads by tensor name for the
    /// exact reference.
    fn write_model(tag: &str) -> (std::path::PathBuf, Arc<CmfModel>, HashMap<String, Vec<u8>>) {
        let mut rng = Rng(0x5EED_0000 ^ tag.len() as u64);
        let mut specs = vec![
            f32_spec(
                "model.embed_tokens.weight".into(),
                &[VOCAB, HS],
                &mut rng,
                2.0,
                0.0,
            ),
            f32_spec("model.norm.weight".into(), &[HS], &mut rng, 0.2, 1.0),
        ];
        let mut experts = HashMap::new();
        for (li, &kv) in KVH.iter().enumerate() {
            let p = format!("model.layers.{li}.");
            specs.push(f32_spec(
                format!("{p}input_layernorm.weight"),
                &[HS],
                &mut rng,
                0.2,
                1.0,
            ));
            specs.push(f32_spec(
                format!("{p}post_attention_layernorm.weight"),
                &[HS],
                &mut rng,
                0.2,
                1.0,
            ));
            for (n, shape) in [
                ("q_proj", [NH * HD, HS]),
                ("k_proj", [kv * HD, HS]),
                ("v_proj", [kv * VD, HS]),
                ("o_proj", [HS, NH * VD]),
            ] {
                let mut spec = f32_spec(
                    format!("{p}self_attn.{n}.weight"),
                    &shape,
                    &mut rng,
                    0.2,
                    0.0,
                );
                if tag == "attn-graph" {
                    // The real MiMo graph skeleton is q8_2f. F32 fixture
                    // projections deliberately cannot enter the batch GEMM.
                    let scale = 0.2f32 / 127.0;
                    let mut data: Vec<u8> = spec.data.chunks_exact(4)
                        .map(|v| (f32::from_le_bytes(v.try_into().unwrap()) / scale)
                            .round().clamp(-127.0, 127.0) as i8 as u8).collect();
                    for _ in 0..shape[0] { data.extend_from_slice(&f32_to_f16(scale).to_le_bytes()); }
                    for _ in 0..shape[1] { data.extend_from_slice(&f32_to_f16(1.0).to_le_bytes()); }
                    spec.dtype = TensorDtype::Q8_2f;
                    spec.data = data;
                }
                specs.push(spec);
            }
            if li == 1 || li == 2 {
                specs.push(f32_spec(
                    format!("{p}self_attn.sinks"),
                    &[NH],
                    &mut rng,
                    2.0,
                    0.0,
                ));
            }
            if li == 0 {
                for (n, shape) in [
                    ("gate_proj", [DENSE_INTER, HS]),
                    ("up_proj", [DENSE_INTER, HS]),
                    ("down_proj", [HS, DENSE_INTER]),
                ] {
                    specs.push(f32_spec(
                        format!("{p}mlp.{n}.weight"),
                        &shape,
                        &mut rng,
                        0.2,
                        0.0,
                    ));
                }
                continue;
            }
            specs.push(f32_spec(
                format!("{p}mlp.gate.weight"),
                &[NE, HS],
                &mut rng,
                0.4,
                0.0,
            ));
            specs.push(f32_spec(
                format!("{p}mlp.expert_bias"),
                &[NE],
                &mut rng,
                0.2,
                0.0,
            ));
            for e in 0..NE {
                for (n, rows, cols) in [
                    ("gate_proj", INTER, HS),
                    ("up_proj", INTER, HS),
                    ("down_proj", HS, INTER),
                ] {
                    let name = format!("{p}mlp.experts.{e}.{n}.weight");
                    let data = q4tp_bytes(rows, cols, &mut rng);
                    experts.insert(name.clone(), data.clone());
                    specs.push(TensorSpec {
                        name,
                        dtype: TensorDtype::Q4TiledP,
                        shape: vec![rows, cols],
                        data,
                    });
                }
            }
        }
        let header = CmfHeader {
            format: "cmf".into(),
            version: CMF_VERSION,
            arch: arch(),
            quant_type: QuantType::F32,
            provenance: None,
            tokenizer_config: None,
            section_hashes: None,
            skills: Vec::new(),
            shard: None,
            calibration: None,
            routing: None,
        };
        let dir = std::env::temp_dir().join(format!("cmf-mimo-bank-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.cmf");
        CmfModel::write(&path, &header, &specs, None, None).unwrap();
        (dir, Arc::new(CmfModel::open(&path).unwrap()), experts)
    }

    fn moe_layers(p: &Pipeline) -> Vec<(usize, &MoeFfn)> {
        p.weights
            .layers
            .iter()
            .enumerate()
            .filter_map(|(li, lw)| match &lw.ffn {
                FfnKind::Moe(m) => Some((li, m)),
                _ => None,
            })
            .collect()
    }

    fn bank(p: &Pipeline) -> Slot {
        Slot::decide_with(
            &moe_layers(p),
            p.num_layers,
            false,
            Some(MoeMode::Dynamic),
            Some(SLOTS),
        )
    }

    /// max|a−b| / max|b|.
    fn rel(a: &[f32], b: &[f32]) -> f32 {
        let d = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        d / b.iter().map(|v| v.abs()).fold(1e-30f32, f32::max)
    }

    /// Σ w·down(silu(gate·x) ⊙ up·x) from the dequantized payloads, in f64.
    fn exact_moe(
        experts: &HashMap<String, Vec<u8>>,
        li: usize,
        x: &[f32],
        picks: &[usize],
        w: &[f32],
    ) -> Vec<f32> {
        let deq = |n: &str, rows: usize, cols: usize| {
            let mut v = vec![0f32; rows * cols];
            dequant_q4tp(&experts[n], rows, cols, &mut v);
            v
        };
        let mut out = vec![0f64; HS];
        for &e in picks {
            let p = format!("model.layers.{li}.mlp.experts.{e}.");
            let g = deq(&format!("{p}gate_proj.weight"), INTER, HS);
            let u = deq(&format!("{p}up_proj.weight"), INTER, HS);
            let d = deq(&format!("{p}down_proj.weight"), HS, INTER);
            let mut act = vec![0f64; INTER];
            for r in 0..INTER {
                let (mut gv, mut uv) = (0f64, 0f64);
                for c in 0..HS {
                    gv += g[r * HS + c] as f64 * x[c] as f64;
                    uv += u[r * HS + c] as f64 * x[c] as f64;
                }
                act[r] = gv / (1.0 + (-gv).exp()) * uv;
            }
            for r in 0..HS {
                let mut acc = 0f64;
                for c in 0..INTER {
                    acc += d[r * INTER + c] as f64 * act[c];
                }
                out[r] += w[e] as f64 * acc;
            }
        }
        out.into_iter().map(|v| v as f32).collect()
    }

    fn bank_ready() -> bool {
        crate::gpu::enabled()
            && crate::gpu::wgpu_active()
            && crate::gpu_wgpu::dsv4_global_moe_supported()
    }

    fn serial() -> std::sync::MutexGuard<'static, ()> {
        static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
        GPU.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Layer level: every routed call through an 8-slot bank equals the
    /// exact f64 expert sum to f32 summation order, with hits, fills and
    /// cold host picks all exercised.
    #[test]
    fn bank_layer_equals_exact_expert_sum() {
        layer_check("layer", false);
    }

    /// The same check through the generic DSV4 bank frame (the fallback
    /// when the dedicated kernels cannot be built).
    #[test]
    fn generic_bank_layer_equals_exact_expert_sum() {
        layer_check("layer-generic", true);
    }

    fn layer_check(tag: &str, generic: bool) {
        let _g = serial();
        if !bank_ready() {
            eprintln!("skip: no wgpu adapter with an expert bank");
            return;
        }
        let (dir, model, experts) = write_model(tag);
        let p = Pipeline::from_model(&model, SamplerConfig::default()).expect("load");
        let mut slot = bank(&p);
        assert!(
            slot.is_on(),
            "the bank must come up on this adapter: {}",
            last_decision()
        );
        if let Slot::On(d) = &mut slot {
            d.fast = generic.then_some(false);
        }
        let s0 = stats();
        let mut rng = Rng(77);
        let strict = !crate::qtensor::a8w8_enabled();
        let (mut worst_dyn, mut worst_host, mut calls) = (0f32, 0f32, 0usize);
        for round in 0..16 {
            for (li, m) in moe_layers(&p) {
                // Tokens drift slowly, as decode hiddens do: routes repeat
                // (hits) and change (fills, cold picks).
                let x: Vec<f32> = (0..HS)
                    .map(|i| {
                        ((i * 7 + li * 3) as f32 * 0.37).sin() + 0.35 * rng.f() * (round % 3) as f32
                    })
                    .collect();
                let r = crate::pipeline::moe_ffn_route(m, &x, None, None);
                assert_eq!(r.idx.len(), TOPK);
                let mix: Vec<f32> = (0..NE)
                    .map(|e| {
                        if r.idx.contains(&e) {
                            r.p[e] / r.wsum
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let want = exact_moe(&experts, li, &x, &r.idx, &mix);
                let jobs: Vec<_> = r.idx.iter().map(|&e| (&m.experts[e], mix[e])).collect();
                let host = crate::gpu::cpu_scope(|| {
                    crate::pipeline::moe_cold_experts_cpu(&jobs, &x, None)
                });
                let got = slot
                    .forward(li, m, &x, &r, None)
                    .expect("the bank served the layer");
                worst_dyn = worst_dyn.max(rel(&got, &want));
                worst_host = worst_host.max(rel(&host, &want));
                calls += 1;
            }
        }
        let s1 = stats();
        let (hits, fills, cold) = (s1.hits - s0.hits, s1.fills - s0.fills, s1.cold - s0.cold);
        let kernels = match &slot {
            Slot::On(d) => d.fast,
            _ => None,
        };
        assert_eq!(
            kernels,
            Some(!generic),
            "expected the {} kernels",
            if generic { "generic" } else { "bank" }
        );
        eprintln!(
            "bank layer check ({tag}): {calls} calls, {} picks: hits {hits} fills {fills} cold {cold}; \
             max rel |bank−exact| {worst_dyn:.2e}, |host−exact| {worst_host:.2e} (a8w8 {})",
            s1.picks - s0.picks,
            !strict
        );
        assert!(
            hits > 0 && fills > 0 && cold > 0,
            "hits {hits} fills {fills} cold {cold}"
        );
        // Device f32 against an f64 sum of the same dequantized weights; the
        // cold picks ride the host kernels, exact under CMF_SDOT=0 and int8
        // activations otherwise.
        // Under A8W8 the host picks carry the int8-activation error; the
        // bank's result must not be further from exact than the host's.
        let bound = if strict { 1e-5 } else { worst_host.max(1e-5) };
        assert!(
            worst_dyn <= bound,
            "bank vs exact {worst_dyn:.3e} > {bound:e}"
        );
        drop(p);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cold_batch_rows_equal_single_token_kernels() {
        let (dir, model, _) = write_model("cold-rows");
        let p = Pipeline::from_model(&model, SamplerConfig::default()).unwrap();
        let (_, m) = moe_layers(&p)[0];
        let xs: Vec<f32> = (0..4 * HS).map(|i| (i as f32 * 0.17).sin()).collect();
        let jobs = vec![
            vec![(&m.experts[0], 0.3), (&m.experts[1], 0.7)],
            vec![],
            vec![(&m.experts[1], 0.2), (&m.experts[0], 0.8)],
            vec![(&m.experts[0], 1.0)],
        ];
        crate::gpu::cpu_scope(|| {
            let batch = crate::pipeline::moe_cold_experts_rows_cpu(&jobs, &xs, HS, None);
            for (r, jobs) in jobs.iter().enumerate() {
                let one =
                    crate::pipeline::moe_cold_experts_cpu(jobs, &xs[r * HS..(r + 1) * HS], None);
                assert_eq!(batch[r * HS..(r + 1) * HS], one, "cold row {r}");
            }
        });
        drop(p);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn dynamic_attention_graph_batches_preserve_layer_keys_and_rewind() {
        let _g = serial();
        if !bank_ready() {
            return;
        }
        let (dir, model, _) = write_model("attn-graph");
        let mut batch = Pipeline::from_model(&model, SamplerConfig::default()).unwrap();
        let mut single = Pipeline::from_model(&model, SamplerConfig::default()).unwrap();
        batch.mimo_moe = bank(&batch);
        single.mimo_moe = bank(&single);
        // The short-panel head/projection kernel must equal the old
        // eight-lane panel bit for bit, including every batch width.
        let crate::pipeline::AttnKind::Full { wq, .. } = &batch.weights.layers[2].attn else {
            unreachable!()
        };
        let (owner, idx, _, _) = wq.graph_weight().unwrap();
        for b in 1..=4 {
            let xs: Vec<f32> = (0..b * HS).map(|i| (i as f32 * 0.071).cos()).collect();
            let mut actual = vec![0.0; b * NH * HD];
            let mut expected = actual.clone();
            assert!(crate::gpu::q82_short_rows(
                owner,
                idx,
                &xs,
                b,
                NH * HD,
                HS,
                &mut actual
            ));
            assert!(crate::gpu::mimo_q8_short_scope(false, || {
                crate::gpu::q82_short_rows(owner, idx, &xs, b, NH * HD, HS, &mut expected)
            }));
            assert_eq!(actual, expected, "q82 short/wide panel b={b}");
        }
        // Distinct global/SWA geometries; neither may overwrite layer zero.
        // Pass 300 absolute positions to cover split-K full attention as
        // well as many SWA wraps and rejected suffix overwrites.
        batch.kv_cache.max_seq_len = 512;
        single.kv_cache.max_seq_len = 512;
        for li in [2, 3] {
            let mut pos = 0;
            for b in (0..180).map(|i| i % 4 + 1) {
                let positions: Vec<_> = (pos..pos + b).collect();
                let xs: Vec<f32> = (0..b * HS)
                    .map(|i| ((i + pos * HS) as f32 * 0.017).sin())
                    .collect();
                let mut ys = xs.clone();
                assert!(
                    matches!(
                        batch.mimo_graph_layer_rows(li, &mut ys, &positions),
                        crate::gpu::BatchGraphOutcome::Completed
                    ),
                    "batch admission li={li}, b={b}"
                );
                let mut want = xs;
                for (row, &p) in want.chunks_exact_mut(HS).zip(&positions) {
                    let outcome = crate::gpu::mimo_q8_short_scope(false, || {
                        crate::gpu::mimo_attention_scratch_scope(false, || {
                            single.mimo_graph_layer_rows(li, row, &[p])
                        })
                    });
                    assert!(matches!(outcome, crate::gpu::BatchGraphOutcome::Completed));
                }
                assert!(
                    rel(&ys, &want) < 2e-5,
                    "li={li}, b={b}: {}",
                    rel(&ys, &want)
                );
                pos += b;
                assert_eq!(
                    crate::gpu::graph_kv_stored(batch.test_graph_kv_id(), li),
                    Some(pos)
                );
                assert_eq!(
                    crate::gpu::graph_kv_stored(batch.test_graph_kv_id(), 0),
                    None
                );
                // Rejected suffix must be overwritable on both full and SWA KV.
                if b > 1 {
                    pos -= 1;
                    assert!(crate::gpu::graph_kv_set_stored(
                        batch.test_graph_kv_id(),
                        li,
                        pos
                    ));
                    assert!(crate::gpu::graph_kv_set_stored(
                        single.test_graph_kv_id(),
                        li,
                        pos
                    ));
                }
            }
            assert!(pos > 300);
        }
        drop(batch);
        drop(single);
        drop(model);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn hybrid_bank_epoch_advances_at_dynamic_boundary() {
        let _g = serial();
        if !bank_ready() {
            return;
        }
        let (dir, model, _) = write_model("hybrid-epoch");
        let p = Pipeline::from_model(&model, SamplerConfig::default()).unwrap();
        let mut slot = bank(&p);
        let Slot::On(d) = &mut slot else {
            panic!("bank unavailable")
        };
        // The first MoE layer (1) runs in the graph and never visits the
        // bank. Layer 2 must still release last round's eviction pins.
        d.dyn_from = 2;
        d.placement.mode = MoeMode::Hybrid;
        d.placement.prefix_layers = 1;
        let bank = d.bank.clone();
        let initial = bank.lock().unwrap().tok;
        let x: Vec<f32> = (0..HS).map(|i| (i as f32 * 0.13).sin()).collect();
        for li in [2, 3] {
            let FfnKind::Moe(m) = &p.weights.layers[li].ffn else {
                unreachable!()
            };
            let route = crate::pipeline::moe_ffn_route(m, &x, None, None);
            assert!(slot.forward(li, m, &x, &route, None).is_some());
            assert_eq!(bank.lock().unwrap().tok, initial + 1, "decode layer {li}");
        }
        let xs = [x.as_slice(), x.as_slice()].concat();
        for li in [2, 3] {
            let FfnKind::Moe(m) = &p.weights.layers[li].ffn else {
                unreachable!()
            };
            let routes: Vec<_> = xs.chunks_exact(HS)
                .map(|row| crate::pipeline::moe_ffn_route(m, row, None, None))
                .collect();
            assert!(slot.forward_rows(li, m, &xs, &routes, None).is_some());
            assert_eq!(bank.lock().unwrap().tok, initial + 2, "verify layer {li}");
        }
        drop(p);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn bank_batch_frame_equals_single_token_frames() {
        let _g = serial();
        if !bank_ready() {
            return;
        }
        let (dir, model, _) = write_model("batch-frames");
        let p = Pipeline::from_model(&model, SamplerConfig::default()).unwrap();
        let mut slot = bank(&p);
        assert_eq!(slot.graph_prefix_end(), Some(1));
        let Slot::On(d) = &mut slot else {
            panic!("bank unavailable")
        };
        let bank_arc = d.bank.clone();
        {
            let mut b = bank_arc.lock().unwrap();
            b.next_token();
            b.resolve(1, &[0, 1, 2, 3], &d.ids[1]).unwrap();
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let mut b = bank_arc.lock().unwrap();
            b.drain();
            if b.pending() == 0 {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "bank fills timed out");
            drop(b);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let mut b = bank_arc.lock().unwrap();
        let remap = b.resolve(1, &[0, 1, 2, 3], &d.ids[1]).unwrap();
        assert!((0..4).all(|e| remap[e] != u32::MAX));
        for rows in 1..=4 {
            let xs: Vec<f32> = (0..rows * HS).map(|i| (i as f32 * 0.13).cos()).collect();
            let sel: Vec<u32> = (0..rows * TOPK)
                .map(|i| if i % 5 == 0 { u32::MAX } else { remap[i % 4] })
                .collect();
            let wt: Vec<f32> = (0..sel.len())
                .map(|i| 0.1 + 0.03 * (i % 4) as f32)
                .collect();
            let mut batch = vec![0.0; xs.len()];
            assert!(crate::gpu_wgpu::mimo_bank::mimo_bank_rows(
                &model, &xs, &sel, &wt, INTER, rows, &mut batch
            ));
            for row in 0..rows {
                let mut one = vec![0.0; HS];
                assert!(crate::gpu_wgpu::mimo_bank::mimo_bank_frame(
                    &model,
                    &xs[row * HS..(row + 1) * HS],
                    &sel[row * TOPK..(row + 1) * TOPK],
                    &wt[row * TOPK..(row + 1) * TOPK],
                    INTER,
                    &mut one
                ));
                assert_eq!(batch[row * HS..(row + 1) * HS], one, "GPU row {row}/{rows}");
            }
        }
        drop(b);
        drop(p);
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// Model level: greedy decode of a prompt through the bank equals the
    /// pure host walk — logits to summation order, identical tokens.
    #[test]
    fn bank_decode_equals_host_decode() {
        let _g = serial();
        if !bank_ready() {
            eprintln!("skip: no wgpu adapter with an expert bank");
            return;
        }
        let (dir, model, _) = write_model("decode");
        let run = |bank_on: bool| -> (Vec<Vec<f32>>, Vec<u32>) {
            let mut p = Pipeline::from_model(&model, SamplerConfig::default()).expect("load");
            p.mimo_moe = if bank_on { bank(&p) } else { Slot::Off };
            assert_eq!(p.mimo_moe.is_on(), bank_on, "{}", last_decision());
            let n = p.num_layers;
            let mut ids: Vec<u32> = vec![3, 17, 42, 5, 9, 33, 21, 8, 60, 1, 12];
            let prompt = ids.len();
            let mut all = Vec::new();
            for pos in 0..prompt + 12 {
                let id = ids[pos];
                let step = |p: &mut Pipeline| {
                    let emb = p.embed_id(id);
                    let h = p.forward_span(&emb, pos, 0, n - 1, None).unwrap();
                    p.logits_from_hidden(&h)
                };
                let lg = if bank_on {
                    step(&mut p)
                } else {
                    crate::gpu::cpu_scope(|| step(&mut p))
                };
                if pos + 1 >= prompt {
                    let next = lg
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(i, _)| i as u32)
                        .unwrap();
                    ids.push(next);
                }
                all.push(lg);
            }
            (all, ids)
        };
        let s0 = stats();
        let (host, host_ids) = run(false);
        let s1 = stats();
        assert_eq!(s1.calls, s0.calls, "the host run must not touch the bank");
        let (dynm, dyn_ids) = run(true);
        let s2 = stats();
        let worst = host
            .iter()
            .zip(&dynm)
            .map(|(h, d)| rel(d, h))
            .fold(0f32, f32::max);
        let strict = !crate::qtensor::a8w8_enabled();
        eprintln!(
            "bank decode check: {} steps, bank calls {} hits {} fills {} cold {}; logits max rel \
             {worst:.2e}; greedy {:?} vs host {:?}",
            host.len(),
            s2.calls - s1.calls,
            s2.hits - s1.hits,
            s2.fills - s1.fills,
            s2.cold - s1.cold,
            &dyn_ids[11..],
            &host_ids[11..],
        );
        assert_eq!(
            s2.calls - s1.calls,
            (host.len() * 3) as u64,
            "every MoE call took the bank"
        );
        assert!(
            s2.hits > s1.hits && s2.cold > s1.cold,
            "both resident and cold picks ran"
        );
        if !strict {
            // The host arm quantizes activations to int8 (A8W8); on this
            // toy that alone moves its logits by percent. Equality is a
            // claim about the exact arm: rerun with CMF_SDOT=0.
            eprintln!("bank decode check: A8W8 host arm — bounds need CMF_SDOT=0, not asserted");
            let _ = std::fs::remove_dir_all(&dir);
            return;
        }
        assert!(worst < 1e-4, "logits bank vs host {worst:.3e} ≥ 1e-4");
        // Every step's argmax, prompt positions included (teacher forced),
        // not only the generated tail.
        let am = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        let (a_host, a_dyn): (Vec<usize>, Vec<usize>) =
            host.iter().zip(&dynm).map(|(h, d)| (am(h), am(d))).unzip();
        assert_eq!(a_dyn, a_host, "per-step argmax differs");
        assert_eq!(dyn_ids, host_ids, "greedy tokens differ");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
