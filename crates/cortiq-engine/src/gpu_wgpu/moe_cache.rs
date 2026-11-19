//! Card slots over a routed-expert bank in GPU-visible system memory.
//!
//! When a stack's routed experts do not fit the card, whole layers' banks go
//! to system memory (`moe_expert_sysmem`) and their kernels read the picked
//! experts over the bus every token. This module gives such a layer an LRU
//! cache of `cap` experts in card memory, kept entirely on the device inside
//! the token graph: after the select, one workgroup looks each pick up in the
//! layer's table, gives a miss the least recently used slot (never one this
//! token's picks hold) and lists it; a copy kernel brings the listed experts
//! from the bank into their slots; the expert kernels then run unchanged on
//! the slot buffers, the picks replaced by their slot numbers. Nothing waits
//! for the host, and an expert read once over the bus stays on the card while
//! the routing keeps coming back to it.

use super::*;
use std::sync::{Arc, Mutex, OnceLock};

pub(crate) const NONE: u32 = u32::MAX;

pub(crate) const CACHE_WGSL: &str = r#"
// ── the slot table: pick -> slot, misses evict the least recently used ──
// st: [0] tick, [1 .. 1+ne) slot of each bank entry, [1+ne .. 1+ne+cap) the
// entry a slot holds, [1+ne+cap .. 1+ne+2cap) the tick a slot was last used.
// out: [0 .. slots) the slot of each pick; [64 + m] / [128 + m] the bank
// entry and the slot of the m-th copy. ind: the copy's indirect grid
// (chunks, copies, 3): a token whose picks all hit launches no copy.
struct CfP { slots: u32, ne: u32, cap: u32, chunks: u32 };
@group(0) @binding(0) var<storage, read>       cf_sel : array<u32>;
@group(0) @binding(1) var<storage, read_write> cf_st  : array<u32>;
@group(0) @binding(2) var<storage, read_write> cf_out : array<u32>;
@group(0) @binding(3) var<uniform>             cf_p   : CfP;
@group(0) @binding(4) var<storage, read_write> cf_ind : array<u32>;
var<workgroup> cf_v: array<u32, 256>;
var<workgroup> cf_i: array<u32, 256>;
var<workgroup> cf_miss: array<u32, 64>;
var<workgroup> cf_n: u32;

@compute @workgroup_size(256)
fn moe_cache_fill(@builtin(local_invocation_index) lid: u32) {
    let ne = cf_p.ne;
    let cap = cf_p.cap;
    let k = min(cf_p.slots, 64u);
    let ow = 1u + ne;
    let la = 1u + ne + cap;
    let tick = cf_st[0] + 1u;
    if (lid == 0u) { cf_n = 0u; }
    // hits first: their slots are stamped with this tick, so no miss below
    // can take them
    if (lid < k) {
        let e = cf_sel[lid];
        var s = 0xFFFFFFFFu;
        if (e < ne) { s = cf_st[1u + e]; }
        if (s != 0xFFFFFFFFu) {
            cf_st[la + s] = tick;
            cf_out[lid] = s;
            cf_miss[lid] = 0u;
        } else {
            cf_miss[lid] = 1u;
        }
    }
    storageBarrier();
    workgroupBarrier();
    for (var j = 0u; j < k; j = j + 1u) {
        let miss = workgroupUniformLoad(&cf_miss[j]);
        if (miss != 0u) {
            // the least recently used slot, ties to the lower one
            var bv = 0xFFFFFFFFu;
            var bi = 0u;
            for (var s = lid; s < cap; s = s + 256u) {
                let v = cf_st[la + s];
                if (v < bv) { bv = v; bi = s; }
            }
            cf_v[lid] = bv;
            cf_i[lid] = bi;
            workgroupBarrier();
            var st = 128u;
            loop {
                if (st == 0u) { break; }
                if (lid < st) {
                    let a = cf_v[lid];
                    let b = cf_v[lid + st];
                    let ib = cf_i[lid + st];
                    if (b < a || (b == a && ib < cf_i[lid])) {
                        cf_v[lid] = b;
                        cf_i[lid] = ib;
                    }
                }
                workgroupBarrier();
                st = st >> 1u;
            }
            if (lid == 0u) {
                let v = cf_i[0];
                let e = cf_sel[j];
                if (e < ne) {
                    let old = cf_st[ow + v];
                    if (old < ne) { cf_st[1u + old] = 0xFFFFFFFFu; }
                    cf_st[ow + v] = e;
                    cf_st[1u + e] = v;
                    let m = cf_n;
                    cf_out[64u + m] = e;
                    cf_out[128u + m] = v;
                    cf_n = m + 1u;
                }
                cf_st[la + v] = tick;
                cf_out[j] = v;
            }
            storageBarrier();
            workgroupBarrier();
        }
    }
    if (lid == 0u) {
        cf_st[0] = tick;
        cf_ind[0] = cf_p.chunks;
        cf_ind[1] = cf_n;
        cf_ind[2] = 3u;
    }
}

// ── the listed experts from the bank into their slots ──
// grid (chunks, copies, 3), indirect: z picks gate / up / down; a chunk is
// `chunk` words.
struct CcP { gu_words: u32, dn_words: u32, slots: u32, chunk: u32 };
@group(0) @binding(0) var<storage, read>       cc_gs  : array<u32>;
@group(0) @binding(1) var<storage, read>       cc_us  : array<u32>;
@group(0) @binding(2) var<storage, read>       cc_ds  : array<u32>;
@group(0) @binding(3) var<storage, read_write> cc_gd  : array<u32>;
@group(0) @binding(4) var<storage, read_write> cc_ud  : array<u32>;
@group(0) @binding(5) var<storage, read_write> cc_dd  : array<u32>;
@group(0) @binding(6) var<storage, read>       cc_out : array<u32>;
@group(0) @binding(7) var<uniform>             cc_p   : CcP;

@compute @workgroup_size(256)
fn moe_cache_copy(@builtin(workgroup_id) wid: vec3<u32>,
                  @builtin(local_invocation_index) lid: u32) {
    let m = wid.y;
    let e = cc_out[64u + m];
    let v = cc_out[128u + m];
    let z = wid.z;
    var words = cc_p.gu_words;
    if (z == 2u) { words = cc_p.dn_words; }
    let b0 = wid.x * cc_p.chunk;
    if (b0 >= words) { return; }
    let b1 = min(b0 + cc_p.chunk, words);
    let src = e * words;
    let dst = v * words;
    for (var i = b0 + lid; i < b1; i = i + 256u) {
        if (z == 0u) {
            cc_gd[dst + i] = cc_gs[src + i];
        } else if (z == 1u) {
            cc_ud[dst + i] = cc_us[src + i];
        } else {
            cc_dd[dst + i] = cc_ds[src + i];
        }
    }
}
"#;

/// Words a copy workgroup moves (8 a lane).
const CHUNK_WORDS: u32 = 2048;

pub(crate) struct Pipes {
    pub fill: wgpu::ComputePipeline,
    pub fill_l: wgpu::BindGroupLayout,
    pub copy: wgpu::ComputePipeline,
    pub copy_l: wgpu::BindGroupLayout,
}

fn build(c: &Ctx) -> Option<Pipes> {
    if c.device.limits().max_storage_buffers_per_shader_stage < 7 {
        return None;
    }
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-moe-cache"),
        source: wgpu::ShaderSource::Wgsl(CACHE_WGSL.into()),
    });
    let pipe = |ep: &str| {
        c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(ep),
            layout: None,
            module: &module,
            entry_point: Some(ep),
            compilation_options: crate::gpu_wgpu::pco(),
            cache: c.pipeline_cache.as_ref(),
        })
    };
    let fill = pipe("moe_cache_fill");
    let copy = pipe("moe_cache_copy");
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("cmf-moe-cache module rejected ({e}): system-memory experts are read in place");
        return None;
    }
    let fill_l = fill.get_bind_group_layout(0);
    let copy_l = copy.get_bind_group_layout(0);
    Some(Pipes { fill, fill_l, copy, copy_l })
}

pub(crate) fn pipes(c: &Ctx) -> Option<&Pipes> {
    c.moe_cache_pipes.get_or_init(|| build(c)).as_ref()
}

/// Whether the caches may be made: `CMF_WGPU_EXPERT_CACHE=0` keeps every
/// system-memory bank read in place.
pub(crate) fn on() -> bool {
    std::env::var("CMF_WGPU_EXPERT_CACHE").as_deref() != Ok("0")
}

/// Whether a wide batch copies a system-memory bank to the card first
/// (`CMF_WGPU_EXPERT_STAGE=0`: the batch kernels read it in place).
pub(crate) fn stage_on() -> bool {
    std::env::var("CMF_WGPU_EXPERT_STAGE").as_deref() != Ok("0")
}

/// One layer's slots, their table and the kernels' uniforms.
pub(crate) struct Cache {
    pub g: wgpu::Buffer,
    pub u: wgpu::Buffer,
    pub d: wgpu::Buffer,
    pub st: wgpu::Buffer,
    pub out: wgpu::Buffer,
    ind: wgpu::Buffer,
    fill_u: wgpu::Buffer,
    copy_u: wgpu::Buffer,
    pub cap: usize,
}

impl Cache {
    /// The fill and copy dispatches between the select (which wrote `sel`)
    /// and the expert kernels, which then read `self.g/u/d` and `self.out`.
    pub(crate) fn encode(&self, c: &Ctx, pass: &mut wgpu::ComputePass<'_>, sel: &wgpu::Buffer, gs: &wgpu::Buffer, us: &wgpu::Buffer, ds: &wgpu::Buffer) {
        let Some(p) = pipes(c) else { return };
        let bf = bind_pairs(c, &p.fill_l, &[(0, sel), (1, &self.st), (2, &self.out), (3, &self.fill_u), (4, &self.ind)]);
        let bc = bind_pairs(
            c,
            &p.copy_l,
            &[(0, gs), (1, us), (2, ds), (3, &self.g), (4, &self.u), (5, &self.d), (6, &self.out), (7, &self.copy_u)],
        );
        pass.set_pipeline(&p.fill);
        pass.set_bind_group(0, &bf, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&p.copy);
        pass.set_bind_group(0, &bc, &[]);
        pass.dispatch_workgroups_indirect(&self.ind, 0);
    }
}

type Key = (usize, usize);

fn caches() -> &'static Mutex<HashMap<Key, Arc<Cache>>> {
    static M: OnceLock<Mutex<HashMap<Key, Arc<Cache>>>> = OnceLock::new();
    M.get_or_init(Default::default)
}

/// The banks whose cache the card refused (`get` does not ask again).
fn refused() -> &'static Mutex<std::collections::HashSet<Key>> {
    static M: OnceLock<Mutex<std::collections::HashSet<Key>>> = OnceLock::new();
    M.get_or_init(Default::default)
}

/// The keys of the banks `moe_expert_sysmem` placed in system memory: only
/// those get slots.
fn sys_banks() -> &'static Mutex<std::collections::HashSet<Key>> {
    static M: OnceLock<Mutex<std::collections::HashSet<Key>>> = OnceLock::new();
    M.get_or_init(Default::default)
}

pub(crate) fn note_sys_bank(key: Key) {
    sys_banks().lock().unwrap().insert(key);
}

pub(crate) fn is_sys_bank(key: Key) -> bool {
    sys_banks().lock().unwrap().contains(&key)
}

/// The layer's cache of `cap` slots over its system-memory bank (`ne`
/// entries of `gu_len` gate / up and `d_len` down bytes, `slots` picks a
/// token), made on first use; None when the bank is not in system memory,
/// an entry is not whole words, or the card refuses.
#[allow(clippy::too_many_arguments)]
pub(crate) fn get(
    c: &Ctx,
    key: Key,
    cap: usize,
    ne: usize,
    slots: usize,
    gu_len: usize,
    d_len: usize,
) -> Option<Arc<Cache>> {
    if let Some(x) = caches().lock().unwrap().get(&key) {
        return Some(x.clone());
    }
    if cap < slots || ne == 0 || slots == 0 || slots > 64 || !is_sys_bank(key) || pipes(c).is_none() {
        return None;
    }
    if refused().lock().unwrap().contains(&key) {
        return None;
    }
    let want = cap.min(ne);
    if gu_len == 0 || d_len == 0 || gu_len % 4 != 0 || d_len % 4 != 0 {
        return None;
    }
    let max_bind = c.device.limits().max_storage_buffer_binding_size as u64;
    if (want * gu_len) as u64 > max_bind || (want * d_len) as u64 > max_bind {
        return None;
    }
    let mk = |label: &'static str, size: usize| {
        c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    };
    // a card that refuses the planned slots (its other allocations past the
    // plan's count) takes three quarters of them, down to two tokens' picks;
    // a refusal is kept, so a layer does not ask again every call (each ask
    // cost ~20 ms: Hy4-flash on a whole A40, 4 tokens/s against 12)
    let mut cap = want;
    let (g, u, d) = loop {
        let scope = c.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let g = mk("moe-cache-g", cap * gu_len);
        let u = mk("moe-cache-u", cap * gu_len);
        let d = mk("moe-cache-d", cap * d_len);
        match pollster::block_on(scope.pop()) {
            None => break (g, u, d),
            Some(e) => {
                drop((g, u, d));
                let next = cap * 3 / 4;
                if next < 2 * slots {
                    tracing::warn!("wgpu: expert cache of {want} slots refused ({e}), {cap} too; the layer reads its bank in place");
                    refused().lock().unwrap().insert(key);
                    return None;
                }
                cap = next;
            }
        }
    };
    if cap < want {
        tracing::warn!("wgpu: expert cache of {want} slots refused by the card; {cap} slots instead");
    }
    let total = (cap * (2 * gu_len + d_len)) as u64;
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let mut st = vec![0u32; 1 + ne + 2 * cap];
    st[1..1 + ne + cap].fill(NONE);
    let st = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("moe-cache-st"),
        contents: bytemuck::cast_slice(&st),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out = mk("moe-cache-out", 3 * 64 * 4);
    let ind = c.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("moe-cache-ind"),
        contents: bytemuck::cast_slice(&[0u32; 4]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
    });
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("wgpu: expert cache of {cap} slots refused ({e}); the layer reads its bank in place");
        refused().lock().unwrap().insert(key);
        return None;
    }
    let (gu_words, dn_words) = ((gu_len / 4) as u32, (d_len / 4) as u32);
    let chunks = gu_words.max(dn_words).div_ceil(CHUNK_WORDS);
    let fill_u = uniform_u32x4(c, [slots as u32, ne as u32, cap as u32, chunks]);
    let copy_u = uniform_u32x4(c, [gu_words, dn_words, slots as u32, CHUNK_WORDS]);
    c.resident.fetch_add(total, std::sync::atomic::Ordering::Relaxed);
    let x = Arc::new(Cache { g, u, d, st, out, ind, fill_u, copy_u, cap });
    let mut m = caches().lock().unwrap();
    {
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::info!(
                "wgpu: expert cache: {cap} of {ne} experts a layer on the card ({} MB) over the system-memory bank, LRU on the device",
                total >> 20
            );
        }
    }
    Some(m.entry(key).or_insert(x).clone())
}

/// The spill past which the banks go under slots: more than 1/n of the
/// experts (`CMF_WGPU_EXPERT_SPILL`, default 8 — the CUDA stream's rule).
fn spill_div() -> u64 {
    std::env::var("CMF_WGPU_EXPERT_SPILL").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(8).max(1)
}

/// The card's staging bank for a wide batch over a system-memory bank: the
/// layer's gate / up / down stacks are copied in whole (one DMA read over
/// the bus a chunk) and the batch kernels read them on the card. The batch
/// kernels walk an expert's rows once per token group; over the bus that
/// re-read cost Ornith-122B q4tp about a second a layer for 512 positions.
/// One bank for every layer (they run in order), grown to the largest.
pub(crate) fn staging(c: &Ctx, sizes: [u64; 3]) -> Option<Arc<[wgpu::Buffer; 3]>> {
    static S: OnceLock<Mutex<Option<([u64; 3], Arc<[wgpu::Buffer; 3]>)>>> = OnceLock::new();
    let m = S.get_or_init(Default::default);
    let mut g = m.lock().unwrap();
    if let Some((have, b)) = g.as_ref() {
        if (0..3).all(|i| have[i] >= sizes[i]) {
            return Some(b.clone());
        }
    }
    let want = match g.as_ref() {
        Some((have, _)) => [have[0].max(sizes[0]), have[1].max(sizes[1]), have[2].max(sizes[2])],
        None => sizes,
    };
    let max_bind = c.device.limits().max_storage_buffer_binding_size as u64;
    if !stage_on() || want.iter().any(|&w| w == 0 || w > max_bind || w % 4 != 0) {
        return None;
    }
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    // COPY_SRC: a streamed layer's pools are seeded from it (`moe_stream::Stream::warm`)
    let mk = |label: &'static str, size: u64| {
        c.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };
    let b = Arc::new([mk("moe-stage-g", want[0]), mk("moe-stage-u", want[1]), mk("moe-stage-d", want[2])]);
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("wgpu: expert staging bank refused ({e}); the batch reads system-memory banks in place");
        return None;
    }
    let old: u64 = g.as_ref().map_or(0, |(h, _)| h.iter().sum());
    let new: u64 = want.iter().sum();
    c.resident.fetch_add(new.saturating_sub(old), std::sync::atomic::Ordering::Relaxed);
    if old == 0 {
        tracing::info!("wgpu: expert staging bank: {} MB on the card for prompts over system-memory banks", new >> 20);
    }
    *g = Some((want, b.clone()));
    Some(b)
}

/// What a stack's routed experts take when they do not all fit the card.
pub(crate) struct Plan {
    /// The layer's bank goes to system memory.
    pub sys: Vec<bool>,
    /// Card slots over that bank (0: read in place).
    pub cap: Vec<usize>,
    /// Card bytes of those slots.
    pub cache_bytes: Vec<u64>,
    /// Card bytes held for the staging bank (`staging`): the largest bank,
    /// when any is in system memory.
    pub stage_bytes: u64,
    /// Every routed layer streams its experts through card slots
    /// (`moe_stream`): neither the card nor the host's room holds them.
    pub stream: bool,
    /// The slots' card bytes then.
    pub pool_bytes: u64,
}

/// The placement of `layers`' routed experts within `exp_budget` card bytes
/// (the live budget less every layer's dense part and the head) and the
/// host's room for system-memory banks. As few layers' banks as needed go
/// to system memory, from the end, when the caches are off; with them on,
/// as many as the room holds, and the card's bytes left after the layers
/// that stay whole become LRU slots over those banks, spread by bank size:
/// an LRU of a fraction f of a layer's experts misses at most as often as
/// reading a fraction f of the layers in place, and less whenever the
/// routing comes back to an expert.
pub(crate) fn plan(routed: &[(u64, usize, usize)], exp_budget: u64, room: u64, cache: bool, stage_on: bool, stream: bool) -> Plan {
    plan_with(routed, exp_budget, room, room, cache, stage_on, stream)
}

/// [`plan`] with the host's whole room apart from the ¾ the banks plan
/// with: the experts stream only when the spill is past all of it — banks in
/// place read RAM, a stream's misses and staging read the file (Ornith q4tp
/// within 24 GB on a 46 GB container: 9.4 tokens/s and the first token in
/// 6.7 s in place, 9.6 and 84 s streamed). The room is the banks' (see
/// `bank_room_bytes`): where it holds every bank, all of them go to system
/// memory and the card's expert bytes all become slots (the same card:
/// 30.6 tokens/s against 8.7 with 15 layers whole and no slots).
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_with(routed: &[(u64, usize, usize)], exp_budget: u64, room: u64, hard_room: u64, cache: bool, stage_on: bool, stream: bool) -> Plan {
    let n = routed.len();
    let mut sys = vec![false; n];
    let mut cap = vec![0usize; n];
    let mut cache_bytes = vec![0u64; n];
    let total: u64 = routed.iter().map(|r| r.0).sum();
    if total <= exp_budget {
        return Plan { sys, cap, cache_bytes, stage_bytes: 0, stream: false, pool_bytes: 0 };
    }
    // a prompt copies one bank at a time to the card: its room comes first
    let stage = if stage_on { routed.iter().map(|r| r.0).max().unwrap_or(0) } else { 0 };
    let exp_budget = exp_budget.saturating_sub(stage);
    // past the card and the host's room: every routed layer through card
    // slots fed from the file, when they hold a fifth of the experts (a
    // smaller pool misses more than the in-place banks cost: CUDA's stream
    // within 12 / 16 GB of Ornith-122B, 5.7 / 7.0 tokens/s against 7.9 / 8.3)
    if stream && total.saturating_sub(exp_budget) > hard_room && exp_budget.saturating_mul(5) >= total {
        return Plan { sys, cap, cache_bytes, stage_bytes: stage, stream: true, pool_bytes: exp_budget / 20 * 19 };
    }
    // the fewest trailing banks out of the card
    let mut on_card = total;
    let mut k_min = 0usize;
    let moe: Vec<usize> = (0..n).filter(|&i| routed[i].0 > 0).collect();
    for &i in moe.iter().rev() {
        if on_card <= exp_budget {
            break;
        }
        on_card -= routed[i].0;
        k_min += 1;
    }
    // the most trailing banks the host holds
    let (mut held, mut k_max) = (0u64, 0usize);
    for &i in moe.iter().rev() {
        if held + routed[i].0 > room {
            break;
        }
        held += routed[i].0;
        k_max += 1;
    }
    // a small spill stays read in place: the slots' fill and copy cost a
    // few microseconds a layer, more than the few experts the bus carries
    // then (Ornith q2tp-v2 on the whole A40, 3 of 48 banks out: 28.4
    // tokens/s in place, 25.6 with 43 banks under slots)
    let spill: u64 = moe.iter().rev().take(k_min).map(|&i| routed[i].0).sum();
    let wide = spill.saturating_mul(spill_div()) > total;
    let k = if cache && wide && k_max > k_min { k_max } else { k_min };
    let out: Vec<usize> = moe.iter().rev().take(k).copied().collect();
    for &i in &out {
        sys[i] = true;
    }
    let stage_bytes = if k > 0 { stage } else { 0 };
    if !cache {
        return Plan { sys, cap, cache_bytes, stage_bytes, stream: false, pool_bytes: 0 };
    }
    let resident: u64 = moe.iter().filter(|&&i| !sys[i]).map(|&i| routed[i].0).sum();
    let left = exp_budget.saturating_sub(resident) / 20 * 19;
    let banked: u64 = out.iter().map(|&i| routed[i].0).sum();
    if banked == 0 {
        return Plan { sys, cap, cache_bytes, stage_bytes, stream: false, pool_bytes: 0 };
    }
    for &i in &out {
        let (bytes, ne, slots) = routed[i];
        if ne == 0 {
            continue;
        }
        let per = bytes / ne as u64;
        let share = (left as u128 * bytes as u128 / banked as u128) as u64;
        let c = ((share / per.max(1)) as usize).min(ne);
        // fewer than two tokens' picks: nothing would stay
        if c >= 2 * slots {
            cap[i] = c;
            cache_bytes[i] = c as u64 * per;
        }
    }
    Plan { sys, cap, cache_bytes, stage_bytes, stream: false, pool_bytes: 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_fills_the_card_then_banks_the_rest() {
        // 4 layers of 10 GB experts (100 a layer, 9 picks), 25 GB for experts
        let r = vec![(10u64 << 30, 100, 9); 4];
        let p = plan(&r, 25 << 30, 1 << 40, false, false, false);
        assert_eq!(p.sys, vec![false, false, true, true]);
        assert!(p.cap.iter().all(|&c| c == 0));
        // with the caches: every bank the host holds, the card's bytes as slots
        let p = plan(&r, 25 << 30, 1 << 40, true, false, false);
        assert_eq!(p.sys, vec![true; 4]);
        // 25 GB * 0.95 over 40 GB of banks: 59 of 100 experts a layer
        assert!(p.cap.iter().all(|&c| c == 59), "{:?}", p.cap);
        // a host that holds three banks: one layer stays whole
        let p = plan(&r, 25 << 30, 31 << 30, true, false, false);
        assert_eq!(p.sys, vec![false, true, true, true]);
        assert!(p.cap[1..].iter().all(|&c| c == 47), "{:?}", p.cap);
        // everything fits: nothing moves
        let p = plan(&r, 40 << 30, 1 << 40, true, false, false);
        assert!(p.sys.iter().all(|&s| !s));
        // a spill of an eighth or less stays read in place
        let r16 = vec![(1u64 << 30, 100, 9); 16];
        let p = plan(&r16, 14 << 30, 1 << 40, true, false, false);
        assert_eq!(p.sys.iter().filter(|&&s| s).count(), 2);
        assert!(p.cap.iter().all(|&c| c == 0), "{:?}", p.cap);
        let p = plan(&r16, 13 << 30, 1 << 40, true, false, false);
        assert_eq!(p.sys.iter().filter(|&&s| s).count(), 16);
        // dense layers keep no bank
        let r2 = vec![(0u64, 0, 0), (10 << 30, 100, 9), (10 << 30, 100, 9)];
        let p = plan(&r2, 5 << 30, 1 << 40, true, false, false);
        assert_eq!(p.sys, vec![false, true, true]);
        assert!(p.cap[1] == 23 && p.cap[2] == 23, "{:?}", p.cap);
        // the staging bank's room comes off the card first
        let p = plan(&r, 25 << 30, 1 << 40, true, true, false);
        assert_eq!(p.stage_bytes, 10 << 30);
        assert_eq!(p.sys, vec![true; 4]);
        // 15 GB * 0.95 over 40 GB of banks: 35 of 100 a layer
        assert!(p.cap.iter().all(|&c| c == 35), "{:?}", p.cap);
        // too little left for two tokens' picks: the banks read in place
        let p = plan(&r, 25 << 30, 31 << 30, true, true, false);
        assert_eq!(p.sys, vec![false, true, true, true]);
        assert!(p.cap.iter().all(|&c| c == 0), "{:?}", p.cap);
        let p = plan(&r, 40 << 30, 1 << 40, true, true, false);
        assert_eq!(p.stage_bytes, 0);
        // past the card and the host: every layer streams through slots
        let p = plan(&r, 25 << 30, 4 << 30, true, true, true);
        assert!(p.stream && p.sys.iter().all(|&s| !s));
        assert_eq!(p.pool_bytes, ((15u64 << 30) / 20) * 19);
        // a pool under a fifth of the experts: the banks in place instead
        let p = plan(&r, 15 << 30, 4 << 30, true, true, true);
        assert!(!p.stream);
    }

    #[test]
    fn cache_wgsl_validates() {
        let module = wgpu::naga::front::wgsl::parse_str(CACHE_WGSL).expect("moe cache WGSL must parse");
        wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), wgpu::naga::valid::Capabilities::all())
            .validate(&module)
            .expect("moe cache WGSL must validate");
    }
}
