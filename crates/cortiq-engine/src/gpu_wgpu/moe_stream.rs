//! Routed experts streamed through card slots (wgpu), for a stack whose
//! experts fit neither the card nor the host's memory beside it.
//!
//! Pools of expert slots in card memory serve the MoE layers of a model (a
//! group of layers a pool: one storage binding holds ~2 GB, so a pool's three
//! buffers stay under it), the least recently used giving way, never one of
//! the current layer's picks. Below them a RAM tier in GPU-visible system
//! memory keeps raw experts (the file's own layout, which the kernels read)
//! with its own LRU: a miss of the card's slots is a DMA copy from it, and a
//! miss of both a copy from the file's mapping into it first. The token graph
//! runs a streamed layer's router and select, ends its submission there and
//! reads the picks back; the host settles their slots (copies encoded ahead
//! of the expert kernels) and writes their slot numbers into the selection
//! the kernels read — the CUDA hybrid arm's expert stream without its repack.

use super::*;
use std::sync::{Arc, Mutex, OnceLock};

/// Whether a stack past the card and the host's room streams its experts
/// (`CMF_WGPU_EXPERT_STREAM=0`: the banks read in place where they fit).
pub(crate) fn on() -> bool {
    std::env::var("CMF_WGPU_EXPERT_STREAM").as_deref() != Ok("0")
}

/// `CMF_WGPU_EXPERT_STREAM=1`: stream whenever the experts pass the card.
pub(crate) fn forced() -> bool {
    std::env::var("CMF_WGPU_EXPERT_STREAM").as_deref() == Ok("1")
}

/// Without a RAM tier the misses' pages are faulted in on the fill threads
/// before their copies (`CMF_WGPU_EXPERT_FAULT_IN=0`: the copies alone).
fn fault_in_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_WGPU_EXPERT_FAULT_IN").as_deref() != Ok("0"))
}

/// `CMF_STREAM_STATS=1`: the stream's counters every 1024 layer calls.
fn stats_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CMF_STREAM_STATS").is_some())
}

/// The host threads that copy a call's missing experts into the RAM tier
/// together (`CMF_WGPU_EXPERT_FILL_THREADS`, default the cores up to eight):
/// a copy is a read of a few MB from the mapping, from the disk when the file
/// is past the host's memory, and one thread's page faults leave the disk's
/// queue nearly empty.
fn fill_pool() -> &'static crate::pool::Pool {
    static P: OnceLock<crate::pool::Pool> = OnceLock::new();
    P.get_or_init(|| {
        let n = std::env::var("CMF_WGPU_EXPERT_FILL_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()).min(8))
            .max(1);
        crate::pool::Pool::with_spin(n.saturating_sub(1), 0)
    })
}

/// `MADV_WILLNEED` over experts' three tensors: the kernel's read-ahead
/// scheduled, no wait (`CMF_WGPU_EXPERT_WILLNEED=0` off).
pub(crate) fn will_need(model: &CmfModel, experts: impl IntoIterator<Item = (usize, usize, usize)>) {
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
    {
        static ON: OnceLock<bool> = OnceLock::new();
        if !*ON.get_or_init(|| std::env::var("CMF_WGPU_EXPERT_WILLNEED").as_deref() != Ok("0")) {
            return;
        }
        let page = if cfg!(target_os = "macos") { 16384usize } else { 4096 };
        for (g, u, d) in experts {
            for idx in [g, u, d] {
                let Some(entry) = model.tensors.get(idx) else { continue };
                let b = model.entry_bytes(entry);
                if b.is_empty() {
                    continue;
                }
                let p = b.as_ptr() as usize;
                let s = p & !(page - 1);
                // SAFETY: a range inside the model's read-only mapping;
                // MADV_WILLNEED only schedules read-ahead.
                unsafe {
                    libc::madvise(s as *mut libc::c_void, p + b.len() - s, libc::MADV_WILLNEED);
                }
            }
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
    let _ = (model, experts.into_iter().count());
}

/// Fault experts' pages in on `fill_pool`'s threads, a read a page: with no
/// RAM tier the misses' copies come straight from the mapping, and one
/// thread's page faults read a 1.6 MB Maxwell expert from the disk in 4 ms
/// (an M4's shared memory, the file past what the page cache keeps).
fn fault_in(model: &CmfModel, experts: &[(usize, usize, usize)]) {
    const PAGE: usize = 16384;
    let mut spans: Vec<&[u8]> = Vec::with_capacity(experts.len() * 3);
    for &(g, u, d) in experts {
        for idx in [g, u, d] {
            if let Some(te) = model.tensors.get(idx) {
                let b = model.entry_bytes(te);
                if !b.is_empty() {
                    spans.push(b);
                }
            }
        }
    }
    if spans.is_empty() {
        return;
    }
    // pages of every span in one list, split across the threads
    let mut pages: Vec<(usize, usize)> = Vec::new();
    for (i, b) in spans.iter().enumerate() {
        let mut o = 0usize;
        while o < b.len() {
            pages.push((i, o));
            o += PAGE;
        }
    }
    let touch = |a: usize, z: usize| {
        let mut acc = 0u8;
        for &(i, o) in &pages[a..z] {
            // SAFETY: an offset inside the span (o < len), a read of the
            // model's read-only mapping
            acc ^= unsafe { std::ptr::read_volatile(spans[i].as_ptr().add(o)) };
        }
        std::hint::black_box(acc);
    };
    fill_pool().run_rows(pages.len(), &touch);
}

/// One pool of card slots: gate, up and down planes of `n` experts.
pub(crate) struct Seg {
    pub g: wgpu::Buffer,
    pub u: wgpu::Buffer,
    pub d: wgpu::Buffer,
    owner: Vec<Option<(usize, usize)>>,
    last: Vec<u64>,
    at: HashMap<(usize, usize), usize>,
}

/// Raw experts in GPU-visible system memory (`host_mem`), a slot each.
struct Tier {
    bufs: Vec<host_mem::SysBuf>,
    per_buf: usize,
    slot: usize,
    owner: Vec<Option<(usize, usize)>>,
    last: Vec<u64>,
    at: HashMap<(usize, usize), usize>,
}

impl Tier {
    fn place(&self, t: usize) -> (&host_mem::SysBuf, u64) {
        (&self.bufs[t / self.per_buf], ((t % self.per_buf) * self.slot) as u64)
    }
}

/// A model's stream: its pools, the RAM tier, the selection the expert
/// kernels read and the readback of the picks.
pub(crate) struct Stream {
    pub segs: Vec<Seg>,
    tier: Option<Tier>,
    pub sel: wgpu::Buffer,
    pub readback: wgpu::Buffer,
    warm_rb: wgpu::Buffer,
    gu_len: usize,
    d_len: usize,
    tick: u64,
    pub picks: u64,
    pub loads: u64,
    pub fills: u64,
    pub load_ns: u64,
    /// The token graph's waits for the picks (the submission's GPU work and
    /// the round trip), `CMF_STREAM_STATS=1`.
    pub wait_ns: u64,
    pub waits: u64,
    /// `CMF_STREAM_STATS=2`: empty submissions' round trips.
    pub rt_ns: u64,
    /// Host time from the picks' arrival to the next streamed submission
    /// (the GPU idles through it), and when the picks last arrived.
    pub gap_ns: u64,
    pub resumed: Option<std::time::Instant>,
    /// The finish and submit of a streamed submission (host time).
    pub sub_ns: u64,
}

impl Stream {
    fn print_stats(&self) {
        eprintln!(
            "wgpu expert stream: {} picks, {} loaded ({:.1}% hits), {} RAM tier fills ({} slots), {:.2} ms a load, {:.3} ms a wait for the picks ({} waits), {:.3} ms an empty round trip, {:.3} ms a gap to the next submission, {:.3} ms its finish and submit",
            self.picks,
            self.loads,
            100.0 * (1.0 - self.loads as f64 / self.picks.max(1) as f64),
            self.fills,
            self.tier_slots(),
            self.load_ns as f64 / 1e6 / self.loads.max(1) as f64,
            self.wait_ns as f64 / 1e6 / self.waits.max(1) as f64,
            self.waits,
            self.rt_ns as f64 / 1e6 / self.waits.max(1) as f64,
            self.gap_ns as f64 / 1e6 / self.waits.max(1) as f64,
            self.sub_ns as f64 / 1e6 / self.waits.max(1) as f64
        );
    }

    /// Expert `e` of layer `lk` in the RAM tier: its slot there (copied from
    /// the mapping when absent). None: no tier, or every slot is held this tick.
    fn tier_get(&mut self, model: &CmfModel, lk: usize, e: usize, ids: (usize, usize, usize)) -> Option<usize> {
        let tick = self.tick;
        let (gu, dl) = (self.gu_len, self.d_len);
        let tier = self.tier.as_mut()?;
        if let Some(&t) = tier.at.get(&(lk, e)) {
            tier.last[t] = tick;
            return Some(t);
        }
        let t = (0..tier.owner.len()).filter(|&t| tier.last[t] != tick).min_by_key(|&t| (tier.owner[t].is_some(), tier.last[t]))?;
        if let Some(old) = tier.owner[t].take() {
            tier.at.remove(&old);
        }
        let src = |idx: usize, len: usize| -> Option<&[u8]> {
            let te = model.tensors.get(idx)?;
            let b = model.entry_bytes(te);
            (b.len() >= len).then(|| &b[..len])
        };
        let (buf, off) = tier.place(t);
        if !(buf.write(off, src(ids.0, gu)?) && buf.write(off + gu as u64, src(ids.1, gu)?) && buf.write(off + 2 * gu as u64, src(ids.2, dl)?)) {
            return None;
        }
        tier.owner[t] = Some((lk, e));
        tier.at.insert((lk, e), t);
        tier.last[t] = tick;
        self.fills += 1;
        Some(t)
    }

    /// Experts `want` of layer `lk` into the RAM tier at once: their tensors
    /// read ahead, slots taken (never this tick's), the copies from the
    /// mapping on `fill_pool`'s threads; at most half the tier a call.
    fn tier_fill(&mut self, model: &CmfModel, lk: usize, want: &[(usize, (usize, usize, usize))]) {
        let tick = self.tick;
        let (gu, dl) = (self.gu_len, self.d_len);
        let Some(tier) = self.tier.as_mut() else { return };
        if 2 * gu + dl > tier.slot {
            return;
        }
        let src = |idx: usize, len: usize| -> Option<&[u8]> {
            let te = model.tensors.get(idx)?;
            let b = model.entry_bytes(te);
            (b.len() >= len).then(|| &b[..len])
        };
        let cap = tier.owner.len() / 2;
        let mut jobs: Vec<(usize, [&[u8]; 3])> = Vec::with_capacity(want.len().min(cap));
        let mut ahead = Vec::new();
        for &(e, ids) in want {
            if jobs.len() >= cap {
                break;
            }
            if let Some(&t) = tier.at.get(&(lk, e)) {
                tier.last[t] = tick;
                continue;
            }
            let (Some(a), Some(b), Some(d)) = (src(ids.0, gu), src(ids.1, gu), src(ids.2, dl)) else { continue };
            let Some(t) = (0..tier.owner.len()).filter(|&t| tier.last[t] != tick).min_by_key(|&t| (tier.owner[t].is_some(), tier.last[t])) else {
                break;
            };
            if let Some(old) = tier.owner[t].take() {
                tier.at.remove(&old);
            }
            tier.owner[t] = Some((lk, e));
            tier.at.insert((lk, e), t);
            tier.last[t] = tick;
            jobs.push((t, [a, b, d]));
            ahead.push(ids);
        }
        if jobs.is_empty() {
            return;
        }
        self.fills += jobs.len() as u64;
        will_need(model, ahead);
        let tier = &*tier;
        let copy = |i: usize| {
            let (t, [a, b, d]) = jobs[i];
            let (buf, off) = tier.place(t);
            // the slot was taken for this copy; its bounds hold (2·gu + dl ≤ slot)
            let _ = buf.write(off, a) && buf.write(off + gu as u64, b) && buf.write(off + 2 * gu as u64, d);
        };
        if jobs.len() == 1 {
            copy(0);
        } else {
            fill_pool().run_rows(jobs.len(), &|a, b| (a..b).for_each(copy));
        }
    }

    /// Copy expert `e` (RAM tier first, else the mapping) into `dst` gate /
    /// up / down planes at `(off_gu, off_d)`: GPU copies in `enc`, or queue
    /// writes without a tier.
    #[allow(clippy::too_many_arguments)]
    fn bring(
        &mut self,
        c: &Ctx,
        enc: &mut wgpu::CommandEncoder,
        model: &CmfModel,
        lk: usize,
        e: usize,
        ids: (usize, usize, usize),
        dst: [&wgpu::Buffer; 3],
        off_gu: u64,
        off_d: u64,
    ) -> Option<()> {
        let (gu, dl) = (self.gu_len as u64, self.d_len as u64);
        if let Some(t) = self.tier_get(model, lk, e, ids) {
            let tier = self.tier.as_ref()?;
            let (buf, off) = tier.place(t);
            enc.copy_buffer_to_buffer(&buf.buffer, off, dst[0], off_gu, gu);
            enc.copy_buffer_to_buffer(&buf.buffer, off + gu, dst[1], off_gu, gu);
            enc.copy_buffer_to_buffer(&buf.buffer, off + 2 * gu, dst[2], off_d, dl);
            return Some(());
        }
        let src = |idx: usize, len: u64| -> Option<&[u8]> {
            let te = model.tensors.get(idx)?;
            let b = model.entry_bytes(te);
            (b.len() as u64 >= len).then(|| &b[..len as usize])
        };
        c.queue.write_buffer(dst[0], off_gu, src(ids.0, gu)?);
        c.queue.write_buffer(dst[1], off_gu, src(ids.1, gu)?);
        c.queue.write_buffer(dst[2], off_d, src(ids.2, dl)?);
        Some(())
    }

    /// The slots in pool `seg` of a layer's picks (layer key `lk`, its
    /// experts' tensors), the missing ones' copies encoded into `enc`.
    #[allow(clippy::too_many_arguments)]
    pub fn ensure(
        &mut self,
        c: &Ctx,
        enc: &mut wgpu::CommandEncoder,
        model: &CmfModel,
        seg: usize,
        lk: usize,
        experts: &[(usize, usize, usize)],
        picks: &[u32],
    ) -> Option<Vec<u32>> {
        self.tick += 1;
        let tick = self.tick;
        let mut out = vec![0u32; picks.len()];
        let mut miss = Vec::new();
        {
            let sg = self.segs.get_mut(seg)?;
            for (i, &e) in picks.iter().enumerate() {
                match sg.at.get(&(lk, e as usize)) {
                    Some(&s) => {
                        sg.last[s] = tick;
                        out[i] = s as u32;
                    }
                    None => miss.push(i),
                }
            }
        }
        self.picks += picks.len() as u64;
        if stats_on() && self.tick % 1024 == 0 {
            self.print_stats();
        }
        if miss.is_empty() {
            return Some(out);
        }
        let t0 = std::time::Instant::now();
        crate::gpu_wgpu::flush_pass(&*enc);
        // the misses into the RAM tier together first (the copies below then
        // find them there); without one, their pages faulted in together
        if miss.len() > 1 && self.tier.is_some() {
            let want: Vec<(usize, (usize, usize, usize))> =
                miss.iter().filter_map(|&i| experts.get(picks[i] as usize).map(|&ids| (picks[i] as usize, ids))).collect();
            self.tier_fill(model, lk, &want);
        } else if self.tier.is_none() && fault_in_on() {
            let want: Vec<(usize, usize, usize)> = miss.iter().filter_map(|&i| experts.get(picks[i] as usize).copied()).collect();
            will_need(model, want.iter().copied());
            fault_in(model, &want);
        }
        for i in miss {
            let e = picks[i] as usize;
            let ids = *experts.get(e)?;
            let (s, g, u, d) = {
                let sg = self.segs.get_mut(seg)?;
                if let Some(&s) = sg.at.get(&(lk, e)) {
                    out[i] = s as u32;
                    continue;
                }
                let s = (0..sg.owner.len()).filter(|&s| sg.last[s] != tick).min_by_key(|&s| (sg.owner[s].is_some(), sg.last[s]))?;
                if let Some(old) = sg.owner[s].take() {
                    sg.at.remove(&old);
                }
                sg.owner[s] = Some((lk, e));
                sg.at.insert((lk, e), s);
                sg.last[s] = tick;
                (s, sg.g.clone(), sg.u.clone(), sg.d.clone())
            };
            let (gu, dl) = (self.gu_len as u64, self.d_len as u64);
            self.bring(c, enc, model, lk, e, ids, [&g, &u, &d], s as u64 * gu, s as u64 * dl)?;
            self.loads += 1;
            out[i] = s as u32;
        }
        self.load_ns += t0.elapsed().as_nanos() as u64;
        Some(out)
    }

    /// Every expert of a layer into a staging bank of the layer's layout (a
    /// prompt's wide chunk), copies encoded into `enc`: the layer's experts
    /// into the RAM tier on the fill threads first, then the next streamed
    /// layer's (`next`) read ahead while the card runs this one.
    #[allow(clippy::too_many_arguments)]
    pub fn stage(
        &mut self,
        c: &Ctx,
        enc: &mut wgpu::CommandEncoder,
        model: &CmfModel,
        lk: usize,
        experts: &[(usize, usize, usize)],
        dst: [&wgpu::Buffer; 3],
        next: Option<&[(usize, usize, usize)]>,
    ) -> Option<()> {
        self.tick += 1;
        crate::gpu_wgpu::flush_pass(&*enc);
        let want: Vec<(usize, (usize, usize, usize))> = experts.iter().copied().enumerate().collect();
        self.tier_fill(model, lk, &want);
        if let Some(n) = next {
            will_need(model, n.iter().copied());
        }
        let (gu, dl) = (self.gu_len as u64, self.d_len as u64);
        for (e, &ids) in experts.iter().enumerate() {
            self.bring(c, enc, model, lk, e, ids, dst, e as u64 * gu, e as u64 * dl)?;
        }
        Some(())
    }

    /// A staged layer's pools seeded from its staging bank: the experts the
    /// chunk's last rows picked (`n` picks of `msel` from `off`, the last
    /// row's first), so the decode after a prompt starts on warm slots
    /// instead of loading every pick of its first tokens. The picks are read
    /// back after the work so far; the copies go into the next submission
    /// (`enc`), before anything rewrites the staging bank; at most the
    /// layer's share of its pool (`CMF_WGPU_EXPERT_WARM=0` off).
    #[allow(clippy::too_many_arguments)]
    pub fn warm(&mut self, c: &Ctx, enc: &mut wgpu::CommandEncoder, msel: &wgpu::Buffer, off: usize, n: usize, lk: usize, seg: usize, src: [&wgpu::Buffer; 3], n_bank: usize, n_moe: usize) -> Option<()> {
        if n == 0 || n * 4 > self.warm_rb.size() as usize || std::env::var("CMF_WGPU_EXPERT_WARM").as_deref() == Ok("0") {
            return None;
        }
        crate::gpu_wgpu::flush_pass(&*enc);
        enc.copy_buffer_to_buffer(msel, (off * 4) as u64, &self.warm_rb, 0, (n * 4) as u64);
        let done = std::mem::replace(enc, c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("batch-graph") }));
        submit(c, finish_enc(done));
        let picks = read_picks(c, &self.warm_rb, n)?;
        self.tick += 1;
        let tick = self.tick;
        let (gu, dl) = (self.gu_len as u64, self.d_len as u64);
        // the layer's share of its pool and of the RAM tier
        let in_seg = (0..n_moe.max(1)).filter(|&o| self.seg_for(o, n_moe) == seg).count().max(1);
        let tier_share = self.tier.as_ref().map_or(0, |t| t.owner.len() / n_moe.max(1));
        let sg = self.segs.get_mut(seg)?;
        let cap = (sg.owner.len() / in_seg).max(1);
        // the picks, the last rows' first, once each
        let mut seen = std::collections::HashSet::new();
        let order: Vec<usize> = picks.iter().rev().map(|&e| e as usize).filter(|&e| e < n_bank && seen.insert(e)).collect();
        let mut loads = 0usize;
        for &e in &order {
            if let Some(&s) = sg.at.get(&(lk, e)) {
                sg.last[s] = tick;
                continue;
            }
            if loads >= cap {
                break;
            }
            let Some(s) = (0..sg.owner.len()).filter(|&s| sg.last[s] != tick).min_by_key(|&s| (sg.owner[s].is_some(), sg.last[s])) else {
                break;
            };
            if let Some(old) = sg.owner[s].take() {
                sg.at.remove(&old);
            }
            sg.owner[s] = Some((lk, e));
            sg.at.insert((lk, e), s);
            sg.last[s] = tick;
            let (e, s) = (e as u64, s as u64);
            enc.copy_buffer_to_buffer(src[0], e * gu, &sg.g, s * gu, gu);
            enc.copy_buffer_to_buffer(src[1], e * gu, &sg.u, s * gu, gu);
            enc.copy_buffer_to_buffer(src[2], e * dl, &sg.d, s * dl, dl);
            loads += 1;
        }
        // the staging copied the whole layer through the RAM tier: past the
        // pool's share and the tier's own, its experts give way first (the
        // tier otherwise kept the last layers only). Off by default: GLM-5.3
        // streamed on the A40 pod, decode after 512 tokens and its first
        // token: pools seeded 4.25 tok/s and 50.6 s, seeded with the tier's
        // demotion 3.87 and 40.3 s, unseeded 3.90 and 46.9 s;
        // `CMF_WGPU_EXPERT_WARM_TIER=1`.
        let demote = std::env::var("CMF_WGPU_EXPERT_WARM_TIER").as_deref() == Ok("1");
        if let Some(t) = self.tier.as_mut().filter(|_| demote) {
            let keep: std::collections::HashSet<usize> = order.iter().copied().take(cap + tier_share.max(16)).collect();
            for e in (0..n_bank).filter(|e| !keep.contains(e)) {
                if let Some(&s) = t.at.get(&(lk, e)) {
                    t.last[s] = 1;
                }
            }
        }
        Some(())
    }

    /// The pool of the `ord`-th of `n_moe` routed layers (spread in order).
    pub fn seg_for(&self, ord: usize, n_moe: usize) -> usize {
        let nseg = self.segs.len().max(1);
        (ord * nseg / n_moe.max(1)).min(nseg - 1)
    }

    pub fn tier_slots(&self) -> usize {
        self.tier.as_ref().map_or(0, |t| t.owner.len())
    }
}

type Shared = Arc<Mutex<Stream>>;

fn streams() -> &'static Mutex<HashMap<u64, Shared>> {
    static M: OnceLock<Mutex<HashMap<u64, Shared>>> = OnceLock::new();
    M.get_or_init(Default::default)
}

/// The model's stream, made on first use with `bytes` of card slots over
/// experts of `gu_len` gate / up and `d_len` down bytes, in pools whose
/// buffers stay within the binding limit, and a RAM tier of half the host's
/// room (`CMF_WGPU_EXPERT_TIER=0`: none, misses read from the mapping).
pub(crate) fn stream(c: &Ctx, uid: u64, gu_len: usize, d_len: usize, bytes: u64) -> Option<Shared> {
    if let Some(p) = streams().lock().unwrap().get(&uid) {
        // one slot size a model: a layer of other expert sizes has no stream
        let ok = p.lock().map(|s| s.gu_len == gu_len && s.d_len == d_len).unwrap_or(false);
        return ok.then(|| p.clone());
    }
    if gu_len == 0 || d_len == 0 || gu_len % 4 != 0 || d_len % 4 != 0 {
        return None;
    }
    let per = (2 * gu_len + d_len) as u64;
    let max_bind = (c.device.limits().max_storage_buffer_binding_size as u64).min(c.device.limits().max_buffer_size);
    let per_seg = ((max_bind / gu_len as u64).min(max_bind / d_len as u64)) as usize;
    let total = (bytes / per) as usize;
    if total < 64 || per_seg < 64 {
        return None;
    }
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let mk = |label: &'static str, size: usize, usage: wgpu::BufferUsages| {
        c.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size as u64, usage, mapped_at_creation: false })
    };
    let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let mut segs = Vec::new();
    let mut left = total;
    while left > 0 {
        let n = left.min(per_seg);
        segs.push(Seg {
            g: mk("moe-pool-g", n * gu_len, st),
            u: mk("moe-pool-u", n * gu_len, st),
            d: mk("moe-pool-d", n * d_len, st),
            owner: vec![None; n],
            last: vec![0; n],
            at: HashMap::new(),
        });
        left -= n;
    }
    let sel = mk("moe-pool-sel", 64 * 4, st);
    let readback = mk("moe-pool-rb", 64 * 4, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
    let warm_rb = mk("moe-pool-warm", 64 * 64 * 4, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("wgpu: expert stream pools of {total} slots refused ({e})");
        return None;
    }
    c.resident.fetch_add(total as u64 * per, std::sync::atomic::Ordering::Relaxed);
    // the RAM tier: half the host's room, in buffers within the size limit;
    // none on shared memory (the GPU's memory is the host's: a second copy)
    let tier = (c.discrete && std::env::var("CMF_WGPU_EXPERT_TIER").as_deref() != Ok("0")).then(|| {
        let slot = (per as usize).next_multiple_of(256);
        let per_buf = (c.device.limits().max_buffer_size / slot as u64) as usize;
        let want = (host_room_bytes() / 2) as usize / slot;
        let mut bufs = Vec::new();
        let mut got = 0usize;
        while got < want && per_buf > 0 {
            let n = (want - got).min(per_buf);
            match host_mem::sysmem_storage_buffer(&c.device, (n * slot) as u64) {
                Some(b) => {
                    bufs.push(b);
                    got += n;
                }
                None => break,
            }
            // a buffer of `per_buf` slots each, the last one too (its tail unused)
            if n < per_buf {
                break;
            }
        }
        let n = bufs.len() * per_buf.max(1);
        let n = n.min(got);
        (n >= 64).then(|| Tier { bufs, per_buf, slot, owner: vec![None; n], last: vec![0; n], at: HashMap::new() })
    }).flatten();
    tracing::info!(
        "wgpu: expert stream: {total} card slots ({} MB) in {} pools, a RAM tier of {} experts",
        (total as u64 * per) >> 20,
        segs.len(),
        tier.as_ref().map_or(0, |t| t.owner.len())
    );
    let s = Arc::new(Mutex::new(Stream {
        segs,
        tier,
        sel,
        readback,
        warm_rb,
        gu_len,
        d_len,
        tick: 0,
        picks: 0,
        loads: 0,
        fills: 0,
        load_ns: 0,
        wait_ns: 0,
        waits: 0,
        rt_ns: 0,
        gap_ns: 0,
        resumed: None,
        sub_ns: 0,
    }));
    streams().lock().unwrap().insert(uid, s.clone());
    Some(s)
}

/// Read `slots` u32 picks back from the readback buffer after the
/// submission that copied them there.
pub(crate) fn read_picks(c: &Ctx, rb: &wgpu::Buffer, slots: usize) -> Option<Vec<u32>> {
    let sz = (slots * 4) as u64;
    let slice = rb.slice(..sz);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = c.device.poll(wgpu::PollType::wait_indefinitely());
    if !rx.recv().map(|r| r.is_ok()).unwrap_or(false) {
        return None;
    }
    let v = {
        let Ok(data) = slice.get_mapped_range() else {
            rb.unmap();
            return None;
        };
        bytemuck::cast_slice::<u8, u32>(&data).to_vec()
    };
    rb.unmap();
    Some(v)
}
