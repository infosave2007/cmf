//! Host tiers of the Qwen3.8-Flash-Next expert cache.
//!
//! The device path keeps the hottest routed experts in a VRAM arena
//! (`QwenGpuPool`). Before this module everything else came from the
//! memory-mapped file: a miss was a run of page faults inside a memcpy, one
//! 4 KiB page at a time, and the page cache it filled kept copies of
//! experts that already sat in VRAM while it dropped the n-gram rows the
//! next token needed. On a host whose RAM does not hold the 77 GB file that
//! turned most misses into slow, serialised disk reads.
//!
//! Two tiers now sit behind the arena:
//!
//! * **RAM tier** — an explicit cache sized from the memory that is actually
//!   free (container limits included). It prefers experts the arena does
//!   NOT hold: copies of VRAM-resident experts are the first to be dropped,
//!   and an expert the arena evicts is read back in the background. When
//!   VRAM + RAM can hold every expert, a background loader fills the tier
//!   and after warm-up the disk is not touched at all.
//! * **File** — on Linux a non-blocking read first takes whatever the page
//!   cache already has; the rest is one `O_DIRECT` read per expert, issued
//!   from many threads at once, so a fast NVMe drive runs at its rated
//!   speed and the page cache is left to the n-gram rows. Elsewhere a plain
//!   positioned read (one call per expert, no page faults).
//!
//! Measured on a RunPod RTX 5090 (overlay filesystem, 62 GB memory cgroup)
//! the OS page cache, driven well, beats an explicit RAM tier there: it
//! already holds most of the file, and a tier filled with direct reads
//! competes with it for the same memory. Copying straight out of the
//! mapping is also the fastest hit (~20 GB/s from mapped resident pages;
//! a buffered `pread` plus a copy measured 7x slower per expert at a 12 GB
//! budget). So the default hands the arena slices of the mapping, and the
//! n-gram table gets random-access advice. `MADV_WILLNEED` ahead of each
//! copy (`CMF_QWEN_WILLNEED=1`) turns a miss into one asynchronous request
//! for the whole expert, but on resident pages its page-cache walk cost
//! more than it saved there (12 GB budget: 24.8 against 31.4 tok/s), so it
//! is opt-in. The explicit RAM tier, buffered `pread` and direct I/O stay
//! available for hosts where they measure better.
//!
//! Knobs: `CMF_QWEN_IO` (`mmap` default | `pread` | `direct`),
//! `CMF_QWEN_RAM_TIER_MB` (unset/`0` off, `auto` = what is free minus a
//! reserve, `N` MiB), `CMF_QWEN_TIER_THREADS` (background loaders,
//! default 2), `CMF_QWEN_CACHE_HINTS=1` (page-cache mode: release an
//! expert's file pages once it sits in VRAM, read it back ahead of time
//! when the arena evicts it).

use cortiq_core::CmfModel;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// Alignment of direct reads and of every tier slot.
const ALIGN: usize = 4096;
/// Gaps between an expert's three matrices up to this size are read along
/// with them (one request instead of three); larger spans are read per part.
const MAX_GAP: usize = 64 << 10;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Part {
    pub abs: u64,
    pub len: usize,
}

/// One expert's three matrices (gate, up, down) in the file.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExpertLoc {
    pub parts: [Part; 3],
}

impl ExpertLoc {
    fn total(&self) -> usize {
        self.parts.iter().map(|p| p.len).sum()
    }
    fn span(&self) -> (u64, u64) {
        let s = self.parts.iter().map(|p| p.abs).min().unwrap_or(0);
        let e = self
            .parts
            .iter()
            .map(|p| p.abs + p.len as u64)
            .max()
            .unwrap_or(0);
        (s, e)
    }
    /// One read covers the three parts when they sit (nearly) back to back.
    fn contiguous(&self) -> bool {
        let (s, e) = self.span();
        (e - s) as usize <= self.total() + MAX_GAP
    }
}

/// Where each part landed inside a buffer: (offset, len).
pub(crate) type Ranges = [(u32, u32); 3];

/// Bytes a buffer needs to receive any expert of this layout.
fn slot_bytes_for(max_total: usize) -> usize {
    (max_total + MAX_GAP + 4 * ALIGN).next_multiple_of(ALIGN)
}

/// A heap buffer aligned for direct I/O.
struct AlignedBuf {
    ptr: *mut u8,
    cap: usize,
}

impl AlignedBuf {
    const fn empty() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            cap: 0,
        }
    }
    fn ensure(&mut self, n: usize) -> &mut [u8] {
        if self.cap < n {
            self.free();
            let cap = n.next_multiple_of(ALIGN);
            let layout = std::alloc::Layout::from_size_align(cap, ALIGN).expect("layout");
            // SAFETY: non-zero size, valid alignment.
            self.ptr = unsafe { std::alloc::alloc(layout) };
            assert!(!self.ptr.is_null(), "expert scratch allocation failed");
            self.cap = cap;
        }
        // SAFETY: `ptr` holds `cap >= n` bytes owned by this buffer.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, n) }
    }
    fn free(&mut self) {
        if !self.ptr.is_null() {
            let layout = std::alloc::Layout::from_size_align(self.cap, ALIGN).expect("layout");
            // SAFETY: allocated above with this exact layout.
            unsafe { std::alloc::dealloc(self.ptr, layout) };
            self.ptr = std::ptr::null_mut();
            self.cap = 0;
        }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        self.free();
    }
}

thread_local! {
    static SCRATCH: RefCell<AlignedBuf> = const { RefCell::new(AlignedBuf::empty()) };
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum IoMode {
    /// copy out of the memory map (page faults on a miss)
    Mmap,
    /// one positioned read per expert through the page cache
    Pread,
    /// page cache when it has the bytes, else O_DIRECT (Linux)
    Direct,
}

/// Positioned read of the whole buffer; Ok(n) with n < len only at EOF.
fn pread_full(f: &std::fs::File, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
    let mut done = 0usize;
    while done < buf.len() {
        #[cfg(unix)]
        let n = std::os::unix::fs::FileExt::read_at(f, &mut buf[done..], off + done as u64);
        #[cfg(windows)]
        let n = std::os::windows::fs::FileExt::seek_read(f, &mut buf[done..], off + done as u64);
        match n {
            Ok(0) => break,
            Ok(n) => done += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(done)
}

/// Non-blocking read from the page cache: the bytes it could take without
/// waiting for the device (0 when nothing is cached).
#[cfg(target_os = "linux")]
fn pread_cached(f: &std::fs::File, buf: &mut [u8], off: u64) -> Result<usize, ()> {
    use std::os::unix::io::AsRawFd;
    let mut done = 0usize;
    while done < buf.len() {
        let iov = libc::iovec {
            iov_base: buf[done..].as_mut_ptr().cast(),
            iov_len: buf.len() - done,
        };
        // SAFETY: one iovec over a live, writable slice.
        let n = unsafe {
            libc::preadv2(
                f.as_raw_fd(),
                &iov,
                1,
                (off + done as u64) as libc::off_t,
                libc::RWF_NOWAIT,
            )
        };
        if n < 0 {
            let e = std::io::Error::last_os_error().raw_os_error();
            // EAGAIN: not cached; ENOTSUP/EOPNOTSUPP/EINVAL: the flag itself
            // is not supported here
            if matches!(e, Some(libc::EOPNOTSUPP) | Some(libc::EINVAL)) {
                return Err(());
            }
            break;
        }
        if n == 0 {
            break;
        }
        done += n as usize;
    }
    Ok(done)
}

pub(crate) struct FileReader {
    model: Arc<CmfModel>,
    mode: IoMode,
    file: Option<std::fs::File>,
    #[cfg(target_os = "linux")]
    direct: Option<std::fs::File>,
    /// O_DIRECT failed once (filesystem without support): buffered from then on
    direct_broken: AtomicBool,
    /// RWF_NOWAIT works on this filesystem (overlayfs answers ENOTSUP):
    /// without it, page residency is asked of the mapping (mincore)
    nowait_ok: AtomicBool,
}

impl FileReader {
    pub(crate) fn open(model: &Arc<CmfModel>) -> Self {
        let file = std::fs::File::open(&model.path).ok();
        #[cfg(target_os = "linux")]
        let direct = {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECT)
                .open(&model.path)
                .ok()
        };
        let want = std::env::var("CMF_QWEN_IO").unwrap_or_default();
        #[cfg(target_os = "linux")]
        let direct_ok = direct.is_some();
        #[cfg(not(target_os = "linux"))]
        let direct_ok = false;
        let mode = match want.as_str() {
            "direct" if direct_ok && file.is_some() => IoMode::Direct,
            "pread" if file.is_some() => IoMode::Pread,
            _ => IoMode::Mmap,
        };
        Self {
            model: model.clone(),
            mode,
            file,
            #[cfg(target_os = "linux")]
            direct,
            direct_broken: AtomicBool::new(false),
            nowait_ok: AtomicBool::new(true),
        }
    }

    pub(crate) fn mode(&self) -> IoMode {
        self.mode
    }

    /// Read one expert into `buf` (aligned, at least `slot_bytes_for(total)`
    /// long). Returns where its three parts landed, and whether the page
    /// cache served all of it.
    fn read(&self, loc: &ExpertLoc, buf: &mut [u8]) -> Option<(Ranges, bool)> {
        if self.mode == IoMode::Mmap {
            return self.read_mmap(loc, buf).map(|r| (r, true));
        }
        if loc.contiguous() {
            let (s, e) = loc.span();
            let s_al = s & !(ALIGN as u64 - 1);
            let shift = (s - s_al) as usize;
            let need = (e - s_al) as usize;
            if need > buf.len() {
                return None;
            }
            let cached = self.read_span(s, e, s_al, buf)?;
            let base = shift as u64;
            let mut r: Ranges = [(0, 0); 3];
            for (i, p) in loc.parts.iter().enumerate() {
                r[i] = ((base + (p.abs - s)) as u32, p.len as u32);
            }
            return Some((r, cached));
        }
        // parts far apart: one aligned region each
        let mut r: Ranges = [(0, 0); 3];
        let mut off = 0usize;
        let mut all_cached = true;
        for (i, p) in loc.parts.iter().enumerate() {
            let s_al = p.abs & !(ALIGN as u64 - 1);
            let shift = (p.abs - s_al) as usize;
            let need = (shift + p.len).next_multiple_of(ALIGN);
            if off + need > buf.len() {
                return None;
            }
            let cached =
                self.read_span(p.abs, p.abs + p.len as u64, s_al, &mut buf[off..off + need])?;
            all_cached &= cached;
            r[i] = ((off + shift) as u32, p.len as u32);
            off += need;
        }
        Some((r, all_cached))
    }

    /// File bytes [s, e) into `buf`, laid out from the aligned start `s_al`
    /// (byte `s` lands at `s - s_al`). True when the page cache had them.
    fn read_span(&self, s: u64, e: u64, s_al: u64, buf: &mut [u8]) -> Option<bool> {
        let shift = (s - s_al) as usize;
        let len = (e - s) as usize;
        let file = self.file.as_ref()?;
        #[cfg(target_os = "linux")]
        if self.mode == IoMode::Direct && !self.direct_broken.load(Ordering::Relaxed) {
            let got = if self.nowait_ok.load(Ordering::Relaxed) {
                match pread_cached(file, &mut buf[shift..shift + len], s) {
                    Ok(n) => n,
                    Err(()) => {
                        self.nowait_ok.store(false, Ordering::Relaxed);
                        0
                    }
                }
            } else if self.resident(s, len) {
                // the page cache has it all: copy it out of the mapping
                let src = &self.model.primary_bytes()[s as usize..s as usize + len];
                buf[shift..shift + len].copy_from_slice(src);
                len
            } else {
                0
            };
            if got == len {
                return Some(true);
            }
            // the rest straight from the device, aligned
            let r0 = s + got as u64;
            let r0_al = r0 & !(ALIGN as u64 - 1);
            let e_al = e.next_multiple_of(ALIGN as u64);
            let o = (r0_al - s_al) as usize;
            let n = (e_al - r0_al) as usize;
            if let Some(d) = self.direct.as_ref() {
                match pread_full(d, &mut buf[o..o + n], r0_al) {
                    // a short read is fine past EOF as long as it covers `e`
                    Ok(k) if r0_al + k as u64 >= e => return Some(false),
                    Ok(_) => return None,
                    Err(err) if err.raw_os_error() == Some(libc::EINVAL) => {
                        self.direct_broken.store(true, Ordering::Relaxed);
                    }
                    Err(_) => return None,
                }
            }
        }
        match pread_full(file, &mut buf[shift..shift + len], s) {
            Ok(k) if k == len => Some(false),
            _ => None,
        }
    }

    /// Every page of file bytes [s, s+len) is in the page cache (mincore
    /// over the mapping; Linux).
    #[cfg(target_os = "linux")]
    fn resident(&self, s: u64, len: usize) -> bool {
        let bytes = self.model.primary_bytes();
        let page = 4096usize;
        let base = bytes.as_ptr() as usize;
        let a = (base + s as usize) & !(page - 1);
        let end = base + s as usize + len;
        if s as usize + len > bytes.len() {
            return false;
        }
        let n = (end - a).div_ceil(page);
        let mut vec = vec![0u8; n];
        // SAFETY: a page-aligned range inside our mapping, one byte per page.
        let rc = unsafe { libc::mincore(a as *mut libc::c_void, end - a, vec.as_mut_ptr()) };
        rc == 0 && vec.iter().all(|&v| v & 1 != 0)
    }

    fn read_mmap(&self, loc: &ExpertLoc, buf: &mut [u8]) -> Option<Ranges> {
        let bytes = self.model.primary_bytes();
        let mut r: Ranges = [(0, 0); 3];
        let mut off = 0usize;
        for (i, p) in loc.parts.iter().enumerate() {
            let src = bytes.get(p.abs as usize..p.abs as usize + p.len)?;
            buf.get_mut(off..off + p.len)?.copy_from_slice(src);
            r[i] = (off as u32, p.len as u32);
            off = (off + p.len).next_multiple_of(64);
        }
        Some(r)
    }
}

/// Raw segment pointer (the tier hands out disjoint slots by protocol).
struct Seg(*mut u8);
// SAFETY: the memory is plain bytes; slot ownership is arbitrated by `Meta`.
unsafe impl Send for Seg {}
unsafe impl Sync for Seg {}

struct Meta {
    /// key → slot, for slots whose bytes are complete
    slot_of: Vec<u32>,
    /// slot → key (u32::MAX = free)
    key_of: Vec<u32>,
    valid: Vec<bool>,
    pins: Vec<u32>,
    last: Vec<u64>,
    ranges: Vec<Ranges>,
    /// per key: the VRAM arena holds it (its copy here is redundant)
    in_vram: Vec<bool>,
    free: Vec<u32>,
    clock: u64,
    cursor: usize,
    /// slots ever handed out (the high-water mark of allocated segments)
    touched: usize,
}

pub(crate) struct RamTier {
    slot_bytes: usize,
    cap: usize,
    seg_slots: usize,
    segs: Mutex<Vec<Seg>>,
    meta: Mutex<Meta>,
}

impl RamTier {
    fn new(n_keys: usize, cap: usize, slot_bytes: usize) -> Self {
        let seg_slots = ((1usize << 30) / slot_bytes).max(1);
        Self {
            slot_bytes,
            cap,
            seg_slots,
            segs: Mutex::new(Vec::new()),
            meta: Mutex::new(Meta {
                slot_of: vec![u32::MAX; n_keys],
                key_of: vec![u32::MAX; cap],
                valid: vec![false; cap],
                pins: vec![0; cap],
                last: vec![0; cap],
                ranges: vec![[(0, 0); 3]; cap],
                in_vram: vec![false; n_keys],
                // lowest slots first: segments are allocated as they fill
                free: (0..cap as u32).rev().collect(),
                clock: 0,
                cursor: 0,
                touched: 0,
            }),
        }
    }

    /// The slot's bytes; allocates its segment on first use.
    fn slot_ptr(&self, slot: usize) -> Option<*mut u8> {
        let si = slot / self.seg_slots;
        let mut segs = self.segs.lock().unwrap();
        while segs.len() <= si {
            let bytes = self.seg_slots * self.slot_bytes;
            let layout = std::alloc::Layout::from_size_align(bytes, ALIGN).ok()?;
            // SAFETY: non-zero size, valid alignment; freed in Drop.
            let p = unsafe { std::alloc::alloc(layout) };
            if p.is_null() {
                return None;
            }
            segs.push(Seg(p));
        }
        // SAFETY: inside the segment allocated above.
        Some(unsafe { segs[si].0.add((slot % self.seg_slots) * self.slot_bytes) })
    }

    /// Pin a complete copy of `key` for reading.
    fn lookup_pin(&self, key: usize) -> Option<(usize, Ranges)> {
        let mut m = self.meta.lock().unwrap();
        let s = *m.slot_of.get(key)?;
        if s == u32::MAX {
            return None;
        }
        let s = s as usize;
        m.pins[s] += 1;
        m.clock += 1;
        m.last[s] = m.clock;
        Some((s, m.ranges[s]))
    }

    fn unpin(&self, slot: usize) {
        let mut m = self.meta.lock().unwrap();
        m.pins[slot] = m.pins[slot].saturating_sub(1);
    }

    /// A slot to load `key` into, pinned. Takes a free slot, else drops a
    /// copy of an expert the VRAM arena holds, else (only when
    /// `evict_cold`) the least recently used unpinned copy.
    fn reserve(&self, key: usize, evict_cold: bool) -> Option<usize> {
        let mut m = self.meta.lock().unwrap();
        if m.slot_of.get(key).is_none_or(|&s| s != u32::MAX) {
            return None;
        }
        let slot = if let Some(s) = m.free.pop() {
            s as usize
        } else {
            let victim = Self::pick_victim(&mut m, self.cap, evict_cold)?;
            let old = m.key_of[victim] as usize;
            if m.slot_of[old] == victim as u32 {
                m.slot_of[old] = u32::MAX;
            }
            m.valid[victim] = false;
            victim
        };
        m.key_of[slot] = key as u32;
        m.pins[slot] = 1;
        m.touched = m.touched.max(slot + 1);
        Some(slot)
    }

    /// Sampled LRU over the complete, unpinned slots: redundant copies
    /// (arena-resident experts) first.
    fn pick_victim(m: &mut Meta, cap: usize, evict_cold: bool) -> Option<usize> {
        const SAMPLE: usize = 96;
        let mut best_dup: Option<(u64, usize)> = None;
        let mut best_any: Option<(u64, usize)> = None;
        let mut seen = 0usize;
        let mut scanned = 0usize;
        while scanned < cap && (seen < SAMPLE || (best_dup.is_none() && !evict_cold)) {
            let s = m.cursor;
            m.cursor = (m.cursor + 1) % cap;
            scanned += 1;
            if !m.valid[s] || m.pins[s] != 0 {
                continue;
            }
            seen += 1;
            let k = m.key_of[s] as usize;
            let cand = (m.last[s], s);
            if m.in_vram.get(k).copied().unwrap_or(false) {
                best_dup = Some(best_dup.map_or(cand, |b| b.min(cand)));
            }
            best_any = Some(best_any.map_or(cand, |b| b.min(cand)));
        }
        best_dup
            .or(if evict_cold { best_any } else { None })
            .map(|(_, s)| s)
    }

    fn commit(&self, slot: usize, key: usize, ranges: Option<Ranges>) {
        let mut m = self.meta.lock().unwrap();
        // two threads may have loaded the same expert: the first copy wins
        let ranges = ranges.filter(|_| m.slot_of[key] == u32::MAX);
        match ranges {
            Some(r) => {
                m.ranges[slot] = r;
                m.valid[slot] = true;
                m.slot_of[key] = slot as u32;
                m.clock += 1;
                m.last[slot] = m.clock;
                m.pins[slot] = m.pins[slot].saturating_sub(1);
            }
            None => {
                m.key_of[slot] = u32::MAX;
                m.valid[slot] = false;
                m.pins[slot] = 0;
                m.free.push(slot as u32);
            }
        }
    }

    fn holds(&self, key: usize) -> bool {
        let m = self.meta.lock().unwrap();
        m.slot_of.get(key).is_some_and(|&s| s != u32::MAX)
    }

    /// (complete copies, of which redundant, bytes allocated)
    fn usage(&self) -> (usize, usize, usize) {
        let m = self.meta.lock().unwrap();
        let mut n = 0;
        let mut dup = 0;
        for s in 0..self.cap {
            if m.valid[s] {
                n += 1;
                if m.in_vram[m.key_of[s] as usize] {
                    dup += 1;
                }
            }
        }
        let segs = self.segs.lock().unwrap().len();
        (n, dup, segs * self.seg_slots * self.slot_bytes)
    }
}

impl Drop for RamTier {
    fn drop(&mut self) {
        let bytes = self.seg_slots * self.slot_bytes;
        if let Ok(layout) = std::alloc::Layout::from_size_align(bytes, ALIGN) {
            for s in self.segs.lock().unwrap().drain(..) {
                // SAFETY: allocated in `slot_ptr` with this layout.
                unsafe { std::alloc::dealloc(s.0, layout) };
            }
        }
    }
}

/// Counters for the per-token profile line.
#[derive(Default)]
pub(crate) struct StoreStats {
    pub ram_hits: AtomicU64,
    pub file_reads: AtomicU64,
    pub cache_hits: AtomicU64,
    pub file_bytes: AtomicU64,
    pub file_ns: AtomicU64,
    pub bg_loads: AtomicU64,
}

struct Background {
    /// experts the arena just evicted: wanted back soon
    urgent: VecDeque<u32>,
    /// warm-up order (profile ranking, then the rest)
    warm: VecDeque<u32>,
    stop: bool,
}

pub(crate) struct ExpertStore {
    reader: FileReader,
    locs: Vec<Option<ExpertLoc>>,
    n_experts: usize,
    slot_bytes: usize,
    tier: Option<RamTier>,
    pub stats: StoreStats,
    bg: Mutex<Background>,
    bg_cv: Condvar,
    /// foreground reads in flight: background loaders back off meanwhile
    fg_reads: AtomicU64,
    /// page-cache mode: release / prefetch file pages as the arena moves
    hints: bool,
    file: Option<std::fs::File>,
}

/// Counts a foreground read for its lifetime.
struct Foreground<'a>(&'a AtomicU64);

impl<'a> Foreground<'a> {
    fn enter(c: &'a AtomicU64) -> Self {
        c.fetch_add(1, Ordering::Relaxed);
        Self(c)
    }
}

impl Drop for Foreground<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Memory this process can still take, in bytes: MemAvailable, and inside
/// a cgroup its limit minus what it holds that cannot be reclaimed (page
/// cache counts as available: the kernel drops it for us).
pub(crate) fn host_available_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        let avail = meminfo
            .lines()
            .find(|l| l.starts_with("MemAvailable:"))?
            .split_whitespace()
            .nth(1)?
            .parse::<u64>()
            .ok()?
            * 1024;
        let read = |p: &str| -> Option<u64> {
            std::fs::read_to_string(p).ok()?.trim().parse::<u64>().ok()
        };
        let cg = match (
            read("/sys/fs/cgroup/memory.max"),
            read("/sys/fs/cgroup/memory.current"),
        ) {
            (Some(max), Some(cur)) if max < u64::MAX / 2 => {
                let file = std::fs::read_to_string("/sys/fs/cgroup/memory.stat")
                    .ok()
                    .and_then(|s| {
                        s.lines()
                            .find(|l| l.starts_with("file "))
                            .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
                    })
                    .unwrap_or(0);
                Some(max.saturating_sub(cur.saturating_sub(file)))
            }
            // cgroup v1: the page cache counts as available here too
            _ => read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
                .filter(|&v| v < u64::MAX / 2)
                .zip(read("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
                .map(|(l, u)| {
                    let cache = std::fs::read_to_string("/sys/fs/cgroup/memory/memory.stat")
                        .ok()
                        .and_then(|s| {
                            s.lines()
                                .find(|l| l.starts_with("total_cache "))
                                .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
                        })
                        .unwrap_or(0);
                    l.saturating_sub(u.saturating_sub(cache))
                }),
        };
        Some(cg.map_or(avail, |c| c.min(avail)))
    }
    #[cfg(not(target_os = "linux"))]
    {
        crate::fcd::available_ram_bytes()
    }
}

impl ExpertStore {
    /// `triples[layer][expert]` are the (gate, up, down) tensor indices;
    /// `vram_slots` is the arena's capacity (the tier is sized for what the
    /// arena cannot hold).
    pub(crate) fn new(
        model: &Arc<CmfModel>,
        triples: &[Vec<(usize, usize, usize)>],
        n_experts: usize,
        vram_slots: usize,
    ) -> Option<Arc<Self>> {
        let loc_of = |t: (usize, usize, usize)| -> Option<ExpertLoc> {
            let part = |i: usize| -> Option<Part> {
                let e = model.tensors.get(i)?;
                Some(Part {
                    abs: model.entry_abs_offset(e)? as u64,
                    len: e.nbytes as usize,
                })
            };
            Some(ExpertLoc {
                parts: [part(t.0)?, part(t.1)?, part(t.2)?],
            })
        };
        let n_keys = triples.len() * n_experts;
        let mut locs = vec![None; n_keys];
        for (li, row) in triples.iter().enumerate() {
            for (e, &t) in row.iter().enumerate().take(n_experts) {
                locs[li * n_experts + e] = loc_of(t);
            }
        }
        let max_total = locs.iter().flatten().map(|l| l.total()).max()?;
        let slot_bytes = slot_bytes_for(max_total);
        let reader = FileReader::open(model);

        // How many experts the RAM tier may hold.
        let want = std::env::var("CMF_QWEN_RAM_TIER_MB").unwrap_or_default();
        let cap_bytes = match want.as_str() {
            "auto" => host_available_bytes().map_or(0, |avail| {
                // leave room for the OS, the page cache the n-gram rows and
                // the skeleton live in, and the process itself
                let reserve = (avail / 6).max(6 << 30);
                avail.saturating_sub(reserve)
            }),
            v => v.parse::<u64>().map_or(0, |mb| mb << 20),
        };
        // never more slots than experts the arena cannot hold, plus room
        // for the copies that make a miss on a just-evicted expert cheap
        let useful = n_keys.saturating_sub(vram_slots) + vram_slots / 8;
        let cap = ((cap_bytes / slot_bytes as u64) as usize)
            .min(useful)
            .min(n_keys);
        let tier = (cap >= 64).then(|| RamTier::new(n_keys, cap, slot_bytes));
        let store = Arc::new(Self {
            reader,
            locs,
            n_experts,
            slot_bytes,
            tier,
            stats: StoreStats::default(),
            bg: Mutex::new(Background {
                urgent: VecDeque::new(),
                warm: VecDeque::new(),
                stop: false,
            }),
            bg_cv: Condvar::new(),
            fg_reads: AtomicU64::new(0),
            hints: std::env::var("CMF_QWEN_CACHE_HINTS").as_deref() == Ok("1"),
            file: std::fs::File::open(&model.path).ok(),
        });
        tracing::info!(
            "qwen4 expert store: io {:?}, RAM tier {} experts ({} MiB max)",
            store.reader.mode(),
            cap,
            (cap * slot_bytes) >> 20
        );
        Some(store)
    }

    pub(crate) fn io_mode(&self) -> IoMode {
        self.reader.mode()
    }

    pub(crate) fn tier_capacity(&self) -> usize {
        self.tier.as_ref().map_or(0, |t| t.cap)
    }

    fn key(&self, layer: usize, expert: usize) -> usize {
        layer * self.n_experts + expert
    }

    /// Run `f` on the expert's (gate, up, down) bytes: from the RAM tier
    /// when it has them, else read from the file (into the tier when there
    /// is a free or redundant slot, else into thread-local scratch).
    pub(crate) fn with_expert<R>(
        &self,
        layer: usize,
        expert: usize,
        keep: bool,
        f: impl FnOnce([&[u8]; 3]) -> R,
    ) -> Option<R> {
        let key = self.key(layer, expert);
        let loc = (*self.locs.get(key)?)?;
        let _fg = Foreground::enter(&self.fg_reads);
        // the mapping itself, when nothing has to be kept in the tier
        let mmap_direct = self.reader.mode() == IoMode::Mmap && (self.tier.is_none() || !keep);
        if mmap_direct && self.tier.as_ref().is_none_or(|t| !t.holds(key)) {
            return self.with_mapped(&loc, f);
        }
        if let Some(t) = self.tier.as_ref() {
            if let Some((slot, r)) = t.lookup_pin(key) {
                let p = t.slot_ptr(slot)?;
                // SAFETY: the pin keeps the slot's bytes from being reused.
                let buf = unsafe { std::slice::from_raw_parts(p, self.slot_bytes) };
                let out = f(split(buf, &r));
                t.unpin(slot);
                self.stats.ram_hits.fetch_add(1, Ordering::Relaxed);
                return Some(out);
            }
            if keep && let Some(slot) = t.reserve(key, false) {
                let Some(p) = t.slot_ptr(slot) else {
                    t.commit(slot, key, None);
                    return None;
                };
                // SAFETY: the reservation gives this thread the slot alone.
                let buf = unsafe { std::slice::from_raw_parts_mut(p, self.slot_bytes) };
                let Some(r) = self.read_file(&loc, buf) else {
                    t.commit(slot, key, None);
                    return None;
                };
                let out = f(split(buf, &r));
                t.commit(slot, key, Some(r));
                return Some(out);
            }
        }
        SCRATCH.with(|s| {
            let mut s = s.borrow_mut();
            let buf = s.ensure(self.slot_bytes);
            let r = self.read_file(&loc, buf)?;
            Some(f(split(buf, &r)))
        })
    }

    /// `f` over slices of the mapping, after asking the kernel to read the
    /// expert's pages ahead (a no-op for resident pages).
    fn with_mapped<R>(&self, loc: &ExpertLoc, f: impl FnOnce([&[u8]; 3]) -> R) -> Option<R> {
        let bytes = self.reader.model.primary_bytes();
        let t0 = std::time::Instant::now();
        let mut parts: [&[u8]; 3] = [&[], &[], &[]];
        for (i, p) in loc.parts.iter().enumerate() {
            parts[i] = bytes.get(p.abs as usize..p.abs as usize + p.len)?;
        }
        if willneed_on() {
            will_need(bytes, loc);
        }
        let out = f(parts);
        let s = &self.stats;
        s.file_reads.fetch_add(1, Ordering::Relaxed);
        s.file_bytes
            .fetch_add(loc.total() as u64, Ordering::Relaxed);
        s.file_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Some(out)
    }

    /// Ask the kernel to start reading these experts now (the frame's cold
    /// winners, before the parallel copies start): every miss then joins
    /// one deep I/O queue instead of waiting its turn behind a page fault.
    pub(crate) fn prefetch(&self, layer: usize, experts: &[usize]) {
        if self.reader.mode() != IoMode::Mmap || !willneed_on() {
            return;
        }
        let bytes = self.reader.model.primary_bytes();
        for &e in experts {
            if let Some(loc) = self.locs.get(self.key(layer, e)).copied().flatten() {
                will_need(bytes, &loc);
            }
        }
    }

    fn read_file(&self, loc: &ExpertLoc, buf: &mut [u8]) -> Option<Ranges> {
        let t0 = std::time::Instant::now();
        let (r, cached) = self.reader.read(loc, buf)?;
        let s = &self.stats;
        s.file_reads.fetch_add(1, Ordering::Relaxed);
        if cached {
            s.cache_hits.fetch_add(1, Ordering::Relaxed);
        }
        s.file_bytes
            .fetch_add(loc.total() as u64, Ordering::Relaxed);
        s.file_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Some(r)
    }

    /// The arena took (`true`) or dropped (`false`) this expert. A dropped
    /// expert the tier does not hold is read back in the background.
    pub(crate) fn note_vram(&self, layer: usize, expert: usize, resident: bool) {
        let key = self.key(layer, expert);
        let Some(t) = self.tier.as_ref() else {
            if self.hints
                && let Some(loc) = self.locs.get(key).copied().flatten()
            {
                self.advise(&loc, resident);
            }
            return;
        };
        let held = {
            let mut m = t.meta.lock().unwrap();
            if let Some(v) = m.in_vram.get_mut(key) {
                *v = resident;
            }
            m.slot_of.get(key).is_some_and(|&s| s != u32::MAX)
        };
        if !resident && !held {
            let mut bg = self.bg.lock().unwrap();
            if bg.urgent.len() < 4096 {
                bg.urgent.push_back(key as u32);
                self.bg_cv.notify_one();
            }
        }
    }

    /// Page-cache hints for one expert: once it sits in VRAM its file
    /// pages may go (DONTNEED); when the arena drops it they are read back
    /// asynchronously (WILLNEED). Linux; advisory.
    fn advise(&self, loc: &ExpertLoc, in_vram: bool) {
        #[cfg(target_os = "linux")]
        if let Some(f) = self.file.as_ref() {
            use std::os::unix::io::AsRawFd;
            let advice = if in_vram {
                libc::POSIX_FADV_DONTNEED
            } else {
                libc::POSIX_FADV_WILLNEED
            };
            for p in &loc.parts {
                // SAFETY: plain fd + numeric range; advisory by contract.
                unsafe {
                    libc::posix_fadvise(
                        f.as_raw_fd(),
                        p.abs as libc::off_t,
                        p.len as libc::off_t,
                        advice,
                    );
                }
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = (loc, in_vram);
    }

    /// Start the background loaders on `order` (keys, most wanted first).
    pub(crate) fn start_background(self: &Arc<Self>, order: Vec<(usize, usize)>) {
        if self.tier.is_none() {
            return;
        }
        {
            let mut bg = self.bg.lock().unwrap();
            bg.warm = order
                .into_iter()
                .map(|(l, e)| self.key(l, e) as u32)
                .collect();
        }
        let n = std::env::var("CMF_QWEN_TIER_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2)
            .clamp(1, 32);
        for _ in 0..n {
            let me = Arc::downgrade(self);
            let _ = std::thread::Builder::new()
                .name("qwen4-ram-tier".into())
                .spawn(move || Self::loader(me));
        }
    }

    fn loader(me: std::sync::Weak<Self>) {
        loop {
            let Some(store) = me.upgrade() else { return };
            let job = {
                let mut bg = store.bg.lock().unwrap();
                loop {
                    if bg.stop {
                        return;
                    }
                    if let Some(k) = bg.urgent.pop_front() {
                        break Some((k as usize, true));
                    }
                    if let Some(k) = bg.warm.pop_front() {
                        break Some((k as usize, false));
                    }
                    // idle: wait for an eviction (timeout so a dropped
                    // store lets the thread go)
                    let (g, _) = store
                        .bg_cv
                        .wait_timeout(bg, std::time::Duration::from_millis(500))
                        .unwrap();
                    bg = g;
                    if bg.urgent.is_empty() && bg.warm.is_empty() {
                        break None;
                    }
                }
            };
            let Some((key, urgent)) = job else {
                drop(store);
                continue;
            };
            // the token being decoded waits on foreground reads: give them
            // the drive (bounded, so a busy decode cannot starve warm-up)
            for _ in 0..200 {
                if store.fg_reads.load(Ordering::Relaxed) == 0 {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_micros(250));
            }
            let Some(t) = store.tier.as_ref() else { return };
            // warm-up skips what the arena holds; an evicted expert is
            // wanted no matter what
            let in_vram = t.meta.lock().unwrap().in_vram[key];
            if (!urgent && in_vram) || t.holds(key) {
                continue;
            }
            let Some(loc) = store.locs.get(key).copied().flatten() else {
                continue;
            };
            // An evicted expert takes only a free slot or a redundant copy:
            // dropping another cold expert for it just moves the miss (and
            // at small VRAM budgets turned into a re-read loop).
            let Some(slot) = t.reserve(key, false) else {
                if !urgent {
                    // the tier is full of experts the arena lacks: warm-up
                    // is done
                    store.bg.lock().unwrap().warm.clear();
                }
                continue;
            };
            let Some(p) = t.slot_ptr(slot) else {
                t.commit(slot, key, None);
                continue;
            };
            // SAFETY: the reservation gives this thread the slot alone.
            let buf = unsafe { std::slice::from_raw_parts_mut(p, store.slot_bytes) };
            let r = store.read_file(&loc, buf);
            if r.is_some() {
                store.stats.bg_loads.fetch_add(1, Ordering::Relaxed);
            }
            t.commit(slot, key, r);
        }
    }

    /// One line for the profile: tier fill and where misses were served.
    pub(crate) fn report(&self) -> String {
        let s = &self.stats;
        let ld = |a: &AtomicU64| a.swap(0, Ordering::Relaxed);
        let (n, dup, bytes) = self.tier.as_ref().map_or((0, 0, 0), |t| t.usage());
        let reads = ld(&s.file_reads);
        let ns = ld(&s.file_ns);
        format!(
            "ram_hits={} file={} (cached {}) {:.0}MB {:.1}ms bg={} tier={}/{} ({} dup, {} MiB)",
            ld(&s.ram_hits),
            reads,
            ld(&s.cache_hits),
            ld(&s.file_bytes) as f64 / 1e6,
            ns as f64 / 1e6,
            ld(&s.bg_loads),
            n,
            self.tier_capacity(),
            dup,
            bytes >> 20
        )
    }
}

impl Drop for ExpertStore {
    fn drop(&mut self) {
        if let Ok(mut bg) = self.bg.lock() {
            bg.stop = true;
        }
        self.bg_cv.notify_all();
    }
}

fn split<'a>(buf: &'a [u8], r: &Ranges) -> [&'a [u8]; 3] {
    let p = |i: usize| &buf[r[i].0 as usize..(r[i].0 + r[i].1) as usize];
    [p(0), p(1), p(2)]
}

/// `CMF_QWEN_WILLNEED=1`: read-ahead advice before mapped copies.
fn willneed_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_QWEN_WILLNEED").as_deref() == Ok("1"))
}

/// `MADV_WILLNEED` over an expert's three ranges of the mapping (unix;
/// advisory; walks the page cache even when the pages are resident).
fn will_need(bytes: &[u8], loc: &ExpertLoc) {
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
    {
        let base = bytes.as_ptr() as usize;
        let page = 4096usize;
        for p in &loc.parts {
            let s = (base + p.abs as usize) & !(page - 1);
            let e = base + p.abs as usize + p.len;
            // SAFETY: a range inside our read-only mapping; advice only.
            unsafe {
                libc::madvise(s as *mut libc::c_void, e - s, libc::MADV_WILLNEED);
            }
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
    let _ = (bytes, loc);
}

/// Random-access advice for tensors whose name passes `pred` (the n-gram
/// table): a fault then reads its own page only, instead of the 128 KiB
/// read-around that turns each ~0.5 KiB row lookup into a large disk read
/// when the table is not in the page cache. Unix only; advisory.
pub(crate) fn advise_random(model: &CmfModel, pred: impl Fn(&str) -> bool) -> usize {
    #[cfg(any(target_os = "linux", target_os = "android", target_os = "macos"))]
    {
        let base = model.primary_bytes().as_ptr() as usize;
        let map_len = model.primary_bytes().len();
        // SAFETY: plain sysconf query.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(4096) as usize;
        let mut advised = 0usize;
        for e in model.tensors.iter().filter(|e| pred(&e.name)) {
            let Some(abs) = model.entry_abs_offset(e) else {
                continue;
            };
            let s = (base + abs).next_multiple_of(page) - base;
            let end = ((base + abs + e.nbytes as usize).min(base + map_len)) / page * page - base;
            if end > s {
                // SAFETY: an address range inside our read-only mapping;
                // MADV_RANDOM only changes readahead.
                let rc = unsafe {
                    libc::madvise((base + s) as *mut libc::c_void, end - s, libc::MADV_RANDOM)
                };
                if rc == 0 {
                    advised += end - s;
                }
            }
        }
        advised
    }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
    {
        let _ = (model, pred);
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expert_span_and_slot_size() {
        let loc = ExpertLoc {
            parts: [
                Part {
                    abs: 10_000,
                    len: 444_160,
                },
                Part {
                    abs: 454_160,
                    len: 444_160,
                },
                Part {
                    abs: 898_320,
                    len: 862_720,
                },
            ],
        };
        assert!(loc.contiguous());
        assert_eq!(loc.total(), 1_751_040);
        let sb = slot_bytes_for(loc.total());
        assert_eq!(sb % ALIGN, 0);
        let (s, e) = loc.span();
        let s_al = s & !(ALIGN as u64 - 1);
        assert!((e - s_al) as usize <= sb);
        let far = ExpertLoc {
            parts: [
                Part { abs: 0, len: 100 },
                Part {
                    abs: 1 << 30,
                    len: 100,
                },
                Part {
                    abs: 2 << 30,
                    len: 100,
                },
            ],
        };
        assert!(!far.contiguous());
    }

    #[test]
    fn tier_prefers_redundant_copies() {
        let t = RamTier::new(16, 2, ALIGN);
        // fill both slots
        for k in [3usize, 5] {
            let s = t.reserve(k, false).unwrap();
            t.commit(s, k, Some([(0, 1), (1, 1), (2, 1)]));
        }
        // full, nothing redundant: a warm-up load is refused
        assert!(t.reserve(7, false).is_none());
        // key 5 moves into VRAM: its copy is the one to go
        t.meta.lock().unwrap().in_vram[5] = true;
        let s = t.reserve(7, false).unwrap();
        t.commit(s, 7, Some([(0, 1), (1, 1), (2, 1)]));
        assert!(t.holds(3) && t.holds(7) && !t.holds(5));
        // an urgent load may drop a cold copy (LRU)
        assert!(t.reserve(9, true).is_some());
        // a pinned slot is never chosen
        let t2 = RamTier::new(4, 1, ALIGN);
        let s = t2.reserve(0, false).unwrap();
        t2.commit(s, 0, Some([(0, 1), (1, 1), (2, 1)]));
        let _pin = t2.lookup_pin(0).unwrap();
        assert!(t2.reserve(1, true).is_none());
    }
}
