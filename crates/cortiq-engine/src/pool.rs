//! Persistent worker pool for row-parallel matvecs.
//!
//! Threads are spawned once and spin-then-park between calls — vmfcore
//! measured spawn-per-matvec at ~+27% decode cost versus a persistent
//! pool. Parallelism is by disjoint row ranges, so results are
//! bit-identical to the serial path (each row's dot product is computed
//! the same way).
//!
//! Dispatch is one shared job descriptor + a per-worker ticket (roadmap
//! §3 P0): the caller writes the descriptor, hands a ticket to each
//! worker it invites and JOINS THE WORK as the extra worker instead of
//! blocking on a latch. The previous design allocated an `Arc<Latch>`
//! and pushed a message into every worker's mpsc channel for every
//! matvec (~200 dispatches/token) — with decode-grade matvecs that
//! synchronization was its own budget. Workers spin for
//! `CMF_POOL_SPIN` iterations before parking.
//! Default 4000: at ~39 dispatches/token, park-immediately pays the
//! unpark syscall on every worker for every dispatch — measured on an
//! M4 (interleaved A/B, current epoch dispatch + parked-flag design):
//! Qwen-0.5B q8 decode 101→115 tok/s, q4t 117→149, the 50M bench model
//! 549→954 at spin=4000 vs spin=0. An early measurement that showed
//! spinning LOSING (−25% on q8) predates the parked-flag skip and the
//! multi-matrix dispatch cuts; it no longer reproduces. Over-spinning
//! still hurts (200k: −15% vs 4k — spinners steal the caller's serial
//! cycles), so the budget stays bounded. `CMF_POOL_SPIN=0` restores
//! park-immediately for share-the-box serving.
//!
//! `CMF_THREADS` env: 0/1 = serial, N = worker count
//! (default: available_parallelism − 1, capped at 8).

use std::sync::Arc;
#[cfg(any(target_os = "android", target_os = "linux"))]
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Embedder override for the pool size (C ABI `cortiq_set_threads`):
/// 0 = unset, consult CMF_THREADS / topology as before. Read once at
/// pool construction, so set it before the load.
pub static FORCED_THREADS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Kernel thread ids of the last fully constructed pool's workers
/// (Android/Linux) — what ADPF's PerformanceHintManager needs to attribute
/// work to the governor. Published as one complete snapshot after that
/// pool's per-instance registration barrier; empty elsewhere.
pub static WORKER_TIDS: std::sync::Mutex<Vec<i32>> = std::sync::Mutex::new(Vec::new());

/// Keeps a word that one thread writes and others poll off everybody
/// else's cache line (128 bytes: Apple silicon lines, and the adjacent-
/// line prefetcher pair on x86).
#[repr(align(128))]
struct Padded<T>(T);

/// One worker's mailbox, alone on its line: the worker polls it, only the
/// caller writes `ticket`, only the worker writes `parked`.
#[repr(align(128))]
struct Slot {
    /// Jobs handed to this worker so far. The caller bumps it (Release,
    /// after the descriptor) once per job the worker is invited to; the
    /// worker runs exactly one job per bump. The caller issues the next
    /// bump only after this worker's `remaining` decrement for the
    /// previous one, so the worker can never miss or double a ticket.
    ticket: AtomicUsize,
    /// "I am parked" — lets the caller skip the unpark syscall for a
    /// worker that is still spinning.
    parked: AtomicBool,
}

struct Inner {
    /// Bumped once per published job, invited or not. Spinning workers
    /// read it ONLY to re-arm their spin budget (the pool is busy, the
    /// next job is microseconds away); it never makes a worker run
    /// anything or read the descriptor.
    epoch: AtomicUsize,
    /// The published job: closure (fat-pointer halves), participant
    /// count, publisher's GPU device. The device rides along because a
    /// dispatch begun on card 1 must not finish on card 0: worker threads
    /// have their own thread-locals, and the engine resolves its wgpu
    /// context through one.
    ///
    /// Only workers holding a ticket for the current job read these
    /// words, and the caller rewrites them only after `remaining` hit 0,
    /// i.e. after every ticket holder has finished — so a read never
    /// races a write and there is no torn descriptor to detect.
    ///
    /// History, because both earlier protocols failed: (1) one slot read
    /// by EVERY worker on the epoch bump. A worker not invited to job k
    /// is not waited for, so it could be preempted between seeing epoch k
    /// and reading the slot, by which time the slot held job k+1 — it ran
    /// k+1, decremented `remaining`, then saw epoch k+1 as new and ran it
    /// AGAIN: `remaining` wrapped to `usize::MAX` and the caller spun
    /// forever (the 47-minute `cortiq ppl` on the S4 bounded export,
    /// whose 384-row Embryo matrices make every dispatch a limited one).
    /// (2) The same words under a seqlock with the epoch inside: correct
    /// on x86, but no release fence followed the writer's opening
    /// increment, so the memory model did not order it before the word
    /// stores (weakly ordered ARM may show a reader new words under an
    /// old even count), and every limited dispatch still woke every
    /// spinning worker to read the descriptor and skip it. With more
    /// spinners than CPUs (the straggler test's 4× pool on a 2–4-CPU CI
    /// runner) they held the CPUs while the one invited worker waited
    /// for a time slice: ~10 ms per dispatch, 20 k jobs in ~230 s. A
    /// worker now learns it is invited from its own ticket and nothing
    /// else.
    desc_data: AtomicUsize,
    desc_vtable: AtomicUsize,
    desc_n: AtomicUsize,
    desc_dev: AtomicUsize,
    /// Ticket holders still running the current job (excludes the
    /// caller). Its own line: the caller polls it while workers
    /// decrement.
    remaining: Padded<AtomicUsize>,
    /// One mailbox per worker, same order as `Pool::threads`.
    slots: Box<[Slot]>,
    shutdown: AtomicBool,
    /// Spin iterations before a worker parks (0 = park immediately).
    spin_budget: AtomicUsize,
    /// More threads (workers + caller) than CPUs this process may run
    /// on. Spinning cannot help then: a spinner occupies the CPU the
    /// thread that has the work needs, and the job waits for the
    /// scheduler's time slice. So in this mode a worker's spin budget is
    /// re-armed only by its own tickets (an idle worker parks instead of
    /// spinning through other workers' jobs), and every spinner — worker
    /// or waiting caller — yields the CPU between short spin bursts.
    /// The default size (available_parallelism − 1 workers) never is.
    oversubscribed: bool,
    /// Per-pool registration state. `WORKER_TIDS` is a process-wide
    /// snapshot for ADPF and cannot be a construction barrier: another
    /// pool may clear and republish that snapshot concurrently.
    #[cfg(any(target_os = "android", target_os = "linux"))]
    registered: AtomicUsize,
    #[cfg(any(target_os = "android", target_os = "linux"))]
    worker_tids: Mutex<Vec<i32>>,
}

/// Process-wide dispatch counter (roadmap §3 P0 «измерения»): one tick
/// per published job. `bench --json` reports dispatches/token from it.
static DISPATCHES: AtomicUsize = AtomicUsize::new(0);

/// Total pool jobs published since process start (all pools).
pub fn dispatch_count() -> usize {
    DISPATCHES.load(Ordering::Relaxed)
}

/// Persistent thread pool: shared job slot, epoch dispatch, caller
/// participation.
pub struct Pool {
    inner: Arc<Inner>,
    /// Thread handles for `unpark` (same order as `parked`).
    threads: Vec<std::thread::Thread>,
    joins: Vec<std::thread::JoinHandle<()>>,
}

fn spin_budget_from_env() -> usize {
    std::env::var("CMF_POOL_SPIN")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4000)
}

/// Rows per chunk: enough chunks to balance, large enough to keep the SDOT
/// inner loop and the prefetcher in their stride — and never so coarse that
/// ONE worker takes the whole job.
///
/// That last clause was missing. The floor was a flat 32, so any job with
/// fewer than 32 rows went entirely to whichever worker grabbed the cursor
/// first while the other 48 were woken, found nothing, and left. The
/// hyper-connection projection has 24 rows and is called 86 times a token:
/// it paid the full price of a fan-out and ran single-threaded.
pub(crate) fn grain_for(rows: usize, workers: usize) -> usize {
    if rows == 0 || workers <= 1 {
        return rows.max(1);
    }
    let balanced = (rows / (workers * 8)).max(32);
    // One chunk per worker at the very least.
    balanced.min(rows.div_ceil(workers)).max(1)
}

impl Pool {
    pub fn new(n_workers: usize) -> Self {
        Self::with_spin(n_workers, spin_budget_from_env())
    }

    /// Explicit spin budget (tests pin it without touching the env).
    pub fn with_spin(n_workers: usize, spin_budget: usize) -> Self {
        // Affinity- and cgroup-quota-aware on Linux; read once — the NUMA
        // bind never narrows the mask below the pool's thread count.
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let inner = Arc::new(Inner {
            epoch: AtomicUsize::new(0),
            desc_data: AtomicUsize::new(0),
            desc_vtable: AtomicUsize::new(0),
            desc_n: AtomicUsize::new(0),
            desc_dev: AtomicUsize::new(0),
            remaining: Padded(AtomicUsize::new(0)),
            slots: (0..n_workers)
                .map(|_| Slot {
                    ticket: AtomicUsize::new(0),
                    parked: AtomicBool::new(false),
                })
                .collect(),
            shutdown: AtomicBool::new(false),
            spin_budget: AtomicUsize::new(spin_budget),
            oversubscribed: n_workers + 1 > cpus,
            #[cfg(any(target_os = "android", target_os = "linux"))]
            registered: AtomicUsize::new(0),
            #[cfg(any(target_os = "android", target_os = "linux"))]
            worker_tids: Mutex::new(Vec::with_capacity(n_workers)),
        });
        let mut joins = Vec::with_capacity(n_workers);
        for w in 0..n_workers {
            let inner = inner.clone();
            let h = std::thread::Builder::new()
                .name(format!("cmf-pool-{w}"))
                .spawn(move || {
                    #[cfg(any(target_os = "android", target_os = "linux"))]
                    {
                        let tid = unsafe { libc::gettid() } as i32;
                        if let Ok(mut tids) = inner.worker_tids.lock() {
                            tids.push(tid);
                        }
                        inner.registered.fetch_add(1, Ordering::Release);
                    }
                    worker_loop(&inner, w)
                })
                .expect("spawn pool worker");
            joins.push(h);
        }
        // Per-pool registration barrier: `spawn` returns before the closure runs,
        // and the embedder reads `cortiq_worker_tids` right after load —
        // on a phone only the first worker had registered by then (the
        // '· 1 threads' About line that misled the cmfmobile device
        // investigation twice). Thread start is milliseconds; wait for
        // every worker has registered before construction returns.
        #[cfg(any(target_os = "android", target_os = "linux"))]
        while inner.registered.load(Ordering::Acquire) < n_workers {
            std::thread::yield_now();
        }
        #[cfg(any(target_os = "android", target_os = "linux"))]
        if let (Ok(mut global), Ok(local)) = (WORKER_TIDS.lock(), inner.worker_tids.lock()) {
            *global = local.clone();
        }
        let threads = joins.iter().map(|h| h.thread().clone()).collect();
        Self {
            inner,
            threads,
            joins,
        }
    }

    /// Big-core count on heterogeneous ARM (big.LITTLE): the kernel
    /// exposes per-core capacity on Android and most ARM Linux; efficiency
    /// cores in the pool DRAG the big ones on our row-parallel jobs (the
    /// same cliff llama.cpp hits at -t 10 on an M4: 163 → 112 tok/s).
    /// None = capacities absent or homogeneous.
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "linux", target_os = "android")
    ))]
    fn big_cores() -> Option<usize> {
        Self::cores_from_capacities(&core_capacities())
    }

    /// How many cores the pool should use, from the kernel's per-core
    /// capacity values. Capacity folds µarch × clock into one number,
    /// and the two need different treatment: cores of ANOTHER µarch
    /// (A5xx efficiency cluster next to A7xx/X: capacity ratio ≥ ~2)
    /// drag row-parallel work down and are excluded; cores of the SAME
    /// µarch merely clock-binned (JLQ JR510: 8×A55 as 4×2.0 + 4×1.5 GHz,
    /// ratio 1.33) pull their weight and must ALL be used. The 1.6
    /// threshold splits the two regimes: on a Snapdragon 8-class part
    /// it keeps X + A7xx mid cores and drops A5xx.
    #[cfg_attr(
        not(all(
            target_arch = "aarch64",
            any(target_os = "linux", target_os = "android")
        )),
        allow(dead_code)
    )]
    fn cores_from_capacities(caps: &[u64]) -> Option<usize> {
        let max = *caps.iter().max()?;
        let min = *caps.iter().min()?;
        if caps.len() < 2 || max == min {
            return None;
        }
        Some(caps.iter().filter(|&&c| c * 8 >= max * 5).count())
    }

    #[cfg(target_os = "macos")]
    fn big_cores() -> Option<usize> {
        // Apple silicon: the P-only default measured WORSE than mixing the
        // efficiency cores in — the grain-pulling dispatch absorbs the
        // speed skew exactly as designed, and decode is memory-bound
        // enough that E-cores add real serviceable work (M4, dense 3B:
        // 4 threads 8.4 tok/s, 6-9 threads 9.6-10.7). Fall through to
        // available_parallelism - 1; CMF_THREADS still pins by hand.
        // The sysctl probe stays for introspection tooling.
        if true {
            return None;
        }
        #[allow(unreachable_code)]
        unsafe extern "C" {
            fn sysctlbyname(
                name: *const std::ffi::c_char,
                oldp: *mut std::ffi::c_void,
                oldlenp: *mut usize,
                newp: *mut std::ffi::c_void,
                newlen: usize,
            ) -> std::ffi::c_int;
        }
        unsafe {
            let name = std::ffi::CString::new("hw.perflevel0.physicalcpu").ok()?;
            let mut count: i32 = 0;
            let mut size = std::mem::size_of::<i32>();
            let ret = sysctlbyname(
                name.as_ptr(),
                &mut count as *mut i32 as *mut std::ffi::c_void,
                &mut size,
                std::ptr::null_mut(),
                0,
            );
            if ret == 0 && count > 0 {
                Some(count as usize)
            } else {
                None
            }
        }
    }

    #[cfg(not(any(
        all(
            target_arch = "aarch64",
            any(target_os = "linux", target_os = "android")
        ),
        target_os = "macos"
    )))]
    fn big_cores() -> Option<usize> {
        None
    }

    /// The thread count `from_env` would use RIGHT NOW: forced (C ABI)
    /// > CMF_THREADS > big-core topology > available_parallelism−1.
    /// > ≤1 means the model runs serial (no pool). Introspection
    /// > (`execution_mode`, status endpoints) must report THIS, not
    /// > available_parallelism.
    pub fn effective_threads() -> usize {
        let forced = FORCED_THREADS.load(std::sync::atomic::Ordering::Relaxed);
        if forced > 0 {
            return forced;
        }
        match std::env::var("CMF_THREADS") {
            Ok(v) => v.parse::<usize>().unwrap_or(0),
            Err(_) => match Self::big_cores() {
                Some(big) => big,
                None => {
                    // The cap was 8, which left big machines idle: on a
                    // 256-core EPYC, Nanbeige 4.2 decoded at 7.4 tok/s on
                    // the default 8 threads and 14.8 at 32, with prefill
                    // 12 -> ~16 over the same move. Past ~32 it falls off
                    // hard (5.5 at 64, 1.6 at 256) — decode is
                    // memory-bound and the extra threads only add
                    // dispatch barriers — so 32 is a ceiling, not a
                    // target. Machines with 9 cores or fewer are
                    // unaffected: avail-1 already bounds them.
                    let avail = std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(1);
                    avail.saturating_sub(1).min(32)
                }
            },
        }
    }

    /// Pool sized from `CMF_THREADS` (see module docs). `None` = serial.
    /// Without the env, heterogeneous ARM defaults to its BIG cores.
    pub fn from_env() -> Option<Arc<Self>> {
        let n = Self::effective_threads();
        if n <= 1 {
            None
        } else {
            Some(Arc::new(Self::new(n)))
        }
    }

    /// Spawned worker threads (the caller joins each job on top).
    pub fn n_workers(&self) -> usize {
        self.threads.len()
    }

    /// Keep the pool on the NUMA node that holds `regions` (the model's
    /// weight bytes). Linux with two or more nodes only; `CMF_NUMA=0`
    /// turns it off, `CMF_NUMA=node:<n>` forces a node.
    ///
    /// WHY: decode streams every weight once per token, and on a
    /// two-socket host the page cache holds a file on whichever node
    /// read it. Unpinned, the scheduler spreads the workers over both
    /// sockets and half the matvec rows cross the socket link. Measured
    /// on a 2×EPYC 7763 pod with the model's pages all on node 0 (31 CPUs
    /// of cgroup quota): a STREAM-style read over a node-0 buffer gives
    /// 42 GB/s from 31 unpinned threads and 74 GB/s from 31 threads kept
    /// on node 0. The mask is the node's physical cores (first SMT
    /// sibling) when there are enough of them for the pool, else the
    /// whole node; never narrower than the pool, so nothing oversubscribes.
    /// Threads are bound to a SET of cores, not to one core each: the
    /// scheduler still balances inside the node. The calling thread
    /// adopts the same mask on its next dispatch.
    pub fn bind_numa(&self, regions: &[&[u8]]) {
        #[cfg(target_os = "linux")]
        {
            let Some((node, cpus)) = numa::choose(regions, self.threads.len() + 1) else {
                return;
            };
            let mut applied = 0usize;
            if let Ok(tids) = self.inner.worker_tids.lock() {
                for &tid in tids.iter() {
                    if numa::set_affinity(tid, &cpus) {
                        applied += 1;
                    }
                }
            }
            numa::publish(cpus.clone());
            numa::adopt_caller();
            tracing::info!(
                "numa: pool bound to node {node} ({} cpus, {applied}/{} workers)",
                cpus.len(),
                self.threads.len()
            );
            if std::env::var("CMF_NUMA_TRACE").is_ok_and(|v| v != "0") {
                eprintln!(
                    "numa: pool bound to node {node}: {} cpus, {applied}/{} workers",
                    cpus.len(),
                    self.threads.len()
                );
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = regions;
    }

    /// Retune an already-created pool for an architecture with a measured
    /// dispatch cadence. The environment remains the operator override; this
    /// hook only changes the automatic default after model geometry is known.
    pub(crate) fn set_spin_budget(&self, spins: usize) {
        self.inner.spin_budget.store(spins, Ordering::Relaxed);
    }

    /// One job: write the descriptor, hand a ticket to workers `0..nw`,
    /// run the caller's share as participant `nw` of `nw + 1`, and return
    /// once every ticket holder has finished. Only one job is ever in
    /// flight (this drains `remaining` before it returns), so the caller
    /// is the single writer of the descriptor, `remaining` and every
    /// `ticket`; workers without a ticket never touch any of them.
    fn dispatch(&self, f: &(dyn Fn(usize, usize) + Sync), nw: usize) {
        let inner = &*self.inner;
        let n = nw + 1;
        let ptr: *const (dyn Fn(usize, usize) + Sync) = f;
        // SAFETY: a fat pointer is exactly two words on every supported
        // target; the halves are only ever reassembled by `worker_loop`.
        let raw: [usize; 2] = unsafe { std::mem::transmute(ptr) };
        inner.desc_data.store(raw[0], Ordering::Relaxed);
        inner.desc_vtable.store(raw[1], Ordering::Relaxed);
        inner.desc_n.store(n, Ordering::Relaxed);
        inner
            .desc_dev
            .store(crate::gpu::current_device(), Ordering::Relaxed);
        inner.remaining.0.store(nw, Ordering::Relaxed);
        let e = inner.epoch.load(Ordering::Relaxed);
        inner.epoch.store(e.wrapping_add(1), Ordering::Relaxed);
        // Release: a worker that sees its new ticket (Acquire) sees the
        // descriptor and `remaining` written above.
        let slots = &inner.slots[..nw];
        for slot in slots {
            let t = slot.ticket.load(Ordering::Relaxed).wrapping_add(1);
            slot.ticket.store(t, Ordering::Release);
        }
        // Pairs with the fence a worker issues between raising `parked`
        // and re-reading its ticket: either that worker sees the ticket
        // or we see its flag and unpark it — no lost wakeup. One fence
        // for all the tickets instead of a SeqCst store per worker.
        std::sync::atomic::fence(Ordering::SeqCst);
        for (slot, t) in slots.iter().zip(&self.threads) {
            if slot.parked.load(Ordering::Relaxed) {
                t.unpark();
            }
        }

        // The caller's share — the barrier costs nothing while there is
        // real work to do.
        f(nw, n);

        // Wait for the stragglers (bounded by one worker's chunk). When
        // the pool has more threads than CPUs, a ticket holder may be
        // waiting for THIS core: yield it early instead of spinning.
        let spin_limit = if inner.oversubscribed { 64 } else { 10_000 };
        let mut spins = 0usize;
        while inner.remaining.0.load(Ordering::Acquire) != 0 {
            spins += 1;
            if spins < spin_limit {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
    }

    /// Run `f(row_start, row_end)` over `0..rows`, self-balancing.
    ///
    /// One dispatch, but workers pull row-ranges from a shared cursor
    /// instead of each taking a fixed 1/n slice. On a heterogeneous CPU
    /// (Apple Silicon: 4 P-cores + 6 E-cores here) a static split makes
    /// every matvec end at the SLOWEST core's pace while the fast ones
    /// idle at the barrier; pulling by grain lets a P-core take several
    /// chunks for each one an E-core takes, so skew collapses to a
    /// single grain. Row ranges stay disjoint and each row's dot is
    /// computed exactly as in the serial path → bit-identical output.
    pub fn run_rows(&self, rows: usize, f: &(dyn Fn(usize, usize) + Sync)) {
        let grain = grain_for(rows, self.threads.len() + 1);
        let chunks = rows.div_ceil(grain.max(1));
        let next = AtomicUsize::new(0);
        self.run_limited(chunks, &|_w, _n| loop {
            let start = next.fetch_add(grain, Ordering::Relaxed);
            if start >= rows {
                break;
            }
            f(start, (start + grain).min(rows));
        });
    }

    /// `run`, with at most `max_workers` workers PARTICIPATING. Same
    /// grain, same row split, bit-identical results — only the number of
    /// threads handed the job changes: a job with eight grains has no use
    /// for three hundred workers, the unpark syscalls and the
    /// remaining-drain would BE the job (measured: 361 pool dispatches
    /// per DeepSeek-V4 token, and CMF_THREADS=64 vs 380 was 1.3 vs 2.4
    /// tok/s with no other change). Only cursor-style closures (which
    /// ignore their (idx, n) arguments) come through here: the caller
    /// identifies itself as the capped count, which is NOT `n_workers()`.
    fn run_limited(&self, max_workers: usize, f: &(dyn Fn(usize, usize) + Sync)) {
        let nw = self.threads.len().min(max_workers);
        if nw == self.threads.len() {
            return self.run(f);
        }
        #[cfg(target_os = "linux")]
        numa::adopt_caller();
        DISPATCHES.fetch_add(1, Ordering::Relaxed);
        // Workers `nw..` get no ticket: they neither run nor wait.
        self.dispatch(f, nw);
    }

    /// Multi-matrix job: one dispatch serves SEVERAL row spaces
    /// (roadmap §3 P0 — «одна внешняя публикация job на слой»). Parts
    /// are laid out back-to-back in a virtual row space and pulled by
    /// grain from one shared cursor, so QKV or gate+up cost a single
    /// barrier instead of one each. Each part's `f(start, end)` sees its
    /// OWN row indices — per-row math and outputs are bit-identical to
    /// separate `run_rows` calls.
    pub fn run_many(&self, parts: &[(usize, &(dyn Fn(usize, usize) + Sync))]) {
        let total: usize = parts.iter().map(|p| p.0).sum();
        if total == 0 {
            return;
        }
        let grain = grain_for(total, self.threads.len() + 1);
        let chunks = total.div_ceil(grain.max(1));
        let next = AtomicUsize::new(0);
        self.run_limited(chunks, &|_w, _n| loop {
            let s = next.fetch_add(grain, Ordering::Relaxed);
            if s >= total {
                break;
            }
            let e = (s + grain).min(total);
            let mut base = 0usize;
            for &(rows, f) in parts {
                let a = s.max(base);
                let b = e.min(base + rows);
                if a < b {
                    f(a - base, b - base);
                }
                base += rows;
                if base >= e {
                    break;
                }
            }
        });
    }

    /// Run `f(worker_idx, n_participants)` on every worker AND the
    /// calling thread (`worker_idx = n_workers()` for the caller);
    /// returns when all participants have finished.
    pub fn run(&self, f: &(dyn Fn(usize, usize) + Sync)) {
        #[cfg(target_os = "linux")]
        numa::adopt_caller();
        DISPATCHES.fetch_add(1, Ordering::Relaxed);
        self.dispatch(f, self.threads.len());
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::SeqCst);
        for t in &self.threads {
            t.unpark();
        }
        for h in self.joins.drain(..) {
            let _ = h.join();
        }
    }
}

/// NUMA placement for the pool (see `Pool::bind_numa`).
#[cfg(target_os = "linux")]
mod numa {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// The published mask; `EPOCH` bumps on every publish so a calling
    /// thread re-adopts at most once per bind.
    static MASK: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    static EPOCH: AtomicUsize = AtomicUsize::new(0);
    thread_local! {
        static SEEN: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    pub(super) fn publish(cpus: Vec<usize>) {
        if let Ok(mut m) = MASK.lock() {
            *m = cpus;
        }
        EPOCH.fetch_add(1, Ordering::Release);
    }

    /// One relaxed load + one TLS read per dispatch when nothing changed.
    #[inline]
    pub(super) fn adopt_caller() {
        let e = EPOCH.load(Ordering::Acquire);
        if e == 0 || SEEN.with(|c| c.get()) == e {
            return;
        }
        SEEN.with(|c| c.set(e));
        if let Ok(m) = MASK.lock() {
            if !m.is_empty() {
                set_affinity(0, &m);
            }
        }
    }

    pub(super) fn parse_list(s: &str) -> Vec<usize> {
        let mut out = Vec::new();
        for part in s.trim().split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            match part.split_once('-') {
                Some((a, b)) => {
                    if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                        out.extend(a..=b);
                    }
                }
                None => {
                    if let Ok(a) = part.parse() {
                        out.push(a);
                    }
                }
            }
        }
        out
    }

    fn nodes() -> Vec<(usize, Vec<usize>)> {
        let mut v = Vec::new();
        let Ok(rd) = std::fs::read_dir("/sys/devices/system/node") else {
            return v;
        };
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            let Some(id) = name.strip_prefix("node").and_then(|x| x.parse::<usize>().ok()) else {
                continue;
            };
            if let Ok(l) = std::fs::read_to_string(e.path().join("cpulist")) {
                let cpus = parse_list(&l);
                if !cpus.is_empty() {
                    v.push((id, cpus));
                }
            }
        }
        v.sort();
        v
    }

    fn allowed() -> Vec<usize> {
        // SAFETY: plain syscall into a zeroed, correctly sized set.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
                return Vec::new();
            }
            (0..libc::CPU_SETSIZE as usize)
                .filter(|&c| libc::CPU_ISSET(c, &set))
                .collect()
        }
    }

    /// First SMT sibling of its core (or no topology info: count it).
    fn primary(cpu: usize) -> bool {
        let p = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
        match std::fs::read_to_string(p) {
            Ok(l) => parse_list(&l).first().is_none_or(|&f| f == cpu),
            Err(_) => true,
        }
    }

    /// Where the weights live: (pages sampled, sampled pages in the page
    /// cache, mapped pages per node). The sample is ≤ 4096 pages spread
    /// over `regions`; every sampled page that is already cached is mapped
    /// here with one read (a minor fault — `mincore` says it is cached, so
    /// no disk I/O), because both node queries below only see pages mapped
    /// into THIS process. Per-node counts come from `move_pages` in query
    /// mode, or — where a container's seccomp profile refuses that syscall
    /// (EPERM on the RunPod image) — from `/proc/self/numa_maps` for the
    /// mappings that hold the regions.
    fn page_nodes(regions: &[&[u8]]) -> (usize, usize, Vec<usize>) {
        const PAGE: usize = 4096;
        let total: usize = regions.iter().map(|r| r.len() / PAGE).sum();
        if total == 0 {
            return (0, 0, Vec::new());
        }
        let stride = total.div_ceil(4096).max(1);
        let mut pages: Vec<*mut libc::c_void> = Vec::new();
        for r in regions {
            let base = (r.as_ptr() as usize).div_ceil(PAGE) * PAGE;
            let end = r.as_ptr() as usize + r.len();
            let mut a = base;
            while a + PAGE <= end {
                pages.push(a as *mut libc::c_void);
                a += PAGE * stride;
            }
        }
        let mut incore = 0usize;
        for &p in &pages {
            let mut vec = 0u8;
            // SAFETY: `p` is a page-aligned address inside a live mapping.
            let cached = unsafe { libc::mincore(p, PAGE, &mut vec) } == 0 && vec & 1 == 1;
            if cached {
                incore += 1;
                // SAFETY: readable mapped byte; volatile so it is not elided.
                unsafe { std::ptr::read_volatile(p as *const u8) };
            }
        }
        let mut status = vec![-1i32; pages.len()];
        // SAFETY: query-only move_pages on our own mappings; `nodes` is
        // NULL so nothing moves, `status` has one slot per page.
        let rc = unsafe {
            libc::syscall(
                libc::SYS_move_pages,
                0,
                pages.len() as libc::c_ulong,
                pages.as_mut_ptr(),
                std::ptr::null::<libc::c_int>(),
                status.as_mut_ptr(),
                0,
            )
        };
        let mut by = Vec::new();
        if rc == 0 {
            for &st in &status {
                if st >= 0 {
                    let n = st as usize;
                    if by.len() <= n {
                        by.resize(n + 1, 0);
                    }
                    by[n] += 1;
                }
            }
        } else {
            by = numa_maps_nodes(regions);
        }
        (pages.len(), incore, by)
    }

    /// Mapped pages per node of every mapping that overlaps `regions`,
    /// from `/proc/self/maps` (ranges) + `/proc/self/numa_maps` (`N<k>=`).
    fn numa_maps_nodes(regions: &[&[u8]]) -> Vec<usize> {
        let (Ok(maps), Ok(nm)) = (
            std::fs::read_to_string("/proc/self/maps"),
            std::fs::read_to_string("/proc/self/numa_maps"),
        ) else {
            return Vec::new();
        };
        let spans: Vec<(usize, usize)> = regions
            .iter()
            .map(|r| (r.as_ptr() as usize, r.as_ptr() as usize + r.len()))
            .collect();
        let mut starts = std::collections::HashSet::new();
        for line in maps.lines() {
            let Some((range, _)) = line.split_once(' ') else {
                continue;
            };
            let Some((a, b)) = range.split_once('-') else {
                continue;
            };
            let (Ok(a), Ok(b)) = (usize::from_str_radix(a, 16), usize::from_str_radix(b, 16)) else {
                continue;
            };
            if spans.iter().any(|&(s, e)| s < b && a < e) {
                starts.insert(a);
            }
        }
        let mut by = Vec::new();
        for line in nm.lines() {
            let mut it = line.split_whitespace();
            let Some(a) = it.next().and_then(|a| usize::from_str_radix(a, 16).ok()) else {
                continue;
            };
            if !starts.contains(&a) {
                continue;
            }
            for f in it {
                let Some((k, v)) = f.split_once('=') else {
                    continue;
                };
                let (Some(n), Ok(v)) = (
                    k.strip_prefix('N').and_then(|n| n.parse::<usize>().ok()),
                    v.parse::<usize>(),
                ) else {
                    continue;
                };
                if by.len() <= n {
                    by.resize(n + 1, 0);
                }
                by[n] += v;
            }
        }
        by
    }

    /// (node, cpu mask) for a pool of `threads` participants, or None.
    pub(super) fn choose(regions: &[&[u8]], threads: usize) -> Option<(usize, Vec<usize>)> {
        let env = std::env::var("CMF_NUMA").ok();
        if matches!(env.as_deref(), Some("0") | Some("off")) {
            return None;
        }
        let trace = std::env::var("CMF_NUMA_TRACE").is_ok_and(|v| v != "0");
        let nodes = nodes();
        if nodes.len() < 2 {
            if trace {
                eprintln!("numa: {} node(s) visible — nothing to bind", nodes.len());
            }
            return None;
        }
        // `CMF_NUMA=node:<n>` forces a node (plain "0" means OFF).
        let forced = env
            .as_deref()
            .and_then(|v| v.strip_prefix("node:"))
            .and_then(|v| v.parse::<usize>().ok());
        let node = match forced {
            Some(n) => n,
            None => {
                // Auto: only when the weights already sit on ONE node
                // (≥ 90% of the resident sample, and most of the sample
                // resident). A file spread over both nodes is better
                // served by both sockets; a cold file has no home yet.
                let (sampled, incore, by) = page_nodes(regions);
                let resident: usize = by.iter().sum();
                if trace {
                    eprintln!(
                        "numa: sampled {sampled} weight pages, {incore} cached, mapped by node {by:?}"
                    );
                }
                if sampled == 0 || incore * 2 < sampled || resident == 0 {
                    return None;
                }
                let (n, &cnt) = by.iter().enumerate().max_by_key(|(_, c)| **c)?;
                if cnt * 10 < resident * 9 {
                    return None;
                }
                n
            }
        };
        let cpus = &nodes.iter().find(|(id, _)| *id == node)?.1;
        let allowed = allowed();
        let usable: Vec<usize> = cpus.iter().copied().filter(|c| allowed.contains(c)).collect();
        let prim: Vec<usize> = usable.iter().copied().filter(|&c| primary(c)).collect();
        if prim.len() >= threads {
            Some((node, prim))
        } else if usable.len() >= threads {
            Some((node, usable))
        } else {
            None
        }
    }

    /// Bind thread `tid` (0 = the calling thread) to `cpus`.
    pub(super) fn set_affinity(tid: i32, cpus: &[usize]) -> bool {
        // SAFETY: plain syscall with a zeroed, correctly sized set.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            for &c in cpus {
                if c < libc::CPU_SETSIZE as usize {
                    libc::CPU_SET(c, &mut set);
                }
            }
            libc::sched_setaffinity(tid, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0
        }
    }
}

/// Per-core capacity: the kernel's `cpu_capacity` (µarch × clock) when
/// EAS exposes it, else `cpufreq/cpuinfo_max_freq` — same cluster
/// ordering, so the 62.5% big-core rule keeps working on EAS-less
/// kernels (TUNING.md open item: pinning silently did nothing there).
#[cfg(any(
    target_os = "android",
    all(target_arch = "aarch64", target_os = "linux")
))]
fn core_capacities() -> Vec<u64> {
    let read_all = |leaf: &str| -> Vec<u64> {
        let mut vals = Vec::new();
        for cpu in 0.. {
            let path = format!("/sys/devices/system/cpu/cpu{cpu}/{leaf}");
            match std::fs::read_to_string(&path) {
                Ok(v) => match v.trim().parse() {
                    Ok(x) => vals.push(x),
                    Err(_) => break,
                },
                Err(_) => break,
            }
        }
        vals
    };
    let caps = read_all("cpu_capacity");
    if caps.len() >= 2 {
        return caps;
    }
    read_all("cpufreq/cpuinfo_max_freq")
}

#[cfg(target_os = "android")]
fn pin_thread_to_big_cores() {
    use std::mem;
    let caps = core_capacities();
    let max = caps.iter().copied().max().unwrap_or(0);
    let min = caps.iter().copied().min().unwrap_or(0);

    // Only pin if heterogeneous
    if caps.len() < 2 || max == min {
        return;
    }

    unsafe {
        let mut set: libc::cpu_set_t = mem::zeroed();
        for (i, &c) in caps.iter().enumerate() {
            if c * 8 >= max * 5 {
                libc::CPU_SET(i, &mut set);
            }
        }
        libc::sched_setaffinity(0, mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

fn worker_loop(inner: &Inner, idx: usize) {
    #[cfg(target_os = "android")]
    pin_thread_to_big_cores();
    // Apple silicon: ask for the performance cores. Threads spawned
    // without a QoS class land on the efficiency cores when the
    // scheduler feels like it — a user's video-VAE encode on an M4 sat
    // on the E-cores at 100% with the P-cores asleep for 140 s (HF
    // discussion #4). USER_INITIATED is the class an interactive tool's
    // work belongs to; the ~4 P-cores then take the pool's grains.
    #[cfg(target_os = "macos")]
    unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INITIATED, 0);
    }

    let slot = &inner.slots[idx];
    // Baselines are the construction values (0), never a fresh read: if
    // the caller hands out a ticket before the OS actually starts this
    // thread, adopting the live value as "already seen" would skip that
    // job and deadlock the caller's wait.
    let mut seen = 0usize;
    let mut seen_epoch = 0usize;
    loop {
        // Wait for a ticket: spin first (decode publishes the next matvec
        // within microseconds), park only when idle for real.
        let mut spins = 0usize;
        loop {
            let t = slot.ticket.load(Ordering::Acquire);
            if t != seen {
                seen = t;
                break;
            }
            if inner.shutdown.load(Ordering::Relaxed) {
                return;
            }
            if !inner.oversubscribed {
                // Any job — even one this worker sits out — means the
                // pool is busy: stay hot for the next one.
                let e = inner.epoch.load(Ordering::Relaxed);
                if e != seen_epoch {
                    seen_epoch = e;
                    spins = 0;
                }
            }
            if spins < inner.spin_budget.load(Ordering::Relaxed) {
                spins += 1;
                if inner.oversubscribed && spins.is_multiple_of(64) {
                    std::thread::yield_now();
                } else {
                    std::hint::spin_loop();
                }
            } else {
                slot.parked.store(true, Ordering::Relaxed);
                // Pairs with the caller's fence between writing tickets
                // and reading `parked`: either it sees our flag (and
                // unparks) or we see its ticket here — a missed wakeup is
                // impossible. Spurious unparks just loop.
                std::sync::atomic::fence(Ordering::SeqCst);
                if slot.ticket.load(Ordering::Relaxed) == seen
                    && !inner.shutdown.load(Ordering::Relaxed)
                {
                    std::thread::park();
                }
                slot.parked.store(false, Ordering::Relaxed);
            }
        }
        // The ticket's Acquire made the descriptor visible, and the caller
        // cannot rewrite it before our decrement below: it waits for
        // `remaining`, which counts this ticket.
        let data = inner.desc_data.load(Ordering::Relaxed);
        let vtable = inner.desc_vtable.load(Ordering::Relaxed);
        let n = inner.desc_n.load(Ordering::Relaxed);
        let dev = inner.desc_dev.load(Ordering::Relaxed);
        // SAFETY: the two words are the fat pointer `dispatch` split, and
        // the caller of this job is blocked on our decrement below, so the
        // closure it borrows is alive for the whole call.
        let task: *const (dyn Fn(usize, usize) + Sync + 'static) =
            unsafe { std::mem::transmute([data, vtable]) };
        let f = unsafe { &*task };
        crate::gpu::set_current_device(dev);
        f(idx, n);
        inner.remaining.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Row-parallel dense matvec: `out[o] = Σ_j w[o·in + j]·x[j]`.
/// Bit-identical to the serial loop (row order does not change math).
pub fn matvec_rows(pool: Option<&Pool>, w: &[f32], x: &[f32], out: &mut [f32]) {
    let in_dim = x.len();
    let out_dim = out.len();
    debug_assert!(w.len() >= out_dim * in_dim);

    let row_dot = |o: usize| -> f32 {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        let mut sum = 0.0f32;
        for j in 0..in_dim {
            sum += row[j] * x[j];
        }
        sum
    };

    let out_addr = SendMut(out.as_mut_ptr());
    let run_range = move |start: usize, end: usize| {
        let mut o = start;
        // Four independent reduction chains hide add latency and reuse x.
        // Each row still sums j=0..in_dim in exactly the scalar order: no
        // horizontal SIMD reduction, FMA, or quantization approximation.
        while end - o >= 4 {
            let base = o * in_dim;
            let w0 = &w[base..base + in_dim];
            let w1 = &w[base + in_dim..base + 2 * in_dim];
            let w2 = &w[base + 2 * in_dim..base + 3 * in_dim];
            let w3 = &w[base + 3 * in_dim..base + 4 * in_dim];
            let (mut a, mut b, mut c, mut d) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            for j in 0..in_dim {
                let v = x[j];
                a += w0[j] * v;
                b += w1[j] * v;
                c += w2[j] * v;
                d += w3[j] * v;
            }
            // run_rows assigns disjoint ranges within 0..out_dim.
            unsafe {
                *out_addr.at(o) = a;
                *out_addr.at(o + 1) = b;
                *out_addr.at(o + 2) = c;
                *out_addr.at(o + 3) = d;
            }
            o += 4;
        }
        for o in o..end {
            unsafe { *out_addr.at(o) = row_dot(o) };
        }
    };
    match pool {
        Some(pool) if out_dim >= 256 => pool.run_rows(out_dim, &run_range),
        _ => run_range(0, out_dim),
    }
}

/// Two-input row matvec: one pass over the weight rows serves BOTH
/// inputs — CPU decode is memory-bound, so the second position costs a
/// fraction of the first (this is where MTP speculative verify wins).
/// Per-output accumulation order matches the single-input path exactly
/// → bit-identical results.
pub fn matvec_rows2(
    pool: Option<&Pool>,
    w: &[f32],
    x1: &[f32],
    x2: &[f32],
    out1: &mut [f32],
    out2: &mut [f32],
) {
    let in_dim = x1.len();
    debug_assert_eq!(x2.len(), in_dim);
    let out_dim = out1.len();
    debug_assert_eq!(out2.len(), out_dim);
    debug_assert!(w.len() >= out_dim * in_dim);

    let row_dots = |o: usize| -> (f32, f32) {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        let (mut s1, mut s2) = (0.0f32, 0.0f32);
        for j in 0..in_dim {
            s1 += row[j] * x1[j];
            s2 += row[j] * x2[j];
        }
        (s1, s2)
    };

    match pool {
        Some(pool) if out_dim >= 256 => {
            let o1 = SendMut(out1.as_mut_ptr());
            let o2 = SendMut(out2.as_mut_ptr());
            let run_range = move |start: usize, end: usize| {
                for o in start..end {
                    let (s1, s2) = row_dots(o);
                    unsafe {
                        *o1.at(o) = s1;
                        *o2.at(o) = s2;
                    }
                }
            };
            pool.run_rows(out_dim, &run_range);
        }
        _ => {
            for o in 0..out_dim {
                let (s1, s2) = row_dots(o);
                out1[o] = s1;
                out2[o] = s2;
            }
        }
    }
}

/// `SendMut` for any element type — the sampler's sparse chain writes
/// per-grain candidate lists.
pub(crate) struct SendMutT<T>(*mut T);
unsafe impl<T> Send for SendMutT<T> {}
unsafe impl<T> Sync for SendMutT<T> {}
impl<T> Clone for SendMutT<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SendMutT<T> {}
impl<T> SendMutT<T> {
    #[inline]
    pub(crate) fn new(p: *mut T) -> Self {
        Self(p)
    }
    /// Same contract as `SendMut::at`: disjoint indices, pointee outlives
    /// the joined dispatch.
    #[inline]
    pub(crate) fn at(self, i: usize) -> *mut T {
        unsafe { self.0.add(i) }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct SendMut(*mut f32);
unsafe impl Send for SendMut {}
unsafe impl Sync for SendMut {}

impl SendMut {
    /// The caller promises the threads it hands this to write disjoint
    /// indices, and that the pointee outlives them.
    #[inline]
    pub(crate) fn new(p: *mut f32) -> Self {
        Self(p)
    }

    /// Method receiver forces the closure to capture the whole (Sync)
    /// wrapper, not the bare `*mut f32` field (edition-2021 precise capture).
    #[inline]
    pub(crate) fn at(self, i: usize) -> *mut f32 {
        unsafe { self.0.add(i) }
    }
}

#[cfg(test)]
mod tests {
    /// A worker NOT invited to a limited job is not waited for; if it is
    /// preempted around the moment the job is published, the caller may
    /// already be on the next job. The first protocol then ran that next
    /// job twice and wrapped `remaining` — the caller spun forever.
    /// Oversubscribe the machine with spinning workers, hammer limited
    /// dispatches, and count every closure entry: each job must run
    /// exactly (limit + 1) times, and the loop must finish (a watchdog
    /// turns a hang into a failure). On a 2–4-CPU box the second protocol
    /// ran ~100 dispatches/s here (every spinner woke for every job and
    /// held the CPUs the invited worker needed) and hit the watchdog at
    /// half the jobs — slow, not wrong, but a pool that needs a time
    /// slice per dispatch is broken too.
    #[test]
    fn uninvited_straggler_never_runs_a_job_twice_or_wraps_the_barrier() {
        use std::sync::atomic::AtomicUsize;
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let workers = (cores * 4).clamp(8, 64);
        let pool = Arc::new(Pool::with_spin(workers, 1_000_000));
        let entries = Arc::new(AtomicUsize::new(0));
        let expected = Arc::new(AtomicUsize::new(0));
        let progress = Arc::new(AtomicUsize::new(0));
        let done = Arc::new(AtomicBool::new(false));
        let iters = 20_000usize;
        let p = pool.clone();
        let (en, ex, pr, dn) = (
            entries.clone(),
            expected.clone(),
            progress.clone(),
            done.clone(),
        );
        let driver = std::thread::spawn(move || {
            for i in 0..iters {
                pr.store(i, Ordering::Relaxed);
                // Few grains → `run_limited` with limit < workers: most
                // workers are uninvited.  Alternate with a full `run`.
                let limit = 1 + i % 3;
                let f = |_w: usize, _n: usize| {
                    en.fetch_add(1, Ordering::Relaxed);
                };
                p.run_limited(limit, &f);
                ex.fetch_add(limit.min(p.n_workers()) + 1, Ordering::Relaxed);
                if i % 97 == 0 {
                    p.run(&f);
                    ex.fetch_add(p.n_workers() + 1, Ordering::Relaxed);
                }
            }
            dn.store(true, Ordering::SeqCst);
        });
        let t0 = std::time::Instant::now();
        while !done.load(Ordering::SeqCst) {
            assert!(
                t0.elapsed() < std::time::Duration::from_secs(120),
                "pool dispatch loop did not finish: at iteration {} of {}, entries {} \
                 expected {} (the difference is the job in flight), remaining {}",
                progress.load(Ordering::Relaxed),
                iters,
                entries.load(Ordering::Relaxed),
                expected.load(Ordering::Relaxed),
                pool.inner.remaining.0.load(Ordering::Relaxed)
            );
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        driver.join().unwrap();
        // A late straggler could still be inside its (single) job; one
        // more full barrier drains it.
        pool.run(&|_w, _n| {});
        assert_eq!(
            entries.load(Ordering::Relaxed),
            expected.load(Ordering::Relaxed),
            "some job ran a closure more or fewer times than its participants"
        );
    }

    /// Participation is decided by the ticket alone: a limited job hands
    /// one ticket to each of workers `0..limit` and to nobody else, a full
    /// `run` one to every worker. Spin 0 also drives the park/unpark path.
    #[test]
    fn limited_dispatch_hands_tickets_only_to_invited_workers() {
        let pool = Pool::with_spin(6, 0);
        let hits: Vec<AtomicUsize> = (0..7).map(|_| AtomicUsize::new(0)).collect();
        let bad = AtomicUsize::new(0);
        // No panics inside the job: a worker that dies never decrements.
        let f = |w: usize, n: usize| match hits.get(w) {
            Some(h) if w < n && (n == 3 || n == 7) => {
                h.fetch_add(1, Ordering::Relaxed);
            }
            _ => {
                bad.fetch_add(1, Ordering::Relaxed);
            }
        };
        let tickets = |p: &Pool| -> Vec<usize> {
            p.inner
                .slots
                .iter()
                .map(|s| s.ticket.load(Ordering::Acquire))
                .collect()
        };
        pool.run_limited(2, &f);
        assert_eq!(tickets(&pool), [1, 1, 0, 0, 0, 0]);
        pool.run(&f);
        assert_eq!(tickets(&pool), [2, 2, 1, 1, 1, 1]);
        assert_eq!(
            bad.load(Ordering::Relaxed),
            0,
            "participant index or count off"
        );
        let hits: Vec<usize> = hits.iter().map(|h| h.load(Ordering::Relaxed)).collect();
        // Index 2 is the caller of the limited job and worker 2 of the run.
        assert_eq!(hits, [2, 2, 2, 1, 1, 1, 1]);
    }

    #[test]
    fn f32_four_row_matvec_matches_scalar_bits_and_preserves_tail() {
        let pool = super::Pool::new(3);
        for rows in [0, 1, 3, 4, 7, 255, 256, 259, 1024] {
            for cols in [0, 1, 3, 32, 65, 384] {
                let w: Vec<f32> = (0..rows * cols)
                    .map(|i| (i as f32 * 0.173).sin() * [1e-3, 1.0, 1e3][i % 3])
                    .collect();
                let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.41).cos()).collect();
                let want: Vec<u32> = (0..rows)
                    .map(|r| {
                        let mut sum = 0.0f32;
                        for j in 0..cols {
                            sum += w[r * cols + j] * x[j];
                        }
                        sum.to_bits()
                    })
                    .collect();
                for workers in [None, Some(&pool)] {
                    let mut out = vec![17.0f32; rows + 5];
                    super::matvec_rows(workers, &w, &x, &mut out[..rows]);
                    let bits: Vec<u32> = out[..rows].iter().map(|v| v.to_bits()).collect();
                    assert_eq!(bits, want, "shape {rows}x{cols}");
                    assert_eq!(&out[rows..], &[17.0; 5]);
                }
                // A short output is an intentional public matvec contract.
                let mut prefix = vec![0.0f32; rows / 2];
                super::matvec_rows(None, &w, &x, &mut prefix);
                assert_eq!(
                    prefix.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want[..rows / 2]
                );
            }
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn numa_cpulist_parses_ranges_and_singles() {
        assert_eq!(super::numa::parse_list("0-3,8,10-11\n"), vec![0, 1, 2, 3, 8, 10, 11]);
        assert_eq!(super::numa::parse_list(""), Vec::<usize>::new());
    }

    #[test]
    #[cfg(any(target_os = "android", target_os = "linux"))]
    fn worker_tids_registered_before_new_returns() {
        // WORKER_TIDS is only the last completed pool's process-wide
        // snapshot; another test can publish a different valid snapshot
        // immediately after `new` returns. Check this pool's private
        // registration state instead.
        use std::collections::HashSet;
        let p = super::Pool::new(3);
        let local: Vec<_> = p.inner.worker_tids.lock().unwrap().clone();
        let registered = p.inner.registered.load(Ordering::Acquire);
        let unique: HashSet<_> = local.iter().copied().collect();
        assert!(
            registered == 3
                && local.len() == 3
                && unique.len() == 3
                && local.iter().all(|&tid| tid > 0),
            "all worker tids must be privately registered before new returns \
             (registered {registered}, local {}, unique {})",
            local.len(),
            unique.len()
        );
    }

    #[test]
    fn forced_threads_overrides_env_and_topology() {
        use std::sync::atomic::Ordering;
        super::FORCED_THREADS.store(3, Ordering::Relaxed);
        let pool = super::Pool::from_env().expect("forced 3 → pool");
        assert_eq!(pool.n_workers(), 3);
        super::FORCED_THREADS.store(1, Ordering::Relaxed);
        assert!(super::Pool::from_env().is_none(), "forced 1 → serial");
        super::FORCED_THREADS.store(0, Ordering::Relaxed);
    }

    #[test]
    #[cfg(any(target_os = "android", target_os = "linux"))]
    fn concurrent_pool_constructors_complete_without_registration_race() {
        // WORKER_TIDS is a process-wide publication target. Before the
        // per-pool counter, a larger constructor could have all its workers
        // append, then a concurrent one could clear that vector; the larger
        // constructor would wait forever for a length that could never return.
        // Start unlike-sized constructors together so that regression is
        // exercised without relying on the test harness' scheduling.
        use std::sync::{Barrier, mpsc};
        use std::time::Duration;

        for round in 0..16 {
            let start = Arc::new(Barrier::new(3));
            let (done_tx, done_rx) = mpsc::channel();
            let mut joins = Vec::new();
            for workers in [8usize, 1usize] {
                let start = start.clone();
                let done_tx = done_tx.clone();
                joins.push(std::thread::spawn(move || {
                    start.wait();
                    let pool = Pool::with_spin(workers, 0);
                    done_tx.send(pool.n_workers()).unwrap();
                }));
            }
            drop(done_tx);
            start.wait();
            let mut sizes = Vec::with_capacity(2);
            for _ in 0..2 {
                sizes.push(
                    done_rx
                        .recv_timeout(Duration::from_secs(10))
                        .unwrap_or_else(|_| panic!("pool constructor stalled in round {round}")),
                );
            }
            sizes.sort_unstable();
            assert_eq!(sizes, [1, 8]);
            for join in joins {
                join.join().unwrap();
            }
        }
    }

    #[test]
    fn capacity_split_clock_bins_vs_microarch() {
        type P = super::Pool;
        // JR510: all-A55, two clock bins — use every core.
        assert_eq!(
            P::cores_from_capacities(&[1024, 1024, 1024, 1024, 768, 768, 768, 768]),
            Some(8)
        );
        // Classic big.LITTLE (A78 + A55) — big only.
        assert_eq!(
            P::cores_from_capacities(&[1024, 1024, 1024, 1024, 350, 350, 350, 350]),
            Some(4)
        );
        // Three-tier flagship: X + A7xx mids stay, A5xx littles go.
        assert_eq!(
            P::cores_from_capacities(&[1024, 800, 800, 800, 800, 300, 300, 300]),
            Some(5)
        );
        // Uniform: no signal, caller falls back.
        assert_eq!(P::cores_from_capacities(&[1024; 8]), None);
        assert_eq!(P::cores_from_capacities(&[]), None);
    }

    use super::*;

    #[test]
    fn parallel_matvec_equals_serial_bitexact() {
        let (out_dim, in_dim) = (512, 64);
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| (i as f32 * 0.013).sin())
            .collect();
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.07).cos()).collect();

        let mut serial = vec![0.0f32; out_dim];
        matvec_rows(None, &w, &x, &mut serial);

        let pool = Pool::new(4);
        let mut parallel = vec![0.0f32; out_dim];
        matvec_rows(Some(&pool), &w, &x, &mut parallel);

        assert_eq!(serial, parallel, "row-parallel must be bit-identical");
    }

    #[test]
    fn fused_pair_equals_two_singles_bitexact() {
        let (out_dim, in_dim) = (300, 48);
        let w: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| (i as f32 * 0.011).sin())
            .collect();
        let x1: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.03).cos()).collect();
        let x2: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.09).sin()).collect();

        let mut a1 = vec![0.0f32; out_dim];
        let mut a2 = vec![0.0f32; out_dim];
        matvec_rows(None, &w, &x1, &mut a1);
        matvec_rows(None, &w, &x2, &mut a2);

        for pool in [None, Some(Pool::new(3))] {
            let mut b1 = vec![0.0f32; out_dim];
            let mut b2 = vec![0.0f32; out_dim];
            matvec_rows2(pool.as_ref(), &w, &x1, &x2, &mut b1, &mut b2);
            assert_eq!(a1, b1, "fused lane 1 must be bit-identical");
            assert_eq!(a2, b2, "fused lane 2 must be bit-identical");
        }
    }

    #[test]
    fn pool_survives_many_runs() {
        let pool = Pool::new(3);
        let counter = AtomicUsize::new(0);
        for _ in 0..100 {
            pool.run(&|_, _| {
                counter.fetch_add(1, Ordering::Relaxed);
            });
        }
        // 3 workers + the participating caller = 4 executions per run.
        assert_eq!(counter.load(Ordering::Relaxed), 400);
    }

    #[test]
    fn pool_wakes_after_park() {
        // Force immediate parking (no spin) — the epoch/parked handshake
        // must still never miss a wakeup.
        let pool = Pool::with_spin(2, 0);
        let counter = AtomicUsize::new(0);
        for _ in 0..50 {
            pool.run(&|_, _| {
                counter.fetch_add(1, Ordering::Relaxed);
            });
            // Give workers time to actually park between jobs.
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
        assert_eq!(counter.load(Ordering::Relaxed), 150);
    }

    #[test]
    fn worker_indices_are_distinct_and_cover_range() {
        let pool = Pool::new(3);
        let hits: Vec<AtomicUsize> = (0..4).map(|_| AtomicUsize::new(0)).collect();
        for _ in 0..20 {
            pool.run(&|widx, n| {
                assert_eq!(n, 4);
                hits[widx].fetch_add(1, Ordering::Relaxed);
            });
        }
        for (i, h) in hits.iter().enumerate() {
            assert_eq!(h.load(Ordering::Relaxed), 20, "participant {i} missed runs");
        }
    }
}

#[cfg(test)]
mod grain_tests {
    use super::grain_for;

    #[test]
    fn a_short_job_still_reaches_every_worker() {
        // 24 rows, 49 workers: the old flat floor of 32 handed all 24 to the
        // first worker and woke the rest for nothing.
        assert_eq!(grain_for(24, 49), 1);
        // Wide jobs keep the stride the SDOT loop wants.
        assert_eq!(grain_for(4096, 49), 32);
        assert_eq!(grain_for(32768, 49), 83);
        // Degenerate shapes must not divide by zero or return zero.
        assert_eq!(grain_for(0, 49), 1);
        assert_eq!(grain_for(7, 1), 7);
        assert!(grain_for(1, 49) >= 1);
    }
}
