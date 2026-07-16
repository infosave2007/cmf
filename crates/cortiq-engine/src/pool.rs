//! Persistent worker pool for row-parallel matvecs.
//!
//! Threads are spawned once and spin-then-park between calls — vmfcore
//! measured spawn-per-matvec at ~+27% decode cost versus a persistent
//! pool. Parallelism is by disjoint row ranges, so results are
//! bit-identical to the serial path (each row's dot product is computed
//! the same way).
//!
//! Dispatch is a single shared job slot + atomic epoch (roadmap §3 P0):
//! the caller publishes one pointer, bumps the epoch and JOINS THE WORK
//! as the extra worker instead of blocking on a latch. The previous
//! design allocated an `Arc<Latch>` and pushed a message into every
//! worker's mpsc channel for every matvec (~200 dispatches/token) —
//! with decode-grade matvecs that synchronization was its own budget.
//! Workers spin for `CMF_POOL_SPIN` iterations before parking.
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

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// A `*const dyn Fn` that may cross a thread boundary. Safety is
/// provided by `Pool::run`: the caller blocks until every worker has
/// finished, so the borrow outlives all uses.
#[derive(Clone, Copy)]
struct TaskPtr(*const (dyn Fn(usize, usize) + Sync));
unsafe impl Send for TaskPtr {}

struct Inner {
    /// Bumped once per published job; workers watch it.
    epoch: AtomicUsize,
    /// Workers still running the current job (excludes the caller).
    remaining: AtomicUsize,
    /// The published job: closure pointer + total participant count.
    /// Written by the caller BEFORE the epoch bump, read by workers
    /// AFTER they observe the new epoch (acquire/release pairing).
    slot: UnsafeCell<Option<(TaskPtr, usize)>>,
    shutdown: AtomicBool,
    /// Spin iterations before a worker parks (0 = park immediately).
    spin_budget: usize,
    /// Per-worker "I am parked" flags — lets the caller skip the unpark
    /// syscall for workers that are still spinning.
    parked: Box<[AtomicBool]>,
}

// SAFETY: `slot` is only written while no job is in flight (run()
// returns after `remaining` hits 0) and only read after the epoch
// publication that follows the write.
unsafe impl Sync for Inner {}

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

impl Pool {
    pub fn new(n_workers: usize) -> Self {
        Self::with_spin(n_workers, spin_budget_from_env())
    }

    /// Explicit spin budget (tests pin it without touching the env).
    pub fn with_spin(n_workers: usize, spin_budget: usize) -> Self {
        let inner = Arc::new(Inner {
            epoch: AtomicUsize::new(0),
            remaining: AtomicUsize::new(0),
            slot: UnsafeCell::new(None),
            shutdown: AtomicBool::new(false),
            spin_budget,
            parked: (0..n_workers).map(|_| AtomicBool::new(false)).collect(),
        });
        let mut joins = Vec::with_capacity(n_workers);
        for w in 0..n_workers {
            let inner = inner.clone();
            let h = std::thread::Builder::new()
                .name(format!("cmf-pool-{w}"))
                .spawn(move || worker_loop(&inner, w))
                .expect("spawn pool worker");
            joins.push(h);
        }
        let threads = joins.iter().map(|h| h.thread().clone()).collect();
        Self { inner, threads, joins }
    }

    /// Pool sized from `CMF_THREADS` (see module docs). `None` = serial.
    pub fn from_env() -> Option<Arc<Self>> {
        let n = match std::env::var("CMF_THREADS") {
            Ok(v) => v.parse::<usize>().unwrap_or(0),
            Err(_) => {
                let avail = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1);
                avail.saturating_sub(1).min(8)
            }
        };
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
        // Enough chunks to balance, large enough to keep the SDOT inner
        // loop and the hardware prefetcher in their stride.
        let grain = (rows / ((self.threads.len() + 1) * 8)).max(32);
        let next = AtomicUsize::new(0);
        self.run(&|_w, _n| loop {
            let start = next.fetch_add(grain, Ordering::Relaxed);
            if start >= rows {
                break;
            }
            f(start, (start + grain).min(rows));
        });
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
        let grain = (total / ((self.threads.len() + 1) * 8)).max(32);
        let next = AtomicUsize::new(0);
        self.run(&|_w, _n| loop {
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
        DISPATCHES.fetch_add(1, Ordering::Relaxed);
        let nw = self.threads.len();
        let n = nw + 1; // caller participates
        // SAFETY: the wait loop below blocks until every worker is done,
        // so extending the borrow to 'static never outlives the call.
        let ptr: *const (dyn Fn(usize, usize) + Sync) = f;
        let ptr: *const (dyn Fn(usize, usize) + Sync + 'static) =
            unsafe { std::mem::transmute(ptr) };
        // SAFETY: no job in flight (previous run() drained `remaining`),
        // so the slot is not being read.
        unsafe { *self.inner.slot.get() = Some((TaskPtr(ptr), n)) };
        self.inner.remaining.store(nw, Ordering::Relaxed);
        self.inner.epoch.fetch_add(1, Ordering::SeqCst);
        for (i, t) in self.threads.iter().enumerate() {
            if self.inner.parked[i].load(Ordering::SeqCst) {
                t.unpark();
            }
        }

        // The caller's share — the barrier costs nothing while there is
        // real work to do.
        f(nw, n);

        // Wait for the stragglers (bounded by one worker's chunk).
        let mut spins = 0usize;
        while self.inner.remaining.load(Ordering::Acquire) != 0 {
            spins += 1;
            if spins < 10_000 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
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

fn worker_loop(inner: &Inner, idx: usize) {
    // The pool is created at epoch 0; baseline MUST be 0, not a fresh
    // epoch read — if the caller publishes a job before the OS actually
    // starts this thread, reading the live epoch would adopt that job's
    // epoch as "already seen", skip it, and deadlock the caller's wait.
    let mut seen = 0usize;
    loop {
        // Wait for a new epoch: spin first (decode publishes the next
        // matvec within microseconds), park only when idle for real.
        let mut spins = 0usize;
        loop {
            let e = inner.epoch.load(Ordering::Acquire);
            if e != seen {
                seen = e;
                break;
            }
            if inner.shutdown.load(Ordering::Relaxed) {
                return;
            }
            if spins < inner.spin_budget {
                spins += 1;
                std::hint::spin_loop();
            } else {
                inner.parked[idx].store(true, Ordering::SeqCst);
                // Re-check under SeqCst: the caller bumps the epoch
                // BEFORE reading `parked`, so either it sees our flag
                // (and unparks) or we see its epoch here — a missed
                // wakeup is impossible. Spurious unparks just loop.
                if inner.epoch.load(Ordering::SeqCst) == seen
                    && !inner.shutdown.load(Ordering::Relaxed)
                {
                    std::thread::park();
                }
                inner.parked[idx].store(false, Ordering::SeqCst);
            }
        }
        // SAFETY: the slot was written before the epoch bump we just
        // observed (release/acquire), and stays valid until `remaining`
        // drops to zero — which happens only after `f` returns below.
        let (task, n) = unsafe { (*inner.slot.get()).expect("job published with epoch") };
        let f = unsafe { &*task.0 };
        f(idx, n);
        inner.remaining.fetch_sub(1, Ordering::AcqRel);
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

    match pool {
        // Small outputs are not worth the barrier round-trip.
        Some(pool) if out_dim >= 256 => {
            let out_addr = SendMut(out.as_mut_ptr());
            pool.run(&move |widx, n| {
                let chunk = out_dim.div_ceil(n);
                let start = widx * chunk;
                let end = (start + chunk).min(out_dim);
                for o in start..end {
                    // SAFETY: workers write disjoint index ranges.
                    unsafe { *out_addr.at(o) = row_dot(o) };
                }
            });
        }
        _ => {
            for (o, dst) in out.iter_mut().enumerate() {
                *dst = row_dot(o);
            }
        }
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
            pool.run(&move |widx, n| {
                let chunk = out_dim.div_ceil(n);
                let start = widx * chunk;
                let end = (start + chunk).min(out_dim);
                for o in start..end {
                    let (s1, s2) = row_dots(o);
                    // SAFETY: workers write disjoint index ranges.
                    unsafe {
                        *o1.at(o) = s1;
                        *o2.at(o) = s2;
                    }
                }
            });
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

#[derive(Clone, Copy)]
struct SendMut(*mut f32);
unsafe impl Send for SendMut {}
unsafe impl Sync for SendMut {}

impl SendMut {
    /// Method receiver forces the closure to capture the whole (Sync)
    /// wrapper, not the bare `*mut f32` field (edition-2021 precise capture).
    #[inline]
    fn at(self, i: usize) -> *mut f32 {
        unsafe { self.0.add(i) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_matvec_equals_serial_bitexact() {
        let (out_dim, in_dim) = (512, 64);
        let w: Vec<f32> = (0..out_dim * in_dim).map(|i| (i as f32 * 0.013).sin()).collect();
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
        let w: Vec<f32> = (0..out_dim * in_dim).map(|i| (i as f32 * 0.011).sin()).collect();
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
