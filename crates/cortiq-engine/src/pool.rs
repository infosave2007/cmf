//! Persistent worker pool for row-parallel matvecs.
//!
//! Threads are spawned once and parked between calls — vmfcore measured
//! spawn-per-matvec at ~+27% decode cost versus a persistent pool.
//! Parallelism is by disjoint row ranges, so results are bit-identical
//! to the serial path (each row's dot product is computed the same way).
//!
//! `CMF_THREADS` env: 0/1 = serial, N = worker count
//! (default: available_parallelism − 1, capped at 8).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Condvar, Mutex};

/// A `*const dyn Fn` that may cross a thread boundary. Safety is
/// provided by `Pool::run`: the caller blocks until every worker has
/// finished, so the borrow outlives all uses.
struct TaskPtr(*const (dyn Fn(usize, usize) + Sync));
unsafe impl Send for TaskPtr {}

struct Latch {
    remaining: AtomicUsize,
    lock: Mutex<()>,
    cv: Condvar,
}

impl Latch {
    fn new(n: usize) -> Arc<Self> {
        Arc::new(Self {
            remaining: AtomicUsize::new(n),
            lock: Mutex::new(()),
            cv: Condvar::new(),
        })
    }

    fn count_down(&self) {
        if self.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            let _g = self.lock.lock().unwrap();
            self.cv.notify_all();
        }
    }

    fn wait(&self) {
        let mut g = self.lock.lock().unwrap();
        while self.remaining.load(Ordering::Acquire) != 0 {
            g = self.cv.wait(g).unwrap();
        }
    }
}

struct Job {
    task: TaskPtr,
    worker_idx: usize,
    n_workers: usize,
    latch: Arc<Latch>,
}

/// Persistent thread pool. Workers park on a channel between jobs.
pub struct Pool {
    txs: Vec<Sender<Job>>,
}

impl Pool {
    pub fn new(n_workers: usize) -> Self {
        let mut txs = Vec::with_capacity(n_workers);
        for w in 0..n_workers {
            let (tx, rx) = channel::<Job>();
            std::thread::Builder::new()
                .name(format!("cmf-pool-{w}"))
                .spawn(move || {
                    while let Ok(job) = rx.recv() {
                        // SAFETY: Pool::run blocks on the latch until this
                        // call returns, keeping the closure borrow alive.
                        let f = unsafe { &*job.task.0 };
                        f(job.worker_idx, job.n_workers);
                        job.latch.count_down();
                    }
                })
                .expect("spawn pool worker");
            txs.push(tx);
        }
        Self { txs }
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

    pub fn n_workers(&self) -> usize {
        self.txs.len()
    }

    /// Run `f(row_start, row_end)` over `0..rows`, self-balancing.
    ///
    /// One dispatch, but workers pull row-ranges from a shared cursor
    /// instead of each taking a fixed 1/n slice. On a heterogeneous CPU
    /// (Apple Silicon: 4 P-cores + 6 E-cores here) a static split makes
    /// every matvec end at the SLOWEST core's pace while the fast ones
    /// idle at the latch; pulling by grain lets a P-core take several
    /// chunks for each one an E-core takes, so skew collapses to a
    /// single grain. Row ranges stay disjoint and each row's dot is
    /// computed exactly as in the serial path → bit-identical output.
    pub fn run_rows(&self, rows: usize, f: &(dyn Fn(usize, usize) + Sync)) {
        // Enough chunks to balance, large enough to keep the SDOT inner
        // loop and the hardware prefetcher in their stride.
        let grain = (rows / (self.txs.len() * 8)).max(32);
        let next = AtomicUsize::new(0);
        self.run(&|_w, _n| loop {
            let start = next.fetch_add(grain, Ordering::Relaxed);
            if start >= rows {
                break;
            }
            f(start, (start + grain).min(rows));
        });
    }

    /// Run `f(worker_idx, n_workers)` on every worker; blocks until all
    /// have finished.
    pub fn run(&self, f: &(dyn Fn(usize, usize) + Sync)) {
        let n = self.txs.len();
        let latch = Latch::new(n);
        // SAFETY: `wait()` below blocks until every worker is done, so
        // extending the borrow to 'static never outlives the call.
        let ptr: *const (dyn Fn(usize, usize) + Sync) = f;
        let ptr: *const (dyn Fn(usize, usize) + Sync + 'static) =
            unsafe { std::mem::transmute(ptr) };
        for (i, tx) in self.txs.iter().enumerate() {
            let job = Job {
                task: TaskPtr(ptr),
                worker_idx: i,
                n_workers: n,
                latch: latch.clone(),
            };
            tx.send(job).expect("pool worker died");
        }
        latch.wait();
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
        // Small outputs are not worth the latch round-trip.
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
        assert_eq!(counter.load(Ordering::Relaxed), 300);
    }
}
