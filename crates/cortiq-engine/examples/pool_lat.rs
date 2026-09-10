//! Pool dispatch-latency probe (scratch diagnostic, not a product).
//!
//! Times `Pool::run` with an empty closure: the pure cost of waking every
//! worker and collecting the latch. Decode issues one dispatch per matvec
//! (~200/token), so this cost is paid ~200 times per token — compare it
//! against the per-worker work of a typical MLP row-slice.
//!
//! Usage: cargo run --release --example pool_lat

use cortiq_engine::pool::Pool;
use std::time::Instant;

fn main() {
    for nt in [2usize, 4, 6, 8, 10] {
        let pool = Pool::new(nt);
        let noop: &(dyn Fn(usize, usize) + Sync) = &|_w, _n| {};
        // Warm: first dispatch spawns/parks the workers.
        for _ in 0..100 {
            pool.run(noop);
        }
        let iters = 20_000;
        let t0 = Instant::now();
        for _ in 0..iters {
            pool.run(noop);
        }
        let el = t0.elapsed().as_secs_f64();
        let per = el / iters as f64 * 1e6;
        println!(
            "threads={nt:2}  {per:7.2} us/dispatch  -> {:6.2} ms/token at 200 matvecs",
            per * 200.0 / 1e3
        );
    }
}
