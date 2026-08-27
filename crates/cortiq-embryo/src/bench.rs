//! Real TFLOPS of OUR training GEMM on this machine — the number the
//! birth budget (docs §4.1) is computed from. Reports GPU time from the
//! command buffer's own timestamps, not wall clock.

use crate::metal::{Cmd, GBuf, Op, ctx};
use crate::ops::lcg_vec;

pub struct Shape {
    pub name: &'static str,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub ta: Op,
    pub tb: Op,
}

/// Embryo-0 training shapes at B·T = 16384 (16 × 1024), all three
/// orientations of the hottest layers, plus a square peak probe.
pub fn embryo_shapes() -> Vec<Shape> {
    let bt = 16384;
    vec![
        Shape {
            name: "peak 4096³ NN",
            m: 4096,
            n: 4096,
            k: 4096,
            ta: Op::N,
            tb: Op::N,
        },
        Shape {
            name: "peak 4096³ NT",
            m: 4096,
            n: 4096,
            k: 4096,
            ta: Op::N,
            tb: Op::T,
        },
        Shape {
            name: "peak 4096³ TN",
            m: 4096,
            n: 4096,
            k: 4096,
            ta: Op::T,
            tb: Op::N,
        },
        Shape {
            name: "ffn up  y=x·Wᵀ  [bt×384]·[768×384]ᵀ",
            m: bt,
            n: 768,
            k: 384,
            ta: Op::N,
            tb: Op::T,
        },
        Shape {
            name: "ffn up  dx=dy·W [bt×768]·[768×384]",
            m: bt,
            n: 384,
            k: 768,
            ta: Op::N,
            tb: Op::N,
        },
        Shape {
            name: "ffn up  dW=dyᵀ·x [768×bt]·[bt×384]",
            m: 768,
            n: 384,
            k: bt,
            ta: Op::T,
            tb: Op::N,
        },
        Shape {
            name: "ffn down y=h·Wᵀ [bt×768]·[384×768]ᵀ",
            m: bt,
            n: 384,
            k: 768,
            ta: Op::N,
            tb: Op::T,
        },
        Shape {
            name: "mixer qkv [bt×384]·[1536×384]ᵀ",
            m: bt,
            n: 1536,
            k: 384,
            ta: Op::N,
            tb: Op::T,
        },
        Shape {
            name: "mixer out [bt×1024]·[384×1024]ᵀ",
            m: bt,
            n: 384,
            k: 1024,
            ta: Op::N,
            tb: Op::T,
        },
        Shape {
            name: "lm_head full [bt×384]·[32768×384]ᵀ",
            m: bt,
            n: 32768,
            k: 384,
            ta: Op::N,
            tb: Op::T,
        },
        Shape {
            name: "lm_head dE  [32768×bt]·[bt×384]",
            m: 32768,
            n: 384,
            k: bt,
            ta: Op::T,
            tb: Op::N,
        },
        Shape {
            name: "hier head  [bt×384]·[256×384]ᵀ",
            m: bt,
            n: 256,
            k: 384,
            ta: Op::N,
            tb: Op::T,
        },
    ]
}

pub struct BenchRow {
    pub name: &'static str,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub gpu_ms: f64,
    pub tflops: f64,
    pub max_abs_err: f32,
}

/// Run each shape `reps` times inside ONE command buffer (amortises the
/// submit), verify a sampled corner against the CPU reference, report.
pub fn run(reps: usize, verify: bool) -> Option<Vec<BenchRow>> {
    let c = ctx()?;
    let mut rows = Vec::new();
    for s in embryo_shapes() {
        let (arows, acols) = if s.ta == Op::N {
            (s.m, s.k)
        } else {
            (s.k, s.m)
        };
        let (brows, bcols) = if s.tb == Op::N {
            (s.k, s.n)
        } else {
            (s.n, s.k)
        };
        let a = lcg_vec(1, arows * acols);
        let b = lcg_vec(2, brows * bcols);
        let ga = GBuf::from_slice(c, &a);
        let gb = GBuf::from_slice(c, &b);
        let gc = GBuf::zeros(c, s.m * s.n);
        // warm-up (pipeline + page-in)
        {
            let cmd = Cmd::new(c);
            cmd.gemm(
                s.ta, s.tb, s.m, s.n, s.k, 1.0, &ga, 0, acols, &gb, 0, bcols, 0.0, &gc, 0, s.n,
            );
            cmd.commit();
        }
        let cmd = Cmd::new(c);
        for _ in 0..reps {
            cmd.gemm(
                s.ta, s.tb, s.m, s.n, s.k, 1.0, &ga, 0, acols, &gb, 0, bcols, 0.0, &gc, 0, s.n,
            );
        }
        let ms = cmd.commit() / reps as f64;
        let flop = 2.0 * s.m as f64 * s.n as f64 * s.k as f64;
        let tflops = flop / (ms * 1e-3) / 1e12;
        // spot-check: 64 sampled elements against the f64 reference
        let mut max_err = 0.0f32;
        if verify {
            let cv = gc.as_slice();
            for t in 0..64usize {
                let i = (t * 7919) % s.m;
                let j = (t * 104_729) % s.n;
                let mut acc = 0.0f64;
                for kk in 0..s.k {
                    let av = if s.ta == Op::T {
                        a[kk * acols + i]
                    } else {
                        a[i * acols + kk]
                    } as f64;
                    let bv = if s.tb == Op::T {
                        b[j * bcols + kk]
                    } else {
                        b[kk * bcols + j]
                    } as f64;
                    acc += av * bv;
                }
                let err = (cv[i * s.n + j] as f64 - acc).abs() as f32;
                max_err = max_err.max(err);
            }
        }
        rows.push(BenchRow {
            name: s.name,
            m: s.m,
            n: s.n,
            k: s.k,
            gpu_ms: ms,
            tflops,
            max_abs_err: max_err,
        });
    }
    Some(rows)
}
