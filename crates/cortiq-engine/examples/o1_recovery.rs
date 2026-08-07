//! Injection/recovery for the O(1) streaming operator.
//!
//! `--o1`'s cost is quoted as a scalar: ×1.296 perplexity against exact
//! attention. A scalar cannot be allocated. It says the operator costs
//! something on average over one corpus at one setting, and says nothing
//! about which setting to spend the next landmark on.
//!
//! This measures the same operator the way a detection experiment
//! measures a search: inject a signal of KNOWN strength at a KNOWN
//! depth, and count how much of it comes back out. The needle is one
//! (key, value) pair whose logit sits `amp` standard deviations above
//! the background — the same construction as `tests/vacuum.rs`, run
//! Monte Carlo over seeds. Because the background values are zero in
//! dimension 0 and the needle's is one, the output's component 0 IS the
//! needle's contribution: recovery = got[0] / exact[0], a fraction with
//! no fitting and no threshold.
//!
//! What it is for. If recovery rises SMOOTHLY with the landmark budget,
//! the budget is a continuous resource and can be allocated across
//! layers by an equalizing rule. If it is a STEP — the sketch either
//! kept the needle or did not — then allocation is a covering problem
//! instead, and averaging landmarks across layers buys nothing. That
//! question has to be answered before any allocation rule is chosen,
//! which is what this prints.
//!
//! Measured (w=128, sink=4, depth=512, 300 trials a cell): the response
//! is a CONTINUUM, decisively. 758–1496 of every 1500 trials in a row
//! land strictly between 5% and 95% recovery, where a step would leave
//! that column near zero; and the two criteria cross at different budgets
//! (at amp 12, half the trials clear 50% by m=4 but need m=32 to clear
//! 95%). The deficit 1−r falls as a shallow power law in m, exponent
//! −0.18 to −0.38 across injection strengths, clustering near −1/4.
//!
//! That exponent is the number an allocation rule needs, and it is not
//! the one a physical detection experiment would hand over. Equalizing a
//! response that grows as r^β needs budget ∝ c^(1/β): at β = 1/2 — the
//! √t law of an integration-time search — that is the familiar c², but at
//! β = 1/4 it is c⁴, quadratically more aggressive. Assuming the physics
//! exponent here would under-serve the weak layers by a wide margin.
//!
//! Two honest limits on that number. Recovery is NOT monotone in
//! injection strength — it dips near amp 6 and recovers by amp 12 —
//! because the statistic divides by the exact answer, and in the middle
//! band the needle is a large minority of the softmax mass, so error in
//! the estimated DENOMINATOR (all background) scales the ratio directly.
//! And the keys here are iid Gaussian, which is the worst case a
//! landmark sketch can be handed: real attention keys are strongly
//! low-rank, which is why the method works at all. Treat β ≈ −1/4 as a
//! floor measured on the hardest input, not as the figure for a model.
//!
//!     cargo run --release -p cortiq-engine --example o1_recovery
//!     TRIALS=300 cargo run --release -p cortiq-engine --example o1_recovery

use cortiq_engine::nystrom::NystromState;

fn unif(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

/// One trial. Returns the fraction of the needle's exact contribution
/// that survived the streaming kernel.
fn trial(m: usize, w: usize, sink: usize, amp: f32, depth: usize, seed: u64) -> f32 {
    let (d, dv) = (64usize, 8usize);
    let t = 8 * m + w + depth;
    let mut s = seed;
    let rd = (d as f32).sqrt();

    let mut unit = |s: &mut u64| -> Vec<f32> {
        let v: Vec<f32> = (0..d).map(|_| unif(s)).collect();
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    };
    let qhat = unit(&mut s);
    let q: Vec<f32> = qhat.iter().map(|x| x * rd).collect();

    let mut qs = Vec::with_capacity(t * d);
    for _ in 0..t {
        let u = unit(&mut s);
        qs.extend(u.iter().map(|x| x * rd));
    }

    let mut ks: Vec<f32> = (0..t * d).map(|_| unif(&mut s) * 1.732).collect();
    let mut vs: Vec<f32> = (0..t * dv).map(|_| unif(&mut s)).collect();
    for j in 0..t {
        vs[j * dv] = 0.0;
    }
    let p = t - depth;
    for c in 0..d {
        ks[p * d + c] += amp * qhat[c];
    }
    vs[p * dv] = 1.0;

    let mut st = NystromState::new(m, w, sink);
    st.prefill(&qs, &ks, &vs, t, d, dv);

    let k_new: Vec<f32> = (0..d).map(|_| unif(&mut s) * 1.732).collect();
    let mut v_new: Vec<f32> = (0..dv).map(|_| unif(&mut s)).collect();
    v_new[0] = 0.0;
    let mut got = vec![0f32; dv];
    st.step(&q, &k_new, &v_new, &mut got);

    let mut logits = Vec::with_capacity(t + 1);
    for j in 0..t {
        let dot: f32 = (0..d).map(|c| q[c] * ks[j * d + c]).sum();
        logits.push(dot / rd);
    }
    let dot: f32 = (0..d).map(|c| q[c] * k_new[c]).sum();
    logits.push(dot / rd);
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let (mut den, mut num0) = (0f64, 0f64);
    for (j, &l) in logits.iter().enumerate() {
        let e = ((l - mx) as f64).exp();
        den += e;
        num0 += e * if j == t { v_new[0] } else { vs[j * dv] } as f64;
    }
    let want0 = (num0 / den) as f32;
    if want0.abs() < 1e-4 {
        return f32::NAN;
    }
    got[0] / want0
}

struct Cell {
    mean: f32,
    sd: f32,
    /// Fraction of trials that kept at least half the needle.
    detected: f32,
    /// Fraction that kept at least 95% — the strict criterion.
    strict: f32,
    /// Trials landing strictly between 5% and 95% — the population a
    /// step response is not allowed to have.
    partial: usize,
}

fn cell(m: usize, w: usize, sink: usize, amp: f32, depth: usize, trials: usize) -> Cell {
    let mut rs = Vec::with_capacity(trials);
    for i in 0..trials {
        let r = trial(m, w, sink, amp, depth, 0x1000 + i as u64 * 0x9E3779B9);
        if r.is_finite() {
            rs.push(r);
        }
    }
    let n = rs.len().max(1) as f32;
    let mean = rs.iter().sum::<f32>() / n;
    let var = rs.iter().map(|r| (r - mean) * (r - mean)).sum::<f32>() / n;
    Cell {
        mean,
        sd: var.sqrt(),
        detected: rs.iter().filter(|&&r| r >= 0.5).count() as f32 / n,
        strict: rs.iter().filter(|&&r| r >= 0.95).count() as f32 / n,
        partial: rs.iter().filter(|&&r| r > 0.05 && r < 0.95).count(),
    }
}

fn main() {
    let trials: usize = std::env::var("TRIALS").ok().and_then(|v| v.parse().ok()).unwrap_or(48);
    let (w, sink, depth) = (128usize, 4usize, 512usize);
    let ms = [4usize, 8, 16, 32, 64];
    let amps = [2.0f32, 4.0, 6.0, 8.0, 12.0];

    println!("O(1) injection/recovery — w={w} sink={sink} depth={depth}, {trials} trials/cell");
    println!("needle logit sits `amp` background sd above the field; recovery = kept / exact\n");

    println!("mean recovery");
    print!("{:>6}", "amp\\m");
    for m in ms {
        print!("{m:>12}");
    }
    println!();
    let mut grid = Vec::new();
    for &amp in &amps {
        print!("{amp:>6.1}");
        let mut row = Vec::new();
        for &m in &ms {
            let c = cell(m, w, sink, amp, depth, trials);
            print!("{:>9.1}±{:<2.0}", c.mean * 100.0, c.sd * 100.0);
            row.push(c);
        }
        println!();
        grid.push((amp, row));
    }

    println!("\nfraction of trials at or above 95% recovery — the strict criterion");
    print!("{:>6}", "amp\\m");
    for m in ms {
        print!("{m:>8}");
    }
    println!();
    for (amp, row) in &grid {
        print!("{amp:>6.1}");
        for c in row {
            print!("{:>7.0}%", c.strict * 100.0);
        }
        println!();
    }

    println!("\nfraction at or above 50% — the loose criterion");
    print!("{:>6}", "amp\\m");
    for m in ms {
        print!("{m:>8}");
    }
    println!();
    for (amp, row) in &grid {
        print!("{amp:>6.1}");
        for c in row {
            print!("{:>7.0}%", c.detected * 100.0);
        }
        println!();
    }

    // Step or continuum? A step in m would show as the 50% and 95%
    // criteria crossing at the SAME m — the needle is either kept whole
    // or lost whole, with no partial states in between. A continuum
    // shows the two criteria crossing at different m, with a spread of
    // partial recoveries in between.
    println!("\nshape of the response in m");
    for (amp, row) in &grid {
        let m50 = ms.iter().zip(row).find(|(_, c)| c.detected >= 0.5).map(|(m, _)| *m);
        let m95 = ms.iter().zip(row).find(|(_, c)| c.strict >= 0.5).map(|(m, _)| *m);
        // Deficit vs m on a log-log slope: a power law gives a stable
        // exponent, a step gives a slope that runs away at the edge.
        let slope = {
            let pts: Vec<(f32, f32)> = ms
                .iter()
                .zip(row)
                .filter(|(_, c)| c.mean < 0.999 && c.mean > 0.0)
                .map(|(m, c)| ((*m as f32).ln(), (1.0 - c.mean).max(1e-4).ln()))
                .collect();
            if pts.len() < 2 {
                f32::NAN
            } else {
                let n = pts.len() as f32;
                let (sx, sy) = pts.iter().fold((0f32, 0f32), |(a, b), (x, y)| (a + x, b + y));
                let (mx, my) = (sx / n, sy / n);
                let num: f32 = pts.iter().map(|(x, y)| (x - mx) * (y - my)).sum();
                let den: f32 = pts.iter().map(|(x, _)| (x - mx) * (x - mx)).sum();
                num / den
            }
        };
        let mid: usize = row.iter().map(|c| c.partial).sum();
        println!(
            "amp {amp:>4.1}: m@50% {:>6}  m@95% {:>6}  log-log slope of (1−r) vs m {slope:>7.2}  \
             partial trials {mid:>4}",
            m50.map(|v| v.to_string()).unwrap_or("—".into()),
            m95.map(|v| v.to_string()).unwrap_or("—".into()),
        );
    }
    println!(
        "\n`partial trials` counts runs landing strictly between 5% and 95% recovery, summed over \
         the row.\nA pure step would leave that column near zero: every trial all or nothing."
    );
}
