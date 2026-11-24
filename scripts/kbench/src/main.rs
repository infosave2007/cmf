//! kbench: time the skill bake's WGSL kernels on a card and check them against f64, without
//! rebuilding the engine — a shader edit is a rerun.
//!
//!   kbench gemm nt|nn SRC ENTRY NB COLS ROWS TILE_ROWS TILE_NB [KSPAN]
//!   kbench attnbwd QSRC QENTRY KVSRC KVENTRY B T NH NKV HD [QROWS KROWS]
//!   kbench attnfwd SRC ENTRY B T NH NKV HD [QROWS]
//!   kbench peak SRC ENTRY WORKGROUPS ITERS
//!
//! SRC is a .wgsl file, or `@NAME` for the raw string `const NAME` of the engine's
//! `crates/cortiq-engine/src/gpu_wgpu.rs` (`KBENCH_SRC` overrides the path), e.g.
//! `kbench gemm nt @F32T_NT_SRC gemm_nt_f32t 256 1024 6144 64 64`.
//! GEMM: uniform [cols/4, rows, nb, kspan], bindings (w, x, y, u); NN reads w as [cols][rows];
//! split-K partial planes are summed on the host. Attention: the bake kernels' bindings and grid
//! (t / QROWS, b·heads). `TRUSTED=1`: the module without naga's bounds checks (as the engine
//! builds these kernels); `NOZERO=1`: no workgroup-memory zeroing; `METAL=1`: the Metal backend.
use std::time::Instant;
use wgpu::util::DeviceExt;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn gpu() -> Gpu {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: if std::env::var("METAL").is_ok() { wgpu::Backends::METAL } else { wgpu::Backends::VULKAN },
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .expect("adapter");
    let info = adapter.get_info();
    let sg = adapter.features().contains(wgpu::Features::SUBGROUP);
    let lim = adapter.limits();
    eprintln!(
        "{} | subgroups {} [{}..{}] | wg storage {} | invocations {}",
        info.name, sg, info.subgroup_min_size, info.subgroup_max_size, lim.max_compute_workgroup_storage_size, lim.max_compute_invocations_per_workgroup
    );
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_limits: lim,
        required_features: if sg { wgpu::Features::SUBGROUP } else { wgpu::Features::empty() },
        ..Default::default()
    }))
    .expect("device");
    Gpu { device, queue }
}

fn pipe(g: &Gpu, path: &str, entry: &str) -> wgpu::ComputePipeline {
    let src = match path.strip_prefix('@') {
        Some(name) => {
            let file = std::env::var("KBENCH_SRC").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../crates/cortiq-engine/src/gpu_wgpu.rs").into());
            let all = std::fs::read_to_string(&file).unwrap_or_else(|e| panic!("{file}: {e}"));
            let head = format!("const {name}: &str = r#\"");
            let i = all.find(&head).unwrap_or_else(|| panic!("no `{head}` in {file}")) + head.len();
            let j = all[i..].find("\"#;").expect("unterminated raw string") + i;
            all[i..j].to_string()
        }
        None => std::fs::read_to_string(path).expect("shader file"),
    };
    let desc = wgpu::ShaderModuleDescriptor { label: Some(entry), source: wgpu::ShaderSource::Wgsl(src.into()) };
    let m = if std::env::var("TRUSTED").is_ok() {
        unsafe { g.device.create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked()) }
    } else {
        g.device.create_shader_module(desc)
    };
    g.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: None,
        module: &m,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions {
            zero_initialize_workgroup_memory: std::env::var("NOZERO").is_err(),
            ..Default::default()
        },
        cache: None,
    })
}

fn buf(g: &Gpu, data: &[f32]) -> wgpu::Buffer {
    g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    })
}

fn zeros(g: &Gpu, n: usize) -> wgpu::Buffer {
    buf(g, &vec![0f32; n.max(4)])
}

fn uni(g: &Gpu, v: &[u32]) -> wgpu::Buffer {
    g.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(v),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

fn bind(g: &Gpu, p: &wgpu::ComputePipeline, bufs: &[&wgpu::Buffer]) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = bufs
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
        .collect();
    g.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &p.get_bind_group_layout(0), entries: &entries })
}

/// ms a dispatch: `reps` dispatches (one pass each) in one submit, after a warm-up.
fn time(g: &Gpu, p: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, grid: (u32, u32, u32), reps: usize) -> f64 {
    let go = |n: usize| {
        let mut enc = g.device.create_command_encoder(&Default::default());
        for _ in 0..n {
            let mut ps = enc.begin_compute_pass(&Default::default());
            ps.set_pipeline(p);
            ps.set_bind_group(0, bg, &[]);
            ps.dispatch_workgroups(grid.0, grid.1, grid.2);
        }
        g.queue.submit(Some(enc.finish()));
        g.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    };
    go(2);
    let t = Instant::now();
    go(reps);
    t.elapsed().as_secs_f64() * 1e3 / reps as f64
}

fn read(g: &Gpu, b: &wgpu::Buffer, n: usize) -> Vec<f32> {
    let st = g.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (n * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = g.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(b, 0, &st, 0, (n * 4) as u64);
    g.queue.submit(Some(enc.finish()));
    let sl = st.slice(..);
    sl.map_async(wgpu::MapMode::Read, |_| {});
    g.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    bytemuck::cast_slice(&sl.get_mapped_range().unwrap()).to_vec()
}

fn synth(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect()
}

fn rel(a: &[f32], r: &[f64]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(r) {
        num += (*x as f64 - y) * (*x as f64 - y);
        den += y * y;
    }
    (num / den.max(1e-300)).sqrt()
}

fn arg(a: &[String], i: usize) -> usize {
    a[i].parse().unwrap_or_else(|_| panic!("argument {i}: {}", a[i]))
}

fn gemm(g: &Gpu, a: &[String]) {
    let nn = a[2] == "nn";
    let (file, entry) = (&a[3], &a[4]);
    let (nb, cols, rows, tr, tnb) = (arg(a, 5), arg(a, 6), arg(a, 7), arg(a, 8), arg(a, 9));
    let kspan = if a.len() > 10 { arg(a, 10) } else { 0 };
    let parts = if kspan > 0 { cols.div_ceil(kspan) } else { 1 };
    let x = synth(nb * cols, 1);
    let w = synth(cols * rows, 2);
    let p = pipe(g, file, entry);
    let (xb, wb, yb) = (buf(g, &x), buf(g, &w), zeros(g, parts * nb * rows));
    let u = uni(g, &[(cols / 4) as u32, rows as u32, nb as u32, kspan as u32]);
    let bg = bind(g, &p, &[&wb, &xb, &yb, &u]);
    let grid = (rows.div_ceil(tr) as u32, nb.div_ceil(tnb) as u32, parts as u32);
    let ms = time(g, &p, &bg, grid, 20);
    let y = read(g, &yb, parts * nb * rows);
    // f64 check on sampled outputs
    let mut got = Vec::new();
    let mut r64 = Vec::new();
    let mut s = 12345u64;
    for _ in 0..512 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let m = (s >> 33) as usize % nb;
        let q = (s >> 13) as usize % rows;
        let mut acc = 0f64;
        for k in 0..cols {
            let wv = if nn { w[k * rows + q] } else { w[q * cols + k] };
            acc += x[m * cols + k] as f64 * wv as f64;
        }
        let mut v = 0f32;
        for z in 0..parts {
            v += y[z * nb * rows + m * rows + q];
        }
        got.push(v);
        r64.push(acc);
    }
    let gflop = 2.0 * (nb * cols * rows) as f64 / 1e9;
    println!(
        "gemm {} {entry} nb {nb} cols {cols} rows {rows} parts {parts}: {ms:.3} ms, {:.2} TFLOPS, rel. error {:.2e}",
        if nn { "NN" } else { "NT" },
        gflop / ms,
        rel(&got, &r64)
    );
}

/// f64 attention (causal, no window) of one (b, h): returns o, and with `dout` the grads dq and the
/// head's contributions to dk, dv.
#[allow(clippy::too_many_arguments)]
fn attn_ref(q: &[f32], k: &[f32], v: &[f32], dout: &[f32], b: usize, t: usize, nh: usize, nkv: usize, hd: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (qdim, kvdim, rep) = (nh * hd, nkv * hd, nh / nkv);
    let n = b * t;
    let scale = 1.0 / (hd as f64).sqrt();
    let (mut o, mut dq, mut dk, mut dv) = (vec![0f64; n * qdim], vec![0f64; n * qdim], vec![0f64; n * kvdim], vec![0f64; n * kvdim]);
    for bi in 0..b {
        for h in 0..nh {
            let gi = h / rep;
            let qi = |i: usize, c: usize| q[(bi * t + i) * qdim + h * hd + c] as f64;
            let ki = |j: usize, c: usize| k[(bi * t + j) * kvdim + gi * hd + c] as f64;
            let vi = |j: usize, c: usize| v[(bi * t + j) * kvdim + gi * hd + c] as f64;
            let di = |i: usize, c: usize| dout[(bi * t + i) * qdim + h * hd + c] as f64;
            for i in 0..t {
                let s: Vec<f64> = (0..=i).map(|j| (0..hd).map(|c| qi(i, c) * ki(j, c)).sum::<f64>() * scale).collect();
                let mx = s.iter().cloned().fold(f64::MIN, f64::max);
                let e: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
                let den: f64 = e.iter().sum();
                let p: Vec<f64> = e.iter().map(|x| x / den).collect();
                for c in 0..hd {
                    o[(bi * t + i) * qdim + h * hd + c] = (0..=i).map(|j| p[j] * vi(j, c)).sum();
                }
                let dp: Vec<f64> = (0..=i).map(|j| (0..hd).map(|c| di(i, c) * vi(j, c)).sum()).collect();
                let dsum: f64 = (0..=i).map(|j| p[j] * dp[j]).sum();
                for j in 0..=i {
                    let ds = p[j] * (dp[j] - dsum) * scale;
                    for c in 0..hd {
                        dq[(bi * t + i) * qdim + h * hd + c] += ds * ki(j, c);
                        dk[(bi * t + j) * kvdim + gi * hd + c] += ds * qi(i, c);
                        dv[(bi * t + j) * kvdim + gi * hd + c] += p[j] * di(i, c);
                    }
                }
            }
        }
    }
    (o, dq, dk, dv)
}

fn attnbwd(g: &Gpu, a: &[String]) {
    let (qf, qe, kf, ke) = (&a[2], &a[3], &a[4], &a[5]);
    let (b, t, nh, nkv, hd) = (arg(a, 6), arg(a, 7), arg(a, 8), arg(a, 9), arg(a, 10));
    let (qrows, krows) = if a.len() > 12 { (arg(a, 11), arg(a, 12)) } else { (1, 1) };
    let (qdim, kvdim, n) = (nh * hd, nkv * hd, b * t);
    let (q, k, v, d) = (synth(n * qdim, 1), synth(n * kvdim, 2), synth(n * kvdim, 3), synth(n * qdim, 4));
    let (q, k): (Vec<f32>, Vec<f32>) = (q.iter().map(|x| x * 0.5).collect(), k.iter().map(|x| x * 0.5).collect());
    let (pq, pk) = (pipe(g, qf, qe), pipe(g, kf, ke));
    let (qb, kb, vb, db) = (buf(g, &q), buf(g, &k), buf(g, &v), buf(g, &d));
    let (dqb, dkb, dvb, stb) = (zeros(g, n * qdim), zeros(g, n * kvdim), zeros(g, n * kvdim), zeros(g, n * nh * 4));
    let u = uni(g, &[t as u32, nh as u32, nkv as u32, hd as u32, b as u32, 0, 0, 0]);
    let bq = bind(g, &pq, &[&qb, &kb, &vb, &db, &dqb, &stb, &u]);
    let bk = bind(g, &pk, &[&qb, &kb, &vb, &db, &stb, &dkb, &dvb, &u]);
    let mq = time(g, &pq, &bq, (t.div_ceil(qrows) as u32, (b * nh) as u32, 1), 20);
    let mk = time(g, &pk, &bk, (t.div_ceil(krows) as u32, (b * nkv) as u32, 1), 20);
    let (_, rq, rk, rv) = attn_ref(&q, &k, &v, &d, b, t, nh, nkv, hd);
    let (gq, gk, gv) = (read(g, &dqb, n * qdim), read(g, &dkb, n * kvdim), read(g, &dvb, n * kvdim));
    println!(
        "attnbwd b {b} t {t} nh {nh}:{nkv} hd {hd}: q {qe} {mq:.3} ms, kv {ke} {mk:.3} ms | rel. error dq {:.2e} dk {:.2e} dv {:.2e}",
        rel(&gq, &rq),
        rel(&gk, &rk),
        rel(&gv, &rv)
    );
}

fn attnfwd(g: &Gpu, a: &[String]) {
    let (f, e) = (&a[2], &a[3]);
    let (b, t, nh, nkv, hd) = (arg(a, 4), arg(a, 5), arg(a, 6), arg(a, 7), arg(a, 8));
    let qrows = if a.len() > 9 { arg(a, 9) } else { 1 };
    let (qdim, kvdim, n) = (nh * hd, nkv * hd, b * t);
    let (q, k, v) = (synth(n * qdim, 1), synth(n * kvdim, 2), synth(n * kvdim, 3));
    let (q, k): (Vec<f32>, Vec<f32>) = (q.iter().map(|x| x * 0.5).collect(), k.iter().map(|x| x * 0.5).collect());
    let p = pipe(g, f, e);
    let (qb, kb, vb, ob) = (buf(g, &q), buf(g, &k), buf(g, &v), zeros(g, n * qdim));
    let u = uni(g, &[t as u32, nh as u32, nkv as u32, hd as u32, b as u32, 0, 0, 0]);
    let bg = bind(g, &p, &[&qb, &kb, &vb, &ob, &u]);
    let ms = time(g, &p, &bg, (t.div_ceil(qrows) as u32, (b * nh) as u32, 1), 20);
    let (ro, _, _, _) = attn_ref(&q, &k, &v, &vec![0f32; n * qdim], b, t, nh, nkv, hd);
    let go = read(g, &ob, n * qdim);
    println!("attnfwd b {b} t {t} nh {nh}:{nkv} hd {hd}: {e} {ms:.3} ms | rel. error {:.2e}", rel(&go, &ro));
}

fn peak(g: &Gpu, a: &[String]) {
    let (f, e, wgs, iters) = (&a[2], &a[3], arg(a, 4), arg(a, 5));
    let p = pipe(g, f, e);
    let ob = zeros(g, wgs * 256 * 4);
    let u = uni(g, &[iters as u32, 0, 0, 0]);
    let bg = bind(g, &p, &[&ob, &u]);
    let ms = time(g, &p, &bg, (wgs as u32, 1, 1), 10);
    println!("peak {e}: {ms:.3} ms, {:.2} TFLOPS", (wgs * 256 * iters * 64) as f64 / ms / 1e9);
}

/// kbench raw SRC ENTRY GX GY GZ U0,U1,… N0 N1 …: time any kernel — storage bindings of N_i
/// floats (synthetic data) in order, then the uniform (u32 words; `f:x` for an f32).
fn raw(g: &Gpu, a: &[String]) {
    let (f, e) = (&a[2], &a[3]);
    let grid = (arg(a, 4) as u32, arg(a, 5) as u32, arg(a, 6) as u32);
    let u: Vec<u32> = a[7]
        .split(',')
        .map(|w| match w.strip_prefix("f:") {
            Some(x) => x.parse::<f32>().unwrap().to_bits(),
            None => w.parse().unwrap(),
        })
        .collect();
    let bufs: Vec<wgpu::Buffer> = a[8..].iter().enumerate().map(|(i, n)| buf(g, &synth(n.parse().unwrap(), i as u64 + 1))).collect();
    let ub = uni(g, &u);
    let mut refs: Vec<&wgpu::Buffer> = bufs.iter().collect();
    refs.push(&ub);
    let p = pipe(g, f, e);
    let bg = bind(g, &p, &refs);
    let ms = time(g, &p, &bg, grid, 20);
    println!("raw {e} grid {grid:?}: {ms:.3} ms");
}

/// kbench cmp SRC1 ENTRY1 GX,GY,GZ SRC2 ENTRY2 GX,GY,GZ U OUTS N0 N1 …: the same bindings and
/// uniform through two kernels (grids of their own); OUTS = comma list of the bindings to compare
/// (rel. L2 of the second against the first), both timed.
fn cmp(g: &Gpu, a: &[String]) {
    let grid = |s: &str| {
        let v: Vec<u32> = s.split(',').map(|x| x.parse().unwrap()).collect();
        (v[0], v[1], v[2])
    };
    let (p1, g1, p2, g2) = (pipe(g, &a[2], &a[3]), grid(&a[4]), pipe(g, &a[5], &a[6]), grid(&a[7]));
    let u: Vec<u32> = a[8]
        .split(',')
        .map(|w| match w.strip_prefix("f:") {
            Some(x) => x.parse::<f32>().unwrap().to_bits(),
            None => w.parse().unwrap(),
        })
        .collect();
    let outs: Vec<usize> = a[9].split(',').map(|x| x.parse().unwrap()).collect();
    let sizes: Vec<usize> = a[10..].iter().map(|n| n.parse().unwrap()).collect();
    let ub = uni(g, &u);
    let mut res = Vec::new();
    for (p, gr) in [(&p1, g1), (&p2, g2)] {
        let bufs: Vec<wgpu::Buffer> = sizes.iter().enumerate().map(|(i, &n)| if outs.contains(&i) { zeros(g, n) } else { buf(g, &synth(n, i as u64 + 1)) }).collect();
        let mut refs: Vec<&wgpu::Buffer> = bufs.iter().collect();
        refs.push(&ub);
        let bg = bind(g, p, &refs);
        let ms = time(g, p, &bg, gr, 20);
        let got: Vec<Vec<f32>> = outs.iter().map(|&i| read(g, &bufs[i], sizes[i])).collect();
        res.push((ms, got));
    }
    let errs: Vec<String> = (0..outs.len())
        .map(|k| {
            let r: Vec<f64> = res[0].1[k].iter().map(|&x| x as f64).collect();
            format!("{}: {:.2e}", outs[k], rel(&res[1].1[k], &r))
        })
        .collect();
    println!("cmp {} {:.3} ms | {} {:.3} ms | rel. diff {}", a[3], res[0].0, a[6], res[1].0, errs.join(" "));
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let g = gpu();
    match a.get(1).map(|s| s.as_str()) {
        Some("gemm") => gemm(&g, &a),
        Some("attnbwd") => attnbwd(&g, &a),
        Some("attnfwd") => attnfwd(&g, &a),
        Some("peak") => peak(&g, &a),
        Some("raw") => raw(&g, &a),
        Some("cmp") => cmp(&g, &a),
        _ => eprintln!("kbench gemm|attnbwd|attnfwd ..."),
    }
}
