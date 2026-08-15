#![cfg(feature = "gpu")]
//! What this card's DRAM actually gives a single well-behaved stream.
//!
//! Exists to split a fork the q4tp campaign hit on an RTX 5090: the
//! decode matvec's PURE weight stream (probe=6 — no codes, no
//! activations, no arithmetic) runs at ~1 TB/s of a 1.79 TB/s card,
//! and every kernel-side suspect measured null (unpack, grid, 2-way
//! unroll, second stream). Either the kernel's 16-interleaved-stream
//! shape thrashes DRAM pages — fixable by a v2 weight layout — or the
//! virtualized pod caps effective bandwidth and the kernel is already
//! at the platform ceiling. This kernel is the cleanest stream a GPU
//! can be asked for: each workgroup walks one contiguous slice with
//! vec4 loads. Whatever it reads IS the platform ceiling.
//!
//! MEASURED (RTX 5090, virtualized RunPod): single stream 1570 GB/s,
//! and the 16-row interleaved control — the matvec's own access
//! pattern — 1623 GB/s. Platform exonerated, LAYOUT exonerated. What
//! distinguishes the real kernel from the control is the shared-memory
//! reduction tree: six workgroupBarriers per 256 lanes per 8-row block
//! (8 KB of codes), against the control's barrier-free 64 KB. The
//! missing third of the bus lives in those barriers.
//!
//! Run: `cargo test --release -p cortiq-engine --features gpu \
//!       --test vram_bandwidth -- --nocapture`

#[test]
fn single_stream_read_ceiling() {
    let Some((device, queue)) = pollster::block_on(async {
        // Headless Vulkan needs the backend named: the default instance
        // finds nothing on a pod and the test silently skips.
        let inst = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });
        let adapter = inst
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .ok()?;
        // The adapter's own limits ARE the maximum grantable set —
        // asking for more than it reports fails request_device, and the
        // first version of this test read that failure as "no adapter".
        let limits = adapter.limits();
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: limits,
                ..Default::default()
            })
            .await
            .ok()
    }) else {
        eprintln!("no adapter — skipping");
        return;
    };

    const WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
struct P { vecs_per_wg: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(2) var<uniform> p: P;

// 256 lanes stride a contiguous slice: lane i reads vec i, i+256, ...
// so every 16-load wavefront touches one 16 KB run of DRAM.
@compute @workgroup_size(256)
fn stream_sum(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) lid: u32) {
    let base = wid.x * p.vecs_per_wg;
    var acc = vec4<f32>(0.0);
    var i = lid;
    loop {
        if (i >= p.vecs_per_wg) { break; }
        acc = acc + src[base + i];
        i = i + 256u;
    }
    if (lid == 0u) { dst[wid.x] = acc.x + acc.y + acc.z + acc.w; }
}
"#;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bw"),
        source: wgpu::ShaderSource::Wgsl(WGSL.into()),
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("stream_sum"),
        layout: None,
        module: &module,
        entry_point: Some("stream_sum"),
        compilation_options: Default::default(),
        cache: None,
    });

    // 1.75 GB under every per-binding limit; big enough that L2 is noise.
    let bytes: u64 = 1_750_000_000 / 16 * 16;
    let nvec = (bytes / 16) as u32;
    let src = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("src"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let wgs: u32 = 2048;
    let vecs_per_wg = nvec.div_ceil(wgs);
    let dst = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dst"),
        size: (wgs * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("p"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&ubuf, 0, bytemuck::cast_slice(&[vecs_per_wg, 0u32, 0, 0]));
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: ubuf.as_entire_binding() },
        ],
    });

    let run = || {
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(wgs, 1, 1);
        }
        queue.submit([enc.finish()]);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
    };
    run(); // warmup: first touch maps the pages
    let reps = 5;
    let t = std::time::Instant::now();
    for _ in 0..reps {
        run();
    }
    let per = t.elapsed().as_secs_f64() / reps as f64;
    eprintln!(
        "single-stream read: {:.1} GB in {:.2} ms = {:.0} GB/s",
        bytes as f64 / 1e9,
        per * 1e3,
        bytes as f64 / per / 1e9
    );

    // The control: the SAME bytes read as the matvec reads them — each
    // workgroup interleaves 16 strided row-streams (rows of a q4tp
    // block live `row_bytes` apart). The delta against the stream above
    // is the exact price of the current layout, i.e. the budget a v2
    // layout may recover.
    const WGSL16: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
struct P { vecs_per_row: u32, rows_per_wg: u32, _b: u32, _c: u32 };
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(256)
fn stream16(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(local_invocation_index) lid: u32) {
    // 16 lanes a row, like the matvec: lane's row = lid/16, its
    // stride walks the row 16 vec4 at a time.
    let row = wid.x * p.rows_per_wg + (lid >> 4u);
    let base = row * p.vecs_per_row;
    var acc = vec4<f32>(0.0);
    var i = lid & 15u;
    loop {
        if (i >= p.vecs_per_row) { break; }
        acc = acc + src[base + i];
        i = i + 16u;
    }
    if (lid == 0u) { dst[wid.x] = acc.x + acc.y + acc.z + acc.w; }
}
"#;
    let m16 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bw16"),
        source: wgpu::ShaderSource::Wgsl(WGSL16.into()),
    });
    let p16 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("stream16"),
        layout: None,
        module: &m16,
        entry_point: Some("stream16"),
        compilation_options: Default::default(),
        cache: None,
    });
    // Rows sized like a 27B ffn row: 2048 cols -> 64 groups x 16 B = 1
    // KB of codes a row... model the STRIDE, not the exact bytes: 4 KB
    // rows, 16 rows a workgroup.
    let vecs_per_row: u32 = 256; // 4 KB per row in vec4 units
    let rows_total = nvec / vecs_per_row;
    let rows_per_wg: u32 = 16;
    let wgs16 = rows_total / rows_per_wg;
    queue.write_buffer(&ubuf, 0, bytemuck::cast_slice(&[vecs_per_row, rows_per_wg, 0u32, 0]));
    let bind16 = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &p16.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: ubuf.as_entire_binding() },
        ],
    });
    let run16 = || {
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&p16);
            pass.set_bind_group(0, &bind16, &[]);
            pass.dispatch_workgroups(wgs16, 1, 1);
        }
        queue.submit([enc.finish()]);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
    };
    run16();
    let t = std::time::Instant::now();
    for _ in 0..reps {
        run16();
    }
    let per16 = t.elapsed().as_secs_f64() / reps as f64;
    eprintln!(
        "16-row interleaved read: {:.1} GB in {:.2} ms = {:.0} GB/s",
        bytes as f64 / 1e9,
        per16 * 1e3,
        bytes as f64 / per16 / 1e9
    );

    // The third arm: the SAME bytes through the GRAPH'S STRUCTURE —
    // 320 serialized dispatches in one pass, each a small slice, each
    // barriered against the next by the pass's own semantics. The
    // dual-kernel null said one deleted wave is under the noise; this
    // measures all of them at once with clean kernels. Collapse to
    // ~1 TB/s convicts the structure and prices it; staying at ~1.6
    // buries the dispatch theory the way ten kernel suspects were
    // buried before it.
    let slices: u32 = 320;
    let vecs_per_slice = nvec / slices;
    queue.write_buffer(&ubuf, 0, bytemuck::cast_slice(&[vecs_per_slice, 0u32, 0, 0]));
    let wg_per_slice = 64u32; // ~5.5 MB per slice, 64 workgroups each
    let run320 = || {
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, &bind, &[]);
            for _ in 0..slices {
                pass.dispatch_workgroups(wg_per_slice, 1, 1);
            }
        }
        queue.submit([enc.finish()]);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
    };
    // NOTE: every dispatch reads the same first slice (the bind group is
    // fixed), so the bytes come from L2 after the first — this measures
    // the DISPATCH structure, deliberately without the DRAM cost.
    run320();
    let t = std::time::Instant::now();
    for _ in 0..reps {
        run320();
    }
    let per320 = t.elapsed().as_secs_f64() / reps as f64;
    eprintln!(
        "320 serialized dispatches (structure only): {:.2} ms = {:.2} us per dispatch",
        per320 * 1e3,
        per320 * 1e6 / slices as f64
    );

    // The fourth arm: the same clean stream dragging a FAT REGISTER
    // FRAME — thirty-two live accumulators the compiler cannot fold.
    // The probes stripped the matvec's WORK but never its registers,
    // and occupancy is the one suspect that survives thirteen nulls:
    // fewer resident warps, fewer loads in flight, ~1 TB/s instead of
    // 1.6. If this arm drops the same way, the fix is a slimmer frame,
    // and it finally has a number to chase.
    const WGSL_FAT: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
struct P { vecs_per_wg: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(256)
fn stream_fat(@builtin(workgroup_id) wid: vec3<u32>,
              @builtin(local_invocation_index) lid: u32) {
    let base = wid.x * p.vecs_per_wg;
    var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0);
    var a2 = vec4<f32>(0.0); var a3 = vec4<f32>(0.0);
    var a4 = vec4<f32>(0.0); var a5 = vec4<f32>(0.0);
    var a6 = vec4<f32>(0.0); var a7 = vec4<f32>(0.0);
    var b0 = vec4<f32>(0.0); var b1 = vec4<f32>(0.0);
    var b2 = vec4<f32>(0.0); var b3 = vec4<f32>(0.0);
    var b4 = vec4<f32>(0.0); var b5 = vec4<f32>(0.0);
    var b6 = vec4<f32>(0.0); var b7 = vec4<f32>(0.0);
    var i = lid;
    loop {
        if (i >= p.vecs_per_wg) { break; }
        let v = src[base + i];
        // Rotate through the frame so every register stays live.
        a0 = a0 + v;           a1 = a1 + v.yzwx;
        a2 = a2 + v.zwxy;      a3 = a3 + v.wxyz;
        a4 = a4 + v * 0.5;     a5 = a5 + v * 0.25;
        a6 = a6 + v * 0.125;   a7 = a7 + v * 0.0625;
        b0 = b0 + a0 * 1e-9;   b1 = b1 + a1 * 1e-9;
        b2 = b2 + a2 * 1e-9;   b3 = b3 + a3 * 1e-9;
        b4 = b4 + a4 * 1e-9;   b5 = b5 + a5 * 1e-9;
        b6 = b6 + a6 * 1e-9;   b7 = b7 + a7 * 1e-9;
        i = i + 256u;
    }
    let s = a0+a1+a2+a3+a4+a5+a6+a7+b0+b1+b2+b3+b4+b5+b6+b7;
    if (lid == 0u) { dst[wid.x] = s.x + s.y + s.z + s.w; }
}
"#;
    let mf = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bwfat"),
        source: wgpu::ShaderSource::Wgsl(WGSL_FAT.into()),
    });
    let pf = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("stream_fat"),
        layout: None,
        module: &mf,
        entry_point: Some("stream_fat"),
        compilation_options: Default::default(),
        cache: None,
    });
    queue.write_buffer(&ubuf, 0, bytemuck::cast_slice(&[vecs_per_wg, 0u32, 0, 0]));
    let bindf = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pf.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: src.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: ubuf.as_entire_binding() },
        ],
    });
    let runf = || {
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pf);
            pass.set_bind_group(0, &bindf, &[]);
            pass.dispatch_workgroups(wgs, 1, 1);
        }
        queue.submit([enc.finish()]);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
    };
    runf();
    let t = std::time::Instant::now();
    for _ in 0..reps {
        runf();
    }
    let perf_ = t.elapsed().as_secs_f64() / reps as f64;
    eprintln!(
        "fat-register stream: {:.1} GB in {:.2} ms = {:.0} GB/s",
        bytes as f64 / 1e9,
        perf_ * 1e3,
        bytes as f64 / perf_ / 1e9
    );

    // The fifth arm, and the last suspect standing after fourteen
    // nulls: the FOOTPRINT. A decode walks 15.4 GB across ~320 buffers
    // a token; every synthetic above re-reads one hot 1.75 GB range.
    // On a virtualized GPU the second-level address translation prices
    // every TLB miss, and that cost rides ON the loads — invisible to
    // every in-kernel elimination. Eight buffers, 14 GB, read once
    // each per pass: if THIS streams at ~1 TB/s, the thief is named,
    // and the fix is an arena — few giant weight buffers instead of
    // hundreds of small ones.
    let nbuf = 8usize;
    let mut bufs = Vec::new();
    for _ in 0..nbuf {
        bufs.push(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fp"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
    }
    queue.write_buffer(&ubuf, 0, bytemuck::cast_slice(&[vecs_per_wg, 0u32, 0, 0]));
    let binds: Vec<_> = bufs
        .iter()
        .map(|bf| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipe.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: bf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: dst.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ubuf.as_entire_binding() },
                ],
            })
        })
        .collect();
    let runfp = || {
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipe);
            for b in &binds {
                pass.set_bind_group(0, b, &[]);
                pass.dispatch_workgroups(wgs, 1, 1);
            }
        }
        queue.submit([enc.finish()]);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
    };
    runfp();
    let t = std::time::Instant::now();
    for _ in 0..reps {
        runfp();
    }
    let perfp = t.elapsed().as_secs_f64() / reps as f64;
    let total = bytes as f64 * nbuf as f64;
    eprintln!(
        "14 GB footprint walk: {:.1} GB in {:.2} ms = {:.0} GB/s",
        total / 1e9,
        perfp * 1e3,
        total / perfp / 1e9
    );
}
