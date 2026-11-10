//! The batch graph's q8_2f projection on wgpu, by kernel, for a decode
//! verify's few rows: `cur` is `q8_2f_matvec_b` as it ships (64 lanes a
//! weight row, four rows a workgroup), `mv4k` the token graph's decode
//! kernel on one row (the columns split across four 64-lane sub-blocks),
//! `b4k` that kernel over up to eight batch rows, every weight word read once
//! for all of them — its row 0 must equal `mv4k`'s to the bit (checked).
//! Wall time of many dispatches in one submission.
//!
//! cargo run --release -p cortiq-engine --features gpu --example q82_batch_bench

#[cfg(feature = "gpu")]
fn main() {
    use cortiq_engine::wgpu;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
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
    .expect("no adapter");
    println!("{}", adapter.get_info().name);
    let mut limits = adapter.limits();
    limits.max_compute_workgroup_storage_size =
        limits.max_compute_workgroup_storage_size.max(32768);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: None,
        required_limits: limits,
        required_features: wgpu::Features::empty(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: Default::default(),
    }))
    .expect("no device");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(SRC.into()),
    });
    let pipe = |ep: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(ep),
            layout: None,
            module: &module,
            entry_point: Some(ep),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let kernels = [
        ("cur", pipe("cur"), 4u32),
        ("b4k", pipe("b4k"), 4u32),
        ("mv4k", pipe("mv4k"), 4u32),
    ];
    let mut seed = 0x9e37_79b9u32;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    use wgpu::util::DeviceExt;
    for (rows, cols) in [
        (2560usize, 896usize),
        (896, 2560),
        (896, 896),
        (128, 896),
        (65536, 896),
    ] {
        // q8_2f payload: int8 codes, f16 row scales, f16 column scales
        let words = (rows * cols) / 4 + rows / 2 + cols / 2 + 4;
        let w: Vec<u32> = (0..words)
            .map(|i| {
                if i < rows * cols / 4 {
                    rnd() & 0x3f3f_3f3f
                } else {
                    0x3c00_3c00 // 1.0 in both halves
                }
            })
            .collect();
        let wbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&w),
            usage: wgpu::BufferUsages::STORAGE,
        });
        for nb in [1usize, 4, 6, 8] {
            let x: Vec<f32> = (0..nb * cols)
                .map(|_| (rnd() % 2000) as f32 / 1000.0 - 1.0)
                .collect();
            let xbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&x),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let ysize = (nb * rows * 4) as u64;
            let p = [(cols / 4) as u32, rows as u32, cols as u32, nb as u32];
            let pbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&p),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let mut outs: Vec<Vec<f32>> = Vec::new();
            let mut line = format!("{rows:>6} x {cols:<5} rows {nb}:");
            let umod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(unrolled_src(nb).into()),
            });
            let upipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("unr"),
                layout: None,
                module: &umod,
                entry_point: Some("unr"),
                compilation_options: Default::default(),
                cache: None,
            });
            let all: Vec<(&str, &wgpu::ComputePipeline, u32)> = kernels
                .iter()
                .map(|(n, p, r)| (*n, p, *r))
                .chain(std::iter::once(("unr", &upipe, 4u32)))
                .collect();
            for (name, pl, rows_per_wg) in all {
                let ybuf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: ysize,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let stage = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: ysize,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pl.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wbuf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: xbuf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: ybuf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: pbuf.as_entire_binding(),
                        },
                    ],
                });
                let wgs = (rows as u32).div_ceil(rows_per_wg).min(65535);
                let run = |reps: usize| -> f64 {
                    let mut enc = device.create_command_encoder(&Default::default());
                    {
                        let mut pass = enc.begin_compute_pass(&Default::default());
                        pass.set_pipeline(pl);
                        pass.set_bind_group(0, &bind, &[]);
                        for _ in 0..reps {
                            pass.dispatch_workgroups(wgs, (nb as u32).div_ceil(8), 1);
                        }
                    }
                    let t0 = std::time::Instant::now();
                    queue.submit(Some(enc.finish()));
                    let _ = device.poll(wgpu::PollType::wait_indefinitely());
                    t0.elapsed().as_secs_f64() * 1e6 / reps as f64
                };
                for _ in 0..3 {
                    run(200);
                }
                let us = (0..5).map(|_| run(200)).fold(f64::MAX, f64::min);
                let mut enc = device.create_command_encoder(&Default::default());
                enc.copy_buffer_to_buffer(&ybuf, 0, &stage, 0, ysize);
                queue.submit(Some(enc.finish()));
                stage.slice(..).map_async(wgpu::MapMode::Read, |_| {});
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
                {
                    let view = stage.slice(..).get_mapped_range().expect("map");
                    outs.push(bytemuck::cast_slice::<u8, f32>(&view).to_vec());
                }
                let gbs = (rows * cols) as f64 / us / 1e3;
                line += &format!("  {name} {us:7.1} us ({gbs:5.0} GB/s)");
            }
            // b4k is the decode kernel (mv4k) batched: its row 0 equals mv4k's to the bit
            // b4k and unr batch the decode kernel (mv4k): their row 0 equals mv4k's to the bit
            let eq = |o: &Vec<f32>| {
                o[..rows]
                    .iter()
                    .zip(&outs[2][..rows])
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            };
            println!(
                "{line}  row 0 = mv4k: b4k {} unr {}",
                eq(&outs[1]),
                eq(&outs[3])
            );
        }
    }
}

/// `mv4k` over exactly `nb` batch rows, unrolled: one vec4 accumulator a
/// batch row as its own variable, the partials lane-major (`j·256 + lane`),
/// the tree and the four-quarter sum per batch row as `mv4k`'s.
#[cfg(feature = "gpu")]
fn unrolled_src(nb: usize) -> String {
    let mut decl = String::new();
    let mut body = String::new();
    let mut store = String::new();
    let mut tree = String::new();
    for j in 0..nb {
        decl += &format!("        var a{j} = vec4<f32>(0.0);\n");
        body += &format!(
            "                let x{j} = qb_x[{j}u * ngrp + i] * cs;\n                a{j} = a{j} + vec4<f32>(dot(wa, x{j}), dot(wb, x{j}), dot(wc, x{j}), dot(wd, x{j}));\n"
        );
        store += &format!("        u_part[{j}u * 256u + lid] = a{j};\n");
        tree += &format!(
            "                u_part[{j}u * 256u + lid] = u_part[{j}u * 256u + lid] + u_part[{j}u * 256u + lid + stride];\n"
        );
    }
    let core = r#"
@compute @workgroup_size(256)
fn unr(@builtin(workgroup_id) wid: vec3<u32>,
       @builtin(num_workgroups) nwg: vec3<u32>,
       @builtin(local_invocation_index) lid: u32) {
    let rows = qb_p.rows;
    let ngrp = qb_p.cols4;
    let qbytes = rows * qb_p.cols;
    let rs0 = qbytes >> 2u;
    let cs0h = (qbytes >> 1u) + rows;
    let sub = lid >> 6u;
    let l = lid & 63u;
    let qlen = ngrp / 4u;
    let kend = sub * qlen + qlen;
    let blocks = (rows + 3u) / 4u;
    var wb_ = wid.x;
    loop {
        if (wb_ >= blocks) { break; }
        let r0 = wb_ * 4u;
        let l1 = r0 + 1u < rows;
        let l2 = r0 + 2u < rows;
        let l3 = r0 + 3u < rows;
DECL        if (r0 < rows) {
            let w0b = r0 * ngrp;
            let w1b = select(w0b, (r0 + 1u) * ngrp, l1);
            let w2b = select(w0b, (r0 + 2u) * ngrp, l2);
            let w3b = select(w0b, (r0 + 3u) * ngrp, l3);
            var i = sub * qlen + l;
            loop {
                if (i >= kend) { break; }
                let cs = qb_f16x4(cs0h + i * 4u);
                let wa = qb_i8x4(qb_w[w0b + i]);
                let wb = qb_i8x4(qb_w[w1b + i]);
                let wc = qb_i8x4(qb_w[w2b + i]);
                let wd = qb_i8x4(qb_w[w3b + i]);
BODY                i = i + 64u;
            }
        }
STORE        workgroupBarrier();
        var stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (l < stride) {
TREE            }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        if (lid < NBu && r0 < rows) {
            let j = lid;
            let t = ((u_part[j * 256u] + u_part[j * 256u + 64u]) + u_part[j * 256u + 128u]) + u_part[j * 256u + 192u];
            let rw0 = unpack2x16float(qb_w[rs0 + (r0 >> 1u)]);
            let rw2 = unpack2x16float(qb_w[rs0 + ((r0 + 2u) >> 1u)]);
            let o = j * rows + r0;
            qb_y[o] = t.x * rw0.x;
            if (l1) { qb_y[o + 1u] = t.y * rw0.y; }
            if (l2) { qb_y[o + 2u] = t.z * rw2.x; }
            if (l3) { qb_y[o + 3u] = t.w * rw2.y; }
        }
        workgroupBarrier();
        wb_ = wb_ + nwg.x;
    }
}
"#;
    let head = SRC.split("// q8_2f_matvec_b, verbatim").next().unwrap().replace(
        "var<workgroup> k4_part: array<vec4<f32>, 256>;\nvar<workgroup> k8_part: array<vec4<f32>, 2048>;",
        &format!("var<workgroup> u_part: array<vec4<f32>, {}>;", 256 * nb),
    );
    head + &core
        .replace("DECL", &decl)
        .replace("BODY", &body)
        .replace("STORE", &store)
        .replace("TREE", &tree)
        .replace("NBu", &format!("{nb}u"))
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("q82_batch_bench needs --features gpu");
}

#[cfg(feature = "gpu")]
const SRC: &str = r#"
struct QbP { cols4: u32, rows: u32, cols: u32, nb: u32 };
@group(0) @binding(0) var<storage, read>       qb_w : array<u32>;
@group(0) @binding(1) var<storage, read>       qb_x : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> qb_y : array<f32>;
@group(0) @binding(3) var<uniform>             qb_p : QbP;
var<workgroup> qb_part: array<f32, 2048>;
var<workgroup> k4_part: array<vec4<f32>, 256>;
var<workgroup> k8_part: array<vec4<f32>, 2048>;

fn qb_i8x4(w: u32) -> vec4<f32> {
    let s = i32(w);
    let b0 = (s << 24u) >> 24u;
    let b1 = (s << 16u) >> 24u;
    let b2 = (s <<  8u) >> 24u;
    let b3 =  s          >> 24u;
    return vec4<f32>(f32(b0), f32(b1), f32(b2), f32(b3));
}

fn qb_f16x4(half: u32) -> vec4<f32> {
    let w = half >> 1u;
    let a = unpack2x16float(qb_w[w]);
    let b = unpack2x16float(qb_w[w + 1u]);
    if ((half & 1u) == 0u) {
        return vec4<f32>(a.x, a.y, b.x, b.y);
    }
    let c = unpack2x16float(qb_w[w + 2u]);
    return vec4<f32>(a.y, b.x, b.y, c.x);
}

// q8_2f_matvec_b, verbatim apart from the name.
@compute @workgroup_size(256)
fn cur(@builtin(workgroup_id) wid: vec3<u32>,
       @builtin(num_workgroups) nwg: vec3<u32>,
       @builtin(local_invocation_index) lid: u32) {
    let rows = qb_p.rows;
    let ngrp = qb_p.cols4;
    let qbytes = rows * qb_p.cols;
    let rs0 = qbytes >> 2u;
    let cs0h = (qbytes >> 1u) + rows;
    let sub = lid >> 6u;
    let l = lid & 63u;
    let b0 = wid.y * 8u;
    let nbh = min(8u, qb_p.nb - b0);
    let blocks = (rows + 3u) / 4u;
    var wb = wid.x;
    loop {
        if (wb >= blocks) { break; }
        let row = wb * 4u + sub;
        var acc: array<f32, 8>;
        for (var j = 0u; j < 8u; j = j + 1u) { acc[j] = 0.0; }
        if (row < rows) {
            let roww = row * ngrp;
            var i = l;
            loop {
                if (i >= ngrp) { break; }
                let wq = qb_i8x4(qb_w[roww + i]);
                let cs = qb_f16x4(cs0h + i * 4u);
                for (var j = 0u; j < 8u; j = j + 1u) {
                    if (j < nbh) {
                        acc[j] = acc[j] + dot(wq, qb_x[(b0 + j) * ngrp + i] * cs);
                    }
                }
                i = i + 64u;
            }
        }
        for (var j = 0u; j < 8u; j = j + 1u) { qb_part[lid * 8u + j] = acc[j]; }
        workgroupBarrier();
        var stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (l < stride) {
                for (var j = 0u; j < 8u; j = j + 1u) {
                    qb_part[lid * 8u + j] = qb_part[lid * 8u + j] + qb_part[(lid + stride) * 8u + j];
                }
            }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        if (l == 0u && row < rows) {
            let rw = unpack2x16float(qb_w[rs0 + (row >> 1u)]);
            var sc = rw.x;
            if ((row & 1u) == 1u) { sc = rw.y; }
            for (var j = 0u; j < nbh; j = j + 1u) {
                qb_y[(b0 + j) * rows + row] = qb_part[lid * 8u + j] * sc;
            }
        }
        workgroupBarrier();
        wb = wb + nwg.x;
    }
}

// q8_2f_matvec4k (the token graph's decode kernel on NVIDIA), one row of
// activations: four weight rows a workgroup, the columns split across the
// four 64-lane sub-blocks, a step's x and column scales loaded once for the
// four rows. Same arithmetic order as the engine's (its 4-step unroll adds
// i, i+64, i+128, i+192 in sequence, as this loop does).
@compute @workgroup_size(256)
fn mv4k(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
    let rows = qb_p.rows;
    let ngrp = qb_p.cols4;
    let qbytes = rows * qb_p.cols;
    let rs0 = qbytes >> 2u;
    let cs0h = (qbytes >> 1u) + rows;
    let sub = lid >> 6u;
    let l = lid & 63u;
    let qlen = ngrp / 4u;
    let kend = sub * qlen + qlen;
    let blocks = (rows + 3u) / 4u;
    var wb = wid.x;
    loop {
        if (wb >= blocks) { break; }
        let r0 = wb * 4u;
        let l1 = r0 + 1u < rows;
        let l2 = r0 + 2u < rows;
        let l3 = r0 + 3u < rows;
        var acc = vec4<f32>(0.0);
        if (r0 < rows) {
            let w0b = r0 * ngrp;
            let w1b = select(w0b, (r0 + 1u) * ngrp, l1);
            let w2b = select(w0b, (r0 + 2u) * ngrp, l2);
            let w3b = select(w0b, (r0 + 3u) * ngrp, l3);
            var i = sub * qlen + l;
            loop {
                if (i >= kend) { break; }
                let xs0 = qb_x[i] * qb_f16x4(cs0h + i * 4u);
                acc.x = acc.x + dot(qb_i8x4(qb_w[w0b + i]), xs0);
                acc.y = acc.y + dot(qb_i8x4(qb_w[w1b + i]), xs0);
                acc.z = acc.z + dot(qb_i8x4(qb_w[w2b + i]), xs0);
                acc.w = acc.w + dot(qb_i8x4(qb_w[w3b + i]), xs0);
                i = i + 64u;
            }
        }
        k4_part[lid] = acc;
        workgroupBarrier();
        var stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (l < stride) { k4_part[lid] = k4_part[lid] + k4_part[lid + stride]; }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        if (lid == 0u && r0 < rows) {
            let t = ((k4_part[0u] + k4_part[64u]) + k4_part[128u]) + k4_part[192u];
            let rw0 = unpack2x16float(qb_w[rs0 + (r0 >> 1u)]);
            let rw2 = unpack2x16float(qb_w[rs0 + ((r0 + 2u) >> 1u)]);
            qb_y[r0] = t.x * rw0.x;
            if (l1) { qb_y[r0 + 1u] = t.y * rw0.y; }
            if (l2) { qb_y[r0 + 2u] = t.z * rw2.x; }
            if (l3) { qb_y[r0 + 3u] = t.w * rw2.y; }
        }
        workgroupBarrier();
        wb = wb + nwg.x;
    }
}

// mv4k over up to eight batch rows: the weight words of the four rows are
// read once for all of them; each batch row's sums are mv4k's to the bit.
@compute @workgroup_size(256)
fn b4k(@builtin(workgroup_id) wid: vec3<u32>,
       @builtin(num_workgroups) nwg: vec3<u32>,
       @builtin(local_invocation_index) lid: u32) {
    let rows = qb_p.rows;
    let ngrp = qb_p.cols4;
    let qbytes = rows * qb_p.cols;
    let rs0 = qbytes >> 2u;
    let cs0h = (qbytes >> 1u) + rows;
    let sub = lid >> 6u;
    let l = lid & 63u;
    let qlen = ngrp / 4u;
    let kend = sub * qlen + qlen;
    let b0 = wid.y * 8u;
    let nbh = min(8u, qb_p.nb - b0);
    let blocks = (rows + 3u) / 4u;
    var wb = wid.x;
    loop {
        if (wb >= blocks) { break; }
        let r0 = wb * 4u;
        let l1 = r0 + 1u < rows;
        let l2 = r0 + 2u < rows;
        let l3 = r0 + 3u < rows;
        var acc: array<vec4<f32>, 8>;
        for (var j = 0u; j < 8u; j = j + 1u) { acc[j] = vec4<f32>(0.0); }
        if (r0 < rows) {
            let w0b = r0 * ngrp;
            let w1b = select(w0b, (r0 + 1u) * ngrp, l1);
            let w2b = select(w0b, (r0 + 2u) * ngrp, l2);
            let w3b = select(w0b, (r0 + 3u) * ngrp, l3);
            var i = sub * qlen + l;
            loop {
                if (i >= kend) { break; }
                let cs = qb_f16x4(cs0h + i * 4u);
                let a = qb_i8x4(qb_w[w0b + i]);
                let b = qb_i8x4(qb_w[w1b + i]);
                let c = qb_i8x4(qb_w[w2b + i]);
                let d = qb_i8x4(qb_w[w3b + i]);
                for (var j = 0u; j < 8u; j = j + 1u) {
                    if (j < nbh) {
                        let xs0 = qb_x[(b0 + j) * ngrp + i] * cs;
                        acc[j] = acc[j] + vec4<f32>(dot(a, xs0), dot(b, xs0), dot(c, xs0), dot(d, xs0));
                    }
                }
                i = i + 64u;
            }
        }
        for (var j = 0u; j < 8u; j = j + 1u) { k8_part[lid * 8u + j] = acc[j]; }
        workgroupBarrier();
        var stride = 32u;
        loop {
            if (stride == 0u) { break; }
            if (l < stride) {
                for (var j = 0u; j < 8u; j = j + 1u) {
                    k8_part[lid * 8u + j] = k8_part[lid * 8u + j] + k8_part[(lid + stride) * 8u + j];
                }
            }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        if (lid < nbh && r0 < rows) {
            let j = lid;
            let t = ((k8_part[j] + k8_part[64u * 8u + j]) + k8_part[128u * 8u + j]) + k8_part[192u * 8u + j];
            let rw0 = unpack2x16float(qb_w[rs0 + (r0 >> 1u)]);
            let rw2 = unpack2x16float(qb_w[rs0 + ((r0 + 2u) >> 1u)]);
            let o = (b0 + j) * rows + r0;
            qb_y[o] = t.x * rw0.x;
            if (l1) { qb_y[o + 1u] = t.y * rw0.y; }
            if (l2) { qb_y[o + 2u] = t.z * rw2.x; }
            if (l3) { qb_y[o + 3u] = t.w * rw2.y; }
        }
        workgroupBarrier();
        wb = wb + nwg.x;
    }
}
"#;
