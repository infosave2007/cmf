//! Layout probes for the cooperative-matrix ops the attend tile uses beyond
//! the GEMM's pattern: loads with a stride of 64, a store with a stride of
//! 16, an accumulator stored, scaled and loaded back, and a `var` declared
//! inside a loop (zeroed per iteration?). One 16 x 16 x 16 product each,
//! against the CPU.
//!
//!     CMF_GPU=wgpu cargo test -p cortiq-engine --features gpu --release \
//!         --test gpu_coop_probe -- --nocapture

#[cfg(feature = "gpu")]
fn val(i: usize, salt: usize) -> f32 {
    // small integers over 8: exact in f16, exact sums in f32
    (((i * 7919 + salt * 104_729) % 17) as f32 - 8.0) / 8.0
}

#[cfg(feature = "gpu")]
const HEAD: &str = r#"
enable wgpu_cooperative_matrix;
enable f16;
@group(0) @binding(0) var<storage, read> pa: array<f32>;
@group(0) @binding(1) var<storage, read> pb: array<f32>;
@group(0) @binding(2) var<storage, read_write> po: array<f32>;
var<workgroup> wa: array<f16, 16 * 64>;
var<workgroup> wb: array<f16, 16 * 64>;
var<workgroup> wc: array<f32, 16 * 64>;
var<workgroup> wc16: array<f32, 256>;
var<workgroup> wk: array<f16, 16 * 64>;
"#;

/// A [16 m][16 k] into wa with stride S, B [16 n][16 k] into wb with stride S.
#[cfg(feature = "gpu")]
fn stage(stride: u32) -> String {
    format!(
        r#"
    for (var t = lid; t < 256u; t = t + 128u) {{
        let m = t / 16u; let k = t % 16u;
        wa[m * {stride}u + k] = f16(pa[m * 16u + k]);
        wb[m * {stride}u + k] = f16(pb[m * 16u + k]);
    }}
    workgroupBarrier();
"#
    )
}

#[cfg(feature = "gpu")]
fn probe(name: &str, head: &str, body: &str, a: &[f32], b: &[f32], want: &[f32], failed: &mut Vec<String>) {
    let src = format!(
        "{head}\n@compute @workgroup_size(128)\nfn probe(@builtin(local_invocation_index) lid: u32, @builtin(subgroup_id) sg: u32, @builtin(subgroup_invocation_id) sl: u32) {{\n{body}\n}}\n"
    );
    let mut got = vec![0f32; want.len()];
    if !cortiq_engine::gpu_wgpu::coop_probe_for_test(&src, "probe", a, b, &mut got) {
        println!("{name}: REJECTED");
        failed.push(format!("{name} rejected"));
        return;
    }
    let num: f64 = got.iter().zip(want).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = want.iter().map(|y| (*y as f64).powi(2)).sum::<f64>().max(1e-30);
    let rel = (num / den).sqrt();
    println!("{name}: rel {rel:.3e}  got[0..6] {:?}  want[0..6] {:?}", &got[..6], &want[..6]);
    if rel > 1e-5 {
        failed.push(format!("{name} rel {rel:.3e}"));
    }
}

#[cfg(feature = "gpu")]
#[test]
fn coop_layout_probes_needs_cmf_gpu() {
    match cortiq_engine::gpu_wgpu::selected_and_up() {
        None => {
            cortiq_engine::gpu_wgpu::skip_or_fail(module_path!());
            return;
        }
        Some(false) => panic!("wgpu selected but the context did not come up"),
        Some(true) => {}
    }
    let a: Vec<f32> = (0..256).map(|i| val(i, 1)).collect();
    let b: Vec<f32> = (0..256).map(|i| val(i, 2)).collect();
    // C[m][n] = sum_k A[m][k] B[n][k]
    let mut c = vec![0f32; 256];
    for m in 0..16 {
        for n in 0..16 {
            c[m * 16 + n] = (0..16).map(|k| a[m * 16 + k] * b[n * 16 + k]).sum();
        }
    }
    let mut failed = Vec::new();
    // 1. The GEMM's pattern: stride 32 loads, stride 64 store, copy out.
    let st32 = stage(32);
    let body = format!(
        r#"{st32}
    let am = coopLoadT<coop_mat16x16<f16, A>>(&wa[0u], 32u);
    let bm = coopLoad<coop_mat16x16<f16, B>>(&wb[0u], 32u);
    var cm: coop_mat16x16<f32, C>;
    cm = coopMultiplyAdd(am, bm, cm);
    workgroupBarrier();
    if (sg == 0u) {{ coopStoreT(cm, &wc[0u], 64u); }}
    workgroupBarrier();
    for (var t = lid; t < 256u; t = t + 128u) {{ po[t] = wc[(t / 16u) * 64u + t % 16u]; }}
"#
    );
    probe("control stride32", HEAD, &body, &a, &b, &c, &mut failed);
    // 2. Loads with stride 64 (the attend's P and V^T tiles).
    let st64 = stage(64);
    let body = format!(
        r#"{st64}
    let am = coopLoadT<coop_mat16x16<f16, A>>(&wa[0u], 64u);
    let bm = coopLoad<coop_mat16x16<f16, B>>(&wb[0u], 64u);
    var cm: coop_mat16x16<f32, C>;
    cm = coopMultiplyAdd(am, bm, cm);
    workgroupBarrier();
    if (sg == 0u) {{ coopStoreT(cm, &wc[0u], 64u); }}
    workgroupBarrier();
    for (var t = lid; t < 256u; t = t + 128u) {{ po[t] = wc[(t / 16u) * 64u + t % 16u]; }}
"#
    );
    probe("loads stride64", HEAD, &body, &a, &b, &c, &mut failed);
    // 3. Store with stride 16, copied out by subgroup 0's lanes as the
    //    attend's output copy does (row sl>>1, eight columns at (sl&1)*8).
    let body = format!(
        r#"{st32}
    let am = coopLoadT<coop_mat16x16<f16, A>>(&wa[0u], 32u);
    let bm = coopLoad<coop_mat16x16<f16, B>>(&wb[0u], 32u);
    var cm: coop_mat16x16<f32, C>;
    cm = coopMultiplyAdd(am, bm, cm);
    workgroupBarrier();
    if (sg == 0u) {{ coopStoreT(cm, &wc16[0u], 16u); }}
    workgroupBarrier();
    if (sg == 0u) {{
        let src = (sl >> 1u) * 16u + (sl & 1u) * 8u;
        for (var j = 0u; j < 8u; j = j + 1u) {{ po[src + j] = wc16[src + j]; }}
    }}
"#
    );
    probe("store stride16 + lane copy", HEAD, &body, &a, &b, &c, &mut failed);
    // 4. Accumulator round trip: store, scale row m by (m + 1), load back as C, store again.
    let mut c_scaled = c.clone();
    for m in 0..16 {
        for n in 0..16 {
            c_scaled[m * 16 + n] *= (m + 1) as f32;
        }
    }
    let body = format!(
        r#"{st32}
    let am = coopLoadT<coop_mat16x16<f16, A>>(&wa[0u], 32u);
    let bm = coopLoad<coop_mat16x16<f16, B>>(&wb[0u], 32u);
    var cm: coop_mat16x16<f32, C>;
    cm = coopMultiplyAdd(am, bm, cm);
    workgroupBarrier();
    if (sg == 0u) {{ coopStoreT(cm, &wc[0u], 64u); }}
    workgroupBarrier();
    for (var t = lid; t < 1024u; t = t + 128u) {{ wc[t] = wc[t] * f32((t / 64u) + 1u); }}
    workgroupBarrier();
    cm = coopLoadT<coop_mat16x16<f32, C>>(&wc[0u], 64u);
    workgroupBarrier();
    if (sg == 0u) {{ coopStoreT(cm, &wc[0u], 64u); }}
    workgroupBarrier();
    for (var t = lid; t < 256u; t = t + 128u) {{ po[t] = wc[(t / 16u) * 64u + t % 16u]; }}
"#
    );
    probe("accumulator store/scale/load", HEAD, &body, &a, &b, &c_scaled, &mut failed);
    // 5. A `var` accumulator declared inside a loop: zeroed per iteration?
    //    Two iterations of the same product → A·B if zeroed, 2·A·B if not.
    let body = format!(
        r#"{st32}
    let am = coopLoadT<coop_mat16x16<f16, A>>(&wa[0u], 32u);
    let bm = coopLoad<coop_mat16x16<f16, B>>(&wb[0u], 32u);
    for (var it = 0u; it < 2u; it = it + 1u) {{
        var cm: coop_mat16x16<f32, C>;
        cm = coopMultiplyAdd(am, bm, cm);
        workgroupBarrier();
        if (sg == 0u) {{ coopStoreT(cm, &wc[0u], 64u); }}
        workgroupBarrier();
    }}
    for (var t = lid; t < 256u; t = t + 128u) {{ po[t] = wc[(t / 16u) * 64u + t % 16u]; }}
"#
    );
    probe("var in loop zeroed", HEAD, &body, &a, &b, &c, &mut failed);
    // 6. Sixteen resident accumulators, each the same product; the fifth
    //    and the last read back (register pressure / spills).
    let mut body = st32.clone();
    body.push_str(
        r#"
    let am = coopLoadT<coop_mat16x16<f16, A>>(&wa[0u], 32u);
    let bm = coopLoad<coop_mat16x16<f16, B>>(&wb[0u], 32u);
"#,
    );
    for i in 0..16 {
        body.push_str(&format!("    var o{i}: coop_mat16x16<f32, C>;\n"));
    }
    body.push_str("    for (var it = 0u; it < 3u; it = it + 1u) {\n");
    for i in 0..16 {
        body.push_str(&format!("        o{i} = coopMultiplyAdd(am, bm, o{i});\n"));
    }
    body.push_str("    }\n    workgroupBarrier();\n    if (sg == 0u) { coopStoreT(o4, &wc[0u], 64u); coopStoreT(o15, &wc[16u], 64u); }\n    workgroupBarrier();\n");
    body.push_str("    for (var t = lid; t < 512u; t = t + 128u) { po[t] = wc[(t / 32u) * 64u + t % 32u]; }\n");
    let mut c16 = vec![0f32; 512];
    for m in 0..16 {
        for n in 0..16 {
            c16[m * 32 + n] = 3.0 * c[m * 16 + n];
            c16[m * 32 + 16 + n] = 3.0 * c[m * 16 + n];
        }
    }
    probe("16 accumulators x3", HEAD, &body, &a, &b, &c16, &mut failed);
    // 7. P·V shape: A = P [16 rows][64 keys] (stride 64, k offset 16), B = V^T
    //    [16 dims][64 keys] (stride 64, same offset): C = P[:,16..32]·V[16..32,:].
    let body = format!(
        r#"
    for (var t = lid; t < 256u; t = t + 128u) {{
        let m = t / 16u; let k = t % 16u;
        wa[m * 64u + 16u + k] = f16(pa[m * 16u + k]);
        wb[m * 64u + 16u + k] = f16(pb[m * 16u + k]);
    }}
    workgroupBarrier();
    let am = coopLoadT<coop_mat16x16<f16, A>>(&wa[16u], 64u);
    let bm = coopLoad<coop_mat16x16<f16, B>>(&wb[16u], 64u);
    var cm: coop_mat16x16<f32, C>;
    cm = coopMultiplyAdd(am, bm, cm);
    workgroupBarrier();
    if (sg == 0u) {{ coopStoreT(cm, &wc[0u], 64u); }}
    workgroupBarrier();
    for (var t = lid; t < 256u; t = t + 128u) {{ po[t] = wc[(t / 16u) * 64u + t % 16u]; }}
"#
    );
    probe("stride64 with k offset 16", HEAD, &body, &a, &b, &c, &mut failed);
    assert!(failed.is_empty(), "coop probes failed: {failed:?}");
}

/// The attend tile's two phases as they stand in the kernel, one workgroup of
/// four subgroups each owning 16 rows: the score phase (64 rows x 64 keys over
/// k = 32, staged as the kernel stages Q and K), the same with sixteen resident
/// accumulators alive, the value phase (P 64 x 64 against V^T, the output through
/// the stride-16 store and the lane copy), and the A operand loaded straight from
/// an f16 storage buffer (no staging) — the shape the next GEMM wants.
#[cfg(feature = "gpu")]
const HEAD2: &str = r#"
enable wgpu_cooperative_matrix;
enable f16;
@group(0) @binding(0) var<storage, read> pa: array<f32>;
@group(0) @binding(1) var<storage, read> pb: array<f32>;
@group(0) @binding(2) var<storage, read_write> po: array<f32>;
var<workgroup> xa: array<f16, 64 * 32>;
var<workgroup> xb: array<f16, 64 * 32>;
var<workgroup> xs: array<f32, 64 * 64>;
var<workgroup> xp: array<f16, 64 * 64>;
var<workgroup> xvt: array<f16, 64 * 64>;
var<workgroup> xc: array<f32, 4 * 256>;
"#;
#[cfg(feature = "gpu")]
const HEAD5: &str = r#"
enable wgpu_cooperative_matrix;
enable f16;
@group(0) @binding(0) var<storage, read> pa: array<f32>;
@group(0) @binding(1) var<storage, read> pb: array<f32>;
@group(0) @binding(2) var<storage, read_write> po: array<f32>;
var<workgroup> xp: array<f16, 64 * 64>;
var<workgroup> xvt: array<f16, 64 * 64>;
var<workgroup> xc: array<f32, 4 * 256>;
var<workgroup> pflag: u32;
var<workgroup> pwid: vec2<u32>;
"#;
#[cfg(feature = "gpu")]
const HEAD6: &str = r#"
enable wgpu_cooperative_matrix;
enable f16;
@group(0) @binding(0) var<storage, read> pa: array<f32>;
@group(0) @binding(1) var<storage, read> pb: array<f32>;
@group(0) @binding(2) var<storage, read_write> po: array<f32>;
var<workgroup> xp: array<f16, 64 * 64>;
var<workgroup> xvt: array<f16, 64 * 64>;
var<workgroup> xc: array<f32, 4 * 256>;
var<workgroup> qflag: vec4<u32>;
var<workgroup> qwid: vec4<u32>;
"#;
#[cfg(feature = "gpu")]
const HEAD3: &str = r#"
enable wgpu_cooperative_matrix;
enable f16;
@group(0) @binding(0) var<storage, read> pa: array<f16>;
@group(0) @binding(1) var<storage, read> pb: array<f32>;
@group(0) @binding(2) var<storage, read_write> po: array<f32>;
var<workgroup> xb: array<f16, 64 * 32>;
var<workgroup> xs: array<f32, 64 * 64>;
"#;
#[cfg(feature = "gpu")]
const HEAD4: &str = r#"
enable wgpu_cooperative_matrix;
enable f16;
@group(0) @binding(0) var<storage, read> pa: array<f16>;
@group(0) @binding(1) var<storage, read> pb: array<f16>;
@group(0) @binding(2) var<storage, read_write> po: array<f32>;
var<workgroup> xs: array<f32, 64 * 64>;
"#;
/// Stage A [64][32] and B [64][32] from pa/pb as the kernel stages Q and K.
#[cfg(feature = "gpu")]
const STAGE_AB: &str = r#"
    for (var t = lid; t < 512u; t = t + 128u) {
        let r = t >> 3u;
        let kq = t & 7u;
        let dst = r * 32u + kq * 4u;
        let src = r * 32u + kq * 4u;
        xa[dst] = f16(pa[src]); xa[dst + 1u] = f16(pa[src + 1u]);
        xa[dst + 2u] = f16(pa[src + 2u]); xa[dst + 3u] = f16(pa[src + 3u]);
        xb[dst] = f16(pb[src]); xb[dst + 1u] = f16(pb[src + 1u]);
        xb[dst + 2u] = f16(pb[src + 2u]); xb[dst + 3u] = f16(pb[src + 3u]);
    }
    workgroupBarrier();
"#;
#[cfg(feature = "gpu")]
const STAGE_B: &str = r#"
    for (var t = lid; t < 512u; t = t + 128u) {
        let r = t >> 3u;
        let kq = t & 7u;
        let dst = r * 32u + kq * 4u;
        xb[dst] = f16(pb[dst]); xb[dst + 1u] = f16(pb[dst + 1u]);
        xb[dst + 2u] = f16(pb[dst + 2u]); xb[dst + 3u] = f16(pb[dst + 3u]);
    }
    workgroupBarrier();
"#;
/// S = A·Bᵀ for the subgroup's 16 rows against 64 columns, `asrc` names the
/// A array (xa or pa) and `bsrc` the B array; stored to xs then copied out.
#[cfg(feature = "gpu")]
fn s_phase(asrc: &str, bsrc: &str) -> String {
    format!(
        r#"
    var c0m: coop_mat16x16<f32, C>; var c1m: coop_mat16x16<f32, C>;
    var c2m: coop_mat16x16<f32, C>; var c3m: coop_mat16x16<f32, C>;
    {{
        let a = coopLoadT<coop_mat16x16<f16, A>>(&{asrc}[sg * 512u], 32u);
        let b0t = coopLoad<coop_mat16x16<f16, B>>(&{bsrc}[0u], 32u);
        let b1t = coopLoad<coop_mat16x16<f16, B>>(&{bsrc}[512u], 32u);
        let b2t = coopLoad<coop_mat16x16<f16, B>>(&{bsrc}[1024u], 32u);
        let b3t = coopLoad<coop_mat16x16<f16, B>>(&{bsrc}[1536u], 32u);
        c0m = coopMultiplyAdd(a, b0t, c0m);
        c1m = coopMultiplyAdd(a, b1t, c1m);
        c2m = coopMultiplyAdd(a, b2t, c2m);
        c3m = coopMultiplyAdd(a, b3t, c3m);
    }}
    {{
        let a = coopLoadT<coop_mat16x16<f16, A>>(&{asrc}[sg * 512u + 16u], 32u);
        let b0t = coopLoad<coop_mat16x16<f16, B>>(&{bsrc}[16u], 32u);
        let b1t = coopLoad<coop_mat16x16<f16, B>>(&{bsrc}[512u + 16u], 32u);
        let b2t = coopLoad<coop_mat16x16<f16, B>>(&{bsrc}[1024u + 16u], 32u);
        let b3t = coopLoad<coop_mat16x16<f16, B>>(&{bsrc}[1536u + 16u], 32u);
        c0m = coopMultiplyAdd(a, b0t, c0m);
        c1m = coopMultiplyAdd(a, b1t, c1m);
        c2m = coopMultiplyAdd(a, b2t, c2m);
        c3m = coopMultiplyAdd(a, b3t, c3m);
    }}
    workgroupBarrier();
    coopStoreT(c0m, &xs[sg * 1024u], 64u);
    coopStoreT(c1m, &xs[sg * 1024u + 16u], 64u);
    coopStoreT(c2m, &xs[sg * 1024u + 32u], 64u);
    coopStoreT(c3m, &xs[sg * 1024u + 48u], 64u);
    workgroupBarrier();
    for (var t = lid; t < 4096u; t = t + 128u) {{ po[t] = xs[t]; }}
"#
    )
}
#[cfg(feature = "gpu")]
fn f16_bits(x: f32) -> u16 {
    // exact for the probe values (k / 8, |k| <= 8): normal range, short mantissa
    let b = x.to_bits();
    let s = (b >> 16) & 0x8000;
    if x == 0.0 {
        return s as u16;
    }
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    assert!((1..31).contains(&e), "f16_bits: {x} out of the normal range");
    (s | ((e as u32) << 10) | ((b & 0x7f_ffff) >> 13)) as u16
}
#[cfg(feature = "gpu")]
fn pack_f16(v: &[f32]) -> Vec<f32> {
    v.chunks(2)
        .map(|c| f32::from_bits(f16_bits(c[0]) as u32 | ((f16_bits(c[1]) as u32) << 16)))
        .collect()
}
#[cfg(feature = "gpu")]
#[test]
fn coop_attend_phase_probes_needs_cmf_gpu() {
    match cortiq_engine::gpu_wgpu::selected_and_up() {
        None => {
            cortiq_engine::gpu_wgpu::skip_or_fail(module_path!());
            return;
        }
        Some(false) => panic!("wgpu selected but the context did not come up"),
        Some(true) => {}
    }
    let mut failed = Vec::new();
    // A [64 m][32 k], B [64 n][32 k]; C[m][n] = sum_k A B
    let a: Vec<f32> = (0..64 * 32).map(|i| val(i, 1)).collect();
    let b: Vec<f32> = (0..64 * 32).map(|i| val(i, 2)).collect();
    let mut c = vec![0f32; 64 * 64];
    for m in 0..64 {
        for n in 0..64 {
            c[m * 64 + n] = (0..32).map(|k| a[m * 32 + k] * b[n * 32 + k]).sum();
        }
    }
    // 8. The score phase: four subgroups, four C tiles each, staged A and B.
    let body = format!("{STAGE_AB}{}", s_phase("xa", "xb"));
    probe("S phase 4 subgroups", HEAD2, &body, &a, &b, &c, &mut failed);
    // 9. The same with sixteen resident accumulators alive through it.
    let mut body = String::from(STAGE_AB);
    for i in 0..16 {
        body.push_str(&format!("    var o{i}: coop_mat16x16<f32, C>;\n"));
    }
    body.push_str(
        r#"
    {
        let a = coopLoadT<coop_mat16x16<f16, A>>(&xa[sg * 512u], 32u);
        let b0t = coopLoad<coop_mat16x16<f16, B>>(&xb[0u], 32u);
"#,
    );
    for i in 0..16 {
        body.push_str(&format!("        o{i} = coopMultiplyAdd(a, b0t, o{i});\n"));
    }
    body.push_str("    }\n");
    body.push_str(&s_phase("xa", "xb"));
    body.push_str("    workgroupBarrier();\n");
    for i in 0..16 {
        body.push_str(&format!("    coopStoreT(o{i}, &xc[sg * 256u], 16u);\n"));
    }
    probe("S phase + 16 accumulators", HEAD2, &body, &a, &b, &c, &mut failed);
    // 10. The value phase: P [64 rows][64 keys] against V [64 keys][64 dims],
    //     V^T staged as the kernel does, four accumulators a subgroup, out
    //     through the stride-16 store and the lane copy.
    let p: Vec<f32> = (0..64 * 64).map(|i| val(i, 3)).collect();
    let v: Vec<f32> = (0..64 * 64).map(|i| val(i, 4)).collect();
    let mut o = vec![0f32; 64 * 64];
    for r in 0..64 {
        for d in 0..64 {
            o[r * 64 + d] = (0..64).map(|k| p[r * 64 + k] * v[k * 64 + d]).sum();
        }
    }
    let mut pv_body = String::from(
        r#"
    for (var t = lid; t < 4096u; t = t + 128u) { xp[t] = f16(pa[t]); }
    for (var t = lid; t < 64u * 16u; t = t + 128u) {
        let p = t >> 4u;
        let d4 = t & 15u;
        let dd = d4 * 4u;
        xvt[dd * 64u + p] = f16(pb[p * 64u + dd]);
        xvt[(dd + 1u) * 64u + p] = f16(pb[p * 64u + dd + 1u]);
        xvt[(dd + 2u) * 64u + p] = f16(pb[p * 64u + dd + 2u]);
        xvt[(dd + 3u) * 64u + p] = f16(pb[p * 64u + dd + 3u]);
    }
    workgroupBarrier();
    var o0: coop_mat16x16<f32, C>; var o1: coop_mat16x16<f32, C>;
    var o2: coop_mat16x16<f32, C>; var o3: coop_mat16x16<f32, C>;
    for (var ks = 0u; ks < 64u; ks = ks + 16u) {
        let a = coopLoadT<coop_mat16x16<f16, A>>(&xp[sg * 1024u + ks], 64u);
        let v0 = coopLoad<coop_mat16x16<f16, B>>(&xvt[ks], 64u);
        let v1 = coopLoad<coop_mat16x16<f16, B>>(&xvt[1024u + ks], 64u);
        let v2 = coopLoad<coop_mat16x16<f16, B>>(&xvt[2048u + ks], 64u);
        let v3 = coopLoad<coop_mat16x16<f16, B>>(&xvt[3072u + ks], 64u);
        o0 = coopMultiplyAdd(a, v0, o0);
        o1 = coopMultiplyAdd(a, v1, o1);
        o2 = coopMultiplyAdd(a, v2, o2);
        o3 = coopMultiplyAdd(a, v3, o3);
    }
    let orow = sg * 16u + (sl >> 1u);
    let ocol = (sl & 1u) * 8u;
"#,
    );
    for j in 0..4 {
        pv_body.push_str(&format!(
            r#"
    coopStoreT(o{j}, &xc[sg * 256u], 16u);
    workgroupBarrier();
    {{
        let src = sg * 256u + (sl >> 1u) * 16u + ocol;
        let dst = orow * 64u + {j}u * 16u + ocol;
        for (var q = 0u; q < 8u; q = q + 1u) {{ po[dst + q] = xc[src + q]; }}
    }}
    workgroupBarrier();
"#
        ));
    }
    probe("PV phase", HEAD2, &pv_body, &p, &v, &o, &mut failed);
    // 10b. The same with the attend kernel's small workgroup scalars declared
    //      and read through workgroupUniformLoad ahead of it (a 4-byte and an
    //      8-byte variable next to the arrays: does their presence shift the
    //      cooperative loads and stores?).
    let body = format!(
        r#"
    if (lid == 0u) {{ pwid = vec2<u32>(0u); pflag = 0u; }}
    let uw = workgroupUniformLoad(&pwid);
    let need = workgroupUniformLoad(&pflag);
    if ((uw.x == 0u) & (need == 0u)) {{
{pv_body}
    }}
"#
    );
    // The alignment trap itself, kept as a record and NOT as a gate: a u32
    // and a vec2<u32> next to the tile arrays shift every cooperative load
    // and store by one element on NVIDIA (rel 1.35 on the RTX 2000 Ada).
    // A driver that gets this right is not a failure either way.
    let mut trap = Vec::new();
    probe("PV phase + u32/vec2 scalars (trap)", HEAD5, &body, &p, &v, &o, &mut trap);
    println!(
        "trap probe: {}",
        if trap.is_empty() { "this driver aligns small workgroup scalars" } else { "REPRODUCED (expected on NVIDIA)" }
    );
    let body = format!(
        r#"
    if (lid == 0u) {{ qwid = vec4<u32>(0u); qflag = vec4<u32>(0u); }}
    let uw = workgroupUniformLoad(&qwid);
    let need = workgroupUniformLoad(&qflag).x;
    if ((uw.x == 0u) & (need == 0u)) {{
{pv_body}
    }}
"#
    );
    probe("PV phase + vec4 scalars", HEAD6, &body, &p, &v, &o, &mut failed);
    // 11. A straight from an f16 storage buffer (row-major, stride 32), B staged.
    let body = format!("{STAGE_B}{}", s_phase("pa", "xb"));
    probe("A from storage f16", HEAD3, &body, &pack_f16(&a), &b, &c, &mut failed);
    // 12. Both operands straight from f16 storage buffers.
    let body = s_phase("pa", "pb");
    probe("A and B from storage f16", HEAD4, &body, &pack_f16(&a), &pack_f16(&b), &c, &mut failed);
    // 13. B stored [k][n] (a V plane: keys down, dims across), stride 64, with
    // the transposing load — and with the plain one, so the record says which
    // of the two reads that layout as the B role. `gqa_attend_part_xc2` reads
    // V this way and takes whichever is exact here (CMF_ATTEND_XC2_VLOAD).
    let bt: Vec<f32> = (0..32 * 64).map(|i| b[(i % 64) * 32 + i / 64]).collect();
    let mut which = Vec::new();
    for (name, ld) in [("B [k][n] from storage via coopLoadT", "coopLoadT"), ("B [k][n] from storage via coopLoad", "coopLoad")] {
        let body = format!(
            r#"
    var c0m: coop_mat16x16<f32, C>; var c1m: coop_mat16x16<f32, C>;
    var c2m: coop_mat16x16<f32, C>; var c3m: coop_mat16x16<f32, C>;
    for (var kk = 0u; kk < 32u; kk = kk + 16u) {{
        let a = coopLoadT<coop_mat16x16<f16, A>>(&pa[sg * 512u + kk], 32u);
        let b0t = {ld}<coop_mat16x16<f16, B>>(&pb[kk * 64u], 64u);
        let b1t = {ld}<coop_mat16x16<f16, B>>(&pb[kk * 64u + 16u], 64u);
        let b2t = {ld}<coop_mat16x16<f16, B>>(&pb[kk * 64u + 32u], 64u);
        let b3t = {ld}<coop_mat16x16<f16, B>>(&pb[kk * 64u + 48u], 64u);
        c0m = coopMultiplyAdd(a, b0t, c0m);
        c1m = coopMultiplyAdd(a, b1t, c1m);
        c2m = coopMultiplyAdd(a, b2t, c2m);
        c3m = coopMultiplyAdd(a, b3t, c3m);
    }}
    workgroupBarrier();
    coopStoreT(c0m, &xs[sg * 1024u], 64u);
    coopStoreT(c1m, &xs[sg * 1024u + 16u], 64u);
    coopStoreT(c2m, &xs[sg * 1024u + 32u], 64u);
    coopStoreT(c3m, &xs[sg * 1024u + 48u], 64u);
    workgroupBarrier();
    for (var t = lid; t < 4096u; t = t + 128u) {{ po[t] = xs[t]; }}
"#
        );
        let mut f = Vec::new();
        probe(name, HEAD4, &body, &pack_f16(&a), &pack_f16(&bt), &c, &mut f);
        if f.is_empty() {
            which.push(ld);
        }
    }
    println!("B [k][n] layout reads as the B role through: {which:?}");
    if which.is_empty() {
        failed.push("no load reads a [k][n] f16 storage layout as the B role".into());
    }
    // 14. A from storage with a wide row (stride 256, the Q plane's hd): the
    // operand's 32 k values sit at the start of each 256-wide row.
    let aw: Vec<f32> = (0..64 * 256).map(|i| if i % 256 < 32 { a[(i / 256) * 32 + i % 256] } else { 0.0 }).collect();
    let body = format!(
        "{STAGE_B}{}",
        s_phase("pa", "xb").replace("&pa[sg * 512u], 32u", "&pa[sg * 4096u], 256u").replace("&pa[sg * 512u + 16u], 32u", "&pa[sg * 4096u + 16u], 256u")
    );
    probe("A from storage f16 stride 256", HEAD3, &body, &pack_f16(&aw), &b, &c, &mut failed);
    assert!(failed.is_empty(), "coop phase probes failed: {failed:?}");
}
