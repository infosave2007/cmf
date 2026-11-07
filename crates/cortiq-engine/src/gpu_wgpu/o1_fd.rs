//! A sealed layer's decode attention in two dispatches.
//!
//! `o1_stages` takes the far-field attention of one row apart into seven dispatches a
//! layer (near scores, landmark scores, the near max, the far part, the near weights, the
//! value sum in pieces, the pieces added). On a 42-layer 2B model the dispatches, not the
//! arithmetic, set the pace: CherryPit-1.2-2B Turbo decoded 68 tok/s on an RTX 5090 past
//! the seal, where the same model with exact attention over 2 000 rows ran at 156.
//!
//! Here the near rows (sinks, then the ring of the window and the exact rows) go through
//! `attend_fd`'s flash-decoding pass: a workgroup of eight 32-lane subgroups a chunk,
//! whole coalesced K/V rows, every query head of the group through one `subgroupAdd`,
//! the online softmax in registers, the subgroups met in workgroup memory. One merge
//! workgroup a query head then scores its 32 landmarks, builds the far part exactly as
//! `o1s_far` does (weights through the landmark inverse, one gain per landmark, the far
//! mass), takes the common exponent over the chunks' maxima and the far terms, and writes
//! (Σ near + Σ far) / (Σ near weights + far mass) — `o1s_soft` and `o1s_add`'s result.
//! `CMF_O1_FD=0` keeps the stages.

use super::*;

/// The part and merge pipelines for one (heads a group, head width).
pub(crate) struct O1Fd {
    pub(crate) part: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    pub(crate) merge: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
}

pub(crate) type O1FdPipe = std::sync::Arc<O1Fd>;

fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_O1_FD").as_deref() != Ok("0"))
}

fn sg32(c: &Ctx) -> bool {
    c.device.features().contains(wgpu::Features::SUBGROUP)
        && c.adapter_info.subgroup_min_size == 32
        && c.adapter_info.subgroup_max_size == 32
}

const HEAD: &str = r#"
struct OU { hpg: u32, m: u32, w: u32, ns: u32, d: u32, g: u32, scale: f32, ncap: u32 };
struct FU { ck: u32, nc: u32, _a: u32, _b: u32 };
"#;

fn part_src(hpg: usize, hd: usize) -> String {
    let hd4 = hd / 4;
    let slots = hd4.div_ceil(32);
    let guard = hd4 % 32 != 0;
    let w = crate::gpu_wgpu::attend_fd::FD_WARPS;
    let mut s = String::from(HEAD);
    s.push_str(
        "@group(0) @binding(0) var<uniform>             pu    : OU;\n\
         @group(0) @binding(1) var<storage, read>       pmeta : array<u32>;\n\
         @group(0) @binding(2) var<storage, read>       pq    : array<vec4<f32>>;\n\
         @group(0) @binding(3) var<storage, read>       prk   : array<vec4<f32>>;\n\
         @group(0) @binding(4) var<storage, read>       psk   : array<vec4<f32>>;\n\
         @group(0) @binding(5) var<storage, read>       prv   : array<vec4<f32>>;\n\
         @group(0) @binding(6) var<storage, read>       psv   : array<vec4<f32>>;\n\
         @group(0) @binding(7) var<storage, read_write> pacc  : array<f32>;\n\
         @group(0) @binding(8) var<storage, read_write> pml   : array<vec2<f32>>;\n\
         @group(0) @binding(9) var<uniform>             pf    : FU;\n",
    );
    s.push_str(&format!(
        "var<workgroup> fd_m: array<f32, {}>;\nvar<workgroup> fd_l: array<f32, {}>;\nvar<workgroup> fd_acc: array<f32, {}>;\n",
        w * hpg,
        w * hpg,
        w * hd
    ));
    s.push_str(
        "@compute @workgroup_size(256)\n\
         fn o1fd_part(@builtin(workgroup_id) wid: vec3<u32>,\n\
                      @builtin(local_invocation_index) lid: u32,\n\
                      @builtin(subgroup_invocation_id) lane: u32) {\n\
             let g = wid.x;\n\
             let ch = wid.y;\n\
             let warp = lid >> 5u;\n",
    );
    s.push_str(&format!("    let hd4 = {hd4}u;\n    let hpk = {hpg}u;\n"));
    s.push_str(
        "    let n = pu.ns + pmeta[g * 4u];\n\
             let per = pf.ck / 8u;\n\
             let p0 = ch * pf.ck + warp * per;\n\
             let pend = min(n, min((ch + 1u) * pf.ck, p0 + per));\n\
             let scale = pu.scale;\n",
    );
    for j in 0..slots {
        if guard {
            s.push_str(&format!("    let ok{j} = lane + {}u < hd4;\n", 32 * j));
        }
    }
    for h in 0..hpg {
        for j in 0..slots {
            let idx = format!("(g * hpk + {h}u) * hd4 + lane + {}u", 32 * j);
            if guard {
                s.push_str(&format!(
                    "    var q{h}_{j} = vec4<f32>(0.0);\n    if (ok{j}) {{ q{h}_{j} = pq[{idx}]; }}\n"
                ));
            } else {
                s.push_str(&format!("    let q{h}_{j} = pq[{idx}];\n"));
            }
            s.push_str(&format!("    var a{h}_{j} = vec4<f32>(0.0);\n"));
        }
        s.push_str(&format!("    var m{h} = -1.0e30;\n    var l{h} = 0.0;\n"));
    }
    s.push_str("    for (var p = p0; p < pend; p = p + 1u) {\n");
    for j in 0..slots {
        let o = 32 * j;
        let load = format!(
            "if (p < pu.ns) {{ let r = (g * pu.ns + p) * hd4 + lane + {o}u; k{j} = psk[r]; v{j} = psv[r]; }} else {{ let r = (g * pu.w + p - pu.ns) * hd4 + lane + {o}u; k{j} = prk[r]; v{j} = prv[r]; }}"
        );
        s.push_str(&format!("        var k{j} = vec4<f32>(0.0);\n        var v{j} = vec4<f32>(0.0);\n"));
        if guard {
            s.push_str(&format!("        if (ok{j}) {{ {load} }}\n"));
        } else {
            s.push_str(&format!("        {load}\n"));
        }
    }
    for h in 0..hpg {
        let dots: Vec<String> = (0..slots).map(|j| format!("dot(q{h}_{j}, k{j})")).collect();
        s.push_str(&format!(
            "        let s{h} = subgroupAdd({}) * scale;\n        let n{h} = max(m{h}, s{h});\n        let f{h} = exp(m{h} - n{h});\n        let w{h} = exp(s{h} - n{h});\n        l{h} = l{h} * f{h} + w{h};\n        m{h} = n{h};\n",
            dots.join(" + ")
        ));
        for j in 0..slots {
            s.push_str(&format!("        a{h}_{j} = a{h}_{j} * f{h} + v{j} * w{h};\n"));
        }
    }
    s.push_str("    }\n");
    for h in 0..hpg {
        s.push_str(&format!(
            "    if (lane == 0u) {{ fd_m[warp * hpk + {h}u] = m{h}; fd_l[warp * hpk + {h}u] = l{h}; }}\n"
        ));
        for j in 0..slots {
            let st = format!(
                "{{ let d = (lane + {o}u) * 4u; fd_acc[warp * {hd}u + d] = a{h}_{j}.x; fd_acc[warp * {hd}u + d + 1u] = a{h}_{j}.y; fd_acc[warp * {hd}u + d + 2u] = a{h}_{j}.z; fd_acc[warp * {hd}u + d + 3u] = a{h}_{j}.w; }}",
                o = 32 * j
            );
            if guard {
                s.push_str(&format!("    if (ok{j}) {st}\n"));
            } else {
                s.push_str(&format!("    {st}\n"));
            }
        }
        s.push_str(&format!(
            "    workgroupBarrier();\n\
                 if (lid < {hd}u) {{\n\
                     var mm = -1.0e30;\n\
                     for (var w = 0u; w < {w}u; w = w + 1u) {{ mm = max(mm, fd_m[w * hpk + {h}u]); }}\n\
                     var ll = 0.0;\n\
                     var aa = 0.0;\n\
                     for (var w = 0u; w < {w}u; w = w + 1u) {{\n\
                         let e = exp(fd_m[w * hpk + {h}u] - mm);\n\
                         ll = ll + fd_l[w * hpk + {h}u] * e;\n\
                         aa = aa + fd_acc[w * {hd}u + lid] * e;\n\
                     }}\n\
                     let idx = (g * hpk + {h}u) * pf.nc + ch;\n\
                     pacc[idx * {hd}u + lid] = aa;\n\
                     if (lid == 0u) {{ pml[idx] = vec2<f32>(mm, ll); }}\n\
                 }}\n\
                 workgroupBarrier();\n"
        ));
    }
    s.push_str("}\n");
    s
}

fn merge_src(hpg: usize, hd: usize) -> String {
    let hd4 = hd / 4;
    let mut s = String::from(HEAD);
    s.push_str(&format!(
        "@group(0) @binding(0) var<uniform>             mu0   : OU;\n\
         @group(0) @binding(1) var<storage, read>       mmeta : array<u32>;\n\
         @group(0) @binding(2) var<storage, read>       mq    : array<vec4<f32>>;\n\
         @group(0) @binding(3) var<storage, read>       mkt   : array<vec4<f32>>;\n\
         @group(0) @binding(4) var<storage, read>       mmu   : array<f32>;\n\
         @group(0) @binding(5) var<storage, read>       mmz   : array<f32>;\n\
         @group(0) @binding(6) var<storage, read>       mth   : array<vec4<f32>>;\n\
         @group(0) @binding(7) var<storage, read>       macc  : array<f32>;\n\
         @group(0) @binding(8) var<storage, read>       mml   : array<vec2<f32>>;\n\
         @group(0) @binding(9) var<uniform>             mf    : FU;\n\
         @group(0) @binding(10) var<storage, read_write> mout : array<f32>;\n\
         var<workgroup> fe: array<f32, 32>;\n\
         var<workgroup> fu: array<f32, 32>;\n\
         var<workgroup> fg: array<f32, 32>;\n\
         @compute @workgroup_size(256)\n\
         fn o1fd_merge(@builtin(workgroup_id) wid: vec3<u32>,\n\
                       @builtin(local_invocation_index) lid: u32) {{\n\
             let gh = wid.x;\n\
             let g = gh / {hpg}u;\n\
             let m = mu0.m;\n\
             let hd4 = {hd4}u;\n\
             let nc = mf.nc;\n\
             let far = mmeta[g * 4u + 2u] > 0u;\n\
             // Landmark scores of this head.\n\
             if (far && lid < m) {{\n\
                 var a4 = vec4<f32>(0.0);\n\
                 for (var e = 0u; e < hd4; e = e + 1u) {{ a4 = a4 + mq[gh * hd4 + e] * mkt[(g * m + lid) * hd4 + e]; }}\n\
                 fe[lid] = (a4.x + a4.y + a4.z + a4.w) * mu0.scale;\n\
             }}\n\
             workgroupBarrier();\n\
             var f = -3.0e38;\n\
             if (far) {{ for (var a = 0u; a < m; a = a + 1u) {{ f = max(f, fe[a]); }} }}\n\
             if (far && lid < m) {{\n\
                 var u = 0.0;\n\
                 for (var a = 0u; a < m; a = a + 1u) {{ u = u + exp(fe[a] - f) * mmu[(gh * m + a) * m + lid]; }}\n\
                 fu[lid] = u;\n\
             }}\n\
             // The common exponent: the chunks' maxima and the far terms.\n\
             var c_all = -1.0e30;\n\
             for (var c = 0u; c < nc; c = c + 1u) {{ c_all = max(c_all, mml[gh * nc + c].x); }}\n\
             let mzb = gh * 2u * m;\n\
             if (far) {{ for (var b = 0u; b < m; b = b + 1u) {{ c_all = max(c_all, f + mmz[mzb + b]); }} }}\n\
             workgroupBarrier();\n\
             if (far && lid < m) {{ fg[lid] = fu[lid] * exp(f + mmz[mzb + lid] - c_all); }}\n\
             workgroupBarrier();\n\
             var far_den = 0.0;\n\
             var have = false;\n\
             if (far) {{\n\
                 for (var b = 0u; b < m; b = b + 1u) {{ far_den = far_den + fg[b] * mmz[mzb + m + b]; }}\n\
                 if (far_den >= 0.0) {{ have = true; }} else {{ far_den = 0.0; }}\n\
             }}\n\
             var den = far_den;\n\
             for (var c = 0u; c < nc; c = c + 1u) {{\n\
                 let ml = mml[gh * nc + c];\n\
                 den = den + ml.y * exp(ml.x - c_all);\n\
             }}\n\
             let inv = 1.0 / max(den, 1.0e-30);\n\
             for (var d = lid; d < {hd}u; d = d + 256u) {{\n\
                 var acc = 0.0;\n\
                 for (var c = 0u; c < nc; c = c + 1u) {{\n\
                     acc = acc + macc[(gh * nc + c) * {hd}u + d] * exp(mml[gh * nc + c].x - c_all);\n\
                 }}\n\
                 if (have) {{\n\
                     let dw = d >> 2u;\n\
                     let dc = d & 3u;\n\
                     for (var b = 0u; b < m; b = b + 1u) {{ acc = acc + fg[b] * mth[(gh * m + b) * hd4 + dw][dc]; }}\n\
                 }}\n\
                 mout[gh * {hd}u + d] = acc * inv;\n\
             }}\n\
         }}\n"
    ));
    s
}

fn compile(c: &Ctx, label: &str, src: &str, entry: &str) -> Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> {
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let p = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: crate::gpu_wgpu::pco(),
        cache: c.pipeline_cache.as_ref(),
    });
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("{label} rejected ({e}): sealed layers keep the o1 stages");
        return None;
    }
    let l = p.get_bind_group_layout(0);
    Some((p, l))
}

/// The two kernels for a sealed layer of `hpg` query heads per KV head of width `hd` and
/// `m` landmarks; None where they do not apply (no 32-lane subgroups, hd not 64/128/256,
/// more than 8 heads a group, more than 32 landmarks, or `CMF_O1_FD=0`).
pub(crate) fn pipe(c: &Ctx, hpg: usize, hd: usize, m: usize) -> Option<O1FdPipe> {
    if !enabled() || hpg == 0 || hpg > 8 || m > 32 || !matches!(hd, 64 | 128 | 256) || !sg32(c) {
        return None;
    }
    let mut map = c.o1_fd_pipes.lock().ok()?;
    map.entry((hpg, hd))
        .or_insert_with(|| {
            let part = compile(c, "cmf-o1fd-part", &part_src(hpg, hd), "o1fd_part")?;
            let merge = compile(c, "cmf-o1fd-merge", &merge_src(hpg, hd), "o1fd_merge")?;
            Some(std::sync::Arc::new(O1Fd { part, merge }))
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every shape parses and validates (naga, subgroup capability on), with no GPU.
    #[test]
    fn sources_validate() {
        for hpg in 1..=8 {
            for hd in [64, 128, 256] {
                for (name, s) in [("part", part_src(hpg, hd)), ("merge", merge_src(hpg, hd))] {
                    let module = wgpu::naga::front::wgsl::parse_str(&s)
                        .unwrap_or_else(|e| panic!("{name} hpg {hpg} hd {hd}: {e}"));
                    wgpu::naga::valid::Validator::new(
                        wgpu::naga::valid::ValidationFlags::all(),
                        wgpu::naga::valid::Capabilities::all(),
                    )
                    .validate(&module)
                    .unwrap_or_else(|e| panic!("{name} hpg {hpg} hd {hd}: {e:?}"));
                }
            }
        }
    }
}
