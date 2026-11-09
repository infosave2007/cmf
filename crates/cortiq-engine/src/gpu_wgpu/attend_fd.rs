//! Flash-decoding attend for the whole-token graph: the split-K part kernel
//! with every lane at work.
//!
//! `gqa_attend_gpart` gives one 256-lane workgroup a 256-position chunk of a
//! KV group: lanes are positions for the scores (each lane walks its own K
//! row, uncoalesced), then 128 of them walk the chunk's 256 V rows one after
//! another, with two eight-head reduction trees in between. A model with two
//! KV heads gets 2 × ⌈n/256⌉ workgroups — 32 at 4 096 positions on a 170-SM
//! RTX 5090 — and every one is a long serial chain: CherryPit-1.2-2B decoded
//! at 110 tok/s from 300 positions on against 165 at 41.
//!
//! Here a workgroup is eight subgroups of 32 lanes and a chunk is
//! `8 · P` positions (`P` from the host, so that the card holds about 128
//! workgroups). A subgroup takes P consecutive positions; for each it reads
//! the K and V rows WHOLE and coalesced (lane `l` holds the vec4 `l` of the
//! row, and `l + 32` for 256-wide heads), gets every query head's score with
//! one `subgroupAdd`, and keeps an online softmax per head in registers — no
//! barrier in the loop. The eight subgroups' (m, l, acc) meet in workgroup
//! memory one head at a time, and the chunk's partial lands in the layout
//! `gqa_attend_merge` already reads (`[h·nc + ch]`, `ml = (max, sum)`), so the
//! merge is unchanged. Same math as the other split kernels up to the order
//! of the sums.
//!
//! The source is generated per (heads per KV group, head width): the head
//! loop is unrolled into named registers (indexed register arrays spill).
//! Needs 32-lane subgroups; `CMF_ATTEND_FD=0` keeps the old kernels.

use super::*;

/// The pipeline and its bind-group layout for one (hpk, hd).
pub(crate) type FdPipe = std::sync::Arc<(wgpu::ComputePipeline, wgpu::BindGroupLayout)>;

/// Subgroups per workgroup.
pub(crate) const FD_WARPS: usize = 8;

fn sg32(c: &Ctx) -> bool {
    c.device.features().contains(wgpu::Features::SUBGROUP)
        && c.adapter_info.subgroup_min_size == 32
        && c.adapter_info.subgroup_max_size == 32
}

pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_ATTEND_FD").as_deref() != Ok("0"))
}

/// Two cache rows a loop step (`CMF_ATTEND_FD_PAIR=1`; default one).
fn pair() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_ATTEND_FD_PAIR").as_deref() == Ok("1"))
}

/// Cache rows a loop step of the blocked loop (`CMF_ATTEND_FD_BLK`: 2 or
/// 4; default 1 = one row a step): the block's K and V rows all load
/// before its first score, and the running max moves once a block (B + 1
/// exponentials a head instead of 2·B). Measured no change (Qwen3.5-4B on
/// an RTX A5000 at 32 000 positions: the part streams the cache at ~650
/// GB/s either way), so the row loop stays the default.
fn blk() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| match std::env::var("CMF_ATTEND_FD_BLK").as_deref() {
        Ok("2") => 2,
        Ok("4") => 4,
        _ => 1,
    })
}

/// Positions from which the decode graph takes this kernel instead of the
/// one-workgroup-per-head attend (`CMF_ATTEND_FD_MIN`, default 16: at 41
/// positions it already decoded 196 tok/s against the old kernel's 163–188).
pub(crate) fn min_positions() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("CMF_ATTEND_FD_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16)
    })
}

/// Positions per subgroup for `n` cached positions over `nkv` KV groups: the
/// grid holds about 64 workgroups (`CMF_ATTEND_FD_WGS`) until a subgroup has
/// 32 positions, then more chunks. RTX 5090, CherryPit-1.2-2B q4tp, tok/s at
/// 512 / 4 096 / 16 384 positions: 64 workgroups 189 / 150 / 92, 128 168 /
/// 141 / 103, 256 140 / 116 / 85 — the merge walks every chunk, a long
/// subgroup chain costs at depth.
pub(crate) fn per_warp(n: usize, nkv: usize) -> usize {
    static WGS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let wgs = *WGS.get_or_init(|| {
        std::env::var("CMF_ATTEND_FD_WGS").ok().and_then(|v| v.parse().ok()).filter(|&v| v > 0).unwrap_or(64)
    });
    let chunks = wgs.div_ceil(nkv.max(1)).max(1);
    n.div_ceil(chunks * FD_WARPS).clamp(1, 32)
}

fn src(hpk: usize, hd: usize, half: bool) -> String {
    let hd4 = hd / 4;
    let slots = hd4.div_ceil(32);
    let guard = hd4 % 32 != 0;
    // No `enable subgroups;`: naga takes the builtins through the device
    // capability and rejects the directive itself.
    // An f16 mirror (`kv16`): K and V rows load as f16, every sum in f32.
    let kvt = if half { "vec4<f16>" } else { "vec4<f32>" };
    let mut s = String::new();
    if half {
        s.push_str("enable f16;\n");
    }
    s.push_str(&format!(
        "struct ApP {{ nh: u32, hpk: u32, hd: u32, cap: u32, n: u32, ck: u32, nc: u32, scale: f32 }};\n\
         @group(0) @binding(0) var<storage, read>       ap_q  : array<vec4<f32>>;\n\
         @group(0) @binding(1) var<storage, read>       ap_k  : array<{kvt}>;\n\
         @group(0) @binding(2) var<storage, read>       ap_v  : array<{kvt}>;\n\
         @group(0) @binding(3) var<storage, read_write> ap_acc: array<f32>;\n\
         @group(0) @binding(4) var<storage, read_write> ap_ml : array<vec2<f32>>;\n\
         @group(0) @binding(5) var<uniform>             ap_p  : ApP;\n"
    ));
    s.push_str(&format!(
        "var<workgroup> fd_m: array<f32, {}>;\nvar<workgroup> fd_l: array<f32, {}>;\nvar<workgroup> fd_acc: array<f32, {}>;\n",
        FD_WARPS * hpk,
        FD_WARPS * hpk,
        FD_WARPS * hd
    ));
    s.push_str(
        "@compute @workgroup_size(256)\n\
         fn gqa_attend_fd(@builtin(workgroup_id) wid: vec3<u32>,\n\
                          @builtin(local_invocation_index) lid: u32,\n\
                          @builtin(subgroup_invocation_id) lane: u32) {\n\
             let g = wid.x;\n\
             let ch = wid.y;\n\
             let warp = lid >> 5u;\n",
    );
    s.push_str(&format!("    let hd4 = {hd4}u;\n    let hpk = {hpk}u;\n"));
    s.push_str(
        "    let kbase = g * ap_p.cap * hd4;\n\
             let per = ap_p.ck / 8u;\n\
             let p0 = ch * ap_p.ck + warp * per;\n\
             let pend = min(ap_p.n, min((ch + 1u) * ap_p.ck, p0 + per));\n\
             let scale = ap_p.scale;\n",
    );
    // The lane's slots of the head width, and its query slices.
    for j in 0..slots {
        if guard {
            s.push_str(&format!("    let ok{j} = lane + {}u < hd4;\n", 32 * j));
        }
    }
    for h in 0..hpk {
        for j in 0..slots {
            let idx = format!("(g * hpk + {h}u) * hd4 + lane + {}u", 32 * j);
            if guard {
                s.push_str(&format!(
                    "    var q{h}_{j} = vec4<f32>(0.0);\n    if (ok{j}) {{ q{h}_{j} = ap_q[{idx}]; }}\n"
                ));
            } else {
                s.push_str(&format!("    let q{h}_{j} = ap_q[{idx}];\n"));
            }
            s.push_str(&format!("    var a{h}_{j} = vec4<f32>(0.0);\n"));
        }
        s.push_str(&format!("    var m{h} = -1.0e30;\n    var l{h} = 0.0;\n"));
    }
    // `CMF_ATTEND_FD_PAIR=1`: rows two at a time, both rows' loads in
    // flight before either score is reduced. Measured no change (Qwen3.5-4B
    // on an RTX A5000 at 32 000 positions, 75.2 against 75.1 tok/s: the
    // merge was the wall, see `gqa_attend_merge_w`), so one row a step stays
    // the default.
    let load = |sfx: &str, rowv: &str, s: &mut String| {
        for j in 0..slots {
            if guard {
                s.push_str(&format!(
                    "        var k{sfx}{j} = vec4<f32>(0.0);\n        var v{sfx}{j} = vec4<f32>(0.0);\n        if (ok{j}) {{ k{sfx}{j} = vec4<f32>(ap_k[{rowv} + lane + {o}u]); v{sfx}{j} = vec4<f32>(ap_v[{rowv} + lane + {o}u]); }}\n",
                    o = 32 * j
                ));
            } else {
                s.push_str(&format!(
                    "        let k{sfx}{j} = vec4<f32>(ap_k[{rowv} + lane + {o}u]);\n        let v{sfx}{j} = vec4<f32>(ap_v[{rowv} + lane + {o}u]);\n",
                    o = 32 * j
                ));
            }
        }
    };
    let one = |s: &mut String| {
        s.push_str("        let row = kbase + p * hd4;\n");
        load("", "row", s);
        for h in 0..hpk {
            let dots: Vec<String> = (0..slots).map(|j| format!("dot(q{h}_{j}, k{j})")).collect();
            s.push_str(&format!(
                "        let s{h} = subgroupAdd({}) * scale;\n        let n{h} = max(m{h}, s{h});\n        let f{h} = exp(m{h} - n{h});\n        let w{h} = exp(s{h} - n{h});\n        l{h} = l{h} * f{h} + w{h};\n        m{h} = n{h};\n",
                dots.join(" + ")
            ));
            for j in 0..slots {
                s.push_str(&format!("        a{h}_{j} = a{h}_{j} * f{h} + v{j} * w{h};\n"));
            }
        }
    };
    let nb = blk();
    if !pair() && nb > 1 {
        s.push_str(&format!("    var p = p0;\n    loop {{\n        if (p + {}u >= pend) {{ break; }}\n", nb - 1));
        for b in 0..nb {
            s.push_str(&format!("        let r{b} = kbase + (p + {b}u) * hd4;\n"));
        }
        for b in 0..nb {
            load(&format!("b{b}_"), &format!("r{b}"), &mut s);
        }
        for h in 0..hpk {
            for b in 0..nb {
                let d: Vec<String> = (0..slots).map(|j| format!("dot(q{h}_{j}, kb{b}_{j})")).collect();
                s.push_str(&format!("        let s{h}_{b} = subgroupAdd({}) * scale;\n", d.join(" + ")));
            }
            let mx = (0..nb).fold(format!("m{h}"), |acc, b| format!("max({acc}, s{h}_{b})"));
            s.push_str(&format!("        let n{h} = {mx};\n        let f{h} = exp(m{h} - n{h});\n"));
            for b in 0..nb {
                s.push_str(&format!("        let w{h}_{b} = exp(s{h}_{b} - n{h});\n"));
            }
            let ws: Vec<String> = (0..nb).map(|b| format!("w{h}_{b}")).collect();
            s.push_str(&format!("        l{h} = l{h} * f{h} + ({});\n        m{h} = n{h};\n", ws.join(" + ")));
            for j in 0..slots {
                let vs: Vec<String> = (0..nb).map(|b| format!("vb{b}_{j} * w{h}_{b}")).collect();
                s.push_str(&format!("        a{h}_{j} = a{h}_{j} * f{h} + {};\n", vs.join(" + ")));
            }
        }
        s.push_str(&format!("        p = p + {nb}u;\n    }}\n    for (; p < pend; p = p + 1u) {{\n"));
        one(&mut s);
        s.push_str("    }\n");
    } else if pair() {
        s.push_str("    var p = p0;\n    loop {\n        if (p + 1u >= pend) { break; }\n        let ra = kbase + p * hd4;\n        let rb = ra + hd4;\n");
        load("a", "ra", &mut s);
        load("b", "rb", &mut s);
        for h in 0..hpk {
            let da: Vec<String> = (0..slots).map(|j| format!("dot(q{h}_{j}, ka{j})")).collect();
            let db: Vec<String> = (0..slots).map(|j| format!("dot(q{h}_{j}, kb{j})")).collect();
            s.push_str(&format!(
                "        let sa{h} = subgroupAdd({}) * scale;\n        let sb{h} = subgroupAdd({}) * scale;\n        let n{h} = max(m{h}, max(sa{h}, sb{h}));\n        let f{h} = exp(m{h} - n{h});\n        let wa{h} = exp(sa{h} - n{h});\n        let wb{h} = exp(sb{h} - n{h});\n        l{h} = l{h} * f{h} + wa{h} + wb{h};\n        m{h} = n{h};\n",
                da.join(" + "),
                db.join(" + ")
            ));
            for j in 0..slots {
                s.push_str(&format!("        a{h}_{j} = a{h}_{j} * f{h} + va{j} * wa{h} + vb{j} * wb{h};\n"));
            }
        }
        s.push_str("        p = p + 2u;\n    }\n    if (p < pend) {\n");
        one(&mut s);
        s.push_str("    }\n");
    } else {
        s.push_str("    for (var p = p0; p < pend; p = p + 1u) {\n");
        one(&mut s);
        s.push_str("    }\n");
    }
    // The eight subgroups meet per head; the chunk's partial goes out.
    for h in 0..hpk {
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
                     for (var w = 0u; w < {FD_WARPS}u; w = w + 1u) {{ mm = max(mm, fd_m[w * hpk + {h}u]); }}\n\
                     var ll = 0.0;\n\
                     var aa = 0.0;\n\
                     for (var w = 0u; w < {FD_WARPS}u; w = w + 1u) {{\n\
                         let e = exp(fd_m[w * hpk + {h}u] - mm);\n\
                         ll = ll + fd_l[w * hpk + {h}u] * e;\n\
                         aa = aa + fd_acc[w * {hd}u + lid] * e;\n\
                     }}\n\
                     let idx = (g * hpk + {h}u) * ap_p.nc + ch;\n\
                     ap_acc[idx * {hd}u + lid] = aa;\n\
                     if (lid == 0u) {{ ap_ml[idx] = vec2<f32>(mm, ll); }}\n\
                 }}\n\
                 workgroupBarrier();\n"
        ));
    }
    s.push_str("}\n");
    s
}

/// The source, for the device-free tests of other modules.
#[cfg(test)]
pub(crate) fn src_for_test(hpk: usize, hd: usize, half: bool) -> String {
    src(hpk, hd, half)
}

fn compile(c: &Ctx, hpk: usize, hd: usize, half: bool) -> Option<FdPipe> {
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-attend-fd"),
        source: wgpu::ShaderSource::Wgsl(src(hpk, hd, half).into()),
    });
    let p = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("gqa_attend_fd"),
        layout: None,
        module: &module,
        entry_point: Some("gqa_attend_fd"),
        compilation_options: crate::gpu_wgpu::pco(),
        cache: c.pipeline_cache.as_ref(),
    });
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("cmf-attend-fd rejected ({e}): the decode graph keeps the split kernels");
        return None;
    }
    let l = p.get_bind_group_layout(0);
    Some(std::sync::Arc::new((p, l)))
}

/// The kernel for a layer of `hpk` query heads per KV head of width `hd`;
/// None where it does not apply (no 32-lane subgroups, hd not 64/128/256,
/// more than 8 heads a group, or `CMF_ATTEND_FD=0`).
/// `half`: the twin over an f16 mirror (`kv16`).
pub(crate) fn pipe(c: &Ctx, hpk: usize, hd: usize, half: bool) -> Option<FdPipe> {
    if !enabled() || hpk == 0 || hpk > 8 || !matches!(hd, 64 | 128 | 256) || !sg32(c) {
        return None;
    }
    let mut map = c.attend_fd_pipes.lock().ok()?;
    map.entry((hpk, hd, half)).or_insert_with(|| compile(c, hpk, hd, half)).clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_warp_spreads_the_grid() {
        // Two KV groups: 32 chunks a group, 8 subgroups a chunk, at most 32
        // positions a subgroup.
        assert_eq!(per_warp(300, 2), 2);
        assert_eq!(per_warp(4096, 2), 16);
        assert_eq!(per_warp(16_384, 2), 32);
        assert_eq!(per_warp(131_072, 2), 32);
        assert_eq!(per_warp(1, 8), 1);
    }

    /// Every shape the decode graph may ask for parses and validates (naga,
    /// subgroup capability on), with no GPU.
    #[test]
    fn sources_validate() {
        for (hpk, hd, half) in (1..=8).flat_map(|h| [64, 128, 256].into_iter().flat_map(move |d| [(h, d, false), (h, d, true)])) {
            {
                let s = src(hpk, hd, half);
                let module = wgpu::naga::front::wgsl::parse_str(&s)
                    .unwrap_or_else(|e| panic!("hpk {hpk} hd {hd}: {e}"));
                wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap_or_else(|e| panic!("hpk {hpk} hd {hd}: {e:?}"));
            }
        }
    }

    #[test]
    fn source_names_every_head() {
        let s = src(8, 128, false);
        assert!(s.contains("let s7 = subgroupAdd(dot(q7_0, k0)) * scale;"));
        assert!(src(4, 256, true).contains("ap_k  : array<vec4<f16>>"));
        assert!(!s.contains("ok0"));
        let s = src(2, 64, false);
        assert!(s.contains("if (ok0)"));
        let s = src(4, 256, false);
        assert!(s.contains("dot(q3_0, k0) + dot(q3_1, k1)"));
    }
}
