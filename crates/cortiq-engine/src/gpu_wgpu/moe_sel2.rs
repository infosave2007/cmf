//! The MoE select of the wgpu graphs for 257–512 experts (GLM-5.3: 288):
//! `moe_select` / `moe_select_b`'s arithmetic with two experts a thread of
//! the 256-wide workgroup — each thread folds its pair before the same
//! reduction trees, ties to the lower expert. Same bindings and uniform as
//! the kernels it stands in for; its own pipelines and layouts.

use super::*;
use std::sync::OnceLock;

const TOKEN_SRC: &str = r#"
struct MoeSelP { n_exp: u32, top_k: u32, norm: u32, pk: u32, scale: f32, _s0: u32, _s1: u32, _s2: u32 };
@group(0) @binding(0) var<storage, read>       ms_logit : array<f32>;
@group(0) @binding(1) var<storage, read>       ms_slog  : array<f32>;
@group(0) @binding(2) var<storage, read_write> ms_sel   : array<u32>;
@group(0) @binding(3) var<storage, read_write> ms_w     : array<f32>;
@group(0) @binding(4) var<uniform>             ms_p     : MoeSelP;
@group(0) @binding(5) var<storage, read>       ms_sgw   : array<u32>;
@group(0) @binding(6) var<storage, read>       ms_x     : array<f32>;
@group(0) @binding(7) var<storage, read>       ms_bias  : array<f32>;
var<workgroup> lg2: array<f32, 512>;
var<workgroup> sc2: array<f32, 512>;
var<workgroup> red: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;
var<workgroup> pick: u32;
var<workgroup> sgv: f32;

@compute @workgroup_size(256)
fn moe_select2(@builtin(local_invocation_index) lid: u32) {
    let sg_kind = ms_p.pk & 0xFFu;
    let sg_hidden = ms_p.pk >> 8u;
    if (sg_kind == 4u) {
        var d = 0.0;
        for (var i = lid; i < sg_hidden; i = i + 256u) { d = d + bitcast<f32>(ms_sgw[i]) * ms_x[i]; }
        red[lid] = d;
        workgroupBarrier();
        for (var st = 128u; st > 0u; st = st >> 1u) {
            if (lid < st) { red[lid] = red[lid] + red[lid + st]; }
            workgroupBarrier();
        }
        if (lid == 0u) { sgv = red[0]; }
    } else {
        if (lid == 0u) { sgv = ms_slog[0]; }
    }
    workgroupBarrier();
    let n = ms_p.n_exp;
    let i0 = lid;
    let i1 = lid + 256u;
    var v0 = -3.0e38;
    var v1 = -3.0e38;
    if (i0 < n) { v0 = ms_logit[i0]; }
    if (i1 < n) { v1 = ms_logit[i1]; }
    lg2[i0] = v0;
    lg2[i1] = v1;
    red[lid] = max(v0, v1);
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (lid < st) { red[lid] = max(red[lid], red[lid + st]); }
        workgroupBarrier();
    }
    let mx = red[0];
    workgroupBarrier();
    red[lid] = select(0.0, exp(v0 - mx), i0 < n) + select(0.0, exp(v1 - mx), i1 < n);
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (lid < st) { red[lid] = red[lid] + red[lid + st]; }
        workgroupBarrier();
    }
    let denom = red[0];
    workgroupBarrier();
    let mflags = ms_p.norm;
    let msig = (mflags & 2u) != 0u;
    var s0 = 0.0;
    var s1 = 0.0;
    if (i0 < n) { s0 = select(exp(v0 - mx) / denom, 1.0 / (1.0 + exp(-v0)), msig); }
    if (i1 < n) { s1 = select(exp(v1 - mx) / denom, 1.0 / (1.0 + exp(-v1)), msig); }
    sc2[i0] = s0;
    sc2[i1] = s1;
    if (msig) {
        var k0 = s0;
        var k1 = s1;
        if ((mflags & 4u) != 0u) {
            if (i0 < n) { k0 = k0 + ms_bias[i0]; }
            if (i1 < n) { k1 = k1 + ms_bias[i1]; }
        }
        lg2[i0] = select(-3.0e38, k0, i0 < n);
        lg2[i1] = select(-3.0e38, k1, i1 < n);
    }
    workgroupBarrier();
    let k = ms_p.top_k;
    var wsum = 0.0;
    for (var slot = 0u; slot < k; slot = slot + 1u) {
        let a = lg2[i0];
        let b = lg2[i1];
        var bv = a;
        var bi = i0;
        if (b > a) { bv = b; bi = i1; }
        red[lid] = bv;
        ri[lid] = bi;
        workgroupBarrier();
        for (var st = 128u; st > 0u; st = st >> 1u) {
            if (lid < st) {
                let x = red[lid];
                let y = red[lid + st];
                let ix = ri[lid];
                let iy = ri[lid + st];
                if (y > x || (y == x && iy < ix)) {
                    red[lid] = y;
                    ri[lid] = iy;
                }
            }
            workgroupBarrier();
        }
        if (lid == 0u) {
            let bx = ri[0];
            ms_sel[slot] = bx;
            ms_w[slot] = sc2[bx];
            pick = bx;
        }
        workgroupBarrier();
        wsum = wsum + sc2[ri[0]];
        if (i0 == pick) { lg2[i0] = -3.0e38; }
        if (i1 == pick) { lg2[i1] = -3.0e38; }
        workgroupBarrier();
    }
    if (lid == 0u) {
        if ((mflags & 1u) != 0u) {
            let eps = select(1e-6, 1e-20, (mflags & 32u) != 0u);
            let dn = select(wsum, wsum + eps, msig);
            for (var slot = 0u; slot < k; slot = slot + 1u) { ms_w[slot] = ms_w[slot] / dn; }
        }
        for (var slot = 0u; slot < k; slot = slot + 1u) { ms_w[slot] = ms_w[slot] * ms_p.scale; }
        if ((mflags & 8u) != 0u) {
            ms_sel[k] = n;
            if ((mflags & 16u) != 0u) {
                ms_w[k] = 1.0;
            } else {
                ms_w[k] = 1.0 / (1.0 + exp(-sgv));
            }
        }
    }
}
"#;

const BATCH_SRC: &str = r#"
struct MoeSelBP { n_exp: u32, top_k: u32, norm: u32, pk: u32, scale: f32, _s0: u32, _s1: u32, _s2: u32 };
@group(0) @binding(0) var<storage, read>       sb_lgin: array<f32>;
@group(0) @binding(1) var<storage, read>       sb_x   : array<f32>;
@group(0) @binding(2) var<storage, read_write> sb_sel : array<u32>;
@group(0) @binding(3) var<storage, read_write> sb_w   : array<f32>;
@group(0) @binding(4) var<uniform>             sb_p   : MoeSelBP;
@group(0) @binding(5) var<storage, read>       sb_sgw : array<u32>;
@group(0) @binding(6) var<storage, read>       sb_bias: array<f32>;
var<workgroup> lg2: array<f32, 512>;
var<workgroup> sc2: array<f32, 512>;
var<workgroup> red: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;
var<workgroup> pick: u32;
var<workgroup> sgv: f32;

@compute @workgroup_size(256)
fn moe_select2_b(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    let t = wid.x;
    let n = sb_p.n_exp;
    let sg_kind = sb_p.pk & 0xFFu;
    let hidden = sb_p.pk >> 8u;
    let xb = t * hidden;
    let i0 = lid;
    let i1 = lid + 256u;
    var v0 = -3.0e38;
    var v1 = -3.0e38;
    if (i0 < n) { v0 = sb_lgin[t * n + i0]; }
    if (i1 < n) { v1 = sb_lgin[t * n + i1]; }
    if (sg_kind == 4u) {
        var d = 0.0;
        for (var i = lid; i < hidden; i = i + 256u) { d = d + bitcast<f32>(sb_sgw[i]) * sb_x[xb + i]; }
        red[lid] = d;
        workgroupBarrier();
        for (var st = 128u; st > 0u; st = st >> 1u) {
            if (lid < st) { red[lid] = red[lid] + red[lid + st]; }
            workgroupBarrier();
        }
        if (lid == 0u) { sgv = red[0]; }
        workgroupBarrier();
    }
    lg2[i0] = v0;
    lg2[i1] = v1;
    red[lid] = max(v0, v1);
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (lid < st) { red[lid] = max(red[lid], red[lid + st]); }
        workgroupBarrier();
    }
    let mx = red[0];
    workgroupBarrier();
    red[lid] = select(0.0, exp(v0 - mx), i0 < n) + select(0.0, exp(v1 - mx), i1 < n);
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (lid < st) { red[lid] = red[lid] + red[lid + st]; }
        workgroupBarrier();
    }
    let denom = red[0];
    workgroupBarrier();
    let mflags = sb_p.norm;
    let msig = (mflags & 2u) != 0u;
    var s0 = 0.0;
    var s1 = 0.0;
    if (i0 < n) { s0 = select(exp(v0 - mx) / denom, 1.0 / (1.0 + exp(-v0)), msig); }
    if (i1 < n) { s1 = select(exp(v1 - mx) / denom, 1.0 / (1.0 + exp(-v1)), msig); }
    sc2[i0] = s0;
    sc2[i1] = s1;
    if (msig) {
        var k0 = s0;
        var k1 = s1;
        if ((mflags & 4u) != 0u) {
            if (i0 < n) { k0 = k0 + sb_bias[i0]; }
            if (i1 < n) { k1 = k1 + sb_bias[i1]; }
        }
        lg2[i0] = select(-3.0e38, k0, i0 < n);
        lg2[i1] = select(-3.0e38, k1, i1 < n);
    }
    workgroupBarrier();
    let kk = sb_p.top_k;
    let has_sh = (mflags & 8u) != 0u;
    let ob = t * (kk + select(0u, 1u, has_sh));
    var wsum = 0.0;
    for (var slot = 0u; slot < kk; slot = slot + 1u) {
        let a = lg2[i0];
        let b = lg2[i1];
        var bv = a;
        var bi = i0;
        if (b > a) { bv = b; bi = i1; }
        red[lid] = bv;
        ri[lid] = bi;
        workgroupBarrier();
        for (var st = 128u; st > 0u; st = st >> 1u) {
            if (lid < st) {
                let x = red[lid];
                let y = red[lid + st];
                let ix = ri[lid];
                let iy = ri[lid + st];
                if (y > x || (y == x && iy < ix)) {
                    red[lid] = y;
                    ri[lid] = iy;
                }
            }
            workgroupBarrier();
        }
        if (lid == 0u) {
            let bx = ri[0];
            sb_sel[ob + slot] = bx;
            sb_w[ob + slot] = sc2[bx];
            pick = bx;
        }
        workgroupBarrier();
        wsum = wsum + sc2[ri[0]];
        if (i0 == pick) { lg2[i0] = -3.0e38; }
        if (i1 == pick) { lg2[i1] = -3.0e38; }
        workgroupBarrier();
    }
    if (lid == 0u) {
        if ((mflags & 1u) != 0u) {
            let eps = select(1e-6, 1e-20, (mflags & 32u) != 0u);
            let dn = select(wsum, wsum + eps, msig);
            for (var slot = 0u; slot < kk; slot = slot + 1u) { sb_w[ob + slot] = sb_w[ob + slot] / dn; }
        }
        for (var slot = 0u; slot < kk; slot = slot + 1u) { sb_w[ob + slot] = sb_w[ob + slot] * sb_p.scale; }
        if (has_sh) {
            sb_sel[ob + kk] = n;
            if ((mflags & 16u) != 0u) {
                sb_w[ob + kk] = 1.0;
            } else {
                sb_w[ob + kk] = 1.0 / (1.0 + exp(-sgv));
            }
        }
    }
}
"#;

pub(crate) struct Pipes {
    pub token: wgpu::ComputePipeline,
    pub token_l: wgpu::BindGroupLayout,
    pub batch: wgpu::ComputePipeline,
    pub batch_l: wgpu::BindGroupLayout,
}

static PIPES: OnceLock<Option<Pipes>> = OnceLock::new();

/// The two kernels, compiled on first use (None: the device refused them).
pub(crate) fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    PIPES
        .get_or_init(|| {
            let sc = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let mk = |src: &str, entry: &str| {
                let m = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(entry),
                    source: wgpu::ShaderSource::Wgsl(src.into()),
                });
                c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None,
                    module: &m,
                    entry_point: Some(entry),
                    compilation_options: crate::gpu_wgpu::pco(),
                    cache: None,
                })
            };
            let token = mk(TOKEN_SRC, "moe_select2");
            let batch = mk(BATCH_SRC, "moe_select2_b");
            if let Some(e) = pollster::block_on(sc.pop()) {
                tracing::warn!("512-expert MoE select rejected: {e}");
                return None;
            }
            let (token_l, batch_l) = (token.get_bind_group_layout(0), batch.get_bind_group_layout(0));
            Some(Pipes { token, token_l, batch, batch_l })
        })
        .as_ref()
}

#[cfg(test)]
mod tests {
    #[test]
    fn select2_wgsl_validates() {
        for src in [super::TOKEN_SRC, super::BATCH_SRC] {
            let module = wgpu::naga::front::wgsl::parse_str(src).expect("select WGSL must parse");
            wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), wgpu::naga::valid::Capabilities::all())
                .validate(&module)
                .expect("select WGSL must validate");
        }
    }
}
