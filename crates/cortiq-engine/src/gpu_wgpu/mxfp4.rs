//! `mxfp4` (OCP MX FP4, dtype 21) on the wgpu backend.
//!
//! - **Per-op kernels** (`mxfp4_mv`, `mxfp4_mm`): one 64-lane workgroup per
//!   output row; a lane walks 32-element groups (`g = lid, lid + 64, …`),
//!   decodes the group's four code words with the exact E2M1 bit
//!   construction (`0x3F000000 + mag << 22` for magnitudes ≥ 2, 0 and 0.5
//!   patched), dots them with the activation, multiplies by the exact E8M0
//!   scale `2^(k−127)` (built from the exponent bits) and the workgroup
//!   tree-reduces. `mxfp4_mm` carries up to eight tokens per dispatch (one
//!   decode of the group serves all of them); `b = 1` is the decode matvec.
//! - **Global MoE bank kernels** (`dsv4_global_{gate_up,down}_mxfp4`): the
//!   descriptor-indexed segmented expert bank of `dsv4_moe_frame` (the
//!   DeepSeek-V4.1 dynamic pool), with the q4tp twins' bindings, route and
//!   BF16 semantics, over mxfp4 slots. A slot holds the tensor bytes
//!   verbatim, so `dsv4_global_slot_fill` is unchanged; its byte length must
//!   be a multiple of four (word-aligned code plane).
//!
//! Both shader modules are built lazily, on the first mxfp4 tensor, in
//! their own modules: a driver that rejects them leaves every other codec's
//! pipelines untouched and mxfp4 on the CPU kernels.

use super::*;

/// Layout of one expert in the segmented global bank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeBankCodec {
    /// gate/up/down q4tp.
    Q4tp,
    /// gate/up q2tp, down q4tp (the 2-bit MoE profile).
    Q2tpGateUp,
    /// gate/up/down mxfp4 (`convert --quant fp4`).
    Mxfp4,
}

impl MoeBankCodec {
    pub fn from_gu_q2(gu_q2: bool) -> Self {
        if gu_q2 { Self::Q2tpGateUp } else { Self::Q4tp }
    }

    /// The historical `gu_q2` flag of the q4tp/q2tp banks (false for mxfp4,
    /// whose pipelines are selected by the bank, not by the frame geometry).
    pub fn gu_q2(self) -> bool {
        self == Self::Q2tpGateUp
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Q4tp => "q4tp",
            Self::Q2tpGateUp => "q2tp",
            Self::Mxfp4 => "mxfp4",
        }
    }

    /// `(gate/up bytes, down bytes)` of one expert, `[inter, hidden]` and
    /// `[hidden, inter]`. None for a shape the codec cannot hold — for
    /// mxfp4 also a length that is not word-aligned.
    pub fn lens(self, inter: usize, hidden: usize) -> Option<(usize, usize)> {
        use cortiq_core::TensorDtype as D;
        let (gu, dn) = match self {
            Self::Q4tp => (D::Q4TiledP, D::Q4TiledP),
            Self::Q2tpGateUp => (D::Q2TiledP, D::Q4TiledP),
            Self::Mxfp4 => (D::Mxfp4, D::Mxfp4),
        };
        let g = cortiq_core::quant::expected_nbytes(gu, &[inter, hidden])?;
        let d = cortiq_core::quant::expected_nbytes(dn, &[hidden, inter])?;
        if self == Self::Mxfp4 && (g % 4 != 0 || d % 4 != 0) {
            return None;
        }
        Some((g, d))
    }
}

/// E2M1/E8M0 decode shared by both shader modules.
const DECODE: &str = r#"
fn mx_e2m1(n: u32) -> f32 {
    let m = n & 7u;
    var v = bitcast<f32>(0x3F000000u + (m << 22u));
    if (m == 0u) { v = 0.0; }
    if (m == 1u) { v = 0.5; }
    return select(v, -v, (n & 8u) != 0u);
}

fn mx_e8m0(k: u32) -> f32 {
    if (k == 255u) { return bitcast<f32>(0x7FC00000u); }
    if (k == 0u) { return bitcast<f32>(0x00400000u); }
    return bitcast<f32>(k << 23u);
}
"#;

const PEROP: &str = r#"
// rows; gpr = cols/32; sbase = WORD offset of the scale plane (rows·cols/8);
// nb = tokens in the batch; cols.
struct MxP { rows: u32, gpr: u32, sbase: u32, nb: u32, cols: u32, _a: u32, _b: u32, _c: u32 };
@group(0) @binding(0) var<storage, read>       mw : array<u32>;
@group(0) @binding(1) var<storage, read>       mx : array<f32>;
@group(0) @binding(2) var<storage, read_write> my : array<f32>;
@group(0) @binding(3) var<uniform>             mp : MxP;
var<workgroup> mpart: array<f32, 512>;

fn mx_scale(row: u32, g: u32) -> f32 {
    let i = row * mp.gpr + g;
    return mx_e8m0((mw[mp.sbase + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xFFu);
}

@compute @workgroup_size(64)
fn mxfp4_mv(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(num_workgroups) nwg: vec3<u32>,
            @builtin(local_invocation_index) lid: u32) {
    let row = wid.x + wid.y * nwg.x;
    if (row >= mp.rows) { return; }
    var acc = 0.0;
    for (var g = lid; g < mp.gpr; g = g + 64u) {
        let base = (row * mp.gpr + g) * 4u;
        let xb = g * 32u;
        var d = 0.0;
        for (var k = 0u; k < 4u; k = k + 1u) {
            let word = mw[base + k];
            for (var j = 0u; j < 8u; j = j + 1u) {
                d = d + mx_e2m1((word >> (j * 4u)) & 15u) * mx[xb + k * 8u + j];
            }
        }
        acc = acc + mx_scale(row, g) * d;
    }
    mpart[lid] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) { mpart[lid] = mpart[lid] + mpart[lid + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid == 0u) { my[row] = mpart[0]; }
}

// Up to eight tokens per workgroup (z = token chunk): the group's 32 weights
// are decoded once into registers and spent on every token of the chunk.
@compute @workgroup_size(64)
fn mxfp4_mm(@builtin(workgroup_id) wid: vec3<u32>,
            @builtin(num_workgroups) nwg: vec3<u32>,
            @builtin(local_invocation_index) lid: u32) {
    let row = wid.x + wid.y * nwg.x;
    if (row >= mp.rows) { return; }
    let t0 = wid.z * 8u;
    let nt = min(8u, mp.nb - t0);
    var acc: array<f32, 8>;
    for (var t = 0u; t < 8u; t = t + 1u) { acc[t] = 0.0; }
    var wv: array<f32, 32>;
    for (var g = lid; g < mp.gpr; g = g + 64u) {
        let base = (row * mp.gpr + g) * 4u;
        for (var k = 0u; k < 4u; k = k + 1u) {
            let word = mw[base + k];
            for (var j = 0u; j < 8u; j = j + 1u) {
                wv[k * 8u + j] = mx_e2m1((word >> (j * 4u)) & 15u);
            }
        }
        let s = mx_scale(row, g);
        for (var t = 0u; t < nt; t = t + 1u) {
            let xb = (t0 + t) * mp.cols + g * 32u;
            var d = 0.0;
            for (var j = 0u; j < 32u; j = j + 1u) { d = d + wv[j] * mx[xb + j]; }
            acc[t] = acc[t] + s * d;
        }
    }
    for (var t = 0u; t < 8u; t = t + 1u) { mpart[t * 64u + lid] = acc[t]; }
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) {
            for (var t = 0u; t < 8u; t = t + 1u) {
                mpart[t * 64u + lid] = mpart[t * 64u + lid] + mpart[t * 64u + lid + stride];
            }
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid == 0u) {
        for (var t = 0u; t < nt; t = t + 1u) { my[(t0 + t) * mp.rows + row] = mpart[t * 64u]; }
    }
}
"#;

/// The global-bank kernels. Bindings, uniforms and the gate/up tail are the
/// q4tp twins' (`DSV4_GLOBAL_MOE_SRC`), so `dsv4_moe_frame` binds them with
/// the same groups; only the weight walk differs. `mat16` is the slot
/// stride in u16 units (a slot's word base is `local · mat16/2`).
const GLOBAL: &str = r#"
enable wgpu_binding_array;

struct WordBank { words: array<u32> };
struct GGuP {
    gpr: u32, inter: u32, slots: u32, mat16: u32,
    lim: f32, segment_slots: u32, _p0: u32, _p1: u32,
};
@group(0) @binding(0) var<storage, read>       gg_gw  : binding_array<WordBank, 8>;
@group(0) @binding(1) var<storage, read>       gg_uw  : binding_array<WordBank, 8>;
@group(0) @binding(2) var<storage, read>       gg_x   : array<f32>;
@group(0) @binding(3) var<storage, read>       gg_sel : array<u32>;
@group(0) @binding(4) var<storage, read_write> gg_act : array<f32>;
@group(0) @binding(5) var<storage, read>       gg_wt  : array<f32>;
@group(1) @binding(0) var<uniform>             gg_p   : GGuP;
var<workgroup> gg_pg: array<f32, 64>;
var<workgroup> gg_pu: array<f32, 64>;

fn mxg_bf16(v: f32) -> f32 {
    let bits = bitcast<u32>(v);
    if ((bits & 0x7F800000u) == 0x7F800000u) { return v; }
    let rounded = bits + 0x7FFFu + ((bits >> 16u) & 1u);
    return bitcast<f32>(rounded & 0xFFFF0000u);
}

fn gg_g8(seg: u32, o: u32) -> u32 {
    return (gg_gw[seg].words[o >> 2u] >> ((o & 3u) * 8u)) & 0xFFu;
}
fn gg_u8(seg: u32, o: u32) -> u32 {
    return (gg_uw[seg].words[o >> 2u] >> ((o & 3u) * 8u)) & 0xFFu;
}

@compute @workgroup_size(64)
fn dsv4_global_gate_up_mxfp4(@builtin(workgroup_id) wid: vec3<u32>,
                             @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    let slot = wid.y;
    let batch = wid.z;
    let bslot = batch * gg_p.slots + slot;
    let flat = gg_sel[bslot];
    let seg = flat / gg_p.segment_slots;
    let local = flat - seg * gg_p.segment_slots;
    let gpr = gg_p.gpr;
    let rows = gg_p.inter;
    let base_w = local * (gg_p.mat16 >> 1u);
    let code_w = base_w + row * gpr * 4u;
    let sc8 = base_w * 4u + rows * gpr * 16u + row * gpr;
    var ag = 0.0;
    var au = 0.0;
    for (var g = lid; g < gpr; g = g + 64u) {
        let sg = mx_e8m0(gg_g8(seg, sc8 + g));
        let su = mx_e8m0(gg_u8(seg, sc8 + g));
        let xb = batch * gpr * 32u + g * 32u;
        var dg = 0.0;
        var du = 0.0;
        for (var k = 0u; k < 4u; k = k + 1u) {
            let wg = gg_gw[seg].words[code_w + g * 4u + k];
            let wu = gg_uw[seg].words[code_w + g * 4u + k];
            for (var j = 0u; j < 8u; j = j + 1u) {
                let xv = gg_x[xb + k * 8u + j];
                dg = dg + mx_e2m1((wg >> (j * 4u)) & 15u) * xv;
                du = du + mx_e2m1((wu >> (j * 4u)) & 15u) * xv;
            }
        }
        ag = ag + sg * dg;
        au = au + su * du;
    }
    gg_pg[lid] = ag;
    gg_pu[lid] = au;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) {
            gg_pg[lid] = gg_pg[lid] + gg_pg[lid + stride];
            gg_pu[lid] = gg_pu[lid] + gg_pu[lid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid == 0u) {
        var gate = gg_pg[0];
        var up = gg_pu[0];
        if (gg_p._p0 != 0u) {
            gate = mxg_bf16(gate);
            up = mxg_bf16(up);
        }
        if (gg_p.lim > 0.0) {
            up = clamp(up, -gg_p.lim, gg_p.lim);
            gate = min(gate, gg_p.lim);
        }
        var act = (gate / (1.0 + exp(-gate))) * up;
        if (gg_p._p0 != 0u) {
            act = mxg_bf16(gg_wt[bslot] * act);
        }
        gg_act[(bslot * gg_p.inter) + row] = act;
    }
}

struct GDnP {
    gpr: u32, hidden: u32, slots: u32, mat16: u32,
    segment_slots: u32, _p0: u32, _p1: u32, _p2: u32,
};
@group(0) @binding(0) var<storage, read>       gd_w   : binding_array<WordBank, 8>;
@group(0) @binding(1) var<storage, read>       gd_act : array<f32>;
@group(0) @binding(2) var<storage, read>       gd_sel : array<u32>;
@group(0) @binding(3) var<storage, read>       gd_wt  : array<f32>;
@group(0) @binding(4) var<storage, read_write> gd_y   : array<f32>;
@group(1) @binding(0) var<uniform>             gd_p   : GDnP;
var<workgroup> gd_pt: array<f32, 64>;

fn gd_8(seg: u32, o: u32) -> u32 {
    return (gd_w[seg].words[o >> 2u] >> ((o & 3u) * 8u)) & 0xFFu;
}

@compute @workgroup_size(64)
fn dsv4_global_down_mxfp4(@builtin(workgroup_id) wid: vec3<u32>,
                          @builtin(local_invocation_index) lid: u32) {
    let row = wid.x;
    let batch = wid.y;
    let gpr = gd_p.gpr;
    let rows = gd_p.hidden;
    let total = gd_p.slots * gpr;
    var acc = 0.0;
    for (var i = lid; i < total; i = i + 64u) {
        let slot = i / gpr;
        let g = i % gpr;
        let bslot = batch * gd_p.slots + slot;
        let flat = gd_sel[bslot];
        let seg = flat / gd_p.segment_slots;
        let local = flat - seg * gd_p.segment_slots;
        let base_w = local * (gd_p.mat16 >> 1u);
        let scale = mx_e8m0(gd_8(seg, base_w * 4u + rows * gpr * 16u + row * gpr + g));
        let code_w = base_w + (row * gpr + g) * 4u;
        let xb = (bslot * gpr + g) * 32u;
        var d = 0.0;
        for (var k = 0u; k < 4u; k = k + 1u) {
            let w = gd_w[seg].words[code_w + k];
            for (var j = 0u; j < 8u; j = j + 1u) {
                d = d + mx_e2m1((w >> (j * 4u)) & 15u) * gd_act[xb + k * 8u + j];
            }
        }
        // V4.1's gate/up pass already folded the route weight into the BF16
        // activation; the generic path keeps the f32 weighting here.
        if (gd_p._p0 != 0u) {
            acc = acc + scale * d;
        } else {
            acc = acc + gd_wt[bslot] * scale * d;
        }
    }
    gd_pt[lid] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) { gd_pt[lid] = gd_pt[lid] + gd_pt[lid + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (lid == 0u) {
        gd_y[batch * gd_p.hidden + row] = select(gd_pt[0], mxg_bf16(gd_pt[0]), gd_p._p0 != 0u);
    }
}
"#;

pub(crate) fn perop_source() -> String {
    format!("{DECODE}{PEROP}")
}

/// The global-bank source for one descriptor-array width (8 or 16).
pub(crate) fn global_source(segments: usize) -> Option<String> {
    let base = format!("{GLOBAL}{DECODE}");
    match segments {
        DSV4_GLOBAL_MOE_SEGMENTS => Some(base),
        DSV4_GLOBAL_MOE_SEGMENTS_S16 => {
            let src = base.replace("binding_array<WordBank, 8>", "binding_array<WordBank, 16>");
            (src.matches("binding_array<WordBank, 16>").count() == 3).then_some(src)
        }
        _ => None,
    }
}

pub(crate) struct Pipes {
    mv: wgpu::ComputePipeline,
    mm: wgpu::ComputePipeline,
}

pub(crate) struct GlobalPipes {
    pub(crate) gu: wgpu::ComputePipeline,
    pub(crate) dn: wgpu::ComputePipeline,
}

fn build_perop(c: &Ctx) -> Option<Pipes> {
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-mxfp4"),
        source: wgpu::ShaderSource::Wgsl(perop_source().into()),
    });
    let mk = |entry: &str| {
        trace_pipeline(entry);
        c.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: c.pipeline_cache.as_ref(),
            })
    };
    let (mv, mm) = (mk("mxfp4_mv"), mk("mxfp4_mm"));
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("cmf-mxfp4 module rejected ({e}): mxfp4 stays on the CPU kernels");
        return None;
    }
    Some(Pipes { mv, mm })
}

fn build_global(c: &Ctx, segments: usize) -> Option<GlobalPipes> {
    let feats = c.device.features();
    if !feats.contains(wgpu::Features::BUFFER_BINDING_ARRAY)
        || !feats.contains(wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY)
    {
        return None;
    }
    let source = global_source(segments)?;
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cmf-mxfp4-global"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let storage = |binding: u32, read_only: bool, count: Option<u32>| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: count.and_then(std::num::NonZeroU32::new),
    };
    let seg = Some(segments as u32);
    let gu0 = c
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mxfp4-global-gu0"),
            entries: &[
                storage(0, true, seg),
                storage(1, true, seg),
                storage(2, true, None),
                storage(3, true, None),
                storage(4, false, None),
                storage(5, true, None),
            ],
        });
    let dn0 = c
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mxfp4-global-dn0"),
            entries: &[
                storage(0, true, seg),
                storage(1, true, None),
                storage(2, true, None),
                storage(3, true, None),
                storage(4, false, None),
            ],
        });
    let params = c
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mxfp4-global-params"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
    let layout = |l0: &wgpu::BindGroupLayout, label: &str| {
        c.device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(l0), Some(&params)],
                immediate_size: 0,
            })
    };
    let (gl, dl) = (
        layout(&gu0, "mxfp4-global-gu"),
        layout(&dn0, "mxfp4-global-dn"),
    );
    let mk = |entry: &str, l: &wgpu::PipelineLayout| {
        trace_pipeline(entry);
        c.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(l),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: c.pipeline_cache.as_ref(),
            })
    };
    let gu = mk("dsv4_global_gate_up_mxfp4", &gl);
    let dn = mk("dsv4_global_down_mxfp4", &dl);
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("cmf-mxfp4-global module rejected ({e}): mxfp4 experts stay on the CPU");
        return None;
    }
    Some(GlobalPipes { gu, dn })
}

pub(crate) fn pipes(c: &Ctx) -> Option<&Pipes> {
    c.mxfp4_pipes.get_or_init(|| build_perop(c)).as_ref()
}

/// The mxfp4 global-bank pipelines for one descriptor-array width.
pub(crate) fn global_pipes(
    c: &Ctx,
    segments: usize,
) -> Option<(&wgpu::ComputePipeline, &wgpu::ComputePipeline)> {
    let cell = match segments {
        DSV4_GLOBAL_MOE_SEGMENTS => &c.mxfp4_global_s8,
        DSV4_GLOBAL_MOE_SEGMENTS_S16 => &c.mxfp4_global_s16,
        _ => return None,
    };
    cell.get_or_init(|| build_global(c, segments))
        .as_ref()
        .map(|p| (&p.gu, &p.dn))
}

/// `out[t·rows + r] = W[r,:] · xs[t·cols ..]` for `t < b` (b = 1: matvec),
/// on the resident weight. false = not taken (no context, wrong dtype or
/// shape, over the VRAM budget, pipeline unavailable): the CPU kernels run.
pub fn matmat(
    model: &Arc<CmfModel>,
    idx: usize,
    xs: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    let _op = c.op_gate.enter();
    if b == 0 || rows == 0 || cols % 32 != 0 || xs.len() < b * cols || out.len() < b * rows {
        return false;
    }
    let Some(e) = model.tensors.get(idx) else {
        return false;
    };
    if e.dtype != cortiq_core::TensorDtype::Mxfp4 || e.shape != [rows, cols] {
        return false;
    }
    let Some(need) = cortiq_core::mxfp4::nbytes(rows, cols) else {
        return false;
    };
    let payload = model.entry_bytes(e);
    if payload.len() < need {
        return false;
    }
    let Some(p) = pipes(c) else { return false };
    note_layer((model.uid() as usize, idx), &e.name);
    let Some(w) = weight_buffer_l(
        c,
        (model.uid() as usize, idx),
        &payload[..need],
        layer_of_name(&e.name),
    ) else {
        return false;
    };
    let max_wg = MAX_WG as usize;
    let wg_x = rows.min(max_wg) as u32;
    let wg_y = rows.div_ceil(max_wg) as u32;
    let chunks = b.div_ceil(8);
    if chunks > max_wg {
        return false;
    }
    let chunks = chunks as u32;
    let mut sc = c.scratch.lock().unwrap();
    let xb = Scratch::ensure(
        &c.device,
        &mut sc.xs,
        (b * cols * 4) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        "mxfp4-x",
    );
    c.queue
        .write_buffer(&xb, 0, bytemuck::cast_slice(&xs[..b * cols]));
    let y_size = (b * rows * 4) as u64;
    let yb = Scratch::ensure(
        &c.device,
        &mut sc.y,
        y_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        "mxfp4-y",
    );
    let stage = Scratch::ensure(
        &c.device,
        &mut sc.stage,
        y_size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        "mxfp4-stage",
    );
    let u = uniform_u32x8(
        c,
        [
            rows as u32,
            (cols / 32) as u32,
            (rows * cols / 8) as u32,
            b as u32,
            cols as u32,
            0,
            0,
            0,
        ],
    );
    let pipe = if b == 1 { &p.mv } else { &p.mm };
    let bind = c.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mxfp4"),
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            bind_buf(0, &w),
            bind_buf(1, &xb),
            bind_buf(2, &yb),
            bind_buf(3, &u),
        ],
    });
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mxfp4"),
        });
    {
        let mut pass = begin_pass(&mut enc);
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, if b == 1 { 1 } else { chunks });
    }
    let ok = readback(c, enc, &yb, &stage, y_size, &mut out[..b * rows]);
    drop(sc);
    ok
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(src: &str) {
        let module = wgpu::naga::front::wgsl::parse_str(src).expect("mxfp4 WGSL must parse");
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("mxfp4 WGSL must validate");
    }

    #[test]
    fn mxfp4_shaders_validate() {
        validate(&perop_source());
        validate(&global_source(8).unwrap());
        let s16 = global_source(16).unwrap();
        assert!(!s16.contains("binding_array<WordBank, 8>"));
        validate(&s16);
    }

    /// Waits for the shared device at the end of a test (the context is
    /// process-wide; a test must not leave work in flight for the next).
    struct Drain(wgpu::Device);

    impl Drop for Drain {
        fn drop(&mut self) {
            let _ = self.0.poll(wgpu::PollType::wait_indefinitely());
        }
    }

    /// Deterministic MXFP4 payload with moderate scales (2^-8 .. 2^-4). The
    /// scale changes from group to group (adjacent groups always differ:
    /// the step is 7 or 8 mod 5), so a kernel that reads another group's
    /// scale byte fails the comparison instead of passing on a constant
    /// plane.
    fn payload(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
        let codes: Vec<u8> = (0..rows * cols / 2)
            .map(|i| (i.wrapping_mul(2654435761).wrapping_add(seed * 977) >> 9) as u8)
            .collect();
        let scales: Vec<u8> = (0..rows * cols / 32)
            .map(|i| 119 + ((i * 7 + i / 3 + seed) % 5) as u8)
            .collect();
        cortiq_core::mxfp4::repack(&codes, &scales, rows, cols).unwrap()
    }

    fn act(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|j| ((j * 37 + seed * 101) % 211) as f32 / 211.0 - 0.5)
            .collect()
    }

    /// A CMF holding the given `[rows, cols]` mxfp4 tensors.
    fn model_with(
        tag: &str,
        tensors: &[(&str, usize, usize, Vec<u8>)],
    ) -> (Arc<CmfModel>, std::path::PathBuf) {
        use cortiq_core::{CmfHeader, TensorSpec};
        let header: CmfHeader = serde_json::from_value(serde_json::json!({
            "format": "cmf", "version": cortiq_core::CMF_VERSION,
            "arch": {"arch_name": "tiny-mxfp4", "hidden_size": 64,
                "intermediate_size": 64, "num_layers": 1, "num_attention_heads": 2,
                "num_kv_heads": 1, "head_dim": 32, "vocab_size": 64,
                "layer_types": ["FullAttention"], "rms_norm_eps": 1e-6,
                "max_position_embeddings": 8},
            "quant_type": "VBIT"
        }))
        .unwrap();
        let specs: Vec<TensorSpec> = tensors
            .iter()
            .map(|(n, r, c, d)| TensorSpec {
                name: (*n).into(),
                dtype: cortiq_core::TensorDtype::Mxfp4,
                shape: vec![*r, *c],
                data: d.clone(),
            })
            .collect();
        let dir = std::env::temp_dir().join(format!("cmf-mxfp4-wgpu-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.cmf");
        CmfModel::write(&path, &header, &specs, None, None).unwrap();
        (Arc::new(CmfModel::open(&path).unwrap()), dir)
    }

    fn reference(p: &[u8], x: &[f32], rows: usize, cols: usize) -> (Vec<f64>, Vec<f64>) {
        let mut w = vec![0f32; rows * cols];
        cortiq_core::mxfp4::dequant(p, rows, cols, &mut w);
        let mut want = vec![0f64; rows];
        let mut mag = vec![0f64; rows];
        for r in 0..rows {
            for j in 0..cols {
                let t = w[r * cols + j] as f64 * x[j] as f64;
                want[r] += t;
                mag[r] += t.abs();
            }
        }
        (want, mag)
    }

    /// The device matvec and batched matvec against an f64 reference of the
    /// dequantized matrix (f32 summation noise only), the batch equal to the
    /// matvec per token bit for bit (one group walk, one reduction tree), and
    /// a > 65535-row tensor through the two-dimensional grid. Run on macOS
    /// with `CMF_GPU=wgpu` (wgpu over Metal); skips without an adapter.
    #[test]
    fn device_matvec_and_batch_match_the_reference() {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping mxfp4 device matvec test");
            return;
        };
        let _drain = Drain(c.device.clone());
        if pipes(c).is_none() {
            eprintln!("mxfp4 pipelines unavailable on this adapter — skipping");
            return;
        }
        eprintln!(
            "mxfp4 matvec on the device: {} ({:?})",
            c.adapter_info.name, c.adapter_info.backend
        );
        let shapes = [(300usize, 2304usize), (70_000, 64), (5, 5120)];
        let tensors: Vec<(&str, usize, usize, Vec<u8>)> = shapes
            .iter()
            .enumerate()
            .map(|(i, &(r, cl))| (["a", "b", "c"][i], r, cl, payload(r, cl, i + 1)))
            .collect();
        let (model, dir) = model_with("mv", &tensors);
        for (i, (_, rows, cols, p)) in tensors.iter().enumerate() {
            let (rows, cols) = (*rows, *cols);
            let x = act(cols, i);
            let mut got = vec![0f32; rows];
            assert!(
                matmat(&model, i, &x, 1, rows, cols, &mut got),
                "matvec {rows}x{cols}"
            );
            let (want, mag) = reference(p, &x, rows, cols);
            for r in 0..rows {
                let tol = mag[r] * 4e-6 + 1e-30;
                assert!(
                    (got[r] as f64 - want[r]).abs() <= tol,
                    "{rows}x{cols} row {r}: {} vs {} (tol {tol})",
                    got[r],
                    want[r]
                );
            }
            if rows > 1000 && cols < 128 {
                continue; // the tall tensor checks the grid; batch it below once
            }
            for b in [2usize, 8, 9, 19] {
                let xs: Vec<f32> = (0..b).flat_map(|t| act(cols, t * 3 + i)).collect();
                let mut mm = vec![0f32; b * rows];
                assert!(matmat(&model, i, &xs, b, rows, cols, &mut mm), "batch {b}");
                for t in 0..b {
                    let mut one = vec![0f32; rows];
                    assert!(matmat(
                        &model,
                        i,
                        &xs[t * cols..(t + 1) * cols],
                        1,
                        rows,
                        cols,
                        &mut one
                    ));
                    assert!(
                        one.iter()
                            .zip(&mm[t * rows..(t + 1) * rows])
                            .all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{rows}x{cols} batch {b} token {t} differs from its matvec"
                    );
                }
            }
        }
        // A wrong dtype or shape is refused, not misread.
        let mut o = vec![0f32; 300];
        assert!(!matmat(&model, 0, &act(2304, 0), 1, 300, 2272, &mut o));
        drop(model);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn bf16(v: f32) -> f32 {
        let bits = v.to_bits();
        f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
    }

    /// The DeepSeek-V4.1 dynamic expert path on an mxfp4 bank: the segmented
    /// global pool is created for the mxfp4 codec, slots are filled with the
    /// tensors' bytes verbatim, and `dsv4_moe_frame` in V4.1's forced +
    /// preweighted + BF16 mode matches an independent host computation of the
    /// same experts (BF16 boundaries as in `dsv41::moe`), with a cold winner
    /// exported to the host. Skips without an adapter or descriptor indexing.
    #[test]
    fn global_bank_frame_runs_mxfp4_experts() {
        global_bank_case("bank", 128, 64, 0.5, 16, 0.02, Dsv41Segments::Auto);
    }

    /// [`global_bank_frame_runs_mxfp4_experts`] on an S16 bank (the
    /// `CMF_DSV41_GLOBAL_SEGMENTS=16` policy, which for mxfp4 builds the
    /// mxfp4 S16 kernels on first need): 32 slots, two per segment, so the
    /// five experts land in segments 0, 0, 1, 1, 2 and a wrong segment or
    /// local index costs O(1). Skips where the adapter cannot do S16.
    #[test]
    fn global_bank_frame_runs_mxfp4_experts_on_s16() {
        global_bank_case("bank-s16", 128, 64, 0.5, 32, 0.02, Dsv41Segments::Eager16);
    }

    /// [`global_bank_frame_runs_mxfp4_experts`] at the V4.1 release expert
    /// geometry (gate/up 2304x5120, down 5120x2304): 160 groups per gate/up
    /// row, so each lane of `dsv4_global_gate_up_mxfp4` walks 2-3 groups,
    /// and 72 per down row, so the down kernel's flat `slot · gpr + g` walk
    /// hands one lane groups of different slots (lane 10: slot 0 g10, slot
    /// 1 g2, slot 1 g66, slot 2 g58). V4.1's SwiGLU limit 10. Eight slots
    /// (150 MB) of bank.
    #[test]
    fn global_bank_frame_runs_mxfp4_experts_at_release_geometry() {
        global_bank_case("bank-1017", 5120, 2304, 10.0, 8, 0.01, Dsv41Segments::Auto);
    }

    /// One bank-frame comparison: four routed experts plus a shared one,
    /// `[inter, dim]` gate/up and `[dim, inter]` down, three routes (the
    /// last with a cold winner), bounded by `max_nrmse` and cos > 0.999
    /// against the host. `policy` is the V4.1 bank's segment policy; under
    /// `Eager16` the bank must come out S16.
    fn global_bank_case(
        tag: &str,
        dim: usize,
        inter: usize,
        limit: f32,
        extra_slots: usize,
        max_nrmse: f64,
        policy: Dsv41Segments,
    ) {
        unsafe { std::env::set_var("CMF_GPU", "wgpu") };
        let Some(c) = ctx() else {
            eprintln!("no wgpu adapter — skipping mxfp4 global bank test");
            return;
        };
        let _drain = Drain(c.device.clone());
        if !dsv4_global_moe_supported() || global_pipes(c, DSV4_GLOBAL_MOE_SEGMENTS).is_none() {
            eprintln!("descriptor-indexed global bank unavailable — skipping");
            return;
        }
        let want_s16 = policy == Dsv41Segments::Eager16;
        if want_s16
            && !(c.dsv4_global_s16_capable
                && global_pipes(c, DSV4_GLOBAL_MOE_SEGMENTS_S16).is_some())
        {
            eprintln!("S16 mxfp4 bank unavailable on this adapter — skipping");
            return;
        }
        eprintln!(
            "mxfp4 global bank {dim}x{inter} on the device: {} ({:?})",
            c.adapter_info.name, c.adapter_info.backend
        );
        let n_exp = 4usize;
        let mut tensors = Vec::new();
        let names: Vec<[String; 3]> = (0..=n_exp)
            .map(|e| {
                let p = if e == n_exp {
                    "model.layers.0.mlp.shared_expert".to_string()
                } else {
                    format!("model.layers.0.mlp.experts.{e}")
                };
                [
                    format!("{p}.gate_proj.weight"),
                    format!("{p}.up_proj.weight"),
                    format!("{p}.down_proj.weight"),
                ]
            })
            .collect();
        for (e, n) in names.iter().enumerate() {
            tensors.push((n[0].as_str(), inter, dim, payload(inter, dim, 10 * e + 1)));
            tensors.push((n[1].as_str(), inter, dim, payload(inter, dim, 10 * e + 2)));
            tensors.push((n[2].as_str(), dim, inter, payload(dim, inter, 10 * e + 3)));
        }
        let (model, dir) = model_with(tag, &tensors);
        let triples: Vec<(usize, usize, usize)> =
            (0..=n_exp).map(|e| (3 * e, 3 * e + 1, 3 * e + 2)).collect();
        let (gu_len, d_len) = MoeBankCodec::Mxfp4.lens(inter, dim).unwrap();
        let gib = 1024u64 * 1024 * 1024;
        let budget = dsv4_vram_budget().unwrap();
        let workspace = (budget / 10).clamp(2 * gib, 4 * gib);
        let request = (workspace / (2 * gu_len + d_len) as u64) as usize + extra_slots;
        let Some((capacity, segment_slots)) = dsv41_global_moe_create_with_policy(
            &model,
            request,
            inter,
            dim,
            MoeBankCodec::Mxfp4,
            policy,
        ) else {
            eprintln!("global mxfp4 bank could not be allocated here — skipping");
            let _ = std::fs::remove_dir_all(&dir);
            return;
        };
        assert!(
            capacity >= 8 && capacity <= extra_slots,
            "capacity {capacity}"
        );
        let segments = if want_s16 {
            DSV4_GLOBAL_MOE_SEGMENTS_S16
        } else {
            DSV4_GLOBAL_MOE_SEGMENTS
        };
        assert_eq!(
            dsv4_global_moe_geometry(&model),
            Some((segments, capacity)),
            "bank segment family"
        );
        assert_eq!(capacity, segments * segment_slots);
        // A second create for another codec on the same model is refused.
        assert!(
            dsv41_global_moe_create_with_policy(
                &model,
                request,
                inter,
                dim,
                MoeBankCodec::Q4tp,
                policy
            )
            .is_none()
        );
        for (slot, t) in triples.iter().copied().enumerate() {
            assert!(dsv4_global_slot_fill(&model, slot, t), "fill slot {slot}");
        }
        let q: Vec<[crate::qtensor::QTensor; 3]> = names
            .iter()
            .map(|n| {
                n.clone()
                    .map(|name| crate::qtensor::QTensor::from_model(&model, &name).unwrap())
            })
            .collect();
        let host_expert = |e: usize, x: &[f32], wgt: f32| -> Vec<f32> {
            let (mut g, mut u) = (vec![0f32; inter], vec![0f32; inter]);
            crate::gpu::cpu_scope(|| {
                crate::qtensor::with_float_activations(|| {
                    q[e][0].matvec(x, &mut g, None);
                    q[e][1].matvec(x, &mut u, None);
                })
            });
            for i in 0..inter {
                let (mut gi, mut ui) = (bf16(g[i]), bf16(u[i]));
                ui = ui.clamp(-limit, limit);
                gi = gi.min(limit);
                g[i] = bf16(gi / (1.0 + (-gi).exp()) * ui * wgt);
            }
            let mut out = vec![0f32; dim];
            crate::gpu::cpu_scope(|| {
                crate::qtensor::with_float_activations(|| q[e][2].matvec(&g, &mut out, None))
            });
            out.iter().map(|&v| bf16(v)).collect()
        };
        for (case, (picks, wts, cold)) in [
            ([0usize, 1usize], [0.25f32, 0.75f32], false),
            ([2, 3], [0.9, 0.1], false),
            ([3, 0], [0.6, 0.4], true),
        ]
        .into_iter()
        .enumerate()
        {
            let x: Vec<f32> = act(dim, case + 7).iter().map(|&v| bf16(v * 2.0)).collect();
            let mut weights = vec![0f32; n_exp];
            weights[picks[0]] = wts[0];
            weights[picks[1]] = wts[1];
            let logits: Vec<f32> = (0..n_exp)
                .map(|e| if picks.contains(&e) { 1.0 } else { 0.0 })
                .collect();
            let mut remap: Vec<u32> = (0..n_exp as u32).collect();
            if cold {
                remap[picks[1]] = u32::MAX;
            }
            let w = Dsv4MoeW {
                router: &[],
                experts: &triples[..n_exp],
                logits: &logits,
                bias: Some(&weights),
                mask: None,
                forced: Some(&picks),
                remap: Some(&remap),
                global: Some(Dsv4GlobalMoe {
                    pool_uid: model.uid(),
                    shared_slot: n_exp as u32,
                    segment_slots: segment_slots as u32,
                }),
                has_shared: true,
                shared_weight: 1.0,
                preweighted: true,
                qwen_softmax: false,
            };
            let geom = Dsv4MoeGeom {
                hidden: dim,
                inter,
                top_k: 2,
                route_scale: 1.0,
                swiglu_limit: limit,
                gu_q2: false,
                bf16: true,
            };
            let (mut got, mut cold_ids, mut cold_x) = (vec![0f32; dim], Vec::new(), Vec::new());
            assert!(
                dsv4_moe_frame(
                    &model,
                    &w,
                    geom,
                    &x,
                    &mut cold_ids,
                    &mut cold_x,
                    None,
                    None,
                    &mut got
                ),
                "frame case {case}"
            );
            let mut want = vec![0f32; dim];
            for (k, &e) in picks.iter().enumerate() {
                if cold && k == 1 {
                    continue;
                }
                for (d, v) in want.iter_mut().zip(host_expert(e, &x, wts[k])) {
                    *d += v;
                }
            }
            for (d, v) in want.iter_mut().zip(host_expert(n_exp, &x, 1.0)) {
                *d += v;
            }
            if cold {
                assert_eq!(cold_ids.len(), 1, "case {case}: {cold_ids:?}");
                assert_eq!(cold_ids[0].0, picks[1]);
            } else {
                assert!(cold_ids.is_empty(), "case {case}: {cold_ids:?}");
            }
            // The device and host sum in different orders and both round to
            // BF16 at three boundaries: compare as vectors.
            let (mut num, mut den, mut dot, mut gg) = (0f64, 0f64, 0f64, 0f64);
            for (a, b) in got.iter().zip(&want) {
                num += ((a - b) as f64).powi(2);
                den += (*b as f64).powi(2);
                dot += *a as f64 * *b as f64;
                gg += (*a as f64).powi(2);
            }
            let cos = dot / (den * gg).sqrt();
            let nrmse = (num / den).sqrt();
            eprintln!("  {tag} case {case}: nrmse {nrmse:.3e} cos {cos:.7}");
            assert!(den > 0.0, "case {case}: the experts contribute nothing");
            assert!(
                nrmse < max_nrmse && cos > 0.999,
                "case {case}: nrmse {nrmse} cos {cos}"
            );
        }
        drop(q);
        drop(model);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bank_codec_lengths() {
        let (g, d) = MoeBankCodec::Mxfp4.lens(2304, 5120).unwrap();
        assert_eq!((g, d), (2304 * 5120 * 17 / 32, 5120 * 2304 * 17 / 32));
        assert_eq!(
            MoeBankCodec::Q4tp.lens(64, 128),
            Some((
                cortiq_core::quant::expected_nbytes(cortiq_core::TensorDtype::Q4TiledP, &[64, 128])
                    .unwrap(),
                cortiq_core::quant::expected_nbytes(cortiq_core::TensorDtype::Q4TiledP, &[128, 64])
                    .unwrap()
            ))
        );
        // 1×32 → 17 bytes: not word-aligned, refused for the bank.
        assert_eq!(MoeBankCodec::Mxfp4.lens(1, 32), None);
        assert!(!MoeBankCodec::Mxfp4.gu_q2() && MoeBankCodec::Q2tpGateUp.gu_q2());
    }
}
