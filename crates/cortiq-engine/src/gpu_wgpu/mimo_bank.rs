//! MiMo-V2 expert-bank frame: the resident top-k experts of one token and
//! one layer, read straight out of the model-wide segmented slot bank
//! (`dsv4_global_*` buffers), in one submit and one readback.
//!
//! Rules (the `zimage` submodule's): reach the parent's context and helpers
//! through `super::`, never edit parent functions, keep pipelines and
//! per-model buffers in module-local caches.
//!
//! Why its own kernels: the bank's generic `dsv4_global_*` kernels read the
//! q4tp planes 16 bits at a time and evaluate an `exp2` per group; on the
//! RTX PRO 6000 a MiMo frame (8 × 13.1 MB of experts) measured 0.36 ms of
//! card time — under 300 GB/s. These read each 32-weight group as one
//! 16-byte vector, take the row's scales from a ladder built once per
//! workgroup in shared memory (the dense `q4tp_matvec4` recipe), run gate
//! and up off the same activation loads, and skip slots the host computes
//! (cold picks) instead of running them at weight zero.
//!
//! Numerics: f32 throughout, the same nibble decode and the same ladder
//! form (`exp2(lo + code·step)`) as the parent's q4tp kernels; only the
//! summation order differs from the host path.

use std::sync::{Arc, Mutex};

use cortiq_core::CmfModel;

use super::Ctx;

const COMMON: &str = r#"
enable wgpu_binding_array;

struct Bank { v: array<vec4<u32>> };
struct MbP {
    gpr_h: u32, inter: u32, hidden: u32, gpr_i: u32,
    seg_slots: u32, gu16: u32, d16: u32, slots: u32,
};

fn mb_nib(w: u32, s: u32) -> f32 { return f32((w >> s) & 0xFu) - 8.0; }
fn mb_dot8(w: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return mb_nib(w, 0u) * a.x
         + mb_nib(w, 4u) * a.y
         + mb_nib(w, 8u) * a.z
         + mb_nib(w, 12u) * a.w
         + mb_nib(w, 16u) * b.x
         + mb_nib(w, 20u) * b.y
         + mb_nib(w, 24u) * b.z
         + mb_nib(w, 28u) * b.w;
}
fn mb_word(v: vec4<u32>, i: u32) -> u32 {
    var r = v.x;
    if (i == 1u) { r = v.y; }
    if (i == 2u) { r = v.z; }
    if (i == 3u) { r = v.w; }
    return r;
}
"#;

const GATE_UP: &str = r#"
@group(0) @binding(0) var<storage, read>       gu_g   : binding_array<Bank, SEGS>;
@group(0) @binding(1) var<storage, read>       gu_u   : binding_array<Bank, SEGS>;
@group(0) @binding(2) var<storage, read>       gu_x   : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       gu_sel : array<u32>;
@group(0) @binding(4) var<storage, read_write> gu_act : array<f32>;
@group(1) @binding(0) var<uniform>             mb_p   : MbP;

var<workgroup> gu_lad: array<f32, 256>;
var<workgroup> gu_pg: array<f32, 256>;
var<workgroup> gu_pu: array<f32, 256>;

fn gu_gw(seg: u32, w: u32) -> u32 { return mb_word(gu_g[seg].v[w >> 2u], w & 3u); }
fn gu_uw(seg: u32, w: u32) -> u32 { return mb_word(gu_u[seg].v[w >> 2u], w & 3u); }
fn gu_gb(seg: u32, b: u32) -> u32 { return (gu_gw(seg, b >> 2u) >> ((b & 3u) * 8u)) & 0xFFu; }
fn gu_ub(seg: u32, b: u32) -> u32 { return (gu_uw(seg, b >> 2u) >> ((b & 3u) * 8u)) & 0xFFu; }

// Workgroup = 4 rows of one slot, a 64-lane sub-block per row, gate and up
// together off the same activation vectors. act = silu(gate)·up; a cold
// slot (NONE) writes zeros and reads nothing.
@compute @workgroup_size(256)
fn mimo_bank_gate_up(@builtin(workgroup_id) wid: vec3<u32>,
                     @builtin(local_invocation_index) lid: u32) {
    let token = wid.z;
    let slot = token * mb_p.slots + wid.y;
    let flat = gu_sel[slot];
    let live = flat != 0xFFFFFFFFu;
    var seg = 0u;
    var local = 0u;
    if (live) {
        seg = flat / mb_p.seg_slots;
        local = flat - seg * mb_p.seg_slots;
    }
    let gpr = mb_p.gpr_h;
    let rows = mb_p.inter;
    let base = local * mb_p.gu16;
    let par_w = base * 4u + rows * gpr * 4u;
    let cod_b = par_w * 4u + rows * 4u;
    let cst = (gpr * 5u + 7u) / 8u;
    let row0 = wid.x * 4u;
    {
        // Ladders: [matrix][row][rung], one rung per lane.
        let m = lid >> 7u;
        let r = (lid >> 5u) & 3u;
        var lv = 0.0;
        if (live && row0 + r < rows) {
            var pw = 0u;
            if (m == 0u) { pw = gu_gw(seg, par_w + row0 + r); }
            else { pw = gu_uw(seg, par_w + row0 + r); }
            let pr = unpack2x16float(pw);
            lv = exp2(pr.x + f32(lid & 31u) * pr.y);
        }
        gu_lad[lid] = lv;
    }
    workgroupBarrier();
    let sub = lid >> 6u;
    let l = lid & 63u;
    let row = row0 + sub;
    var ag = 0.0;
    var au = 0.0;
    if (live && row < rows) {
        let crow = cod_b + row * cst;
        let wrow = base + row * gpr;
        var g = l;
        loop {
            if (g >= gpr) { break; }
            let bit = g * 5u;
            let cb = bit >> 3u;
            let sh = bit & 7u;
            var cg = gu_gb(seg, crow + cb);
            var cu = gu_ub(seg, crow + cb);
            if (sh > 3u) {
                cg = cg | (gu_gb(seg, crow + cb + 1u) << 8u);
                cu = cu | (gu_ub(seg, crow + cb + 1u) << 8u);
            }
            let vg = gu_g[seg].v[wrow + g];
            let vu = gu_u[seg].v[wrow + g];
            let xq = token * (mb_p.hidden / 4u) + g * 8u;
            let x0 = gu_x[xq];      let x1 = gu_x[xq + 1u];
            let x2 = gu_x[xq + 2u]; let x3 = gu_x[xq + 3u];
            let x4 = gu_x[xq + 4u]; let x5 = gu_x[xq + 5u];
            let x6 = gu_x[xq + 6u]; let x7 = gu_x[xq + 7u];
            let sg = gu_lad[(sub << 5u) + ((cg >> sh) & 31u)];
            let su = gu_lad[128u + (sub << 5u) + ((cu >> sh) & 31u)];
            ag = ag + sg * (mb_dot8(vg.x, x0, x1) + mb_dot8(vg.y, x2, x3)
                          + mb_dot8(vg.z, x4, x5) + mb_dot8(vg.w, x6, x7));
            au = au + su * (mb_dot8(vu.x, x0, x1) + mb_dot8(vu.y, x2, x3)
                          + mb_dot8(vu.z, x4, x5) + mb_dot8(vu.w, x6, x7));
            g = g + 64u;
        }
    }
    gu_pg[lid] = ag;
    gu_pu[lid] = au;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (l < stride) {
            gu_pg[lid] = gu_pg[lid] + gu_pg[lid + stride];
            gu_pu[lid] = gu_pu[lid] + gu_pu[lid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (l == 0u && row < rows) {
        let gg = gu_pg[sub << 6u];
        let uu = gu_pu[sub << 6u];
        var a = 0.0;
        if (live) { a = (gg / (1.0 + exp(-gg))) * uu; }
        gu_act[slot * rows + row] = a;
    }
}
"#;

const DOWN: &str = r#"
@group(0) @binding(0) var<storage, read>       dn_d   : binding_array<Bank, SEGS>;
@group(0) @binding(1) var<storage, read>       dn_act : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       dn_sel : array<u32>;
@group(0) @binding(3) var<storage, read>       dn_wt  : array<f32>;
@group(0) @binding(4) var<storage, read_write> dn_out : array<f32>;
@group(1) @binding(0) var<uniform>             mb_p   : MbP;

var<workgroup> dn_pt: array<f32, 256>;

fn dn_dw(seg: u32, w: u32) -> u32 { return mb_word(dn_d[seg].v[w >> 2u], w & 3u); }
fn dn_db(seg: u32, b: u32) -> u32 { return (dn_dw(seg, b >> 2u) >> ((b & 3u) * 8u)) & 0xFFu; }

// Workgroup = 4 output rows, a 64-lane sub-block per row walking every
// (slot, group) term; cold slots are skipped.
@compute @workgroup_size(256)
fn mimo_bank_down(@builtin(workgroup_id) wid: vec3<u32>,
                  @builtin(local_invocation_index) lid: u32) {
    let token = wid.y;
    let sub = lid >> 6u;
    let l = lid & 63u;
    let row = wid.x * 4u + sub;
    let rows = mb_p.hidden;
    let gpr = mb_p.gpr_i;
    let cst = (gpr * 5u + 7u) / 8u;
    let act4 = mb_p.inter / 4u;
    var acc = 0.0;
    if (row < rows) {
        let total = mb_p.slots * gpr;
        var i = l;
        loop {
            if (i >= total) { break; }
            let local_slot = i / gpr;
            let g = i - local_slot * gpr;
            let slot = token * mb_p.slots + local_slot;
            let flat = dn_sel[slot];
            let w = dn_wt[slot];
            if (flat != 0xFFFFFFFFu && w != 0.0) {
                let seg = flat / mb_p.seg_slots;
                let local = flat - seg * mb_p.seg_slots;
                let base = local * mb_p.d16;
                let par_w = base * 4u + rows * gpr * 4u;
                let pr = unpack2x16float(dn_dw(seg, par_w + row));
                let crow = par_w * 4u + rows * 4u + row * cst;
                let bit = g * 5u;
                let cb = bit >> 3u;
                let sh = bit & 7u;
                var cv = dn_db(seg, crow + cb);
                if (sh > 3u) { cv = cv | (dn_db(seg, crow + cb + 1u) << 8u); }
                let scale = exp2(pr.x + f32((cv >> sh) & 31u) * pr.y);
                let v = dn_d[seg].v[base + row * gpr + g];
                let aq = slot * act4 + g * 8u;
                let a0 = dn_act[aq];      let a1 = dn_act[aq + 1u];
                let a2 = dn_act[aq + 2u]; let a3 = dn_act[aq + 3u];
                let a4 = dn_act[aq + 4u]; let a5 = dn_act[aq + 5u];
                let a6 = dn_act[aq + 6u]; let a7 = dn_act[aq + 7u];
                acc = acc + w * scale * (mb_dot8(v.x, a0, a1) + mb_dot8(v.y, a2, a3)
                                       + mb_dot8(v.z, a4, a5) + mb_dot8(v.w, a6, a7));
            }
            i = i + 64u;
        }
    }
    dn_pt[lid] = acc;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if (stride == 0u) { break; }
        if (l < stride) { dn_pt[lid] = dn_pt[lid] + dn_pt[lid + stride]; }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    if (l == 0u && row < rows) { dn_out[token * mb_p.hidden + row] = dn_pt[sub << 6u]; }
}
"#;

struct Pipes {
    gu: wgpu::ComputePipeline,
    dn: wgpu::ComputePipeline,
}

/// Pipelines per descriptor-array width (8 or 16 segments).
static PIPES: Mutex<Vec<(usize, Arc<Pipes>)>> = Mutex::new(Vec::new());

fn pipes(c: &Ctx, segs: usize) -> Option<Arc<Pipes>> {
    let mut cache = PIPES.lock().unwrap();
    if let Some((_, p)) = cache.iter().find(|(s, _)| *s == segs) {
        return Some(p.clone());
    }
    let dev = &c.device;
    let scope = dev.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = |label: &str, body: &str| {
        let src = format!("{COMMON}{}", body.replace("SEGS", &segs.to_string()));
        dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        })
    };
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
    let s = Some(segs as u32);
    let gu0 = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mimo-bank-gu0"),
        entries: &[
            storage(0, true, s),
            storage(1, true, s),
            storage(2, true, None),
            storage(3, true, None),
            storage(4, false, None),
        ],
    });
    let dn0 = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mimo-bank-dn0"),
        entries: &[
            storage(0, true, s),
            storage(1, true, None),
            storage(2, true, None),
            storage(3, true, None),
            storage(4, false, None),
        ],
    });
    let params = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mimo-bank-params"),
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
    let pipeline =
        |label: &str, bgl: &wgpu::BindGroupLayout, m: &wgpu::ShaderModule, entry: &str| {
            let layout = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(bgl), Some(&params)],
                immediate_size: 0,
            });
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: m,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: c.pipeline_cache.as_ref(),
            })
        };
    let gm = module("mimo-bank-gu", GATE_UP);
    let dm = module("mimo-bank-dn", DOWN);
    let p = Pipes {
        gu: pipeline("mimo-bank-gate-up", &gu0, &gm, "mimo_bank_gate_up"),
        dn: pipeline("mimo-bank-down", &dn0, &dm, "mimo_bank_down"),
    };
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("MiMo bank kernels unavailable ({e}) — generic bank frame");
        return None;
    }
    let p = Arc::new(p);
    cache.push((segs, p.clone()));
    Some(p)
}

/// Per-model frame state: the static bind groups over the bank and the
/// frame's own small buffers.
struct State {
    pipes: Arc<Pipes>,
    x: wgpu::Buffer,
    sel: wgpu::Buffer,
    wt: wgpu::Buffer,
    out: wgpu::Buffer,
    stage: wgpu::Buffer,
    bg_gu: wgpu::BindGroup,
    bg_dn: wgpu::BindGroup,
    bg_p: wgpu::BindGroup,
    bg_pd: wgpu::BindGroup,
    hidden: usize,
    inter: usize,
    slots: usize,
    rows: usize,
}

static STATES: Mutex<Vec<(u64, Arc<State>)>> = Mutex::new(Vec::new());

fn state(
    c: &Ctx,
    model: &Arc<CmfModel>,
    hidden: usize,
    inter: usize,
    slots: usize,
    rows: usize,
) -> Option<Arc<State>> {
    let mut cache = STATES.lock().unwrap();
    if let Some((_, s)) = cache
        .iter()
        .find(|(u, s)| *u == model.uid() && s.rows == rows)
    {
        return (s.hidden == hidden && s.inter == inter && s.slots == slots).then(|| s.clone());
    }
    let bank = c
        .dsv4_global_moe
        .lock()
        .unwrap()
        .get(&model.uid())
        .cloned()?;
    // 16-byte vector reads need 16-byte slot strides; q2tp gate/up is a
    // different weight plane.
    if bank.gu_q2
        || bank.gu_len % 16 != 0
        || bank.d_len % 16 != 0
        || hidden % 32 != 0
        || inter % 32 != 0
        || slots == 0
        || slots > 64
        || rows == 0
        || rows > 4
    {
        return None;
    }
    let gu16 = (bank.gu_len / 16) as u32;
    let d16 = (bank.d_len / 16) as u32;
    // Word/byte offsets inside one segment must stay below 2^32.
    if (bank.segment_slots as u64) * (bank.gu_len.max(bank.d_len) as u64) >= (1u64 << 32) {
        return None;
    }
    let pipes = pipes(c, bank.segments)?;
    let dev = &c.device;
    let mk = |label: &str, size: usize, usage: wgpu::BufferUsages| {
        dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16) as u64,
            usage,
            mapped_at_creation: false,
        })
    };
    let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let x = mk("mimo-bank-x", rows * hidden * 4, st);
    let sel = mk("mimo-bank-sel", rows * slots * 4, st);
    let wt = mk("mimo-bank-wt", rows * slots * 4, st);
    let act = mk(
        "mimo-bank-act",
        rows * slots * inter * 4,
        wgpu::BufferUsages::STORAGE,
    );
    let out = mk(
        "mimo-bank-out",
        rows * hidden * 4,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let stage = mk(
        "mimo-bank-stage",
        rows * hidden * 4,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let params = mk(
        "mimo-bank-params",
        32,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let p: [u32; 8] = [
        (hidden / 32) as u32,
        inter as u32,
        hidden as u32,
        (inter / 32) as u32,
        bank.segment_slots as u32,
        gu16,
        d16,
        slots as u32,
    ];
    c.queue.write_buffer(&params, 0, bytemuck::cast_slice(&p));
    fn arr(bufs: &[wgpu::Buffer]) -> Vec<wgpu::BufferBinding<'_>> {
        bufs.iter()
            .map(wgpu::Buffer::as_entire_buffer_binding)
            .collect()
    }
    let (ga, ua, da) = (arr(&bank.gate), arr(&bank.up), arr(&bank.down));
    let bg_gu = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mimo-bank-gu"),
        layout: &pipes.gu.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::BufferArray(&ga),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::BufferArray(&ua),
            },
            super::bind_buf(2, &x),
            super::bind_buf(3, &sel),
            super::bind_buf(4, &act),
        ],
    });
    let bg_dn = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mimo-bank-dn"),
        layout: &pipes.dn.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::BufferArray(&da),
            },
            super::bind_buf(1, &act),
            super::bind_buf(2, &sel),
            super::bind_buf(3, &wt),
            super::bind_buf(4, &out),
        ],
    });
    let bg_p = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mimo-bank-p"),
        layout: &pipes.gu.get_bind_group_layout(1),
        entries: &[super::bind_buf(0, &params)],
    });
    let bg_pd = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mimo-bank-pd"),
        layout: &pipes.dn.get_bind_group_layout(1),
        entries: &[super::bind_buf(0, &params)],
    });
    let s = Arc::new(State {
        pipes,
        x,
        sel,
        wt,
        out,
        stage,
        bg_gu,
        bg_dn,
        bg_p,
        bg_pd,
        hidden,
        inter,
        slots,
        rows,
    });
    cache.push((model.uid(), s.clone()));
    Some(s)
}

/// Whether the dedicated kernels serve this model's bank.
pub fn mimo_bank_ready(model: &Arc<CmfModel>, hidden: usize, inter: usize, slots: usize) -> bool {
    super::ctx().is_some_and(|c| state(c, model, hidden, inter, slots, 1).is_some())
}

/// Σ over `sel`'s resident slots of `wt[s]·down(silu(gate·x)⊙up·x)` into
/// `out`. `sel[s] = u32::MAX` marks a pick the host computes (it
/// contributes nothing here). `false` = not served; `out` is untouched.
pub fn mimo_bank_frame(
    model: &Arc<CmfModel>,
    x: &[f32],
    sel: &[u32],
    wt: &[f32],
    inter: usize,
    out: &mut [f32],
) -> bool {
    mimo_bank_rows(model, x, sel, wt, inter, 1, out)
}

/// K independent token routes in two dispatches and one readback. Token
/// indexing is outside every reduction, so each row uses the same f32
/// arithmetic as `mimo_bank_frame`, even when several rows pick one expert.
/// The caller holds the bank lock, pinning the UNION of all selected slots.
pub fn mimo_bank_rows(
    model: &Arc<CmfModel>,
    x: &[f32],
    sel: &[u32],
    wt: &[f32],
    inter: usize,
    rows: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = super::ctx() else { return false };
    if rows == 0 || rows > 4 || x.len() % rows != 0 || sel.len() % rows != 0 {
        return false;
    }
    let hidden = x.len() / rows;
    let slots = sel.len() / rows;
    if sel.len() != wt.len() || out.len() < x.len() {
        return false;
    }
    let Some(s) = state(c, model, hidden, inter, slots, rows) else {
        return false;
    };
    c.queue.write_buffer(&s.x, 0, bytemuck::cast_slice(x));
    c.queue.write_buffer(&s.sel, 0, bytemuck::cast_slice(sel));
    c.queue.write_buffer(&s.wt, 0, bytemuck::cast_slice(wt));
    let mut enc = c
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mimo-bank"),
        });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mimo-bank"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&s.pipes.gu);
        pass.set_bind_group(0, &s.bg_gu, &[]);
        pass.set_bind_group(1, &s.bg_p, &[]);
        pass.dispatch_workgroups(inter.div_ceil(4) as u32, slots as u32, rows as u32);
        pass.set_pipeline(&s.pipes.dn);
        pass.set_bind_group(0, &s.bg_dn, &[]);
        pass.set_bind_group(1, &s.bg_pd, &[]);
        pass.dispatch_workgroups(hidden.div_ceil(4) as u32, rows as u32, 1);
    }
    let bytes = (x.len() * 4) as u64;
    enc.copy_buffer_to_buffer(&s.out, 0, &s.stage, 0, bytes);
    super::submit(c, enc.finish());
    let slice = s.stage.slice(..bytes);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r.is_ok());
    });
    if c.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        return false;
    }
    if !rx.recv().unwrap_or(false) {
        return false;
    }
    let ok = match slice.get_mapped_range() {
        Ok(data) => {
            out[..x.len()].copy_from_slice(bytemuck::cast_slice(&data[..x.len() * 4]));
            true
        }
        Err(_) => false,
    };
    s.stage.unmap();
    ok
}
