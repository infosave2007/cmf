//! Metal training context — device, queue, the compiled kernel set, and
//! thin typed dispatchers. Apple Silicon only (unified memory: every
//! buffer is StorageModeShared, so the host reads results without a
//! blit and the CPU reference ops in `ops.rs` can check any tensor).
//!
//! No MPS. The kernels are ours (`shaders.metal`, compiled at first use).

use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize,
};
use std::ffi::c_void;
use std::sync::OnceLock;

const MSL: &str = include_str!("shaders.metal");

/// GEMM operand layout: which of the two orientations a matrix is stored
/// in. See the kernel header for the exact meaning of TA/TB.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// stored as it is used ([M,K] for A, [K,N] for B)
    N,
    /// stored transposed ([K,M] for A, [N,K] for B)
    T,
}

pub struct Ctx {
    pub device: Device,
    pub queue: CommandQueue,
    _lib: Library,
    /// gemm_f32 specialised on (TA, TB): index = TA·2 + TB.
    gemm: [ComputePipelineState; 4],
    axpby: ComputePipelineState,
    adamw: ComputePipelineState,
    sumsq: ComputePipelineState,
    rms_fwd: ComputePipelineState,
    rms_bwd_dx: ComputePipelineState,
    rms_dw: ComputePipelineState,
    conv_fwd: ComputePipelineState,
    conv_dx: ComputePipelineState,
    conv_dw: ComputePipelineState,
    swiglu_fwd: ComputePipelineState,
    swiglu_bwd: ComputePipelineState,
    embed_gather: ComputePipelineState,
    softmax_ce: ComputePipelineState,
    hk_phi: ComputePipelineState,
    hk_dtheta: ComputePipelineState,
    hk_kv: ComputePipelineState,
    hk_dkv_split: ComputePipelineState,
    hk_states_fwd: ComputePipelineState,
    hk_chunk_fwd: ComputePipelineState,
    hk_dstates_bwd: ComputePipelineState,
    hk_chunk_bwd: ComputePipelineState,
    rope: ComputePipelineState,
    causal_softmax: ComputePipelineState,
    softmax_bwd: ComputePipelineState,
    sigmoid_fwd: ComputePipelineState,
    sigmoid_bwd: ComputePipelineState,
    embed_scatter_add: ComputePipelineState,
    copy: ComputePipelineState,
    kappa_fwd: ComputePipelineState,
    kappa_bwd: ComputePipelineState,
    hk_scale: ComputePipelineState,
    hk_unscale: ComputePipelineState,
    hk_states_par: ComputePipelineState,
    hk_dstates_par: ComputePipelineState,
    gather_rows: ComputePipelineState,
    scatter_add_rows: ComputePipelineState,
    softmax_ce_idx: ComputePipelineState,
    group_sum: ComputePipelineState,
    route: ComputePipelineState,
    route_group: ComputePipelineState,
    moe_gather: ComputePipelineState,
    moe_scatter_add: ComputePipelineState,
    moe_stats: ComputePipelineState,
    moe_update: ComputePipelineState,
    moe_indirect: ComputePipelineState,
    moe_init_mu: ComputePipelineState,
    moe_center: ComputePipelineState,
    mask_fwd: ComputePipelineState,
    mask_bwd_dm: ComputePipelineState,
    mask_bwd_dh: ComputePipelineState,
    hk_pow_from_alog: ComputePipelineState,
    hk_scale_ktp: ComputePipelineState,
    hk_dgamma: ComputePipelineState,
}
unsafe impl Send for Ctx {}
unsafe impl Sync for Ctx {}

static CTX: OnceLock<Result<Ctx, String>> = OnceLock::new();

/// The process-wide context; `None` (with the reason logged once) when
/// there is no usable Metal device.
pub fn ctx() -> Option<&'static Ctx> {
    match CTX.get_or_init(init) {
        Ok(c) => Some(c),
        Err(e) => {
            static ONCE: OnceLock<()> = OnceLock::new();
            ONCE.get_or_init(|| eprintln!("cortiq-embryo: Metal unavailable: {e}"));
            None
        }
    }
}

fn init() -> Result<Ctx, String> {
    let device = Device::system_default().ok_or("no Metal device")?;
    if !device.has_unified_memory() {
        return Err(format!("device '{}' has no unified memory", device.name()));
    }
    let opts = metal::CompileOptions::new();
    opts.set_language_version(metal::MTLLanguageVersion::V3_0);
    // Trainer: keep IEEE semantics (no fast-math contraction surprises
    // between the GPU and the CPU reference the gradchecks compare to).
    opts.set_fast_math_enabled(false);
    let lib = device
        .new_library_with_source(MSL, &opts)
        .map_err(|e| format!("MSL compile: {e}"))?;
    let pso = |name: &str| -> Result<ComputePipelineState, String> {
        let f = lib
            .get_function(name, None)
            .map_err(|e| format!("kernel {name}: {e}"))?;
        device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(|e| format!("pipeline {name}: {e}"))
    };
    let gemm_pso = |ta: bool, tb: bool| -> Result<ComputePipelineState, String> {
        let fcv = metal::FunctionConstantValues::new();
        fcv.set_constant_value_at_index(
            &ta as *const bool as *const c_void,
            metal::MTLDataType::Bool,
            0,
        );
        fcv.set_constant_value_at_index(
            &tb as *const bool as *const c_void,
            metal::MTLDataType::Bool,
            1,
        );
        let f = lib
            .get_function("gemm_f32", Some(fcv))
            .map_err(|e| format!("gemm_f32({ta},{tb}): {e}"))?;
        device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(|e| format!("pipeline gemm_f32({ta},{tb}): {e}"))
    };
    let gemm = [
        gemm_pso(false, false)?,
        gemm_pso(false, true)?,
        gemm_pso(true, false)?,
        gemm_pso(true, true)?,
    ];
    let queue = device.new_command_queue();
    Ok(Ctx {
        axpby: pso("axpby_f32")?,
        adamw: pso("adamw_f32")?,
        sumsq: pso("sumsq_f32")?,
        rms_fwd: pso("rmsnorm_fwd_f32")?,
        rms_bwd_dx: pso("rmsnorm_bwd_dx_f32")?,
        rms_dw: pso("rmsnorm_dw_f32")?,
        conv_fwd: pso("conv1d_fwd_f32")?,
        conv_dx: pso("conv1d_bwd_dx_f32")?,
        conv_dw: pso("conv1d_dw_f32")?,
        swiglu_fwd: pso("swiglu_fwd_f32")?,
        swiglu_bwd: pso("swiglu_bwd_f32")?,
        embed_gather: pso("embed_gather_f32")?,
        softmax_ce: pso("softmax_ce_f32")?,
        hk_phi: pso("hk_phi_f32")?,
        hk_dtheta: pso("hk_dtheta_f32")?,
        hk_kv: pso("hk_kv_f32")?,
        hk_dkv_split: pso("hk_dkv_split_f32")?,
        hk_states_fwd: pso("hk_states_fwd_f32")?,
        hk_chunk_fwd: pso("hk_chunk_fwd_f32")?,
        hk_dstates_bwd: pso("hk_dstates_bwd_f32")?,
        hk_chunk_bwd: pso("hk_chunk_bwd_f32")?,
        rope: pso("rope_f32")?,
        causal_softmax: pso("causal_softmax_rows_f32")?,
        softmax_bwd: pso("softmax_bwd_rows_f32")?,
        sigmoid_fwd: pso("sigmoid_fwd_f32")?,
        sigmoid_bwd: pso("sigmoid_bwd_f32")?,
        embed_scatter_add: pso("embed_scatter_add_f32")?,
        copy: pso("copy_f32")?,
        kappa_fwd: pso("kappa_fwd_f32")?,
        kappa_bwd: pso("kappa_bwd_f32")?,
        hk_scale: pso("hk_scale_f32")?,
        hk_unscale: pso("hk_unscale_f32")?,
        hk_states_par: pso("hk_states_fwd_par_f32")?,
        hk_dstates_par: pso("hk_dstates_bwd_par_f32")?,
        gather_rows: pso("gather_rows_f32")?,
        scatter_add_rows: pso("scatter_add_rows_f32")?,
        softmax_ce_idx: pso("softmax_ce_idx_f32")?,
        group_sum: pso("group_sum_heads_f32")?,
        route: pso("route_f32")?,
        route_group: pso("route_group_f32")?,
        moe_gather: pso("moe_gather_f32")?,
        moe_scatter_add: pso("moe_scatter_add_f32")?,
        moe_stats: pso("moe_stats_f32")?,
        moe_update: pso("moe_update_f32")?,
        moe_indirect: pso("moe_indirect_args_f32")?,
        moe_init_mu: pso("moe_init_mu_f32")?,
        moe_center: pso("moe_center_f32")?,
        mask_fwd: pso("mask_fwd_f32")?,
        mask_bwd_dm: pso("mask_bwd_dm_f32")?,
        mask_bwd_dh: pso("mask_bwd_dh_f32")?,
        hk_pow_from_alog: pso("hk_pow_from_alog_f32")?,
        hk_scale_ktp: pso("hk_scale_ktp_f32")?,
        hk_dgamma: pso("hk_dgamma_f32")?,
        gemm,
        _lib: lib,
        queue,
        device,
    })
}

/// A device buffer of `len` f32 (StorageModeShared: host-visible).
pub struct GBuf {
    pub buf: Buffer,
    pub len: usize,
}
unsafe impl Send for GBuf {}
unsafe impl Sync for GBuf {}

impl GBuf {
    pub fn zeros(c: &Ctx, len: usize) -> GBuf {
        let bytes = (len.max(1) * 4) as u64;
        let buf = c.device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        // new_buffer memory is zero-filled by Metal for shared storage
        // in practice, but the contract does not promise it — do it.
        unsafe { std::ptr::write_bytes(buf.contents() as *mut u8, 0, bytes as usize) };
        GBuf { buf, len }
    }
    pub fn from_slice(c: &Ctx, x: &[f32]) -> GBuf {
        let bytes = (x.len().max(1) * 4) as u64;
        let buf = c.device.new_buffer_with_data(
            x.as_ptr() as *const c_void,
            bytes,
            MTLResourceOptions::StorageModeShared,
        );
        GBuf { buf, len: x.len() }
    }
    pub fn from_u32(c: &Ctx, x: &[u32]) -> GBuf {
        let bytes = (x.len().max(1) * 4) as u64;
        let buf = c.device.new_buffer_with_data(
            x.as_ptr() as *const c_void,
            bytes,
            MTLResourceOptions::StorageModeShared,
        );
        GBuf { buf, len: x.len() }
    }
    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.buf.contents() as *const f32, self.len) }
    }
    /// Host view for writes. Only valid while no command buffer touching
    /// this buffer is in flight — the caller sequences that.
    #[allow(clippy::mut_from_ref)]
    pub fn as_mut_slice(&self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.buf.contents() as *mut f32, self.len) }
    }
    pub fn read_to(&self, out: &mut [f32]) {
        out.copy_from_slice(&self.as_slice()[..out.len()]);
    }
    pub fn to_vec(&self) -> Vec<f32> {
        self.as_slice().to_vec()
    }
    pub fn write_from(&self, x: &[f32]) {
        self.as_mut_slice()[..x.len()].copy_from_slice(x);
    }
    pub fn fill(&self, v: f32) {
        self.as_mut_slice().fill(v);
    }
}

/// One command buffer's worth of encoded work. `commit()` submits and
/// waits; kernels encoded in one Cmd run in order (one compute encoder).
pub struct Cmd<'a> {
    c: &'a Ctx,
    cb: metal::CommandBuffer,
    enc: metal::ComputeCommandEncoder,
}

impl<'a> Cmd<'a> {
    pub fn new(c: &'a Ctx) -> Cmd<'a> {
        let cb = c.queue.new_command_buffer().to_owned();
        let enc = cb.new_compute_command_encoder().to_owned();
        Cmd { c, cb, enc }
    }

    /// C[M,N] = alpha·op(A)·op(B) + beta·C. Leading dimensions are the
    /// stored row lengths (lda = K or M, ldb = N or K, ldc = N).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
        ta: Op,
        tb: Op,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &GBuf,
        a_off: usize,
        lda: usize,
        b: &GBuf,
        b_off: usize,
        ldb: usize,
        beta: f32,
        cbuf: &GBuf,
        c_off: usize,
        ldc: usize,
    ) {
        self.gemm_ex(
            ta, tb, m, n, k, alpha, a, a_off, lda, b, b_off, ldb, beta, cbuf, c_off, ldc, &GemmBatch::none(), false,
        );
    }

    /// Batched GEMM: one tile grid per (b, h, c) triple, operands offset by
    /// `batch` strides; `causal` zeroes C[i,j] for j > i in the epilogue.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_ex(
        &self,
        ta: Op,
        tb: Op,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &GBuf,
        a_off: usize,
        lda: usize,
        b: &GBuf,
        b_off: usize,
        ldb: usize,
        beta: f32,
        cbuf: &GBuf,
        c_off: usize,
        ldc: usize,
        batch: &GemmBatch,
        causal: bool,
    ) {
        self.gemm_dyn(ta, tb, m, n, k, alpha, a, a_off, lda, b, b_off, ldb, beta, cbuf, c_off, ldc, batch, causal, &GemmDyn::none());
    }

    /// GEMM with a GPU-decided shape: `dynamic.indirect` = (buffer, byte
    /// offset) holding the threadgroup grid (rows may be fewer than m/64
    /// tiles), `dynamic.kcount` = (buffer, element offset) with per-batch
    /// row counts → K = min(round64(count), k). Used by the routed experts so
    /// no expert computes its empty capacity rows.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_dyn(
        &self,
        ta: Op,
        tb: Op,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &GBuf,
        a_off: usize,
        lda: usize,
        b: &GBuf,
        b_off: usize,
        ldb: usize,
        beta: f32,
        cbuf: &GBuf,
        c_off: usize,
        ldc: usize,
        batch: &GemmBatch,
        causal: bool,
        dynamic: &GemmDyn<'_>,
    ) {
        assert!(m % 64 == 0 && n % 64 == 0 && k % 32 == 0, "gemm tile alignment: m={m} n={n} k={k}");
        assert!(lda % 4 == 0 && ldb % 4 == 0 && ldc % 4 == 0 && a_off % 4 == 0 && b_off % 4 == 0 && c_off % 4 == 0);
        let (arows, acols) = if ta == Op::N { (m, k) } else { (k, m) };
        let (brows, bcols) = if tb == Op::N { (k, n) } else { (n, k) };
        assert!(lda >= acols && ldb >= bcols && ldc >= n);
        let (nb, nh, nc) = (batch.nb.max(1), batch.nh.max(1), batch.nc.max(1));
        let last = |s: [usize; 3]| (nb - 1) * s[0] + (nh - 1) * s[1] + (nc - 1) * s[2];
        assert!(a_off + last(batch.sa) + (arows - 1) * lda + acols <= a.len, "gemm: A out of range");
        assert!(b_off + last(batch.sb) + (brows - 1) * ldb + bcols <= b.len, "gemm: B out of range");
        assert!(c_off + last(batch.sc) + (m - 1) * ldc + n <= cbuf.len, "gemm: C out of range");
        for s in [batch.sa, batch.sb, batch.sc] {
            assert!(s.iter().all(|x| x % 4 == 0), "gemm: batch strides must be multiples of 4 floats");
        }
        #[repr(C)]
        struct Args {
            sa: [u64; 3],
            sb: [u64; 3],
            sc: [u64; 3],
            m: u32,
            n: u32,
            k: u32,
            lda: u32,
            ldb: u32,
            ldc: u32,
            alpha: f32,
            beta: f32,
            nb_h: u32,
            nb_c: u32,
            mask: u32,
            kdyn: u32,
        }
        let cv = |s: [usize; 3]| [s[0] as u64, s[1] as u64, s[2] as u64];
        let args = Args {
            sa: cv(batch.sa),
            sb: cv(batch.sb),
            sc: cv(batch.sc),
            m: m as u32,
            n: n as u32,
            k: k as u32,
            lda: lda as u32,
            ldb: ldb as u32,
            ldc: ldc as u32,
            alpha,
            beta,
            nb_h: nh as u32,
            nb_c: nc as u32,
            mask: causal as u32,
            kdyn: dynamic.kcount.is_some() as u32,
        };
        let idx = (ta == Op::T) as usize * 2 + (tb == Op::T) as usize;
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.gemm[idx]);
        e.set_buffer(0, Some(&a.buf), (a_off * 4) as u64);
        e.set_buffer(1, Some(&b.buf), (b_off * 4) as u64);
        e.set_buffer(2, Some(&cbuf.buf), (c_off * 4) as u64);
        e.set_bytes(3, std::mem::size_of::<Args>() as u64, &args as *const Args as *const c_void);
        match dynamic.kcount {
            Some((kb, koff)) => e.set_buffer(4, Some(&kb.buf), (koff * 4) as u64),
            None => e.set_buffer(4, Some(&a.buf), 0), // never read (kdyn = 0)
        }
        match dynamic.indirect {
            Some((ib, ioff)) => e.dispatch_thread_groups_indirect(&ib.buf, ioff as u64, MTLSize::new(128, 1, 1)),
            None => e.dispatch_thread_groups(
                MTLSize::new((n / 64) as u64, (m / 64) as u64, (nb * nh * nc) as u64),
                MTLSize::new(128, 1, 1),
            ),
        }
    }

    fn set_u32x4(&self, idx: u64, v: [u32; 4]) {
        self.enc.set_bytes(idx, 16, v.as_ptr() as *const c_void);
    }

    fn set_u32(&self, idx: u64, v: u32) {
        self.enc.set_bytes(idx, 4, &v as *const u32 as *const c_void);
    }
    fn set_f32(&self, idx: u64, v: f32) {
        self.enc.set_bytes(idx, 4, &v as *const f32 as *const c_void);
    }
    fn grid1(&self, n: usize, tg: usize) {
        let groups = n.div_ceil(tg).max(1);
        self.enc.dispatch_thread_groups(
            MTLSize::new(groups as u64, 1, 1),
            MTLSize::new(tg as u64, 1, 1),
        );
    }

    /// y = a·x + b·y over n floats.
    pub fn axpby(&self, a: f32, x: &GBuf, b: f32, y: &GBuf, n: usize) {
        assert!(x.len >= n && y.len >= n);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.axpby);
        e.set_buffer(0, Some(&x.buf), 0);
        e.set_buffer(1, Some(&y.buf), 0);
        self.set_f32(2, a);
        self.set_f32(3, b);
        self.set_u32(4, n as u32);
        self.grid1(n, 256);
    }

    /// AdamW step over `n` parameters (see the kernel for the update).
    #[allow(clippy::too_many_arguments)]
    pub fn adamw(
        &self,
        p: &GBuf,
        g: &GBuf,
        m: &GBuf,
        v: &GBuf,
        n: usize,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
        step: u32,
        gscale: f32,
    ) {
        assert!(p.len >= n && g.len >= n && m.len >= n && v.len >= n);
        #[repr(C)]
        struct Args {
            n: u32,
            lr: f32,
            beta1: f32,
            beta2: f32,
            eps: f32,
            wd: f32,
            bc1: f32,
            bc2: f32,
            gscale: f32,
        }
        let t = step.max(1) as f64;
        let args = Args {
            n: n as u32,
            lr,
            beta1,
            beta2,
            eps,
            wd,
            bc1: (1.0 / (1.0 - (beta1 as f64).powf(t))) as f32,
            bc2: (1.0 / (1.0 - (beta2 as f64).powf(t))) as f32,
            gscale,
        };
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.adamw);
        e.set_buffer(0, Some(&p.buf), 0);
        e.set_buffer(1, Some(&g.buf), 0);
        e.set_buffer(2, Some(&m.buf), 0);
        e.set_buffer(3, Some(&v.buf), 0);
        e.set_bytes(4, std::mem::size_of::<Args>() as u64, &args as *const Args as *const c_void);
        self.grid1(n, 256);
    }

    /// AdamW over the sub-range [off, off+n) of p/g/m/v (same offset in each).
    #[allow(clippy::too_many_arguments)]
    pub fn adamw_at(
        &self,
        p: &GBuf,
        g: &GBuf,
        m: &GBuf,
        v: &GBuf,
        off: usize,
        n: usize,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
        step: u32,
        gscale: f32,
    ) {
        assert!(p.len >= off + n && g.len >= off + n && m.len >= off + n && v.len >= off + n);
        #[repr(C)]
        struct Args {
            n: u32,
            lr: f32,
            beta1: f32,
            beta2: f32,
            eps: f32,
            wd: f32,
            bc1: f32,
            bc2: f32,
            gscale: f32,
        }
        let t = step.max(1) as f64;
        let args = Args {
            n: n as u32,
            lr,
            beta1,
            beta2,
            eps,
            wd,
            bc1: (1.0 / (1.0 - (beta1 as f64).powf(t))) as f32,
            bc2: (1.0 / (1.0 - (beta2 as f64).powf(t))) as f32,
            gscale,
        };
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.adamw);
        e.set_buffer(0, Some(&p.buf), (off * 4) as u64);
        e.set_buffer(1, Some(&g.buf), (off * 4) as u64);
        e.set_buffer(2, Some(&m.buf), (off * 4) as u64);
        e.set_buffer(3, Some(&v.buf), (off * 4) as u64);
        e.set_bytes(4, std::mem::size_of::<Args>() as u64, &args as *const Args as *const c_void);
        self.grid1(n, 256);
    }

    /// Sum of squares of x[off..off+n] into partial[part_off..]; returns groups.
    pub fn sumsq_at(&self, x: &GBuf, off: usize, n: usize, partial: &GBuf, part_off: usize) -> usize {
        let groups = n.div_ceil(256).clamp(1, 4096).min(partial.len - part_off);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.sumsq);
        e.set_buffer(0, Some(&x.buf), (off * 4) as u64);
        e.set_buffer(1, Some(&partial.buf), (part_off * 4) as u64);
        self.set_u32(2, n as u32);
        e.dispatch_thread_groups(MTLSize::new(groups as u64, 1, 1), MTLSize::new(256, 1, 1));
        groups
    }

    /// Partial sums of squares of x[..n] into `partial` (one per
    /// threadgroup; the caller sums the first `groups` on the host).
    /// Returns the number of partials written.
    pub fn sumsq(&self, x: &GBuf, n: usize, partial: &GBuf) -> usize {
        let groups = n.div_ceil(256).clamp(1, 4096).min(partial.len);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.sumsq);
        e.set_buffer(0, Some(&x.buf), 0);
        e.set_buffer(1, Some(&partial.buf), 0);
        self.set_u32(2, n as u32);
        e.dispatch_thread_groups(MTLSize::new(groups as u64, 1, 1), MTLSize::new(256, 1, 1));
        groups
    }

    /// RMSNorm forward over `rows` rows of width d: y, inv.
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_fwd(&self, x: &GBuf, w: &GBuf, y: &GBuf, inv: &GBuf, rows: usize, d: usize, eps: f32) {
        assert!(x.len >= rows * d && y.len >= rows * d && w.len >= d && inv.len >= rows);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.rms_fwd);
        e.set_buffer(0, Some(&x.buf), 0);
        e.set_buffer(1, Some(&w.buf), 0);
        e.set_buffer(2, Some(&y.buf), 0);
        e.set_buffer(3, Some(&inv.buf), 0);
        self.set_u32(4, d as u32);
        self.set_f32(5, eps);
        e.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(128, 1, 1));
    }

    /// RMSNorm backward: dx = beta·dx + ∂/∂x, and dw += Σ_rows.
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_bwd(
        &self,
        x: &GBuf,
        w: &GBuf,
        dy: &GBuf,
        inv: &GBuf,
        dx: &GBuf,
        beta: f32,
        dw: &GBuf,
        rows: usize,
        d: usize,
    ) {
        assert!(x.len >= rows * d && dy.len >= rows * d && dx.len >= rows * d && dw.len >= d);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.rms_bwd_dx);
        e.set_buffer(0, Some(&x.buf), 0);
        e.set_buffer(1, Some(&w.buf), 0);
        e.set_buffer(2, Some(&dy.buf), 0);
        e.set_buffer(3, Some(&inv.buf), 0);
        e.set_buffer(4, Some(&dx.buf), 0);
        self.set_u32(5, d as u32);
        self.set_f32(6, beta);
        e.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(128, 1, 1));
        e.set_compute_pipeline_state(&self.c.rms_dw);
        e.set_buffer(0, Some(&x.buf), 0);
        e.set_buffer(1, Some(&dy.buf), 0);
        e.set_buffer(2, Some(&inv.buf), 0);
        e.set_buffer(3, Some(&dw.buf), 0);
        self.set_u32(4, d as u32);
        self.set_u32(5, rows as u32);
        self.grid1(d, 128);
    }

    /// h = silu(gate)·up over n.
    pub fn swiglu_fwd(&self, gate: &GBuf, up: &GBuf, h: &GBuf, n: usize) {
        assert!(gate.len >= n && up.len >= n && h.len >= n);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.swiglu_fwd);
        e.set_buffer(0, Some(&gate.buf), 0);
        e.set_buffer(1, Some(&up.buf), 0);
        e.set_buffer(2, Some(&h.buf), 0);
        self.set_u32(3, n as u32);
        self.grid1(n, 256);
    }

    /// dgate, dup from dh.
    pub fn swiglu_bwd(&self, gate: &GBuf, up: &GBuf, dh: &GBuf, dgate: &GBuf, dup: &GBuf, n: usize) {
        assert!(gate.len >= n && up.len >= n && dh.len >= n && dgate.len >= n && dup.len >= n);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.swiglu_bwd);
        e.set_buffer(0, Some(&gate.buf), 0);
        e.set_buffer(1, Some(&up.buf), 0);
        e.set_buffer(2, Some(&dh.buf), 0);
        e.set_buffer(3, Some(&dgate.buf), 0);
        e.set_buffer(4, Some(&dup.buf), 0);
        self.set_u32(5, n as u32);
        self.grid1(n, 256);
    }

    /// out[row,:] = E[tok[row],:] for `rows` rows of width d.
    pub fn embed_gather(&self, e_tab: &GBuf, tok: &GBuf, out: &GBuf, rows: usize, d: usize) {
        assert!(tok.len >= rows && out.len >= rows * d);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.embed_gather);
        e.set_buffer(0, Some(&e_tab.buf), 0);
        e.set_buffer(1, Some(&tok.buf), 0);
        e.set_buffer(2, Some(&out.buf), 0);
        self.set_u32(3, d as u32);
        let tgx = 64u64.min(d as u64).max(1);
        e.dispatch_thread_groups(
            MTLSize::new((d as u64).div_ceil(tgx), rows as u64, 1),
            MTLSize::new(tgx, 1, 1),
        );
    }

    /// Fused softmax-CE over `rows` rows of `n` logits: loss[row], and
    /// logits ← (p − onehot)·scale in place.
    pub fn softmax_ce(&self, logits: &GBuf, target: &GBuf, loss: &GBuf, rows: usize, n: usize, scale: f32) {
        assert!(logits.len >= rows * n && target.len >= rows && loss.len >= rows);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.softmax_ce);
        e.set_buffer(0, Some(&logits.buf), 0);
        e.set_buffer(1, Some(&target.buf), 0);
        e.set_buffer(2, Some(&loss.buf), 0);
        self.set_u32(3, n as u32);
        self.set_f32(4, scale);
        e.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    // ---------------- hybrid_k mixer (chunk C = 64) ----------------

    fn hk_args(&self, idx: u64, d: &HkDims) {
        #[repr(C)]
        struct Args {
            b: u32,
            t: u32,
            nh: u32,
            nph: u32,
            dv: u32,
        }
        let a = Args { b: d.b as u32, t: d.t as u32, nh: d.nh as u32, nph: d.nph as u32, dv: d.dv as u32 };
        self.enc.set_bytes(idx, std::mem::size_of::<Args>() as u64, &a as *const Args as *const c_void);
    }

    /// Forward of one hybrid_k mixer layer (after the projections):
    /// inputs thq/thk [B·T, nh·nph], v [B·T, nh·dv], kappa [B·T, nh]; scratch
    /// phq/phk [B·T, nh·2nph], kv [B·T, nh·dv], states [B·nh·(T/64+1)·2nph·dv];
    /// output out [B·T, nh·dv]. `pow` from `hk_pow_table`.
    pub fn hk_forward(&self, d: &HkDims, w: &HkWork<'_>) {
        hk_check(d, w);
        let e = &self.enc;
        let rows = d.b * d.t;
        // φ tables
        for (th, ph) in [(w.thq, w.phq), (w.thk, w.phk)] {
            e.set_compute_pipeline_state(&self.c.hk_phi);
            e.set_buffer(0, Some(&th.buf), 0);
            e.set_buffer(1, Some(&ph.buf), 0);
            self.hk_args(2, d);
            self.grid1(rows * d.nh * d.nph, 256);
        }
        // kv = κ⊙v
        e.set_compute_pipeline_state(&self.c.hk_kv);
        e.set_buffer(0, Some(&w.v.buf), 0);
        e.set_buffer(1, Some(&w.kappa.buf), 0);
        e.set_buffer(2, Some(&w.kv.buf), 0);
        self.hk_args(3, d);
        self.grid1(rows * d.nh * d.dv, 256);
        // chunk-boundary states
        e.set_compute_pipeline_state(&self.c.hk_states_fwd);
        e.set_buffer(0, Some(&w.phk.buf), 0);
        e.set_buffer(1, Some(&w.kv.buf), 0);
        e.set_buffer(2, Some(&w.pow.buf), (w.pow_off * 4) as u64);
        e.set_buffer(3, Some(&w.states.buf), 0);
        self.hk_args(4, d);
        e.dispatch_thread_groups(MTLSize::new((d.b * d.nh) as u64, 1, 1), MTLSize::new((d.dv * (2 * d.nph).div_ceil(16)) as u64, 1, 1));
        // per-chunk outputs
        let nchunks = d.t / 64;
        e.set_compute_pipeline_state(&self.c.hk_chunk_fwd);
        e.set_buffer(0, Some(&w.phq.buf), 0);
        e.set_buffer(1, Some(&w.phk.buf), 0);
        e.set_buffer(2, Some(&w.kv.buf), 0);
        e.set_buffer(3, Some(&w.pow.buf), (w.pow_off * 4) as u64);
        e.set_buffer(4, Some(&w.states.buf), 0);
        e.set_buffer(5, Some(&w.out.buf), 0);
        self.hk_args(6, d);
        e.dispatch_thread_groups(
            MTLSize::new((d.b * d.nh * nchunks) as u64, 1, 1),
            MTLSize::new(d.dv.max(128) as u64, 1, 1),
        );
    }

    /// Backward of one hybrid_k mixer layer given dout [B·T, nh·dv].
    /// Requires the forward's phq/phk/kv/states. Writes dthq/dthk (beta-
    /// accumulated), dv, dkappa; scratch dstates (same size as states),
    /// dkv [B·T, nh·dv], dphq/dphk [B·T, nh·2nph].
    pub fn hk_backward(&self, d: &HkDims, w: &HkWork<'_>, g: &HkGrads<'_>, beta_th: f32) {
        hk_check(d, w);
        let e = &self.enc;
        let rows = d.b * d.t;
        let nchunks = d.t / 64;
        assert!(g.dstates.len >= w.states.len && g.dkv.len >= rows * d.nh * d.dv);
        assert!(g.dphq.len >= rows * d.nh * 2 * d.nph && g.dphk.len >= rows * d.nh * 2 * d.nph);
        assert!(g.dout.len >= rows * d.nh * d.dv && g.dv.len >= rows * d.nh * d.dv);
        assert!(g.dthq.len >= rows * d.nh * d.nph && g.dthk.len >= rows * d.nh * d.nph && g.dkappa.len >= rows * d.nh);
        // reverse state-gradient scan
        e.set_compute_pipeline_state(&self.c.hk_dstates_bwd);
        e.set_buffer(0, Some(&w.phq.buf), 0);
        e.set_buffer(1, Some(&g.dout.buf), 0);
        e.set_buffer(2, Some(&w.pow.buf), (w.pow_off * 4) as u64);
        e.set_buffer(3, Some(&g.dstates.buf), 0);
        self.hk_args(4, d);
        e.dispatch_thread_groups(MTLSize::new((d.b * d.nh) as u64, 1, 1), MTLSize::new((d.dv * (2 * d.nph).div_ceil(16)) as u64, 1, 1));
        // per-chunk gradients
        e.set_compute_pipeline_state(&self.c.hk_chunk_bwd);
        e.set_buffer(0, Some(&w.phq.buf), 0);
        e.set_buffer(1, Some(&w.phk.buf), 0);
        e.set_buffer(2, Some(&w.kv.buf), 0);
        e.set_buffer(3, Some(&w.pow.buf), (w.pow_off * 4) as u64);
        e.set_buffer(4, Some(&w.states.buf), 0);
        e.set_buffer(5, Some(&g.dstates.buf), 0);
        e.set_buffer(6, Some(&g.dout.buf), 0);
        e.set_buffer(7, Some(&g.dkv.buf), 0);
        e.set_buffer(8, Some(&g.dphq.buf), 0);
        e.set_buffer(9, Some(&g.dphk.buf), 0);
        self.hk_args(10, d);
        e.dispatch_thread_groups(
            MTLSize::new((d.b * d.nh * nchunks) as u64, 1, 1),
            MTLSize::new(d.dv.max(128) as u64, 1, 1),
        );
        // dv, dκ from dkv
        e.set_compute_pipeline_state(&self.c.hk_dkv_split);
        e.set_buffer(0, Some(&w.v.buf), 0);
        e.set_buffer(1, Some(&w.kappa.buf), 0);
        e.set_buffer(2, Some(&g.dkv.buf), 0);
        e.set_buffer(3, Some(&g.dv.buf), 0);
        e.set_buffer(4, Some(&g.dkappa.buf), 0);
        self.hk_args(5, d);
        self.grid1(rows * d.nh, 128);
        // dθ from dφ
        for (th, dph, dth) in [(w.thq, g.dphq, g.dthq), (w.thk, g.dphk, g.dthk)] {
            e.set_compute_pipeline_state(&self.c.hk_dtheta);
            e.set_buffer(0, Some(&th.buf), 0);
            e.set_buffer(1, Some(&dph.buf), 0);
            e.set_buffer(2, Some(&dth.buf), 0);
            self.hk_args(3, d);
            self.set_f32(4, beta_th);
            self.grid1(rows * d.nh * d.nph, 256);
        }
    }

    // ---------------- anchor attention companions ----------------

    /// RoPE in place on x[rows, nheads·hd] (neox halves), position = row % t.
    /// `inverse` applies the transpose rotation (the backward).
    #[allow(clippy::too_many_arguments)]
    pub fn rope(&self, x: &GBuf, x_off: usize, rows: usize, t: usize, nheads: usize, hd: usize, base: f32, inverse: bool) {
        assert!(hd % 2 == 0 && x.len >= x_off + rows * nheads * hd);
        #[repr(C)]
        struct Args {
            t: u32,
            nheads: u32,
            hd: u32,
            base: f32,
            sign: f32,
        }
        let a = Args { t: t as u32, nheads: nheads as u32, hd: hd as u32, base, sign: if inverse { -1.0 } else { 1.0 } };
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.rope);
        e.set_buffer(0, Some(&x.buf), (x_off * 4) as u64);
        e.set_bytes(1, std::mem::size_of::<Args>() as u64, &a as *const Args as *const c_void);
        self.grid1(rows * nheads * (hd / 2), 256);
    }

    /// Causal row softmax in place on an [t,t] block at `off` (row stride t).
    pub fn causal_softmax(&self, s: &GBuf, off: usize, t: usize) {
        assert!(s.len >= off + t * t);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.causal_softmax);
        e.set_buffer(0, Some(&s.buf), (off * 4) as u64);
        self.set_u32(1, t as u32);
        e.dispatch_thread_groups(MTLSize::new(t as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// dS = P ⊙ (dP − rowsum(P⊙dP)) in place on dP ([t,t] blocks at offsets).
    pub fn softmax_bwd(&self, p: &GBuf, p_off: usize, dp: &GBuf, dp_off: usize, t: usize) {
        assert!(p.len >= p_off + t * t && dp.len >= dp_off + t * t);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.softmax_bwd);
        e.set_buffer(0, Some(&p.buf), (p_off * 4) as u64);
        e.set_buffer(1, Some(&dp.buf), (dp_off * 4) as u64);
        self.set_u32(2, t as u32);
        e.dispatch_thread_groups(MTLSize::new(t as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// y = σ(x + bias) over n.
    pub fn sigmoid_fwd(&self, x: &GBuf, y: &GBuf, bias: f32, n: usize) {
        assert!(x.len >= n && y.len >= n);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.sigmoid_fwd);
        e.set_buffer(0, Some(&x.buf), 0);
        e.set_buffer(1, Some(&y.buf), 0);
        self.set_f32(2, bias);
        self.set_u32(3, n as u32);
        self.grid1(n, 256);
    }

    /// dx = dy·y·(1−y) over n.
    pub fn sigmoid_bwd(&self, y: &GBuf, dy: &GBuf, dx: &GBuf, n: usize) {
        assert!(y.len >= n && dy.len >= n && dx.len >= n);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.sigmoid_bwd);
        e.set_buffer(0, Some(&y.buf), 0);
        e.set_buffer(1, Some(&dy.buf), 0);
        e.set_buffer(2, Some(&dx.buf), 0);
        self.set_u32(3, n as u32);
        self.grid1(n, 256);
    }

    /// dE[tok[row],:] += dx[row,:] (atomic).
    pub fn embed_scatter_add(&self, de: &GBuf, de_off: usize, tok: &GBuf, dx: &GBuf, rows: usize, d: usize) {
        assert!(tok.len >= rows && dx.len >= rows * d);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.embed_scatter_add);
        e.set_buffer(0, Some(&de.buf), (de_off * 4) as u64);
        e.set_buffer(1, Some(&tok.buf), 0);
        e.set_buffer(2, Some(&dx.buf), 0);
        self.set_u32(3, d as u32);
        let tgx = 64u64.min(d as u64).max(1);
        e.dispatch_thread_groups(
            MTLSize::new((d as u64).div_ceil(tgx), rows as u64, 1),
            MTLSize::new(tgx, 1, 1),
        );
    }

    /// dst[dst_off..+n] = src[src_off..+n].
    pub fn copy(&self, src: &GBuf, src_off: usize, dst: &GBuf, dst_off: usize, n: usize) {
        assert!(src.len >= src_off + n && dst.len >= dst_off + n);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.copy);
        e.set_buffer(0, Some(&src.buf), (src_off * 4) as u64);
        e.set_buffer(1, Some(&dst.buf), (dst_off * 4) as u64);
        self.set_u32(2, n as u32);
        self.grid1(n, 256);
    }

    /// Embedding gather with a table offset (the tied table lives inside
    /// the parameter arena).
    pub fn embed_gather_at(&self, e_tab: &GBuf, e_off: usize, tok: &GBuf, out: &GBuf, rows: usize, d: usize) {
        assert!(tok.len >= rows && out.len >= rows * d);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.embed_gather);
        e.set_buffer(0, Some(&e_tab.buf), (e_off * 4) as u64);
        e.set_buffer(1, Some(&tok.buf), 0);
        e.set_buffer(2, Some(&out.buf), 0);
        self.set_u32(3, d as u32);
        let tgx = 64u64.min(d as u64).max(1);
        e.dispatch_thread_groups(
            MTLSize::new((d as u64).div_ceil(tgx), rows as u64, 1),
            MTLSize::new(tgx, 1, 1),
        );
    }

    /// RMSNorm forward where w lives at an offset inside a bigger buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_fwd_at(&self, x: &GBuf, w: &GBuf, w_off: usize, y: &GBuf, inv: &GBuf, rows: usize, d: usize, eps: f32) {
        assert!(x.len >= rows * d && y.len >= rows * d && w.len >= w_off + d && inv.len >= rows);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.rms_fwd);
        e.set_buffer(0, Some(&x.buf), 0);
        e.set_buffer(1, Some(&w.buf), (w_off * 4) as u64);
        e.set_buffer(2, Some(&y.buf), 0);
        e.set_buffer(3, Some(&inv.buf), 0);
        self.set_u32(4, d as u32);
        self.set_f32(5, eps);
        e.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(128, 1, 1));
    }

    /// Causal depthwise conv1d [b, t, h] with k taps, weights at an offset
    /// inside the parameter arena. Zero left pad per sequence.
    pub fn conv1d_fwd_at(&self, x: &GBuf, w: &GBuf, w_off: usize, y: &GBuf, b: usize, t: usize, h: usize, k: usize) {
        assert!(x.len >= b * t * h && y.len >= b * t * h && w.len >= w_off + h * k);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.conv_fwd);
        e.set_buffer(0, Some(&x.buf), 0);
        e.set_buffer(1, Some(&w.buf), (w_off * 4) as u64);
        e.set_buffer(2, Some(&y.buf), 0);
        self.set_u32x4(3, [b as u32, t as u32, h as u32, k as u32]);
        let n = (b * t * h) as u64;
        e.dispatch_thread_groups(MTLSize::new(n.div_ceil(128), 1, 1), MTLSize::new(128, 1, 1));
    }

    /// dX and dW of the causal depthwise conv. `dx` may alias nothing; dw
    /// accumulates (+=) into the grad arena at `dw_off`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d_bwd_at(&self, x: &GBuf, w: &GBuf, w_off: usize, dy: &GBuf, dx: &GBuf, dw: &GBuf, dw_off: usize, b: usize, t: usize, h: usize, k: usize) {
        assert!(x.len >= b * t * h && dy.len >= b * t * h && dx.len >= b * t * h);
        assert!(w.len >= w_off + h * k && dw.len >= dw_off + h * k);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.conv_dx);
        e.set_buffer(0, Some(&dy.buf), 0);
        e.set_buffer(1, Some(&w.buf), (w_off * 4) as u64);
        e.set_buffer(2, Some(&dx.buf), 0);
        self.set_u32x4(3, [b as u32, t as u32, h as u32, k as u32]);
        let n = (b * t * h) as u64;
        e.dispatch_thread_groups(MTLSize::new(n.div_ceil(128), 1, 1), MTLSize::new(128, 1, 1));
        e.set_compute_pipeline_state(&self.c.conv_dw);
        e.set_buffer(0, Some(&x.buf), 0);
        e.set_buffer(1, Some(&dy.buf), 0);
        e.set_buffer(2, Some(&dw.buf), (dw_off * 4) as u64);
        self.set_u32x4(3, [b as u32, t as u32, h as u32, k as u32]);
        let hk = (h * k) as u64;
        e.dispatch_thread_groups(MTLSize::new(hk.div_ceil(128), 1, 1), MTLSize::new(128, 1, 1));
    }

    /// RMSNorm backward with w and dw at offsets inside bigger buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_bwd_at(
        &self,
        x: &GBuf,
        w: &GBuf,
        w_off: usize,
        dy: &GBuf,
        inv: &GBuf,
        dx: &GBuf,
        beta: f32,
        dw: &GBuf,
        dw_off: usize,
        rows: usize,
        d: usize,
    ) {
        assert!(x.len >= rows * d && dy.len >= rows * d && dx.len >= rows * d && dw.len >= dw_off + d && w.len >= w_off + d);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.rms_bwd_dx);
        e.set_buffer(0, Some(&x.buf), 0);
        e.set_buffer(1, Some(&w.buf), (w_off * 4) as u64);
        e.set_buffer(2, Some(&dy.buf), 0);
        e.set_buffer(3, Some(&inv.buf), 0);
        e.set_buffer(4, Some(&dx.buf), 0);
        self.set_u32(5, d as u32);
        self.set_f32(6, beta);
        e.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(128, 1, 1));
        e.set_compute_pipeline_state(&self.c.rms_dw);
        e.set_buffer(0, Some(&x.buf), 0);
        e.set_buffer(1, Some(&dy.buf), 0);
        e.set_buffer(2, Some(&inv.buf), 0);
        e.set_buffer(3, Some(&dw.buf), (dw_off * 4) as u64);
        self.set_u32(4, d as u32);
        self.set_u32(5, rows as u32);
        self.grid1(d, 128);
    }

    /// Fused softmax-CE where the logits block sits at an offset.
    pub fn softmax_ce_at(&self, logits: &GBuf, l_off: usize, target: &GBuf, t_off: usize, loss: &GBuf, l2_off: usize, rows: usize, n: usize, scale: f32) {
        assert!(logits.len >= l_off + rows * n && target.len >= t_off + rows && loss.len >= l2_off + rows);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.softmax_ce);
        e.set_buffer(0, Some(&logits.buf), (l_off * 4) as u64);
        e.set_buffer(1, Some(&target.buf), (t_off * 4) as u64);
        e.set_buffer(2, Some(&loss.buf), (l2_off * 4) as u64);
        self.set_u32(3, n as u32);
        self.set_f32(4, scale);
        e.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// κ = σ(pre[:, :nh] + bias) from a padded [rows, ld] pre-activation.
    pub fn kappa_fwd(&self, pre: &GBuf, kap: &GBuf, rows: usize, nh: usize, ld: usize, bias: f32) {
        assert!(pre.len >= rows * ld && kap.len >= rows * nh);
        #[repr(C)]
        struct Args { rows: u32, nh: u32, ld: u32, bias: f32 }
        let a = Args { rows: rows as u32, nh: nh as u32, ld: ld as u32, bias };
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.kappa_fwd);
        e.set_buffer(0, Some(&pre.buf), 0);
        e.set_buffer(1, Some(&kap.buf), 0);
        e.set_bytes(2, std::mem::size_of::<Args>() as u64, &a as *const Args as *const c_void);
        self.grid1(rows * nh, 256);
    }

    /// dpre (padded [rows, ld]) from dκ.
    pub fn kappa_bwd(&self, kap: &GBuf, dkap: &GBuf, dpre: &GBuf, rows: usize, nh: usize, ld: usize) {
        assert!(dpre.len >= rows * ld && kap.len >= rows * nh && dkap.len >= rows * nh);
        #[repr(C)]
        struct Args { rows: u32, nh: u32, ld: u32, bias: f32 }
        let a = Args { rows: rows as u32, nh: nh as u32, ld: ld as u32, bias: 0.0 };
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.kappa_bwd);
        e.set_buffer(0, Some(&kap.buf), 0);
        e.set_buffer(1, Some(&dkap.buf), 0);
        e.set_buffer(2, Some(&dpre.buf), 0);
        e.set_bytes(3, std::mem::size_of::<Args>() as u64, &a as *const Args as *const c_void);
        self.grid1(rows * ld, 256);
    }

    // ---------- hybrid_k, GEMM formulation (the fast path) ----------

    fn hk_scale(&self, d: &HkDims, w: &HkWork<'_>, sc: &HkScratch<'_>) {
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.hk_scale);
        e.set_buffer(0, Some(&w.phq.buf), 0);
        e.set_buffer(1, Some(&w.phk.buf), 0);
        e.set_buffer(2, Some(&w.pow.buf), (w.pow_off * 4) as u64);
        e.set_buffer(3, Some(&sc.qt.buf), 0);
        e.set_buffer(4, Some(&sc.kt.buf), 0);
        e.set_buffer(5, Some(&sc.qp.buf), 0);
        e.set_buffer(6, Some(&sc.kh.buf), 0);
        self.hk_args(7, d);
        self.grid1(d.b * d.t * d.nh * 2 * d.nph, 256);
    }

    /// A = causal(Q̃·K̃ᵀ) for every chunk (into sc.a).
    fn hk_intra_a(&self, d: &HkDims, sc: &HkScratch<'_>) {
        let (p2, nch) = (2 * d.nph, d.t / 64);
        let cm = HkScratch::chunk_major(d);
        let sa = [d.nh * nch * 4096, nch * 4096, 4096];
        let bt = GemmBatch { nb: d.b, nh: d.nh, nc: nch, sa: cm, sb: cm, sc: sa };
        self.gemm_ex(Op::N, Op::T, 64, 64, p2, 1.0, sc.qt, 0, p2, sc.kt, 0, p2, 0.0, sc.a, 0, 64, &bt, true);
    }

    /// Forward, GEMM formulation. Same contract as `hk_forward` (φ, kv,
    /// chunk states via the scan kernel), the per-chunk output as three
    /// batched GEMMs. Needs `sc` scratch (see `HkScratch`).
    pub fn hk_forward_gemm(&self, d: &HkDims, w: &HkWork<'_>, sc: &HkScratch<'_>) {
        hk_check(d, w);
        sc.check(d);
        let e = &self.enc;
        let rows = d.b * d.t;
        let (p2, dv, nch) = (2 * d.nph, d.dv, d.t / 64);
        for (th, ph) in [(w.thq, w.phq), (w.thk, w.phk)] {
            e.set_compute_pipeline_state(&self.c.hk_phi);
            e.set_buffer(0, Some(&th.buf), 0);
            e.set_buffer(1, Some(&ph.buf), 0);
            self.hk_args(2, d);
            self.grid1(rows * d.nh * d.nph, 256);
        }
        e.set_compute_pipeline_state(&self.c.hk_kv);
        e.set_buffer(0, Some(&w.v.buf), 0);
        e.set_buffer(1, Some(&w.kappa.buf), 0);
        e.set_buffer(2, Some(&w.kv.buf), 0);
        self.hk_args(3, d);
        self.grid1(rows * d.nh * d.dv, 256);
        self.hk_states_only(d, w);
        self.hk_scale(d, w, sc);
        self.hk_intra_a(d, sc);
        // out = A·KV + Q⁺·S_c
        let rm = HkScratch::row_major(d, dv);
        let st = HkScratch::states(d);
        let sa = [d.nh * nch * 4096, nch * 4096, 4096];
        let cm = HkScratch::chunk_major(d);
        let bt1 = GemmBatch { nb: d.b, nh: d.nh, nc: nch, sa, sb: rm, sc: rm };
        self.gemm_ex(Op::N, Op::N, 64, dv, 64, 1.0, sc.a, 0, 64, w.kv, 0, d.nh * dv, 0.0, w.out, 0, d.nh * dv, &bt1, false);
        let bt2 = GemmBatch { nb: d.b, nh: d.nh, nc: nch, sa: cm, sb: st, sc: rm };
        self.gemm_ex(Op::N, Op::N, 64, dv, p2, 1.0, sc.qp, 0, p2, w.states, 0, dv, 1.0, w.out, 0, d.nh * dv, &bt2, false);
    }

    /// Backward, GEMM formulation. Same contract as `hk_backward`.
    pub fn hk_backward_gemm(&self, d: &HkDims, w: &HkWork<'_>, g: &HkGrads<'_>, sc: &HkScratch<'_>, beta_th: f32) {
        hk_check(d, w);
        sc.check(d);
        let e = &self.enc;
        let rows = d.b * d.t;
        let (p2, dv, nch) = (2 * d.nph, d.dv, d.t / 64);
        assert!(g.dstates.len >= w.states.len && g.dkv.len >= rows * d.nh * dv);
        assert!(g.dphq.len >= rows * d.nh * p2 && g.dphk.len >= rows * d.nh * p2);
        assert!(g.dout.len >= rows * d.nh * dv && g.dv.len >= rows * d.nh * dv);
        assert!(g.dthq.len >= rows * d.nh * d.nph && g.dthk.len >= rows * d.nh * d.nph && g.dkappa.len >= rows * d.nh);
        // reverse state-gradient scan (exact scan kernel)
        self.hk_dstates_only(d, w, g);
        // scaled tables + A (recomputed: cheaper than keeping them per layer)
        self.hk_scale(d, w, sc);
        self.hk_intra_a(d, sc);
        let rm = HkScratch::row_major(d, dv);
        let st = HkScratch::states(d);
        let cm = HkScratch::chunk_major(d);
        let sa = [d.nh * nch * 4096, nch * 4096, 4096];
        let nb = d.b;
        // dKV = Aᵀ·dO + K̂·G_{c+1}
        let bt = GemmBatch { nb, nh: d.nh, nc: nch, sa, sb: rm, sc: rm };
        self.gemm_ex(Op::T, Op::N, 64, dv, 64, 1.0, sc.a, 0, 64, g.dout, 0, d.nh * dv, 0.0, g.dkv, 0, d.nh * dv, &bt, false);
        let bt = GemmBatch { nb, nh: d.nh, nc: nch, sa: cm, sb: st, sc: rm };
        self.gemm_ex(Op::N, Op::N, 64, dv, p2, 1.0, sc.kh, 0, p2, g.dstates, p2 * dv, dv, 1.0, g.dkv, 0, d.nh * dv, &bt, false);
        // dA = causal(dO·KVᵀ)  (overwrites A)
        let bt = GemmBatch { nb, nh: d.nh, nc: nch, sa: rm, sb: rm, sc: sa };
        self.gemm_ex(Op::N, Op::T, 64, 64, dv, 1.0, g.dout, 0, d.nh * dv, w.kv, 0, d.nh * dv, 0.0, sc.a, 0, 64, &bt, true);
        // dK̃ = dAᵀ·Q̃ ; dQ̃ = dA·K̃
        let bt = GemmBatch { nb, nh: d.nh, nc: nch, sa, sb: cm, sc: cm };
        self.gemm_ex(Op::T, Op::N, 64, p2, 64, 1.0, sc.a, 0, 64, sc.qt, 0, p2, 0.0, sc.dkt, 0, p2, &bt, false);
        self.gemm_ex(Op::N, Op::N, 64, p2, 64, 1.0, sc.a, 0, 64, sc.kt, 0, p2, 0.0, sc.dqt, 0, p2, &bt, false);
        // inter terms: dqi = dO·S_cᵀ ; dki = KV·G_{c+1}ᵀ
        let bt = GemmBatch { nb, nh: d.nh, nc: nch, sa: rm, sb: st, sc: cm };
        self.gemm_ex(Op::N, Op::T, 64, p2, dv, 1.0, g.dout, 0, d.nh * dv, w.states, 0, dv, 0.0, sc.dqi, 0, p2, &bt, false);
        self.gemm_ex(Op::N, Op::T, 64, p2, dv, 1.0, w.kv, 0, d.nh * dv, g.dstates, p2 * dv, dv, 0.0, sc.dki, 0, p2, &bt, false);
        // back to dφq/dφk (row-major)
        e.set_compute_pipeline_state(&self.c.hk_unscale);
        e.set_buffer(0, Some(&sc.dqt.buf), 0);
        e.set_buffer(1, Some(&sc.dkt.buf), 0);
        e.set_buffer(2, Some(&sc.dqi.buf), 0);
        e.set_buffer(3, Some(&sc.dki.buf), 0);
        e.set_buffer(4, Some(&w.pow.buf), (w.pow_off * 4) as u64);
        e.set_buffer(5, Some(&g.dphq.buf), 0);
        e.set_buffer(6, Some(&g.dphk.buf), 0);
        self.hk_args(7, d);
        self.grid1(rows * d.nh * p2, 256);
        // dv, dκ from dkv
        e.set_compute_pipeline_state(&self.c.hk_dkv_split);
        e.set_buffer(0, Some(&w.v.buf), 0);
        e.set_buffer(1, Some(&w.kappa.buf), 0);
        e.set_buffer(2, Some(&g.dkv.buf), 0);
        e.set_buffer(3, Some(&g.dv.buf), 0);
        e.set_buffer(4, Some(&g.dkappa.buf), 0);
        self.hk_args(5, d);
        self.grid1(rows * d.nh, 128);
        // dθ from dφ
        for (th, dph, dth) in [(w.thq, g.dphq, g.dthq), (w.thk, g.dphk, g.dthk)] {
            e.set_compute_pipeline_state(&self.c.hk_dtheta);
            e.set_buffer(0, Some(&th.buf), 0);
            e.set_buffer(1, Some(&dph.buf), 0);
            e.set_buffer(2, Some(&dth.buf), 0);
            self.hk_args(3, d);
            self.set_f32(4, beta_th);
            self.grid1(rows * d.nh * d.nph, 256);
        }
    }

    /// Cell-parallel forward chunk-state scan.
    pub fn hk_states_par(&self, d: &HkDims, w: &HkWork<'_>) {
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.hk_states_par);
        e.set_buffer(0, Some(&w.phk.buf), 0);
        e.set_buffer(1, Some(&w.kv.buf), 0);
        e.set_buffer(2, Some(&w.pow.buf), (w.pow_off * 4) as u64);
        e.set_buffer(3, Some(&w.states.buf), 0);
        self.hk_args(4, d);
        let p2 = 2 * d.nph;
        e.dispatch_thread_groups(
            MTLSize::new((d.b * d.nh) as u64, p2.div_ceil(8) as u64, d.dv.div_ceil(32) as u64),
            MTLSize::new(32, 8, 1),
        );
    }
    /// Cell-parallel reverse state-gradient scan.
    pub fn hk_dstates_par(&self, d: &HkDims, w: &HkWork<'_>, g: &HkGrads<'_>) {
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.hk_dstates_par);
        e.set_buffer(0, Some(&w.phq.buf), 0);
        e.set_buffer(1, Some(&g.dout.buf), 0);
        e.set_buffer(2, Some(&w.pow.buf), (w.pow_off * 4) as u64);
        e.set_buffer(3, Some(&g.dstates.buf), 0);
        self.hk_args(4, d);
        let p2 = 2 * d.nph;
        e.dispatch_thread_groups(
            MTLSize::new((d.b * d.nh) as u64, p2.div_ceil(8) as u64, d.dv.div_ceil(32) as u64),
            MTLSize::new(32, 8, 1),
        );
    }
    /// (profiling) only the forward chunk-state scan
    pub fn hk_states_only(&self, d: &HkDims, w: &HkWork<'_>) {
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.hk_states_fwd);
        e.set_buffer(0, Some(&w.phk.buf), 0);
        e.set_buffer(1, Some(&w.kv.buf), 0);
        e.set_buffer(2, Some(&w.pow.buf), (w.pow_off * 4) as u64);
        e.set_buffer(3, Some(&w.states.buf), 0);
        self.hk_args(4, d);
        e.dispatch_thread_groups(MTLSize::new((d.b * d.nh) as u64, 1, 1), MTLSize::new((d.dv * (2 * d.nph).div_ceil(16)) as u64, 1, 1));
    }
    /// (profiling) only the reverse state-gradient scan
    pub fn hk_dstates_only(&self, d: &HkDims, w: &HkWork<'_>, g: &HkGrads<'_>) {
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.hk_dstates_bwd);
        e.set_buffer(0, Some(&w.phq.buf), 0);
        e.set_buffer(1, Some(&g.dout.buf), 0);
        e.set_buffer(2, Some(&w.pow.buf), (w.pow_off * 4) as u64);
        e.set_buffer(3, Some(&g.dstates.buf), 0);
        self.hk_args(4, d);
        e.dispatch_thread_groups(MTLSize::new((d.b * d.nh) as u64, 1, 1), MTLSize::new((d.dv * (2 * d.nph).div_ceil(16)) as u64, 1, 1));
    }

    /// dst[i,:] = src[idx[i],:] (idx < 0 → zero row), `rows` rows of width d.
    pub fn gather_rows(&self, src: &GBuf, idx: &GBuf, dst: &GBuf, rows: usize, d: usize) {
        assert!(idx.len >= rows && dst.len >= rows * d);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.gather_rows);
        e.set_buffer(0, Some(&src.buf), 0);
        e.set_buffer(1, Some(&idx.buf), 0);
        e.set_buffer(2, Some(&dst.buf), 0);
        self.set_u32(3, d as u32);
        let tgx = 64u64.min(d as u64).max(1);
        e.dispatch_thread_groups(MTLSize::new((d as u64).div_ceil(tgx), rows as u64, 1), MTLSize::new(tgx, 1, 1));
    }

    /// dst[idx[i],:] += src[i,:] (idx < 0 skipped; unique indices).
    pub fn scatter_add_rows(&self, dst: &GBuf, idx: &GBuf, src: &GBuf, rows: usize, d: usize) {
        assert!(idx.len >= rows && src.len >= rows * d);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.scatter_add_rows);
        e.set_buffer(0, Some(&dst.buf), 0);
        e.set_buffer(1, Some(&idx.buf), 0);
        e.set_buffer(2, Some(&src.buf), 0);
        self.set_u32(3, d as u32);
        let tgx = 64u64.min(d as u64).max(1);
        e.dispatch_thread_groups(MTLSize::new((d as u64).div_ceil(tgx), rows as u64, 1), MTLSize::new(tgx, 1, 1));
    }

    /// Within-cluster CE with an index map (see the kernel).
    pub fn softmax_ce_idx(&self, logits: &GBuf, idx: &GBuf, tgt: &GBuf, loss2: &GBuf, rows: usize, n: usize, scale: f32) {
        assert!(logits.len >= rows * n && idx.len >= rows);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.softmax_ce_idx);
        e.set_buffer(0, Some(&logits.buf), 0);
        e.set_buffer(1, Some(&idx.buf), 0);
        e.set_buffer(2, Some(&tgt.buf), 0);
        e.set_buffer(3, Some(&loss2.buf), 0);
        self.set_u32(4, n as u32);
        self.set_f32(5, scale);
        e.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Causal row softmax in place on `blocks` consecutive [t,t] blocks at `off`.
    pub fn causal_softmax_blocks(&self, s: &GBuf, off: usize, t: usize, blocks: usize) {
        assert!(s.len >= off + blocks * t * t);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.causal_softmax);
        e.set_buffer(0, Some(&s.buf), (off * 4) as u64);
        self.set_u32(1, t as u32);
        e.dispatch_thread_groups(MTLSize::new(t as u64, blocks as u64, 1), MTLSize::new(256, 1, 1));
    }

    /// dS = P ⊙ (dP − rowsum(P⊙dP)) in place on dP, `blocks` consecutive blocks.
    pub fn softmax_bwd_blocks(&self, p: &GBuf, p_off: usize, dp: &GBuf, dp_off: usize, t: usize, blocks: usize) {
        assert!(p.len >= p_off + blocks * t * t && dp.len >= dp_off + blocks * t * t);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.softmax_bwd);
        e.set_buffer(0, Some(&p.buf), (p_off * 4) as u64);
        e.set_buffer(1, Some(&dp.buf), (dp_off * 4) as u64);
        self.set_u32(2, t as u32);
        e.dispatch_thread_groups(MTLSize::new(t as u64, blocks as u64, 1), MTLSize::new(256, 1, 1));
    }

    /// dst[b·T+t][g·hd+d] = Σ_j src[b][g·group+j][t][d] (head-major → row-major GQA reduce).
    #[allow(clippy::too_many_arguments)]
    pub fn group_sum_heads(&self, src: &GBuf, dst: &GBuf, b: usize, t: usize, qh: usize, kvh: usize, hd: usize) {
        assert!(src.len >= b * qh * t * hd && dst.len >= b * t * kvh * hd && qh % kvh == 0);
        #[repr(C)]
        struct Args {
            b: u32,
            t: u32,
            qh: u32,
            kvh: u32,
            hd: u32,
        }
        let a = Args { b: b as u32, t: t as u32, qh: qh as u32, kvh: kvh as u32, hd: hd as u32 };
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.group_sum);
        e.set_buffer(0, Some(&src.buf), 0);
        e.set_buffer(1, Some(&dst.buf), 0);
        e.set_bytes(2, std::mem::size_of::<Args>() as u64, &a as *const Args as *const c_void);
        self.grid1(b * t * kvh * hd, 256);
    }

    // ---------------- routed experts ----------------

    fn route_args(&self, idx: u64, r: &RouteDims) {
        #[repr(C)]
        struct Args {
            rows: u32,
            h: u32,
            e: u32,
            k: u32,
            cap: u32,
        }
        let a = Args { rows: r.rows as u32, h: r.h as u32, e: r.e as u32, k: r.k as u32, cap: r.cap as u32 };
        self.enc.set_bytes(idx, std::mem::size_of::<Args>() as u64, &a as *const Args as *const c_void);
    }

    /// assign[row] = argmin_e resonance − bias; res[row] = winning resonance.
    #[allow(clippy::too_many_arguments)]
    pub fn route(&self, r: &RouteDims, x: &GBuf, mu: &GBuf, mu_off: usize, u: &GBuf, u_off: usize, bias: &GBuf, bias_off: usize, assign: &GBuf, res: &GBuf) {
        assert!(r.e <= 64 && x.len >= r.rows * r.h && assign.len >= r.rows && res.len >= r.rows);
        assert!(mu.len >= mu_off + r.e * r.h && bias.len >= bias_off + r.e && u.len >= u_off + r.e * r.k * r.h);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.route);
        e.set_buffer(0, Some(&x.buf), 0);
        e.set_buffer(1, Some(&mu.buf), (mu_off * 4) as u64);
        e.set_buffer(2, Some(&u.buf), (u_off * 4) as u64);
        e.set_buffer(3, Some(&bias.buf), (bias_off * 4) as u64);
        e.set_buffer(4, Some(&assign.buf), 0);
        e.set_buffer(5, Some(&res.buf), 0);
        self.route_args(6, r);
        e.dispatch_thread_groups(MTLSize::new(r.rows as u64, 1, 1), MTLSize::new(64, 1, 1));
    }

    /// slot[row] = rank within its expert; count[e].
    pub fn route_group(&self, r: &RouteDims, assign: &GBuf, slot: &GBuf, count: &GBuf, count_off: usize) {
        assert!(slot.len >= r.rows && count.len >= count_off + r.e);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.route_group);
        e.set_buffer(0, Some(&assign.buf), 0);
        e.set_buffer(1, Some(&slot.buf), 0);
        e.set_buffer(2, Some(&count.buf), (count_off * 4) as u64);
        self.route_args(3, r);
        e.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
    }

    /// hg[e][slot] = x[row] (slot < cap); zero the buffer first.
    pub fn moe_gather(&self, r: &RouteDims, x: &GBuf, assign: &GBuf, slot: &GBuf, hg: &GBuf) {
        assert!(hg.len >= r.e * r.cap * r.h && x.len >= r.rows * r.h);
        self.axpby(0.0, hg, 0.0, hg, r.e * r.cap * r.h);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.moe_gather);
        e.set_buffer(0, Some(&x.buf), 0);
        e.set_buffer(1, Some(&assign.buf), 0);
        e.set_buffer(2, Some(&slot.buf), 0);
        e.set_buffer(3, Some(&hg.buf), 0);
        self.route_args(4, r);
        let tgx = 64u64.min(r.h as u64).max(1);
        e.dispatch_thread_groups(MTLSize::new((r.h as u64).div_ceil(tgx), r.rows as u64, 1), MTLSize::new(tgx, 1, 1));
    }

    /// out[row] += yh[e][slot] (slot < cap).
    pub fn moe_scatter_add(&self, r: &RouteDims, out: &GBuf, assign: &GBuf, slot: &GBuf, yh: &GBuf) {
        assert!(yh.len >= r.e * r.cap * r.h && out.len >= r.rows * r.h);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.moe_scatter_add);
        e.set_buffer(0, Some(&out.buf), 0);
        e.set_buffer(1, Some(&assign.buf), 0);
        e.set_buffer(2, Some(&slot.buf), 0);
        e.set_buffer(3, Some(&yh.buf), 0);
        self.route_args(4, r);
        let tgx = 64u64.min(r.h as u64).max(1);
        e.dispatch_thread_groups(MTLSize::new((r.h as u64).div_ceil(tgx), r.rows as u64, 1), MTLSize::new(tgx, 1, 1));
    }

    /// sums[e][j] over the filled slots of expert e.
    pub fn moe_stats(&self, r: &RouteDims, hg: &GBuf, count: &GBuf, count_off: usize, sums: &GBuf, sums_off: usize) {
        assert!(sums.len >= sums_off + r.e * r.h);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.moe_stats);
        e.set_buffer(0, Some(&hg.buf), 0);
        e.set_buffer(1, Some(&count.buf), (count_off * 4) as u64);
        e.set_buffer(2, Some(&sums.buf), (sums_off * 4) as u64);
        self.route_args(3, r);
        self.grid1(r.e * r.h, 128);
    }

    /// μ EMA + balancing bias update from the step's routing statistics.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_update(&self, r: &RouteDims, mu: &GBuf, mu_off: usize, bias: &GBuf, bias_off: usize, sums: &GBuf, sums_off: usize, count: &GBuf, count_off: usize, res: &GBuf, alpha: f32, eta: f32, frozen_below: usize) {
        #[repr(C)]
        struct Args {
            rows: u32,
            h: u32,
            e: u32,
            alpha: f32,
            eta: f32,
            frozen_below: u32,
        }
        let a = Args { rows: r.rows as u32, h: r.h as u32, e: r.e as u32, alpha, eta, frozen_below: frozen_below as u32 };
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.moe_update);
        e.set_buffer(0, Some(&mu.buf), (mu_off * 4) as u64);
        e.set_buffer(1, Some(&bias.buf), (bias_off * 4) as u64);
        e.set_buffer(2, Some(&sums.buf), (sums_off * 4) as u64);
        e.set_buffer(3, Some(&count.buf), (count_off * 4) as u64);
        e.set_buffer(4, Some(&res.buf), 0);
        e.set_bytes(5, std::mem::size_of::<Args>() as u64, &a as *const Args as *const c_void);
        self.grid1(r.e * r.h, 128);
    }

    /// μ_e := x[rows[e]] (data init of the descriptors).
    pub fn moe_init_mu(&self, r: &RouteDims, x: &GBuf, rows: &GBuf, mu: &GBuf, mu_off: usize) {
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.moe_init_mu);
        e.set_buffer(0, Some(&x.buf), 0);
        e.set_buffer(1, Some(&rows.buf), 0);
        e.set_buffer(2, Some(&mu.buf), (mu_off * 4) as u64);
        self.route_args(3, r);
        self.grid1(r.e * r.h, 128);
    }

    /// Per-expert indirect grids from the routing counts: two sets of
    /// {n/64, ceil(min(count,cap)/64), 1} (n1 columns, n2 columns) at
    /// args[off..off + 2·E·3] (u32 elements).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_indirect_args(&self, count: &GBuf, count_off: usize, args: &GBuf, args_off: usize, e_n: usize, cap: usize, n1: usize, n2: usize) {
        assert!(args.len >= args_off + 2 * e_n * 3 && count.len >= count_off + e_n);
        #[repr(C)]
        struct A {
            e: u32,
            cap: u32,
            n1: u32,
            n2: u32,
        }
        let a = A { e: e_n as u32, cap: cap as u32, n1: n1 as u32, n2: n2 as u32 };
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.moe_indirect);
        e.set_buffer(0, Some(&count.buf), (count_off * 4) as u64);
        e.set_buffer(1, Some(&args.buf), (args_off * 4) as u64);
        e.set_bytes(2, std::mem::size_of::<A>() as u64, &a as *const A as *const c_void);
        self.grid1(e_n, 64);
    }

    /// hgc = hg − μ over the filled slots (zeros beyond).
    pub fn moe_center(&self, r: &RouteDims, hg: &GBuf, mu: &GBuf, mu_off: usize, count: &GBuf, count_off: usize, hgc: &GBuf) {
        assert!(hgc.len >= r.e * r.cap * r.h && hg.len >= r.e * r.cap * r.h);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.moe_center);
        e.set_buffer(0, Some(&hg.buf), 0);
        e.set_buffer(1, Some(&mu.buf), (mu_off * 4) as u64);
        e.set_buffer(2, Some(&count.buf), (count_off * 4) as u64);
        e.set_buffer(3, Some(&hgc.buf), 0);
        self.route_args(4, r);
        self.grid1(r.e * r.cap * r.h, 256);
    }

    // ---------------- skill masks ----------------
    fn mask_args(&self, idx: u64, rows: usize, n: usize, hard: bool, tau: f32, l1: f32) {
        #[repr(C)]
        struct A {
            rows: u32,
            n: u32,
            hard: u32,
            tau: f32,
            l1: f32,
        }
        let a = A { rows: rows as u32, n: n as u32, hard: hard as u32, tau, l1 };
        self.enc.set_bytes(idx, std::mem::size_of::<A>() as u64, &a as *const A as *const c_void);
    }
    /// hh[rows, n] *= mask(m[m_off..]) (soft σ or hard 1[σ>τ]).
    pub fn mask_fwd(&self, hh: &GBuf, m: &GBuf, m_off: usize, rows: usize, n: usize, hard: bool, tau: f32) {
        assert!(hh.len >= rows * n && m.len >= m_off + n);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.mask_fwd);
        e.set_buffer(0, Some(&hh.buf), 0);
        e.set_buffer(1, Some(&m.buf), (m_off * 4) as u64);
        self.mask_args(2, rows, n, hard, tau, 0.0);
        self.grid1(rows * n, 256);
    }
    /// dm += column reduction (soft), then dhhm *= mask (in place).
    #[allow(clippy::too_many_arguments)]
    pub fn mask_bwd(&self, dhhm: &GBuf, hh_pre: &GBuf, m: &GBuf, m_off: usize, dm: &GBuf, rows: usize, n: usize, hard: bool, tau: f32, l1: f32) {
        assert!(dhhm.len >= rows * n && hh_pre.len >= rows * n && m.len >= m_off + n && dm.len >= m_off + n);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.mask_bwd_dm);
        e.set_buffer(0, Some(&dhhm.buf), 0);
        e.set_buffer(1, Some(&hh_pre.buf), 0);
        e.set_buffer(2, Some(&m.buf), (m_off * 4) as u64);
        e.set_buffer(3, Some(&dm.buf), (m_off * 4) as u64);
        self.mask_args(4, rows, n, hard, tau, l1);
        self.grid1(n, 128);
        e.set_compute_pipeline_state(&self.c.mask_bwd_dh);
        e.set_buffer(0, Some(&dhhm.buf), 0);
        e.set_buffer(1, Some(&m.buf), (m_off * 4) as u64);
        self.mask_args(2, rows, n, hard, tau, 0.0);
        self.grid1(rows * n, 256);
    }

    /// pow[h][δ][f] = γ^δ from A_log (γ = exp(−exp(A_log))), written at pow_off.
    pub fn hk_pow_from_alog(&self, d: &HkDims, alog: &GBuf, alog_off: usize, pow: &GBuf, pow_off: usize) {
        let p2 = 2 * d.nph;
        assert!(alog.len >= alog_off + d.nh * p2 && pow.len >= pow_off + d.nh * 65 * p2);
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.hk_pow_from_alog);
        e.set_buffer(0, Some(&alog.buf), (alog_off * 4) as u64);
        e.set_buffer(1, Some(&pow.buf), (pow_off * 4) as u64);
        self.hk_args(2, d);
        self.grid1(d.nh * p2, 128);
    }

    /// dA_log += ∂L/∂A_log of one hybrid_k layer. Call right AFTER
    /// `hk_backward_gemm` (sc.a holds dA, sc.dqt/dqi/dki are fresh, the
    /// scaled tables are in sc). `ktp`/`dqtp` are two more chunk-major
    /// scratch buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn hk_dgamma(&self, d: &HkDims, w: &HkWork<'_>, g: &HkGrads<'_>, sc: &HkScratch<'_>, ktp: &GBuf, dqtp: &GBuf, dalog: &GBuf, dalog_off: usize) {
        let rows = d.b * d.t;
        let (p2, nch) = (2 * d.nph, d.t / 64);
        let cl = HkScratch::chunk_len(d);
        assert!(ktp.len >= cl && dqtp.len >= cl && dalog.len >= dalog_off + d.nh * p2);
        let e = &self.enc;
        // K̃′ table
        e.set_compute_pipeline_state(&self.c.hk_scale_ktp);
        e.set_buffer(0, Some(&w.phk.buf), 0);
        e.set_buffer(1, Some(&w.pow.buf), (w.pow_off * 4) as u64);
        e.set_buffer(2, Some(&ktp.buf), 0);
        self.hk_args(3, d);
        self.grid1(rows * d.nh * p2, 256);
        // dqtp = dA·K̃′
        let cm = HkScratch::chunk_major(d);
        let sa = [d.nh * nch * 4096, nch * 4096, 4096];
        let bt = GemmBatch { nb: d.b, nh: d.nh, nc: nch, sa, sb: cm, sc: cm };
        self.gemm_ex(Op::N, Op::N, 64, p2, 64, 1.0, sc.a, 0, 64, ktp, 0, p2, 0.0, dqtp, 0, p2, &bt, false);
        // reduction → dA_log
        e.set_compute_pipeline_state(&self.c.hk_dgamma);
        e.set_buffer(0, Some(&w.phq.buf), 0);
        e.set_buffer(1, Some(&w.pow.buf), (w.pow_off * 4) as u64);
        e.set_buffer(2, Some(&sc.qt.buf), 0);
        e.set_buffer(3, Some(&sc.qp.buf), 0);
        e.set_buffer(4, Some(&sc.kh.buf), 0);
        e.set_buffer(5, Some(&sc.dqt.buf), 0);
        e.set_buffer(6, Some(&dqtp.buf), 0);
        e.set_buffer(7, Some(&sc.dqi.buf), 0);
        e.set_buffer(8, Some(&sc.dki.buf), 0);
        e.set_buffer(9, Some(&w.states.buf), 0);
        e.set_buffer(10, Some(&g.dstates.buf), 0);
        e.set_buffer(11, Some(&dalog.buf), (dalog_off * 4) as u64);
        self.hk_args(12, d);
        e.dispatch_thread_groups(MTLSize::new((d.nh * p2) as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Submit and wait. Returns GPU time in milliseconds (GPUEndTime −
    /// GPUStartTime) — the number the TFLOPS bench reports.
    pub fn commit(self) -> f64 {
        self.enc.end_encoding();
        self.cb.commit();
        self.cb.wait_until_completed();
        gpu_ms(&self.cb)
    }
}

/// GPU time of a completed command buffer in milliseconds — metal-rs
/// does not surface the getters, raw objc does (same as the runtime).
#[allow(unexpected_cfgs)]
fn gpu_ms(cmd: &metal::CommandBufferRef) -> f64 {
    use metal::foreign_types::ForeignTypeRef;
    use metal::objc::{msg_send, sel, sel_impl};
    unsafe {
        let p = cmd.as_ptr();
        let s: f64 = msg_send![p, GPUStartTime];
        let e: f64 = msg_send![p, GPUEndTime];
        (e - s) * 1000.0
    }
}

pub use crate::ops::HkDims;

/// Buffers of one hybrid_k layer's forward (inputs + scratch + output).
pub struct HkWork<'a> {
    pub thq: &'a GBuf,
    pub thk: &'a GBuf,
    pub v: &'a GBuf,
    pub kappa: &'a GBuf,
    /// γ^δ table from `hk_pow_table` (or `hk_pow_from_alog`), at `pow_off`
    pub pow: &'a GBuf,
    pub pow_off: usize,
    pub phq: &'a GBuf,
    pub phk: &'a GBuf,
    pub kv: &'a GBuf,
    pub states: &'a GBuf,
    pub out: &'a GBuf,
}

/// Buffers of one hybrid_k layer's backward.
pub struct HkGrads<'a> {
    pub dout: &'a GBuf,
    pub dstates: &'a GBuf,
    pub dkv: &'a GBuf,
    pub dphq: &'a GBuf,
    pub dphk: &'a GBuf,
    pub dthq: &'a GBuf,
    pub dthk: &'a GBuf,
    pub dv: &'a GBuf,
    pub dkappa: &'a GBuf,
}

fn hk_check(d: &HkDims, w: &HkWork<'_>) {
    assert!(d.t % 64 == 0, "hybrid_k: T must be a multiple of the chunk (64), got {}", d.t);
    assert!(d.nph <= 32 && d.dv <= 128, "hybrid_k kernels: nph ≤ 32, dv ≤ 128");
    let rows = d.b * d.t;
    assert!(w.thq.len >= rows * d.nh * d.nph && w.thk.len >= rows * d.nh * d.nph);
    assert!(w.v.len >= rows * d.nh * d.dv && w.out.len >= rows * d.nh * d.dv && w.kv.len >= rows * d.nh * d.dv);
    assert!(w.kappa.len >= rows * d.nh);
    assert!(w.phq.len >= rows * d.nh * 2 * d.nph && w.phk.len >= rows * d.nh * 2 * d.nph);
    assert!(w.pow.len >= w.pow_off + d.nh * 65 * 2 * d.nph);
    assert!(w.states.len >= d.b * d.nh * (d.t / 64 + 1) * 2 * d.nph * d.dv);
}

/// γ_{h,f}^δ for δ = 0..=64, laid out [nh][65][p2] — the kernels' `pow`.
pub fn hk_pow_table(decay: &[f32], nh: usize, nph: usize) -> Vec<f32> {
    let p2 = 2 * nph;
    let mut t = vec![0.0f32; nh * 65 * p2];
    for h in 0..nh {
        for f in 0..p2 {
            let g = decay[h * p2 + f] as f64;
            for delta in 0..=64usize {
                t[(h * 65 + delta) * p2 + f] = g.powi(delta as i32) as f32;
            }
        }
    }
    t
}

/// Batch decomposition of a GEMM grid's z axis: z = (b·nh + h)·nc + c,
/// operand offsets = b·s[0] + h·s[1] + c·s[2] (elements).
#[derive(Clone, Copy, Debug)]
pub struct GemmBatch {
    pub nb: usize,
    pub nh: usize,
    pub nc: usize,
    pub sa: [usize; 3],
    pub sb: [usize; 3],
    pub sc: [usize; 3],
}

impl GemmBatch {
    pub fn none() -> GemmBatch {
        GemmBatch { nb: 1, nh: 1, nc: 1, sa: [0; 3], sb: [0; 3], sc: [0; 3] }
    }
}

/// Scratch of the GEMM-formulated hybrid_k (shared by all layers; the
/// backward recomputes the tables from phq/phk).
pub struct HkScratch<'a> {
    /// chunk-major [B, nh, T/64, 64, 2nph]
    pub qt: &'a GBuf,
    pub kt: &'a GBuf,
    pub qp: &'a GBuf,
    pub kh: &'a GBuf,
    pub dqt: &'a GBuf,
    pub dkt: &'a GBuf,
    pub dqi: &'a GBuf,
    pub dki: &'a GBuf,
    /// [B, nh, T/64, 64, 64] — A, then dA
    pub a: &'a GBuf,
}

impl HkScratch<'_> {
    pub fn chunk_len(d: &HkDims) -> usize {
        d.b * d.t * d.nh * 2 * d.nph
    }
    pub fn a_len(d: &HkDims) -> usize {
        d.b * d.nh * (d.t / 64) * 4096
    }
    fn check(&self, d: &HkDims) {
        assert!((2 * d.nph) % 64 == 0 && d.dv % 64 == 0, "hybrid_k GEMM path: 2·nph and dv must be multiples of 64 (got {} and {})", 2 * d.nph, d.dv);
        let n = Self::chunk_len(d);
        for b in [self.qt, self.kt, self.qp, self.kh, self.dqt, self.dkt, self.dqi, self.dki] {
            assert!(b.len >= n, "hk scratch: chunk-major buffer too small");
        }
        assert!(self.a.len >= Self::a_len(d), "hk scratch: A too small");
    }
    /// batch strides of a chunk-major [B, nh, nch, 64, p2] buffer
    pub fn chunk_major(d: &HkDims) -> [usize; 3] {
        let (p2, nch) = (2 * d.nph, d.t / 64);
        [d.nh * nch * 64 * p2, nch * 64 * p2, 64 * p2]
    }
    /// batch strides of a row-major [B·T, nh·x] activation, chunk = 64 rows
    pub fn row_major(d: &HkDims, x: usize) -> [usize; 3] {
        [d.t * d.nh * x, x, 64 * d.nh * x]
    }
    /// batch strides of the states buffer [B, nh, nch+1, p2, dv] (S_c)
    pub fn states(d: &HkDims) -> [usize; 3] {
        let (p2, nch) = (2 * d.nph, d.t / 64);
        [d.nh * (nch + 1) * p2 * d.dv, (nch + 1) * p2 * d.dv, p2 * d.dv]
    }
}

/// Routing dimensions of one expert block.
#[derive(Clone, Copy, Debug)]
pub struct RouteDims {
    pub rows: usize,
    pub h: usize,
    pub e: usize,
    /// principal directions per descriptor (0 = distance-to-mean only)
    pub k: usize,
    pub cap: usize,
}

/// GPU-decided GEMM shape (see `Cmd::gemm_dyn`).
#[derive(Clone, Copy)]
pub struct GemmDyn<'a> {
    /// (buffer, BYTE offset) of MTLDispatchThreadgroupsIndirectArguments
    pub indirect: Option<(&'a GBuf, usize)>,
    /// (buffer, element offset) of per-batch u32 row counts
    pub kcount: Option<(&'a GBuf, usize)>,
}

impl GemmDyn<'_> {
    pub fn none() -> GemmDyn<'static> {
        GemmDyn { indirect: None, kcount: None }
    }
}
