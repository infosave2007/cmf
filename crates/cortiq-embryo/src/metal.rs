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
    swiglu_fwd: ComputePipelineState,
    swiglu_bwd: ComputePipelineState,
    embed_gather: ComputePipelineState,
    softmax_ce: ComputePipelineState,
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
        swiglu_fwd: pso("swiglu_fwd_f32")?,
        swiglu_bwd: pso("swiglu_bwd_f32")?,
        embed_gather: pso("embed_gather_f32")?,
        softmax_ce: pso("softmax_ce_f32")?,
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
        assert!(m % 64 == 0 && n % 64 == 0 && k % 32 == 0, "gemm tile alignment: m={m} n={n} k={k}");
        assert!(lda % 4 == 0 && ldb % 4 == 0 && ldc % 4 == 0 && a_off % 4 == 0 && b_off % 4 == 0 && c_off % 4 == 0);
        let (arows, acols) = if ta == Op::N { (m, k) } else { (k, m) };
        let (brows, bcols) = if tb == Op::N { (k, n) } else { (n, k) };
        assert!(lda >= acols && ldb >= bcols && ldc >= n);
        assert!(a_off + (arows - 1) * lda + acols <= a.len, "gemm: A out of range");
        assert!(b_off + (brows - 1) * ldb + bcols <= b.len, "gemm: B out of range");
        assert!(c_off + (m - 1) * ldc + n <= cbuf.len, "gemm: C out of range");
        #[repr(C)]
        struct Args {
            m: u32,
            n: u32,
            k: u32,
            lda: u32,
            ldb: u32,
            ldc: u32,
            alpha: f32,
            beta: f32,
        }
        let args = Args {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            lda: lda as u32,
            ldb: ldb as u32,
            ldc: ldc as u32,
            alpha,
            beta,
        };
        let idx = (ta == Op::T) as usize * 2 + (tb == Op::T) as usize;
        let e = &self.enc;
        e.set_compute_pipeline_state(&self.c.gemm[idx]);
        e.set_buffer(0, Some(&a.buf), (a_off * 4) as u64);
        e.set_buffer(1, Some(&b.buf), (b_off * 4) as u64);
        e.set_buffer(2, Some(&cbuf.buf), (c_off * 4) as u64);
        e.set_bytes(3, std::mem::size_of::<Args>() as u64, &args as *const Args as *const c_void);
        e.dispatch_thread_groups(
            MTLSize::new((n / 64) as u64, (m / 64) as u64, 1),
            MTLSize::new(128, 1, 1),
        );
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
