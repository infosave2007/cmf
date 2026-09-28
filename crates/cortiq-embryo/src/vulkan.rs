//! Native resident wgpu/Vulkan training backend for the Embryo graph.
//!
//! The Metal implementation and this module intentionally expose the same
//! `Ctx`/`Cmd`/`GBuf` seam.  The model graph therefore stays single-source:
//! projections, RMSNorm, hybrid-k/Phase-Delta, route, CE and AdamW are all
//! encoded into one ordered command stream.  `GBuf` owns device-local storage;
//! host mirrors exist only for explicit checkpoint/telemetry readback and are
//! synchronised on demand (never used as an execution fallback).

use std::cell::{RefCell, UnsafeCell};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

pub use crate::ops::{GdnScanDims, HkDims};

const SLOTS: usize = 16;
const PARAM_WORDS: usize = 64;

/// A small bit-packed parameter block.  Keeping operation parameters in a
/// storage buffer avoids per-dispatch uniform-size/alignment limits on Vulkan.
#[derive(Clone)]
struct OpRec {
    slots: [DeviceBuf; SLOTS],
    params: Vec<u32>,
    groups: u32,
    /// 0 = the `main` op switch, 1 = the register-tiled 64×64 GEMM entry
    /// (`gemm64`, a separate pipeline of the same module), 2/3 = the GDN
    /// mixer scan entries (`gdn_fwd` / `gdn_bwd`, 128-lane workgroups).
    entry: u32,
}

#[derive(Clone)]
pub struct DeviceBuf(Arc<BufInner>);

struct BufInner {
    raw: wgpu::Buffer,
    len: usize,
    host: UnsafeCell<Vec<u32>>,
    host_dirty: std::sync::atomic::AtomicBool,
    gpu_dirty: std::sync::atomic::AtomicBool,
    ctx: Arc<CtxInner>,
}

struct TsState {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    stage: wgpu::Buffer,
    period_ns: f32,
}

/// Process-wide device context.  The card is deliberately selected through
/// Vulkan only; a software adapter is rejected unless CMF_GPU_SOFTWARE=1.
struct CtxInner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Fallback pipeline for shader validation and controlled A/B runs.
    pipeline: Arc<wgpu::ComputePipeline>,
    pipeline_layout: Arc<wgpu::PipelineLayout>,
    /// A pipeline compiled with a literal operation id dead-strips the other
    /// ~50 branches in EMBRYO_WGSL.  Pipelines are lazy because most graph
    /// shapes use only a subset of the operation table.
    pipelines: Mutex<HashMap<u32, Arc<wgpu::ComputePipeline>>>,
    /// The register-tiled 64×64×32 GEMM (`gemm64` entry point): 256 lanes,
    /// a 4×4 accumulator micro-tile per lane, 16 KB of shared A/B tiles.
    /// Selected by `CMF_VULKAN_GEMM64=1` for tile-aligned GEMMs.
    pipeline64: Arc<wgpu::ComputePipeline>,
    /// GDN mixer token scan entries (one 128-lane workgroup per (b, head)).
    pipeline_gdn_fwd: Arc<wgpu::ComputePipeline>,
    pipeline_gdn_bwd: Arc<wgpu::ComputePipeline>,
    gemm64: bool,
    specialize: bool,
    tile_gemm: bool,
    profile_ops: bool,
    ts: Option<TsState>,
    ts_lock: Mutex<()>,
    layout: wgpu::BindGroupLayout,
    dummy: wgpu::Buffer,
    adapter_name: String,
}
pub struct Ctx {
    inner: Arc<CtxInner>,
}

static CTX: OnceLock<Result<Ctx, String>> = OnceLock::new();

/// Process-wide Vulkan context.  `None` is an honest capability refusal; the
/// caller must not silently run a CPU implementation under the GPU name.
pub fn ctx() -> Option<&'static Ctx> {
    match CTX.get_or_init(init) {
        Ok(c) => Some(c),
        Err(e) => {
            static ONCE: OnceLock<()> = OnceLock::new();
            ONCE.get_or_init(|| eprintln!("cortiq-embryo: Vulkan unavailable: {e}"));
            None
        }
    }
}

fn init() -> Result<Ctx, String> {
    // On Linux this is the production Vulkan backend. On macOS this file is
    // only the private `vulkan_validation` build (lib.rs): the same WGSL runs
    // through wgpu's Metal backend so its kernel tests can execute locally
    // (the public macOS trainer stays on the native Metal kernels).
    let backends = if cfg!(target_os = "macos") {
        wgpu::Backends::METAL
    } else {
        wgpu::Backends::VULKAN
    };
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        ..Default::default()
    }))
    .map_err(|e| format!("no Vulkan adapter: {e}"))?;
    let info = adapter.get_info();
    if info.device_type == wgpu::DeviceType::Cpu
        && std::env::var("CMF_GPU_SOFTWARE").ok().as_deref() != Some("1")
    {
        return Err(format!(
            "Vulkan adapter {} is software; refusing CPU masquerade (set CMF_GPU_SOFTWARE=1 only for shader validation)",
            info.name
        ));
    }
    let want_ts = std::env::var("CMF_VULKAN_OP_TS").ok().as_deref() == Some("1")
        && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    if std::env::var("CMF_VULKAN_OP_TS").ok().as_deref() == Some("1") && !want_ts {
        eprintln!("cortiq-embryo: Vulkan timestamp queries unavailable on this adapter");
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("cortiq-embryo-vulkan"),
        required_limits: adapter.limits(),
        required_features: if want_ts {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        },
        ..Default::default()
    }))
    .map_err(|e| format!("request Vulkan device: {e}"))?;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("cortiq-embryo-train"),
        source: wgpu::ShaderSource::Wgsl(EMBRYO_WGSL.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("embryo-bindings"),
        entries: &(0..=SLOTS as u32)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: if binding == SLOTS as u32 {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }
                } else {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }
                },
                count: None,
            })
            .collect::<Vec<_>>(),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("embryo-pipeline-layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("embryo-op"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let pipeline64 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("embryo-gemm64"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("gemm64"),
        compilation_options: Default::default(),
        cache: None,
    });
    let entry_pipeline = |name: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(name),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let pipeline_gdn_fwd = entry_pipeline("gdn_fwd");
    let pipeline_gdn_bwd = entry_pipeline("gdn_bwd");
    let gemm64 = std::env::var("CMF_VULKAN_GEMM64")
        .map(|v| v != "0")
        .unwrap_or(false);
    let dummy = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("embryo-dummy"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let specialize = std::env::var("CMF_VULKAN_SPECIALIZE")
        .map(|v| v != "0")
        // A/B measurements on the RTX PRO4000 show the driver already
        // predicates the uniform operation switch efficiently; compiling and
        // switching hundreds of per-op pipelines adds ~2.7% to this graph.
        // Keep specialization available for shader experiments, but retain
        // the faster single-pipeline default in production.
        .unwrap_or(false);
    let tile_gemm = std::env::var("CMF_VULKAN_TILE_GEMM")
        .map(|v| v != "0")
        .unwrap_or(true);
    let profile_ops = std::env::var("CMF_VULKAN_PROFILE_OPS").ok().as_deref() == Some("1");
    // Profiling is intentionally opt-in: it splits the command pass and
    // performs a tiny mapped readback after each commit, so it must never be
    // part of the resident training path.  2048 pairs cover the observed
    // full Embryo graph (829 operations at B1T64).
    let ts = want_ts.then(|| {
        const MAX_PAIRS: u32 = 2048;
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("embryo-op-timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: MAX_PAIRS * 2,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("embryo-op-timestamp-resolve"),
            size: (MAX_PAIRS as u64) * 2 * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("embryo-op-timestamp-stage"),
            size: (MAX_PAIRS as u64) * 2 * 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        TsState {
            query_set,
            resolve,
            stage,
            period_ns: queue.get_timestamp_period(),
        }
    });
    Ok(Ctx {
        inner: Arc::new(CtxInner {
            device,
            queue,
            pipeline: Arc::new(pipeline),
            pipeline_layout: Arc::new(pipeline_layout),
            pipelines: Mutex::new(HashMap::new()),
            pipeline64: Arc::new(pipeline64),
            pipeline_gdn_fwd: Arc::new(pipeline_gdn_fwd),
            pipeline_gdn_bwd: Arc::new(pipeline_gdn_bwd),
            gemm64,
            specialize,
            tile_gemm,
            profile_ops,
            ts,
            ts_lock: Mutex::new(()),
            layout,
            dummy,
            adapter_name: info.name,
        }),
    })
}

impl CtxInner {
    /// Return the operation-specialized pipeline, or the single fallback
    /// pipeline when explicitly disabled for an A/B comparison.  The shader
    /// keeps the exact same code and bindings; only the root operation id is
    /// made a literal so the backend compiler can remove unrelated branches.
    fn pipeline_for(&self, op: u32) -> Arc<wgpu::ComputePipeline> {
        if !self.specialize {
            return self.pipeline.clone();
        }
        if let Some(p) = self.pipelines.lock().unwrap().get(&op).cloned() {
            return p;
        }
        let source = EMBRYO_WGSL.replacen("let op=U(0u);", &format!("let op={op}u;"), 1);
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("cortiq-embryo-vulkan-specialized"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let pipeline = Arc::new(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("cortiq-embryo-vulkan-specialized-op"),
                layout: Some(&self.pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            },
        ));
        // Another encoder thread may have won the race; retaining either
        // equivalent pipeline is safe, and avoids holding the mutex while the
        // driver compiles WGSL.
        self.pipelines
            .lock()
            .unwrap()
            .entry(op)
            .or_insert_with(|| pipeline.clone())
            .clone()
    }

    fn bind_op(&self, op: &OpRec) -> (wgpu::Buffer, wgpu::BindGroup) {
        let pb = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("embryo-op-params"),
                contents: bytemuck::cast_slice(&op.params),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let mut entries = Vec::with_capacity(SLOTS + 1);
        for (i, s) in op.slots.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: s.raw().as_entire_binding(),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: SLOTS as u32,
            resource: pb.as_entire_binding(),
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("embryo-op-bind"),
            layout: &self.layout,
            entries: &entries,
        });
        (pb, bg)
    }
}

/// A resident device buffer with an explicit, demand-driven readback mirror.
pub struct GBuf {
    pub buf: DeviceBuf,
    pub len: usize,
}

impl DeviceBuf {
    fn raw(&self) -> &wgpu::Buffer {
        &self.0.raw
    }
    fn len(&self) -> usize {
        self.0.len
    }
    /// Return the host-mirror pointer after synchronising a device write.
    ///
    /// # Safety
    /// The returned pointer is borrowed from the mirror held by this buffer.
    /// The caller must keep it out of every Vulkan submission and must not
    /// call a method which can read or write the mirror (including
    /// `contents`, `as_slice`, `as_u32_slice`, `as_mut_slice`, `write_from`,
    /// `fill`, or readback) until all dereferences of the pointer are done.
    /// In particular, a borrowed host slice may not survive a GPU dispatch;
    /// the mirror is an explicit, exclusive host/device boundary rather than
    /// shared memory.
    pub unsafe fn contents(&self) -> *mut c_void {
        // The old Metal seam exposes a shared-memory pointer.  On discrete
        // Vulkan this is an explicit readback/writeback boundary instead:
        // synchronise before handing the pointer out, and conservatively mark
        // it dirty so host writes are uploaded before the next dispatch.
        self.sync_host();
        self.0
            .host_dirty
            .store(true, std::sync::atomic::Ordering::Release);
        unsafe { (*self.0.host.get()).as_mut_ptr() as *mut c_void }
    }
    fn sync_host(&self) {
        if !self
            .0
            .gpu_dirty
            .swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            return;
        }
        let nbytes = (self.0.len.max(1) * 4) as u64;
        let stage = self.0.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("embryo-readback"),
            size: nbytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .0
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("embryo-readback-encoder"),
            });
        enc.copy_buffer_to_buffer(self.raw(), 0, &stage, 0, nbytes);
        self.0.ctx.queue.submit(Some(enc.finish()));
        stage.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.0.ctx.device.poll(wgpu::PollType::wait_indefinitely());
        let mapped = stage
            .slice(..)
            .get_mapped_range()
            .expect("readback mapping");
        let src: &[u32] = bytemuck::cast_slice(&mapped);
        unsafe {
            (*self.0.host.get()).copy_from_slice(&src[..self.0.len]);
        }
        drop(mapped);
        stage.unmap();
    }
    fn upload_host(&self) {
        if !self
            .0
            .host_dirty
            .swap(false, std::sync::atomic::Ordering::AcqRel)
        {
            return;
        }
        let host = unsafe { &*self.0.host.get() };
        self.0
            .ctx
            .queue
            .write_buffer(self.raw(), 0, bytemuck::cast_slice(host));
    }
    fn mark_gpu(&self) {
        self.0
            .gpu_dirty
            .store(true, std::sync::atomic::Ordering::Release);
    }
}

impl GBuf {
    /// Match the Metal seam's `GBuf::len()` accessor while retaining the
    /// public field used by Vulkan-side allocation code.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn zeros(c: &Ctx, len: usize) -> GBuf {
        let inner = new_buf(&c.inner, len, None);
        GBuf {
            buf: DeviceBuf(inner),
            len,
        }
    }
    pub fn from_slice(c: &Ctx, x: &[f32]) -> GBuf {
        let words: Vec<u32> = x.iter().map(|v| v.to_bits()).collect();
        let inner = new_buf(&c.inner, x.len(), Some(&words));
        GBuf {
            buf: DeviceBuf(inner),
            len: x.len(),
        }
    }
    pub fn from_u32(c: &Ctx, x: &[u32]) -> GBuf {
        let inner = new_buf(&c.inner, x.len(), Some(x));
        GBuf {
            buf: DeviceBuf(inner),
            len: x.len(),
        }
    }
    /// Borrow the f32 host mirror.
    ///
    /// # Safety
    /// No GPU command or other mirror-mutating/readback operation may occur
    /// for the lifetime of the returned slice. Use `to_vec` for an owned
    /// snapshot when that lifetime cannot be proven locally.
    pub unsafe fn as_slice(&self) -> &[f32] {
        self.buf.sync_host();
        std::slice::from_raw_parts(self.buf.contents() as *const f32, self.len)
    }
    /// Borrow the u32 host mirror; see [`GBuf::as_slice`] for the safety
    /// contract.
    pub unsafe fn as_u32_slice(&self) -> &[u32] {
        self.buf.sync_host();
        std::slice::from_raw_parts(self.buf.contents() as *const u32, self.len)
    }
    #[allow(clippy::mut_from_ref)]
    /// Borrow the mutable f32 host mirror; see [`GBuf::as_slice`] for the
    /// safety contract. Prefer `write_from`/`fill` for ordinary writes.
    pub unsafe fn as_mut_slice(&self) -> &mut [f32] {
        self.buf.sync_host();
        self.buf
            .0
            .host_dirty
            .store(true, std::sync::atomic::Ordering::Release);
        std::slice::from_raw_parts_mut(self.buf.contents() as *mut f32, self.len)
    }
    pub fn read_to(&self, out: &mut [f32]) {
        // `read_to` returns an owned copy, so no mirror borrow escapes this
        // method and callers cannot accidentally hold it across a dispatch.
        unsafe { out.copy_from_slice(&self.as_slice()[..out.len()]) };
    }
    pub fn to_vec(&self) -> Vec<f32> {
        unsafe { self.as_slice().to_vec() }
    }
    pub fn write_from(&self, x: &[f32]) {
        let host = unsafe { &mut *self.buf.0.host.get() };
        for (dst, src) in host[..x.len()].iter_mut().zip(x) {
            *dst = src.to_bits();
        }
        self.buf
            .0
            .host_dirty
            .store(true, std::sync::atomic::Ordering::Release);
        self.buf.upload_host();
    }
    pub fn fill(&self, v: f32) {
        let bits = v.to_bits();
        unsafe {
            (&mut *self.buf.0.host.get())[..self.len].fill(bits);
        }
        self.buf
            .0
            .host_dirty
            .store(true, std::sync::atomic::Ordering::Release);
        self.buf.upload_host();
    }
}

fn new_buf(ctx: &Arc<CtxInner>, len: usize, init: Option<&[u32]>) -> Arc<BufInner> {
    let n = len.max(1);
    let raw = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("embryo-resident"),
        size: (n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut host = vec![0u32; n];
    if let Some(x) = init {
        host[..x.len()].copy_from_slice(x);
    }
    ctx.queue.write_buffer(&raw, 0, bytemuck::cast_slice(&host));
    Arc::new(BufInner {
        raw,
        len,
        host: UnsafeCell::new(host),
        host_dirty: std::sync::atomic::AtomicBool::new(false),
        gpu_dirty: std::sync::atomic::AtomicBool::new(false),
        ctx: Arc::clone(ctx),
    })
}

/// One ordered command stream.  Every operation below records a real compute
/// dispatch; commit submits once, then waits for completion before allowing a
/// host readback.
pub struct Cmd<'a> {
    c: &'a Ctx,
    ops: RefCell<Vec<OpRec>>,
    /// hybrid_k forward scans recorded while set start from checkpoint slot
    /// 0 (state carried across windows) instead of S = 0 (op 29 p[9]).
    hk_carry: std::cell::Cell<bool>,
}

impl<'a> Cmd<'a> {
    pub fn new(c: &'a Ctx) -> Cmd<'a> {
        Cmd {
            c,
            ops: RefCell::new(Vec::new()),
            hk_carry: std::cell::Cell::new(false),
        }
    }
    /// Same contract as the Metal `Cmd::set_hk_carry`.
    pub fn set_hk_carry(&self, on: bool) {
        self.hk_carry.set(on);
    }

    fn rec(&self, op: u32, slots: &[(&GBuf, usize)], mut p: Vec<u32>, n: usize) {
        let mut binds = std::array::from_fn(|_| DeviceBuf(Arc::clone(&self.c.inner_dummy().0)));
        // `inner_dummy` is only a helper for array initialisation; replace all
        // slots with the context's real 4-byte dummy below.
        let dummy = DeviceBuf(Arc::new(BufInner {
            raw: self.c.inner.dummy.clone(),
            len: 1,
            host: UnsafeCell::new(vec![0]),
            host_dirty: std::sync::atomic::AtomicBool::new(false),
            gpu_dirty: std::sync::atomic::AtomicBool::new(false),
            ctx: Arc::clone(&self.c.inner),
        }));
        binds = std::array::from_fn(|_| dummy.clone());
        for (x, i) in slots {
            binds[*i] = x.buf.clone();
        }
        p.resize(PARAM_WORDS, 0);
        p[0] = op;
        let groups = n.div_ceil(256).max(1) as u32;
        p[PARAM_WORDS - 1] = groups.min(65_535);
        let entry = p[PARAM_WORDS - 2];
        p[PARAM_WORDS - 2] = 0;
        self.ops.borrow_mut().push(OpRec {
            slots: binds,
            params: p,
            groups,
            entry,
        });
    }

    fn p_u(v: usize) -> u32 {
        v as u32
    }
    fn p_f(v: f32) -> u32 {
        v.to_bits()
    }

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
        c: &GBuf,
        c_off: usize,
        ldc: usize,
    ) {
        self.gemm_ex(
            ta,
            tb,
            m,
            n,
            k,
            alpha,
            a,
            a_off,
            lda,
            b,
            b_off,
            ldb,
            beta,
            c,
            c_off,
            ldc,
            &GemmBatch::none(),
            false,
        )
    }
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
        c: &GBuf,
        c_off: usize,
        ldc: usize,
        batch: &GemmBatch,
        causal: bool,
    ) {
        self.gemm_dyn(
            ta,
            tb,
            m,
            n,
            k,
            alpha,
            a,
            a_off,
            lda,
            b,
            b_off,
            ldb,
            beta,
            c,
            c_off,
            ldc,
            batch,
            causal,
            &GemmDyn::none(),
        )
    }
    /// One grouped dispatch for `nbatch` GEMMs of different row counts and
    /// arbitrary offsets: batch z multiplies `op(A)[rows_z, k]·op(B)[k, n]`
    /// (or, with `rows_bound_k`, `op(A)[m, rows_z]·op(B)[rows_z, n]`) at the
    /// element offsets `table[toff + 4z .. +3]` on top of `a_off/b_off/c_off`;
    /// `m`/`k` are the maxima over the table. See `GemmDyn::table`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_table(
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
        c: &GBuf,
        c_off: usize,
        ldc: usize,
        table: &GBuf,
        toff: usize,
        nbatch: usize,
        rows_bound_k: bool,
    ) {
        assert!(table.len >= toff + 4 * nbatch);
        let batch = GemmBatch {
            nb: nbatch,
            nh: 1,
            nc: 1,
            sa: [0; 3],
            sb: [0; 3],
            sc: [0; 3],
        };
        self.gemm_dyn_table(
            ta,
            tb,
            m,
            n,
            k,
            alpha,
            a,
            a_off,
            lda,
            b,
            b_off,
            ldb,
            beta,
            c,
            c_off,
            ldc,
            &batch,
            false,
            &GemmDyn::none(),
            Some((table, toff, rows_bound_k)),
        )
    }
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
        c: &GBuf,
        c_off: usize,
        ldc: usize,
        batch: &GemmBatch,
        causal: bool,
        dynamic: &GemmDyn<'_>,
    ) {
        self.gemm_dyn_table(
            ta, tb, m, n, k, alpha, a, a_off, lda, b, b_off, ldb, beta, c, c_off, ldc, batch,
            causal, dynamic, None,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn gemm_dyn_table(
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
        c: &GBuf,
        c_off: usize,
        ldc: usize,
        batch: &GemmBatch,
        causal: bool,
        dynamic: &GemmDyn<'_>,
        table: GemmTable<'_>,
    ) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = (m * n * batch.nb.max(1) * batch.nh.max(1) * batch.nc.max(1)) as u32;
        p[2..18].copy_from_slice(&[
            m as u32,
            n as u32,
            k as u32,
            (ta == Op::T) as u32,
            (tb == Op::T) as u32,
            lda as u32,
            ldb as u32,
            ldc as u32,
            a_off as u32,
            b_off as u32,
            c_off as u32,
            alpha.to_bits(),
            beta.to_bits(),
            batch.nb.max(1) as u32,
            batch.nh.max(1) as u32,
            batch.nc.max(1) as u32,
        ]);
        p[18..27].copy_from_slice(&[
            batch.sa[0] as u32,
            batch.sa[1] as u32,
            batch.sa[2] as u32,
            batch.sb[0] as u32,
            batch.sb[1] as u32,
            batch.sb[2] as u32,
            batch.sc[0] as u32,
            batch.sc[1] as u32,
            batch.sc[2] as u32,
        ]);
        // Match Metal's dynamic GEMM contract. A dynamic K count is rounded
        // up to the 64-row tile and clamped to K in the shader; this is used
        // by expert weight gradients so padded capacity rows never contribute
        // to an update. The indirect expert arguments use the same resident
        // `{n/64, ceil(rows/64), 1}` records as Metal; the scalar dispatch is
        // deliberately fixed, but rows outside the GPU-produced count are
        // guarded in the shader and never reach a weight update.
        p[27] = causal as u32;
        p[28] = dynamic.kcount.is_some() as u32;
        if let Some((_, koff)) = dynamic.kcount {
            p[29] = koff as u32;
        }
        if let Some((indirect, off)) = dynamic.indirect {
            p[30] = 1;
            p[29] = (off / 4) as u32;
        }
        // The tiled path assigns one 16x16 output tile to each workgroup and
        // keeps the original scalar summation order inside each output. This
        // preserves f32 parity while allowing the GPU to reuse A/B tiles.
        // Keep an environment escape hatch for exact A/B measurements and for
        // devices with unusually small workgroup memory.
        let tile_m = m.div_ceil(16);
        let tile_n = n.div_ceil(16);
        let batches = batch.nb.max(1) * batch.nh.max(1) * batch.nc.max(1);
        let tile_groups = tile_m.saturating_mul(tile_n).saturating_mul(batches);
        // Register-tiled 64×64×32 entry for tile-aligned, non-dynamic GEMMs
        // (same ascending-k summation per output as the other paths).
        if self.c.inner.gemm64
            && m % 64 == 0
            && n % 64 == 0
            && k % 32 == 0
            && k > 0
            && dynamic.kcount.is_none()
            && dynamic.indirect.is_none()
            && table.is_none()
        {
            let groups = (m / 64) * (n / 64) * batches;
            p[PARAM_WORDS - 2] = 1;
            self.rec(OP_GEMM, &[(a, 0), (b, 1), (c, 2)], p, groups * 256);
            return;
        }
        let use_tile = self.c.inner.tile_gemm && tile_groups > 0;
        p[31] = use_tile as u32;
        p[32] = tile_n as u32;
        p[33] = tile_m as u32;
        p[34] = tile_groups.min(65_535) as u32;
        let mut slots = vec![(a, 0), (b, 1), (c, 2)];
        if let Some((kcount, _)) = dynamic.kcount {
            slots.push((kcount, 3));
        } else if let Some((indirect, _)) = dynamic.indirect {
            slots.push((indirect, 3));
        } else if let Some((table, toff, kmode)) = table {
            // table-batched: batch z reads (a_off, b_off, c_off, rows) at
            // table[toff + 4z]; rows bound M (kmode = false) or K (kmode = true)
            p[35] = 1;
            p[36] = toff as u32;
            p[37] = kmode as u32;
            slots.push((table, 3));
        }
        self.rec(
            OP_GEMM,
            &slots,
            p,
            if use_tile {
                // `rec` rounds its logical invocation count to 256-lane
                // workgroups.  A tiled GEMM needs one whole workgroup per
                // output tile, not one invocation per tile.
                tile_groups.saturating_mul(256)
            } else {
                m * n * batches
            },
        );
    }

    pub fn axpby(&self, aa: f32, x: &GBuf, bb: f32, y: &GBuf, n: usize) {
        self.rec(
            OP_AXPBY,
            &[(x, 0), (y, 1)],
            vec![0, n as u32, aa.to_bits(), bb.to_bits()],
            n,
        );
    }
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
        self.adamw_at(p, g, m, v, 0, n, lr, beta1, beta2, eps, wd, step, gscale)
    }
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
        let t = step.max(1) as f64;
        let mut q = vec![0; n.min(PARAM_WORDS)];
        q.resize(PARAM_WORDS, 0);
        q[1] = n as u32;
        q[2] = off as u32;
        q[3] = lr.to_bits();
        q[4] = beta1.to_bits();
        q[5] = beta2.to_bits();
        q[6] = eps.to_bits();
        q[7] = wd.to_bits();
        q[8] = ((1.0 / (1.0 - (beta1 as f64).powf(t))) as f32).to_bits();
        q[9] = ((1.0 / (1.0 - (beta2 as f64).powf(t))) as f32).to_bits();
        q[10] = gscale.to_bits();
        self.rec(OP_ADAMW, &[(p, 0), (g, 1), (m, 2), (v, 3)], q, n)
    }
    pub fn sumsq_at(&self, x: &GBuf, off: usize, n: usize, part: &GBuf, poff: usize) -> usize {
        let groups = n.div_ceil(256).clamp(1, 4096).min(part.len - poff);
        let mut p = vec![0; PARAM_WORDS];
        p[1] = n as u32;
        p[2] = off as u32;
        p[3] = groups as u32;
        p[4] = poff as u32;
        self.rec(OP_SUMSQ, &[(x, 0), (part, 1)], p, groups * 256);
        groups
    }
    pub fn sumsq(&self, x: &GBuf, n: usize, part: &GBuf) -> usize {
        self.sumsq_at(x, 0, n, part, 0)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_fwd(
        &self,
        x: &GBuf,
        w: &GBuf,
        y: &GBuf,
        inv: &GBuf,
        rows: usize,
        d: usize,
        eps: f32,
    ) {
        self.rmsnorm_fwd_at(x, w, 0, y, inv, rows, d, eps)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_fwd_at(
        &self,
        x: &GBuf,
        w: &GBuf,
        woff: usize,
        y: &GBuf,
        inv: &GBuf,
        rows: usize,
        d: usize,
        eps: f32,
    ) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = rows as u32;
        p[2] = d as u32;
        p[3] = woff as u32;
        p[4] = eps.to_bits();
        self.rec(
            OP_RMS_FWD,
            &[(x, 0), (w, 1), (y, 2), (inv, 3)],
            p,
            rows * 256,
        )
    }
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
        self.rmsnorm_bwd_at(x, w, 0, dy, inv, dx, beta, dw, 0, rows, d)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_bwd_at(
        &self,
        x: &GBuf,
        w: &GBuf,
        woff: usize,
        dy: &GBuf,
        inv: &GBuf,
        dx: &GBuf,
        beta: f32,
        dw: &GBuf,
        dwoff: usize,
        rows: usize,
        d: usize,
    ) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = rows as u32;
        p[2] = d as u32;
        p[3] = woff as u32;
        p[4] = dwoff as u32;
        p[5] = beta.to_bits();
        self.rec(
            OP_RMS_BWD,
            &[(x, 0), (w, 1), (dy, 2), (inv, 3), (dx, 4), (dw, 5)],
            p,
            rows * d.max(256),
        );
    }
    pub fn swiglu_fwd(&self, g: &GBuf, u: &GBuf, h: &GBuf, n: usize) {
        self.rec(
            OP_SWIGLU_FWD,
            &[(g, 0), (u, 1), (h, 2)],
            vec![0, n as u32],
            n,
        )
    }
    pub fn swiglu_bwd(&self, g: &GBuf, u: &GBuf, dh: &GBuf, dg: &GBuf, du: &GBuf, n: usize) {
        self.rec(
            OP_SWIGLU_BWD,
            &[(g, 0), (u, 1), (dh, 2), (dg, 3), (du, 4)],
            vec![0, n as u32],
            n,
        )
    }
    pub fn embed_gather(&self, e: &GBuf, tok: &GBuf, out: &GBuf, rows: usize, d: usize) {
        self.embed_gather_at(e, 0, tok, out, rows, d)
    }
    pub fn embed_gather_at(
        &self,
        e: &GBuf,
        eoff: usize,
        tok: &GBuf,
        out: &GBuf,
        rows: usize,
        d: usize,
    ) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = rows as u32;
        p[2] = d as u32;
        p[3] = eoff as u32;
        self.rec(OP_EMBED_GATHER, &[(e, 0), (tok, 1), (out, 2)], p, rows * d)
    }
    /// dE[tok[row], :] += dx[row, :] (tied embedding backward), deterministic
    /// and parallel: op 58 builds a per-row chain on the device (bit 31 =
    /// this row is the FIRST occurrence of its token, low bits = the next
    /// row with the same token or 0x7fffffff), then op 10 gives every
    /// (first row, column) pair one owner that walks its chain in ascending
    /// row order starting from the current dE value — the exact f32 sum
    /// order of the old single-invocation loop, so dE is bit-identical to it
    /// (`CMF_VULKAN_SCATTER_SERIAL=1` keeps the old loop for A/B).
    pub fn embed_scatter_add(
        &self,
        de: &GBuf,
        deoff: usize,
        tok: &GBuf,
        dx: &GBuf,
        rows: usize,
        d: usize,
    ) {
        if std::env::var("CMF_VULKAN_SCATTER_SERIAL").ok().as_deref() == Some("1") {
            return self.embed_scatter_add_serial(de, deoff, tok, dx, rows, d);
        }
        let chain = GBuf::zeros(self.c, rows);
        let mut p = vec![0; PARAM_WORDS];
        p[1] = rows as u32;
        self.rec(OP_EMBED_CHAIN, &[(tok, 1), (&chain, 3)], p, rows);
        let mut p = vec![0; PARAM_WORDS];
        p[1] = rows as u32;
        p[2] = d as u32;
        p[3] = deoff as u32;
        p[4] = 0;
        self.rec(
            OP_EMBED_SCATTER,
            &[(de, 0), (tok, 1), (dx, 2), (&chain, 3)],
            p,
            rows * d,
        )
    }
    /// The pre-S6c single-invocation scatter (one thread over rows·d), kept
    /// as the bit-exact reference of `embed_scatter_add`.
    pub fn embed_scatter_add_serial(
        &self,
        de: &GBuf,
        deoff: usize,
        tok: &GBuf,
        dx: &GBuf,
        rows: usize,
        d: usize,
    ) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = rows as u32;
        p[2] = d as u32;
        p[3] = deoff as u32;
        p[4] = 1;
        self.rec(OP_EMBED_SCATTER, &[(de, 0), (tok, 1), (dx, 2)], p, rows * d)
    }
    pub fn softmax_ce(
        &self,
        logits: &GBuf,
        target: &GBuf,
        loss: &GBuf,
        rows: usize,
        n: usize,
        scale: f32,
    ) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = rows as u32;
        p[2] = n as u32;
        p[3] = scale.to_bits();
        self.rec(
            OP_SOFTMAX_CE,
            &[(logits, 0), (target, 1), (loss, 2)],
            p,
            rows,
        )
    }
    pub fn softmax_ce_at(
        &self,
        logits: &GBuf,
        l_off: usize,
        target: &GBuf,
        t_off: usize,
        loss: &GBuf,
        l2_off: usize,
        rows: usize,
        n: usize,
        scale: f32,
    ) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = rows as u32;
        p[2] = n as u32;
        p[3] = scale.to_bits();
        p[4] = l_off as u32;
        p[5] = t_off as u32;
        p[6] = l2_off as u32;
        self.rec(
            OP_SOFTMAX_CE,
            &[(logits, 0), (target, 1), (loss, 2)],
            p,
            rows,
        )
    }
    pub fn softmax_ce_idx(
        &self,
        logits: &GBuf,
        idx: &GBuf,
        tgt: &GBuf,
        loss: &GBuf,
        rows: usize,
        n: usize,
        scale: f32,
    ) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = rows as u32;
        p[2] = n as u32;
        p[3] = scale.to_bits();
        self.rec(
            OP_SOFTMAX_CE_IDX,
            &[(logits, 0), (idx, 1), (tgt, 2), (loss, 3)],
            p,
            rows,
        )
    }
    /// Same contract as the Metal `block_copy`.
    #[allow(clippy::too_many_arguments)]
    pub fn block_copy(
        &self,
        src: &GBuf,
        src_off: usize,
        src_stride: usize,
        dst: &GBuf,
        dst_off: usize,
        dst_stride: usize,
        nblk: usize,
        len: usize,
        mask: Option<(&GBuf, usize)>,
    ) {
        assert!(src.len >= src_off + (nblk - 1) * src_stride + len);
        assert!(dst.len >= dst_off + (nblk - 1) * dst_stride + len);
        let mut p = vec![0; PARAM_WORDS];
        p[1] = nblk as u32;
        p[2] = len as u32;
        p[3] = src_off as u32;
        p[4] = src_stride as u32;
        p[5] = dst_off as u32;
        p[6] = dst_stride as u32;
        p[7] = mask.map_or(1, |m| m.1.max(1)) as u32;
        p[8] = mask.is_some() as u32;
        let mut slots = vec![(src, 0), (dst, 1)];
        if let Some((m, _)) = mask {
            slots.push((m, 2));
        }
        self.rec(OP_BLOCK_COPY, &slots, p, nblk * len)
    }
    pub fn copy(&self, s: &GBuf, soff: usize, d: &GBuf, doff: usize, n: usize) {
        self.rec(
            OP_COPY,
            &[(s, 0), (d, 1)],
            vec![0, n as u32, soff as u32, doff as u32],
            n,
        )
    }
    pub fn slice_cols(&self, s: &GBuf, dst: &GBuf, rows: usize, src_cols: usize, cols: usize) {
        self.rec(
            OP_SLICE,
            &[(s, 0), (dst, 1)],
            vec![0, rows as u32, src_cols as u32, cols as u32],
            rows * cols,
        )
    }
    pub fn pad_cols(&self, s: &GBuf, dst: &GBuf, rows: usize, src_cols: usize, cols: usize) {
        self.rec(
            OP_PAD,
            &[(s, 0), (dst, 1)],
            vec![0, rows as u32, src_cols as u32, cols as u32],
            rows * cols,
        )
    }
    pub fn dot_accum(&self, a: &GBuf, b: &GBuf, d: &GBuf, off: usize, n: usize) {
        self.rec(
            OP_DOT,
            &[(a, 0), (b, 1), (d, 2)],
            vec![0, n as u32, off as u32],
            1,
        )
    }
    pub fn sigmoid_fwd(&self, x: &GBuf, y: &GBuf, bias: f32, n: usize) {
        self.rec(
            OP_SIGMOID_FWD,
            &[(x, 0), (y, 1)],
            vec![0, n as u32, bias.to_bits()],
            n,
        )
    }
    pub fn sigmoid_bwd(&self, y: &GBuf, dy: &GBuf, dx: &GBuf, n: usize) {
        self.rec(
            OP_SIGMOID_BWD,
            &[(y, 0), (dy, 1), (dx, 2)],
            vec![0, n as u32],
            n,
        )
    }
    pub fn rope(
        &self,
        x: &GBuf,
        off: usize,
        rows: usize,
        t: usize,
        heads: usize,
        hd: usize,
        base: f32,
        inverse: bool,
    ) {
        self.rope_at(x, off, rows, t, heads, hd, base, inverse, 0)
    }
    /// `rope` with the positions shifted by `pos0` (same contract as Metal).
    #[allow(clippy::too_many_arguments)]
    /// In-place RoPE over `rows` rows of `[heads, hd]` (position = row % t + pos0).
    /// The WGSL guard is `rows·heads·hd/2` (the dispatch size); until 24.09 it
    /// also multiplied by `t`, which wrapped u32 to 0 at exactly
    /// rows·t·heads·hd = 2^32 (B=8/T=1024 with 8 heads of 128: the S4/S6b
    /// production shape) and silently skipped the rotation of q.
    pub fn rope_at(
        &self,
        x: &GBuf,
        off: usize,
        rows: usize,
        t: usize,
        heads: usize,
        hd: usize,
        base: f32,
        inverse: bool,
        pos0: usize,
    ) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = rows as u32;
        p[2] = t as u32;
        p[3] = heads as u32;
        p[4] = hd as u32;
        p[5] = off as u32;
        p[6] = base.to_bits();
        p[7] = inverse as u32;
        p[8] = pos0 as u32;
        self.rec(OP_ROPE, &[(x, 0)], p, rows * heads * hd / 2)
    }
    pub fn causal_softmax(&self, s: &GBuf, off: usize, t: usize) {
        self.causal_softmax_blocks(s, off, t, 1)
    }
    pub fn causal_softmax_blocks(&self, s: &GBuf, off: usize, t: usize, blocks: usize) {
        self.banded_softmax_blocks(s, off, t, t, 0, 0, 0, blocks)
    }
    /// Band + sink softmax over `[t, ld]` blocks (mirror of the Metal
    /// dispatcher, same predicate: keep columns `0..sink`, zero
    /// `sink..sink_pad`, keep causal column `sink_pad + j` iff `j ≤ row`
    /// and `row − j < window`; `window = 0` = full causal). `(t, 0, 0, 0)`
    /// is the legacy op 19 exactly.
    #[allow(clippy::too_many_arguments)]
    pub fn banded_softmax_blocks(
        &self,
        s: &GBuf,
        off: usize,
        t: usize,
        ld: usize,
        sink: usize,
        sink_pad: usize,
        window: usize,
        blocks: usize,
    ) {
        self.banded_softmax_carry(s, off, t, ld, sink, sink_pad, window, blocks, 0, 1, None)
    }
    /// Same contract as the Metal `banded_softmax_carry`.
    #[allow(clippy::too_many_arguments)]
    pub fn banded_softmax_carry(
        &self,
        s: &GBuf,
        off: usize,
        t: usize,
        ld: usize,
        sink: usize,
        sink_pad: usize,
        window: usize,
        blocks: usize,
        carry_pad: usize,
        bps: usize,
        cmask: Option<&GBuf>,
    ) {
        assert!(ld == sink_pad + carry_pad + t && sink <= sink_pad);
        assert!(carry_pad == 0 || cmask.is_some_and(|m| m.len * bps >= blocks));
        let mut p = vec![0; PARAM_WORDS];
        p[1] = t as u32;
        p[2] = blocks as u32;
        p[3] = off as u32;
        p[4] = ld as u32;
        p[5] = sink as u32;
        p[6] = sink_pad as u32;
        p[7] = window as u32;
        p[8] = carry_pad as u32;
        p[9] = bps.max(1) as u32;
        let mut slots = vec![(s, 0)];
        if let Some(m) = cmask {
            slots.push((m, 1));
        }
        self.rec(OP_CAUSAL, &slots, p, blocks * t)
    }
    pub fn softmax_bwd(&self, pb: &GBuf, poff: usize, dp: &GBuf, dpoff: usize, t: usize) {
        self.softmax_bwd_blocks(pb, poff, dp, dpoff, t, 1)
    }
    pub fn softmax_bwd_blocks(
        &self,
        pb: &GBuf,
        poff: usize,
        dp: &GBuf,
        dpoff: usize,
        t: usize,
        blocks: usize,
    ) {
        self.softmax_bwd_blocks_ld(pb, poff, dp, dpoff, t, t, blocks)
    }
    /// `softmax_bwd_blocks` over `[t, ld]` blocks (row stride `ld`).
    #[allow(clippy::too_many_arguments)]
    pub fn softmax_bwd_blocks_ld(
        &self,
        pb: &GBuf,
        poff: usize,
        dp: &GBuf,
        dpoff: usize,
        t: usize,
        ld: usize,
        blocks: usize,
    ) {
        assert!(ld >= t);
        let mut p = vec![0; PARAM_WORDS];
        p[1] = t as u32;
        p[2] = blocks as u32;
        p[3] = poff as u32;
        p[4] = dpoff as u32;
        p[5] = ld as u32;
        self.rec(OP_SOFTMAX_BWD, &[(pb, 0), (dp, 1)], p, blocks * t)
    }
    /// Bounded-anchor sink gradient fold (mirror of Metal `sink_grad_accum`):
    /// `dst[off + (g·S + s)·hd + d] += alpha · Σ_{b<nb} Σ_{j<group}
    /// src[((b·qh + g·group + j)·SINK_PAD + s)·hd + d]`.
    #[allow(clippy::too_many_arguments)]
    pub fn sink_grad_accum(
        &self,
        src: &GBuf,
        dst: &GBuf,
        dst_off: usize,
        nb: usize,
        kvh: usize,
        group: usize,
        sink: usize,
        hd: usize,
        alpha: f32,
    ) {
        let n = kvh * sink * hd;
        assert!(
            src.len >= nb * kvh * group * crate::model::SINK_PAD * hd && dst.len >= dst_off + n
        );
        let mut p = vec![0; PARAM_WORDS];
        p[1] = n as u32;
        p[2] = sink as u32;
        p[3] = hd as u32;
        p[4] = group as u32;
        p[5] = dst_off as u32;
        p[6] = alpha.to_bits();
        p[7] = crate::model::SINK_PAD as u32;
        p[8] = nb as u32;
        p[9] = (kvh * group) as u32;
        self.rec(OP_SINK_ACCUM, &[(src, 0), (dst, 1)], p, n)
    }
    pub fn gather_rows(&self, src: &GBuf, idx: &GBuf, dst: &GBuf, rows: usize, d: usize) {
        self.rec(
            OP_GATHER,
            &[(src, 0), (idx, 1), (dst, 2)],
            vec![0, rows as u32, d as u32],
            rows * d,
        )
    }
    pub fn scatter_add_rows(&self, dst: &GBuf, idx: &GBuf, src: &GBuf, rows: usize, d: usize) {
        assert!(
            d > 0 && dst.len % d == 0,
            "scatter destination must be row-major"
        );
        self.rec(
            OP_SCATTER,
            &[(dst, 0), (idx, 1), (src, 2)],
            // Keep the stride in p[3], matching the Metal OP22 contract.
            // Putting `d` in p[2] makes the WGSL kernel see a zero row width
            // and silently do no work.
            vec![0, rows as u32, 0, d as u32, (dst.len / d) as u32],
            dst.len,
        )
    }
    pub fn group_sum_heads(
        &self,
        src: &GBuf,
        dst: &GBuf,
        b: usize,
        t: usize,
        qh: usize,
        kvh: usize,
        hd: usize,
    ) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = b as u32;
        p[2] = t as u32;
        p[3] = qh as u32;
        p[4] = kvh as u32;
        p[5] = hd as u32;
        self.rec(OP_GROUP_SUM, &[(src, 0), (dst, 1)], p, b * t * kvh * hd)
    }

    // ---- convolution / kappa helpers ----
    pub fn conv1d_fwd_at(
        &self,
        x: &GBuf,
        w: &GBuf,
        woff: usize,
        y: &GBuf,
        b: usize,
        t: usize,
        h: usize,
        k: usize,
    ) {
        self.conv1d_fwd_hist(x, w, woff, y, None, b, t, h, k)
    }
    /// Same contract as the Metal `conv1d_fwd_hist`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d_fwd_hist(
        &self,
        x: &GBuf,
        w: &GBuf,
        woff: usize,
        y: &GBuf,
        hist: Option<&GBuf>,
        b: usize,
        t: usize,
        h: usize,
        k: usize,
    ) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = b as u32;
        p[2] = t as u32;
        p[3] = h as u32;
        p[4] = k as u32;
        p[5] = woff as u32;
        p[7] = hist.is_some() as u32;
        let mut slots = vec![(x, 0), (w, 1), (y, 2)];
        if let Some(hh) = hist {
            assert!(hh.len >= b * (k - 1) * h);
            slots.push((hh, 3));
        }
        self.rec(OP_CONV_FWD, &slots, p, b * t * h)
    }
    pub fn conv1d_bwd_at(
        &self,
        x: &GBuf,
        w: &GBuf,
        woff: usize,
        dy: &GBuf,
        dx: &GBuf,
        dw: &GBuf,
        dwoff: usize,
        b: usize,
        t: usize,
        h: usize,
        k: usize,
    ) {
        self.conv1d_bwd_hist(x, w, woff, dy, dx, dw, dwoff, None, b, t, h, k)
    }
    /// Same contract as the Metal `conv1d_bwd_hist`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d_bwd_hist(
        &self,
        x: &GBuf,
        w: &GBuf,
        woff: usize,
        dy: &GBuf,
        dx: &GBuf,
        dw: &GBuf,
        dwoff: usize,
        hist: Option<&GBuf>,
        b: usize,
        t: usize,
        h: usize,
        k: usize,
    ) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = b as u32;
        p[2] = t as u32;
        p[3] = h as u32;
        p[4] = k as u32;
        p[5] = woff as u32;
        p[6] = dwoff as u32;
        p[7] = hist.is_some() as u32;
        let mut slots = vec![(x, 0), (w, 1), (dy, 2), (dx, 3), (dw, 4)];
        if let Some(hh) = hist {
            assert!(hh.len >= b * (k - 1) * h);
            slots.push((hh, 5));
        }
        self.rec(OP_CONV_BWD, &slots, p, b * t * h + h * k)
    }
    pub fn kappa_fwd(&self, pre: &GBuf, kap: &GBuf, rows: usize, nh: usize, ld: usize, bias: f32) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = rows as u32;
        p[2] = nh as u32;
        p[3] = ld as u32;
        p[4] = bias.to_bits();
        self.rec(OP_KAPPA_FWD, &[(pre, 0), (kap, 1)], p, rows * nh)
    }
    pub fn kappa_bwd(
        &self,
        kap: &GBuf,
        dkap: &GBuf,
        dpre: &GBuf,
        rows: usize,
        nh: usize,
        ld: usize,
    ) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = rows as u32;
        p[2] = nh as u32;
        p[3] = ld as u32;
        self.rec(
            OP_KAPPA_BWD,
            &[(kap, 0), (dkap, 1), (dpre, 2)],
            p,
            rows * ld,
        )
    }

    // ---- fixed hybrid-k / Phase-Delta graph ----
    pub fn hk_forward(&self, d: &HkDims, w: &HkWork<'_>) {
        self.hk_forward_mode(d, w, false)
    }
    /// Forward, GEMM formulation (port of Metal `hk_forward_gemm`): φ, kv,
    /// the chunk-boundary states by the value-parallel scan (op 29 in
    /// states-only mode), the scaled chunk-major tables (op 53), the
    /// intra-chunk causal A = Q̃·K̃ᵀ and `out = A·KV + Q⁺·S_c` as batched
    /// GEMMs over (b, h, chunk). Same contract as `hk_forward`.
    pub fn hk_forward_gemm(&self, d: &HkDims, w: &HkWork<'_>, sc: &HkScratch<'_>) {
        hk_gemm_check(d, w, sc);
        self.hk_phi(w.thq, w.phq, d, false);
        self.hk_phi(w.thk, w.phk, d, false);
        self.hk_kv(w.v, w.kappa, w.kv, d);
        self.hk_states_only(d, w);
        self.hk_scale(d, w, sc);
        self.hk_intra_a(d, sc);
        let (p2, dv, nch) = (2 * d.nph, d.dv, d.t / 64);
        let rm = HkScratch::row_major(d, dv);
        let st = HkScratch::states(d);
        let sa = [d.nh * nch * 4096, nch * 4096, 4096];
        let cm = HkScratch::chunk_major(d);
        // out = A·KV
        let bt1 = GemmBatch {
            nb: d.b,
            nh: d.nh,
            nc: nch,
            sa,
            sb: rm,
            sc: rm,
        };
        self.gemm_ex(
            Op::N,
            Op::N,
            64,
            dv,
            64,
            1.0,
            sc.a,
            0,
            64,
            w.kv,
            0,
            d.nh * dv,
            0.0,
            w.out,
            0,
            d.nh * dv,
            &bt1,
            false,
        );
        // out += Q⁺·S_c
        let bt2 = GemmBatch {
            nb: d.b,
            nh: d.nh,
            nc: nch,
            sa: cm,
            sb: st,
            sc: rm,
        };
        self.gemm_ex(
            Op::N,
            Op::N,
            64,
            dv,
            p2,
            1.0,
            sc.qp,
            0,
            p2,
            w.states,
            0,
            dv,
            1.0,
            w.out,
            0,
            d.nh * dv,
            &bt2,
            false,
        );
    }
    /// Only the forward chunk-boundary state scan (op 29 with p[8] = 1:
    /// the per-token output is skipped; the GEMM path produces it).
    pub fn hk_states_only(&self, d: &HkDims, w: &HkWork<'_>) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = d.b as u32;
        p[2] = d.t as u32;
        p[3] = d.nh as u32;
        p[4] = d.nph as u32;
        p[5] = d.dv as u32;
        p[6] = w.pow_off as u32;
        p[7] = 0;
        p[8] = 1;
        p[9] = self.hk_carry.get() as u32;
        self.rec(
            OP_HK_FWD,
            &[
                (w.phq, 0),
                (w.phk, 1),
                (w.v, 2),
                (w.kappa, 3),
                (w.pow, 4),
                (w.states, 5),
                (w.out, 6),
                (w.kv, 7),
            ],
            p,
            d.b * d.nh * d.dv,
        )
    }
    pub fn hk_states_par(&self, d: &HkDims, w: &HkWork<'_>) {
        self.hk_states_only(d, w)
    }
    /// Scaled chunk-major tables (op 53, mirror of Metal `hk_scale_f32`):
    /// Q̃ = φq·γ^t, K̃ = φk/γ^t, Q⁺ = φq·γ^{t+1}, K̂ = φk·γ^{63−t}.
    fn hk_scale(&self, d: &HkDims, w: &HkWork<'_>, sc: &HkScratch<'_>) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = d.b as u32;
        p[2] = d.t as u32;
        p[3] = d.nh as u32;
        p[4] = d.nph as u32;
        p[6] = w.pow_off as u32;
        self.rec(
            OP_HK_SCALE,
            &[
                (w.phq, 0),
                (w.phk, 1),
                (w.pow, 2),
                (sc.qt, 3),
                (sc.kt, 4),
                (sc.qp, 5),
                (sc.kh, 6),
            ],
            p,
            d.b * d.t * d.nh * 2 * d.nph,
        )
    }
    /// A = causal(Q̃·K̃ᵀ) for every chunk (into sc.a), batched over (b, h, c).
    fn hk_intra_a(&self, d: &HkDims, sc: &HkScratch<'_>) {
        let (p2, nch) = (2 * d.nph, d.t / 64);
        let cm = HkScratch::chunk_major(d);
        let sa = [d.nh * nch * 4096, nch * 4096, 4096];
        let bt = GemmBatch {
            nb: d.b,
            nh: d.nh,
            nc: nch,
            sa: cm,
            sb: cm,
            sc: sa,
        };
        self.gemm_ex(
            Op::N,
            Op::T,
            64,
            64,
            p2,
            1.0,
            sc.qt,
            0,
            p2,
            sc.kt,
            0,
            p2,
            0.0,
            sc.a,
            0,
            64,
            &bt,
            true,
        );
    }
    fn hk_forward_mode(&self, d: &HkDims, w: &HkWork<'_>, phase: bool) {
        self.hk_phi(w.thq, w.phq, d, phase);
        self.hk_phi(w.thk, w.phk, d, phase);
        self.hk_kv(w.v, w.kappa, w.kv, d);
        let mut p = vec![0; PARAM_WORDS];
        p[1] = d.b as u32;
        p[2] = d.t as u32;
        p[3] = d.nh as u32;
        p[4] = d.nph as u32;
        p[5] = d.dv as u32;
        p[6] = w.pow_off as u32;
        p[7] = phase as u32;
        p[9] = self.hk_carry.get() as u32;
        self.rec(
            OP_HK_FWD,
            &[
                (w.phq, 0),
                (w.phk, 1),
                (w.v, 2),
                (w.kappa, 3),
                (w.pow, 4),
                (w.states, 5),
                (w.out, 6),
                (w.kv, 7),
            ],
            p,
            // Both recurrence laws own one complete (batch, head, value)
            // column.  The legacy path used to dispatch one invocation per
            // token while carrying only a two-row local state, which silently
            // discarded history for T>2.  Keep the whole scan resident just
            // like Phase-Delta; the shader selects the law via p[7].
            d.b * d.nh * d.dv,
        )
    }
    fn hk_phi(&self, th: &GBuf, ph: &GBuf, d: &HkDims, phase: bool) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = (d.b * d.t) as u32;
        p[2] = d.nh as u32;
        p[3] = d.nph as u32;
        p[4] = phase as u32;
        self.rec(OP_HK_PHI, &[(th, 0), (ph, 1)], p, d.b * d.t * d.nh * d.nph)
    }
    fn hk_kv(&self, v: &GBuf, kappa: &GBuf, kv: &GBuf, d: &HkDims) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = (d.b * d.t) as u32;
        p[2] = d.nh as u32;
        p[3] = d.dv as u32;
        self.rec(
            OP_HK_KV,
            &[(v, 0), (kappa, 1), (kv, 2)],
            p,
            d.b * d.t * d.nh * d.dv,
        )
    }
    pub fn phase_delta_forward(&self, d: &HkDims, w: &HkWork<'_>) {
        self.hk_forward_mode(d, w, true)
    }
    pub fn phase_delta_forward_reset(&self, d: &HkDims, w: &HkWork<'_>) {
        self.axpby(0.0, w.states, 0.0, w.states, w.states.len);
        self.phase_delta_forward(d, w)
    }
    pub fn hk_backward(&self, d: &HkDims, w: &HkWork<'_>, g: &HkGrads<'_>, beta: f32) {
        self.hk_backward_mode(d, w, g, beta, false)
    }
    /// Backward, GEMM formulation (port of Metal `hk_backward_gemm`, same
    /// contract as `hk_backward`): the reverse chunk-boundary state-gradient
    /// scan (op 55, value-parallel), the scaled tables + A recomputed, then
    ///   dKV = Aᵀ·dO + K̂·G_{c+1};  dA = causal(dO·KVᵀ);
    ///   dK̃ = dAᵀ·Q̃;  dQ̃ = dA·K̃;  dqi = dO·S_cᵀ;  dki = KV·G_{c+1}ᵀ
    /// as batched GEMMs, `hk_unscale` (op 54) back to dφq/dφk, `hk_dkv_split`
    /// (op 56) → dv/dκ and `hk_dtheta` (op 57, with `beta`) → dθq/dθk.
    pub fn hk_backward_gemm(
        &self,
        d: &HkDims,
        w: &HkWork<'_>,
        g: &HkGrads<'_>,
        sc: &HkScratch<'_>,
        beta: f32,
    ) {
        hk_gemm_check(d, w, sc);
        let rows = d.b * d.t;
        let (p2, dv, nch) = (2 * d.nph, d.dv, d.t / 64);
        assert!(g.dstates.len >= w.states.len && g.dkv.len >= rows * d.nh * dv);
        assert!(g.dphq.len >= rows * d.nh * p2 && g.dphk.len >= rows * d.nh * p2);
        self.hk_dstates_only(d, w, g);
        self.hk_scale(d, w, sc);
        self.hk_intra_a(d, sc);
        let rm = HkScratch::row_major(d, dv);
        let st = HkScratch::states(d);
        let cm = HkScratch::chunk_major(d);
        let sa = [d.nh * nch * 4096, nch * 4096, 4096];
        let nb = d.b;
        // dKV = Aᵀ·dO + K̂·G_{c+1}
        let bt = GemmBatch {
            nb,
            nh: d.nh,
            nc: nch,
            sa,
            sb: rm,
            sc: rm,
        };
        self.gemm_ex(
            Op::T,
            Op::N,
            64,
            dv,
            64,
            1.0,
            sc.a,
            0,
            64,
            g.dout,
            0,
            d.nh * dv,
            0.0,
            g.dkv,
            0,
            d.nh * dv,
            &bt,
            false,
        );
        let bt = GemmBatch {
            nb,
            nh: d.nh,
            nc: nch,
            sa: cm,
            sb: st,
            sc: rm,
        };
        self.gemm_ex(
            Op::N,
            Op::N,
            64,
            dv,
            p2,
            1.0,
            sc.kh,
            0,
            p2,
            g.dstates,
            p2 * dv,
            dv,
            1.0,
            g.dkv,
            0,
            d.nh * dv,
            &bt,
            false,
        );
        // dA = causal(dO·KVᵀ) (overwrites A)
        let bt = GemmBatch {
            nb,
            nh: d.nh,
            nc: nch,
            sa: rm,
            sb: rm,
            sc: sa,
        };
        self.gemm_ex(
            Op::N,
            Op::T,
            64,
            64,
            dv,
            1.0,
            g.dout,
            0,
            d.nh * dv,
            w.kv,
            0,
            d.nh * dv,
            0.0,
            sc.a,
            0,
            64,
            &bt,
            true,
        );
        // dK̃ = dAᵀ·Q̃ ; dQ̃ = dA·K̃
        let bt = GemmBatch {
            nb,
            nh: d.nh,
            nc: nch,
            sa,
            sb: cm,
            sc: cm,
        };
        self.gemm_ex(
            Op::T,
            Op::N,
            64,
            p2,
            64,
            1.0,
            sc.a,
            0,
            64,
            sc.qt,
            0,
            p2,
            0.0,
            sc.dkt,
            0,
            p2,
            &bt,
            false,
        );
        self.gemm_ex(
            Op::N,
            Op::N,
            64,
            p2,
            64,
            1.0,
            sc.a,
            0,
            64,
            sc.kt,
            0,
            p2,
            0.0,
            sc.dqt,
            0,
            p2,
            &bt,
            false,
        );
        // inter terms: dqi = dO·S_cᵀ ; dki = KV·G_{c+1}ᵀ
        let bt = GemmBatch {
            nb,
            nh: d.nh,
            nc: nch,
            sa: rm,
            sb: st,
            sc: cm,
        };
        self.gemm_ex(
            Op::N,
            Op::T,
            64,
            p2,
            dv,
            1.0,
            g.dout,
            0,
            d.nh * dv,
            w.states,
            0,
            dv,
            0.0,
            sc.dqi,
            0,
            p2,
            &bt,
            false,
        );
        self.gemm_ex(
            Op::N,
            Op::T,
            64,
            p2,
            dv,
            1.0,
            w.kv,
            0,
            d.nh * dv,
            g.dstates,
            p2 * dv,
            dv,
            0.0,
            sc.dki,
            0,
            p2,
            &bt,
            false,
        );
        // back to dφq/dφk (row-major)
        let mut p = vec![0; PARAM_WORDS];
        p[1] = d.b as u32;
        p[2] = d.t as u32;
        p[3] = d.nh as u32;
        p[4] = d.nph as u32;
        p[6] = w.pow_off as u32;
        self.rec(
            OP_HK_UNSCALE,
            &[
                (sc.dqt, 0),
                (sc.dkt, 1),
                (sc.dqi, 2),
                (sc.dki, 3),
                (w.pow, 4),
                (g.dphq, 5),
                (g.dphk, 6),
            ],
            p,
            rows * d.nh * p2,
        );
        // dv, dκ from dkv
        let mut p = vec![0; PARAM_WORDS];
        p[1] = d.b as u32;
        p[2] = d.t as u32;
        p[3] = d.nh as u32;
        p[5] = d.dv as u32;
        self.rec(
            OP_HK_DKV_SPLIT,
            &[(w.v, 0), (w.kappa, 1), (g.dkv, 2), (g.dv, 3), (g.dkappa, 4)],
            p,
            rows * d.nh,
        );
        // dθ from dφ
        for (th, dph, dth) in [(w.thq, g.dphq, g.dthq), (w.thk, g.dphk, g.dthk)] {
            let mut p = vec![0; PARAM_WORDS];
            p[1] = rows as u32;
            p[3] = d.nh as u32;
            p[4] = d.nph as u32;
            p[7] = beta.to_bits();
            self.rec(
                OP_HK_DTHETA,
                &[(th, 0), (dph, 1), (dth, 2)],
                p,
                rows * d.nh * d.nph,
            );
        }
    }
    pub fn phase_delta_backward(&self, d: &HkDims, w: &HkWork<'_>, g: &HkGrads<'_>) {
        self.phase_delta_backward_blocks(d, w, g);
        self.phase_delta_fold(d, w, g);
    }
    pub fn phase_delta_backward_blocks(&self, d: &HkDims, w: &HkWork<'_>, g: &HkGrads<'_>) {
        // Own the reset just like the Metal block pass. This keeps direct
        // block/fold callers deterministic as well as the combined wrapper.
        self.phase_delta_zero_grads(d, w, g);
        let (Some(chunk), Some(partial)) = (w.phase_chunk, w.phase_partial) else {
            // Tiny seam-only callers may not allocate the optional scratch;
            // retain the finite legacy fallback rather than dereferencing a
            // missing Phase-Delta arena.
            self.hk_backward_mode(d, w, g, 0.0, true);
            return;
        };
        let nch = d.t.div_ceil(64);
        let nblocks = d.dv.div_ceil(32);
        for chunk_id in (0..nch).rev() {
            let start = chunk_id * 64;
            let clen = (d.t - start).min(64);
            let mut p = vec![0; PARAM_WORDS];
            p[1] = d.b as u32;
            p[2] = d.t as u32;
            p[3] = d.nh as u32;
            p[4] = d.nph as u32;
            p[5] = d.dv as u32;
            p[6] = w.pow_off as u32;
            p[7] = chunk_id as u32;
            p[8] = start as u32;
            p[9] = clen as u32;
            self.rec(
                OP_PHASE_REPLAY,
                &[
                    (w.phk, 0),
                    (w.v, 1),
                    (w.kappa, 2),
                    (w.pow, 3),
                    (w.states, 4),
                    (chunk, 5),
                ],
                p,
                d.b * d.nh * d.dv,
            );
            let mut p = vec![0; PARAM_WORDS];
            p[1] = d.b as u32;
            p[2] = d.t as u32;
            p[3] = d.nh as u32;
            p[4] = d.nph as u32;
            p[5] = d.dv as u32;
            p[6] = w.pow_off as u32;
            p[7] = chunk_id as u32;
            p[8] = start as u32;
            p[9] = clen as u32;
            self.rec(
                OP_PHASE_BWD,
                &[
                    (w.phq, 0),
                    (w.phk, 1),
                    (w.v, 2),
                    (w.kappa, 3),
                    (w.pow, 4),
                    (g.dout, 5),
                    (chunk, 6),
                    (partial, 7),
                    (g.dv, 8),
                    (g.dstates, 9),
                ],
                p,
                d.b * d.nh * nblocks,
            );
        }
    }
    pub fn phase_delta_zero_grads(&self, d: &HkDims, w: &HkWork<'_>, g: &HkGrads<'_>) {
        for x in [g.dthq, g.dthk, g.dv, g.dkappa, g.dstates] {
            self.axpby(0.0, x, 0.0, x, x.len)
        }
        let _ = (d, w);
    }
    pub fn phase_delta_fold(&self, d: &HkDims, w: &HkWork<'_>, g: &HkGrads<'_>) {
        let Some(partial) = w.phase_partial else {
            return;
        };
        let mut p = vec![0; PARAM_WORDS];
        p[1] = d.b as u32;
        p[2] = d.t as u32;
        p[3] = d.nh as u32;
        p[4] = d.nph as u32;
        p[5] = d.dv as u32;
        self.rec(
            OP_PHASE_FOLD,
            &[(partial, 0), (g.dthq, 1), (g.dthk, 2), (g.dkappa, 3)],
            p,
            d.b * d.t * d.nh,
        );
    }
    /// Only the reverse chunk-boundary state-gradient scan (op 55: one
    /// invocation per (b, h, value channel), G ← γ·(G + φq_s·dout_s) over
    /// the tokens in reverse, `dstates[c]` = the gradient of the state
    /// entering chunk c, `dstates[nch]` = 0). Mirror of Metal
    /// `hk_dstates_bwd_f32`.
    pub fn hk_dstates_only(&self, d: &HkDims, w: &HkWork<'_>, g: &HkGrads<'_>) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = d.b as u32;
        p[2] = d.t as u32;
        p[3] = d.nh as u32;
        p[4] = d.nph as u32;
        p[5] = d.dv as u32;
        p[6] = w.pow_off as u32;
        self.rec(
            OP_HK_DSTATES,
            &[(w.phq, 0), (g.dout, 1), (w.pow, 2), (g.dstates, 3)],
            p,
            d.b * d.nh * d.dv,
        )
    }
    pub fn hk_dstates_par(&self, d: &HkDims, w: &HkWork<'_>, g: &HkGrads<'_>) {
        self.hk_dstates_only(d, w, g)
    }

    /// Fold value-parallel legacy-HK partials.  Kept as a small seam method so
    /// the resident fold can be contract-tested independently of the model.
    pub fn hk_fold_fast(
        &self,
        b: usize,
        t: usize,
        nh: usize,
        nph: usize,
        dv: usize,
        chunk: &GBuf,
        dthq: &GBuf,
        dthk: &GBuf,
        dkappa: &GBuf,
        beta: f32,
    ) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = b as u32;
        p[2] = t as u32;
        p[3] = nh as u32;
        p[4] = nph as u32;
        p[5] = dv as u32;
        p[6] = beta.to_bits();
        self.rec(
            OP_HK_FOLD_FAST,
            &[(chunk, 0), (dthq, 1), (dthk, 2), (dkappa, 3)],
            p,
            b * t * nh,
        );
    }

    fn hk_backward_mode(
        &self,
        d: &HkDims,
        w: &HkWork<'_>,
        g: &HkGrads<'_>,
        beta: f32,
        phase: bool,
    ) {
        // Native Vulkan receives an extra resident scratch arena for the
        // legacy path.  Give every value channel its own reverse-scan owner;
        // the old block kernel made one invocation serially walk 32 values
        // and was measured at ~13.6 s of a 13.9 s B1T64 step.  The fast
        // kernel writes disjoint `(token, feature, value)` partials and a
        // separate fold performs the deterministic value reduction.
        if !phase && std::env::var("CMF_VULKAN_HK_FAST").ok().as_deref() != Some("0") {
            if let Some(chunk) = w.phase_chunk {
                let mut p = vec![0; PARAM_WORDS];
                p[1] = d.b as u32;
                p[2] = d.t as u32;
                p[3] = d.nh as u32;
                p[4] = d.nph as u32;
                p[5] = d.dv as u32;
                p[6] = w.pow_off as u32;
                p[7] = beta.to_bits();
                p[10] = 2; // OP30 dispatches the value-parallel variant
                self.rec(
                    OP_HK_BWD,
                    &[
                        (w.phq, 0),
                        (w.phk, 1),
                        (w.v, 2),
                        (w.kappa, 3),
                        (w.pow, 4),
                        (w.states, 5),
                        (g.dout, 6),
                        (g.dthq, 7),
                        (g.dthk, 8),
                        (g.dv, 9),
                        (g.dkappa, 10),
                        (g.dstates, 11),
                        (w.phase_partial.unwrap_or(chunk), 12),
                        (chunk, 13),
                    ],
                    p,
                    d.b * d.nh * d.dv,
                );
                let mut fp = vec![0; PARAM_WORDS];
                fp[1] = d.b as u32;
                fp[2] = d.t as u32;
                fp[3] = d.nh as u32;
                fp[4] = d.nph as u32;
                fp[5] = d.dv as u32;
                fp[6] = beta.to_bits();
                self.rec(
                    OP_HK_FOLD_FAST,
                    &[(chunk, 0), (g.dthq, 1), (g.dthk, 2), (g.dkappa, 3)],
                    fp,
                    d.b * d.t * d.nh,
                );
                return;
            }
        }
        // On native Vulkan the legacy reverse scan uses the same deterministic
        // partial-row/fold scheme as Phase-Delta.  One invocation owns a
        // (batch, head, 32-value) block, so the expensive reverse recurrence
        // is parallel over value channels rather than one invocation scanning
        // every value dimension.  Keep the serial path for seam callers that
        // deliberately omit the optional scratch arena (and for Metal).
        if !phase {
            if let Some(partial) = w.phase_partial {
                let nblocks = d.dv.div_ceil(32);
                let mut p = vec![0; PARAM_WORDS];
                p[1] = d.b as u32;
                p[2] = d.t as u32;
                p[3] = d.nh as u32;
                p[4] = d.nph as u32;
                p[5] = d.dv as u32;
                p[6] = w.pow_off as u32;
                p[7] = beta.to_bits();
                p[8] = 0;
                p[9] = 1; // legacy block mode
                self.rec(
                    OP_HK_BWD,
                    &[
                        (w.phq, 0),
                        (w.phk, 1),
                        (w.v, 2),
                        (w.kappa, 3),
                        (w.pow, 4),
                        (w.states, 5),
                        (g.dout, 6),
                        (g.dthq, 7),
                        (g.dthk, 8),
                        (g.dv, 9),
                        (g.dkappa, 10),
                        (g.dstates, 11),
                        (partial, 12),
                    ],
                    p,
                    d.b * d.nh * nblocks,
                );
                // Fold owns the only writers of dθq/dθk/dκ after all value
                // blocks have completed.  p[7] carries the requested beta
                // for the angle gradients; κ gradients are freshly written.
                let mut fp = vec![0; PARAM_WORDS];
                fp[1] = d.b as u32;
                fp[2] = d.t as u32;
                fp[3] = d.nh as u32;
                fp[4] = d.nph as u32;
                fp[5] = d.dv as u32;
                fp[7] = beta.to_bits();
                self.rec(
                    OP_PHASE_FOLD,
                    &[(partial, 0), (g.dthq, 1), (g.dthk, 2), (g.dkappa, 3)],
                    fp,
                    d.b * d.t * d.nh,
                );
                return;
            }
        }
        let mut p = vec![0; PARAM_WORDS];
        p[1] = d.b as u32;
        p[2] = d.t as u32;
        p[3] = d.nh as u32;
        p[4] = d.nph as u32;
        p[5] = d.dv as u32;
        p[6] = w.pow_off as u32;
        p[7] = beta.to_bits();
        p[8] = phase as u32;
        self.rec(
            OP_HK_BWD,
            &[
                (w.phq, 0),
                (w.phk, 1),
                (w.v, 2),
                (w.kappa, 3),
                (w.pow, 4),
                (w.states, 5),
                (g.dout, 6),
                (g.dthq, 7),
                (g.dthk, 8),
                (g.dv, 9),
                (g.dkappa, 10),
                (g.dstates, 11),
            ],
            p,
            // One owner per (batch, head) performs the exact reverse scan
            // for all value channels and reduces dκ without float atomics.
            d.b * d.nh,
        )
    }

    // ---- routing API ----
    pub fn route(
        &self,
        r: &RouteDims,
        x: &GBuf,
        mu: &GBuf,
        muoff: usize,
        u: &GBuf,
        uoff: usize,
        bias: &GBuf,
        eoff: usize,
        assign: &GBuf,
        res: &GBuf,
    ) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = r.rows as u32;
        p[2] = r.h as u32;
        p[3] = r.e as u32;
        p[4] = r.k as u32;
        p[5] = muoff as u32;
        p[6] = uoff as u32;
        p[7] = eoff as u32;
        self.rec(
            OP_ROUTE,
            &[(x, 0), (mu, 1), (u, 2), (bias, 3), (assign, 4), (res, 5)],
            p,
            r.rows,
        )
    }
    pub fn route_smooth_k4(
        &self,
        r: &RouteDims,
        _t: usize,
        x: &GBuf,
        mu: &GBuf,
        muoff: usize,
        u: &GBuf,
        uoff: usize,
        bias: &GBuf,
        eoff: usize,
        assign: &GBuf,
        res: &GBuf,
    ) {
        let _ = (r, _t, x, mu, muoff, u, uoff, bias, eoff, assign, res);
        panic!("Vulkan backend rejects router_smooth_k4; use fixed top-1 routing");
    }
    pub fn route_top2(
        &self,
        r: &RouteDims,
        _t: usize,
        x: &GBuf,
        mu: &GBuf,
        muoff: usize,
        u: &GBuf,
        uoff: usize,
        bias: &GBuf,
        eoff: usize,
        threshold: f32,
        assign: &GBuf,
        runner: &GBuf,
        margin: &GBuf,
        rw: &GBuf,
        res: &GBuf,
        fallback: &GBuf,
    ) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = r.rows as u32;
        p[2] = r.h as u32;
        p[3] = r.e as u32;
        p[4] = r.k as u32;
        p[5] = muoff as u32;
        p[6] = uoff as u32;
        p[7] = eoff as u32;
        p[8] = threshold.to_bits();
        self.rec(
            OP_ROUTE_TOP2,
            &[
                (x, 0),
                (mu, 1),
                (u, 2),
                (bias, 3),
                (assign, 4),
                (runner, 5),
                (margin, 6),
                (rw, 7),
                (res, 8),
                (fallback, 9),
            ],
            p,
            r.rows,
        )
    }
    pub fn route_group(
        &self,
        r: &RouteDims,
        assign: &GBuf,
        slot: &GBuf,
        count: &GBuf,
        eoff: usize,
    ) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = r.rows as u32;
        p[2] = r.e as u32;
        p[3] = eoff as u32;
        self.rec(
            OP_ROUTE_GROUP,
            &[(assign, 0), (slot, 1), (count, 2)],
            p,
            r.rows.max(r.e),
        )
    }
    pub fn moe_gather(&self, r: &RouteDims, x: &GBuf, assign: &GBuf, slot: &GBuf, hg: &GBuf) {
        // The native Metal path clears the expert staging arena before the
        // sparse gather.  Keep the same invariant here: a dropped route (or
        // an empty expert slot) must present a zero row to the dense expert
        // GEMMs rather than stale data from the previous layer/batch.
        self.axpby(0.0, hg, 0.0, hg, r.e * r.cap * r.h);
        let mut p = vec![0; PARAM_WORDS];
        p[1] = r.rows as u32;
        p[2] = r.h as u32;
        p[3] = r.cap as u32;
        self.rec(
            OP_MOE_GATHER,
            &[(x, 0), (assign, 1), (slot, 2), (hg, 3)],
            p,
            r.rows * r.h,
        )
    }
    pub fn moe_gather_weighted(
        &self,
        r: &RouteDims,
        x: &GBuf,
        assign: &GBuf,
        slot: &GBuf,
        w: &GBuf,
        hg: &GBuf,
        _primary: bool,
    ) {
        let _ = (r, x, assign, slot, w, hg, _primary);
        panic!("Vulkan backend rejects weighted top-2 expert routing");
    }
    pub fn moe_scatter_add(&self, r: &RouteDims, dst: &GBuf, assign: &GBuf, slot: &GBuf, y: &GBuf) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = r.rows as u32;
        p[2] = r.h as u32;
        p[3] = r.cap as u32;
        self.rec(
            OP_MOE_SCATTER,
            &[(dst, 0), (assign, 1), (slot, 2), (y, 3)],
            p,
            r.rows * r.h,
        )
    }
    pub fn moe_scatter_add_weighted(
        &self,
        r: &RouteDims,
        dst: &GBuf,
        assign: &GBuf,
        slot: &GBuf,
        w: &GBuf,
        y: &GBuf,
        primary: bool,
    ) {
        let _ = (r, dst, assign, slot, w, y, primary);
        panic!("Vulkan backend rejects weighted top-2 expert routing");
    }
    pub fn moe_stats(
        &self,
        r: &RouteDims,
        hg: &GBuf,
        count: &GBuf,
        eoff: usize,
        sums: &GBuf,
        muoff: usize,
    ) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = r.e as u32;
        p[2] = r.h as u32;
        p[3] = r.cap as u32;
        p[4] = eoff as u32;
        p[5] = muoff as u32;
        self.rec(
            OP_MOE_STATS,
            &[(hg, 0), (count, 1), (sums, 2)],
            p,
            r.e * r.h,
        )
    }
    pub fn moe_update(
        &self,
        r: &RouteDims,
        mu: &GBuf,
        muoff: usize,
        bias: &GBuf,
        eoff: usize,
        sums: &GBuf,
        summoff: usize,
        count: &GBuf,
        countoff: usize,
        res: &GBuf,
        alpha: f32,
        eta: f32,
        frozen: usize,
        bias_frozen_from: usize,
    ) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = r.e as u32;
        p[2] = r.h as u32;
        p[3] = muoff as u32;
        p[4] = eoff as u32;
        p[5] = summoff as u32;
        p[6] = countoff as u32;
        p[7] = alpha.to_bits();
        p[8] = eta.to_bits();
        p[9] = frozen as u32;
        p[10] = r.rows as u32;
        // experts `e >= bias_frozen_from` keep their balancing bias (growth)
        p[11] = bias_frozen_from.min(u32::MAX as usize) as u32;
        // the μ mean divides by the slots moe_stats summed: min(count, cap)
        p[12] = r.cap as u32;
        self.rec(
            OP_MOE_UPDATE,
            &[(mu, 0), (bias, 1), (sums, 2), (count, 3), (res, 4)],
            p,
            r.e * r.h,
        )
    }
    pub fn moe_init_mu(&self, r: &RouteDims, x: &GBuf, rows: &GBuf, mu: &GBuf, off: usize) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = r.e as u32;
        p[2] = r.h as u32;
        p[3] = off as u32;
        self.rec(OP_MOE_INIT, &[(x, 0), (rows, 1), (mu, 2)], p, r.e * r.h)
    }
    pub fn moe_indirect_args(
        &self,
        count: &GBuf,
        coff: usize,
        indir: &GBuf,
        ioff: usize,
        e: usize,
        cap: usize,
        i: usize,
        h: usize,
    ) {
        assert!(count.len >= coff + e);
        assert!(indir.len >= ioff + 2 * e * 3);
        let mut p = vec![0; PARAM_WORDS];
        p[1] = e as u32;
        p[2] = cap as u32;
        p[3] = i as u32;
        p[4] = h as u32;
        p[5] = coff as u32;
        p[6] = ioff as u32;
        self.rec(OP_MOE_INDIRECT, &[(count, 0), (indir, 1)], p, e);
    }
    pub fn moe_center(
        &self,
        r: &RouteDims,
        hg: &GBuf,
        mu: &GBuf,
        off: usize,
        count: &GBuf,
        coff: usize,
        out: &GBuf,
    ) {
        assert!(hg.len >= r.e * r.cap * r.h);
        assert!(out.len >= r.e * r.cap * r.h);
        let mut p = vec![0; PARAM_WORDS];
        p[1] = r.e as u32;
        p[2] = r.h as u32;
        p[3] = r.cap as u32;
        p[4] = off as u32;
        p[5] = coff as u32;
        self.rec(
            OP_MOE_CENTER,
            &[(hg, 0), (mu, 1), (count, 2), (out, 3)],
            p,
            r.e * r.cap * r.h,
        );
    }
    pub fn mask_fwd(
        &self,
        hh: &GBuf,
        logits: &GBuf,
        loff: usize,
        rows: usize,
        i: usize,
        hard: bool,
        tau: f32,
    ) {
        // Same contract as the Metal `mask_fwd_f32`: element z = (row, j)
        // reads logit `loff + j` (j = z mod i), and the hard mask is the
        // 0/1 indicator 1[σ > τ] — not σ gated by it — so a baked skill's
        // folded tensors equal what phase A/B trained under.
        assert!(i > 0 && hh.len >= rows * i && logits.len >= loff + i);
        let mut p = vec![0; PARAM_WORDS];
        p[1] = (rows * i) as u32;
        p[2] = loff as u32;
        p[3] = hard as u32;
        p[4] = tau.to_bits();
        p[5] = i as u32;
        self.rec(OP_MASK_FWD, &[(hh, 0), (logits, 1)], p, rows * i)
    }
    pub fn mask_bwd(
        &self,
        dh: &GBuf,
        hh: &GBuf,
        logits: &GBuf,
        loff: usize,
        dm: &GBuf,
        rows: usize,
        i: usize,
        hard: bool,
        tau: f32,
        l1: f32,
    ) {
        let mut p = vec![0; PARAM_WORDS];
        // The shader owns both the per-row activation derivative and the
        // per-neuron reduction for the mask-logit gradient.  Keep the row
        // count and neuron width separate so a multi-layer SkillState writes
        // only its selected `[offset .. offset + i]` slice.
        p[1] = rows as u32;
        p[2] = i as u32;
        p[3] = hard as u32;
        p[4] = tau.to_bits();
        p[5] = l1.to_bits();
        p[6] = loff as u32;
        // Reduce the mask-logit derivative before the in-place activation
        // gradient pass below; otherwise concurrent rows would race while
        // reading `dh` as both source and destination.
        self.rec(
            OP_MASK_DM,
            &[(dh, 0), (hh, 1), (logits, 2), (dm, 3)],
            p.clone(),
            i,
        );
        self.rec(
            OP_MASK_BWD,
            &[(dh, 0), (hh, 1), (logits, 2), (dm, 3)],
            p,
            rows * i,
        )
    }
    /// Hierarchical-head evaluation GEMM without padding each cluster bucket
    /// to a 64-row tile.  `head_idx` maps grouped rows back to source tokens;
    /// the target-cluster buffer selects the corresponding contiguous
    /// vocabulary slice.  This is evaluation-only: training keeps the
    /// existing grouped buffers because its weight-gradient reduction has a
    /// separate deterministic contract.
    #[allow(clippy::too_many_arguments)]
    pub fn head_group_fwd(
        &self,
        x: &GBuf,
        p: &GBuf,
        embed_off: usize,
        head_idx: &GBuf,
        tgt_cluster: &GBuf,
        logits: &GBuf,
        rows: usize,
        cols: usize,
        h: usize,
    ) {
        let mut args = vec![0; PARAM_WORDS];
        args[1] = rows as u32;
        args[2] = cols as u32;
        args[3] = h as u32;
        args[4] = embed_off as u32;
        self.rec(
            OP_HEAD_GROUP_FWD,
            &[(x, 0), (p, 1), (head_idx, 2), (tgt_cluster, 3), (logits, 4)],
            args,
            rows * cols,
        )
    }
    /// The Vulkan graph uses a resident fixed-decay table generated on the
    /// host (`hk_pow_table`).  It does not yet expose the trainable-decay
    /// `γ^δ` materialisation pass from the Metal backend.  Refusing the
    /// direct call is important: a success-shaped no-op would leave stale
    /// powers (and silently corrupt every subsequent recurrent result).
    pub fn hk_pow_from_alog(&self, _a: &GBuf, _p: &GBuf, _off: usize, _nh: usize, _np: usize) {
        panic!("Vulkan backend rejects trainable-decay hk_pow_from_alog")
    }

    /// Trainable-decay γ gradients are outside the fixed-decay Vulkan
    /// contract.  Do not silently return with zero gradients: callers must
    /// choose the supported fixed-decay path or fail closed.
    pub fn hk_dgamma(&self, _d: &HkDims, _w: &HkWork<'_>, _g: &HkGrads<'_>) {
        panic!("Vulkan backend rejects trainable-decay hk_dgamma")
    }
    pub fn gdn_forward(
        &self,
        _q: &GBuf,
        _k: &GBuf,
        _v: &GBuf,
        _z: &GBuf,
        _ab: &GBuf,
        _p: &GBuf,
        _co: usize,
        _no: usize,
        _ao: usize,
        _dt: usize,
        _qcv: &GBuf,
        _kcv: &GBuf,
        _vcv: &GBuf,
        _beta: &GBuf,
        _raw: &GBuf,
        _inv: &GBuf,
        _out: &GBuf,
        _states: &GBuf,
        _b: usize,
        _t: usize,
        _eps: f32,
    ) {
        panic!("GDN lane is not enabled by the native Vulkan P1 shape")
    }
    pub fn gdn_forward_parallel(
        &self,
        q: &GBuf,
        k: &GBuf,
        v: &GBuf,
        z: &GBuf,
        ab: &GBuf,
        p: &GBuf,
        co: usize,
        no: usize,
        ao: usize,
        dt: usize,
        qcv: &GBuf,
        kcv: &GBuf,
        vcv: &GBuf,
        beta: &GBuf,
        raw: &GBuf,
        inv: &GBuf,
        out: &GBuf,
        states: &GBuf,
        b: usize,
        t: usize,
        eps: f32,
    ) {
        self.gdn_forward(
            q, k, v, z, ab, p, co, no, ao, dt, qcv, kcv, vcv, beta, raw, inv, out, states, b, t,
            eps,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_backward(
        &self,
        _qraw: &GBuf,
        _kraw: &GBuf,
        _vraw: &GBuf,
        _qcv: &GBuf,
        _kcv: &GBuf,
        _vcv: &GBuf,
        _ab: &GBuf,
        _beta: &GBuf,
        _raw_o: &GBuf,
        _inv: &GBuf,
        _norm: &GBuf,
        _states: &GBuf,
        _dnorm: &GBuf,
        _dz: &GBuf,
        _dq: &GBuf,
        _dk: &GBuf,
        _dv: &GBuf,
        _dab: &GBuf,
        _p: &GBuf,
        _conv_off: usize,
        _norm_off: usize,
        _alog_off: usize,
        _dt_off: usize,
        _g: &GBuf,
        _gconv_off: usize,
        _gnorm_off: usize,
        _galog_off: usize,
        _gdt_off: usize,
        _b: usize,
        _t: usize,
    ) {
        panic!("GDN lane is not enabled by the native Vulkan P1 shape")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_backward_parallel(
        &self,
        qraw: &GBuf,
        kraw: &GBuf,
        vraw: &GBuf,
        qcv: &GBuf,
        kcv: &GBuf,
        vcv: &GBuf,
        ab: &GBuf,
        beta: &GBuf,
        raw_o: &GBuf,
        inv: &GBuf,
        norm: &GBuf,
        states: &GBuf,
        dnorm: &GBuf,
        dz: &GBuf,
        dq: &GBuf,
        dk: &GBuf,
        dv: &GBuf,
        dab: &GBuf,
        p: &GBuf,
        conv_off: usize,
        norm_off: usize,
        alog_off: usize,
        dt_off: usize,
        g: &GBuf,
        gconv_off: usize,
        gnorm_off: usize,
        galog_off: usize,
        gdt_off: usize,
        b: usize,
        t: usize,
    ) {
        self.gdn_backward(
            qraw, kraw, vraw, qcv, kcv, vcv, ab, beta, raw_o, inv, norm, states, dnorm, dz, dq, dk,
            dv, dab, p, conv_off, norm_off, alog_off, dt_off, g, gconv_off, gnorm_off, galog_off,
            gdt_off, b, t,
        )
    }
    /// y = SiLU(x) over n.
    pub fn silu_fwd(&self, x: &GBuf, y: &GBuf, n: usize) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = n as u32;
        self.rec(OP_SILU_FWD, &[(x, 0), (y, 1)], p, n)
    }
    /// dx = dy·SiLU'(x) over n (x = the pre-activation).
    pub fn silu_bwd(&self, x: &GBuf, dy: &GBuf, dx: &GBuf, n: usize) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = n as u32;
        self.rec(OP_SILU_BWD, &[(x, 0), (dy, 1), (dx, 2)], p, n)
    }

    fn gdn_scan_params(d: &GdnScanDims, flags: u32, alog_off: usize, dt_off: usize, entry: u32) -> Vec<u32> {
        assert!(d.dk >= 4 && d.dk <= 128 && d.dv >= 4 && d.dv <= 128, "gdn scan: dk/dv ≤ 128");
        assert!(d.c_dim >= 2 * d.nv * d.dk + d.nv * d.dv && d.ab_ld >= d.nv);
        let mut p = vec![0; PARAM_WORDS];
        p[1] = d.b as u32;
        p[2] = d.t as u32;
        p[3] = d.nv as u32;
        p[4] = d.dk as u32;
        p[5] = d.dv as u32;
        p[6] = d.c_dim as u32;
        p[7] = d.ab_ld as u32;
        p[8] = flags;
        p[9] = alog_off as u32;
        p[10] = dt_off as u32;
        p[PARAM_WORDS - 2] = entry;
        p
    }

    /// GDN mixer scan forward — same contract as the Metal `gdn_scan_fwd`
    /// (`gdn_fwd` entry, one 128-lane workgroup per (sequence, head)).
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_scan_fwd(
        &self,
        d: &GdnScanDims,
        qkv_cv: &GBuf,
        a_pre: &GBuf,
        b_pre: &GBuf,
        p: &GBuf,
        alog_off: usize,
        dt_off: usize,
        raw_o: &GBuf,
        states: &GBuf,
        live: &GBuf,
        s0_from_ckpt: bool,
        beta_one: bool,
    ) {
        let rows = d.rows();
        assert!(qkv_cv.len >= rows * d.c_dim && a_pre.len >= rows * d.ab_ld && b_pre.len >= rows * d.ab_ld);
        assert!(raw_o.len >= rows * d.nv * d.dv && p.len >= alog_off + d.nv && p.len >= dt_off + d.nv);
        assert!(states.len >= d.b * d.nv * (d.nch() + 1) * d.state() && live.len >= d.b * d.nv * d.state());
        let pp = Self::gdn_scan_params(d, s0_from_ckpt as u32 | ((beta_one as u32) << 1), alog_off, dt_off, 2);
        self.rec(
            OP_GDN_SCAN_FWD,
            &[(qkv_cv, 0), (a_pre, 1), (b_pre, 2), (p, 3), (raw_o, 4), (states, 5), (live, 6)],
            pp,
            d.b * d.nv * 256,
        )
    }

    /// GDN mixer scan backward — same contract as the Metal `gdn_scan_bwd`
    /// (`gdn_bwd` entry).
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_scan_bwd(
        &self,
        d: &GdnScanDims,
        qkv_cv: &GBuf,
        a_pre: &GBuf,
        b_pre: &GBuf,
        p: &GBuf,
        alog_off: usize,
        dt_off: usize,
        states: &GBuf,
        doo: &GBuf,
        chunk: &GBuf,
        dlive: &GBuf,
        ds_init: bool,
        beta_one: bool,
        dcv: &GBuf,
        da: &GBuf,
        db: &GBuf,
        part: &GBuf,
    ) {
        let rows = d.rows();
        assert!(qkv_cv.len >= rows * d.c_dim && dcv.len >= rows * d.c_dim);
        assert!(a_pre.len >= rows * d.ab_ld && b_pre.len >= rows * d.ab_ld && da.len >= rows * d.ab_ld && db.len >= rows * d.ab_ld);
        assert!(doo.len >= rows * d.nv * d.dv && part.len >= d.b * d.nv * 2);
        assert!(states.len >= d.b * d.nv * (d.nch() + 1) * d.state());
        assert!(chunk.len >= d.b * d.nv * 65 * d.state() && dlive.len >= d.b * d.nv * d.state());
        let pp = Self::gdn_scan_params(d, ds_init as u32 | ((beta_one as u32) << 1), alog_off, dt_off, 3);
        self.rec(
            OP_GDN_SCAN_BWD,
            &[
                (qkv_cv, 0),
                (a_pre, 1),
                (b_pre, 2),
                (p, 3),
                (states, 4),
                (doo, 5),
                (chunk, 6),
                (dlive, 7),
                (dcv, 8),
                (da, 9),
                (db, 10),
                (part, 11),
            ],
            pp,
            d.b * d.nv * 256,
        )
    }

    /// g[galog_off + h] += Σ_b part[(b·nv+h)·2]; g[gdt_off + h] += Σ_b part[..+1].
    pub fn gdn_scan_fold(&self, part: &GBuf, b: usize, nv: usize, g: &GBuf, galog_off: usize, gdt_off: usize) {
        assert!(part.len >= b * nv * 2 && g.len >= galog_off + nv && g.len >= gdt_off + nv);
        let mut p = vec![0; PARAM_WORDS];
        p[1] = b as u32;
        p[2] = nv as u32;
        p[3] = galog_off as u32;
        p[4] = gdt_off as u32;
        self.rec(OP_GDN_FOLD, &[(part, 0), (g, 1)], p, nv)
    }

    // ---- GDN chunked WY/UT form (gdn_wy.rs) ----
    fn wy_p(d: &GdnScanDims) -> Vec<u32> {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = d.b as u32;
        p[2] = d.t as u32;
        p[3] = d.nv as u32;
        p[4] = d.dk as u32;
        p[5] = d.dv as u32;
        p[6] = d.c_dim as u32;
        p[7] = d.ab_ld as u32;
        p[8] = (d.t / 64) as u32;
        p
    }
    /// zero checkpoint slot 0 of every (b, h): states[(bh·slots)·ss ..]
    pub fn gdn_wy_zero_s0(&self, states: &GBuf, nb: usize, slots: usize, ss: usize) {
        let mut p = vec![0; PARAM_WORDS];
        p[1] = nb as u32;
        p[2] = slots as u32;
        p[3] = ss as u32;
        self.rec(OP_WY_ZERO_S0, &[(states, 0)], p, nb * ss)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_wy_scal(&self, d: &GdnScanDims, a_pre: &GBuf, b_pre: &GBuf, p: &GBuf, alog_off: usize, dt_off: usize, lg: &GBuf, gam: &GBuf, bet: &GBuf, rho: &GBuf, beta_one: bool) {
        let mut pp = Self::wy_p(d);
        pp[9] = alog_off as u32;
        pp[10] = dt_off as u32;
        pp[11] = beta_one as u32;
        self.rec(OP_WY_SCAL, &[(a_pre, 0), (b_pre, 1), (p, 2), (lg, 3), (gam, 4), (bet, 5), (rho, 6)], pp, d.b * d.nv * (d.t / 64))
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_wy_norm(&self, d: &GdnScanDims, qkv_cv: &GBuf, gam: &GBuf, qn: &GBuf, kn: &GBuf, qg: &GBuf, kg: &GBuf, vcm: &GBuf) {
        let pp = Self::wy_p(d);
        self.rec(OP_WY_NORM, &[(qkv_cv, 0), (gam, 1), (qn, 2), (kn, 3), (qg, 4), (kg, 5), (vcm, 6)], pp, d.rows() * d.nv)
    }
    pub fn gdn_wy_mask(&self, d: &GdnScanDims, kk: &GBuf, qk: &GBuf, lg: &GBuf, bet: &GBuf, lmat: &GBuf) {
        let pp = Self::wy_p(d);
        self.rec(OP_WY_MASK, &[(kk, 0), (qk, 1), (lg, 2), (bet, 3), (lmat, 4)], pp, d.b * d.nv * (d.t / 64) * 4096)
    }
    pub fn gdn_wy_ut(&self, d: &GdnScanDims, lmat: &GBuf, tmat: &GBuf) {
        let pp = Self::wy_p(d);
        self.rec(OP_WY_UT, &[(lmat, 0), (tmat, 1)], pp, d.b * d.nv * (d.t / 64) * 64)
    }
    pub fn gdn_wy_sscale(&self, d: &GdnScanDims, states: &GBuf, gam: &GBuf, c: usize) {
        let mut pp = Self::wy_p(d);
        pp[12] = c as u32;
        self.rec(OP_WY_SSCALE, &[(states, 0), (gam, 1)], pp, d.b * d.nv * d.state())
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_wy_u(&self, d: &GdnScanDims, epm: &GBuf, bet: &GBuf, rho: &GBuf, um: &GBuf, utm: &GBuf, c: usize) {
        let mut pp = Self::wy_p(d);
        pp[12] = c as u32;
        self.rec(OP_WY_U, &[(epm, 0), (bet, 1), (rho, 2), (um, 3), (utm, 4)], pp, d.b * d.nv * 64 * d.dv)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_wy_du(&self, d: &GdnScanDims, du: &GBuf, dut: &GBuf, um: &GBuf, epm: &GBuf, bet: &GBuf, rho: &GBuf, dep: &GBuf, drho: &GBuf, dbet: &GBuf, c: usize) {
        let mut pp = Self::wy_p(d);
        pp[12] = c as u32;
        self.rec(OP_WY_DU, &[(du, 0), (dut, 1), (um, 2), (epm, 3), (bet, 4), (rho, 5), (dep, 6), (drho, 7), (dbet, 8)], pp, d.b * d.nv * 64)
    }
    pub fn gdn_wy_dstate(&self, d: &GdnScanDims, dsn: &GBuf, dlive: &GBuf, gam: &GBuf, c: usize) {
        let mut pp = Self::wy_p(d);
        pp[12] = c as u32;
        self.rec(OP_WY_DSTATE, &[(dsn, 0), (dlive, 1), (gam, 2)], pp, d.b * d.nv * d.state())
    }
    pub fn gdn_wy_dgend(&self, d: &GdnScanDims, dlive: &GBuf, states: &GBuf, dgend: &GBuf, c: usize) {
        let mut pp = Self::wy_p(d);
        pp[12] = c as u32;
        self.rec(OP_WY_DGEND, &[(dlive, 0), (states, 1), (dgend, 2)], pp, d.b * d.nv)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_wy_mask_bwd_red(&self, d: &GdnScanDims, dp: &GBuf, qk: &GBuf, dl: &GBuf, lmat: &GBuf, kk: &GBuf, lg: &GBuf, bet: &GBuf, dlgm: &GBuf, dbet: &GBuf) {
        let pp = Self::wy_p(d);
        self.rec(OP_WY_MASK_BWD_RED, &[(dp, 0), (qk, 1), (dl, 2), (lmat, 3), (kk, 4), (lg, 5), (bet, 6), (dlgm, 7), (dbet, 8)], pp, d.b * d.nv * (d.t / 64) * 64)
    }
    pub fn gdn_wy_mask_bwd(&self, d: &GdnScanDims, dp: &GBuf, dl: &GBuf, lg: &GBuf, bet: &GBuf) {
        let pp = Self::wy_p(d);
        self.rec(OP_WY_MASK_BWD, &[(dp, 0), (dl, 1), (lg, 2), (bet, 3)], pp, d.b * d.nv * (d.t / 64) * 4096)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_wy_dlg(&self, d: &GdnScanDims, dqn: &GBuf, dkn: &GBuf, dqg: &GBuf, dkg: &GBuf, qn: &GBuf, kn: &GBuf, drho: &GBuf, dlgm: &GBuf, dgend: &GBuf, rho: &GBuf, gam: &GBuf, dla: &GBuf) {
        let pp = Self::wy_p(d);
        self.rec(OP_WY_DLG, &[(dqn, 0), (dkn, 1), (dqg, 2), (dkg, 3), (qn, 4), (kn, 5), (drho, 6), (dlgm, 7), (dgend, 8), (rho, 9), (gam, 10), (dla, 11)], pp, d.b * d.nv * (d.t / 64))
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_wy_dscal(&self, d: &GdnScanDims, dla: &GBuf, a_pre: &GBuf, p: &GBuf, alog_off: usize, dt_off: usize, dbet: &GBuf, bet: &GBuf, da: &GBuf, db: &GBuf, part: &GBuf, beta_one: bool) {
        let mut pp = Self::wy_p(d);
        pp[9] = alog_off as u32;
        pp[10] = dt_off as u32;
        pp[11] = beta_one as u32;
        self.rec(OP_WY_DSCAL, &[(dla, 0), (a_pre, 1), (p, 2), (dbet, 3), (bet, 4), (da, 5), (db, 6), (part, 7)], pp, d.b * d.nv)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_wy_dnorm(&self, d: &GdnScanDims, dqn: &GBuf, dkn: &GBuf, dqg: &GBuf, dkg: &GBuf, qkv_cv: &GBuf, gam: &GBuf, dcv: &GBuf) {
        let pp = Self::wy_p(d);
        self.rec(OP_WY_DNORM, &[(dqn, 0), (dkn, 1), (dqg, 2), (dkg, 3), (qkv_cv, 4), (gam, 5), (dcv, 6)], pp, d.rows() * d.nv)
    }

    pub fn commit(self) -> f64 {
        let ops = self.ops.into_inner();
        for op in &ops {
            for s in &op.slots {
                s.upload_host();
            }
        }
        let encode_start = std::time::Instant::now();
        let mut enc = self
            .c
            .inner
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("embryo-train-submit"),
            });
        const TS_MAX_PAIRS: usize = 2048;
        // Timestamp staging/query resources are process-wide and native tests
        // may issue independent commands from multiple threads. Serialize
        // only this opt-in diagnostic path; normal training never touches it.
        let _ts_guard = self
            .c
            .inner
            .ts
            .as_ref()
            .map(|_| self.c.inner.ts_lock.lock().unwrap());
        if let Some(ts) = self.c.inner.ts.as_ref() {
            // Diagnostic-only path: a separate pass gives each operation its
            // own begin/end timestamp.  It intentionally pays extra pass and
            // synchronization overhead and is never selected in training
            // unless CMF_VULKAN_OP_TS=1 is set.
            for (i, op) in ops.iter().enumerate() {
                let (_pb, bg) = self.c.inner.bind_op(op);
                let pipeline = match op.entry {
                    1 => self.c.inner.pipeline64.clone(),
                    2 => self.c.inner.pipeline_gdn_fwd.clone(),
                    3 => self.c.inner.pipeline_gdn_bwd.clone(),
                    _ => self.c.inner.pipeline_for(op.params[0]),
                };
                let timestamp_writes =
                    (i < TS_MAX_PAIRS).then_some(wgpu::ComputePassTimestampWrites {
                        query_set: &ts.query_set,
                        beginning_of_pass_write_index: Some((2 * i) as u32),
                        end_of_pass_write_index: Some((2 * i + 1) as u32),
                    });
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("embryo-op-profile"),
                    timestamp_writes,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                let gx = op.groups.min(65_535);
                let gy = op.groups.div_ceil(gx);
                pass.dispatch_workgroups(gx, gy, 1);
                drop(pass);
                for s in &op.slots {
                    s.mark_gpu();
                }
            }
            let n = ops.len().min(TS_MAX_PAIRS);
            if n > 0 {
                enc.resolve_query_set(&ts.query_set, 0..(2 * n) as u32, &ts.resolve, 0);
                enc.copy_buffer_to_buffer(&ts.resolve, 0, &ts.stage, 0, (2 * n * 8) as u64);
            }
        } else {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("embryo-train"),
                timestamp_writes: None,
            });
            for op in &ops {
                let pipeline = match op.entry {
                    1 => self.c.inner.pipeline64.clone(),
                    2 => self.c.inner.pipeline_gdn_fwd.clone(),
                    3 => self.c.inner.pipeline_gdn_bwd.clone(),
                    _ => self.c.inner.pipeline_for(op.params[0]),
                };
                pass.set_pipeline(&pipeline);
                let (_pb, bg) = self.c.inner.bind_op(op);
                pass.set_bind_group(0, &bg, &[]);
                // Vulkan limits each dispatch dimension to 65,535 workgroups.
                // Keep a flat logical invocation index in WGSL while folding
                // very large parameter-arena sweeps into the y dimension.
                let gx = op.groups.min(65_535);
                let gy = op.groups.div_ceil(gx);
                pass.dispatch_workgroups(gx, gy, 1);
                for s in &op.slots {
                    s.mark_gpu();
                }
            }
        }
        if self.c.inner.profile_ops {
            eprintln!(
                "vulkan encode ops={} ms={:.3} specialized={}",
                ops.len(),
                encode_start.elapsed().as_secs_f64() * 1000.0,
                self.c.inner.specialize
            );
        }
        let t = std::time::Instant::now();
        self.c.inner.queue.submit(Some(enc.finish()));
        let _ = self
            .c
            .inner
            .device
            .poll(wgpu::PollType::wait_indefinitely());
        if let Some(ts) = self.c.inner.ts.as_ref() {
            let n = ops.len().min(TS_MAX_PAIRS);
            if n > 0 {
                let bytes = (2 * n * 8) as u64;
                let (tx, rx) = std::sync::mpsc::channel();
                ts.stage.map_async(wgpu::MapMode::Read, ..bytes, move |r| {
                    let _ = tx.send(r);
                });
                let _ = self
                    .c
                    .inner
                    .device
                    .poll(wgpu::PollType::wait_indefinitely());
                if rx.recv().map(|r| r.is_ok()).unwrap_or(false) {
                    let mapped = ts
                        .stage
                        .get_mapped_range(..bytes)
                        .expect("timestamp readback");
                    let raw: &[u64] = bytemuck::cast_slice(&mapped);
                    let mut totals = HashMap::<u32, (f64, usize)>::new();
                    for (i, op) in ops.iter().take(n).enumerate() {
                        let ticks = raw[2 * i + 1].saturating_sub(raw[2 * i]);
                        let ms = ticks as f64 * ts.period_ns as f64 / 1.0e6;
                        let e = totals.entry(op.params[0]).or_insert((0.0, 0));
                        e.0 += ms;
                        e.1 += 1;
                    }
                    let mut rows: Vec<_> = totals.into_iter().collect();
                    rows.sort_unstable_by_key(|(op, _)| *op);
                    for (op, (ms, count)) in rows {
                        eprintln!("vulkan op-ts op={op} count={count} gpu_ms={ms:.3}");
                    }
                    drop(mapped);
                    ts.stage.unmap();
                } else {
                    eprintln!("cortiq-embryo: Vulkan timestamp map failed");
                }
            }
        }
        t.elapsed().as_secs_f64() * 1000.0
    }
}

// `create_buffer_init` is the only utility trait used by the backend.
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    N,
    T,
}

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
    pub fn none() -> Self {
        Self {
            nb: 1,
            nh: 1,
            nc: 1,
            sa: [0; 3],
            sb: [0; 3],
            sc: [0; 3],
        }
    }
}
#[derive(Clone, Copy)]
pub struct GemmDyn<'a> {
    pub indirect: Option<(&'a GBuf, usize)>,
    pub kcount: Option<(&'a GBuf, usize)>,
}
impl GemmDyn<'_> {
    pub fn none() -> GemmDyn<'static> {
        GemmDyn {
            indirect: None,
            kcount: None,
        }
    }
}

/// Table-batched GEMM (Vulkan only): `(table, word offset, rows-bound-K)`.
/// Batch z takes `(a_off, b_off, c_off, rows)` from `table[off + 4z..]`
/// (element offsets added to the base offsets; the stride batches are still
/// applied); `rows` bounds M, or K when the flag is set. Used for the
/// grouped hierarchical head, whose clusters have different row counts and
/// non-affine offsets. Kept out of `GemmDyn` so that struct stays identical
/// to the Metal seam's.
type GemmTable<'a> = Option<(&'a GBuf, usize, bool)>;

/// Build the resident decay-power table used by the hybrid-k recurrence.
pub fn hk_pow_table(decay: &[f32], nh: usize, nph: usize) -> Vec<f32> {
    let p2 = 2 * nph;
    let mut out = vec![0.0f32; nh * 65 * p2];
    for h in 0..nh {
        for f in 0..p2 {
            let base = decay[h * p2 + f] as f64;
            for delta in 0..=64usize {
                out[(h * 65 + delta) * p2 + f] = base.powi(delta as i32) as f32;
            }
        }
    }
    out
}

/// Shape contract of the GEMM-form hybrid_k (mirror of Metal `hk_check` +
/// `HkScratch::check`): T a multiple of the 64-token chunk, the resident
/// tables sized for B·T rows, the A scratch for every (b, h, chunk).
fn hk_gemm_check(d: &HkDims, w: &HkWork<'_>, sc: &HkScratch<'_>) {
    assert!(
        d.t % 64 == 0,
        "hybrid_k GEMM form: T must be a multiple of the chunk (64), got {}",
        d.t
    );
    assert!(d.nph <= 32 && d.dv <= 128, "hybrid_k kernels: nph ≤ 32, dv ≤ 128");
    let rows = d.b * d.t;
    let p2 = 2 * d.nph;
    assert!(w.phq.len >= rows * d.nh * p2 && w.phk.len >= rows * d.nh * p2);
    assert!(w.kv.len >= rows * d.nh * d.dv && w.out.len >= rows * d.nh * d.dv);
    assert!(w.pow.len >= w.pow_off + d.nh * 65 * p2);
    assert!(w.states.len >= d.b * d.nh * (d.t / 64 + 1) * p2 * d.dv);
    let cl = HkScratch::chunk_len(d);
    for x in [sc.qt, sc.kt, sc.qp, sc.kh, sc.dqt, sc.dkt, sc.dqi, sc.dki] {
        assert!(x.len >= cl, "hybrid_k GEMM scratch table too small");
    }
    assert!(sc.a.len >= HkScratch::a_len(d), "hybrid_k GEMM A scratch too small");
}

pub struct HkWork<'a> {
    pub thq: &'a GBuf,
    pub thk: &'a GBuf,
    pub v: &'a GBuf,
    pub kappa: &'a GBuf,
    pub pow: &'a GBuf,
    pub pow_off: usize,
    pub phq: &'a GBuf,
    pub phk: &'a GBuf,
    pub kv: &'a GBuf,
    pub states: &'a GBuf,
    pub out: &'a GBuf,
    pub phase_chunk: Option<&'a GBuf>,
    pub phase_partial: Option<&'a GBuf>,
}
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
pub struct HkScratch<'a> {
    pub qt: &'a GBuf,
    pub kt: &'a GBuf,
    pub qp: &'a GBuf,
    pub kh: &'a GBuf,
    pub dqt: &'a GBuf,
    pub dkt: &'a GBuf,
    pub dqi: &'a GBuf,
    pub dki: &'a GBuf,
    pub a: &'a GBuf,
}
impl HkScratch<'_> {
    pub fn chunk_len(d: &HkDims) -> usize {
        d.b * d.t * d.nh * 2 * d.nph
    }
    pub fn a_len(d: &HkDims) -> usize {
        d.b * d.nh * (d.t / 64) * 4096
    }
    pub fn chunk_major(d: &HkDims) -> [usize; 3] {
        let (p2, nch) = (2 * d.nph, d.t / 64);
        [d.nh * nch * 64 * p2, nch * 64 * p2, 64 * p2]
    }
    pub fn row_major(d: &HkDims, x: usize) -> [usize; 3] {
        [d.t * d.nh * x, x, 64 * d.nh * x]
    }
    pub fn states(d: &HkDims) -> [usize; 3] {
        let (p2, nch) = (2 * d.nph, d.t / 64);
        [
            d.nh * (nch + 1) * p2 * d.dv,
            (nch + 1) * p2 * d.dv,
            p2 * d.dv,
        ]
    }
}
#[derive(Clone, Copy, Debug)]
pub struct RouteDims {
    pub rows: usize,
    pub h: usize,
    pub e: usize,
    pub k: usize,
    pub cap: usize,
}

// Operation numbers shared with the WGSL switch.
const OP_NOOP: u32 = 0;
const OP_GEMM: u32 = 1;
const OP_AXPBY: u32 = 2;
const OP_ADAMW: u32 = 3;
const OP_SUMSQ: u32 = 4;
const OP_RMS_FWD: u32 = 5;
const OP_RMS_BWD: u32 = 6;
const OP_SWIGLU_FWD: u32 = 7;
const OP_SWIGLU_BWD: u32 = 8;
const OP_EMBED_GATHER: u32 = 9;
const OP_EMBED_SCATTER: u32 = 10;
const OP_SOFTMAX_CE: u32 = 11;
const OP_COPY: u32 = 12;
const OP_SLICE: u32 = 13;
const OP_PAD: u32 = 14;
const OP_DOT: u32 = 15;
const OP_SIGMOID_FWD: u32 = 16;
const OP_SIGMOID_BWD: u32 = 17;
const OP_ROPE: u32 = 18;
const OP_CAUSAL: u32 = 19;
const OP_SOFTMAX_BWD: u32 = 20;
const OP_GATHER: u32 = 21;
const OP_SCATTER: u32 = 22;
const OP_CONV_FWD: u32 = 23;
const OP_CONV_BWD: u32 = 24;
const OP_KAPPA_FWD: u32 = 25;
const OP_KAPPA_BWD: u32 = 26;
const OP_HK_PHI: u32 = 27;
const OP_HK_KV: u32 = 28;
const OP_HK_FWD: u32 = 29;
const OP_HK_BWD: u32 = 30;
const OP_ROUTE: u32 = 31;
const OP_ROUTE_GROUP: u32 = 32;
const OP_MOE_GATHER: u32 = 33;
const OP_MOE_SCATTER: u32 = 34;
const OP_MOE_STATS: u32 = 35;
const OP_MOE_UPDATE: u32 = 36;
const OP_MOE_INIT: u32 = 37;
const OP_ROUTE_TOP2: u32 = 38;
const OP_MASK_FWD: u32 = 39;
const OP_MASK_BWD: u32 = 40;
const OP_SOFTMAX_CE_IDX: u32 = 41;
const OP_GROUP_SUM: u32 = 42;
const OP_PHASE_REPLAY: u32 = 43;
const OP_PHASE_BWD: u32 = 44;
const OP_PHASE_FOLD: u32 = 45;
const OP_MOE_INDIRECT: u32 = 46;
const OP_MOE_CENTER: u32 = 47;
const OP_MASK_DM: u32 = 48;
const OP_HEAD_GROUP_FWD: u32 = 49;
const OP_HK_BWD_FAST: u32 = 50;
const OP_HK_FOLD_FAST: u32 = 51;
const OP_SINK_ACCUM: u32 = 52;
const OP_HK_SCALE: u32 = 53;
const OP_HK_UNSCALE: u32 = 54;
const OP_HK_DSTATES: u32 = 55;
const OP_HK_DKV_SPLIT: u32 = 56;
const OP_HK_DTHETA: u32 = 57;
const OP_EMBED_CHAIN: u32 = 58;
const OP_SILU_FWD: u32 = 59;
const OP_SILU_BWD: u32 = 60;
const OP_GDN_FOLD: u32 = 61;
/// Separate entry points (`gdn_fwd` / `gdn_bwd`): the op id is only a label.
const OP_GDN_SCAN_FWD: u32 = 62;
const OP_GDN_SCAN_BWD: u32 = 63;
// GDN chunked WY/UT form (gdn_wy.rs): per-token scalars, norms, masks, UT
// solve, per-chunk elementwise passes and the backward assembly.
const OP_WY_SCAL: u32 = 64;
const OP_WY_NORM: u32 = 65;
const OP_WY_MASK: u32 = 66;
const OP_WY_UT: u32 = 67;
const OP_WY_SSCALE: u32 = 68;
const OP_WY_U: u32 = 69;
const OP_WY_DU: u32 = 70;
const OP_WY_DSTATE: u32 = 71;
const OP_WY_DGEND: u32 = 72;
const OP_WY_MASK_BWD_RED: u32 = 73;
const OP_WY_MASK_BWD: u32 = 74;
const OP_WY_DLG: u32 = 75;
const OP_WY_DSCAL: u32 = 76;
const OP_WY_DNORM: u32 = 77;
const OP_WY_ZERO_S0: u32 = 78;
const OP_BLOCK_COPY: u32 = 79;

/// WGSL source.  Input/output storage buffers all use u32 words so a single
/// bind layout can serve f32 and token/index operations without host copies.
const EMBRYO_WGSL: &str = r#"
struct P { v: array<u32> };
@group(0) @binding(0) var<storage,read_write> a: array<u32>;
@group(0) @binding(1) var<storage,read_write> b: array<u32>;
@group(0) @binding(2) var<storage,read_write> c: array<u32>;
@group(0) @binding(3) var<storage,read_write> d: array<u32>;
@group(0) @binding(4) var<storage,read_write> e: array<u32>;
@group(0) @binding(5) var<storage,read_write> f: array<u32>;
@group(0) @binding(6) var<storage,read_write> g: array<u32>;
@group(0) @binding(7) var<storage,read_write> h: array<u32>;
@group(0) @binding(8) var<storage,read_write> i: array<u32>;
@group(0) @binding(9) var<storage,read_write> j: array<u32>;
@group(0) @binding(10) var<storage,read_write> k: array<u32>;
@group(0) @binding(11) var<storage,read_write> l: array<u32>;
@group(0) @binding(12) var<storage,read_write> m: array<u32>;
@group(0) @binding(13) var<storage,read_write> nbuf: array<u32>;
@group(0) @binding(14) var<storage,read_write> obuf: array<u32>;
@group(0) @binding(15) var<storage,read_write> pbuf: array<u32>;
@group(0) @binding(16) var<storage,read> par: array<u32>;
fn U(x:u32)->u32{return par[x];} fn F(x:u32)->f32{return bitcast<f32>(par[x]);}
fn AF(x:u32)->f32{return bitcast<f32>(a[x]);} fn BF(x:u32)->f32{return bitcast<f32>(b[x]);} fn CF(x:u32)->f32{return bitcast<f32>(c[x]);} fn DF(x:u32)->f32{return bitcast<f32>(d[x]);}
fn DU(x:u32)->u32{return d[x];}
fn CU(x:u32)->u32{return c[x];}
fn AU(x:u32)->u32{return a[x];}
fn BU(x:u32)->u32{return b[x];}
fn EF(x:u32)->f32{return bitcast<f32>(e[x]);} fn GF(x:u32)->f32{return bitcast<f32>(g[x]);} fn HF(x:u32)->f32{return bitcast<f32>(h[x]);} fn IF(x:u32)->f32{return bitcast<f32>(i[x]);}
fn JF(x:u32)->f32{return bitcast<f32>(j[x]);} fn KF(x:u32)->f32{return bitcast<f32>(k[x]);}
fn FF(x:u32)->f32{return bitcast<f32>(f[x]);}
fn MF(x:u32)->f32{return bitcast<f32>(m[x]);}
fn NF(x:u32)->f32{return bitcast<f32>(nbuf[x]);}
fn WF(x:f32)->u32{return bitcast<u32>(x);}
fn silu(x:f32)->f32 { return x/(1.0+exp(-x)); }
fn safe_exp(x:f32)->f32 { return exp(clamp(x,-80.0,80.0)); }

// One invocation owns one (batch, head, value) column for legacy hybrid-k.
// The recurrence is S_t = γ·S_{t-1} + φk_t·(κ_t v_t), with output q_tᵀS_t.
// Chunk boundaries are materialised for the reverse pass; every token is
// still computed on the device and no host mirror participates in the scan.
fn hybrid_k_forward(z:u32) {
 let bh=z/U(5u); let dd=z%U(5u); if(bh>=U(1u)*U(3u)){return;}
 let b0=bh/U(3u); let hh=bh%U(3u); let p2=2u*U(4u); let nch=(U(2u)+63u)/64u;
 let stbase=bh*(nch+1u)*p2*U(5u); let gamoff=U(6u)+hh*65u*p2;
 var S:array<f32,64>;
 // p[9] = 1: start from checkpoint slot 0 (state carried across windows)
 for(var ff=0u;ff<64u;ff=ff+1u){S[ff]=select(0.0,FF(stbase+ff*U(5u)+dd),U(9u)==1u && ff<p2);}
 for(var tt=0u;tt<U(2u);tt=tt+1u){
   if((tt%64u)==0u){for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){f[stbase+((tt/64u)*p2+ff)*U(5u)+dd]=WF(S[ff]);}}}
   let rr=(b0*U(2u)+tt)*U(3u)+hh; let kap=DF(rr); let vv=CF(rr*U(5u)+dd);
   for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){S[ff]=EF(gamoff+p2+ff)*S[ff]+BF(rr*p2+ff)*(kap*vv);}}
   // p[8] = 1: chunk-boundary states only (the GEMM form owns the output)
   if(U(8u)==0u){
     var outv=0.0;
     for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){outv=outv+AF(rr*p2+ff)*S[ff];}}
     g[rr*U(5u)+dd]=WF(outv);
   }
 }
 for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){f[stbase+(nch*p2+ff)*U(5u)+dd]=WF(S[ff]);}}
}

// Exact legacy reverse scan. One (batch,head) owner walks all value columns
// so dκ is reduced without racy float atomics.  The state entering a token is
// replayed from its resident 64-token boundary; this is intentionally the
// simple reference path and is fail-closed by the caller's bounded proof
// geometry until the production chunked variant is promoted.
fn hybrid_k_backward(z:u32) {
 let bh=z; if(bh>=U(1u)*U(3u)){return;}
 let b0=bh/U(3u); let hh=bh%U(3u); let p2=2u*U(4u); let nch=(U(2u)+63u)/64u;
 let stbase=bh*(nch+1u)*p2*U(5u); let gamoff=U(6u)+hh*65u*p2; let beta=F(7u);
 // Clear/rebase the reductions owned by this (batch, head) column.
 for(var tt0=0u;tt0<U(2u);tt0=tt0+1u){
   let rh0=(b0*U(2u)+tt0)*U(3u)+hh;
   k[rh0]=WF(0.0);
   for(var ii=0u;ii<32u;ii=ii+1u){if(ii<U(4u)){
     h[rh0*U(4u)+ii]=WF(beta*HF(rh0*U(4u)+ii));
     i[rh0*U(4u)+ii]=WF(beta*IF(rh0*U(4u)+ii));
   }}
 }
 for(var d0=0u;d0<U(5u);d0=d0+1u){
   var G:array<f32,64>; for(var ff=0u;ff<64u;ff=ff+1u){G[ff]=0.0;}
   // The terminal state has no successor contribution.
   for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){l[stbase+(nch*p2+ff)*U(5u)+d0]=WF(0.0);}}
   for(var rev=0u;rev<U(2u);rev=rev+1u){
     let tt=U(2u)-1u-rev; let cid=tt/64u; let start=cid*64u;
     var S:array<f32,64>;
     for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){S[ff]=FF(stbase+(cid*p2+ff)*U(5u)+d0);}}
     for(var ss=0u;ss<64u;ss=ss+1u){if(start+ss<=tt){
       let rrs=(b0*U(2u)+start+ss)*U(3u)+hh; let kp=DF(rrs); let vv=CF(rrs*U(5u)+d0);
       for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){S[ff]=EF(gamoff+p2+ff)*S[ff]+BF(rrs*p2+ff)*(kp*vv);}}
     }}
     let rh=(b0*U(2u)+tt)*U(3u)+hh; let qbase=rh*p2; let dout=GF(rh*U(5u)+d0);
     let kap=DF(rh); let vv=CF(rh*U(5u)+d0); let kvt=kap*vv;
     // Add the direct output contribution before propagating through the
     // recurrent write.  The old scalar path added q*dout only after dkv,
     // which dropped every token's contribution from dφk/dv/dκ.
     for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){G[ff]=G[ff]+AF(qbase+ff)*dout;}}
     // dθ from dφ, then dkv from the state adjoint.  This owner is the
     // only writer for this row/head, so all feature/value reductions are
     // deterministic and no float atomics are required.
     for(var ii=0u;ii<32u;ii=ii+1u){if(ii<U(4u)){
       let dqc=dout*S[ii]; let dqs=dout*S[U(4u)+ii];
       let dkc=G[ii]*kvt; let dks=G[U(4u)+ii]*kvt;
       let qi=rh*U(4u)+ii; let ki=rh*U(4u)+ii;
       h[qi]=WF(HF(qi)-AF(qbase+U(4u)+ii)*dqc+AF(qbase+ii)*dqs);
       i[ki]=WF(IF(ki)-BF(qbase+U(4u)+ii)*dkc+BF(qbase+ii)*dks);
     }}
     var dkv=0.0;
     for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){dkv=dkv+G[ff]*BF(qbase+ff);}}
     j[rh*U(5u)+d0]=WF(kap*dkv);
     k[rh]=WF(KF(rh)+dkv*vv);
     for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){G[ff]=EF(gamoff+p2+ff)*G[ff];}}
     if((tt%64u)==0u){for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){l[stbase+(cid*p2+ff)*U(5u)+d0]=WF(G[ff]);}}}
   }
 }
}

// Value-parallel legacy reverse scan.  Each invocation owns one
// (batch,head,value) column and writes only its own value/channel partials to
// nbuf, avoiding the serial 32-value loop in hybrid_k_backward_block.  The
// following fold reduces those resident partials in a deterministic f32 order.
fn hybrid_k_backward_fast(z:u32) {
 let bh=z/U(5u); let d0=z%U(5u); if(bh>=U(1u)*U(3u)){return;}
 let b0=bh/U(3u); let hh=bh%U(3u); let p2=2u*U(4u); let stride=(p2+1u)*U(5u);
 let nch=(U(2u)+63u)/64u; let stbase=bh*(nch+1u)*p2*U(5u); let gamoff=U(6u)+hh*65u*p2;
 let chbase=bh*U(2u)*stride; var G:array<f32,64>;
 for(var ff=0u;ff<64u;ff=ff+1u){G[ff]=0.0;if(ff<p2){l[stbase+(nch*p2+ff)*U(5u)+d0]=WF(0.0);}}
 for(var rev=0u;rev<U(2u);rev=rev+1u){
   let tt=U(2u)-1u-rev; let cid=tt/64u; let start=cid*64u;
   var S:array<f32,64>;
   for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){S[ff]=FF(stbase+(cid*p2+ff)*U(5u)+d0);}}
   for(var ss=0u;ss<64u;ss=ss+1u){if(start+ss<=tt){
     let rr=(b0*U(2u)+start+ss)*U(3u)+hh; let kp=DF(rr); let vv=CF(rr*U(5u)+d0);
     for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){S[ff]=EF(gamoff+p2+ff)*S[ff]+BF(rr*p2+ff)*(kp*vv);}}
   }}
   let rh=(b0*U(2u)+tt)*U(3u)+hh; let qbase=rh*p2; let dout=GF(rh*U(5u)+d0);
   let kap=DF(rh); let vv=CF(rh*U(5u)+d0); let kvt=kap*vv;
   for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){G[ff]=G[ff]+AF(qbase+ff)*dout;}}
   let base=chbase+tt*stride;
   for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){
     var part=0.0;
     if(ff<U(4u)){let dqc=dout*S[ff];let dkc=G[ff]*kvt;part=-AF(qbase+U(4u)+ff)*dqc+AF(qbase+ff)*(dout*S[U(4u)+ff]);}
     else{let ii=ff-U(4u);let dkc=G[ii]*kvt;let dks=G[U(4u)+ii]*kvt;part=-BF(qbase+U(4u)+ii)*dkc+BF(qbase+ii)*dks;}
     nbuf[base+ff*U(5u)+d0]=WF(part);
   }}
   var dkv=0.0;for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){dkv=dkv+G[ff]*BF(qbase+ff);}}
   j[rh*U(5u)+d0]=WF(kap*dkv);
   nbuf[base+p2*U(5u)+d0]=WF(dkv*vv);
   for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){G[ff]=EF(gamoff+p2+ff)*G[ff];}}
   if((tt%64u)==0u){for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){l[stbase+(cid*p2+ff)*U(5u)+d0]=WF(G[ff]);}}}
 }
}

fn hybrid_k_fold_fast(z:u32) {
 let total=U(1u)*U(2u)*U(3u); if(z>=total){return;}
 let row=z/U(3u); let hh=z%U(3u); let b0=row/U(2u); let tt=row%U(2u); let bh=b0*U(3u)+hh;
 let p2=2u*U(4u); let stride=(p2+1u)*U(5u); let base=bh*U(2u)*stride+tt*stride; let angle=row*U(3u)*U(4u)+hh*U(4u); let beta=F(6u);
 for(var ii=0u;ii<32u;ii=ii+1u){if(ii<U(4u)){
   var sq=0.0;var sk=0.0;
   for(var d0=0u;d0<U(5u);d0=d0+1u){sq=sq+AF(base+ii*U(5u)+d0);sk=sk+AF(base+(U(4u)+ii)*U(5u)+d0);}
   b[angle+ii]=WF(beta*BF(angle+ii)+sq);c[angle+ii]=WF(beta*CF(angle+ii)+sk);
 }}
 var sb=0.0;for(var d0=0u;d0<U(5u);d0=d0+1u){sb=sb+AF(base+p2*U(5u)+d0);}
 d[row*U(3u)+hh]=WF(beta*DF(row*U(3u)+hh)+sb);
}

// Block-parallel legacy reverse scan.  This is algebraically the same
// recurrence as hybrid_k_backward, but each invocation owns a disjoint block
// of 32 value channels.  Angle and κ reductions are accumulated into one
// deterministic partial row per (batch, head, block, token), then folded by
// phase_delta_fold.  The old serial function remains available for tiny seam
// callers that do not provide the optional partial arena.
fn hybrid_k_backward_block(z:u32) {
 let nb=(U(5u)+31u)/32u; let bh=z/nb; let block=z%nb; if(bh>=U(1u)*U(3u)){return;}
 let b0=bh/U(3u); let hh=bh%U(3u); let p2=2u*U(4u); let width=1u+U(4u)*2u;
 let nch=(U(2u)+63u)/64u; let stbase=bh*(nch+1u)*p2*U(5u); let gamoff=U(6u)+hh*65u*p2;
 // Each block owns a complete row for every token; clear it before adding
 // the value-channel contributions below.  No float atomics are needed.
 for(var tt0=0u;tt0<U(2u);tt0=tt0+1u){
   let base=((bh*nb+block)*U(2u)+tt0)*width;
   for(var q=0u;q<width;q=q+1u){m[base+q]=0u;}
 }
 for(var d0=block*32u;d0<min((block+1u)*32u,U(5u));d0=d0+1u){
   var G:array<f32,64>; for(var ff=0u;ff<64u;ff=ff+1u){G[ff]=0.0;}
   // The terminal state has no successor contribution.  The command writes
   // every owned value column, matching the serial operator's overwrite.
   for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){l[stbase+(nch*p2+ff)*U(5u)+d0]=WF(0.0);}}
   for(var rev=0u;rev<U(2u);rev=rev+1u){
     let tt=U(2u)-1u-rev; let cid=tt/64u; let start=cid*64u;
     var S:array<f32,64>;
     for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){S[ff]=FF(stbase+(cid*p2+ff)*U(5u)+d0);}}
     for(var ss=0u;ss<64u;ss=ss+1u){if(start+ss<=tt){
       let rr=(b0*U(2u)+start+ss)*U(3u)+hh; let kap=DF(rr); let vv=CF(rr*U(5u)+d0);
       for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){S[ff]=EF(gamoff+p2+ff)*S[ff]+BF(rr*p2+ff)*(kap*vv);}}
     }}
     let rh=(b0*U(2u)+tt)*U(3u)+hh; let qbase=rh*p2; let dout=GF(rh*U(5u)+d0);
     let kap=DF(rh); let vv=CF(rh*U(5u)+d0); let kvt=kap*vv;
     for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){G[ff]=G[ff]+AF(qbase+ff)*dout;}}
     let base=((bh*nb+block)*U(2u)+tt)*width;
     for(var ii=0u;ii<32u;ii=ii+1u){if(ii<U(4u)){
       let dqc=dout*S[ii]; let dqs=dout*S[U(4u)+ii];
       let dkc=G[ii]*kvt; let dks=G[U(4u)+ii]*kvt;
       m[base+1u+ii]=WF(MF(base+1u+ii)-AF(qbase+U(4u)+ii)*dqc+AF(qbase+ii)*dqs);
       m[base+1u+U(4u)+ii]=WF(MF(base+1u+U(4u)+ii)-BF(qbase+U(4u)+ii)*dkc+BF(qbase+ii)*dks);
     }}
     var dkv=0.0;
     for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){dkv=dkv+G[ff]*BF(qbase+ff);}}
     j[rh*U(5u)+d0]=WF(kap*dkv);
     m[base]=WF(MF(base)+dkv*vv);
     for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){G[ff]=EF(gamoff+p2+ff)*G[ff];}}
     if((tt%64u)==0u){for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){l[stbase+(cid*p2+ff)*U(5u)+d0]=WF(G[ff]);}}}
   }
 }
}

// One invocation owns one (batch,head,value) column for Phase-Delta.  This
// is the same causal recurrence as the Metal lane, with chunk-boundary state
// checkpoints retained for continuation/telemetry.  The host dispatches only
// these column owners when parameter 7 selects Phase-Delta.
fn phase_delta_forward(z:u32) {
 let bh=z/U(5u); let dd=z%U(5u); if(bh>=U(1u)*U(3u)){return;}
 let b0=bh/U(3u); let hh=bh%U(3u); let p2=2u*U(4u); let nch=(U(2u)+63u)/64u;
 let stbase=bh*(nch+1u)*p2*U(5u); let gamoff=U(6u)+hh*65u*p2;
 var S:array<f32,64>;
 for(var ff=0u;ff<64u;ff=ff+1u){S[ff]=FF(stbase+ff*U(5u)+dd);}
 for(var tt=0u;tt<U(2u);tt=tt+1u){
   if((tt%64u)==0u){for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){f[stbase+((tt/64u)*p2+ff)*U(5u)+dd]=WF(S[ff]);}}}
   let rr=(b0*U(2u)+tt)*U(3u)+hh; let vv=CF(rr*U(5u)+dd); let kap=DF(rr);
   var rsum=0.0;
   for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){rsum=rsum+BF(rr*p2+ff)*EF(gamoff+p2+ff)*S[ff];}}
   for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){S[ff]=EF(gamoff+p2+ff)*S[ff]+kap*BF(rr*p2+ff)*(vv-rsum);}}
   var outv=0.0;
   for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){outv=outv+AF(rr*p2+ff)*S[ff];}}
   g[rr*U(5u)+dd]=WF(outv);
 }
 for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){f[stbase+(nch*p2+ff)*U(5u)+dd]=WF(S[ff]);}}
}

// Replay one 64-token chunk from its forward boundary into the bounded
// scratch arena.  The backward block shader consumes this arena immediately;
// host-side command ordering makes descending chunks deterministic.
fn phase_delta_replay(z:u32) {
 let bh=z/U(5u); let dd=z%U(5u); if(bh>=U(1u)*U(3u)){return;}
 let b0=bh/U(3u); let hh=bh%U(3u); let p2=2u*U(4u); let cid=U(7u);
 let start=U(8u); let clen=U(9u); let nch=(U(2u)+63u)/64u;
 let stbase=bh*(nch+1u)*p2*U(5u); let chbase=bh*65u*p2*U(5u);
 let gamoff=U(6u)+hh*65u*p2; var S:array<f32,64>;
 for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){S[ff]=EF(stbase+(cid*p2+ff)*U(5u)+dd);}}
 for(var jj=0u;jj<64u;jj=jj+1u){if(jj<clen){
   for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){f[chbase+(jj*p2+ff)*U(5u)+dd]=WF(S[ff]);}}
   let rr=(b0*U(2u)+start+jj)*U(3u)+hh;
   var rsum=0.0; let vv=BF(rr*U(5u)+dd); let kap=CF(rr);
   for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){rsum=rsum+AF(rr*p2+ff)*DF(gamoff+p2+ff)*S[ff];}}
   for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){S[ff]=DF(gamoff+p2+ff)*S[ff]+kap*AF(rr*p2+ff)*(vv-rsum);}}
 }}
 for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){f[chbase+(clen*p2+ff)*U(5u)+dd]=WF(S[ff]);}}
}

// Exact reverse recurrence for one 32-value block.  q/k angle and κ
// reductions are written to disjoint partial rows; the fold below owns the
// final deterministic reduction.  This is deliberately scalar WGSL (rather
// than a CPU wrapper) so the same resident graph works on non-Metal Vulkan.
fn phase_delta_backward_block(z:u32) {
 let nb=(U(5u)+31u)/32u; let bh=z/nb; let block=z%nb; if(bh>=U(1u)*U(3u)){return;}
 let b0=bh/U(3u); let hh=bh%U(3u); let p2=2u*U(4u); let cid=U(7u); let start=U(8u); let clen=U(9u);
 let width=1u+2u*U(4u); let nch=(U(2u)+63u)/64u; let chbase=bh*65u*p2*U(5u); let stbase=bh*(nch+1u)*p2*U(5u); let gamoff=U(6u)+hh*65u*p2;
 for(var jj=0u;jj<64u;jj=jj+1u){if(jj<clen){
   let base=((bh*nb+block)*U(2u)+start+jj)*width;
   for(var q=0u;q<width;q=q+1u){h[base+q]=0u;}
 }}
 for(var d0=block*32u;d0<min((block+1u)*32u,U(5u));d0=d0+1u){
   // The host launches chunks in descending order. Carry the adjoint of the
   // chunk entry state through the resident boundary arena so later-chunk
   // gradients are not dropped at every 64-token boundary.
   var G:array<f32,64>; for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){G[ff]=JF(stbase+((cid+1u)*p2+ff)*U(5u)+d0);}}
   for(var rev=0u;rev<64u;rev=rev+1u){if(rev<clen){
     let jj=clen-1u-rev; let rr=(b0*U(2u)+start+jj)*U(3u)+hh;
     let qbase=rr*p2; let dout=FF(rr*U(5u)+d0); let vv=CF(rr*U(5u)+d0); let kap=DF(rr);
     var rsum=0.0;
     for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){let sprev=GF(chbase+(jj*p2+ff)*U(5u)+d0);rsum=rsum+BF(qbase+ff)*EF(gamoff+p2+ff)*sprev;}}
     let e0=vv-rsum;
     for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){G[ff]=G[ff]+AF(qbase+ff)*dout;}}
     var u0=0.0; var dbeta=0.0;
     for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){let kf=BF(qbase+ff);u0=u0+G[ff]*kf;dbeta=dbeta+G[ff]*kf*e0;}}
     let base=((bh*nb+block)*U(2u)+start+jj)*width;
     h[base]=WF(HF(base)+dbeta);
     for(var ii=0u;ii<32u;ii=ii+1u){if(ii<U(4u)){
       let f0=ii; let f1=U(4u)+ii;
       let sprev0=GF(chbase+(jj*p2+f0)*U(5u)+d0); let sprev1=GF(chbase+(jj*p2+f1)*U(5u)+d0);
       let scur0=GF(chbase+((jj+1u)*p2+f0)*U(5u)+d0); let scur1=GF(chbase+((jj+1u)*p2+f1)*U(5u)+d0);
       let qgc0=dout*scur0; let qgs0=dout*scur1;
       let qtc0=-AF(qbase+U(4u)+ii)*qgc0+AF(qbase+ii)*qgs0;
       let gkc0=G[ii]*kap*e0-kap*u0*EF(gamoff+p2+ii)*sprev0;
       let gks0=G[U(4u)+ii]*kap*e0-kap*u0*EF(gamoff+p2+U(4u)+ii)*sprev1;
       let kt0=-BF(qbase+U(4u)+ii)*gkc0+BF(qbase+ii)*gks0;
       h[base+1u+ii]=WF(HF(base+1u+ii)+qtc0);
       h[base+1u+U(4u)+ii]=WF(HF(base+1u+U(4u)+ii)+kt0);
     }}
     for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){G[ff]=EF(gamoff+p2+ff)*(G[ff]-kap*u0*BF(qbase+ff));}}
     i[rr*U(5u)+d0]=WF(kap*u0);
   }}
   for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){j[stbase+(cid*p2+ff)*U(5u)+d0]=WF(G[ff]);}}
 }
}

fn phase_delta_fold(z:u32) {
 let total=U(1u)*U(2u)*U(3u); if(z>=total){return;} let row=z/U(3u); let hh=z%U(3u);
 let nb=(U(5u)+31u)/32u; let width=1u+2u*U(4u); let b0=row/U(2u); let tt=row%U(2u); let bh=b0*U(3u)+hh; let angle=(row*U(3u)+hh)*U(4u); let beta=F(7u); var sb=0.0;
 for(var block=0u;block<32u;block=block+1u){if(block<nb){let base=((bh*nb+block)*U(2u)+tt)*width;sb=sb+AF(base);}}
 for(var ii=0u;ii<32u;ii=ii+1u){if(ii<U(4u)){var sq=0.0;var sk=0.0;for(var block=0u;block<32u;block=block+1u){if(block<nb){let base=((bh*nb+block)*U(2u)+tt)*width;sq=sq+AF(base+1u+ii);sk=sk+AF(base+1u+U(4u)+ii);}}b[angle+ii]=WF(beta*BF(angle+ii)+sq);c[angle+ii]=WF(beta*CF(angle+ii)+sk);}}
 d[row*U(3u)+hh]=WF(sb);
}

// Shared tiles for the optional GEMM path.  Each workgroup computes one
// 16x16 output tile; all lanes still accumulate their own output in the same
// k order as the scalar reference, so this is a scheduling optimization rather
// than a change in reduction semantics.
var<workgroup> gemm_tile_a: array<f32,256>;
var<workgroup> gemm_tile_b: array<f32,256>;

// Register-tiled GEMM (a separate entry point; op 1's parameter block,
// bindings 0..2 = A, B, C). One workgroup = one 64×64 C tile of one batch
// entry; 256 lanes, lane (ty, tx) owns the 4×4 micro-tile at rows ty*4..,
// cols tx*4..; K in 32-wide steps through 16 KB of shared A[64][32] /
// B[32][64]. Every output accumulates its products in ascending k like the
// scalar and 16×16 paths. Host guarantees M%64, N%64, K%32 and no dynamic
// K / indirect / table mode.
var<workgroup> g64_a: array<f32,2048>;
var<workgroup> g64_b: array<f32,2048>;

// ---------------------------------------------------------------------
// GDN mixer token scan (plan S7) — the port of shaders.metal
// `gdn_scan_fwd_f32` / `gdn_scan_bwd_f32`: one 128-lane workgroup per
// (sequence, head), lane = state row / output column, running state in
// storage (`live`), one checkpoint per 64 tokens (`states`, slot 0 = S_0),
// the backward replays each chunk into `chunk` and walks it in reverse.
// Every reduction is summed by lane 0 in lane order (bit-exact, same
// arithmetic as the Metal kernel).
// fwd slots: a=qkv_cv b=a_pre c=b_pre d=params(alog@U9, dt@U10) e=raw_o f=states g=live
// bwd slots: a=qkv_cv b=a_pre c=b_pre d=params e=states f=doo g=chunk h=dlive i=dcv j=da k=db l=part
// params: U1=B U2=T U3=nv U4=dk U5=dv U6=c_dim U7=ab_ld U8=flags
// ---------------------------------------------------------------------
var<workgroup> gs_q: array<f32,128>;
var<workgroup> gs_k: array<f32,128>;
var<workgroup> gs_v: array<f32,128>;
var<workgroup> gs_do: array<f32,128>;
var<workgroup> gs_u: array<f32,128>;
var<workgroup> gs_dkv: array<f32,128>;
var<workgroup> gs_ra: array<f32,128>;
var<workgroup> gs_rb: array<f32,128>;
fn gdn_softplus(x: f32) -> f32 { if (x > 20.0) { return x; } return log(1.0 + exp(x)); }

@compute @workgroup_size(128)
fn gdn_fwd(@builtin(local_invocation_id) lidv: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
 let tid=lidv.x; let tg=wg.x+wg.y*nwg.x;
 let B=U(1u); let T=U(2u); let nv=U(3u); let dk=U(4u); let dv=U(5u); let cd=U(6u); let ld=U(7u); let flags=U(8u);
 if(tg>=B*nv){return;}
 let bi=tg/nv; let hh=tg%nv; let nch=(T+63u)/64u; let ss=dk*dv;
 let sbase=tg*ss; let ckbase=tg*(nch+1u)*ss;
 let ea=exp(DF(U(9u)+hh)); let dt_h=DF(U(10u)+hh); let sdk=inverseSqrt(f32(dk));
 if(tid<dv){
   if((flags&1u)!=0u){for(var ii=0u;ii<dk;ii=ii+1u){g[sbase+ii*dv+tid]=f[ckbase+ii*dv+tid];}}
   else{for(var ii=0u;ii<dk;ii=ii+1u){g[sbase+ii*dv+tid]=WF(0.0);f[ckbase+ii*dv+tid]=WF(0.0);}}
 }
 for(var ti=0u;ti<T;ti=ti+1u){
   let row=bi*T+ti; let base=row*cd;
   if((ti%64u)==0u && ti>0u && tid<dv){let ck0=ckbase+(ti/64u)*ss;for(var ii=0u;ii<dk;ii=ii+1u){f[ck0+ii*dv+tid]=g[sbase+ii*dv+tid];}}
   var qq=0.0; var kk=0.0;
   if(tid<dk){let qv=AF(base+hh*dk+tid);let kv=AF(base+nv*dk+hh*dk+tid);gs_q[tid]=qv;gs_k[tid]=kv;qq=qv*qv;kk=kv*kv;}
   if(tid<dv){gs_v[tid]=AF(base+2u*nv*dk+hh*dv+tid);}
   gs_ra[tid]=qq; gs_rb[tid]=kk;
   workgroupBarrier();
   if(tid==0u){var sa=0.0;var sb=0.0;for(var ii=0u;ii<dk;ii=ii+1u){sa=sa+gs_ra[ii];sb=sb+gs_rb[ii];}gs_ra[0]=sa;gs_rb[0]=sb;}
   workgroupBarrier();
   let qn=gs_ra[0]; let kn=gs_rb[0];
   workgroupBarrier();
   let iq=inverseSqrt(qn+1.0e-6)*sdk; let ik=inverseSqrt(kn+1.0e-6);
   let aa=BF(row*ld+hh)+dt_h; let sp=gdn_softplus(aa); let gg=exp(-ea*sp);
   let beta=select(1.0/(1.0+exp(-CF(row*ld+hh))),1.0,(flags&2u)!=0u);
   if(tid<dv){
     var kv=0.0; for(var ii=0u;ii<dk;ii=ii+1u){kv=kv+GF(sbase+ii*dv+tid)*gs_k[ii];} kv=kv*ik;
     let u=beta*(gs_v[tid]-gg*kv); var o=0.0;
     for(var ii=0u;ii<dk;ii=ii+1u){let cc=gg*GF(sbase+ii*dv+tid)+(gs_k[ii]*ik)*u;g[sbase+ii*dv+tid]=WF(cc);o=o+(gs_q[ii]*iq)*cc;}
     e[row*(nv*dv)+hh*dv+tid]=WF(o);
   }
   workgroupBarrier();
 }
 if(tid<dv){let ck0=ckbase+nch*ss;for(var ii=0u;ii<dk;ii=ii+1u){f[ck0+ii*dv+tid]=g[sbase+ii*dv+tid];}}
}

@compute @workgroup_size(128)
fn gdn_bwd(@builtin(local_invocation_id) lidv: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
 let tid=lidv.x; let tg=wg.x+wg.y*nwg.x;
 let B=U(1u); let T=U(2u); let nv=U(3u); let dk=U(4u); let dv=U(5u); let cd=U(6u); let ld=U(7u); let flags=U(8u);
 if(tg>=B*nv){return;}
 let bi=tg/nv; let hh=tg%nv; let nch=(T+63u)/64u; let ss=dk*dv;
 let chbase=tg*65u*ss; let dsbase=tg*ss; let ckbase=tg*(nch+1u)*ss;
 let ea=exp(DF(U(9u)+hh)); let dt_h=DF(U(10u)+hh); let sdk=inverseSqrt(f32(dk));
 if((flags&1u)==0u && tid<dv){for(var ii=0u;ii<dk;ii=ii+1u){h[dsbase+ii*dv+tid]=WF(0.0);}}
 var p_alog=0.0; var p_dt=0.0;
 for(var cc0=nch;cc0>0u;cc0=cc0-1u){
   let c0=cc0-1u; let start=c0*64u; let clen=min(64u,T-start);
   if(tid<dv){for(var ii=0u;ii<dk;ii=ii+1u){g[chbase+ii*dv+tid]=e[ckbase+c0*ss+ii*dv+tid];}}
   storageBarrier(); workgroupBarrier();
   for(var jj=0u;jj<clen;jj=jj+1u){
     let row=bi*T+start+jj; let base=row*cd;
     var qq=0.0; var kk=0.0;
     if(tid<dk){let qv=AF(base+hh*dk+tid);let kv=AF(base+nv*dk+hh*dk+tid);gs_q[tid]=qv;gs_k[tid]=kv;qq=qv*qv;kk=kv*kv;}
     if(tid<dv){gs_v[tid]=AF(base+2u*nv*dk+hh*dv+tid);}
     gs_ra[tid]=qq; gs_rb[tid]=kk;
     workgroupBarrier();
     if(tid==0u){var sa=0.0;var sb=0.0;for(var ii=0u;ii<dk;ii=ii+1u){sa=sa+gs_ra[ii];sb=sb+gs_rb[ii];}gs_ra[0]=sa;gs_rb[0]=sb;}
     workgroupBarrier();
     let kn=gs_rb[0];
     workgroupBarrier();
     let ik=inverseSqrt(kn+1.0e-6);
     let aa=BF(row*ld+hh)+dt_h; let sp=gdn_softplus(aa); let gg=exp(-ea*sp);
     let beta=select(1.0/(1.0+exp(-CF(row*ld+hh))),1.0,(flags&2u)!=0u);
     let sp0=chbase+jj*ss; let sc0=chbase+(jj+1u)*ss;
     if(tid<dv){
       var kv=0.0; for(var ii=0u;ii<dk;ii=ii+1u){kv=kv+GF(sp0+ii*dv+tid)*gs_k[ii];} kv=kv*ik;
       let u=beta*(gs_v[tid]-gg*kv);
       for(var ii=0u;ii<dk;ii=ii+1u){g[sc0+ii*dv+tid]=WF(gg*GF(sp0+ii*dv+tid)+(gs_k[ii]*ik)*u);}
     }
     workgroupBarrier();
   }
   for(var jr=clen;jr>0u;jr=jr-1u){
     let jj=jr-1u;
     let row=bi*T+start+jj; let base=row*cd;
     var qq=0.0; var kk=0.0;
     if(tid<dk){let qv=AF(base+hh*dk+tid);let kv=AF(base+nv*dk+hh*dk+tid);gs_q[tid]=qv;gs_k[tid]=kv;qq=qv*qv;kk=kv*kv;}
     if(tid<dv){gs_v[tid]=AF(base+2u*nv*dk+hh*dv+tid);gs_do[tid]=FF(row*(nv*dv)+hh*dv+tid);}
     gs_ra[tid]=qq; gs_rb[tid]=kk;
     workgroupBarrier();
     if(tid==0u){var sa=0.0;var sb=0.0;for(var ii=0u;ii<dk;ii=ii+1u){sa=sa+gs_ra[ii];sb=sb+gs_rb[ii];}gs_ra[0]=sa;gs_rb[0]=sb;}
     workgroupBarrier();
     let qn=gs_ra[0]; let kn=gs_rb[0];
     workgroupBarrier();
     let iq=inverseSqrt(qn+1.0e-6)*sdk; let ik=inverseSqrt(kn+1.0e-6);
     let aa=BF(row*ld+hh)+dt_h; let sp=gdn_softplus(aa); let sig=1.0/(1.0+exp(-aa)); let gg=exp(-ea*sp);
     let beta=select(1.0/(1.0+exp(-CF(row*ld+hh))),1.0,(flags&2u)!=0u);
     let sp0=chbase+jj*ss; let sc0=chbase+(jj+1u)*ss;
     // A (column): kv, u; dS += q̂ ⊗ do
     var kvc=0.0;
     if(tid<dv){
       for(var ii=0u;ii<dk;ii=ii+1u){kvc=kvc+GF(sp0+ii*dv+tid)*gs_k[ii];} kvc=kvc*ik*gg;
       gs_u[tid]=beta*(gs_v[tid]-kvc);
       let dd=gs_do[tid];
       for(var ii=0u;ii<dk;ii=ii+1u){h[dsbase+ii*dv+tid]=WF(HF(dsbase+ii*dv+tid)+(gs_q[ii]*iq)*dd);}
     }
     // B (row): dq̂ = S_t·do
     var dqh=0.0;
     if(tid<dk){for(var jx=0u;jx<dv;jx=jx+1u){dqh=dqh+GF(sc0+tid*dv+jx)*gs_do[jx];}}
     storageBarrier(); workgroupBarrier();
     // C (column): du, dv, dkv, dβ partial
     var bpart=0.0;
     if(tid<dv){
       var du=0.0; for(var ii=0u;ii<dk;ii=ii+1u){du=du+HF(dsbase+ii*dv+tid)*(gs_k[ii]*ik);}
       i[base+2u*nv*dk+hh*dv+tid]=WF(beta*du);
       gs_dkv[tid]=-beta*du;
       bpart=du*(gs_v[tid]-kvc);
     }
     storageBarrier(); workgroupBarrier();
     // D (row): dk̂, dg partial, dS ← g·(dS + k̂ ⊗ dkv)
     var dkh=0.0; var gpart=0.0;
     if(tid<dk){
       let kf=gs_k[tid]*ik;
       for(var jx=0u;jx<dv;jx=jx+1u){
         let sprev=GF(sp0+tid*dv+jx); let dsv=HF(dsbase+tid*dv+jx); let dspre=dsv+kf*gs_dkv[jx];
         dkh=dkh+dsv*gs_u[jx]+(gg*sprev)*gs_dkv[jx]; gpart=gpart+dspre*sprev;
         h[dsbase+tid*dv+jx]=WF(gg*dspre);
       }
     }
     gs_ra[tid]=bpart; gs_rb[tid]=gpart;
     workgroupBarrier();
     if(tid==0u){var sa=0.0;var sb=0.0;for(var ii=0u;ii<128u;ii=ii+1u){sa=sa+gs_ra[ii];sb=sb+gs_rb[ii];}gs_ra[0]=sa;gs_rb[0]=sb;}
     workgroupBarrier();
     let dbeta=gs_ra[0]; let dg=gs_rb[0];
     workgroupBarrier();
     var pq=0.0; var pk=0.0;
     if(tid<dk){pq=dqh*gs_q[tid];pk=dkh*gs_k[tid];}
     gs_ra[tid]=pq; gs_rb[tid]=pk;
     workgroupBarrier();
     if(tid==0u){var sa=0.0;var sb=0.0;for(var ii=0u;ii<dk;ii=ii+1u){sa=sa+gs_ra[ii];sb=sb+gs_rb[ii];}gs_ra[0]=sa;gs_rb[0]=sb;}
     workgroupBarrier();
     let dqdot=gs_ra[0]; let dkdot=gs_rb[0];
     workgroupBarrier();
     if(tid<dk){
       i[base+hh*dk+tid]=WF(iq*dqh-gs_q[tid]*iq*iq*iq*f32(dk)*dqdot);
       i[base+nv*dk+hh*dk+tid]=WF(ik*dkh-gs_k[tid]*ik*ik*ik*dkdot);
     }
     if(tid==0u){
       let d_a=dg*(-ea*sig*gg);
       j[row*ld+hh]=WF(d_a);
       k[row*ld+hh]=WF(dbeta*beta*(1.0-beta));
       p_alog=p_alog+dg*(-ea*sp*gg);
       p_dt=p_dt+d_a;
     }
     storageBarrier(); workgroupBarrier();
   }
 }
 if(tid==0u){l[tg*2u]=WF(p_alog);l[tg*2u+1u]=WF(p_dt);}
}

@compute @workgroup_size(256)
fn gemm64(@builtin(local_invocation_id) lidv: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
 let lid=lidv.x; let group=wg.x+wg.y*nwg.x;
 let m0=U(2u);let n0=U(3u);let k0=U(4u);let ta=U(5u);let tb=U(6u);let lda=U(7u);let ldb=U(8u);let ldc=U(9u);let ao=U(10u);let bo=U(11u);let co=U(12u);let al=F(13u);let be=F(14u);let nb=U(15u);let nh=U(16u);let nc=U(17u);
 let tiles_n=n0/64u; let tiles_m=m0/64u; let per_batch=tiles_n*tiles_m;
 let batch_id=group/per_batch; if(batch_id>=nb*nh*nc){return;}
 let tile=group%per_batch; let tm=tile/tiles_n; let tn=tile%tiles_n;
 let bc=batch_id%nc;let bh=batch_id/nc;let ch=bh%nh;let cb=bh/nh;
 let oa=ao+cb*U(18u)+ch*U(19u)+bc*U(20u);let ob=bo+cb*U(21u)+ch*U(22u)+bc*U(23u);let oc=co+cb*U(24u)+ch*U(25u)+bc*U(26u);
 let r0=tm*64u; let c0=tn*64u; let ty=lid/16u; let tx=lid%16u;
 var acc:array<f32,16>; for(var i=0u;i<16u;i=i+1u){acc[i]=0.0;}
 for(var kb=0u;kb<k0;kb=kb+32u){
   for(var i=0u;i<8u;i=i+1u){let idx=lid+i*256u;let row=idx/32u;let kk=idx%32u;let gr=r0+row;let gk=kb+kk;let ia=select(oa+gr*lda+gk,oa+gk*lda+gr,ta==1u);g64_a[row*32u+kk]=AF(ia);}
   for(var i=0u;i<8u;i=i+1u){let idx=lid+i*256u;let kk=idx/64u;let col=idx%64u;let gk=kb+kk;let gc=c0+col;let ib=select(ob+gk*ldb+gc,ob+gc*ldb+gk,tb==1u);g64_b[kk*64u+col]=BF(ib);}
   workgroupBarrier();
   for(var kk=0u;kk<32u;kk=kk+1u){
     let a0=g64_a[(ty*4u)*32u+kk];let a1=g64_a[(ty*4u+1u)*32u+kk];let a2=g64_a[(ty*4u+2u)*32u+kk];let a3=g64_a[(ty*4u+3u)*32u+kk];
     let b0=g64_b[kk*64u+tx*4u];let b1=g64_b[kk*64u+tx*4u+1u];let b2=g64_b[kk*64u+tx*4u+2u];let b3=g64_b[kk*64u+tx*4u+3u];
     acc[0]=acc[0]+a0*b0;acc[1]=acc[1]+a0*b1;acc[2]=acc[2]+a0*b2;acc[3]=acc[3]+a0*b3;
     acc[4]=acc[4]+a1*b0;acc[5]=acc[5]+a1*b1;acc[6]=acc[6]+a1*b2;acc[7]=acc[7]+a1*b3;
     acc[8]=acc[8]+a2*b0;acc[9]=acc[9]+a2*b1;acc[10]=acc[10]+a2*b2;acc[11]=acc[11]+a2*b3;
     acc[12]=acc[12]+a3*b0;acc[13]=acc[13]+a3*b1;acc[14]=acc[14]+a3*b2;acc[15]=acc[15]+a3*b3;
   }
   workgroupBarrier();
 }
 for(var i=0u;i<4u;i=i+1u){for(var j=0u;j<4u;j=j+1u){
   let rr=r0+ty*4u+i; let cc=c0+tx*4u+j; let ix=oc+rr*ldc+cc;
   var outv=al*acc[i*4u+j]+be*CF(ix);
   if(U(27u)==1u && cc>rr){outv=0.0;}
   c[ix]=WF(outv);
 }}
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
 let z=id.x + id.y*U(63u)*256u; let op=U(0u); if(op==0u){return;}
 if(op==1u){ // batched row-major GEMM
   if(U(31u)==1u){
    let lid=id.x%256u;
    let tile_id=(id.x/256u)+id.y*U(34u);
    let per_batch=U(32u)*U(33u);
    let batch_id=tile_id/per_batch;
    // The grid is folded into y above 65535 workgroups, so up to 65534
    // tile ids past the last batch are dispatched: they must not run (in
    // table mode their offsets would come from the NEXT table section and
    // land inside live buffers). Uniform per workgroup, so no barrier splits.
    if(batch_id>=U(15u)*U(16u)*U(17u)){return;}
    let tile=tile_id%per_batch;
    let rr=(tile/U(32u))*16u+lid/16u;
    let cc=(tile%U(32u))*16u+lid%16u;
    let m0=U(2u);let n0=U(3u);let k0=U(4u);let ta=U(5u);let tb=U(6u);let lda=U(7u);let ldb=U(8u);let ldc=U(9u);let ao=U(10u);let bo=U(11u);let co=U(12u);let al=F(13u);let be=F(14u);let nb=U(15u);let nh=U(16u);let nc=U(17u);
    let bc=batch_id%nc;let bh=batch_id/nc;let ch=bh%nh;let cb=bh/nh;
    var oa=ao+cb*U(18u)+ch*U(19u)+bc*U(20u);var ob=bo+cb*U(21u)+ch*U(22u)+bc*U(23u);var oc=co+cb*U(24u)+ch*U(25u)+bc*U(26u);
    var kdim=k0; var mrows=m0;
    if(U(28u)==1u){kdim=min(((DU(U(29u)+batch_id)+63u)/64u)*64u,k0);}
    // table-batched (p[35]): per-batch offsets and a row count bounding M or K
    if(U(35u)==1u){let tq=U(36u)+batch_id*4u;oa=oa+DU(tq);ob=ob+DU(tq+1u);oc=oc+DU(tq+2u);let rz=DU(tq+3u);if(U(37u)==1u){kdim=min(rz,k0);}else{mrows=min(rz,m0);}}
    var sum=0.0;
    for(var kb=0u;kb<kdim;kb=kb+16u){
      let aq=kb+lid%16u; let bk=kb+lid/16u;
      var av=0.0; var bv=0.0;
      if(rr<mrows && aq<kdim){let ia=select(oa+rr*lda+aq,oa+aq*lda+rr,ta==1u);av=AF(ia);}
      if(bk<kdim && cc<n0){let ib=select(ob+bk*ldb+cc,ob+cc*ldb+bk,tb==1u);bv=BF(ib);}
      gemm_tile_a[lid]=av;gemm_tile_b[lid]=bv;
      workgroupBarrier();
      if(rr<mrows && cc<n0){for(var q=0u;q<16u;q=q+1u){let kk=kb+q;if(kk<kdim){let av0=gemm_tile_a[(lid/16u)*16u+q];let bv0=gemm_tile_b[q*16u+(lid%16u)];sum=sum+av0*bv0;}}}
      workgroupBarrier();
    }
    if(rr<mrows && cc<n0){
      let out_ix=oc+rr*ldc+cc;
      var outv=al*sum+be*CF(out_ix);
      if(U(27u)==1u && cc>rr){outv=0.0;}
      // Indirect expert records store (x, y, z); y is the active-row count
      // used by the scalar contract, hence the fixed +1 offset here.
      if(U(30u)==1u && rr>=min(m0,DU(U(29u)+1u)*64u)){if(be==0.0){c[out_ix]=0u;}}
      else{c[out_ix]=WF(outv);}
    }
    return;
   }
   if(z>=U(1u)){return;} let m0=U(2u);let n0=U(3u);let k0=U(4u);let ta=U(5u);let tb=U(6u);let lda=U(7u);let ldb=U(8u);let ldc=U(9u);let ao=U(10u);let bo=U(11u);let co=U(12u);let al=F(13u);let be=F(14u);let nb=U(15u);let nh=U(16u);let nc=U(17u);let one=m0*n0;let zz=z/one;let rr=(z%one)/n0;let cc=z%n0;let bc=zz%nc;let bh=zz/nc;let ch=bh%nh;let cb=bh/nh;var oa=ao+cb*U(18u)+ch*U(19u)+bc*U(20u);var ob=bo+cb*U(21u)+ch*U(22u)+bc*U(23u);var oc=co+cb*U(24u)+ch*U(25u)+bc*U(26u);if(U(30u)==1u && rr>=min(m0,DU(U(29u)+1u)*64u)){if(be==0.0){c[oc+rr*ldc+cc]=0u;}return;}var kdim=k0;if(U(28u)==1u){kdim=min(((DU(U(29u)+zz)+63u)/64u)*64u,k0);}if(U(35u)==1u){let tq=U(36u)+zz*4u;oa=oa+DU(tq);ob=ob+DU(tq+1u);oc=oc+DU(tq+2u);let rz=DU(tq+3u);if(U(37u)==1u){kdim=min(rz,k0);}else if(rr>=rz){return;}}var sum=0.0;for(var q=0u;q<kdim;q=q+1u){let ia=select(oa+rr*lda+q,oa+q*lda+rr,ta==1u);let ib=select(ob+q*ldb+cc,ob+cc*ldb+q,tb==1u);sum=sum+AF(ia)*BF(ib);}var outv=al*sum+be*CF(oc+rr*ldc+cc);if(U(27u)==1u && cc>rr){outv=0.0;}c[oc+rr*ldc+cc]=WF(outv);return;
 }
 if(op==2u){if(z<U(1u)){b[z]=WF(F(2u)*AF(z)+F(3u)*BF(z));}return;}
 if(op==3u){if(z>=U(1u)){return;}let q=U(2u)+z;let lr=F(3u);let b1=F(4u);let b2=F(5u);let ep=F(6u);let wd=F(7u);let bc1=F(8u);let bc2=F(9u);let gs=F(10u);let gg=BF(q)*gs;let mm=b1*CF(q)+(1.0-b1)*gg;let vv=b2*DF(q)+(1.0-b2)*gg*gg;c[q]=WF(mm);d[q]=WF(vv);a[q]=WF((1.0-lr*wd)*AF(q)-lr*(mm*bc1/(sqrt(vv*bc2)+ep)));return;}
 // Owner z reduces every 256-element tile t = z, z+groups, z+2*groups, ...
 // (grid-stride over tiles, the same discipline as Metal `sumsq_f32`).
 // The host clamps `groups` to the partial buffer, so without the stride a
 // 56M-element arena was reduced only over its first groups*256 = 1M
 // elements (1.87%) and the printed grad norm, the clip and the non-finite
 // guard were blind to the rest.  Every tail element (including a short
 // final tile) is covered exactly once.
 if(op==4u){let groups=U(3u);if(z>=groups){return;}let n=U(1u);let base=U(2u);var s=0.0;for(var t=z*256u;t<n;t=t+groups*256u){let end=min(t+256u,n);for(var q=base+t;q<base+end;q=q+1u){let v=AF(q);s=s+v*v;}}b[U(4u)+z]=WF(s);return;}
 if(op==5u){let row=z;if(row>=U(1u)){return;}let dd=U(2u);let wo=U(3u);var s=0.0;for(var q=0u;q<dd;q=q+1u){let x=AF(row*dd+q);s=s+x*x;}let inv=1.0/sqrt(s/f32(dd)+F(4u));d[row]=WF(inv);for(var r0=0u;r0<dd;r0=r0+1u){c[row*dd+r0]=WF(AF(row*dd+r0)*BF(wo+r0)*inv);}return;}
 if(op==6u){let dd=U(2u);let rows=U(1u);let q=z;if(q>=rows*dd){return;}let row=q/dd;let col=q%dd;let inv=DF(row);let x=AF(row*dd+col);let wv=BF(U(3u)+col);var dot=0.0;for(var j0=0u;j0<dd;j0=j0+1u){dot=dot+CF(row*dd+j0)*BF(U(3u)+j0)*AF(row*dd+j0);}let c0=inv*inv*inv*dot/f32(dd);let val=inv*CF(q)*wv-c0*x;let old=EF(q);e[q]=WF(select(val,old*F(5u)+val,F(5u)!=0.0));if(z<dd){var sw=0.0;for(var r=0u;r<rows;r=r+1u){sw=sw+CF(r*dd+z)*AF(r*dd+z)*DF(r);}f[U(4u)+z]=WF(FF(U(4u)+z)+sw);}return;}
 if(op==7u){if(z<U(1u)){c[z]=WF(silu(AF(z))*BF(z));}return;}
 if(op==8u){if(z<U(1u)){let gg=AF(z);let uu=BF(z);let dh=CF(z);let sig=1.0/(1.0+exp(-gg));let sg=gg*sig;let ds=sig*(1.0+gg*(1.0-sig));d[z]=WF(dh*ds*uu);e[z]=WF(dh*sg);}return;}
 if(op==9u){if(z>=U(1u)*U(2u)){return;}let row=z/U(2u);let col=z%U(2u);let tok=b[row];c[z]=a[U(3u)+tok*U(2u)+col];return;}
 // op 10: tied-embedding scatter-add. p[4]=1: the legacy single-invocation loop
 // (bit-exact reference). p[4]=0: one owner per (first-occurrence row, column)
 // walks the op-58 chain (slot 3) in ascending row order from the current dE
 // value — the same f32 addition sequence, no atomics, no serial invocation.
 if(op==10u){
   if(U(4u)==1u){if(z!=0u){return;}for(var row=0u;row<U(1u);row=row+1u){let tok=b[row];if(tok!=0xffffffffu){for(var col=0u;col<U(2u);col=col+1u){let dst=U(3u)+tok*U(2u)+col;a[dst]=WF(AF(dst)+CF(row*U(2u)+col));}}}return;}
   let rows=U(1u);let dd=U(2u);if(z>=rows*dd){return;}let r=z/dd;let col=z%dd;let v=d[r];if((v&0x80000000u)==0u){return;}
   let tok=b[r];let dst=U(3u)+tok*dd+col;var acc=AF(dst);var cur=r;
   loop{acc=acc+CF(cur*dd+col);let nxt=d[cur]&0x7fffffffu;if(nxt==0x7fffffffu){break;}cur=nxt;}
   a[dst]=WF(acc);return;}
 // op 58: per-row token chain for op 10 (slot 1 = tokens, slot 3 = chain).
 if(op==58u){let rows=U(1u);if(z>=rows){return;}let tok=b[z];if(tok==0xffffffffu){d[z]=0u;return;}var first=1u;for(var r=0u;r<z;r=r+1u){if(b[r]==tok){first=0u;break;}}var nxt=0x7fffffffu;for(var r=z+1u;r<rows;r=r+1u){if(b[r]==tok){nxt=r;break;}}d[z]=(first<<31u)|nxt;return;}
 if(op==59u){if(z<U(1u)){b[z]=WF(silu(AF(z)));}return;}
 if(op==60u){if(z<U(1u)){let x=AF(z);let sg=1.0/(1.0+exp(-x));c[z]=WF(BF(z)*sg*(1.0+x*(1.0-sg)));}return;}
 // op 61: GDN scan fold — g[galog+h] += Σ_b part[(b·nv+h)·2], g[gdt+h] += Σ_b part[..+1]
 if(op==61u){if(z>=U(2u)){return;}var sa=0.0;var sd=0.0;for(var bb=0u;bb<U(1u);bb=bb+1u){sa=sa+AF((bb*U(2u)+z)*2u);sd=sd+AF((bb*U(2u)+z)*2u+1u);}b[U(3u)+z]=WF(BF(U(3u)+z)+sa);b[U(4u)+z]=WF(BF(U(4u)+z)+sd);return;}
 // ---- GDN chunked WY/UT form (gdn_wy.rs). Common params: U1=B U2=T U3=nv U4=dk U5=dv U6=c_dim U7=ab_ld U8=nch; U12=chunk.
 // op 79: strided block copy with an optional per-row mask (a=src b=dst c=mask): U1=nblk U2=len U3=src_off U4=src_stride U5=dst_off U6=dst_stride U7=mask_div U8=use_mask
 if(op==79u){if(z>=U(1u)*U(2u)){return;}let blk=z/U(2u);let ii=z%U(2u);var on=true;if(U(8u)==1u){on=CU(blk/U(7u))!=0u;}b[U(5u)+blk*U(6u)+ii]=select(WF(0.0),a[U(3u)+blk*U(4u)+ii],on);return;}
 // op 78: zero checkpoint slot 0 of every (b,h). a=states; U1=nb U2=slots U3=ss
 if(op==78u){if(z>=U(1u)*U(3u)){return;}let bh=z/U(3u);let ee=z%U(3u);a[(bh*U(2u))*U(3u)+ee]=WF(0.0);return;}
 // op 64: per-(b,h,chunk) scalars: lg (cumulative log α in the chunk), γ, β, ρ = γ_end/γ_j. a=a_pre b=b_pre c=p(alog@U9,dt@U10) d=lg e=gam f=bet g=rho; U11=β≡1
 if(op==64u){let nch=U(8u);if(z>=U(1u)*U(3u)*nch){return;}let bh=z/nch;let cc=z%nch;let bb=bh/U(3u);let hh=bh%U(3u);let T=U(2u);let ld=U(7u);
   let ea=exp(CF(U(9u)+hh));let dt=CF(U(10u)+hh);var acc=0.0;
   for(var ii=0u;ii<64u;ii=ii+1u){let t=cc*64u+ii;let row=bb*T+t;let aa=AF(row*ld+hh)+dt;let sp=gdn_softplus(aa);acc=acc-ea*sp;d[bh*T+t]=WF(acc);e[bh*T+t]=WF(exp(acc));
     f[bh*T+t]=WF(select(1.0/(1.0+exp(-BF(row*ld+hh))),1.0,U(11u)==1u));}
   let lend=acc;for(var ii=0u;ii<64u;ii=ii+1u){let t=cc*64u+ii;g[bh*T+t]=WF(exp(lend-DF(bh*T+t)));}return;}
 // op 65: per-(row,head) norms: q̂,k̂ (q̂ /√dk), γq̂, γk̂, V chunk-major. a=qkv_cv b=gam c=qn d=kn e=qg f=kg g=vcm
 if(op==65u){let nv=U(3u);if(z>=U(1u)*U(2u)*nv){return;}let row=z/nv;let hh=z%nv;let T=U(2u);let dk=U(4u);let dv=U(5u);let cd=U(6u);let nch=U(8u);
   let bb=row/T;let t=row%T;let bh=bb*nv+hh;let gm=BF(bh*T+t);var qn2=0.0;var kn2=0.0;
   for(var ii=0u;ii<dk;ii=ii+1u){let qv=AF(row*cd+hh*dk+ii);let kv=AF(row*cd+nv*dk+hh*dk+ii);qn2=qn2+qv*qv;kn2=kn2+kv*kv;}
   let iq=inverseSqrt(qn2+1.0e-6)*inverseSqrt(f32(dk));let ik=inverseSqrt(kn2+1.0e-6);
   for(var ii=0u;ii<dk;ii=ii+1u){let o=row*nv*dk+hh*dk+ii;let qv=AF(row*cd+hh*dk+ii)*iq;let kv=AF(row*cd+nv*dk+hh*dk+ii)*ik;c[o]=WF(qv);d[o]=WF(kv);e[o]=WF(gm*qv);f[o]=WF(gm*kv);}
   let cc=t/64u;let il=t%64u;for(var vv=0u;vv<dv;vv=vv+1u){g[((bh*nch+cc)*64u+il)*dv+vv]=a[row*cd+2u*nv*dk+hh*dv+vv];}return;}
 // op 66: masks. a=kk(raw) b=qk(in→P out) c=lg d=bet e=lmat(out)
 if(op==66u){let nch=U(8u);let T=U(2u);if(z>=U(1u)*U(3u)*nch*4096u){return;}let bhc=z/4096u;let ee=z%4096u;let ii=ee/64u;let jj=ee%64u;let bh=bhc/nch;let cc=bhc%nch;
   let ti=bh*T+cc*64u+ii;let tj=bh*T+cc*64u+jj;var G=0.0;if(ii>=jj){G=exp(CF(ti)-CF(tj));}
   e[z]=WF(select(0.0,AF(z)*G*DF(tj),ii>jj));b[z]=WF(select(0.0,BF(z)*G,ii>=jj));return;}
 // op 67: T = (I+L)^{-1} by forward substitution, one thread per (chunk, column). a=lmat b=tmat
 if(op==67u){let nch=U(8u);if(z>=U(1u)*U(3u)*nch*64u){return;}let bhc=z/64u;let col=z%64u;let base=bhc*4096u;
   for(var ii=0u;ii<64u;ii=ii+1u){var s=select(0.0,1.0,ii==col);for(var jj=0u;jj<ii;jj=jj+1u){s=s-AF(base+ii*64u+jj)*BF(base+jj*64u+col);}b[base+ii*64u+col]=WF(s);}return;}
 // op 68: S_{c+1} := γ_end·S_c. a=states b=gam
 if(op==68u){let ss=U(4u)*U(5u);let T=U(2u);let nch=U(8u);let cc=U(12u);if(z>=U(1u)*U(3u)*ss){return;}let bh=z/ss;let ee=z%ss;let gend=BF(bh*T+cc*64u+63u);
   a[(bh*(nch+1u)+cc+1u)*ss+ee]=WF(gend*AF((bh*(nch+1u)+cc)*ss+ee));return;}
 // op 69: U = β E', Ũ = ρ U for chunk c. a=epm b=bet c=rho d=um e=utm
 if(op==69u){let dv=U(5u);let T=U(2u);let nch=U(8u);let cc=U(12u);if(z>=U(1u)*U(3u)*64u*dv){return;}let bh=z/(64u*dv);let r=z%(64u*dv);let ii=r/dv;let vv=r%dv;let t=cc*64u+ii;
   let idx=((bh*nch+cc)*64u+ii)*dv+vv;let u=BF(bh*T+t)*AF(idx);d[idx]=WF(u);e[idx]=WF(CF(bh*T+t)*u);return;}
 // op 70: dU += ρ dŨ; dρ, dβ; dE' = β dU. a=du b=dut c=um d=epm e=bet f=rho g=dep h=drho i=dbet
 if(op==70u){let dv=U(5u);let T=U(2u);let nch=U(8u);let cc=U(12u);if(z>=U(1u)*U(3u)*64u){return;}let bh=z/64u;let ii=z%64u;let t=cc*64u+ii;
   let base=((bh*nch+cc)*64u+ii)*dv;let dub=(bh*64u+ii)*dv;let rh=FF(bh*T+t);let be=EF(bh*T+t);var dr=0.0;var dbv=0.0;
   for(var vv=0u;vv<dv;vv=vv+1u){let dutv=BF(dub+vv);let duv=AF(dub+vv)+rh*dutv;a[dub+vv]=WF(duv);dr=dr+dutv*CF(base+vv);dbv=dbv+duv*DF(base+vv);g[dub+vv]=WF(be*duv);}
   h[bh*T+t]=WF(dr);i[bh*T+t]=WF(dbv);return;}
 // op 71: dS_c += γ_end dS_{c+1}. a=dsn b=dlive c=gam
 if(op==71u){let ss=U(4u)*U(5u);let T=U(2u);let cc=U(12u);if(z>=U(1u)*U(3u)*ss){return;}let bh=z/ss;a[z]=WF(AF(z)+CF(bh*T+cc*64u+63u)*BF(z));return;}
 // op 72: dgend[bh,c] = Σ dS_{c+1}·S_c. a=dlive b=states c=dgend
 if(op==72u){let ss=U(4u)*U(5u);let nch=U(8u);let cc=U(12u);if(z>=U(1u)*U(3u)){return;}let bh=z;var s=0.0;for(var ee=0u;ee<ss;ee=ee+1u){s=s+AF(bh*ss+ee)*BF((bh*(nch+1u)+cc)*ss+ee);}c[bh*nch+cc]=WF(s);return;}
 // op 73: mask backward reductions per (chunk,i): dlgm_i = Σ_j t1(i,j) − Σ_k t1(k,i), dβ_i += Σ_{k>i} dL_ki A_ki. a=dp b=P c=dl d=L e=kk f=lg g=bet h=dlgm i=dbet
 if(op==73u){let nch=U(8u);let T=U(2u);if(z>=U(1u)*U(3u)*nch*64u){return;}let bhc=z/64u;let ii=z%64u;let bh=bhc/nch;let cc=bhc%nch;let base=bhc*4096u;let ti=bh*T+cc*64u+ii;
   var rs=0.0;for(var jj=0u;jj<=ii;jj=jj+1u){let q=base+ii*64u+jj;rs=rs+AF(q)*BF(q)+CF(q)*DF(q);}
   var cs=0.0;var dbv=0.0;for(var kk0=ii;kk0<64u;kk0=kk0+1u){let q=base+kk0*64u+ii;cs=cs+AF(q)*BF(q)+CF(q)*DF(q);if(kk0>ii){let tk=bh*T+cc*64u+kk0;dbv=dbv+CF(q)*EF(q)*exp(FF(tk)-FF(ti));}}
   h[ti]=WF(rs-cs);i[ti]=WF(IF(ti)+dbv);return;}
 // op 74: dP := dP⊙Γ (lower incl), dL := dL⊙Γ·β_j (strict lower). a=dp b=dl c=lg d=bet
 if(op==74u){let nch=U(8u);let T=U(2u);if(z>=U(1u)*U(3u)*nch*4096u){return;}let bhc=z/4096u;let ee=z%4096u;let ii=ee/64u;let jj=ee%64u;let bh=bhc/nch;let cc=bhc%nch;
   let ti=bh*T+cc*64u+ii;let tj=bh*T+cc*64u+jj;var G=0.0;if(ii>=jj){G=exp(CF(ti)-CF(tj));}
   a[z]=WF(select(0.0,AF(z)*G,ii>=jj));b[z]=WF(select(0.0,BF(z)*G*DF(tj),ii>jj));return;}
 // op 75: per-(b,h,chunk) decay gradient: dlg_i = γ_i(dQγ_i·q̂_i + dKγ_i·k̂_i) + dlgm_i − dρ_i ρ_i (+ Σ_j dρ_j ρ_j + γ_end dgend at i = 63); dla = reverse cumsum.
 //        a=dqn b=dkn c=dqg d=dkg e=qn f=kn g=drho h=dlgm i=dgend j=rho k=gam l=dla
 if(op==75u){let nch=U(8u);let T=U(2u);let nv=U(3u);let dk=U(4u);if(z>=U(1u)*nv*nch){return;}let bh=z/nch;let cc=z%nch;let bb=bh/nv;let hh=bh%nv;
   var srho=0.0;for(var ii=0u;ii<64u;ii=ii+1u){let t=bh*T+cc*64u+ii;srho=srho+GF(t)*JF(t);}
   var acc=0.0;
   for(var ir=64u;ir>0u;ir=ir-1u){let ii=ir-1u;let t=cc*64u+ii;let tt=bh*T+t;let row=bb*T+t;var gq=0.0;var gk=0.0;
     for(var dd=0u;dd<dk;dd=dd+1u){let o=row*nv*dk+hh*dk+dd;gq=gq+CF(o)*EF(o);gk=gk+DF(o)*FF(o);}
     var dlg=KF(tt)*(gq+gk)+HF(tt)-GF(tt)*JF(tt);if(ii==63u){dlg=dlg+srho+KF(tt)*IF(bh*nch+cc);}
     acc=acc+dlg;l[tt]=WF(acc);}
   return;}
 // op 76: per-(b,h): da = dla·(−e^A σ(a+dt)), db = dβ·β(1−β), part = (Σ dla·(−e^A sp), Σ da). a=dla b=a_pre c=p d=dbet e=bet f=da g=db h=part; U11=β≡1
 if(op==76u){let nv=U(3u);let T=U(2u);let ld=U(7u);if(z>=U(1u)*nv){return;}let bh=z;let bb=bh/nv;let hh=bh%nv;let ea=exp(CF(U(9u)+hh));let dt=CF(U(10u)+hh);var pa=0.0;var pd=0.0;
   for(var t=0u;t<T;t=t+1u){let row=bb*T+t;let aa=BF(row*ld+hh)+dt;let sp=gdn_softplus(aa);let sg=1.0/(1.0+exp(-aa));let dl=AF(bh*T+t);let d_a=dl*(-ea*sg);
     f[row*ld+hh]=WF(d_a);pa=pa+dl*(-ea*sp);pd=pd+d_a;let be=EF(bh*T+t);g[row*ld+hh]=WF(select(DF(bh*T+t)*be*(1.0-be),0.0,U(11u)==1u));}
   h[bh*2u]=WF(pa);h[bh*2u+1u]=WF(pd);return;}
 // op 77: per-(row,head) q/k norm backward: dQ̂ = dqn + γ dqg, dK̂ = dkn + γ dkg → dcv q/k columns. a=dqn b=dkn c=dqg d=dkg e=qkv_cv f=gam g=dcv
 if(op==77u){let nv=U(3u);if(z>=U(1u)*U(2u)*nv){return;}let row=z/nv;let hh=z%nv;let T=U(2u);let dk=U(4u);let cd=U(6u);let bb=row/T;let t=row%T;let gm=FF((bb*nv+hh)*T+t);
   var qn2=0.0;var kn2=0.0;var dq=0.0;var dkd=0.0;
   for(var ii=0u;ii<dk;ii=ii+1u){let o=row*nv*dk+hh*dk+ii;let qv=EF(row*cd+hh*dk+ii);let kv=EF(row*cd+nv*dk+hh*dk+ii);qn2=qn2+qv*qv;kn2=kn2+kv*kv;dq=dq+(AF(o)+gm*CF(o))*qv;dkd=dkd+(BF(o)+gm*DF(o))*kv;}
   let iq=inverseSqrt(qn2+1.0e-6)*inverseSqrt(f32(dk));let ik=inverseSqrt(kn2+1.0e-6);
   for(var ii=0u;ii<dk;ii=ii+1u){let o=row*nv*dk+hh*dk+ii;let qv=EF(row*cd+hh*dk+ii);let kv=EF(row*cd+nv*dk+hh*dk+ii);let dQ=AF(o)+gm*CF(o);let dK=BF(o)+gm*DF(o);
     g[row*cd+hh*dk+ii]=WF(iq*dQ-qv*iq*iq*iq*f32(dk)*dq);g[row*cd+nv*dk+hh*dk+ii]=WF(ik*dK-kv*ik*ik*ik*dkd);}
   return;}
 if(op==11u){let row=z;if(row>=U(1u)){return;}let nn=U(2u);let lo=U(4u)+row*nn;let to=U(5u)+row;let oo=U(6u)+row;let tgt=b[to];if(tgt==0xffffffffu){for(var q=0u;q<nn;q=q+1u){a[lo+q]=0u;}c[oo]=0u;return;}var mx=-3.4e38;for(var q=0u;q<nn;q=q+1u){mx=max(mx,AF(lo+q));}var den=0.0;for(var q=0u;q<nn;q=q+1u){den=den+safe_exp(AF(lo+q)-mx);}var loss=0.0;for(var q=0u;q<nn;q=q+1u){let pr=safe_exp(AF(lo+q)-mx)/den;let is_t=(q==tgt);a[lo+q]=WF((pr-select(0.0,1.0,is_t))*F(3u));if(is_t){loss=-log(max(pr,1e-30));}}c[oo]=WF(loss);return;}
 if(op==41u){let row=z;if(row>=U(1u)){return;}let nn=U(2u);let lo=U(4u)+row*nn;let ix=bitcast<i32>(b[row]);if(ix<0){for(var q=0u;q<nn;q=q+1u){a[lo+q]=0u;}return;}let tgtv=c[u32(ix)];if(tgtv==0xffffffffu){for(var q=0u;q<nn;q=q+1u){a[lo+q]=0u;}d[u32(ix)]=0u;return;}let tgt=tgtv%nn;var mx=-3.4e38;for(var q=0u;q<nn;q=q+1u){mx=max(mx,AF(lo+q));}var den=0.0;for(var q=0u;q<nn;q=q+1u){den=den+safe_exp(AF(lo+q)-mx);}let lse=mx+log(den);let loss=lse-AF(lo+tgt);for(var q=0u;q<nn;q=q+1u){let pr=safe_exp(AF(lo+q)-mx)/den;a[lo+q]=WF((pr-select(0.0,1.0,q==tgt))*F(3u));}d[u32(ix)]=WF(loss);return;}
 if(op==12u){if(z<U(1u)){b[U(3u)+z]=a[U(2u)+z];}return;}
 if(op==13u){if(z>=U(1u)*U(3u)){return;}let row=z/U(3u);let col=z%U(3u);b[row*U(3u)+col]=a[row*U(2u)+col];return;}
 if(op==14u){if(z>=U(1u)*U(3u)){return;}let row=z/U(3u);let col=z%U(3u);b[row*U(3u)+col]=select(0u,a[row*U(2u)+col],col<U(2u));return;}
 // The destination is binding `c`; read its existing value from `c` too
 // when this reduction is requested as an accumulation.
 if(op==15u){if(z==0u){var s=0.0;for(var q=0u;q<U(1u);q=q+1u){s=s+AF(q)*BF(q);}c[U(2u)]=WF(CF(U(2u))+s);}return;}
 if(op==16u){if(z<U(1u)){b[z]=WF(1.0/(1.0+exp(-(AF(z)+F(2u)))));}return;}
 if(op==17u){if(z<U(1u)){let y=AF(z);c[z]=WF(BF(z)*y*(1.0-y));}return;}
 if(op==18u){if(z>=U(1u)*U(3u)*U(4u)/2u){return;}let hd=U(4u);let pair=z;let row=pair/(U(3u)*hd/2u);let rem=pair%(U(3u)*hd/2u);let head=rem/(hd/2u);let j0=rem%(hd/2u);let pos=row%U(2u)+U(8u);let ang=f32(pos)*pow(F(6u),-f32(2u*j0)/f32(hd));let cs=cos(ang);let sg=select(1.0,-1.0,U(7u)==1u);let sn=sg*sin(ang);let ix=U(5u)+row*U(3u)*hd+head*hd+j0;let iy=ix+hd/2u;let x=AF(ix);let y=AF(iy);a[ix]=WF(x*cs-y*sn);a[iy]=WF(x*sn+y*cs);return;}
 // Causal softmax is an in-place operation on binding `a` (the host
 // contract supplies only the logits buffer at slot 0).  Writing `c`
 // silently targeted the dummy buffer, leaving the attention probabilities
 // as raw logits on Vulkan while Metal normalized them in place.
 // op 19: band + sink softmax over [t, ld] rows (ld = sink_pad + t). Row r keeps
 // sink columns 0..sink, zeroes sink..sink_pad, and keeps causal column sink_pad+j
 // iff j <= r and r - j < window (window 0 = full causal). (ld,sink,sink_pad,window)
 // = (t,0,0,0) visits exactly the legacy columns in the legacy order.
 // op 19: band + sink softmax; p[8] = carry_pad (columns [sp, sp+cp) = keys of the previous window,
 // valid for a sequence when its u32 in slot b is set; p[9] = blocks per sequence). cp = 0 → legacy order.
 if(op==19u){let zrow=z;if(zrow>=U(1u)*U(2u)){return;}let t=U(1u);let ld=U(4u);let sk=U(5u);let sp=U(6u);let w=U(7u);let cp=U(8u);let bps=max(U(9u),1u);let block=zrow/t;let row=zrow%t;let base=U(3u)+block*t*ld+row*ld;let cb=sp+cp;let jmax=cb+row+1u;var lo=cb;if(w>0u && row+1u>w){lo=cb+row+1u-w;}
   var clo=cb;if(cp>0u && BU(block/bps)!=0u){var back=cp;if(w>0u){back=w-1u;}if(back>row){let span=back-row;if(span>=cp){clo=sp;}else{clo=cb-span;}}}
   var mx=-3.4e38;for(var q=clo;q<cb;q=q+1u){mx=max(mx,AF(base+q));}for(var q=0u;q<jmax;q=q+1u){if(q<sk || q>=lo){mx=max(mx,AF(base+q));}}
   var den=0.0;for(var q=clo;q<cb;q=q+1u){den=den+safe_exp(AF(base+q)-mx);}for(var q=0u;q<jmax;q=q+1u){if(q<sk || q>=lo){den=den+safe_exp(AF(base+q)-mx);}}
   for(var q=0u;q<ld;q=q+1u){let ok=(q<sk)||(q>=clo && q<cb)||(q>=lo && q<jmax);a[base+q]=WF(select(0.0,safe_exp(AF(base+q)-mx)/den,ok));}return;}
 // Softmax backward receives logits/probabilities at slot 0 and the
 // upstream row gradient at slot 1; the result is written back to slot 1.
 // Binding `d` is the dummy slot and is not part of this host call.
 // op 20: softmax backward over [t, ld] rows (p[5] = ld; legacy passes ld = t).
 if(op==20u){let zrow=z;if(zrow>=U(1u)*U(2u)){return;}let t=U(1u);let ld=U(5u);let block=zrow/t;let row=zrow%t;let pb=U(3u)+block*t*ld+row*ld;let db=U(4u)+block*t*ld+row*ld;var dot=0.0;for(var q=0u;q<ld;q=q+1u){dot=dot+AF(pb+q)*BF(db+q);}for(var q=0u;q<ld;q=q+1u){b[db+q]=WF(AF(pb+q)*(BF(db+q)-dot));}return;}
 if(op==21u){if(z<U(1u)*U(2u)){let row=z/U(2u);let col=z%U(2u);let ix=bitcast<i32>(b[row]);if(ix<0){c[z]=0u;}else{c[z]=a[u32(ix)*U(2u)+col];}}return;}
 // Deterministic float scatter-add.  The old row-owner dispatch performed
 // integer addition on the raw f32 bit patterns and also raced whenever two
 // source rows named the same destination.  One invocation owns each
 // destination word and folds all matching source rows in f32 order.
 if(op==22u){let d0=U(3u);let dst_rows=U(4u);if(z>=dst_rows*d0){return;}let dst=z/d0;let col=z%d0;var sum=0.0;for(var row=0u;row<U(1u);row=row+1u){let ix=bitcast<i32>(b[row]);if(ix>=0 && u32(ix)==dst){sum=sum+CF(row*d0+col);}}a[z]=WF(AF(z)+sum);return;}
 // op 23: causal depthwise conv; p[7] = 1: slot d holds the carried history [b, k−1, h] read for time < back
 if(op==23u){if(z>=U(1u)*U(2u)*U(3u)){return;}let row=z/U(3u);let col=z%U(3u);let batch=row/U(2u);let time=row%U(2u);let hh=U(3u);let kk=U(4u);let wo=U(5u);var s=0.0;for(var q=0u;q<kk;q=q+1u){let back=kk-1u-q;if(time>=back){let src=((batch*U(2u)+time-back)*hh)+col;s=s+BF(wo+col*kk+q)*AF(src);}else if(U(7u)==1u){let hr=kk-1u+time-back;s=s+BF(wo+col*kk+q)*DF((batch*(kk-1u)+hr)*hh+col);}}c[z]=WF(s);return;}
 // op 24: conv backward (dx, then dW += with the carried history in slot f when p[7] = 1; the history gets no gradient)
 if(op==24u){let bth=U(1u)*U(2u)*U(3u);let hk=U(3u)*U(4u);if(z<bth){let row=z/U(3u);let col=z%U(3u);let batch=row/U(2u);let time=row%U(2u);let hh=U(3u);let kk=U(4u);var s=0.0;for(var q=0u;q<kk;q=q+1u){let back=kk-1u-q;let dst=time+back;if(dst<U(2u)){s=s+BF(U(5u)+col*kk+q)*CF(((batch*U(2u)+dst)*hh)+col);}}d[z]=WF(s);return;}let q=z-bth;if(q>=hk){return;}let col=q/U(4u);let tap=q%U(4u);let back=U(4u)-1u-tap;var sw=0.0;for(var bi=0u;bi<U(1u);bi=bi+1u){for(var ti=back;ti<U(2u);ti=ti+1u){sw=sw+CF(((bi*U(2u)+ti)*U(3u))+col)*AF(((bi*U(2u)+(ti-back))*U(3u))+col);}if(U(7u)==1u){for(var ti=0u;ti<back && ti<U(2u);ti=ti+1u){sw=sw+CF(((bi*U(2u)+ti)*U(3u))+col)*FF((bi*(U(4u)-1u)+(U(4u)-1u+ti-back))*U(3u)+col);}}}e[U(6u)+col*U(4u)+tap]=WF(EF(U(6u)+col*U(4u)+tap)+sw);return;}
 if(op==25u){if(z<U(1u)*U(2u)){let row=z/U(2u);let head=z%U(2u);let ld=U(3u);let x=AF(row*ld+head);b[row*U(2u)+head]=WF(1.0/(1.0+exp(-(x+F(4u)))));}return;}
 if(op==26u){let rows=U(1u);let nh=U(2u);let ld=U(3u);if(z>=rows*ld){return;}let row=z/ld;let head=z%ld;if(head>=nh){c[row*ld+head]=WF(0.0);return;}let dy=BF(row*nh+head);let kap=AF(row*nh+head);c[row*ld+head]=WF(dy*kap*(1.0-kap));return;}
 if(op==27u){if(z>=U(1u)*U(2u)*U(3u)){return;}let per=U(2u)*U(3u);let row=z/per;let rem=z%per;let hh=rem/U(3u);let feat=rem%U(3u);let theta=AF(row*per+hh*U(3u)+feat);let scale=select(1.0,1.0/sqrt(f32(U(3u))),U(4u)==1u);let p2=2u*U(3u);b[(row*U(2u)+hh)*p2+feat]=WF(scale*cos(theta));b[(row*U(2u)+hh)*p2+U(3u)+feat]=WF(scale*sin(theta));return;}
 // hk_kv's host contract is p[1]=rows, p[2]=heads, p[3]=dv.  p[4]
 // is intentionally unused; using it as the value width makes the old
 // shader divide by zero and leave the resident κ·v buffer untouched.
 if(op==28u){if(z<U(1u)*U(2u)*U(3u)){let row=z/(U(2u)*U(3u));let rem=z%(U(2u)*U(3u));let head=rem/U(3u);let dd=rem%U(3u);c[z]=WF(AF(z)*BF(row*U(2u)+head));}return;}
 if(op==29u && U(7u)==1u){phase_delta_forward(z);return;}
 if(op==29u){hybrid_k_forward(z);return;}
 if(op==30u && U(10u)==2u){hybrid_k_backward_fast(z);return;}
 if(op==30u && U(9u)==1u){hybrid_k_backward_block(z);return;}
 if(op==30u){hybrid_k_backward(z);return;}
 if(op==31u){if(z>=U(1u)){return;}let row=z;let hh=U(2u);let ee=U(3u);let kk=U(4u);var best=3.4e38;var best_raw=0.0;var bi=0u;for(var q=0u;q<ee;q=q+1u){var dist=0.0;var proj=0.0;for(var col=0u;col<hh;col=col+1u){let dx=AF(row*hh+col)-BF(U(5u)+q*hh+col);dist=dist+dx*dx;}for(var qi=0u;qi<kk;qi=qi+1u){var pp=0.0;for(var col=0u;col<hh;col=col+1u){let dx=AF(row*hh+col)-BF(U(5u)+q*hh+col);pp=pp+dx*CF(U(6u)+(q*kk+qi)*hh+col);}proj=proj+pp*pp;}let raw=dist-proj;let score=raw-DF(U(7u)+q);if(score<best){best=score;best_raw=raw;bi=q;}}e[row]=bi;f[row]=WF(best_raw);return;}
 if(op==38u){if(z!=0u){return;}let rows=U(1u);let hh=U(2u);let ee=U(3u);let kk=U(4u);let threshold=F(8u);var fall=0u;for(var row=0u;row<rows;row=row+1u){var best=3.4e38;var second=3.4e38;var best_raw=0.0;var second_raw=0.0;var bi=0u;var b2=0xffffffffu;for(var q=0u;q<ee;q=q+1u){var dist=0.0;var proj=0.0;for(var col=0u;col<hh;col=col+1u){let dx=AF(row*hh+col)-BF(U(5u)+q*hh+col);dist=dist+dx*dx;}for(var qi=0u;qi<kk;qi=qi+1u){var pp=0.0;for(var col=0u;col<hh;col=col+1u){let dx=AF(row*hh+col)-BF(U(5u)+q*hh+col);pp=pp+dx*CF(U(6u)+(q*kk+qi)*hh+col);}proj=proj+pp*pp;}let raw=dist-proj;let score=raw-DF(U(7u)+q);if(score<best){second=best;second_raw=best_raw;best=score;best_raw=raw;b2=bi;bi=q;}else if(score<second){second=score;second_raw=raw;b2=q;}}e[row]=bi;f[row]=b2;let margin=second-best;g[row]=WF(margin);h[row]=WF(select(0.0,0.5,margin<threshold));i[row]=WF(best_raw);if(margin<threshold){fall=fall+1u;}}j[0]=fall;return;}
 if(op==32u){if(z!=0u){return;}let ee=U(2u);for(var q=0u;q<ee;q=q+1u){c[U(3u)+q]=0u;}for(var row=0u;row<U(1u);row=row+1u){let who=a[row];if(who>=ee){b[row]=0xffffffffu;continue;}let slot=c[U(3u)+who];b[row]=slot;c[U(3u)+who]=slot+1u;}}
 if(op==33u){if(z<U(1u)*U(2u)){let row=z/U(2u);let col=z%U(2u);let who=b[row];let slot=c[row];if(slot<U(3u)){d[(who*U(3u)+slot)*U(2u)+col]=a[row*U(2u)+col];}}}
 if(op==34u){if(z<U(1u)*U(2u)){let row=z/U(2u);let col=z%U(2u);let who=b[row];let slot=c[row];if(slot<U(3u)){a[row*U(2u)+col]=WF(AF(row*U(2u)+col)+DF((who*U(3u)+slot)*U(2u)+col));}}}
 if(op==35u){if(z<U(1u)*U(2u)){let who=z/U(2u);let col=z%U(2u);let n=min(BU(U(4u)+who),U(3u));var ss=0.0;for(var q=0u;q<n;q=q+1u){ss=ss+AF((who*U(3u)+q)*U(2u)+col);}c[U(5u)+who*U(2u)+col]=WF(ss);return;}}
 // op 36: descriptor update (port of moe_update_f32). Slots: a=mu b=bias c=sums d=count e=res.
 // The count MUST come from slot d: reading it from the bias buffer (the old `BU`) left
 // every μ frozen and moved every bias by the same amount — routing collapsed to one expert.
 // μ ← (1−α)·μ + α·mean as in Metal (the old port had α and 1−α swapped).
 // p[11] = bias_frozen_from: experts who >= p[11] keep their balancing bias (growth).
 // p[12] = cap: moe_stats (op 35) sums only the filled slots min(count, cap), so the mean
 // divides by the same n_mu = min(count, cap) — dividing the capped sum by the full count
 // pulled the μ of an over-capacity expert toward the origin by cap/count every step.
 // The balancing frac keeps the full count (it measures the load, not the slots).
 if(op==36u){if(z<U(1u)*U(2u)){let who=z/U(2u);let col=z%U(2u);if(who<U(9u)){return;}let cnt=DU(U(6u)+who);let n_mu=min(cnt,U(12u));let old=AF(U(3u)+who*U(2u)+col);if(n_mu>0u){let sum=CF(U(5u)+who*U(2u)+col)/f32(n_mu);a[U(3u)+who*U(2u)+col]=WF((1.0-F(7u))*old+F(7u)*sum);}if(col==0u && who<U(11u)){var rs=0.0;for(var row=0u;row<U(10u);row=row+1u){rs=rs+EF(row);}let scale=rs/f32(max(U(10u),1u));let frac=f32(cnt)/f32(max(U(10u),1u));b[U(4u)+who]=WF(BF(U(4u)+who)+F(8u)*scale*(1.0/f32(U(1u))-frac));}return;}}
 if(op==37u){if(z<U(1u)*U(2u)){let who=z/U(2u);let col=z%U(2u);let row=BU(who);c[U(3u)+who*U(2u)+col]=WF(AF(row*U(2u)+col));return;}}
 if(op==39u){if(z<U(1u)){let l=BF(U(2u)+z%U(5u));let s=1.0/(1.0+exp(-l));let hardw=select(0.0,1.0,s>F(4u));let w=select(s,hardw,U(3u)==1u);a[z]=WF(AF(z)*w);}return;}
 if(op==40u){let rows=U(1u);let nn=U(2u);let lo=U(6u);if(z<rows*nn){let col=z%nn;let l=CF(lo+col);let s=1.0/(1.0+exp(-l));let hardw=select(0.0,1.0,s>F(4u));let w=select(s,hardw,U(3u)==1u);a[z]=WF(AF(z)*w);}return;}
 if(op==48u){let rows=U(1u);let nn=U(2u);let lo=U(6u);if(z<nn && U(3u)==0u){let l=CF(lo+z);let s=1.0/(1.0+exp(-l));let ds=s*(1.0-s);var acc=0.0;for(var rr=0u;rr<rows;rr=rr+1u){acc=acc+AF(rr*nn+z)*BF(rr*nn+z);}let old=DF(lo+z);d[lo+z]=WF(old+(acc+F(5u))*ds);}return;}
 if(op==49u){let rows=U(1u);let cols=U(2u);let hh=U(3u);let eo=U(4u);if(z>=rows*cols){return;}let row=z/cols;let col=z%cols;let token=bitcast<u32>(c[row]);if(token==0xffffffffu){e[z]=0u;return;}let cluster=d[token];if(cluster==0xffffffffu){e[z]=0u;return;}var sum=0.0;for(var q=0u;q<1024u;q=q+1u){if(q<hh){sum=sum+AF(token*hh+q)*BF(eo+cluster*cols*hh+col*hh+q);}}e[z]=WF(sum);return;}
 if(op==51u){hybrid_k_fold_fast(z);return;}
 // op 52: bounded-anchor sink gradient fold: dst[off+z] += alpha * sum_j src[((g*group+j)*pad+s)*hd+d], z=(g*S+s)*hd+d
 // ops 53-57: the GEMM-form hybrid_k (ports of Metal hk_scale_f32 / hk_unscale_f32 /
 // hk_dstates_bwd_f32 / hk_dkv_split_f32 / hk_dtheta_f32). Chunk-major tables are
 // [(b*nh+h)][c][64][p2]; pow is [nh][65][p2] at p[6].
 if(op==53u){let rows=U(1u)*U(2u);let nh=U(3u);let p2=2u*U(4u);if(z>=rows*nh*p2){return;}let row=z/(nh*p2);let r=z%(nh*p2);let hh=r/p2;let ff=r%p2;let b0=row/U(2u);let tt=row%U(2u);let cc=tt/64u;let t=tt%64u;let nch=U(2u)/64u;let dst=(((b0*nh+hh)*nch+cc)*64u+t)*p2+ff;let pw=U(6u)+hh*65u*p2;let q=AF(z);let kq=BF(z);let gt=CF(pw+t*p2+ff);d[dst]=WF(q*gt);e[dst]=WF(kq/gt);f[dst]=WF(q*CF(pw+(t+1u)*p2+ff));g[dst]=WF(kq*CF(pw+(63u-t)*p2+ff));return;}
 if(op==54u){let rows=U(1u)*U(2u);let nh=U(3u);let p2=2u*U(4u);if(z>=rows*nh*p2){return;}let row=z/(nh*p2);let r=z%(nh*p2);let hh=r/p2;let ff=r%p2;let b0=row/U(2u);let tt=row%U(2u);let cc=tt/64u;let t=tt%64u;let nch=U(2u)/64u;let src=(((b0*nh+hh)*nch+cc)*64u+t)*p2+ff;let pw=U(6u)+hh*65u*p2;let gt=EF(pw+t*p2+ff);f[z]=WF(AF(src)*gt+CF(src)*EF(pw+(t+1u)*p2+ff));g[z]=WF(BF(src)/gt+DF(src)*EF(pw+(63u-t)*p2+ff));return;}
 if(op==55u){let bh=z/U(5u);let d0=z%U(5u);if(bh>=U(1u)*U(3u)){return;}let b0=bh/U(3u);let hh=bh%U(3u);let p2=2u*U(4u);let nch=U(2u)/64u;let stbase=bh*(nch+1u)*p2*U(5u);let gamoff=U(6u)+hh*65u*p2+p2;var G:array<f32,64>;for(var ff=0u;ff<64u;ff=ff+1u){G[ff]=0.0;if(ff<p2){d[stbase+(nch*p2+ff)*U(5u)+d0]=WF(0.0);}}for(var rev=0u;rev<U(2u);rev=rev+1u){let tt=U(2u)-1u-rev;let rr=(b0*U(2u)+tt)*U(3u)+hh;let dov=BF(rr*U(5u)+d0);for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){G[ff]=CF(gamoff+ff)*(G[ff]+AF(rr*p2+ff)*dov);}}if((tt%64u)==0u){for(var ff=0u;ff<64u;ff=ff+1u){if(ff<p2){d[stbase+((tt/64u)*p2+ff)*U(5u)+d0]=WF(G[ff]);}}}}return;}
 if(op==56u){let rows=U(1u)*U(2u);let nh=U(3u);if(z>=rows*nh){return;}let dv=U(5u);let kk=BF(z);let base=z*dv;var s=0.0;for(var q=0u;q<dv;q=q+1u){let gq=CF(base+q);d[base+q]=WF(kk*gq);s=s+gq*AF(base+q);}e[z]=WF(s);return;}
 if(op==57u){let rows=U(1u);let nh=U(3u);let nph=U(4u);if(z>=rows*nh*nph){return;}let r=z%(nh*nph);let hh=r/nph;let ii=r%nph;let row=z/(nh*nph);let p2=2u*nph;let base=row*nh*p2+hh*p2;let t=AF(z);let gv=-sin(t)*BF(base+ii)+cos(t)*BF(base+nph+ii);let beta=F(7u);c[z]=WF(select(beta*CF(z)+gv,gv,beta==0.0));return;}
 if(op==52u){if(z>=U(1u)){return;}let sk=U(2u);let hd=U(3u);let grp=U(4u);let goff=U(5u);let al=F(6u);let pad=U(7u);let nb=U(8u);let qh=U(9u);let d=z%hd;let s=(z/hd)%sk;let g=z/(hd*sk);var acc=0.0;for(var bb=0u;bb<nb;bb=bb+1u){for(var j=0u;j<grp;j=j+1u){acc=acc+AF(((bb*qh+g*grp+j)*pad+s)*hd+d);}}b[goff+z]=WF(BF(goff+z)+al*acc);return;}
 if(op==42u){if(z>=U(1u)*U(2u)*U(4u)*U(5u)){return;}let row=z/(U(4u)*U(5u));let dd=z%(U(4u)*U(5u));let batch=row/U(2u);let time=row%U(2u);let qh=U(3u);let kvh=U(4u);let hd=U(5u);let group=qh/kvh;let kvhead=dd/hd;var sum=0.0;for(var q=0u;q<qh;q=q+1u){if(q/group==kvhead){sum=sum+AF(((batch*qh+q)*U(2u)+time)*hd+(dd%hd));}}b[((batch*U(2u)+time)*kvh+kvhead)*hd+(dd%hd)]=WF(sum);return;}
 if(op==43u){phase_delta_replay(z);return;}
 if(op==44u){phase_delta_backward_block(z);return;}
 if(op==45u){phase_delta_fold(z);return;}
 // `moe_indirect_args` binds counts at slot 0 and the dispatch-record
 // destination at slot 1.  Reading BU here used the zeroed destination as
 // the count, emitted mt=0 for every expert, and silently skipped all routed
 // expert GEMMs on Vulkan.
 if(op==46u){if(z>=U(1u)){return;}let who=z;let ee=U(1u);let cap=U(2u);let n1=U(3u);let n2=U(4u);let co=U(5u);let io=U(6u);let cnt=min(AU(co+who),cap);let mt=(cnt+63u)/64u;b[(io+who*3u)+0u]=n1/64u;b[(io+who*3u)+1u]=mt;b[(io+who*3u)+2u]=1u;b[(io+ee*3u+who*3u)+0u]=n2/64u;b[(io+ee*3u+who*3u)+1u]=mt;b[(io+ee*3u+who*3u)+2u]=1u;return;}
 if(op==47u){if(z>=U(1u)*U(2u)*U(3u)){return;}let es=z/(U(2u)*U(3u));let rem=z%(U(2u)*U(3u));let slot=rem/U(2u);let col=rem%U(2u);let n=min(BU(U(5u)+es),U(3u));let src=(es*U(3u)+slot)*U(2u)+col;if(slot<n){d[src]=WF(AF(src)-BF(U(4u)+es*U(2u)+col));}else{d[src]=0u;}return;}
}
"#;

impl Ctx {
    fn inner_dummy(&self) -> DeviceBuf {
        DeviceBuf(Arc::new(BufInner {
            raw: self.inner.dummy.clone(),
            len: 1,
            host: UnsafeCell::new(vec![0]),
            host_dirty: std::sync::atomic::AtomicBool::new(false),
            gpu_dirty: std::sync::atomic::AtomicBool::new(false),
            ctx: Arc::clone(&self.inner),
        }))
    }
    pub fn adapter_name(&self) -> &str {
        &self.inner.adapter_name
    }
}

#[cfg(all(test, not(target_os = "macos")))]
mod tests {
    use super::*;

    #[test]
    fn native_vulkan_gemm_and_adamw_change_device_state() {
        let c = ctx().expect("native Vulkan adapter is required for this test");
        let a = GBuf::from_slice(c, &[1.0, 2.0, 3.0, 4.0]);
        let b = GBuf::from_slice(c, &[2.0, 0.0, 1.0, 2.0]);
        let out = GBuf::zeros(c, 4);
        let mut cmd = Cmd::new(c);
        cmd.gemm(
            Op::N,
            Op::N,
            2,
            2,
            2,
            1.0,
            &a,
            0,
            2,
            &b,
            0,
            2,
            0.0,
            &out,
            0,
            2,
        );
        cmd.commit();
        assert_eq!(out.to_vec(), vec![4.0, 4.0, 10.0, 8.0]);

        let p = GBuf::from_slice(c, &[1.0, 2.0]);
        let g = GBuf::from_slice(c, &[1.0, -1.0]);
        let m = GBuf::zeros(c, 2);
        let v = GBuf::zeros(c, 2);
        let mut cmd = Cmd::new(c);
        cmd.adamw(&p, &g, &m, &v, 2, 1e-2, 0.9, 0.999, 1e-8, 0.0, 1, 1.0);
        cmd.commit();
        assert_ne!(p.to_vec(), vec![1.0, 2.0]);
        assert!(p.to_vec().iter().all(|x| x.is_finite()));
    }

    #[test]
    fn native_vulkan_gemm_dyn_honors_kcount_and_causal_mask() {
        let c = ctx().expect("native Vulkan adapter is required for this test");
        let (m, n, k) = (5usize, 7usize, 96usize);
        let a_host: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.01).collect();
        let b_host: Vec<f32> = (0..n * k).map(|i| (i % 17) as f32 * -0.02).collect();
        let a = GBuf::from_slice(c, &a_host);
        let b = GBuf::from_slice(c, &b_host);
        // Metal rounds the dynamic count to a tile, then clamps to K.  A
        // count of 5 therefore uses the first 64 columns of A/B here.
        let count = GBuf::from_u32(c, &[5]);
        let out = GBuf::zeros(c, m * n);
        let dyn_k = GemmDyn {
            indirect: None,
            kcount: Some((&count, 0)),
        };
        let cmd = Cmd::new(c);
        cmd.gemm_dyn(
            Op::N,
            Op::T,
            m,
            n,
            k,
            1.0,
            &a,
            0,
            k,
            &b,
            0,
            k,
            0.0,
            &out,
            0,
            n,
            &GemmBatch::none(),
            true,
            &dyn_k,
        );
        cmd.commit();
        let got = out.to_vec();
        let mut want = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                if col > row {
                    continue;
                }
                want[row * n + col] = (0..64)
                    .map(|q| a_host[row * k + q] * b_host[col * k + q])
                    .sum();
            }
        }
        let err = got
            .iter()
            .zip(&want)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 2e-5, "dynamic/causal GEMM max error {err:e}");
    }

    #[test]
    fn native_vulkan_indexed_ce_writes_each_loss_row() {
        let c = ctx().expect("native Vulkan adapter is required for this test");
        let rows = 1536usize;
        let n = 256usize;
        let logits = GBuf::zeros(c, rows * n);
        let idx = GBuf::from_u32(c, &(0..rows as u32).collect::<Vec<_>>());
        let tgt = GBuf::from_u32(c, &vec![1; rows]);
        let loss = GBuf::from_slice(c, &vec![-3.4e38; rows]);
        let cmd = Cmd::new(c);
        cmd.softmax_ce_idx(&logits, &idx, &tgt, &loss, rows, n, 1.0);
        cmd.commit();
        let xs = loss.to_vec();
        assert!(
            xs.iter()
                .all(|x| x.is_finite() && (*x - (n as f32).ln()).abs() < 1e-3)
        );
    }

    #[test]
    fn native_vulkan_hk_forward_stays_finite_across_rows() {
        let c = ctx().expect("native Vulkan adapter is required for this test");
        let d = HkDims {
            b: 1,
            t: 64,
            nh: 8,
            nph: 32,
            dv: 128,
        };
        let rows = d.b * d.t;
        let p2 = 2 * d.nph;
        let thq = GBuf::from_slice(c, &vec![0.25; rows * d.nh * d.nph]);
        let thk = GBuf::from_slice(c, &vec![0.5; rows * d.nh * d.nph]);
        let phq = GBuf::zeros(c, rows * d.nh * p2);
        let phk = GBuf::zeros(c, rows * d.nh * p2);
        let v = GBuf::from_slice(c, &vec![0.125; rows * d.nh * d.dv]);
        let kv = GBuf::zeros(c, rows * d.nh * d.dv);
        let kap = GBuf::from_slice(c, &vec![0.75; rows * d.nh]);
        let pow = GBuf::from_slice(c, &hk_pow_table(&vec![0.99; d.nh * p2], d.nh, d.nph));
        let states = GBuf::zeros(c, d.nh * 2 * p2 * d.dv);
        let out = GBuf::zeros(c, rows * d.nh * d.dv);
        let w = HkWork {
            thq: &thq,
            thk: &thk,
            v: &v,
            kappa: &kap,
            pow: &pow,
            pow_off: 0,
            phq: &phq,
            phk: &phk,
            kv: &kv,
            states: &states,
            out: &out,
            phase_chunk: None,
            phase_partial: None,
        };
        let cmd = Cmd::new(c);
        cmd.hk_forward(&d, &w);
        cmd.commit();
        let xs = out.to_vec();
        assert!(xs.iter().all(|x| x.is_finite()));

        // The selected Phase-Delta path has a distinct normalized φ and
        // column-owned causal scan; exercise it separately from legacy HK.
        let cmd = Cmd::new(c);
        cmd.phase_delta_forward_reset(&d, &w);
        cmd.commit();
        assert!(out.to_vec().iter().all(|x| x.is_finite()));
    }

    #[test]
    fn native_vulkan_phase_delta_angle_gradient_matches_difference() {
        let c = ctx().expect("native Vulkan adapter is required for this test");
        // More than one 64-token chunk is intentional: this catches a
        // tempting but incorrect implementation that resets the reverse
        // state at each chunk boundary instead of carrying `dstates`.
        let d = HkDims {
            b: 1,
            t: 130,
            nh: 1,
            nph: 2,
            dv: 32,
        };
        let rows = d.b * d.t;
        let p2 = 2 * d.nph;
        let mut thq_host = vec![0.0f32; rows * d.nh * d.nph];
        let mut thk_host = vec![0.0f32; rows * d.nh * d.nph];
        for (i, x) in thq_host.iter_mut().enumerate() {
            *x = 0.01 * (i % 11) as f32;
        }
        for (i, x) in thk_host.iter_mut().enumerate() {
            *x = -0.02 * (i % 7) as f32;
        }
        let thq = GBuf::from_slice(c, &thq_host);
        let thk = GBuf::from_slice(c, &thk_host);
        let phq = GBuf::zeros(c, rows * d.nh * p2);
        let phk = GBuf::zeros(c, rows * d.nh * p2);
        let v = GBuf::from_slice(c, &vec![0.1; rows * d.nh * d.dv]);
        let kap = GBuf::from_slice(c, &vec![0.7; rows * d.nh]);
        let pow = GBuf::from_slice(c, &hk_pow_table(&vec![0.97; d.nh * p2], d.nh, d.nph));
        let states = GBuf::zeros(c, d.nh * 2 * p2 * d.dv);
        let out = GBuf::zeros(c, rows * d.nh * d.dv);
        let chunk = GBuf::zeros(c, d.nh * 65 * p2 * d.dv);
        let partial = GBuf::zeros(c, d.nh * (d.dv.div_ceil(32)) * d.t * (1 + 2 * d.nph));
        let w = HkWork {
            thq: &thq,
            thk: &thk,
            v: &v,
            kappa: &kap,
            pow: &pow,
            pow_off: 0,
            phq: &phq,
            phk: &phk,
            kv: &v,
            states: &states,
            out: &out,
            phase_chunk: Some(&chunk),
            phase_partial: Some(&partial),
        };
        let dout_host: Vec<f32> = (0..out.len)
            .map(|i| 0.01 * ((i % 9) as f32 - 4.0))
            .collect();
        let dout = GBuf::from_slice(c, &dout_host);
        let dthq = GBuf::zeros(c, rows * d.nh * d.nph);
        let dthk = GBuf::zeros(c, rows * d.nh * d.nph);
        let dv = GBuf::zeros(c, rows * d.nh * d.dv);
        let dkap = GBuf::zeros(c, rows * d.nh);
        let dstates = GBuf::zeros(c, states.len);
        let dkv = GBuf::zeros(c, rows * d.nh * d.dv);
        let dphq = GBuf::zeros(c, rows * d.nh * p2);
        let dphk = GBuf::zeros(c, rows * d.nh * p2);
        let gr = HkGrads {
            dout: &dout,
            dstates: &dstates,
            dkv: &dkv,
            dphq: &dphq,
            dphk: &dphk,
            dthq: &dthq,
            dthk: &dthk,
            dv: &dv,
            dkappa: &dkap,
        };
        let loss = |x: &[f32]| {
            thq.write_from(x);
            let cmd = Cmd::new(c);
            cmd.phase_delta_forward_reset(&d, &w);
            cmd.commit();
            out.to_vec()
                .iter()
                .zip(&dout_host)
                .map(|(a, b)| a * b)
                .sum::<f32>()
        };
        let cmd = Cmd::new(c);
        cmd.phase_delta_forward_reset(&d, &w);
        cmd.phase_delta_backward(&d, &w, &gr);
        cmd.commit();
        let analytic = dthq.to_vec()[5];
        let eps = 1e-3;
        let mut plus = thq_host.clone();
        plus[5] += eps;
        let mut minus = thq_host.clone();
        minus[5] -= eps;
        let numeric = (loss(&plus) - loss(&minus)) / (2.0 * eps);
        assert!(
            (analytic - numeric).abs() < 2e-2,
            "analytic={analytic} numeric={numeric}"
        );
    }
}

/// The DTG-MA skill-mask kernels (ops 39/48/40) against the Metal contract
/// (`mask_fwd_f32`, `mask_bwd_dm_f32`, `mask_bwd_dh_f32`) on a multi-layer
/// logit buffer: every row of a layer reads the SAME `I` logits at its
/// `loff`, and the hard mask is the 0/1 indicator. Runs on Linux (native
/// Vulkan, required) and on macOS through wgpu's Metal backend (skipped
/// when no adapter is available there).
#[cfg(test)]
mod skill_mask_tests {
    use super::*;

    fn device() -> Option<&'static Ctx> {
        let c = ctx();
        if c.is_none() && !cfg!(target_os = "macos") {
            panic!("native Vulkan adapter is required for this test");
        }
        c
    }

    fn sig(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    #[test]
    fn mask_kernels_match_the_metal_contract_on_a_multi_layer_state() {
        let Some(c) = device() else {
            eprintln!("skip: no wgpu adapter");
            return;
        };
        let (rows, i, layers) = (5usize, 96usize, 2usize);
        let (tau, l1) = (0.5f32, 0.03f32);
        // layer 0 logits all-on, layer 1 mixed around τ; the kernel is asked
        // for layer 1 (loff = i), so a row-major read past the slice shows up
        let logits_h: Vec<f32> = (0..layers * i)
            .map(|k| {
                if k < i {
                    3.0
                } else {
                    ((k * 37 % 11) as f32 - 5.0) * 0.4
                }
            })
            .collect();
        let hh_h: Vec<f32> = (0..rows * i)
            .map(|k| ((k * 13 % 17) as f32 - 8.0) * 0.1)
            .collect();
        let dh_h: Vec<f32> = (0..rows * i)
            .map(|k| ((k * 7 % 19) as f32 - 9.0) * 0.05)
            .collect();
        let dm0: Vec<f32> = (0..layers * i).map(|k| k as f32 * 1e-3).collect();
        let loff = i;
        let logits = GBuf::from_slice(c, &logits_h);
        for hard in [false, true] {
            let w = |j: usize| {
                let s = sig(logits_h[loff + j]);
                if hard {
                    if s > tau { 1.0 } else { 0.0 }
                } else {
                    s
                }
            };
            // forward
            let hh = GBuf::from_slice(c, &hh_h);
            let cmd = Cmd::new(c);
            cmd.mask_fwd(&hh, &logits, loff, rows, i, hard, tau);
            cmd.commit();
            let got = hh.to_vec();
            for r in 0..rows {
                for j in 0..i {
                    let want = hh_h[r * i + j] * w(j);
                    let g = got[r * i + j];
                    assert!(
                        (g - want).abs() <= 1e-6 * (1.0 + want.abs()),
                        "mask_fwd hard={hard} row {r} col {j}: {g} vs {want}"
                    );
                }
            }
            // backward: dm (soft only, accumulated into the layer's slice) and dh *= w
            let dh = GBuf::from_slice(c, &dh_h);
            let pre = GBuf::from_slice(c, &hh_h);
            let dm = GBuf::from_slice(c, &dm0);
            let cmd = Cmd::new(c);
            cmd.mask_bwd(&dh, &pre, &logits, loff, &dm, rows, i, hard, tau, l1);
            cmd.commit();
            let (got_dh, got_dm) = (dh.to_vec(), dm.to_vec());
            for r in 0..rows {
                for j in 0..i {
                    let want = dh_h[r * i + j] * w(j);
                    assert!((got_dh[r * i + j] - want).abs() <= 1e-6 * (1.0 + want.abs()));
                }
            }
            for k in 0..layers * i {
                let want = if !hard && k >= loff && k < loff + i {
                    let j = k - loff;
                    let s = sig(logits_h[k]);
                    let acc: f32 = (0..rows).map(|r| dh_h[r * i + j] * hh_h[r * i + j]).sum();
                    dm0[k] + (acc + l1) * s * (1.0 - s)
                } else {
                    dm0[k]
                };
                assert!(
                    (got_dm[k] - want).abs() <= 1e-5 * (1.0 + want.abs()),
                    "mask_bwd dm hard={hard} [{k}]: {} vs {want}",
                    got_dm[k]
                );
            }
        }
    }
}
