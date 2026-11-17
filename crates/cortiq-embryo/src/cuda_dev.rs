// The CUDA device layer of the trainer (`--features cuda`), included by
// `vulkan.rs` in place of its wgpu layer: the same `Cmd` recording, the
// same kernels (`EMBRYO_WGSL` through `wgsl_cuda::translate`, compiled by
// NVRTC once per process and kept on disk as PTX), one launch per recorded
// operation on one stream. A buffer is a device allocation with the same
// demand-driven host mirror as the wgpu layer's.

use cudarc::cublas::sys as bl;
use cudarc::driver::{result as cu, sys as cs};

/// `OpRec::entry` of a GEMM that runs on cuBLAS (`Cmd::cublas_gemm`).
const ENTRY_CUBLAS: u32 = 100;

/// A device allocation's address (the split-K pool's element).
type RawBuf = u64;

struct BufInner {
    ptr: u64,
    len: usize,
    /// A view (the dummy, a split-K scratch) frees nothing.
    owned: bool,
    host: UnsafeCell<Vec<u32>>,
    host_dirty: std::sync::atomic::AtomicBool,
    gpu_dirty: std::sync::atomic::AtomicBool,
    ctx: Arc<CtxInner>,
}

impl Drop for BufInner {
    fn drop(&mut self) {
        if self.owned {
            self.ctx.current();
            unsafe {
                let _ = cu::free_sync(self.ptr);
            }
        }
    }
}

/// The entry points by `OpRec::entry`: kernel name and block size.
const ENTRIES: [(&str, u32); 11] = [
    ("emb_main", 256),
    ("emb_gemm64", 256),
    ("emb_gdn_fwd", 128),
    ("emb_gdn_bwd", 128),
    ("emb_gemm128", 256),
    ("emb_rms_bwd_dx", 256),
    ("emb_rms_dw", 256),
    ("emb_wy_norm_rows", 128),
    ("emb_wy_dnorm_rows", 128),
    ("emb_band_softmax_rows", 256),
    ("emb_softmax_bwd_rows", 256),
];

/// CUDA-only forms of the operations whose WGSL form wastes the card: one
/// serial thread (`ROUTE_GROUP`), one thread per row or per column reading
/// 3 KB apart (`RMS_FWD`, the dW of `RMS_BWD`, the dW of `CONV_BWD`), a few
/// thousand threads each walking its own run (`SUMSQ`, the WY dots and
/// scalars). Every float sum keeps the order and the expression of the WGSL
/// (a row's or a group's sum stays one thread's sequential loop, read from
/// shared memory after a coalesced load), so the results are the same bits;
/// the slot counts are integers. `CMF_CUDA_NATIVE=0` turns them all off,
/// `CMF_CUDA_NATIVE_SKIP=70,76` the listed operation codes.
///
/// The same bits are not a property of the source alone: NVRTC fuses a
/// product into the sum that follows it, and the PTX assembler fuses a
/// plain `mul` + `add`/`sub` pair once more, depending on the other uses of
/// the product. A change here is checked on the card: a few `birth` steps
/// with and without the kernel must leave checkpoints with one sha256.
const NATIVE_CUDA: &str = r#"
// op 32: slot[row] = how many earlier rows chose the same expert,
// count[e] = rows per expert. 256 lanes count their own run of rows, lane 0
// turns the counts into exclusive prefixes, the lanes number their rows.
extern "C" __global__ void __launch_bounds__(256) emb_x_route_group(EWD) {
 __shared__ u32 cnt[256*16];
 const u32 rows=U(1u), ee=U(2u), eo=U(3u), t=threadIdx.x;
 const u32 chunk=(rows+255u)/256u; const u32 r0=min(t*chunk,rows); const u32 r1=min(r0+chunk,rows);
 u32 loc[16];
 for(u32 q=0u;q<16u;q++){loc[q]=0u;}
 for(u32 r=r0;r<r1;r++){const u32 who=S_a[r]; if(who<ee){loc[who]++;}}
 for(u32 q=0u;q<16u;q++){cnt[t*16u+q]=loc[q];}
 __syncthreads();
 if(t==0u){
  for(u32 q=0u;q<ee;q++){u32 run=0u; for(u32 i=0u;i<256u;i++){const u32 v=cnt[i*16u+q]; cnt[i*16u+q]=run; run+=v;} S_c[eo+q]=run;}
 }
 __syncthreads();
 for(u32 q=0u;q<16u;q++){loc[q]=cnt[t*16u+q];}
 for(u32 r=r0;r<r1;r++){const u32 who=S_a[r]; if(who>=ee){S_b[r]=0xffffffffu;}else{S_b[r]=loc[who]; loc[who]++;}}
}
// op 5: a warp per row (8 rows per block). The row is loaded coalesced into
// shared memory, lane 0 sums it in the WGSL's order, every lane scales.
extern "C" __global__ void __launch_bounds__(256) emb_x_rms_fwd(EWD) {
 extern __shared__ float xrow[];
 const u32 rows=U(1u), dd=U(2u), wo=U(3u);
 const u32 w=threadIdx.x>>5u, lane=threadIdx.x&31u;
 const u32 row=(blockIdx.x+blockIdx.y*gridDim.x)*8u+w;
 if(row>=rows){return;}
 volatile float* xs=xrow+w*(dd+1u);
 for(u32 q=lane;q<dd;q+=32u){xs[q]=AF(row*dd+q);}
 __syncwarp();
 if(lane==0u){
  float s=0.0f;
  for(u32 q=0u;q<dd;q++){const float x=xs[q]; s=s+x*x;}
  const float inv0=1.0f/sqrt(s/(float)(dd)+F(4u));
  S_d[row]=as_u32(inv0); xs[dd]=inv0;
 }
 __syncwarp();
 const float inv=xs[dd];
 for(u32 r0=lane;r0<dd;r0+=32u){S_c[row*dd+r0]=as_u32(xs[r0]*BF(wo+r0)*inv);}
}
// op 4: a warp per 32 neighbouring groups. Their 32 chunks of one round
// are 8192 contiguous floats: loaded coalesced, then each lane adds its own
// 256 in order (the 257 stride keeps the lanes on different banks).
extern "C" __global__ void __launch_bounds__(32) emb_x_sumsq(EWD) {
 __shared__ float xs[31*257+256];
 const u32 groups=U(3u), n=U(1u), base=U(2u), lane=threadIdx.x;
 const u32 z0=(blockIdx.x+blockIdx.y*gridDim.x)*32u; const u32 z=z0+lane;
 if(z0>=groups){return;}
 volatile float* v=xs;
 float s=0.0f;
 for(u32 t0=z0*256u;t0<n;t0+=groups*256u){
  for(u32 j=0u;j<256u;j++){const u32 e=j*32u+lane; if(t0+e<n){v[(e>>8u)*257u+(e&255u)]=AF(base+t0+e);}}
  __syncwarp();
  if(z<groups){
   const u32 t=t0+lane*256u;
   if(t<n){const u32 end=min(256u,n-t); for(u32 q=0u;q<end;q++){const float x=v[lane*257u+q]; s=s+x*x;}}
  }
  __syncwarp();
 }
 if(z<groups){S_b[U(4u)+z]=as_u32(s);}
}
// entry 6 (dW of the RMS norm): 32 neighbouring columns per block, so a
// warp reads one cache line of dy and of x per row instead of 32. A thread
// keeps the partial sums of 32 of the WGSL's 256 lanes (rows L, L+256, ...
// in order); the tree over the 256 lanes is the WGSL's.
extern "C" __global__ void __launch_bounds__(256) emb_x_rms_dw(EWD) {
 __shared__ float red[256*32];
 const u32 rows=U(1u), dd=U(2u);
 const u32 w=threadIdx.x>>5u, c=threadIdx.x&31u;
 const u32 col=(blockIdx.x+blockIdx.y*gridDim.x)*32u+c;
 const bool live=col<dd;
 float acc[32];
 #pragma unroll
 for(u32 i=0u;i<32u;i++){acc[i]=0.0f;}
 if(live){
  for(u32 r0=0u;r0<rows;r0+=256u){
   #pragma unroll
   for(u32 i=0u;i<32u;i++){
    const u32 r=r0+w+8u*i;
    if(r<rows){acc[i]=acc[i]+CF(r*dd+col)*AF(r*dd+col)*DF(r);}
   }
  }
 }
 #pragma unroll
 for(u32 i=0u;i<32u;i++){red[(w+8u*i)*32u+c]=acc[i];}
 __syncthreads();
 for(u32 st=128u;st>0u;st=st>>1u){
  for(u32 i=0u;i<32u;i++){const u32 L=w+8u*i; if(L<st){red[L*32u+c]=red[L*32u+c]+red[(L+st)*32u+c];}}
  __syncthreads();
 }
 if(w==0u&&live){S_f[U(4u)+col]=as_u32(FF(U(4u)+col)+red[c]);}
}
// op 70 (WY dU): a warp per (b, h, i) row, lanes over the value channels;
// the two dots of the row are lane 0's, in the WGSL's order.
extern "C" __global__ void __launch_bounds__(256) emb_x_wy_du(EWD) {
 extern __shared__ float xrow[];
 const u32 dv=U(5u), T=U(2u), nch=U(8u), cc=U(12u);
 const u32 w=threadIdx.x>>5u, lane=threadIdx.x&31u;
 const u32 z=(blockIdx.x+blockIdx.y*gridDim.x)*8u+w;
 if(z>=U(1u)*U(3u)*64u){return;}
 const u32 bh=z/64u, ii=z%64u; const u32 t=cc*64u+ii;
 const u32 base=((bh*nch+cc)*64u+ii)*dv; const u32 dub=(bh*64u+ii)*dv;
 const float rh=FF(bh*T+t); const float be=EF(bh*T+t);
 volatile float* s0=xrow+w*4u*dv;
 for(u32 vv=lane;vv<dv;vv+=32u){
  const float dutv=BF(dub+vv); const float duv=AF(dub+vv)+rh*dutv;
  S_a[dub+vv]=as_u32(duv); S_g[dub+vv]=as_u32(be*duv);
  s0[vv]=dutv; s0[dv+vv]=CF(base+vv); s0[2u*dv+vv]=duv; s0[3u*dv+vv]=DF(base+vv);
 }
 __syncwarp();
 if(lane==0u){
  float dr=0.0f; float dbv=0.0f;
  for(u32 vv=0u;vv<dv;vv++){const float dutv=s0[vv]; const float duv=s0[2u*dv+vv]; dr=dr+dutv*s0[dv+vv]; dbv=dbv+duv*s0[3u*dv+vv];}
  S_h[bh*T+t]=as_u32(dr); S_i[bh*T+t]=as_u32(dbv);
 }
}
// op 72 (WY dgend partials): a warp per 32 neighbouring chunks of one
// (b, h); 64 elements of every chunk per round, loaded 32 in a row.
extern "C" __global__ void __launch_bounds__(32) emb_x_wy_dgend(EWD) {
 __shared__ float sa[31*65+64]; __shared__ float sb[31*65+64];
 const u32 ss=U(4u)*U(5u), nch=U(8u), cc=U(12u), lane=threadIdx.x;
 const u32 z0=(blockIdx.x+blockIdx.y*gridDim.x)*32u; const u32 z=z0+lane;
 const u32 total=U(1u)*U(3u)*64u;
 if(z0>=total){return;}
 const u32 bh=z0/64u, p0=z0%64u; const u32 ch=(ss+63u)/64u;
 const u32 abase=bh*ss, bbase=(bh*(nch+1u)+cc)*ss;
 volatile float* va=sa; volatile float* vb=sb;
 float s=0.0f;
 for(u32 r=0u;r<ch;r+=64u){
  const u32 len=min(64u,ch-r);
  for(u32 q=0u;q<32u;q++){
   const u32 e0=(p0+q)*ch+r;
   for(u32 h=0u;h<len;h+=32u){
    const u32 j=h+lane;
    if(j<len&&e0+j<ss){va[q*65u+j]=AF(abase+e0+j); vb[q*65u+j]=BF(bbase+e0+j);}
   }
  }
  __syncwarp();
  if(z<total){
   const u32 lo=(p0+lane)*ch+r;
   for(u32 j=0u;j<len;j++){if(lo+j<ss){s=s+va[lane*65u+j]*vb[lane*65u+j];}}
  }
  __syncwarp();
 }
 if(z<total){S_d[z]=as_u32(s);}
}
// op 85 (WY dlg dots): a warp per 32 neighbouring (row, head) units, each a
// run of dk floats; the q dot, then the k dot.
extern "C" __global__ void __launch_bounds__(32) emb_x_wy_dlg_dot(EWD) {
 __shared__ float sa[31*65+64]; __shared__ float sb[31*65+64];
 const u32 T=U(2u), nv=U(3u), dk=U(4u), lane=threadIdx.x;
 const u32 z0=(blockIdx.x+blockIdx.y*gridDim.x)*32u; const u32 z=z0+lane; const u32 total=U(1u)*T*nv;
 if(z0>=total){return;}
 volatile float* va=sa; volatile float* vb=sb;
 float gq=0.0f; float gk=0.0f;
 for(u32 pass=0u;pass<2u;pass++){
  for(u32 r=0u;r<dk;r+=64u){
   const u32 len=min(64u,dk-r);
   for(u32 q=0u;q<32u;q++){
    const u32 e0=(z0+q)*dk+r;
    for(u32 h=0u;h<len;h+=32u){
     const u32 j=h+lane;
     if(j<len&&z0+q<total){
      if(pass==0u){va[q*65u+j]=CF(e0+j); vb[q*65u+j]=EF(e0+j);}else{va[q*65u+j]=DF(e0+j); vb[q*65u+j]=FF(e0+j);}
     }
    }
   }
   __syncwarp();
   if(z<total){
    if(pass==0u){for(u32 j=0u;j<len;j++){gq=gq+va[lane*65u+j]*vb[lane*65u+j];}}
    else{for(u32 j=0u;j<len;j++){gk=gk+va[lane*65u+j]*vb[lane*65u+j];}}
   }
   __syncwarp();
  }
 }
 if(z<total){const u32 row=z/nv, hh=z%nv; const u32 bb=row/T, t=row%T; S_a[(bb*nv+hh)*T+t]=as_u32(gq+gk);}
}
// op 76 (WY decay scalars): a block per (b, h), lanes over the tokens; the
// two sums over the tokens are lane 0's, in the WGSL's order. The WGSL
// form compiles to PTX `mul` + `sub` for both sums; the PTX assembler then
// fuses the pair of `pa` (its product has no other use) and leaves `pd`
// (its product is also stored as `da`): hence one `fmaf` and one plain add.
extern "C" __global__ void __launch_bounds__(256) emb_x_wy_dscal(EWD) {
 extern __shared__ float xrow[];
 const u32 nv=U(3u), T=U(2u), ld=U(7u); const u32 bh=blockIdx.x+blockIdx.y*gridDim.x;
 if(bh>=U(1u)*nv){return;}
 const u32 bb=bh/nv, hh=bh%nv; const float ea=exp(CF(U(9u)+hh)); const float dt=CF(U(10u)+hh);
 volatile float* s0=xrow;
 for(u32 t=threadIdx.x;t<T;t+=256u){
  const u32 row=bb*T+t; const float aa=BF(row*ld+hh)+dt; const float sp=gdn_softplus(EWP, aa); const float sg=1.0f/(1.0f+exp(-aa));
  const float dl=AF(bh*T+t); const float d_a=dl*(-ea*sg);
  S_f[row*ld+hh]=as_u32(d_a);
  s0[t]=dl; s0[T+t]=(-ea*sp); s0[2u*T+t]=d_a;
  const float be=EF(bh*T+t);
  S_g[row*ld+hh]=as_u32(((U(11u)==1u)?(0.0f):(DF(bh*T+t)*be*(1.0f-be))));
 }
 __syncthreads();
 if(threadIdx.x==0u){
  float pa=0.0f; float pd=0.0f;
  for(u32 t=0u;t<T;t++){pa=fmaf(s0[t],s0[T+t],pa); pd=pd+s0[2u*T+t];}
  S_h[bh*2u]=as_u32(pa); S_h[bh*2u+1u]=as_u32(pd);
 }
}
// op 24 (conv backward): dx as in the WGSL; dW with a warp per tap over 32
// neighbouring channels, the operands of 16 rows loaded ahead of their
// (still sequential) sum.
extern "C" __global__ void __launch_bounds__(256) emb_x_conv_bwd(EWD) {
 const u32 B=U(1u), T=U(2u), H=U(3u), K=U(4u);
 const u32 bth=B*T*H; const u32 nb=(bth+255u)/256u;
 const u32 blk=blockIdx.x+blockIdx.y*gridDim.x;
 if(blk<nb){
  const u32 z=blk*256u+threadIdx.x; if(z>=bth){return;}
  const u32 row=z/H, col=z%H; const u32 batch=row/T, time=row%T;
  float s=0.0f;
  for(u32 q=0u;q<K;q++){const u32 back=K-1u-q; const u32 dst=time+back; if(dst<T){s=s+BF(U(5u)+col*K+q)*CF(((batch*T+dst)*H)+col);}}
  S_d[z]=as_u32(s); return;
 }
 const u32 tap=threadIdx.x>>5u; const u32 col=(blk-nb)*32u+(threadIdx.x&31u);
 if(tap>=K||col>=H){return;}
 const u32 back=K-1u-tap; float sw=0.0f;
 for(u32 bi=0u;bi<B;bi++){
  u32 ti=back;
  for(;ti+16u<=T;ti+=16u){
   float dy[16]; float xv[16];
   #pragma unroll
   for(u32 j=0u;j<16u;j++){dy[j]=CF(((bi*T+ti+j)*H)+col); xv[j]=AF(((bi*T+(ti+j-back))*H)+col);}
   #pragma unroll
   for(u32 j=0u;j<16u;j++){sw=sw+dy[j]*xv[j];}
  }
  for(;ti<T;ti++){sw=sw+CF(((bi*T+ti)*H)+col)*AF(((bi*T+(ti-back))*H)+col);}
  if(U(7u)==1u){for(u32 t2=0u;t2<back&&t2<T;t2++){sw=sw+CF(((bi*T+t2)*H)+col)*FF((bi*(K-1u)+(K-1u+t2-back))*H+col);}}
 }
 S_e[U(6u)+col*K+tap]=as_u32(EF(U(6u)+col*K+tap)+sw);
}
"#;

/// The kernels of `NATIVE_CUDA`.
struct Native {
    route_group: cs::CUfunction,
    rms_fwd: cs::CUfunction,
    sumsq: cs::CUfunction,
    rms_dw: cs::CUfunction,
    wy_du: cs::CUfunction,
    wy_dgend: cs::CUfunction,
    wy_dlg_dot: cs::CUfunction,
    wy_dscal: cs::CUfunction,
    conv_bwd: cs::CUfunction,
    /// Operation codes kept on the WGSL form (`CMF_CUDA_NATIVE_SKIP=70,76`).
    skip: Vec<u32>,
}

/// The parameter blocks of one commit, uploaded together.
struct ParArena {
    ptr: u64,
    words: usize,
}

struct CtxInner {
    cuctx: cs::CUcontext,
    stream: cs::CUstream,
    funcs: [cs::CUfunction; 11],
    /// `emb_main_op<N>` by operation code; null = the generic `emb_main`.
    main_spec: Vec<cs::CUfunction>,
    /// The CUDA-only kernels (`CMF_CUDA_NATIVE=0` = none).
    native: Option<Native>,
    big_gpu: bool,
    gemm128: bool,
    gemm64: bool,
    tile_gemm: bool,
    profile_ops: bool,
    op_ts: bool,
    /// cuBLAS for the large plain GEMMs (`CMF_CUDA_CUBLAS=0` = the translated
    /// kernels everywhere, `all` = every plain GEMM).
    blas: Option<bl::cublasHandle_t>,
    /// A plain single-batch GEMM of at least this many multiply-adds goes to
    /// cuBLAS (`CMF_CUDA_CUBLAS_MIN`).
    blas_min: u64,
    /// cuBLAS also for the dynamic-K / dynamic-row GEMMs (`CMF_CUDA_CUBLAS_DYN=0` = off).
    blas_dyn: bool,
    /// The cuBLAS handle is in the TF32 tensor-core mode (`CMF_CUDA_TF32=0` = never).
    tf32: bool,
    /// Live [`ExactF32`] guards: while any lives, cuBLAS runs full f32.
    exact_depth: Mutex<usize>,
    /// 64 KB of zeros: the slots an operation leaves unbound.
    dummy: u64,
    adapter_name: String,
    splitk_pool: Mutex<Vec<(RawBuf, usize)>>,
    /// One commit at a time owns the parameter arena.
    commit: Mutex<ParArena>,
}

// SAFETY: the context, stream and functions are driver handles valid on any
// thread once the context is current there (`current`), and every use of the
// stream is serialised by `commit`.
unsafe impl Send for CtxInner {}
unsafe impl Sync for CtxInner {}

impl CtxInner {
    fn current(&self) {
        unsafe {
            let _ = cu::ctx::set_current(self.cuctx);
        }
    }
}

pub struct Ctx {
    inner: Arc<CtxInner>,
}

static CTX: OnceLock<Result<Ctx, String>> = OnceLock::new();

/// Process-wide CUDA context. `None` is an honest capability refusal; the
/// caller must not run a CPU implementation under the GPU name.
pub fn ctx() -> Option<&'static Ctx> {
    match CTX.get_or_init(init) {
        Ok(c) => {
            c.inner.current();
            Some(c)
        }
        Err(e) => {
            static ONCE: OnceLock<()> = OnceLock::new();
            ONCE.get_or_init(|| eprintln!("cortiq-embryo: CUDA unavailable: {e}"));
            None
        }
    }
}

fn derr(e: cu::DriverError) -> String {
    format!("{e:?}")
}

/// Where the kernels' PTX is kept: `CMF_CUDA_CACHE` (`0` = no cache), else
/// `$XDG_CACHE_HOME/cortiq/cuda` or `~/.cache/cortiq/cuda`.
fn ptx_cache(src: &str, arch: Option<&str>) -> Option<std::path::PathBuf> {
    let dir = match std::env::var("CMF_CUDA_CACHE") {
        Ok(v) if v == "0" => return None,
        Ok(v) if !v.is_empty() => std::path::PathBuf::from(v),
        _ => std::env::var_os("XDG_CACHE_HOME")
            .map(std::path::PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))?
            .join("cortiq")
            .join("cuda"),
    };
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in src.bytes().chain(arch.unwrap_or("").bytes()).chain(env!("CARGO_PKG_VERSION").bytes()) {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    Some(dir.join(format!("embryo-train-{h:016x}.ptx")))
}

fn init() -> Result<Ctx, String> {
    unsafe {
        if !cs::is_culib_present() {
            return Err("the NVIDIA driver library (libcuda) was not found".into());
        }
        if !cudarc::nvrtc::sys::is_culib_present() {
            return Err("libnvrtc (CUDA toolkit) was not found: it compiles the kernels".into());
        }
    }
    cu::init().map_err(derr)?;
    let ordinal: i32 = std::env::var("CMF_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let dev = cu::device::get(ordinal).map_err(derr)?;
    let cuctx = unsafe { cu::primary_ctx::retain(dev) }.map_err(derr)?;
    unsafe { cu::ctx::set_current(cuctx) }.map_err(derr)?;
    let name = cu::device::get_name(dev).unwrap_or_default();
    let attr = |a| unsafe { cu::device::get_attribute(dev, a) }.unwrap_or(0);
    let major = attr(cs::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    let minor = attr(cs::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    let arch = match (major, minor) {
        (7, 5) => Some("compute_75"),
        (8, 0) => Some("compute_80"),
        (8, 6) => Some("compute_86"),
        (8, 7) => Some("compute_87"),
        (8, 9) => Some("compute_89"),
        (9, 0) => Some("compute_90"),
        (10, 0) => Some("compute_100"),
        (12, 0) => Some("compute_120"),
        _ => None,
    };
    // A blocking stream: the synchronous copies of the host mirrors (the
    // legacy stream) are ordered with the launches.
    let stream = cu::stream::create(cu::stream::StreamKind::Default).map_err(derr)?;
    // `emb_main` once per operation code (`CMF_CUDA_MAIN_SPEC=0` = the one
    // kernel with the run-time switch): same arithmetic, no dispatch chain.
    let (src, spec_ops) = if std::env::var("CMF_CUDA_MAIN_SPEC").as_deref() == Ok("0") {
        (crate::wgsl_cuda::translate(EMBRYO_WGSL), Vec::new())
    } else {
        crate::wgsl_cuda::specialize_main(&crate::wgsl_cuda::translate(EMBRYO_WGSL))
    };
    let native_on = std::env::var("CMF_CUDA_NATIVE").as_deref() != Ok("0");
    let src = if native_on { src + NATIVE_CUDA } else { src };
    let cached = ptx_cache(&src, arch);
    let ptx = match cached.as_ref().and_then(|p| std::fs::read_to_string(p).ok()) {
        Some(t) if t.contains(".entry") => t,
        _ => {
            let opts = cudarc::nvrtc::CompileOptions { arch, ..Default::default() };
            let t = cudarc::nvrtc::compile_ptx_with_opts(&src, opts)
                .map_err(|e| format!("NVRTC: {e:?}"))?
                .to_src();
            if let Some(p) = &cached {
                if let Some(dir) = p.parent() {
                    let _ = std::fs::create_dir_all(dir);
                }
                let tmp = p.with_extension("tmp");
                if std::fs::write(&tmp, &t).is_ok() {
                    let _ = std::fs::rename(&tmp, p);
                }
            }
            t
        }
    };
    let text = std::ffi::CString::new(ptx).map_err(|e| e.to_string())?;
    let module = unsafe { cu::module::load_data(text.as_ptr().cast()) }.map_err(derr)?;
    let mut funcs = [std::ptr::null_mut(); 11];
    for (f, (n, _)) in funcs.iter_mut().zip(ENTRIES) {
        *f = unsafe { cu::module::get_function(module, std::ffi::CString::new(n).unwrap()) }
            .map_err(|e| format!("kernel {n}: {e:?}"))?;
    }
    let mut main_spec: Vec<cs::CUfunction> = vec![std::ptr::null_mut(); 128];
    for op in spec_ops {
        if let Some(slot) = main_spec.get_mut(op as usize) {
            *slot = unsafe { cu::module::get_function(module, std::ffi::CString::new(format!("emb_main_op{op}")).unwrap()) }
                .map_err(|e| format!("kernel emb_main_op{op}: {e:?}"))?;
        }
    }
    let native = if native_on {
        let get = |n: &str| {
            unsafe { cu::module::get_function(module, std::ffi::CString::new(n).unwrap()) }
                .map_err(|e| format!("kernel {n}: {e:?}"))
        };
        Some(Native {
            route_group: get("emb_x_route_group")?,
            rms_fwd: get("emb_x_rms_fwd")?,
            sumsq: get("emb_x_sumsq")?,
            rms_dw: get("emb_x_rms_dw")?,
            wy_du: get("emb_x_wy_du")?,
            wy_dgend: get("emb_x_wy_dgend")?,
            wy_dlg_dot: get("emb_x_wy_dlg_dot")?,
            wy_dscal: get("emb_x_wy_dscal")?,
            conv_bwd: get("emb_x_conv_bwd")?,
            skip: std::env::var("CMF_CUDA_NATIVE_SKIP")
                .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
                .unwrap_or_default(),
        })
    } else {
        None
    };
    let on = |k: &str| std::env::var(k).map(|v| v != "0").unwrap_or(true);
    // cuBLAS takes the LARGE plain GEMMs, on the tensor cores (TF32: the
    // operands' mantissas are rounded to 10 bits inside the multiply, sums
    // stay f32; `CMF_CUDA_TF32=0` keeps full f32). The thousands of small
    // batched GEMMs of a step stay on the translated kernels: cuBLAS runs
    // them at 0.1-0.3 TFLOPS (an A40, a 340M GDN genome: a step of 3157 ms
    // with every plain GEMM on cuBLAS against 2340 ms without it).
    // `CMF_CUDA_CUBLAS=0` turns it off (the bit identity of a GEMM's loop and
    // table forms), `all` sends every plain GEMM.
    let blas_env = std::env::var("CMF_CUDA_CUBLAS").unwrap_or_default();
    let tf32 = std::env::var("CMF_CUDA_TF32").as_deref() != Ok("0");
    let blas = if blas_env != "0" && unsafe { bl::is_culib_present() } {
        let mut h: bl::cublasHandle_t = std::ptr::null_mut();
        let mut ok = unsafe { bl::cublasCreate_v2(&mut h) } as u32 == 0
            && unsafe { bl::cublasSetStream_v2(h, stream.cast()) } as u32 == 0;
        if ok && tf32 {
            ok = unsafe { bl::cublasSetMathMode(h, bl::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH) } as u32 == 0;
        }
        ok.then_some(h)
    } else {
        None
    };
    let tf32 = tf32 && blas.is_some();
    let blas_min: u64 = match blas_env.as_str() {
        "all" | "1" => 0,
        _ => std::env::var("CMF_CUDA_CUBLAS_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(1 << 26),
    };
    let dummy = unsafe { cu::malloc_sync(64 << 10) }.map_err(derr)?;
    unsafe { cu::memset_d8_sync(dummy, 0, 64 << 10) }.map_err(derr)?;
    Ok(Ctx {
        inner: Arc::new(CtxInner {
            cuctx,
            stream,
            funcs,
            main_spec,
            native,
            big_gpu: name.contains("RTX PRO 6000") || name.contains("RTX 5090"),
            gemm128: on("CMF_VULKAN_GEMM128"),
            gemm64: on("CMF_VULKAN_GEMM64"),
            tile_gemm: on("CMF_VULKAN_TILE_GEMM"),
            profile_ops: std::env::var("CMF_VULKAN_PROFILE_OPS").ok().as_deref() == Some("1"),
            op_ts: std::env::var("CMF_VULKAN_OP_TS").ok().as_deref() == Some("1"),
            blas,
            blas_min,
            blas_dyn: std::env::var("CMF_CUDA_CUBLAS_DYN").as_deref() != Ok("0"),
            tf32,
            exact_depth: Mutex::new(0),
            dummy,
            adapter_name: name,
            splitk_pool: Mutex::new(Vec::new()),
            commit: Mutex::new(ParArena { ptr: 0, words: 0 }),
        }),
    })
}

impl DeviceBuf {
    /// Return the host-mirror pointer after synchronising a device write;
    /// the contract of the wgpu layer's `contents`.
    ///
    /// # Safety
    /// The pointer borrows this buffer's mirror: no command may run and no
    /// mirror method be called while it is dereferenced.
    pub unsafe fn contents(&self) -> *mut c_void {
        self.sync_host();
        self.0.host_dirty.store(true, std::sync::atomic::Ordering::Release);
        unsafe { (*self.0.host.get()).as_mut_ptr() as *mut c_void }
    }
    fn sync_host(&self) {
        if !self.0.gpu_dirty.swap(false, std::sync::atomic::Ordering::AcqRel) {
            return;
        }
        self.0.ctx.current();
        let host = unsafe { &mut *self.0.host.get() };
        let n = self.0.len.min(host.len());
        unsafe { cu::memcpy_dtoh_sync(&mut host[..n], self.0.ptr) }.expect("CUDA readback");
    }
    fn upload_host(&self) {
        if !self.0.host_dirty.swap(false, std::sync::atomic::Ordering::AcqRel) {
            return;
        }
        self.0.ctx.current();
        let host = unsafe { &*self.0.host.get() };
        unsafe { cu::memcpy_htod_sync(self.0.ptr, &host[..]) }.expect("CUDA upload");
    }
    fn mark_gpu(&self) {
        self.0.gpu_dirty.store(true, std::sync::atomic::Ordering::Release);
    }
}

fn new_buf(ctx: &Arc<CtxInner>, len: usize, init: Option<&[u32]>) -> Arc<BufInner> {
    let n = len.max(1);
    ctx.current();
    // a little past the end: a kernel's guarded tail may still form the address
    let bytes = n * 4 + 256;
    let ptr = unsafe { cu::malloc_sync(bytes) }.unwrap_or_else(|e| panic!("CUDA allocation of {bytes} bytes: {e:?}"));
    let mut host = vec![0u32; n];
    unsafe { cu::memset_d8_sync(ptr, 0, bytes) }.expect("CUDA clear");
    if let Some(x) = init {
        host[..x.len()].copy_from_slice(x);
        unsafe { cu::memcpy_htod_sync(ptr, &host[..]) }.expect("CUDA upload");
    }
    Arc::new(BufInner {
        ptr,
        len,
        owned: true,
        host: UnsafeCell::new(host),
        host_dirty: std::sync::atomic::AtomicBool::new(false),
        gpu_dirty: std::sync::atomic::AtomicBool::new(false),
        ctx: Arc::clone(ctx),
    })
}

impl Ctx {
    fn inner_dummy(&self) -> DeviceBuf {
        DeviceBuf(Arc::new(BufInner {
            ptr: self.inner.dummy,
            len: 1,
            owned: false,
            host: UnsafeCell::new(vec![0]),
            host_dirty: std::sync::atomic::AtomicBool::new(false),
            gpu_dirty: std::sync::atomic::AtomicBool::new(false),
            ctx: Arc::clone(&self.inner),
        }))
    }
}

/// While it lives, the GEMMs committed run in full f32 ([`Ctx::exact_f32`]).
pub struct ExactF32<'a>(&'a Ctx);

impl Ctx {
    /// The GEMMs committed while the guard lives run in full f32: cuBLAS
    /// leaves the TF32 tensor-core mode (operands rounded to 10 mantissa bits
    /// inside the multiply). For numbers another implementation must
    /// reproduce: the skill router's φ is checked against the runtime's CPU
    /// reference within 1e-5, and TF32 moves it by 5e-5 on Embryo-56M (full
    /// f32: 6e-8). Training steps keep TF32. Guards nest.
    pub fn exact_f32(&self) -> ExactF32<'_> {
        self.tf32_depth(1);
        ExactF32(self)
    }

    fn tf32_depth(&self, step: isize) {
        let inner = &self.inner;
        let Some(h) = inner.blas.filter(|_| inner.tf32) else {
            return;
        };
        // under the commit lock: no GEMM of another stream is executing while the mode flips
        let _arena = inner.commit.lock().unwrap();
        let mut d = inner.exact_depth.lock().unwrap();
        let before = *d;
        *d = (before as isize + step).max(0) as usize;
        let mode = match (before, *d) {
            (0, 1) => bl::cublasMath_t::CUBLAS_DEFAULT_MATH,
            (1, 0) => bl::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
            _ => return,
        };
        inner.current();
        let st = unsafe { bl::cublasSetMathMode(h, mode) } as u32;
        assert_eq!(st, 0, "cublasSetMathMode({mode:?}) failed: {st}");
    }
}

impl Drop for ExactF32<'_> {
    fn drop(&mut self) {
        self.0.tf32_depth(-1);
    }
}

impl Cmd<'_> {
    /// Record a GEMM for cuBLAS when it is a plain one (no causal mask, no
    /// dynamic K or rows, no table, no band, K > 0): the op-1 parameter block
    /// with `entry` = `ENTRY_CUBLAS`. False = the caller records the kernels.
    #[allow(clippy::too_many_arguments)]
    fn cublas_gemm(
        &self,
        p: &[u32],
        a: &GBuf,
        b: &GBuf,
        c: &GBuf,
        causal: bool,
        dynamic: &GemmDyn<'_>,
        table: &GemmTable<'_>,
        band: Option<&GemmBand>,
    ) -> bool {
        if self.c.inner.blas.is_none() || causal || table.is_some() || band.is_some_and(|b| b.mode != 0) || p[4] == 0 {
            return false;
        }
        let single = p[15] * p[16] * p[17] == 1;
        let min = self.c.inner.blas_min;
        if min > 0 && (!single || (p[2] as u64) * (p[3] as u64) * (p[4] as u64) < min) {
            return false;
        }
        // Dynamic K (expert weight gradients) and dynamic rows (expert
        // forward and input gradients): the count lives on the card, so the
        // call is sized when it is issued (`commit` reads the count back).
        // Rows past the count follow the kernels' contract C = beta·C, which
        // this path writes for beta 0 and 1 over a dense C only.
        let beta = f32::from_bits(p[14]);
        let dyn_ok = self.c.inner.blas_dyn
            && single
            && !(dynamic.kcount.is_some() && dynamic.indirect.is_some())
            && (dynamic.indirect.is_none() || ((beta == 0.0 || beta == 1.0) && p[9] == p[3]));
        if (dynamic.kcount.is_some() || dynamic.indirect.is_some()) && !dyn_ok {
            return false;
        }
        let mut q = p.to_vec();
        q[PARAM_WORDS - 2] = ENTRY_CUBLAS;
        let mut slots = vec![(a, 0), (b, 1), (c, 2)];
        if let Some((kb, koff)) = dynamic.kcount {
            q[28] = 1;
            q[29] = koff as u32;
            slots.push((kb, 3));
        } else if let Some((ib, off)) = dynamic.indirect {
            q[30] = 1;
            q[29] = (off / 4) as u32;
            slots.push((ib, 3));
        }
        self.rec(OP_GEMM, &slots, q, 256);
        true
    }

    /// The split-K partial buffer of this stream (≥ `n` floats): borrowed
    /// from the context pool on first use, kept until `commit`.
    fn splitk_scratch(&self, n: usize) -> GBuf {
        let mut mine = self.splitk.borrow_mut();
        if !matches!(*mine, Some((_, len)) if len >= n) {
            let mut pool = self.c.inner.splitk_pool.lock().unwrap();
            if let Some(b) = mine.take() {
                pool.push(b);
            }
            *mine = match pool.iter().position(|(_, len)| *len >= n) {
                Some(i) => Some(pool.swap_remove(i)),
                None => {
                    let len = n.max(1 << 22);
                    self.c.inner.current();
                    // kept for the process: the pool lends it to every later stream
                    let raw = unsafe { cu::malloc_sync(len * 4) }.expect("CUDA split-K scratch");
                    Some((raw, len))
                }
            };
        }
        let (raw, len) = mine.as_ref().unwrap();
        // device-only scratch: no host mirror (never uploaded or read back)
        GBuf {
            buf: DeviceBuf(Arc::new(BufInner {
                ptr: *raw,
                len: *len,
                owned: false,
                host: UnsafeCell::new(Vec::new()),
                host_dirty: std::sync::atomic::AtomicBool::new(false),
                gpu_dirty: std::sync::atomic::AtomicBool::new(false),
                ctx: Arc::clone(&self.c.inner),
            })),
            len: *len,
        }
    }

    /// Launch every recorded operation in order on the stream and wait for
    /// the card; the milliseconds from the first launch to the end.
    pub fn commit(self) -> f64 {
        let splitk = self.splitk.borrow_mut().take();
        let ops = self.ops.into_inner();
        let inner = &self.c.inner;
        inner.current();
        let mut arena = inner.commit.lock().unwrap();
        for op in &ops {
            for s in &op.slots {
                s.upload_host();
            }
        }
        let encode_start = std::time::Instant::now();
        let words = ops.len() * PARAM_WORDS;
        if arena.words < words {
            if arena.ptr != 0 {
                unsafe {
                    let _ = cu::free_sync(arena.ptr);
                }
            }
            let want = words.next_power_of_two().max(PARAM_WORDS * 1024);
            arena.ptr = unsafe { cu::malloc_sync(want * 4) }.expect("CUDA parameter arena");
            arena.words = want;
        }
        let mut all = Vec::with_capacity(words);
        for op in &ops {
            all.extend_from_slice(&op.params);
        }
        if !all.is_empty() {
            unsafe { cu::memcpy_htod_sync(arena.ptr, &all) }.expect("CUDA parameter upload");
        }
        let t = std::time::Instant::now();
        let mut events = Vec::new();
        let ev = || cu::event::create(cs::CUevent_flags::CU_EVENT_DEFAULT).expect("CUDA event");
        if inner.op_ts {
            let e = ev();
            unsafe { cu::event::record(e, inner.stream) }.expect("CUDA event");
            events.push(e);
        }
        // Count buffers of the dynamic cuBLAS GEMMs: a count is read back
        // once and reused until another operation touches its buffer.
        let dyn_bufs: std::collections::HashSet<u64> = ops
            .iter()
            .filter(|o| o.entry == ENTRY_CUBLAS && (o.params[28] == 1 || o.params[30] == 1))
            .map(|o| o.slots[3].0.ptr)
            .collect();
        let mut dyn_seen: HashMap<(u64, u32), [u32; 2]> = HashMap::new();
        for (i, op) in ops.iter().enumerate() {
            if op.entry != ENTRY_CUBLAS && !dyn_bufs.is_empty() {
                for s in &op.slots {
                    if dyn_bufs.contains(&s.0.ptr) {
                        dyn_seen.retain(|(ptr, _), _| *ptr != s.0.ptr);
                    }
                }
            }
            if op.entry == ENTRY_CUBLAS {
                let dynv = (op.params[28] == 1 || op.params[30] == 1).then(|| {
                    let key = (op.slots[3].0.ptr, op.params[29]);
                    *dyn_seen.entry(key).or_insert_with(|| {
                        let mut w = [0u32; 2];
                        unsafe {
                            cu::stream::synchronize(inner.stream).expect("CUDA sync before a dynamic GEMM");
                            cu::memcpy_dtoh_sync(&mut w[..], key.0 + 4 * key.1 as u64).expect("CUDA count readback");
                        }
                        w
                    })
                });
                unsafe { blas_gemm(inner, op, dynv) };
                if inner.op_ts {
                    let e = ev();
                    unsafe { cu::event::record(e, inner.stream) }.expect("CUDA event");
                    events.push(e);
                }
                for s in &op.slots {
                    s.mark_gpu();
                }
                continue;
            }
            let entry = (op.entry as usize).min(ENTRIES.len() - 1);
            let (mut f, mut block) = (inner.funcs[entry], ENTRIES[entry].1);
            let (mut groups, mut shared) = (op.groups, 0u32);
            if entry == 0 {
                if let Some(s) = inner.main_spec.get(op.params[0] as usize).filter(|s| !s.is_null()) {
                    f = *s;
                }
                // (kernel, block, blocks, dynamic shared bytes) of a CUDA-only form
                let p = &op.params;
                let nat = inner.native.as_ref().filter(|x| !x.skip.contains(&p[0])).and_then(|x| match p[0] {
                    OP_ROUTE_GROUP if p[2] <= 16 => Some((x.route_group, 256, 1, 0)),
                    OP_RMS_FWD if (1..=1535).contains(&p[2]) => Some((x.rms_fwd, 256, p[1].div_ceil(8), 32 * (p[2] + 1))),
                    OP_SUMSQ => Some((x.sumsq, 32, p[3].div_ceil(32), 0)),
                    OP_WY_DU if (1..=384).contains(&p[5]) => Some((x.wy_du, 256, (p[1] * p[3] * 64).div_ceil(8), 128 * p[5])),
                    OP_WY_DGEND => Some((x.wy_dgend, 32, (p[1] * p[3] * 64).div_ceil(32), 0)),
                    OP_WY_DLG_DOT => Some((x.wy_dlg_dot, 32, (p[1] * p[2] * p[3]).div_ceil(32), 0)),
                    OP_WY_DSCAL if (1..=4096).contains(&p[2]) => Some((x.wy_dscal, 256, p[1] * p[3], 12 * p[2])),
                    OP_CONV_BWD if (1..=8).contains(&p[4]) => {
                        Some((x.conv_bwd, 256, (p[1] * p[2] * p[3]).div_ceil(256) + p[3].div_ceil(32), 0))
                    }
                    _ => None,
                });
                if let Some(n) = nat {
                    (f, block, groups, shared) = n;
                }
            } else if entry == 6 {
                if let Some(x) = inner.native.as_ref().filter(|x| !x.skip.contains(&op.params[0])) {
                    (f, groups) = (x.rms_dw, op.params[2].div_ceil(32));
                }
            }
            let mut vals = [0u64; SLOTS + 1];
            for (v, s) in vals.iter_mut().zip(&op.slots) {
                *v = s.0.ptr;
            }
            vals[SLOTS] = arena.ptr + (i * PARAM_WORDS * 4) as u64;
            let mut args: Vec<*mut c_void> = vals.iter_mut().map(|v| v as *mut u64 as *mut c_void).collect();
            let gx = groups.clamp(1, 65_535);
            let gy = groups.div_ceil(gx).max(1);
            unsafe { cu::launch_kernel(f, (gx, gy, 1), (block, 1, 1), shared, inner.stream, &mut args) }
                .unwrap_or_else(|e| panic!("CUDA launch of op {} (entry {entry}): {e:?}", op.params[0]));
            if inner.op_ts {
                let e = ev();
                unsafe { cu::event::record(e, inner.stream) }.expect("CUDA event");
                events.push(e);
            }
            for s in &op.slots {
                s.mark_gpu();
            }
        }
        if inner.profile_ops {
            eprintln!("cuda encode ops={} ms={:.3}", ops.len(), encode_start.elapsed().as_secs_f64() * 1000.0);
        }
        unsafe { cu::stream::synchronize(inner.stream) }.expect("CUDA commit");
        if inner.op_ts && events.len() == ops.len() + 1 {
            let mut totals = HashMap::<u32, (f64, usize)>::new();
            let mut gemms = HashMap::<[u32; 7], (f64, usize)>::new();
            let mut span = 0.0;
            for (i, op) in ops.iter().enumerate() {
                let ms = unsafe { cu::event::elapsed(events[i], events[i + 1]) }.unwrap_or(0.0) as f64;
                span += ms;
                // a dedicated entry of a non-GEMM operation is listed apart: op + 1000 * entry
                let split = op.params[0] != OP_GEMM && op.entry != 0 && op.entry != ENTRY_CUBLAS;
                let e = totals.entry(op.params[0] + if split { 1000 * op.entry } else { 0 }).or_insert((0.0, 0));
                e.0 += ms;
                e.1 += 1;
                if op.params[0] == OP_GEMM {
                    let p = &op.params;
                    let key = [op.entry, p[5] * 2 + p[6], p[2], p[3], p[4], p[15] * p[16] * p[17], p[31]];
                    let g = gemms.entry(key).or_insert((0.0, 0));
                    g.0 += ms;
                    g.1 += 1;
                }
            }
            eprintln!("cuda commit-ts ops={} span_ms={span:.3}", ops.len());
            let mut rows: Vec<_> = totals.into_iter().collect();
            rows.sort_unstable_by_key(|(op, _)| *op);
            for (op, (ms, count)) in rows {
                eprintln!("cuda op-ts op={op} count={count} gpu_ms={ms:.3}");
            }
            for (k, (ms, count)) in gemms {
                eprintln!(
                    "cuda gemm-ts entry={} t={} m={} n={} k={} batches={} tile={} count={count} gpu_ms={ms:.3}",
                    k[0], k[1], k[2], k[3], k[4], k[5], k[6]
                );
            }
        }
        for e in events {
            unsafe {
                let _ = cu::event::destroy(e);
            }
        }
        drop(arena);
        if let Some(b) = splitk {
            inner.splitk_pool.lock().unwrap().push(b);
        }
        t.elapsed().as_secs_f64() * 1000.0
    }
}

/// The op-1 GEMM of `op` (row-major, three batch levels with their strides)
/// as cuBLAS calls: row-major C = op(A)·op(B) is column-major
/// Cᵀ = op(B)ᵀ·op(A)ᵀ over the same storage, so B goes first; the innermost
/// batch level is one strided call, the outer two a loop.
unsafe fn blas_gemm(inner: &CtxInner, op: &OpRec, dynv: Option<[u32; 2]>) {
    let p = &op.params;
    let (m0, n, k0) = (p[2] as i32, p[3] as i32, p[4] as i32);
    // The op-1 dynamic contract of the kernels: K = min(round64(count), K);
    // rows >= 64 * (the dispatch record's second word) are C = beta * C.
    let (mut m, mut k) = (m0, k0);
    if let Some([w0, w1]) = dynv {
        if p[28] == 1 {
            k = k0.min((w0.div_ceil(64) * 64).min(k0 as u32) as i32);
        }
        if p[30] == 1 {
            m = m0.min((w1.saturating_mul(64)).min(m0 as u32) as i32);
        }
    }
    if m < m0 && f32::from_bits(p[14]) == 0.0 {
        // dense C (ldc == n, checked when recorded): the skipped rows are zero
        let tail = op.slots[2].0.ptr + 4 * (p[12] as u64 + m as u64 * p[9] as u64);
        unsafe { cu::memset_d8_async(tail, 0, 4 * (m0 - m) as usize * n as usize, inner.stream) }
            .expect("CUDA clear of the skipped rows");
    }
    if m == 0 {
        return;
    }
    if k == 0 {
        // no rows to sum: C = beta * C (cuBLAS is not asked to scale)
        let beta = f32::from_bits(p[14]);
        if beta == 0.0 {
            for r in 0..m as u64 {
                let row = op.slots[2].0.ptr + 4 * (p[12] as u64 + r * p[9] as u64);
                unsafe { cu::memset_d8_async(row, 0, 4 * n as usize, inner.stream) }.expect("CUDA clear of C");
            }
            return;
        }
        if beta == 1.0 {
            return;
        }
    }
    let (ta, tb) = (p[5] == 1, p[6] == 1);
    let (lda, ldb, ldc) = (p[7] as i32, p[8] as i32, p[9] as i32);
    let (al, be) = (f32::from_bits(p[13]), f32::from_bits(p[14]));
    let (nb, nh, nc) = (p[15] as u64, p[16] as u64, p[17] as u64);
    let opr = |t| if t { bl::cublasOperation_t::CUBLAS_OP_T } else { bl::cublasOperation_t::CUBLAS_OP_N };
    let h = inner.blas.expect("recorded only with a handle");
    let base = |slot: usize, off: u32| op.slots[slot].0.ptr + 4 * off as u64;
    let (a0, b0, c0) = (base(0, p[10]), base(1, p[11]), base(2, p[12]));
    for cb in 0..nb {
        for ch in 0..nh {
            let a = a0 + 4 * (cb * p[18] as u64 + ch * p[19] as u64);
            let b = b0 + 4 * (cb * p[21] as u64 + ch * p[22] as u64);
            let c = c0 + 4 * (cb * p[24] as u64 + ch * p[25] as u64);
            let s = unsafe {
                if nc == 1 {
                    bl::cublasSgemm_v2(h, opr(tb), opr(ta), n, m, k, &al, b as *const f32, ldb, a as *const f32, lda, &be, c as *mut f32, ldc)
                } else {
                    bl::cublasSgemmStridedBatched(
                        h,
                        opr(tb),
                        opr(ta),
                        n,
                        m,
                        k,
                        &al,
                        b as *const f32,
                        ldb,
                        p[23] as i64,
                        a as *const f32,
                        lda,
                        p[20] as i64,
                        &be,
                        c as *mut f32,
                        ldc,
                        p[26] as i64,
                        nc as i32,
                    )
                }
            };
            assert_eq!(s as u32, 0, "cuBLAS GEMM m={m} n={n} k={k}: {s:?}");
        }
    }
}
