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
  #pragma unroll 16
  for(u32 q=0u;q<dd;q++){const float x=xs[q]; s=s+x*x;}
  const float inv0=1.0f/sqrt(s/(float)(dd)+F(4u));
  S_d[row]=as_u32(inv0); xs[dd]=inv0;
 }
 __syncwarp();
 const float inv=xs[dd];
 for(u32 r0=lane;r0<dd;r0+=32u){S_c[row*dd+r0]=as_u32(xs[r0]*BF(wo+r0)*inv);}
}
// op 4: a warp per group (4096 groups of a 341 M vector: every SM busy).
// A round's 256 floats are loaded coalesced, the next round's while lane 0
// adds this one in the WGSL's order.
extern "C" __global__ void __launch_bounds__(256) emb_x_sumsq(EWD) {
 __shared__ float xs[8*256];
 const u32 groups=U(3u), n=U(1u), base=U(2u);
 const u32 w=threadIdx.x>>5u, lane=threadIdx.x&31u;
 const u32 z=(blockIdx.x+blockIdx.y*gridDim.x)*8u+w;
 if(z>=groups){return;}
 float* v=xs+w*256u;
 const u32 stride=groups*256u;
 float r[8];
 u32 t=z*256u;
 #pragma unroll
 for(u32 j=0u;j<8u;j++){const u32 e=j*32u+lane; r[j]=(t<n&&e<n-t)?AF(base+t+e):0.0f;}
 float s=0.0f;
 while(t<n){
  #pragma unroll
  for(u32 j=0u;j<8u;j++){v[j*32u+lane]=r[j];}
  __syncwarp();
  const u32 tn=(n-t>stride)?(t+stride):n;
  #pragma unroll
  for(u32 j=0u;j<8u;j++){const u32 e=j*32u+lane; r[j]=(tn<n&&e<n-tn)?AF(base+tn+e):0.0f;}
  if(lane==0u){const u32 end=min(256u,n-t);
   #pragma unroll 16
   for(u32 q=0u;q<end;q++){const float x=v[q]; s=s+x*x;}}
  __syncwarp();
  t=tn;
 }
 if(lane==0u){S_b[U(4u)+z]=as_u32(s);}
}
// entry 6 (dW of the RMS norm), in two launches. First the WGSL's 256 lane
// sums as chains of their own: a thread per (lane L, column), rows L, L+256,
// ... in order, operands of 8 rows loaded ahead of their (sequential) sum,
// into the scratch (slot m) as [256][dd]. Then a block per 32 columns runs
// the WGSL's tree over the 256 lanes and adds the result to dW.
extern "C" __global__ void __launch_bounds__(256) emb_x_rms_dw_part(EWD) {
 const u32 rows=U(1u), dd=U(2u);
 const u32 ncg=(dd+31u)/32u; const u32 blk=blockIdx.x+blockIdx.y*gridDim.x;
 const u32 col=(blk%ncg)*32u+(threadIdx.x&31u); const u32 L=(blk/ncg)*8u+(threadIdx.x>>5u);
 if(col>=dd||L>=256u){return;}
 float acc=0.0f;
 u32 r=L;
 for(;r+7u*256u<rows;r+=8u*256u){
  float cv[8]; float av[8]; float dv[8];
  #pragma unroll
  for(u32 j=0u;j<8u;j++){const u32 rr=r+j*256u; cv[j]=CF(rr*dd+col); av[j]=AF(rr*dd+col); dv[j]=DF(rr);}
  #pragma unroll
  for(u32 j=0u;j<8u;j++){acc=acc+cv[j]*av[j]*dv[j];}
 }
 for(;r<rows;r+=256u){acc=acc+CF(r*dd+col)*AF(r*dd+col)*DF(r);}
 S_m[L*dd+col]=as_u32(acc);
}
extern "C" __global__ void __launch_bounds__(256) emb_x_rms_dw_tree(EWD) {
 __shared__ float red[256*32];
 const u32 dd=U(2u);
 const u32 w=threadIdx.x>>5u, c=threadIdx.x&31u;
 const u32 col=(blockIdx.x+blockIdx.y*gridDim.x)*32u+c;
 const bool live=col<dd;
 for(u32 i=0u;i<32u;i++){const u32 L=w+8u*i; red[L*32u+c]=live?as_f32(S_m[L*dd+col]):0.0f;}
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
  #pragma unroll 16
  for(u32 vv=0u;vv<dv;vv++){const float dutv=s0[vv]; const float duv=s0[2u*dv+vv]; dr=dr+dutv*s0[dv+vv]; dbv=dbv+duv*s0[3u*dv+vv];}
  S_h[bh*T+t]=as_u32(dr); S_i[bh*T+t]=as_u32(dbv);
 }
}
// op 72 (WY dgend partials): a warp per partial (b, h, i), its run of
// ceil(dk·dv/64) products loaded 256 at a time, lane 0's sum in order.
extern "C" __global__ void __launch_bounds__(256) emb_x_wy_dgend(EWD) {
 __shared__ float sa[8*256]; __shared__ float sb[8*256];
 const u32 ss=U(4u)*U(5u), nch=U(8u), cc=U(12u);
 const u32 w=threadIdx.x>>5u, lane=threadIdx.x&31u;
 const u32 z=(blockIdx.x+blockIdx.y*gridDim.x)*8u+w;
 if(z>=U(1u)*U(3u)*64u){return;}
 const u32 bh=z/64u, pi=z%64u; const u32 ch=(ss+63u)/64u;
 const u32 lo=pi*ch; const u32 hi=min(lo+ch,ss);
 const u32 abase=bh*ss, bbase=(bh*(nch+1u)+cc)*ss;
 float* va=sa+w*256u; float* vb=sb+w*256u;
 float s=0.0f;
 for(u32 r0=lo;r0<hi;r0+=256u){
  const u32 len=min(256u,hi-r0);
  for(u32 j=lane;j<len;j+=32u){va[j]=AF(abase+r0+j); vb[j]=BF(bbase+r0+j);}
  __syncwarp();
  if(lane==0u){
   #pragma unroll 16
   for(u32 j=0u;j<len;j++){s=s+va[j]*vb[j];}}
  __syncwarp();
 }
 if(lane==0u){S_d[z]=as_u32(s);}
}
// op 85 (WY dlg dots): a warp per (row, head) unit, its four runs of dk
// (<= 128) floats loaded coalesced; lane 0 adds the q dot and the k dot.
extern "C" __global__ void __launch_bounds__(256) emb_x_wy_dlg_dot(EWD) {
 __shared__ float xs[8*512];
 const u32 T=U(2u), nv=U(3u), dk=U(4u);
 const u32 w=threadIdx.x>>5u, lane=threadIdx.x&31u;
 const u32 z=(blockIdx.x+blockIdx.y*gridDim.x)*8u+w;
 if(z>=U(1u)*T*nv){return;}
 float* v=xs+w*512u;
 const u32 o=z*dk;
 for(u32 j=lane;j<dk;j+=32u){v[j]=CF(o+j); v[128u+j]=EF(o+j); v[256u+j]=DF(o+j); v[384u+j]=FF(o+j);}
 __syncwarp();
 if(lane==0u){
  float gq=0.0f; float gk=0.0f;
  #pragma unroll 16
  for(u32 dd=0u;dd<dk;dd++){gq=gq+v[dd]*v[128u+dd]; gk=gk+v[256u+dd]*v[384u+dd];}
  const u32 row=z/nv, hh=z%nv; const u32 bb=row/T, t=row%T;
  S_a[(bb*nv+hh)*T+t]=as_u32(gq+gk);
 }
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
  #pragma unroll 16
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
// op 24, dW for K <= 4 (launched after `emb_x_conv_bwd` has done dx): a
// block per 32 neighbouring channels, warp `tap` (< K) keeps the WGSL's
// chain of that tap (batch by batch, the rows in order, then the carried
// rows). Warps 4..7 bring the rows: tiles of 128 rows of dy and 131 of x
// (3 rows of history above) in two shared-memory stages. A loader stores
// the tile it loaded one round earlier and issues the loads of the next, so
// their latency passes while the tap warps add the current tile.
#define CONV_TR 128u
extern "C" __global__ void __launch_bounds__(256) emb_x_conv_dw(EWD) {
 extern __shared__ float sm[];
 const u32 B=U(1u), T=U(2u), H=U(3u), K=U(4u);
 const u32 w=threadIdx.x>>5u, c=threadIdx.x&31u;
 const u32 col=(blockIdx.x+blockIdx.y*gridDim.x)*32u+c;
 const bool live=col<H;
 const u32 XR=CONV_TR+3u; const u32 STG=(CONV_TR+XR)*32u;
 const u32 ntt=(T+CONV_TR-1u)/CONV_TR; const u32 total=B*ntt;
 const u32 tap=w; const u32 back=(tap<K)?(K-1u-tap):0u;
 const u32 lw=w-4u;
 float rv[65];
 if(w>=4u){
  #pragma unroll
  for(u32 j=0u;j<32u;j++){const u32 ti=lw+4u*j; rv[j]=(live&&ti<T)?CF((ti*H)+col):0.0f;}
  #pragma unroll
  for(u32 j=0u;j<33u;j++){const int ti=(int)(lw+4u*j)-3; rv[32u+j]=(live&&ti>=0&&(u32)ti<T)?AF(((u32)ti*H)+col):0.0f;}
 }
 float sw=0.0f;
 for(u32 q=0u;q<=total;q++){
  if(w>=4u){
   if(q<total){
    float* st=sm+(q&1u)*STG;
    #pragma unroll
    for(u32 j=0u;j<32u;j++){st[(lw+4u*j)*32u+c]=rv[j];}
    #pragma unroll
    for(u32 j=0u;j<33u;j++){const u32 r=lw+4u*j; if(r<XR){st[CONV_TR*32u+r*32u+c]=rv[32u+j];}}
    if(q+1u<total){
     const u32 bi=(q+1u)/ntt, t0=((q+1u)%ntt)*CONV_TR;
     #pragma unroll
     for(u32 j=0u;j<32u;j++){const u32 ti=t0+lw+4u*j; rv[j]=(live&&ti<T)?CF(((bi*T+ti)*H)+col):0.0f;}
     #pragma unroll
     for(u32 j=0u;j<33u;j++){const int ti=(int)t0+(int)(lw+4u*j)-3; rv[32u+j]=(live&&ti>=0&&(u32)ti<T)?AF(((bi*T+(u32)ti)*H)+col):0.0f;}
    }
   }
  } else if(q>0u&&tap<K&&live){
   const u32 qq=q-1u; const u32 bi=qq/ntt, t0=(qq%ntt)*CONV_TR; const float* st=sm+(qq&1u)*STG;
   const u32 lo=max(t0,back); const u32 hi=min(t0+CONV_TR,T);
   #pragma unroll 8
   for(u32 ti=lo;ti<hi;ti++){sw=sw+st[(ti-t0)*32u+c]*st[CONV_TR*32u+(ti-t0+3u-back)*32u+c];}
   if(t0+CONV_TR>=T&&U(7u)==1u){for(u32 t2=0u;t2<back&&t2<T;t2++){sw=sw+CF(((bi*T+t2)*H)+col)*FF((bi*(K-1u)+(K-1u+t2-back))*H+col);}}
  }
  __syncthreads();
 }
 if(tap<K&&live){S_e[U(6u)+col*K+tap]=as_u32(EF(U(6u)+col*K+tap)+sw);}
}
// op 35 (MoE row sums per expert column): a thread per (expert, column) as
// in the WGSL, 32 to a block, its rows loaded 32 ahead of the sum.
extern "C" __global__ void __launch_bounds__(32) emb_x_moe_stats(EWD) {
 const u32 z=(blockIdx.x+blockIdx.y*gridDim.x)*32u+threadIdx.x;
 const u32 D=U(2u), cap=U(3u);
 if(z>=U(1u)*D){return;}
 const u32 who=z/D, col=z%D;
 const u32 n=min(BU(U(4u)+who),cap);
 float ss=0.0f; u32 q=0u;
 for(;q+32u<=n;q+=32u){
  float v[32];
  #pragma unroll
  for(u32 j=0u;j<32u;j++){v[j]=AF((who*cap+q+j)*D+col);}
  #pragma unroll
  for(u32 j=0u;j<32u;j++){ss=ss+v[j];}
 }
 for(;q<n;q++){ss=ss+AF((who*cap+q)*D+col);}
 S_c[U(5u)+who*D+col]=as_u32(ss);
}
// op 36 (MoE centre and balancing-bias update): the centres a thread per
// (expert, column) as in the WGSL; the bias of each live expert a block of
// its own, the residual rows loaded 2048 at a time and summed in order by
// thread 0 (the WGSL's column-0 thread).
extern "C" __global__ void __launch_bounds__(256) emb_x_moe_update(EWD) {
 __shared__ float rows_s[2048];
 const u32 E=U(1u), D=U(2u);
 const u32 nb=(E*D+255u)/256u;
 const u32 blk=blockIdx.x+blockIdx.y*gridDim.x;
 if(blk<nb){
  const u32 z=blk*256u+threadIdx.x; if(z>=E*D){return;}
  const u32 who=z/D, col=z%D; if(who<U(9u)){return;}
  const u32 cnt=DU(U(6u)+who); const u32 n_mu=min(cnt,U(12u));
  const float old=AF(U(3u)+who*D+col);
  if(n_mu>0u){const float sum=CF(U(5u)+who*D+col)/(float)(n_mu);S_a[U(3u)+who*D+col]=as_u32((1.0f-F(7u))*old+F(7u)*sum);}
  return;
 }
 const u32 who=blk-nb;
 if(who>=E||who<U(9u)||who>=U(11u)){return;}
 const u32 R=U(10u);
 float rs=0.0f;
 for(u32 r0=0u;r0<R;r0+=2048u){
  const u32 len=min(2048u,R-r0);
  for(u32 j=threadIdx.x;j<len;j+=256u){rows_s[j]=EF(r0+j);}
  __syncthreads();
  if(threadIdx.x==0u){
   #pragma unroll 16
   for(u32 j=0u;j<len;j++){rs=rs+rows_s[j];}}
  __syncthreads();
 }
 if(threadIdx.x==0u){
  const u32 cnt=DU(U(6u)+who);
  const float scale=rs/(float)(max(R,1u));const float frac=(float)(cnt)/(float)(max(R,1u));
  S_b[U(4u)+who]=as_u32(BF(U(4u)+who)+F(8u)*scale*(1.0f/(float)(U(1u))-frac));
 }
}
// ops 7, 8 on expert rows (p[2] = 1): SWIGLU_ROWS_BLOCKS blocks per expert
// stride over that expert's live elements (its first 64·y rows, y = word 1
// of its indirect record), each element the WGSL's formula. The grid is
// fixed: the dead rows (three quarters under dropless) cost no blocks.
#define SWIGLU_ROWS_BLOCKS 512u
extern "C" __global__ void __launch_bounds__(256) emb_x_swiglu_rows(EWD) {
 const u32 cap=U(3u), ii=U(4u), io=U(5u);
 const u32 blk=blockIdx.x+blockIdx.y*gridDim.x; const u32 ex=blk/SWIGLU_ROWS_BLOCKS; const u32 xb=blk%SWIGLU_ROWS_BLOCKS;
 if(ex*cap*ii>=U(1u)){return;}
 const u32 live=min(S_d[io+ex*3u+1u]*64u,cap)*ii; const u32 base=ex*cap*ii;
 for(u32 q=xb*256u+threadIdx.x;q<live;q+=SWIGLU_ROWS_BLOCKS*256u){const u32 z=base+q; S_c[z]=as_u32(silu(EWP, AF(z))*BF(z));}
}
extern "C" __global__ void __launch_bounds__(256) emb_x_swiglu_bwd_rows(EWD) {
 const u32 cap=U(3u), ii=U(4u), io=U(5u);
 const u32 blk=blockIdx.x+blockIdx.y*gridDim.x; const u32 ex=blk/SWIGLU_ROWS_BLOCKS; const u32 xb=blk%SWIGLU_ROWS_BLOCKS;
 if(ex*cap*ii>=U(1u)){return;}
 const u32 live=min(S_f[io+ex*3u+1u]*64u,cap)*ii; const u32 base=ex*cap*ii;
 for(u32 q=xb*256u+threadIdx.x;q<live;q+=SWIGLU_ROWS_BLOCKS*256u){
  const u32 z=base+q;
  const float gg=AF(z); const float uu=BF(z); const float dh=CF(z);
  const float sig=1.0f/(1.0f+exp(-gg)); const float sg=gg*sig; const float ds=sig*(1.0f+gg*(1.0f-sig));
  S_d[z]=as_u32(dh*ds*uu); S_e[z]=as_u32(dh*sg);
 }
}
// op 120 (CUDA only): the chunk chain of the WY forward (`gdn_wy::wy_fwd`'s
// loop) in one launch. A block per (b, h, 32 value columns) walks the chunks
// in order and does what the per-chunk operations did, with their
// arithmetic: S' = γ_end·S_c (op 68), R = V − Kγ·S_c (GEMM α −1, β 1),
// E' = T·R, U = β·E', Ũ = ρ·U (op 69), S_{c+1} = S' + K̂ᵀ·Ũ (GEMM β 1),
// each GEMM output one fma chain over k in order (gemm64 / gemm128) with
// their epilogue. It writes states, vcm (R), epm, um and utm as they did.
// Slots: a states, b γ, c Kγ, d K̂, e vcm, f T, g β, h ρ, i epm, j um,
// k utm. Shared: S [dk][32], Kγ then K̂ [64][dk], T [64][64], R, Ũ [64][32].
extern "C" __global__ void __launch_bounds__(256) emb_x_wy_fwd_chain(EWD) {
 extern __shared__ float sm[];
 const u32 T=U(2u), nv=U(3u), dk=U(4u), dv=U(5u), nch=U(8u);
 const u32 ss=dk*dv; const u32 ntile=dv/32u;
 const u32 blk=blockIdx.x+blockIdx.y*gridDim.x;
 const u32 bh=blk/ntile; const u32 j=(blk%ntile)*32u+(threadIdx.x&31u);
 if(bh>=U(1u)*nv){return;}
 const u32 bb=bh/nv, hh=bh%nv;
 const u32 tid=threadIdx.x, lane=tid&31u, w=tid>>5u;
 const u32 rpw=dk/8u;   // state rows per warp (dk <= 128)
 float* Ss=sm; float* Ks=Ss+dk*32u; float* Ts=Ks+64u*dk; float* Rs=Ts+4096u; float* Us=Rs+2048u;
 const u32 k4n=dk/4u;   // float4 per key row
 // a [64][dk] block of token rows (Kγ or K̂) as float4, every load of the thread in flight at once
 #define WY_LOAD_K(SRC) { float4 kv[8]; \
  _Pragma("unroll") for(u32 m=0u;m<8u;m++){const u32 q=tid+256u*m; kv[m]=make_float4(0.0f,0.0f,0.0f,0.0f); \
   if(q<16u*dk){const u32 i=q/k4n, k4=q%k4n; kv[m]=*((const float4*)((SRC)+(bb*T+tb+i)*nv*dk+hh*dk)+k4);}} \
  _Pragma("unroll") for(u32 m=0u;m<8u;m++){const u32 q=tid+256u*m; if(q<16u*dk){((float4*)Ks)[q]=kv[m];}} }
 {
  float sv[16];
  #pragma unroll
  for(u32 m=0u;m<16u;m++){const u32 r=w+8u*m; sv[m]=(r<dk)?AF((bh*(nch+1u))*ss+r*dv+j):0.0f;}
  #pragma unroll
  for(u32 m=0u;m<16u;m++){const u32 r=w+8u*m; if(r<dk){Ss[r*32u+lane]=sv[m];}}
 }
 for(u32 c=0u;c<nch;c++){
  const u32 tb=c*64u;
  {
   float4 tv[4];
   #pragma unroll
   for(u32 m=0u;m<4u;m++){tv[m]=*((const float4*)(S_f+(bh*nch+c)*4096u)+tid+256u*m);}
   WY_LOAD_K(S_c)
   #pragma unroll
   for(u32 m=0u;m<4u;m++){((float4*)Ts)[tid+256u*m]=tv[m];}
  }
  __syncthreads();
  {
   float acc[8];
   #pragma unroll
   for(u32 a=0u;a<8u;a++){acc[a]=0.0f;}
   for(u32 k=0u;k<dk;k+=4u){
    const float s0=Ss[k*32u+lane], s1=Ss[(k+1u)*32u+lane], s2=Ss[(k+2u)*32u+lane], s3=Ss[(k+3u)*32u+lane];
    #pragma unroll
    for(u32 a=0u;a<8u;a++){
     const float4 kv=*((const float4*)(Ks+(w*8u+a)*dk+k));
     acc[a]=fmaf(kv.x,s0,acc[a]); acc[a]=fmaf(kv.y,s1,acc[a]); acc[a]=fmaf(kv.z,s2,acc[a]); acc[a]=fmaf(kv.w,s3,acc[a]);
    }
   }
   #pragma unroll
   for(u32 a=0u;a<8u;a++){
    const u32 i=w*8u+a; const u32 ix=((bh*nch+c)*64u+i)*dv+j;
    float outv=-1.0f*acc[a]; outv=fmaf(1.0f,EF(ix),outv);
    S_e[ix]=as_u32(outv); Rs[i*32u+lane]=outv;
   }
  }
  __syncthreads();
  WY_LOAD_K(S_d)
  {
   float acc[8];
   #pragma unroll
   for(u32 a=0u;a<8u;a++){acc[a]=0.0f;}
   for(u32 k=0u;k<64u;k+=4u){
    const float r0=Rs[k*32u+lane], r1=Rs[(k+1u)*32u+lane], r2=Rs[(k+2u)*32u+lane], r3=Rs[(k+3u)*32u+lane];
    #pragma unroll
    for(u32 a=0u;a<8u;a++){
     const float4 tv=*((const float4*)(Ts+(w*8u+a)*64u+k));
     acc[a]=fmaf(tv.x,r0,acc[a]); acc[a]=fmaf(tv.y,r1,acc[a]); acc[a]=fmaf(tv.z,r2,acc[a]); acc[a]=fmaf(tv.w,r3,acc[a]);
    }
   }
   #pragma unroll
   for(u32 a=0u;a<8u;a++){
    const u32 i=w*8u+a; const u32 ix=((bh*nch+c)*64u+i)*dv+j; const u32 t=tb+i;
    const float ep=1.0f*acc[a];
    S_i[ix]=as_u32(ep);
    const float u=GF(bh*T+t)*ep; S_j[ix]=as_u32(u);
    const float ut=HF(bh*T+t)*u; S_k[ix]=as_u32(ut); Us[i*32u+lane]=ut;
   }
  }
  __syncthreads();
  {
   const float gend=BF(bh*T+tb+63u);
   float acc[16];
   #pragma unroll
   for(u32 a=0u;a<16u;a++){acc[a]=0.0f;}
   for(u32 kk=0u;kk<64u;kk++){
    const float uv=Us[kk*32u+lane];
    float kr[16];
    #pragma unroll
    for(u32 a4=0u;a4<4u;a4++){
     if(a4*4u<rpw){const float4 kv=*((const float4*)(Ks+kk*dk+w*rpw+a4*4u)); kr[a4*4u]=kv.x; kr[a4*4u+1u]=kv.y; kr[a4*4u+2u]=kv.z; kr[a4*4u+3u]=kv.w;}
    }
    #pragma unroll
    for(u32 a=0u;a<16u;a++){if(a<rpw){acc[a]=fmaf(kr[a],uv,acc[a]);}}
   }
   #pragma unroll
   for(u32 a=0u;a<16u;a++){
    if(a<rpw){
     const u32 r=w*rpw+a;
     // op 68 stored this product before the GEMM read it: no fusion into the sum
     const float sp=__fmul_rn(gend,Ss[r*32u+lane]);
     float outv=1.0f*acc[a]; outv=fmaf(1.0f,sp,outv);
     S_a[(bh*(nch+1u)+c+1u)*ss+r*dv+j]=as_u32(outv); Ss[r*32u+lane]=outv;
    }
   }
  }
  __syncthreads();
 }
 #undef WY_LOAD_K
}
// op 121 (CUDA only): the chunk chain of the WY backward (`gdn_wy::wy_bwd`'s
// loop) in one launch. Only dS is carried and every product of the chain is
// separable by value column, so a block per (b, h, 32 columns) walks the
// chunks in reverse with its dS columns in shared memory. Per chunk, with the
// arithmetic of the per-chunk operations: dsl[c] = dS_{c+1}; dŨ = K̂_c·dS_{c+1}
// (GEMM β 0); op 70's dU = dU⁰ + ρ·dŨ and dE' = β·dU; dR = T_cᵀ·dE' (GEMM β 0,
// into dcv); dS_c = (dS⁰_c + γ_end·dS_{c+1}) (op 71) − Kγ_cᵀ·dR (GEMM α −1,
// β 1). dŨ and dU stay for op 122 (op 70's dρ, dβ dots), dE' for the batched
// dT, dS_{c+1} for the batched dK̂ and op 123 (dγ_end); dS_0 goes to dlive.
// Slots: a dlive, b dU⁰ → dU (chunk-outer), c dS⁰ (chunk-outer), d K̂, e Kγ,
// f β, g ρ, h γ, j T, k dsl (chunk-outer), l dE' (chunk-outer), m dŨ
// (chunk-outer), nbuf dcv. Shared (80 KB at dk = 128, within an A40's 99):
// dS [dk][32], K̂ then Kγ [64][dk] (Kγ is loaded into registers while dR is
// summed and stored once K̂ is done), T [64][64], dE' and dR [64][32].
extern "C" __global__ void __launch_bounds__(256) emb_x_wy_bwd_chain(EWD) {
 extern __shared__ float sm[];
 const u32 T=U(2u), nv=U(3u), dk=U(4u), dv=U(5u), cdim=U(6u), nch=U(8u);
 const u32 NB=U(1u)*nv; const u32 ss=dk*dv; const u32 ntile=dv/32u;
 const u32 blk=blockIdx.x+blockIdx.y*gridDim.x;
 const u32 bh=blk/ntile; const u32 j=(blk%ntile)*32u+(threadIdx.x&31u);
 if(bh>=NB){return;}
 const u32 bb=bh/nv, hh=bh%nv;
 const u32 tid=threadIdx.x, lane=tid&31u, w=tid>>5u;
 const u32 rpw=dk/8u; const u32 k4n=dk/4u;
 float* Sd=sm; float* Ks=Sd+dk*32u; float* Ts=Ks+64u*dk; float* Ep=Ts+4096u; float* Dr=Ep+2048u; float* Kg=Ks;
 {
  float v[16];
  #pragma unroll
  for(u32 m=0u;m<16u;m++){const u32 r=w+8u*m; v[m]=(r<dk)?AF(bh*ss+r*dv+j):0.0f;}
  #pragma unroll
  for(u32 m=0u;m<16u;m++){const u32 r=w+8u*m; if(r<dk){Sd[r*32u+lane]=v[m];}}
 }
 __syncthreads();
 for(u32 cr=0u;cr<nch;cr++){
  const u32 c=nch-1u-cr; const u32 tb=c*64u;
  {
   float4 kh[8]; float4 tv[4];
   #pragma unroll
   for(u32 m=0u;m<8u;m++){
    const u32 q=tid+256u*m; kh[m]=make_float4(0.0f,0.0f,0.0f,0.0f);
    if(q<16u*dk){const u32 i=q/k4n, k4=q%k4n; kh[m]=*((const float4*)(S_d+(bb*T+tb+i)*nv*dk+hh*dk)+k4);}
   }
   #pragma unroll
   for(u32 m=0u;m<4u;m++){tv[m]=*((const float4*)(S_j+(bh*nch+c)*4096u)+tid+256u*m);}
   #pragma unroll
   for(u32 m=0u;m<16u;m++){const u32 r=w+8u*m; if(r<dk){S_k[(c*NB+bh)*ss+r*dv+j]=as_u32(Sd[r*32u+lane]);}}
   #pragma unroll
   for(u32 m=0u;m<8u;m++){const u32 q=tid+256u*m; if(q<16u*dk){((float4*)Ks)[q]=kh[m];}}
   #pragma unroll
   for(u32 m=0u;m<4u;m++){((float4*)Ts)[tid+256u*m]=tv[m];}
  }
  __syncthreads();
  {
   // dŨ = K̂·dS: rows w*8 .. w*8+7, column j
   float acc[8];
   #pragma unroll
   for(u32 a=0u;a<8u;a++){acc[a]=0.0f;}
   for(u32 k=0u;k<dk;k+=4u){
    const float s0=Sd[k*32u+lane], s1=Sd[(k+1u)*32u+lane], s2=Sd[(k+2u)*32u+lane], s3=Sd[(k+3u)*32u+lane];
    #pragma unroll
    for(u32 a=0u;a<8u;a++){
     const float4 kv=*((const float4*)(Ks+(w*8u+a)*dk+k));
     acc[a]=fmaf(kv.x,s0,acc[a]); acc[a]=fmaf(kv.y,s1,acc[a]); acc[a]=fmaf(kv.z,s2,acc[a]); acc[a]=fmaf(kv.w,s3,acc[a]);
    }
   }
   #pragma unroll
   for(u32 a=0u;a<8u;a++){
    const u32 i=w*8u+a; const u32 t=tb+i; const u32 ix=((c*NB+bh)*64u+i)*dv+j;
    const float rh=GF(bh*T+t); const float be=FF(bh*T+t);
    const float dutv=1.0f*acc[a];
    const float duv=BF(ix)+rh*dutv;
    const float ep=be*duv;
    S_m[ix]=as_u32(dutv); S_b[ix]=as_u32(duv); S_l[ix]=as_u32(ep); Ep[i*32u+lane]=ep;
   }
  }
  __syncthreads();
  {
   // Kγ_c into registers now, into the K̂ buffer after dR (K̂ is done)
   float4 kgv[8];
   #pragma unroll
   for(u32 m=0u;m<8u;m++){
    const u32 q=tid+256u*m; kgv[m]=make_float4(0.0f,0.0f,0.0f,0.0f);
    if(q<16u*dk){const u32 i=q/k4n, k4=q%k4n; kgv[m]=*((const float4*)(S_e+(bb*T+tb+i)*nv*dk+hh*dk)+k4);}
   }
   // dR = Tᵀ·dE': rows w*8 .. w*8+7 (A(i, k) = T[k][i]), column j
   float acc[8];
   #pragma unroll
   for(u32 a=0u;a<8u;a++){acc[a]=0.0f;}
   for(u32 k=0u;k<64u;k++){
    const float4 t0=*((const float4*)(Ts+k*64u+w*8u)); const float4 t1=*((const float4*)(Ts+k*64u+w*8u+4u));
    const float ev=Ep[k*32u+lane];
    acc[0]=fmaf(t0.x,ev,acc[0]); acc[1]=fmaf(t0.y,ev,acc[1]); acc[2]=fmaf(t0.z,ev,acc[2]); acc[3]=fmaf(t0.w,ev,acc[3]);
    acc[4]=fmaf(t1.x,ev,acc[4]); acc[5]=fmaf(t1.y,ev,acc[5]); acc[6]=fmaf(t1.z,ev,acc[6]); acc[7]=fmaf(t1.w,ev,acc[7]);
   }
   #pragma unroll
   for(u32 a=0u;a<8u;a++){
    const u32 i=w*8u+a; const float dr=1.0f*acc[a];
    S_nbuf[(bb*T+tb+i)*cdim+2u*nv*dk+hh*dv+j]=as_u32(dr); Dr[i*32u+lane]=dr;
   }
   #pragma unroll
   for(u32 m=0u;m<8u;m++){const u32 q=tid+256u*m; if(q<16u*dk){((float4*)Kg)[q]=kgv[m];}}
  }
  __syncthreads();
  {
   // dS_c: rows w*rpw .. (A(r, kk) = Kγ[kk][r]), column j
   const float gend=HF(bh*T+tb+63u);
   float acc[16];
   #pragma unroll
   for(u32 a=0u;a<16u;a++){acc[a]=0.0f;}
   for(u32 kk=0u;kk<64u;kk++){
    const float dr=Dr[kk*32u+lane];
    float kr[16];
    #pragma unroll
    for(u32 a4=0u;a4<4u;a4++){
     if(a4*4u<rpw){const float4 kv=*((const float4*)(Kg+kk*dk+w*rpw+a4*4u)); kr[a4*4u]=kv.x; kr[a4*4u+1u]=kv.y; kr[a4*4u+2u]=kv.z; kr[a4*4u+3u]=kv.w;}
    }
    #pragma unroll
    for(u32 a=0u;a<16u;a++){if(a<rpw){acc[a]=fmaf(kr[a],dr,acc[a]);}}
   }
   #pragma unroll
   for(u32 a=0u;a<16u;a++){
    if(a<rpw){
     const u32 r=w*rpw+a;
     const float dq=CF((c*NB+bh)*ss+r*dv+j); const float dl=Sd[r*32u+lane];
     const float dsnv=dq+gend*dl;
     float outv=-1.0f*acc[a]; outv=fmaf(1.0f,dsnv,outv);
     Sd[r*32u+lane]=outv;
    }
   }
  }
  __syncthreads();
 }
 #pragma unroll
 for(u32 m=0u;m<16u;m++){const u32 r=w+8u*m; if(r<dk){S_a[bh*ss+r*dv+j]=as_u32(Sd[r*32u+lane]);}}
}
// op 122 (CUDA only): op 70's dρ, dβ dots for every chunk after the chain:
// a warp per (b, h, chunk, row), lane 0's sums in op 70's order.
// Slots: a dŨ (chunk-outer), b U, c dU (chunk-outer), d E', e dρ, f dβ.
extern "C" __global__ void __launch_bounds__(256) emb_x_wy_du_dots(EWD) {
 __shared__ float xs[8*512];
 const u32 T=U(2u), nv=U(3u), dv=U(5u), nch=U(8u); const u32 NB=U(1u)*nv;
 const u32 w=threadIdx.x>>5u, lane=threadIdx.x&31u;
 const u32 z=(blockIdx.x+blockIdx.y*gridDim.x)*8u+w;
 if(z>=NB*nch*64u){return;}
 const u32 bh=z/(nch*64u); const u32 rem=z%(nch*64u); const u32 c=rem/64u, i=rem%64u;
 const u32 co=((c*NB+bh)*64u+i)*dv; const u32 bm=((bh*nch+c)*64u+i)*dv;
 float* v=xs+w*512u;
 for(u32 j=lane;j<dv;j+=32u){v[j]=AF(co+j); v[128u+j]=BF(bm+j); v[256u+j]=CF(co+j); v[384u+j]=DF(bm+j);}
 __syncwarp();
 if(lane==0u){
  float dr=0.0f; float dbv=0.0f;
  #pragma unroll 16
  for(u32 vv=0u;vv<dv;vv++){const float dutv=v[vv]; const float duv=v[256u+vv]; dr=dr+dutv*v[128u+vv]; dbv=dbv+duv*v[384u+vv];}
  S_e[bh*T+c*64u+i]=as_u32(dr); S_f[bh*T+c*64u+i]=as_u32(dbv);
 }
}
// op 123 (CUDA only): the dγ_end partials of every chunk after the chain (op
// 72 over (b, h, chunk)), then op 80's fold per (b, h, chunk) in order.
// Slots: a dS_{c+1} (chunk-outer), b states, d partials [NB, nch, 64], e dγ_end.
extern "C" __global__ void __launch_bounds__(256) emb_x_wy_dgend_all(EWD) {
 __shared__ float sa[8*256]; __shared__ float sb[8*256];
 const u32 ss=U(4u)*U(5u), nch=U(8u); const u32 NB=U(1u)*U(3u);
 const u32 w=threadIdx.x>>5u, lane=threadIdx.x&31u;
 const u32 z=(blockIdx.x+blockIdx.y*gridDim.x)*8u+w;
 if(z>=NB*nch*64u){return;}
 const u32 bh=z/(nch*64u); const u32 rem=z%(nch*64u); const u32 c=rem/64u, pi=rem%64u;
 const u32 ch=(ss+63u)/64u; const u32 lo=pi*ch; const u32 hi=min(lo+ch,ss);
 const u32 abase=(c*NB+bh)*ss, bbase=(bh*(nch+1u)+c)*ss;
 float* va=sa+w*256u; float* vb=sb+w*256u;
 float s=0.0f;
 for(u32 r0=lo;r0<hi;r0+=256u){
  const u32 len=min(256u,hi-r0);
  for(u32 j=lane;j<len;j+=32u){va[j]=AF(abase+r0+j); vb[j]=BF(bbase+r0+j);}
  __syncwarp();
  if(lane==0u){
   #pragma unroll 16
   for(u32 j=0u;j<len;j++){s=s+va[j]*vb[j];}}
  __syncwarp();
 }
 if(lane==0u){S_d[(bh*nch+c)*64u+pi]=as_u32(s);}
}
extern "C" __global__ void __launch_bounds__(256) emb_x_wy_dgend_fold_all(EWD) {
 const u32 nch=U(8u); const u32 NB=U(1u)*U(3u);
 const u32 z=(blockIdx.x+blockIdx.y*gridDim.x)*256u+threadIdx.x;
 if(z>=NB*nch){return;}
 float s=0.0f;
 for(u32 i=0u;i<64u;i++){s=s+DF(z*64u+i);}
 S_e[z]=as_u32(s);
}
// op 64 (WY decay scalars of a chunk): a warp per (b, h, chunk). The lanes
// form every token's softplus and β, lane 0 runs the cumulative log decay in
// order, the lanes write γ and ρ from it; the WGSL's expressions.
// Slots: a a_pre, b b_pre, c arena, d lg, e γ, f β, g ρ.
extern "C" __global__ void __launch_bounds__(256) emb_x_wy_scal(EWD) {
 __shared__ float sh[8*64];
 const u32 nch=U(8u), T=U(2u), ld=U(7u), nv=U(3u);
 const u32 w=threadIdx.x>>5u, lane=threadIdx.x&31u;
 const u32 z=(blockIdx.x+blockIdx.y*gridDim.x)*8u+w;
 if(z>=U(1u)*nv*nch){return;}
 const u32 bh=z/nch, cc=z%nch; const u32 bb=bh/nv, hh=bh%nv;
 const float ea=exp(CF(U(9u)+hh)); const float dt=CF(U(10u)+hh);
 float* v=sh+w*64u;
 for(u32 ii=lane;ii<64u;ii+=32u){
  const u32 t=cc*64u+ii; const u32 row=bb*T+t;
  const float aa=AF(row*ld+hh)+dt; v[ii]=gdn_softplus(EWP, aa);
  S_f[bh*T+t]=as_u32(((U(11u)==1u)?(1.0f):(1.0f/(1.0f+exp(-BF(row*ld+hh))))));
 }
 __syncwarp();
 if(lane==0u){float acc=0.0f; for(u32 ii=0u;ii<64u;ii++){acc=acc-ea*v[ii]; v[ii]=acc;}}
 __syncwarp();
 const float lend=v[63];
 for(u32 ii=lane;ii<64u;ii+=32u){const u32 t=cc*64u+ii; const float acc=v[ii]; S_d[bh*T+t]=as_u32(acc); S_e[bh*T+t]=as_u32(exp(acc)); S_g[bh*T+t]=as_u32(exp(lend-acc));}
}
// op 67 (T = (I+L)⁻¹ by forward substitution): a block of 64 per (b, h,
// chunk) with L in shared memory; thread `col` substitutes its column in the
// WGSL's order, reading its earlier rows from shared memory. Slots: a L, b T.
extern "C" __global__ void __launch_bounds__(64) emb_x_wy_ut(EWD) {
 __shared__ float La[4096]; __shared__ float Tb[4096];
 const u32 bhc=blockIdx.x+blockIdx.y*gridDim.x; if(bhc>=U(1u)*U(3u)*U(8u)){return;}
 const u32 col=threadIdx.x; const u32 base=bhc*4096u;
 for(u32 q=col;q<4096u;q+=64u){La[q]=AF(base+q);}
 __syncthreads();
 for(u32 ii=0u;ii<64u;ii++){
  float s=((ii==col)?(1.0f):(0.0f));
  for(u32 jj=0u;jj<ii;jj++){s=s-La[ii*64u+jj]*Tb[jj*64u+col];}
  Tb[ii*64u+col]=s; S_b[base+ii*64u+col]=as_u32(s);
 }
}
// op 75 (per-token decay gradient of a chunk): a warp per (b, h, chunk);
// lane 0 sums Σ dρ·ρ, the lanes form every token's term, lane 0 accumulates
// them from the chunk's end in order. The WGSL's expressions.
// Slots: a dots, g dρ, h dlg mask term, i dγ_end, j ρ, k γ, l d log α.
extern "C" __global__ void __launch_bounds__(256) emb_x_wy_dlg(EWD) {
 __shared__ float sh[8*192];
 const u32 nch=U(8u), T=U(2u), nv=U(3u);
 const u32 w=threadIdx.x>>5u, lane=threadIdx.x&31u;
 const u32 z=(blockIdx.x+blockIdx.y*gridDim.x)*8u+w;
 if(z>=U(1u)*nv*nch){return;}
 const u32 bh=z/nch, cc=z%nch;
 float* gv=sh+w*192u; float* jv=gv+64u; float* tv=gv+128u;
 for(u32 ii=lane;ii<64u;ii+=32u){const u32 t=bh*T+cc*64u+ii; gv[ii]=GF(t); jv[ii]=JF(t);}
 __syncwarp();
 float srho=0.0f;
 if(lane==0u){for(u32 ii=0u;ii<64u;ii++){srho=srho+gv[ii]*jv[ii];}}
 srho=__shfl_sync(0xffffffffu,srho,0);
 for(u32 ii=lane;ii<64u;ii+=32u){
  const u32 tt=bh*T+cc*64u+ii;
  float dlg=KF(tt)*AF(tt)+HF(tt)-GF(tt)*JF(tt); if(ii==63u){dlg=dlg+srho+KF(tt)*IF(bh*nch+cc);}
  tv[ii]=dlg;
 }
 __syncwarp();
 if(lane==0u){float acc=0.0f; for(u32 ir=64u;ir>0u;ir--){const u32 ii=ir-1u; acc=acc+tv[ii]; tv[ii]=acc;}}
 __syncwarp();
 for(u32 ii=lane;ii<64u;ii+=32u){S_l[bh*T+cc*64u+ii]=as_u32(tv[ii]);}
}
// ops 33, 34 (MoE row gather into the expert slots / scatter-add back): a
// block of 64 per row, float4 columns (h % 4 == 0); a copy and a plain add.
extern "C" __global__ void __launch_bounds__(64) emb_x_moe_gather(EWD) {
 const u32 row=blockIdx.x+blockIdx.y*gridDim.x; if(row>=U(1u)){return;}
 const u32 H=U(2u), cap=U(3u); const u32 who=S_b[row], slot=S_c[row];
 if(slot>=cap){return;}
 const float4* src=(const float4*)(S_a+row*H); float4* dst=(float4*)(S_d+(who*cap+slot)*H);
 for(u32 q=threadIdx.x;q<H/4u;q+=64u){dst[q]=src[q];}
}
extern "C" __global__ void __launch_bounds__(64) emb_x_moe_scatter(EWD) {
 const u32 row=blockIdx.x+blockIdx.y*gridDim.x; if(row>=U(1u)){return;}
 const u32 H=U(2u), cap=U(3u); const u32 who=S_b[row], slot=S_c[row];
 if(slot>=cap){return;}
 float4* dst=(float4*)(S_a+row*H); const float4* y=(const float4*)(S_d+(who*cap+slot)*H);
 for(u32 q=threadIdx.x;q<H/4u;q+=64u){float4 a=dst[q]; const float4 b=y[q]; a.x=a.x+b.x; a.y=a.y+b.y; a.z=a.z+b.z; a.w=a.w+b.w; dst[q]=a;}
}
// op 1 in table mode (p[35] = 1, the tiled scalar contract: the hierarchical
// head's per-cluster GEMMs): 64×64 output tiles, 4×4 per thread, in place of
// one output per thread. Every output is the WGSL's sum (k in order, the same
// expression) and epilogue; a tile wholly past its batch's rows leaves first.
extern "C" __global__ void __launch_bounds__(256) emb_x_gemm_table(EWD) {
 __shared__ float As[16][68]; __shared__ float Bs[16][68];
 const u32 m0=U(2u), n0=U(3u), k0=U(4u), ta=U(5u), tb=U(6u), lda=U(7u), ldb=U(8u), ldc=U(9u);
 const u32 ao=U(10u), bo=U(11u), co=U(12u); const float al=F(13u), be=F(14u); const u32 nb=U(15u), nh=U(16u), nc=U(17u);
 const u32 tmn=(m0+63u)/64u, tnn=(n0+63u)/64u; const u32 per=tmn*tnn;
 const u32 blk=blockIdx.x+blockIdx.y*gridDim.x; const u32 batch_id=blk/per;
 if(batch_id>=nb*nh*nc){return;}
 const u32 tile=blk%per; const u32 r0=(tile/tnn)*64u, c0=(tile%tnn)*64u;
 const u32 bc=batch_id%nc; const u32 bh=batch_id/nc; const u32 ch=bh%nh; const u32 cb=bh/nh;
 u32 oa=ao+cb*U(18u)+ch*U(19u)+bc*U(20u); u32 ob=bo+cb*U(21u)+ch*U(22u)+bc*U(23u); u32 oc=co+cb*U(24u)+ch*U(25u)+bc*U(26u);
 u32 kdim=k0; u32 mrows=m0;
 {const u32 tq=U(36u)+batch_id*4u; oa=oa+DU(tq); ob=ob+DU(tq+1u); oc=oc+DU(tq+2u); const u32 rz=DU(tq+3u); if(U(37u)==1u){kdim=min(rz,k0);}else{mrows=min(rz,m0);}}
 if(r0>=mrows){return;}
 const u32 tid=threadIdx.x; const u32 ty=tid>>4u, tx=tid&15u;
 float acc[4][4];
 #pragma unroll
 for(u32 i=0u;i<4u;i++){
  #pragma unroll
  for(u32 j=0u;j<4u;j++){acc[i][j]=0.0f;}
 }
 for(u32 kb=0u;kb<kdim;kb+=16u){
  #pragma unroll
  for(u32 e=0u;e<4u;e++){
   const u32 idx=tid+256u*e;
   {const u32 row=idx>>4u, kk=idx&15u; const u32 gr=r0+row, gk=kb+kk; float av=0.0f;
    if(gr<mrows&&gk<kdim){av=AF((ta==1u)?(oa+gk*lda+gr):(oa+gr*lda+gk));} As[kk][row]=av;}
   {const u32 kk=idx>>6u, col=idx&63u; const u32 gk=kb+kk, gc=c0+col; float bv=0.0f;
    if(gk<kdim&&gc<n0){bv=BF((tb==1u)?(ob+gc*ldb+gk):(ob+gk*ldb+gc));} Bs[kk][col]=bv;}
  }
  __syncthreads();
  const u32 kn=min(16u,kdim-kb);
  for(u32 q=0u;q<kn;q++){
   float a[4]; float b[4];
   #pragma unroll
   for(u32 i=0u;i<4u;i++){a[i]=As[q][ty*4u+i]; b[i]=Bs[q][tx*4u+i];}
   #pragma unroll
   for(u32 i=0u;i<4u;i++){
    #pragma unroll
    for(u32 j=0u;j<4u;j++){acc[i][j]=acc[i][j]+a[i]*b[j];}
   }
  }
  __syncthreads();
 }
 #pragma unroll
 for(u32 i=0u;i<4u;i++){
  #pragma unroll
  for(u32 j=0u;j<4u;j++){
   const u32 rr=r0+ty*4u+i, cc=c0+tx*4u+j;
   if(rr<mrows&&cc<n0){
    const u32 out_ix=oc+rr*ldc+cc;
    float outv=al*acc[i][j]+be*CF(out_ix);
    if(U(27u)==1u&&cc>rr){outv=0.0f;}
    S_c[out_ix]=as_u32(outv);
   }
  }
 }
}
// entry 10 (softmax backward over the rows of P): a block per row as in the
// WGSL. P is read once into registers and dP only where P ≠ 0: off the band
// P is exactly 0 and its term adds nothing to the lane sums (which never sit
// at −0). The lane sums in lane order and the WGSL's tree; dS = P·(dP − Σ)
// where P ≠ 0 and +0 elsewhere (the WGSL's 0·(dP − Σ) is ±0, which every
// later sum absorbs the same way).
#define SMB_PER 14u
extern "C" __global__ void __launch_bounds__(256) emb_x_softmax_bwd(EWD) {
 __shared__ float red[256];
 const u32 lid=threadIdx.x; const u32 zrow=blockIdx.x+blockIdx.y*gridDim.x;
 if(zrow>=U(1u)*U(2u)){return;}
 const u32 t=U(1u), ld=U(5u); const u32 block=zrow/t, row=zrow%t;
 const u32 pb=U(3u)+block*t*ld+row*ld; const u32 db=U(4u)+block*t*ld+row*ld;
 float pv[SMB_PER]; float dv[SMB_PER];
 #pragma unroll
 for(u32 i=0u;i<SMB_PER;i++){const u32 q=lid+256u*i; pv[i]=0.0f; dv[i]=0.0f; if(q<ld){pv[i]=AF(pb+q);}}
 #pragma unroll
 for(u32 i=0u;i<SMB_PER;i++){const u32 q=lid+256u*i; if(q<ld&&pv[i]!=0.0f){dv[i]=BF(db+q);}}
 float dot=0.0f;
 #pragma unroll
 for(u32 i=0u;i<SMB_PER;i++){if(pv[i]!=0.0f){dot=dot+pv[i]*dv[i];}}
 red[lid]=dot; __syncthreads();
 for(u32 st=128u;st>0u;st=st>>1u){if(lid<st){red[lid]=red[lid]+red[lid+st];} __syncthreads();}
 const float gdot=red[0];
 #pragma unroll
 for(u32 i=0u;i<SMB_PER;i++){const u32 q=lid+256u*i; if(q<ld){S_b[db+q]=as_u32((pv[i]!=0.0f)?(pv[i]*(dv[i]-gdot)):0.0f);}}
}
// op 84 (route finalize): a warp per row, the row loaded coalesced, lane 0's
// norm and expert scan in the WGSL's order.
extern "C" __global__ void __launch_bounds__(256) emb_x_route_fin(EWD) {
 extern __shared__ float xrow[];
 const u32 rows=U(1u), hh=U(2u), ee=U(3u), kk=U(4u), os=U(8u), om=U(9u);
 const u32 w=threadIdx.x>>5u, lane=threadIdx.x&31u;
 const u32 row=(blockIdx.x+blockIdx.y*gridDim.x)*8u+w;
 if(row>=rows){return;}
 float* xs=xrow+w*hh;
 for(u32 q=lane;q<hh;q+=32u){xs[q]=AF(row*hh+q);}
 __syncwarp();
 if(lane==0u){
  float n2=0.0f;
  #pragma unroll 16
  for(u32 col=0u;col<hh;col++){const float v=xs[col]; n2=n2+v*v;}
  float best=3.4e38f; float best_raw=0.0f; u32 bi=0u;
  for(u32 q=0u;q<ee;q++){
   float proj=0.0f;
   for(u32 qi=0u;qi<kk;qi++){const u32 ix=q*kk+qi; const float pp=BF(row*ee*kk+ix)-BF(om+ix); proj=proj+pp*pp;}
   const float raw=n2-2.0f*BF(os+row*ee+q)+BF(om+ee*kk+q)-proj; const float score=raw-CF(U(7u)+q);
   if(score<best){best=score;best_raw=raw;bi=q;}
  }
  S_d[row]=bi; S_e[row]=as_u32(best_raw);
 }
}
// op 23 (causal conv forward): a thread per (sequence, 8 steps, channel),
// each output summed over the taps in the WGSL's order. Away from the start
// of a sequence the 8+K−1 inputs are loaded together (static registers);
// the first K−1 steps (history or a skipped tap) take the WGSL's form.
extern "C" __global__ void __launch_bounds__(256) emb_x_conv_fwd(EWD) {
 const u32 B=U(1u), T=U(2u), H=U(3u), K=U(4u), wo=U(5u);
 const u32 nt=(T+7u)/8u;
 const u32 z=(blockIdx.x+blockIdx.y*gridDim.x)*256u+threadIdx.x;
 if(z>=B*nt*H){return;}
 const u32 col=z%H; const u32 bt=z/H; const u32 batch=bt/nt; const u32 t0=(bt%nt)*8u;
 if(t0>=K-1u&&t0+8u<=T){
  float w[8]; float xv[15];
  #pragma unroll
  for(u32 q=0u;q<8u;q++){w[q]=(q<K)?BF(wo+col*K+q):0.0f;}
  #pragma unroll
  for(u32 j=0u;j<15u;j++){xv[j]=(j<K+7u)?AF((batch*T+t0+j-(K-1u))*H+col):0.0f;}
  #pragma unroll
  for(u32 i=0u;i<8u;i++){
   float s=0.0f;
   #pragma unroll
   for(u32 q=0u;q<8u;q++){if(q<K){s=s+w[q]*xv[i+q];}}
   S_c[((batch*T+t0+i)*H)+col]=as_u32(s);
  }
  return;
 }
 for(u32 t=t0;t<min(t0+8u,T);t++){
  float s=0.0f;
  for(u32 q=0u;q<K;q++){
   const u32 back=K-1u-q;
   if(t>=back){s=s+BF(wo+col*K+q)*AF(((batch*T+t-back)*H)+col);}
   else if(U(7u)==1u){const u32 hr=K-1u+t-back; s=s+BF(wo+col*K+q)*DF((batch*(K-1u)+hr)*H+col);}
  }
  S_c[((batch*T+t)*H)+col]=as_u32(s);
 }
}
// op 24, dx: a thread per (sequence, 8 steps, channel), each output summed
// over the taps in the WGSL's order (q ascending: dy at t+K−1 down to t).
// Away from the end of a sequence the 8+K−1 values of dy are loaded
// together; the last steps take the WGSL's form.
extern "C" __global__ void __launch_bounds__(256) emb_x_conv_dx(EWD) {
 const u32 B=U(1u), T=U(2u), H=U(3u), K=U(4u);
 const u32 nt=(T+7u)/8u;
 const u32 z=(blockIdx.x+blockIdx.y*gridDim.x)*256u+threadIdx.x;
 if(z>=B*nt*H){return;}
 const u32 col=z%H; const u32 bt=z/H; const u32 batch=bt/nt; const u32 t0=(bt%nt)*8u;
 if(t0+7u+K-1u<T){
  // wr[j] = tap K−1−j: the tap that meets dy at t + j
  float wr[8]; float yv[15];
  #pragma unroll
  for(u32 j=0u;j<8u;j++){wr[j]=(j<K)?BF(U(5u)+col*K+(K-1u-j)):0.0f;}
  #pragma unroll
  for(u32 j=0u;j<15u;j++){yv[j]=(j<K+7u)?CF(((batch*T+t0+j)*H)+col):0.0f;}
  #pragma unroll
  for(u32 i=0u;i<8u;i++){
   float s=0.0f;
   #pragma unroll
   for(int j=7;j>=0;j--){if((u32)j<K){s=s+wr[j]*yv[i+(u32)j];}}
   S_d[((batch*T+t0+i)*H)+col]=as_u32(s);
  }
  return;
 }
 for(u32 t=t0;t<min(t0+8u,T);t++){
  float s=0.0f;
  for(u32 q=0u;q<K;q++){const u32 back=K-1u-q; const u32 dst=t+back; if(dst<T){s=s+BF(U(5u)+col*K+q)*CF(((batch*T+dst)*H)+col);}}
  S_d[((batch*T+t)*H)+col]=as_u32(s);
 }
}
"#;

/// The kernels of `NATIVE_CUDA`.
struct Native {
    route_group: cs::CUfunction,
    rms_fwd: cs::CUfunction,
    sumsq: cs::CUfunction,
    rms_dw_part: cs::CUfunction,
    rms_dw_tree: cs::CUfunction,
    wy_du: cs::CUfunction,
    wy_dgend: cs::CUfunction,
    wy_dlg_dot: cs::CUfunction,
    wy_dscal: cs::CUfunction,
    conv_bwd: cs::CUfunction,
    /// `emb_x_conv_dw`; null when the card refuses its shared memory
    conv_dw: cs::CUfunction,
    moe_stats: cs::CUfunction,
    moe_update: cs::CUfunction,
    swiglu_rows: cs::CUfunction,
    swiglu_bwd_rows: cs::CUfunction,
    conv_fwd: cs::CUfunction,
    conv_dx: cs::CUfunction,
    /// `emb_x_wy_fwd_chain`; null when the card refuses its shared memory
    wy_fwd_chain: cs::CUfunction,
    /// `emb_x_wy_bwd_chain`; null when the card refuses its shared memory
    wy_bwd_chain: cs::CUfunction,
    wy_du_dots: cs::CUfunction,
    wy_dgend_all: cs::CUfunction,
    wy_dgend_fold_all: cs::CUfunction,
    route_fin: cs::CUfunction,
    moe_gather: cs::CUfunction,
    moe_scatter: cs::CUfunction,
    gemm_table: cs::CUfunction,
    softmax_bwd: cs::CUfunction,
    wy_scal: cs::CUfunction,
    wy_ut: cs::CUfunction,
    wy_dlg: cs::CUfunction,
    /// Operation codes kept on the WGSL form (`CMF_CUDA_NATIVE_SKIP=70,76`).
    skip: Vec<u32>,
}

/// Dynamic shared memory of `emb_x_conv_dw`: two stages of 128 rows of dy
/// and 131 of x, 32 channels each.
const CONV_DW_SMEM: u32 = 2 * (128 + 131) * 32 * 4;

/// Dynamic shared memory of `emb_x_wy_fwd_chain` for key width `dk`.
fn wy_chain_smem(dk: u32) -> u32 {
    (dk * 32 + 64 * dk + 4096 + 2048 + 2048) * 4
}

/// The op code of the WY forward chain (CUDA only: no WGSL form).
const OP_WY_FWD_CHAIN: u32 = 120;
/// The WY backward chain and its two batched companions (CUDA only).
const OP_WY_BWD_CHAIN: u32 = 121;
const OP_WY_DU_DOTS: u32 = 122;
const OP_WY_DGEND_ALL: u32 = 123;
/// Dynamic shared memory of `emb_x_wy_bwd_chain` for key width `dk`.
fn wy_bwd_chain_smem(dk: u32) -> u32 {
    (dk * 32 + 64 * dk + 4096 + 2048 + 2048) * 4
}

impl Ctx {
    /// The WY forward chain runs as one CUDA kernel for this geometry
    /// (`Cmd::gdn_wy_fwd_chain`); otherwise the per-chunk operations.
    pub fn wy_fwd_chain_ok(&self, dk: usize, dv: usize) -> bool {
        self.inner.native.as_ref().is_some_and(|x| !x.wy_fwd_chain.is_null() && !x.skip.contains(&OP_WY_FWD_CHAIN))
            && dk > 0
            && dk <= 128
            && dk % 32 == 0
            && dv > 0
            && dv % 32 == 0
    }
    /// The WY backward chain runs as one CUDA kernel (`Cmd::gdn_wy_bwd_chain`).
    pub fn wy_bwd_chain_ok(&self, dk: usize, dv: usize) -> bool {
        self.inner.native.as_ref().is_some_and(|x| !x.wy_bwd_chain.is_null() && !x.skip.contains(&OP_WY_BWD_CHAIN))
            && dk > 0
            && dk <= 128
            && dk % 32 == 0
            && dv > 0
            && dv % 32 == 0
    }
}

/// The parameter blocks of one commit, uploaded together; with the scratch
/// of the two-launch CUDA-only forms (the 256 lane sums of the RMS dW).
struct ParArena {
    ptr: u64,
    words: usize,
    scratch: u64,
    scratch_words: usize,
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
    /// cuBLAS GEMMs in the bf16 tensor-core mode (`CMF_CUDA_GEMM_BF16=1`: f32
    /// in and out, operands rounded to bf16 inside the multiply, f32 sums;
    /// twice TF32's rate on an A40). Off by default; an [`ExactF32`] guard
    /// still runs full f32.
    bf16: bool,
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
            rms_dw_part: get("emb_x_rms_dw_part")?,
            rms_dw_tree: get("emb_x_rms_dw_tree")?,
            wy_du: get("emb_x_wy_du")?,
            wy_dgend: get("emb_x_wy_dgend")?,
            wy_dlg_dot: get("emb_x_wy_dlg_dot")?,
            wy_dscal: get("emb_x_wy_dscal")?,
            conv_bwd: get("emb_x_conv_bwd")?,
            conv_dw: {
                let f = get("emb_x_conv_dw")?;
                let a = cs::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
                if unsafe { cu::function::set_function_attribute(f, a, CONV_DW_SMEM as i32) }.is_ok() { f } else { std::ptr::null_mut() }
            },
            moe_stats: get("emb_x_moe_stats")?,
            moe_update: get("emb_x_moe_update")?,
            swiglu_rows: get("emb_x_swiglu_rows")?,
            swiglu_bwd_rows: get("emb_x_swiglu_bwd_rows")?,
            conv_fwd: get("emb_x_conv_fwd")?,
            conv_dx: get("emb_x_conv_dx")?,
            wy_fwd_chain: {
                let f = get("emb_x_wy_fwd_chain")?;
                let a = cs::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
                if unsafe { cu::function::set_function_attribute(f, a, wy_chain_smem(128) as i32) }.is_ok() { f } else { std::ptr::null_mut() }
            },
            wy_bwd_chain: {
                let f = get("emb_x_wy_bwd_chain")?;
                let a = cs::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
                if unsafe { cu::function::set_function_attribute(f, a, wy_bwd_chain_smem(128) as i32) }.is_ok() { f } else { std::ptr::null_mut() }
            },
            wy_du_dots: get("emb_x_wy_du_dots")?,
            wy_dgend_all: get("emb_x_wy_dgend_all")?,
            wy_dgend_fold_all: get("emb_x_wy_dgend_fold_all")?,
            route_fin: get("emb_x_route_fin")?,
            moe_gather: get("emb_x_moe_gather")?,
            moe_scatter: get("emb_x_moe_scatter")?,
            gemm_table: get("emb_x_gemm_table")?,
            softmax_bwd: get("emb_x_softmax_bwd")?,
            wy_scal: get("emb_x_wy_scal")?,
            wy_ut: get("emb_x_wy_ut")?,
            wy_dlg: get("emb_x_wy_dlg")?,
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
            bf16: tf32 && std::env::var("CMF_CUDA_GEMM_BF16").as_deref() == Ok("1"),
            exact_depth: Mutex::new(0),
            dummy,
            adapter_name: name,
            splitk_pool: Mutex::new(Vec::new()),
            commit: Mutex::new(ParArena { ptr: 0, words: 0, scratch: 0, scratch_words: 0 }),
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
        // once and reused until another operation touches its buffer. One
        // read takes up to 256 words from the count on, so the neighbouring
        // experts' counts come with it (one wait for the card per layer and
        // pass instead of one per expert).
        let dyn_bufs: std::collections::HashSet<u64> = ops
            .iter()
            .filter(|o| o.entry == ENTRY_CUBLAS && (o.params[28] == 1 || o.params[30] == 1))
            .map(|o| o.slots[3].0.ptr)
            .collect();
        let mut dyn_seen: HashMap<u64, (u32, Vec<u32>)> = HashMap::new();
        // host time spent waiting for the card before a dynamic GEMM's count
        let (mut syncs, mut sync_s) = (0usize, 0.0f64);
        for (i, op) in ops.iter().enumerate() {
            if op.entry != ENTRY_CUBLAS && !dyn_bufs.is_empty() {
                for (si, s) in op.slots.iter().enumerate() {
                    if dyn_bufs.contains(&s.0.ptr) && !reads_only(op, si) {
                        dyn_seen.remove(&s.0.ptr);
                    }
                }
            }
            if op.entry == ENTRY_CUBLAS {
                let dynv = (op.params[28] == 1 || op.params[30] == 1).then(|| {
                    let (ptr, off) = (op.slots[3].0.ptr, op.params[29]);
                    let hit = dyn_seen.get(&ptr).and_then(|(start, w)| {
                        let at = off.checked_sub(*start)? as usize;
                        (at + 2 <= w.len()).then(|| [w[at], w[at + 1]])
                    });
                    hit.unwrap_or_else(|| {
                        // the allocation runs 256 bytes past the buffer: two words always read
                        let len = op.slots[3].0.len.saturating_sub(off as usize).clamp(2, 256);
                        let mut w = vec![0u32; len];
                        let t0 = std::time::Instant::now();
                        unsafe {
                            cu::stream::synchronize(inner.stream).expect("CUDA sync before a dynamic GEMM");
                            cu::memcpy_dtoh_sync(&mut w[..], ptr + 4 * off as u64).expect("CUDA count readback");
                        }
                        syncs += 1;
                        sync_s += t0.elapsed().as_secs_f64();
                        let first = [w[0], w[1]];
                        dyn_seen.insert(ptr, (off, w));
                        first
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
            // A large copy is the bits of the driver's device copy; axpby(0, x,
            // 0, x) is a zero fill (0·x + 0·x differs from it only in the sign
            // of a zero and in a NaN or infinity). Both leave the kernel path
            // for the driver's (`CMF_CUDA_NATIVE_SKIP=12,2` keeps the kernels).
            if op.entry == 0 {
                let p = &op.params;
                let (src, dst) = (op.slots[0].0.ptr, op.slots[1].0.ptr);
                let fast = inner.native.as_ref().is_some_and(|x| !x.skip.contains(&p[0]));
                let done = fast
                    && match p[0] {
                        OP_COPY if p[1] >= (1 << 20) && (src != dst || p[2].abs_diff(p[3]) >= p[1]) => {
                            unsafe { cu::memcpy_dtod_async(dst + 4 * p[3] as u64, src + 4 * p[2] as u64, 4 * p[1] as usize, inner.stream) }
                                .expect("CUDA device copy");
                            true
                        }
                        OP_AXPBY if p[2] == 0 && p[3] == 0 && src == dst && p[1] > 0 => {
                            unsafe { cu::memset_d8_async(dst, 0, 4 * p[1] as usize, inner.stream) }.expect("CUDA zero fill");
                            true
                        }
                        _ => false,
                    };
                if done {
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
            }
            let entry = (op.entry as usize).min(ENTRIES.len() - 1);
            let (mut f, mut block) = (inner.funcs[entry], ENTRIES[entry].1);
            let (mut groups, mut shared) = (op.groups, 0u32);
            // a second launch of a two-launch CUDA-only form, and whether the
            // scratch goes into slot m
            let mut then: Option<(cs::CUfunction, u32, u32, u32)> = None;
            let mut scratch_words = 0usize;
            let p = &op.params;
            if entry == 0 {
                if let Some(s) = inner.main_spec.get(p[0] as usize).filter(|s| !s.is_null()) {
                    f = *s;
                }
                // (kernel, block, blocks, dynamic shared bytes) of a CUDA-only form
                let nat = inner.native.as_ref().filter(|x| !x.skip.contains(&p[0])).and_then(|x| match p[0] {
                    OP_ROUTE_GROUP if p[2] <= 16 => Some((x.route_group, 256, 1, 0)),
                    OP_RMS_FWD if (1..=1535).contains(&p[2]) => Some((x.rms_fwd, 256, p[1].div_ceil(8), 32 * (p[2] + 1))),
                    OP_SUMSQ => Some((x.sumsq, 256, p[3].div_ceil(8), 0)),
                    OP_WY_DU if (1..=384).contains(&p[5]) => Some((x.wy_du, 256, (p[1] * p[3] * 64).div_ceil(8), 128 * p[5])),
                    OP_WY_DGEND => Some((x.wy_dgend, 256, (p[1] * p[3] * 64).div_ceil(8), 0)),
                    OP_WY_DLG_DOT if (1..=128).contains(&p[4]) => Some((x.wy_dlg_dot, 256, (p[1] * p[2] * p[3]).div_ceil(8), 0)),
                    OP_WY_DSCAL if (1..=4096).contains(&p[2]) => Some((x.wy_dscal, 256, p[1] * p[3], 12 * p[2])),
                    OP_CONV_BWD if (1..=4).contains(&p[4]) && !x.conv_dw.is_null() => {
                        then = Some((x.conv_dw, 256, p[3].div_ceil(32), CONV_DW_SMEM));
                        Some((x.conv_dx, 256, (p[1] * p[2].div_ceil(8) * p[3]).div_ceil(256), 0))
                    }
                    OP_CONV_BWD if (1..=8).contains(&p[4]) => {
                        Some((x.conv_bwd, 256, (p[1] * p[2] * p[3]).div_ceil(256) + p[3].div_ceil(32), 0))
                    }
                    OP_MOE_STATS => Some((x.moe_stats, 32, (p[1] * p[2]).div_ceil(32), 0)),
                    // the scalar tiled GEMM with a table (no dynamic K or rows, no band)
                    OP_GEMM if p[31] == 1 && p[35] == 1 && p[28] != 1 && p[30] != 1 => {
                        let tiles = p[2].div_ceil(64) * p[3].div_ceil(64);
                        Some((x.gemm_table, 256, p[15] * p[16] * p[17] * tiles, 0))
                    }
                    OP_MOE_GATHER if p[2] % 4 == 0 => Some((x.moe_gather, 64, p[1], 0)),
                    OP_MOE_SCATTER if p[2] % 4 == 0 => Some((x.moe_scatter, 64, p[1], 0)),
                    OP_WY_SCAL => Some((x.wy_scal, 256, (p[1] * p[3] * p[8]).div_ceil(8), 0)),
                    OP_WY_UT => Some((x.wy_ut, 64, p[1] * p[3] * p[8], 0)),
                    OP_WY_DLG => Some((x.wy_dlg, 256, (p[1] * p[3] * p[8]).div_ceil(8), 0)),
                    OP_ROUTE_FIN if (1..=1536).contains(&p[2]) => Some((x.route_fin, 256, p[1].div_ceil(8), 32 * p[2])),
                    OP_WY_FWD_CHAIN => Some((x.wy_fwd_chain, 256, p[1] * p[3] * (p[5] / 32), wy_chain_smem(p[4]))),
                    OP_WY_BWD_CHAIN => Some((x.wy_bwd_chain, 256, p[1] * p[3] * (p[5] / 32), wy_bwd_chain_smem(p[4]))),
                    OP_WY_DU_DOTS => Some((x.wy_du_dots, 256, (p[1] * p[3] * p[8] * 64).div_ceil(8), 0)),
                    OP_WY_DGEND_ALL => {
                        then = Some((x.wy_dgend_fold_all, 256, (p[1] * p[3] * p[8]).div_ceil(256), 0));
                        Some((x.wy_dgend_all, 256, (p[1] * p[3] * p[8] * 64).div_ceil(8), 0))
                    }
                    // p[1] / (p[3]·p[4]) experts, SWIGLU_ROWS_BLOCKS blocks each
                    OP_SWIGLU_FWD if p[2] == 1 && p[3] * p[4] > 0 => Some((x.swiglu_rows, 256, (p[1] / (p[3] * p[4])) * 512, 0)),
                    OP_SWIGLU_BWD if p[2] == 1 && p[3] * p[4] > 0 => Some((x.swiglu_bwd_rows, 256, (p[1] / (p[3] * p[4])) * 512, 0)),
                    OP_CONV_FWD if (1..=8).contains(&p[4]) => Some((x.conv_fwd, 256, (p[1] * p[2].div_ceil(8) * p[3]).div_ceil(256), 0)),
                    OP_MOE_UPDATE => Some((x.moe_update, 256, (p[1] * p[2]).div_ceil(256) + p[1], 0)),
                    _ => None,
                });
                if let Some(n) = nat {
                    (f, block, groups, shared) = n;
                }
            } else if entry == 10 && p[5] <= 256 * 14 {
                if let Some(x) = inner.native.as_ref().filter(|x| !x.skip.contains(&(p[0] + 10000))) {
                    f = x.softmax_bwd;
                }
            } else if entry == 6 && p[2] > 0 {
                if let Some(x) = inner.native.as_ref().filter(|x| !x.skip.contains(&p[0])) {
                    let ncg = p[2].div_ceil(32);
                    (f, groups) = (x.rms_dw_part, ncg * 32);
                    then = Some((x.rms_dw_tree, 256, ncg, 0));
                    scratch_words = 256 * p[2] as usize;
                }
            }
            if scratch_words > arena.scratch_words {
                if arena.scratch != 0 {
                    unsafe {
                        // the stream is ordered: a previous user of the scratch has finished
                        cu::stream::synchronize(inner.stream).expect("CUDA sync before a scratch resize");
                        let _ = cu::free_sync(arena.scratch);
                    }
                }
                arena.scratch = unsafe { cu::malloc_sync(scratch_words * 4) }.expect("CUDA native scratch");
                arena.scratch_words = scratch_words;
            }
            let mut vals = [0u64; SLOTS + 1];
            for (v, s) in vals.iter_mut().zip(&op.slots) {
                *v = s.0.ptr;
            }
            if scratch_words > 0 {
                vals[12] = arena.scratch;
            }
            vals[SLOTS] = arena.ptr + (i * PARAM_WORDS * 4) as u64;
            let mut args: Vec<*mut c_void> = vals.iter_mut().map(|v| v as *mut u64 as *mut c_void).collect();
            for (f, block, groups, shared) in std::iter::once((f, block, groups, shared)).chain(then) {
                let gx = groups.clamp(1, 65_535);
                let gy = groups.div_ceil(gx).max(1);
                unsafe { cu::launch_kernel(f, (gx, gy, 1), (block, 1, 1), shared, inner.stream, &mut args) }
                    .unwrap_or_else(|e| panic!("CUDA launch of op {} (entry {entry}): {e:?}", p[0]));
            }
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
            eprintln!(
                "cuda encode ops={} ms={:.3} dyn_syncs={syncs} sync_wait_ms={:.3}",
                ops.len(),
                encode_start.elapsed().as_secs_f64() * 1000.0,
                sync_s * 1000.0
            );
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

/// Slot `si` of `op` is only read by it: a count read there leaves the
/// counts a dynamic GEMM read back valid (the expert row bound of the row
/// SwiGLU, the counts of the MoE statistics and update).
fn reads_only(op: &OpRec, si: usize) -> bool {
    let p = &op.params;
    op.entry == 0
        && match p[0] {
            OP_SWIGLU_FWD => p[2] == 1 && si == 3,
            OP_SWIGLU_BWD => p[2] == 1 && si == 5,
            OP_MOE_STATS => si == 1,
            OP_MOE_UPDATE => si == 3,
            _ => false,
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
    let bf16 = inner.bf16 && *inner.exact_depth.lock().unwrap() == 0;
    let (f32t, ct, algo) = (
        bl::cudaDataType::CUDA_R_32F,
        bl::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF,
        bl::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
    );
    let (a0, b0, c0) = (base(0, p[10]), base(1, p[11]), base(2, p[12]));
    for cb in 0..nb {
        for ch in 0..nh {
            let a = a0 + 4 * (cb * p[18] as u64 + ch * p[19] as u64);
            let b = b0 + 4 * (cb * p[21] as u64 + ch * p[22] as u64);
            let c = c0 + 4 * (cb * p[24] as u64 + ch * p[25] as u64);
            let s = unsafe {
                if bf16 && nc == 1 {
                    bl::cublasGemmEx(
                        h, opr(tb), opr(ta), n, m, k, (&al as *const f32).cast(), b as *const c_void, f32t, ldb,
                        a as *const c_void, f32t, lda, (&be as *const f32).cast(), c as *mut c_void, f32t, ldc, ct, algo,
                    )
                } else if bf16 {
                    bl::cublasGemmStridedBatchedEx(
                        h, opr(tb), opr(ta), n, m, k, (&al as *const f32).cast(), b as *const c_void, f32t, ldb,
                        p[23] as i64, a as *const c_void, f32t, lda, p[20] as i64, (&be as *const f32).cast(),
                        c as *mut c_void, f32t, ldc, p[26] as i64, nc as i32, ct, algo,
                    )
                } else if nc == 1 {
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
