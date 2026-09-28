struct Params { n:u32, h:u32, heads:u32, dh:u32, x:u32, y:u32, z:u32, w:u32, b:u32, k:u32, m:u32, mode:u32, eps:u32, pad0:u32, pad1:u32, pad2:u32 }
@group(0) @binding(0) var<storage, read> weights:array<f32>;
@group(0) @binding(1) var<storage, read_write> arena:array<f32>;
@group(0) @binding(2) var<storage, read> input:array<u32>;
@group(0) @binding(3) var<uniform> p:Params;
var<workgroup> partial:array<f32,128>;
var<workgroup> coefficient:f32;
var<workgroup> a:array<f32,128>;
// Transposed + padded: avoid a 16-way NVIDIA shared-memory bank conflict.
var<workgroup> b:array<f32,528>;
// Works for any hardware subgroup width dividing this 128-thread workgroup.
fn reduce(v:f32, lid:u32, lane:u32, size:u32)->f32 {
    let sum=subgroupAdd(v);
    if(lane==0u) { partial[lid/size]=sum; }
    workgroupBarrier();
    if(lid==0u) {
        var total=0.0;
        for(var i=0u;i<128u/size;i++) { total+=partial[i]; }
        coefficient=total;
    }
    workgroupBarrier();
    return coefficient;
}
fn gelu(x:f32)->f32 {
    let z=x/sqrt(2.0);
    let t=1.0/(1.0+0.5*abs(z));
    let e=t*exp(-z*z-1.26551223+t*(1.00002368+t*(0.37409196+t*(0.09678418+t*(-0.18628806+t*(0.27886807+t*(-1.13520398+t*(1.48851587+t*(-0.82215223+t*0.17087277)))))))));
    let erf_z=select(e-1.0,1.0-e,z>=0.0);
    return (x*(erf_z+1.0))*0.5;
}
@compute @workgroup_size(128)
fn embedding(@builtin(global_invocation_id) gid:vec3<u32>) {
    let i=gid.x;
    if(i<p.n*p.h) { arena[p.y+i]=(weights[p.w+input[i/p.h]*p.h+i%p.h]+weights[p.z+i%p.h])+weights[p.b+i]; }
}
// 8x32 output, 16-wide K tiles; each lane owns two FP32 accumulators.
@compute @workgroup_size(128)
fn linear(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_index) lid:u32) {
    let r0=group.y*8u;
    let m0=group.x*32u;
    let row=lid/32u;
    let col=lid%32u;
    var c0=0.0; var c1=0.0;
    for(var k0=0u;k0<p.k;k0+=16u) {
        let ar=r0+lid/16u;
        let ac=k0+lid%16u;
        var av=0.0;
        if(ar<p.n && ac<p.k) { av=arena[p.x+ar*p.k+ac]; }
        a[lid]=av;
        for(var i=lid;i<512u;i+=128u) {
            let wr=m0+i/16u;
            let wc=k0+i%16u;
            var bv=0.0;
            if(wr<p.m && wc<p.k) { bv=weights[p.w+wr*p.k+wc]; }
            b[(i%16u)*33u+i/16u]=bv;
        }
        workgroupBarrier();
        for(var k=0u;k<16u;k++) {
            let bv=b[k*33u+col];
            c0+=a[row*16u+k]*bv;
            c1+=a[(row+4u)*16u+k]*bv;
        }
        workgroupBarrier();
    }
    let m=m0+col;
    if(m<p.m) {
        if(r0+row<p.n) {
            let v=c0+weights[p.b+m];
            if(p.mode!=0u) { arena[p.y+(r0+row)*p.m+m]=gelu(v); }
            else { arena[p.y+(r0+row)*p.m+m]=v; }
        }
        if(r0+row+4u<p.n) {
            let v=c1+weights[p.b+m];
            if(p.mode!=0u) { arena[p.y+(r0+row+4u)*p.m+m]=gelu(v); }
            else { arena[p.y+(r0+row+4u)*p.m+m]=v; }
        }
    }
}
@compute @workgroup_size(128)
fn norm(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_index) lid:u32,
    @builtin(subgroup_invocation_id) lane:u32, @builtin(subgroup_size) size:u32) {
    let row=group.x;
    var sum=0.0;
    for(var j=lid;j<p.h;j+=128u) {
        var v=arena[p.x+row*p.h+j];
        if(p.mode!=0u) { v+=arena[p.z+row*p.h+j]; }
        sum+=v;
    }
    let mean=reduce(sum,lid,lane,size)/f32(p.h);
    var varsum=0.0;
    for(var j=lid;j<p.h;j+=128u) {
        var v=arena[p.x+row*p.h+j];
        if(p.mode!=0u) { v+=arena[p.z+row*p.h+j]; }
        let diff=v-mean;
        varsum+=diff*diff;
    }
    let inv=1.0/sqrt(reduce(varsum,lid,lane,size)/f32(p.h)+bitcast<f32>(p.eps));
    for(var j=lid;j<p.h;j+=128u) {
        var v=arena[p.x+row*p.h+j];
        if(p.mode!=0u) { v+=arena[p.z+row*p.h+j]; }
        arena[p.y+row*p.h+j]=((v-mean)*inv)*weights[p.w+j]+weights[p.b+j];
    }
}
@compute @workgroup_size(128)
fn scores(@builtin(global_invocation_id) gid:vec3<u32>) {
    let i=gid.x;
    if(i>=p.heads*p.n*p.n) { return; }
    let key=i%p.n; let query=(i/p.n)%p.n; let head=i/(p.n*p.n);
    var sum=0.0;
    for(var j=0u;j<p.dh;j++) { sum+=arena[p.x+query*3u*p.h+head*p.dh+j]*arena[p.x+key*3u*p.h+p.h+head*p.dh+j]; }
    arena[p.y+i]=sum/sqrt(f32(p.dh));
}
@compute @workgroup_size(128)
fn softmax(@builtin(global_invocation_id) gid:vec3<u32>) {
    let row=gid.x;
    if(row>=p.heads*p.n) { return; }
    let start=p.x+row*p.n;
    var mx=-3.402823466e+38;
    for(var j=0u;j<p.n;j++) { mx=max(mx,arena[start+j]); }
    var sum=0.0;
    for(var j=0u;j<p.n;j++) { let v=exp(arena[start+j]-mx); arena[start+j]=v; sum+=v; }
    let inv=1.0/sum;
    for(var j=0u;j<p.n;j++) { arena[start+j]*=inv; }
}
@compute @workgroup_size(128)
fn context(@builtin(global_invocation_id) gid:vec3<u32>) {
    let i=gid.x;
    if(i>=p.n*p.h) { return; }
    let query=i/p.h; let feature=i%p.h; let head=feature/p.dh;
    var sum=0.0;
    for(var key=0u;key<p.n;key++) { sum+=arena[p.z+(head*p.n+query)*p.n+key]*arena[p.x+key*3u*p.h+2u*p.h+feature]; }
    arena[p.y+i]=sum;
}
@compute @workgroup_size(128)
fn pool(@builtin(global_invocation_id) gid:vec3<u32>) {
    let col=gid.x;
    if(col<p.h) {
        var v=0.0;
        for(var row=0u;row<p.n;row++) { v+=arena[p.x+row*p.h+col]; }
        arena[p.y+col]=v/f32(p.n);
    } else if(col<p.h+4096u) { arena[p.y+col]=0.5*bitcast<f32>(input[p.z+col-p.h]); }
}
@compute @workgroup_size(128)
fn normalize(@builtin(local_invocation_index) lid:u32,
    @builtin(subgroup_invocation_id) lane:u32, @builtin(subgroup_size) size:u32) {
    var sum=0.0;
    for(var j=lid;j<p.h;j+=128u) { let v=arena[p.x+j]; sum+=v*v; }
    let den=sqrt(reduce(sum,lid,lane,size))+1e-12;
    sum=0.0;
    for(var j=lid;j<p.h;j+=128u) { let v=arena[p.x+j]/den; arena[p.x+j]=v; sum+=v*v; }
    let n=sqrt(reduce(sum,lid,lane,size));
    if(n>1e-12) {
        let inv=1.0/n;
        for(var j=lid;j<p.h;j+=128u) { arena[p.x+j]*=inv; }
    }
}

// Small text sequences do not fill enough tiled-GEMM workgroups. Distribute
// the row dot products across subgroups instead; adjacent tokens reuse L2.
@compute @workgroup_size(128)
fn linear_short(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_index) lid:u32,
    @builtin(subgroup_invocation_id) lane:u32, @builtin(subgroup_size) size:u32) {
    let row=group.y;
    if(size==32u) {
        let m=group.x*4u+lid/32u;
        var sum=0.0;
        if(m<p.m) {
            for(var k=lane;k<p.k;k+=32u) { sum+=arena[p.x+row*p.k+k]*weights[p.w+m*p.k+k]; }
        }
        let dot=subgroupAdd(sum);
        if(lane==0u && m<p.m) {
            let v=dot+weights[p.b+m];
            if(p.mode!=0u) { arena[p.y+row*p.m+m]=gelu(v); }
            else { arena[p.y+row*p.m+m]=v; }
        }
    } else {
        // Correct fallback if a device selects a different subgroup width.
        // The host only selects this fast path for NVIDIA's 32-lane warps.
        let m=group.x*4u+lid;
        if(lid<4u && m<p.m) {
            var dot=0.0;
            for(var k=0u;k<p.k;k++) { dot+=arena[p.x+row*p.k+k]*weights[p.w+m*p.k+k]; }
            let v=dot+weights[p.b+m];
            if(p.mode!=0u) { arena[p.y+row*p.m+m]=gelu(v); }
            else { arena[p.y+row*p.m+m]=v; }
        }
    }
}

// Four query/head rows per workgroup; online softmax stays in registers.
// FP32, no score matrix or intermediate dispatch. Selected only on NVIDIA
// with head_dim <= 32; scalar score fallback preserves other subgroup sizes.
@compute @workgroup_size(128)
fn attention_fused(@builtin(workgroup_id) group:vec3<u32>, @builtin(local_invocation_index) lid:u32,
    @builtin(subgroup_invocation_id) lane:u32, @builtin(subgroup_size) size:u32) {
    let row=group.x*4u+lid/32u;
    let col=lid%32u;
    let valid_row=row<p.heads*p.n;
    let head=row/p.n; let query=row%p.n;
    let feature=head*p.dh+col;
    var q=0.0;
    if(valid_row && col<p.dh) { q=arena[p.x+query*3u*p.h+feature]; }
    var maximum=-3.402823466e+38; var denominator=0.0; var value=0.0;
    for(var key=0u;key<p.n;key++) {
        var k=0.0; var v=0.0;
        if(valid_row && col<p.dh) {
            k=arena[p.x+key*3u*p.h+p.h+feature];
            v=arena[p.x+key*3u*p.h+2u*p.h+feature];
        }
        var dot=0.0;
        if(size==32u) { dot=subgroupAdd(q*k); }
        else if(valid_row) {
            for(var j=0u;j<p.dh;j++) { dot+=arena[p.x+query*3u*p.h+head*p.dh+j]*arena[p.x+key*3u*p.h+p.h+head*p.dh+j]; }
        }
        let score=dot/sqrt(f32(p.dh));
        let next=max(maximum,score);
        let old_scale=exp(maximum-next); let weight=exp(score-next);
        value=value*old_scale+v*weight;
        denominator=denominator*old_scale+weight;
        maximum=next;
    }
    if(valid_row && col<p.dh) { arena[p.y+query*p.h+feature]=value/denominator; }
}
