const DIM:u32=SIGNAL_DIM;
struct Params { offset:u32, pad0:u32, pad1:u32, pad2:u32 }
@group(0) @binding(0) var<storage, read> input:array<f32>;
@group(0) @binding(1) var<storage, read> weights:array<f32>;
// Per topology: mean offset, basis offset, rank, padding.
@group(0) @binding(2) var<storage, read> tasks:array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> output:array<f32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> partial:array<f32,256>;
var<workgroup> coefficient:f32;
fn reduce(v:f32,lid:u32,lane:u32,size:u32)->f32 {
    let sum=subgroupAdd(v);
    if(lane==0u) { partial[lid/size]=sum; }
    workgroupBarrier();
    if(lid==0u) {
        var total=0.0;
        for(var i=0u;i<256u/size;i++) { total+=partial[i]; }
        coefficient=total;
    }
    workgroupBarrier();
    return coefficient;
}
@compute @workgroup_size(256)
fn reconstruction(@builtin(workgroup_id) wid:vec3<u32>, @builtin(local_invocation_index) lid:u32,
    @builtin(subgroup_invocation_id) lane:u32, @builtin(subgroup_size) size:u32) {
    let task=tasks[wid.x];
    // REGISTER_INIT
    for(var k=0u;k<task.z;k++) {
        var sum=0.0;
        // REGISTER_DOT
        let c=reduce(sum,lid,lane,size);
        // REGISTER_UPDATE
    }
    var sum=0.0;
    // REGISTER_ERROR
    let error=reduce(sum,lid,lane,size);
    if(lid==0u) { output[wid.x]=error; }
}
