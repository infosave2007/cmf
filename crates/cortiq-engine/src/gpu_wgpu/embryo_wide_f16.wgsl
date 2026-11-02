// Row kernels of the wide token path for f16 matrices: 16 bytes hold eight values.
// A matrix word in the layer record is its offset in values, a multiple of 8.
fn d8(q: vec4<u32>, v0: vec4<f32>, v1: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(q.x), unpack2x16float(q.y)) * v0
         + vec4<f32>(unpack2x16float(q.z), unpack2x16float(q.w)) * v1;
}

// A matrix reads the vector as its stage left it.
fn x_in(_i: u32) -> u32 { return S_NORM; }
fn fold4(_e: u32, _cell: u32, v: vec4<f32>) -> vec4<f32> { return v; }
fn fold1(_e: u32, _i: u32, v: f32) -> f32 { return v; }
fn fold_lane(_i: u32, _e: u32, _lid: u32, _v: vec4<f32>) {}

// One chunk of a row.
fn mv_part(e: u32, row: u32, ch: u32, cols: u32, x: u32) -> f32 {
    let first = ch * CHUNK;
    let n = min(CHUNK, cols - first) >> 3u;
    let o = (e + row * cols + first) >> 3u;
    let xb = (x + first) >> 2u;
    var a = vec4<f32>(0.0);
    for (var j = 0u; j < n; j += 1u) {
        a += d8(mats[o + j], s4[xb + 2u * j], s4[xb + 2u * j + 1u]);
    }
    return hsum(a);
}

// One row times the `cols` values of the scratch image at `x`.
fn mv(e: u32, row: u32, cols: u32, x: u32) -> f32 {
    let n = cols >> 3u;
    let o = (e + row * cols) >> 3u;
    let xb = x >> 2u;
    var a = vec4<f32>(0.0);
    for (var j = 0u; j < n; j += 1u) {
        a += d8(mats[o + j], s4[xb + 2u * j], s4[xb + 2u * j + 1u]);
    }
    return hsum(a);
}

// Two rows over the same vector: one pass over its values.
fn mv2(ea: u32, eb: u32, row: u32, cols: u32, x: u32, xb: u32) -> vec2<f32> {
    return mv2r(ea, row, eb, row, cols, x, xb);
}

fn mv2r(ea: u32, ra: u32, eb: u32, rb_: u32, cols: u32, x: u32, _xb: u32) -> vec2<f32> {
    let n = cols >> 3u;
    let oa = (ea + ra * cols) >> 3u;
    let ob = (eb + rb_ * cols) >> 3u;
    let xb = x >> 2u;
    var a = vec4<f32>(0.0);
    var b = vec4<f32>(0.0);
    for (var j = 0u; j < n; j += 1u) {
        let v0 = s4[xb + 2u * j];
        let v1 = s4[xb + 2u * j + 1u];
        a += d8(mats[oa + j], v0, v1);
        b += d8(mats[ob + j], v0, v1);
    }
    return vec2<f32>(hsum(a), hsum(b));
}

// One vocabulary row times the normed token.
fn mv_head(row: u32) -> f32 {
    let n = md[0u] >> 3u;
    let o = row * n;
    var a = vec4<f32>(0.0);
    for (var j = 0u; j < n; j += 1u) {
        a += d8(head_mat[o + j], s4[2u * j], s4[2u * j + 1u]);
    }
    return hsum(a);
}
