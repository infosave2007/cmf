// Row kernels of the wide token path for q8_2f matrices: w[o, i] = q[o, i] · row[o] ·
// col[i], 16 bytes hold sixteen codes. A matrix word in the layer record is an index
// in the meta: (byte offset of the codes, offset of the row scales, of the column
// field), both in the small tensors. The stage that makes a matrix's input multiplies
// in the column field once, so a row reads only its codes and one vector, and the
// codes unpack as integers: on an M4 unpack4x8snorm costs 2.5 times the whole kernel.
const S_FOLD: u32 = 40960u;

fn x_in(i: u32) -> u32 { return S_FOLD + i * 1024u; }
fn fold4(e: u32, cell: u32, v: vec4<f32>) -> vec4<f32> { return v * small4[(md[e + 2u] >> 2u) + cell]; }
fn fold1(e: u32, i: u32, v: f32) -> f32 { return v * sm(md[e + 2u] + i); }
fn fold_lane(i: u32, e: u32, lid: u32, v: vec4<f32>) { s4[(x_in(i) >> 2u) + lid] = fold4(e, lid, v); }

fn u4(v: u32) -> vec4<f32> { return vec4<f32>(unpack4xI8(v)); }

fn q16(w: vec4<u32>, xb: u32) -> vec4<f32> {
    return u4(w.x) * s4[xb] + u4(w.y) * s4[xb + 1u] + u4(w.z) * s4[xb + 2u] + u4(w.w) * s4[xb + 3u];
}

// One chunk of a row.
fn mv_part(e: u32, row: u32, ch: u32, cols: u32, x: u32) -> f32 {
    let first = ch * CHUNK;
    let n = min(CHUNK, cols - first) >> 4u;
    let o = (md[e] + row * cols + first) >> 4u;
    let xb = (x + first) >> 2u;
    var a = vec4<f32>(0.0);
    for (var j = 0u; j < n; j += 1u) {
        a += q16(mats[o + j], xb + 4u * j);
    }
    return hsum(a) * sm(md[e + 1u] + row);
}

// One row times the `cols` values of the scratch image at `x`.
fn mv(e: u32, row: u32, cols: u32, x: u32) -> f32 {
    let n = cols >> 4u;
    let o = (md[e] + row * cols) >> 4u;
    let xb = x >> 2u;
    var a = vec4<f32>(0.0);
    for (var j = 0u; j < n; j += 1u) {
        a += q16(mats[o + j], xb + 4u * j);
    }
    return hsum(a) * sm(md[e + 1u] + row);
}

// Two rows, each over its own vector.
fn mv2(ea: u32, eb: u32, row: u32, cols: u32, xa: u32, xb: u32) -> vec2<f32> {
    return mv2r(ea, row, eb, row, cols, xa, xb);
}

fn mv2r(ea: u32, ra: u32, eb: u32, rb_: u32, cols: u32, xa: u32, xb: u32) -> vec2<f32> {
    let n = cols >> 4u;
    let oa = (md[ea] + ra * cols) >> 4u;
    let ob = (md[eb] + rb_ * cols) >> 4u;
    let va = xa >> 2u;
    let vb = xb >> 2u;
    var a = vec4<f32>(0.0);
    var b = vec4<f32>(0.0);
    for (var j = 0u; j < n; j += 1u) {
        a += q16(mats[oa + j], va + 4u * j);
        b += q16(mats[ob + j], vb + 4u * j);
    }
    return vec2<f32>(hsum(a) * sm(md[ea + 1u] + ra), hsum(b) * sm(md[eb + 1u] + rb_));
}

// One vocabulary row times the normed token.
fn mv_head(row: u32) -> f32 {
    let e = md[30u];
    let hidden = md[0u];
    let n = hidden >> 4u;
    let o = (row * hidden) >> 4u;
    let xb = x_in(0u) >> 2u;
    var a = vec4<f32>(0.0);
    for (var j = 0u; j < n; j += 1u) {
        a += q16(head_mat[o + j], xb + 4u * j);
    }
    return hsum(a) * sm(md[e + 1u] + row);
}
