//! Native GGUF → `.cmf` importer. Parses a GGUF file, dequantizes every common
//! ggml tensor type (F32, F16, BF16, Q4_0/1, Q5_0/1, Q8_0, and the K-quants
//! Q2_K–Q6_K + Q8_K — all faithful ports of ggml `dequantize_row_*`), maps ggml
//! tensor names to HF names, reconstructs a Hugging Face tokenizer.json from the
//! embedded ggml metadata, and writes a `.cmf`. No Python. A GGUF repo id can be
//! passed directly (the matching `.gguf` is downloaded). IQ4_NL / IQ4_XS (the
//! non-linear 4-bit codebook, used inside q2_k/q3_k mixes) are handled; the
//! IQ1/IQ2/IQ3 grid-codebook types are the only ggml types not yet supported.

use crate::convert::{self, Quant};
use cortiq_core::format::{CMF_VERSION, CmfHeader, CmfModel, TensorSpec, TokenizerBundle};
use cortiq_core::quant::f16_to_f32;
use cortiq_core::types::{LayerType, ModelArch, NormStyle, QuantType, TensorDtype};
use std::collections::BTreeMap;
use std::fs;

// GGUF metadata value types.
const T_U8: u32 = 0;
const T_I8: u32 = 1;
const T_U16: u32 = 2;
const T_I16: u32 = 3;
const T_U32: u32 = 4;
const T_I32: u32 = 5;
const T_F32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STR: u32 = 8;
const T_ARR: u32 = 9;
const T_U64: u32 = 10;
const T_I64: u32 = 11;
const T_F64: u32 = 12;

// ggml tensor dtypes (ggml.h enum ids). All of these are dequantized natively.
const GGML_F32: u32 = 0;
const GGML_F16: u32 = 1;
const GGML_Q4_0: u32 = 2;
const GGML_Q4_1: u32 = 3;
const GGML_Q5_0: u32 = 6;
const GGML_Q5_1: u32 = 7;
const GGML_Q8_0: u32 = 8;
const GGML_Q2_K: u32 = 10;
const GGML_Q3_K: u32 = 11;
const GGML_Q4_K: u32 = 12;
const GGML_Q5_K: u32 = 13;
const GGML_Q6_K: u32 = 14;
const GGML_Q8_K: u32 = 15;
const GGML_IQ4_NL: u32 = 20;
const GGML_IQ4_XS: u32 = 23;
const GGML_BF16: u32 = 30;

/// Non-linear 4-bit codebook shared by IQ4_NL and IQ4_XS.
const KVALUES_IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

/// A parsed GGUF metadata value (only the parts we need are typed richly).
#[derive(Clone)]
enum Val {
    U64(u64),
    I64(i64),
    F64(f64),
    Str(String),
    /// Array of strings (tokens / merges).
    StrArr(Vec<String>),
    /// Array of ints (token_type).
    IntArr(Vec<i64>),
    Other,
}

impl Val {
    fn as_u64(&self) -> Option<u64> {
        match self {
            Val::U64(v) => Some(*v),
            Val::I64(v) => Some(*v as u64),
            _ => None,
        }
    }
    fn as_f64(&self) -> Option<f64> {
        match self {
            Val::F64(v) => Some(*v),
            Val::U64(v) => Some(*v as f64),
            Val::I64(v) => Some(*v as f64),
            _ => None,
        }
    }
    fn as_str(&self) -> Option<&str> {
        match self {
            Val::Str(s) => Some(s),
            _ => None,
        }
    }
}

struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> anyhow::Result<&'a [u8]> {
        if self.p + n > self.b.len() {
            anyhow::bail!("gguf: truncated");
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u32(&mut self) -> anyhow::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> anyhow::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn gstr(&mut self) -> anyhow::Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn scalar(&mut self, t: u32) -> anyhow::Result<Val> {
        Ok(match t {
            T_U8 | T_BOOL => Val::U64(self.take(1)?[0] as u64),
            T_I8 => Val::I64(self.take(1)?[0] as i8 as i64),
            T_U16 => Val::U64(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as u64),
            T_I16 => Val::I64(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            T_U32 => Val::U64(self.u32()? as u64),
            T_I32 => Val::I64(i32::from_le_bytes(self.take(4)?.try_into().unwrap()) as i64),
            T_F32 => Val::F64(f32::from_le_bytes(self.take(4)?.try_into().unwrap()) as f64),
            T_U64 => Val::U64(self.u64()?),
            T_I64 => Val::I64(i64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            T_F64 => Val::F64(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            T_STR => Val::Str(self.gstr()?),
            other => anyhow::bail!("gguf: bad value type {other}"),
        })
    }
    fn value(&mut self, t: u32) -> anyhow::Result<Val> {
        if t != T_ARR {
            return self.scalar(t);
        }
        let et = self.u32()?;
        let n = self.u64()? as usize;
        if et == T_STR {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(self.gstr()?);
            }
            Ok(Val::StrArr(v))
        } else if matches!(et, T_U8 | T_I8 | T_U16 | T_I16 | T_U32 | T_I32 | T_U64 | T_I64 | T_BOOL) {
            let mut v = Vec::with_capacity(n);
            for _ in 0..n {
                v.push(self.scalar(et)?.as_u64().map(|x| x as i64).unwrap_or(0));
            }
            Ok(Val::IntArr(v))
        } else {
            // arrays of floats etc. — consume and ignore
            for _ in 0..n {
                let _ = self.scalar(et)?;
            }
            Ok(Val::Other)
        }
    }
}

struct GgufTensor {
    name: String,
    dims: Vec<u64>, // ggml order (ne[0] fastest)
    ggml_type: u32,
    offset: u64, // relative to data section
}

struct Gguf {
    md: BTreeMap<String, Val>,
    tensors: Vec<GgufTensor>,
    bytes: Vec<u8>,
    data_start: usize,
}

fn align_up(x: usize, a: usize) -> usize {
    x.div_ceil(a) * a
}

fn parse(path: &std::path::Path) -> anyhow::Result<Gguf> {
    let bytes = fs::read(path)?;
    let mut c = Cursor { b: &bytes, p: 0 };
    if c.take(4)? != b"GGUF" {
        anyhow::bail!("not a GGUF file");
    }
    let _ver = c.u32()?;
    let n_tensors = c.u64()? as usize;
    let n_kv = c.u64()? as usize;
    let mut md = BTreeMap::new();
    for _ in 0..n_kv {
        let key = c.gstr()?;
        let t = c.u32()?;
        md.insert(key, c.value(t)?);
    }
    let mut tensors = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name = c.gstr()?;
        let nd = c.u32()? as usize;
        let mut dims = Vec::with_capacity(nd);
        for _ in 0..nd {
            dims.push(c.u64()?);
        }
        let ggml_type = c.u32()?;
        let offset = c.u64()?;
        tensors.push(GgufTensor { name, dims, ggml_type, offset });
    }
    let align = md.get("general.alignment").and_then(|v| v.as_u64()).unwrap_or(32) as usize;
    let data_start = align_up(c.p, align.max(1));
    Ok(Gguf { md, tensors, bytes, data_start })
}

/// Dequantize `n` elements of a ggml tensor into f32. Every codec below is a
/// faithful port of ggml's `dequantize_row_*` (ggml-quants.c); output order and
/// scale packing match byte-for-byte.
fn dequant(ggml_type: u32, raw: &[u8], n: usize) -> anyhow::Result<Vec<f32>> {
    Ok(match ggml_type {
        GGML_F32 => raw.chunks_exact(4).take(n).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        GGML_F16 => raw.chunks_exact(2).take(n).map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect(),
        GGML_BF16 => dequant_bf16(raw, n),
        GGML_Q4_0 => dequant_q4_0(raw, n),
        GGML_Q4_1 => dequant_q4_1(raw, n),
        GGML_Q5_0 => dequant_q5_0(raw, n),
        GGML_Q5_1 => dequant_q5_1(raw, n),
        GGML_Q8_0 => dequant_q8_0(raw, n),
        GGML_Q2_K => dequant_q2_k(raw, n),
        GGML_Q3_K => dequant_q3_k(raw, n),
        GGML_Q4_K => dequant_q4_k(raw, n),
        GGML_Q5_K => dequant_q5_k(raw, n),
        GGML_Q6_K => dequant_q6_k(raw, n),
        GGML_Q8_K => dequant_q8_k(raw, n),
        GGML_IQ4_NL => dequant_iq4_nl(raw, n),
        GGML_IQ4_XS => dequant_iq4_xs(raw, n),
        other => anyhow::bail!(
            "ggml tensor type {other} not supported by the native importer (supported: F32, F16, BF16, \
             Q4_0/1, Q5_0/1, Q8_0, Q2_K..Q6_K, Q8_K, IQ4_NL, IQ4_XS; the IQ1/IQ2/IQ3 grid codebooks are not)"
        ),
    })
}

/// On-disk byte length of `n` elements for a given ggml type.
fn nbytes(ggml_type: u32, n: usize) -> anyhow::Result<usize> {
    let blk = |elems: usize, bytes: usize| n.div_ceil(elems) * bytes;
    Ok(match ggml_type {
        GGML_F32 => n * 4,
        GGML_F16 | GGML_BF16 => n * 2,
        GGML_Q4_0 => blk(32, 18),
        GGML_Q4_1 => blk(32, 20),
        GGML_Q5_0 => blk(32, 22),
        GGML_Q5_1 => blk(32, 24),
        GGML_Q8_0 => blk(32, 34),
        GGML_Q2_K => blk(256, 84),
        GGML_Q3_K => blk(256, 110),
        GGML_Q4_K => blk(256, 144),
        GGML_Q5_K => blk(256, 176),
        GGML_Q6_K => blk(256, 210),
        GGML_Q8_K => blk(256, 292),
        GGML_IQ4_NL => blk(32, 18),
        GGML_IQ4_XS => blk(256, 136),
        other => anyhow::bail!("ggml type {other} unsupported by the native importer"),
    })
}

// BF16: contiguous 2-byte little-endian bfloat16 (top 16 bits of an f32).
fn dequant_bf16(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for (j, o) in out.iter_mut().enumerate() {
        let bits = u16::from_le_bytes([raw[j * 2], raw[j * 2 + 1]]) as u32;
        *o = f32::from_bits(bits << 16);
    }
    out
}

// block_q8_0 (34 bytes, 32 elems): [d: f16 LE][qs[32]: i8]
fn dequant_q8_0(raw: &[u8], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for blk in raw.chunks_exact(34) {
        let scale = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for &q in &blk[2..34] {
            out.push(q as i8 as f32 * scale);
        }
    }
    out.truncate(n);
    out
}

// block_q4_0 (18 bytes, 32 elems): [d: f16 LE][qs[16]: u8]; low nibble -> j, high -> j+16, minus 8.
fn dequant_q4_0(raw: &[u8], n: usize) -> Vec<f32> {
    const QK: usize = 32;
    const BB: usize = 18;
    let nb = n.div_ceil(QK);
    let mut out = vec![0.0f32; nb * QK];
    for i in 0..nb {
        let b = &raw[i * BB..i * BB + BB];
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let qs = &b[2..2 + QK / 2];
        for j in 0..QK / 2 {
            out[i * QK + j] = ((qs[j] & 0x0F) as i32 - 8) as f32 * d;
            out[i * QK + j + QK / 2] = ((qs[j] >> 4) as i32 - 8) as f32 * d;
        }
    }
    out.truncate(n);
    out
}

// block_q4_1 (20 bytes, 32 elems): [d: f16 LE][m: f16 LE][qs[16]: u8]; value = nibble*d + m (unsigned).
fn dequant_q4_1(raw: &[u8], n: usize) -> Vec<f32> {
    const QK: usize = 32;
    const BB: usize = 20;
    let nb = n.div_ceil(QK);
    let mut out = vec![0.0f32; nb * QK];
    for i in 0..nb {
        let b = &raw[i * BB..i * BB + BB];
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let m = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
        let qs = &b[4..4 + QK / 2];
        for j in 0..QK / 2 {
            out[i * QK + j] = (qs[j] & 0x0F) as f32 * d + m;
            out[i * QK + j + QK / 2] = (qs[j] >> 4) as f32 * d + m;
        }
    }
    out.truncate(n);
    out
}

// block_q5_0 (22 bytes, 32 elems): [d: f16 LE][qh[4]: u32 LE][qs[16]: u8]; 5th bit from qh, minus 16.
fn dequant_q5_0(raw: &[u8], n: usize) -> Vec<f32> {
    const QK: usize = 32;
    const BB: usize = 22;
    let nb = n.div_ceil(QK);
    let mut out = vec![0.0f32; nb * QK];
    for i in 0..nb {
        let b = &raw[i * BB..i * BB + BB];
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
        let qs = &b[6..6 + QK / 2];
        for j in 0..QK / 2 {
            let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
            let x0 = (((qs[j] & 0x0F) | xh_0) as i32) - 16;
            let x1 = (((qs[j] >> 4) | xh_1) as i32) - 16;
            out[i * QK + j] = x0 as f32 * d;
            out[i * QK + j + QK / 2] = x1 as f32 * d;
        }
    }
    out.truncate(n);
    out
}

// block_q5_1 (24 bytes, 32 elems): [d: f16 LE][m: f16 LE][qh[4]: u32 LE][qs[16]: u8]; value = (nibble|xh)*d + m.
fn dequant_q5_1(raw: &[u8], n: usize) -> Vec<f32> {
    const QK: usize = 32;
    const BB: usize = 24;
    let nb = n.div_ceil(QK);
    let mut out = vec![0.0f32; nb * QK];
    for i in 0..nb {
        let b = &raw[i * BB..i * BB + BB];
        let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
        let m = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
        let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
        let qs = &b[8..8 + QK / 2];
        for j in 0..QK / 2 {
            let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
            let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;
            let x0 = ((qs[j] & 0x0F) | xh_0) as f32;
            let x1 = ((qs[j] >> 4) | xh_1) as f32;
            out[i * QK + j] = x0 * d + m;
            out[i * QK + j + QK / 2] = x1 * d + m;
        }
    }
    out.truncate(n);
    out
}

// block_q2_K (84 bytes, 256 elems): scales[16], qs[64], d(f16), dmin(f16).
fn dequant_q2_k(raw: &[u8], n: usize) -> Vec<f32> {
    const QK_K: usize = 256;
    const BB: usize = 84;
    let nb = n.div_ceil(QK_K);
    let mut y: Vec<f32> = Vec::with_capacity(nb * QK_K);
    for i in 0..nb {
        let b = &raw[i * BB..i * BB + BB];
        let scales = &b[0..16];
        let qs = &b[16..80];
        let d = f16_to_f32(u16::from_le_bytes([b[80], b[81]]));
        let min = f16_to_f32(u16::from_le_bytes([b[82], b[83]]));
        let mut is = 0usize;
        let mut q_off = 0usize;
        let mut nn = 0usize;
        while nn < QK_K {
            let mut shift = 0u32;
            for _ in 0..4 {
                let sc = scales[is];
                is += 1;
                let (dl, ml) = (d * (sc & 0xF) as f32, min * (sc >> 4) as f32);
                for l in 0..16 {
                    let q = ((qs[q_off + l] >> shift) & 3) as f32;
                    y.push(dl * q - ml);
                }
                let sc = scales[is];
                is += 1;
                let (dl, ml) = (d * (sc & 0xF) as f32, min * (sc >> 4) as f32);
                for l in 0..16 {
                    let q = ((qs[q_off + 16 + l] >> shift) & 3) as f32;
                    y.push(dl * q - ml);
                }
                shift += 2;
            }
            q_off += 32;
            nn += 128;
        }
    }
    y.truncate(n);
    y
}

// block_q3_K (110 bytes, 256 elems): hmask[32], qs[64], scales[12], d(f16).
fn dequant_q3_k(raw: &[u8], n: usize) -> Vec<f32> {
    const QK_K: usize = 256;
    const BB: usize = 110;
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let nb = n.div_ceil(QK_K);
    let mut y: Vec<f32> = Vec::with_capacity(nb * QK_K);
    for i in 0..nb {
        let b = &raw[i * BB..i * BB + BB];
        let hm = &b[0..32];
        let qs = &b[32..96];
        let d_all = f16_to_f32(u16::from_le_bytes([b[108], b[109]]));

        // Unpack the 16 six-bit scales via ggml's aux[4] uint32 shuffle.
        let a0 = u32::from_le_bytes([b[96], b[97], b[98], b[99]]);
        let a1 = u32::from_le_bytes([b[100], b[101], b[102], b[103]]);
        let tmp = u32::from_le_bytes([b[104], b[105], b[106], b[107]]);
        let na0 = (a0 & KMASK2) | ((tmp & KMASK1) << 4);
        let na1 = (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        let na2 = ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        let na3 = ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        let mut sb = [0u8; 16];
        sb[0..4].copy_from_slice(&na0.to_le_bytes());
        sb[4..8].copy_from_slice(&na1.to_le_bytes());
        sb[8..12].copy_from_slice(&na2.to_le_bytes());
        sb[12..16].copy_from_slice(&na3.to_le_bytes());
        let scales: [i8; 16] = std::array::from_fn(|k| sb[k] as i8);

        let mut m: u8 = 1;
        let mut is = 0usize;
        let mut q_off = 0usize;
        let mut nn = 0usize;
        while nn < QK_K {
            let mut shift = 0u32;
            for _ in 0..4 {
                let dl = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let low = ((qs[q_off + l] >> shift) & 3) as i32;
                    let high = if (hm[l] & m) != 0 { 0 } else { 4 };
                    y.push(dl * (low - high) as f32);
                }
                let dl = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let low = ((qs[q_off + 16 + l] >> shift) & 3) as i32;
                    let high = if (hm[16 + l] & m) != 0 { 0 } else { 4 };
                    y.push(dl * (low - high) as f32);
                }
                shift += 2;
                m <<= 1;
            }
            q_off += 32;
            nn += 128;
        }
    }
    y.truncate(n);
    y
}

// 6-bit scale/min unpack — faithful port of ggml get_scale_min_k4 (shared by Q4_K/Q5_K).
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

// block_q4_K (144 bytes, 256 elems): d(f16), dmin(f16), scales[12], qs[128].
fn dequant_q4_k(raw: &[u8], n: usize) -> Vec<f32> {
    const QK_K: usize = 256;
    const BB: usize = 144;
    let nb = n.div_ceil(QK_K);
    let mut out: Vec<f32> = Vec::with_capacity(nb * QK_K);
    for i in 0..nb {
        let base = i * BB;
        let d = f16_to_f32(u16::from_le_bytes([raw[base], raw[base + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([raw[base + 2], raw[base + 3]]));
        let scales = &raw[base + 4..base + 16];
        let qs = &raw[base + 16..base + 144];
        let mut is = 0usize;
        let mut q_off = 0usize;
        for _ in 0..4 {
            let (sc1, m1u) = get_scale_min_k4(is, scales);
            let (sc2, m2u) = get_scale_min_k4(is + 1, scales);
            let (d1, m1) = (d * sc1 as f32, dmin * m1u as f32);
            let (d2, m2) = (d * sc2 as f32, dmin * m2u as f32);
            for l in 0..32 {
                out.push(d1 * (qs[q_off + l] & 0xF) as f32 - m1);
            }
            for l in 0..32 {
                out.push(d2 * (qs[q_off + l] >> 4) as f32 - m2);
            }
            q_off += 32;
            is += 2;
        }
    }
    out.truncate(n);
    out
}

// block_q5_K (176 bytes, 256 elems): d(f16), dmin(f16), scales[12], qh[32], qs[128].
fn dequant_q5_k(raw: &[u8], n: usize) -> Vec<f32> {
    const QK_K: usize = 256;
    const BB: usize = 176;
    let nb = n.div_ceil(QK_K);
    let mut out: Vec<f32> = Vec::with_capacity(nb * QK_K);
    for i in 0..nb {
        let base = i * BB;
        let d = f16_to_f32(u16::from_le_bytes([raw[base], raw[base + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([raw[base + 2], raw[base + 3]]));
        let scales = &raw[base + 4..base + 16];
        let qh = &raw[base + 16..base + 48];
        let ql = &raw[base + 48..base + 176];
        let mut is = 0usize;
        let mut ql_off = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for _ in 0..4 {
            let (sc1, m1u) = get_scale_min_k4(is, scales);
            let (sc2, m2u) = get_scale_min_k4(is + 1, scales);
            let (d1, m1) = (d * sc1 as f32, dmin * m1u as f32);
            let (d2, m2) = (d * sc2 as f32, dmin * m2u as f32);
            for l in 0..32 {
                let hi = if qh[l] & u1 != 0 { 16u32 } else { 0 };
                out.push(d1 * ((ql[ql_off + l] & 0xF) as u32 + hi) as f32 - m1);
            }
            for l in 0..32 {
                let hi = if qh[l] & u2 != 0 { 16u32 } else { 0 };
                out.push(d2 * ((ql[ql_off + l] >> 4) as u32 + hi) as f32 - m2);
            }
            ql_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    out.truncate(n);
    out
}

// block_q6_K (210 bytes, 256 elems): ql[128], qh[64], scales[16] (i8), d(f16).
fn dequant_q6_k(raw: &[u8], n: usize) -> Vec<f32> {
    const QK_K: usize = 256;
    const BB: usize = 210;
    let nb = n.div_ceil(QK_K);
    let mut out = vec![0.0f32; nb * QK_K];
    for i in 0..nb {
        let base = i * BB;
        let ql = &raw[base..base + 128];
        let qh = &raw[base + 128..base + 192];
        let sc = &raw[base + 192..base + 208];
        let d = f16_to_f32(u16::from_le_bytes([raw[base + 208], raw[base + 209]]));
        for half in 0..2 {
            let ql_off = half * 64;
            let qh_off = half * 32;
            let sc_off = half * 8;
            let y_off = i * QK_K + half * 128;
            for l in 0..32 {
                let is = l / 16;
                let ql0 = ql[ql_off + l] as i32;
                let ql32 = ql[ql_off + l + 32] as i32;
                let qhb = qh[qh_off + l] as i32;
                let q1 = ((ql0 & 0xF) | ((qhb & 3) << 4)) - 32;
                let q2 = ((ql32 & 0xF) | (((qhb >> 2) & 3) << 4)) - 32;
                let q3 = ((ql0 >> 4) | (((qhb >> 4) & 3) << 4)) - 32;
                let q4 = ((ql32 >> 4) | (((qhb >> 6) & 3) << 4)) - 32;
                out[y_off + l] = d * (sc[sc_off + is] as i8 as i32 * q1) as f32;
                out[y_off + l + 32] = d * (sc[sc_off + is + 2] as i8 as i32 * q2) as f32;
                out[y_off + l + 64] = d * (sc[sc_off + is + 4] as i8 as i32 * q3) as f32;
                out[y_off + l + 96] = d * (sc[sc_off + is + 6] as i8 as i32 * q4) as f32;
            }
        }
    }
    out.truncate(n);
    out
}

// block_q8_K (292 bytes, 256 elems): d(f32), qs[256] (i8), bsums[16] (i16, unused for dequant).
fn dequant_q8_k(raw: &[u8], n: usize) -> Vec<f32> {
    const QK_K: usize = 256;
    const BB: usize = 292;
    let nb = n.div_ceil(QK_K);
    let mut out = vec![0.0f32; nb * QK_K];
    for i in 0..nb {
        let base = i * BB;
        let d = f32::from_le_bytes([raw[base], raw[base + 1], raw[base + 2], raw[base + 3]]);
        let qs = &raw[base + 4..base + 4 + QK_K];
        for j in 0..QK_K {
            out[i * QK_K + j] = d * qs[j] as i8 as f32;
        }
    }
    out.truncate(n);
    out
}

// block_iq4_nl (18 bytes, 32 elems): d(f16), qs[16]; nibble indexes the non-linear
// codebook. Low nibble -> j, high nibble -> j+16 (same interleave as Q4_0).
fn dequant_iq4_nl(raw: &[u8], n: usize) -> Vec<f32> {
    const QK: usize = 32;
    const BB: usize = 18;
    let nb = n.div_ceil(QK);
    let mut out = vec![0.0f32; nb * QK];
    for i in 0..nb {
        let base = i * BB;
        let d = f16_to_f32(u16::from_le_bytes([raw[base], raw[base + 1]]));
        let qs = &raw[base + 2..base + 18];
        for j in 0..QK / 2 {
            out[i * QK + j] = d * KVALUES_IQ4NL[(qs[j] & 0xf) as usize] as f32;
            out[i * QK + j + QK / 2] = d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
        }
    }
    out.truncate(n);
    out
}

// block_iq4_xs (136 bytes, 256 elems): d(f16), scales_h(u16), scales_l[4], qs[128].
// 8 sub-blocks of 32; per-sub-block 6-bit scale (ls-32) from scales_l/scales_h.
fn dequant_iq4_xs(raw: &[u8], n: usize) -> Vec<f32> {
    const QK_K: usize = 256;
    const BB: usize = 136;
    let nb = n.div_ceil(QK_K);
    let mut out = vec![0.0f32; nb * QK_K];
    for i in 0..nb {
        let base = i * BB;
        let d = f16_to_f32(u16::from_le_bytes([raw[base], raw[base + 1]]));
        let scales_h = u16::from_le_bytes([raw[base + 2], raw[base + 3]]);
        let scales_l = &raw[base + 4..base + 8];
        let qs = &raw[base + 8..base + 136];
        for ib in 0..QK_K / 32 {
            let ls = (((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as i32) | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
            let dl = d * (ls - 32) as f32;
            let q_off = ib * 16;
            let y_off = i * QK_K + ib * 32;
            for j in 0..16 {
                out[y_off + j] = dl * KVALUES_IQ4NL[(qs[q_off + j] & 0xf) as usize] as f32;
                out[y_off + j + 16] = dl * KVALUES_IQ4NL[(qs[q_off + j] >> 4) as usize] as f32;
            }
        }
    }
    out.truncate(n);
    out
}

/// ggml tensor name → HF name (`None` = skip, e.g. rope freqs).
fn map_name(g: &str) -> Option<String> {
    match g {
        "token_embd.weight" => return Some("model.embed_tokens.weight".into()),
        "output_norm.weight" => return Some("model.norm.weight".into()),
        "output.weight" => return Some("lm_head.weight".into()),
        _ => {}
    }
    let rest = g.strip_prefix("blk.")?;
    let (idx, suffix) = rest.split_once('.')?;
    let mapped = match suffix {
        "attn_norm.weight" => "input_layernorm.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_q.bias" => "self_attn.q_proj.bias",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_k.bias" => "self_attn.k_proj.bias",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_v.bias" => "self_attn.v_proj.bias",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        _ => return None,
    };
    Some(format!("model.layers.{idx}.{mapped}"))
}

fn arch_from_md(md: &BTreeMap<String, Val>) -> anyhow::Result<ModelArch> {
    let arch = md.get("general.architecture").and_then(|v| v.as_str()).unwrap_or("qwen2").to_string();
    let g = |k: &str| md.get(&format!("{arch}.{k}"));
    let gu = |k: &str| g(k).and_then(|v| v.as_u64()).map(|x| x as usize);
    let n_layers = gu("block_count").ok_or_else(|| anyhow::anyhow!("gguf: no block_count"))?;
    let hidden = gu("embedding_length").ok_or_else(|| anyhow::anyhow!("gguf: no embedding_length"))?;
    let n_heads = gu("attention.head_count").ok_or_else(|| anyhow::anyhow!("gguf: no head_count"))?;
    let vocab = md.get("tokenizer.ggml.tokens").and_then(|v| if let Val::StrArr(a) = v { Some(a.len()) } else { None }).unwrap_or(0);
    let norm_style = if arch.contains("gemma") { NormStyle::Gemma } else { NormStyle::Qwen };
    Ok(ModelArch {
        arch_name: arch.clone(),
        hidden_size: hidden,
        intermediate_size: gu("feed_forward_length").unwrap_or(0),
        num_layers: n_layers,
        num_attention_heads: n_heads,
        num_kv_heads: gu("attention.head_count_kv").unwrap_or(n_heads),
        head_dim: gu("attention.key_length").unwrap_or(hidden / n_heads.max(1)),
        vocab_size: vocab,
        layer_types: vec![LayerType::FullAttention; n_layers],
        rms_norm_eps: g("attention.layer_norm_rms_epsilon").and_then(|v| v.as_f64()).unwrap_or(1e-6),
        norm_style,
        rope_theta: g("rope.freq_base").and_then(|v| v.as_f64()).unwrap_or(10_000.0),
        tie_word_embeddings: false,
        partial_rotary_factor: 1.0,
        yarn: None,
        attention_heads_per_layer: None,
        mtp: None,
        moe: None,
        linear_core: None,
        max_position_embeddings: gu("context_length").unwrap_or(32_768),
        linear_conv_kernel_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        hidden_act: "silu".into(),
        embed_multiplier: 1.0,
        query_pre_attn_scalar: None,
        sliding_window: None,
        sliding_window_pattern: None,
        rope_local_base_freq: None,
        local_partial_rotary_factor: None,
        global_head_dim: None,
        num_global_kv_heads: None,
        global_partial_rotary_factor: None,
        final_logit_softcapping: None,
        attn_v_norm: false,
        num_loops: 1,
        loop_final_norm: false,
    })
}

/// Reconstruct a HF byte-level-BPE tokenizer.json + chat bundle from ggml metadata.
fn tokenizer(md: &BTreeMap<String, Val>) -> (Option<Vec<u8>>, TokenizerBundle) {
    let empty = TokenizerBundle {
        chat_template: None,
        eos_token_ids: Vec::new(),
        bos_token_id: None,
        pad_token_id: None,
    };
    let tokens = match md.get("tokenizer.ggml.tokens") {
        Some(Val::StrArr(a)) => a,
        _ => return (None, empty),
    };
    let types: &[i64] = match md.get("tokenizer.ggml.token_type") {
        Some(Val::IntArr(a)) => a,
        _ => &[],
    };
    let merges = match md.get("tokenizer.ggml.merges") {
        Some(Val::StrArr(a)) => a.clone(),
        _ => Vec::new(),
    };
    // vocab: token -> id
    let vocab: serde_json::Map<String, serde_json::Value> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), serde_json::json!(i))).collect();
    // added/special tokens: ggml token_type CONTROL(3) / USER_DEFINED(4).
    let added: Vec<serde_json::Value> = tokens
        .iter()
        .enumerate()
        .filter(|(i, _)| matches!(types.get(*i).copied(), Some(3) | Some(4)))
        .map(|(i, t)| {
            serde_json::json!({
                "id": i, "content": t, "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true
            })
        })
        .collect();
    let tj = serde_json::json!({
        "version": "1.0",
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": true },
        "post_processor": null,
        "decoder": { "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true },
        "model": {
            "type": "BPE", "dropout": null, "unk_token": null,
            "continuing_subword_prefix": null, "end_of_word_suffix": null,
            "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
            "vocab": vocab, "merges": merges
        }
    });
    let eos = md.get("tokenizer.ggml.eos_token_id").and_then(|v| v.as_u64()).map(|x| x as u32);
    let bos = md.get("tokenizer.ggml.bos_token_id").and_then(|v| v.as_u64()).map(|x| x as u32);
    let pad = md.get("tokenizer.ggml.padding_token_id").and_then(|v| v.as_u64()).map(|x| x as u32);
    let bundle = TokenizerBundle {
        chat_template: md.get("tokenizer.chat_template").and_then(|v| v.as_str().map(String::from)),
        eos_token_ids: eos.into_iter().collect(),
        bos_token_id: bos,
        pad_token_id: pad,
    };
    (Some(serde_json::to_vec(&tj).unwrap()), bundle)
}

/// Import a GGUF file into a `.cmf` (quantized with `quant`).
/// Resolve a GGUF source spec to a local path: a local file, an HF repo id
/// (auto-pick the best `.gguf`), or `owner/repo/path/file.gguf`.
fn resolve_gguf_source(spec: &str, token: Option<&str>) -> anyhow::Result<std::path::PathBuf> {
    let p = std::path::Path::new(spec);
    if p.exists() {
        return Ok(p.to_path_buf());
    }
    let segs: Vec<&str> = spec.trim_matches('/').split('/').collect();
    if spec.to_lowercase().ends_with(".gguf") && segs.len() >= 3 {
        // owner/repo/<file...>.gguf — a specific file inside a repo.
        let repo = format!("{}/{}", segs[0], segs[1]);
        let file = segs[2..].join("/");
        eprintln!("downloading {file} from {repo}…");
        return convert::hf_fetch_file(&repo, &file, token);
    }
    if convert::looks_like_repo(spec) {
        let files = convert::hf_repo_files(spec, token);
        let ggufs: Vec<&String> = files.iter().filter(|f| f.to_lowercase().ends_with(".gguf")).collect();
        if ggufs.is_empty() {
            anyhow::bail!("'{spec}': the HF repo has no .gguf files");
        }
        let pick = pick_gguf(&ggufs);
        eprintln!("selected {pick} from {spec} ({} .gguf files available)", ggufs.len());
        return convert::hf_fetch_file(spec, pick, token);
    }
    anyhow::bail!("'{spec}': not a local .gguf file, an HF repo id (owner/name), or owner/name/file.gguf")
}

/// Pick the highest-fidelity natively-supported `.gguf` from a repo's file list.
/// (IQ* codebook types are skipped — the importer does not decode them.)
fn pick_gguf<'a>(files: &[&'a String]) -> &'a str {
    const PREF: &[&str] = &["q8_0", "bf16", "f16", "fp16", "q6_k", "q5_k", "q5_1", "q5_0", "q4_k", "q4_1", "q4_0", "q3_k", "q2_k"];
    for key in PREF {
        if let Some(f) = files.iter().find(|f| f.to_lowercase().contains(key) && !f.to_lowercase().contains("iq")) {
            return f.as_str();
        }
    }
    // Fall back to the first non-IQ file, else the very first.
    files.iter().find(|f| !f.to_lowercase().contains("iq")).unwrap_or(&files[0]).as_str()
}

pub fn run_import_gguf(gguf: &str, quant: &str, output: &str, hf_token: Option<&str>, mut progress: impl FnMut(f32)) -> anyhow::Result<()> {
    let quant = convert::parse_quant(quant)?;
    // Source: a local .gguf, an HF repo id (auto-pick a .gguf), or owner/repo/file.gguf.
    let path = resolve_gguf_source(gguf, hf_token)?;
    let g = parse(&path)?;

    // Honest guard: the native GGUF importer maps standard transformer tensors
    // only. A linear-attention / GatedDeltaNet (SSM) hybrid — llama.cpp `qwen35`
    // / `qwen3_next`, or Mamba — would silently lose every mixer tensor. Refuse
    // it clearly instead of writing a broken model; the safetensors path works.
    if let Some(t) = g.tensors.iter().find(|t| t.name.contains("ssm")) {
        anyhow::bail!(
            "GGUF '{}' is a linear-attention / SSM hybrid (e.g. tensor '{}') — the native \
             GGUF importer handles standard transformer layouts only. Convert the model's \
             safetensors repo with `cortiq convert` instead (GatedDeltaNet is supported there).",
            path.display(),
            t.name
        );
    }

    let arch = arch_from_md(&g.md)?;
    let is_llama = arch.arch_name == "llama";
    let n_heads = arch.num_attention_heads;
    let n_kv = arch.num_kv_heads;

    let total = g.tensors.len().max(1);
    let mut tensors: Vec<TensorSpec> = Vec::with_capacity(total);
    for (idx, t) in g.tensors.iter().enumerate() {
        progress((idx + 1) as f32 / total as f32);
        let Some(name) = map_name(&t.name) else {
            continue;
        };
        let numel: usize = t.dims.iter().map(|&d| d as usize).product();
        let nb = nbytes(t.ggml_type, numel)?;
        let raw = &g.bytes[g.data_start + t.offset as usize..g.data_start + t.offset as usize + nb];
        let mut vals = dequant(t.ggml_type, raw, numel)?;
        // HF shape = ggml dims reversed (ne[0] is fastest / the input dim).
        let shape: Vec<usize> = t.dims.iter().rev().map(|&d| d as usize).collect();

        // llama.cpp permutes q/k weights for its rope; undo it for HF layout.
        if is_llama && shape.len() == 2 {
            if name.ends_with("self_attn.q_proj.weight") {
                vals = unpermute(&vals, shape[0], shape[1], n_heads);
            } else if name.ends_with("self_attn.k_proj.weight") {
                vals = unpermute(&vals, shape[0], shape[1], n_kv);
            }
        }

        let two_d = shape.len() == 2 && numel >= 32;
        let (dt, data) = if two_d { convert::quantize_2d(quant, &vals, shape[0], shape[1]) } else { (TensorDtype::F16, convert::encode_f16(&vals)) };
        tensors.push(TensorSpec { name, dtype: dt, shape, data });
    }

    let (vocab, bundle) = tokenizer(&g.md);
    let quant_type = match quant {
        Quant::Q8Row => QuantType::Q8Row,
        Quant::Q8_2f => QuantType::Q8_2f,
        Quant::Q4Block => QuantType::Q4Block,
        Quant::F16 => QuantType::F16,
        Quant::Vbit => QuantType::Vbit,
        Quant::Q4Tiled => QuantType::Q4Block,
        Quant::Q1 | Quant::Q1p | Quant::Q1s | Quant::Q1t => QuantType::Vbit,
    };
    let header = CmfHeader {
        format: "cmf".into(),
        version: CMF_VERSION,
        arch,
        quant_type,
        provenance: Some(serde_json::json!({ "tool": "cortiq import-gguf", "source": gguf })),
        tokenizer_config: Some(bundle),
        section_hashes: None,
        skills: Vec::new(),
        shard: None,
        calibration: None,
    };
    CmfModel::write(output, &header, &tensors, None, vocab.as_deref()).map_err(|e| anyhow::anyhow!("write {output}: {e}"))?;
    progress(1.0);
    Ok(())
}

/// Undo llama.cpp's q/k rope permutation: rows are interleaved (d/2, 2) → (2, d/2).
fn unpermute(vals: &[f32], out_dim: usize, in_dim: usize, n_heads: usize) -> Vec<f32> {
    if n_heads == 0 || out_dim % n_heads != 0 {
        return vals.to_vec();
    }
    let hd = out_dim / n_heads; // head dim
    if hd % 2 != 0 {
        return vals.to_vec();
    }
    let half = hd / 2;
    let mut out = vec![0f32; vals.len()];
    for h in 0..n_heads {
        for r in 0..hd {
            // permuted row r ← original row: interleave halves
            let src_r = if r < half { r * 2 } else { (r - half) * 2 + 1 };
            let dst = (h * hd + r) * in_dim;
            let srco = (h * hd + src_r) * in_dim;
            out[dst..dst + in_dim].copy_from_slice(&vals[srco..srco + in_dim]);
        }
    }
    out
}

#[cfg(test)]
mod dequant_tests {
    use super::*;
    fn unhex(h: &str) -> Vec<u8> {
        (0..h.len()).step_by(2).map(|i| u8::from_str_radix(&h[i..i + 2], 16).unwrap()).collect()
    }
    #[test]
    fn q6k_matches_ggml_reference() {
        // A real blk.0.ffn_down block from Qwen2.5-0.5B-Instruct q6_k.gguf.
        let raw = unhex(
            "277930fb06d815ad9bed79c397dd1c7a10f175bd78508d65a71ebb10484afc7187ec41365560eff0fc04dee1790a59b6168bb16da04dc9126a1092d41793bbe4fef1c110260a98efde182bc43d8ba932f61201521b56897d1d6f33265ea1f9afd84a1093dd31cbecfbb73ac8397e4a084eab57fe90da90d431fdce0b6ff67e07aa61a8896a964b59611565505695a6ac865b67a46da544ad4961b94322a25c4049d69204276592554aa96a56599296299ac66b964ad651e9e9b415a6a628d52531dabc1b91ce6c4b503bd580adaaca262b81",
        );
        assert_eq!(raw.len(), 210);
        let out = dequant_q6_k(&raw, 256);
        let expect: &[(usize, f32)] = &[
            (0, -0.006113),
            (1, 0.006113),
            (2, 0.027945),
            (3, 0.004366),
            (4, -0.00524),
            (5, -0.006986),
            (6, -0.018339),
            (7, 0.00262),
            (32, 0.008483),
            (64, 0.003956),
            (96, -0.015398),
            (127, 0.002673),
        ];
        for &(i, e) in expect {
            assert!((out[i] - e).abs() < 2e-4, "idx {i}: got {} want {}", out[i], e);
        }
    }

    #[test]
    fn q4k_matches_ggml_reference() {
        // Real blk.11.ffn_down block from Qwen2.5-0.5B q4_k_m.gguf (max err vs fp16 = 7e-4).
        let raw = unhex(
            "72016409bafff4f3beffe2f58d5554628a96507978a697c576bb2d98d59693c0a756bf48ed5889a9e6ac0996cc74db3841c402c583f596c7865b6495dc90c7628442475e3b6570a44396e922b0e1b87083f6499396d2844a747f596892629433c95b593770fd9196846b850159d3b3b8cb87d56697488005d44bf48ff9dbf5c8b795d877680ca876ca5981a742a139a8",
        );
        assert_eq!(raw.len(), 144);
        let out = dequant_q4_k(&raw, 256);
        for &(i, e) in &[(0usize, 0.002592f32), (1, -0.002525), (2, -0.0102), (31, 3.3e-5), (32, 0.000751), (64, -0.004447), (128, -0.003603), (200, -0.004132), (255, 0.002143)] {
            assert!((out[i] - e).abs() < 2e-4, "q4k idx {i}: got {} want {}", out[i], e);
        }
    }

    #[test]
    fn q5k_matches_ggml_reference() {
        // Real blk.11.ffn_down block from Qwen2.5-0.5B q5_k_m.gguf (max err vs fp16 = 3e-4).
        let raw = unhex(
            "ab008d09bdfff4f7bffee2f26f482846e3aa902ba12aaa1e885791dbeecaaac2ba90d7054771a32be25bad821baa6ff1164cc104f16e3eacfe974c41db3e47906fbe8fa1ebb22363fd5a025ea908c8718278058bf6fc2d700ca8b92baa218fb4097580af68bae14a762db43460b261e127fd93253bb32784f8ffb2d123d3295781b6a17ee0fb322d08c70a02b2a67670760d99cb3d7f001a9886f71ee2a5ea806e19a0fdc0075fec83a1015e74525140",
        );
        assert_eq!(raw.len(), 176);
        let out = dequant_q5_k(&raw, 256);
        for &(i, e) in &[(0usize, 0.003006f32), (1, -0.003211), (2, -0.01005), (31, -0.000102), (32, 0.000413), (64, -0.004699), (128, -0.003084), (200, -0.003904), (255, 0.002199)] {
            assert!((out[i] - e).abs() < 2e-4, "q5k idx {i}: got {} want {}", out[i], e);
        }
    }
}
