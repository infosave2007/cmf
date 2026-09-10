//! Native GGUF → `.cmf` importer. Parses a GGUF file, dequantizes every common
//! ggml tensor type (F32, F16, BF16, Q4_0/1, Q5_0/1, Q8_0, and the K-quants
//! Q2_K–Q6_K + Q8_K — all faithful ports of ggml `dequantize_row_*`), maps ggml
//! tensor names to HF names, reconstructs a Hugging Face tokenizer.json from the
//! embedded ggml metadata, and writes a `.cmf`. No Python. A GGUF repo id can be
//! passed directly (the matching `.gguf` is downloaded). IQ4_NL / IQ4_XS (the
//! non-linear 4-bit codebook, used inside q2_k/q3_k mixes) are handled; the
//! IQ1/IQ2/IQ3 grid-codebook types are the only ggml types not yet supported.
//! Qwen Image diffusion-transformer GGUFs use a separate component branch
//! which preserves source tensor names and stores the official image
//! transformer config instead of interpreting the file as an LLM.

use crate::convert::{self, Quant};
use cortiq_core::format::{
    CMF_VERSION, CmfHeader, CmfModel, CmfStreamWriter, TensorSpec, TokenizerBundle,
};
use cortiq_core::quant::f16_to_f32;
use cortiq_core::types::{LayerType, ModelArch, MoeConfig, NormStyle, QuantType, TensorDtype};
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
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

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
        } else if matches!(
            et,
            T_U8 | T_I8 | T_U16 | T_I16 | T_U32 | T_I32 | T_U64 | T_I64 | T_BOOL
        ) {
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
    /// The whole file, memory-mapped — a 20 GB+ MoE GGUF must not be
    /// slurped into RAM on a 24 GB machine.
    bytes: memmap2::Mmap,
    data_start: usize,
}

fn align_up(x: usize, a: usize) -> usize {
    x.div_ceil(a) * a
}

fn parse(path: &std::path::Path) -> anyhow::Result<Gguf> {
    let file = fs::File::open(path)?;
    // SAFETY: read-only map of a file we just opened.
    let bytes = unsafe { memmap2::Mmap::map(&file)? };
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
        tensors.push(GgufTensor {
            name,
            dims,
            ggml_type,
            offset,
        });
    }
    let align = md
        .get("general.alignment")
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as usize;
    let data_start = align_up(c.p, align.max(1));
    Ok(Gguf {
        md,
        tensors,
        bytes,
        data_start,
    })
}

/// Dequantize `n` elements of a ggml tensor into f32. Every codec below is a
/// faithful port of ggml's `dequantize_row_*` (ggml-quants.c); output order and
/// scale packing match byte-for-byte.
fn dequant(ggml_type: u32, raw: &[u8], n: usize) -> anyhow::Result<Vec<f32>> {
    Ok(match ggml_type {
        GGML_F32 => raw
            .chunks_exact(4)
            .take(n)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        GGML_F16 => raw
            .chunks_exact(2)
            .take(n)
            .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
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
            let ls = (((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as i32)
                | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
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
        // ── qwen35moe (KAT-Coder / Qwen3.6-MoE class): GDN hybrid ──
        // llama.cpp pre-splits the fused HF projections; names map 1:1
        // onto the engine's linear_attn.* / mlp.* layout. The routed
        // expert tensors (ffn_*_exps) are 3-D and handled separately.
        "post_attention_norm.weight" => "post_attention_layernorm.weight",
        "attn_qkv.weight" => "linear_attn.in_proj_qkv.weight",
        "attn_gate.weight" => "linear_attn.in_proj_z.weight",
        "ssm_alpha.weight" => "linear_attn.in_proj_a.weight",
        "ssm_beta.weight" => "linear_attn.in_proj_b.weight",
        "ssm_a" => "linear_attn.A_log",
        "ssm_dt.bias" => "linear_attn.dt_bias",
        "ssm_conv1d.weight" => "linear_attn.conv1d.weight",
        "ssm_norm.weight" => "linear_attn.norm.weight",
        "ssm_out.weight" => "linear_attn.out_proj.weight",
        "ffn_gate_inp.weight" => "mlp.gate.weight",
        "ffn_gate_inp_shexp.weight" => "mlp.shared_expert_gate.weight",
        "ffn_gate_shexp.weight" => "mlp.shared_expert.gate_proj.weight",
        "ffn_up_shexp.weight" => "mlp.shared_expert.up_proj.weight",
        "ffn_down_shexp.weight" => "mlp.shared_expert.down_proj.weight",
        _ => return None,
    };
    Some(format!("model.layers.{idx}.{mapped}"))
}

fn arch_from_md(md: &BTreeMap<String, Val>, tensors: &[GgufTensor]) -> anyhow::Result<ModelArch> {
    let arch = md
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("qwen2")
        .to_string();
    let g = |k: &str| md.get(&format!("{arch}.{k}"));
    let gu = |k: &str| g(k).and_then(|v| v.as_u64()).map(|x| x as usize);
    let gf = |k: &str| g(k).and_then(|v| v.as_f64());
    let n_layers = gu("block_count").ok_or_else(|| anyhow::anyhow!("gguf: no block_count"))?;
    let hidden =
        gu("embedding_length").ok_or_else(|| anyhow::anyhow!("gguf: no embedding_length"))?;
    let n_heads =
        gu("attention.head_count").ok_or_else(|| anyhow::anyhow!("gguf: no head_count"))?;
    let vocab = md
        .get("tokenizer.ggml.tokens")
        .and_then(|v| {
            if let Val::StrArr(a) = v {
                Some(a.len())
            } else {
                None
            }
        })
        .unwrap_or(0);
    let is_q35 = arch.starts_with("qwen35");
    let is_gemma2 = arch == "gemma2";
    let norm_style = if arch.contains("gemma") || is_q35 {
        // qwen3.5 / qwen3.6 use zero-centered x̂·(1+w) norms.
        NormStyle::Gemma
    } else {
        NormStyle::Qwen
    };
    // qwen35moe (GDN hybrid + MoE): the attention/linear schedule comes
    // from tensor PRESENCE (full-attention layers carry attn_q, GDN
    // layers carry attn_qkv + ssm_*) — more robust than trusting the
    // full_attention_interval semantics.
    let layer_types = if is_q35 {
        (0..n_layers)
            .map(|i| {
                if tensors
                    .iter()
                    .any(|t| t.name == format!("blk.{i}.attn_q.weight"))
                {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect()
    } else {
        vec![LayerType::FullAttention; n_layers]
    };
    let head_dim = gu("attention.key_length").unwrap_or(hidden / n_heads.max(1));
    let moe = if is_q35 {
        gu("expert_count").filter(|&n| n > 0).map(|ne| MoeConfig {
            num_experts: ne,
            top_k: gu("expert_used_count").unwrap_or(8),
            moe_intermediate_size: gu("expert_feed_forward_length").unwrap_or(0),
            // Qwen3.5/3.6 renormalize the top-k softmax weights.
            norm_topk_prob: true,
            shared_expert_intermediate_size: gu("expert_shared_feed_forward_length"),
            router_sigmoid: false,
            routed_scaling_factor: None,
            router_resonance: false,
        })
    } else {
        None
    };
    // GDN geometry (qwen35moe): key heads = ssm.group_count at
    // state_size dims each; value heads = inner_size / state_size.
    let ssm_state = gu("ssm.state_size");
    let ssm_vheads = match (gu("ssm.inner_size"), ssm_state) {
        (Some(inner), Some(st)) if st > 0 => Some(inner / st),
        _ => None,
    };
    Ok(ModelArch {
        arch_name: if is_q35 {
            "qwen3_5_moe".into()
        } else {
            arch.clone()
        },
        hidden_size: hidden,
        intermediate_size: gu("feed_forward_length").unwrap_or(0),
        num_layers: n_layers,
        num_attention_heads: n_heads,
        num_kv_heads: gu("attention.head_count_kv").unwrap_or(n_heads),
        head_dim,
        vocab_size: vocab,
        layer_types,
        rms_norm_eps: g("attention.layer_norm_rms_epsilon")
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6),
        norm_style,
        rope_theta: g("rope.freq_base")
            .and_then(|v| v.as_f64())
            .unwrap_or(10_000.0),
        // No separate output head in the file = tied embeddings.
        tie_word_embeddings: !tensors.iter().any(|t| t.name == "output.weight"),
        partial_rotary_factor: match gu("rope.dimension_count") {
            Some(rd) if head_dim > 0 && rd < head_dim => rd as f32 / head_dim as f32,
            _ => 1.0,
        },
        yarn: None,
        attention_heads_per_layer: None,
        mtp: None,
        moe,
        qwen4_exp: None,
        deepseek_v41: None,
        linear_core: if is_q35 {
            Some(cortiq_core::types::LinearCoreConfig {
                kind: "gated_delta_net".into(),
                num_heads: ssm_vheads.unwrap_or(0),
                nphase: None,
                value_head_dim: ssm_state.unwrap_or(0),
            })
        } else {
            None
        },
        head_clusters: None,
        max_position_embeddings: gu("context_length").unwrap_or(32_768),
        linear_conv_kernel_dim: gu("ssm.conv_kernel"),
        linear_num_key_heads: gu("ssm.group_count"),
        linear_num_value_heads: ssm_vheads,
        linear_key_head_dim: ssm_state,
        linear_value_head_dim: ssm_state,
        hidden_act: "silu".into(),
        embed_multiplier: 1.0,
        query_pre_attn_scalar: None,
        sliding_window: if is_gemma2 {
            gu("attention.sliding_window")
        } else {
            None
        },
        sliding_window_pattern: if is_gemma2 { Some(2) } else { None },
        rope_local_base_freq: None,
        local_partial_rotary_factor: None,
        global_head_dim: None,
        num_global_kv_heads: None,
        global_partial_rotary_factor: None,
        final_logit_softcapping: gf("attn_logit_softcapping")
            .is_some()
            .then(|| gf("final_logit_softcapping"))
            .flatten()
            .map(|v| v),
        attn_logit_softcapping: gf("attn_logit_softcapping").map(|v| v),
        mla: None,
        activation_situ_beta: None,
        activation_situ_linear_beta: None,
        attn_v_norm: false,
        num_loops: 1,
        kda_gate_lower_bound: None,
        g3n: None,
        rope_freq_factors: None,
        logit_multiplier: None,
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
    let vocab: serde_json::Map<String, serde_json::Value> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), serde_json::json!(i)))
        .collect();
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
    let eos = md
        .get("tokenizer.ggml.eos_token_id")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32);
    let bos = md
        .get("tokenizer.ggml.bos_token_id")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32);
    let pad = md
        .get("tokenizer.ggml.padding_token_id")
        .and_then(|v| v.as_u64())
        .map(|x| x as u32);
    let bundle = TokenizerBundle {
        chat_template: md
            .get("tokenizer.chat_template")
            .and_then(|v| v.as_str().map(String::from)),
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
        let ggufs: Vec<&String> = files
            .iter()
            .filter(|f| f.to_lowercase().ends_with(".gguf"))
            .collect();
        if ggufs.is_empty() {
            anyhow::bail!("'{spec}': the HF repo has no .gguf files");
        }
        let pick = pick_gguf(&ggufs);
        eprintln!(
            "selected {pick} from {spec} ({} .gguf files available)",
            ggufs.len()
        );
        return convert::hf_fetch_file(spec, pick, token);
    }
    anyhow::bail!(
        "'{spec}': not a local .gguf file, an HF repo id (owner/name), or owner/name/file.gguf"
    )
}

/// Pick the highest-fidelity natively-supported `.gguf` from a repo's file list.
/// (IQ* codebook types are skipped — the importer does not decode them.)
fn pick_gguf<'a>(files: &[&'a String]) -> &'a str {
    const PREF: &[&str] = &[
        "q8_0", "bf16", "f16", "fp16", "q6_k", "q5_k", "q5_1", "q5_0", "q4_k", "q4_1", "q4_0",
        "q3_k", "q2_k",
    ];
    for key in PREF {
        if let Some(f) = files
            .iter()
            .find(|f| f.to_lowercase().contains(key) && !f.to_lowercase().contains("iq"))
        {
            return f.as_str();
        }
    }
    // Fall back to the first non-IQ file, else the very first.
    files
        .iter()
        .find(|f| !f.to_lowercase().contains("iq"))
        .unwrap_or(&files[0])
        .as_str()
}

fn quant_type_for(quant: Quant) -> QuantType {
    match quant {
        Quant::Q8Row => QuantType::Q8Row,
        Quant::Q8_2f => QuantType::Q8_2f,
        Quant::Q4Block => QuantType::Q4Block,
        Quant::F16 => QuantType::F16,
        Quant::Vbit => QuantType::Vbit,
        Quant::Q4Tiled | Quant::Q4TiledP | Quant::Q2TiledP => QuantType::Q4Block,
        Quant::Q1 | Quant::Q1p | Quant::Q1s | Quant::Q1t => QuantType::Vbit,
    }
}

struct QwenImageGeometry {
    hidden_size: usize,
    intermediate_size: usize,
    num_layers: usize,
    num_attention_heads: usize,
    head_dim: usize,
    in_channels: usize,
    joint_attention_dim: usize,
    out_channels: usize,
}

fn qwen_image_named_shape(tensors: &[GgufTensor], name: &str) -> anyhow::Result<Vec<usize>> {
    let tensor = tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| anyhow::anyhow!("qwen_image GGUF is missing required tensor '{name}'"))?;
    Ok(qwen_image_shape(tensor)?.0)
}

fn qwen_image_geometry(tensors: &[GgufTensor]) -> anyhow::Result<QwenImageGeometry> {
    let hidden_shape = qwen_image_named_shape(tensors, "img_in.bias")?;
    if hidden_shape.len() != 1 {
        anyhow::bail!("qwen_image img_in.bias must be 1-D, got {hidden_shape:?}");
    }
    let hidden_size = hidden_shape[0];

    let img_in = qwen_image_named_shape(tensors, "img_in.weight")?;
    if img_in.len() != 2 || img_in[0] != hidden_size {
        anyhow::bail!(
            "qwen_image img_in.weight must have framework shape [{hidden_size}, in_channels], got {img_in:?}"
        );
    }
    let in_channels = img_in[1];

    let txt_in = qwen_image_named_shape(tensors, "txt_in.weight")?;
    if txt_in.len() != 2 || txt_in[0] != hidden_size {
        anyhow::bail!(
            "qwen_image txt_in.weight must have framework shape [{hidden_size}, joint_attention_dim], got {txt_in:?}"
        );
    }
    let joint_attention_dim = txt_in[1];

    let proj_out = qwen_image_named_shape(tensors, "proj_out.weight")?;
    const PATCH_SIZE: usize = 2;
    let patch_area = PATCH_SIZE * PATCH_SIZE;
    if proj_out.len() != 2 || proj_out[1] != hidden_size || proj_out[0] % patch_area != 0 {
        anyhow::bail!(
            "qwen_image proj_out.weight must have framework shape [out_channels*{}, {hidden_size}], got {proj_out:?}",
            patch_area
        );
    }
    let out_channels = proj_out[0] / patch_area;

    let mut layer_ids = BTreeMap::new();
    for tensor in tensors {
        let Some(rest) = tensor.name.strip_prefix("transformer_blocks.") else {
            continue;
        };
        let Some(raw_id) = rest.split('.').next() else {
            continue;
        };
        if let Ok(id) = raw_id.parse::<usize>() {
            layer_ids.insert(id, ());
        }
    }
    let Some(&last_layer) = layer_ids.keys().next_back() else {
        anyhow::bail!("qwen_image GGUF has no transformer_blocks.* tensors");
    };
    for id in 0..=last_layer {
        if !layer_ids.contains_key(&id) {
            anyhow::bail!("qwen_image transformer block ids are not contiguous at {id}");
        }
    }
    let num_layers = last_layer + 1;
    let first_block = 0;
    let norm_q = qwen_image_named_shape(
        tensors,
        &format!("transformer_blocks.{first_block}.attn.norm_q.weight"),
    )?;
    if norm_q.len() != 1 {
        anyhow::bail!("qwen_image attention norm_q must be 1-D, got {norm_q:?}");
    }
    let head_dim = norm_q[0];
    // Qwen Image's three RoPE axes are [16, 56, 56], so the canonical head
    // dimension is 128. Read it from the tensor but reject a different family
    // rather than writing a file whose static RoPE contract is false.
    if head_dim != 128 {
        anyhow::bail!(
            "qwen_image attention head dimension {head_dim} is unsupported; expected the canonical 128"
        );
    }
    if hidden_size % head_dim != 0 {
        anyhow::bail!(
            "qwen_image hidden size {hidden_size} is not divisible by head dimension {head_dim}"
        );
    }
    let num_attention_heads = hidden_size / head_dim;

    let img_mlp = qwen_image_named_shape(
        tensors,
        &format!("transformer_blocks.{first_block}.img_mlp.net.0.proj.weight"),
    )?;
    if img_mlp.len() != 2 || img_mlp[1] != hidden_size {
        anyhow::bail!(
            "qwen_image image MLP projection must have framework shape [intermediate_size, {hidden_size}], got {img_mlp:?}"
        );
    }
    let intermediate_size = img_mlp[0];

    Ok(QwenImageGeometry {
        hidden_size,
        intermediate_size,
        num_layers,
        num_attention_heads,
        head_dim,
        in_channels,
        joint_attention_dim,
        out_channels,
    })
}

fn qwen_image_config(geometry: &QwenImageGeometry) -> serde_json::Value {
    // The transformer component's official diffusers config. Geometry that is
    // represented by tensors is derived above; patching and the three RoPE
    // axes are the canonical Qwen Image family contract.
    serde_json::json!({
        "_class_name": "QwenImageTransformer2DModel",
        "_diffusers_version": "0.36.0.dev0",
        "attention_head_dim": geometry.head_dim,
        "axes_dims_rope": [16, 56, 56],
        "guidance_embeds": false,
        "in_channels": geometry.in_channels,
        "joint_attention_dim": geometry.joint_attention_dim,
        "num_attention_heads": geometry.num_attention_heads,
        "num_layers": geometry.num_layers,
        "out_channels": geometry.out_channels,
        "patch_size": 2
    })
}

fn qwen_image_arch(geometry: &QwenImageGeometry) -> ModelArch {
    ModelArch {
        arch_name: "qwen_image".into(),
        hidden_size: geometry.hidden_size,
        intermediate_size: geometry.intermediate_size,
        num_layers: geometry.num_layers,
        num_attention_heads: geometry.num_attention_heads,
        num_kv_heads: geometry.num_attention_heads,
        head_dim: geometry.head_dim,
        vocab_size: 0,
        layer_types: vec![LayerType::FullAttention; geometry.num_layers],
        rms_norm_eps: 1e-5,
        norm_style: NormStyle::Qwen,
        rope_theta: 10_000.0,
        tie_word_embeddings: false,
        partial_rotary_factor: 1.0,
        yarn: None,
        attention_heads_per_layer: None,
        // Qwen Image's MLPs use GELU with the PyTorch tanh approximation.
        hidden_act: "gelu_tanh".into(),
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
        attn_logit_softcapping: None,
        mla: None,
        activation_situ_beta: None,
        activation_situ_linear_beta: None,
        attn_v_norm: false,
        mtp: None,
        moe: None,
        qwen4_exp: None,
        deepseek_v41: None,
        linear_core: None,
        head_clusters: None,
        max_position_embeddings: 0,
        linear_conv_kernel_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        rope_freq_factors: None,
        logit_multiplier: None,
        g3n: None,
        kda_gate_lower_bound: None,
        num_loops: 1,
        loop_final_norm: false,
    }
}

fn qwen_image_raw<'a>(g: &'a Gguf, t: &GgufTensor, numel: usize) -> anyhow::Result<&'a [u8]> {
    let nb = nbytes(t.ggml_type, numel)?;
    let offset = usize::try_from(t.offset)
        .map_err(|_| anyhow::anyhow!("gguf tensor '{}': offset overflows usize", t.name))?;
    let start = g
        .data_start
        .checked_add(offset)
        .ok_or_else(|| anyhow::anyhow!("gguf tensor '{}': data offset overflows", t.name))?;
    let end = start
        .checked_add(nb)
        .ok_or_else(|| anyhow::anyhow!("gguf tensor '{}': data range overflows", t.name))?;
    if end > g.bytes.len() {
        anyhow::bail!(
            "gguf tensor '{}' is truncated: needs bytes [{start}, {end}), file has {}",
            t.name,
            g.bytes.len()
        );
    }
    Ok(&g.bytes[start..end])
}

fn qwen_image_temp_path(output: &std::path::Path) -> anyhow::Result<std::path::PathBuf> {
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let stem = output
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("output.cmf");
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    for attempt in 0..32u32 {
        let candidate = parent.join(format!(
            ".{stem}.qwen-image-{}-{nonce}-{attempt}.partial",
            std::process::id()
        ));
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(_) => return Ok(candidate),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
    anyhow::bail!(
        "could not allocate a unique temporary CMF beside {}",
        output.display()
    )
}

fn qwen_image_sync_parent(path: &std::path::Path) -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| std::path::Path::new("."));
        fs::File::open(parent)?.sync_all()?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

fn qwen_image_shape(t: &GgufTensor) -> anyhow::Result<(Vec<usize>, usize)> {
    if t.dims.is_empty() || t.dims.len() > 2 {
        anyhow::bail!(
            "qwen_image tensor '{}' has unsupported rank {}; only 1-D and 2-D tensors are supported",
            t.name,
            t.dims.len()
        );
    }
    let shape: Vec<usize> = t
        .dims
        .iter()
        .rev()
        .map(|&d| {
            if d == 0 {
                return Err(anyhow::anyhow!(
                    "qwen_image tensor '{}' has a zero dimension",
                    t.name
                ));
            }
            usize::try_from(d).map_err(|_| {
                anyhow::anyhow!(
                    "qwen_image tensor '{}': dimension {d} overflows usize",
                    t.name
                )
            })
        })
        .collect::<anyhow::Result<_>>()?;
    let numel = shape.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d).ok_or_else(|| {
            anyhow::anyhow!("qwen_image tensor '{}': shape product overflows", t.name)
        })
    })?;
    Ok((shape, numel))
}

/// Import the Qwen Image diffusion transformer component. GGUF stores matrix
/// dimensions in ggml order (the fastest dimension first), so CMF receives
/// the reversed, framework-facing shape while retaining the source name.
/// Quantized tensors are decoded and re-encoded one at a time through the
/// existing native codecs; CmfStreamWriter keeps the 16.8 GB input plus output
/// bounded without a temporary whole-model copy.
fn run_import_qwen_image(
    g: &Gguf,
    source: &std::path::Path,
    source_spec: &str,
    quant: Quant,
    output: &str,
    mut progress: impl FnMut(f32),
) -> anyhow::Result<()> {
    if g.tensors.is_empty() {
        anyhow::bail!("qwen_image GGUF has no tensors");
    }
    let geometry = qwen_image_geometry(&g.tensors)?;
    let config = serde_json::to_vec(&qwen_image_config(&geometry))?;
    // Validate the complete directory before creating the output. A malformed
    // or unsupported source therefore cannot leave a partial CMF that looks
    // resumable or valid to a later caller.
    let mut seen = BTreeMap::new();
    if g.tensors.iter().any(|t| t.name == "image.config_json") {
        anyhow::bail!("qwen_image source already contains reserved tensor 'image.config_json'");
    }
    for t in &g.tensors {
        if seen.insert(&t.name, ()).is_some() {
            anyhow::bail!("qwen_image GGUF contains duplicate tensor '{}'", t.name);
        }
        let (_, numel) = qwen_image_shape(t)?;
        if !matches!(
            t.ggml_type,
            GGML_F32
                | GGML_F16
                | GGML_BF16
                | GGML_Q4_0
                | GGML_Q4_1
                | GGML_Q5_0
                | GGML_Q5_1
                | GGML_Q8_0
                | GGML_Q2_K
                | GGML_Q3_K
                | GGML_Q4_K
                | GGML_Q5_K
                | GGML_Q6_K
                | GGML_Q8_K
                | GGML_IQ4_NL
                | GGML_IQ4_XS
        ) {
            anyhow::bail!(
                "qwen_image tensor '{}' uses ggml type {} with no native dequantizer",
                t.name,
                t.ggml_type
            );
        }
        let _ = qwen_image_raw(g, t, numel)?;
    }
    let total = g.tensors.len() + 1;
    let avg_name = g
        .tensors
        .iter()
        .map(|t| t.name.len())
        .sum::<usize>()
        .checked_div(g.tensors.len().max(1))
        .unwrap_or(64)
        .max(32);
    let gap = CmfStreamWriter::head_reserve_for(total, avg_name);
    let output_path = std::path::Path::new(output);
    let temp_path = qwen_image_temp_path(output_path)?;
    let conversion: anyhow::Result<()> = (|| {
        let mut writer = CmfStreamWriter::new(&temp_path, gap)
            .map_err(|e| anyhow::anyhow!("create streamed CMF {output}: {e}"))?;
        writer
            .push(
                "image.config_json",
                TensorDtype::U8,
                &[config.len()],
                &config,
            )
            .map_err(|e| anyhow::anyhow!("write image config: {e}"))?;
        progress(1.0 / total as f32);

        for (idx, t) in g.tensors.iter().enumerate() {
            let (shape, numel) = qwen_image_shape(t)?;
            let raw = qwen_image_raw(g, t, numel)?;
            let (dtype, data) = match t.ggml_type {
                // Controls and root matrices in the real Qwen Image GGUF use
                // these native dtypes; retain their bytes exactly.
                GGML_F32 => (TensorDtype::F32, raw.to_vec()),
                GGML_F16 => (TensorDtype::F16, raw.to_vec()),
                GGML_BF16 => (TensorDtype::Bf16, raw.to_vec()),
                // Every other type already supported by dequant() follows the
                // ordinary CMF requantization path, including Q4/Q5/Q6/Q8 K
                // blocks and IQ4 codebooks.
                _ => {
                    let vals = dequant(t.ggml_type, raw, numel)?;
                    if shape.len() == 2 {
                        convert::quantize_2d(quant, &vals, shape[0], shape[1])
                    } else {
                        (TensorDtype::F16, convert::encode_f16(&vals))
                    }
                }
            };
            writer
                .push(&t.name, dtype, &shape, &data)
                .map_err(|e| anyhow::anyhow!("write qwen_image tensor '{}': {e}", t.name))?;
            progress((idx + 2) as f32 / total as f32);
        }

        let arch = qwen_image_arch(&geometry);
        let header = CmfHeader {
            format: "cmf".into(),
            version: CMF_VERSION,
            arch,
            quant_type: quant_type_for(quant),
            provenance: Some(serde_json::json!({
                "tool": "cortiq import-gguf",
                "source": source_spec,
                "source_path": source.display().to_string(),
                "source_arch": "qwen_image",
                "model_kind": "image",
                "component": "transformer",
                "component_class": "QwenImageTransformer2DModel",
                "pipeline_family": "Qwen Image",
                "artifact_scope": "transformer_only",
                "runnable_image_pipeline": false,
                "tensor_name_policy": "source_names_unchanged",
                "source_tensor_count": g.tensors.len(),
                "source_quantization_version": g
                    .md
                    .get("general.quantization_version")
                    .and_then(|v| v.as_u64()),
                "source_file_type": g
                    .md
                    .get("general.file_type")
                    .and_then(|v| v.as_u64()),
                "output_quant": convert::quant_name(quant),
                "external_components": {
                    "text_encoder": "Qwen2.5-VL family",
                    "tokenizer": "Qwen2 tokenizer/processor",
                    "vae": "AutoencoderKLQwenImage",
                    "scheduler": "FlowMatch Euler"
                }
            })),
            // The transformer GGUF has no tokenizer metadata. An image
            // component must not acquire a fabricated text tokenizer section.
            tokenizer_config: None,
            section_hashes: None,
            skills: Vec::new(),
            shard: None,
            calibration: None,
            routing: None,
        };
        writer
            .finish(&header, None, None)
            .map_err(|e| anyhow::anyhow!("finish streamed CMF {output}: {e}"))?;
        Ok(())
    })();
    if let Err(err) = conversion {
        let _ = fs::remove_file(&temp_path);
        return Err(err);
    }
    if let Err(err) = fs::rename(&temp_path, output_path) {
        let _ = fs::remove_file(&temp_path);
        return Err(anyhow::anyhow!(
            "atomically install streamed CMF {output}: {err}"
        ));
    }
    qwen_image_sync_parent(output_path)?;
    progress(1.0);
    Ok(())
}

pub fn run_import_gguf(
    gguf: &str,
    quant: &str,
    output: &str,
    hf_token: Option<&str>,
    mut progress: impl FnMut(f32),
) -> anyhow::Result<()> {
    let quant = convert::parse_quant(quant)?;
    // Source: a local .gguf, an HF repo id (auto-pick a .gguf), or owner/repo/file.gguf.
    let path = resolve_gguf_source(gguf, hf_token)?;
    let g = parse(&path)?;

    // Qwen Image is a diffusion transformer component, not an LLM. It has no
    // block_count metadata and must bypass the generic tokenizer/LLM mapper.
    if g.md
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .is_some_and(|arch| arch == "qwen_image")
    {
        return run_import_qwen_image(&g, &path, gguf, quant, output, progress);
    }

    let arch = arch_from_md(&g.md, &g.tensors)?;
    let is_llama = arch.arch_name == "llama";
    let is_q35 = arch.arch_name == "qwen3_5_moe";

    // Honest guard: the native GGUF importer maps standard transformer tensors
    // plus the qwen35moe GDN-hybrid layout. Any OTHER SSM family (Mamba,
    // plain qwen3_next) would silently lose its mixer tensors — refuse it
    // clearly instead of writing a broken model; the safetensors path works.
    if !is_q35 {
        if let Some(t) = g.tensors.iter().find(|t| t.name.contains("ssm")) {
            anyhow::bail!(
                "GGUF '{}' is a linear-attention / SSM hybrid (e.g. tensor '{}') — the native \
                 GGUF importer handles standard transformer layouts and qwen35moe only. Convert \
                 the model's safetensors repo with `cortiq convert` instead (GatedDeltaNet is \
                 supported there).",
                path.display(),
                t.name
            );
        }
    }

    let n_heads = arch.num_attention_heads;
    let n_kv = arch.num_kv_heads;

    let total = g.tensors.len().max(1);
    let mut tensors: Vec<TensorSpec> = Vec::with_capacity(total);
    for (idx, t) in g.tensors.iter().enumerate() {
        progress((idx + 1) as f32 / total as f32);
        // qwen35moe routed experts: one 3-D [n_exp, out, in] tensor per
        // projection — split into the per-expert 2-D matrices the engine
        // loads (`mlp.experts.E.{gate,up,down}_proj`).
        if is_q35 && t.dims.len() == 3 {
            let proj = if t.name.ends_with("ffn_gate_exps.weight") {
                Some("gate_proj")
            } else if t.name.ends_with("ffn_up_exps.weight") {
                Some("up_proj")
            } else if t.name.ends_with("ffn_down_exps.weight") {
                Some("down_proj")
            } else {
                None
            };
            if let (Some(proj), Some(layer)) = (
                proj,
                t.name
                    .strip_prefix("blk.")
                    .and_then(|r| r.split('.').next())
                    .map(String::from),
            ) {
                let numel: usize = t.dims.iter().map(|&d| d as usize).product();
                let nb = nbytes(t.ggml_type, numel)?;
                let raw = &g.bytes
                    [g.data_start + t.offset as usize..g.data_start + t.offset as usize + nb];
                let vals = dequant(t.ggml_type, raw, numel)?;
                let shape: Vec<usize> = t.dims.iter().rev().map(|&d| d as usize).collect();
                let (nexp, out, inn) = (shape[0], shape[1], shape[2]);
                for e in 0..nexp {
                    let sl = &vals[e * out * inn..(e + 1) * out * inn];
                    let (dt, data) = convert::quantize_2d(quant, sl, out, inn);
                    tensors.push(TensorSpec {
                        name: format!("model.layers.{layer}.mlp.experts.{e}.{proj}.weight"),
                        dtype: dt,
                        shape: vec![out, inn],
                        data,
                    });
                }
                continue;
            }
        }
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

        if is_q35 {
            // Undo llama.cpp's tiled V-head order on every V-indexed
            // tensor (see v_head_untile). Geometry from the arch: nk
            // K groups, nv V heads of dv dims, dk key dims.
            let nk = arch.linear_num_key_heads.unwrap_or(0);
            let nv = arch.linear_num_value_heads.unwrap_or(0);
            let dk = arch.linear_key_head_dim.unwrap_or(0);
            let dv = arch.linear_value_head_dim.unwrap_or(0);
            if nk > 0 && nv > nk {
                let hid = *shape.last().unwrap_or(&0);
                if name.ends_with("linear_attn.in_proj_qkv.weight") {
                    let voff = 2 * nk * dk * hid;
                    v_head_untile(&mut vals[voff..], nv, dv * hid, nk);
                } else if name.ends_with("linear_attn.in_proj_z.weight") {
                    v_head_untile(&mut vals, nv, dv * hid, nk);
                } else if name.ends_with("linear_attn.in_proj_a.weight")
                    || name.ends_with("linear_attn.in_proj_b.weight")
                {
                    v_head_untile(&mut vals, nv, hid, nk);
                } else if name.ends_with("linear_attn.A_log")
                    || name.ends_with("linear_attn.dt_bias")
                {
                    v_head_untile(&mut vals, nv, 1, nk);
                } else if name.ends_with("linear_attn.conv1d.weight") {
                    // channels [2·nk·dk | nv·dv], k taps each — untile
                    // the V channel blocks.
                    let k = *shape.last().unwrap_or(&1);
                    let coff = 2 * nk * dk * k;
                    v_head_untile(&mut vals[coff..], nv, dv * k, nk);
                } else if name.ends_with("linear_attn.out_proj.weight") {
                    // input columns are V-head-indexed.
                    v_head_untile_cols(&mut vals, shape[0], nv, dv, nk);
                }
            }
            // llama.cpp bakes the zero-centered (1+w) shift into the RMS
            // norm weights of gemma-style models; the engine adds the 1
            // itself. Detect by magnitude (shifted weights sit near 1,
            // raw near 0) so either convention imports right. The GDN
            // gated norm (linear_attn.norm) is EXCLUDED on both sides:
            // llama.cpp stores it raw, and its weights initialize at 1 —
            // the magnitude heuristic would mangle them (found the hard
            // way: it was the one broken tensor in the layer diff).
            if name.ends_with("layernorm.weight")
                || name.ends_with("_norm.weight")
                || name == "model.norm.weight"
            {
                let mean = vals.iter().sum::<f32>() / vals.len().max(1) as f32;
                if mean > 0.5 {
                    for v in vals.iter_mut() {
                        *v -= 1.0;
                    }
                }
            }
            // llama.cpp stores GDN decay as -exp(A_log) (the Mamba
            // convention); the engine wants raw A_log. All-negative
            // values mean the transform was applied.
            if name.ends_with("linear_attn.A_log") && vals.iter().all(|&v| v < 0.0) {
                for v in vals.iter_mut() {
                    *v = (-*v).ln();
                }
            }
        }

        // Precision-critical small tensors ride as raw f32: the GDN a/b
        // projections, the depthwise conv taps and the MoE routers (a
        // bit-flip there is costly — same policy as the safetensors
        // converter).
        let keep_f32 = is_q35
            && (name.ends_with("mlp.gate.weight")
                || name.contains("linear_attn.in_proj_a")
                || name.contains("linear_attn.in_proj_b")
                || name.contains("conv1d")
                || name.ends_with("shared_expert_gate.weight"));
        // The shared-expert sigmoid gate is a 1-output Linear — the
        // engine loads it as a matrix, so carry it as [1, hidden].
        let shape = if name.ends_with("shared_expert_gate.weight") && shape.len() == 1 {
            vec![1, shape[0]]
        } else {
            shape
        };
        let two_d = shape.len() == 2 && numel >= 32 && !keep_f32;
        let (dt, data) = if two_d {
            convert::quantize_2d(quant, &vals, shape[0], shape[1])
        } else if keep_f32 {
            (
                TensorDtype::F32,
                vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
            )
        } else {
            (TensorDtype::F16, convert::encode_f16(&vals))
        };
        tensors.push(TensorSpec {
            name,
            dtype: dt,
            shape,
            data,
        });
    }

    let (vocab, bundle) = tokenizer(&g.md);
    let quant_type = quant_type_for(quant);
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
        routing: None,
    };
    CmfModel::write(output, &header, &tensors, None, vocab.as_deref())
        .map_err(|e| anyhow::anyhow!("write {output}: {e}"))?;
    progress(1.0);
    Ok(())
}

/// llama.cpp stores qwen35moe V-head-indexed blocks in "tiled" order
/// (v-within-group major: block v·nk + g); the HF layout the engine
/// runs is grouped by K head (block g·r + v). Pure block permutation.
fn v_head_untile(vals: &mut [f32], nblk: usize, blk: usize, nk: usize) {
    if nk == 0 || nblk % nk != 0 || vals.len() != nblk * blk {
        return;
    }
    let r = nblk / nk;
    let src = vals.to_vec();
    for g in 0..nk {
        for v in 0..r {
            let dst = (g * r + v) * blk;
            let s = (v * nk + g) * blk;
            vals[dst..dst + blk].copy_from_slice(&src[s..s + blk]);
        }
    }
}

/// The same untile applied to COLUMN blocks of every row (out_proj's
/// input dimension is V-head-indexed).
fn v_head_untile_cols(vals: &mut [f32], rows: usize, nblk: usize, blk: usize, nk: usize) {
    if nk == 0 || nblk % nk != 0 || vals.len() != rows * nblk * blk {
        return;
    }
    let r = nblk / nk;
    let stride = nblk * blk;
    let src = vals.to_vec();
    for row in 0..rows {
        for g in 0..nk {
            for v in 0..r {
                let dst = row * stride + (g * r + v) * blk;
                let s = row * stride + (v * nk + g) * blk;
                vals[dst..dst + blk].copy_from_slice(&src[s..s + blk]);
            }
        }
    }
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

    /// tiled block v·nk+g → grouped g·r+v, and the column variant.
    #[test]
    fn v_head_untile_roundtrip() {
        let (nk, r, blk) = (3usize, 2usize, 4usize);
        let nv = nk * r;
        // grouped ground truth: block id = g·r+v encoded in the values
        let grouped: Vec<f32> = (0..nv * blk).map(|i| (i / blk) as f32).collect();
        // build the tiled layout llama.cpp stores
        let mut tiled = vec![0f32; nv * blk];
        for g in 0..nk {
            for v in 0..r {
                let t = (v * nk + g) * blk;
                let s = (g * r + v) * blk;
                tiled[t..t + blk].copy_from_slice(&grouped[s..s + blk]);
            }
        }
        let mut got = tiled.clone();
        v_head_untile(&mut got, nv, blk, nk);
        assert_eq!(got, grouped);

        // column variant: 2 rows of the same block structure
        let rows = 2usize;
        let mut tiled2 = Vec::new();
        for _ in 0..rows {
            tiled2.extend_from_slice(&tiled);
        }
        let mut got2 = tiled2.clone();
        v_head_untile_cols(&mut got2, rows, nv, blk, nk);
        for row in 0..rows {
            assert_eq!(&got2[row * nv * blk..(row + 1) * nv * blk], &grouped[..]);
        }
    }
    fn unhex(h: &str) -> Vec<u8> {
        (0..h.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&h[i..i + 2], 16).unwrap())
            .collect()
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
            assert!(
                (out[i] - e).abs() < 2e-4,
                "idx {i}: got {} want {}",
                out[i],
                e
            );
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
        for &(i, e) in &[
            (0usize, 0.002592f32),
            (1, -0.002525),
            (2, -0.0102),
            (31, 3.3e-5),
            (32, 0.000751),
            (64, -0.004447),
            (128, -0.003603),
            (200, -0.004132),
            (255, 0.002143),
        ] {
            assert!(
                (out[i] - e).abs() < 2e-4,
                "q4k idx {i}: got {} want {}",
                out[i],
                e
            );
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
        for &(i, e) in &[
            (0usize, 0.003006f32),
            (1, -0.003211),
            (2, -0.01005),
            (31, -0.000102),
            (32, 0.000413),
            (64, -0.004699),
            (128, -0.003084),
            (200, -0.003904),
            (255, 0.002199),
        ] {
            assert!(
                (out[i] - e).abs() < 2e-4,
                "q5k idx {i}: got {} want {}",
                out[i],
                e
            );
        }
    }

    fn put_gstr(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    fn test_qwen_image_gguf() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
        // The payloads use the same GGML ids and block layout as the real
        // Qwen Image file: F32 controls, BF16 root matrices, and quantized
        // projection weights. The K blocks are deliberately nonzero so the
        // output check exercises dequantization rather than only shape paths.
        let f32_bias: Vec<u8> = (0..256usize)
            .flat_map(|i| (i as f32 * 0.25 - 0.5).to_le_bytes())
            .collect();
        let img_in_weight: Vec<u8> = (0..(256usize * 64))
            .flat_map(|i| {
                let bits = (0x3f80u16).wrapping_add((i as u16) & 7);
                bits.to_le_bytes()
            })
            .collect();
        let txt_in_weight: Vec<u8> = (0..(256usize * 3584))
            .flat_map(|i| {
                let bits = (0x3f80u16).wrapping_add((i as u16) & 7);
                bits.to_le_bytes()
            })
            .collect();
        let proj_out_weight: Vec<u8> = (0..(64usize * 256))
            .flat_map(|i| {
                let bits = (0x3f80u16).wrapping_add((i as u16) & 7);
                bits.to_le_bytes()
            })
            .collect();
        let mut q6 = vec![0u8; 1024 * 210]; // [1024, 256] in CMF after dim reversal
        for block in q6.chunks_exact_mut(210) {
            block[0] = 0x0f;
            block[32] = 0xf0;
            block[192] = 1; // first sub-scale
            block[208..210].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1
        }
        let mut q4 = vec![0u8; 2048 * 18]; // [256, 256] in CMF after dim reversal
        for block in q4.chunks_exact_mut(18) {
            block[..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1
            block[2..].fill(0x88); // centered zero nibbles
        }
        let norm: Vec<u8> = (0..128usize).flat_map(|_| (1.0f32).to_le_bytes()).collect();
        let defs = vec![
            ("img_in.bias", vec![256u64], GGML_F32, f32_bias.clone()),
            (
                "img_in.weight",
                vec![64, 256],
                GGML_BF16,
                img_in_weight.clone(),
            ),
            ("txt_in.weight", vec![3584, 256], GGML_BF16, txt_in_weight),
            ("proj_out.weight", vec![256, 64], GGML_BF16, proj_out_weight),
            (
                "transformer_blocks.0.img_mlp.net.0.proj.weight",
                vec![256, 1024],
                GGML_Q6_K,
                q6.clone(),
            ),
            (
                "transformer_blocks.0.attn.to_q.weight",
                vec![256, 256],
                GGML_Q4_0,
                q4,
            ),
            (
                "transformer_blocks.0.attn.norm_q.weight",
                vec![128],
                GGML_F32,
                norm,
            ),
        ];
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(defs.len() as u64).to_le_bytes());
        out.extend_from_slice(&3u64.to_le_bytes());
        put_gstr(&mut out, "general.architecture");
        out.extend_from_slice(&T_STR.to_le_bytes());
        put_gstr(&mut out, "qwen_image");
        put_gstr(&mut out, "general.quantization_version");
        out.extend_from_slice(&T_U32.to_le_bytes());
        out.extend_from_slice(&2u32.to_le_bytes());
        put_gstr(&mut out, "general.file_type");
        out.extend_from_slice(&T_U32.to_le_bytes());
        out.extend_from_slice(&18u32.to_le_bytes());
        let mut rel = 0usize;
        let mut offsets = Vec::with_capacity(defs.len());
        for (name, dims, ggml_type, data) in &defs {
            rel = align_up(rel, 32);
            offsets.push(rel);
            put_gstr(&mut out, name);
            out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for &d in dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&ggml_type.to_le_bytes());
            out.extend_from_slice(&(rel as u64).to_le_bytes());
            rel += data.len();
        }
        let data_start = align_up(out.len(), 32);
        out.resize(data_start, 0);
        for ((_, _, _, data), &offset) in defs.iter().zip(&offsets) {
            let start = data_start + offset;
            if out.len() < start {
                out.resize(start, 0);
            }
            out.extend_from_slice(data);
        }
        (out, q6, f32_bias, img_in_weight)
    }

    #[test]
    fn qwen_image_import_preserves_controls_and_requantizes_q6k() {
        let (gguf_bytes, q6_raw, f32_bias, img_in_weight) = test_qwen_image_gguf();
        let id = format!(
            "cortiq-qwen-image-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let src = std::env::temp_dir().join(format!("{id}.gguf"));
        let out = std::env::temp_dir().join(format!("{id}.cmf"));
        std::fs::write(&src, gguf_bytes).unwrap();
        run_import_gguf(
            src.to_str().unwrap(),
            "q8",
            out.to_str().unwrap(),
            None,
            |_| {},
        )
        .unwrap();

        let model = cortiq_core::CmfModel::open(&out).unwrap();
        let errors = model.verify();
        assert!(errors.is_empty(), "CMF verification errors: {errors:?}");
        assert_eq!(model.tensors.len(), 8); // seven source tensors + config
        assert_eq!(model.header.arch.arch_name, "qwen_image");
        assert_eq!(model.header.arch.hidden_size, 256);
        assert_eq!(model.header.arch.intermediate_size, 1024);
        assert_eq!(model.header.arch.num_layers, 1);
        assert_eq!(model.header.arch.num_attention_heads, 2);
        assert_eq!(model.header.arch.head_dim, 128);
        assert_eq!(model.header.arch.hidden_act, "gelu_tanh");
        let provenance = model.header.provenance.as_ref().unwrap();
        assert_eq!(provenance["model_kind"], "image");
        assert_eq!(provenance["component"], "transformer");
        assert_eq!(provenance["artifact_scope"], "transformer_only");
        assert_eq!(provenance["runnable_image_pipeline"], false);
        assert_eq!(provenance["tensor_name_policy"], "source_names_unchanged");
        assert_eq!(provenance["source_tensor_count"], 7);

        let config = model.tensor("image.config_json").unwrap();
        assert_eq!(config.dtype, TensorDtype::U8);
        let config_json: serde_json::Value =
            serde_json::from_slice(model.entry_bytes(config)).unwrap();
        assert_eq!(config_json["_class_name"], "QwenImageTransformer2DModel");
        assert_eq!(
            config_json["axes_dims_rope"],
            serde_json::json!([16, 56, 56])
        );
        assert_eq!(config_json["attention_head_dim"], 128);
        assert_eq!(config_json["num_attention_heads"], 2);
        assert_eq!(config_json["num_layers"], 1);
        assert_eq!(config_json["in_channels"], 64);
        assert_eq!(config_json["joint_attention_dim"], 3584);
        assert_eq!(config_json["out_channels"], 16);

        let bias = model.tensor("img_in.bias").unwrap();
        assert_eq!(bias.dtype, TensorDtype::F32);
        assert_eq!(bias.shape, vec![256]);
        let bias_bytes = model.entry_bytes(bias);
        assert_eq!(bias_bytes, f32_bias.as_slice());

        let root_weight = model.tensor("img_in.weight").unwrap();
        assert_eq!(root_weight.dtype, TensorDtype::Bf16);
        assert_eq!(root_weight.shape, vec![256, 64]);
        assert_eq!(root_weight.n_elems(), 256 * 64);
        assert_eq!(model.entry_bytes(root_weight), img_in_weight.as_slice());

        let txt_weight = model.tensor("txt_in.weight").unwrap();
        assert_eq!(txt_weight.dtype, TensorDtype::Bf16);
        assert_eq!(txt_weight.shape, vec![256, 3584]);

        let proj_out = model.tensor("proj_out.weight").unwrap();
        assert_eq!(proj_out.dtype, TensorDtype::Bf16);
        assert_eq!(proj_out.shape, vec![64, 256]);

        let q6 = model
            .tensor("transformer_blocks.0.img_mlp.net.0.proj.weight")
            .unwrap();
        assert_eq!(q6.dtype, TensorDtype::Q8Row);
        assert_eq!(q6.shape, vec![1024, 256]);
        let mut decoded = vec![0.0f32; q6.n_elems()];
        cortiq_core::quant::dequant_tensor(q6, model.entry_bytes(q6), &mut decoded).unwrap();
        let expected = dequant_q6_k(&q6_raw, q6.n_elems());
        assert!(decoded.iter().all(|v| v.is_finite()));
        assert!(decoded.iter().any(|v| v.abs() > 1.0));
        let max_err = decoded
            .iter()
            .zip(expected)
            .map(|(got, want)| (got - want).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.25, "Q6_K→Q8 max error {max_err}");

        let q4 = model
            .tensor("transformer_blocks.0.attn.to_q.weight")
            .unwrap();
        assert_eq!(q4.dtype, TensorDtype::Q8Row);
        assert_eq!(q4.shape, vec![256, 256]);
        let mut q4_decoded = vec![0.0f32; q4.n_elems()];
        cortiq_core::quant::dequant_tensor(q4, model.entry_bytes(q4), &mut q4_decoded).unwrap();
        assert!(q4_decoded.iter().all(|v| v.is_finite()));

        drop(model);
        let _ = std::fs::remove_file(src);
        let _ = std::fs::remove_file(out);
    }

    #[test]
    fn qwen_image_import_is_atomic_for_alias_and_failure() {
        let (gguf_bytes, _, _, _) = test_qwen_image_gguf();
        let id = format!(
            "cortiq-qwen-image-atomic-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let source = std::env::temp_dir().join(format!("{id}.gguf"));
        std::fs::write(&source, &gguf_bytes).unwrap();
        // The output aliases the mmap'ed input. The temporary sibling and
        // final rename keep the input mapping valid until all source reads are
        // complete, then atomically install the CMF in its place.
        run_import_gguf(
            source.to_str().unwrap(),
            "q8",
            source.to_str().unwrap(),
            None,
            |_| {},
        )
        .unwrap();
        let model = cortiq_core::CmfModel::open(&source).unwrap();
        assert!(model.verify().is_empty());
        assert_eq!(model.header.arch.hidden_size, 256);
        drop(model);

        // A pre-existing output survives a malformed source: preflight fails
        // before any temporary writer is created.
        let bad_source = std::env::temp_dir().join(format!("{id}.bad.gguf"));
        let bad_output = std::env::temp_dir().join(format!("{id}.old.cmf"));
        let mut truncated = gguf_bytes;
        truncated.truncate(truncated.len().saturating_sub(1));
        std::fs::write(&bad_source, truncated).unwrap();
        std::fs::write(&bad_output, b"previous-valid-artifact").unwrap();
        assert!(
            run_import_gguf(
                bad_source.to_str().unwrap(),
                "q8",
                bad_output.to_str().unwrap(),
                None,
                |_| {},
            )
            .is_err()
        );
        assert_eq!(
            std::fs::read(&bad_output).unwrap(),
            b"previous-valid-artifact"
        );

        let _ = std::fs::remove_file(source);
        let _ = std::fs::remove_file(bad_source);
        let _ = std::fs::remove_file(bad_output);
    }
}
