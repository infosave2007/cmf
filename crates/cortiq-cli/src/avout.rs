//! Writing the render out: a baseline JPEG encoder, a RIFF/AVI muxer
//! and a WAV writer, so `cortiq animate` produces a file that plays
//! with sound and nothing has to be installed to get it.
//!
//! MJPEG in AVI is the one container-and-codec pair that is small
//! enough to write honestly in a few hundred lines and still opens in
//! every player: intra-only frames, no motion estimation, no bitstream
//! conformance beyond Annex K's own tables. The alternative was
//! shelling out to ffmpeg, which is precisely the dependency this
//! project exists to not have.

use std::io::Write;

// ── baseline JPEG ───────────────────────────────────────────────────

#[rustfmt::skip]
const ZIGZAG: [usize; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

#[rustfmt::skip]
const QUANT_LUMA: [u8; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68,109,103, 77,
    24, 35, 55, 64, 81,104,113, 92,
    49, 64, 78, 87,103,121,120,101,
    72, 92, 95, 98,112,100,103, 99,
];

#[rustfmt::skip]
const QUANT_CHROMA: [u8; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
];

const DC_LUMA_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
const DC_CHROMA_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
const DC_VALS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

const AC_LUMA_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d];
#[rustfmt::skip]
const AC_LUMA_VALS: [u8; 162] = [
    0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,0x07,
    0x22,0x71,0x14,0x32,0x81,0x91,0xa1,0x08,0x23,0x42,0xb1,0xc1,0x15,0x52,0xd1,0xf0,
    0x24,0x33,0x62,0x72,0x82,0x09,0x0a,0x16,0x17,0x18,0x19,0x1a,0x25,0x26,0x27,0x28,
    0x29,0x2a,0x34,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,
    0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,0x69,
    0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x83,0x84,0x85,0x86,0x87,0x88,0x89,
    0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,
    0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,
    0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe1,0xe2,
    0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,
    0xf9,0xfa,
];

const AC_CHROMA_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77];
#[rustfmt::skip]
const AC_CHROMA_VALS: [u8; 162] = [
    0x00,0x01,0x02,0x03,0x11,0x04,0x05,0x21,0x31,0x06,0x12,0x41,0x51,0x07,0x61,0x71,
    0x13,0x22,0x32,0x81,0x08,0x14,0x42,0x91,0xa1,0xb1,0xc1,0x09,0x23,0x33,0x52,0xf0,
    0x15,0x62,0x72,0xd1,0x0a,0x16,0x24,0x34,0xe1,0x25,0xf1,0x17,0x18,0x19,0x1a,0x26,
    0x27,0x28,0x29,0x2a,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,
    0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,
    0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x82,0x83,0x84,0x85,0x86,0x87,
    0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,
    0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,
    0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,
    0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,
    0xf9,0xfa,
];

/// (code, length) per symbol value, built from a canonical bits/vals pair.
fn huff_table(bits: &[u8; 16], vals: &[u8]) -> Vec<(u16, u8)> {
    let mut out = vec![(0u16, 0u8); 256];
    let mut code = 0u16;
    let mut k = 0usize;
    for (i, &n) in bits.iter().enumerate() {
        for _ in 0..n {
            out[vals[k] as usize] = (code, i as u8 + 1);
            code += 1;
            k += 1;
        }
        code <<= 1;
    }
    out
}

/// Quantization tables at `quality` (1..100), the libjpeg scaling.
fn scaled(q: &[u8; 64], quality: u32) -> [u8; 64] {
    let s = if quality < 50 {
        5000 / quality.max(1)
    } else {
        200 - quality.min(100) * 2
    };
    let mut out = [0u8; 64];
    for i in 0..64 {
        out[i] = (((q[i] as u32 * s) + 50) / 100).clamp(1, 255) as u8;
    }
    out
}

/// The AAN-free straightforward separable FDCT. A frame is 8×8 blocks
/// and there are a few thousand of them; this is not the bottleneck
/// next to a 36-layer decoder.
fn fdct(block: &mut [f32; 64]) {
    let mut tmp = [0f32; 64];
    let c = |u: usize| {
        if u == 0 {
            std::f32::consts::FRAC_1_SQRT_2
        } else {
            1.0
        }
    };
    let mut cos_t = [[0f32; 8]; 8];
    for (x, row) in cos_t.iter_mut().enumerate() {
        for (u, v) in row.iter_mut().enumerate() {
            *v = (((2 * x + 1) as f32) * u as f32 * std::f32::consts::PI / 16.0).cos();
        }
    }
    for y in 0..8 {
        for u in 0..8 {
            let mut s = 0f32;
            for x in 0..8 {
                s += block[y * 8 + x] * cos_t[x][u];
            }
            tmp[y * 8 + u] = 0.5 * c(u) * s;
        }
    }
    for u in 0..8 {
        for v in 0..8 {
            let mut s = 0f32;
            for y in 0..8 {
                s += tmp[y * 8 + u] * cos_t[y][v];
            }
            block[v * 8 + u] = 0.5 * c(v) * s;
        }
    }
}

struct BitWriter {
    out: Vec<u8>,
    acc: u32,
    n: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            acc: 0,
            n: 0,
        }
    }
    fn put(&mut self, code: u16, len: u8) {
        self.acc = (self.acc << len) | code as u32;
        self.n += len as u32;
        while self.n >= 8 {
            self.n -= 8;
            let b = ((self.acc >> self.n) & 0xFF) as u8;
            self.out.push(b);
            // 0xFF in entropy-coded data is escaped with a zero byte.
            if b == 0xFF {
                self.out.push(0);
            }
        }
    }
    fn flush(&mut self) {
        if self.n > 0 {
            let pad = 8 - self.n;
            self.put((1u16 << pad) - 1, pad as u8);
        }
    }
}

/// Magnitude category and the value's low bits, as the JPEG spec's
/// `EXTEND` expects them.
fn magnitude(v: i32) -> (u8, u16) {
    let a = v.unsigned_abs();
    let mut s = 0u8;
    while (1u32 << s) <= a {
        s += 1;
    }
    let bits = if v < 0 { (v - 1) & ((1 << s) - 1) } else { v } as u16;
    (s, bits)
}

/// One RGB frame → a complete JFIF byte stream, 4:4:4.
pub fn encode_jpeg(rgb: &[f32], h: usize, w: usize, quality: u32) -> Vec<u8> {
    let ql = scaled(&QUANT_LUMA, quality);
    let qc = scaled(&QUANT_CHROMA, quality);
    let dc_l = huff_table(&DC_LUMA_BITS, &DC_VALS);
    let ac_l = huff_table(&AC_LUMA_BITS, &AC_LUMA_VALS);
    let dc_c = huff_table(&DC_CHROMA_BITS, &DC_VALS);
    let ac_c = huff_table(&AC_CHROMA_BITS, &AC_CHROMA_VALS);

    let mut f = Vec::new();
    f.extend_from_slice(&[0xFF, 0xD8]); // SOI
    // APP0/JFIF — MJPEG-in-AVI decoders expect it.
    f.extend_from_slice(&[0xFF, 0xE0, 0, 16]);
    f.extend_from_slice(b"JFIF\0");
    f.extend_from_slice(&[1, 1, 0, 0, 1, 0, 1, 0, 0]);
    for (id, q) in [(0u8, &ql), (1u8, &qc)] {
        f.extend_from_slice(&[0xFF, 0xDB, 0, 67, id]);
        for &z in ZIGZAG.iter() {
            f.push(q[z]);
        }
    }
    // SOF0, three components, all 1×1 sampled (4:4:4).
    f.extend_from_slice(&[0xFF, 0xC0, 0, 17, 8]);
    f.extend_from_slice(&(h as u16).to_be_bytes());
    f.extend_from_slice(&(w as u16).to_be_bytes());
    f.push(3);
    for (i, tq) in [(1u8, 0u8), (2, 1), (3, 1)] {
        f.extend_from_slice(&[i, 0x11, tq]);
    }
    for (class_id, bits, vals) in [
        (0x00u8, &DC_LUMA_BITS, &DC_VALS[..]),
        (0x10, &AC_LUMA_BITS, &AC_LUMA_VALS[..]),
        (0x01, &DC_CHROMA_BITS, &DC_VALS[..]),
        (0x11, &AC_CHROMA_BITS, &AC_CHROMA_VALS[..]),
    ] {
        let len = 2 + 1 + 16 + vals.len();
        f.extend_from_slice(&[0xFF, 0xC4]);
        f.extend_from_slice(&(len as u16).to_be_bytes());
        f.push(class_id);
        f.extend_from_slice(bits);
        f.extend_from_slice(vals);
    }
    f.extend_from_slice(&[0xFF, 0xDA, 0, 12, 3]);
    for (i, t) in [(1u8, 0x00u8), (2, 0x11), (3, 0x11)] {
        f.extend_from_slice(&[i, t]);
    }
    f.extend_from_slice(&[0, 63, 0]);

    let mut bw = BitWriter::new();
    let mut prev = [0i32; 3];
    let (mbh, mbw) = (h.div_ceil(8), w.div_ceil(8));
    let plane = h * w;
    for by in 0..mbh {
        for bx in 0..mbw {
            for comp in 0..3 {
                let mut blk = [0f32; 64];
                for y in 0..8 {
                    for x in 0..8 {
                        let sy = (by * 8 + y).min(h - 1);
                        let sx = (bx * 8 + x).min(w - 1);
                        let i = sy * w + sx;
                        let (r, g, b) = (
                            rgb[i] * 255.0,
                            rgb[plane + i] * 255.0,
                            rgb[2 * plane + i] * 255.0,
                        );
                        blk[y * 8 + x] = match comp {
                            0 => 0.299 * r + 0.587 * g + 0.114 * b - 128.0,
                            1 => -0.168_736 * r - 0.331_264 * g + 0.5 * b,
                            _ => 0.5 * r - 0.418_688 * g - 0.081_312 * b,
                        };
                    }
                }
                fdct(&mut blk);
                let (q, dct, act) = if comp == 0 {
                    (&ql, &dc_l, &ac_l)
                } else {
                    (&qc, &dc_c, &ac_c)
                };
                let mut zz = [0i32; 64];
                for (k, &z) in ZIGZAG.iter().enumerate() {
                    zz[k] = (blk[z] / q[z] as f32).round() as i32;
                }
                let diff = zz[0] - prev[comp];
                prev[comp] = zz[0];
                let (s, bits) = magnitude(diff);
                let (c, l) = dct[s as usize];
                bw.put(c, l);
                if s > 0 {
                    bw.put(bits & ((1 << s) - 1), s);
                }
                let mut run = 0u8;
                for k in 1..64 {
                    if zz[k] == 0 {
                        run += 1;
                        continue;
                    }
                    while run > 15 {
                        let (c, l) = act[0xF0];
                        bw.put(c, l);
                        run -= 16;
                    }
                    let (s, bits) = magnitude(zz[k]);
                    let (c, l) = act[(run << 4 | s) as usize];
                    bw.put(c, l);
                    bw.put(bits & ((1 << s) - 1), s);
                    run = 0;
                }
                if run > 0 {
                    let (c, l) = act[0x00];
                    bw.put(c, l);
                }
            }
        }
    }
    bw.flush();
    f.extend_from_slice(&bw.out);
    f.extend_from_slice(&[0xFF, 0xD9]); // EOI
    f
}

// ── WAV ─────────────────────────────────────────────────────────────

/// 16-bit stereo PCM. `audio` is `[2, n]` in [-1, 1].
pub fn wav_bytes(audio: &[f32], n: usize, rate: usize) -> Vec<u8> {
    let mut pcm = Vec::with_capacity(n * 4);
    for i in 0..n {
        for ch in 0..2 {
            let v = (audio[ch * n + i].clamp(-1.0, 1.0) * 32767.0).round() as i16;
            pcm.extend_from_slice(&v.to_le_bytes());
        }
    }
    let mut out = Vec::with_capacity(pcm.len() + 44);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + pcm.len()) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&(rate as u32).to_le_bytes());
    out.extend_from_slice(&((rate * 4) as u32).to_le_bytes());
    out.extend_from_slice(&4u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
    out.extend_from_slice(&pcm);
    out
}

// ── AVI ─────────────────────────────────────────────────────────────

fn list(tag: &[u8; 4], body: Vec<u8>) -> Vec<u8> {
    let mut v = Vec::with_capacity(body.len() + 12);
    v.extend_from_slice(b"LIST");
    v.extend_from_slice(&((body.len() + 4) as u32).to_le_bytes());
    v.extend_from_slice(tag);
    v.extend_from_slice(&body);
    v
}

fn chunk(tag: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(body.len() + 9);
    v.extend_from_slice(tag);
    v.extend_from_slice(&(body.len() as u32).to_le_bytes());
    v.extend_from_slice(body);
    if body.len() % 2 == 1 {
        v.push(0); // RIFF chunks are word-aligned
    }
    v
}

/// MJPEG video plus 16-bit stereo PCM, interleaved one audio packet per
/// frame so a player can stream it.
pub fn write_avi(
    path: &std::path::Path,
    frames: &[Vec<u8>],
    w: usize,
    h: usize,
    fps: usize,
    audio: &[f32],
    samples: usize,
    rate: usize,
) -> std::io::Result<()> {
    let n = frames.len();
    let per_frame = if n > 0 { samples.div_ceil(n) } else { 0 };

    let mut movi = Vec::new();
    let mut index: Vec<(u32, u32, u32)> = Vec::new(); // (fourcc-id, offset, size)
    for (i, f) in frames.iter().enumerate() {
        index.push((0, movi.len() as u32 + 4, f.len() as u32));
        movi.extend_from_slice(&chunk(b"00dc", f));
        let a = i * per_frame;
        let b = ((i + 1) * per_frame).min(samples);
        let mut pcm = Vec::with_capacity((b.saturating_sub(a)) * 4);
        for s in a..b {
            for ch in 0..2 {
                let v = (audio[ch * samples + s].clamp(-1.0, 1.0) * 32767.0).round() as i16;
                pcm.extend_from_slice(&v.to_le_bytes());
            }
        }
        index.push((1, movi.len() as u32 + 4, pcm.len() as u32));
        movi.extend_from_slice(&chunk(b"01wb", &pcm));
    }

    let mut avih = Vec::new();
    avih.extend_from_slice(&((1_000_000 / fps) as u32).to_le_bytes()); // µs per frame
    avih.extend_from_slice(&0u32.to_le_bytes()); // max bytes/sec
    avih.extend_from_slice(&0u32.to_le_bytes()); // padding granularity
    avih.extend_from_slice(&0x10u32.to_le_bytes()); // AVIF_HASINDEX
    avih.extend_from_slice(&(n as u32).to_le_bytes());
    avih.extend_from_slice(&0u32.to_le_bytes());
    avih.extend_from_slice(&2u32.to_le_bytes()); // streams
    avih.extend_from_slice(&0u32.to_le_bytes());
    avih.extend_from_slice(&(w as u32).to_le_bytes());
    avih.extend_from_slice(&(h as u32).to_le_bytes());
    avih.extend_from_slice(&[0u8; 16]);

    let mut vstrh = Vec::new();
    vstrh.extend_from_slice(b"vidsMJPG");
    vstrh.extend_from_slice(&[0u8; 12]);
    vstrh.extend_from_slice(&1u32.to_le_bytes()); // scale
    vstrh.extend_from_slice(&(fps as u32).to_le_bytes()); // rate
    vstrh.extend_from_slice(&0u32.to_le_bytes());
    vstrh.extend_from_slice(&(n as u32).to_le_bytes());
    vstrh.extend_from_slice(&0u32.to_le_bytes());
    vstrh.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
    vstrh.extend_from_slice(&0u32.to_le_bytes());
    vstrh.extend_from_slice(&[0u8; 8]);

    let mut vstrf = Vec::new();
    vstrf.extend_from_slice(&40u32.to_le_bytes());
    vstrf.extend_from_slice(&(w as u32).to_le_bytes());
    vstrf.extend_from_slice(&(h as u32).to_le_bytes());
    vstrf.extend_from_slice(&1u16.to_le_bytes());
    vstrf.extend_from_slice(&24u16.to_le_bytes());
    vstrf.extend_from_slice(b"MJPG");
    vstrf.extend_from_slice(&((w * h * 3) as u32).to_le_bytes());
    vstrf.extend_from_slice(&[0u8; 16]);

    let mut astrh = Vec::new();
    astrh.extend_from_slice(b"auds\0\0\0\0");
    astrh.extend_from_slice(&[0u8; 12]);
    astrh.extend_from_slice(&4u32.to_le_bytes()); // scale = block align
    astrh.extend_from_slice(&((rate * 4) as u32).to_le_bytes());
    astrh.extend_from_slice(&0u32.to_le_bytes());
    astrh.extend_from_slice(&(samples as u32).to_le_bytes());
    astrh.extend_from_slice(&0u32.to_le_bytes());
    astrh.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
    astrh.extend_from_slice(&0u32.to_le_bytes());
    astrh.extend_from_slice(&[0u8; 8]);

    let mut astrf = Vec::new();
    astrf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    astrf.extend_from_slice(&2u16.to_le_bytes());
    astrf.extend_from_slice(&(rate as u32).to_le_bytes());
    astrf.extend_from_slice(&((rate * 4) as u32).to_le_bytes());
    astrf.extend_from_slice(&4u16.to_le_bytes());
    astrf.extend_from_slice(&16u16.to_le_bytes());
    astrf.extend_from_slice(&0u16.to_le_bytes());

    let mut hdrl = chunk(b"avih", &avih);
    let mut vstrl = chunk(b"strh", &vstrh);
    vstrl.extend_from_slice(&chunk(b"strf", &vstrf));
    hdrl.extend_from_slice(&list(b"strl", vstrl));
    let mut astrl = chunk(b"strh", &astrh);
    astrl.extend_from_slice(&chunk(b"strf", &astrf));
    hdrl.extend_from_slice(&list(b"strl", astrl));

    let mut idx1 = Vec::with_capacity(index.len() * 16);
    for (id, off, size) in &index {
        idx1.extend_from_slice(if *id == 0 { b"00dc" } else { b"01wb" });
        idx1.extend_from_slice(&0x10u32.to_le_bytes()); // AVIIF_KEYFRAME
        idx1.extend_from_slice(&off.to_le_bytes());
        idx1.extend_from_slice(&size.to_le_bytes());
    }

    let mut body = Vec::new();
    body.extend_from_slice(&list(b"hdrl", hdrl));
    body.extend_from_slice(&list(b"movi", movi));
    body.extend_from_slice(&chunk(b"idx1", &idx1));

    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"RIFF")?;
    f.write_all(&((body.len() + 4) as u32).to_le_bytes())?;
    f.write_all(b"AVI ")?;
    f.write_all(&body)?;
    f.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grey ramp survives the round trip well enough that the file is
    /// a real JPEG: markers in place, entropy data escaped, and the
    /// stream terminated.
    #[test]
    fn jpeg_is_well_formed() {
        let (h, w) = (16usize, 24usize);
        let mut rgb = vec![0f32; 3 * h * w];
        for y in 0..h {
            for x in 0..w {
                let v = x as f32 / w as f32;
                for c in 0..3 {
                    rgb[c * h * w + y * w + x] = v;
                }
            }
        }
        let j = encode_jpeg(&rgb, h, w, 90);
        assert_eq!(&j[..2], &[0xFF, 0xD8]);
        assert_eq!(&j[j.len() - 2..], &[0xFF, 0xD9]);
        assert!(j.len() > 300, "suspiciously short: {}", j.len());
        // Every 0xFF inside the entropy segment must be followed by 0x00
        // or be a marker; find SOS and check the tail.
        let sos = j.windows(2).position(|p| p == [0xFF, 0xDA]).unwrap();
        let data = &j[sos + 14..j.len() - 2];
        for p in data.windows(2) {
            if p[0] == 0xFF {
                assert_eq!(p[1], 0x00, "unescaped 0xFF in the entropy stream");
            }
        }
    }

    #[test]
    fn huffman_tables_are_canonical() {
        let t = huff_table(&DC_LUMA_BITS, &DC_VALS);
        // Value 0 is the shortest code in the standard DC table.
        assert_eq!(t[0].1, 2);
        assert_eq!(t[0].0, 0);
        let ac = huff_table(&AC_LUMA_BITS, &AC_LUMA_VALS);
        assert_eq!(ac[0x01].1, 2); // (0,1) is the most common AC symbol
        assert!(ac[0xFA].1 >= 16);
    }

    #[test]
    fn wav_header_is_44_bytes_and_declares_its_payload() {
        let n = 100;
        let a = vec![0f32; 2 * n];
        let w = wav_bytes(&a, n, 32000);
        assert_eq!(w.len(), 44 + n * 4);
        assert_eq!(&w[..4], b"RIFF");
        assert_eq!(&w[8..12], b"WAVE");
        assert_eq!(
            u32::from_le_bytes(w[40..44].try_into().unwrap()) as usize,
            n * 4
        );
    }
}
