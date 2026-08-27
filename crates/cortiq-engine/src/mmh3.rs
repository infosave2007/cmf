//! MiniMax-H3: the packed-token audio-video DiT.
//!
//! One stream of tokens — `[text | audio | video]` — denoised by fifty
//! blocks of full self-attention, 3-axis RoPE and adaLN modulation, with
//! the video and audio latents riding different flow schedules and a
//! separate output head each.
//!
//! Three things make it unlike the image DiT next door:
//!
//! * **The sequence is packed, not batched.** Text, audio rows and video
//!   rows sit in one sequence and attend to each other; there is no
//!   cross-attention. Modulation is what tells a token which modality it
//!   is: every block emits six modulation vectors for each of three
//!   modality tags at each of the (at most two, for t2va) distinct
//!   timesteps, and a segment table says which row each token reads.
//!
//! * **adaLN arrives as a curve, not a matrix.** The released weight is
//!   [96768, 2688] per block — 13 B parameters, 40% of the model — for a
//!   map whose input is one number. `cortiq animate-pack` collapses it
//!   onto a rank-24 basis of the timestep curve that already carries the
//!   Turbo LoRA, so what lands here is [96768, 24] and a [1025, 24]
//!   table to interpolate. Measured against the full matrix: rms 8.7e-5
//!   on a signal of rms 0.46.
//!
//! * **Two clocks.** The sampler hands in the video sigma; the audio
//!   stream's own sigma is a closed-form remap of it (shift 12 → 3), and
//!   the two are integrated separately. `forward` returns both
//!   velocities unscaled, each on its own schedule — the reference
//!   returns the audio one pre-multiplied by d(σ_a)/d(σ_v) so that a
//!   single-schedule sampler is approximately right, which at four steps
//!   it is not.
//!
//! Parity: `tools/mk_mmh3_toy.py` builds a toy checkpoint carrying the
//! release's real tensor names and a golden forward from ComfyUI's own
//! module; `tools/mmh3_toy_gate.sh` diffs this port against it.

use crate::dit::Proj;
use crate::ltxlora::{LoraBank, LoraBranch};

/// Per-phase microseconds of the DiT block, under `CMF_MMH3_PROF=1`:
/// 0 norm+modulate · 1 qkv GEMM · 2 qk-norm+RoPE · 3 attention ·
/// 4 out GEMM+residual · 5 fc1 GEMM · 6 SwiGLU · 7 fc2 GEMM.
pub static MMH3_PROF: [std::sync::atomic::AtomicU64; 9] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

pub(crate) fn mmh3_prof_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_MMH3_PROF").is_ok())
}

/// One line per phase, sorted by cost — the map that says which kernel
/// to write next.
pub fn mmh3_prof_report() -> Option<String> {
    if !mmh3_prof_on() {
        return None;
    }
    const NAMES: [&str; 9] = [
        "norm+mod",
        "qkv gemm",
        "qknorm+rope",
        "attention",
        "out gemm",
        "fc1 gemm",
        "swiglu",
        "fc2 gemm",
        "residual",
    ];
    let mut v: Vec<(u64, &str)> = MMH3_PROF
        .iter()
        .map(|a| a.load(std::sync::atomic::Ordering::Relaxed))
        .zip(NAMES)
        .collect();
    let total: u64 = v.iter().map(|(us, _)| *us).sum();
    v.sort_by(|a, b| b.0.cmp(&a.0));
    let mut out = format!("mmh3 phases (total {:.1} s):", total as f64 / 1e6);
    for (us, name) in v {
        out.push_str(&format!(
            "\n  {name:<14} {:>7.1} s  {:>5.1}%",
            us as f64 / 1e6,
            100.0 * us as f64 / total.max(1) as f64
        ));
    }
    Some(out)
}

use crate::pool::Pool;
use cortiq_core::CmfModel;
use std::sync::Arc;

/// Frames a video latent token spans, cycling with period 5 — the
/// checkpoint's temporal grid, which is NOT uniform.
const FRAME_PER_TOKEN: [f64; 5] = [1.0, 4.0, 4.0, 4.0, 4.0];
const FRAME_RESCALE: f64 = 5.0 / 3.0;

/// Modality tags, as the adaLN row layout orders them.
const TAG_VIDEO: usize = 0;
const TAG_TEXT: usize = 1;
const TAG_AUDIO: usize = 2;
const MODALITIES: usize = 3;
/// shift/scale/gate for attention, then the same three for the MLP.
const EXPAND: usize = 6;
/// Where a visual condition row sits on the schedule: all but arrived.
const VISUAL_COND_TIMESTEP: f64 = 0.999;
/// An audio condition is not noised at all.
const AUDIO_COND_TIMESTEP: f64 = 1.0;

// ── the flow-schedule remap ─────────────────────────────────────────

/// σ on `to`'s schedule at the same point of the shared base grid.
pub fn time_shift_sigma(sigma: f64, from_shift: f64, to_shift: f64) -> f64 {
    let base = sigma / (from_shift + sigma * (1.0 - from_shift));
    to_shift * base / (1.0 + (to_shift - 1.0) * base)
}

/// d(σ_to)/d(σ_from) at the same base-grid point.
pub fn time_shift_slope(sigma: f64, from_shift: f64, to_shift: f64) -> f64 {
    let base = sigma / (from_shift + sigma * (1.0 - from_shift));
    (to_shift * (1.0 + (from_shift - 1.0) * base).powi(2))
        / (from_shift * (1.0 + (to_shift - 1.0) * base).powi(2))
}

// ── the packed layout ───────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Kind {
    Text,
    /// A keyframe's latent, re-injected every step and never denoised.
    Cond,
    /// A reference block's video rows — an image, or a clip's frames.
    RefImg,
    /// A reference block's audio rows.
    RefAudio,
    Audio,
    Video,
    /// Streaming context: video rows of an already-generated chunk, fed
    /// back as the student's own x0 at timestep 0. Not denoised, not
    /// decoded — they exist so the current chunk can attend to them.
    CtxVideo,
    /// The same for audio.
    CtxAudio,
}

/// One contiguous run of rows sharing a modality tag and a timestep.
#[derive(Clone, Copy, Debug)]
pub struct Segment {
    pub start: usize,
    pub stop: usize,
    pub kind: Kind,
}

/// The static structure of one shape signature: where each stream sits
/// in the sequence and what 3-D position every row carries.
pub struct Layout {
    pub seq_len: usize,
    pub segments: Vec<Segment>,
    /// [seq_len, 3] — (t, h, w), f64 because the axes are fractional.
    pub pos: Vec<[f64; 3]>,
    pub text_len: usize,
    pub audio_t: usize,
    pub latent_t: usize,
    pub lat_h: usize,
    pub lat_w: usize,
    /// Rows per latent frame after the 2×2 patch.
    pub frame_rows: usize,
    /// Per-token modality tag for the text span. A vision block inside
    /// the prompt carries the VIDEO tag, not the text one.
    pub text_tags: Vec<u8>,
}

/// One reference block, in the order it was given.
#[derive(Clone, Debug)]
pub enum Ref {
    /// A still, at its own latent size.
    Image { lat_h: usize, lat_w: usize },
    /// Standalone audio, `t` latent frames of it.
    Audio { t: usize },
    /// Frames, with an optional soundtrack that packs immediately
    /// before them and shares their origin.
    Video {
        latent_t: usize,
        lat_h: usize,
        lat_w: usize,
        audio_t: usize,
    },
}

/// Channel-major stereo rows: `t` frames per channel, the two channels
/// pinned to the grid's extreme w coordinates so RoPE can tell them
/// apart, h flat at zero.
fn push_audio_grid(pos: &mut Vec<[f64; 3]>, cursor: f64, t: usize, w_low: f64, w_high: f64) {
    for ch in 0..2 {
        let w = if ch == 0 { w_low } else { w_high };
        for i in 0..t {
            pos.push([cursor + i as f64, 0.0, w]);
        }
    }
}

/// `linspace((1 − ratio)/2, (1 + ratio)/2, dim/patch, endpoint=False) · 32`
fn axis_from_sqrt_area(dim: usize, patch: usize, sqrt_area: f64) -> Vec<f64> {
    let ratio = dim as f64 / sqrt_area;
    let n = dim / patch;
    (0..n)
        .map(|i| (i as f64 * (ratio / n as f64) + (1.0 - ratio) / 2.0) * 32.0)
        .collect()
}

impl Layout {
    /// t2va: `[text | audio | video]`, the target streams last and in
    /// that order. Keyframe and reference blocks would slot between the
    /// text and the audio; this port does text-to-video only.
    pub fn t2va(
        text_len: usize,
        latent_t: usize,
        lat_h: usize,
        lat_w: usize,
        audio_t: usize,
    ) -> Self {
        Self::build(text_len, latent_t, lat_h, lat_w, audio_t, &[], &[])
    }

    /// `fl2va`: keyframe condition rows sit between the text and the
    /// audio, sharing the TARGET spatial grid, each pinned to the time
    /// coordinate of the frame it stands for — the first frame at the
    /// text's end, the last one a whole clip further on, minus one
    /// span. They never advance the cursor, so audio and video still
    /// start where they would have.
    ///
    /// `frames` gives each keyframe's pixel index and the clip's total,
    /// and `text_tags` the per-token modality of the prompt span.
    pub fn fl2va(
        text_len: usize,
        latent_t: usize,
        lat_h: usize,
        lat_w: usize,
        audio_t: usize,
        frames: &[(usize, usize)],
        text_tags: &[u8],
    ) -> Self {
        Self::build(text_len, latent_t, lat_h, lat_w, audio_t, frames, text_tags)
    }

    /// `ref2va`: reference images, audio and clips ahead of the target
    /// streams. Unlike a keyframe, a reference ADVANCES the cursor —
    /// each block occupies its own stretch of the time axis, and the
    /// target audio and video begin after the last of them.
    pub fn ref2va(
        text_len: usize,
        latent_t: usize,
        lat_h: usize,
        lat_w: usize,
        audio_t: usize,
        refs: &[Ref],
        text_tags: &[u8],
    ) -> Self {
        Self::build_full(
            text_len,
            latent_t,
            lat_h,
            lat_w,
            audio_t,
            &[],
            refs,
            text_tags,
        )
    }

    fn build(
        text_len: usize,
        latent_t: usize,
        lat_h: usize,
        lat_w: usize,
        audio_t: usize,
        frames: &[(usize, usize)],
        text_tags: &[u8],
    ) -> Self {
        Self::build_full(
            text_len,
            latent_t,
            lat_h,
            lat_w,
            audio_t,
            frames,
            &[],
            text_tags,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_full(
        text_len: usize,
        latent_t: usize,
        lat_h: usize,
        lat_w: usize,
        audio_t: usize,
        frames: &[(usize, usize)],
        refs: &[Ref],
        text_tags: &[u8],
    ) -> Self {
        let area = ((lat_h * lat_w) as f64).sqrt();
        let h_axis = axis_from_sqrt_area(lat_h, 2, area);
        let w_axis = axis_from_sqrt_area(lat_w, 2, area);
        let frame_rows = h_axis.len() * w_axis.len();

        let mut pos: Vec<[f64; 3]> = Vec::new();
        let mut segments = Vec::new();

        segments.push(Segment {
            start: 0,
            stop: text_len,
            kind: Kind::Text,
        });
        for i in 0..text_len {
            pos.push([i as f64, 0.0, 0.0]);
        }

        // Both target streams share this origin: the text runs out at
        // `text_len` and audio and video start together from there —
        // unless references push it along.
        let mut cursor = text_len as f64;
        let (w_low_t, w_high_t) = (w_axis[0], w_axis[w_axis.len() - 1]);

        for r in refs {
            match r {
                Ref::Image { lat_h, lat_w } => {
                    let (rh, rw) = (
                        axis_from_sqrt_area(*lat_h, 2, ((lat_h * lat_w) as f64).sqrt()),
                        axis_from_sqrt_area(*lat_w, 2, ((lat_h * lat_w) as f64).sqrt()),
                    );
                    let start = pos.len();
                    for &h in &rh {
                        for &w in &rw {
                            pos.push([cursor, h, w]);
                        }
                    }
                    segments.push(Segment {
                        start,
                        stop: pos.len(),
                        kind: Kind::RefImg,
                    });
                    cursor += 1.0;
                }
                Ref::Audio { t } => {
                    if *t > 0 {
                        let start = pos.len();
                        push_audio_grid(&mut pos, cursor, *t, w_low_t, w_high_t);
                        segments.push(Segment {
                            start,
                            stop: pos.len(),
                            kind: Kind::RefAudio,
                        });
                    }
                    cursor += *t as f64;
                }
                Ref::Video {
                    latent_t: vt,
                    lat_h: rh_,
                    lat_w: rw_,
                    audio_t: rt,
                } => {
                    let area = ((rh_ * rw_) as f64).sqrt();
                    let rh = axis_from_sqrt_area(*rh_, 2, area);
                    let rw = axis_from_sqrt_area(*rw_, 2, area);
                    // The block's audio packs immediately BEFORE its
                    // frames, both from the same origin, and takes its
                    // w extremes from the block's own grid.
                    if *rt > 0 {
                        let start = pos.len();
                        push_audio_grid(&mut pos, cursor, *rt, rw[0], rw[rw.len() - 1]);
                        segments.push(Segment {
                            start,
                            stop: pos.len(),
                            kind: Kind::RefAudio,
                        });
                    }
                    let start = pos.len();
                    let mut t_coord = cursor;
                    for k in 0..*vt {
                        for &h in &rh {
                            for &w in &rw {
                                pos.push([t_coord, h, w]);
                            }
                        }
                        t_coord += FRAME_RESCALE * FRAME_PER_TOKEN[k % 5];
                    }
                    segments.push(Segment {
                        start,
                        stop: pos.len(),
                        kind: Kind::RefImg,
                    });
                    let spans: f64 = (0..*vt)
                        .map(|k| FRAME_RESCALE * FRAME_PER_TOKEN[k % 5])
                        .sum();
                    cursor += (*rt as f64).max(spans);
                }
            }
        }
        let cursor = cursor;

        // Keyframes, in the order given.
        let spans: f64 = (0..latent_t)
            .map(|k| FRAME_RESCALE * FRAME_PER_TOKEN[k % 5])
            .sum();
        for &(pixel_index, frame_count) in frames {
            let cond_t = if pixel_index == 0 {
                cursor
            } else if frame_count > 0 && pixel_index == frame_count - 1 {
                cursor + spans - FRAME_RESCALE
            } else {
                panic!("only the first and last frame can anchor a keyframe");
            };
            let start = pos.len();
            for &h in &h_axis {
                for &w in &w_axis {
                    pos.push([cond_t, h, w]);
                }
            }
            segments.push(Segment {
                start,
                stop: pos.len(),
                kind: Kind::Cond,
            });
        }

        // Audio is channel-major stereo: every latent frame once per
        // channel, the two channels pinned to the frame grid's extreme
        // w coordinates so they are distinguishable under RoPE.
        let a_start = pos.len();
        push_audio_grid(&mut pos, cursor, audio_t, w_low_t, w_high_t);
        segments.push(Segment {
            start: a_start,
            stop: pos.len(),
            kind: Kind::Audio,
        });

        // Video: the t axis advances by the per-token frame spans, not
        // by one; h/w come from the shared frame grid.
        let v_start = pos.len();
        let mut t_coord = cursor;
        for k in 0..latent_t {
            for &h in &h_axis {
                for &w in &w_axis {
                    pos.push([t_coord, h, w]);
                }
            }
            t_coord += FRAME_RESCALE * FRAME_PER_TOKEN[k % 5];
        }
        segments.push(Segment {
            start: v_start,
            stop: pos.len(),
            kind: Kind::Video,
        });

        Self {
            seq_len: pos.len(),
            segments,
            pos,
            text_len,
            audio_t,
            latent_t,
            lat_h,
            lat_w,
            frame_rows,
            text_tags: if text_tags.is_empty() {
                vec![TAG_TEXT as u8; text_len]
            } else {
                text_tags.to_vec()
            },
        }
    }

    /// The chunk-causal layout: one chunk being denoised, and the chunks
    /// it is allowed to see.
    ///
    /// RAVEN generates a clip chunk by chunk, each one extrapolated from
    /// what came before instead of denoised as one bidirectional clip.
    /// Its attention pattern is `sink` chunks from the start plus a
    /// sliding `window` of recent ones — the reference implements that
    /// with a KV cache; the same *pattern* falls out of simply not
    /// packing the rows a chunk may not see, which is what this builds.
    /// The cache is then an optimization of this, not a prerequisite.
    ///
    /// Positions stay ABSOLUTE. A chunk five steps in must carry the RoPE
    /// coordinates it would have had in the whole clip, or the model is
    /// told it is generating the opening again.
    ///
    /// Rows come out as `[text | audio: ctx then current | video: ctx then
    /// current]`: the audio and video blocks stay contiguous because the
    /// patchifier hands them over that way, and audio is channel-major, so
    /// its context is two runs — one per channel.
    #[allow(clippy::too_many_arguments)]
    pub fn streaming(
        text_len: usize,
        text_tags: &[u8],
        lat_h: usize,
        lat_w: usize,
        ctx_video: &[usize],
        cur_video: &[usize],
        ctx_audio: &[usize],
        cur_audio: &[usize],
    ) -> Self {
        let area = ((lat_h * lat_w) as f64).sqrt();
        let h_axis = axis_from_sqrt_area(lat_h, 2, area);
        let w_axis = axis_from_sqrt_area(lat_w, 2, area);
        let frame_rows = h_axis.len() * w_axis.len();
        let (w_low, w_high) = (w_axis[0], w_axis[w_axis.len() - 1]);

        let mut pos: Vec<[f64; 3]> = Vec::new();
        let mut segments = Vec::new();
        segments.push(Segment {
            start: 0,
            stop: text_len,
            kind: Kind::Text,
        });
        for i in 0..text_len {
            pos.push([i as f64, 0.0, 0.0]);
        }
        let cursor = text_len as f64;

        // The absolute t coordinate of latent frame k: the same running
        // sum the bidirectional layout walks, evaluated at k.
        let t_at = |k: usize| -> f64 {
            cursor
                + (0..k)
                    .map(|j| FRAME_RESCALE * FRAME_PER_TOKEN[j % 5])
                    .sum::<f64>()
        };

        // Audio, channel-major: for each channel, the context frames then
        // the current ones, so each channel's half is [ctx | cur].
        // Channel-major INSIDE each role, not across them: the packer
        // hands over [ch0 | ch1] for a set of frames, so context and
        // current each have to be one run of rows — the output side
        // looks the current one up by kind and slices it whole.
        for (idxs, kind) in [(ctx_audio, Kind::CtxAudio), (cur_audio, Kind::Audio)] {
            if idxs.is_empty() {
                continue;
            }
            let start = pos.len();
            for ch in 0..2 {
                let w = if ch == 0 { w_low } else { w_high };
                for &i in idxs {
                    pos.push([cursor + i as f64, 0.0, w]);
                }
            }
            segments.push(Segment {
                start,
                stop: pos.len(),
                kind,
            });
        }

        for (idxs, kind) in [(ctx_video, Kind::CtxVideo), (cur_video, Kind::Video)] {
            if idxs.is_empty() {
                continue;
            }
            let start = pos.len();
            for &k in idxs {
                let t = t_at(k);
                for &h in &h_axis {
                    for &w in &w_axis {
                        pos.push([t, h, w]);
                    }
                }
            }
            segments.push(Segment {
                start,
                stop: pos.len(),
                kind,
            });
        }

        Self {
            seq_len: pos.len(),
            segments,
            pos,
            text_len,
            audio_t: ctx_audio.len() + cur_audio.len(),
            latent_t: ctx_video.len() + cur_video.len(),
            lat_h,
            lat_w,
            frame_rows,
            text_tags: if text_tags.is_empty() {
                vec![TAG_TEXT as u8; text_len]
            } else {
                text_tags.to_vec()
            },
        }
    }

    /// The contiguous span the video rows occupy — context and current
    /// together, because the patchifier produces them as one block.
    pub fn video_block(&self) -> (usize, usize) {
        self.span(&[Kind::CtxVideo, Kind::Video])
    }

    /// The same for audio.
    pub fn audio_block(&self) -> (usize, usize) {
        self.span(&[Kind::CtxAudio, Kind::Audio])
    }

    fn span(&self, kinds: &[Kind]) -> (usize, usize) {
        let mut lo = usize::MAX;
        let mut hi = 0usize;
        for s in self.segments.iter().filter(|s| kinds.contains(&s.kind)) {
            lo = lo.min(s.start);
            hi = hi.max(s.stop);
        }
        if lo == usize::MAX { (0, 0) } else { (lo, hi) }
    }

    /// How many keyframe condition rows the layout carries.
    pub fn cond_rows(&self) -> usize {
        self.segments
            .iter()
            .filter(|s| s.kind == Kind::Cond)
            .map(|s| s.stop - s.start)
            .sum()
    }

    fn segment(&self, kind: Kind) -> Segment {
        *self
            .segments
            .iter()
            .find(|s| s.kind == kind)
            .expect("every layout carries all three streams")
    }
}

// ── weights ─────────────────────────────────────────────────────────

struct Adaln {
    /// [out, rank] — the curve basis, already carrying the LoRA.
    w: Proj,
    b: Vec<f32>,
    /// [grid, rank]
    table: Vec<f32>,
    rank: usize,
    out: usize,
}

impl Adaln {
    fn load(model: &Arc<CmfModel>, prefix: &str) -> Result<Self, String> {
        let w = Proj::from_model(model, &format!("{prefix}.weight"))?;
        let table = crate::dit::cmf_f32(model, &format!("{prefix}.table"))?;
        let b = crate::dit::cmf_f32(model, &format!("{prefix}.bias"))?;
        let out = w.rows();
        let rank = table.len() / CURVE_GRID;
        Ok(Self {
            w,
            b,
            table,
            rank,
            out,
        })
    }

    /// The modulation vectors at `ts`: `[ts.len() · MODALITIES, expand ·
    /// hidden]` laid out so row `t·MODALITIES + tag`, chunk `e`, starts
    /// at `(t·MODALITIES + tag)·expand·hidden + e·hidden`.
    fn eval(&self, ts: &[f64], pool: Option<&Pool>) -> Vec<f32> {
        let g = CURVE_GRID;
        let mut coords = vec![0f32; ts.len() * self.rank];
        for (i, &t) in ts.iter().enumerate() {
            // t → fractional grid index; out-of-range clamps to the ends,
            // and the last interval is kept whole so t = 1 does not read
            // past the table.
            let p = (t.clamp(0.0, 1.0) * (g - 1) as f64) as f32;
            let i0 = (p.floor() as usize).min(g - 2);
            let f = p - i0 as f32;
            for k in 0..self.rank {
                let (a, b) = (
                    self.table[i0 * self.rank + k],
                    self.table[(i0 + 1) * self.rank + k],
                );
                coords[i * self.rank + k] = a + (b - a) * f;
            }
        }
        let mut out = vec![0f32; ts.len() * self.out];
        self.w.matmat(&coords, ts.len(), &mut out, pool);
        for row in out.chunks_exact_mut(self.out) {
            for (v, &bv) in row.iter_mut().zip(&self.b) {
                *v += bv;
            }
        }
        out
    }
}

struct Block {
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    qkv: Proj, // [3·heads·hd, hidden]
    out: Proj, // [hidden, heads·hd]
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    fc1: Proj, // [2·ffn, hidden]
    fc2: Proj, // [hidden, ffn]
    adaln: Option<Adaln>,
    /// Runtime low-rank branches, one per projection an adapter names.
    /// A branch is why the fused device paths below stand down: they
    /// keep qkv, the attention output and the FFN's middle on the card,
    /// and the branch has to read exactly those.
    lora: BlockLora,
}

/// The four projections an adapter can reach in a packed block. `adaln`
/// is deliberately absent: this container carries the modulation as a
/// rank-24 curve, and folding an adaLN update into it needs the time
/// embedding the packer had (`animate-pack --lora --time-embedder`).
#[derive(Default)]
struct BlockLora {
    qkv: Option<LoraBranch>,
    out: Option<LoraBranch>,
    fc1: Option<LoraBranch>,
    fc2: Option<LoraBranch>,
}

/// A projection and its branch, in one device submission where the platform
/// has that kernel and the shapes fit — the branch reads the activation the
/// base GEMM already uploaded and accumulates into the output it just wrote,
/// so it costs no transfer of its own. Falls back to `matmat` + `add`, which
/// is correct everywhere and pays a round trip for the branch.
///
/// The router and the probe both need the branch and the base separated to
/// measure one against the other, which the fused kernel does not do — so
/// when either is asked for, a branch takes the split path until it HAS been
/// measured, and the fused one from the next step onward. Without that the
/// probe reports 0.0000 for every branch and reads as "this adapter does
/// nothing", which is what it did in the first run of it.
fn proj_with_lora(
    p: &Proj,
    br: &Option<LoraBranch>,
    x: &[f32],
    n: usize,
    out: &mut [f32],
    pool: Option<&Pool>,
) {
    #[cfg(target_os = "macos")]
    if let Some(l) = br.as_ref().filter(|l| l.live()) {
        if let Some((model, idx)) = p.q4tp_mapped() {
            let (rows, cols) = (p.rows(), p.cols());
            if n >= 32
                && cols % 32 == 0
                && crate::gpu::enabled_here()
                && !crate::gpu::mm_killed()
                && (!crate::ltxlora::wants_measurement() || l.resonance() != 0.0)
                && crate::gpu_metal::q4tp_matmat_lora(model, idx, x, n, rows, cols, out, &l.side())
            {
                return;
            }
        }
    }
    p.matmat(x, n, out, pool);
    if let Some(l) = br {
        l.add(x, n, out, pool);
    }
}

impl BlockLora {
    /// Live, not merely present: a branch the router has switched off
    /// gives the fused device path back for the rest of the render.
    fn live(br: &Option<LoraBranch>) -> bool {
        br.as_ref().is_some_and(|l| l.live())
    }
    fn ffn_any(&self) -> bool {
        Self::live(&self.fc1) || Self::live(&self.fc2)
    }
}

pub(crate) const CURVE_GRID: usize = 1025;

pub struct MiniMaxH3 {
    video_patch: Proj,
    video_patch_b: Vec<f32>,
    audio_patch: Proj,
    audio_patch_b: Vec<f32>,
    condition: Proj,
    condition_b: Vec<f32>,
    refiner: Vec<Block>,
    refiner_norm: Vec<f32>,
    blocks: Vec<Block>,
    final_norm: Vec<f32>,
    final_adaln: Adaln,
    video_out: Proj,
    video_out_b: Vec<f32>,
    audio_out: Proj,
    audio_out_b: Vec<f32>,
    inv_freq: Vec<f32>,
    pool: Option<Arc<Pool>>,
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub latents_dim: usize,
    pub audio_dim: usize,
    pub text_dim: usize,
    pub shift_video: f64,
    pub shift_audio: f64,
    eps: f64,
    qk_eps: f64,
    final_eps: f64,
    /// How much of a keyframe latent survives the noise blend, and
    /// therefore where its rows sit on the schedule. `VISUAL_COND_
    /// TIMESTEP` is the reference's default; 1.0 turns the blend off.
    pub cond_aug: f64,
    /// The same, for a reference soundtrack. The reference's default is
    /// 1.0 — an audio condition is not noised at all.
    pub cond_aug_audio: f64,
    /// Adapter keys that found a projection, for the caller's report.
    lora_bound: std::collections::BTreeSet<String>,
}

fn rms_norm_into(x: &[f32], w: &[f32], eps: f64, dst: &mut [f32]) {
    let ss = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ss + eps).sqrt();
    for ((d, &v), &g) in dst.iter_mut().zip(x).zip(w) {
        *d = (v as f64 * inv) as f32 * g;
    }
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

impl MiniMaxH3 {
    pub fn from_cmf(model: &Arc<CmfModel>) -> Result<Self, String> {
        Self::from_cmf_lora(model, None)
    }

    /// The same, with an adapter consulted for every projection.
    ///
    /// The base weights are q4tp and stay untouched: a rank-32 update
    /// cannot be folded into a four-bit ladder without dequantizing the
    /// whole DiT, so the branch rides beside the base GEMM the way
    /// [`crate::ltxlora`] does it for LTX.
    pub fn from_cmf_lora(model: &Arc<CmfModel>, bank: Option<&LoraBank>) -> Result<Self, String> {
        let cfg: serde_json::Value = serde_json::from_slice(
            model
                .tensor_bytes("dit.config_json")
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("dit.config_json: {e}"))?;
        let u = |k: &str| cfg[k].as_u64().unwrap_or(0) as usize;
        let f = |k: &str, d: f64| cfg[k].as_f64().unwrap_or(d);
        let n_blocks = u("num_layers");
        let n_refiner = u("token_refiner_num_layers");
        let f32v = |n: &str| crate::dit::cmf_f32(model, n);

        // Which of the adapter's names actually found a projection here.
        let bound: std::cell::RefCell<std::collections::BTreeSet<String>> = Default::default();
        let load_block = |prefix: &str, with_adaln: bool| -> Result<Block, String> {
            let qkv = Proj::from_model(model, &format!("dit.{prefix}.attn.qkv_proj.weight"))?;
            let out = Proj::from_model(model, &format!("dit.{prefix}.attn.out_proj.weight"))?;
            let fc1 = Proj::from_model(model, &format!("dit.{prefix}.mlp.fc1.weight"))?;
            let fc2 = Proj::from_model(model, &format!("dit.{prefix}.mlp.fc2.weight"))?;
            // A branch is bound against the SHAPE of the projection it will
            // ride: `add` writes n·out floats through a raw pointer, so an
            // adapter for another model has to be refused by name, not
            // discovered as a corrupted panel.
            let lora = match bank {
                None => BlockLora::default(),
                Some(k) => {
                    let mut take = |suffix: &str, p: &Proj| -> Result<Option<LoraBranch>, String> {
                        let key = format!("{prefix}.{suffix}");
                        let br = k.branch_for(&format!("dit.{key}"), p.rows(), p.cols())?;
                        if br.is_some() {
                            bound.borrow_mut().insert(key);
                        }
                        Ok(br)
                    };
                    BlockLora {
                        qkv: take("attn.qkv_proj", &qkv)?,
                        out: take("attn.out_proj", &out)?,
                        fc1: take("mlp.fc1", &fc1)?,
                        fc2: take("mlp.fc2", &fc2)?,
                    }
                }
            };
            Ok(Block {
                norm1: f32v(&format!("dit.{prefix}.norm1.weight"))?,
                norm2: f32v(&format!("dit.{prefix}.norm2.weight"))?,
                qkv,
                out,
                q_norm: f32v(&format!("dit.{prefix}.attn.q_norm.weight"))?,
                k_norm: f32v(&format!("dit.{prefix}.attn.k_norm.weight"))?,
                fc1,
                fc2,
                adaln: if with_adaln {
                    Some(Adaln::load(model, &format!("dit.{prefix}.adaln"))?)
                } else {
                    None
                },
                lora,
            })
        };
        let blocks = (0..n_blocks)
            .map(|i| load_block(&format!("blocks.{i}"), true))
            .collect::<Result<Vec<_>, _>>()?;
        let refiner = (0..n_refiner)
            .map(|i| load_block(&format!("token_refiner.blocks.{i}"), false))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            video_patch: Proj::from_model(model, "dit.video_patch_proj.weight")?,
            video_patch_b: f32v("dit.video_patch_proj.bias")?,
            audio_patch: Proj::from_model(model, "dit.audio_patch_proj.weight")?,
            audio_patch_b: f32v("dit.audio_patch_proj.bias")?,
            condition: Proj::from_model(model, "dit.condition_proj.weight")?,
            condition_b: f32v("dit.condition_proj.bias")?,
            refiner,
            refiner_norm: f32v("dit.token_refiner.final_norm.weight")?,
            blocks,
            final_norm: f32v("dit.final_layer.norm.weight")?,
            final_adaln: Adaln::load(model, "dit.final_layer.adaln")?,
            video_out: Proj::from_model(model, "dit.final_layer.video_out.weight")?,
            video_out_b: f32v("dit.final_layer.video_out.bias")?,
            audio_out: Proj::from_model(model, "dit.final_layer.audio_out.weight")?,
            audio_out_b: f32v("dit.final_layer.audio_out.bias")?,
            inv_freq: f32v("dit.rope_inv_freq")?,
            pool: Pool::from_env(),
            hidden: u("hidden_size"),
            heads: u("num_attention_heads"),
            head_dim: u("attention_head_dim"),
            ffn: u("ffn_hidden_size"),
            latents_dim: u("latents_dim"),
            audio_dim: u("audio_latents_dim"),
            text_dim: u("text_dim"),
            shift_video: f("sigma_shift_video", 12.0),
            shift_audio: f("sigma_shift_audio", 3.0),
            eps: f("norm_eps", 1e-5),
            qk_eps: f("qk_norm_eps", 1e-5),
            final_eps: f("final_norm_eps", 1e-5),
            cond_aug: VISUAL_COND_TIMESTEP,
            cond_aug_audio: AUDIO_COND_TIMESTEP,
            lora_bound: bound.into_inner(),
        })
    }

    /// How many of the adapter's branches found a projection.
    pub fn lora_bound(&self) -> usize {
        self.lora_bound.len()
    }

    /// Whether this adapter key landed on a projection of ours.
    pub fn lora_binds(&self, key: &str) -> bool {
        self.lora_bound.contains(key)
    }

    /// Under `CMF_LORA_PROBE=1`: every branch by measured contribution,
    /// loudest first, and which ones the router switched off. This is the
    /// map that says where an adapter actually lives — for the Realism
    /// adapter it is not uniform across the fifty blocks.
    pub fn lora_report(&self) -> Option<String> {
        if !crate::ltxlora::probe_report_on() {
            return None;
        }
        let mut rows: Vec<(f32, bool, String)> = Vec::new();
        let mut walk = |blocks: &[Block], pre: &str| {
            for (i, b) in blocks.iter().enumerate() {
                for (name, br) in [
                    ("attn.qkv_proj", &b.lora.qkv),
                    ("attn.out_proj", &b.lora.out),
                    ("mlp.fc1", &b.lora.fc1),
                    ("mlp.fc2", &b.lora.fc2),
                ] {
                    if let Some(l) = br {
                        rows.push((l.resonance(), l.live(), format!("{pre}{i}.{name}")));
                    }
                }
            }
        };
        walk(&self.blocks, "blocks.");
        walk(&self.refiner, "token_refiner.blocks.");
        if rows.is_empty() {
            return None;
        }
        rows.sort_by(|a, b| b.0.total_cmp(&a.0));
        let live = rows.iter().filter(|r| r.1).count();
        let mut s = format!(
            "lora branches by contribution ‖sΔY‖/‖Y‖ ({} of {} live):\n",
            live,
            rows.len()
        );
        for (r, on, name) in &rows {
            s.push_str(&format!(
                "  {:>8.4}  {}  {name}\n",
                r,
                if *on { "on " } else { "off" }
            ));
        }
        Some(s)
    }

    /// Qwen3-VL states `[n, text_dim]` → refined text embeds
    /// `[n, hidden]`. Prompt-only, so the caller does this once per
    /// generation rather than once per step.
    pub fn refine_text(&self, states: &[f32], n: usize) -> Vec<f32> {
        let pool = self.pool.as_deref();
        let mut h = vec![0f32; n * self.hidden];
        self.condition.matmat(states, n, &mut h, pool);
        for row in h.chunks_exact_mut(self.hidden) {
            for (v, &b) in row.iter_mut().zip(&self.condition_b) {
                *v += b;
            }
        }
        let ids: Vec<[f64; 3]> = Vec::new();
        for blk in &self.refiner {
            self.block_forward(blk, &mut h, n, None, &ids, &[]);
        }
        let mut out = vec![0f32; n * self.hidden];
        for (o, x) in out
            .chunks_exact_mut(self.hidden)
            .zip(h.chunks_exact(self.hidden))
        {
            rms_norm_into(x, &self.refiner_norm, self.final_eps, o);
        }
        out
    }

    /// The rotation angles of every row: `[n, 48]`. Each of the three
    /// position axes contributes `inv_freq.len()` angles, and the pair
    /// (j, j+48) of the first 96 head dims rotates by angle j.
    fn rope_angles(&self, pos: &[[f64; 3]]) -> Vec<f32> {
        let k = self.inv_freq.len();
        let mut out = Vec::with_capacity(pos.len() * 3 * k);
        for p in pos {
            for axis in 0..3 {
                for j in 0..k {
                    out.push((p[axis] * self.inv_freq[j] as f64) as f32);
                }
            }
        }
        out
    }

    /// Per-head RMSNorm then the partial split-half rotation, in place.
    /// `angles` carries 48 angles a row — three axes × 16 frequencies —
    /// and pair (j, j+48) of the head's first 96 dims turns by angle j.
    /// Dims 96..128 are not rotated at all, which is the checkpoint's
    /// own `rope_inv_freq_len` arithmetic and not a truncation.
    /// In place on a STRIDED view of the fused qkv buffer: `stride` is
    /// the row pitch and `off` the plane's start. Normalizing q and k
    /// through a scratch copy cost four full passes over `n·heads·hd`
    /// per block — 10 G element copies over a render — for arithmetic
    /// that touches each element once.
    #[allow(clippy::too_many_arguments)]
    fn norm_rope_w(
        &self,
        v: &mut [f32],
        n: usize,
        heads: usize,
        w: &[f32],
        angles: &[f32],
        stride: usize,
        off: usize,
    ) {
        let hd = self.head_dim;
        let pairs = if angles.is_empty() {
            0
        } else {
            angles.len() / n
        };
        let pool = self.pool.as_deref();
        let ptr = SendPtr(v.as_mut_ptr());
        let work = |lo: usize, hi: usize| {
            for p in lo..hi {
                for h in 0..heads {
                    // SAFETY: workers own disjoint token ranges, and the
                    // heads of one token are disjoint within it.
                    let x = unsafe { ptr.row(p * stride + off + h * hd, hd) };
                    let ss = x.iter().map(|&a| (a as f64) * (a as f64)).sum::<f64>() / hd as f64;
                    let inv = 1.0 / (ss + self.qk_eps).sqrt();
                    for (d, &g) in x.iter_mut().zip(w) {
                        *d = (*d as f64 * inv) as f32 * g;
                    }
                    for j in 0..pairs {
                        let a = angles[p * pairs + j];
                        let (s, c) = a.sin_cos();
                        let (lo_v, hi_v) = (x[j], x[j + pairs]);
                        x[j] = lo_v * c - hi_v * s;
                        x[j + pairs] = lo_v * s + hi_v * c;
                    }
                }
            }
        };
        match pool {
            Some(pl) => pl.run_rows(n, &work),
            None => work(0, n),
        }
    }

    /// Full bidirectional attention over the packed sequence.
    /// `nr` = (rope angles, q norm weights, k norm weights, eps) when the
    /// DEVICE should apply qk-norm and RoPE. Passing it means the caller
    /// skipped `norm_rope_w`, which is the only reason the qkv panel had
    /// to come back to the host at all.
    fn attention(
        &self,
        qkv: &[f32],
        n: usize,
        attn: &mut [f32],
        nr: Option<(&[f32], &[f32], &[f32], f32)>,
    ) {
        let t_repack = std::time::Instant::now();
        let (nh, hd) = (self.heads, self.head_dim);
        let inner = nh * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let pool = self.pool.as_deref();
        // Device path: scores, softmax and the PV product all stay in
        // device buffers (`dit_qk` → `dit_softmax` → `dit_pv`), so the
        // n×n score plane — 144 MB at render size — never crosses the
        // bus. Measured share of a denoise step before this: 41.5%.
        // The kernels exist and Lumina's DiT already rides them; this
        // block is what puts MiniMax on the same road. CMF_MMH3_ATTN=cpu
        // forces the host loop (the A/B that proved the parity).
        if std::env::var("CMF_MMH3_ATTN").as_deref() != Ok("cpu")
            && crate::gpu::enabled_here()
            && n >= 256
        {
            // The qkv panel goes up in one piece and is split into
            // head-major planes ON the card. The host repack below cost
            // 4.4 s of a 7.3 s attention phase — more than the device
            // work it fed. Measured at render size: repack 4.4 → 1.5 s,
            // render 130.0 → 111.8 s, frames bit-identical.
            // `CMF_MMH3_ATTN=repack` forces the host form.
            // The device can do qk-norm and RoPE itself (kernel and
            // plumbing are in; pass Some((angles, q_norm, k_norm, eps))).
            // `attention` cannot reach them: they live in the caller,
            // and handing them over means changing this signature — the
            // last step of task #33, and the point where the panel stops
            // coming home at all.
            if std::env::var("CMF_MMH3_ATTN").as_deref() != Ok("repack")
                && crate::gpu::dit_attention_packed(qkv, nh, n, hd, scale, nr, attn)
            {
                // No stamp: the caller's slot-3 stopwatch already spans
                // this whole call, and CMF_DIT_ATTN_PROF breaks the
                // device half into its three walls. Two overlapping
                // stopwatches on the same work is how a 19.2 s step
                // came to read as 26.1.
                return;
            }
            assert!(
                nr.is_none(),
                "device qk-norm was requested but the device path refused; \
                 the host loops below expect q/k already normalized"
            );
            let mut qh = vec![0f32; nh * n * hd];
            let mut kh = vec![0f32; nh * n * hd];
            let mut vh = vec![0f32; nh * n * hd];
            {
                let (pq, pk, pv) = (
                    SendPtr(qh.as_mut_ptr()),
                    SendPtr(kh.as_mut_ptr()),
                    SendPtr(vh.as_mut_ptr()),
                );
                pool_rows(pool, n, &|lo, hi| {
                    for p in lo..hi {
                        let base = p * 3 * inner;
                        for h in 0..nh {
                            let dst = (h * n + p) * hd;
                            // SAFETY: workers own disjoint token ranges,
                            // and each token writes its own head slots.
                            unsafe {
                                pq.row(dst, hd)
                                    .copy_from_slice(&qkv[base + h * hd..base + (h + 1) * hd]);
                                pk.row(dst, hd).copy_from_slice(
                                    &qkv[base + inner + h * hd..base + inner + (h + 1) * hd],
                                );
                                pv.row(dst, hd).copy_from_slice(
                                    &qkv[base + 2 * inner + h * hd
                                        ..base + 2 * inner + (h + 1) * hd],
                                );
                            }
                        }
                    }
                });
            }
            // Split the phase: the repack above is host work, this call
            // is device work. The last round wrote a tensor-core QK on
            // the assumption that the GEMMs dominate and the step did
            // not move — so measure the halves before writing anything
            // else. Slot 2 (qknorm+rope, unused on this path) takes the
            // repack; slot 3 keeps the device call.
            Self::prof(2, t_repack);
            // No stamp on the device call: the caller's slot-3 stopwatch
            // already spans it, and stamping both double-counts.
            if crate::gpu::dit_attention(&qh, &kh, &vh, nh, nh, n, hd, scale, attn) {
                return;
            }
        }
        let mut qh = vec![0f32; n * hd];
        let mut kh = vec![0f32; n * hd];
        let mut vt = vec![0f32; hd * n];
        let mut scores = vec![0f32; n * n];
        let mut oh = vec![0f32; n * hd];
        for h in 0..nh {
            for p in 0..n {
                let base = p * 3 * inner;
                let qs = &qkv[base + h * hd..base + (h + 1) * hd];
                for (d, &val) in qs.iter().enumerate() {
                    qh[p * hd + d] = val * scale;
                }
                kh[p * hd..(p + 1) * hd]
                    .copy_from_slice(&qkv[base + inner + h * hd..base + inner + (h + 1) * hd]);
                let vs = &qkv[base + 2 * inner + h * hd..base + 2 * inner + (h + 1) * hd];
                for (d, &val) in vs.iter().enumerate() {
                    vt[d * n + p] = val;
                }
            }
            crate::fcd_ops::gemm_nt(&qh, &kh, &mut scores, n, hd, n, pool);
            let sp = SendPtr(scores.as_mut_ptr());
            let soft = |lo: usize, hi: usize| {
                for r in lo..hi {
                    // SAFETY: workers own disjoint score rows.
                    softmax_inplace(unsafe { sp.row(r * n, n) });
                }
            };
            match pool {
                Some(pl) => pl.run_rows(n, &soft),
                None => soft(0, n),
            }
            crate::fcd_ops::gemm_nt(&scores, &vt, &mut oh, n, n, hd, pool);
            for p in 0..n {
                attn[p * inner + h * hd..p * inner + (h + 1) * hd]
                    .copy_from_slice(&oh[p * hd..(p + 1) * hd]);
            }
        }
    }

    /// Where a denoise step's wall actually goes, in microseconds, under
    /// `CMF_MMH3_PROF=1`. Optimizing a 60-second step without this is
    /// guesswork, and guesswork on a rented card is expensive.
    fn prof(slot: usize, t: std::time::Instant) {
        if !mmh3_prof_on() {
            return;
        }
        MMH3_PROF[slot].fetch_add(
            t.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// One block. `mods` is the block's modulation buffer and `rows` the
    /// per-token row index into it; both empty for the refiner, which is
    /// unmodulated and unrotated.
    fn block_forward(
        &self,
        blk: &Block,
        x: &mut [f32],
        n: usize,
        mods: Option<&[f32]>,
        pos: &[[f64; 3]],
        rows: &[u32],
    ) {
        let hs = self.hidden;
        let pool = self.pool.as_deref();
        let inner = self.heads * self.head_dim;
        let angles = if pos.is_empty() {
            Vec::new()
        } else {
            self.rope_angles(pos)
        };

        let t = std::time::Instant::now();
        let mut xn = vec![0f32; n * hs];
        self.norm_rows(&mut xn, x, n, hs, &blk.norm1, self.eps);
        if let (Some(m), false) = (mods, rows.is_empty()) {
            self.modulate(&mut xn, hs, m, rows, 0, 1);
        }
        Self::prof(0, t);
        let t = std::time::Instant::now();
        // The projection writes its panel into a device buffer and the
        // split reads it THERE — with qk-norm on the device too, nothing
        // touches that panel on the host, so 160 MB down and the same
        // back up per block simply stop happening. Measured at render
        // size: 104.9 → 98.1 s, frames bit-identical.
        // `CMF_MMH3_FUSEQKV=0` sends the panel home again.
        let t_qkv = std::time::Instant::now();
        // Ask whether the device can TAKE the norm, not merely whether a
        // device exists. Skipping the host loop for a backend with no
        // packed kernel hands the attention unnormalized q/k and there is
        // no way back from there — which is why the refusal below used to
        // be an assert.
        let qk_gpu = std::env::var("CMF_MMH3_QKNORM").as_deref() != Ok("cpu")
            && crate::gpu::enabled_here()
            && crate::gpu::dit_attention_packed_available()
            && n >= 256;
        // An adapter on qkv needs the panel the fused paths never let
        // out of the card; one on `out` needs the attention output. Each
        // stands down only for the projection it owns — a qkv-only
        // adapter still gets device attention through the chain below.
        if qk_gpu
            && !BlockLora::live(&blk.lora.qkv)
            && std::env::var("CMF_MMH3_FUSEQKV").as_deref() != Ok("0")
        {
            // Best case first: qkv, attention and the output projection
            // with nothing crossing the bus between them. It refuses at
            // the door when anything is missing, so falling through to
            // the chain below never repeats work.
            let fuse_out = !BlockLora::live(&blk.lora.out)
                && std::env::var("CMF_MMH3_FUSEOUT").as_deref() != Ok("0");
            if std::env::var("CMF_GPU_DEBUG").is_ok() {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    eprintln!(
                        "mmh3 fuse-out gate: qkv_q={} out_q={} qkv_map={} out_map={}",
                        matches!(&blk.qkv, Proj::Q(_)),
                        matches!(&blk.out, Proj::Q(_)),
                        matches!(&blk.qkv, Proj::Q(q) if q.mapped_device_gemm().is_some()),
                        matches!(&blk.out, Proj::Q(o) if o.mapped_device_gemm().is_some()),
                    )
                });
            }
            if let (true, Proj::Q(q), Proj::Q(o)) = (fuse_out, &blk.qkv, &blk.out) {
                if let (Some((m, i)), Some((_, oi))) =
                    (q.mapped_device_gemm(), o.mapped_device_gemm())
                {
                    let mut proj = vec![0f32; n * hs];
                    if crate::gpu::dit_qkv_attn_out(
                        m,
                        i,
                        oi,
                        &xn,
                        n,
                        hs,
                        self.heads,
                        self.head_dim,
                        1.0 / (self.head_dim as f32).sqrt(),
                        (
                            &angles[..],
                            &blk.q_norm[..],
                            &blk.k_norm[..],
                            self.qk_eps as f32,
                        ),
                        &mut proj,
                    ) {
                        Self::prof(1, t_qkv);
                        let t_res = std::time::Instant::now();
                        self.residual(x, hs, &proj, mods, rows, 2);
                        Self::prof(8, t_res);
                        self.ffn_tail(blk, x, n, mods, rows);
                        return;
                    }
                }
            }
            if let Proj::Q(q) = &blk.qkv {
                if let Some((m, i)) = q.mapped_device_gemm() {
                    let mut attn = vec![0f32; n * inner];
                    if crate::gpu::dit_qkv_attention(
                        m,
                        i,
                        &xn,
                        n,
                        hs,
                        self.heads,
                        self.head_dim,
                        1.0 / (self.head_dim as f32).sqrt(),
                        (
                            &angles[..],
                            &blk.q_norm[..],
                            &blk.k_norm[..],
                            self.qk_eps as f32,
                        ),
                        &mut attn,
                    ) {
                        // Stamp the same slots the unfused path does, or
                        // the profile reads 5.5 s for a step the device
                        // spends 6+ in: an instrument with a blind spot
                        // is worse than none, and this one has cost two
                        // wrong conclusions already.
                        Self::prof(1, t_qkv);
                        let t_out = std::time::Instant::now();
                        let mut proj = vec![0f32; n * hs];
                        proj_with_lora(&blk.out, &blk.lora.out, &attn, n, &mut proj, pool);
                        Self::prof(4, t_out);
                        let t_res = std::time::Instant::now();
                        self.residual(x, hs, &proj, mods, rows, 2);
                        Self::prof(8, t_res);
                        self.ffn_tail(blk, x, n, mods, rows);
                        return;
                    }
                }
            }
        }
        let mut qkv = vec![0f32; n * 3 * inner];
        proj_with_lora(&blk.qkv, &blk.lora.qkv, &xn, n, &mut qkv, pool);
        Self::prof(1, t);
        // q and k are the first two thirds of every row; normalize and
        // rotate them where they lie, leaving v alone.
        // qk-norm and RoPE ride the same device pass that scatters the
        // panel head-major, so the host neither walks the panel nor
        // needs it back. Parity holds over a whole render: 1, 2 and 4
        // steps all match this loop's output exactly (delta 0.000).
        //
        // A 7.7% "drift" was measured first and was an artefact — the
        // reference had been rendered by an EARLIER binary, so the
        // comparison carried every change since, not this one. Same
        // binary, both arms, or the number means nothing.
        // `CMF_MMH3_QKNORM=cpu` restores the host loop.
        let qk_on_gpu = qk_gpu;
        let t = std::time::Instant::now();
        if !qk_on_gpu {
            for (which, w) in [(0usize, &blk.q_norm), (1usize, &blk.k_norm)] {
                self.norm_rope_w(
                    &mut qkv,
                    n,
                    self.heads,
                    w,
                    &angles,
                    3 * inner,
                    which * inner,
                );
            }
        }
        Self::prof(2, t);
        let t = std::time::Instant::now();
        let mut attn = vec![0f32; n * inner];
        self.attention(
            &qkv,
            n,
            &mut attn,
            qk_on_gpu.then_some((
                &angles[..],
                &blk.q_norm[..],
                &blk.k_norm[..],
                self.qk_eps as f32,
            )),
        );
        Self::prof(3, t);
        let t = std::time::Instant::now();
        let mut proj = vec![0f32; n * hs];
        proj_with_lora(&blk.out, &blk.lora.out, &attn, n, &mut proj, pool);
        Self::prof(4, t);
        // Own slot: the residual is host-side elementwise work with
        // modulation, and billing it to the projection hid which of the
        // two actually costs (they were 9.8 s together at 512×288).
        let t = std::time::Instant::now();
        self.residual(x, hs, &proj, mods, rows, 2);
        Self::prof(8, t);

        self.ffn_tail(blk, x, n, mods, rows);
    }

    /// The FFN half of a block: norm, modulate, SwiGLU, residual. Its
    /// own function so the attention half can return early — the fused
    /// path (`dit_qkv_attention`) finishes attention on the device and
    /// has nowhere to jump to otherwise.
    fn ffn_tail(&self, blk: &Block, x: &mut [f32], n: usize, mods: Option<&[f32]>, rows: &[u32]) {
        let hs = self.hidden;
        let pool = self.pool.as_deref();
        let mut xn = vec![0f32; n * hs];
        let mut proj = vec![0f32; n * hs];
        let t = std::time::Instant::now();
        self.norm_rows(&mut xn, x, n, hs, &blk.norm2, self.eps);
        if let (Some(m), false) = (mods, rows.is_empty()) {
            self.modulate(&mut xn, hs, m, rows, 3, 4);
        }
        Self::prof(0, t);
        let t = std::time::Instant::now();
        // Device-resident FFN: fc1 → SwiGLU → fc2 without the
        // intermediate crossing the bus. At render size that panel is
        // hundreds of megabytes each way, per block, per step.
        // CMF_MMH3_FFN=cpu forces the host chain below.
        if std::env::var("CMF_MMH3_FFN").as_deref() != Ok("cpu")
            && !blk.lora.ffn_any()
            && crate::gpu::enabled_here()
            && n >= 64
        {
            if let (Proj::Q(q1), Proj::Q(q2)) = (&blk.fc1, &blk.fc2) {
                if let (Some((m, i1)), Some((_, i2))) =
                    (q1.mapped_device_gemm(), q2.mapped_device_gemm())
                {
                    let mut fout = vec![0f32; n * hs];
                    if crate::gpu::q4tp_ffn_packed(m, i1, i2, &xn, n, hs, self.ffn, None, &mut fout)
                    {
                        Self::prof(5, t);
                        self.residual(x, hs, &fout, mods, rows, 5);
                        return;
                    }
                }
            }
        }
        let mut gu = vec![0f32; n * 2 * self.ffn];
        proj_with_lora(&blk.fc1, &blk.lora.fc1, &xn, n, &mut gu, pool);
        Self::prof(5, t);
        let t = std::time::Instant::now();
        // SwiGLU: fc1's output is [gate | up] per row.
        let ffn = self.ffn;
        let mut act = vec![0f32; n * ffn];
        let ap = SendPtr(act.as_mut_ptr());
        pool_rows(pool, n, &|lo, hi| {
            for p in lo..hi {
                let row = &gu[p * 2 * ffn..(p + 1) * 2 * ffn];
                let (g, up) = row.split_at(ffn);
                // SAFETY: workers own disjoint token ranges.
                for (o, (&a, &b)) in unsafe { ap.row(p * ffn, ffn) }
                    .iter_mut()
                    .zip(g.iter().zip(up))
                {
                    *o = silu(a) * b;
                }
            }
        });
        Self::prof(6, t);
        let t = std::time::Instant::now();
        proj_with_lora(&blk.fc2, &blk.lora.fc2, &act, n, &mut proj, pool);
        Self::prof(7, t);
        self.residual(x, hs, &proj, mods, rows, 5);
    }

    /// RMSNorm every row of `src` into `dst`, across the pool. One
    /// block does this four times over `n·hidden`; on a 1 879-token
    /// pack that is 40 M elements a block, and it was running on one
    /// thread while forty-seven sat idle.
    fn norm_rows(&self, dst: &mut [f32], src: &[f32], n: usize, hs: usize, w: &[f32], eps: f64) {
        let ptr = SendPtr(dst.as_mut_ptr());
        pool_rows(self.pool.as_deref(), n, &|lo, hi| {
            for p in lo..hi {
                // SAFETY: workers own disjoint token ranges.
                rms_norm_into(&src[p * hs..(p + 1) * hs], w, eps, unsafe {
                    ptr.row(p * hs, hs)
                });
            }
        });
    }

    /// `x = x·(1 + scale[row]) + shift[row]`, per token, across the pool.
    fn modulate(
        &self,
        x: &mut [f32],
        hs: usize,
        mods: &[f32],
        rows: &[u32],
        shift_e: usize,
        scale_e: usize,
    ) {
        let stride = EXPAND * hs;
        let ptr = SendPtr(x.as_mut_ptr());
        pool_rows(self.pool.as_deref(), rows.len(), &|lo, hi| {
            for p in lo..hi {
                let base = rows[p] as usize * stride;
                let shift = &mods[base + shift_e * hs..base + (shift_e + 1) * hs];
                let scale = &mods[base + scale_e * hs..base + (scale_e + 1) * hs];
                // SAFETY: workers own disjoint token ranges.
                for ((v, &sc), &sh) in unsafe { ptr.row(p * hs, hs) }
                    .iter_mut()
                    .zip(scale)
                    .zip(shift)
                {
                    *v = *v * (1.0 + sc) + sh;
                }
            }
        });
    }

    /// The gated residual — or a plain one where there is no modulation
    /// (the token refiner).
    fn residual(
        &self,
        x: &mut [f32],
        hs: usize,
        other: &[f32],
        mods: Option<&[f32]>,
        rows: &[u32],
        gate_e: usize,
    ) {
        let n = x.len() / hs;
        let stride = EXPAND * hs;
        let ptr = SendPtr(x.as_mut_ptr());
        let gated = mods.filter(|_| !rows.is_empty());
        pool_rows(self.pool.as_deref(), n, &|lo, hi| {
            for p in lo..hi {
                // SAFETY: workers own disjoint token ranges.
                let row = unsafe { ptr.row(p * hs, hs) };
                let src = &other[p * hs..(p + 1) * hs];
                match gated {
                    Some(m) => {
                        let base = rows[p] as usize * stride;
                        let gate = &m[base + gate_e * hs..base + (gate_e + 1) * hs];
                        for ((v, &g), &o) in row.iter_mut().zip(gate).zip(src) {
                            *v += g * o;
                        }
                    }
                    None => {
                        for (v, &o) in row.iter_mut().zip(src) {
                            *v += o;
                        }
                    }
                }
            }
        });
    }

    /// One denoise evaluation.
    ///
    /// `video` is `[latents_dim, latent_t, lat_h, lat_w]` and `audio` is
    /// `[audio_dim, 2, audio_t]`, both in the reference's channel-major
    /// order. `text` is the refined `[n, hidden]` stream. Returns the
    /// two velocities, EACH ON ITS OWN SCHEDULE and unscaled.
    pub fn forward(
        &self,
        layout: &Layout,
        text: &[f32],
        video: &[f32],
        audio: &[f32],
        sigma_v: f64,
        cond: &[Vec<f32>],
    ) -> (Vec<f32>, Vec<f32>) {
        let hs = self.hidden;
        let pool = self.pool.as_deref();
        let sigma_v = sigma_v.max(1e-6);
        let t_v = 1.0 - sigma_v;
        let t_a = 1.0 - time_shift_sigma(sigma_v, self.shift_video, self.shift_audio);

        // Distinct timesteps, sorted — the adaLN row index is a position
        // in this list, so the order is part of the contract. A keyframe
        // pins its rows near 1: they are conditions, not noise being
        // removed.
        let has_cond = layout
            .segments
            .iter()
            .any(|s| matches!(s.kind, Kind::Cond | Kind::RefImg));
        let has_ref_audio = layout.segments.iter().any(|s| s.kind == Kind::RefAudio);
        // A reference soundtrack pins to the AUDIO clock's condition
        // timestep, which is its own number.
        let t_cond_a = t_a.max(self.cond_aug_audio);
        // The condition rows' timestep IS the noise-augmentation figure:
        // the reference blends `aug` of the latent with `1 − aug` of
        // noise and then tells the block the row sits at `aug`. Turning
        // the blend off means aug = 1, and the timestep moves with it.
        let t_cond = t_v.max(self.cond_aug);
        // Streaming context is the student's OWN x0, handed back at
        // timestep 0 — the reference passes literal zeros for the clean
        // role, and a chunk conditioned on anything else is conditioned
        // on a frame the previous chunk never produced.
        let has_ctx = layout
            .segments
            .iter()
            .any(|s| matches!(s.kind, Kind::CtxVideo | Kind::CtxAudio));
        let mut ts = vec![t_v, t_a];
        if has_ctx {
            ts.push(0.0);
        }
        if has_cond {
            ts.push(t_cond);
        }
        if has_ref_audio {
            ts.push(t_cond_a);
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ts.dedup();
        let row_of = |t: f64| ts.iter().position(|&x| x == t).unwrap();
        let (row_v, row_a) = (row_of(t_v), row_of(t_a));
        let row_c = if has_cond { row_of(t_cond) } else { 0 };
        let row_ctx = if has_ctx { row_of(0.0) } else { 0 };
        let row_ca = if has_ref_audio { row_of(t_cond_a) } else { 0 };

        // Per-token modulation row: t_row · MODALITIES + tag. The text
        // span is not uniform once a vision block is in it — those
        // positions carry the VIDEO tag.
        let mut rows = vec![0u32; layout.seq_len];
        for s in &layout.segments {
            match s.kind {
                Kind::Text => {
                    for (i, v) in rows[s.start..s.stop].iter_mut().enumerate() {
                        let tag = *layout.text_tags.get(i).unwrap_or(&(TAG_TEXT as u8));
                        *v = (row_v * MODALITIES + tag as usize) as u32;
                    }
                }
                // A reference's rows are conditions too: same timestep
                // near 1, and the modality of whichever stream they
                // belong to.
                Kind::Cond | Kind::RefImg => {
                    for v in rows[s.start..s.stop].iter_mut() {
                        *v = (row_c * MODALITIES + TAG_VIDEO) as u32;
                    }
                }
                Kind::RefAudio => {
                    for v in rows[s.start..s.stop].iter_mut() {
                        *v = (row_ca * MODALITIES + TAG_AUDIO) as u32;
                    }
                }
                Kind::Video => {
                    for v in rows[s.start..s.stop].iter_mut() {
                        *v = (row_v * MODALITIES + TAG_VIDEO) as u32;
                    }
                }
                Kind::Audio => {
                    for v in rows[s.start..s.stop].iter_mut() {
                        *v = (row_a * MODALITIES + TAG_AUDIO) as u32;
                    }
                }
                Kind::CtxVideo => {
                    for v in rows[s.start..s.stop].iter_mut() {
                        *v = (row_ctx * MODALITIES + TAG_VIDEO) as u32;
                    }
                }
                Kind::CtxAudio => {
                    for v in rows[s.start..s.stop].iter_mut() {
                        *v = (row_ctx * MODALITIES + TAG_AUDIO) as u32;
                    }
                }
            }
        }

        // ── embed ──
        let v_rows = patchify_video(
            video,
            self.latents_dim,
            layout.latent_t,
            layout.lat_h,
            layout.lat_w,
        );
        let v_n = v_rows.len() / (self.latents_dim * 4);
        let a_rows = pack_audio(audio, self.audio_dim, layout.audio_t);
        // Condition rows go through the SAME patch projection as the
        // target, so they are patchified the same way — one frame each.
        let vd = self.latents_dim * 4;
        let cond_rows: Vec<Vec<f32>> = cond
            .iter()
            .enumerate()
            .map(|(i, z)| {
                let mut r = patchify_video(z, self.latents_dim, 1, layout.lat_h, layout.lat_w);
                if self.cond_aug < 1.0 {
                    // The reference draws this from a torch generator
                    // reseeded per condition; ours is its own stream, so
                    // the 0.1% it contributes differs — deliberately, and
                    // it is 0.1% of a unit normal.
                    let noise = crate::videogen::gauss_pub(r.len(), 0x5EED ^ i as u64);
                    let a = self.cond_aug as f32;
                    for (v, n) in r.iter_mut().zip(&noise) {
                        *v = a * *v + (1.0 - a) * n;
                    }
                }
                r
            })
            .collect();

        let mut h = vec![0f32; layout.seq_len * hs];
        let mut ci = 0usize;
        for s in layout.segments.iter().filter(|s| s.kind == Kind::Cond) {
            let n = s.stop - s.start;
            let r = cond_rows.get(ci).unwrap_or_else(|| {
                panic!(
                    "layout has {} cond segments, {} latents given",
                    ci + 1,
                    cond_rows.len()
                )
            });
            self.video_patch
                .matmat(r, n, &mut h[s.start * hs..s.stop * hs], pool);
            for row in h[s.start * hs..s.stop * hs].chunks_exact_mut(hs) {
                for (v, &b) in row.iter_mut().zip(&self.video_patch_b) {
                    *v += b;
                }
            }
            ci += 1;
        }
        // The embed writes the whole block — context and current — while
        // the OUTPUT is only the current chunk's rows, which is why these
        // are two different spans in streaming and the same one otherwise.
        let (vblk_start, vblk_stop) = layout.video_block();
        let (ablk_start, ablk_stop) = layout.audio_block();
        let vseg = layout.segment(Kind::Video);
        let aseg = layout.segment(Kind::Audio);
        let tseg = layout.segment(Kind::Text);
        h[tseg.start * hs..tseg.stop * hs].copy_from_slice(&text[..(tseg.stop - tseg.start) * hs]);
        self.video_patch
            .matmat(&v_rows, v_n, &mut h[vblk_start * hs..vblk_stop * hs], pool);
        self.audio_patch.matmat(
            &a_rows,
            ablk_stop - ablk_start,
            &mut h[ablk_start * hs..ablk_stop * hs],
            pool,
        );
        for row in h[vblk_start * hs..vblk_stop * hs].chunks_exact_mut(hs) {
            for (v, &b) in row.iter_mut().zip(&self.video_patch_b) {
                *v += b;
            }
        }
        for row in h[ablk_start * hs..ablk_stop * hs].chunks_exact_mut(hs) {
            for (v, &b) in row.iter_mut().zip(&self.audio_patch_b) {
                *v += b;
            }
        }

        // ── blocks ──
        for (i, blk) in self.blocks.iter().enumerate() {
            let mods = blk.adaln.as_ref().unwrap().eval(&ts, pool);
            self.block_forward(blk, &mut h, layout.seq_len, Some(&mods), &layout.pos, &rows);
            // Per-block watch on the row that dies: growing magnitude is
            // an overflow, a sudden jump from finite to NaN is a kernel.
            if let Ok(w) = std::env::var("CMF_MMH3_WATCHROW") {
                if let Ok(r) = w.parse::<usize>() {
                    let row = &h[r * hs..(r + 1) * hs];
                    let amax = row
                        .iter()
                        .filter(|v| v.is_finite())
                        .fold(0f32, |a, v| a.max(v.abs()));
                    let bad = row.iter().filter(|v| !v.is_finite()).count();
                    if bad > 0 || i == 0 || amax > 1e3 {
                        eprintln!("  watch blk {i}: row {r} absmax {amax:.3e} nonfinite {bad}");
                    }
                }
            }
            if std::env::var_os("CMF_DIT_PROGRESS").is_some() {
                eprint!("\r  block {}/{}", i + 1, self.blocks.len());
            }
        }

        // ── heads ──
        if std::env::var("CMF_MMH3_NANPROBE").is_ok() {
            let bad: Vec<usize> = (0..h.len() / hs)
                .filter(|r| h[r * hs..(r + 1) * hs].iter().any(|v| !v.is_finite()))
                .collect();
            if !bad.is_empty() {
                eprintln!(
                    "  nanprobe rows: {} of {} bad, first 8 {:?}, video seg {}..{}, audio seg {}..{}",
                    bad.len(),
                    h.len() / hs,
                    &bad[..bad.len().min(8)],
                    vseg.start,
                    vseg.stop,
                    aseg.start,
                    aseg.stop
                );
            }
        }
        let fm = self.final_adaln.eval(&ts, pool);
        let mut video_out = vec![0f32; (vseg.stop - vseg.start) * vd];
        let mut audio_out = vec![0f32; (aseg.stop - aseg.start) * self.audio_dim];
        for (seg, row, w, b, dst, dim) in [
            (
                vseg,
                row_v,
                &self.video_out,
                &self.video_out_b,
                &mut video_out,
                vd,
            ),
            (
                aseg,
                row_a,
                &self.audio_out,
                &self.audio_out_b,
                &mut audio_out,
                self.audio_dim,
            ),
        ] {
            let n = seg.stop - seg.start;
            let mut hn = vec![0f32; n * hs];
            for (o, src) in hn
                .chunks_exact_mut(hs)
                .zip(h[seg.start * hs..seg.stop * hs].chunks_exact(hs))
            {
                rms_norm_into(src, &self.final_norm, self.final_eps, o);
            }
            // The final layer's adaLN has one modality, so the row IS
            // the timestep index.
            let shift = &fm[row * 2 * hs..row * 2 * hs + hs];
            let scale = &fm[row * 2 * hs + hs..(row + 1) * 2 * hs];
            for r in hn.chunks_exact_mut(hs) {
                for ((v, &sc), &sh) in r.iter_mut().zip(scale).zip(shift) {
                    *v = *v * (1.0 + sc) + sh;
                }
            }
            // `CMF_MMH3_NANPROBE=1`: which side of the head's GEMM a NaN
            // is on. The stream that dies is the one to instrument, and
            // "the input was already bad" and "this GEMM made it bad" are
            // different bugs in different files.
            let probe = std::env::var("CMF_MMH3_NANPROBE").is_ok();
            if probe {
                let bad_in = hn.iter().filter(|v| !v.is_finite()).count();
                eprintln!(
                    "  nanprobe dim={dim} n={n} in_bad={bad_in} in_absmax={:.4}",
                    hn.iter()
                        .filter(|v| v.is_finite())
                        .fold(0f32, |a, v| a.max(v.abs()))
                );
            }
            w.matmat(&hn, n, dst, pool);
            if probe {
                let bad_out = dst.iter().filter(|v| !v.is_finite()).count();
                eprintln!(
                    "  nanprobe dim={dim} out_bad={bad_out}/{} out_absmax={:.4}",
                    dst.len(),
                    dst.iter()
                        .filter(|v| v.is_finite())
                        .fold(0f32, |a, v| a.max(v.abs()))
                );
            }
            for r in dst.chunks_exact_mut(dim) {
                for (v, &bv) in r.iter_mut().zip(b.iter()) {
                    *v += bv;
                }
            }
        }

        // The reference predicts toward the data and the sampler steps
        // σ down, hence the sign; the audio velocity is returned on its
        // OWN clock rather than pre-scaled by d(σ_a)/d(σ_v).
        // The output covers the rows the sampler owns — which in the
        // chunk-causal layout is the CURRENT chunk, not every visible
        // frame. Sizing this by `layout.latent_t` (context included) fed
        // one chunk's rows into a buffer shaped for five and walked off
        // the end on the first chunk that had any context.
        let out_t = (vseg.stop - vseg.start) / layout.frame_rows;
        let out_at = (aseg.stop - aseg.start) / 2;
        let video = unpatchify_video(
            &video_out,
            self.latents_dim,
            out_t,
            layout.lat_h,
            layout.lat_w,
        );
        let audio = unpack_audio(&audio_out, self.audio_dim, out_at);
        (
            video.iter().map(|&v| -v).collect(),
            audio.iter().map(|&v| -v).collect(),
        )
    }
}

// ── modulation helpers ──────────────────────────────────────────────

/// Rows of `n` items split across pool workers (serial without a pool).
fn pool_rows(pool: Option<&Pool>, n: usize, f: &(dyn Fn(usize, usize) + Sync)) {
    match pool {
        Some(p) => p.run_rows(n, f),
        None => f(0, n),
    }
}

// ── stream (un)packing ──────────────────────────────────────────────

/// `[C, T, H, W]` → `[T·(H/2)·(W/2), C·4]`, the 2×2 spatial patch
/// flattened channel-major-outer as `einsum("nctrhpwq->nthwcrpq")`.
pub fn patchify_video(x: &[f32], c: usize, t: usize, h: usize, w: usize) -> Vec<f32> {
    let (ph, pw) = (h / 2, w / 2);
    let mut out = vec![0f32; t * ph * pw * c * 4];
    let mut i = 0;
    for ti in 0..t {
        for hi in 0..ph {
            for wi in 0..pw {
                for ci in 0..c {
                    for p in 0..2 {
                        for q in 0..2 {
                            out[i] = x[((ci * t + ti) * h + hi * 2 + p) * w + wi * 2 + q];
                            i += 1;
                        }
                    }
                }
            }
        }
    }
    out
}

/// The inverse of `patchify_video`.
pub fn unpatchify_video(rows: &[f32], c: usize, t: usize, h: usize, w: usize) -> Vec<f32> {
    let (ph, pw) = (h / 2, w / 2);
    let mut out = vec![0f32; c * t * h * w];
    let mut i = 0;
    for ti in 0..t {
        for hi in 0..ph {
            for wi in 0..pw {
                for ci in 0..c {
                    for p in 0..2 {
                        for q in 0..2 {
                            out[((ci * t + ti) * h + hi * 2 + p) * w + wi * 2 + q] = rows[i];
                            i += 1;
                        }
                    }
                }
            }
        }
    }
    out
}

/// `[C, 2, T]` → `[2·T, C]`, channel-major: channel 0's frames then
/// channel 1's.
pub fn pack_audio(x: &[f32], c: usize, t: usize) -> Vec<f32> {
    let mut out = vec![0f32; 2 * t * c];
    for ch in 0..2 {
        for ti in 0..t {
            for ci in 0..c {
                out[(ch * t + ti) * c + ci] = x[(ci * 2 + ch) * t + ti];
            }
        }
    }
    out
}

/// The inverse of `pack_audio`.
pub fn unpack_audio(rows: &[f32], c: usize, t: usize) -> Vec<f32> {
    let mut out = vec![0f32; c * 2 * t];
    for ch in 0..2 {
        for ti in 0..t {
            for ci in 0..c {
                out[(ci * 2 + ch) * t + ti] = rows[(ch * t + ti) * c + ci];
            }
        }
    }
    out
}

// ── small shared bits ───────────────────────────────────────────────

struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    /// SAFETY: caller guarantees disjoint `[off, off+len)` per worker.
    #[allow(clippy::mut_from_ref)]
    unsafe fn row(&self, off: usize, len: usize) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(off), len) }
    }
}

fn softmax_inplace(row: &mut [f32]) {
    let mx = row.iter().cloned().fold(f32::MIN, f32::max);
    let mut den = 0f32;
    for r in row.iter_mut() {
        *r = (*r - mx).exp();
        den += *r;
    }
    if den > 0.0 {
        let inv = 1.0 / den;
        for r in row.iter_mut() {
            *r *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    /// A streaming layout whose "current chunk" is the whole clip and
    /// which sees no context must be the bidirectional layout, row for
    /// row. Absolute positions are the whole point of the chunked path:
    /// if this drifts, every chunk after the first is told it is
    /// generating the opening again.
    #[test]
    fn streaming_layout_degenerates_to_the_bidirectional_one() {
        let (lat_h, lat_w, latent_t, audio_t, text_len) = (16, 24, 7, 12, 5);
        let base = Layout::t2va(text_len, latent_t, lat_h, lat_w, audio_t);
        let cur_v: Vec<usize> = (0..latent_t).collect();
        let cur_a: Vec<usize> = (0..audio_t).collect();
        let stream = Layout::streaming(text_len, &[], lat_h, lat_w, &[], &cur_v, &[], &cur_a);
        assert_eq!(stream.seq_len, base.seq_len, "row count");
        assert_eq!(stream.latent_t, base.latent_t);
        assert_eq!(stream.audio_t, base.audio_t);
        for (i, (a, b)) in stream.pos.iter().zip(&base.pos).enumerate() {
            for axis in 0..3 {
                assert!(
                    (a[axis] - b[axis]).abs() < 1e-9,
                    "row {i} axis {axis}: {a:?} vs {b:?}"
                );
            }
        }
        let seg = |l: &Layout, k: Kind| {
            l.segments
                .iter()
                .find(|s| s.kind == k)
                .map(|s| (s.start, s.stop))
        };
        assert_eq!(seg(&stream, Kind::Video), seg(&base, Kind::Video));
        assert_eq!(seg(&stream, Kind::Audio), seg(&base, Kind::Audio));
    }

    /// A later chunk keeps the coordinates it would have had in the whole
    /// clip — that is what makes the chunks line up into one video.
    #[test]
    fn streaming_positions_are_absolute() {
        let (lat_h, lat_w) = (16, 24);
        let whole = Layout::streaming(
            3,
            &[],
            lat_h,
            lat_w,
            &[],
            &(0..9).collect::<Vec<_>>(),
            &[],
            &[],
        );
        let tail = Layout::streaming(
            3,
            &[],
            lat_h,
            lat_w,
            &[],
            &(6..9).collect::<Vec<_>>(),
            &[],
            &[],
        );
        let rows = whole.frame_rows;
        let (ws, _) = whole.video_block();
        let (ts_, _) = tail.video_block();
        for k in 0..3 {
            let a = whole.pos[ws + (6 + k) * rows];
            let b = tail.pos[ts_ + k * rows];
            assert!((a[0] - b[0]).abs() < 1e-9, "frame {k}: {a:?} vs {b:?}");
        }
    }

    use super::*;

    #[test]
    fn schedule_remap_is_an_involution() {
        for &s in &[1e-3, 0.25, 0.5, 0.8, 0.972_973, 1.0] {
            let a = time_shift_sigma(s, 12.0, 3.0);
            let back = time_shift_sigma(a, 3.0, 12.0);
            assert!((back - s).abs() < 1e-9, "{s} -> {a} -> {back}");
        }
    }

    #[test]
    fn slope_matches_a_finite_difference() {
        let h = 1e-6;
        for &s in &[0.2, 0.5, 0.9] {
            let num = (time_shift_sigma(s + h, 12.0, 3.0) - time_shift_sigma(s - h, 12.0, 3.0))
                / (2.0 * h);
            let got = time_shift_slope(s, 12.0, 3.0);
            assert!((num - got).abs() < 1e-5, "{s}: {num} vs {got}");
        }
    }

    #[test]
    fn patchify_round_trips() {
        let (c, t, h, w) = (3usize, 2usize, 4usize, 6usize);
        let x: Vec<f32> = (0..c * t * h * w).map(|i| i as f32).collect();
        let rows = patchify_video(&x, c, t, h, w);
        assert_eq!(rows.len(), t * (h / 2) * (w / 2) * c * 4);
        assert_eq!(unpatchify_video(&rows, c, t, h, w), x);
    }

    #[test]
    fn audio_pack_round_trips() {
        let (c, t) = (5usize, 7usize);
        let x: Vec<f32> = (0..c * 2 * t).map(|i| i as f32).collect();
        let rows = pack_audio(&x, c, t);
        assert_eq!(unpack_audio(&rows, c, t), x);
    }

    #[test]
    fn keyframes_sit_between_the_text_and_the_audio() {
        let (tl, lt, lh, lw, at) = (8usize, 3usize, 8usize, 12usize, 5usize);
        let base = Layout::t2va(tl, lt, lh, lw, at);
        // First and last frame of a 39-frame clip.
        let l = Layout::fl2va(tl, lt, lh, lw, at, &[(0, 39), (38, 39)], &[]);
        assert_eq!(l.cond_rows(), 2 * l.frame_rows);
        assert_eq!(l.seq_len, base.seq_len + 2 * l.frame_rows);
        let kinds: Vec<_> = l.segments.iter().map(|s| s.kind).collect();
        assert_eq!(
            kinds,
            vec![Kind::Text, Kind::Cond, Kind::Cond, Kind::Audio, Kind::Video]
        );
        // A keyframe never advances the cursor: audio and video start
        // where they would have without it.
        let (a0, v0) = (l.segment(Kind::Audio), l.segment(Kind::Video));
        let (ba, bv) = (base.segment(Kind::Audio), base.segment(Kind::Video));
        assert_eq!(l.pos[a0.start][0], base.pos[ba.start][0]);
        assert_eq!(l.pos[v0.start][0], base.pos[bv.start][0]);
        // The first frame's rows sit at the text's end; the last one's a
        // whole clip further on, minus one span.
        let c: Vec<_> = l.segments.iter().filter(|s| s.kind == Kind::Cond).collect();
        assert_eq!(l.pos[c[0].start][0], tl as f64);
        let spans: f64 = (0..lt)
            .map(|k| FRAME_RESCALE * FRAME_PER_TOKEN[k % 5])
            .sum();
        assert!((l.pos[c[1].start][0] - (tl as f64 + spans - FRAME_RESCALE)).abs() < 1e-12);
        // Both share the TARGET spatial grid.
        assert_eq!(l.pos[c[0].start][1], l.pos[v0.start][1]);
        assert_eq!(l.pos[c[0].start][2], l.pos[v0.start][2]);
    }

    #[test]
    fn layout_places_the_target_streams_last() {
        let l = Layout::t2va(8, 3, 8, 12, 5);
        assert_eq!(l.frame_rows, 4 * 6);
        assert_eq!(l.seq_len, 8 + 2 * 5 + 3 * 24);
        assert_eq!(l.segments[0].kind, Kind::Text);
        assert_eq!(l.segments[1].kind, Kind::Audio);
        assert_eq!(l.segments[2].kind, Kind::Video);
        // The video t axis advances by the 1,4,4,4,4 span pattern.
        let v = l.segment(Kind::Video);
        let t0 = l.pos[v.start][0];
        let t1 = l.pos[v.start + l.frame_rows][0];
        assert!((t1 - t0 - FRAME_RESCALE).abs() < 1e-12);
    }
}
