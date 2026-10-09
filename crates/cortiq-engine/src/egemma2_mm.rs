//! EmbeddingGemma 2 multimodal inputs: text, images and video — alone or
//! interleaved in one sequence — embedded into the one 768-d space.
//!
//! An input is a text that may hold placeholders (`<|image|>`, `<|video|>`)
//! and the media that fill them, in order. Exactly as
//! `EmbeddingGemma2Processor` lays it out:
//!
//! * each `<|image|>` becomes `<|image>` + N × `<|image|>` + `<image|>`,
//!   N = the image's soft tokens (256 for a square photo at the default
//!   budget of 280);
//! * each `<|video|>` becomes, per sampled frame, `<|image>` + M ×
//!   `<|video|>` + `<image|>` (M = 120 for a 16:9 frame at 140);
//! * the whole is `[BOS] … [EOS]`, at most 8192 tokens (media are never
//!   cut: an input that does not fit is an error);
//! * media given without placeholders in the text come first, separated by
//!   spaces, then the text (the processor's own layout for a text-less
//!   input is `"<|image|> <|image|>"`).
//!
//! The text is tokenized with the placeholders in it (they are added
//! tokens, so they split it exactly where the processor's expansion
//! would), then each placeholder id is expanded. The token rows are
//! `E[id]·sqrt(512)`; the placeholder rows are replaced by the tower's
//! soft tokens; the text model runs over the merged rows.
//!
//! Task prompts follow sentence-transformers: they prefix text-only
//! inputs. An input with media gets a prompt only when one is set on that
//! input itself (then it prefixes its text, placeholders included).

use crate::egemma2::{EmbeddingGemma2, TextInput};
use crate::egemma2_vision::{VisionInput, VisionProcessor, VisionTower};
use crate::media::RgbFrame;
use crate::pool::Pool;
use cortiq_core::CmfModel;
use serde_json::Value;
use std::sync::{Arc, OnceLock};

pub const IMAGE_PLACEHOLDER: &str = "<|image|>";
pub const VIDEO_PLACEHOLDER: &str = "<|video|>";

/// The special token ids of the layout (`config.json`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpecialIds {
    pub image: u32,
    pub video: u32,
    pub boi: u32,
    pub eoi: u32,
}

impl Default for SpecialIds {
    fn default() -> Self {
        SpecialIds {
            image: 258_880,
            video: 258_884,
            boi: 255_999,
            eoi: 258_882,
        }
    }
}

impl SpecialIds {
    pub fn from_config(cfg: &Value) -> Self {
        let d = SpecialIds::default();
        let g = |k: &str, v: u32| cfg.get(k).and_then(|x| x.as_u64()).map_or(v, |x| x as u32);
        SpecialIds {
            image: g("image_token_id", d.image),
            video: g("video_token_id", d.video),
            boi: g("boi_token_id", d.boi),
            eoi: g("eoi_token_id", d.eoi),
        }
    }
}

/// One media item, preprocessed.
#[derive(Clone, Debug)]
pub enum Media {
    Image(VisionInput),
    /// the sampled frames, each at the video budget
    Video(Vec<VisionInput>),
}

impl Media {
    fn placeholder(&self) -> &'static str {
        match self {
            Media::Image(_) => IMAGE_PLACEHOLDER,
            Media::Video(_) => VIDEO_PLACEHOLDER,
        }
    }
}

/// One input: text (may hold placeholders), its media, and its prompt.
#[derive(Clone, Debug, Default)]
pub struct MixedInput {
    pub text: String,
    pub media: Vec<Media>,
    pub prompt_name: Option<String>,
    pub title: Option<String>,
    pub prompt: Option<String>,
}

impl MixedInput {
    pub fn text(t: TextInput) -> Self {
        MixedInput {
            text: t.text,
            media: Vec::new(),
            prompt_name: t.prompt_name,
            title: t.title,
            prompt: t.prompt,
        }
    }

    pub fn image(img: VisionInput) -> Self {
        MixedInput {
            media: vec![Media::Image(img)],
            ..Default::default()
        }
    }

    pub fn video(frames: Vec<VisionInput>) -> Self {
        MixedInput {
            media: vec![Media::Video(frames)],
            ..Default::default()
        }
    }
}

/// The text with every medium's placeholder in place: the text as given
/// when its placeholders match the media (per kind, in order); otherwise —
/// no placeholders at all — the media's placeholders first, space
/// separated, then the text.
pub fn layout_text(text: &str, media: &[Media]) -> Result<String, String> {
    let n_img = text.matches(IMAGE_PLACEHOLDER).count();
    let n_vid = text.matches(VIDEO_PLACEHOLDER).count();
    let want_img = media
        .iter()
        .filter(|m| matches!(m, Media::Image(_)))
        .count();
    let want_vid = media.len() - want_img;
    if n_img == 0 && n_vid == 0 {
        let mut parts: Vec<&str> = media.iter().map(|m| m.placeholder()).collect();
        if !text.is_empty() {
            parts.push(text);
        }
        return Ok(parts.join(" "));
    }
    if n_img != want_img || n_vid != want_vid {
        return Err(format!(
            "the text holds {n_img} {IMAGE_PLACEHOLDER} and {n_vid} {VIDEO_PLACEHOLDER} \
             but {want_img} image(s) and {want_vid} video(s) were given"
        ));
    }
    Ok(text.to_string())
}

/// The order the placeholders occur in `text`: `true` for an image.
fn placeholder_order(text: &str) -> Vec<bool> {
    let mut out = Vec::new();
    let mut rest = text;
    loop {
        let (i, v) = (rest.find(IMAGE_PLACEHOLDER), rest.find(VIDEO_PLACEHOLDER));
        match (i, v) {
            (None, None) => break,
            (Some(a), b) if b.is_none_or(|b| a < b) => {
                out.push(true);
                rest = &rest[a + IMAGE_PLACEHOLDER.len()..];
            }
            (_, Some(b)) => {
                out.push(false);
                rest = &rest[b + VIDEO_PLACEHOLDER.len()..];
            }
            _ => unreachable!(),
        }
    }
    out
}

/// The text encoder with the vision tower beside it (loaded on first use).
pub struct MediaEncoder {
    pub text: EmbeddingGemma2,
    pub proc: VisionProcessor,
    pub ids: SpecialIds,
    model: Arc<CmfModel>,
    pool: Option<Arc<Pool>>,
    vision: OnceLock<Result<VisionTower, String>>,
}

impl MediaEncoder {
    pub fn load(model: &Arc<CmfModel>, pool: Option<Arc<Pool>>) -> Result<Self, String> {
        let text = EmbeddingGemma2::load(model, pool.clone())?;
        let prov = model.header.provenance.clone().unwrap_or(Value::Null);
        let eg = &prov["embedding_gemma2"];
        Ok(MediaEncoder {
            text,
            proc: VisionProcessor::from_config(&eg["processor"]),
            ids: SpecialIds::from_config(&eg["config"]),
            model: model.clone(),
            pool,
            vision: OnceLock::new(),
        })
    }

    /// Does the file carry the vision tower?
    pub fn has_vision(&self) -> bool {
        VisionTower::present(&self.model)
    }

    /// The vision tower, loaded on first use (~0.7 GB of f32 weights).
    pub fn vision(&self) -> Result<&VisionTower, String> {
        self.vision
            .get_or_init(|| VisionTower::load(&self.model, self.pool.clone()))
            .as_ref()
            .map_err(|e| e.clone())
    }

    /// Load the tower now (a server does it at startup).
    pub fn warm_vision(&self) -> Result<(), String> {
        self.vision().map(|_| ())
    }

    /// An image at `budget` soft tokens (`None`: the processor's default).
    pub fn prepare_image(&self, img: &RgbFrame, budget: Option<usize>) -> Result<Media, String> {
        Ok(Media::Image(self.proc.prepare_image(
            img,
            budget.unwrap_or(self.proc.image_budget),
        )?))
    }

    /// Sampled video frames at `budget` soft tokens a frame.
    pub fn prepare_frames(
        &self,
        frames: &[RgbFrame],
        budget: Option<usize>,
    ) -> Result<Media, String> {
        Ok(Media::Video(self.proc.prepare_frames(
            frames,
            budget.unwrap_or(self.proc.video_budget),
        )?))
    }

    /// Token ids of one input, every placeholder expanded.
    pub fn input_ids(&self, input: &MixedInput) -> Result<Vec<u32>, String> {
        Ok(self.layout(input)?.0)
    }

    /// Token ids of one input with every placeholder expanded, and the
    /// order its encoded images / frames (numbered in `media` order, a
    /// video's frames consecutively) occupy the placeholder rows.
    pub fn layout(&self, input: &MixedInput) -> Result<(Vec<u32>, Vec<usize>), String> {
        if input.media.is_empty() {
            let ids = self.text.input_ids(&TextInput {
                text: input.text.clone(),
                prompt_name: input.prompt_name.clone(),
                title: input.title.clone(),
                prompt: input.prompt.clone(),
            })?;
            return Ok((ids, Vec::new()));
        }
        let k = self.proc.pool_k;
        let laid = layout_text(&input.text, &input.media)?;
        let full = if input.prompt_name.is_some() || input.title.is_some() || input.prompt.is_some()
        {
            self.text.prompts().format(
                &laid,
                input.prompt_name.as_deref(),
                input.title.as_deref(),
                input.prompt.as_deref(),
            )?
        } else {
            laid
        };
        // first encoded row of each medium
        let mut first = Vec::with_capacity(input.media.len());
        let mut r = 0usize;
        for m in &input.media {
            first.push(r);
            r += match m {
                Media::Image(_) => 1,
                Media::Video(f) => f.len(),
            };
        }
        // media of each kind, in the order their placeholders occur
        let order = placeholder_order(&full);
        let mut imgs =
            (0..input.media.len()).filter(|&i| matches!(input.media[i], Media::Image(_)));
        let mut vids =
            (0..input.media.len()).filter(|&i| matches!(input.media[i], Media::Video(_)));
        let mut seq_media: Vec<usize> = Vec::with_capacity(order.len());
        for is_img in order {
            let m = if is_img { imgs.next() } else { vids.next() };
            seq_media.push(m.ok_or("a placeholder without a medium")?);
        }
        if imgs.next().is_some() || vids.next().is_some() {
            return Err("a medium without a placeholder".into());
        }
        let raw = self.text.tokenizer().encode(&full);
        let mut ids = Vec::with_capacity(raw.len() + 512);
        let mut rows_order = Vec::with_capacity(r);
        ids.push(self.text.bos);
        let mut next = seq_media.iter();
        for &t in &raw {
            if t == self.ids.image || t == self.ids.video {
                let m = *next.next().ok_or("more placeholder tokens than media")?;
                match &input.media[m] {
                    Media::Image(x) => {
                        ids.push(self.ids.boi);
                        ids.extend(std::iter::repeat_n(self.ids.image, x.n_soft(k)));
                        ids.push(self.ids.eoi);
                        rows_order.push(first[m]);
                    }
                    Media::Video(frames) => {
                        for (j, f) in frames.iter().enumerate() {
                            ids.push(self.ids.boi);
                            ids.extend(std::iter::repeat_n(self.ids.video, f.n_soft(k)));
                            ids.push(self.ids.eoi);
                            rows_order.push(first[m] + j);
                        }
                    }
                }
            } else {
                ids.push(t);
            }
        }
        if next.next().is_some() {
            return Err("a placeholder did not tokenize to its token".into());
        }
        ids.push(self.text.eos);
        if ids.len() > self.text.max_tokens {
            return Err(format!(
                "{} tokens > the model's {} token context (media are not truncated: \
                 lower the image budget or the number of frames)",
                ids.len(),
                self.text.max_tokens
            ));
        }
        Ok((ids, rows_order))
    }

    /// Embed inputs: unit-length vectors of the model's width, in order.
    pub fn embed(&self, inputs: &[MixedInput]) -> Result<Vec<Vec<f32>>, String> {
        let laid: Vec<(Vec<u32>, Vec<usize>)> = inputs
            .iter()
            .enumerate()
            .map(|(i, x)| self.layout(x).map_err(|e| format!("input {i}: {e}")))
            .collect::<Result<_, _>>()?;
        let seqs: Vec<Vec<u32>> = laid.iter().map(|(s, _)| s.clone()).collect();
        if inputs.iter().all(|x| x.media.is_empty()) {
            return self.text.embed_ids(&seqs);
        }
        // every image / frame of every input through the tower, packed
        let mut flat: Vec<VisionInput> = Vec::new();
        let mut base = Vec::with_capacity(inputs.len());
        for x in inputs {
            base.push(flat.len());
            for m in &x.media {
                match m {
                    Media::Image(v) => flat.push(v.clone()),
                    Media::Video(f) => flat.extend(f.iter().cloned()),
                }
            }
        }
        let soft = self.vision()?.encode(&flat)?;
        drop(flat);
        let d = self.text.hidden();
        let (image, video) = (self.ids.image, self.ids.video);
        let mut out: Vec<Option<Vec<f32>>> = vec![None; inputs.len()];
        // pack sequences up to the context into one forward
        let mut start = 0usize;
        while start < inputs.len() {
            let mut end = start;
            let mut toks = 0usize;
            while end < inputs.len()
                && (end == start || toks + seqs[end].len() <= self.text.max_tokens)
            {
                toks += seqs[end].len();
                end += 1;
            }
            let mut x = self.text.embed_rows(&seqs[start..end]);
            let mut row = 0usize;
            for i in start..end {
                let (ids, order) = &laid[i];
                let mut src = order
                    .iter()
                    .flat_map(|&j| soft[base[i] + j].chunks_exact(d));
                for &t in ids {
                    if t == image || t == video {
                        let s = src.next().ok_or("fewer soft tokens than placeholders")?;
                        x[row * d..(row + 1) * d].copy_from_slice(s);
                    }
                    row += 1;
                }
                if src.next().is_some() {
                    return Err("more soft tokens than placeholders".into());
                }
            }
            let lens: Vec<usize> = seqs[start..end].iter().map(|s| s.len()).collect();
            for (j, v) in self.text.embed_merged(x, &lens).into_iter().enumerate() {
                out[start + j] = Some(v);
            }
            start = end;
        }
        Ok(out.into_iter().map(|v| v.unwrap()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(pw: usize, ph: usize) -> Media {
        Media::Image(VisionInput {
            pw,
            ph,
            pixels: Vec::new(),
        })
    }

    fn vid(frames: usize) -> Media {
        Media::Video(
            (0..frames)
                .map(|_| VisionInput {
                    pw: 45,
                    ph: 24,
                    pixels: Vec::new(),
                })
                .collect(),
        )
    }

    #[test]
    fn layout_places_media_without_placeholders_first() {
        assert_eq!(layout_text("", &[img(3, 3)]).unwrap(), "<|image|>");
        assert_eq!(
            layout_text("", &[img(3, 3), img(3, 3)]).unwrap(),
            "<|image|> <|image|>"
        );
        assert_eq!(
            layout_text("a fox", &[img(3, 3), vid(2)]).unwrap(),
            "<|image|> <|video|> a fox"
        );
        assert_eq!(
            layout_text("A <|image|> b", &[img(3, 3)]).unwrap(),
            "A <|image|> b"
        );
        assert!(layout_text("A <|image|> <|image|>", &[img(3, 3)]).is_err());
        assert!(layout_text("A <|video|>", &[img(3, 3)]).is_err());
    }

    #[test]
    fn placeholder_order_interleaves_kinds() {
        assert_eq!(
            placeholder_order("x <|video|> y <|image|><|image|> <|video|>"),
            vec![false, true, true, false]
        );
        assert!(placeholder_order("plain").is_empty());
    }
}
