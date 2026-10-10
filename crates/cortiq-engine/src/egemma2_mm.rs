//! EmbeddingGemma 2 multimodal inputs: text, images, video and audio —
//! alone or interleaved in one sequence — embedded into the one 768-d space.
//!
//! An input is a text that may hold placeholders (`<|image|>`, `<|video|>`,
//! `<|audio|>`) and the media that fill them, in order. Exactly as
//! `EmbeddingGemma2Processor` lays it out:
//!
//! * each `<|image|>` becomes `<|image>` + N × `<|image|>` + `<image|>`,
//!   N = the image's soft tokens (256 for a square photo at the default
//!   budget of 280);
//! * each `<|video|>` becomes, per sampled frame, `<|image>` + M ×
//!   `<|video|>` + `<image|>` (M = 120 for a 16:9 frame at 140);
//! * each `<|audio|>` becomes `<|audio>` + K × `<|audio|>` + `<audio|>`,
//!   K = 25 per second of the clip (mono 16 kHz, cut at 30 s);
//! * the whole is `[BOS] … [EOS]`, at most 8192 tokens (media are never
//!   cut: an input that does not fit is an error);
//! * media given without placeholders in the text come first — images, then
//!   videos, then audio clips, separated by spaces — then the text (the
//!   processor's own layout for a text-less input is `"<|image|> <|image|>"`).
//!
//! The text is tokenized with the placeholders in it (they are added
//! tokens, so they split it exactly where the processor's expansion
//! would), then each placeholder id is expanded. The token rows are
//! `E[id]·sqrt(512)`; the placeholder rows are replaced by the towers' soft
//! tokens (vision for images and frames, audio for clips); the text model
//! runs over the merged rows. Every input of a call goes through each tower
//! together, then through the text model packed up to the context.
//!
//! Task prompts follow sentence-transformers: they prefix text-only
//! inputs. An input with media gets a prompt only when one is set on that
//! input itself (then it prefixes its text, placeholders included).

use crate::egemma2::{EmbeddingGemma2, TextInput};
use crate::egemma2_audio::{self as ea, AudioTower};
use crate::egemma2_vision::{VisionInput, VisionProcessor, VisionTower};
use crate::media::RgbFrame;
use crate::pool::Pool;
use cortiq_core::CmfModel;
use serde_json::Value;
use std::sync::{Arc, OnceLock};

pub const IMAGE_PLACEHOLDER: &str = "<|image|>";
pub const VIDEO_PLACEHOLDER: &str = "<|video|>";
pub const AUDIO_PLACEHOLDER: &str = ea::PLACEHOLDER;

/// The special token ids of the layout (`config.json`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpecialIds {
    pub image: u32,
    pub video: u32,
    pub boi: u32,
    pub eoi: u32,
    pub audio: u32,
    pub boa: u32,
    pub eoa: u32,
}

impl Default for SpecialIds {
    fn default() -> Self {
        SpecialIds {
            image: 258_880,
            video: 258_884,
            boi: 255_999,
            eoi: 258_882,
            audio: ea::AUDIO_TOKEN,
            boa: ea::BOA_TOKEN,
            eoa: ea::EOA_TOKEN,
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
            audio: g("audio_token_id", d.audio),
            boa: g("boa_token_id", d.boa),
            // config.json spells this one `eoa_token_index`
            eoa: g("eoa_token_id", g("eoa_token_index", d.eoa)),
        }
    }
}

/// One media item, preprocessed.
#[derive(Clone, Debug)]
pub enum Media {
    Image(VisionInput),
    /// the sampled frames, each at the video budget
    Video(Vec<VisionInput>),
    /// a clip, mono 16 kHz (cut at 30 s by the tower)
    Audio(Vec<f32>),
}

/// The three kinds of placeholder, in the processor's text-less order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    Image,
    Video,
    Audio,
}

impl Kind {
    const ALL: [Kind; 3] = [Kind::Image, Kind::Video, Kind::Audio];

    fn placeholder(self) -> &'static str {
        match self {
            Kind::Image => IMAGE_PLACEHOLDER,
            Kind::Video => VIDEO_PLACEHOLDER,
            Kind::Audio => AUDIO_PLACEHOLDER,
        }
    }
}

impl Media {
    fn kind(&self) -> Kind {
        match self {
            Media::Image(_) => Kind::Image,
            Media::Video(_) => Kind::Video,
            Media::Audio(_) => Kind::Audio,
        }
    }

    /// images / frames this medium sends through the vision tower
    fn vision_rows(&self) -> usize {
        match self {
            Media::Image(_) => 1,
            Media::Video(f) => f.len(),
            Media::Audio(_) => 0,
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

    /// A clip (mono 16 kHz) on its own.
    pub fn audio(wave: Vec<f32>) -> Self {
        MixedInput {
            media: vec![Media::Audio(wave)],
            ..Default::default()
        }
    }
}

/// The text with every medium's placeholder in place: the text as given
/// when its placeholders match the media (per kind, in order); otherwise —
/// no placeholders at all — the media's placeholders first (images, videos,
/// then audio, as the processor lays out a text-less input), space
/// separated, then the text.
pub fn layout_text(text: &str, media: &[Media]) -> Result<String, String> {
    let found: Vec<usize> = Kind::ALL
        .iter()
        .map(|k| text.matches(k.placeholder()).count())
        .collect();
    let want: Vec<usize> = Kind::ALL
        .iter()
        .map(|&k| media.iter().filter(|m| m.kind() == k).count())
        .collect();
    if found.iter().all(|&n| n == 0) {
        let mut kinds: Vec<Kind> = media.iter().map(Media::kind).collect();
        kinds.sort();
        let mut parts: Vec<&str> = kinds.iter().map(|k| k.placeholder()).collect();
        if !text.is_empty() {
            parts.push(text);
        }
        return Ok(parts.join(" "));
    }
    if found != want {
        return Err(format!(
            "the text holds {} {IMAGE_PLACEHOLDER}, {} {VIDEO_PLACEHOLDER} and {} \
             {AUDIO_PLACEHOLDER} but {} image(s), {} video(s) and {} audio clip(s) were given",
            found[0], found[1], found[2], want[0], want[1], want[2]
        ));
    }
    Ok(text.to_string())
}

/// The kinds of the placeholders in `text`, in the order they occur.
fn placeholder_order(text: &str) -> Vec<Kind> {
    let mut out = Vec::new();
    let mut rest = text;
    loop {
        let next = Kind::ALL
            .iter()
            .filter_map(|&k| rest.find(k.placeholder()).map(|at| (at, k)))
            .min();
        let Some((at, k)) = next else { break };
        out.push(k);
        rest = &rest[at + k.placeholder().len()..];
    }
    out
}

/// Where an input's soft tokens come from, in sequence order: the vision
/// rows (its images and frames, numbered in `media` order, a video's
/// frames consecutively) and the audio clips (numbered in `media` order).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Placement {
    pub vision: Vec<usize>,
    pub audio: Vec<usize>,
}

/// The text encoder with the vision and audio towers beside it (each
/// loaded on first use).
pub struct MediaEncoder {
    pub text: EmbeddingGemma2,
    pub proc: VisionProcessor,
    pub ids: SpecialIds,
    model: Arc<CmfModel>,
    pool: Option<Arc<Pool>>,
    vision: OnceLock<Result<VisionTower, String>>,
    audio: OnceLock<Result<AudioTower, String>>,
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
            audio: OnceLock::new(),
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

    /// Does the file carry the audio tower?
    pub fn has_audio(&self) -> bool {
        ea::has_audio(&self.model)
    }

    /// The audio tower, loaded on first use (~1.2 GB of f32 weights).
    pub fn audio(&self) -> Result<&AudioTower, String> {
        self.audio
            .get_or_init(|| {
                if !self.has_audio() {
                    return Err("this file carries no audio tower (a text-only pack?)".into());
                }
                AudioTower::load(&self.model, self.pool.clone())
            })
            .as_ref()
            .map_err(|e| e.clone())
    }

    /// Load the audio tower now.
    pub fn warm_audio(&self) -> Result<(), String> {
        self.audio().map(|_| ())
    }

    /// Load the towers the inputs need (their one-time load stays out of
    /// a timed forward).
    pub fn warm_for(&self, inputs: &[MixedInput]) -> Result<(), String> {
        let media = || inputs.iter().flat_map(|x| x.media.iter());
        if media().any(|m| m.vision_rows() > 0) {
            self.warm_vision()?;
        }
        if media().any(|m| matches!(m, Media::Audio(_))) {
            self.warm_audio()?;
        }
        Ok(())
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

    /// Token ids of one input with every placeholder expanded, and where
    /// its soft tokens come from, in sequence order.
    pub fn layout(&self, input: &MixedInput) -> Result<(Vec<u32>, Placement), String> {
        if input.media.is_empty() {
            let ids = self.text.input_ids(&TextInput {
                text: input.text.clone(),
                prompt_name: input.prompt_name.clone(),
                title: input.title.clone(),
                prompt: input.prompt.clone(),
            })?;
            return Ok((ids, Placement::default()));
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
        // first vision row / audio clip of each medium
        let mut first = Vec::with_capacity(input.media.len());
        let (mut r, mut a) = (0usize, 0usize);
        for m in &input.media {
            match m {
                Media::Audio(_) => {
                    first.push(a);
                    a += 1;
                }
                _ => {
                    first.push(r);
                    r += m.vision_rows();
                }
            }
        }
        // media of each kind, in the order their placeholders occur
        let mut of_kind: Vec<_> = Kind::ALL
            .iter()
            .map(|&kd| {
                (0..input.media.len())
                    .filter(move |&i| input.media[i].kind() == kd)
                    .collect::<Vec<_>>()
                    .into_iter()
            })
            .collect();
        let mut seq_media: Vec<usize> = Vec::new();
        for kd in placeholder_order(&full) {
            let m = of_kind[kd as usize].next();
            seq_media.push(m.ok_or("a placeholder without a medium")?);
        }
        if of_kind.iter_mut().any(|it| it.next().is_some()) {
            return Err("a medium without a placeholder".into());
        }
        let raw = self.text.tokenizer().encode(&full);
        let mut ids = Vec::with_capacity(raw.len() + 512);
        let mut place = Placement::default();
        ids.push(self.text.bos);
        let mut next = seq_media.iter();
        for &t in &raw {
            if t == self.ids.image || t == self.ids.video || t == self.ids.audio {
                let m = *next.next().ok_or("more placeholder tokens than media")?;
                let want = match input.media[m].kind() {
                    Kind::Image => self.ids.image,
                    Kind::Video => self.ids.video,
                    Kind::Audio => self.ids.audio,
                };
                if t != want {
                    return Err("a placeholder did not tokenize to its token".into());
                }
                match &input.media[m] {
                    Media::Image(x) => {
                        ids.push(self.ids.boi);
                        ids.extend(std::iter::repeat_n(self.ids.image, x.n_soft(k)));
                        ids.push(self.ids.eoi);
                        place.vision.push(first[m]);
                    }
                    Media::Video(frames) => {
                        for (j, f) in frames.iter().enumerate() {
                            ids.push(self.ids.boi);
                            ids.extend(std::iter::repeat_n(self.ids.video, f.n_soft(k)));
                            ids.push(self.ids.eoi);
                            place.vision.push(first[m] + j);
                        }
                    }
                    Media::Audio(w) => {
                        ids.push(self.ids.boa);
                        ids.extend(std::iter::repeat_n(self.ids.audio, ea::num_tokens(w.len())));
                        ids.push(self.ids.eoa);
                        place.audio.push(first[m]);
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
                 lower the image budget, the number of frames or the audio length)",
                ids.len(),
                self.text.max_tokens
            ));
        }
        Ok((ids, place))
    }

    /// Embed inputs: unit-length vectors of the model's width, in order.
    pub fn embed(&self, inputs: &[MixedInput]) -> Result<Vec<Vec<f32>>, String> {
        let laid: Vec<(Vec<u32>, Placement)> = inputs
            .iter()
            .enumerate()
            .map(|(i, x)| self.layout(x).map_err(|e| format!("input {i}: {e}")))
            .collect::<Result<_, _>>()?;
        let seqs: Vec<Vec<u32>> = laid.iter().map(|(s, _)| s.clone()).collect();
        if inputs.iter().all(|x| x.media.is_empty()) {
            return self.text.embed_ids(&seqs);
        }
        // every image / frame of every input through the vision tower, and
        // every clip through the audio tower, each packed
        let mut flat: Vec<VisionInput> = Vec::new();
        let mut clips: Vec<&[f32]> = Vec::new();
        let mut base = Vec::with_capacity(inputs.len());
        for x in inputs {
            base.push((flat.len(), clips.len()));
            for m in &x.media {
                match m {
                    Media::Image(v) => flat.push(v.clone()),
                    Media::Video(f) => flat.extend(f.iter().cloned()),
                    Media::Audio(w) => clips.push(w),
                }
            }
        }
        let d = self.text.hidden();
        let soft = if flat.is_empty() {
            Vec::new()
        } else {
            self.vision()?.encode(&flat)?
        };
        drop(flat);
        let asoft = if clips.is_empty() {
            Vec::new()
        } else {
            let tower = self.audio()?;
            if tower.out_dim != d {
                return Err(format!(
                    "audio soft tokens are {}-d, the text encoder is {d}-d",
                    tower.out_dim
                ));
            }
            tower.soft_tokens(&clips)
        };
        let (image, video, audio) = (self.ids.image, self.ids.video, self.ids.audio);
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
                let (ids, place) = &laid[i];
                let (vb, ab) = base[i];
                let mut vsrc = place
                    .vision
                    .iter()
                    .flat_map(|&j| soft[vb + j].chunks_exact(d));
                let mut asrc = place
                    .audio
                    .iter()
                    .flat_map(|&j| asoft[ab + j].chunks_exact(d));
                for &t in ids {
                    let src = if t == image || t == video {
                        Some(vsrc.next().ok_or("fewer soft tokens than placeholders")?)
                    } else if t == audio {
                        Some(asrc.next().ok_or("fewer audio tokens than placeholders")?)
                    } else {
                        None
                    };
                    if let Some(s) = src {
                        x[row * d..(row + 1) * d].copy_from_slice(s);
                    }
                    row += 1;
                }
                if vsrc.next().is_some() || asrc.next().is_some() {
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
        // the processor's text-less order: images, videos, audio
        let clip = Media::Audio(vec![0.0; 1600]);
        assert_eq!(
            layout_text("", &[clip.clone(), vid(1), img(3, 3)]).unwrap(),
            "<|image|> <|video|> <|audio|>"
        );
        assert_eq!(
            layout_text("said: <|audio|>", std::slice::from_ref(&clip)).unwrap(),
            "said: <|audio|>"
        );
        assert!(layout_text("said: <|audio|> <|audio|>", &[clip]).is_err());
    }

    #[test]
    fn placeholder_order_interleaves_kinds() {
        assert_eq!(
            placeholder_order("x <|video|> y <|image|><|audio|><|image|> <|video|>"),
            vec![
                Kind::Video,
                Kind::Image,
                Kind::Audio,
                Kind::Image,
                Kind::Video
            ]
        );
        assert!(placeholder_order("plain").is_empty());
    }
}
