//! Shared MiMo CLI/server ingress. Media stays in ordered content blocks until
//! strict template rendering; only pad rows are replaced by tower embeddings.
//! Text-only requests neither discover nor load multimodal companions.
use crate::{
    Pipeline, media,
    mimo_audio::MimoAudio,
    mimo_mm::{MimoMm, MimoTokenIds},
    mimo_vision::{self, MimoProcessorConfig, MimoVit, VideoSource},
    tokenizer::Tokenizer,
};
use cortiq_core::CmfModel;
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Default)]
pub struct MediaOptions {
    pub companion: Option<PathBuf>,
    pub image_max_pixels: Option<usize>,
    pub video_fps: Option<f64>,
}

pub struct PreparedMimoPrompt {
    pub token_ids: Vec<u32>,
    /// None for pure text: preserve ordinary prefill and prefix reuse.
    pub rows: Option<Vec<f32>>,
}

#[derive(Debug)]
enum Item {
    Image(Value),
    Video(Value),
    Audio(Value),
}

fn record(v: &Value) -> Value {
    if let Some(url) = v.as_str() {
        json!({"url":url})
    } else {
        v.clone()
    }
}

/// Normalize aliases without flattening content or inserting newlines. Reject
/// unknown blocks instead of silently rendering a prompt without its media.
fn normalize(messages: &[Value]) -> Result<(Vec<Value>, Vec<Item>), String> {
    let mut messages = messages.to_vec();
    let mut items = Vec::new();
    for message in &mut messages {
        let Some(content) = message.get_mut("content") else {
            continue;
        };
        if content.is_string() || content.is_null() {
            continue;
        }
        let parts = content
            .as_array_mut()
            .ok_or("message content must be text or an array")?;
        for part in parts {
            if part.is_string() {
                continue;
            }
            let typ = part.get("type").and_then(Value::as_str).unwrap_or("");
            if typ == "image"
                || typ == "image_url"
                || part.get("image").is_some()
                || part.get("image_url").is_some()
            {
                let source = part
                    .get("image_url")
                    .or_else(|| part.get("image"))
                    .unwrap_or(part);
                items.push(Item::Image(record(source)));
                *part = json!({"type":"image"});
            } else if matches!(typ, "audio" | "input_audio" | "audio_url")
                || part.get("input_audio").is_some()
                || part.get("audio_url").is_some()
                || part.get("audio").is_some()
            {
                let source = part
                    .get("input_audio")
                    .or_else(|| part.get("audio_url"))
                    .or_else(|| part.get("audio"))
                    .unwrap_or(part);
                if source
                    .get("format")
                    .and_then(Value::as_str)
                    .is_some_and(|s| s != "wav")
                {
                    return Err("MiMo input_audio currently supports WAV only".into());
                }
                items.push(Item::Audio(record(source)));
                *part = json!({"type":"audio"});
            } else if matches!(typ, "video" | "video_url")
                || part.get("video").is_some()
                || part.get("video_url").is_some()
            {
                let source = part
                    .get("video")
                    .or_else(|| part.get("video_url"))
                    .unwrap_or(part);
                items.push(Item::Video(record(source)));
                *part = json!({"type":"video"});
            } else if part.get("text").and_then(Value::as_str).is_none() {
                return Err(format!("unsupported MiMo content block type '{typ}'"));
            }
        }
    }
    Ok((messages, items))
}

pub fn has_media(messages: &[Value]) -> Result<bool, String> {
    Ok(!normalize(messages)?.1.is_empty())
}

/// Render text-only multipart requests exactly as MiMo's template specifies.
pub fn text_ids(
    tok: &Tokenizer,
    messages: &[Value],
    tools: Option<&[Value]>,
    thinking: Option<bool>,
) -> Result<Vec<u32>, String> {
    let (messages, items) = normalize(messages)?;
    if !items.is_empty() {
        return Err("media requires a MiMo multimodal companion".into());
    }
    if tok.chat_template.is_none() {
        return Err("MiMo requires its embedded chat template".into());
    }
    tok.try_apply_chat_template_json(&messages, tools, thinking)
}

/// Server entry point: use the exact model owned by the checked-out pipeline.
pub fn prepare_for_pipeline(
    pipeline: &Pipeline,
    messages: &[Value],
    tools: Option<&[Value]>,
    thinking: Option<bool>,
    options: &MediaOptions,
) -> Result<PreparedMimoPrompt, String> {
    let model = pipeline
        .model
        .as_ref()
        .ok_or("MiMo media requires a file-backed pipeline")?;
    prepare(model, pipeline, messages, tools, thinking, options)
}

pub fn prepare(
    model: &Arc<CmfModel>,
    pipeline: &Pipeline,
    messages: &[Value],
    tools: Option<&[Value]>,
    thinking: Option<bool>,
    options: &MediaOptions,
) -> Result<PreparedMimoPrompt, String> {
    let (messages, items) = normalize(messages)?;
    let tok = &pipeline.tokenizer;
    if tok.chat_template.is_none() {
        return Err("MiMo requires its embedded chat template".into());
    }
    let ids = tok.try_apply_chat_template_json(&messages, tools, thinking)?;
    if items.is_empty() {
        return Ok(PreparedMimoPrompt {
            token_ids: ids,
            rows: None,
        });
    }
    let mm = MimoMm::attach(model, options.companion.as_deref())?.ok_or(
        "this is a text-only MiMo package; install its .mm.cmf companion or pass --mm PATH",
    )?;
    mm.validate_text(model, tok)?;
    let cfg = mimo_vision::read_mm_config(mm.model())?;
    let processor = MimoProcessorConfig::from_config(&cfg)?;
    let (mut vision, mut audio) = (None, None);
    let (mut images, mut videos, mut audio_tokens) = (Vec::new(), Vec::new(), Vec::new());
    let (mut image_rows, mut video_rows, mut audio_rows) = (Vec::new(), Vec::new(), Vec::new());
    for item in items {
        match item {
            Item::Image(source) => {
                let frame = media::decode_rgb(&media::load_image_bytes(&source)?)?;
                let input =
                    mimo_vision::prepare_image(&frame, &processor, options.image_max_pixels)?;
                if vision.is_none() {
                    vision = Some(MimoVit::from_model(mm.model())?);
                }
                let rows = vision.as_ref().unwrap().forward(&input)?;
                check_rows(&rows, input.tokens(), pipeline.hidden_size)?;
                image_rows.extend(rows);
                images.push(input);
            }
            Item::Video(source) => {
                let path = source
                    .get("url")
                    .or_else(|| source.get("path"))
                    .and_then(Value::as_str)
                    .ok_or("video requires a local Y4M file or frame directory path")?;
                if path.starts_with("http:")
                    || path.starts_with("https:")
                    || path.starts_with("data:")
                {
                    return Err(
                        "MiMo video supports local Y4M files or frame directories only".into(),
                    );
                }
                let fps = source
                    .get("fps")
                    .and_then(Value::as_f64)
                    .or(options.video_fps);
                let src = VideoSource::open(
                    Path::new(path.strip_prefix("file://").unwrap_or(path)),
                    fps,
                )?;
                let input = mimo_vision::prepare_video(&src, &processor, options.image_max_pixels)?;
                if vision.is_none() {
                    vision = Some(MimoVit::from_model(mm.model())?);
                }
                let rows = vision.as_ref().unwrap().forward(&input)?;
                check_rows(&rows, input.tokens(), pipeline.hidden_size)?;
                video_rows.extend(rows);
                videos.push(input);
            }
            Item::Audio(source) => {
                if audio.is_none() {
                    audio = Some(MimoAudio::from_model(mm.model())?);
                }
                let embedded = audio
                    .as_ref()
                    .unwrap()
                    .embed_wav(&media::load_image_bytes(&source)?)?;
                if embedded.dim != pipeline.hidden_size {
                    return Err("audio/text hidden dimension mismatch".into());
                }
                check_rows(&embedded.rows, embedded.n_tokens, pipeline.hidden_size)?;
                audio_tokens.push(embedded.n_tokens);
                audio_rows.extend(embedded.rows);
            }
        }
    }
    let ids = mimo_vision::expand_prompt_ids(
        &ids,
        &images.iter().collect::<Vec<_>>(),
        &videos.iter().collect::<Vec<_>>(),
        &audio_tokens,
        tok,
    )?;
    let rows = inject_rows(
        &ids,
        pipeline.hidden_size,
        [&image_rows, &video_rows, &audio_rows],
        |id| pipeline.embed_id(id),
    )?;
    Ok(PreparedMimoPrompt {
        token_ids: ids,
        rows: Some(rows),
    })
}

fn check_rows(rows: &[f32], n: usize, h: usize) -> Result<(), String> {
    if n == 0 || n.checked_mul(h) != Some(rows.len()) || !rows.iter().all(|v| v.is_finite()) {
        return Err("invalid or non-finite MiMo media embeddings".into());
    }
    Ok(())
}

fn inject_rows(
    ids: &[u32],
    h: usize,
    media: [&[f32]; 3],
    mut embed: impl FnMut(u32) -> Vec<f32>,
) -> Result<Vec<f32>, String> {
    let cap = ids
        .len()
        .checked_mul(h)
        .ok_or("prompt embedding size overflow")?;
    let mut rows = Vec::with_capacity(cap);
    let mut offsets = [0usize; 3];
    let t = MimoTokenIds::PINNED;
    for &id in ids {
        let which = [t.image_pad, t.video_pad, t.audio_pad]
            .iter()
            .position(|&pad| pad == id);
        if let Some(i) = which {
            let slice = media[i]
                .get(offsets[i]..offsets[i] + h)
                .ok_or("media pad/embedding count mismatch")?;
            rows.extend_from_slice(slice);
            offsets[i] += h;
        } else {
            let row = embed(id);
            if row.len() != h {
                return Err("invalid text embedding width".into());
            }
            rows.extend(row);
        }
    }
    if offsets.iter().zip(media).any(|(&n, m)| n != m.len()) {
        return Err("unused MiMo media embedding rows".into());
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn multipart_order_and_aliases_are_preserved() {
        let msgs = vec![
            json!({"role":"user","content":[{"type":"text","text":"before"},
            {"type":"image_url","image_url":{"url":"x.png"}}, {"type":"text","text":"after"},
            {"type":"input_audio","input_audio":{"data":"AA==","format":"wav"}},
            {"type":"video","video":{"path":"frames","fps":1.0}}]}),
        ];
        let (m, items) = normalize(&msgs).unwrap();
        assert_eq!(m[0]["content"][0]["text"], "before");
        assert_eq!(m[0]["content"][2]["text"], "after");
        assert!(matches!(
            &items[..],
            [Item::Image(_), Item::Audio(_), Item::Video(_)]
        ));
        assert!(normalize(&[json!({"content":[{"type":"unknown"}]})]).is_err());
        assert!(
            normalize(&[
                json!({"content":[{"type":"input_audio","input_audio":{"format":"mp3"}}]})
            ])
            .is_err()
        );
    }
    #[test]
    fn injection_consumes_each_modality_in_prompt_order() {
        let t = MimoTokenIds::PINNED;
        let ids = [1, t.audio_pad, t.image_pad, 2, t.video_pad, t.image_pad];
        let media: [&[f32]; 3] = [&[10., 11., 12., 13.], &[20., 21.], &[30., 31.]];
        assert_eq!(
            inject_rows(&ids, 2, media, |id| vec![id as f32; 2]).unwrap(),
            vec![1., 1., 30., 31., 10., 11., 2., 2., 20., 21., 12., 13.]
        );
        assert!(inject_rows(&ids[..5], 2, media, |_| vec![0.; 2]).is_err());
        assert!(inject_rows(&ids, 3, media, |_| vec![0.; 3]).is_err());
    }
}
