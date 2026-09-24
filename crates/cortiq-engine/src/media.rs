//! Media ingress shared by the vision front ends: fetching image bytes from
//! the request shapes the CLI and the OpenAI server accept, and decoding them
//! to 8-bit RGB.
//!
//! `load_image_bytes` used to live in `dsv41_vision`; it moved here unchanged
//! so MiMo (and any later tower) reads images exactly the way DeepSeek-V4.1
//! does. `dsv41_vision::load_image_bytes` re-exports it.

use base64::Engine as _;
use serde_json::Value;

/// Load bytes from raw/base64 data, Anthropic source records, data URLs,
/// HTTP(S), or local paths.  This follows the official loader's precedence.
pub fn load_image_bytes(record: &Value) -> Result<Vec<u8>, String> {
    let map = record
        .as_object()
        .ok_or_else(|| "image record must be an object".to_string())?;
    if let Some(data) = map.get("data") {
        if let Some(s) = data.as_str() {
            return base64::engine::general_purpose::STANDARD
                .decode(s)
                .map_err(|e| format!("invalid base64 image data: {e}"));
        }
    }
    if let Some(source) = map.get("source").and_then(Value::as_object) {
        if let Some(data) = source.get("data").and_then(Value::as_str) {
            return base64::engine::general_purpose::STANDARD
                .decode(data)
                .map_err(|e| format!("invalid base64 Anthropic image data: {e}"));
        }
        if let Some(url) = source.get("url").and_then(Value::as_str) {
            return load_image_bytes(&serde_json::json!({"url": url}));
        }
    }
    let url = map.get("url").and_then(Value::as_str).ok_or_else(|| {
        format!(
            "image record has no data/source/url (keys: {:?})",
            map.keys()
        )
    })?;
    if let Some((header, payload)) = url.split_once(',').filter(|(h, _)| h.starts_with("data:")) {
        if !header.contains(";base64") {
            return Err(format!("unsupported data URL encoding: {header}"));
        }
        return base64::engine::general_purpose::STANDARD
            .decode(payload)
            .map_err(|e| format!("invalid data URL image: {e}"));
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        let response = ureq::get(url)
            .timeout(std::time::Duration::from_secs(30))
            .call()
            .map_err(|e| format!("image download failed: {e}"))?;
        let mut reader = response.into_reader();
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut reader, &mut bytes)
            .map_err(|e| format!("image download read failed: {e}"))?;
        return Ok(bytes);
    }
    let path = url.strip_prefix("file://").unwrap_or(url);
    std::fs::read(path).map_err(|e| format!("image path '{path}' could not be read: {e}"))
}

/// An 8-bit RGB raster, row-major `[height][width][3]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbFrame {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl RgbFrame {
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Result<Self, String> {
        if width == 0 || height == 0 {
            return Err(format!(
                "image dimensions must be non-zero, got {width}x{height}"
            ));
        }
        let want = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| "image size overflow".to_string())?;
        if data.len() != want {
            return Err(format!(
                "RGB buffer has {} bytes, expected {want} for {width}x{height}",
                data.len()
            ));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }
}

/// Decode an encoded image (PNG, JPEG, GIF, WebP, or binary PPM `P6`) to
/// RGB. Alpha is DROPPED, not composited: vLLM and sglang's PIL path both
/// call `convert("RGB")`, which is what `to_rgb8` does. Grayscale is
/// replicated to three channels, as PIL does.
pub fn decode_rgb(bytes: &[u8]) -> Result<RgbFrame, String> {
    if bytes.starts_with(b"P6") {
        return decode_ppm(bytes);
    }
    let img = image::load_from_memory(bytes).map_err(|e| format!("image decode failed: {e}"))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    RgbFrame::new(w, h, rgb.into_raw())
}

/// Read a file from disk and decode it with [`decode_rgb`].
pub fn read_rgb(path: &std::path::Path) -> Result<RgbFrame, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    decode_rgb(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// Binary PPM (`P6`, maxval ≤ 255). `ffmpeg -i x.mp4 frames/%06d.ppm` is the
/// cheapest way to hand a video to the frame-directory source, and the
/// `image` crate is built here without its PNM decoder.
fn decode_ppm(bytes: &[u8]) -> Result<RgbFrame, String> {
    let mut pos = 2usize;
    let mut fields = [0usize; 3];
    for field in &mut fields {
        loop {
            while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
                pos += 1;
            }
            if pos < bytes.len() && bytes[pos] == b'#' {
                while pos < bytes.len() && bytes[pos] != b'\n' {
                    pos += 1;
                }
                continue;
            }
            break;
        }
        let start = pos;
        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
            pos += 1;
        }
        if start == pos {
            return Err("PPM header is malformed".into());
        }
        *field = std::str::from_utf8(&bytes[start..pos])
            .ok()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| "PPM header number is malformed".to_string())?;
    }
    // Exactly one whitespace byte separates the header from the raster.
    pos += 1;
    let [w, h, maxval] = fields;
    if maxval == 0 || maxval > 255 {
        return Err(format!("PPM maxval {maxval} is not supported (8-bit only)"));
    }
    let n = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| "PPM size overflow".to_string())?;
    let raster = bytes
        .get(pos..pos + n)
        .ok_or_else(|| format!("PPM raster is truncated: need {n} bytes"))?;
    let data = if maxval == 255 {
        raster.to_vec()
    } else {
        raster
            .iter()
            .map(|&v| ((v as u32 * 255 + maxval as u32 / 2) / maxval as u32).min(255) as u8)
            .collect()
    };
    RgbFrame::new(w, h, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppm_round_trip() {
        let mut bytes = b"P6\n# c\n2 1\n255\n".to_vec();
        bytes.extend_from_slice(&[1, 2, 3, 4, 5, 6]);
        let f = decode_rgb(&bytes).unwrap();
        assert_eq!((f.width, f.height), (2, 1));
        assert_eq!(f.data, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn data_url_decodes() {
        let rec = serde_json::json!({"url": "data:image/png;base64,AAEC"});
        assert_eq!(load_image_bytes(&rec).unwrap(), vec![0, 1, 2]);
    }
}
