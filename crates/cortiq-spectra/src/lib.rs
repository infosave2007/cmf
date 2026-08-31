//! Deterministic, bounded-memory dual-energy X-ray colorization.
//!
//! The detector is modelled as a row stream. Calibration is reduced to one
//! vector per energy before the stream starts; the colorizer then keeps a
//! three-row halo and emits only mature rows. Consequently a caller may use
//! any chunk size without changing a byte of the result.

use cortiq_core::{
    CmfHeader, CmfModel, LayerType, ModelArch, NormStyle, QuantType, TensorDtype, TensorSpec,
};
use image::{
    DynamicImage, GenericImageView, ImageBuffer, ImageReader, Luma, Rgb, RgbImage, imageops,
};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::io::Cursor;
use std::path::Path;

pub const REFUSAL_INVALID_CALIBRATION: u8 = 1;
pub const REFUSAL_SATURATED: u8 = 2;
pub const REFUSAL_NONFINITE: u8 = 4;

#[derive(Debug, thiserror::Error)]
pub enum SpectraError {
    #[error("image decode: {0}")]
    Image(#[from] image::ImageError),
    #[error("{0}")]
    Invalid(String),
    #[error("CMF profile: {0}")]
    Cmf(#[from] cortiq_core::CmfError),
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InvalidDenominator {
    Refuse,
    Clip,
}

impl Default for InvalidDenominator {
    fn default() -> Self {
        Self::Refuse
    }
}

/// Output-changing deterministic classifier/display constants.  They are
/// carried in CMF provenance so a loaded scanner profile does not depend on
/// source-code defaults.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColorParameters {
    pub ratio_center: f64,
    pub ratio_span: f64,
    pub confidence_scale: f64,
    pub saturation_exponent: f64,
    pub palette_orange: [u8; 3],
    pub palette_green: [u8; 3],
    pub palette_blue: [u8; 3],
}

impl Default for ColorParameters {
    fn default() -> Self {
        Self {
            ratio_center: 0.92,
            ratio_span: 0.37,
            confidence_scale: 0.9,
            saturation_exponent: 0.62,
            palette_orange: [255, 102, 0],
            palette_green: [75, 215, 95],
            palette_blue: [28, 112, 255],
        }
    }
}

impl ColorParameters {
    fn validate(&self) -> Result<(), SpectraError> {
        if !self.ratio_center.is_finite()
            || !self.ratio_span.is_finite()
            || self.ratio_span <= 0.0
            || !self.confidence_scale.is_finite()
            || self.confidence_scale < 0.0
            || !self.saturation_exponent.is_finite()
            || self.saturation_exponent <= 0.0
        {
            return Err(SpectraError::Invalid(
                "scanner colour parameters are non-finite or out of range".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    pub width: usize,
    pub gain_high: Vec<f64>,
    pub offset_high: Vec<f64>,
    pub gain_low: Vec<f64>,
    pub offset_low: Vec<f64>,
    #[serde(default = "default_denominator_epsilon")]
    pub denominator_epsilon: f64,
    #[serde(default)]
    pub invalid_denominator: InvalidDenominator,
    #[serde(default)]
    pub color_parameters: ColorParameters,
    /// Optional scanner-owned colour transfer table.  The table is fitted
    /// from calibrated output/reference pairs, persisted in the CMF profile,
    /// and applied after the bounded stream filter.  Keeping it optional
    /// preserves the physics-only baseline for callers that do not have a
    /// fitted display calibration yet.
    #[serde(default)]
    pub color_lut: Option<ColorLut>,
}

/// Compact deterministic 3-D RGB transfer table.  `size` bins are used for
/// each input channel and `values` stores interleaved RGB bytes in
/// `(r * size + g) * size + b` order.  A 128³ table occupies 6 MiB and is a
/// bounded profile asset, not a spatial/suitcase-specific mask.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ColorLut {
    pub size: usize,
    pub values: Vec<u8>,
}

impl ColorLut {
    pub const DEFAULT_SIZE: usize = 128;

    pub fn validate(&self) -> Result<(), SpectraError> {
        if !(2..=256).contains(&self.size) {
            return Err(SpectraError::Invalid(
                "colour LUT size must be in 2..=256".into(),
            ));
        }
        let entries = self
            .size
            .checked_mul(self.size)
            .and_then(|n| n.checked_mul(self.size))
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| SpectraError::Invalid("colour LUT dimensions overflow".into()))?;
        if self.values.len() != entries {
            return Err(SpectraError::Invalid(format!(
                "colour LUT has {} bytes, expected {entries}",
                self.values.len()
            )));
        }
        Ok(())
    }

    fn map_rgb(&self, rgb: &mut [u8]) {
        debug_assert!(rgb.len() % 3 == 0);
        let size = self.size;
        for px in rgb.chunks_exact_mut(3) {
            let bin = |v: u8| ((usize::from(v) * size) / 256).min(size - 1);
            let i = ((bin(px[0]) * size + bin(px[1])) * size + bin(px[2])) * 3;
            px.copy_from_slice(&self.values[i..i + 3]);
        }
    }
}

fn default_denominator_epsilon() -> f64 {
    1.0
}

impl Calibration {
    /// Average repeated gain/offset rows. The input slices are row-major and
    /// all four arrays must have the same width.
    pub fn from_rows(
        width: usize,
        gain_high: &[u16],
        gain_low: &[u16],
        offset_high: &[u16],
        offset_low: &[u16],
    ) -> Result<Self, SpectraError> {
        fn mean(width: usize, values: &[u16]) -> Result<Vec<f64>, SpectraError> {
            if width == 0 {
                return Err(SpectraError::Invalid(
                    "calibration width must be positive".into(),
                ));
            }
            if values.is_empty() || values.len() % width != 0 {
                return Err(SpectraError::Invalid(format!(
                    "calibration has {} values, not a non-empty multiple of width {width}",
                    values.len()
                )));
            }
            let rows = values.len() / width;
            let mut out = vec![0.0; width];
            for row in values.chunks_exact(width) {
                for (dst, &v) in out.iter_mut().zip(row) {
                    *dst += f64::from(v);
                }
            }
            for x in &mut out {
                *x /= rows as f64;
            }
            Ok(out)
        }
        let gh = mean(width, gain_high)?;
        let gl = mean(width, gain_low)?;
        let oh = mean(width, offset_high)?;
        let ol = mean(width, offset_low)?;
        Ok(Self {
            width,
            gain_high: gh,
            offset_high: oh,
            gain_low: gl,
            offset_low: ol,
            denominator_epsilon: 1.0,
            invalid_denominator: InvalidDenominator::Refuse,
            color_parameters: ColorParameters::default(),
            color_lut: None,
        })
    }

    fn validate(&self) -> Result<(), SpectraError> {
        if self.width == 0
            || self.width > u32::MAX as usize
            || self.gain_high.len() != self.width
            || self.gain_low.len() != self.width
            || self.offset_high.len() != self.width
            || self.offset_low.len() != self.width
        {
            return Err(SpectraError::Invalid(
                "calibration vectors have inconsistent width".into(),
            ));
        }
        if !self.denominator_epsilon.is_finite() || self.denominator_epsilon < 0.0 {
            return Err(SpectraError::Invalid(
                "calibration denominator_epsilon must be finite and non-negative".into(),
            ));
        }
        if let Some(lut) = &self.color_lut {
            lut.validate()?;
        }
        self.color_parameters.validate()?;
        Ok(())
    }

    pub fn invalid_detectors(&self) -> usize {
        if !self.denominator_epsilon.is_finite() || self.denominator_epsilon < 0.0 {
            return self.width;
        }
        self.gain_high
            .iter()
            .zip(&self.offset_high)
            .zip(self.gain_low.iter().zip(&self.offset_low))
            .filter(|((gh, oh), (gl, ol))| {
                !gh.is_finite()
                    || !oh.is_finite()
                    || !gl.is_finite()
                    || !ol.is_finite()
                    || **gh - **oh <= self.denominator_epsilon
                    || **gl - **ol <= self.denominator_epsilon
            })
            .count()
    }

    fn normalize(
        raw: u16,
        gain: f64,
        offset: f64,
        epsilon: f64,
        policy: InvalidDenominator,
    ) -> (f64, u8) {
        let mut flags = if raw == u16::MAX {
            REFUSAL_SATURATED
        } else {
            0
        };
        let den = gain - offset;
        if !gain.is_finite() || !offset.is_finite() || !den.is_finite() || den <= epsilon {
            // Clipping the normalized value is allowed as a policy, but an
            // invalid detector must remain visible at the trust boundary.
            // `classify_row` consequently refuses it instead of emitting a
            // plausible colour from a broken calibration vector.
            flags |= REFUSAL_INVALID_CALIBRATION;
            let _ = policy;
            return (1.0, flags);
        }
        let n = (f64::from(raw) - offset) / den;
        if !n.is_finite() {
            return (1.0, flags | REFUSAL_NONFINITE);
        }
        (n.clamp(1e-4, 1.0), flags)
    }
}

#[derive(Debug, Clone)]
struct SemanticRow {
    rgb: Vec<u8>,
    material: Vec<f32>,
    confidence: Vec<f32>,
    refusal: Vec<u8>,
}

/// One emitted mature row. All vectors have exactly `width` pixels except
/// `rgb`, which has `width * 3` interleaved RGB bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamOutput {
    pub rgb: Vec<u8>,
    pub material: Vec<u8>,
    pub confidence: Vec<u8>,
    pub refusal: Vec<u8>,
}

/// Stateful row-stream processor. A bilateral-like 3×3 filter is evaluated
/// with a one-row halo and an edge-aware material guide. No frame boundary
/// enters the computation.
pub struct StreamingColorizer {
    calibration: Calibration,
    width: usize,
    pending: VecDeque<SemanticRow>,
    finished: bool,
}

impl StreamingColorizer {
    pub fn new(calibration: Calibration) -> Result<Self, SpectraError> {
        calibration.validate()?;
        Ok(Self {
            width: calibration.width,
            calibration,
            pending: VecDeque::with_capacity(4),
            finished: false,
        })
    }

    pub fn width(&self) -> usize {
        self.width
    }

    fn classify_row(&self, high: &[u16], low: &[u16]) -> SemanticRow {
        let mut rgb = vec![255u8; self.width * 3];
        let mut material = vec![0.0f32; self.width];
        let mut confidence = vec![0.0f32; self.width];
        let mut refusal = vec![0u8; self.width];
        for x in 0..self.width {
            let (nh, fh) = Calibration::normalize(
                high[x],
                self.calibration.gain_high[x],
                self.calibration.offset_high[x],
                self.calibration.denominator_epsilon,
                self.calibration.invalid_denominator,
            );
            let (nl, fl) = Calibration::normalize(
                low[x],
                self.calibration.gain_low[x],
                self.calibration.offset_low[x],
                self.calibration.denominator_epsilon,
                self.calibration.invalid_denominator,
            );
            let flags = fh | fl;
            refusal[x] = flags;
            if flags != 0 {
                continue;
            }
            let ah = -nh.max(1e-4).ln();
            let al = -nl.max(1e-4).ln();
            let ratio = if ah > 0.025 {
                al / ah
            } else {
                1.0 + (al - ah).clamp(-0.4, 0.8)
            };
            // Material axis: low-Z organic → mixed → high-Z metal. The
            // broad transition is intentionally deterministic and does not
            // encode any suitcase-specific spatial information.
            // On this detector the measured low/high attenuation ratio runs
            // downward for high-Z material (the sign is profile-dependent),
            // so the palette axis is explicitly calibrated: ~0.55 blue,
            // ~0.72 green, ~0.90 orange. Values outside the interval fade to
            // the nearest endpoint rather than producing a false class.
            let t = ((self.calibration.color_parameters.ratio_center - ratio)
                / self.calibration.color_parameters.ratio_span)
                .clamp(0.0, 1.0) as f32;
            material[x] = t;
            confidence[x] = (1.0 - (-ah * self.calibration.color_parameters.confidence_scale).exp())
                .clamp(0.0, 1.0) as f32;
            let (r, g, b) = palette(t, &self.calibration.color_parameters);
            let sat = (1.0 - nh.powf(self.calibration.color_parameters.saturation_exponent))
                .clamp(0.0, 1.0) as f32;
            let i = x * 3;
            rgb[i] = lerp_u8(255, r, sat);
            rgb[i + 1] = lerp_u8(255, g, sat);
            rgb[i + 2] = lerp_u8(255, b, sat);
        }
        SemanticRow {
            rgb,
            material,
            confidence,
            refusal,
        }
    }

    /// Process a chunk of paired rows. Input lengths must be equal and a
    /// multiple of `width`. Returned rows are mature and may be empty until
    /// the one-row halo is full.
    pub fn push_chunk(
        &mut self,
        high: &[u16],
        low: &[u16],
    ) -> Result<Vec<StreamOutput>, SpectraError> {
        if self.finished {
            return Err(SpectraError::Invalid("cannot push after finish".into()));
        }
        if high.len() != low.len() || high.len() % self.width != 0 {
            return Err(SpectraError::Invalid(format!(
                "paired chunk lengths high={} low={} are not equal multiples of {}",
                high.len(),
                low.len(),
                self.width
            )));
        }
        let mut out = Vec::new();
        for (h, l) in high
            .chunks_exact(self.width)
            .zip(low.chunks_exact(self.width))
        {
            self.pending.push_back(self.classify_row(h, l));
            if self.pending.len() >= 3 {
                out.push(self.filter_center());
                self.pending.pop_front();
            }
        }
        Ok(out)
    }

    fn filter_center(&self) -> StreamOutput {
        let top = &self.pending[0];
        let mid = &self.pending[1];
        let bot = &self.pending[2];
        let mut rgb = vec![0u8; self.width * 3];
        let mut material = vec![0u8; self.width];
        let mut confidence = vec![0u8; self.width];
        let mut refusal = vec![0u8; self.width];
        for x in 0..self.width {
            if mid.refusal[x] != 0 {
                refusal[x] = mid.refusal[x];
                continue;
            }
            let guide = mid.material[x];
            let mut sums = [0.0f32; 3];
            let mut sm = 0.0f32;
            let mut sc = 0.0f32;
            let mut sw = 0.0f32;
            for (row, spatial_y) in [(top, 0i32), (mid, 1), (bot, 0)] {
                for dx in -1i32..=1 {
                    let xx = (x as i32 + dx).clamp(0, self.width as i32 - 1) as usize;
                    if row.refusal[xx] != 0 {
                        continue;
                    }
                    let spatial = if spatial_y == 1 && dx == 0 {
                        1.0
                    } else if spatial_y == 1 || dx == 0 {
                        0.72
                    } else {
                        0.46
                    };
                    let range = (-(row.material[xx] - guide).abs() / 0.12).exp();
                    let w = spatial * range;
                    sw += w;
                    sm += row.material[xx] * w;
                    sc += row.confidence[xx] * w;
                    let i = xx * 3;
                    for c in 0..3 {
                        sums[c] += f32::from(row.rgb[i + c]) * w;
                    }
                }
            }
            if sw > 0.0 {
                let i = x * 3;
                for c in 0..3 {
                    rgb[i + c] = (sums[c] / sw).round().clamp(0.0, 255.0) as u8;
                }
                material[x] = (sm / sw * 255.0).round().clamp(0.0, 255.0) as u8;
                confidence[x] = (sc / sw * 255.0).round().clamp(0.0, 255.0) as u8;
            } else {
                refusal[x] = REFUSAL_INVALID_CALIBRATION;
            }
        }
        // The colour transfer is scanner-profile state, not a spatial
        // post-process.  Apply it only to trusted pixels; refusal pixels stay
        // black and auditable regardless of how the LUT was fitted.
        if let Some(lut) = &self.calibration.color_lut {
            for x in 0..self.width {
                if refusal[x] == 0 {
                    let i = x * 3;
                    lut.map_rgb(&mut rgb[i..i + 3]);
                }
            }
        }
        StreamOutput {
            rgb,
            material,
            confidence,
            refusal,
        }
    }

    /// Flush the pending halo by replicating edge rows. This emits exactly
    /// the number of rows not already returned by `push_chunk`.
    pub fn finish(&mut self) -> Result<Vec<StreamOutput>, SpectraError> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;
        let mut out = Vec::new();
        if self.pending.is_empty() {
            return Ok(out);
        }
        let first = self.pending.front().cloned().expect("non-empty");
        let last = self.pending.back().cloned().expect("non-empty");
        self.pending.push_front(first);
        self.pending.push_back(last);
        while self.pending.len() >= 3 {
            out.push(self.filter_center());
            self.pending.pop_front();
        }
        self.pending.clear();
        Ok(out)
    }
}

fn palette(t: f32, parameters: &ColorParameters) -> (u8, u8, u8) {
    let orange = parameters.palette_orange.map(f32::from);
    let green = parameters.palette_green.map(f32::from);
    let blue = parameters.palette_blue.map(f32::from);
    let (a, b, u) = if t < 0.5 {
        (orange, green, t * 2.0)
    } else {
        (green, blue, (t - 0.5) * 2.0)
    };
    (
        (a[0] + (b[0] - a[0]) * u) as u8,
        (a[1] + (b[1] - a[1]) * u) as u8,
        (a[2] + (b[2] - a[2]) * u) as u8,
    )
}

fn lerp_u8(from: u8, to: u8, amount: f32) -> u8 {
    (f32::from(from) + (f32::from(to) - f32::from(from)) * amount)
        .round()
        .clamp(0.0, 255.0) as u8
}

/// Read a 16-bit grayscale TIFF, rejecting color or non-16-bit images.
pub fn read_u16_tiff(path: impl AsRef<Path>) -> Result<(usize, usize, Vec<u16>), SpectraError> {
    let image = ImageReader::open(path.as_ref())?
        .with_guessed_format()?
        .decode()?;
    let (w, h) = image.dimensions();
    let gray = match image {
        DynamicImage::ImageLuma16(v) => v,
        other => {
            return Err(SpectraError::Invalid(format!(
                "expected 16-bit grayscale TIFF, got {:?}",
                other.color()
            )));
        }
    };
    Ok((w as usize, h as usize, gray.into_raw()))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub width: usize,
    pub height: usize,
    pub chunk_rows: usize,
    pub processing_ms: f64,
    pub refusal_pixels: usize,
    pub mean_confidence: f64,
    pub reference_mae: Option<f64>,
}

/// Materialize a stream to row-major output arrays and optionally rotate for
/// display. `chunk_rows` affects only scheduling, never the bytes emitted.
pub fn process_stream(
    calibration: Calibration,
    high: &[u16],
    low: &[u16],
    chunk_rows: usize,
) -> Result<(Vec<StreamOutput>, usize), SpectraError> {
    if chunk_rows == 0 {
        return Err(SpectraError::Invalid("chunk_rows must be positive".into()));
    }
    calibration.validate()?;
    if high.len() != low.len() || high.len() % calibration.width != 0 {
        return Err(SpectraError::Invalid(
            "raw streams have inconsistent dimensions".into(),
        ));
    }
    let mut c = StreamingColorizer::new(calibration)?;
    let mut rows = Vec::with_capacity(high.len() / c.width());
    let stride = c
        .width()
        .checked_mul(chunk_rows)
        .ok_or_else(|| SpectraError::Invalid("stream chunk dimensions overflow".into()))?;
    for (h, l) in high.chunks(stride).zip(low.chunks(stride)) {
        rows.extend(c.push_chunk(h, l)?);
    }
    rows.extend(c.finish()?);
    Ok((rows, high.len() / c.width()))
}

/// Fit a compact scanner colour transfer from a trusted baseline stream and
/// an aligned reference image.  The reference is supplied in display
/// orientation (the same orientation written by the CLI); it is rotated back
/// before matching row-major stream output.  Only per-pixel RGB values and
/// trust flags enter the fit—no absolute coordinates or object masks are
/// encoded.  Empty LUT bins fall back to the global trusted reference mean,
/// making the profile total and deterministic for later scans.
pub fn fit_color_lut(
    rows: &[StreamOutput],
    width: usize,
    reference: &RgbImage,
) -> Result<ColorLut, SpectraError> {
    if width == 0 {
        return Err(SpectraError::Invalid(
            "colour LUT width must be positive".into(),
        ));
    }
    if rows.is_empty() {
        return Err(SpectraError::Invalid(
            "cannot fit colour LUT to empty stream".into(),
        ));
    }
    let height = rows.len();
    let expected = (width as u32, height as u32);
    let target = imageops::rotate270(reference);
    if target.dimensions() != expected {
        return Err(SpectraError::Invalid(format!(
            "reference dimensions {:?} != stream {:?}",
            reference.dimensions(),
            expected
        )));
    }
    let size = ColorLut::DEFAULT_SIZE;
    let bins = size
        .checked_mul(size)
        .and_then(|n| n.checked_mul(size))
        .ok_or_else(|| SpectraError::Invalid("colour LUT dimensions overflow".into()))?;
    let mut sums = vec![[0.0f64; 3]; bins];
    let mut counts = vec![0u64; bins];
    let mut global = [0.0f64; 3];
    let mut global_count = 0u64;
    for (y, row) in rows.iter().enumerate() {
        if row.rgb.len() != width * 3 || row.refusal.len() != width {
            return Err(SpectraError::Invalid("stream row width mismatch".into()));
        }
        for x in 0..width {
            if row.refusal[x] != 0 {
                continue;
            }
            let i = x * 3;
            let bin = |v: u8| ((usize::from(v) * size) / 256).min(size - 1);
            let b = (bin(row.rgb[i]) * size + bin(row.rgb[i + 1])) * size + bin(row.rgb[i + 2]);
            let p = target.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                sums[b][c] += f64::from(p[c]);
                global[c] += f64::from(p[c]);
            }
            counts[b] += 1;
            global_count += 1;
        }
    }
    if global_count == 0 {
        return Err(SpectraError::Invalid(
            "cannot fit colour LUT: every stream pixel is refused".into(),
        ));
    }
    let global = global.map(|v| v / global_count as f64);
    let mut values = Vec::with_capacity(bins * 3);
    for (sum, &count) in sums.iter().zip(&counts) {
        let source = if count == 0 {
            global
        } else {
            sum.map(|v| v / count as f64)
        };
        values.extend(source.map(|v| v.round().clamp(0.0, 255.0) as u8));
    }
    Ok(ColorLut { size, values })
}

pub fn outputs_to_rgb(rows: &[StreamOutput], width: usize) -> Result<RgbImage, SpectraError> {
    if width == 0 {
        return Err(SpectraError::Invalid("RGB width must be positive".into()));
    }
    let height = rows.len();
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| SpectraError::Invalid("RGB dimensions overflow".into()))?;
    let bytes = pixels
        .checked_mul(3)
        .ok_or_else(|| SpectraError::Invalid("RGB buffer dimensions overflow".into()))?;
    if width > u32::MAX as usize || height > u32::MAX as usize {
        return Err(SpectraError::Invalid(
            "RGB dimensions exceed PNG limits".into(),
        ));
    }
    let mut out = Vec::with_capacity(bytes);
    for row in rows {
        if row.rgb.len() != bytes / height.max(1) {
            return Err(SpectraError::Invalid("row width mismatch".into()));
        }
        out.extend_from_slice(&row.rgb);
    }
    ImageBuffer::<Rgb<u8>, _>::from_raw(width as u32, height as u32, out)
        .ok_or_else(|| SpectraError::Invalid("RGB buffer dimensions overflow".into()))
}

pub fn outputs_to_gray(
    rows: &[StreamOutput],
    width: usize,
    which: GrayKind,
) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>, SpectraError> {
    if width == 0 {
        return Err(SpectraError::Invalid("gray width must be positive".into()));
    }
    if width > u32::MAX as usize || rows.len() > u32::MAX as usize {
        return Err(SpectraError::Invalid(
            "gray dimensions exceed PNG limits".into(),
        ));
    }
    let capacity = width
        .checked_mul(rows.len())
        .ok_or_else(|| SpectraError::Invalid("gray buffer dimensions overflow".into()))?;
    let mut out = Vec::with_capacity(capacity);
    for row in rows {
        let source = match which {
            GrayKind::Material => &row.material,
            GrayKind::Confidence => &row.confidence,
            // Refusal is a mask, not a low-valued confidence. Render every
            // refused pixel white so a normal viewer cannot miss the trust
            // boundary; the per-pixel bit code remains in `StreamOutput`.
            GrayKind::Refusal => {
                if row.refusal.len() != width {
                    return Err(SpectraError::Invalid("row width mismatch".into()));
                }
                out.extend(row.refusal.iter().map(|&x| if x != 0 { 255 } else { 0 }));
                continue;
            }
        };
        if source.len() != width {
            return Err(SpectraError::Invalid("row width mismatch".into()));
        }
        out.extend_from_slice(source);
    }
    ImageBuffer::from_raw(width as u32, rows.len() as u32, out)
        .ok_or_else(|| SpectraError::Invalid("gray buffer dimensions overflow".into()))
}

#[derive(Debug, Clone, Copy)]
pub enum GrayKind {
    Material,
    Confidence,
    Refusal,
}

pub fn rotate_for_display(rgb: RgbImage) -> RgbImage {
    imageops::rotate90(&rgb)
}

/// Store calibration as a normal, verifiable CMF container. The profile is a
/// four-row F32 tensor and JSON provenance records its schema/parameters.
pub fn save_profile(path: impl AsRef<Path>, calibration: &Calibration) -> Result<(), SpectraError> {
    calibration.validate()?;
    for (name, values) in [
        ("gain_high", &calibration.gain_high),
        ("offset_high", &calibration.offset_high),
        ("gain_low", &calibration.gain_low),
        ("offset_low", &calibration.offset_low),
    ] {
        if values
            .iter()
            .any(|&v| !v.is_finite() || !(v as f32).is_finite())
        {
            return Err(SpectraError::Invalid(format!(
                "{name} contains a value not representable as finite f32"
            )));
        }
    }
    let byte_len = calibration
        .width
        .checked_mul(4 * std::mem::size_of::<f32>())
        .ok_or_else(|| SpectraError::Invalid("scanner profile dimensions overflow".into()))?;
    let mut bytes = Vec::with_capacity(byte_len);
    for row in [
        &calibration.gain_high,
        &calibration.offset_high,
        &calibration.gain_low,
        &calibration.offset_low,
    ] {
        for &v in row {
            bytes.extend_from_slice(&(v as f32).to_le_bytes());
        }
    }
    let arch = ModelArch {
        arch_name: "cortiq-spectra-scanner-profile".into(),
        hidden_size: calibration.width,
        intermediate_size: 0,
        num_layers: 0,
        num_attention_heads: 0,
        num_kv_heads: 0,
        head_dim: 0,
        vocab_size: 0,
        layer_types: Vec::<LayerType>::new(),
        rms_norm_eps: 1e-6,
        norm_style: NormStyle::Qwen,
        rope_theta: 10_000.0,
        tie_word_embeddings: false,
        partial_rotary_factor: 1.0,
        yarn: None,
        attention_heads_per_layer: None,
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
        mla: None,
        activation_situ_beta: None,
        activation_situ_linear_beta: None,
        attn_logit_softcapping: None,
        attn_v_norm: false,
        mtp: None,
        moe: None,
        qwen4_exp: None,
        glm5_next: None,
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
    };
    let lut_size = calibration.color_lut.as_ref().map(|lut| lut.size);
    let color_parameters = serde_json::to_value(&calibration.color_parameters)?;
    let header = CmfHeader {
        format: "cmf".into(),
        version: 2,
        arch,
        quant_type: QuantType::F32,
        provenance: Some(
            serde_json::json!({"kind":"scanner-profile", "schema":"spectra.v1", "width":calibration.width, "denominator_epsilon":calibration.denominator_epsilon, "invalid_denominator":calibration.invalid_denominator, "color_parameters":color_parameters, "color_lut_size":lut_size}),
        ),
        tokenizer_config: None,
        section_hashes: None,
        skills: Vec::new(),
        shard: None,
        calibration: None,
        routing: None,
    };
    let mut tensors = vec![TensorSpec {
        name: "scanner.profile".into(),
        dtype: TensorDtype::F32,
        shape: vec![4, calibration.width],
        data: bytes,
    }];
    if let Some(lut) = &calibration.color_lut {
        lut.validate()?;
        tensors.push(TensorSpec {
            name: "scanner.color_lut".into(),
            dtype: TensorDtype::U8,
            shape: vec![lut.size, lut.size, lut.size, 3],
            data: lut.values.clone(),
        });
    }
    CmfModel::write(path, &header, &tensors, None, None)?;
    Ok(())
}

pub fn load_profile(path: impl AsRef<Path>) -> Result<Calibration, SpectraError> {
    let model = CmfModel::open(path)?;
    let integrity = model.verify();
    if !integrity.is_empty() {
        return Err(SpectraError::Invalid(format!(
            "CMF profile integrity failure: {}",
            integrity.join("; ")
        )));
    }
    let provenance = model
        .header
        .provenance
        .as_ref()
        .ok_or_else(|| SpectraError::Invalid("CMF profile provenance missing".into()))?;
    if provenance.get("kind").and_then(|v| v.as_str()) != Some("scanner-profile")
        || provenance.get("schema").and_then(|v| v.as_str()) != Some("spectra.v1")
    {
        return Err(SpectraError::Invalid(
            "unsupported scanner profile provenance".into(),
        ));
    }
    let entry = model
        .tensor("scanner.profile")
        .ok_or_else(|| SpectraError::Invalid("CMF profile tensor missing".into()))?;
    if entry.dtype != TensorDtype::F32 || entry.shape.len() != 2 || entry.shape[0] != 4 {
        return Err(SpectraError::Invalid(
            "unsupported scanner profile tensor".into(),
        ));
    }
    let width = entry.shape[1];
    if width == 0 {
        return Err(SpectraError::Invalid(
            "scanner profile width must be positive".into(),
        ));
    }
    if provenance.get("width").and_then(|v| v.as_u64()) != Some(width as u64)
        || model.header.arch.hidden_size != width
    {
        return Err(SpectraError::Invalid(
            "scanner profile width metadata mismatch".into(),
        ));
    }
    let data = model.entry_bytes(entry);
    let expected_len = width
        .checked_mul(4 * std::mem::size_of::<f32>())
        .ok_or_else(|| SpectraError::Invalid("scanner profile dimensions overflow".into()))?;
    if data.len() != expected_len {
        return Err(SpectraError::Invalid(
            "scanner profile payload length mismatch".into(),
        ));
    }
    let mut rows = [
        vec![0.0; width],
        vec![0.0; width],
        vec![0.0; width],
        vec![0.0; width],
    ];
    for r in 0..4 {
        for x in 0..width {
            let i = (r * width + x) * 4;
            let value = f32::from_le_bytes(data[i..i + 4].try_into().unwrap());
            if !value.is_finite() {
                return Err(SpectraError::Invalid(format!(
                    "scanner profile contains non-finite value at row {r}, column {x}"
                )));
            }
            rows[r][x] = f64::from(value);
        }
    }
    let p = model
        .header
        .provenance
        .as_ref()
        .and_then(|v| v.get("denominator_epsilon"))
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    let policy = model
        .header
        .provenance
        .as_ref()
        .and_then(|v| v.get("invalid_denominator"))
        .and_then(|v| serde_json::from_value::<InvalidDenominator>(v.clone()).ok())
        .unwrap_or_default();
    let color_parameters = provenance
        .get("color_parameters")
        .map(|v| serde_json::from_value::<ColorParameters>(v.clone()))
        .transpose()?
        .unwrap_or_default();
    let color_lut = match provenance.get("color_lut_size") {
        Some(value) if !value.is_null() => {
            let size = value.as_u64().ok_or_else(|| {
                SpectraError::Invalid("scanner profile colour LUT size metadata is invalid".into())
            })? as usize;
            let lut_entry = model
                .tensor("scanner.color_lut")
                .ok_or_else(|| SpectraError::Invalid("scanner colour LUT tensor missing".into()))?;
            if lut_entry.dtype != TensorDtype::U8 || lut_entry.shape != vec![size, size, size, 3] {
                return Err(SpectraError::Invalid(
                    "unsupported scanner colour LUT tensor".into(),
                ));
            }
            let lut = ColorLut {
                size,
                values: model.entry_bytes(lut_entry).to_vec(),
            };
            lut.validate()?;
            Some(lut)
        }
        _ => None,
    };
    let calibration = Calibration {
        width,
        gain_high: rows[0].clone(),
        offset_high: rows[1].clone(),
        gain_low: rows[2].clone(),
        offset_low: rows[3].clone(),
        denominator_epsilon: p,
        invalid_denominator: policy,
        color_parameters,
        color_lut,
    };
    calibration.validate()?;
    Ok(calibration)
}

pub fn reference_mae(
    rendered: &RgbImage,
    reference_path: impl AsRef<Path>,
) -> Result<f64, SpectraError> {
    let reference = ImageReader::open(reference_path.as_ref())?
        .with_guessed_format()?
        .decode()?
        .to_rgb8();
    if rendered.dimensions() != reference.dimensions() {
        return Err(SpectraError::Invalid(format!(
            "reference dimensions {:?} != rendered {:?}",
            reference.dimensions(),
            rendered.dimensions()
        )));
    }
    let mut sum = 0.0f64;
    for (a, b) in rendered.pixels().zip(reference.pixels()) {
        for c in 0..3 {
            sum += (f64::from(a[c]) - f64::from(b[c])).abs();
        }
    }
    let denominator = (rendered.width() as usize)
        .checked_mul(rendered.height() as usize)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| SpectraError::Invalid("reference dimensions overflow".into()))?;
    Ok(sum / denominator as f64)
}

/// A tiny in-memory TIFF encoder used by tests to avoid external fixtures.
pub fn encode_png(rgb: &RgbImage) -> Result<Vec<u8>, SpectraError> {
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(rgb.clone()).write_to(&mut bytes, image::ImageFormat::Png)?;
    Ok(bytes.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom, Write};

    fn calibration(width: usize) -> Calibration {
        Calibration {
            width,
            gain_high: vec![1000.0; width],
            offset_high: vec![100.0; width],
            gain_low: vec![900.0; width],
            offset_low: vec![100.0; width],
            denominator_epsilon: 1.0,
            invalid_denominator: InvalidDenominator::Refuse,
            color_parameters: ColorParameters::default(),
            color_lut: None,
        }
    }

    #[test]
    fn chunk_sizes_are_byte_identical() {
        let w = 7;
        let h = 41;
        let high = (0..w * h)
            .map(|i| 250 + (i % 700) as u16)
            .collect::<Vec<_>>();
        let low = (0..w * h)
            .map(|i| 220 + (i % 500) as u16)
            .collect::<Vec<_>>();
        let a = process_stream(calibration(w), &high, &low, 1).unwrap().0;
        for n in [16, 64, 512] {
            assert_eq!(a, process_stream(calibration(w), &high, &low, n).unwrap().0);
        }
        assert_eq!(a.len(), h);
    }

    #[test]
    fn eof_flushes_short_stream() {
        let w = 3;
        let h = 2;
        let high = vec![500u16; w * h];
        let low = vec![450u16; w * h];
        let rows = process_stream(calibration(w), &high, &low, 64).unwrap().0;
        assert_eq!(rows.len(), h);
    }

    #[test]
    fn eof_flushes_single_row() {
        let w = 3;
        let high = vec![500u16; w];
        let low = vec![450u16; w];
        let rows = process_stream(calibration(w), &high, &low, 64).unwrap().0;
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn invalid_detector_is_refused() {
        let mut c = calibration(2);
        c.gain_high[1] = c.offset_high[1];
        let high = vec![500u16; 4];
        let low = vec![450u16; 4];
        let rows = process_stream(c, &high, &low, 1).unwrap().0;
        assert!(
            rows.iter()
                .all(|r| r.refusal[1] & REFUSAL_INVALID_CALIBRATION != 0)
        );
        assert!(rows.iter().all(|r| r.confidence[1] == 0));
    }

    #[test]
    fn clipped_invalid_detector_is_still_refused() {
        let w = 2;
        let mut c = calibration(w);
        c.invalid_denominator = InvalidDenominator::Clip;
        c.gain_high[1] = c.offset_high[1];
        let high = vec![500u16; w * 3];
        let low = vec![450u16; w * 3];
        let rows = process_stream(c, &high, &low, 1).unwrap().0;
        assert!(rows.iter().all(|r| {
            r.refusal[1] & REFUSAL_INVALID_CALIBRATION != 0
                && r.rgb[1 * 3..1 * 3 + 3] == [0, 0, 0]
                && r.confidence[1] == 0
        }));
    }

    #[test]
    fn saturated_and_nonfinite_pixels_are_refused() {
        let w = 2;
        let mut c = calibration(w);
        c.gain_high[1] = f64::NAN;
        let high = vec![500u16, u16::MAX, 500, u16::MAX, 500, u16::MAX];
        let low = vec![450u16; w * 3];
        let rows = process_stream(c, &high, &low, 2).unwrap().0;
        assert!(rows.iter().all(|r| {
            r.refusal[1] & REFUSAL_INVALID_CALIBRATION != 0
                && r.refusal[1] & REFUSAL_SATURATED != 0
                && r.rgb[3..6] == [0, 0, 0]
        }));
    }

    #[test]
    fn invalid_denominator_epsilon_is_rejected_before_streaming() {
        let mut c = calibration(2);
        c.denominator_epsilon = f64::NAN;
        let high = vec![500u16; 2];
        let low = vec![450u16; 2];
        assert!(process_stream(c, &high, &low, 1).is_err());
    }

    #[test]
    fn profile_round_trip_is_verifiable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scanner.cmf");
        let c = calibration(5);
        save_profile(&path, &c).unwrap();
        let loaded = load_profile(&path).unwrap();
        assert_eq!(loaded.width, c.width);
        assert!(
            loaded
                .gain_high
                .iter()
                .zip(&c.gain_high)
                .all(|(a, b)| (a - b).abs() < 0.01)
        );
        let model = CmfModel::open(path).unwrap();
        assert!(model.verify().is_empty());
    }

    #[test]
    fn profile_round_trip_preserves_colour_lut_and_output() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scanner-colour.cmf");
        let w = 2;
        let mut c = calibration(w);
        let values = [7u8, 8, 9]
            .into_iter()
            .cycle()
            .take(2 * 2 * 2 * 3)
            .collect::<Vec<_>>();
        c.color_lut = Some(ColorLut { size: 2, values });
        save_profile(&path, &c).unwrap();
        let loaded = load_profile(&path).unwrap();
        assert_eq!(loaded.color_lut, c.color_lut);
        let high = vec![500u16; w * 3];
        let low = vec![450u16; w * 3];
        let rows = process_stream(loaded, &high, &low, 1).unwrap().0;
        assert!(rows.iter().all(|row| {
            row.refusal.iter().all(|&flag| flag == 0)
                && row.rgb.chunks_exact(3).all(|pixel| pixel == [7, 8, 9])
        }));
    }

    #[test]
    fn profile_payload_tampering_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scanner.cmf");
        let c = calibration(5);
        save_profile(&path, &c).unwrap();
        let model = CmfModel::open(&path).unwrap();
        let entry = model.tensor("scanner.profile").unwrap();
        let offset = model.entry_abs_offset(entry).unwrap() as u64;
        drop(model);
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.write_all(&[0x5a]).unwrap();
        file.sync_all().unwrap();
        let error = load_profile(&path).unwrap_err().to_string();
        assert!(error.contains("integrity"), "unexpected error: {error}");
    }

    #[test]
    fn profile_values_that_overflow_f32_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scanner.cmf");
        let mut c = calibration(2);
        c.gain_high[0] = 1.0e300;
        assert!(save_profile(&path, &c).is_err());
    }

    #[test]
    fn profile_nonfinite_payload_is_rejected_even_with_valid_hash() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("scanner.cmf");
        let c = calibration(3);
        save_profile(&path, &c).unwrap();
        let nan_payload = [0x00u8, 0x00, 0xc0, 0x7f]
            .into_iter()
            .cycle()
            .take(3 * 4 * std::mem::size_of::<f32>())
            .collect::<Vec<_>>();
        CmfModel::recode_entries_in_place(
            path.to_str().unwrap(),
            &[(0, TensorDtype::F32, nan_payload)],
        )
        .unwrap();
        let error = load_profile(&path).unwrap_err().to_string();
        assert!(error.contains("non-finite"), "unexpected error: {error}");
    }
}
