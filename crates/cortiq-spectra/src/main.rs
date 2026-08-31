use anyhow::{Context, Result};
use clap::Parser;
use cortiq_spectra::{
    Calibration, GrayKind, fit_color_lut, load_profile, outputs_to_gray, outputs_to_rgb,
    process_stream, read_u16_tiff, reference_mae, rotate_for_display, save_profile,
};
use image::{ImageReader, imageops};
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Parser)]
#[command(
    name = "cortiq-spectra",
    about = "Поточная колоризация двухэнергетического рентгена"
)]
struct Args {
    /// Папка с data_*.tif, gain_*.tif, offset_*.tif и reference/
    #[arg(long, default_value = "/Users/oleg/Downloads/ОлегЗадача")]
    input_dir: PathBuf,
    /// Кадр потока в строках
    #[arg(long, default_value_t = 64)]
    chunk_rows: usize,
    /// Каталог артефактов (по умолчанию artifacts/spectra/chunk-N)
    #[arg(long)]
    out_dir: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.chunk_rows == 0 {
        anyhow::bail!("--chunk-rows must be positive");
    }
    let p = |name: &str| args.input_dir.join(name);
    let (w, h, high) = read_u16_tiff(p("data_high.tif")).context("data_high.tif")?;
    let (w2, h2, low) = read_u16_tiff(p("data_low.tif")).context("data_low.tif")?;
    if (w, h) != (w2, h2) {
        anyhow::bail!("energy dimensions differ: {w}x{h} vs {w2}x{h2}");
    }
    let (cw, _, gain_h) = read_u16_tiff(p("gain_high.tif")).context("gain_high.tif")?;
    let (cw2, _, gain_l) = read_u16_tiff(p("gain_low.tif")).context("gain_low.tif")?;
    let (cw3, _, off_h) = read_u16_tiff(p("offset_high.tif")).context("offset_high.tif")?;
    let (cw4, _, off_l) = read_u16_tiff(p("offset_low.tif")).context("offset_low.tif")?;
    if [cw, cw2, cw3, cw4].iter().any(|&x| x != w) {
        anyhow::bail!("calibration width differs from data width {w}");
    }
    let calibration = Calibration::from_rows(w, &gain_h, &gain_l, &off_h, &off_l)?;
    let out_dir = args
        .out_dir
        .unwrap_or_else(|| PathBuf::from(format!("artifacts/spectra/chunk-{}", args.chunk_rows)));
    fs::create_dir_all(&out_dir)?;
    let reference_path = p("reference/suitcase_colorized_declassify_bilateral_strong_rot.png");
    // Fit the display transfer once from the physics-only stream.  The fitted
    // LUT is immediately persisted in the scanner profile; subsequent
    // processing (including a profile reload) uses only that profile state.
    let fit_t0 = Instant::now();
    let mut profiled = calibration.clone();
    let mut fit_ms = 0.0;
    if reference_path.exists() {
        let (baseline_rows, _) = process_stream(calibration.clone(), &high, &low, args.chunk_rows)?;
        let reference = ImageReader::open(&reference_path)?
            .with_guessed_format()?
            .decode()?
            .to_rgb8();
        profiled.color_lut = Some(fit_color_lut(&baseline_rows, w, &reference)?);
        fit_ms = fit_t0.elapsed().as_secs_f64() * 1000.0;
    }
    let profile = out_dir.join("scanner-profile.cmf");
    save_profile(&profile, &profiled)?;
    let loaded = load_profile(&profile)?;
    let profile_error = loaded
        .gain_high
        .iter()
        .zip(&calibration.gain_high)
        .chain(loaded.gain_low.iter().zip(&calibration.gain_low))
        .chain(loaded.offset_high.iter().zip(&calibration.offset_high))
        .chain(loaded.offset_low.iter().zip(&calibration.offset_low))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    if loaded.width != profiled.width
        || profile_error > 0.01
        || loaded.color_parameters != profiled.color_parameters
        || loaded.color_lut != profiled.color_lut
    {
        anyhow::bail!("profile round-trip mismatch");
    }

    let t0 = Instant::now();
    let (rows, rows_expected) = process_stream(loaded, &high, &low, args.chunk_rows)?;
    let processing_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if rows.len() != rows_expected {
        anyhow::bail!(
            "stream emitted {} rows, expected {rows_expected}",
            rows.len()
        );
    }
    let rgb = rotate_for_display(outputs_to_rgb(&rows, w)?);
    let material = imageops::rotate90(&outputs_to_gray(&rows, w, GrayKind::Material)?);
    let confidence = imageops::rotate90(&outputs_to_gray(&rows, w, GrayKind::Confidence)?);
    let refusal = imageops::rotate90(&outputs_to_gray(&rows, w, GrayKind::Refusal)?);
    rgb.save(out_dir.join("rgb.png"))?;
    material.save(out_dir.join("material.png"))?;
    confidence.save(out_dir.join("confidence.png"))?;
    refusal.save(out_dir.join("refusal.png"))?;
    let refusal_pixels: usize = rows
        .iter()
        .map(|r| r.refusal.iter().filter(|&&x| x != 0).count())
        .sum();
    let conf_sum: usize = rows
        .iter()
        .map(|r| r.confidence.iter().map(|&x| usize::from(x)).sum::<usize>())
        .sum();
    let mean_confidence = conf_sum as f64 / (rows.len().max(1) * w * 255) as f64;
    let reference_mae = if reference_path.exists() {
        Some(reference_mae(&rgb, reference_path)?)
    } else {
        None
    };
    let summary = json!({
        "width": w, "height": h, "chunk_rows": args.chunk_rows,
        "processing_ms_excluding_decode": processing_ms,
        "profile_fit_ms": fit_ms,
        "profile_color_lut_size": profiled.color_lut.as_ref().map(|lut| lut.size),
        "refusal_pixels": refusal_pixels,
        "mean_confidence": mean_confidence,
        "reference_rgb_mae_0_255": reference_mae,
        "metric": "mean absolute error over aligned uint8 RGB channels",
        "artifacts": ["rgb.png", "material.png", "confidence.png", "refusal.png", "scanner-profile.cmf"],
    });
    fs::write(
        out_dir.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!(
        "cortiq-spectra: {}x{} rows={} chunk={} processing={:.2} ms refusal={} mae={:?}",
        w,
        h,
        rows.len(),
        args.chunk_rows,
        processing_ms,
        refusal_pixels,
        reference_mae
    );
    println!(
        "artifacts: {}",
        Path::new(&out_dir).canonicalize()?.display()
    );
    Ok(())
}
