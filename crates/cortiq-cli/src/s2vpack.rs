//! Pack a diffusers Kandinsky S2V directory into ONE .cmf that
//! `cortiq s2v file.cmf --image I --audio A` runs with no other flags.
//!
//! Layout of the container:
//! - `dit.*`   the transformer under its diffusers names. The attention and
//!   feed-forward projections of all three block families carry `--quant`;
//!   the modulation layers stay 16-bit because they only feed a per-step
//!   host precompute; the embedders, norms and biases are f32.
//! - `te.*`    the Thinker's text model (28 layers, `lm_head` left out), the
//!   projections at `--te-quant`, `embed_tokens` q8_row, norms f32.
//! - `audio.*` the Thinker's audio tower (32 layers, the bos/eos embedding
//!   left out): the projections at `--audio-quant`, the convolutions and
//!   norms f32. The vision tower is never run, so it is not packed.
//! - `clip.*`  the CLIP text tower at f16, with its own `clip.tokenizer_json`.
//! - `vae.*`   HunyuanVideo VAE encoder + decoder at `--vae-quant`.
//! - `s2v.config_json` the recipe (steps, fps, chunk length, buckets, RoPE
//!   scale, pi-Flow schedule, the default and the system prompt).
//! - VOCAB = `tokenizer/tokenizer.json`.
//!
//! Every tensor is converted while mapped, one bounded window at a time, so
//! the 42.7 GB transformer never sits in RAM. Writes `<out>.sha256`.

use crate::zimagepack::{Codec, StTensor, codec_name, encode, parse_codec, sha256_file, st_header};
use crate::{convert, gguf};
use anyhow::{Context, ensure};
use cortiq_core::format::{CmfHeader, CmfStreamWriter};
use cortiq_core::types::{ModelArch, TensorDtype};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub(crate) const DEFAULT_DIT_CODEC: &str = "q4tp";
pub(crate) const DEFAULT_TE_CODEC: &str = "q8";
pub(crate) const DEFAULT_AUDIO_CODEC: &str = "q8";
pub(crate) const DEFAULT_VAE_CODEC: &str = "f16";

/// `DEFAULT_PROMPT` of the reference pipeline. Text conditioning is not
/// optional there: an empty prompt leaves cross-attention nothing to attend to.
const DEFAULT_PROMPT: &str = "The video presents a stable medium close-up of a subject facing the camera directly. The subject exhibits natural, lifelike body dynamics: the head and shoulders move fluidly with the rhythm of speech. The lighting is even, maintaining a balanced natural exposure to clearly define facial features without washout. Throughout the sequence, the subject engages in continuous speech marked by precise, high-fidelity articulation. The lips move in a rhythmic, natural pattern: alternating between brief firm closures and relaxed openings. During these openings, the tips of the upper teeth are momentarily and realistically visible, exhibiting natural enamel texture. The upper lip lifts subtly over the teeth during enunciation, while the jaw performs minimal vertical oscillations synchronized with the speech cadence. Facial micro-expressions are active yet restrained; eyebrows rise and lower slightly with minute asymmetry. The eyes maintain a steady, engaged gaze forward, punctuated by smooth, deliberate blinks. ";
const SYSTEM_PROMPT: &str = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.";

pub(crate) fn is_s2v_root(root: &Path) -> bool {
    root.join("modular_model_index.json").is_file()
        && std::fs::read(root.join("transformer/config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .is_some_and(|v| v["_class_name"] == "KandinskyS2VTransformer3DModel")
}

pub(crate) struct PackOpts {
    pub dit: Codec,
    pub te: Codec,
    pub audio: Codec,
    pub vae: Codec,
    /// DiT projections kept at q8_2f over the `dit` base: `visual.N`,
    /// a family (`attn`, `cross`, `audio`, `ffn`) or `all`.
    pub dit_keep: Vec<String>,
    pub source_sha: bool,
}

/// Which codec family one transformer tensor belongs to.
#[derive(Clone, Copy, PartialEq)]
enum Fam {
    Proj,
    Keep16,
    F32,
}

fn dit_family(name: &str, shape: &[usize]) -> Fam {
    if !name.ends_with(".weight") || shape.len() != 2 {
        return Fam::F32;
    }
    if name.contains("modulation.out_layer") {
        return Fam::Keep16;
    }
    let attn_proj = [".to_query.", ".to_key.", ".to_value.", ".out_layer."]
        .iter()
        .any(|k| name.contains(k))
        && name.contains("attention");
    if (attn_proj || name.contains(".feed_forward.")) && name.contains("transformer_blocks") {
        return Fam::Proj;
    }
    // Embedders: read once per chunk or once per token on the host.
    if name.contains("_embeddings") && name.contains(".in_layer.") {
        return Fam::Keep16;
    }
    Fam::F32
}

fn dit_keep_match(name: &str, keys: &[String]) -> bool {
    keys.iter().any(|k| match k.as_str() {
        "all" => true,
        "attn" => name.contains(".self_attention."),
        "cross" => name.contains(".cross_attention."),
        "audio" => name.contains(".cross_attention_v2a."),
        "ffn" => name.contains(".feed_forward."),
        k => k
            .strip_prefix("visual.")
            .is_some_and(|n| name.starts_with(&format!("visual_transformer_blocks.{n}."))),
    })
}

fn keep16(t: &StTensor) -> Codec {
    match t.dtype.as_str() {
        "BF16" | "F16" => Codec::Raw,
        _ => Codec::F16,
    }
}

/// The canned prompt's Thinker and CLIP outputs, from the source weights: a
/// throwaway container of the text encoders at their source precision, read
/// back through the engine. A render with the default prompt then opens neither.
fn canned_conditions(root: &Path, out: &str, o: &PackOpts) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    let tmp = format!("{out}.prompt.tmp");
    write_container(root, &tmp, o, true, None)?;
    let r = (|| -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        let m = std::sync::Arc::new(cortiq_core::CmfModel::open_sharded(&tmp)?);
        let d = cortiq_engine::s2vgen::S2vDefaults::of(&m).map_err(|e| anyhow::anyhow!(e))?;
        cortiq_engine::s2vgen::encode_prompt(&m, &d, &d.prompt).map_err(|e| anyhow::anyhow!(e))
    })();
    let _ = std::fs::remove_file(&tmp);
    let _ = std::fs::remove_file(format!("{tmp}.sha256"));
    r
}

pub(crate) fn pack(root: &Path, out: &str, o: &PackOpts) -> anyhow::Result<()> {
    ensure!(
        is_s2v_root(root),
        "{}: not a diffusers Kandinsky S2V directory",
        root.display()
    );
    let t = std::time::Instant::now();
    let cond = canned_conditions(root, out, o)?;
    eprintln!("s2v pack: canned prompt encoded in {:.0}s", t.elapsed().as_secs_f64());
    write_container(root, out, o, false, Some(&cond))
}

fn write_container(
    root: &Path,
    out: &str,
    o: &PackOpts,
    prompt_only: bool,
    cond: Option<&(Vec<f32>, Vec<f32>)>,
) -> anyhow::Result<()> {
    let t_all = std::time::Instant::now();
    let read = |rel: &str| -> anyhow::Result<Vec<u8>> {
        std::fs::read(root.join(rel)).with_context(|| format!("{}/{rel}", root.display()))
    };
    let dit_cfg_raw = read("transformer/config.json")?;
    let te_cfg_raw = read("text_encoder/config.json")?;
    let te_cfg: serde_json::Value = serde_json::from_slice(&te_cfg_raw)?;
    let text_cfg = te_cfg["text_config"].clone();
    let audio_cfg = te_cfg["audio_config"].clone();
    let clip_cfg_raw = read("text_encoder_2/config.json")?;
    let vae_cfg_raw = read("vae/config.json")?;
    let sched_raw = read("scheduler/scheduler_config.json")?;
    let sched: serde_json::Value = serde_json::from_slice(&sched_raw)?;
    let clip_tok = read("tokenizer_2/tokenizer.json")?;
    let vocab = read("tokenizer/tokenizer.json")?;
    let proc_cfg = read("tokenizer/processor_config.json")?;

    let n_grid = sched["n_grid"].as_u64().context("n_grid")?;
    let defaults = serde_json::json!({
        "steps": if n_grid > 1 { 16 } else { 50 },
        "fps": 25,
        "latent_frames": 32,
        "seed": 5,
        "buckets": [[512, 512], [768, 512], [512, 768]],
        "scale_factor": [1.0, 2.0, 2.0],
        "prompt": DEFAULT_PROMPT,
        "system_prompt": SYSTEM_PROMPT,
        "shift": sched["shift"],
        "n_grid": n_grid,
        "eps": sched["eps"],
        "final_step_size_scale": sched["final_step_size_scale"],
        "num_policy_substeps": sched["num_policy_substeps"],
    });

    let files_of = |dir: &str, single: &str| -> PathBuf { root.join(dir).join(single) };
    let dit_file = files_of("transformer", "diffusion_pytorch_model.safetensors");
    let te_file = files_of("text_encoder", "model.safetensors");
    let clip_file = files_of("text_encoder_2", "model.safetensors");
    let vae_file = files_of("vae", "diffusion_pytorch_model.safetensors");

    let mut source_sha = serde_json::Map::new();
    if o.source_sha && !prompt_only {
        let list = [&dit_file, &te_file, &clip_file, &vae_file];
        let hashes: Vec<anyhow::Result<String>> = std::thread::scope(|s| {
            let hs: Vec<_> = list.iter().map(|f| s.spawn(|| sha256_file(f))).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (f, h) in list.iter().zip(hashes) {
            let rel = f.strip_prefix(root).unwrap_or(f).display().to_string();
            source_sha.insert(rel, serde_json::json!(h?));
        }
    }

    struct Item {
        out: String,
        file: usize,
        t: usize,
        codec: Codec,
    }
    let mut files: Vec<(PathBuf, Vec<StTensor>, usize)> = Vec::new();
    let mut items: Vec<Item> = Vec::new();

    // ── transformer ──
    let q8 = Codec::Q(convert::Quant::Q8_2f);
    let (ts, size) = st_header(&dit_file)?;
    for (ti, t) in ts.iter().enumerate() {
        if prompt_only {
            break;
        }
        // The speech and prompt towers run a handful of tokens through blocks
        // shared by every frame: at q4tp the speech stream leaves its four
        // blocks 78 % away from the reference (measured), so they stay 8-bit.
        let tower = t.name.starts_with("text_transformer_blocks") || t.name.starts_with("audio_transformer_blocks");
        let codec = match dit_family(&t.name, &t.shape) {
            Fam::Proj if tower && matches!(o.dit, Codec::Q(_)) => q8,
            Fam::Proj if dit_keep_match(&t.name, &o.dit_keep) => q8,
            Fam::Proj => o.dit,
            Fam::Keep16 => keep16(t),
            Fam::F32 => Codec::F32,
        };
        items.push(Item {
            out: format!("dit.{}", t.name),
            file: 0,
            t: ti,
            codec,
        });
    }
    files.push((dit_file.clone(), ts, size));
    let n_dit = items.len();

    // ── Thinker: text model and audio tower; the vision tower is never run ──
    let (ts, size) = st_header(&te_file)?;
    for (ti, t) in ts.iter().enumerate() {
        let (out_name, codec) = if let Some(n) = t.name.strip_prefix("model.") {
            let c = if prompt_only {
                Codec::Raw
            } else if n == "embed_tokens.weight" {
                match o.te {
                    Codec::Q(_) => Codec::Q(convert::Quant::Q8Row),
                    c => c,
                }
            } else if t.shape.len() == 2 && n.ends_with("_proj.weight") {
                o.te
            } else {
                Codec::F32
            };
            (format!("te.{n}"), c)
        } else if let Some(n) = t.name.strip_prefix("audio_tower.") {
            if n == "audio_bos_eos_token.weight" || prompt_only {
                continue;
            }
            let proj = t.shape.len() == 2
                && n.ends_with(".weight")
                && (n.contains("_proj.") || n.contains(".fc1.") || n.contains(".fc2.") || n == "proj.weight");
            (format!("audio.{n}"), if proj { o.audio } else { Codec::F32 })
        } else {
            continue;
        };
        items.push(Item {
            out: out_name,
            file: 1,
            t: ti,
            codec,
        });
    }
    files.push((te_file.clone(), ts, size));

    // ── CLIP text tower ──
    let (ts, size) = st_header(&clip_file)?;
    for (ti, t) in ts.iter().enumerate() {
        if t.name.ends_with("position_ids") {
            continue;
        }
        let n = t.name.strip_prefix("text_model.").unwrap_or(&t.name);
        items.push(Item {
            out: format!("clip.{n}"),
            file: 2,
            t: ti,
            codec: if prompt_only { Codec::F32 } else { Codec::F16 },
        });
    }
    files.push((clip_file.clone(), ts, size));

    // ── VAE ──
    let (ts, size) = st_header(&vae_file)?;
    for (ti, t) in ts.iter().enumerate() {
        if prompt_only {
            break;
        }
        items.push(Item {
            out: format!("vae.{}", t.name),
            file: 3,
            t: ti,
            codec: o.vae,
        });
    }
    files.push((vae_file.clone(), ts, size));

    let extras: Vec<(&str, Vec<u8>)> = vec![
        ("dit.config_json", dit_cfg_raw.clone()),
        ("te.config_json", serde_json::to_vec_pretty(&text_cfg)?),
        ("audio.config_json", serde_json::to_vec_pretty(&audio_cfg)?),
        ("audio.processor_json", proc_cfg.clone()),
        ("clip.config_json", clip_cfg_raw.clone()),
        ("clip.tokenizer_json", clip_tok.clone()),
        ("vae.config_json", vae_cfg_raw.clone()),
        ("s2v.config_json", serde_json::to_vec_pretty(&defaults)?),
        ("s2v.scheduler_json", sched_raw.clone()),
    ];
    let count = items.len() + extras.len() + 2;
    eprintln!(
        "s2v pack: {n_dit} dit ({}), {} other tensors, te {}, audio {}, vae {}",
        codec_name(o.dit),
        items.len() - n_dit,
        codec_name(o.te),
        codec_name(o.audio),
        codec_name(o.vae),
    );

    let out_path = PathBuf::from(out);
    let temp = gguf::qwen_image_temp_path(&out_path)?;
    let threads: usize = std::env::var("CMF_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8)
        .max(2);
    let mut dtype_bytes: BTreeMap<String, u64> = BTreeMap::new();
    let result = (|| -> anyhow::Result<()> {
        let mut w = CmfStreamWriter::new(&temp, CmfStreamWriter::head_reserve_for(count, 64))?;
        for (n, b) in &extras {
            w.push(n, TensorDtype::U8, &[b.len()], b)?;
        }
        if let Some((text, pooled)) = cond {
            let hidden = text_cfg["hidden_size"].as_u64().context("hidden")? as usize;
            let tb: Vec<u8> = text.iter().flat_map(|v| v.to_le_bytes()).collect();
            let pb: Vec<u8> = pooled.iter().flat_map(|v| v.to_le_bytes()).collect();
            w.push("s2v.default_text", TensorDtype::F32, &[text.len() / hidden, hidden], &tb)?;
            w.push("s2v.default_pooled", TensorDtype::F32, &[1, pooled.len()], &pb)?;
        }
        let mut done = 0usize;
        let mut maps: Vec<Option<memmap2::Mmap>> = (0..files.len()).map(|_| None).collect();
        for chunk in items.chunks(threads) {
            let need: std::collections::BTreeSet<usize> = chunk.iter().map(|it| it.file).collect();
            for (i, m) in maps.iter_mut().enumerate() {
                if !need.contains(&i) {
                    *m = None;
                }
            }
            for it in chunk {
                if maps[it.file].is_none() {
                    let (p, _, size) = &files[it.file];
                    let fh = std::fs::File::open(p)?;
                    let m = unsafe { memmap2::Mmap::map(&fh)? };
                    ensure!(m.len() == *size, "{}: size changed", p.display());
                    maps[it.file] = Some(m);
                }
            }
            let encoded: Vec<anyhow::Result<(TensorDtype, Vec<u8>)>> = std::thread::scope(|s| {
                let hs: Vec<_> = chunk
                    .iter()
                    .map(|it| {
                        let t = &files[it.file].1[it.t];
                        let raw = &maps[it.file].as_ref().unwrap()[t.range.clone()];
                        s.spawn(move || encode(t, raw, it.codec))
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for (it, enc) in chunk.iter().zip(encoded) {
                let (dt, bytes) = enc.with_context(|| it.out.clone())?;
                let t = &files[it.file].1[it.t];
                *dtype_bytes.entry(format!("{:?}", dt)).or_default() += bytes.len() as u64;
                w.push(&it.out, dt, &t.shape, &bytes)?;
                done += 1;
            }
            if done % 256 < chunk.len() || done == items.len() {
                eprintln!(
                    "  {done}/{} tensors ({:.0}s)",
                    items.len(),
                    t_all.elapsed().as_secs_f64()
                );
            }
        }
        drop(maps);
        let n_layers = text_cfg["num_hidden_layers"].as_u64().context("layers")? as usize;
        let hidden = text_cfg["hidden_size"].as_u64().context("hidden")? as usize;
        let heads = text_cfg["num_attention_heads"].as_u64().context("heads")? as usize;
        let arch: ModelArch = serde_json::from_value(serde_json::json!({
            "arch_name": "kandinsky_s2v",
            "hidden_size": hidden,
            "intermediate_size": text_cfg["intermediate_size"],
            "num_layers": n_layers,
            "num_attention_heads": heads,
            "num_kv_heads": text_cfg["num_key_value_heads"],
            "head_dim": hidden / heads,
            "vocab_size": text_cfg["vocab_size"],
            "layer_types": vec!["FullAttention"; n_layers],
            "rms_norm_eps": text_cfg["rms_norm_eps"],
            "max_position_embeddings": text_cfg["max_position_embeddings"],
            "linear_conv_kernel_dim": 0, "linear_num_key_heads": 0, "linear_num_value_heads": 0,
        }))?;
        let quant_type = match o.dit {
            Codec::Q(q) => gguf::quant_type_for(q),
            _ => cortiq_core::types::QuantType::F16,
        };
        let header = CmfHeader {
            format: "cmf".into(),
            version: cortiq_core::CMF_VERSION,
            arch,
            quant_type,
            provenance: Some(serde_json::json!({
                "tool": "cortiq s2v-pack",
                "pipeline": "kandinsky-s2v",
                "components": {"te": "qwen2.5-omni thinker text model",
                               "audio": "qwen2.5-omni audio tower",
                               "clip": "clip-vit-l text tower (pooled)",
                               "dit": "KandinskyS2VTransformer3DModel",
                               "vae": "AutoencoderKLHunyuanVideo"},
                "packed_from": root.file_name().and_then(|s| s.to_str()).unwrap_or("kandinsky-s2v"),
                "dit_codec": codec_name(o.dit),
                "te_codec": codec_name(o.te),
                "audio_codec": codec_name(o.audio),
                "vae_codec": codec_name(o.vae),
                "dit_keep": o.dit_keep,
                "defaults": defaults,
                "source_sha256": source_sha,
                "tensor_name_policy": "diffusers names under dit./te./audio./clip./vae.",
            })),
            tokenizer_config: None,
            section_hashes: None,
            skills: vec![],
            shard: None,
            calibration: None,
            routing: None,
            genome: None,
            lineage: Vec::new(),
            router: None,
            segments: Vec::new(),
        };
        w.finish(&header, None, Some(&vocab))?;
        Ok(())
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_file(&temp);
        return Err(e);
    }
    std::fs::rename(&temp, &out_path)?;
    gguf::qwen_image_sync_parent(&out_path)?;
    let sha = sha256_file(&out_path)?;
    let fname = out_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(out)
        .to_string();
    std::fs::write(format!("{out}.sha256"), format!("{sha}  {fname}\n"))?;
    let size = std::fs::metadata(&out_path)?.len();
    println!(
        "{out}: {count} tensors, {:.3} GB ({:.3} GiB), sha256 {sha}, {:.0}s",
        size as f64 / 1e9,
        size as f64 / (1u64 << 30) as f64,
        t_all.elapsed().as_secs_f64()
    );
    for (k, v) in dtype_bytes {
        println!("  {k}: {:.3} GB", v as f64 / 1e9);
    }
    Ok(())
}

/// `--quant` etc. → the options the CLI hands in.
pub(crate) fn opts(
    dit: Option<&str>,
    te: Option<&str>,
    audio: Option<&str>,
    vae: Option<&str>,
    dit_keep: Vec<String>,
    source_sha: bool,
) -> anyhow::Result<PackOpts> {
    Ok(PackOpts {
        dit: parse_codec(dit.unwrap_or(DEFAULT_DIT_CODEC))?,
        te: parse_codec(te.unwrap_or(DEFAULT_TE_CODEC))?,
        audio: parse_codec(audio.unwrap_or(DEFAULT_AUDIO_CODEC))?,
        vae: parse_codec(vae.unwrap_or(DEFAULT_VAE_CODEC))?,
        dit_keep: dit_keep.into_iter().filter(|k| k != "none").collect(),
        source_sha,
    })
}
