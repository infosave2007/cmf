//! GigaAM end to end on tiny synthetic checkpoints (random weights, the real
//! layout): `cortiq convert` → `cortiq transcribe`, a CTC revision and an
//! RNN-T revision.
//!
//! * the converter writes every profile (`f16`, `q8_2f`, `q4tp`), stores the
//!   stack projection scaled with its transposed twin and leaves the
//!   per-stack copies of the rotary table out;
//! * a clip transcribes to the same text whether it runs alone or in a
//!   batch of files, in one process or two (the ragged batch pads nothing);
//! * a recording over 30 s is cut by the file's voice-activity model and
//!   the segments tile it in order; `--timestamps` gives word times inside
//!   the recording;
//! * the `ssl` revision (no recognition head) is refused.
//!
//! CPU (`CMF_GPU=0`).

use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn cortiq(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_cortiq")).args(args).env("CMF_GPU", "0").env("RUST_LOG", "warn").output().expect("spawn cortiq")
}

fn ok(out: &Output) -> String {
    assert!(out.status.success(), "cortiq failed: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// A deterministic stream of small values.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, shape, vals) in tensors {
        assert_eq!(shape.iter().product::<usize>().max(1), vals.len(), "{name}");
        let start = data.len();
        for v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        header.insert(name.clone(), json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}));
    }
    let mut h = serde_json::to_vec(&Value::Object(header)).unwrap();
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(data);
    std::fs::write(path, out).unwrap();
}

const W: usize = 64; // model width
const MEL: usize = 64;
const DIMS: [usize; 3] = [64, 96, 64];
const HEADS: [usize; 3] = [1, 2, 1];
const KERNELS: [usize; 3] = [5, 0, 5];
const FOLD: [usize; 3] = [1, 2, 1];

/// The encoder and the front end of a tiny Zapformer, named as the checkpoint names them.
fn encoder_tensors(r: &mut Rng) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let mut t: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut lin = |t: &mut Vec<_>, r: &mut Rng, p: &str, rows: usize, cols: usize| {
        t.push((format!("{p}.weight"), vec![rows, cols], r.vec(rows * cols, 2.0 / (cols as f32).sqrt())));
        t.push((format!("{p}.bias"), vec![rows], r.vec(rows, 0.1)));
    };
    let e = "model.encoder.encoder_embed";
    for (i, (ci, co)) in [(1usize, 4usize), (4, 8), (8, 16)].into_iter().enumerate() {
        t.push((format!("{e}.conv{}.weight", i + 1), vec![co, ci, 3, 3], r.vec(co * ci * 9, 0.6)));
        t.push((format!("{e}.conv{}.bias", i + 1), vec![co], r.vec(co, 0.1)));
    }
    t.push((format!("{e}.convnext.depthwise_conv.weight"), vec![16, 1, 3, 3], r.vec(16 * 9, 0.5)));
    t.push((format!("{e}.convnext.depthwise_conv.bias"), vec![16], r.vec(16, 0.1)));
    t.push((format!("{e}.convnext.pointwise_conv1.weight"), vec![48, 16, 1, 1], r.vec(48 * 16, 0.5)));
    t.push((format!("{e}.convnext.pointwise_conv1.bias"), vec![48], r.vec(48, 0.1)));
    t.push((format!("{e}.convnext.pointwise_conv2.weight"), vec![16, 48, 1, 1], r.vec(16 * 48, 0.3)));
    t.push((format!("{e}.convnext.pointwise_conv2.bias"), vec![16], r.vec(16, 0.1)));
    lin(&mut t, r, &format!("{e}.out"), W, 16 * 15);
    let enc = "model.encoder.encoder";
    for s in 0..3 {
        let (d, hw, p) = (DIMS[s], HEADS[s] * 32, format!("{enc}.encoders.{s}"));
        t.push((format!("{p}.proj.weight"), vec![d, FOLD[s] * W], r.vec(d * FOLD[s] * W, 0.6)));
        t.push((format!("{p}.residual_scales"), vec![2], vec![-1.0, 1.0]));
        t.push((format!("{p}.input_scale"), vec![1], vec![1.0]));
        // a stray copy of the rotary table, as the checkpoint carries one a stack
        t.push((format!("{p}.rope_bank.tables.ed32_S10000_b10000.cos"), vec![1, 1, 2, 16], vec![1.0; 32]));
        let l = format!("{p}.layers.0");
        lin(&mut t, r, &format!("{l}.feed_forward1.in_proj"), 4 * d, d);
        lin(&mut t, r, &format!("{l}.feed_forward1.out_proj"), d, 4 * d);
        t.push((format!("{l}.self_attn.in_norm.weight"), vec![d], vec![1.0; d]));
        lin(&mut t, r, &format!("{l}.self_attn.qk_in_proj"), 2 * hw, d);
        lin(&mut t, r, &format!("{l}.self_attn.vg_in_proj"), 3 * hw, d);
        lin(&mut t, r, &format!("{l}.self_attn.out_proj"), d, hw);
        if KERNELS[s] > 0 {
            lin(&mut t, r, &format!("{l}.conv_module.in_proj"), 2 * d, d);
            t.push((format!("{l}.conv_module.depthwise_conv.weight"), vec![d, 1, KERNELS[s]], r.vec(d * KERNELS[s], 0.6)));
            t.push((format!("{l}.conv_module.depthwise_conv.bias"), vec![d], r.vec(d, 0.1)));
            t.push((format!("{l}.conv_module.weighted_mean.weights"), vec![d], r.vec(d, 0.5)));
            lin(&mut t, r, &format!("{l}.conv_module.out_proj"), d, d);
            lin(&mut t, r, &format!("{l}.feed_forward2.in_proj"), 4 * d, d);
            lin(&mut t, r, &format!("{l}.feed_forward2.out_proj"), d, 4 * d);
        }
        t.push((format!("{l}.norm.weight"), vec![d], vec![1.0; d]));
    }
    t.push((format!("{enc}.out_norm.scale"), vec![], vec![0.2]));
    t.push((format!("{enc}.out_norm.eps"), vec![], vec![0.1]));
    // the model's own rotary table: 10 000 positions of 16 angles
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for p in 0..10_000 {
        for i in 0..16 {
            let a = p as f32 / 10000f32.powf(i as f32 / 16.0);
            cos.push(a.cos());
            sin.push(a.sin());
        }
    }
    t.push((format!("{enc}.rope_bank.tables.ed32_S10000_b10000.cos"), vec![1, 1, 10_000, 16], cos));
    t.push((format!("{enc}.rope_bank.tables.ed32_S10000_b10000.sin"), vec![1, 1, 10_000, 16], sin));
    front_end(&mut t, "model.preprocessor");
    t
}

/// A Hann window and a triangular filter bank (the values do not matter, the shapes do).
fn front_end(t: &mut Vec<(String, Vec<usize>, Vec<f32>)>, p: &str) {
    let window: Vec<f32> = (0..320).map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / 320.0).cos()).collect();
    let mut fb = vec![0f32; 161 * MEL];
    for m in 0..MEL {
        for k in 0..161 {
            let c = (m as f32 + 1.0) * 160.0 / (MEL as f32 + 1.0);
            fb[k * MEL + m] = (1.0 - (k as f32 - c).abs() / 4.0).max(0.0);
        }
    }
    t.push((format!("{p}.featurizer.0.spectrogram.window"), vec![320], window));
    t.push((format!("{p}.featurizer.0.mel_scale.fb"), vec![161, MEL], fb));
}

/// A tiny voice-activity Conformer (two layers of 32, one head) whose head bias makes every step speech.
fn write_vad(dir: &Path, r: &mut Rng) {
    let (d, mut t): (usize, Vec<(String, Vec<usize>, Vec<f32>)>) = (32, Vec::new());
    let mut lin = |t: &mut Vec<_>, r: &mut Rng, p: &str, shape: Vec<usize>, scale: f32| {
        let (rows, n): (usize, usize) = (shape[0], shape.iter().product());
        t.push((format!("{p}.weight"), shape, r.vec(n, scale)));
        t.push((format!("{p}.bias"), vec![rows], r.vec(rows, 0.05)));
    };
    lin(&mut t, r, "encoder.pre_encode.conv.0", vec![d, MEL, 3], 0.05);
    lin(&mut t, r, "encoder.pre_encode.conv.2", vec![d, d, 3], 0.1);
    for l in 0..2 {
        let p = format!("encoder.layers.{l}");
        for n in ["norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out", "conv.batch_norm"] {
            t.push((format!("{p}.{n}.weight"), vec![d], vec![1.0; d]));
            t.push((format!("{p}.{n}.bias"), vec![d], vec![0.0; d]));
        }
        for ff in ["feed_forward1", "feed_forward2"] {
            lin(&mut t, r, &format!("{p}.{ff}.linear1"), vec![2 * d, d], 0.3);
            lin(&mut t, r, &format!("{p}.{ff}.linear2"), vec![d, 2 * d], 0.3);
        }
        for a in ["linear_q", "linear_k", "linear_v", "linear_out"] {
            lin(&mut t, r, &format!("{p}.self_attn.{a}"), vec![d, d], 0.3);
        }
        lin(&mut t, r, &format!("{p}.conv.pointwise_conv1"), vec![2 * d, d, 1], 0.3);
        lin(&mut t, r, &format!("{p}.conv.depthwise_conv"), vec![d, 1, 15], 0.2);
        lin(&mut t, r, &format!("{p}.conv.pointwise_conv2"), vec![d, d, 1], 0.3);
    }
    t.push(("head.decoder_layers.0.weight".into(), vec![1, d, 1], r.vec(d, 0.01)));
    t.push(("head.decoder_layers.0.bias".into(), vec![1], vec![4.0]));
    front_end(&mut t, "preprocessor");
    write_safetensors(&dir.join("vad.safetensors"), &t);
    let cfg = json!({
        "model_name": "vad",
        "encoder": {"feat_in": MEL, "subsampling_factor": 4, "n_layers": 2, "d_model": d, "n_heads": 1, "ff_expansion_factor": 2,
            "conv_kernel_size": 15, "conv_norm_type": "layer_norm", "subsampling": "conv1d", "subs_kernel_size": 3,
            "self_attention_model": "rotary", "pos_emb_max_len": 10000},
        "head": {"feat_in": d, "num_classes": 1},
        "vad": {"window_sec": 10.0},
    });
    std::fs::write(dir.join("vad_config.json"), serde_json::to_vec(&cfg).unwrap()).unwrap();
}

fn encoder_cfg() -> Value {
    json!({
        "_target_": "gigaam.encoder.ZapformerEncoder", "activation": "SiLU", "causal_frontend": true, "conv_params": "5,0,5",
        "downsampling_factor": "1,2,1", "dual_mode": true, "encoder_dim": "64,96,64", "feedforward_multiple": "4,4,4",
        "frontend_channels": "4,8,16", "in_channels": MEL, "num_encoder_layers": "1,1,1", "num_heads": "1,2,1",
        "query_head_dim": "32,32,32", "value_head_dim": "32,32,32",
    })
}

const CHARS: [&str; 5] = [" ", "а", "б", "в", "г"];

/// A CTC checkpoint directory: character heads over five symbols.
fn write_ctc(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let mut r = Rng(7);
    let mut t = encoder_tensors(&mut r);
    for head in ["e2e", "raw"] {
        t.push((format!("model.heads.{head}.decoder_layers.0.weight"), vec![CHARS.len() + 1, W, 1], r.vec((CHARS.len() + 1) * W, 1.5)));
        t.push((format!("model.heads.{head}.decoder_layers.0.bias"), vec![CHARS.len() + 1], r.vec(CHARS.len() + 1, 0.1)));
    }
    write_safetensors(&dir.join("model.safetensors"), &t);
    let head = json!({"_target_": "gigaam.decoder.CTCHead", "feat_in": W, "num_classes": CHARS.len() + 1});
    let dec = json!({"_target_": "gigaam.decoding.CTCGreedyDecoding", "model_path": null, "vocabulary": CHARS});
    let cfg = json!({"model_type": "gigaam", "cfg": {"model_name": "ctc", "encoder": encoder_cfg(),
        "heads": {"e2e": head, "raw": head}, "decodings": {"e2e": dec, "raw": dec}}});
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&cfg).unwrap()).unwrap();
    write_vad(dir, &mut r);
}

/// A SentencePiece model file holding `pieces` (the pieces field only).
fn sentencepiece(pieces: &[&str]) -> Vec<u8> {
    let mut out = Vec::new();
    for p in pieces {
        let mut m = vec![0x0a, p.len() as u8];
        m.extend_from_slice(p.as_bytes());
        m.extend_from_slice(&[0x15, 0, 0, 0, 0]);
        out.push(0x0a);
        out.push(m.len() as u8);
        out.extend(m);
    }
    out
}

/// An RNN-T checkpoint directory: an LSTM of 16 and a joint of 16 over six units.
fn write_rnnt(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let mut r = Rng(11);
    let mut t = encoder_tensors(&mut r);
    let pieces = ["<unk>", "\u{2581}да", "\u{2581}нет", "к", "а", "\u{2581}"];
    let (h, classes) = (16usize, pieces.len() + 1);
    for head in ["e2e", "raw"] {
        let p = format!("model.heads.{head}");
        t.push((format!("{p}.decoder.embed.weight"), vec![classes, h], r.vec(classes * h, 1.0)));
        for (n, len) in [("weight_ih_l0", 4 * h * h), ("weight_hh_l0", 4 * h * h), ("bias_ih_l0", 4 * h), ("bias_hh_l0", 4 * h)] {
            let shape = if n.starts_with("weight") { vec![4 * h, h] } else { vec![4 * h] };
            t.push((format!("{p}.decoder.lstm.{n}"), shape, r.vec(len, 0.6)));
        }
        t.push((format!("{p}.joint.enc.weight"), vec![h, W], r.vec(h * W, 1.5)));
        t.push((format!("{p}.joint.enc.bias"), vec![h], r.vec(h, 0.1)));
        t.push((format!("{p}.joint.pred.weight"), vec![h, h], r.vec(h * h, 0.6)));
        t.push((format!("{p}.joint.pred.bias"), vec![h], r.vec(h, 0.1)));
        t.push((format!("{p}.joint.joint_net.1.weight"), vec![classes, h], r.vec(classes * h, 1.5)));
        // a blank that wins most frames, so the transcript is a sentence rather than a wall of units
        let mut bias = r.vec(classes, 0.1);
        bias[classes - 1] = 1.2;
        t.push((format!("{p}.joint.joint_net.1.bias"), vec![classes], bias));
    }
    write_safetensors(&dir.join("model.safetensors"), &t);
    for f in ["e2e_tokenizer.model", "raw_tokenizer.model"] {
        std::fs::write(dir.join(f), sentencepiece(&pieces)).unwrap();
    }
    let head = json!({"_target_": "gigaam.decoder.RNNTHead",
        "decoder": {"num_classes": classes, "pred_hidden": h, "pred_rnn_layers": 1},
        "joint": {"enc_hidden": W, "joint_hidden": h, "num_classes": classes, "pred_hidden": h}});
    let dec = |f: &str| json!({"_target_": "gigaam.decoding.RNNTGreedyDecoding", "max_symbols_per_step": 10, "model_path": f});
    let cfg = json!({"model_type": "gigaam", "cfg": {"model_name": "rnnt", "encoder": encoder_cfg(),
        "heads": {"e2e": head, "raw": head}, "decodings": {"e2e": dec("e2e_tokenizer.model"), "raw": dec("raw_tokenizer.model")}}});
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&cfg).unwrap()).unwrap();
    write_vad(dir, &mut r);
}

/// A 16 kHz mono PCM16 WAV of `secs` seconds: a few drifting tones.
fn write_wav(path: &Path, secs: f32, seed: u32) {
    let n = (secs * 16000.0) as usize;
    let mut data = Vec::with_capacity(n * 2);
    for i in 0..n {
        let t = i as f32 / 16000.0;
        let f = 180.0 + 40.0 * seed as f32 + 300.0 * (t * (0.7 + 0.1 * seed as f32)).sin();
        let v = 0.3 * (2.0 * std::f32::consts::PI * f * t).sin() + 0.2 * (2.0 * std::f32::consts::PI * 2.3 * f * t).sin() * (3.0 * t).cos();
        data.extend_from_slice(&((v * 20000.0) as i16).to_le_bytes());
    }
    let mut w = Vec::new();
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&16000u32.to_le_bytes());
    w.extend_from_slice(&32000u32.to_le_bytes());
    w.extend_from_slice(&2u16.to_le_bytes());
    w.extend_from_slice(&16u16.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&(data.len() as u32).to_le_bytes());
    w.extend_from_slice(&data);
    std::fs::write(path, w).unwrap();
}

fn s(p: &PathBuf) -> &str {
    p.to_str().unwrap()
}

/// The JSON line of every file from `cortiq transcribe … --json`.
fn transcribe(model: &PathBuf, clips: &[&PathBuf], extra: &[&str]) -> Vec<Value> {
    let mut args = vec!["transcribe", s(model)];
    args.extend(clips.iter().map(|c| s(c)));
    args.extend(["--json", "--head", "both"]);
    args.extend(extra);
    ok(&cortiq(&args)).lines().filter(|l| l.starts_with('{')).map(|l| serde_json::from_str(l).unwrap()).collect()
}

#[test]
fn ctc_and_rnnt_convert_and_transcribe() {
    let tmp = tempfile::tempdir().unwrap();
    let (a, b, long) = (tmp.path().join("a.wav"), tmp.path().join("b.wav"), tmp.path().join("long.wav"));
    write_wav(&a, 2.33, 1);
    write_wav(&b, 5.0, 2);
    write_wav(&long, 47.0, 3);
    for (rev, write) in [("ctc", write_ctc as fn(&Path)), ("rnnt", write_rnnt as fn(&Path))] {
        let src = tmp.path().join(format!("src-{rev}"));
        write(&src);
        let mut texts: Vec<String> = Vec::new();
        for quant in ["f16", "q8_2f", "q4tp"] {
            let model = tmp.path().join(format!("{rev}-{quant}.cmf"));
            ok(&cortiq(&["convert", "--model", s(&src), "--output", s(&model), "--quant", quant]));
            let info = ok(&cortiq(&["info", s(&model), "--tensors", "model.encoder.encoder"]));
            assert!(info.contains("encoders.1.proj_t.weight"), "{rev} {quant}: the transposed projection is stored");
            assert!(!info.contains("encoders.0.rope_bank"), "{rev} {quant}: a stack's copy of the rotary table is left out");
            // alone, in a batch of files, and again: the same transcripts
            let alone = transcribe(&model, &[&a], &[]);
            let both = transcribe(&model, &[&a, &b], &[]);
            assert_eq!(alone[0]["e2e"]["text"], both[0]["e2e"]["text"], "{rev} {quant}");
            assert_eq!(alone[0]["raw"]["text"], both[0]["raw"]["text"], "{rev} {quant}");
            for (x, y) in both.iter().zip(&transcribe(&model, &[&a, &b], &[])) {
                assert_eq!((&x["e2e"], &x["raw"]), (&y["e2e"], &y["raw"]), "{rev} {quant}: a second run");
            }
            assert!((both[1]["duration"].as_f64().unwrap() - 5.0).abs() < 1e-6);
            texts.push(alone[0]["e2e"]["text"].as_str().unwrap().to_string());
            if quant != "f16" {
                continue;
            }
            // a recording over 30 s: the voice-activity model cuts it, the segments tile it in order
            let l = &transcribe(&model, &[&long], &["--timestamps"])[0];
            let segs = l["e2e"]["segments"].as_array().unwrap();
            assert!(segs.len() >= 2, "{rev}: {} segments for 47 s", segs.len());
            let mut prev_end = 0.0;
            for sg in segs {
                let (st, en) = (sg["start"].as_f64().unwrap(), sg["end"].as_f64().unwrap());
                assert!(st >= prev_end - 1e-9 && en > st && en - st <= 30.0 + 1e-6 && en <= 47.0 + 1e-6, "{rev}: segment {st}..{en}");
                for w in sg["words"].as_array().unwrap() {
                    let (ws, we) = (w["start"].as_f64().unwrap(), w["end"].as_f64().unwrap());
                    assert!(ws >= st - 1e-6 && we <= en + 0.05 && we > ws, "{rev}: word {ws}..{we} in {st}..{en}");
                }
                prev_end = en;
            }
            let joined: Vec<&str> = segs.iter().map(|x| x["text"].as_str().unwrap()).collect();
            assert_eq!(l["e2e"]["text"].as_str().unwrap(), joined.join(" "));
            // --vad on a short clip takes the same path and still covers the clip
            let v = &transcribe(&model, &[&b], &["--vad"])[0];
            assert!(!v["e2e"]["segments"].as_array().unwrap().is_empty());
        }
        assert!(!texts[0].is_empty(), "{rev}: an empty transcript says nothing about the path");
    }
    // the encoder-only revision has no head to transcribe with
    let ssl = tmp.path().join("src-ssl");
    std::fs::create_dir_all(&ssl).unwrap();
    std::fs::write(ssl.join("config.json"), serde_json::to_vec(&json!({"model_type": "gigaam", "cfg": {"model_name": "ssl", "encoder": encoder_cfg()}})).unwrap()).unwrap();
    let out = cortiq(&["convert", "--model", s(&ssl), "--output", s(&tmp.path().join("ssl.cmf")), "--quant", "q8_2f"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("no recognition head"));
}
