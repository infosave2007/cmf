//! `cortiq quantize-moe` end to end on the engine's synthetic five-layer
//! DeepSeek-V4.1 fixture (`cortiq-engine/tests/common/dsv41_fixture.rs`:
//! ratio-0/2/1 layers, two KV sources, an index consumer, candidate
//! pre-filtering, Engram, q4tp and q2tp gate/up expert slots), run as the
//! real binary on the CPU (`CMF_GPU=0`, `CMF_MOE_HOST=1`):
//!
//! * the capture through the per-token forward, every window led by BOS,
//!   the V4.1 prior default;
//! * `CMF_MOE_CAPTURE_DIR`: saved by the first run, reused by the next ones
//!   without a calibration forward (same output bytes), refused for another
//!   run's tokens;
//! * the slot checks (a q4tp payload does not fit the q2tp slots) and the
//!   hole-punching probe before any capture;
//! * q2tp `--output`, q2tp `--in-place`, and DIAG with route weighting.
//!
//!     cargo test -p cortiq-cli --test moegptq_dsv41_e2e -- --nocapture

// The fixture's `json!` config nests past the default macro depth.
#![recursion_limit = "256"]

#[path = "../../cortiq-engine/tests/common/dsv41_fixture.rs"]
#[allow(dead_code)]
mod dsv41_fixture;

use dsv41_fixture::{DIM, EXPERTS, Fixture, INTER, N_LAYERS};
use std::path::{Path, PathBuf};
use std::process::Command;

fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, v) in tensors {
        let start = data.len();
        data.extend(v.iter().flat_map(|x| x.to_le_bytes()));
        header.insert(name.clone(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}));
    }
    let h = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    out.extend_from_slice(&data);
    std::fs::write(path, out).unwrap();
}

/// The "original checkpoint" of the fixture's experts: plain F32 `w1`
/// (gate), `w3` (up), `w2` (down) per routed expert and the shared expert,
/// one shard per layer, with its index.
fn write_source(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let mut rng = dsv41_fixture::Rng::new(0x5EED);
    let mut mat = |rows: usize, cols: usize| -> Vec<f32> { (0..rows * cols).map(|_| (rng.normal() * 0.08) as f32).collect() };
    let mut wmap = serde_json::Map::new();
    for li in 0..N_LAYERS {
        let shard = format!("model-{:05}-of-{:05}.safetensors", li + 1, N_LAYERS);
        let mut t = Vec::new();
        let stems = (0..EXPERTS).map(|e| format!("layers.{li}.ffn.experts.{e}")).chain([format!("layers.{li}.ffn.shared_experts")]);
        for stem in stems {
            t.push((format!("{stem}.w1.weight"), vec![INTER, DIM], mat(INTER, DIM)));
            t.push((format!("{stem}.w3.weight"), vec![INTER, DIM], mat(INTER, DIM)));
            t.push((format!("{stem}.w2.weight"), vec![DIM, INTER], mat(DIM, INTER)));
        }
        for (n, ..) in &t {
            wmap.insert(n.clone(), serde_json::json!(shard));
        }
        write_safetensors(&dir.join(&shard), &t);
    }
    std::fs::write(dir.join("model.safetensors.index.json"), serde_json::to_vec(&serde_json::json!({"weight_map": wmap})).unwrap()).unwrap();
}

/// Run the binary on the CPU; (success, stderr).
fn cortiq(args: &[&str], env: &[(&str, &str)]) -> (bool, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_cortiq"));
    cmd.args(args).env("CMF_GPU", "0").env("CMF_MOE_WORKERS", "2");
    for k in [
        "CMF_MOE_HOST",
        "CMF_MOE_DIAG",
        "CMF_MOE_RANGE",
        "CMF_MOE_PRIOR",
        "CMF_MOE_ROUTE_W",
        "CMF_MOE_CAPTURE_DIR",
        "CMF_MOE_CAPTURE_ONLY",
        "CMF_MOE_CAPTURE_FORCE",
        "CMF_MOE_NO_PUNCH",
        "CMF_MOE_NO_SPACE_CHECK",
        "CMF_MOE_STOP_AFTER",
        "CMF_MOE_HOLDOUT",
        "CMF_DSV41_O_FP8",
    ] {
        cmd.env_remove(k);
    }
    cmd.envs(env.iter().copied());
    let out = cmd.output().expect("run cortiq");
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    (out.status.success(), err)
}

#[test]
fn quantize_moe_on_the_v41_fixture() {
    let root = std::env::temp_dir().join(format!("cortiq-moe-v41-e2e-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let fx = Fixture::build(1);
    let calib: PathBuf = fx.write("moe-e2e");
    let src = root.join("src");
    write_source(&src);
    let text = root.join("calib.txt");
    std::fs::write(&text, "The quick brown fox jumps over the lazy dog while the five boxing wizards jump quickly. ".repeat(40)).unwrap();
    let cap = root.join("capture");
    let (calib_s, src_s, text_s, cap_s) = (calib.to_str().unwrap(), src.to_str().unwrap(), text.to_str().unwrap(), cap.to_str().unwrap());
    let out1 = root.join("q2tp-1.cmf");
    let out2 = root.join("q2tp-2.cmf");
    let base = |out: &Path, tokens: &'static str| -> Vec<String> {
        ["quantize-moe", calib_s, "--hf", src_s, "--calib", text_s, "--tokens", tokens, "--window", "50", "--threads", "2", "--scratch", root.join("scratch").to_str().unwrap(), "--output", out.to_str().unwrap()]
            .iter()
            .map(|s| s.to_string())
            .collect()
    };
    fn args(v: &[String]) -> Vec<&str> {
        v.iter().map(String::as_str).collect()
    }
    let host = ("CMF_MOE_HOST", "1");
    let capdir = ("CMF_MOE_CAPTURE_DIR", cap_s);

    // without a GPU and without CMF_MOE_HOST=1 the run stops at once
    let (ok, err) = cortiq(&args(&base(&out1, "100")), &[]);
    assert!(!ok && err.contains("no wgpu GPU (set CMF_MOE_HOST=1"), "{err}");

    // --output naming the calibration file (here through a symlink) is
    // refused before anything opens it for writing
    let calib_bytes = std::fs::read(&calib).unwrap();
    #[cfg(unix)]
    {
        let link = root.join("calib-link.cmf");
        std::os::unix::fs::symlink(&calib, &link).unwrap();
        let (ok, err) = cortiq(&args(&base(&link, "100")), &[host]);
        assert!(!ok && err.contains("is the calibration file"), "{err}");
    }
    let (ok, err) = cortiq(&args(&base(&calib, "100")), &[host]);
    assert!(!ok && err.contains("is the calibration file"), "{err}");
    assert_eq!(std::fs::read(&calib).unwrap(), calib_bytes, "the calibration file is untouched");

    // 1: q2tp into a new file; the capture is saved
    let (ok, err) = cortiq(&args(&base(&out1, "100")), &[host, capdir]);
    eprintln!("{err}");
    assert!(ok, "first run failed:\n{err}");
    assert!(err.contains("every calibration window starts with BOS"), "{err}");
    assert!(err.contains("prior: n0 = 64 pseudo-rows") && err.contains("V4.1 default"), "the V4.1 prior default: {err}");
    assert!(err.contains("captured 2/2 windows"), "{err}");
    assert!(err.contains("saving the capture"), "{err}");
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(cap.join("meta.json")).unwrap()).unwrap();
    assert_eq!(meta["layers"], serde_json::json!((0..N_LAYERS).collect::<Vec<_>>()));
    assert_eq!((meta["ntok"].as_u64(), meta["windows"].as_u64(), meta["route_weights"].as_bool()), (Some(100), Some(2), Some(true)));
    let m1 = cortiq_core::CmfModel::open(&out1).unwrap();
    assert!(m1.verify().is_empty(), "{:?}", m1.verify());
    let prov = &m1.header.provenance.as_ref().unwrap()["gptq_moe"];
    assert_eq!((prov["prior"].as_f64(), prov["codec"].as_str()), (Some(64.0), Some("q2tp")));
    // the provenance names the capture: its tokens, forward settings, binary
    let cp = &prov["calibration_capture"];
    assert_eq!(cp["token_hash"], meta["token_hash"]);
    assert_eq!((cp["env"]["CMF_GPU"].as_str(), meta["env"]["CMF_GPU"].as_str()), (Some("0"), Some("0")));
    assert!(cp["binary"].as_str().is_some_and(|b| b.contains("cortiq")), "{cp}");
    let gate = m1.tensor("model.layers.0.mlp.experts.0.gate_proj.weight").unwrap();
    assert_eq!(gate.dtype, cortiq_core::types::TensorDtype::Q2TiledP);

    // capture only, into a second directory: the same rows as the first
    // run saved, and no output file
    let cap2 = root.join("capture2");
    let none = root.join("not-written.cmf");
    let (ok, err) = cortiq(&args(&base(&none, "100")), &[host, ("CMF_MOE_CAPTURE_DIR", cap2.to_str().unwrap()), ("CMF_MOE_CAPTURE_ONLY", "1")]);
    assert!(ok, "capture-only run failed:\n{err}");
    assert!(err.contains("no expert was encoded") && !err.contains("layer 0: routed rows"), "{err}");
    assert!(!none.exists(), "capture only writes no model file");
    for li in 0..N_LAYERS {
        for ext in ["x", "picks", "w"] {
            let f = format!("L{li}.{ext}");
            assert_eq!(std::fs::read(cap.join(&f)).unwrap(), std::fs::read(cap2.join(&f)).unwrap(), "{f}");
        }
    }

    // 2: the same tokens again: loaded from the capture directory, no
    // forward, the same bytes
    let (ok, err) = cortiq(&args(&base(&out2, "100")), &[host, capdir]);
    assert!(ok, "second run failed:\n{err}");
    assert!(err.contains("no calibration forward") && !err.contains("captured "), "{err}");
    assert_eq!(std::fs::read(&out1).unwrap(), std::fs::read(&out2).unwrap(), "a reused capture writes the same file");

    // a capture directory that cannot be written: the run warns and goes on
    // from the rows in memory (the same file); capture-only fails
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let ro = root.join("capture-ro");
        std::fs::create_dir_all(&ro).unwrap();
        std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(0o555)).unwrap();
        if std::fs::write(ro.join("probe"), b"").is_err() {
            let out3 = root.join("q2tp-3.cmf");
            let (ok, err) = cortiq(&args(&base(&out3, "100")), &[host, ("CMF_MOE_CAPTURE_DIR", ro.to_str().unwrap())]);
            assert!(ok && err.contains("warning: saving layer 0's capture failed"), "{err}");
            // the same tensors (the provenance names another capture directory)
            let m3 = cortiq_core::CmfModel::open(&out3).unwrap();
            assert_eq!(m3.tensors.len(), m1.tensors.len());
            for t in &m1.tensors {
                assert_eq!(m3.entry_bytes(m3.tensor(&t.name).unwrap()), m1.entry_bytes(t), "{}", t.name);
            }
            let (ok, err) = cortiq(&args(&base(&none, "100")), &[host, ("CMF_MOE_CAPTURE_DIR", ro.to_str().unwrap()), ("CMF_MOE_CAPTURE_ONLY", "1")]);
            assert!(!ok && err.contains("saving layer 0's capture"), "{err}");
            assert_eq!(std::fs::read_dir(&ro).unwrap().count(), 0, "no partial file is left");
        }
        std::fs::set_permissions(&ro, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    // 3: other tokens: the saved capture is refused, never overwritten
    let (ok, err) = cortiq(&args(&base(&root.join("x.cmf"), "120")), &[host, capdir]);
    assert!(!ok && err.contains("holds the capture of another run") && err.contains("calibration tokens differ"), "{err}");
    // the same tokens with another forward setting (the reference
    // act_quant(o)): refused unless forced
    let (ok, err) = cortiq(&args(&base(&root.join("x.cmf"), "100")), &[host, capdir, ("CMF_DSV41_O_FP8", "1")]);
    assert!(!ok && err.contains("CMF_DSV41_O_FP8 unset saved, \"1\" now") && err.contains("CMF_MOE_CAPTURE_FORCE=1"), "{err}");
    assert!(!root.join("x.cmf").exists(), "refused before the output is opened");

    // 4: DIAG with both down Hessians on two layers, from the saved capture
    let ad: Vec<String> = base(Path::new("/dev/null"), "100").into_iter().chain(["--min-rows".into(), "8".into()]).collect();
    let (ok, err) = cortiq(&args(&ad), &[host, capdir, ("CMF_MOE_DIAG", "0,64"), ("CMF_MOE_ROUTE_W", "diag"), ("CMF_MOE_RANGE", "2..4"), ("CMF_MOE_HOLDOUT", "2")]);
    eprintln!("{err}");
    assert!(ok, "diag failed:\n{err}");
    assert!(err.contains("DIAG total") && err.contains("\"64+w\"") && err.contains("route-weighted"), "{err}");

    // 5: in place. A q4tp payload does not fit the fixture's q2tp gate/up
    // slots (odd layers): refused before any capture
    let inplace = root.join("inplace.cmf");
    std::fs::copy(&calib, &inplace).unwrap();
    let mut a = base(&out1, "100");
    a.truncate(a.len() - 2);
    a[1] = inplace.to_str().unwrap().into();
    let a4: Vec<String> = a.iter().cloned().chain(["--in-place".into(), "--codec".into(), "q4tp".into()]).collect();
    let (ok, err) = cortiq(&args(&a4), &[host, capdir]);
    assert!(!ok && err.contains("a q4tp payload fits only a q4tp slot") && !err.contains("captured"), "{err}");
    // q2tp in place: the hole-punching probe first
    let a2: Vec<String> = a.iter().cloned().chain(["--in-place".into(), "--codec".into(), "q2tp".into()]).collect();
    if cfg!(not(target_os = "linux")) {
        let (ok, err) = cortiq(&args(&a2), &[host, capdir]);
        assert!(!ok && err.contains("CMF_MOE_NO_PUNCH=1"), "{err}");
    }
    let before = std::fs::read(&inplace).unwrap();
    // the capture was taken from the original file: another calibration
    // file path is refused unless forced (the copy holds the same bytes)
    let (ok, err) = cortiq(&args(&a2), &[host, capdir, ("CMF_MOE_NO_PUNCH", "1")]);
    assert!(!ok && err.contains("captured differently") && err.contains("calibration file"), "{err}");
    assert_eq!(std::fs::read(&inplace).unwrap(), before, "refused before anything is written");
    let (ok, err) = cortiq(&args(&a2), &[host, capdir, ("CMF_MOE_NO_PUNCH", "1"), ("CMF_MOE_CAPTURE_FORCE", "1")]);
    assert!(ok, "q2tp in place failed:\n{err}");
    assert!(err.contains("no calibration forward"), "{err}");
    let mi = cortiq_core::CmfModel::open(&inplace).unwrap();
    assert!(mi.verify().is_empty(), "{:?}", mi.verify());
    assert_eq!(std::fs::metadata(&inplace).unwrap().len() as usize, before.len(), "in place keeps the file length");
    // the header keeps the converter's provenance; the recipe is in a sidecar
    let side: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("inplace.cmf.gptq_moe.json")).unwrap()).unwrap();
    let run0 = &side["runs"][0];
    assert_eq!((run0["weight_quant"].as_str(), run0["gptq_moe"]["layers"].as_str()), (Some("q2tp"), Some("0..5")), "{side}");
    assert_eq!(run0["gptq_moe"]["calibration_capture"]["token_hash"], meta["token_hash"]);
    assert!(run0["gptq_moe"]["calibration_capture"]["reused_despite"][0].as_str().is_some_and(|w| w.contains("calibration file")), "{side}");
    // the in-place experts are the --output run's experts
    for li in 0..N_LAYERS {
        for name in [format!("model.layers.{li}.mlp.experts.1.up_proj.weight"), format!("model.layers.{li}.mlp.shared_experts.down_proj.weight")] {
            let (a, b) = (mi.tensor(&name).unwrap(), m1.tensor(&name).unwrap());
            assert_eq!((a.dtype, mi.entry_bytes(a)), (b.dtype, m1.entry_bytes(b)), "{name}");
        }
    }
    let _ = std::fs::remove_dir_all(&root);
    let _ = std::fs::remove_dir_all(calib.parent().unwrap());
}
