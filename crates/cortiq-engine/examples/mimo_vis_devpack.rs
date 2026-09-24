//! Dev-only packer: copy the MiMo vision tower (`visual.*`) out of an HF
//! safetensors shard into a small CMF, byte-for-byte in the source dtype,
//! with the checkpoint `config.json` as the `mm.config_json` blob.
//!
//! It exists so the vision work (parity, q4tp quality) is not blocked on
//! the companion converter: the engine loads `visual.*` by name from any
//! CMF, and `cortiq requant --quant q4tp-quantize` turns this BF16 file
//! into the q4tp variant with the converter's own codec.
//!
//!     cargo run --release -p cortiq-engine --example mimo_vis_devpack -- \
//!         SHARD.safetensors config.json OUT.cmf [PREFIX=visual.]

use cortiq_core::format::{CmfHeader, CmfModel, TensorSpec};
use cortiq_core::types::{ModelArch, QuantType, TensorDtype};
use std::io::{Read, Seek, SeekFrom};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: mimo_vis_devpack SHARD.safetensors config.json OUT.cmf [PREFIX]");
        std::process::exit(2);
    }
    let prefix = args.get(4).map(String::as_str).unwrap_or("visual.");
    let mut f = std::fs::File::open(&args[1]).expect("open safetensors");
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8).unwrap();
    let hlen = u64::from_le_bytes(len8) as usize;
    let mut hbuf = vec![0u8; hlen];
    f.read_exact(&mut hbuf).unwrap();
    let header: serde_json::Value = serde_json::from_slice(&hbuf).expect("header json");
    let base = 8 + hlen as u64;
    let mut names: Vec<&String> = header
        .as_object()
        .unwrap()
        .keys()
        .filter(|k| k.starts_with(prefix))
        .collect();
    names.sort();
    let mut specs = Vec::with_capacity(names.len() + 1);
    let config = std::fs::read(&args[2]).expect("read config.json");
    serde_json::from_slice::<serde_json::Value>(&config).expect("config.json parses");
    specs.push(TensorSpec {
        name: "mm.config_json".into(),
        dtype: TensorDtype::U8,
        shape: vec![config.len()],
        data: config,
    });
    let mut bytes_total = 0usize;
    for name in &names {
        let meta = &header[name.as_str()];
        let dtype = match meta["dtype"].as_str().unwrap() {
            "BF16" => TensorDtype::Bf16,
            "F16" => TensorDtype::F16,
            "F32" => TensorDtype::F32,
            other => panic!("{name}: dtype {other} not supported"),
        };
        let shape: Vec<usize> = meta["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let off = meta["data_offsets"].as_array().unwrap();
        let (a, b) = (off[0].as_u64().unwrap(), off[1].as_u64().unwrap());
        let mut data = vec![0u8; (b - a) as usize];
        f.seek(SeekFrom::Start(base + a)).unwrap();
        f.read_exact(&mut data).unwrap();
        bytes_total += data.len();
        specs.push(TensorSpec {
            name: (*name).clone(),
            dtype,
            shape,
            data,
        });
    }
    let arch: ModelArch = serde_json::from_value(serde_json::json!({
        "arch_name": "mimo_v2_mm",
        "hidden_size": 4096,
        "intermediate_size": 0,
        "num_layers": 0,
        "num_attention_heads": 1,
        "num_kv_heads": 1,
        "head_dim": 1,
        "vocab_size": 0,
        "layer_types": [],
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 0,
        "linear_conv_kernel_dim": 0,
        "linear_num_key_heads": 0,
        "linear_num_value_heads": 0
    }))
    .unwrap();
    let hdr = CmfHeader {
        format: "cmf".into(),
        version: cortiq_core::CMF_VERSION,
        arch,
        quant_type: QuantType::BF16,
        provenance: Some(serde_json::json!({
            "tool": "mimo_vis_devpack (dev only)",
            "source": args[1],
            "prefix": prefix,
        })),
        tokenizer_config: None,
        section_hashes: None,
        skills: vec![],
        shard: None,
        calibration: None,
        routing: None,
    };
    CmfModel::write(&args[3], &hdr, &specs, None, None).expect("write cmf");
    println!(
        "wrote {} ({} tensors + config, {:.1} MB payload)",
        args[3],
        names.len(),
        bytes_total as f64 / 1e6
    );
}
