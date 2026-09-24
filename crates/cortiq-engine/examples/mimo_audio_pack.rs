//! DEVELOPMENT ONLY: write an audio-only MiMo companion CMF straight from the
//! HF checkpoint, every tensor in its source dtype (BF16 matrices, F32 RVQ
//! codebooks) under the companion names (`audio_tokenizer.encoder.*`,
//! `audio_encoder.*`, `speech_embeddings.*`) plus the two config blobs.
//!
//! It exists so the CMF load path and the q4tp codec can be gated before the
//! real converter (`cortiq convert --mimo-towers`) lands:
//!
//! ```text
//! mimo_audio_pack --src /root/mimo/src --out audio.exact.mm.cmf
//! cortiq requant audio.exact.mm.cmf --output audio.q4tp.mm.cmf --quant q4tp-quantize
//! ```
//!
//! `requant q4tp-quantize` turns the 2-D matrices into q4tp and leaves the
//! codebooks, the speech-embedding tables and every rank≠2 tensor as they are.
//!
//! That plain q4tp fails the audio gates (G9.4, G10.1). The companion that
//! passes keeps the tokenizer layers at q8_2f and puts GPTQ q4tp on the
//! LLM-side encoder, with Hessians from `mimo_audio_dump calib`:
//!
//! ```text
//! mimo_audio_dump calib --src audio.exact.mm.cmf --wav-dir CALIB_WAVS --out hess.bin
//! cortiq quantize-gptq audio.exact.mm.cmf --calib /dev/null --output audio.mm.cmf \
//!     --codec q4tp --hessians hess.bin --act-order \
//!     --tensor-quant 'audio_tokenizer.encoder.layers.*=q8_2f'
//! ```

use cortiq_core::format::{CmfHeader, CmfModel, TensorSpec};
use cortiq_core::types::{ModelArch, QuantType, TensorDtype};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |n: &str| {
        args.iter()
            .position(|a| a == n)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let src = PathBuf::from(get("--src").expect("--src HF_DIR"));
    let out = PathBuf::from(get("--out").expect("--out X.mm.cmf"));
    let (raw, blobs) =
        cortiq_engine::mimo_audio::read_hf_audio_tensors(&src).expect("read checkpoint");
    let mut specs = Vec::with_capacity(raw.len() + blobs.len());
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    for t in raw {
        let dtype = match t.dtype.as_str() {
            "BF16" => TensorDtype::Bf16,
            "F16" => TensorDtype::F16,
            "F32" => TensorDtype::F32,
            other => panic!("{}: dtype {other}", t.name),
        };
        let group = t.name.split('.').next().unwrap_or("").to_string();
        *counts.entry(group).or_default() += 1;
        specs.push(TensorSpec {
            name: t.name,
            dtype,
            shape: t.shape,
            data: t.bytes,
        });
    }
    for (name, bytes) in blobs {
        specs.push(TensorSpec {
            name,
            dtype: TensorDtype::U8,
            shape: vec![bytes.len()],
            data: bytes,
        });
    }
    let arch: ModelArch = serde_json::from_value(serde_json::json!({
        "arch_name": "mimo_v2_mm",
        "hidden_size": 4096,
        "intermediate_size": 0,
        "num_layers": 0,
        "num_attention_heads": 1,
        "num_kv_heads": 1,
        "head_dim": 64,
        "vocab_size": 0,
        "layer_types": [],
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 0,
        "linear_conv_kernel_dim": 0,
        "linear_num_key_heads": 0,
        "linear_num_value_heads": 0
    }))
    .expect("arch");
    let header = CmfHeader {
        format: "cmf".into(),
        version: cortiq_core::CMF_VERSION,
        arch,
        quant_type: QuantType::BF16,
        provenance: Some(serde_json::json!({
            "tool": "mimo_audio_pack (development; audio towers only)",
            "base_arch": "mimo_v2",
            "source": src.display().to_string(),
            "codec": "source dtypes: BF16 tower tensors, F32 RVQ codebooks",
            "groups": counts,
        })),
        tokenizer_config: None,
        section_hashes: None,
        skills: vec![],
        shard: None,
        calibration: None,
        routing: None,
    };
    CmfModel::write(&out, &header, &specs, None, None).expect("write");
    println!("{}: {} tensors {:?}", out.display(), specs.len(), counts);
}
