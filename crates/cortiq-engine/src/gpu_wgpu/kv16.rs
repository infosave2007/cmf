//! f16 K/V mirrors for the whole-token and batch graphs.
//!
//! The device mirror of a full-attention layer held f32 rows: 64 KB a token
//! on Qwen3.5-4B (8 layers × 4 KV heads × 256 wide × K and V) where
//! llama.cpp's cache keeps f16, and 2.1 GB read for every decoded token at
//! 32 000 positions. Here a uniform-geometry mirror (`kv_mirror_ensure`)
//! stores f16 rows when its sequence asked for it (`note`), and the kernels
//! that touch it come in f16 twins: the append writes f16, the attends load
//! f16 and do every sum in f32 as before. The prompt's matrix-unit tile
//! (`gqa_attend_part_xc2`) already ate f16 planes made from the f32 mirror;
//! an f16 mirror IS that plane, so the copy goes away.
//!
//! A sequence asks only when nothing it will run needs f32 rows: no MTP head,
//! no drafter, no O(1) seal, no block preview (the pipeline decides, see
//! `Pipeline::kv16_allowed`). A mirror keeps the dtype it was made with; a
//! path without an f16 twin refuses an f16 mirror instead of reading it.
//! `CMF_GPU_KV16=0` keeps f32 mirrors everywhere.

use super::*;

/// The device can hold and read f16 mirrors: shader f16, and the 32-lane
/// subgroups the flash-decoding attend needs.
pub(crate) fn device_ok(c: &Ctx) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CMF_GPU_KV16").as_deref() != Ok("0"))
        && c.device.features().contains(wgpu::Features::SHADER_F16)
        && c.device.features().contains(wgpu::Features::SUBGROUP)
        && c.adapter_info.subgroup_min_size == 32
        && c.adapter_info.subgroup_max_size == 32
}

/// Whether each sequence (graph kv id) wants f16 mirrors: the first note
/// after a reset decides and holds until `forget`.
fn notes() -> &'static std::sync::Mutex<HashMap<u64, bool>> {
    static N: std::sync::OnceLock<std::sync::Mutex<HashMap<u64, bool>>> = std::sync::OnceLock::new();
    N.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

pub fn note(kv_id: u64, allowed: bool) {
    let mut n = notes().lock().unwrap();
    if !n.contains_key(&kv_id) && std::env::var("CMF_GPU_DEBUG").is_ok() {
        eprintln!("kv16: sequence {kv_id} allowed={allowed}");
    }
    n.entry(kv_id).or_insert(allowed);
}

pub(crate) fn forget(kv_id: u64) {
    notes().lock().unwrap().remove(&kv_id);
}

/// A new uniform mirror of this sequence is made f16: it asked, and both
/// graphs' f16 twins came up on this device.
pub(crate) fn wanted(c: &Ctx, kv_id: u64) -> bool {
    device_ok(c)
        && notes().lock().unwrap().get(&kv_id).copied().unwrap_or(false)
        && pipes(c).is_some()
        && super::dense_batch::attn_for(c, true).is_some()
}

/// `src` with the storage arrays `names` in f16: their declarations take
/// f16 elements (`array<f32>` → `array<f16>`, `array<vec4<f32>>` →
/// `array<vec4<f16>>`), every read `name[i]` becomes `f32(name[i])` /
/// `vec4<f32>(name[i])` and every store `name[i] = e;` stores `f16(e)` /
/// `vec4<f16>(e)`. Everything else — the order of every sum — is the f32
/// kernel's. Panics on a name the source does not declare (a renamed
/// binding must not silently stay f32 under an f16 buffer).
pub(crate) fn half_src(src: &str, names: &[&str]) -> String {
    let mut vec4 = HashMap::new();
    let mut lines = Vec::new();
    for line in src.lines() {
        let mut l = line.to_string();
        if l.contains("var<storage") {
            for &n in names {
                let at = l.find(&format!(" {n} ")).or_else(|| l.find(&format!(" {n}:")));
                if let Some(p) = at.filter(|&p| l[p..].contains("array<")) {
                    let tail = l[p..].to_string();
                    let (t, v4) = if tail.contains("array<vec4<f32>>") {
                        (tail.replacen("array<vec4<f32>>", "array<vec4<f16>>", 1), true)
                    } else if tail.contains("array<f32>") {
                        (tail.replacen("array<f32>", "array<f16>", 1), false)
                    } else {
                        panic!("half_src: {n} is not an f32 or vec4<f32> array");
                    };
                    l = format!("{}{}", &l[..p], t);
                    vec4.insert(n, v4);
                }
            }
        }
        lines.push(l);
    }
    for &n in names {
        assert!(vec4.contains_key(n), "half_src: {n} is not declared");
    }
    let mut out = String::from("enable f16;\n");
    out.push_str(&rewrite(&lines.join("\n"), names, &vec4));
    out.push('\n');
    out
}

/// The access rewrite of `half_src` over `text` (recursive: an index or a
/// stored expression may itself read an f16 array).
fn rewrite(text: &str, names: &[&str], vec4: &HashMap<&str, bool>) -> String {
    let ident = |ch: char| ch.is_ascii_alphanumeric() || ch == '_';
    let b = text.as_bytes();
    let mut out = String::with_capacity(text.len() + 64);
    let mut i = 0;
    while i < text.len() {
        // Comments are copied as they are.
        if text[i..].starts_with("//") {
            let end = text[i..].find('\n').map_or(text.len(), |e| i + e);
            out.push_str(&text[i..end]);
            i = end;
            continue;
        }
        let hit = names.iter().find(|&&n| {
            text[i..].starts_with(n)
                && text[i + n.len()..].starts_with('[')
                && (i == 0 || !ident(text[..i].chars().next_back().unwrap()))
        });
        let Some(&n) = hit else {
            let ch = text[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
            continue;
        };
        let open = i + n.len();
        let mut depth = 0usize;
        let mut close = open;
        for (j, &c) in b.iter().enumerate().skip(open) {
            match c {
                b'[' => depth += 1,
                b']' => {
                    depth -= 1;
                    if depth == 0 {
                        close = j;
                        break;
                    }
                }
                _ => {}
            }
        }
        let access = format!("{n}[{}]", rewrite(&text[open + 1..close], names, vec4));
        let rest = &text[close + 1..];
        let trimmed = rest.trim_start();
        let (t32, t16) = if vec4[n] { ("vec4<f32>", "vec4<f16>") } else { ("f32", "f16") };
        if trimmed.starts_with('=') && !trimmed.starts_with("==") {
            // A store: `name[i] = e;` → `name[i] = f16(e);`.
            let eq = close + 1 + (rest.len() - trimmed.len());
            let semi = eq + text[eq..].find(';').expect("half_src: store without ';'");
            out.push_str(&access);
            out.push_str(" = ");
            out.push_str(t16);
            out.push('(');
            out.push_str(&rewrite(text[eq + 1..semi].trim(), names, vec4));
            out.push_str(");");
            i = semi + 1;
        } else {
            out.push_str(t32);
            out.push('(');
            out.push_str(&access);
            out.push(')');
            i = close + 1;
        }
    }
    out
}

/// The slice of `WGSL` (the main module) between two markers, both
/// expected exactly once.
fn cut<'a>(src: &'a str, from: &str, to: &str) -> &'a str {
    let a = src.find(from).unwrap_or_else(|| panic!("kv16: marker {from:?} missing"));
    let b = a + src[a..].find(to).unwrap_or_else(|| panic!("kv16: marker {to:?} missing"));
    &src[a..b]
}

/// `kv_append` with f16 mirror rows.
pub(crate) fn kv_append_src() -> String {
    half_src(
        cut(
            super::WGSL,
            "struct KvP { nkv: u32, hd: u32, cap: u32, stored: u32 };",
            "// Grouped decode attention, one 32-thread workgroup per Q-head.",
        ),
        &["kv_kb", "kv_vb"],
    )
}

/// `gqa_attend_dec` over f16 mirror rows.
pub(crate) fn attend_dec_src() -> String {
    half_src(
        cut(
            super::WGSL,
            "struct AtP { nh: u32, hpk: u32, hd: u32, cap: u32, n: u32, scale: f32, _b: u32, _c: u32 };",
            "// `gqa_attend_dec` with RoPE and the KV append folded in",
        ),
        &["at_k", "at_v"],
    )
}

pub(crate) struct Pipes {
    pub(crate) kv_append: wgpu::ComputePipeline,
    pub(crate) attend_dec: wgpu::ComputePipeline,
}

fn compile(c: &Ctx, label: &str, src: &str, entry: &str) -> Option<wgpu::ComputePipeline> {
    let scope = c.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = c.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let p = c.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: None,
        module: &module,
        entry_point: Some(entry),
        compilation_options: crate::gpu_wgpu::pco(),
        cache: c.pipeline_cache.as_ref(),
    });
    if let Some(e) = pollster::block_on(scope.pop()) {
        tracing::warn!("{label} rejected ({e}): K/V mirrors stay f32");
        return None;
    }
    Some(p)
}

/// The token graph's f16 twins; None (f16 mirrors are then never made)
/// when the device rejects one.
pub(crate) fn pipes(c: &Ctx) -> Option<&'static Pipes> {
    static P: std::sync::OnceLock<Option<Pipes>> = std::sync::OnceLock::new();
    P.get_or_init(|| {
        Some(Pipes {
            kv_append: compile(c, "cmf-kv16-append", &kv_append_src(), "kv_append")?,
            attend_dec: compile(c, "cmf-kv16-dec", &attend_dec_src(), "gqa_attend_dec")?,
        })
    })
    .as_ref()
}

/// f32 rows → the bytes of an f16 mirror.
pub(crate) fn to_half_bytes(rows: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(rows.len() * 2);
    for &v in rows {
        b.extend_from_slice(&cortiq_core::quant::f32_to_f16(v).to_le_bytes());
    }
    b
}

/// Bytes read back from a mirror → f32 values.
pub(crate) fn from_mirror_bytes(bytes: &[u8], half: bool) -> Vec<f32> {
    if half {
        bytes
            .chunks_exact(2)
            .map(|h| cortiq_core::quant::f16_to_f32(u16::from_le_bytes([h[0], h[1]])))
            .collect()
    } else {
        bytemuck::cast_slice::<u8, f32>(bytes).to_vec()
    }
}

/// Write host rows (f32) into a mirror at element offset `elem` (the
/// mirror's own element type).
pub(crate) fn write_rows(c: &Ctx, buf: &wgpu::Buffer, elem: usize, rows: &[f32], half: bool) {
    if half {
        c.queue.write_buffer(buf, (elem * 2) as u64, &to_half_bytes(rows));
    } else {
        c.queue.write_buffer(buf, (elem * 4) as u64, bytemuck::cast_slice(rows));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(src: &str) {
        let module = wgpu::naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{e:?}"));
    }

    /// The prefill GEMM with the q4tp loader and the device-scale store
    /// parses and validates (naga, all capabilities), with no GPU.
    #[test]
    fn zi_mm_q4_validates() {
        use super::super::zimage::{default_cfg, mm_src, Epi, MmCfg};
        let g = MmCfg { q4: true, dscale: true, ..default_cfg(Epi::F32) };
        assert!(g.valid());
        let src = mm_src(g);
        assert!(src.contains("fn qcode"));
        let module = wgpu::naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{e}"));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{e:?}"));
    }

    #[test]
    fn half_src_rewrites_reads_and_stores() {
        let src = "@group(0) @binding(0) var<storage, read_write> kb : array<f32>;\n\
                   @group(0) @binding(1) var<storage, read>       kv : array<vec4<f32>>;\n\
                   @group(0) @binding(2) var<storage, read_write> out : array<f32>;\n\
                   @compute @workgroup_size(1)\n\
                   fn main() {\n    kb[1u] = out[0u] * 2.0;\n    out[1u] = kv[kb[0u] == 0.0 && true][2u] + out[2u];\n    kb[0u] = 0.0;\n}\n";
        let h = half_src(src, &["kb", "kv"]);
        assert!(h.contains("kb : array<f16>"), "{h}");
        assert!(h.contains("kv : array<vec4<f16>>"), "{h}");
        assert!(h.contains("kb[1u] = f16(out[0u] * 2.0);"), "{h}");
        assert!(h.contains("vec4<f32>(kv[f32(kb[0u]) == 0.0 && true])[2u]"), "{h}");
        assert!(h.contains("kb[0u] = f16(0.0);"), "{h}");
        assert!(!h.contains("out : array<f16>"), "{h}");
    }

    /// The f16 twins cut from the main module parse and validate (naga,
    /// all capabilities), with no GPU.
    #[test]
    fn twins_validate() {
        validate(&kv_append_src());
        validate(&attend_dec_src());
        assert!(attend_dec_src().contains("at_k : array<vec4<f16>>"));
        validate(&super::super::attend_fd::src_for_test(4, 256, true));
        validate(&half_src(super::super::dense_batch::ATTN_BT_WGSL, super::super::dense_batch::HALF_NAMES));
    }
}
