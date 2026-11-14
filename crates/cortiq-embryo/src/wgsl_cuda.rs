//! The trainer's WGSL (`vulkan.rs`'s `EMBRYO_WGSL`) as CUDA C, for the
//! `cuda` build: one source of the kernels for both backends. The WGSL is
//! written in a narrow dialect (u32 storage slots `a`..`m`, `nbuf`, `obuf`,
//! `pbuf` and the parameter block `par`, the one-line accessors `U`, `F`,
//! `AF`.., scalar code with a few vec4 tiles), and this pass rewrites that
//! dialect, not WGSL at large:
//!
//! - every storage slot becomes a kernel argument `S_<name>`, and every user
//!   function takes the slots first (`EWD` / `EWP`), so a helper reads the
//!   same buffers its entry point was launched with;
//! - the accessors become macros over those arguments;
//! - `let`/`var` become `const auto`/`auto` (a typed `var` keeps its type and
//!   WGSL's zero initialisation), `select(f, t, c)` a conditional, float
//!   literals f32, `loop` a `while(true)`, the barriers `__syncthreads`;
//! - an entry point (`@compute @workgroup_size(n) fn name(builtins)`) becomes
//!   `extern "C" __global__ void emb_<name>`, its builtins read from the
//!   launch (`global_invocation_id` as x = block·n + thread, y = block y).

use fancy_regex::{Captures, Regex};

/// The slots of the WGSL, in binding order; `par` last.
pub const SLOTS: [&str; 17] = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "nbuf", "obuf", "pbuf", "par",
];

const PRELUDE: &str = r#"
typedef unsigned int u32;
#define EWD u32* S_a, u32* S_b, u32* S_c, u32* S_d, u32* S_e, u32* S_f, u32* S_g, u32* S_h, u32* S_i, \
    u32* S_j, u32* S_k, u32* S_l, u32* S_m, u32* S_nbuf, u32* S_obuf, u32* S_pbuf, const u32* S_par
#define EWP S_a, S_b, S_c, S_d, S_e, S_f, S_g, S_h, S_i, S_j, S_k, S_l, S_m, S_nbuf, S_obuf, S_pbuf, S_par
__device__ __forceinline__ float as_f32(u32 x) { return __uint_as_float(x); }
__device__ __forceinline__ float as_f32(int x) { return __int_as_float(x); }
__device__ __forceinline__ float as_f32(float x) { return x; }
__device__ __forceinline__ u32 as_u32(float x) { return __float_as_uint(x); }
__device__ __forceinline__ u32 as_u32(u32 x) { return x; }
__device__ __forceinline__ u32 as_u32(int x) { return (u32)x; }
__device__ __forceinline__ int as_i32(u32 x) { return (int)x; }
__device__ __forceinline__ int as_i32(int x) { return x; }
__device__ __forceinline__ int as_i32(float x) { return __float_as_int(x); }
#define U(x) (S_par[(x)])
#define F(x) as_f32(S_par[(x)])
#define AF(x) as_f32(S_a[(x)])
#define BF(x) as_f32(S_b[(x)])
#define CF(x) as_f32(S_c[(x)])
#define DF(x) as_f32(S_d[(x)])
#define EF(x) as_f32(S_e[(x)])
#define FF(x) as_f32(S_f[(x)])
#define GF(x) as_f32(S_g[(x)])
#define HF(x) as_f32(S_h[(x)])
#define IF(x) as_f32(S_i[(x)])
#define JF(x) as_f32(S_j[(x)])
#define KF(x) as_f32(S_k[(x)])
#define MF(x) as_f32(S_m[(x)])
#define NF(x) as_f32(S_nbuf[(x)])
#define AU(x) (S_a[(x)])
#define BU(x) (S_b[(x)])
#define CU(x) (S_c[(x)])
#define DU(x) (S_d[(x)])
#define WF(x) as_u32((float)(x))
#define F4I(x) ((float*)&(x))
__device__ __forceinline__ float clamp(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
__device__ __forceinline__ float4 V4(float a) { return make_float4(a, a, a, a); }
__device__ __forceinline__ float4 V4(float a, float b, float c, float d) { return make_float4(a, b, c, d); }
__device__ __forceinline__ float4 operator+(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ __forceinline__ float4 operator-(float4 a, float4 b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
__device__ __forceinline__ float4 operator*(float4 a, float4 b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
__device__ __forceinline__ float4 operator*(float s, float4 b) { return make_float4(s * b.x, s * b.y, s * b.z, s * b.w); }
__device__ __forceinline__ float4 operator*(float4 a, float s) { return make_float4(a.x * s, a.y * s, a.z * s, a.w * s); }
__device__ __forceinline__ float4 fma(float4 a, float4 b, float4 c) {
    return make_float4(fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y), fmaf(a.z, b.z, c.z), fmaf(a.w, b.w, c.w));
}
"#;

fn re(p: &str) -> Regex {
    Regex::new(p).expect("translator pattern")
}

/// Split `s` (the text after an opening parenthesis) at its matching close:
/// (the top-level comma-separated arguments, the rest after the close).
fn args_of(s: &str) -> Option<(Vec<String>, &str)> {
    let mut depth = 0i32;
    let mut cur = String::new();
    let mut out = Vec::new();
    for (i, ch) in s.char_indices() {
        match ch {
            '(' | '[' => {
                depth += 1;
                cur.push(ch);
            }
            ')' | ']' if depth > 0 => {
                depth -= 1;
                cur.push(ch);
            }
            ')' => {
                out.push(cur.trim().to_string());
                return Some((out, &s[i + 1..]));
            }
            ',' if depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }
    None
}

/// A vec4's component by index: `v[j]` of a vec4 value `v` and `t[i][j]` of
/// a vec4 array `t` read and write through a float view (`F4I`).
fn vec4_index(mut s: String, values: &[String], arrays: &[String]) -> String {
    for n in values {
        s = re(&format!(r"\b{n}\[")).replace_all(&s, format!("F4I({n})[").as_str()).into_owned();
    }
    for n in arrays {
        let pat = format!("{n}[");
        let mut out = String::new();
        let mut rest = s.as_str();
        while let Some(at) = rest.find(&pat) {
            let ident = at == 0 || !rest[..at].chars().next_back().is_some_and(|c| c.is_alphanumeric() || c == '_');
            out.push_str(&rest[..at]);
            let body = &rest[at + pat.len()..];
            let mut depth = 1;
            let mut end = 0;
            for (i, ch) in body.char_indices() {
                match ch {
                    '[' => depth += 1,
                    ']' => {
                        depth -= 1;
                        if depth == 0 {
                            end = i;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let after = &body[end + 1..];
            if ident && after.starts_with('[') {
                out.push_str(&format!("F4I({n}[{}])", &body[..end]));
            } else {
                out.push_str(&format!("{n}[{}]", &body[..end]));
            }
            rest = after;
        }
        out.push_str(rest);
        s = out;
    }
    s
}

/// `select(f, t, c)` → `((c) ? (t) : (f))`, innermost first.
fn selects(mut s: String) -> String {
    while let Some(at) = s.rfind("select(") {
        let before_ok = at == 0 || !s[..at].chars().next_back().is_some_and(|c| c.is_alphanumeric() || c == '_');
        if !before_ok {
            // an identifier ending in `select`: not ours (none in the dialect)
            break;
        }
        let Some((args, rest)) = args_of(&s[at + 7..]) else { break };
        assert_eq!(args.len(), 3, "select takes three arguments");
        let rep = format!("(({})?({}):({}))", args[2], args[1], args[0]);
        s = format!("{}{}{}", &s[..at], rep, rest);
    }
    s
}

fn ty(t: &str) -> String {
    let t = t.trim();
    match t {
        "u32" => "u32".into(),
        "f32" => "float".into(),
        "i32" => "int".into(),
        "bool" => "bool".into(),
        "vec4<f32>" => "float4".into(),
        "vec3<u32>" => "uint3".into(),
        other => panic!("WGSL type {other} outside the dialect"),
    }
}

/// The CUDA C of `wgsl`; the entry points are `emb_<name>`.
pub fn translate(wgsl: &str) -> String {
    let mut s = wgsl.to_string();
    // bindings, the parameter struct and the accessors (macros in the prelude)
    s = re(r"(?m)^struct P \{[^}]*\};").replace_all(&s, "").into_owned();
    s = re(r"(?m)^@group\(0\) @binding\(\d+\) var<storage,[a-z_]+> \w+: array<u32>;\n?").replace_all(&s, "").into_owned();
    s = re(r"fn (?:U|F|[A-N]F|[A-D]U|WF)\(x:[uf]32\)->[uf]32\{return [^}]*\}").replace_all(&s, "").into_owned();
    // slots as kernel arguments
    s = re(r"\b(a|b|c|d|e|f|g|h|i|j|k|l|m|nbuf|obuf|pbuf|par)\[").replace_all(&s, "S_$1[").into_owned();
    s = re(r"var<workgroup> (\w+): array<([^,]+),(\d+)>;")
        .replace_all(&s, |c: &Captures| format!("__shared__ {} {}[{}];", ty(&c[2]), &c[1], &c[3]))
        .into_owned();
    // entry points
    let entry = re(r"@compute\s+@workgroup_size\((\d+)\)\s*fn (\w+)\(((?:[^()]|\([^()]*\))*)\)\s*\{");
    s = entry
        .replace_all(&s, |c: &Captures| {
            let n = &c[1];
            let mut head = format!("extern \"C\" __global__ void __launch_bounds__({n}) emb_{}(EWD) {{", &c[2]);
            for p in c[3].split(',').filter(|p| !p.trim().is_empty()) {
                let p = p.trim();
                let b = re(r"@builtin\((\w+)\)\s*(\w+)\s*:").captures(p).unwrap().expect("a builtin argument");
                let (kind, name) = (b[1].to_string(), b[2].to_string());
                head += &match kind.as_str() {
                    "local_invocation_id" => format!(" const uint3 {name} = threadIdx;"),
                    "workgroup_id" => format!(" const uint3 {name} = blockIdx;"),
                    "num_workgroups" => format!(" const uint3 {name} = gridDim;"),
                    "local_invocation_index" => format!(" const u32 {name} = threadIdx.x;"),
                    "global_invocation_id" => {
                        format!(" const uint3 {name} = make_uint3(blockIdx.x * {n}u + threadIdx.x, blockIdx.y, blockIdx.z);")
                    }
                    k => panic!("builtin {k} outside the dialect"),
                };
            }
            head
        })
        .into_owned();
    // user functions: the slots first; calls marked through `@@(` meanwhile
    let def = re(r"\bfn (\w+)\(([^)]*)\)\s*(?:->\s*([\w<>]+)\s*)?\{");
    let mut names = Vec::new();
    let mut vec4_values = Vec::new();
    s = def
        .replace_all(&s, |c: &Captures| {
            names.push(c[1].to_string());
            let mut params = vec!["EWD".to_string()];
            for p in c[2].split(',').filter(|p| !p.trim().is_empty()) {
                let (n, t) = p.split_once(':').expect("a typed parameter");
                if ty(t) == "float4" {
                    vec4_values.push(n.trim().to_string());
                }
                params.push(format!("{} {}", ty(t), n.trim()));
            }
            let ret = c.get(3).map(|r| ty(r.as_str())).unwrap_or_else(|| "void".into());
            format!("__device__ {ret} {}@@({}) {{", &c[1], params.join(", "))
        })
        .into_owned();
    for n in &names {
        s = re(&format!(r"\b{n}\((\s*\))?"))
            .replace_all(&s, |c: &Captures| if c.get(1).is_some() { format!("{n}(EWP)") } else { format!("{n}(EWP, ") })
            .into_owned();
    }
    s = s.replace("@@(", "(");
    // declarations
    s = re(r"\bvar (\w+)\s*:\s*array<([^,]+),(\d+)>;")
        .replace_all(&s, |c: &Captures| format!("{} {}[{}] = {{}};", ty(&c[2]), &c[1], &c[3]))
        .into_owned();
    s = re(r"\bvar (\w+)\s*:\s*([\w<>]+)\s*=")
        .replace_all(&s, |c: &Captures| format!("{} {} =", ty(&c[2]), &c[1]))
        .into_owned();
    s = re(r"\bvar (\w+)\s*:\s*([\w<>]+)\s*;")
        .replace_all(&s, |c: &Captures| format!("{} {}{{}};", ty(&c[2]), &c[1]))
        .into_owned();
    s = re(r"\blet\s+").replace_all(&s, "const auto ").into_owned();
    s = re(r"\bvar\s+").replace_all(&s, "auto ").into_owned();
    // expressions
    s = s.replace("bitcast<f32>(", "as_f32(").replace("bitcast<u32>(", "as_u32(").replace("bitcast<i32>(", "as_i32(");
    s = s.replace("vec4<f32>(", "V4(");
    s = re(r"\bf32\(").replace_all(&s, "(float)(").into_owned();
    s = re(r"\bu32\(").replace_all(&s, "(u32)(").into_owned();
    s = re(r"\bi32\(").replace_all(&s, "(int)(").into_owned();
    s = re(r"\bloop\s*\{").replace_all(&s, "while(true){").into_owned();
    s = s.replace("workgroupBarrier()", "__syncthreads()").replace("storageBarrier()", "__threadfence_block()");
    s = re(r"\binverseSqrt\(").replace_all(&s, "rsqrtf(").into_owned();
    s = re(r"(?<![\w.])(\d+\.\d*(?:e[-+]?\d+)?|\d+e[-+]?\d+)(?![\w.])").replace_all(&s, "${1}f").into_owned();
    s = selects(s);
    let vec4_arrays: Vec<String> = re(r"__shared__ float4 (\w+)\[")
        .captures_iter(&s)
        .filter_map(|c| c.ok().map(|c| c[1].to_string()))
        .collect();
    vec4_values.sort();
    vec4_values.dedup();
    s = vec4_index(s, &vec4_values, &vec4_arrays);
    format!("{PRELUDE}\n{s}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_nest_and_keep_commas() {
        let s = selects("x=select(min(a,b),select(1u,2u,q),c[0]==1u);".into());
        assert_eq!(s, "x=((c[0]==1u)?(((q)?(2u):(1u))):(min(a,b)));");
    }

    #[test]
    fn vec4_components_go_through_a_float_view() {
        let s = vec4_index("x=v[j]; t[i+1u][k%4u]=y; t[0]=w;".into(), &["v".into()], &["t".into()]);
        assert_eq!(s, "x=F4I(v)[j]; F4I(t[i+1u])[k%4u]=y; t[0]=w;");
    }

    #[test]
    fn a_dialect_sample_translates() {
        let w = "@group(0) @binding(0) var<storage,read_write> a: array<u32>;\n\
                 fn U(x:u32)->u32{return par[x];}\n\
                 var<workgroup> t: array<f32,256>;\n\
                 fn sq(x:f32)->f32 { return x*x+0.5; }\n\
                 @compute @workgroup_size(256)\n\
                 fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n\
                  let z=id.x; var s:array<f32,4>; var q=0.0; loop{ break; }\n\
                  a[z]=bitcast<u32>(select(1.0,sq(f32(z)),z<U(1u)));\n\
                 }\n";
        let c = translate(w);
        assert!(c.contains("extern \"C\" __global__ void __launch_bounds__(256) emb_main(EWD)"), "{c}");
        assert!(c.contains("__device__ float sq(EWD, float x)"), "{c}");
        assert!(c.contains("sq(EWP, (float)(z))"), "{c}");
        assert!(c.contains("float s[4] = {};"), "{c}");
        assert!(c.contains("auto q=0.0f;"), "{c}");
        assert!(c.contains("S_a[z]=as_u32(((z<U(1u))?(sq(EWP, (float)(z))):(1.0f)));"), "{c}");
        assert!(c.contains("__shared__ float t[256];"), "{c}");
        assert!(!c.contains("@group"), "{c}");
    }
}
