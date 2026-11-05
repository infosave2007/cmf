//! A prompt chunk's FFNs on the Neural Engine, through Apple's private client, the service behind
//! Core ML: a MIL program over IOSurfaces the GPU shares, after Splash (Inco AI, Apache-2.0; see
//! THIRD_PARTY_NOTICES.md). The interface may change with any macOS, so everything here falls back
//! to the GPU. `CMF_ANE=1` asks for it.
//!
//! q4tp weights are four bits under a scale per tile, so eight bits under a scale per row hold them
//! nearly whole: no rotation is needed to spread outliers, and the activations go as fp16. The CPU
//! decodes them, a NEON table lookup per 16 weights, while the ANE runs the layer before and the
//! GPU its mixer: three engines at once, with the file out of the GPU's working set for the FFNs.

use super::*;
use objc::rc::autoreleasepool;
use objc::runtime::{BOOL, Class, NO, Object};
use objc::{class, msg_send, sel, sel_impl};
use std::ffi::{CStr, CString, c_void};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

type Id = *mut Object;

#[link(name = "IOSurface", kind = "framework")]
unsafe extern "C" {
    fn IOSurfaceCreate(properties: *const c_void) -> *mut c_void;
    fn IOSurfaceGetBaseAddress(surface: *mut c_void) -> *mut c_void;
    fn IOSurfaceGetBytesPerRow(surface: *mut c_void) -> usize;
    fn IOSurfaceGetAllocSize(surface: *mut c_void) -> usize;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFRelease(cf: *mut c_void);
}

unsafe extern "C" {
    fn dlopen(path: *const std::ffi::c_char, mode: i32) -> *mut c_void;
}

const QOS_DEFAULT: u32 = 0x15;

fn ns_string(s: &str) -> Id {
    let c = CString::new(s).unwrap();
    unsafe { msg_send![class!(NSString), stringWithUTF8String: c.as_ptr()] }
}

fn rust_string(s: Id) -> String {
    if s.is_null() {
        return String::new();
    }
    unsafe {
        let p: *const std::ffi::c_char = msg_send![s, UTF8String];
        if p.is_null() { String::new() } else { CStr::from_ptr(p).to_string_lossy().into_owned() }
    }
}

fn ns_u32(v: u32) -> Id {
    unsafe { msg_send![class!(NSNumber), numberWithUnsignedInt: v] }
}

fn ns_u64(v: u64) -> Id {
    unsafe { msg_send![class!(NSNumber), numberWithUnsignedLongLong: v] }
}

fn ns_array(items: &[Id]) -> Id {
    unsafe { msg_send![class!(NSArray), arrayWithObjects: items.as_ptr() count: items.len()] }
}

fn ns_dict(pairs: &[(Id, Id)]) -> Id {
    let keys: Vec<Id> = pairs.iter().map(|p| p.0).collect();
    let vals: Vec<Id> = pairs.iter().map(|p| p.1).collect();
    unsafe { msg_send![class!(NSDictionary), dictionaryWithObjects: vals.as_ptr() forKeys: keys.as_ptr() count: pairs.len()] }
}

fn describe(error: Id) -> String {
    if error.is_null() {
        return "no reason".into();
    }
    let d: Id = unsafe { msg_send![error, description] };
    rust_string(d)
}

/// The classes of the private interface, once.
fn api() -> Result<(), String> {
    static DONE: std::sync::OnceLock<Result<(), String>> = std::sync::OnceLock::new();
    DONE.get_or_init(|| {
        let p = CString::new("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine").unwrap();
        if unsafe { dlopen(p.as_ptr(), 2) }.is_null() {
            return Err("AppleNeuralEngine does not load".into());
        }
        for c in ["_ANEModel", "_ANEClient", "_ANEIOSurfaceObject", "_ANERequest"] {
            if Class::get(c).is_none() {
                return Err(format!("AppleNeuralEngine lacks {c}"));
            }
        }
        Ok(())
    })
    .clone()
}

/// Memory the GPU and the Neural Engine share: `rows` rows of `width` fp16 or int8 values, each
/// row padded to 64 bytes.
pub(crate) struct Surface {
    pub(crate) raw: *mut c_void,
    pub(crate) stride: usize,
    pub(crate) bytes: usize,
    pub(crate) int8: bool,
}

unsafe impl Send for Surface {}
unsafe impl Sync for Surface {}

impl Surface {
    pub(crate) fn new(rows: usize, width: usize, int8: bool) -> Result<Surface, String> {
        let elem = if int8 { 1usize } else { 2 };
        let stride = (width * elem).div_ceil(64) * 64;
        let bytes = (stride * rows).div_ceil(16384) * 16384;
        let format: u32 = if int8 { 0x4c303038 } else { 0x4c303068 }; // 'L008', 'L00h'
        let raw = autoreleasepool(|| {
            let props = ns_dict(&[
                (ns_string("IOSurfaceWidth"), ns_u64(width as u64)),
                (ns_string("IOSurfaceHeight"), ns_u64(rows as u64)),
                (ns_string("IOSurfaceBytesPerElement"), ns_u64(elem as u64)),
                (ns_string("IOSurfaceBytesPerRow"), ns_u64(stride as u64)),
                (ns_string("IOSurfaceAllocSize"), ns_u64(bytes as u64)),
                (ns_string("IOSurfacePixelFormat"), ns_u32(format)),
            ]);
            unsafe { IOSurfaceCreate(props as *const c_void) }
        });
        if raw.is_null() {
            return Err("IOSurface creation failed".into());
        }
        let (got_stride, got_size) = unsafe { (IOSurfaceGetBytesPerRow(raw), IOSurfaceGetAllocSize(raw)) };
        if got_stride != stride || got_size != bytes {
            unsafe { CFRelease(raw) };
            return Err(format!("IOSurface took stride {got_stride} size {got_size}, not {stride} {bytes}"));
        }
        Ok(Surface { raw, stride, bytes, int8 })
    }

    pub(crate) fn base(&self) -> *mut u8 {
        unsafe { IOSurfaceGetBaseAddress(self.raw) as *mut u8 }
    }

    fn elem(&self) -> usize {
        if self.int8 { 1 } else { 2 }
    }

    /// The MIL type of its leading `rows` × `width` values, as a program reads or writes them.
    pub(crate) fn buffer_type(&self, rows: usize, width: usize) -> String {
        format!(
            "tensor_buffer<{}, shape=[1, 1, {rows}, {width}], strides={}, interleave_factors=[1, 1, 1, 1]>",
            if self.int8 { "int8" } else { "fp16" },
            self.strides(rows)
        )
    }

    pub(crate) fn strides(&self, rows: usize) -> String {
        let stride = self.stride / self.elem();
        let plane = rows * stride;
        format!("[{plane}, {plane}, {stride}, 1]")
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe { CFRelease(self.raw) };
    }
}

fn fnv(data: &[u8], mut h: u64) -> u64 {
    for b in data {
        h = (h ^ *b as u64).wrapping_mul(0x100000001b3);
    }
    h
}

fn cache_dir() -> PathBuf {
    std::env::temp_dir().join("cortiq-ane")
}

/// A MIL program compiled and loaded on the Neural Engine.
pub(crate) struct Program {
    client: Id,
    model: Id,
    /// By function: name, input symbol names and indices, output symbol.
    procedures: Vec<(String, Vec<(String, u32)>, u32)>,
}

unsafe impl Send for Program {}
unsafe impl Sync for Program {}

impl Program {
    pub(crate) fn new(mil: &str, weights: &[u8]) -> Result<Program, String> {
        api()?;
        let key = format!("{:016x}", fnv(weights, fnv(mil.as_bytes(), 0xcbf29ce484222325)));
        let dir: PathBuf = cache_dir().join(&key);
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        std::fs::write(dir.join("model.mil"), mil).map_err(|e| e.to_string())?;
        std::fs::write(dir.join("weights.bin"), weights).map_err(|e| e.to_string())?;
        Self::load_dir(&dir, &key)
    }

    fn load_dir(dir: &Path, key: &str) -> Result<Program, String> {
        autoreleasepool(|| unsafe {
            let client: Id = msg_send![Class::get("_ANEClient").unwrap(), sharedConnection];
            if client.is_null() {
                return Err("no shared connection".to_string());
            }
            let url: Id = msg_send![class!(NSURL), fileURLWithPath: ns_string(dir.to_str().unwrap()) isDirectory: true];
            let model: Id = msg_send![Class::get("_ANEModel").unwrap(), modelAtURL: url key: ns_string(key)];
            if model.is_null() {
                return Err("model creation failed".to_string());
            }
            let model: Id = msg_send![model, retain];
            let client: Id = msg_send![client, retain];
            let exists: BOOL = msg_send![client, compiledModelExistsFor: model];
            let empty = ns_dict(&[]);
            let mut error: Id = std::ptr::null_mut();
            if exists == NO {
                let options = ns_dict(&[
                    (ns_string("kANEFModelType"), ns_string("kANEFModelMIL")),
                    (ns_string("kANEFNetPlistFilenameKey"), ns_string("model.mil")),
                ]);
                let ok: BOOL = msg_send![client, compileModel: model options: options qos: QOS_DEFAULT error: &mut error];
                if ok == NO {
                    return Err(format!("compile: {}", describe(error)));
                }
            }
            let ok: BOOL = msg_send![client, loadModel: model options: empty qos: QOS_DEFAULT error: &mut error];
            if ok == NO {
                return Err(format!("load: {}", describe(error)));
            }
            let attrs: Id = msg_send![model, modelAttributes];
            let desc: Id = msg_send![attrs, objectForKey: ns_string("ANEFModelDescription")];
            if desc.is_null() {
                return Err("the program does not describe its procedures".to_string());
            }
            let symbols: Id = msg_send![desc, objectForKey: ns_string("kANEFModelInputSymbolsArrayKey")];
            let names: Id = msg_send![desc, objectForKey: ns_string("kANEFModelProcedureNameToIDMapKey")];
            let entries: Id = msg_send![desc, objectForKey: ns_string("ANEFModelProcedures")];
            let n: usize = msg_send![entries, count];
            let mut procedures = vec![(String::new(), Vec::new(), 0u32); n];
            let keys: Id = msg_send![names, allKeys];
            let nk: usize = msg_send![keys, count];
            for i in 0..nk {
                let k: Id = msg_send![keys, objectAtIndex: i];
                let v: Id = msg_send![names, objectForKey: k];
                let idx: usize = msg_send![v, unsignedIntegerValue];
                procedures[idx].0 = rust_string(k);
            }
            for i in 0..n {
                let e: Id = msg_send![entries, objectAtIndex: i];
                let id_num: Id = msg_send![e, objectForKey: ns_string("ANEFModelProcedureID")];
                let idx: usize = msg_send![id_num, unsignedIntegerValue];
                let outs: Id = msg_send![e, objectForKey: ns_string("ANEFModelOutputSymbolIndexArray")];
                let o0: Id = msg_send![outs, objectAtIndex: 0usize];
                procedures[idx].2 = msg_send![o0, unsignedIntValue];
                let ins: Id = msg_send![e, objectForKey: ns_string("ANEFModelInputSymbolIndexArray")];
                let ni: usize = msg_send![ins, count];
                for j in 0..ni {
                    let s: Id = msg_send![ins, objectAtIndex: j];
                    let sym: u32 = msg_send![s, unsignedIntValue];
                    let name: Id = msg_send![symbols, objectAtIndex: sym as usize];
                    procedures[idx].1.push((rust_string(name), sym));
                }
            }
            Ok(Program { client, model, procedures })
        })
    }

    /// One evaluation of `function`, waiting for it. `inputs` by the names of the program.
    pub(crate) fn eval(&self, function: &str, inputs: &[(&str, &Surface)], output: &Surface) -> Result<(), String> {
        let (index, proc_) = self
            .procedures
            .iter()
            .enumerate()
            .find(|(_, p)| p.0 == function)
            .ok_or_else(|| format!("no function {function}"))?;
        autoreleasepool(|| unsafe {
            let obj = |s: &Surface| -> Id { msg_send![Class::get("_ANEIOSurfaceObject").unwrap(), objectWithIOSurface: s.raw] };
            let mut objs = Vec::new();
            let mut idxs = Vec::new();
            for (name, sym) in &proc_.1 {
                let s = inputs
                    .iter()
                    .find(|(n, _)| *n == name.as_str())
                    .ok_or_else(|| format!("no surface for the input {name}"))?;
                objs.push(obj(s.1));
                idxs.push(ns_u32(*sym));
            }
            let nil: Id = std::ptr::null_mut();
            let request: Id = msg_send![Class::get("_ANERequest").unwrap(),
                requestWithInputs: ns_array(&objs)
                inputIndices: ns_array(&idxs)
                outputs: ns_array(&[obj(output)])
                outputIndices: ns_array(&[ns_u32(proc_.2)])
                weightsBuffer: nil
                perfStats: nil
                procedureIndex: ns_u32(index as u32)
                sharedEvents: nil
                transactionHandle: nil];
            if request.is_null() {
                return Err("request creation failed".to_string());
            }
            let mut error: Id = std::ptr::null_mut();
            let empty = ns_dict(&[]);
            let ok: BOOL = msg_send![self.client, evaluateWithModel: self.model options: empty request: request qos: QOS_DEFAULT error: &mut error];
            if ok == NO {
                return Err(format!("evaluation: {}", describe(error)));
            }
            Ok(())
        })
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        autoreleasepool(|| unsafe {
            let mut error: Id = std::ptr::null_mut();
            let empty = ns_dict(&[]);
            let _: BOOL = msg_send![self.client, unloadModel: self.model options: empty qos: QOS_DEFAULT error: &mut error];
            let _: () = msg_send![self.model, release];
            let _: () = msg_send![self.client, release];
        });
    }
}

/// A program's blob: it names no constant, and the service wants the file.
fn empty_blob() -> Vec<u8> {
    let mut blob = vec![0u8; 132];
    blob[0..4].copy_from_slice(&1u32.to_le_bytes());
    blob[4..8].copy_from_slice(&2u32.to_le_bytes());
    blob[64..68].copy_from_slice(&0xdeadbeefu32.to_le_bytes());
    blob[68..72].copy_from_slice(&1u32.to_le_bytes());
    blob[72..80].copy_from_slice(&4u64.to_le_bytes());
    blob[80..88].copy_from_slice(&128u64.to_le_bytes());
    blob
}

fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let m = b & 0x7fffff;
    if e <= 0 {
        if e < -10 {
            return s;
        }
        let m = (m | 0x800000) >> (1 - e);
        return s | ((m + 0x1000) >> 13) as u16;
    }
    if e >= 31 {
        return s | 0x7c00;
    }
    s | (((e as u16) << 10) + ((m + 0x1000) >> 13) as u16)
}

/// One segment of a matmul: its inputs a program takes in 2560 channels each.
pub(crate) const SEGMENT: usize = 2560;

/// The surfaces a function's text takes its types from.
pub(crate) struct FfnTypes<'a> {
    pub(crate) x: Vec<&'a Surface>,
    pub(crate) wg: Vec<&'a Surface>,
    pub(crate) wu: Vec<&'a Surface>,
    pub(crate) wd: Vec<&'a Surface>,
    pub(crate) sg: &'a Surface,
    pub(crate) su: &'a Surface,
    pub(crate) sd: &'a Surface,
    pub(crate) out: &'a Surface,
}

fn tensor(ty: &str, rows: usize, width: usize) -> String {
    format!("tensor<{ty}, [1, 1, {rows}, {width}]>")
}

/// An fp16 constant as a hex float, written exactly: MIL rounds it to the nearest fp16.
fn fp16(v: f64) -> String {
    let b = v.to_bits();
    let e = ((b >> 52) & 0x7ff) as i64 - 1023;
    let m = b & ((1u64 << 52) - 1);
    let sign = if v < 0.0 { "-" } else { "" };
    format!("fp16({sign}0x1.{m:013x}p{e:+})")
}

/// The MIL of one function of `rows` rows: int8 weights, fp16 activations. A token's intermediate
/// row still comes down to one peak, against fp16 overflow in the down product.
pub(crate) fn ffn_function(hidden: usize, channels: usize, rows: usize, s: &FfnTypes<'_>) -> String {
    let unit = fp16(1.0 / 128.0);
    let segs = hidden / SEGMENT;
    let mut params: Vec<String> = Vec::new();
    let mut body = String::new();
    let mut line = |t: String| {
        body.push_str("        ");
        body.push_str(&t);
        body.push_str(";\n");
    };
    let mut input = |name: String, surf: &Surface, r: usize, w: usize, line: &mut dyn FnMut(String)| {
        params.push(format!("{} {name}", surf.buffer_type(r, w)));
        let ty = if surf.int8 { "int8" } else { "fp16" };
        line(format!("{} {name}_t = tensor_buffer_to_tensor<ios17>(input = {name})", tensor(ty, r, w)));
    };
    for k in 0..segs {
        input(format!("x{k}"), s.x[k], SEGMENT, rows, &mut line);
    }
    for k in 0..segs {
        input(format!("wg{k}"), s.wg[k], channels, SEGMENT, &mut line);
        input(format!("wu{k}"), s.wu[k], channels, SEGMENT, &mut line);
    }
    input("sg".into(), s.sg, channels, 1, &mut line);
    input("su".into(), s.su, channels, 1, &mut line);
    let down: Vec<usize> = (0..s.wd.len()).map(|i| SEGMENT.min(channels - i * SEGMENT)).collect();
    for (i, &w) in down.iter().enumerate() {
        input(format!("wd{i}"), s.wd[i], hidden, w, &mut line);
    }
    input("sd".into(), s.sd, hidden, 1, &mut line);
    let f16 = |name: &str, h: usize, w: usize, expr: String| format!("{} {name} = {expr}", tensor("fp16", h, w));
    let matmul = |name: String, weights: String, values: String, h: usize, w: usize, line: &mut dyn FnMut(String)| {
        line(f16(&format!("{weights}_d"), h, w, format!("dequantize(input = {weights}_t, scale = {unit})")));
        line(format!(
            "{} {name} = matmul(transpose_x = bool(false), transpose_y = bool(false), x = {weights}_d, y = {values})",
            tensor("fp16", h, rows)
        ));
    };
    let mut line2 = |t: String| line(t);
    let sum = |prefix: &str, terms: usize, h: usize, line: &mut dyn FnMut(String)| -> String {
        let mut total = format!("{prefix}0");
        for t in 1..terms {
            let next = format!("{prefix}_sum{t}");
            line(f16(&next, h, rows, format!("add(x = {total}, y = {prefix}{t})")));
            total = next;
        }
        total
    };
    for p in ["g", "u"] {
        for k in 0..segs {
            matmul(format!("{p}m{k}"), format!("w{p}{k}"), format!("x{k}_t"), channels, SEGMENT, &mut line2);
        }
        let total = sum(&format!("{p}m"), segs, channels, &mut line2);
        line2(f16(&format!("{p}s"), channels, rows, format!("mul(x = {total}, y = s{p}_t)")));
    }
    let (c, r) = (channels, rows);
    line2(f16("gh", c, r, format!("mul(x = gs, y = {})", fp16(0.5))));
    line2(f16("th", c, r, "tanh(x = gh)".into()));
    line2(f16("tp", c, r, format!("add(x = th, y = {})", fp16(1.0))));
    line2(f16("silu", c, r, "mul(x = gh, y = tp)".into()));
    line2(f16("h", c, r, "mul(x = silu, y = us)".into()));
    line2(f16("habs", c, r, "abs(x = h)".into()));
    line2(f16("peak", 1, r, "reduce_max(x = habs, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))".into()));
    line2(f16("floor", 1, r, format!("maximum(x = peak, y = {})", fp16(1.0 / 512.0))));
    line2(f16("inverse", 1, r, format!("real_div(x = {}, y = floor)", fp16(1.0))));
    line2(f16("hd", c, r, "mul(x = h, y = inverse)".into()));
    let mut begin = 0usize;
    for (i, &w) in down.iter().enumerate() {
        line2(f16(
            &format!("hd{i}"),
            w,
            r,
            format!("slice_by_size(x = hd, begin = tensor<int32, [4]>([0, 0, {begin}, 0]), size = tensor<int32, [4]>([1, 1, {w}, {r}]))"),
        ));
        matmul(format!("dm{i}"), format!("wd{i}"), format!("hd{i}"), hidden, w, &mut line2);
        begin += w;
    }
    let total = sum("dm", down.len(), hidden, &mut line2);
    line2(f16("ds", hidden, r, format!("mul(x = {total}, y = sd_t)")));
    line2(f16("yt", hidden + 1, r, "concat(axis = int32(2), interleave = bool(false), values = (ds, floor))".into()));
    line2(format!(
        "{} y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>({}))",
        s.out.buffer_type(hidden + 1, r),
        s.out.strides(hidden + 1)
    ));
    format!("    func ffn{rows}<ios18>({}) {{\n{body}    }} -> (y);\n", params.join(", "))
}

const MSL: &str = include_str!("ane_msl.metal");

/// The rows of the program's functions. A chunk runs on the smallest that holds it.
const FUNCTIONS: [usize; 3] = [128, 256, 512];

struct Shared {
    s: Surface,
    b: Buffer,
}

fn shared_surface(c: &Ctx, rows: usize, width: usize, int8: bool) -> Result<Shared, String> {
    let s = Surface::new(rows, width, int8)?;
    let b = c._device.new_buffer_with_bytes_no_copy(
        s.base() as *const c_void,
        s.bytes as u64,
        MTLResourceOptions::StorageModeShared,
        None,
    );
    Ok(Shared { s, b })
}

struct WeightSet {
    wg: Vec<Shared>,
    wu: Vec<Shared>,
    wd: Vec<Shared>,
    sg: Shared,
    su: Shared,
    sd: Shared,
}

/// One projection in the model's file: its absolute offset, rows and columns.
#[derive(Clone, Copy)]
pub(crate) struct Proj {
    pub(crate) abs: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

pub(crate) struct AneFfn {
    prog: Program,
    pack: ComputePipelineState,
    join: ComputePipelineState,
    hidden: usize,
    inter: usize,
    sets: [WeightSet; 2],
    x: Vec<Shared>,
    tx: Shared,
    out: Shared,
    status: Buffer,
    layers: Vec<[Proj; 3]>,
    /// What each set holds, once a staging is done.
    staged: Mutex<[Option<usize>; 2]>,
    /// A staging on its way: the thread, the set it writes and the layer.
    staging: Mutex<Option<(std::thread::JoinHandle<()>, usize, usize)>>,
    /// Milliseconds: waiting for weights, the ANE, the GPU's part, and the evaluations.
    pub(crate) ms: Mutex<(f64, f64, f64, u64)>,
}

unsafe impl Send for AneFfn {}
unsafe impl Sync for AneFfn {}

fn seg_widths(channels: usize) -> Vec<usize> {
    (0..channels.div_ceil(SEGMENT)).map(|i| SEGMENT.min(channels - i * SEGMENT)).collect()
}

fn weight_set(c: &Ctx, hidden: usize, inter: usize) -> Result<WeightSet, String> {
    let mk = |r, w, i| shared_surface(c, r, w, i);
    let (mut wg, mut wu, mut wd) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..hidden / SEGMENT {
        wg.push(mk(inter, SEGMENT, true)?);
        wu.push(mk(inter, SEGMENT, true)?);
    }
    for w in seg_widths(inter) {
        wd.push(mk(hidden, w, true)?);
    }
    Ok(WeightSet { wg, wu, wd, sg: mk(inter, 1, false)?, su: mk(inter, 1, false)?, sd: mk(hidden, 1, false)? })
}

impl AneFfn {
    /// `layers`: gate, up and down of each FFN, q4tp, of one shape.
    pub(crate) fn new(c: &Ctx, layers: Vec<[Proj; 3]>) -> Result<AneFfn, String> {
        api()?;
        let first = layers.first().ok_or("no layers")?;
        let (hidden, inter) = (first[0].cols, first[0].rows);
        if hidden % SEGMENT != 0 || hidden / SEGMENT > 4 {
            return Err(format!("hidden {hidden} is not whole segments"));
        }
        if inter % 32 != 0 {
            return Err(format!("intermediate {inter} is not whole tiles"));
        }
        if layers.iter().any(|l| {
            l[0].cols != hidden || l[0].rows != inter || l[1].cols != hidden || l[1].rows != inter || l[2].rows != hidden || l[2].cols != inter
        }) {
            return Err("layers differ in shape".into());
        }
        let opts = metal::CompileOptions::new();
        opts.set_language_version(metal::MTLLanguageVersion::V3_0);
        let lib = c._device.new_library_with_source(MSL, &opts).map_err(|e| format!("ane MSL: {e}"))?;
        let pso = |name: &str| -> Result<ComputePipelineState, String> {
            let f = lib.get_function(name, None).map_err(|e| format!("kernel {name}: {e}"))?;
            c._device.new_compute_pipeline_state_with_function(&f).map_err(|e| format!("pipeline {name}: {e}"))
        };
        let (pack, join) = (pso("ane_pack16")?, pso("ane_join")?);
        let sets = [weight_set(c, hidden, inter)?, weight_set(c, hidden, inter)?];
        let max_rows = *FUNCTIONS.last().unwrap();
        let x: Result<Vec<Shared>, String> =
            (0..hidden / SEGMENT).map(|_| shared_surface(c, SEGMENT, max_rows, false)).collect();
        let x = x?;
        let tx = shared_surface(c, 1, max_rows, false)?;
        // The join's input scale: there is nothing to bring back.
        let p = tx.s.base() as *mut u16;
        for i in 0..tx.s.bytes / 2 {
            // SAFETY: inside the surface, which no command uses yet.
            unsafe { *p.add(i) = f32_to_f16(1.0) };
        }
        let out = shared_surface(c, hidden + 1, max_rows, false)?;
        let status = c._device.new_buffer(16, MTLResourceOptions::StorageModeShared);
        let types = FfnTypes {
            x: x.iter().map(|a| &a.s).collect(),
            wg: sets[0].wg.iter().map(|a| &a.s).collect(),
            wu: sets[0].wu.iter().map(|a| &a.s).collect(),
            wd: sets[0].wd.iter().map(|a| &a.s).collect(),
            sg: &sets[0].sg.s,
            su: &sets[0].su.s,
            sd: &sets[0].sd.s,
            out: &out.s,
        };
        let mut text = String::from("program(1.3)\n{\n");
        for rows in FUNCTIONS {
            text.push_str(&ffn_function(hidden, inter, rows, &types));
        }
        text.push_str("}\n");
        drop(types);
        let t0 = std::time::Instant::now();
        let prog = Program::new(&text, &empty_blob())?;
        tracing::info!(
            "neural engine: FFN of {inter} channels over {} layers, program {:.1} s",
            layers.len(),
            t0.elapsed().as_secs_f64()
        );
        Ok(AneFfn {
            prog,
            pack,
            join,
            hidden,
            inter,
            sets,
            x,
            tx,
            out,
            status,
            layers,
            staged: Mutex::new([None, None]),
            staging: Mutex::new(None),
            ms: Mutex::new((0.0, 0.0, 0.0, 0)),
        })
    }

    pub(crate) fn layer_of(&self, gate_abs: usize) -> Option<usize> {
        self.layers.iter().position(|l| l[0].abs == gate_abs)
    }

    pub(crate) fn takes(&self, rows: usize) -> bool {
        (64..=*FUNCTIONS.last().unwrap()).contains(&rows)
    }

    /// Layer `li`'s int8 weights into set `set` on the CPU.
    fn stage(&self, li: usize, set: usize, file: &[usize; 3]) {
        let t = &self.sets[set];
        for m in 0..3 {
            let pr = self.layers[li][m];
            let q = file[m];
            let (outs, scale): (&[Shared], &Shared) = match m {
                0 => (&t.wg, &t.sg),
                1 => (&t.wu, &t.su),
                _ => (&t.wd, &t.sd),
            };
            let outs: Vec<(usize, usize)> = outs.iter().map(|o| (o.s.base() as usize, o.s.stride)).collect();
            let (sbase, sstride) = (scale.s.base() as usize, scale.s.stride);
            let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(8);
            let per = pr.rows.div_ceil(threads);
            std::thread::scope(|sc| {
                for th in 0..threads {
                    let outs = &outs;
                    sc.spawn(move || {
                        let (r0, r1) = (th * per, ((th + 1) * per).min(pr.rows));
                        // SAFETY: rows of the mapped tensor and of this set's surfaces, which no
                        // engine reads until this staging is joined; threads own disjoint rows.
                        unsafe { stage_rows(q, pr.rows, pr.cols, r0, r1, outs, sbase, sstride) };
                    });
                }
            });
        }
    }

    /// The chunk's normalized rows as the program's fp16 input, transposed.
    pub(crate) fn encode_input(&self, enc: &metal::ComputeCommandEncoderRef, n_b: &Buffer, rows: usize) {
        let hidden = self.hidden as u32;
        for (k, x) in self.x.iter().enumerate() {
            enc.set_compute_pipeline_state(&self.pack);
            enc.set_buffer(0, Some(n_b), 0);
            enc.set_buffer(1, Some(&x.b), 0);
            let (ch, stride) = ((k * SEGMENT) as u32, (x.s.stride / 2) as u32);
            enc.set_bytes(2, 4, &hidden as *const u32 as *const c_void);
            enc.set_bytes(3, 4, &ch as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &stride as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new((rows as u64).div_ceil(32), (SEGMENT / 32) as u64, 1),
                MTLSize::new(32, 8, 1),
            );
        }
    }

    fn inputs(&self, set: usize) -> Vec<(String, &Surface)> {
        let t = &self.sets[set];
        let mut v: Vec<(String, &Surface)> = Vec::new();
        for (k, x) in self.x.iter().enumerate() {
            v.push((format!("x{k}"), &x.s));
            v.push((format!("wg{k}"), &t.wg[k].s));
            v.push((format!("wu{k}"), &t.wu[k].s));
        }
        for (i, w) in t.wd.iter().enumerate() {
            v.push((format!("wd{i}"), &w.s));
        }
        v.push(("sg".into(), &t.sg.s));
        v.push(("su".into(), &t.su.s));
        v.push(("sd".into(), &t.sd.s));
        v
    }

    fn file_ptrs(&self, fbuf: &WeightArena, li: usize) -> [usize; 3] {
        let mut p = [0usize; 3];
        for m in 0..3 {
            let (b, off) = fbuf.locate(self.layers[li][m].abs);
            p[m] = b.contents() as usize + off;
        }
        p
    }

    /// Layer `li`'s FFN over the input `encode_input` wrote, whose command is done. A thread
    /// stages the next layer's weights while the ANE runs this one and the GPU goes on to the
    /// next layer's mixer.
    pub(crate) fn run(self: &Arc<Self>, fbuf: &WeightArena, li: usize, rows: usize) -> Result<(), String> {
        let t0 = std::time::Instant::now();
        if let Some((h, set, l)) = self.staging.lock().unwrap().take() {
            h.join().map_err(|_| "weights staging panicked".to_string())?;
            self.staged.lock().unwrap()[set] = Some(l);
        }
        let found = self.staged.lock().unwrap().iter().position(|v| *v == Some(li));
        let set = match found {
            Some(s) => s,
            None => {
                self.stage(li, 0, &self.file_ptrs(fbuf, li));
                self.staged.lock().unwrap()[0] = Some(li);
                0
            }
        };
        let t1 = std::time::Instant::now();
        let next = (li + 1) % self.layers.len();
        let other = set ^ 1;
        if self.staged.lock().unwrap()[other] != Some(next) {
            self.staged.lock().unwrap()[other] = None;
            let file = self.file_ptrs(fbuf, next);
            let me = self.clone();
            let h = std::thread::spawn(move || me.stage(next, other, &file));
            *self.staging.lock().unwrap() = Some((h, other, next));
        }
        let func = FUNCTIONS.iter().copied().find(|&f| f >= rows).ok_or("too many rows")?;
        let names = self.inputs(set);
        let refs: Vec<(&str, &Surface)> = names.iter().map(|(n, s)| (n.as_str(), *s)).collect();
        self.prog.eval(&format!("ffn{func}"), &refs, &self.out.s)?;
        let mut m = self.ms.lock().unwrap();
        m.0 += (t1 - t0).as_secs_f64() * 1e3;
        m.1 += t1.elapsed().as_secs_f64() * 1e3;
        m.3 += 1;
        if m.3 % 64 == 0 && std::env::var("CMF_ANE_PROF").is_ok() {
            eprintln!(
                "ane-prof: {} evaluations, each: the GPU's part {:.1} ms, waited for weights {:.1} ms, the ANE {:.1} ms",
                m.3,
                m.2 / m.3 as f64,
                m.0 / m.3 as f64,
                m.1 / m.3 as f64
            );
        }
        Ok(())
    }

    /// The output of `run` as the residual's delta.
    pub(crate) fn encode_join(&self, enc: &metal::ComputeCommandEncoderRef, d_b: &Buffer, rows: usize) {
        let (hidden, stride) = (self.hidden as u32, (self.out.s.stride / 2) as u32);
        enc.set_compute_pipeline_state(&self.join);
        enc.set_buffer(0, Some(d_b), 0);
        enc.set_buffer(1, Some(&self.out.b), 0);
        enc.set_buffer(2, Some(&self.tx.b), 0);
        enc.set_buffer(3, Some(&self.status), 0);
        enc.set_bytes(4, 4, &hidden as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &stride as *const u32 as *const c_void);
        enc.dispatch_threads(MTLSize::new(self.hidden as u64, rows as u64, 1), MTLSize::new(32, 1, 1));
    }

    /// Whether a value the ANE gave was not finite, since the last call.
    pub(crate) fn not_finite(&self) -> bool {
        let p = self.status.contents() as *mut u32;
        // SAFETY: a shared buffer of 16 bytes, and no command writes it now.
        unsafe {
            let v = *p != 0;
            *p = 0;
            v
        }
    }
}

/// Rows [r0, r1) of a q4tp tensor as int8 under each row's scale. `outs`: per segment of 80 tiles,
/// its surface's base and stride. The row's scale goes to its fp16 surface.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
unsafe fn stage_rows(q: usize, rows: usize, cols: usize, r0: usize, r1: usize, outs: &[(usize, usize)], sbase: usize, sstride: usize) {
    use std::arch::aarch64::*;
    let q = q as *const u8;
    let gpr = cols / 32;
    let params = rows * gpr * 16;
    let codes = params + rows * 4;
    let stride = (gpr * 5).div_ceil(8);
    let per_seg = SEGMENT / 32;
    let mut code = vec![0u8; gpr];
    for r in r0..r1 {
        unsafe {
            let ph = q.add(params + r * 4);
            let lo = f16_to_f32(u16::from_le_bytes([*ph, *ph.add(1)]));
            let step = f16_to_f32(u16::from_le_bytes([*ph.add(2), *ph.add(3)]));
            let cp = q.add(codes + r * stride);
            let mut top = 0u8;
            for (g, c) in code.iter_mut().enumerate() {
                let bit = g * 5;
                let (at, shf) = (bit >> 3, bit & 7);
                let v = *cp.add(at) as u32 | if shf > 3 { (*cp.add(at + 1) as u32) << 8 } else { 0 };
                *c = ((v >> shf) & 31) as u8;
                top = top.max(*c);
            }
            let peak = (8.0 * (lo + top as f32 * step).exp2()).max(1.0 / 16384.0);
            let bits = f32_to_f16(peak / 127.0 * 128.0);
            *((sbase + r * sstride) as *mut u16) = bits;
            let inverse = 128.0 / f16_to_f32(bits);
            let mut luts = [[0i8; 16]; 32];
            let mut have = 0u32;
            let n_lo = vdupq_n_u8(15);
            for g in 0..gpr {
                let c = code[g] as usize;
                if have & (1 << c) == 0 {
                    have |= 1 << c;
                    let k = (lo + c as f32 * step).exp2() * inverse;
                    for (n, slot) in luts[c].iter_mut().enumerate() {
                        *slot = ((n as f32 - 8.0) * k).round().clamp(-127.0, 127.0) as i8;
                    }
                }
                let lut = vld1q_s8(luts[c].as_ptr());
                let tile = vld1q_u8(q.add((r * gpr + g) * 16));
                let (l, h) = (vandq_u8(tile, n_lo), vshrq_n_u8(tile, 4));
                let (seg, within) = (g / per_seg, g % per_seg);
                let (base, st) = outs[seg];
                let out = (base + r * st + within * 32) as *mut i8;
                vst1q_s8(out, vqtbl1q_s8(lut, vzip1q_u8(l, h)));
                vst1q_s8(out.add(16), vqtbl1q_s8(lut, vzip2q_u8(l, h)));
            }
        }
    }
}

/// Macs without NEON have no Neural Engine: `wanted` keeps them away from this.
#[cfg(not(target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
unsafe fn stage_rows(_q: usize, _rows: usize, _cols: usize, _r0: usize, _r1: usize, _outs: &[(usize, usize)], _sbase: usize, _sstride: usize) {}

fn f16_to_f32(bits: u16) -> f32 {
    let s = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = ((bits >> 10) & 0x1f) as i32;
    let m = (bits & 0x3ff) as f32;
    if e == 0 {
        s * m * 2f32.powi(-24)
    } else {
        s * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
    }
}


static ANE: Mutex<Option<Arc<AneFfn>>> = Mutex::new(None);
static REFUSED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether a prompt asks for the Neural Engine, which takes larger chunks.
pub(crate) fn wanted() -> bool {
    cfg!(target_arch = "aarch64")
        && std::env::var("CMF_ANE").as_deref() == Ok("1")
        && !REFUSED.load(std::sync::atomic::Ordering::Relaxed)
        && api().is_ok()
}

/// The FFN for these layers, built once; any refusal keeps the GPU's.
pub(crate) fn ffn_for(c: &Ctx, layers: Vec<[Proj; 3]>) -> Option<Arc<AneFfn>> {
    if !wanted() {
        return None;
    }
    let mut slot = ANE.lock().unwrap();
    if let Some(a) = slot.as_ref() {
        if a.layers.len() == layers.len() && a.layers.first().map(|l| l[0].abs) == layers.first().map(|l| l[0].abs) {
            return Some(a.clone());
        }
    }
    match AneFfn::new(c, layers) {
        Ok(a) => {
            let a = Arc::new(a);
            *slot = Some(a.clone());
            Some(a)
        }
        Err(e) => {
            tracing::warn!("neural engine off: {e}");
            REFUSED.store(true, std::sync::atomic::Ordering::Relaxed);
            None
        }
    }
}

/// The ANE stopped: the GPU's for the rest of the process.
pub(crate) fn retire(why: &str) {
    tracing::warn!("neural engine off: {why}");
    REFUSED.store(true, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One FFN of noise against the same arithmetic on the host, with the ANE's speed. It talks to
    /// a private service, so it runs by hand:
    /// `cargo test -p cortiq-engine --release --lib ane_ffn_matches_host -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn ane_ffn_matches_host() {
        let (hidden, channels, rows) = (5120usize, 2560usize, 128usize);
        let mk = |r, w, i| Surface::new(r, w, i).expect("surface");
        let x: Vec<Surface> = (0..2).map(|_| mk(SEGMENT, rows, false)).collect();
        let wg: Vec<Surface> = (0..2).map(|_| mk(channels, SEGMENT, true)).collect();
        let wu: Vec<Surface> = (0..2).map(|_| mk(channels, SEGMENT, true)).collect();
        let wd = vec![mk(hidden, channels, true)];
        let (sg, su, sd) = (mk(channels, 1, false), mk(channels, 1, false), mk(hidden, 1, false));
        let out = mk(hidden + 1, rows, false);
        let mut seed = 12345u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
            (seed >> 56) as i8
        };
        for g in wg.iter().chain(&wu).chain(&wd) {
            for i in 0..g.bytes {
                unsafe { *g.base().add(i) = (rnd() / 4) as u8 };
            }
        }
        for g in &x {
            let p = g.base() as *mut u16;
            for i in 0..g.bytes / 2 {
                unsafe { *p.add(i) = f32_to_f16(rnd() as f32 / 128.0) };
            }
        }
        for g in [&sg, &su, &sd] {
            let p = g.base() as *mut u16;
            for i in 0..g.bytes / 2 {
                unsafe { *p.add(i) = f32_to_f16(0.1) };
            }
        }
        let types = FfnTypes {
            x: x.iter().collect(),
            wg: wg.iter().collect(),
            wu: wu.iter().collect(),
            wd: wd.iter().collect(),
            sg: &sg,
            su: &su,
            sd: &sd,
            out: &out,
        };
        let mil = format!("program(1.3)\n{{\n{}}}\n", ffn_function(hidden, channels, rows, &types));
        let Ok(prog) = Program::new(&mil, &empty_blob()) else {
            eprintln!("no Neural Engine here");
            return;
        };
        let mut names: Vec<(String, &Surface)> = Vec::new();
        for k in 0..2 {
            names.push((format!("x{k}"), &x[k]));
            names.push((format!("wg{k}"), &wg[k]));
            names.push((format!("wu{k}"), &wu[k]));
        }
        names.push(("wd0".into(), &wd[0]));
        names.push(("sg".into(), &sg));
        names.push(("su".into(), &su));
        names.push(("sd".into(), &sd));
        let refs: Vec<(&str, &Surface)> = names.iter().map(|(n, s)| (n.as_str(), *s)).collect();
        let mut best = f64::MAX;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            prog.eval(&format!("ffn{rows}"), &refs, &out).expect("evaluation");
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        let w = |s: &Surface, r: usize, c: usize| unsafe { *s.base().add(r * s.stride + c) as i8 as f32 } / 128.0;
        let h16 = |s: &Surface, r: usize, c: usize| unsafe { f16_to_f32(*(s.base().add(r * s.stride) as *const u16).add(c)) };
        let (mut e, mut t) = (0f64, 0f64);
        for row in [0usize, 7, rows - 1] {
            let xv: Vec<f32> = (0..hidden).map(|c| h16(&x[c / SEGMENT], c % SEGMENT, row)).collect();
            let mv = |m: &[Surface], ch: usize| -> f32 {
                let mut acc = 0f32;
                for (c, v) in xv.iter().enumerate() {
                    acc += w(&m[c / SEGMENT], ch, c % SEGMENT) * v;
                }
                acc * 0.1
            };
            let h: Vec<f32> = (0..channels)
                .map(|ch| {
                    let g = mv(&wg, ch) * 0.5;
                    g * (g.tanh() + 1.0) * mv(&wu, ch)
                })
                .collect();
            for r in (0..hidden).step_by(97) {
                let want: f32 = h.iter().enumerate().map(|(c, v)| w(&wd[0], r, c) * v).sum::<f32>() * 0.1;
                let got = h16(&out, r, row) * h16(&out, hidden, row);
                e += ((want - got) as f64).powi(2);
                t += (want as f64).powi(2);
            }
        }
        let rel = (e / t.max(1e-30)).sqrt();
        eprintln!("hidden {hidden} channels {channels} rows {rows}: {best:.1} ms, relative rms {rel:.4}");
        assert!(rel < 0.15, "relative rms {rel}");
    }
}
