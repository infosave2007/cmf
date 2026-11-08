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
use std::sync::{Condvar, Mutex};

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

/// vImage's view of a plane: `height` rows of `width` values, `row_bytes` apart.
#[repr(C)]
struct Plane {
    data: *mut c_void,
    height: usize,
    width: usize,
    row_bytes: usize,
}

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn vImageConvert_Planar16FtoPlanarF(src: *const Plane, dest: *const Plane, flags: u32) -> isize;
    fn vImageConvert_PlanarFtoPlanar16F(src: *const Plane, dest: *const Plane, flags: u32) -> isize;
    fn vDSP_mtrans(a: *const f32, ia: isize, c: *mut f32, ic: isize, m: usize, n: usize);
    fn vDSP_svesq(a: *const f32, ia: isize, c: *mut f32, n: usize);
    fn vDSP_vsmul(a: *const f32, ia: isize, b: *const f32, c: *mut f32, ic: isize, n: usize);
    fn vDSP_vmul(a: *const f32, ia: isize, b: *const f32, ib: isize, c: *mut f32, ic: isize, n: usize);
    fn cblas_saxpy(n: i32, alpha: f32, x: *const f32, incx: i32, y: *mut f32, incy: i32);
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

/// The text of one MIL function, a line at a time.
struct Mil {
    params: Vec<String>,
    body: String,
}

impl Mil {
    fn new() -> Mil {
        Mil { params: Vec::new(), body: String::new() }
    }

    fn line(&mut self, t: String) {
        self.body.push_str("        ");
        self.body.push_str(&t);
        self.body.push_str(";\n");
    }

    fn f16(&mut self, name: &str, h: usize, w: usize, expr: String) {
        self.line(format!("{} {name} = {expr}", tensor("fp16", h, w)));
    }

    /// A surface as the input `name`, and its tensor `name_t`.
    fn input(&mut self, name: &str, surf: &Surface, r: usize, w: usize) -> String {
        self.params.push(format!("{} {name}", surf.buffer_type(r, w)));
        let ty = if surf.int8 { "int8" } else { "fp16" };
        self.line(format!("{} {name}_t = tensor_buffer_to_tensor<ios17>(input = {name})", tensor(ty, r, w)));
        format!("{name}_t")
    }

    /// Weights as an input, and the name of their fp16 tensor: int8 ones count in 1/128.
    fn weights(&mut self, name: &str, surf: &Surface, r: usize, w: usize) -> String {
        let t = self.input(name, surf, r, w);
        if !surf.int8 {
            return t;
        }
        self.f16(&format!("{name}_d"), r, w, format!("dequantize(input = {t}, scale = {})", fp16(1.0 / 128.0)));
        format!("{name}_d")
    }

    /// The sum over the segments of `weights[k]` · `values[k]`: `h` × `rows`.
    fn product(&mut self, out: &str, weights: &[String], values: &[String], h: usize, rows: usize) -> String {
        for (k, (w, v)) in weights.iter().zip(values).enumerate() {
            self.f16(
                &format!("{out}{k}"),
                h,
                rows,
                format!("matmul(transpose_x = bool(false), transpose_y = bool(false), x = {w}, y = {v})"),
            );
        }
        let mut total = format!("{out}0");
        for k in 1..weights.len() {
            let next = format!("{out}_sum{k}");
            self.f16(&next, h, rows, format!("add(x = {total}, y = {out}{k})"));
            total = next;
        }
        total
    }

    /// `value` (`h` × `rows`) as the function's output on `surf`.
    fn finish(mut self, name: &str, value: &str, surf: &Surface, h: usize, rows: usize) -> String {
        self.line(format!(
            "{} y = tensor_to_tensor_buffer<ios17>(input = {value}, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>({}))",
            surf.buffer_type(h, rows),
            surf.strides(h)
        ));
        format!("    func {name}<ios18>({}) {{\n{}    }} -> (y);\n", self.params.join(", "), self.body)
    }
}

/// The MIL of one FFN over `rows` rows: fp16 activations, weights of the surfaces' type. A
/// token's intermediate row still comes down to one peak, against fp16 overflow in the down
/// product.
pub(crate) fn ffn_function(hidden: usize, channels: usize, rows: usize, s: &FfnTypes<'_>) -> String {
    let segs = hidden / SEGMENT;
    let mut m = Mil::new();
    let xs: Vec<String> = (0..segs).map(|k| m.input(&format!("x{k}"), s.x[k], SEGMENT, rows)).collect();
    let (mut wg, mut wu) = (Vec::new(), Vec::new());
    for k in 0..segs {
        wg.push(m.weights(&format!("wg{k}"), s.wg[k], channels, SEGMENT));
        wu.push(m.weights(&format!("wu{k}"), s.wu[k], channels, SEGMENT));
    }
    m.input("sg", s.sg, channels, 1);
    m.input("su", s.su, channels, 1);
    let down = seg_widths(channels);
    let wd: Vec<String> = down.iter().enumerate().map(|(i, &w)| m.weights(&format!("wd{i}"), s.wd[i], hidden, w)).collect();
    m.input("sd", s.sd, hidden, 1);
    for (p, w) in [("g", &wg), ("u", &wu)] {
        let total = m.product(&format!("{p}m"), w, &xs, channels, rows);
        m.f16(&format!("{p}s"), channels, rows, format!("mul(x = {total}, y = s{p}_t)"));
    }
    let (c, r) = (channels, rows);
    if ACT.load(std::sync::atomic::Ordering::Relaxed) == 1 {
        // GELU, tanh form: 0.5·g·(1 + tanh(√(2/π)·(g + 0.044715·g³))). The activations are
        // fp16 here, where it stays within the exact (erf) form's rounding.
        m.f16("g2", c, r, "mul(x = gs, y = gs)".into());
        m.f16("g3", c, r, "mul(x = g2, y = gs)".into());
        m.f16("g3c", c, r, format!("mul(x = g3, y = {})", fp16(0.044715)));
        m.f16("gin", c, r, "add(x = gs, y = g3c)".into());
        m.f16("gk", c, r, format!("mul(x = gin, y = {})", fp16(0.797_884_6)));
        m.f16("th", c, r, "tanh(x = gk)".into());
        m.f16("tp", c, r, format!("add(x = th, y = {})", fp16(1.0)));
        m.f16("gh", c, r, format!("mul(x = gs, y = {})", fp16(0.5)));
        m.f16("silu", c, r, "mul(x = gh, y = tp)".into());
    } else {
        m.f16("gh", c, r, format!("mul(x = gs, y = {})", fp16(0.5)));
        m.f16("th", c, r, "tanh(x = gh)".into());
        m.f16("tp", c, r, format!("add(x = th, y = {})", fp16(1.0)));
        m.f16("silu", c, r, "mul(x = gh, y = tp)".into());
    }
    m.f16("h", c, r, "mul(x = silu, y = us)".into());
    m.f16("habs", c, r, "abs(x = h)".into());
    m.f16("peak", 1, r, "reduce_max(x = habs, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))".into());
    m.f16("floor", 1, r, format!("maximum(x = peak, y = {})", fp16(1.0 / 512.0)));
    m.f16("inverse", 1, r, format!("real_div(x = {}, y = floor)", fp16(1.0)));
    m.f16("hd", c, r, "mul(x = h, y = inverse)".into());
    let mut parts = Vec::new();
    let mut begin = 0usize;
    for (i, &w) in down.iter().enumerate() {
        m.f16(
            &format!("hd{i}"),
            w,
            r,
            format!("slice_by_size(x = hd, begin = tensor<int32, [4]>([0, 0, {begin}, 0]), size = tensor<int32, [4]>([1, 1, {w}, {r}]))"),
        );
        parts.push(format!("hd{i}"));
        begin += w;
    }
    let total = m.product("dm", &wd, &parts, hidden, rows);
    m.f16("ds", hidden, r, format!("mul(x = {total}, y = sd_t)"));
    m.f16("yt", hidden + 1, r, "concat(axis = int32(2), interleave = bool(false), values = (ds, floor))".into());
    m.finish(&format!("ffn{rows}"), "yt", s.out, hidden + 1, rows)
}

/// The MIL of a group of projections of one input over `rows` rows, their outputs one under
/// another. `projs`: each one's weight segments, its row scales and its rows.
pub(crate) fn lin_function(name: &str, rows: usize, x: &[&Surface], projs: &[(Vec<&Surface>, &Surface, usize)], out: &Surface) -> String {
    let mut m = Mil::new();
    let xs: Vec<String> = x.iter().enumerate().map(|(k, s)| m.input(&format!("x{k}"), s, SEGMENT, rows)).collect();
    let mut parts = Vec::new();
    let mut total_rows = 0usize;
    for (j, (w, scale, h)) in projs.iter().enumerate() {
        let ws: Vec<String> = w.iter().enumerate().map(|(k, surf)| m.weights(&format!("w{j}_{k}"), surf, *h, SEGMENT)).collect();
        m.input(&format!("s{j}"), scale, *h, 1);
        let total = m.product(&format!("p{j}m"), &ws, &xs, *h, rows);
        m.f16(&format!("p{j}"), *h, rows, format!("mul(x = {total}, y = s{j}_t)"));
        parts.push(format!("p{j}"));
        total_rows += h;
    }
    let value = if parts.len() == 1 {
        parts[0].clone()
    } else {
        m.f16("yt", total_rows, rows, format!("concat(axis = int32(2), interleave = bool(false), values = ({}))", parts.join(", ")));
        "yt".into()
    };
    m.finish(name, &value, out, total_rows, rows)
}

const MSL: &str = include_str!("ane_msl.metal");

/// The rows of the program's functions. A chunk runs on the smallest that holds it.
const FUNCTIONS: [usize; 3] = [128, 256, 512];

/// Halves of a chunk in flight at once.
pub(crate) const LANES: usize = 2;

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

/// One projection as the ANE reads it: its weight segments and the scale of each row.
struct Staged {
    w: Vec<Shared>,
    s: Shared,
}

fn staged(c: &Ctx, rows: usize, cols: usize, int8: bool) -> Result<Staged, String> {
    let w: Result<Vec<Shared>, String> = seg_widths(cols).into_iter().map(|w| shared_surface(c, rows, w, int8)).collect();
    Ok(Staged { w: w?, s: shared_surface(c, rows, 1, false)? })
}

/// One projection in the model's file: its absolute offset, rows and columns.
#[derive(Clone, Copy)]
pub(crate) struct Proj {
    pub(crate) abs: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

/// What the ANE takes of one layer: its FFN's gate, up and down, and the projections of the
/// mixer's input.
#[derive(Clone, Default)]
pub(crate) struct LayerProjs {
    pub(crate) ffn: Option<[Proj; 3]>,
    pub(crate) mix: Vec<Proj>,
}

/// The weights the ANE reads go in two groups, each with two sets that take turns. A third
/// set of FFN weights lets the next layer be staged before the second half lets one go, and
/// lost: Qwen3.8-27B on the M4, first token 27.8 and 27.7 s with two sets, 31.3 s with three
/// for the FFNs, 28.6 s with three for both groups. With two the halves keep a layer apart.
const FFN: usize = 0;
const MIX: usize = 1;

#[derive(Clone, Copy, Default)]
struct Slot {
    layer: Option<usize>,
    ready: bool,
    /// Evaluations of its layer the halves of this chunk still owe.
    uses: usize,
    /// The chunk that counted `uses`.
    chunk: u64,
    /// The layer a thread is to stage here, or is staging.
    pending: Option<usize>,
}

#[derive(Default)]
struct Stock {
    groups: [[Slot; 2]; 2],
    lanes: usize,
    chunk: u64,
    dead: bool,
    /// The lane that keeps the ANE from its FFN through the mixer call that follows.
    hold: Option<usize>,
}

/// A lane's surfaces: the input segments, the FFN's output, and the mixer projections' output
/// of each shape.
struct LaneIo {
    x: Vec<Shared>,
    out: Shared,
    mix: Vec<Shared>,
    /// The host's planes for `host_norm`: channels by rows, and rows by channels.
    host: Mutex<(Vec<f32>, Vec<f32>)>,
}

#[derive(Default)]
struct Times {
    /// Milliseconds by group: waiting for weights, the ANE, and the evaluations.
    weights: [f64; 2],
    ane: [f64; 2],
    count: [u64; 2],
    /// Waiting for the GPU's commands, and for the ANE while the other half had it.
    gpu: f64,
    queue: f64,
    /// The CPU's milliseconds on weights, and the layers staged.
    staged: [f64; 2],
    layers: [u64; 2],
    /// The CPU's milliseconds between an FFN and the next mixer.
    host: f64,
    /// The GPU's own milliseconds on commands of GDN and of attention layers, and their count.
    cmd: [f64; 2],
    cmds: [u64; 2],
}

pub(crate) struct AneFfn {
    prog: Program,
    pack: ComputePipelineState,
    join: ComputePipelineState,
    unpack: ComputePipelineState,
    hidden: usize,
    inter: usize,
    layers: Vec<LayerProjs>,
    /// The shapes of mixer projections: the rows of each projection of a group.
    shapes: Vec<Vec<usize>>,
    /// Each layer's shape, when the ANE takes its mixer's input.
    shape_of: Vec<Option<usize>>,
    ffn_sets: [[Staged; 3]; 2],
    /// By set, then shape, then projection.
    mix_sets: [Vec<Vec<Staged>>; 2],
    io: [LaneIo; LANES],
    tx: Shared,
    stock: Mutex<Stock>,
    turn: Condvar,
    /// One evaluation at a time.
    eval: Mutex<()>,
    times: Mutex<Times>,
    born: std::time::Instant,
}

unsafe impl Send for AneFfn {}
unsafe impl Sync for AneFfn {}

fn seg_widths(channels: usize) -> Vec<usize> {
    (0..channels.div_ceil(SEGMENT)).map(|i| SEGMENT.min(channels - i * SEGMENT)).collect()
}

/// Whether the mixer's projections go as fp16, which holds q4tp weights whole; int8 holds them
/// to 1.4 % and halves their memory (`CMF_ANE_MIX=int8`).
fn mix_int8() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("CMF_ANE_MIX").is_ok_and(|v| v.split(',').any(|t| t == "int8")))
}

unsafe extern "C" {
    fn mach_absolute_time() -> u64;
    fn mach_timebase_info(info: *mut [u32; 2]) -> i32;
}

/// The clock Metal stamps its commands with, in milliseconds.
pub(crate) fn clock_ms() -> f64 {
    static BASE: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    let scale = *BASE.get_or_init(|| {
        let mut info = [1u32, 1u32];
        // SAFETY: a two-word struct the call fills.
        unsafe { mach_timebase_info(&mut info) };
        info[0] as f64 / info[1] as f64 / 1e6
    });
    // SAFETY: no arguments and no state.
    unsafe { mach_absolute_time() as f64 * scale }
}

/// `CMF_ANE_TRACE=1` prints every stage of a chunk's halves with its start and its length, for
/// reading the schedule: `ane-trace START lane LANE layer LAYER WHAT LENGTH`, in milliseconds.
pub(crate) fn trace(lane: usize, layer: usize, what: &str, start: f64, end: f64) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *ON.get_or_init(|| std::env::var("CMF_ANE_TRACE").is_ok()) {
        eprintln!("ane-trace {start:.2} lane {lane} layer {layer} {what} {:.2}", end - start);
    }
}

/// What the ANE takes besides the FFNs (`CMF_ANE_MIX`): 0 nothing (`ffn`), 1 a GDN mixer's `z`
/// (`z`), 2 that and an attention layer's `q` (the default), 3 every input projection (`all`);
/// `,int8` stages them as int8. The default leaves the GPU and the ANE about equally busy in
/// both kinds of layer. The half that ran an FFN keeps the ANE for the next layer's call, so
/// the short call does not wait behind the other half's FFN. Qwen3.8-27B on the M4, 3 600
/// tokens, first token: 32.2 and 33.0 s with the FFNs alone, 30.5 and 28.7 s with `z`.
pub(crate) fn mix_level() -> u8 {
    static V: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let v = std::env::var("CMF_ANE_MIX").unwrap_or_default();
        let has = |t: &str| v.split(',').any(|x| x == t);
        if has("ffn") || has("0") {
            0
        } else if has("all") {
            3
        } else if has("z") {
            1
        } else {
            2
        }
    })
}

fn mix_on() -> bool {
    mix_level() > 0
}

impl AneFfn {
    /// `layers`: what the ANE may take of each layer, q4tp, the FFNs of one shape.
    pub(crate) fn new(c: &Ctx, mut layers: Vec<LayerProjs>) -> Result<AneFfn, String> {
        api()?;
        let first = layers.iter().find_map(|l| l.ffn).ok_or("no layers")?;
        let (hidden, inter) = (first[0].cols, first[0].rows);
        if hidden % SEGMENT != 0 || hidden / SEGMENT > 4 {
            return Err(format!("hidden {hidden} is not whole segments"));
        }
        if inter % 32 != 0 {
            return Err(format!("intermediate {inter} is not whole tiles"));
        }
        if layers.iter().filter_map(|l| l.ffn).any(|l| {
            l[0].cols != hidden || l[0].rows != inter || l[1].cols != hidden || l[1].rows != inter || l[2].rows != hidden || l[2].cols != inter
        }) {
            return Err("layers differ in shape".into());
        }
        // A mixer's projections go when they read the same rows as the FFN and fit one output.
        let mut shapes: Vec<Vec<usize>> = Vec::new();
        let mut shape_of = Vec::with_capacity(layers.len());
        for l in layers.iter_mut() {
            let rows: Vec<usize> = l.mix.iter().map(|p| p.rows).collect();
            let fits = mix_on() && !l.mix.is_empty() && l.mix.iter().all(|p| p.cols == hidden) && rows.iter().sum::<usize>() <= 16384;
            if !fits {
                l.mix.clear();
                shape_of.push(None);
                continue;
            }
            let at = shapes.iter().position(|s| *s == rows).unwrap_or_else(|| {
                shapes.push(rows.clone());
                shapes.len() - 1
            });
            shape_of.push(Some(at));
        }
        let opts = metal::CompileOptions::new();
        opts.set_language_version(metal::MTLLanguageVersion::V3_0);
        let lib = c._device.new_library_with_source(MSL, &opts).map_err(|e| format!("ane MSL: {e}"))?;
        let pso = |name: &str| -> Result<ComputePipelineState, String> {
            let f = lib.get_function(name, None).map_err(|e| format!("kernel {name}: {e}"))?;
            c._device.new_compute_pipeline_state_with_function(&f).map_err(|e| format!("pipeline {name}: {e}"))
        };
        let (pack, join, unpack) = (pso("ane_pack16")?, pso("ane_join")?, pso("ane_unpack")?);
        let ffn_set = || -> Result<[Staged; 3], String> {
            Ok([staged(c, inter, hidden, true)?, staged(c, inter, hidden, true)?, staged(c, hidden, inter, true)?])
        };
        let mix_set = || -> Result<Vec<Vec<Staged>>, String> {
            shapes.iter().map(|s| s.iter().map(|&r| staged(c, r, hidden, mix_int8())).collect()).collect()
        };
        let ffn_sets = [ffn_set()?, ffn_set()?];
        let mix_sets = [mix_set()?, mix_set()?];
        let max_rows = *FUNCTIONS.last().unwrap();
        let lane = || -> Result<LaneIo, String> {
            let x: Result<Vec<Shared>, String> =
                (0..hidden / SEGMENT).map(|_| shared_surface(c, SEGMENT, max_rows, false)).collect();
            let mix: Result<Vec<Shared>, String> =
                shapes.iter().map(|s| shared_surface(c, s.iter().sum(), max_rows, false)).collect();
            Ok(LaneIo {
                x: x?,
                out: shared_surface(c, hidden + 1, max_rows, false)?,
                mix: mix?,
                host: Mutex::new((Vec::new(), Vec::new())),
            })
        };
        let io = [lane()?, lane()?];
        let tx = shared_surface(c, 1, max_rows, false)?;
        // The join's input scale: there is nothing to bring back.
        let p = tx.s.base() as *mut u16;
        for i in 0..tx.s.bytes / 2 {
            // SAFETY: inside the surface, which no command uses yet.
            unsafe { *p.add(i) = f32_to_f16(1.0) };
        }
        let x: Vec<&Surface> = io[0].x.iter().map(|a| &a.s).collect();
        fn seg(t: &Staged) -> Vec<&Surface> {
            t.w.iter().map(|a| &a.s).collect()
        }
        let f = &ffn_sets[0];
        let types = FfnTypes { x: x.clone(), wg: seg(&f[0]), wu: seg(&f[1]), wd: seg(&f[2]), sg: &f[0].s.s, su: &f[1].s.s, sd: &f[2].s.s, out: &io[0].out.s };
        let mut text = String::from("program(1.3)\n{\n");
        for rows in FUNCTIONS {
            text.push_str(&ffn_function(hidden, inter, rows, &types));
            for (k, shape) in shapes.iter().enumerate() {
                let projs: Vec<(Vec<&Surface>, &Surface, usize)> =
                    mix_sets[0][k].iter().zip(shape).map(|(t, &r)| (seg(t), &t.s.s, r)).collect();
                text.push_str(&lin_function(&format!("mix{k}_{rows}"), rows, &x, &projs, &io[0].mix[k].s));
            }
        }
        text.push_str("}\n");
        drop(types);
        let t0 = std::time::Instant::now();
        let prog = Program::new(&text, &empty_blob())?;
        tracing::info!(
            "neural engine: FFN of {inter} channels over {} layers, mixer projections {:?} over {} ({}), program {:.1} s",
            layers.iter().filter(|l| l.ffn.is_some()).count(),
            shapes,
            shape_of.iter().flatten().count(),
            if mix_int8() { "int8" } else { "fp16" },
            t0.elapsed().as_secs_f64()
        );
        Ok(AneFfn {
            prog,
            pack,
            join,
            unpack,
            hidden,
            inter,
            layers,
            shapes,
            shape_of,
            ffn_sets,
            mix_sets,
            io,
            tx,
            stock: Mutex::new(Stock { lanes: 1, ..Stock::default() }),
            turn: Condvar::new(),
            eval: Mutex::new(()),
            times: Mutex::new(Times::default()),
            born: std::time::Instant::now(),
        })
    }

    /// The layer whose FFN has this gate.
    pub(crate) fn layer_of(&self, gate_abs: usize) -> Option<usize> {
        self.layers.iter().position(|l| l.ffn.is_some_and(|f| f[0].abs == gate_abs))
    }

    /// The layer whose mixer's first projection this is, when the ANE takes them.
    pub(crate) fn mix_layer(&self, first_abs: usize) -> Option<usize> {
        self.layers.iter().position(|l| l.mix.first().is_some_and(|p| p.abs == first_abs))
    }

    pub(crate) fn takes(&self, rows: usize) -> bool {
        (64..=*FUNCTIONS.last().unwrap()).contains(&rows)
    }

    /// A chunk begins, in `lanes` halves: each owes every layer one evaluation of a group.
    pub(crate) fn begin(&self, lanes: usize) {
        let mut st = self.stock.lock().unwrap();
        st.lanes = lanes.clamp(1, LANES);
        st.chunk += 1;
        st.hold = None;
        // Whatever an abandoned chunk still owed is void.
        for g in st.groups.iter_mut() {
            for s in g.iter_mut() {
                s.uses = 0;
            }
        }
        self.turn.notify_all();
    }

    /// No evaluation is to come: every wait ends.
    fn stop(&self) {
        self.stock.lock().unwrap().dead = true;
        self.turn.notify_all();
    }

    fn group_projs(&self, group: usize, li: usize) -> Vec<Proj> {
        match group {
            FFN => self.layers[li].ffn.map(|f| f.to_vec()).unwrap_or_default(),
            _ => self.layers[li].mix.clone(),
        }
    }

    fn group_set(&self, group: usize, li: usize, set: usize) -> &[Staged] {
        match group {
            FFN => &self.ffn_sets[set],
            _ => &self.mix_sets[set][self.shape_of[li].unwrap()],
        }
    }

    /// Layer `li`'s weights of a group into set `set` on the CPU: `file` holds each projection's
    /// address.
    fn stage(&self, group: usize, li: usize, set: usize, file: &[usize]) {
        let t0 = std::time::Instant::now();
        let projs = self.group_projs(group, li);
        for ((pr, t), &q) in projs.iter().zip(self.group_set(group, li, set)).zip(file) {
            let outs: Vec<(usize, usize)> = t.w.iter().map(|o| (o.s.base() as usize, o.s.stride)).collect();
            let wide = !t.w[0].s.int8;
            let (sbase, sstride) = (t.s.s.base() as usize, t.s.s.stride);
            let threads = stage_threads();
            let per = pr.rows.div_ceil(threads);
            let pr = *pr;
            std::thread::scope(|sc| {
                for th in 0..threads {
                    let outs = &outs;
                    sc.spawn(move || {
                        let (r0, r1) = (th * per, ((th + 1) * per).min(pr.rows));
                        // SAFETY: rows of the mapped tensor and of this set's surfaces, which no
                        // engine reads until the set is ready; threads own disjoint rows.
                        unsafe { stage_rows(q, pr.rows, pr.cols, r0, r1, outs, sbase, sstride, wide) };
                    });
                }
            });
        }
        let mut m = self.times.lock().unwrap();
        m.staged[group] += t0.elapsed().as_secs_f64() * 1e3;
        m.layers[group] += 1;
    }

    /// A thread stages layer `li` of a group into `set` once no half needs what the set holds.
    fn stage_later(self: &Arc<Self>, fbuf: &WeightArena, group: usize, li: usize, set: usize) {
        let file: Vec<usize> = self
            .group_projs(group, li)
            .iter()
            .map(|p| {
                let (b, off) = fbuf.locate(p.abs);
                b.contents() as usize + off
            })
            .collect();
        let me = self.clone();
        std::thread::spawn(move || {
            let mut st = me.stock.lock().unwrap();
            while st.groups[group][set].uses > 0 && !st.dead {
                st = me.turn.wait(st).unwrap();
            }
            if st.dead {
                st.groups[group][set].pending = None;
                me.turn.notify_all();
                return;
            }
            st.groups[group][set] = Slot { layer: Some(li), ready: false, uses: 0, chunk: 0, pending: Some(li) };
            drop(st);
            me.stage(group, li, set, &file);
            let mut st = me.stock.lock().unwrap();
            st.groups[group][set].ready = true;
            st.groups[group][set].pending = None;
            me.turn.notify_all();
        });
    }

    /// The set that holds layer `li`'s weights of a group, staged here if nobody did, with the
    /// next layer's begun on the other set. `release` returns it.
    fn acquire(self: &Arc<Self>, fbuf: &WeightArena, group: usize, li: usize, lane: usize) -> Result<usize, String> {
        let mut st = self.stock.lock().unwrap();
        let set = loop {
            if st.dead {
                return Err("the Neural Engine stopped".into());
            }
            let g = &mut st.groups[group];
            if let Some(s) = g.iter().position(|s| s.layer == Some(li) && s.ready) {
                break s;
            }
            if !g.iter().any(|s| s.pending == Some(li)) {
                if let Some(s) = g.iter().position(|s| s.pending.is_none() && s.uses == 0) {
                    g[s].pending = Some(li);
                    self.stage_later(fbuf, group, li, s);
                }
            }
            // A half that waits for weights lets the ANE go: the set it waits for may be one
            // the other half has yet to read.
            if st.hold == Some(lane) {
                st.hold = None;
                self.turn.notify_all();
            }
            st = self.turn.wait(st).unwrap();
        };
        let (chunk, lanes) = (st.chunk, st.lanes);
        let slot = &mut st.groups[group][set];
        if slot.chunk != chunk {
            slot.chunk = chunk;
            slot.uses = lanes;
        }
        // The layer after it, ahead of time. A chunk's last layer stages nothing: no thread
        // outlives the chunk, whose model may go.
        let next = li + 1;
        if next < self.layers.len() && !self.group_projs(group, next).is_empty() {
            let other = set ^ 1;
            let g = &mut st.groups[group];
            if g[other].layer != Some(next) && g[other].pending.is_none() {
                g[other].pending = Some(next);
                self.stage_later(fbuf, group, next, other);
            }
        }
        Ok(set)
    }

    fn release(&self, group: usize, set: usize) {
        let mut st = self.stock.lock().unwrap();
        let s = &mut st.groups[group][set];
        s.uses = s.uses.saturating_sub(1);
        self.turn.notify_all();
    }

    /// The chunk's normalized rows as the program's fp16 input, transposed.
    pub(crate) fn encode_input(&self, enc: &metal::ComputeCommandEncoderRef, n_b: &Buffer, rows: usize, lane: usize) {
        let hidden = self.hidden as u32;
        for (k, x) in self.io[lane].x.iter().enumerate() {
            enc.set_compute_pipeline_state(&self.pack);
            enc.set_buffer(0, Some(n_b), 0);
            enc.set_buffer(1, Some(&x.b), 0);
            let (ch, stride) = ((k * SEGMENT) as u32, (x.s.stride / 2) as u32);
            enc.set_bytes(2, 4, &hidden as *const u32 as *const c_void);
            enc.set_bytes(3, 4, &ch as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &stride as *const u32 as *const c_void);
            let rows_u = rows as u32;
            enc.set_bytes(5, 4, &rows_u as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new((rows as u64).div_ceil(32), (SEGMENT / 32) as u64, 1),
                MTLSize::new(32, 8, 1),
            );
        }
    }

    /// What stands between an FFN and the next layer's mixer, on the host, so that no command
    /// waits behind the other half's on the GPU: with `add` the FFN's output joins the stream
    /// `h` (`rows` × hidden), which is normed by `gain` into `n` and into this lane's input.
    /// The arithmetic of `add_rmsnorm_rows`.
    ///
    /// # Safety
    /// `h` and `n` hold `rows` × hidden floats that no command reads or writes now, and the
    /// lane's last evaluation is over.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn host_norm(&self, lane: usize, rows: usize, h: *mut f32, n: *mut f32, gain: &[f32], gemma: bool, eps: f32, add: bool) {
        let t0 = std::time::Instant::now();
        let hidden = self.hidden;
        let io = &self.io[lane];
        let mut planes = io.host.lock().unwrap();
        let (by_channel, by_row) = &mut *planes;
        by_channel.resize((hidden + 1) * rows, 0.0);
        by_row.resize(rows * hidden, 0.0);
        let gain: Vec<f32> = if gemma { gain.iter().map(|w| 1.0 + w).collect() } else { gain.to_vec() };
        unsafe {
            if add {
                let src = Plane { data: io.out.s.base() as *mut c_void, height: hidden + 1, width: rows, row_bytes: io.out.s.stride };
                let dst = Plane { data: by_channel.as_mut_ptr() as *mut c_void, height: hidden + 1, width: rows, row_bytes: rows * 4 };
                vImageConvert_Planar16FtoPlanarF(&src, &dst, 0);
                vDSP_mtrans(by_channel.as_ptr(), 1, by_row.as_mut_ptr(), 1, rows, hidden);
            }
            for t in 0..rows {
                let (hr, nr) = (h.add(t * hidden), n.add(t * hidden));
                if add {
                    // The row under the channels is each token's scale.
                    cblas_saxpy(hidden as i32, by_channel[hidden * rows + t], by_row.as_ptr().add(t * hidden), 1, hr, 1);
                }
                let mut squares = 0f32;
                vDSP_svesq(hr, 1, &mut squares, hidden);
                let inverse = 1.0 / (squares / hidden as f32 + eps).sqrt();
                vDSP_vsmul(hr, 1, &inverse, nr, 1, hidden);
                vDSP_vmul(nr, 1, gain.as_ptr(), 1, nr, 1, hidden);
            }
            vDSP_mtrans(n, 1, by_channel.as_mut_ptr(), 1, hidden, rows);
            for (k, x) in io.x.iter().enumerate() {
                let src = Plane { data: by_channel.as_mut_ptr().add(k * SEGMENT * rows) as *mut c_void, height: SEGMENT, width: rows, row_bytes: rows * 4 };
                let dst = Plane { data: x.s.base() as *mut c_void, height: SEGMENT, width: rows, row_bytes: x.s.stride };
                vImageConvert_PlanarFtoPlanar16F(&src, &dst, 0);
            }
        }
        self.times.lock().unwrap().host += t0.elapsed().as_secs_f64() * 1e3;
    }

    /// The GPU's time on a finished command of a GDN or an attention layer, for the profile.
    pub(crate) fn note_gpu(&self, attention: bool, ms: f64) {
        let mut m = self.times.lock().unwrap();
        m.cmd[attention as usize] += ms;
        m.cmds[attention as usize] += 1;
    }

    /// One evaluation of a group's function for layer `li` over the input `encode_input` wrote
    /// for this lane, whose command is done. `waited`: the milliseconds the caller spent on it.
    fn run(self: &Arc<Self>, fbuf: &WeightArena, group: usize, li: usize, rows: usize, lane: usize, waited: f64) -> Result<(), String> {
        let t0 = std::time::Instant::now();
        let set = self.acquire(fbuf, group, li, lane)?;
        let t1 = std::time::Instant::now();
        let func = FUNCTIONS.iter().copied().find(|&f| f >= rows).ok_or("too many rows")?;
        let io = &self.io[lane];
        let mut v: Vec<(String, &Surface)> = io.x.iter().enumerate().map(|(k, x)| (format!("x{k}"), &x.s)).collect();
        let (name, out, out_rows) = match group {
            FFN => {
                for (p, t) in ["g", "u", "d"].iter().zip(&self.ffn_sets[set]) {
                    for (k, w) in t.w.iter().enumerate() {
                        v.push((format!("w{p}{k}"), &w.s));
                    }
                    v.push((format!("s{p}"), &t.s.s));
                }
                (format!("ffn{func}"), &io.out, self.hidden + 1)
            }
            _ => {
                let shape = self.shape_of[li].ok_or("no mixer projections")?;
                for (j, t) in self.mix_sets[set][shape].iter().enumerate() {
                    for (k, w) in t.w.iter().enumerate() {
                        v.push((format!("w{j}_{k}"), &w.s));
                    }
                    v.push((format!("s{j}"), &t.s.s));
                }
                (format!("mix{shape}_{func}"), &io.mix[shape], self.shapes[shape].iter().sum())
            }
        };
        let refs: Vec<(&str, &Surface)> = v.iter().map(|(n, s)| (n.as_str(), *s)).collect();
        // A half keeps the ANE from its FFN through the next layer's mixer call. That call is
        // short, and behind the other half's FFN it waited 11 ms of a 60 ms layer with the GPU
        // idle; this way the other half's FFN waits for it instead, 6 ms at most.
        // The second half follows the first at every layer, so the first never waits for it
        // without a bound: it gives the second 3 ms to get to its call and then goes ahead.
        {
            let mut st = self.stock.lock().unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(3);
            while st.hold.is_some_and(|l| l != lane) && !st.dead {
                if lane == 0 {
                    let left = deadline.saturating_duration_since(std::time::Instant::now());
                    if left.is_zero() {
                        break;
                    }
                    st = self.turn.wait_timeout(st, left).unwrap().0;
                } else {
                    st = self.turn.wait(st).unwrap();
                }
            }
            if group == FFN && st.lanes > 1 && self.layers.get(li + 1).is_some_and(|l| !l.mix.is_empty()) {
                st.hold = Some(lane);
            }
        }
        let (result, t2) = {
            let _one = self.eval.lock().unwrap();
            let t2 = std::time::Instant::now();
            let c0 = clock_ms();
            let r = self.prog.eval(&name, &refs, &out.s);
            trace(lane, li, if group == FFN { "ane-ffn" } else { "ane-mix" }, c0, clock_ms());
            (r, t2)
        };
        self.release(group, set);
        if group == MIX || result.is_err() {
            let mut st = self.stock.lock().unwrap();
            if st.hold == Some(lane) {
                st.hold = None;
                self.turn.notify_all();
            }
        }
        result?;
        if !finite(&out.s, out_rows, rows) {
            return Err("an output that is not finite".into());
        }
        let mut m = self.times.lock().unwrap();
        m.weights[group] += (t1 - t0).as_secs_f64() * 1e3;
        m.queue += (t2 - t1).as_secs_f64() * 1e3;
        m.ane[group] += t2.elapsed().as_secs_f64() * 1e3;
        m.gpu += waited;
        m.count[group] += 1;
        if group == FFN && m.count[FFN] % 128 == 0 && std::env::var("CMF_ANE_PROF").is_ok() {
            let n = |g: usize| m.count[g].max(1) as f64;
            let l = |g: usize| m.layers[g].max(1) as f64;
            eprintln!(
                "ane-prof: {} FFNs {:.1} ms each, {} mixer inputs {:.1} ms each; the GPU's commands {:.1} ms a GDN layer, {:.1} ms an attention layer; a half waited {:.1} ms for the GPU, {:.1} ms for the ANE, {:.1} ms for weights, the host's norm took {:.1} ms; the CPU {:.1} + {:.1} ms a layer of weights; {:.1} ms of wall clock an FFN",
                m.count[FFN],
                m.ane[FFN] / n(FFN),
                m.count[MIX],
                m.ane[MIX] / n(MIX),
                m.cmd[0] / m.cmds[0].max(1) as f64,
                m.cmd[1] / m.cmds[1].max(1) as f64,
                m.gpu / n(FFN),
                m.queue / n(FFN),
                (m.weights[FFN] + m.weights[MIX]) / n(FFN),
                m.host / n(FFN),
                m.staged[FFN] / l(FFN),
                m.staged[MIX] / l(MIX),
                self.born.elapsed().as_secs_f64() * 1e3 / n(FFN)
            );
            super::cut_report();
        }
        Ok(())
    }

    /// Layer `li`'s FFN: `encode_join` brings its output.
    pub(crate) fn run_ffn(self: &Arc<Self>, fbuf: &WeightArena, li: usize, rows: usize, lane: usize, waited: f64) -> Result<(), String> {
        self.run(fbuf, FFN, li, rows, lane, waited)
    }

    /// Layer `li`'s mixer projections: `encode_unpack` brings each one's output.
    pub(crate) fn run_mix(self: &Arc<Self>, fbuf: &WeightArena, li: usize, rows: usize, lane: usize, waited: f64) -> Result<(), String> {
        self.run(fbuf, MIX, li, rows, lane, waited)
    }

    /// The output of `run_ffn` as the residual's delta.
    pub(crate) fn encode_join(&self, enc: &metal::ComputeCommandEncoderRef, d_b: &Buffer, rows: usize, lane: usize) {
        let out = &self.io[lane].out;
        let (hidden, stride) = (self.hidden as u32, (out.s.stride / 2) as u32);
        enc.set_compute_pipeline_state(&self.join);
        enc.set_buffer(0, Some(d_b), 0);
        enc.set_buffer(1, Some(&out.b), 0);
        enc.set_buffer(2, Some(&self.tx.b), 0);
        enc.set_bytes(3, 4, &hidden as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &stride as *const u32 as *const c_void);
        let rows_u = rows as u32;
        enc.set_bytes(5, 4, &rows_u as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new((rows as u64).div_ceil(32), (self.hidden as u64).div_ceil(32), 1),
            MTLSize::new(32, 8, 1),
        );
    }

    /// Projection `j` of layer `li`'s mixer out of `run_mix`'s output: `rows` rows of its width.
    pub(crate) fn encode_unpack(&self, enc: &metal::ComputeCommandEncoderRef, li: usize, j: usize, y: &Buffer, rows: usize, lane: usize) {
        let Some(shape) = self.shape_of[li] else { return };
        let out = &self.io[lane].mix[shape];
        let first: usize = self.shapes[shape][..j].iter().sum();
        let (width, stride, first) = (self.shapes[shape][j] as u32, (out.s.stride / 2) as u32, first as u32);
        enc.set_compute_pipeline_state(&self.unpack);
        enc.set_buffer(0, Some(y), 0);
        enc.set_buffer(1, Some(&out.b), 0);
        enc.set_bytes(2, 4, &width as *const u32 as *const c_void);
        enc.set_bytes(3, 4, &stride as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &first as *const u32 as *const c_void);
        let rows_u = rows as u32;
        enc.set_bytes(5, 4, &rows_u as *const u32 as *const c_void);
        enc.dispatch_thread_groups(
            MTLSize::new((rows as u64).div_ceil(32), (width as u64).div_ceil(32), 1),
            MTLSize::new(32, 8, 1),
        );
    }
}

/// Threads a staging takes.
fn stage_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("CMF_ANE_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n| (1..=32).contains(n))
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()).min(8))
    })
}

/// Whether the leading `rows` × `width` fp16 values of a surface are all finite.
fn finite(s: &Surface, rows: usize, width: usize) -> bool {
    let base = s.base() as *const u16;
    let mut bad = 0u16;
    for r in 0..rows {
        // SAFETY: inside the surface; the ANE's evaluation is over.
        let row = unsafe { std::slice::from_raw_parts(base.add(r * s.stride / 2), width) };
        let mut top = 0u16;
        for v in row {
            top |= ((*v & 0x7c00) == 0x7c00) as u16;
        }
        bad |= top;
    }
    bad == 0
}

/// Rows [r0, r1) of a q4tp tensor under each row's scale, which goes to its fp16 surface: int8
/// in 1/128, or with `wide` fp16, which holds the four bits and a tile's scale whole. `outs`:
/// per segment of 80 tiles, its surface's base and stride.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
unsafe fn stage_rows(q: usize, rows: usize, cols: usize, r0: usize, r1: usize, outs: &[(usize, usize)], sbase: usize, sstride: usize, wide: bool) {
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
            let bits = f32_to_f16(if wide { peak } else { peak / 127.0 * 128.0 });
            *((sbase + r * sstride) as *mut u16) = bits;
            let inverse = if wide { 1.0 } else { 128.0 } / f16_to_f32(bits);
            // By code: the sixteen values of a tile, as int8 or as the two bytes of fp16.
            let mut luts = [[0u8; 32]; 32];
            let mut have = 0u32;
            let n_lo = vdupq_n_u8(15);
            for g in 0..gpr {
                let c = code[g] as usize;
                if have & (1 << c) == 0 {
                    have |= 1 << c;
                    let k = (lo + c as f32 * step).exp2() * inverse;
                    for n in 0..16 {
                        let v = (n as f32 - 8.0) * k;
                        if wide {
                            let h = f32_to_f16(v).to_le_bytes();
                            luts[c][n] = h[0];
                            luts[c][16 + n] = h[1];
                        } else {
                            luts[c][n] = v.round().clamp(-127.0, 127.0) as i8 as u8;
                        }
                    }
                }
                let tile = vld1q_u8(q.add((r * gpr + g) * 16));
                let (l, h) = (vandq_u8(tile, n_lo), vshrq_n_u8(tile, 4));
                let (i0, i1) = (vzip1q_u8(l, h), vzip2q_u8(l, h));
                let (seg, within) = (g / per_seg, g % per_seg);
                let (base, st) = outs[seg];
                let low = vld1q_u8(luts[c].as_ptr());
                if wide {
                    let high = vld1q_u8(luts[c].as_ptr().add(16));
                    let out = (base + r * st + within * 64) as *mut u8;
                    let (a0, b0) = (vqtbl1q_u8(low, i0), vqtbl1q_u8(high, i0));
                    let (a1, b1) = (vqtbl1q_u8(low, i1), vqtbl1q_u8(high, i1));
                    vst1q_u8(out, vzip1q_u8(a0, b0));
                    vst1q_u8(out.add(16), vzip2q_u8(a0, b0));
                    vst1q_u8(out.add(32), vzip1q_u8(a1, b1));
                    vst1q_u8(out.add(48), vzip2q_u8(a1, b1));
                } else {
                    let out = (base + r * st + within * 32) as *mut u8;
                    vst1q_u8(out, vqtbl1q_u8(low, i0));
                    vst1q_u8(out.add(16), vqtbl1q_u8(low, i1));
                }
            }
        }
    }
}

/// Macs without NEON have no Neural Engine: `wanted` keeps them away from this.
#[cfg(not(target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
unsafe fn stage_rows(_q: usize, _rows: usize, _cols: usize, _r0: usize, _r1: usize, _outs: &[(usize, usize)], _sbase: usize, _sstride: usize, _wide: bool) {}

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
/// The FFN activation the Neural Engine's program computes: 0 SiLU, 1 GELU (Spark-X2.6).
static ACT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// The file's FFN activation, before its program is built.
pub(crate) fn set_act(act: u8) {
    ACT.store(act, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the Neural Engine's FFN program computes this activation.
pub(crate) fn takes_act(act: u8) -> bool {
    act == ACT.load(std::sync::atomic::Ordering::Relaxed) && act <= 1
}

pub(crate) fn wanted() -> bool {
    cfg!(target_arch = "aarch64")
        && std::env::var("CMF_ANE").as_deref() == Ok("1")
        && !REFUSED.load(std::sync::atomic::Ordering::Relaxed)
        && api().is_ok()
}

/// The ANE's side for these layers, built once; any refusal keeps the GPU's.
pub(crate) fn ffn_for(c: &Ctx, layers: Vec<LayerProjs>) -> Option<Arc<AneFfn>> {
    if !wanted() {
        return None;
    }
    let mut slot = ANE.lock().unwrap();
    let first = |l: &[LayerProjs]| l.iter().find_map(|l| l.ffn).map(|f| f[0].abs);
    if let Some(a) = slot.as_ref() {
        if a.layers.len() == layers.len() && first(&a.layers) == first(&layers) {
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

/// The ANE stopped: the GPU's for the rest of the process, and nobody waits for it.
pub(crate) fn retire(why: &str) {
    if !REFUSED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!("neural engine off: {why}");
    }
    if let Some(a) = ANE.lock().unwrap().as_ref() {
        a.stop();
    }
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
