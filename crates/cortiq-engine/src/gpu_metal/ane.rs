//! The Neural Engine through Apple's private client, the service behind Core ML. A MIL program
//! over IOSurfaces the GPU shares, after Splash (Inco AI, Apache-2.0): see THIRD_PARTY_NOTICES.
//! The interface may change with any macOS: everything here fails softly to the GPU.

use objc::rc::autoreleasepool;
use objc::runtime::{BOOL, Class, NO, Object};
use objc::{class, msg_send, sel, sel_impl};
use std::ffi::{CStr, CString, c_void};
use std::path::{Path, PathBuf};

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

/// The blob of one fp16 constant, which a program names at offset 64.
pub(crate) fn constant_blob(values: &[half_bits::F16]) -> Vec<u8> {
    let data = 128usize;
    let n = std::mem::size_of_val(values);
    let mut blob = vec![0u8; data + n];
    blob[0..4].copy_from_slice(&1u32.to_le_bytes());
    blob[4..8].copy_from_slice(&2u32.to_le_bytes());
    blob[64..68].copy_from_slice(&0xdeadbeefu32.to_le_bytes());
    blob[68..72].copy_from_slice(&1u32.to_le_bytes());
    blob[72..80].copy_from_slice(&(n as u64).to_le_bytes());
    blob[80..88].copy_from_slice(&(data as u64).to_le_bytes());
    for (i, v) in values.iter().enumerate() {
        blob[data + i * 2..data + i * 2 + 2].copy_from_slice(&v.0.to_le_bytes());
    }
    blob
}

/// fp16 by hand: the stable toolchain has no f16.
pub(crate) mod half_bits {
    #[derive(Clone, Copy)]
    pub(crate) struct F16(pub(crate) u16);
    impl F16 {
        pub(crate) fn from_f32(v: f32) -> F16 {
            let b = v.to_bits();
            let s = ((b >> 16) & 0x8000) as u16;
            let e = ((b >> 23) & 0xff) as i32 - 127 + 15;
            let m = b & 0x7fffff;
            if e <= 0 {
                if e < -10 {
                    return F16(s);
                }
                let m = (m | 0x800000) >> (1 - e);
                return F16(s | ((m + 0x1000) >> 13) as u16);
            }
            if e >= 31 {
                return F16(s | 0x7c00);
            }
            F16(s | ((e as u16) << 10) + (((m + 0x1000) >> 13) as u16))
        }
    }
}

/// How one layer's FFN runs on the ANE: `hidden` inputs, `channels` of its intermediate width,
/// `rows` rows. Mirrors Splash's AneFfn::function: int8 weights and activations, a Hadamard
/// rotation of the intermediate row, a per-token int8 scale.
pub(crate) const SEGMENT: usize = 2560;
pub(crate) const BLOCK: usize = 512;

/// The surfaces a function's text takes its types from.
pub(crate) struct FfnTypes<'a> {
    pub(crate) x: Vec<&'a Surface>,
    pub(crate) tx: &'a Surface,
    pub(crate) wg: Vec<&'a Surface>,
    pub(crate) wu: Vec<&'a Surface>,
    pub(crate) wd: Vec<&'a Surface>,
    pub(crate) sg: &'a Surface,
    pub(crate) su: &'a Surface,
    pub(crate) sd: &'a Surface,
    pub(crate) out: &'a Surface,
}

pub(crate) struct FfnSurfaces {
    pub(crate) x: Vec<Surface>,
    pub(crate) tx: Surface,
    pub(crate) wg: Vec<Surface>,
    pub(crate) wu: Vec<Surface>,
    pub(crate) wd: Vec<Surface>,
    pub(crate) sg: Surface,
    pub(crate) su: Surface,
    pub(crate) sd: Surface,
    pub(crate) out: Surface,
}

pub(crate) fn ffn_surfaces(hidden: usize, channels: usize, max_rows: usize) -> Result<FfnSurfaces, String> {
    let segs = hidden / SEGMENT;
    let down = channels.div_ceil(SEGMENT);
    let mk = |r, w, i| Surface::new(r, w, i);
    let mut x = Vec::new();
    let (mut wg, mut wu, mut wd) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..segs {
        x.push(mk(SEGMENT, max_rows, true)?);
        wg.push(mk(channels, SEGMENT, true)?);
        wu.push(mk(channels, SEGMENT, true)?);
    }
    for i in 0..down {
        let w = SEGMENT.min(channels - i * SEGMENT);
        wd.push(mk(hidden, w, true)?);
    }
    Ok(FfnSurfaces {
        x,
        tx: mk(1, max_rows, false)?,
        wg,
        wu,
        wd,
        sg: mk(channels, 1, false)?,
        su: mk(channels, 1, false)?,
        sd: mk(hidden, 1, false)?,
        out: mk(hidden + 1, max_rows, false)?,
    })
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

/// The MIL of one function of `rows` rows, as Splash's AneFfn::function writes it.
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
    input("tx".into(), s.tx, 1, rows, &mut line);
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
    for k in 0..segs {
        line2(f16(&format!("x{k}_d"), SEGMENT, rows, format!("dequantize(input = x{k}_t, scale = {unit})")));
    }
    for p in ["g", "u"] {
        for k in 0..segs {
            matmul(format!("{p}m{k}"), format!("w{p}{k}"), format!("x{k}_d"), channels, SEGMENT, &mut line2);
        }
        let total = sum(&format!("{p}m"), segs, channels, &mut line2);
        line2(f16(&format!("{p}s"), channels, rows, format!("mul(x = {total}, y = s{p}_t)")));
    }
    line2(f16("gt", channels, rows, "mul(x = gs, y = tx_t)".into()));
    line2(f16("gh", channels, rows, format!("mul(x = gt, y = {})", fp16(0.5))));
    line2(f16("th", channels, rows, "tanh(x = gh)".into()));
    line2(f16("tp", channels, rows, format!("add(x = th, y = {})", fp16(1.0))));
    line2(f16("silu", channels, rows, "mul(x = gh, y = tp)".into()));
    line2(f16("h", channels, rows, "mul(x = silu, y = us)".into()));
    let (c, r, b) = (channels, rows, BLOCK);
    line2(format!("tensor<fp16, [1, {c}, 1, {r}]> h4 = reshape(x = h, shape = tensor<int32, [4]>([1, {c}, 1, {r}]))"));
    line2(format!(
        "tensor<fp16, [{c}, {b}, 1, 1]> rotation = const()[name = string(\"rotation\"), val = tensor<fp16, [{c}, {b}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64(64)))]"
    ));
    line2(format!(
        "tensor<fp16, [1, {c}, 1, {r}]> hr4 = conv(dilations = tensor<int32, [2]>([1, 1]), groups = int32({}), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), strides = tensor<int32, [2]>([1, 1]), weight = rotation, x = h4)",
        c / b
    ));
    line2(f16("hr", c, r, format!("reshape(x = hr4, shape = tensor<int32, [4]>([1, 1, {c}, {r}]))")));
    line2(f16("habs", c, r, "abs(x = hr)".into()));
    line2(f16("peak", 1, r, "reduce_max(x = habs, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))".into()));
    line2(f16("floor", 1, r, format!("maximum(x = peak, y = {})", fp16(1.0 / 512.0))));
    line2(f16("inverse", 1, r, format!("real_div(x = {}, y = floor)", fp16(127.0))));
    line2(f16("hs", c, r, "mul(x = hr, y = inverse)".into()));
    line2(format!("{} hq = quantize(input = hs, scale = fp16(1), output_dtype = string(\"int8\"))", tensor("int8", c, r)));
    line2(f16("hd", c, r, format!("dequantize(input = hq, scale = {unit})")));
    line2(f16("hscale", 1, r, format!("mul(x = floor, y = {})", fp16(128.0 / 127.0))));
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
    line2(f16("yt", hidden + 1, r, "concat(axis = int32(2), interleave = bool(false), values = (ds, hscale))".into()));
    line2(format!(
        "{} y = tensor_to_tensor_buffer<ios17>(input = yt, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>({}))",
        s.out.buffer_type(hidden + 1, r),
        s.out.strides(hidden + 1)
    ));
    format!("    func ffn{rows}<ios18>({}) {{\n{body}    }} -> (y);\n", params.join(", "))
}

/// The Hadamard rotation constant of the intermediate rows.
pub(crate) fn rotation_blob(channels: usize) -> Vec<u8> {
    let signs = rotation_signs();
    let mut v = Vec::with_capacity(channels * BLOCK);
    let norm = 1.0 / (BLOCK as f32).sqrt();
    for ch in 0..channels {
        let out = ch % BLOCK;
        for i in 0..BLOCK {
            let h = if ((out & i).count_ones() & 1) == 1 { -1.0 } else { 1.0 };
            v.push(half_bits::F16::from_f32(signs[out] * h * norm));
        }
    }
    constant_blob(&v)
}

pub(crate) fn rotation_signs() -> Vec<f32> {
    let mut x: u64 = 20260930;
    (0..BLOCK)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            if x & 1 == 1 { -1.0 } else { 1.0 }
        })
        .collect()
}

// ── A dense FFN of a prompt chunk on the ANE ─────────────────────────────────────────────────

use super::*;

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

struct Pipes {
    rotate: ComputePipelineState,
    pack: ComputePipelineState,
    w_in: ComputePipelineState,
    w_int: ComputePipelineState,
    s_in: ComputePipelineState,
    s_int: ComputePipelineState,
    join: ComputePipelineState,
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
    p: Pipes,
    hidden: usize,
    inter: usize,
    sets: [WeightSet; 2],
    x: Vec<Shared>,
    tx: Shared,
    out: Shared,
    rotated: Buffer,
    signs: Buffer,
    status: Buffer,
    row_scales: Buffer,
    layers: Vec<[Proj; 3]>,
    /// What each set holds, once a staging is done.
    staged: Mutex<[Option<usize>; 2]>,
    /// A staging on its way, for the set it writes.
    staging: Mutex<Option<(metal::CommandBuffer, usize, usize)>>,
    pub(crate) ms: Mutex<(f64, f64, f64, u64)>,
}

unsafe impl Send for AneFfn {}
unsafe impl Sync for AneFfn {}

fn seg_widths(channels: usize) -> Vec<usize> {
    (0..channels.div_ceil(SEGMENT)).map(|i| SEGMENT.min(channels - i * SEGMENT)).collect()
}

fn weight_set(c: &Ctx, hidden: usize, inter: usize) -> Result<WeightSet, String> {
    let segs = hidden / SEGMENT;
    let mk = |r, w, i| shared_surface(c, r, w, i);
    let (mut wg, mut wu, mut wd) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..segs {
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
    pub(crate) fn new(c: &Ctx, fbuf: &WeightArena, layers: Vec<[Proj; 3]>) -> Result<AneFfn, String> {
        api()?;
        let first = layers.first().ok_or("no layers")?;
        let (hidden, inter) = (first[0].cols, first[0].rows);
        if hidden % SEGMENT != 0 || hidden % 1024 != 0 || hidden / SEGMENT > 4 {
            return Err(format!("hidden {hidden} is not whole segments"));
        }
        if inter % BLOCK != 0 || inter % WEIGHT_ROWS_U != 0 {
            return Err(format!("intermediate {inter} is not whole blocks"));
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
        let p = Pipes {
            rotate: pso("ane_rotate")?,
            pack: pso("ane_pack")?,
            w_in: pso("ane_weights_inputs")?,
            w_int: pso("ane_weights_intermediate")?,
            s_in: pso("ane_row_scale_inputs")?,
            s_int: pso("ane_row_scale_intermediate")?,
            join: pso("ane_join")?,
        };
        let sets = [weight_set(c, hidden, inter)?, weight_set(c, hidden, inter)?];
        let max_rows = *FUNCTIONS.last().unwrap();
        let x: Result<Vec<Shared>, String> =
            (0..hidden / SEGMENT).map(|_| shared_surface(c, SEGMENT, max_rows, true)).collect();
        let x = x?;
        let tx = shared_surface(c, 1, max_rows, false)?;
        let out = shared_surface(c, hidden + 1, max_rows, false)?;
        let shared = |bytes: usize| c._device.new_buffer(bytes.max(16) as u64, MTLResourceOptions::StorageModeShared);
        let signs = shared(BLOCK * 4);
        let sv = rotation_signs();
        // SAFETY: fresh shared buffers of these sizes.
        unsafe { std::ptr::copy_nonoverlapping(sv.as_ptr(), signs.contents() as *mut f32, BLOCK) };
        let rotated = shared(max_rows * hidden * 2);
        let status = shared(16);
        let per = 2 * inter + hidden;
        let row_scales = shared(layers.len() * per * 2);
        // The program, with one surface set's types: both sets have one stride.
        let view = FfnSurfacesRef { x: &x, wg: &sets[0].wg, wu: &sets[0].wu, wd: &sets[0].wd, tx: &tx, sg: &sets[0].sg, su: &sets[0].su, sd: &sets[0].sd, out: &out };
        let mut text = String::from("program(1.3)\n{\n");
        for rows in FUNCTIONS {
            text.push_str(&ffn_function_ref(hidden, inter, rows, &view));
        }
        text.push_str("}\n");
        let t0 = std::time::Instant::now();
        let prog = Program::new(&text, &rotation_blob(inter))?;
        let me = AneFfn {
            prog,
            p,
            hidden,
            inter,
            sets,
            x,
            tx,
            out,
            rotated,
            signs,
            status,
            row_scales,
            layers,
            staged: Mutex::new([None, None]),
            staging: Mutex::new(None),
            ms: Mutex::new((0.0, 0.0, 0.0, 0)),
        };
        let t1 = std::time::Instant::now();
        me.row_scales_cached(c, fbuf)?;
        tracing::info!(
            "neural engine: FFN of {inter} channels over {} layers, program {:.1} s, row scales {:.1} s",
            me.layers.len(),
            (t1 - t0).as_secs_f64(),
            t1.elapsed().as_secs_f64()
        );
        Ok(me)
    }

    fn scale_off(&self, layer: usize, matrix: usize) -> usize {
        let per = 2 * self.inter + self.hidden;
        (layer * per + matrix * self.inter) * 2
    }

    /// The row scales of a model seen before come from a small file: reading them takes every
    /// layer's weights through the GPU once.
    fn row_scales_cached(&self, c: &Ctx, fbuf: &WeightArena) -> Result<(), String> {
        let n = self.layers.len() * (2 * self.inter + self.hidden) * 2;
        let mut key = 0xcbf29ce484222325u64;
        for l in &self.layers {
            for pr in l {
                key = fnv(&(pr.abs as u64).to_le_bytes(), key);
                key = fnv(&(pr.rows as u64).to_le_bytes(), key);
            }
        }
        let first = self.layers[0][0].abs;
        // SAFETY: the first gate's params are inside the mapped file that `fbuf` covers.
        let (buf, off) = fbuf.locate(first + self.inter * (self.hidden / 32) * 16);
        let tag = unsafe { std::slice::from_raw_parts((buf.contents() as *const u8).add(off), 64) };
        key = fnv(tag, key);
        let file = cache_dir().join(format!("row-scales-{key:016x}.bin"));
        if let Ok(data) = std::fs::read(&file) {
            if data.len() == n {
                // SAFETY: a shared buffer of `n` bytes that no command is using.
                unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.row_scales.contents() as *mut u8, n) };
                return Ok(());
            }
        }
        self.compute_row_scales(c, fbuf)?;
        // SAFETY: as above; the command is done.
        let data = unsafe { std::slice::from_raw_parts(self.row_scales.contents() as *const u8, n) };
        let _ = std::fs::create_dir_all(cache_dir());
        let _ = std::fs::write(&file, data);
        Ok(())
    }

    fn compute_row_scales(&self, c: &Ctx, fbuf: &WeightArena) -> Result<(), String> {
        for (li, l) in self.layers.iter().enumerate() {
            let cmd = c.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            for m in 0..3 {
                let pr = l[m];
                let inputs = m < 2;
                enc.set_compute_pipeline_state(if inputs { &self.p.s_in } else { &self.p.s_int });
                fbuf.bind(enc, 0, pr.abs);
                enc.set_buffer(1, Some(&self.row_scales), self.scale_off(li, m) as u64);
                enc.set_buffer(2, Some(&self.signs), 0);
                let params = [(pr.cols / 32) as u32, pr.rows as u32, 0, pr.cols as u32, 0, 0];
                enc.set_bytes(3, 24, params.as_ptr() as *const c_void);
                enc.dispatch_thread_groups(MTLSize::new((pr.rows / WEIGHT_ROWS_U) as u64, 1, 1), MTLSize::new(256, 1, 1));
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            if cmd.status() == metal::MTLCommandBufferStatus::Error {
                return Err("row scales: the command failed".into());
            }
        }
        Ok(())
    }

    pub(crate) fn layer_of(&self, gate_abs: usize) -> Option<usize> {
        self.layers.iter().position(|l| l[0].abs == gate_abs)
    }

    pub(crate) fn takes(&self, rows: usize) -> bool {
        FUNCTIONS.iter().any(|&f| f == rows) || (rows >= 64 && rows <= *FUNCTIONS.last().unwrap())
    }

    /// Layer `li`'s int8 weights into set `set`.
    fn encode_weights(&self, enc: &metal::ComputeCommandEncoderRef, fbuf: &WeightArena, li: usize, set: usize) {
        let t = &self.sets[set];
        let l = &self.layers[li];
        let mut add = |m: usize, out: &Shared, scale: &Shared, input: usize, width: usize| {
            let pr = l[m];
            let inputs = m < 2;
            enc.set_compute_pipeline_state(if inputs { &self.p.w_in } else { &self.p.w_int });
            fbuf.bind(enc, 0, pr.abs);
            enc.set_buffer(1, Some(&self.row_scales), self.scale_off(li, m) as u64);
            enc.set_buffer(2, Some(&out.b), 0);
            enc.set_buffer(3, Some(&scale.b), 0);
            enc.set_buffer(4, Some(&self.signs), 0);
            let params = [
                (pr.cols / 32) as u32,
                pr.rows as u32,
                input as u32,
                width as u32,
                out.s.stride as u32,
                (scale.s.stride / 2) as u32,
            ];
            enc.set_bytes(5, 24, params.as_ptr() as *const c_void);
            let block = if inputs { 128 } else { BLOCK };
            enc.dispatch_thread_groups(
                MTLSize::new((pr.rows / WEIGHT_ROWS_U) as u64, (width / block) as u64, 1),
                MTLSize::new(256, 1, 1),
            );
        };
        for k in 0..t.wg.len() {
            add(0, &t.wg[k], &t.sg, k * SEGMENT, SEGMENT);
            add(1, &t.wu[k], &t.su, k * SEGMENT, SEGMENT);
        }
        for (i, w) in seg_widths(self.inter).into_iter().enumerate() {
            add(2, &t.wd[i], &t.sd, i * SEGMENT, w);
        }
    }

    /// The chunk's normalized rows as the program's int8 input.
    pub(crate) fn encode_input(&self, enc: &metal::ComputeCommandEncoderRef, n_b: &Buffer, rows: usize) {
        let hidden = self.hidden as u32;
        enc.set_compute_pipeline_state(&self.p.rotate);
        enc.set_buffer(0, Some(n_b), 0);
        enc.set_buffer(1, Some(&self.signs), 0);
        enc.set_buffer(2, Some(&self.rotated), 0);
        enc.set_buffer(3, Some(&self.tx.b), 0);
        enc.set_bytes(4, 4, &hidden as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(256, 1, 1));
        for (k, x) in self.x.iter().enumerate() {
            enc.set_compute_pipeline_state(&self.p.pack);
            enc.set_buffer(0, Some(&self.rotated), 0);
            enc.set_buffer(1, Some(&x.b), 0);
            let (ch, stride) = ((k * SEGMENT) as u32, x.s.stride as u32);
            enc.set_bytes(2, 4, &hidden as *const u32 as *const c_void);
            enc.set_bytes(3, 4, &ch as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &stride as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize::new((rows as u64).div_ceil(32), (SEGMENT / 32) as u64, 1),
                MTLSize::new(32, 8, 1),
            );
        }
    }

    fn inputs<'a>(&'a self, set: usize) -> Vec<(String, &'a Surface)> {
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
        v.push(("tx".into(), &self.tx.s));
        v.push(("sg".into(), &t.sg.s));
        v.push(("su".into(), &t.su.s));
        v.push(("sd".into(), &t.sd.s));
        v
    }

    fn wait_staging(&self) -> Result<(), String> {
        if let Some((cmd, set, li)) = self.staging.lock().unwrap().take() {
            cmd.wait_until_completed();
            if cmd.status() == metal::MTLCommandBufferStatus::Error {
                return Err("weights staging failed".into());
            }
            self.staged.lock().unwrap()[set] = Some(li);
        }
        Ok(())
    }

    /// Layer `li`'s FFN over the inputs `encode_input` wrote, whose command is done: staged here
    /// if it is not ready, the next layer staged while the ANE runs.
    pub(crate) fn run(&self, c: &Ctx, fbuf: &WeightArena, li: usize, rows: usize) -> Result<(), String> {
        let t0 = std::time::Instant::now();
        self.wait_staging()?;
        let found = {
            let s = self.staged.lock().unwrap();
            s.iter().position(|v| *v == Some(li))
        };
        let set = match found {
            Some(s) => s,
            None => {
                let cmd = c.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                self.encode_weights(enc, fbuf, li, 0);
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                if cmd.status() == metal::MTLCommandBufferStatus::Error {
                    return Err("weights staging failed".into());
                }
                self.staged.lock().unwrap()[0] = Some(li);
                0
            }
        };
        let t1 = std::time::Instant::now();
        // The other set takes the next layer while this one runs.
        let next = (li + 1) % self.layers.len();
        let other = set ^ 1;
        if self.staged.lock().unwrap()[other] != Some(next) {
            self.staged.lock().unwrap()[other] = None;
            let cmd = c.queue.new_command_buffer().to_owned();
            let enc = cmd.new_compute_command_encoder();
            self.encode_weights(enc, fbuf, next, other);
            enc.end_encoding();
            cmd.commit();
            *self.staging.lock().unwrap() = Some((cmd, other, next));
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
            eprintln!("ane-prof: {} evaluations, each: the GPU's part {:.1} ms, waited for weights {:.1} ms, the ANE {:.1} ms", m.3, m.2 / m.3 as f64, m.0 / m.3 as f64, m.1 / m.3 as f64);
        }
        Ok(())
    }

    /// The output of `run` as the residual's delta.
    pub(crate) fn encode_join(&self, enc: &metal::ComputeCommandEncoderRef, d_b: &Buffer, rows: usize) {
        let (hidden, stride) = (self.hidden as u32, (self.out.s.stride / 2) as u32);
        enc.set_compute_pipeline_state(&self.p.join);
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

const WEIGHT_ROWS_U: usize = 8;

struct FfnSurfacesRef<'a> {
    x: &'a [Shared],
    wg: &'a [Shared],
    wu: &'a [Shared],
    wd: &'a [Shared],
    tx: &'a Shared,
    sg: &'a Shared,
    su: &'a Shared,
    sd: &'a Shared,
    out: &'a Shared,
}

fn ffn_function_ref(hidden: usize, inter: usize, rows: usize, v: &FfnSurfacesRef<'_>) -> String {
    // Types only, from surfaces of the same strides: the text names no memory.
    let s = FfnTypes {
        x: v.x.iter().map(|a| &a.s).collect(),
        tx: &v.tx.s,
        wg: v.wg.iter().map(|a| &a.s).collect(),
        wu: v.wu.iter().map(|a| &a.s).collect(),
        wd: v.wd.iter().map(|a| &a.s).collect(),
        sg: &v.sg.s,
        su: &v.su.s,
        sd: &v.sd.s,
        out: &v.out.s,
    };
    ffn_function(hidden, inter, rows, &s)
}

static ANE: Mutex<Option<Arc<AneFfn>>> = Mutex::new(None);
static REFUSED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether a prompt asks for the Neural Engine, which takes larger chunks.
pub(crate) fn wanted() -> bool {
    std::env::var("CMF_ANE").as_deref() == Ok("1")
        && !REFUSED.load(std::sync::atomic::Ordering::Relaxed)
        && api().is_ok()
}

/// The FFN for these layers, built once. `CMF_ANE=1` asks for it; any refusal keeps the GPU's.
pub(crate) fn ffn_for(c: &Ctx, fbuf: &WeightArena, layers: Vec<[Proj; 3]>) -> Option<Arc<AneFfn>> {
    if std::env::var("CMF_ANE").as_deref() != Ok("1") || REFUSED.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    let mut slot = ANE.lock().unwrap();
    if let Some(a) = slot.as_ref() {
        if a.layers.len() == layers.len() && a.layers.first().map(|l| l[0].abs) == layers.first().map(|l| l[0].abs) {
            return Some(a.clone());
        }
    }
    match AneFfn::new(c, fbuf, layers) {
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

    /// How fast the ANE of this Mac runs one FFN of realistic size, on noise.
    /// `cargo test -p cortiq-engine --release --lib ane_ffn_speed -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn ane_ffn_speed() {
        let hidden: usize = std::env::var("ANE_H").ok().and_then(|v| v.parse().ok()).unwrap_or(5120);
        let list = std::env::var("ANE_C").unwrap_or("2560,5120,10240,15360".into());
        let rows: usize = std::env::var("ANE_R").ok().and_then(|v| v.parse().ok()).unwrap_or(256);
        for channels in list.split(',').map(|v| v.parse::<usize>().unwrap()) {
            let s = ffn_surfaces(hidden, channels, rows).expect("surfaces");
            let mut seed = 12345u64;
            let mut fill = |surf: &Surface| {
                let p = surf.base();
                for i in 0..surf.bytes {
                    seed = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
                    unsafe { *p.add(i) = ((seed >> 56) as i8 / 4) as u8 };
                }
            };
            for g in s.x.iter().chain(&s.wg).chain(&s.wu).chain(&s.wd) {
                fill(g);
            }
            let one = half_bits::F16::from_f32(0.1).0;
            for surf in [&s.tx, &s.sg, &s.su, &s.sd] {
                let p = surf.base() as *mut u16;
                for i in 0..surf.bytes / 2 {
                    unsafe { *p.add(i) = one };
                }
            }
            let types = FfnTypes {
                x: s.x.iter().collect(),
                tx: &s.tx,
                wg: s.wg.iter().collect(),
                wu: s.wu.iter().collect(),
                wd: s.wd.iter().collect(),
                sg: &s.sg,
                su: &s.su,
                sd: &s.sd,
                out: &s.out,
            };
            let mil = format!("program(1.3)\n{{\n{}}}\n", ffn_function(hidden, channels, rows, &types));
            let t0 = std::time::Instant::now();
            let prog = match Program::new(&mil, &rotation_blob(channels)) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("channels {channels}: program: {e}");
                    continue;
                }
            };
            let t_load = t0.elapsed().as_secs_f64();
            let mut names: Vec<(String, &Surface)> = Vec::new();
            for (k, x) in s.x.iter().enumerate() {
                names.push((format!("x{k}"), x));
                names.push((format!("wg{k}"), &s.wg[k]));
                names.push((format!("wu{k}"), &s.wu[k]));
            }
            for (i, w) in s.wd.iter().enumerate() {
                names.push((format!("wd{i}"), w));
            }
            names.push(("tx".into(), &s.tx));
            names.push(("sg".into(), &s.sg));
            names.push(("su".into(), &s.su));
            names.push(("sd".into(), &s.sd));
            let refs: Vec<(&str, &Surface)> = names.iter().map(|(n, s)| (n.as_str(), *s)).collect();
            let mut ms = Vec::new();
            for _ in 0..6 {
                let t = std::time::Instant::now();
                if let Err(e) = prog.eval(&format!("ffn{rows}"), &refs, &s.out) {
                    eprintln!("channels {channels}: {e}");
                    break;
                }
                ms.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let finite = {
                let p = s.out.base() as *const u16;
                (0..s.out.bytes / 2).filter(|i| unsafe { *p.add(*i) } & 0x7c00 == 0x7c00).count()
            };
            // The same arithmetic on the host, for a few rows.
            let i8_at = |surf: &Surface, r: usize, c: usize| unsafe { *surf.base().add(r * surf.stride + c) as i8 as f32 } / 128.0;
            let f16_of = |bits: u16| -> f32 {
                let s = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
                let e = ((bits >> 10) & 0x1f) as i32;
                let m = (bits & 0x3ff) as f32;
                if e == 0 { s * m * 2f32.powi(-24) } else { s * (1.0 + m / 1024.0) * 2f32.powi(e - 15) }
            };
            let signs = rotation_signs();
            let norm = 1.0 / (BLOCK as f32).sqrt();
            let mut worst = 0f32;
            let mut rms = (0f64, 0f64);
            for row in [0usize, 7, rows - 1] {
                let segs = hidden / SEGMENT;
                let x: Vec<f32> = (0..hidden).map(|c| i8_at(&s.x[c / SEGMENT], c % SEGMENT, row)).collect();
                let mv = |w: &Vec<Surface>, ch: usize| -> f32 {
                    let mut acc = 0f32;
                    for k in 0..segs {
                        for c in 0..SEGMENT {
                            acc += i8_at(&w[k], ch, c) * x[k * SEGMENT + c];
                        }
                    }
                    acc * 0.1
                };
                let h: Vec<f32> = (0..channels)
                    .map(|ch| {
                        let g = mv(&s.wg, ch) * 0.1 * 0.5;
                        g * (g.tanh() + 1.0) * mv(&s.wu, ch)
                    })
                    .collect();
                let mut hr = vec![0f32; channels];
                for ch in 0..channels {
                    let base = ch / BLOCK * BLOCK;
                    let out = ch % BLOCK;
                    let mut acc = 0f32;
                    for i in 0..BLOCK {
                        let sign = if ((out & i).count_ones() & 1) == 1 { -1.0 } else { 1.0 };
                        acc += sign * h[base + i];
                    }
                    hr[ch] = acc * signs[out] * norm;
                }
                let peak = hr.iter().fold(0f32, |a, v| a.max(v.abs())).max(1.0 / 512.0);
                let hd: Vec<f32> = hr.iter().map(|v| (v * 127.0 / peak).round().clamp(-128.0, 127.0) / 128.0).collect();
                let hscale = peak * 128.0 / 127.0;
                let out_at = |r: usize| unsafe { *((s.out.base().add(r * s.out.stride) as *const u16).add(row)) };
                let got_scale = f16_of(out_at(hidden));
                for r in (0..hidden).step_by(97) {
                    let mut acc = 0f32;
                    for (i, w) in s.wd.iter().enumerate() {
                        let width = SEGMENT.min(channels - i * SEGMENT);
                        for c in 0..width {
                            acc += i8_at(w, r, c) * hd[i * SEGMENT + c];
                        }
                    }
                    let want = acc * 0.1 * hscale;
                    let got = f16_of(out_at(r)) * got_scale;
                    worst = worst.max((want - got).abs());
                    rms.0 += ((want - got) as f64).powi(2);
                    rms.1 += (want as f64).powi(2);
                }
                eprintln!("  row {row}: token scale host {hscale:.4} ane {got_scale:.4}");
            }
            eprintln!("  against the host: max|Δ| {worst:.3e}, relative rms {:.3e}", (rms.0 / rms.1.max(1e-30)).sqrt());
            let macs = (3 * channels * hidden * rows) as f64;
            let best = ms.iter().cloned().fold(f64::MAX, f64::min);
            eprintln!(
                "hidden {hidden} channels {channels} rows {rows}: load {t_load:.1} s, eval {:?} ms, {:.0} GMAC/s, non-finite {finite}",
                ms.iter().map(|m| m.round() as i64).collect::<Vec<_>>(),
                macs / best / 1e6
            );
        }
    }
}

