//! Apple silicon: a MoE layer's routed experts read in place from the file's
//! mapping. The GPU's memory is the host's, so a bank needs no copy: one
//! no-copy Metal buffer over the layer's experts (page-aligned around them),
//! the gate, up and down planes bound at their offsets in it, an expert the
//! file's stride on (`expert_stride`), the kernels unchanged. The OS keeps
//! what fits of the file in memory and reads the rest from the disk; there is
//! no stream, no slot and no wait for the picks.
//!
//! Measured on a MacBook Air M4 (24 GB) over Maxwell-V1-Flash's 4-bit file:
//! 42 such banks (17.7 GB of the file) bound in one command buffer, 504
//! random experts a "token" read back exact at 100 GB/s in a process of its
//! own; two 13.6 GB windows over the whole file instead made Metal drop the
//! command buffer without an error. Under the model's graphs the banks do not
//! stay in memory there (see `on`).

use super::*;
use std::sync::{Arc, Mutex, OnceLock};

/// The VM page of Apple silicon: a no-copy buffer starts and ends on one.
const PAGE: usize = 16384;

/// `CMF_WGPU_NOCOPY=1`: banks read in place. Off by default: Metal makes a
/// command buffer's buffers resident whole, so every bank of a token is in
/// memory at once — on a 24 GB M4 with the 4-bit Maxwell file (17.7 GB of
/// banks) the page cache held 10-12 GB of it during a run and each token
/// re-read banks from the disk (minutes for 57 tokens; a 128-token run
/// swapped ~50 GB). A machine whose memory holds the banks beside the system
/// is where it pays (unmeasured here).
pub(crate) fn on(c: &Ctx) -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    !c.discrete && *ON.get_or_init(|| std::env::var("CMF_WGPU_NOCOPY").as_deref() == Ok("1"))
}

/// Where a layer's experts lie in the mapping: the page-aligned span around
/// them, each plane's first expert in it, the stride from one expert to the
/// next (the same for the three planes).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Layout {
    pub start: usize,
    pub len: usize,
    pub off: [u64; 3],
    pub stride: u64,
}

/// The layout of `experts` (gate, up, down tensor indices) when they are
/// evenly spaced in the file's primary mapping with equal lengths and
/// offsets the binding alignment admits; None otherwise.
pub(crate) fn layout(model: &CmfModel, experts: &[(usize, usize, usize)], align: u64) -> Option<Layout> {
    if experts.len() < 2 {
        return None;
    }
    let primary = model.primary_bytes();
    let (lo, hi) = (primary.as_ptr() as usize, primary.as_ptr() as usize + primary.len());
    let at = |i: usize| -> Option<(usize, usize)> {
        let e = model.tensors.get(i)?;
        if e.shard != 0 {
            return None;
        }
        let b = model.entry_bytes(e);
        let p = b.as_ptr() as usize;
        (p >= lo && p + b.len() <= hi && !b.is_empty()).then_some((p, b.len()))
    };
    let (g0, gl) = at(experts[0].0)?;
    let (u0, ul) = at(experts[0].1)?;
    let (d0, dl) = at(experts[0].2)?;
    let (g1, _) = at(experts[1].0)?;
    let stride = g1.checked_sub(g0)?;
    if gl != ul || stride == 0 || stride % 16 != 0 {
        return None;
    }
    for (e, &(g, u, d)) in experts.iter().enumerate() {
        let want = e * stride;
        let ok = at(g) == Some((g0 + want, gl)) && at(u) == Some((u0 + want, ul)) && at(d) == Some((d0 + want, dl));
        if !ok {
            return None;
        }
    }
    let first = g0.min(u0).min(d0);
    let last = experts.len() - 1;
    let end = (g0 + last * stride + gl).max(u0 + last * stride + ul).max(d0 + last * stride + dl);
    let start = first & !(PAGE - 1);
    let len = (end - start).next_multiple_of(PAGE);
    if start < lo & !(PAGE - 1) || start + len > (hi + PAGE - 1) & !(PAGE - 1) {
        return None;
    }
    let off = [(g0 - start) as u64, (u0 - start) as u64, (d0 - start) as u64];
    if off.iter().any(|&o| o % align.max(4) != 0) {
        return None;
    }
    Some(Layout { start, len, off, stride: stride as u64 })
}

/// A layer's bank: the no-copy buffer and its layout.
pub(crate) struct Bank {
    pub buf: wgpu::Buffer,
    pub lay: Layout,
}

/// The bank of a layer (cached a model and layer, by its first gate tensor).
pub(crate) fn bank(c: &Ctx, model: &CmfModel, experts: &[(usize, usize, usize)]) -> Option<Arc<Bank>> {
    static CACHE: OnceLock<Mutex<HashMap<(u64, usize), Option<Arc<Bank>>>>> = OnceLock::new();
    let key = (model.uid(), experts.first()?.0);
    let cache = CACHE.get_or_init(Default::default);
    if let Some(b) = cache.lock().unwrap().get(&key) {
        return b.clone();
    }
    let align = c.device.limits().min_storage_buffer_offset_alignment as u64;
    let made = layout(model, experts, align).and_then(|lay| make(c, lay).map(|buf| Arc::new(Bank { buf, lay })));
    cache.lock().unwrap().insert(key, made.clone());
    made
}

/// The wgpu buffer over `lay`'s span of the mapping (no copy).
fn make(c: &Ctx, lay: Layout) -> Option<wgpu::Buffer> {
    use metal::foreign_types::ForeignTypeRef;
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    if lay.len as u64 > c.device.limits().max_buffer_size {
        return None;
    }
    // SAFETY: the device is wgpu's own Metal device; the span lies inside the
    // model's read-only mapping, which outlives the buffer (the model is held
    // by the graphs' caller for as long as its banks are cached); the buffer
    // is bound read-only (STORAGE, never written).
    unsafe {
        let hal = c.device.as_hal::<wgpu::hal::api::Metal>()?;
        let raw: *const ProtocolObject<dyn objc2_metal::MTLDevice> = Retained::as_ptr(hal.raw_device());
        let dev = metal::DeviceRef::from_ptr(raw as *mut metal::MTLDevice);
        let mb = dev.new_buffer_with_bytes_no_copy(
            lay.start as *const std::ffi::c_void,
            lay.len as u64,
            metal::MTLResourceOptions::StorageModeShared,
            None,
        );
        let ptr = metal::foreign_types::ForeignType::as_ptr(&mb) as *mut ProtocolObject<dyn objc2_metal::MTLBuffer>;
        let raw_buf = Retained::retain(ptr)?;
        drop(mb);
        let hb = wgpu::hal::metal::Device::buffer_from_raw(raw_buf, lay.len as u64);
        Some(c.device.create_buffer_from_hal::<wgpu::hal::api::Metal>(
            hb,
            &wgpu::BufferDescriptor {
                label: Some("moe-nocopy-bank"),
                size: lay.len as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            },
        ))
    }
}
