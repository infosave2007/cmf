//! GPU-visible buffers in plain system memory (Vulkan only).
//!
//! wgpu places every `MAP_WRITE` buffer — including the queue's own
//! `write_buffer` staging — through its allocator's "CPU to GPU" location,
//! which prefers DEVICE_LOCAL | HOST_VISIBLE memory: without resizable BAR
//! that is the 256 MB PCIe window, so the CPU writes every byte across the
//! bus itself (write-combined stores into the card). Measured on an RTX 3090
//! (PCIe 4.0 x16, 256 MB BAR) that path tops out near the rate the prompt
//! frames' expert admissions ran at.
//!
//! A buffer allocated here lives in a HOST_VISIBLE, non-DEVICE_LOCAL memory
//! type (the host heap), is mapped once for its lifetime and can be the
//! source of `copy_buffer_to_buffer`: the card's copy engine then pulls the
//! bytes over PCIe (DMA), and the CPU only ever writes system RAM.

/// A system-memory buffer mapped for its whole life. `ptr` stays valid
/// while `buffer` lives (wgpu frees the memory when the buffer drops).
pub struct SysBuf {
    pub buffer: wgpu::Buffer,
    ptr: *mut u8,
    size: u64,
    /// HOST_CACHED memory (CPU reads are cheap, the card snoops).
    pub cached: bool,
}

// SAFETY: the pointer is to a persistent mapping of device memory owned by
// `buffer`; callers write disjoint ranges.
unsafe impl Send for SysBuf {}
unsafe impl Sync for SysBuf {}

impl SysBuf {
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Raw pointer to the mapping (valid for `size` bytes).
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Copy `src` to `off`. The caller keeps writers to one range apart and
    /// keeps a range from being rewritten while a copy out of it is pending.
    pub fn write(&self, off: u64, src: &[u8]) -> bool {
        if off.checked_add(src.len() as u64).is_none_or(|e| e > self.size) {
            return false;
        }
        // SAFETY: bounds checked above; the mapping is live while self is.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.add(off as usize), src.len());
        }
        true
    }
}

/// Allocate `size` bytes of host-heap memory as a `COPY_SRC | COPY_DST`
/// buffer: HOST_VISIBLE | HOST_COHERENT, with HOST_CACHED when `cached`,
/// never DEVICE_LOCAL. None off Vulkan (and off Linux) or when the driver
/// refuses.
#[cfg(target_os = "linux")]
pub fn sysmem_buffer(device: &wgpu::Device, size: u64, cached: bool) -> Option<SysBuf> {
    use ash::vk;
    if size == 0 || size > device.limits().max_buffer_size {
        return None;
    }
    // SAFETY: plain Vulkan object creation on the device wgpu owns; every
    // failure path releases what was created.
    unsafe {
        let hal = device.as_hal::<wgpu::hal::api::Vulkan>()?;
        let raw = hal.raw_device();
        let props = hal
            .shared_instance()
            .raw_instance()
            .get_physical_device_memory_properties(hal.raw_physical_device());
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buf = raw.create_buffer(&info, None).ok()?;
        let req = raw.get_buffer_memory_requirements(buf);
        let want = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT
            | if cached {
                vk::MemoryPropertyFlags::HOST_CACHED
            } else {
                vk::MemoryPropertyFlags::empty()
            };
        let ty = (0..props.memory_type_count).find(|&i| {
            let f = props.memory_types[i as usize].property_flags;
            req.memory_type_bits & (1 << i) != 0
                && f.contains(want)
                && !f.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                && (cached || !f.contains(vk::MemoryPropertyFlags::HOST_CACHED))
        });
        let Some(ty) = ty else {
            raw.destroy_buffer(buf, None);
            return None;
        };
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(ty);
        let Ok(mem) = raw.allocate_memory(&alloc, None) else {
            raw.destroy_buffer(buf, None);
            return None;
        };
        if raw.bind_buffer_memory(buf, mem, 0).is_err() {
            raw.destroy_buffer(buf, None);
            raw.free_memory(mem, None);
            return None;
        }
        let Ok(ptr) = raw.map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) else {
            raw.destroy_buffer(buf, None);
            raw.free_memory(mem, None);
            return None;
        };
        let hb = wgpu::hal::vulkan::Buffer::from_raw_managed(buf, mem, 0, req.size);
        drop(hal);
        let buffer = device.create_buffer_from_hal::<wgpu::hal::api::Vulkan>(
            hb,
            &wgpu::BufferDescriptor {
                label: Some("cortiq-sysmem"),
                size,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        );
        Some(SysBuf {
            buffer,
            ptr: ptr as *mut u8,
            size,
            cached,
        })
    }
}

#[cfg(not(target_os = "linux"))]
pub fn sysmem_buffer(_device: &wgpu::Device, _size: u64, _cached: bool) -> Option<SysBuf> {
    None
}

/// The card's CPU-writable VRAM window is the legacy BAR (at most 512 MiB:
/// no resizable BAR), so every `write_buffer` byte crosses PCIe as a CPU
/// store into that window. False off Vulkan or when there is no such type.
#[cfg(target_os = "linux")]
pub fn small_bar(device: &wgpu::Device) -> bool {
    use ash::vk;
    // SAFETY: a read-only query on the physical device wgpu owns.
    unsafe {
        let Some(hal) = device.as_hal::<wgpu::hal::api::Vulkan>() else {
            return false;
        };
        let props = hal
            .shared_instance()
            .raw_instance()
            .get_physical_device_memory_properties(hal.raw_physical_device());
        let both = vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE;
        let window = (0..props.memory_type_count as usize)
            .filter(|&i| props.memory_types[i].property_flags.contains(both))
            .map(|i| props.memory_heaps[props.memory_types[i].heap_index as usize].size)
            .max();
        window.is_some_and(|w| w <= 512 << 20)
    }
}

#[cfg(not(target_os = "linux"))]
pub fn small_bar(_device: &wgpu::Device) -> bool {
    false
}
