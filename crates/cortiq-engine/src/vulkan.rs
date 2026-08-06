//! A native Vulkan compute lane, for the one thing wgpu cannot reach.
//!
//! The card's matrix units — tensor cores on NVIDIA — are exposed through
//! `VK_KHR_cooperative_matrix`, and the shape they implement is f16 16x16
//! with an f32 accumulator. Measured on the same card, same GEMM, same
//! shape, against the WGSL kernel the engine uses today:
//!
//! ```text
//! wgpu, scalar fp32          25 082 GFLOP/s    3.53 ms
//! this, cooperative f16      62 776 GFLOP/s    1.41 ms
//! ```
//!
//! Against the format's scalar dequant the kernel lands within 9e-5 of the
//! row's magnitude — f16 operands, f32 accumulator. `tests/vk_coop.rs`
//! holds it there, and holds it at tile edges: the first version of this
//! kernel stored its f32 accumulators into an f16 plane, wrote nothing
//! usable, and measured 81 315 GFLOP/s for it.
//!
//! This is an accelerator and never a requirement. Everything here sits
//! behind a capability probe, the way the CPU kernels sit behind AVX-512
//! VNNI or ARM's dotprod: no extension, no device, no matching driver —
//! and the wgpu path runs exactly as before. `ash` links nothing; it opens
//! `libvulkan.so.1` at run time, which is what wgpu already does, so the
//! binary's dependencies do not change.
//!
//! `CMF_VK=0` turns the lane off.

use ash::vk;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Compiled ahead of time by glslang and carried in the binary: a
/// pure-Rust install cannot ask the user for a C++ toolchain. Rebuild with
/// `tools/vkcoop/build.sh` after touching the source next to it.
const Q4TP_COOP_SPV: &[u8] = include_bytes!("shaders/q4tp_coop.spv");

struct Buf {
    b: vk::Buffer,
    m: vk::DeviceMemory,
    size: u64,
}

pub struct Ctx {
    _entry: ash::Entry,
    instance: ash::Instance,
    pd: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    set_layout: vk::DescriptorSetLayout,
    desc_pool: vk::DescriptorPool,
    pipe_layout: vk::PipelineLayout,
    q4tp_mm: vk::Pipeline,
    /// Weights already on the device, keyed as in the wgpu cache.
    weights: Mutex<HashMap<(usize, usize), Buf>>,
    /// Growable scratch: activations in, results out, staging between.
    scratch: Mutex<Scratch>,
}

#[derive(Default)]
struct Scratch {
    xs: Option<Buf>,
    ys: Option<Buf>,
    stage: Option<Buf>,
    params: Option<Buf>,
}

// The handles are opaque; every use goes through `&Ctx` and the queue is
// touched under the scratch lock.
unsafe impl Send for Ctx {}
unsafe impl Sync for Ctx {}

static CTX: OnceLock<Option<Ctx>> = OnceLock::new();

/// Is the fast lane up? False on any machine without the extension, which
/// is the normal case and costs nothing: the wgpu path is unchanged.
pub fn available() -> bool {
    ctx().is_some()
}

fn ctx() -> Option<&'static Ctx> {
    CTX.get_or_init(|| {
        if std::env::var("CMF_VK").is_ok_and(|v| v == "0") {
            return None;
        }
        match unsafe { init() } {
            Ok(c) => {
                tracing::info!("vulkan cooperative-matrix lane: on");
                Some(c)
            }
            Err(e) => {
                tracing::debug!("vulkan lane off: {e}");
                None
            }
        }
    })
    .as_ref()
}

/// Does this device implement the f16 16x16 shape with an f32 accumulator?
/// Anything else — and the driver here reports six other configurations —
/// is not what the kernel below was written against.
unsafe fn has_coop_f16(entry: &ash::Entry, inst: &ash::Instance, pd: vk::PhysicalDevice) -> bool {
    unsafe {
        let khr = ash::khr::cooperative_matrix::Instance::new(entry, inst);
        let Ok(list) = khr.get_physical_device_cooperative_matrix_properties(pd) else {
            return false;
        };
        list.iter().any(|c| {
            c.m_size == 16
                && c.n_size == 16
                && c.k_size == 16
                && c.a_type == vk::ComponentTypeKHR::FLOAT16
                && c.b_type == vk::ComponentTypeKHR::FLOAT16
                && c.c_type == vk::ComponentTypeKHR::FLOAT32
                && c.result_type == vk::ComponentTypeKHR::FLOAT32
                && c.scope == vk::ScopeKHR::SUBGROUP
        })
    }
}

unsafe fn init() -> Result<Ctx, String> {
    unsafe {
        let entry = ash::Entry::load().map_err(|e| format!("no vulkan loader: {e}"))?;
        let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 3, 0));
        let instance = entry
            .create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
            .map_err(|e| format!("instance: {e}"))?;
        let pds = instance
            .enumerate_physical_devices()
            .map_err(|e| format!("devices: {e}"))?;
        // A discrete card that has the shape, else any card that has it.
        let pick = pds
            .iter()
            .find(|&&d| {
                instance.get_physical_device_properties(d).device_type
                    == vk::PhysicalDeviceType::DISCRETE_GPU
                    && has_coop_f16(&entry, &instance, d)
            })
            .or_else(|| pds.iter().find(|&&d| has_coop_f16(&entry, &instance, d)))
            .copied();
        let Some(pd) = pick else {
            instance.destroy_instance(None);
            return Err("no device with f16 16x16 cooperative matrices".into());
        };
        let qf = instance
            .get_physical_device_queue_family_properties(pd)
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or("no compute queue")? as u32;

        let ext = [c"VK_KHR_cooperative_matrix".as_ptr()];
        let mut f_coop =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        let mut f_f16 = vk::PhysicalDeviceShaderFloat16Int8Features::default()
            .shader_float16(true)
            .shader_int8(true);
        let mut f_16bit =
            vk::PhysicalDevice16BitStorageFeatures::default().storage_buffer16_bit_access(true);
        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qf)
            .queue_priorities(&prio)];
        let device = instance
            .create_device(
                pd,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&qci)
                    .enabled_extension_names(&ext)
                    .push_next(&mut f_coop)
                    .push_next(&mut f_f16)
                    .push_next(&mut f_16bit),
                None,
            )
            .map_err(|e| format!("device: {e}"))?;
        let queue = device.get_device_queue(qf, 0);
        let cmd_pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(qf)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .map_err(|e| format!("command pool: {e}"))?;
        let cmd = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .map_err(|e| format!("command buffer: {e}"))?[0];

        let binds = [
            binding(0, vk::DescriptorType::STORAGE_BUFFER),
            binding(1, vk::DescriptorType::STORAGE_BUFFER),
            binding(2, vk::DescriptorType::STORAGE_BUFFER),
            binding(3, vk::DescriptorType::UNIFORM_BUFFER),
        ];
        let set_layout = device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binds),
                None,
            )
            .map_err(|e| format!("set layout: {e}"))?;
        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(3 * 64),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(64),
        ];
        let desc_pool = device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(64)
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .pool_sizes(&sizes),
                None,
            )
            .map_err(|e| format!("descriptor pool: {e}"))?;
        let layouts = [set_layout];
        let pipe_layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                None,
            )
            .map_err(|e| format!("pipeline layout: {e}"))?;

        let words: Vec<u32> = Q4TP_COOP_SPV
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let module = device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            .map_err(|e| format!("shader module: {e}"))?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let q4tp_mm = device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(pipe_layout)],
                None,
            )
            .map_err(|(_, e)| format!("pipeline: {e}"))?[0];
        device.destroy_shader_module(module, None);

        Ok(Ctx {
            _entry: entry,
            instance,
            pd,
            device,
            queue,
            cmd_pool,
            cmd,
            set_layout,
            desc_pool,
            pipe_layout,
            q4tp_mm,
            weights: Mutex::new(HashMap::new()),
            scratch: Mutex::new(Scratch::default()),
        })
    }
}

fn binding(b: u32, t: vk::DescriptorType) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(b)
        .descriptor_type(t)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
}

impl Ctx {
    unsafe fn mem_type(&self, bits: u32, want: vk::MemoryPropertyFlags) -> Option<u32> {
        unsafe {
            let mp = self.instance.get_physical_device_memory_properties(self.pd);
            (0..mp.memory_type_count).find(|&i| {
                bits & (1 << i) != 0
                    && mp.memory_types[i as usize]
                        .property_flags
                        .contains(want)
            })
        }
    }

    unsafe fn alloc(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        props: vk::MemoryPropertyFlags,
    ) -> Option<Buf> {
        unsafe {
            let b = self
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size.max(4))
                        .usage(usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .ok()?;
            let req = self.device.get_buffer_memory_requirements(b);
            let idx = self.mem_type(req.memory_type_bits, props)?;
            let m = self
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(idx),
                    None,
                )
                .ok()?;
            self.device.bind_buffer_memory(b, m, 0).ok()?;
            Some(Buf { b, m, size })
        }
    }

    /// Grow-only scratch: a render asks for the same shapes step after
    /// step, so this settles after the first block.
    unsafe fn ensure(
        &self,
        slot: &mut Option<Buf>,
        size: u64,
        usage: vk::BufferUsageFlags,
        props: vk::MemoryPropertyFlags,
    ) -> Option<vk::Buffer> {
        unsafe {
            if let Some(b) = slot.as_ref() {
                if b.size >= size {
                    return Some(b.b);
                }
                self.device.destroy_buffer(b.b, None);
                self.device.free_memory(b.m, None);
            }
            let nb = self.alloc(size, usage, props)?;
            let h = nb.b;
            *slot = Some(nb);
            Some(h)
        }
    }

    unsafe fn run_once(&self, record: impl FnOnce(vk::CommandBuffer)) -> bool {
        unsafe {
            let d = &self.device;
            if d.begin_command_buffer(
                self.cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .is_err()
            {
                return false;
            }
            record(self.cmd);
            if d.end_command_buffer(self.cmd).is_err() {
                return false;
            }
            let cbs = [self.cmd];
            let si = [vk::SubmitInfo::default().command_buffers(&cbs)];
            if d.queue_submit(self.queue, &si, vk::Fence::null()).is_err() {
                return false;
            }
            d.device_wait_idle().is_ok()
        }
    }

    /// Upload host bytes into a device-local buffer through the staging
    /// slot. Used for weights (once per tensor) and activations (per call).
    unsafe fn upload(&self, sc: &mut Scratch, data: &[u8], dst: vk::Buffer) -> bool {
        unsafe {
            let Some(stage) = self.ensure(
                &mut sc.stage,
                data.len() as u64,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ) else {
                return false;
            };
            let mem = sc.stage.as_ref().unwrap().m;
            let Ok(p) = self
                .device
                .map_memory(mem, 0, data.len() as u64, vk::MemoryMapFlags::empty())
            else {
                return false;
            };
            std::ptr::copy_nonoverlapping(data.as_ptr(), p as *mut u8, data.len());
            self.device.unmap_memory(mem);
            self.run_once(|cb| {
                self.device.cmd_copy_buffer(
                    cb,
                    stage,
                    dst,
                    &[vk::BufferCopy::default().size(data.len() as u64)],
                );
            })
        }
    }
}

/// The batched q4tp GEMM on the matrix units. `key` identifies the weight
/// tensor so it is uploaded once and stays; everything else rides the
/// per-call scratch. Returns false if anything is missing, and the caller
/// then runs whatever it would have run before.
pub fn q4tp_matmat(
    key: (usize, usize),
    weights: &[u8],
    xs: &[f32],
    b: usize,
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> bool {
    let Some(c) = ctx() else { return false };
    if cols % 32 != 0 || rows == 0 || b == 0 || xs.len() < b * cols || out.len() < b * rows {
        return false;
    }
    unsafe {
        let d = &c.device;
        let mut sc = c.scratch.lock().unwrap();
        // Weights, once.
        {
            let mut w = c.weights.lock().unwrap();
            if !w.contains_key(&key) {
                let Some(buf) = c.alloc(
                    weights.len() as u64,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                ) else {
                    return false;
                };
                if !c.upload(&mut sc, weights, buf.b) {
                    d.destroy_buffer(buf.b, None);
                    d.free_memory(buf.m, None);
                    return false;
                }
                w.insert(key, buf);
            }
        }
        let wbuf = c.weights.lock().unwrap().get(&key).map(|b| b.b);
        let Some(wbuf) = wbuf else { return false };

        let xbytes = (b * cols * 4) as u64;
        let ybytes = (b * rows * 4) as u64;
        let Some(xbuf) = c.ensure(
            &mut sc.xs,
            xbytes,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ) else {
            return false;
        };
        let Some(ybuf) = c.ensure(
            &mut sc.ys,
            ybytes,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ) else {
            return false;
        };
        let Some(pbuf) = c.ensure(
            &mut sc.params,
            16,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ) else {
            return false;
        };
        let xb: &[u8] =
            std::slice::from_raw_parts(xs.as_ptr() as *const u8, b * cols * 4);
        if !c.upload(&mut sc, xb, xbuf) {
            return false;
        }
        let p = [(cols / 4) as u32, rows as u32, b as u32, 0u32];
        let pb: &[u8] = std::slice::from_raw_parts(p.as_ptr() as *const u8, 16);
        if !c.upload(&mut sc, pb, pbuf) {
            return false;
        }

        let layouts = [c.set_layout];
        let Ok(sets) = d.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(c.desc_pool)
                .set_layouts(&layouts),
        ) else {
            return false;
        };
        let set = sets[0];
        let i0 = [vk::DescriptorBufferInfo::default()
            .buffer(wbuf)
            .range(weights.len() as u64)];
        let i1 = [vk::DescriptorBufferInfo::default()
            .buffer(xbuf)
            .range(xbytes)];
        let i2 = [vk::DescriptorBufferInfo::default()
            .buffer(ybuf)
            .range(ybytes)];
        let i3 = [vk::DescriptorBufferInfo::default().buffer(pbuf).range(16)];
        fn w<'a>(
            set: vk::DescriptorSet,
            bind: u32,
            t: vk::DescriptorType,
            info: &'a [vk::DescriptorBufferInfo],
        ) -> vk::WriteDescriptorSet<'a> {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(bind)
                .descriptor_type(t)
                .buffer_info(info)
        }
        d.update_descriptor_sets(
            &[
                w(set, 0, vk::DescriptorType::STORAGE_BUFFER, &i0),
                w(set, 1, vk::DescriptorType::STORAGE_BUFFER, &i1),
                w(set, 2, vk::DescriptorType::STORAGE_BUFFER, &i2),
                w(set, 3, vk::DescriptorType::UNIFORM_BUFFER, &i3),
            ],
            &[],
        );
        let ok = c.run_once(|cb| {
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, c.q4tp_mm);
            d.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                c.pipe_layout,
                0,
                &[set],
                &[],
            );
            d.cmd_dispatch(cb, (rows as u32).div_ceil(64), (b as u32).div_ceil(64), 1);
        });
        let _ = d.free_descriptor_sets(c.desc_pool, &[set]);
        if !ok {
            return false;
        }

        // Results back through the staging slot.
        let Some(stage) = c.ensure(
            &mut sc.stage,
            ybytes,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) else {
            return false;
        };
        if !c.run_once(|cb| {
            d.cmd_copy_buffer(
                cb,
                ybuf,
                stage,
                &[vk::BufferCopy::default().size(ybytes)],
            );
        }) {
            return false;
        }
        let mem = sc.stage.as_ref().unwrap().m;
        let Ok(p) = d.map_memory(mem, 0, ybytes, vk::MemoryMapFlags::empty()) else {
            return false;
        };
        std::ptr::copy_nonoverlapping(p as *const f32, out.as_mut_ptr(), b * rows);
        d.unmap_memory(mem);
        true
    }
}
