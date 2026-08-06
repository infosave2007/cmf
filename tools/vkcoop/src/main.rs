use ash::vk;
use std::time::Instant;

const ROWS: usize = 9216;
const COLS: usize = 2304;
const NB: usize = 2085;

unsafe fn find_mem(
    inst: &ash::Instance,
    pd: vk::PhysicalDevice,
    bits: u32,
    want: vk::MemoryPropertyFlags,
) -> u32 {
    let mp = inst.get_physical_device_memory_properties(pd);
    for i in 0..mp.memory_type_count {
        if bits & (1 << i) != 0 && mp.memory_types[i as usize].property_flags.contains(want) {
            return i;
        }
    }
    panic!("no memory type");
}

struct Buf {
    b: vk::Buffer,
    m: vk::DeviceMemory,
}

unsafe fn mk_buf(
    inst: &ash::Instance,
    pd: vk::PhysicalDevice,
    dev: &ash::Device,
    size: u64,
    usage: vk::BufferUsageFlags,
    props: vk::MemoryPropertyFlags,
) -> Buf {
    let b = dev
        .create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
        .unwrap();
    let req = dev.get_buffer_memory_requirements(b);
    let mi = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(find_mem(inst, pd, req.memory_type_bits, props));
    let m = dev.allocate_memory(&mi, None).unwrap();
    dev.bind_buffer_memory(b, m, 0).unwrap();
    Buf { b, m }
}

fn main() {
    unsafe {
        let entry = ash::Entry::load().unwrap();
        let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 3, 0));
        let inst = entry
            .create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None)
            .unwrap();
        let pd = inst
            .enumerate_physical_devices()
            .unwrap()
            .into_iter()
            .find(|&d| {
                inst.get_physical_device_properties(d).device_type
                    == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .expect("no discrete gpu");
        let qf = inst
            .get_physical_device_queue_family_properties(pd)
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .unwrap() as u32;

        let ext = [c"VK_KHR_cooperative_matrix".as_ptr()];
        let mut f_coop =
            vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
        let mut f_f16 = vk::PhysicalDeviceShaderFloat16Int8Features::default()
            .shader_float16(true)
            .shader_int8(true);
        let mut f_16bit = vk::PhysicalDevice16BitStorageFeatures::default()
            .storage_buffer16_bit_access(true);
        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qf)
            .queue_priorities(&prio)];
        let dci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qci)
            .enabled_extension_names(&ext)
            .push_next(&mut f_coop)
            .push_next(&mut f_f16)
            .push_next(&mut f_16bit);
        let dev = inst.create_device(pd, &dci, None).unwrap();
        let queue = dev.get_device_queue(qf, 0);

        // buffers
        let gpr = COLS / 32;
        let wbytes = ROWS * gpr * 16 + ROWS * 4 + ROWS * ((gpr * 5 + 7) / 8);
        let wbytes = (wbytes + 3) & !3;
        let xbytes = NB * COLS * 4;
        let ybytes = NB * ROWS * 4;
        let dl = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let hv = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let su = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST;
        let wb = mk_buf(&inst, pd, &dev, wbytes as u64, su, dl);
        let xb = mk_buf(&inst, pd, &dev, xbytes as u64, su, dl);
        let yb = mk_buf(&inst, pd, &dev, ybytes as u64, vk::BufferUsageFlags::STORAGE_BUFFER, dl);
        let pb = mk_buf(
            &inst, pd, &dev, 16,
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, dl,
        );
        let stage = mk_buf(
            &inst, pd, &dev,
            wbytes.max(xbytes) as u64,
            vk::BufferUsageFlags::TRANSFER_SRC, hv,
        );

        // synthetic weights: valid f16 row params so the scales are finite
        let mut w = vec![0u8; wbytes];
        for (i, v) in w.iter_mut().enumerate() {
            *v = (i * 37 % 251) as u8;
        }
        let params_off = ROWS * gpr * 16;
        let lo = half_bits(-4.0);
        let step = half_bits(0.1);
        for r in 0..ROWS {
            let o = params_off + r * 4;
            w[o..o + 2].copy_from_slice(&lo.to_le_bytes());
            w[o + 2..o + 4].copy_from_slice(&step.to_le_bytes());
        }
        let xs: Vec<f32> = (0..NB * COLS).map(|i| ((i % 97) as f32 - 48.0) / 48.0).collect();

        let cmd_pool = dev
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(qf),
                None,
            )
            .unwrap();
        let cb = dev
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .unwrap()[0];

        let upload = |data: &[u8], dst: vk::Buffer, size: u64| {
            let p = dev
                .map_memory(stage.m, 0, size, vk::MemoryMapFlags::empty())
                .unwrap() as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), p, size as usize);
            dev.unmap_memory(stage.m);
            dev.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();
            dev.cmd_copy_buffer(cb, stage.b, dst, &[vk::BufferCopy::default().size(size)]);
            dev.end_command_buffer(cb).unwrap();
            let cbs = [cb];
            dev.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&cbs)], vk::Fence::null())
                .unwrap();
            dev.device_wait_idle().unwrap();
        };
        upload(&w, wb.b, wbytes as u64);
        upload(bytemuck_cast(&xs), xb.b, xbytes as u64);
        let p_words = [(COLS / 4) as u32, ROWS as u32, NB as u32, 0u32];
        upload(bytemuck_cast(&p_words), pb.b, 16);

        // descriptors
        let binds = [
            dsl(0, vk::DescriptorType::STORAGE_BUFFER),
            dsl(1, vk::DescriptorType::STORAGE_BUFFER),
            dsl(2, vk::DescriptorType::STORAGE_BUFFER),
            dsl(3, vk::DescriptorType::UNIFORM_BUFFER),
        ];
        let set_layout = dev
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binds),
                None,
            )
            .unwrap();
        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(3),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
        ];
        let pool = dev
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes),
                None,
            )
            .unwrap();
        let layouts = [set_layout];
        let set = dev
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
            .unwrap()[0];
        let bi = |b: vk::Buffer, sz: u64| [vk::DescriptorBufferInfo::default().buffer(b).offset(0).range(sz)];
        let (i0, i1, i2, i3) = (
            bi(wb.b, wbytes as u64),
            bi(xb.b, xbytes as u64),
            bi(yb.b, ybytes as u64),
            bi(pb.b, 16),
        );
        dev.update_descriptor_sets(
            &[
                wds(set, 0, vk::DescriptorType::STORAGE_BUFFER, &i0),
                wds(set, 1, vk::DescriptorType::STORAGE_BUFFER, &i1),
                wds(set, 2, vk::DescriptorType::STORAGE_BUFFER, &i2),
                wds(set, 3, vk::DescriptorType::UNIFORM_BUFFER, &i3),
            ],
            &[],
        );

        let spv = std::fs::read("/root/vkprobe/q4tp_coop.spv").unwrap();
        let words: Vec<u32> = spv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let module = dev
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            .unwrap();
        let pl = dev
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                None,
            )
            .unwrap();
        let stage_ci = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let pipe = dev
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default().stage(stage_ci).layout(pl)],
                None,
            )
            .unwrap()[0];

        let reps = 20u32;
        let run = || {
            dev.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe);
            dev.cmd_bind_descriptor_sets(
                cb, vk::PipelineBindPoint::COMPUTE, pl, 0, &[set], &[],
            );
            for _ in 0..reps {
                dev.cmd_dispatch(cb, (ROWS as u32).div_ceil(64), (NB as u32).div_ceil(64), 1);
            }
            dev.end_command_buffer(cb).unwrap();
            let cbs = [cb];
            dev.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&cbs)], vk::Fence::null())
                .unwrap();
            dev.device_wait_idle().unwrap();
        };
        run();
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            run();
            best = best.min(t.elapsed().as_secs_f64());
        }
        let flops = 2.0 * ROWS as f64 * COLS as f64 * NB as f64 * reps as f64;
        println!(
            "vulkan coop q4tp {ROWS}x{COLS} n={NB}: {:.2} ms/call  {:.0} GFLOP/s",
            best * 1e3 / reps as f64,
            flops / best / 1e9
        );
    }
}

fn dsl(b: u32, t: vk::DescriptorType) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(b)
        .descriptor_type(t)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
}

fn wds<'a>(
    set: vk::DescriptorSet,
    b: u32,
    t: vk::DescriptorType,
    info: &'a [vk::DescriptorBufferInfo],
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(b)
        .descriptor_type(t)
        .buffer_info(info)
}

fn half_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    let m = (b & 0x7FFFFF) >> 13;
    if e <= 0 { s } else { s | ((e as u16) << 10) | m as u16 }
}

fn bytemuck_cast<T>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
