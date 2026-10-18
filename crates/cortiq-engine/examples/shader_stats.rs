//! Driver statistics (registers, shared memory, …) of the trainer's
//! generated compute kernels, through VK_KHR_pipeline_executable_properties
//! on a device of our own (wgpu does not ask for capture):
//!
//! ```text
//! cargo run --release -p cortiq-engine --features gpu --example shader_stats [filter]
//! ```

#[cfg(all(feature = "gpu", target_os = "linux"))]
mod imp {
    use ash::vk;
    use std::ffi::CStr;

    fn spirv(src: &str, entry: &str) -> Result<Vec<u32>, String> {
        let module = naga::front::wgsl::parse_str(src).map_err(|e| e.emit_to_string(src))?;
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .map_err(|e| format!("{e:?}"))?;
        let mut opts = naga::back::spv::Options { lang_version: (1, 3), ..Default::default() };
        opts.bounds_check_policies = naga::proc::BoundsCheckPolicies {
            index: naga::proc::BoundsCheckPolicy::Unchecked,
            buffer: naga::proc::BoundsCheckPolicy::Unchecked,
            image_load: naga::proc::BoundsCheckPolicy::Unchecked,
            binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
        };
        opts.flags.remove(naga::back::spv::WriterFlags::DEBUG);
        let po = naga::back::spv::PipelineOptions { shader_stage: naga::ShaderStage::Compute, entry_point: entry.into() };
        let words = naga::back::spv::write_vec(&module, &info, &opts, Some(&po)).map_err(|e| format!("{e:?}"))?;
        Ok(words)
    }

    /// Bindings of group 0 used by `entry` (uniform or storage), from naga.
    fn bindings(src: &str) -> Vec<(u32, vk::DescriptorType)> {
        let module = naga::front::wgsl::parse_str(src).unwrap();
        let mut v = Vec::new();
        for (_, g) in module.global_variables.iter() {
            if let Some(b) = &g.binding {
                let t = match g.space {
                    naga::AddressSpace::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
                    naga::AddressSpace::Storage { .. } => vk::DescriptorType::STORAGE_BUFFER,
                    _ => continue,
                };
                if b.group == 0 {
                    v.push((b.binding, t));
                }
            }
        }
        v.sort();
        v.dedup_by_key(|x| x.0);
        v
    }

    pub fn main() {
        let filter = std::env::args().nth(1).unwrap_or_default();
        let mut srcs: Vec<(&str, String, Vec<&str>)> = cortiq_engine::gpu_wgpu::gdn_train::debug_sources();
        srcs.retain(|_| true);
        unsafe {
            let entry = ash::Entry::load().expect("libvulkan");
            let app = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 3, 0));
            let inst = entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None).expect("instance");
            let pd = inst.enumerate_physical_devices().unwrap()[0];
            let props = inst.get_physical_device_properties(pd);
            println!("device: {:?}", CStr::from_ptr(props.device_name.as_ptr()));
            let qf = inst
                .get_physical_device_queue_family_properties(pd)
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .unwrap() as u32;
            let prio = [1.0f32];
            let qci = [vk::DeviceQueueCreateInfo::default().queue_family_index(qf).queue_priorities(&prio)];
            let exts = [ash::khr::pipeline_executable_properties::NAME.as_ptr()];
            let mut pef = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default().pipeline_executable_info(true);
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default().shader_float16(true).vulkan_memory_model(true).vulkan_memory_model_device_scope(true);
            let mut f11 = vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
            let mut f13 = vk::PhysicalDeviceVulkan13Features::default().subgroup_size_control(true);
            let dci = vk::DeviceCreateInfo::default()
                .queue_create_infos(&qci)
                .enabled_extension_names(&exts)
                .push_next(&mut pef)
                .push_next(&mut f12)
                .push_next(&mut f11)
                .push_next(&mut f13);
            let dev = inst.create_device(pd, &dci, None).expect("device");
            let pep = ash::khr::pipeline_executable_properties::Device::new(&inst, &dev);
            for (name, src, eps) in &srcs {
                let binds = bindings(src);
                let lb: Vec<vk::DescriptorSetLayoutBinding> = binds
                    .iter()
                    .map(|&(b, t)| vk::DescriptorSetLayoutBinding::default().binding(b).descriptor_type(t).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE))
                    .collect();
                let dsl = dev.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&lb), None).unwrap();
                let sets = [dsl];
                let pl = dev.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sets), None).unwrap();
                for ep in eps {
                    if !filter.is_empty() && !ep.contains(filter.as_str()) {
                        continue;
                    }
                    let words = match spirv(src, ep) {
                        Ok(w) => w,
                        Err(e) => {
                            println!("{name}/{ep}: naga error {e}");
                            continue;
                        }
                    };
                    let sm = dev.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None).unwrap();
                    let cname = std::ffi::CString::new(*ep).unwrap();
                    let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(sm).name(&cname);
                    let ci = [vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(pl)
                        .flags(vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR | vk::PipelineCreateFlags::CAPTURE_INTERNAL_REPRESENTATIONS_KHR)];
                    let pipe = match dev.create_compute_pipelines(vk::PipelineCache::null(), &ci, None) {
                        Ok(p) => p[0],
                        Err((_, e)) => {
                            println!("{name}/{ep}: pipeline error {e:?}");
                            continue;
                        }
                    };
                    let pinfo = vk::PipelineInfoKHR::default().pipeline(pipe);
                    let exes = pep.get_pipeline_executable_properties(&pinfo).unwrap();
                    for (ei, _) in exes.iter().enumerate() {
                        let einfo = vk::PipelineExecutableInfoKHR::default().pipeline(pipe).executable_index(ei as u32);
                        let stats = pep.get_pipeline_executable_statistics(&einfo).unwrap();
                        let mut line = format!("{name}/{ep}:");
                        for st in &stats {
                            let n = CStr::from_ptr(st.name.as_ptr()).to_string_lossy();
                            let v = match st.format {
                                vk::PipelineExecutableStatisticFormatKHR::BOOL32 => format!("{}", st.value.b32),
                                vk::PipelineExecutableStatisticFormatKHR::INT64 => format!("{}", st.value.i64),
                                vk::PipelineExecutableStatisticFormatKHR::UINT64 => format!("{}", st.value.u64),
                                _ => format!("{}", st.value.f64),
                            };
                            line += &format!("  {n}={v}");
                        }
                        println!("{line}");
                        if let Ok(dir) = std::env::var("SHADER_IR") {
                            // two calls: sizes, then the data into our buffers
                            let fp = pep.fp();
                            let mut count = 0u32;
                            let _ = (fp.get_pipeline_executable_internal_representations_khr)(dev.handle(), &einfo, &mut count, std::ptr::null_mut());
                            let mut irs = vec![vk::PipelineExecutableInternalRepresentationKHR::default(); count as usize];
                            let _ = (fp.get_pipeline_executable_internal_representations_khr)(dev.handle(), &einfo, &mut count, irs.as_mut_ptr());
                            let mut bufs: Vec<Vec<u8>> = irs.iter().map(|ir| vec![0u8; ir.data_size]).collect();
                            for (ir, b) in irs.iter_mut().zip(bufs.iter_mut()) {
                                ir.p_data = b.as_mut_ptr() as *mut std::ffi::c_void;
                            }
                            let _ = (fp.get_pipeline_executable_internal_representations_khr)(dev.handle(), &einfo, &mut count, irs.as_mut_ptr());
                            for (k, (ir, b)) in irs.iter().zip(&bufs).enumerate() {
                                let n = CStr::from_ptr(ir.name.as_ptr()).to_string_lossy();
                                let path = format!("{dir}/{ep}.{k}.{}", if ir.is_text != 0 { "txt" } else { "bin" });
                                let _ = std::fs::write(&path, b);
                                println!("  IR {n}: {} bytes -> {path}", ir.data_size);
                            }
                        }
                    }
                    dev.destroy_pipeline(pipe, None);
                    dev.destroy_shader_module(sm, None);
                }
            }
        }
    }
}

fn main() {
    #[cfg(all(feature = "gpu", target_os = "linux"))]
    imp::main();
}
