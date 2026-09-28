//! Shared, explicitly selected native Vulkan device. No software/CPU fallback.
use anyhow::{Context, Result, bail, ensure};
use std::sync::{Arc, OnceLock};
use wgpu::util::DeviceExt;

pub(crate) struct ContextGpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub name: String,
    pub nvidia: bool,
    error: Arc<parking_lot::Mutex<Option<String>>>,
}
static CONTEXT: OnceLock<parking_lot::Mutex<Option<Arc<ContextGpu>>>> = OnceLock::new();

impl ContextGpu {
    pub fn get() -> Result<Arc<Self>> {
        let mut slot = CONTEXT.get_or_init(|| parking_lot::Mutex::new(None)).lock();
        let selector = std::env::var("CORTIQ_DECISION_VULKAN_ADAPTER").ok();
        if let Some(context) = slot.as_ref() {
            ensure!(
                selector
                    .as_ref()
                    .is_none_or(|s| context.name.to_lowercase().contains(&s.to_lowercase())),
                "Vulkan device already initialized as {}; cannot change adapter in a running process",
                context.name
            );
            context.check()?;
            return Ok(context.clone());
        }
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });
        let mut adapters: Vec<_> =
            pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN))
                .into_iter()
                .filter(|a| {
                    let i = a.get_info();
                    i.device_type != wgpu::DeviceType::Cpu
                        && i.backend == wgpu::Backend::Vulkan
                        && selector
                            .as_ref()
                            .is_none_or(|s| i.name.to_lowercase().contains(&s.to_lowercase()))
                })
                .collect();
        ensure!(
            adapters.len() == 1,
            "expected one hardware Vulkan adapter, found {}; set CORTIQ_DECISION_VULKAN_ADAPTER to a unique GPU name",
            adapters.len()
        );
        let adapter = adapters.remove(0);
        ensure!(
            adapter.features().contains(wgpu::Features::SUBGROUP),
            "Vulkan GPU lacks FP32 subgroup reductions"
        );
        let profile = std::env::var_os("CORTIQ_DECISION_VULKAN_PROFILE").is_some();
        ensure!(
            !profile || adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY),
            "Vulkan timestamp profiling unsupported"
        );
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Cortiq Decision Vulkan"),
            required_features: wgpu::Features::SUBGROUP
                | if profile {
                    wgpu::Features::TIMESTAMP_QUERY
                } else {
                    wgpu::Features::empty()
                },
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .context("create decision Vulkan device")?;
        let error = Arc::new(parking_lot::Mutex::new(None));
        let handler_error = error.clone();
        device.on_uncaptured_error(Arc::new(move |e| {
            *handler_error.lock() = Some(format!("{e}"));
        }));
        let context = Arc::new(Self {
            device,
            queue,
            name: adapter.get_info().name,
            nvidia: adapter.get_info().vendor == 0x10de,
            error,
        });
        *slot = Some(context.clone());
        Ok(context)
    }
    pub fn check(&self) -> Result<()> {
        if let Some(e) = self.error.lock().as_ref() {
            bail!("decision Vulkan device error: {e}");
        }
        Ok(())
    }
    pub fn upload(&self, data: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("decision resident data"),
                contents: if data.is_empty() { &[0; 4] } else { data },
                usage,
            })
    }
    pub fn buffer(&self, bytes: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("decision scratch"),
            size: bytes.max(4),
            usage,
            mapped_at_creation: false,
        })
    }
    /// Exactly one blocking completion wait, with an explicit timeout.
    pub fn readback(
        &self,
        buffer: &wgpu::Buffer,
        bytes: u64,
        submission: wgpu::SubmissionIndex,
    ) -> Result<Vec<f32>> {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        buffer
            .slice(..bytes)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        let polled = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(std::time::Duration::from_secs(30)),
        });
        if let Err(e) = polled {
            buffer.unmap();
            bail!("Vulkan completion: {e}");
        }
        let mapped = match rx.recv_timeout(std::time::Duration::from_secs(1)) {
            Ok(value) => value,
            Err(e) => {
                buffer.unmap();
                bail!("Vulkan readback callback: {e}");
            }
        };
        if let Err(e) = mapped {
            buffer.unmap();
            bail!("Vulkan readback: {e}");
        }
        let view = match buffer.slice(..bytes).get_mapped_range() {
            Ok(view) => view,
            Err(e) => {
                buffer.unmap();
                bail!("Vulkan mapped range: {e}");
            }
        };
        let result = bytemuck::try_cast_slice::<u8, f32>(&view)
            .map(<[f32]>::to_vec)
            .map_err(|e| anyhow::anyhow!("Vulkan readback alignment: {e}"));
        drop(view);
        buffer.unmap();
        self.check()?;
        result
    }
}

pub(crate) fn entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
pub(crate) fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
