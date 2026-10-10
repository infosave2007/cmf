//! Host-to-card transfer ceilings for the Qwen3.8-Flash-Next expert
//! admissions (scratch diagnostic, not a product).
//!
//! An admitted expert is ~1.67 MB in three parts (gate, up, down), copied
//! from the memory-mapped model file into the VRAM arena. This measures, in
//! GB/s, every leg such a copy can take on this host:
//!
//! - `memcpy` out of the page-cached mapping into private RAM (1..N
//!   threads; first touch of the pages in this process, then again);
//! - `memcpy` into a wgpu `MAP_WRITE` buffer (where `write_buffer` stages);
//! - `memcpy` into host-heap memory (`host_mem::sysmem_buffer`, write-
//!   combined and cached types);
//! - `queue.write_buffer` from the mapping (1..N threads, submit + wait);
//! - DMA: `copy_buffer_to_buffer` from host-heap / `MAP_WRITE` buffers into
//!   a device-local buffer, as one large copy and as expert-sized parts.
//!
//! Usage (Linux): cargo build --release -p cortiq-engine --features gpu \
//!          --example qwen4_xfer_bench && qwen4_xfer_bench <model.cmf>

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const PART: usize = 557_056; // a third of a 1.67 MB expert, 4 KiB aligned
const EXPERT: usize = 3 * PART;

struct Map {
    base: *const u8,
    len: usize,
}
unsafe impl Send for Map {}
unsafe impl Sync for Map {}

// Linux only, like the host-heap buffers it measures: `libc` is not a
// dependency on Windows, where `--features gpu --all-targets` builds this too.
#[cfg(not(target_os = "linux"))]
fn map_file(_path: &str) -> Map {
    panic!("qwen4_xfer_bench runs on Linux only (mmap of the model, Vulkan host-heap buffers)");
}

#[cfg(target_os = "linux")]
fn map_file(path: &str) -> Map {
    use std::os::unix::io::AsRawFd;
    let f = std::fs::File::open(path).expect("open");
    let len = f.metadata().expect("stat").len() as usize;
    // SAFETY: read-only shared mapping of a file we keep open for the run.
    let p = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            f.as_raw_fd(),
            0,
        )
    };
    assert!(p != libc::MAP_FAILED, "mmap");
    std::mem::forget(f);
    Map {
        base: p as *const u8,
        len,
    }
}

/// `n` random expert-sized source offsets (4 KiB aligned) in the file.
fn offsets(m: &Map, n: usize, seed: u64) -> Vec<usize> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s as usize) % (m.len - 2 * EXPERT)) & !4095
        })
        .collect()
}

/// Run `f(i)` for i in 0..n on `threads` threads; seconds of wall time.
fn par(threads: usize, n: usize, f: &(dyn Fn(usize) + Sync)) -> f64 {
    let next = AtomicUsize::new(0);
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    f(i);
                }
            });
        }
    });
    t0.elapsed().as_secs_f64()
}

fn gbs(bytes: usize, secs: f64) -> f64 {
    bytes as f64 / secs / 1e9
}

struct Dst(*mut u8, usize);
unsafe impl Send for Dst {}
unsafe impl Sync for Dst {}
impl Dst {
    fn at(&self, off: usize) -> *mut u8 {
        // SAFETY: callers stay in bounds
        unsafe { self.0.add(off) }
    }
}
impl Map {
    fn at(&self, off: usize) -> *const u8 {
        // SAFETY: callers stay in bounds
        unsafe { self.base.add(off) }
    }
}

/// memcpy every expert of `offs` from the mapping into `dst` (slot i % slots).
fn copy_into(m: &Map, offs: &[usize], dst: &Dst, threads: usize) -> f64 {
    let slots = dst.1 / EXPERT;
    par(threads, offs.len(), &|i| {
        let d = (i % slots) * EXPERT;
        // SAFETY: in-bounds source and destination; slots of concurrent
        // copies may coincide (benchmark scratch, never read back).
        unsafe {
            std::ptr::copy_nonoverlapping(m.at(offs[i]), dst.at(d), EXPERT);
        }
    })
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: qwen4_xfer_bench <model.cmf>");
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024);
    let thr: Vec<usize> = std::env::var("THREADS")
        .unwrap_or_else(|_| "1,4,8,16,32".into())
        .split(',')
        .filter_map(|v| v.parse().ok())
        .collect();
    let m = map_file(&path);
    let bytes = n * EXPERT;
    println!(
        "file {:.1} GB, {n} experts of {:.2} MB per test = {:.2} GB",
        m.len as f64 / 1e9,
        EXPERT as f64 / 1e6,
        bytes as f64 / 1e9
    );
    let mut seed = 1u64;
    let mut fresh = || {
        seed += 1;
        offsets(&m, n, seed)
    };

    // 1. mapping -> private RAM
    let mut heap = vec![0u8; 64 * EXPERT];
    let hd = Dst(heap.as_mut_ptr(), heap.len());
    for &t in &thr {
        let o = fresh();
        let a = copy_into(&m, &o, &hd, t);
        let b = copy_into(&m, &o, &hd, t);
        println!(
            "memcpy mmap->heap      threads={t:3}: first touch {:6.2} GB/s, again {:6.2} GB/s",
            gbs(bytes, a),
            gbs(bytes, b)
        );
    }

    // wgpu device
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .expect("adapter");
    let mut limits = adapter.limits();
    limits.max_buffer_size = limits.max_buffer_size.min(u32::MAX as u64);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("xfer"),
        required_limits: limits,
        ..Default::default()
    }))
    .expect("device");
    println!("adapter: {}", adapter.get_info().name);
    let wait = || {
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
    };
    let gib = 1u64 << 30;
    let vram = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vram"),
        size: gib,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let vram_slots = (gib as usize) / EXPERT;

    // 2. mapping -> MAP_WRITE buffer (where wgpu's own staging lands)
    let mw_size = (96 * EXPERT) as u64;
    let mw = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("map-write"),
        size: mw_size,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: true,
    });
    {
        let mut view = mw.slice(..).get_mapped_range_mut().expect("mapped");
        let len = view.len();
        let d = Dst(view.slice(..).as_raw_element_ptr().as_ptr(), len);
        for &t in &thr {
            let o = fresh();
            copy_into(&m, &o, &hd, t.max(16)); // pages touched, not timed
            let a = copy_into(&m, &o, &d, t);
            println!(
                "memcpy mmap->MAP_WRITE threads={t:3}: {:6.2} GB/s",
                gbs(bytes, a)
            );
        }
    }
    mw.unmap();

    // 3. mapping / heap -> host-heap buffers
    let ta = Instant::now();
    let sys_wc = cortiq_engine::gpu_wgpu::host_mem::sysmem_buffer(&device, gib, false);
    let a_wc = ta.elapsed().as_secs_f64() * 1e3;
    let ta = Instant::now();
    let sys_c = cortiq_engine::gpu_wgpu::host_mem::sysmem_buffer(&device, gib, true);
    let a_c = ta.elapsed().as_secs_f64() * 1e3;
    println!(
        "host-heap buffers (1 GiB): write-combined {} ({a_wc:.1} ms to allocate), cached {} ({a_c:.1} ms)",
        sys_wc.is_some(),
        sys_c.is_some()
    );
    for (name, sb) in [("sysWC", sys_wc.as_ref()), ("sysCached", sys_c.as_ref())] {
        let Some(sb) = sb else { continue };
        let d = Dst(sb.as_ptr(), sb.size() as usize);
        // fault the whole mapping in once
        par(32, (sb.size() as usize) / EXPERT, &|i| unsafe {
            std::ptr::write_bytes(d.at(i * EXPERT), 0, EXPERT);
        });
        for &t in &thr {
            let o = fresh();
            let a = copy_into(&m, &o, &d, t);
            let b = copy_into(&m, &o, &d, t);
            let src = Dst(heap.as_mut_ptr(), heap.len());
            let c = par(t, n, &|i| unsafe {
                std::ptr::copy_nonoverlapping(
                    src.at((i % 64) * EXPERT),
                    d.at((i % (d.1 / EXPERT)) * EXPERT),
                    EXPERT,
                );
            });
            println!(
                "memcpy mmap->{name:9} threads={t:3}: first touch {:6.2} GB/s, again {:6.2} GB/s; heap->{name} {:6.2} GB/s",
                gbs(bytes, a),
                gbs(bytes, b),
                gbs(bytes, c)
            );
        }
    }

    // 4. queue.write_buffer from the mapping
    for &t in &thr {
        let o = fresh();
        let t0 = Instant::now();
        par(t, n, &|i| {
            let slot = (i % vram_slots) * EXPERT;
            for p in 0..3 {
                // SAFETY: in-bounds slice of the mapping
                let src = unsafe { std::slice::from_raw_parts(m.at(o[i] + p * PART), PART) };
                queue.write_buffer(&vram, (slot + p * PART) as u64, src);
            }
        });
        let host = t0.elapsed().as_secs_f64();
        queue.submit(std::iter::empty());
        wait();
        let all = t0.elapsed().as_secs_f64();
        println!(
            "write_buffer mmap->vram threads={t:3}: host {:6.2} GB/s, with submit+wait {:6.2} GB/s",
            gbs(bytes, host),
            gbs(bytes, all)
        );
    }

    // 5. DMA into VRAM
    let dma = |src: &wgpu::Buffer, src_size: u64, parts: Option<usize>, label: &str| {
        for rep in 0..3 {
            let mut enc = device.create_command_encoder(&Default::default());
            let moved = match parts {
                None => {
                    let len = src_size.min(gib);
                    enc.copy_buffer_to_buffer(src, 0, &vram, 0, len);
                    len as usize
                }
                Some(k) => {
                    let src_slots = (src_size as usize) / EXPERT;
                    for e in 0..k {
                        let so = (e % src_slots) * EXPERT;
                        let dof = (e % vram_slots) * EXPERT;
                        for p in 0..3 {
                            enc.copy_buffer_to_buffer(
                                src,
                                (so + p * PART) as u64,
                                &vram,
                                (dof + p * PART) as u64,
                                PART as u64,
                            );
                        }
                    }
                    k * EXPERT
                }
            };
            let cb = enc.finish();
            let t0 = Instant::now();
            queue.submit([cb]);
            wait();
            let s = t0.elapsed().as_secs_f64();
            println!(
                "DMA {label:28} rep {rep}: {:7.1} MB in {:6.2} ms = {:6.2} GB/s",
                moved as f64 / 1e6,
                s * 1e3,
                gbs(moved, s)
            );
        }
    };
    for (name, sb) in [("sysWC", sys_wc.as_ref()), ("sysCached", sys_c.as_ref())] {
        let Some(sb) = sb else { continue };
        dma(&sb.buffer, sb.size(), None, &format!("{name} 1 GiB one copy"));
        dma(&sb.buffer, sb.size(), Some(400), &format!("{name} 400 experts x3"));
        dma(&sb.buffer, sb.size(), Some(10), &format!("{name} 10 experts x3"));
    }
    dma(&mw, mw_size, None, "MAP_WRITE one copy");
    dma(&mw, mw_size, Some(96), "MAP_WRITE 96 experts x3");
    let big_mw = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("map-write-1g"),
        size: gib,
        usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    dma(&big_mw, gib, None, "MAP_WRITE 1 GiB one copy");

    // 6. DMA while the host copies into the other half (overlap check)
    if let Some(sb) = sys_wc.as_ref() {
        let d = Dst(sb.as_ptr(), sb.size() as usize);
        let o = fresh();
        let half = (sb.size() / 2) as usize;
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&sb.buffer, 0, &vram, 0, half as u64);
        let cb = enc.finish();
        let t0 = Instant::now();
        queue.submit([cb]);
        let dh = Dst(d.at(half), half);
        let a = copy_into(&m, &o, &dh, 32);
        wait();
        let s = t0.elapsed().as_secs_f64();
        println!(
            "overlap: DMA {:.0} MB + host memcpy {:.0} MB (32 thr, {:.2} GB/s alone-in-run) in {:.1} ms",
            half as f64 / 1e6,
            bytes as f64 / 1e6,
            gbs(bytes, a),
            s * 1e3
        );
    }
}
