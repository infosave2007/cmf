//! Keep the working set resident.
//!
//! An mmap'd model is demand-paged, and the page cache evicts by age. For a
//! mixture of experts that is the wrong policy: one cold expert, touched
//! once, can evict a hot one that every token needs. The routing field says
//! which experts a task actually uses — measured on this model, 99 of 256
//! carry 95% of the mass — so the fix is not to load the rest on demand
//! (mmap already does) but to stop the rest from evicting what matters.
//!
//! `CMF_MOE_PIN=<stats.json>` pins the skeleton plus the covered experts;
//! `CMF_MOE_PIN_COVER` sets the fraction (default 0.95).
//!
//! Two things make this fail quietly if unattended. `RLIMIT_MEMLOCK` is 8 MB
//! on a stock container, so the first `mlock` past that returns ENOMEM and
//! every later one does too — the limit is raised here, and if that is
//! refused the caller is told rather than left with 8 MB pinned. And pinning
//! more than physical memory is a way to invite the OOM killer, so the
//! budget is checked against what the machine has.

use std::collections::HashMap;

#[cfg(not(any(target_os = "linux", target_os = "android")))]
mod imp {
    /// Pinning is a server-side concern and the pieces it needs — mlock
    /// through libc, /proc/meminfo — are Linux's. Elsewhere it reports that
    /// it did nothing rather than pretending otherwise.
    pub fn raise_memlock_limit() -> Option<u64> {
        None
    }
    pub fn lock_slice(_b: &[u8]) -> std::io::Result<()> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "закрепление памяти поддержано только на Linux",
        ))
    }
    pub fn phys_mem() -> Option<(u64, u64)> {
        None
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
mod imp {
    /// Raise the locked-memory limit as far as the process is allowed to.
    /// Returns the new soft limit in bytes, or None if it could not be read.
    pub fn raise_memlock_limit() -> Option<u64> {
        unsafe {
            let mut rl = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            if libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut rl) != 0 {
                return None;
            }
            // Raising the soft limit to the hard one is not enough: a stock
            // container ships 8 MB for BOTH, and the first tensor of any real
            // model is bigger than that. With CAP_SYS_RESOURCE the hard limit
            // can go too, so try that first and fall back.
            for want in [
                libc::rlimit {
                    rlim_cur: libc::RLIM_INFINITY,
                    rlim_max: libc::RLIM_INFINITY,
                },
                libc::rlimit {
                    rlim_cur: rl.rlim_max,
                    rlim_max: rl.rlim_max,
                },
            ] {
                if libc::setrlimit(libc::RLIMIT_MEMLOCK, &want) == 0 {
                    rl.rlim_cur = want.rlim_cur;
                    break;
                }
            }
            Some(rl.rlim_cur as u64)
        }
    }

    /// Total and available physical memory in bytes, if the kernel will say.
    pub fn phys_mem() -> Option<(u64, u64)> {
        let s = std::fs::read_to_string("/proc/meminfo").ok()?;
        let get = |k: &str| -> Option<u64> {
            s.lines()
                .find(|l| l.starts_with(k))?
                .split_whitespace()
                .nth(1)?
                .parse::<u64>()
                .ok()
                .map(|kb| kb * 1024)
        };
        Some((get("MemTotal:")?, get("MemAvailable:")?))
    }

    /// mlock one byte range. The kernel wants page-aligned addresses; a slice
    /// from the middle of a mapping is not, so the range is widened outward.
    pub fn lock_slice(b: &[u8]) -> std::io::Result<()> {
        if b.is_empty() {
            return Ok(());
        }
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let start = b.as_ptr() as usize;
        let aligned = start & !(page - 1);
        let len = (start - aligned) + b.len();
        let rc = unsafe { libc::mlock(aligned as *const libc::c_void, len) };
        if rc == 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error())
        }
    }
}

pub use imp::raise_memlock_limit;
use imp::{lock_slice, phys_mem};

/// The experts holding `cover` of a layer's routing mass, per layer.
pub fn hot_experts(stats_path: &str, cover: f64) -> Option<HashMap<usize, Vec<usize>>> {
    let text = std::fs::read_to_string(stats_path)
        .map_err(|e| tracing::warn!("CMF_MOE_PIN: cannot read {stats_path}: {e}"))
        .ok()?;
    let map: HashMap<String, Vec<u64>> = serde_json::from_str(&text)
        .map_err(|e| tracing::warn!("CMF_MOE_PIN: bad JSON in {stats_path}: {e}"))
        .ok()?;
    let mut out = HashMap::new();
    for (k, counts) in map {
        let Ok(li) = k.parse::<usize>() else { continue };
        let total: u64 = counts.iter().sum();
        if total == 0 {
            continue;
        }
        let mut order: Vec<usize> = (0..counts.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(counts[i]));
        let mut acc = 0u64;
        let mut keep = Vec::new();
        for i in order {
            if counts[i] == 0 {
                break;
            }
            acc += counts[i];
            keep.push(i);
            if acc as f64 >= cover * total as f64 {
                break;
            }
        }
        out.insert(li, keep);
    }
    Some(out)
}

/// What a pinning pass did, for the log and for the caller to judge.
pub struct Pinned {
    pub bytes: u64,
    pub tensors: usize,
    pub skipped: usize,
    pub limit: Option<u64>,
}

/// Pin every named tensor that exists, stopping cleanly at the first refusal
/// rather than hammering the kernel with thousands of doomed calls.
pub fn pin_tensors(model: &cortiq_core::CmfModel, names: &[String]) -> Pinned {
    let limit = raise_memlock_limit();
    let mut want: u64 = 0;
    for n in names {
        if let Ok(b) = model.tensor_bytes(n) {
            want += b.len() as u64;
        }
    }
    if let Some((total, avail)) = phys_mem() {
        if want > avail {
            tracing::warn!(
                "закрепление {:.1} ГБ при доступных {:.1} ГБ из {:.1} — \
                 закрепляю столько, сколько поместится, остальное останется \
                 подкачиваемым",
                want as f64 / 1e9,
                avail as f64 / 1e9,
                total as f64 / 1e9
            );
        }
    }
    let mut out = Pinned {
        bytes: 0,
        tensors: 0,
        skipped: 0,
        limit,
    };
    let mut fails = 0usize;
    for n in names {
        let Ok(b) = model.tensor_bytes(n) else {
            out.skipped += 1;
            continue;
        };
        match lock_slice(b) {
            Ok(()) => {
                out.bytes += b.len() as u64;
                out.tensors += 1;
            }
            Err(e) => {
                // One oversized tensor is not a reason to abandon the rest —
                // the first attempt gave up on the embedding and pinned
                // nothing at all. Give up only when it is clearly hopeless.
                out.skipped += 1;
                fails += 1;
                if fails == 1 {
                    tracing::warn!("первое закрепление не удалось ({n}): {e}");
                }
                if fails >= 64 && out.bytes < 1 << 30 {
                    tracing::warn!(
                        "закрепление упирается в RLIMIT_MEMLOCK = {:.0} МБ и поднять \
                         его не дали (контейнеры обычно снимают CAP_SYS_RESOURCE). \
                         Лечится вне процесса: `--ulimit memlock=-1` у докера, \
                         LimitMEMLOCK=infinity в systemd, или запуск с этой \
                         привилегией. Пропущено {} тензоров.",
                        limit.unwrap_or(0) as f64 / 1e6,
                        names.len() - out.skipped
                    );
                    break;
                }
            }
        }
    }
    out
}
