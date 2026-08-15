//! What a node is worth RIGHT NOW, read from the OS rather than declared.
//!
//! A phone's budget is not a constant. On a Snapdragon 778G the same
//! worker, same model, same room temperature (36 °C, no throttling in
//! sight) served a layer span at 22.6 ms one minute and 42.4 ms the next
//! — the difference was the DVFS state, because a task that computes for
//! a few milliseconds and then blocks on a socket never convinces
//! `schedutil` to raise the clock. A planner that trusts a number
//! measured at connect time will keep a plan that stopped being true.
//!
//! So the worker reports the three things that actually move: how hot it
//! is, whether it is on the wall, and what its fastest core is clocked at
//! this instant. Nothing here is a promise — every field is `Option`,
//! absent when the platform does not expose it, and no field is ever
//! guessed.

use serde::{Deserialize, Serialize};

/// A node's live capacity signals. Every field is optional on purpose:
/// "unknown" and "zero" must never be the same value to a scheduler.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct NodeStats {
    /// Hottest CPU/SoC thermal zone, milli-degrees C.
    pub thermal_mc: Option<i64>,
    /// Mains/USB power attached. A phone on battery is a different node.
    pub powered: Option<bool>,
    /// Current clock of the fastest core, kHz — the DVFS state, which on
    /// a bursty network worker is the single largest unmodelled factor.
    pub cpu_khz_cur: Option<u64>,
    /// What that core CAN reach, kHz. `cur / max` is the throttle ratio
    /// the planner should read instead of guessing from temperature.
    pub cpu_khz_max: Option<u64>,
    /// Memory the OS says is available, kB — the ceiling on a layer span.
    pub mem_avail_kb: Option<u64>,
    /// Worker pool size actually in use.
    pub threads: u32,
    /// Human label for logs: "android/aarch64", "macos/aarch64", …
    pub platform: String,
}

impl NodeStats {
    /// Fraction of peak clock the fastest core is running at, when both
    /// numbers are known. Below ~0.5 on a decode loop means the governor
    /// never woke up, not that the part is slow.
    pub fn clock_ratio(&self) -> Option<f64> {
        match (self.cpu_khz_cur, self.cpu_khz_max) {
            (Some(c), Some(m)) if m > 0 => Some(c as f64 / m as f64),
            _ => None,
        }
    }

    /// One line for a log or a status endpoint. Unknown fields say so.
    pub fn summary(&self) -> String {
        let t = self
            .thermal_mc
            .map(|v| format!("{:.1}°C", v as f64 / 1000.0))
            .unwrap_or_else(|| "temp ?".into());
        let p = match self.powered {
            Some(true) => "on mains",
            Some(false) => "on battery",
            None => "power ?",
        };
        let c = match (self.cpu_khz_cur, self.cpu_khz_max) {
            (Some(cur), Some(max)) => format!(
                "{} of {} MHz ({:.0}%)",
                cur / 1000,
                max / 1000,
                100.0 * cur as f64 / max as f64
            ),
            (Some(cur), None) => format!("{} MHz", cur / 1000),
            _ => "clock ?".into(),
        };
        let m = self
            .mem_avail_kb
            .map(|v| format!("{:.1} GB free", v as f64 / 1_048_576.0))
            .unwrap_or_else(|| "mem ?".into());
        format!(
            "{} · {} threads · {} · {} · {} · {}",
            self.platform, self.threads, c, t, p, m
        )
    }
}

fn read_num(path: &str) -> Option<u64> {
    std::fs::read_to_string(path)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
}

/// Hottest zone among the ones whose `type` looks like a CPU/SoC/skin
/// sensor. Battery and charger zones are excluded: they lag the die by
/// minutes and would smooth away exactly the event we care about.
#[cfg(target_os = "android")]
fn thermal_mc() -> Option<i64> {
    let dir = std::fs::read_dir("/sys/class/thermal").ok()?;
    let mut hottest: Option<i64> = None;
    for e in dir.flatten() {
        let p = e.path();
        let ty = std::fs::read_to_string(p.join("type")).unwrap_or_default();
        let ty = ty.trim();
        let interesting = ty.contains("cpu") || ty.contains("soc") || ty.contains("skin");
        if !interesting {
            continue;
        }
        if let Some(v) = read_num(&p.join("temp").to_string_lossy()) {
            let v = v as i64;
            hottest = Some(hottest.map_or(v, |h| h.max(v)));
        }
    }
    hottest
}

#[cfg(not(target_os = "android"))]
fn thermal_mc() -> Option<i64> {
    None
}

/// `online` on any non-battery supply — the USB cable that makes a phone
/// a usable worker is exactly this bit.
#[cfg(target_os = "android")]
fn powered() -> Option<bool> {
    let dir = std::fs::read_dir("/sys/class/power_supply").ok()?;
    let mut any = false;
    for e in dir.flatten() {
        let p = e.path();
        let ty = std::fs::read_to_string(p.join("type")).unwrap_or_default();
        if ty.trim().eq_ignore_ascii_case("battery") {
            continue;
        }
        if let Some(v) = read_num(&p.join("online").to_string_lossy()) {
            any = true;
            if v == 1 {
                return Some(true);
            }
        }
    }
    if any { Some(false) } else { None }
}

#[cfg(not(target_os = "android"))]
fn powered() -> Option<bool> {
    None
}

/// The fastest core's current and peak clock. "Fastest" is decided by
/// `cpuinfo_max_freq`, so this reads the big cluster on big.LITTLE
/// without hard-coding a core index.
#[cfg(any(target_os = "android", target_os = "linux"))]
fn clocks() -> (Option<u64>, Option<u64>) {
    let mut best_max = 0u64;
    let mut best_cur = None;
    for cpu in 0..64 {
        let base = format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq");
        let Some(max) = read_num(&format!("{base}/cpuinfo_max_freq")) else {
            continue;
        };
        if max > best_max {
            best_max = max;
            best_cur = read_num(&format!("{base}/scaling_cur_freq"));
        }
    }
    if best_max == 0 {
        (None, None)
    } else {
        (best_cur, Some(best_max))
    }
}

#[cfg(not(any(target_os = "android", target_os = "linux")))]
fn clocks() -> (Option<u64>, Option<u64>) {
    (None, None)
}

#[cfg(any(target_os = "android", target_os = "linux"))]
fn mem_avail_kb() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            return rest.split_whitespace().next()?.parse().ok();
        }
    }
    None
}

#[cfg(not(any(target_os = "android", target_os = "linux")))]
fn mem_avail_kb() -> Option<u64> {
    None
}

fn platform() -> String {
    format!("{}/{}", std::env::consts::OS, std::env::consts::ARCH)
}

/// Read the node's live signals. Cheap enough to call per request (four
/// small sysfs reads); the coordinator decides how often it wants them.
pub fn read(threads: usize) -> NodeStats {
    let (cur, max) = clocks();
    NodeStats {
        thermal_mc: thermal_mc(),
        powered: powered(),
        cpu_khz_cur: cur,
        cpu_khz_max: max,
        mem_avail_kb: mem_avail_kb(),
        threads: threads as u32,
        platform: platform(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_is_not_zero() {
        // A field the platform does not expose must stay None all the way
        // to the summary — a planner reading 0 MHz would park the node.
        let s = NodeStats {
            platform: "test/arch".into(),
            threads: 4,
            ..Default::default()
        };
        assert!(s.clock_ratio().is_none());
        let line = s.summary();
        assert!(line.contains("clock ?"), "{line}");
        assert!(line.contains("temp ?"), "{line}");
        assert!(line.contains("power ?"), "{line}");
    }

    #[test]
    fn clock_ratio_reads_the_governor() {
        let s = NodeStats {
            cpu_khz_cur: Some(691_200),
            cpu_khz_max: Some(2_400_000),
            ..Default::default()
        };
        // The measured Android case: a third of peak while decoding.
        let r = s.clock_ratio().unwrap();
        assert!(r > 0.28 && r < 0.29, "{r}");
        assert!(
            s.summary().contains("691 of 2400 MHz (29%)"),
            "{}",
            s.summary()
        );
    }

    #[test]
    fn reading_this_host_never_panics() {
        let s = read(4);
        assert_eq!(s.threads, 4);
        assert!(s.platform.contains('/'));
    }
}
