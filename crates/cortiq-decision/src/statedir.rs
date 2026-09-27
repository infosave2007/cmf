//! State directory layout and the `LOCK` (spec §4.14, §4.11).
//!
//! ```text
//! <state>/            mode 0700
//!   LOCK              pid and a random nonce; created with O_EXCL (one serving process)
//!   keys.json         API key records (sha256 of the key only), atomic replace
//!   keys.json.lock    held (O_EXCL) by any process while it changes keys.json
//!   usage/            YYYY-MM.jsonl usage ledger + totals.json snapshot
//!   oracle.jsonl      oracle reservation ledger (cascade)
//!   oracle.state      oracle stop reason and switches (cascade)
//!   learn.log         cache, buffer and learning records (cascade)
//!   shadow.jsonl      router-API shadow comparisons (`serve --shadow-of`), no texts
//!   shadow.key        random HMAC key of the text digests in shadow.jsonl (0600)
//!   generations/      gNNNNNN.cmf overlay generations
//!   CURRENT           "gNNNNNN <sha256>" of the served generation, atomic replace
//! ```
//!
//! [`StateDir::open`] only creates the layout; a serving process also takes the
//! [`StateLock`] ([`StateDir::lock`]). A second lock attempt on the same
//! directory fails with the holder's pid until the lock is dropped;
//! `--break-lock` ([`StateDir::lock`] with `break_lock`) removes a lock left by
//! a dead process. Key management from the CLI writes `keys.json` atomically
//! without the `LOCK`, under `keys.json.lock` like the server's own key
//! changes (neither writes back a file the other changed in between); a
//! server picks the new file up by its mtime
//! ([`crate::keys::KeyStore::maybe_reload`]).
//!
//! [`atomic_write`]: temp file in the same directory (`create_new`, mode 0600),
//! write, `fsync`, `rename` over the target, `fsync` of the directory.

use anyhow::{Context, Result, bail, ensure};
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};

pub const LOCK_FILE: &str = "LOCK";
pub const KEYS_FILE: &str = "keys.json";
pub const USAGE_DIR: &str = "usage";
pub const ORACLE_LEDGER_FILE: &str = "oracle.jsonl";
pub const ORACLE_STATE_FILE: &str = "oracle.state";
pub const LEARN_LOG_FILE: &str = "learn.log";
pub const GENERATIONS_DIR: &str = "generations";
pub const CURRENT_FILE: &str = "CURRENT";

/// A held state directory lock: removed on drop (only while it is still ours).
#[derive(Debug)]
pub struct StateLock {
    path: PathBuf,
    content: String,
}

impl StateLock {
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// A detached way to remove this lock from elsewhere (a signal handler
    /// that ends the process, which runs no destructors): it removes the file
    /// only while it still holds this lock's content.
    pub fn release_handle(&self) -> LockRelease {
        LockRelease {
            path: self.path.clone(),
            content: self.content.clone(),
        }
    }
}

impl Drop for StateLock {
    fn drop(&mut self) {
        LockRelease {
            path: std::mem::take(&mut self.path),
            content: std::mem::take(&mut self.content),
        }
        .release();
    }
}

/// See [`StateLock::release_handle`].
#[derive(Clone, Debug)]
pub struct LockRelease {
    path: PathBuf,
    content: String,
}

impl LockRelease {
    /// Remove the lock file if it is still this lock (a lock broken and
    /// retaken by another process is not ours to remove); `true` when removed.
    pub fn release(&self) -> bool {
        fs::read_to_string(&self.path).is_ok_and(|c| c == self.content)
            && fs::remove_file(&self.path).is_ok()
    }
}

/// Why the lock could not be taken.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "state directory {dir} is locked by pid {pid} ({lock}); one process per state directory — stop it, or remove a stale lock with --break-lock"
)]
pub struct Locked {
    pub dir: String,
    pub lock: String,
    pub pid: String,
}

/// The served generation named by `CURRENT`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Current {
    pub generation: u64,
    /// sha256 of the generation file (generation 0: of the base's `decision.manifest`).
    pub sha256: String,
}

/// The layout of one state directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateDir {
    root: PathBuf,
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: u32) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(mode))
        .with_context(|| format!("chmod {mode:o} {}", path.display()))
}

#[cfg(not(unix))]
fn set_mode(_path: &Path, _mode: u32) -> Result<()> {
    Ok(())
}

fn create_dir_0700(path: &Path) -> Result<()> {
    match fs::create_dir(path) {
        Ok(()) => {}
        Err(e) if e.kind() == ErrorKind::AlreadyExists => {
            ensure!(
                path.is_dir(),
                "{} exists and is not a directory",
                path.display()
            );
        }
        Err(e) => return Err(e).with_context(|| format!("create {}", path.display())),
    }
    set_mode(path, 0o700)
}

/// Open a new file for writing that must not exist yet (mode 0600).
pub fn create_new_private(path: &Path) -> std::io::Result<File> {
    let mut o = OpenOptions::new();
    o.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(0o600);
    }
    o.open(path)
}

/// `fsync` a directory (makes a rename or a new entry durable). A no-op where
/// directories cannot be opened (Windows).
pub fn sync_dir(dir: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        File::open(dir)
            .and_then(|d| d.sync_all())
            .with_context(|| format!("fsync {}", dir.display()))?;
    }
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

/// 16 hex characters from the OS RNG.
fn nonce() -> Result<String> {
    use rand_core::{OsRng, RngCore};
    let mut b = [0u8; 8];
    OsRng
        .try_fill_bytes(&mut b)
        .map_err(|e| anyhow::anyhow!("OS random number generator: {e}"))?;
    Ok(b.iter().map(|x| format!("{x:02x}")).collect())
}

/// Replace `path` with `bytes` atomically: a temp file next to it (mode 0600),
/// `fsync`, `rename`, `fsync` of the directory. A crash leaves the old or the
/// new content, never a mix.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let dir = match path.parent() {
        Some(d) if !d.as_os_str().is_empty() => d.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let name = path
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("{} has no file name", path.display()))?
        .to_string_lossy()
        .into_owned();
    let tmp = dir.join(format!(".{name}.{}.tmp", nonce()?));
    let result = (|| -> Result<()> {
        let mut f =
            create_new_private(&tmp).with_context(|| format!("create {}", tmp.display()))?;
        f.write_all(bytes)
            .and_then(|()| f.sync_all())
            .with_context(|| format!("write {}", tmp.display()))?;
        drop(f);
        fs::rename(&tmp, path)
            .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
        sync_dir(&dir)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

/// `gNNNNNN` (at least six digits).
pub fn generation_name(generation: u64) -> String {
    format!("g{generation:06}")
}

/// The generation number of a `gNNNNNN` name.
pub fn parse_generation_name(name: &str) -> Option<u64> {
    let digits = name.strip_prefix('g')?;
    if digits.len() < 6 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let n: u64 = digits.parse().ok()?;
    (generation_name(n) == name).then_some(n)
}

impl StateDir {
    /// The state directory at `root`: created (mode 0700) with `usage/` and
    /// `generations/` when missing; an existing directory is tightened to 0700.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        create_dir_0700(&root)?;
        create_dir_0700(&root.join(USAGE_DIR))?;
        create_dir_0700(&root.join(GENERATIONS_DIR))?;
        Ok(Self { root })
    }

    /// `<model>.state` (the default of spec §4.13).
    pub fn default_for(model: &Path) -> PathBuf {
        let mut s = model.as_os_str().to_os_string();
        s.push(crate::config::STATE_DIR_SUFFIX);
        PathBuf::from(s)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Take the exclusive `LOCK` (O_EXCL, content `pid nonce`). With
    /// `break_lock` an existing lock is removed first (its holder is reported
    /// by the caller as broken).
    pub fn lock(&self, break_lock: bool) -> Result<StateLock> {
        let path = self.lock_path();
        let content = format!("{} {}\n", std::process::id(), nonce()?);
        for attempt in 0..2 {
            match create_new_private(&path) {
                Ok(mut f) => {
                    f.write_all(content.as_bytes())
                        .and_then(|()| f.sync_all())
                        .with_context(|| format!("write {}", path.display()))?;
                    sync_dir(&self.root)?;
                    return Ok(StateLock { path, content });
                }
                Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                    if break_lock && attempt == 0 {
                        fs::remove_file(&path)
                            .with_context(|| format!("remove {}", path.display()))?;
                        continue;
                    }
                    let mut holder = String::new();
                    let _ = File::open(&path).and_then(|mut f| f.read_to_string(&mut holder));
                    let pid = holder
                        .split_whitespace()
                        .next()
                        .unwrap_or("unknown")
                        .to_string();
                    return Err(Locked {
                        dir: self.root.display().to_string(),
                        lock: path.display().to_string(),
                        pid,
                    }
                    .into());
                }
                Err(e) => return Err(e).with_context(|| format!("create {}", path.display())),
            }
        }
        bail!("could not take {} after breaking it", path.display())
    }

    /// Take the `LOCK` in place of the stale one the caller checked, whose
    /// content was `stale`: that lock is removed only while the file still
    /// holds `stale`, so a lock another process took meanwhile (two runs
    /// breaking the same stale lock at once) is never removed — this call
    /// then fails with [`Locked`] for it. The file is moved aside by an atomic
    /// rename before its content is compared; a lock that is not the stale
    /// one is put back without replacing a newer one.
    pub fn lock_replacing(&self, stale: &str) -> Result<StateLock> {
        let path = self.lock_path();
        let aside = self.root.join(format!("{LOCK_FILE}.breaking-{}", nonce()?));
        match fs::rename(&path, &aside) {
            Ok(()) => {
                let moved = fs::read_to_string(&aside).unwrap_or_default();
                if moved != stale {
                    // Not ours to break: back where it was (never over a lock
                    // taken in between), then the plain attempt names it.
                    let back = fs::hard_link(&aside, &path).is_ok()
                        || create_new_private(&path)
                            .and_then(|mut f| f.write_all(moved.as_bytes()).and(f.sync_all()))
                            .is_ok();
                    if !back {
                        tracing::warn!(
                            lock = %path.display(),
                            "could not put back a LOCK moved aside while breaking a stale one"
                        );
                    }
                }
                fs::remove_file(&aside).with_context(|| format!("remove {}", aside.display()))?;
                sync_dir(&self.root)?;
            }
            Err(e) if e.kind() == ErrorKind::NotFound => {}
            Err(e) => return Err(e).with_context(|| format!("move aside {}", path.display())),
        }
        self.lock(false)
    }

    pub fn lock_path(&self) -> PathBuf {
        self.root.join(LOCK_FILE)
    }

    pub fn keys_path(&self) -> PathBuf {
        self.root.join(KEYS_FILE)
    }

    pub fn usage_dir(&self) -> PathBuf {
        self.root.join(USAGE_DIR)
    }

    pub fn oracle_ledger_path(&self) -> PathBuf {
        self.root.join(ORACLE_LEDGER_FILE)
    }

    pub fn oracle_state_path(&self) -> PathBuf {
        self.root.join(ORACLE_STATE_FILE)
    }

    pub fn learn_log_path(&self) -> PathBuf {
        self.root.join(LEARN_LOG_FILE)
    }

    /// `shadow.jsonl` of `cortiq serve --shadow-of` ([`crate::shadow`]).
    pub fn shadow_log_path(&self) -> PathBuf {
        self.root.join(crate::shadow::SHADOW_LOG_FILE)
    }

    pub fn generations_dir(&self) -> PathBuf {
        self.root.join(GENERATIONS_DIR)
    }

    /// `generations/gNNNNNN.cmf`.
    pub fn generation_path(&self, generation: u64) -> PathBuf {
        self.generations_dir()
            .join(format!("{}.cmf", generation_name(generation)))
    }

    pub fn current_path(&self) -> PathBuf {
        self.root.join(CURRENT_FILE)
    }

    /// The generations present, ascending.
    pub fn generations(&self) -> Result<Vec<(u64, PathBuf)>> {
        let mut out = Vec::new();
        let dir = self.generations_dir();
        for e in fs::read_dir(&dir).with_context(|| format!("read {}", dir.display()))? {
            let e = e?;
            let name = e.file_name().to_string_lossy().into_owned();
            if let Some(stem) = name.strip_suffix(".cmf")
                && let Some(n) = parse_generation_name(stem)
            {
                out.push((n, e.path()));
            }
        }
        out.sort();
        Ok(out)
    }

    /// `CURRENT`, or `None` when it does not exist (the base is served).
    pub fn read_current(&self) -> Result<Option<Current>> {
        let path = self.current_path();
        let text = match fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e).with_context(|| format!("read {}", path.display())),
        };
        let mut it = text.split_whitespace();
        let (Some(name), Some(sha), None) = (it.next(), it.next(), it.next()) else {
            bail!("{}: expected 'gNNNNNN <sha256>'", path.display());
        };
        let generation = parse_generation_name(name)
            .ok_or_else(|| anyhow::anyhow!("{}: bad generation name '{name}'", path.display()))?;
        ensure!(
            crate::manifest::is_sha256_hex(sha),
            "{}: bad sha256 '{sha}'",
            path.display()
        );
        Ok(Some(Current {
            generation,
            sha256: sha.to_string(),
        }))
    }

    /// Replace `CURRENT` atomically with `gNNNNNN <sha256>`.
    pub fn write_current(&self, current: &Current) -> Result<()> {
        ensure!(
            crate::manifest::is_sha256_hex(&current.sha256),
            "CURRENT needs a lowercase sha256, got '{}'",
            current.sha256
        );
        let text = format!(
            "{} {}\n",
            generation_name(current.generation),
            current.sha256
        );
        atomic_write(&self.current_path(), text.as_bytes())
    }

    /// The overlay file of `CURRENT` (`None` for generation 0 or no `CURRENT`).
    pub fn current_overlay(&self) -> Result<Option<PathBuf>> {
        Ok(match self.read_current()? {
            Some(c) if c.generation > 0 => Some(self.generation_path(c.generation)),
            _ => None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_names_round_trip() {
        assert_eq!(generation_name(7), "g000007");
        assert_eq!(parse_generation_name("g000007"), Some(7));
        assert_eq!(parse_generation_name("g1234567"), Some(1_234_567));
        assert_eq!(parse_generation_name("g00007"), None);
        assert_eq!(parse_generation_name("g0000007"), None);
        assert_eq!(parse_generation_name("x000007"), None);
    }
}
