//! State directory layout and the `LOCK` (spec §4.14, §4.11).
//!
//! ```text
//! <state>/            mode 0700
//!   LOCK              pid and a random nonce; flock(2)-held by the one serving process
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
//! [`StateLock`] ([`StateDir::lock`]): an advisory `flock(2)` on `LOCK`, held
//! for the life of the guard, with `pid nonce` in the file to name the holder.
//! The kernel drops a flock when its process ends, however it ends (SIGKILL
//! after a stop timeout, an out-of-memory kill, a crash of the host), so a
//! `LOCK` no process holds is stale: the next lock attempt takes it over and
//! warns with the pid it was left by. No pid check is involved (in a
//! container the server is always pid 1, so a pid could never tell). A second
//! lock attempt fails with the holder's pid while the holder runs; a held lock
//! is never broken. Where the filesystem has no advisory locks (Windows, some
//! network mounts) the file alone (`O_EXCL`) says the directory is taken, and
//! `--break-lock` ([`StateDir::lock`] with `break_lock`) removes a lock left by
//! a dead process. Versions up to 0.7.8 only create the file: one of them
//! refuses a directory while a newer process holds it, but a newer process
//! takes over from a running old one: do not run the two on one directory at
//! the same time. Key management from the CLI writes `keys.json` atomically
//! without the `LOCK`, under `keys.json.lock` like the server's own key
//! changes (neither writes back a file the other changed in between); a
//! server picks the new file up by its mtime
//! ([`crate::keys::KeyStore::maybe_reload`]).
//!
//! [`atomic_write`]: temp file in the same directory (`create_new`, mode 0600),
//! write, `fsync`, `rename` over the target, `fsync` of the directory.

use anyhow::{Context, Result, bail, ensure};
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const LOCK_FILE: &str = "LOCK";
/// Lock attempts before giving up on a `LOCK` that keeps being removed and
/// created again under this process.
const LOCK_ATTEMPTS: usize = 8;
pub const KEYS_FILE: &str = "keys.json";
pub const USAGE_DIR: &str = "usage";
pub const ORACLE_LEDGER_FILE: &str = "oracle.jsonl";
pub const ORACLE_STATE_FILE: &str = "oracle.state";
pub const LEARN_LOG_FILE: &str = "learn.log";
pub const GENERATIONS_DIR: &str = "generations";
pub const CURRENT_FILE: &str = "CURRENT";

/// A held state directory lock: the `flock` on `LOCK` lasts as long as this
/// guard, and at most as long as the process; the file is removed on drop
/// (only while it is still ours).
#[derive(Debug)]
pub struct StateLock {
    path: PathBuf,
    content: String,
    /// The open `LOCK`, whose flock is the lock: closed only after the file
    /// is removed (fields drop after [`Drop::drop`]).
    _file: File,
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Locked {
    pub dir: String,
    pub lock: String,
    pub pid: String,
    /// A running process holds the lock's flock: it is never broken. `false`
    /// where the filesystem has no advisory locks and the file alone says the
    /// directory is taken (`--break-lock` removes it there).
    pub held: bool,
}

impl std::fmt::Display for Locked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { dir, lock, pid, .. } = self;
        if self.held {
            write!(
                f,
                "state directory {dir} is locked by pid {pid} ({lock}), a running process that holds it; one process per state directory — stop that process first (a held lock is never broken, --break-lock included)"
            )
        } else {
            write!(
                f,
                "state directory {dir} is locked by pid {pid} ({lock}); one process per state directory — stop it, or remove a stale lock with --break-lock (the directory's filesystem has no advisory locks, so a lock left by a crash stays until then)"
            )
        }
    }
}

impl std::error::Error for Locked {}

/// What a lock attempt may remove where the filesystem has no advisory locks
/// and the `LOCK` file alone says the directory is taken.
#[derive(Clone, Copy, Debug)]
enum Break<'a> {
    No,
    /// `--break-lock`: the lock found there.
    Found,
    /// The stale lock the caller checked (its content).
    Checked(&'a str),
}

/// Take an exclusive `flock(2)` on `file` without waiting: `Ok(true)` taken,
/// `Ok(false)` another open `LOCK` holds it, `Err` when the filesystem (or
/// the platform) has no advisory locks.
#[cfg(unix)]
fn try_flock(file: &File) -> std::io::Result<bool> {
    use std::os::unix::io::AsRawFd;
    loop {
        // SAFETY: flock(2) on a descriptor `file` owns; it only changes the
        // lock of that open file description.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
            return Ok(true);
        }
        let e = std::io::Error::last_os_error();
        match e.raw_os_error() {
            Some(c) if c == libc::EWOULDBLOCK || c == libc::EAGAIN => return Ok(false),
            Some(libc::EINTR) => {}
            _ => return Err(e),
        }
    }
}

#[cfg(not(unix))]
fn try_flock(_file: &File) -> std::io::Result<bool> {
    Err(std::io::Error::from(ErrorKind::Unsupported))
}

/// Whether `path` still names the file `file` has open: a `LOCK` removed (its
/// holder released it) or replaced between the open and the flock is a lock
/// no other process sees.
#[cfg(unix)]
fn still_named(path: &Path, file: &File) -> std::io::Result<bool> {
    use std::os::unix::fs::MetadataExt;
    let open = file.metadata()?;
    match fs::metadata(path) {
        Ok(m) => Ok(m.dev() == open.dev() && m.ino() == open.ino()),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

#[cfg(not(unix))]
fn still_named(_path: &Path, _file: &File) -> std::io::Result<bool> {
    Ok(true)
}

/// Replace the whole content of the open `LOCK` with `content`, durably.
fn write_lock(file: &mut File, content: &str) -> std::io::Result<()> {
    file.set_len(0)?;
    file.seek(SeekFrom::Start(0))?;
    file.write_all(content.as_bytes())?;
    file.sync_all()
}

/// The pid of a `LOCK`'s content (`pid nonce`), `unknown` when empty.
fn holder_pid(content: &str) -> String {
    content
        .split_whitespace()
        .next()
        .unwrap_or("unknown")
        .to_string()
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

    /// Take the exclusive `LOCK`: its `flock`, content `pid nonce`. A `LOCK`
    /// no process holds (its process ended without removing it) is taken over
    /// with a warning naming the pid it was left by; one a running process
    /// holds fails with [`Locked`] (`held`), whatever `break_lock` says. Where
    /// the filesystem has no advisory locks the file is created with `O_EXCL`
    /// and an existing one fails with [`Locked`] (not `held`), unless
    /// `break_lock` removes it first (only while it still holds what was
    /// found, see [`StateDir::lock_replacing`]).
    pub fn lock(&self, break_lock: bool) -> Result<StateLock> {
        let b = if break_lock { Break::Found } else { Break::No };
        self.take(b, try_flock)
    }

    /// [`StateDir::lock`] in place of the stale lock the caller checked, whose
    /// content was `stale`. With advisory locks this is `lock(false)`: a lock
    /// no process holds is stale whatever it says. Without them the file is
    /// removed only while it still holds `stale`, so a lock another process
    /// took meanwhile (two runs breaking the same stale lock at once) is never
    /// removed — this call then fails with [`Locked`] for it.
    pub fn lock_replacing(&self, stale: &str) -> Result<StateLock> {
        self.take(Break::Checked(stale), try_flock)
    }

    /// [`StateDir::lock`] with the flock primitive given (tests stand in for
    /// a filesystem without advisory locks).
    fn take(&self, brk: Break<'_>, flock: fn(&File) -> std::io::Result<bool>) -> Result<StateLock> {
        let path = self.lock_path();
        let content = format!("{} {}\n", std::process::id(), nonce()?);
        let mut broken = false;
        for _ in 0..LOCK_ATTEMPTS {
            let (mut file, created) = match create_new_private(&path) {
                Ok(f) => (f, true),
                Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                    match OpenOptions::new().read(true).write(true).open(&path) {
                        Ok(f) => (f, false),
                        // Removed in between (its holder released it): again.
                        Err(e) if e.kind() == ErrorKind::NotFound => continue,
                        Err(e) => {
                            return Err(e).with_context(|| format!("open {}", path.display()));
                        }
                    }
                }
                Err(e) => return Err(e).with_context(|| format!("create {}", path.display())),
            };
            match flock(&file) {
                Ok(true) => {
                    if !still_named(&path, &file)
                        .with_context(|| format!("stat {}", path.display()))?
                    {
                        continue;
                    }
                    let mut left = String::new();
                    if !created {
                        let _ = file.read_to_string(&mut left);
                    }
                    write_lock(&mut file, &content)
                        .with_context(|| format!("write {}", path.display()))?;
                    sync_dir(&self.root)?;
                    if !left.trim().is_empty() {
                        tracing::warn!(
                            lock = %path.display(),
                            "took over the LOCK of state directory {} left by pid {}, which ended without releasing it (killed, out of memory or a crash)",
                            self.root.display(),
                            holder_pid(&left)
                        );
                    }
                    return Ok(StateLock {
                        path,
                        content,
                        _file: file,
                    });
                }
                Ok(false) => return Err(self.locked(true).into()),
                Err(no_locks) => {
                    // No advisory locks here: the file alone (O_EXCL) says
                    // the directory is taken.
                    if created {
                        write_lock(&mut file, &content)
                            .with_context(|| format!("write {}", path.display()))?;
                        sync_dir(&self.root)?;
                        if cfg!(unix) {
                            tracing::warn!(
                                lock = %path.display(),
                                "no advisory lock on the LOCK of state directory {} ({no_locks}): a LOCK left by a crash stays until --break-lock removes it",
                                self.root.display()
                            );
                        }
                        return Ok(StateLock {
                            path,
                            content,
                            _file: file,
                        });
                    }
                    drop(file);
                    let stale = match brk {
                        Break::No => None,
                        Break::Found => Some(fs::read_to_string(&path).unwrap_or_default()),
                        Break::Checked(s) => Some(s.to_string()),
                    };
                    match stale {
                        Some(stale) if !broken => {
                            broken = true;
                            self.remove_if_still(&stale)?;
                        }
                        _ => return Err(self.locked(false).into()),
                    }
                }
            }
        }
        bail!(
            "could not take {}: it was removed and created again {LOCK_ATTEMPTS} times while this process tried",
            path.display()
        )
    }

    /// [`Locked`] with the pid the `LOCK` names now.
    fn locked(&self, held: bool) -> Locked {
        let path = self.lock_path();
        let mut holder = String::new();
        let _ = File::open(&path).and_then(|mut f| f.read_to_string(&mut holder));
        Locked {
            dir: self.root.display().to_string(),
            lock: path.display().to_string(),
            pid: holder_pid(&holder),
            held,
        }
    }

    /// Remove the `LOCK` only while it holds `stale` (a filesystem without
    /// advisory locks). The file is moved aside by an atomic rename before
    /// its content is compared; a lock that is not the stale one is put back
    /// without replacing a newer one.
    fn remove_if_still(&self, stale: &str) -> Result<()> {
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
        Ok(())
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

    /// A filesystem without advisory locks.
    fn no_flock(_: &File) -> std::io::Result<bool> {
        Err(std::io::Error::from(ErrorKind::Unsupported))
    }

    #[test]
    fn without_advisory_locks_the_file_alone_is_the_lock() {
        let dir = tempfile::tempdir().unwrap();
        let s = StateDir::open(dir.path().join("st")).unwrap();
        let aside = || {
            fs::read_dir(s.root())
                .unwrap()
                .filter(|e| {
                    e.as_ref()
                        .unwrap()
                        .file_name()
                        .to_string_lossy()
                        .starts_with("LOCK.")
                })
                .count()
        };
        let l1 = s.take(Break::No, no_flock).unwrap();
        let held = fs::read_to_string(s.lock_path()).unwrap();
        let e = s.take(Break::No, no_flock).unwrap_err();
        let l = e.downcast_ref::<Locked>().unwrap();
        assert!(!l.held && l.pid == std::process::id().to_string(), "{l:?}");
        assert!(format!("{e:#}").contains("--break-lock"), "{e:#}");
        // Breaking only the lock that was checked: a stale content that is no
        // longer the file's leaves the lock in place and fails with its holder.
        let e = s
            .take(Break::Checked("99999999 0123abcd\n"), no_flock)
            .unwrap_err();
        assert!(e.downcast_ref::<Locked>().is_some(), "{e:#}");
        assert_eq!(fs::read_to_string(s.lock_path()).unwrap(), held);
        assert_eq!(aside(), 0, "nothing is left aside");
        // --break-lock takes it over; the old guard does not remove the new lock.
        let l2 = s.take(Break::Found, no_flock).unwrap();
        drop(l1);
        assert!(
            s.lock_path().exists(),
            "the broken guard removed the new lock"
        );
        assert!(s.take(Break::No, no_flock).is_err());
        drop(l2);
        assert!(!s.lock_path().exists());
        // The checked stale lock is replaced.
        fs::write(s.lock_path(), "99999999 0123abcd\n").unwrap();
        let l3 = s
            .take(Break::Checked("99999999 0123abcd\n"), no_flock)
            .unwrap();
        assert_ne!(
            fs::read_to_string(s.lock_path()).unwrap(),
            "99999999 0123abcd\n"
        );
        drop(l3);
        assert!(!s.lock_path().exists());
        assert_eq!(aside(), 0, "nothing is left aside");
    }
}
