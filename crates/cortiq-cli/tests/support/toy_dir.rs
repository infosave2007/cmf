//! The directory of a test binary's shared toy fixture (included with
//! `#[path]`).
//!
//! A toy built once per process behind a `OnceLock` is never dropped, so a
//! `tempfile::TempDir` held there would leave its directory in the system
//! temp dir after every run. The toy lives instead under the target directory
//! (`CARGO_TARGET_TMPDIR`), at a fixed path per test binary, profile and tag,
//! emptied when the process builds it: a run replaces the previous run's files
//! and nothing accumulates outside `target/`. A child process of the same
//! binary (the parent test sets [`TOY_CHILD_ENV`]) uses a path of its own, so
//! the parent's toy is never removed under it.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// Set by a test on the child process it spawns from its own binary.
pub const TOY_CHILD_ENV: &str = "CORTIQ_DECISION_TEST_CHILD";

/// An empty directory for the toy `tag` of this test binary.
pub fn toy_dir(tag: &str) -> PathBuf {
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let child = if std::env::var_os(TOY_CHILD_ENV).is_some() {
        "-child"
    } else {
        ""
    };
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("decision-toys")
        .join(format!(
            "{}-{profile}-{tag}{child}",
            env!("CARGO_CRATE_NAME")
        ));
    match std::fs::remove_dir_all(&dir) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => panic!("remove {}: {e}", dir.display()),
    }
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    dir
}
