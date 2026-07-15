# Contributing to CMF

Thanks for your interest in improving **CMF (Cortiq Model Format)**. This
document explains how to build the project, the standards we hold changes to,
and how to submit them.

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Report a bug** — open a [bug report](https://github.com/infosave2007/cmf/issues/new?template=bug_report.md).
- **Request a feature** — open a [feature request](https://github.com/infosave2007/cmf/issues/new?template=feature_request.md).
- **Improve docs** — the README and `docs/` are maintained in English, Russian,
  and Chinese; corrections and translations are very welcome.
- **Send code** — bug fixes, new converters, backend improvements, tests.

For anything large (a new backend, an on-disk format change), please open an
issue to discuss the design **before** writing the code, so we can agree on the
approach and avoid wasted work.

## Project layout

```
crates/
  cortiq-core     on-disk format: envelope, sections, quantization codecs
  cortiq-engine   runtime, no ML framework (CPU + optional wgpu GPU)
  cortiq-server   optional axum HTTP serving layer
  cortiq-cli      the `cortiq` command-line binary
converter/        Python converters (source model -> .cmf)
python/           pure-Python reader for inspecting containers
docs/             specification and format comparison (EN / RU / ZH)
tests/            cross-language fixtures and generators
```

## Building

Requirements: a stable Rust toolchain, **1.85 or newer** (the workspace uses
edition 2024).

```bash
cargo build --workspace            # debug build
cargo build --workspace --release  # optimized build
cargo run -p cortiq-cli -- --help  # run the CLI
```

Optional GPU backend (wgpu — Vulkan / Metal / DX12):

```bash
cargo build -p cortiq-engine --features gpu
```

The Python tooling has no third-party runtime dependencies for the reader; the
converters document their own requirements at the top of each script.

## Before you open a pull request

Run the same checks CI runs, locally:

```bash
cargo build --workspace
cargo test  --workspace        # the full suite is self-contained (no model downloads)
cargo clippy --workspace --all-targets
```

- **Tests must pass.** New behavior needs a test; bug fixes should add a
  regression test that fails before the fix.
- **Keep parity gates green.** Numerical changes must preserve the golden-parity
  and round-trip tests in `crates/*/tests/`. If a golden value legitimately
  changes, regenerate the fixture and explain why in the PR.
- **Clippy** is advisory in CI but please address obvious lints in code you touch.
- **Formatting.** The codebase uses a hand-tuned layout rather than stock
  `rustfmt`; match the style of the surrounding code and keep diffs minimal.
- **Comments and identifiers are English.** Test fixtures may contain
  non-English text on purpose (multilingual tokenizer cases).
- **No new mandatory dependencies** in `cortiq-core` / `cortiq-engine` — the
  no-ML-framework runtime is a core property of the project. Discuss first if
  you believe one is unavoidable.

## Commit and PR guidelines

- Write imperative, present-tense commit subjects ("Add q4 SDOT path", not
  "Added"/"Adds"). Keep the subject under ~72 characters.
- One logical change per PR. Separate mechanical refactors from behavior changes.
- Fill in the pull-request template, link the issue it closes, and describe how
  you tested the change.
- Update `CHANGELOG.md` under the `[Unreleased]` heading for any user-visible
  change.

## Licensing of contributions

CMF is licensed under the **Apache License, Version 2.0**. By submitting a
contribution you agree that it is provided under the same license and, per
Apache-2.0 §5, that your submission includes the patent grant described in
§3 of the license. See [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and
[`PATENTS.md`](PATENTS.md).

Please only submit work that is your own or that you have the right to
contribute under these terms.
