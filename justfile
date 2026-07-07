# CMF — developer shortcuts for `just` (https://github.com/casey/just).
# Run `just` or `just --list` to see all recipes.
# (A Makefile with the same recipes is provided for `make` users.)

# Show available recipes
default:
    @just --list

# Debug build of the whole workspace
build:
    cargo build --workspace

# Optimized release build
release:
    cargo build --workspace --release

# Run the full test suite
test:
    cargo test --workspace

# Format the code
fmt:
    cargo fmt --all

# Check formatting without modifying files
fmt-check:
    cargo fmt --all -- --check

# Run the linter
clippy:
    cargo clippy --workspace --all-targets

# Fast type-check
check:
    cargo check --workspace

# Build API docs
doc:
    cargo doc --workspace --no-deps

# Run the CLI, e.g. `just run --help`
run *ARGS:
    cargo run -p cortiq-cli -- {{ARGS}}

# Build the engine with the wgpu GPU backend
gpu-build:
    cargo build -p cortiq-engine --features gpu

# Dry-run crates.io packaging for cortiq-core
publish-dry:
    cargo publish --dry-run -p cortiq-core

# Remove build artifacts
clean:
    cargo clean
