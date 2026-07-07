# CMF — developer shortcuts. Run `make help` for the list.
# (A justfile with the same recipes is provided for `just` users.)
.DEFAULT_GOAL := help
CARGO ?= cargo

.PHONY: help build release test fmt fmt-check clippy check doc run gpu-build publish-dry clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

build: ## Debug build of the whole workspace
	$(CARGO) build --workspace

release: ## Optimized release build
	$(CARGO) build --workspace --release

test: ## Run the full test suite
	$(CARGO) test --workspace

fmt: ## Format the code
	$(CARGO) fmt --all

fmt-check: ## Check formatting without modifying files
	$(CARGO) fmt --all -- --check

clippy: ## Run the linter
	$(CARGO) clippy --workspace --all-targets

check: ## Fast type-check
	$(CARGO) check --workspace

doc: ## Build API docs
	$(CARGO) doc --workspace --no-deps

run: ## Run the CLI, e.g. `make run ARGS="--help"`
	$(CARGO) run -p cortiq-cli -- $(ARGS)

gpu-build: ## Build the engine with the wgpu GPU backend
	$(CARGO) build -p cortiq-engine --features gpu

publish-dry: ## Dry-run crates.io packaging for cortiq-core
	$(CARGO) publish --dry-run -p cortiq-core

clean: ## Remove build artifacts
	$(CARGO) clean
