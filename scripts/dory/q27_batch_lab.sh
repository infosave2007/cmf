#!/bin/bash
# A batched q4tp matvec against the kernels in the engine, on a model's own FFN weights:
# q27_batch_lab.sh MODEL B ARMS [LAYERS], from the repository's root.
# ARMS: name:rows_per_threadgroup:threads,... where name is an entry point of q27_batch_lab.metal,
# or bk (the scalar batched matvec), n8 (the narrow GEMM), bs (the engine's kernel for 2 or 3 vectors).
CMF_LAB_B=$2 CMF_LAB_ARMS=$3 CMF_LAB_LAYERS=${4:-4} CMF_LAB_MSL=$(dirname "$0")/q27_batch_lab.metal CMF_BENCH_MODEL=$1 \
cargo test --release -p cortiq-engine --test metal_mm_bench batch_kernel_lab -- --ignored --nocapture 2>&1 | grep -E "^LAB|MSL compile|error|panicked|program_source" | cut -c1-300
