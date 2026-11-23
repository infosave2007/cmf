# kbench

Times the skill bake's WGSL kernels on a card and checks them against f64 (or against
another kernel), without rebuilding the engine: a shader edit is a rerun. Built apart from
the workspace (`cargo build --release` here); it takes the engine's own sources by name.

```sh
cd scripts/kbench && cargo build --release
K=target/release/kbench
export TRUSTED=1   # the engine builds these kernels without naga's bounds checks

# GEMM: nt|nn, source, entry, NB COLS ROWS, tile 64×64, optional split-K span
$K gemm nt @F32T_NT_SRC gemm_nt_f32t 256 1024 6144 64 64
$K gemm nn @F32T_NN_SRC gemm_nn_f32t 256 6144 1024 64 64 896

# attention core: b t nh nkv hd
$K attnfwd @BAKE_ATTN_SG_SRC bake_attn_head 4 64 16 8 128
$K attnbwd @BAKE_ATTN_BWD_Q_SG_SRC bake_attn_bwd_q @BAKE_ATTN_BWD_KV_SG_SRC bake_attn_bwd_kv 4 64 16 8 128

# two kernels on the same inputs (grids of their own), outputs compared: here the scalar and
# the subgroup q/k norm + RoPE (n 256, 16:8 heads of 128, RoPE half 64, q and k norms)
$K cmp @BAKE_QKR_SRC bake_qkr 128,1,1 @BAKE_QKR_SG_SRC bake_qkr 4096,1,1 \
  256,64,16,8,128,2048,64,6,f:0.000001,0,0,0 5,6,7,8,9,10 \
  1048576 256 256 4096 8192 524288 262144 262144 524288 4096 2048

# any kernel, timing only; the card's FMA peak
$K raw SRC ENTRY GX GY GZ U0,U1,… N0 N1 …
$K peak my.wgsl peak 4096 20000
```

`@NAME` is the raw string `const NAME` of `crates/cortiq-engine/src/gpu_wgpu.rs`; point
`KBENCH_SRC` at another file (`gpu_wgpu/bake_res.rs` for the resident step's kernels) or
pass a `.wgsl` path. `NOZERO=1` skips workgroup-memory zeroing, `METAL=1` takes the Metal
backend (Vulkan otherwise).
