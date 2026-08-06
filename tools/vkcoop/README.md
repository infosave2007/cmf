# The tensor-core probe

The q4tp GEMM written twice for the same card: once as the engine runs it
today (WGSL through wgpu, scalar fp32) and once against Vulkan directly,
with `VK_KHR_cooperative_matrix` and f16 16×16 — the shape NVIDIA's tensor
cores actually implement.

    NVIDIA RTX PRO 6000 Blackwell, 9216×2304 weights, 2085 tokens

    wgpu, scalar fp32          25 082 GFLOP/s     3.53 ms
    Vulkan, cooperative f16    62 776 GFLOP/s     1.41 ms   2.50×

The first number this probe produced was 81 315 GFLOP/s, from a kernel
that stored its f32 accumulators into an f16 plane and therefore wrote
nothing: a benchmark with no correctness check measures whatever the
shader felt like doing. `crates/cortiq-engine/tests/vk_coop.rs` now pins
the kernel to `dequant_q4tp` before anyone times it.

It exists to answer one question before a large amount of work is done on
its assumption: is the runtime the ceiling, or is the kernel?

The first answer was "the runtime", and it was wrong. Asking wgpu for
cooperative matrices appeared to make a render forty times slower, which
read as a driver falling off the matrix units. It was not: `request_device`
was *failing* — anything wgpu prefixes with EXPERIMENTAL needs an
`ExperimentalFeatures` token that we were not passing — so the GPU path
never came up at all and the whole render ran on the CPU. With the token,
and with `SHADER_F16` requested, wgpu accepts `coop_mat16x16<f16, A>` with
an f32 accumulator: the shape its own documentation says is unsupported.

Vulkan directly reports what the hardware has:

    M16 N16 K16  f16 × f16 → f32     ← what this probe uses
    M16 N16 K32  i8  × i8  → i32     ← 2× again, and our quantisation
                                        already speaks int8
    M16 N8  K16  f16 × f16 → f32
    ...

## Running it

Needs a Vulkan 1.3 driver with `VK_KHR_cooperative_matrix`, and glslang
≥ 14 to build the shader (11.x, which Ubuntu ships, predates the
extension):

```sh
curl -sL -o g.tgz https://github.com/KhronosGroup/glslang/releases/download/16.5.0/glslang-16.5.0-linux-x86_64-release.tar.gz
mkdir -p glslang && tar xzf g.tgz -C glslang
./glslang/bin/glslang --target-env vulkan1.3 -S comp -o q4tp_coop.spv q4tp_coop.comp
cargo run --release
```

The `.spv` is compiled here rather than at build time on purpose: the
engine promises a pure-Rust install and glslang is a C++ toolchain. A
backend built on this would carry the compiled SPIR-V in the binary.

## What it does not do

It is a benchmark, not a backend. It reads synthetic weights, writes to a
buffer nobody consumes, and skips the rest of a DiT block. Using it for
real means a native Vulkan compute layer beside the wgpu one — device,
memory, descriptors, submission, and the block's other kernels in GLSL —
and the block staying on that device end to end, because mixing two
drivers' memory is worse than porting the kernels.
