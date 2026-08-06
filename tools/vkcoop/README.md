# The tensor-core probe

The q4tp GEMM written twice for the same card: once as the engine runs it
today (WGSL through wgpu, scalar fp32) and once against Vulkan directly,
with `VK_KHR_cooperative_matrix` and f16 16×16 — the shape NVIDIA's tensor
cores actually implement.

    NVIDIA RTX PRO 6000 Blackwell, 9216×2304 weights, 2085 tokens

    wgpu, scalar fp32          25 082 GFLOP/s     3.53 ms
    Vulkan, cooperative f16    81 315 GFLOP/s     1.09 ms   3.24×

It exists to answer one question before a large amount of work is done on
its assumption: is the runtime the ceiling, or is the kernel? It is the
runtime. wgpu 30 exposes exactly one cooperative shape, 8×8 f32, which
this driver does not implement on the matrix units — asked for it, a
render went from 0.68 s a step to 52. Vulkan directly reports what the
hardware has:

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
