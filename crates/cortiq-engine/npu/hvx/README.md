# cortiq-npu: the Hexagon module

Our own cDSP module for Snapdragon phones (Hexagon v68 and newer, unsigned
PD), used by `src/npu.rs` when the NPU lane is asked for. Integer HVX only, so
it runs on DSPs without HMX. About 18 KB on the DSP and 15 KB on Android; no
vendor runtime is shipped.

| file | what |
|---|---|
| `cortiq_hvx.idl` | the FastRPC interface: `ping`, `q4_matvec` (one or two matrices), `q4_gemm` (a prompt chunk) |
| `dsp/cortiq_hvx_imp.c` | the DSP side: the tiled 4-bit kernels, the clock vote, HVX threads |
| `host/cq_hvx.c` | the Android side: `cq_npu_init/alloc/add/run/gemm`, the C ABI `npu.rs` loads |
| `host/hvx_test.c`, `host/hvx_multi.c` | accuracy and speed against a reference; several matrices, a second thread |
| `build.sh` | the DSP side in a Linux container holding the Hexagon SDK, the Android side with the NDK |

## Building

Needs the Qualcomm Hexagon SDK 6.7 (Community edition, Linux), Docker and the
Android NDK. The SDK's tools are x86-64 Linux binaries; `build.sh` runs them
in an `ubuntu:22.04` image with `libncurses5 libtinfo5 libatomic1` (tag it
`hexsdk:1` or set `HEXSDK_IMAGE`).

```
HEXAGON_SDK_ROOT=<...>/Hexagon_SDK/6.7.0.0 ./build.sh v68
```

`out/libcortiq_hvx_skel.so` goes where the DSP can read it
(`ADSP_LIBRARY_PATH`; in an app, the native library directory, extracted),
`out/libcortiq_npu.so` beside the engine library.

## The layout and the kernel

A matrix's rows are tiled by 32. One record (640 bytes) is one tile's group
of 32 inputs: four 128-byte vectors of codes — vector `j` holds, in lane `r`,
bytes `4j..4j+3` of row `r`'s 16-byte group — and one of scales, a Q15 ratio
of the row's largest in the upper half of each word. The codes are q4tp's own
bytes, moved, not requantized.

`vrmpy(codes, four int8 inputs)` leaves each row's partial sum in its own
word, so 32 rows advance together with no horizontal reduction; a group takes
eight such products, a subtraction of `8·Σx` and one fractional multiply by
the scales. `q4_gemm` splits a group's codes once and spends them on eight
activation vectors.

Three things the DSP does not forgive, all found on a Snapdragon 778G:

- no hardware prefetch: without `l2fetch` a 7680x2048 matvec takes 4.1 ms,
  with it 0.69 ms;
- an `l2fetch` of an unmapped page is a fault that kills the PD, not a no-op:
  the fetch must stop at the matrix's end;
- without a persistent `fastrpc_mmap` every call maps and unmaps the matrix
  (0.75 ms on a 10 MB buffer); mapped, the buffer is reached by its fd
  (`HAP_mmap_get`), not passed as an argument.

## Measured (Snapdragon 778G, Hexagon V68, LFM3-1.7B q4tp)

- matvec 7680x2048: 0.69 ms on the DSP with one HVX thread (memory-bound: a
  second thread adds nothing), 0.92-1.12 ms with the round trip; the CPU's
  four big cores take 0.99 ms. Relative error against a double-precision
  reference: 1e-6.
- `q4_gemm`, the same matrix: 0.43 ms per vector with one thread, 0.23 with
  two (compute-bound; v68 has two HVX units).
- whole model, a 256-token prompt, decode after it unchanged at 11.8
  tokens/s: 22.7 tokens/s on the CPU; with the lane 26.4 / 30.9 / 35.6 within
  256 / 440 / 640 MB of weights copied to the DSP.
- botpsy-0.8B (qwen3.5): 60.4 -> 65.0; LFM2.5-230M: no gain.
- decode: the split is slower or level (the token is bound by DDR, which the
  CPU and the DSP share), so it is off unless `CMF_NPU_DECODE=1`.

## Knobs

`CMF_NPU=1` / `0`, `CMF_NPU_SHARE` (0.65: the DSP's share of a matrix's
rows), `CMF_NPU_MB` (default: a fifth of the available memory, at most 512),
`CMF_NPU_MIN_M` (2: smallest matrix, millions of weights),
`CMF_NPU_BATCH_THREADS` (2), `CMF_NPU_THREADS` (1), `CMF_NPU_DECODE=1`,
`CMF_NPU_PROBE=0` (no trial, always use the NPU) / `=1` (measure again),
`CMF_NPU_MARGIN` (0.95), `CMF_NPU_LOG=1`, `CMF_NPU_PROF=1`, `CMF_NPU_PIN=0`,
`CMF_NPU_LIB`.
