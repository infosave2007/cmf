# cortiq-npu shim

`cortiq_npu.cpp` is a C ABI over Qualcomm's QNN HTP backend, used by
`src/npu.rs` when `CMF_NPU=1` on Android. It is not part of the cargo build:
it needs the QAIRT SDK headers (Qualcomm AI Runtime, Community edition).

```
CXX=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/<host>/bin/aarch64-linux-android26-clang++
$CXX -O2 -std=c++17 -fPIC -shared -I $QAIRT/include/QNN -o libcortiq_npu.so \
    cortiq_npu.cpp -ldl -static-libstdc++ -Wl,-z,max-page-size=16384
$CXX -O2 -std=c++17 -o npu_test npu_test.cpp -L. -lcortiq_npu -static-libstdc++
```

On the device, next to the binary: `libQnnHtp.so`, `libQnnHtpPrepare.so`,
`libQnnHtpV68Stub.so`, `libQnnSystem.so` (from `lib/aarch64-android`) and
`libQnnHtpV68Skel.so`, `libQnnHtpV68.so` (from `lib/hexagon-v68/unsigned`),
with `LD_LIBRARY_PATH` and `ADSP_LIBRARY_PATH` pointing at them.

`npu_test <rows> <cols> <alpha>` checks one matvec against a double-precision
reference and times it. Snapdragon 778G (Hexagon V68): 8192x2048 in 1.44 ms,
relative L2 error 1.2e-3 at alpha 1.2, 2.0e-3 at 2, 8.1e-3 at 8.

Knobs read by the engine: `CMF_NPU=1`, `CMF_NPU_LIB`, `CMF_NPU_QNN`,
`CMF_NPU_ALPHA` (0.5), `CMF_NPU_MIN_M` (2: smallest matrix, millions of
weights), `CMF_NPU_MB` (1024), `CMF_NPU_PROF=1`.
