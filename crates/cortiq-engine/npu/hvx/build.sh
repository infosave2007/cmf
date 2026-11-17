#!/usr/bin/env bash
# Build the cortiq NPU module: the DSP skeleton(s) in a Linux container holding the Hexagon SDK, the Android
# side with the NDK.   HEXAGON_SDK_ROOT=<.../Hexagon_SDK/6.7.0.0>  ./build.sh [v68 v73 ...]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
sdk="${HEXAGON_SDK_ROOT:?set HEXAGON_SDK_ROOT}"
archs=("${@:-v68}")
out="$here/out"; mkdir -p "$out/gen"
img="${HEXSDK_IMAGE:-hexsdk:1}"
docker run --rm --platform linux/amd64 -v "$sdk":/sdk:ro -v "$here":/work -w /work "$img" bash -euc '
  T=/sdk/tools/HEXAGON_Tools/21.0.04/Tools
  /sdk/ipc/fastrpc/qaic/Ubuntu/qaic -mdll -o out/gen -I /sdk/incs -I /sdk/incs/stddef cortiq_hvx.idl
  for a in '"${archs[*]}"'; do
    q=/sdk/rtos/qurt/compute$a/include
    $T/bin/hexagon-clang -m$a -mhvx -mhvx-length=128B -O2 -G0 -fpic -Wall -fno-zero-initialized-in-bss -fdata-sections \
      -I out/gen -I . -I /sdk/incs -I /sdk/incs/stddef -I $q -I $q/qurt -I $q/posix \
      -shared -Wl,-Bsymbolic -Wl,--no-threads -Wl,--build-id -Wl,-L$T/target/hexagon/lib/$a/G0/pic \
      -o out/libcortiq_hvx_skel_$a.so -Wl,-soname,libcortiq_hvx_skel.so dsp/cortiq_hvx_imp.c out/gen/cortiq_hvx_skel.c -lc
  done
  cp out/libcortiq_hvx_skel_'"${archs[0]}"'.so out/libcortiq_hvx_skel.so
'
ndk="${ANDROID_NDK_HOME:-$HOME/Library/Android/sdk/ndk/28.2.13676358}"
cc="$ndk/toolchains/llvm/prebuilt/$(ls "$ndk/toolchains/llvm/prebuilt" | head -1)/bin/aarch64-linux-android26-clang"
inc=(-I "$out/gen" -I "$sdk/incs" -I "$sdk/incs/stddef")
rpc="$sdk/ipc/fastrpc/remote/ship/android_aarch64"
"$cc" -O2 -Wall -fPIC -shared "${inc[@]}" -o "$out/libcortiq_npu.so" host/cq_hvx.c "$out/gen/cortiq_hvx_stub.c" \
  -L "$rpc" -lcdsprpc -Wl,-z,max-page-size=16384
"$cc" -O2 -Wall "${inc[@]}" -o "$out/hvx_test" host/hvx_test.c -L "$out" -lcortiq_npu -lm
"$cc" -O2 -Wall -o "$out/hvx_multi" host/hvx_multi.c -L "$out" -lcortiq_npu
ls -la "$out"/*.so "$out/hvx_test" | awk "{print \$5, \$9}"
