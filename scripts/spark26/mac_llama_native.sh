#!/bin/bash
# mac_llama_native.sh DIR SCRIPTS CMAKE: llama.cpp built for arm64 on the Mac (the first build came
# out x86_64 from an Intel-Homebrew cmake and ran under Rosetta), then the final suite on it.
D=$1; S=$2; CM=$3; cd "$D/cmp/llama.cpp" || exit 1
"$CM" -B build-arm64 -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.5-a+dotprod+i8mm -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_SERVER=OFF > "$D/llama_arm64_cfg.log" 2>&1 || { echo "configure failed"; tail -20 "$D/llama_arm64_cfg.log"; exit 1; }
"$CM" --build build-arm64 --config Release -j8 --target llama-completion llama-perplexity llama-bench llama-tokenize > "$D/llama_arm64_build.log" 2>&1 || { echo "build failed"; grep -iE "error" "$D/llama_arm64_build.log" | head -20; exit 1; }
file build-arm64/bin/llama-completion
cd "$D"
python3 -c "import time; time.sleep(120)"
exec bash "$S/mac_suite3.sh" "$D" "$S"
