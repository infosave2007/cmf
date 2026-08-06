#!/bin/bash
# Compile the engine's Vulkan shaders to SPIR-V and drop them where
# `include_bytes!` expects them. glslang must be >= 14 — the 11.x that
# distributions ship predates GL_KHR_cooperative_matrix and rejects the
# first line of the shader.
set -e
GLSLANG=${GLSLANG:-glslang}
SRC=$(cd "$(dirname "$0")/../../crates/cortiq-engine/src/shaders" && pwd)
for f in "$SRC"/*.comp; do
  "$GLSLANG" --target-env vulkan1.3 -S comp -o "${f%.comp}.spv" "$f"
  echo "$(basename "${f%.comp}").spv"
done
