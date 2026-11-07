# shared settings of the Flash-Next runs on this card
export XDG_RUNTIME_DIR=/tmp CMF_GPU_VRAM_MB=${CMF_GPU_VRAM_MB:-29000} RUST_LOG=${RUST_LOG:-info}
BIN=${BIN:-/root/rip/cortiq.ft}
FILE=${FILE:-/root/rip/fn/qwen38-flash-next-q2tp-v2.cmf}
PROFILE=${PROFILE:-/root/rip/fn/flashnext.profile}
