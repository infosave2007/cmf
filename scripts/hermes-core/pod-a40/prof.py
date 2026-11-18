"""Per-operation profile of the Embryo CUDA trainer.

    CMF_VULKAN_OP_TS=1 cortiq-embryo step-bench ... 2> prof.log
    python3 prof.py prof.log [N]

Reads the `cuda commit-ts` / `cuda op-ts` / `cuda gemm-ts` lines the trainer
prints per commit with CMF_VULKAN_OP_TS=1 and prints, per training step: the
GEMM time by class, the N heaviest GEMM shapes with their nominal TFLOPS
(a dynamic-K / dynamic-row GEMM is counted with its full shape) and the
other operations. An operation on a dedicated entry point is listed as
op + 1000 * entry.
"""

import collections
import re
import sys

NAMES = {
    1: "GEMM", 2: "AXPBY", 3: "ADAMW", 4: "SUMSQ", 5: "RMS_FWD", 6: "RMS_BWD", 7: "SWIGLU_FWD", 8: "SWIGLU_BWD",
    9: "EMBED_GATHER", 10: "EMBED_SCATTER", 11: "SOFTMAX_CE", 12: "COPY", 13: "SLICE", 14: "PAD", 15: "DOT",
    16: "SIGMOID_FWD", 17: "SIGMOID_BWD", 18: "ROPE", 19: "CAUSAL", 20: "SOFTMAX_BWD", 21: "GATHER", 22: "SCATTER",
    23: "CONV_FWD", 24: "CONV_BWD", 31: "ROUTE", 32: "ROUTE_GROUP", 33: "MOE_GATHER", 34: "MOE_SCATTER",
    35: "MOE_STATS", 36: "MOE_UPDATE", 46: "MOE_INDIRECT", 47: "MOE_CENTER", 49: "HEAD_GROUP_FWD",
    70: "WY_DU", 72: "WY_DGEND", 73: "WY_MASK_BWD_RED", 76: "WY_DSCAL", 85: "WY_DLG_DOT",
    5006: "RMS_BWD dx", 6006: "RMS_BWD dw", 10020: "SOFTMAX_BWD rows",
}


def main() -> None:
    gem = collections.defaultdict(lambda: [0.0, 0])
    ops = collections.defaultdict(lambda: [0.0, 0])
    steps = 0
    # only the training-step commits: the largest commits of the log (an
    # end-of-run held-out pass commits fewer operations and is left out)
    sizes = [int(x) for x in re.findall(r"cuda commit-ts ops=(\d+)", open(sys.argv[1]).read())]
    big = max(sizes) if sizes else 0
    live = False
    for line in open(sys.argv[1]):
        m = re.match(r"cuda commit-ts ops=(\d+) span_ms=([\d.]+)", line)
        if m:
            live = int(m.group(1)) == big
            steps += live
        if not live:
            continue
        m = re.match(r"cuda op-ts op=(\d+) count=(\d+) gpu_ms=([\d.]+)", line)
        if m:
            ops[int(m.group(1))][0] += float(m.group(3))
            ops[int(m.group(1))][1] += int(m.group(2))
        m = re.match(
            r"cuda gemm-ts entry=(\d+) t=(\d+) m=(\d+) n=(\d+) k=(\d+) batches=(\d+) tile=(\d+) count=(\d+) gpu_ms=([\d.]+)",
            line,
        )
        if m:
            k = tuple(int(x) for x in m.groups()[:7])
            gem[k][0] += float(m.group(9))
            gem[k][1] += int(m.group(8))
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    tot = sum(v[0] for v in ops.values())
    gt = sum(v[0] for v in gem.values())
    print(f"steps {steps}; per step: total {tot / steps:.0f} ms, GEMM {gt / steps:.0f} ms, other {(tot - gt) / steps:.0f} ms")
    by = collections.defaultdict(float)
    for k, (ms, _) in gem.items():
        by["cublas" if k[0] == 100 else "kernels"] += ms
    print("GEMM per step:", {c: round(v / steps) for c, v in by.items()})
    for k, (ms, c) in sorted(gem.items(), key=lambda x: -x[1][0])[:top]:
        fl = 2 * k[2] * k[3] * k[4] * k[5] * c / 1e12
        print(
            f"  entry {k[0]:>3} t={k[1]} m={k[2]:>6} n={k[3]:>6} k={k[4]:>6} b={k[5]:>4} x{c // steps:>4}/step"
            f" {ms / steps:7.1f} ms/step {fl / (ms / 1e3):6.1f} TFLOPS"
        )
    print("other ops per step:")
    for op, (ms, c) in sorted(ops.items(), key=lambda x: -x[1][0])[:22]:
        if op != 1:
            print(f"  op {op:>5} {NAMES.get(op, str(op)):<17} {ms / steps:7.1f} ms  x{c // steps}")


if __name__ == "__main__":
    main()
