#!/usr/bin/env python3
"""Cheap regression test for the independent Q6_K checker.

The fixture is intentionally unlike a real model block: all 16 signed scale
bytes are nonuniform and ql/qh contain varying bit patterns.  A small C
oracle is compiled from the upstream GGML dequantize_row_q6_K loop, then all
256 values are compared with the Python checker.  No model, NumPy, or Rust
build is needed.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from check_qwen_image_component import q6k_value, sample_windows  # noqa: E402


UPSTREAM_SOURCE = (
    "https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.c#L1823-L1851"
)
Q6K_ELEMENTS = 256
Q6K_BYTES = 210


def c_array(values: list[int], *, unsigned: bool) -> str:
    if unsigned:
        return ", ".join("0x%02x" % (value & 0xFF) for value in values)
    return ", ".join(str(value) for value in values)


def build_oracle_source(ql: list[int], qh: list[int], scales: list[int]) -> str:
    return f"""
#include <stdint.h>
#include <stdio.h>

enum {{ QK_K = 256 }};
typedef struct {{
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
}} block_q6_K_ref;

/* Transcribed from GGML's dequantize_row_q6_K; the fixture has d=1.0. */
static void dequantize_row_q6_K_ref(const block_q6_K_ref * x, float * y) {{
    const float d = 1.0f;
    const uint8_t * ql = x->ql;
    const uint8_t * qh = x->qh;
    const int8_t  * sc = x->scales;
    for (int n = 0; n < QK_K; n += 128) {{
        for (int l = 0; l < 32; ++l) {{
            int is = l/16;
            const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[n + l +  0] = d * sc[is + 0] * q1;
            y[n + l + 32] = d * sc[is + 2] * q2;
            y[n + l + 64] = d * sc[is + 4] * q3;
            y[n + l + 96] = d * sc[is + 6] * q4;
        }}
        ql += 64;
        qh += 32;
        sc += 8;
    }}
}}

int main(void) {{
    static const block_q6_K_ref x = {{
        {{ {c_array(ql, unsigned=True)} }},
        {{ {c_array(qh, unsigned=True)} }},
        {{ {c_array(scales, unsigned=False)} }},
        0x3c00
    }};
    float y[QK_K];
    dequantize_row_q6_K_ref(&x, y);
    for (int i = 0; i < QK_K; ++i) {{
        printf("%.9g\\n", (double)y[i]);
    }}
    return 0;
}}
"""


def compile_and_run(source: str) -> tuple[str, list[float]]:
    compiler_spec = os.environ.get("CC", "cc")
    compiler = shlex.split(compiler_spec)
    if not compiler or shutil.which(compiler[0]) is None:
        raise RuntimeError("C compiler not found; set CC to a local C99 compiler")
    with tempfile.TemporaryDirectory(prefix="q6k-ggml-check-") as temp:
        executable = Path(temp) / "oracle"
        command = compiler + [
            "-std=c99", "-O2", "-Wall", "-Wextra", "-pedantic",
            "-x", "c", "-o", str(executable), "-",
        ]
        result = subprocess.run(
            command,
            input=source.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                "C oracle compilation failed (exit %d): %s"
                % (result.returncode, result.stderr.decode("utf-8", "replace"))
            )
        result = subprocess.run(
            [str(executable)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                "C oracle failed (exit %d): %s"
                % (result.returncode, result.stderr.decode("utf-8", "replace"))
            )
        try:
            values = [float(line) for line in result.stdout.split()]
        except ValueError as exc:
            raise RuntimeError("C oracle emitted non-numeric output") from exc
        display_command = " ".join(
            compiler + ["-std=c99", "-O2", "-Wall", "-Wextra", "-pedantic",
                        "-x", "c", "-o", "<temporary-oracle>", "-"]
        )
        return display_command, values


def check_sampling() -> dict[str, object]:
    cases = {}
    for count in (256, 512, 1024, 4096, 100_000):
        windows = sample_windows(count, 1024)
        indices = [i for start, size in windows for i in range(start, start + size)]
        if count > 1024:
            assert len(indices) == 1024
            assert len(windows) == 4
            assert all(size == 256 for _, size in windows)
        assert all(start % 256 == 0 for start, _ in windows)
        assert indices == sorted(set(indices))
        assert all(0 <= i < count for i in indices)
        cases[str(count)] = {
            "windows": [{"start": start, "count": size} for start, size in windows],
            "sample_count": len(indices),
        }
    return cases


def main() -> int:
    ql = [(37 * i + 11) & 0xFF for i in range(128)]
    qh = [(53 * i + 7) & 0xFF for i in range(64)]
    scales = [127, -127, 63, -63, 31, -31, 15, -15,
              7, -7, 5, -5, 3, -3, 1, -1]
    block = bytes(ql) + bytes(qh) + bytes(value & 0xFF for value in scales) + b"\x00\x3c"
    assert len(block) == Q6K_BYTES

    try:
        command, oracle = compile_and_run(build_oracle_source(ql, qh, scales))
        got = [q6k_value(memoryview(block), index) for index in range(Q6K_ELEMENTS)]
        if len(oracle) != Q6K_ELEMENTS:
            raise RuntimeError("C oracle emitted %d values, expected %d" %
                               (len(oracle), Q6K_ELEMENTS))
        differences = [abs(actual - expected) for actual, expected in zip(got, oracle)]
        mismatches = [
            index for index, (actual, expected) in enumerate(zip(got, oracle))
            if not math.isclose(actual, expected, rel_tol=2e-6, abs_tol=2e-4)
        ]
        if mismatches:
            raise RuntimeError(
                "Q6_K mismatch at %d/%d positions; first=%d python=%r c=%r"
                % (len(mismatches), Q6K_ELEMENTS, mismatches[0],
                   got[mismatches[0]], oracle[mismatches[0]])
            )
        receipt = {
            "status": "PASS",
            "upstream_source": UPSTREAM_SOURCE,
            "oracle": {
                "compiler_command": command,
                "elements": Q6K_ELEMENTS,
                "block_bytes": Q6K_BYTES,
                "ql_pattern": "(37*i+11)&0xff for i=0..127",
                "qh_pattern": "(53*i+7)&0xff for i=0..63",
                "signed_scales": scales,
                "all_positions_compared": True,
                "max_abs_error": max(differences),
            },
            "sampling": check_sampling(),
        }
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except (AssertionError, OSError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
