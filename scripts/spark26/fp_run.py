#!/usr/bin/env python3
"""Peak memory of a command on macOS: phys_footprint (what Activity Monitor shows as Memory — on
unified memory it includes the process's GPU allocations, which RSS leaves out) and the resident
size, sampled every 0.25 s through proc_pid_rusage, plus the wall time.
fp_run.py OUT_PREFIX CMD [ARGS...]
The command's stdout and stderr go to OUT_PREFIX.out and OUT_PREFIX.err; stdin is /dev/null."""
import ctypes
import subprocess
import sys
import time

lib = ctypes.CDLL("/usr/lib/libproc.dylib")


class RU(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(f"f{i}", ctypes.c_uint64) for i in range(18)]


def usage(pid):
    r = RU()
    if lib.proc_pid_rusage(pid, 2, ctypes.byref(r)) != 0:
        return None
    return r.f6, r.f7  # resident, phys_footprint


def main():
    prefix, cmd = sys.argv[1], sys.argv[2:]
    t0 = time.time()
    with open(prefix + ".out", "w") as out, open(prefix + ".err", "w") as err:
        p = subprocess.Popen(cmd, stdout=out, stderr=err, stdin=subprocess.DEVNULL)
        res = fp = 0
        while p.poll() is None:
            u = usage(p.pid)
            if u:
                res, fp = max(res, u[0]), max(fp, u[1])
            time.sleep(0.25)
    print(f"footprint_peak_gb={fp / 2**30:.2f} resident_peak_gb={res / 2**30:.2f} wall_s={time.time() - t0:.0f} rc={p.returncode}")


if __name__ == "__main__":
    main()
