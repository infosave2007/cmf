"""timerun.py CMD...: run CMD, then print its wall time and peak RSS (children's ru_maxrss) to stderr in
/usr/bin/time -v's wording (the pod has no GNU time). Exit code is CMD's."""
import resource, subprocess, sys, time

t0 = time.time()
rc = subprocess.call(sys.argv[1:])
dt = time.time() - t0
kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
sys.stderr.write(f"\tElapsed (wall clock) time (s): {dt:.1f}\n\tMaximum resident set size (kbytes): {kb}\n")
sys.exit(rc)
