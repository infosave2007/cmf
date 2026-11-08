#!/usr/bin/env python3
"""Parallel range download of one public HF file with resume and a sha256 check.
pdl.py URL OUT SIZE SHA256 [CONNECTIONS] — progress in OUT.progress (done chunks)."""
import hashlib, os, sys, threading, time, urllib.request
url, out, size, sha = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
conns = int(sys.argv[5]) if len(sys.argv) > 5 else 16
CH = 32 << 20
n = (size + CH - 1) // CH
prog = out + ".progress"
done = set(int(x) for x in open(prog).read().split()) if os.path.exists(prog) else set()
if not os.path.exists(out):
    with open(out, "wb") as f:
        f.truncate(size)
lock = threading.Lock()
todo = [i for i in range(n) if i not in done]
t0 = time.time(); got = [0]
def worker():
    fd = os.open(out, os.O_WRONLY)
    while True:
        with lock:
            if not todo:
                break
            i = todo.pop(0)
        a = i * CH; b = min(size, a + CH) - 1
        for attempt in range(20):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                assert len(data) == b - a + 1, (len(data), b - a + 1)
                os.pwrite(fd, data, a)
                break
            except Exception as e:
                time.sleep(2 + attempt)
        else:
            print("chunk failed", i, flush=True); continue
        with lock:
            done.add(i); got[0] += b - a + 1
            open(prog, "a").write(f"{i}\n")
            if len(done) % 10 == 0:
                el = time.time() - t0
                print(f"{len(done)}/{n} chunks, {got[0]/el/1e6:.1f} MB/s", flush=True)
    os.close(fd)
ts = [threading.Thread(target=worker) for _ in range(conns)]
[t.start() for t in ts]; [t.join() for t in ts]
if len(done) == n:
    h = hashlib.sha256()
    with open(out, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    print("sha256", h.hexdigest(), "MATCH" if h.hexdigest() == sha else "MISMATCH", flush=True)
