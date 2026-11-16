#!/usr/bin/env python3
"""Does CUDA host memory from the VMM API (cuMemCreate, location HOST) stay
outside the container's memory limit, as Vulkan's GPU-visible system memory
does? cuMemHostAlloc's pages count against it (the hybrid's 256 MB probe), so
the CUDA RAM tier on a 46 GB container holds 24.5 GB of experts while Vulkan's
banks take the host's whole available memory. Prints the container's usage
growth for 512 MB of each kind, whether the host can write the VMM block, and
the copy rate from it into card memory."""
import ctypes as C, time

cu = C.CDLL("libcuda.so.1")
def ck(rc, what):
    if rc != 0:
        s = C.c_char_p()
        cu.cuGetErrorString(rc, C.byref(s))
        raise SystemExit(f"{what}: {rc} {s.value}")

def usage():
    for p in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            return int(open(p).read())
        except OSError:
            pass
    return 0

class Loc(C.Structure):
    _fields_ = [("type", C.c_int), ("id", C.c_int)]
class Flags(C.Structure):
    _fields_ = [("compressionType", C.c_ubyte), ("gpuDirectRDMACapable", C.c_ubyte), ("usage", C.c_ushort), ("reserved", C.c_ubyte * 4)]
class Prop(C.Structure):
    _fields_ = [("type", C.c_int), ("handles", C.c_int), ("location", Loc), ("win32", C.c_void_p), ("flags", Flags)]
class Access(C.Structure):
    _fields_ = [("location", Loc), ("flags", C.c_int)]

ck(cu.cuInit(0), "cuInit")
dev = C.c_int()
ck(cu.cuDeviceGet(C.byref(dev), 0), "cuDeviceGet")
ctx = C.c_void_p()
ck(cu.cuCtxCreate_v2(C.byref(ctx), 0, dev), "cuCtxCreate")
N = 512 << 20
scratch = (C.c_ubyte * (N + (64 << 20)))()

# cuMemHostAlloc for reference
u0 = usage()
h = C.c_void_p()
ck(cu.cuMemHostAlloc(C.byref(h), C.c_size_t(N), 0x01 | 0x02), "cuMemHostAlloc")
C.memset(h, 1, N)
t = time.time(); C.memset(h, 2, N); tw = time.time() - t
t = time.time(); C.memmove(scratch, h, N); tr = time.time() - t
print(f"cuMemHostAlloc: container usage +{(usage() - u0) >> 20} MB for {N >> 20} MB, host writes {N / tw / 1e9:.1f} GB/s, reads {N / tr / 1e9:.1f} GB/s")
cu.cuMemFreeHost(h)

for ltype, name in ((2, "HOST"), (3, "HOST_NUMA")):
    prop = Prop(type=1, handles=0, location=Loc(ltype, 0))
    gran = C.c_size_t()
    rc = cu.cuMemGetAllocationGranularity(C.byref(gran), C.byref(prop), 0)
    if rc != 0:
        print(f"{name}: granularity refused ({rc})")
        continue
    size = (N + gran.value - 1) // gran.value * gran.value
    u0 = usage()
    hd = C.c_ulonglong()
    rc = cu.cuMemCreate(C.byref(hd), C.c_size_t(size), C.byref(prop), C.c_ulonglong(0))
    if rc != 0:
        print(f"{name}: cuMemCreate refused ({rc})")
        continue
    va = C.c_ulonglong()
    ck(cu.cuMemAddressReserve(C.byref(va), C.c_size_t(size), C.c_size_t(0), C.c_ulonglong(0), C.c_ulonglong(0)), "reserve")
    ck(cu.cuMemMap(va, C.c_size_t(size), C.c_size_t(0), hd, C.c_ulonglong(0)), "map")
    acc = (Access * 2)(Access(Loc(1, 0), 3), Access(Loc(ltype, 0), 3))
    rc = cu.cuMemSetAccess(va, C.c_size_t(size), acc, C.c_size_t(2))
    host_ok = rc == 0
    if not host_ok:
        ck(cu.cuMemSetAccess(va, C.c_size_t(size), acc, C.c_size_t(1)), "access device")
    ck(cu.cuMemsetD8_v2(va, 1, C.c_size_t(size)), "memset")
    ck(cu.cuCtxSynchronize(), "sync")
    if host_ok:
        t = time.time(); C.memset(C.c_void_p(va.value), 2, size); th = time.time() - t
        hw = f"host writes {size / th / 1e9:.1f} GB/s"
    else:
        hw = f"host access refused ({rc})"
    grew = (usage() - u0) >> 20
    if host_ok:
        buf = (C.c_ubyte * size).from_address(va.value)
        t = time.time(); C.memmove(scratch, buf, size); tr = time.time() - t
        hw += f", host reads {size / tr / 1e9:.1f} GB/s"
        mt = C.c_uint()
        cu.cuPointerGetAttribute(C.byref(mt), 2, va)
        hw += f", memory type {mt.value}"
    d = C.c_ulonglong()
    ck(cu.cuMemAlloc_v2(C.byref(d), C.c_size_t(size)), "alloc")
    ck(cu.cuMemcpyDtoD_v2(d, va, C.c_size_t(size)), "copy")
    ck(cu.cuCtxSynchronize(), "sync")
    t = time.time()
    for _ in range(4):
        ck(cu.cuMemcpyDtoD_v2(d, va, C.c_size_t(size)), "copy")
    ck(cu.cuCtxSynchronize(), "sync")
    bw = 4 * size / (time.time() - t) / 1e9
    if host_ok:
        t = time.time()
        ck(cu.cuMemcpyHtoD_v2(d, C.c_void_p(va.value), C.c_size_t(size)), "htod")
        ck(cu.cuCtxSynchronize(), "sync")
        hw += f", HtoD from it {size / (time.time() - t) / 1e9:.1f} GB/s"
        t = time.time()
        ck(cu.cuMemcpy(d, va, C.c_size_t(size)), "memcpy")
        ck(cu.cuCtxSynchronize(), "sync")
        hw += f", unified copy {size / (time.time() - t) / 1e9:.1f} GB/s"
    print(f"VMM {name}: container usage +{grew} MB for {size >> 20} MB, {hw}, to the card {bw:.1f} GB/s")
    cu.cuMemFree_v2(d)
    cu.cuMemUnmap(va, C.c_size_t(size))
    cu.cuMemAddressFree(va, C.c_size_t(size))
    cu.cuMemRelease(hd)
print("VMMPROBE_DONE")
