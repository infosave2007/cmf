# hog.py CARD_GIB: hold the VRAM of a 32 GB card above CARD_GIB (cudaMalloc through Strata's own libcudart),
# so an engine that sizes its caches from free VRAM sees a CARD_GIB card. Prints HOG_READY, sleeps until killed.
import ctypes, sys, time, subprocess
card = float(sys.argv[1])
rt = ctypes.CDLL("/root/fc/strata/engine/libcudart.so.13")
free, total = ctypes.c_size_t(), ctypes.c_size_t()
rt.cudaFree(ctypes.c_void_p(0))  # context
rt.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
want = int(free.value - card * (1 << 30))
ptr = ctypes.c_void_p()
held = 0
while want - held > (64 << 20):
    step = min(want - held, 1 << 30)
    if rt.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(step)) != 0:
        break
    rt.cudaMemset(ptr, 0, ctypes.c_size_t(step))
    held += step
rt.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
print(f"HOG_READY held {held >> 20} MiB, free {free.value >> 20} MiB of {total.value >> 20}", flush=True)
while True:
    time.sleep(3600)
