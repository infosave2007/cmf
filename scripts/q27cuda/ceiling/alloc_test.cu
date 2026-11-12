// How much card memory small cudaMalloc calls take (the driver's rounding).
#include <cstdio>
#include <cuda.h>
int main() {
  cuInit(0); CUdevice d; cuDeviceGet(&d, 0); CUcontext c; cuDevicePrimaryCtxRetain(&c, d); cuCtxSetCurrent(c);
  size_t f0, f1, t; CUdeviceptr p;
  size_t sizes[] = {64, 4096, 65536, 300000, 983040, 1600000, 2200000};
  for (size_t s : sizes) {
    cuMemGetInfo(&f0, &t);
    for (int i = 0; i < 100; i++) cuMemAlloc(&p, s);
    cuMemGetInfo(&f1, &t);
    printf("100 x %zu bytes: %.2f MiB taken (%.1f KiB each)\n", s, (f0 - f1) / 1048576.0, (f0 - f1) / 100.0 / 1024.0);
  }
}
