// Accuracy and speed of the shim against a double-precision reference.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <vector>
extern "C" { int cq_npu_init(const char *); int cq_npu_add(uint32_t, uint32_t, const int8_t *, const float *, float); int cq_npu_run(int, const uint16_t *, uint16_t *); const char *cq_npu_error(); }
static double now() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
  uint32_t rows = argc > 1 ? atoi(argv[1]) : 8192, cols = argc > 2 ? atoi(argv[2]) : 2048; float alpha = argc > 3 ? atof(argv[3]) : 8.0f;
  if (cq_npu_init("libQnnHtp.so")) { printf("init: %s\n", cq_npu_error()); return 1; }
  std::vector<int8_t> w((size_t)rows * cols); std::vector<float> rs(rows), x(cols); uint64_t s = 88172645463325252ull;
  auto rnd = [&] { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
  auto gauss = [&] { double a = 0; for (int i = 0; i < 6; i++) a += (double)(rnd() % 20001) / 10000.0 - 1.0; return a / 1.414; };
  for (auto &b : w) { int v = (int)lround(gauss() * 40); b = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v); }
  double maxn = 0; for (uint32_t r = 0; r < rows; r++) { rs[r] = 0.0005f * (1.0f + (rnd() % 100) / 50.0f); double n = 0; for (uint32_t c = 0; c < cols; c++) { double v = w[(size_t)r * cols + c] * rs[r]; n += v * v; } if (sqrt(n) > maxn) maxn = sqrt(n); }
  for (auto &v : x) v = (float)gauss();
  float out_range = alpha * (float)maxn, out_scale = out_range / 32768.0f;
  double t0 = now(); int h = cq_npu_add(rows, cols, w.data(), rs.data(), out_scale); if (h < 0) { printf("add: %s\n", cq_npu_error()); return 1; }
  printf("rows %u cols %u | graph built in %.2f s | out range ±%.3f (alpha %.1f)\n", rows, cols, now() - t0, out_range, alpha);
  float mx = 0; for (float v : x) mx = fabsf(v) > mx ? fabsf(v) : mx;
  std::vector<uint16_t> xq(cols), yq(rows);
  for (uint32_t c = 0; c < cols; c++) { long v = lround(x[c] / mx * 32767.0) + 32768; xq[c] = (uint16_t)(v < 0 ? 0 : v > 65535 ? 65535 : v); }
  if (cq_npu_run(h, xq.data(), yq.data())) { printf("run: %s\n", cq_npu_error()); return 1; }
  double num = 0, den = 0, maxabs = 0; int sat = 0;
  for (uint32_t r = 0; r < rows; r++) { double ref = 0; for (uint32_t c = 0; c < cols; c++) ref += (double)w[(size_t)r * cols + c] * rs[r] * x[c];
    double got = ((double)yq[r] - 32768.0) * out_scale * mx; num += (got - ref) * (got - ref); den += ref * ref; if (fabs(ref) > maxabs) maxabs = fabs(ref); if (yq[r] == 0 || yq[r] == 65535) sat++; }
  printf("relative L2 error %.2e | saturated rows %d | max|y| %.3f vs range %.3f\n", sqrt(num / den), sat, maxabs, out_range * mx);
  double best = 1e9; for (int it = 0; it < 3; it++) { double t = now(); for (int i = 0; i < 20; i++) cq_npu_run(h, xq.data(), yq.data()); double d = (now() - t) / 20; if (d < best) best = d; }
  printf("%.2f ms per matvec | %.2f G weights/s\n", best * 1e3, rows * (double)cols / best / 1e9);
  fflush(stdout); _exit(0);
}
