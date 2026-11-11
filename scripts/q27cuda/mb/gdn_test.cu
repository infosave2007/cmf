// The GDN step of the hybrid arm's small path (front + scan + out32), on the
// shapes of Qwen3.8-27B, timed over 48 layers' worth of separate states so
// the 3 MB matrices come from memory as in a token.
//   nvcc -O3 -arch=sm_86 -DHPK=6 -DHD=256 -w gdn_test.cu -o gdn_test
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include "hyb.cu"
#include "gdn_scan2.cuh"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

static float* dev_rand(size_t n, float scale) {
    std::vector<float> h(n);
    for (auto& v : h) v = (rand() / (float)RAND_MAX - 0.5f) * scale;
    float* d;
    CK(cudaMalloc(&d, n * 4));
    CK(cudaMemcpy(d, h.data(), n * 4, cudaMemcpyHostToDevice));
    return d;
}

int main() {
    const int nv = 48, nk = 16, dk = 128, dv = 128, kk = 4, hidden = 5120, L = 48;
    const int c_dim = 2 * nk * dk + nv * dv, units = c_dim / dk;
    const int NMAX = 16;
    float* qkv = dev_rand((size_t)NMAX * c_dim, 2.0f);
    float* taps = dev_rand((size_t)c_dim * kk, 0.5f);
    float* xn = dev_rand((size_t)NMAX * hidden, 2.0f);
    std::vector<float*> wa(L), wb(L), S(L), ring(L);
    for (int l = 0; l < L; l++) {
        wa[l] = dev_rand((size_t)nv * hidden, 0.02f);
        wb[l] = dev_rand((size_t)nv * hidden, 0.02f);
        S[l] = dev_rand((size_t)nv * dk * dv, 0.1f);
        ring[l] = dev_rand((size_t)(kk - 1) * c_dim, 1.0f);
    }
    float* a_log = dev_rand(nv, 1.0f);
    float* dt_bias = dev_rand(nv, 1.0f);
    float *cq, *gb, *o;
    CK(cudaMalloc(&cq, (size_t)NMAX * c_dim * 4));
    CK(cudaMalloc(&gb, (size_t)NMAX * 2 * nv * 4));
    CK(cudaMalloc(&o, (size_t)NMAX * nv * dv * 4));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const int cols = nv * dv;
    const int ring_blocks = (c_dim + 255) / 256;
    for (int n : {1, 5, 8}) {
        for (int what = 0; what < 3; what++) {
            // 0 front only, 1 scan only, 2 both
            float best = 1e9f;
            for (int rep = 0; rep < 5; rep++) {
                CK(cudaEventRecord(e0));
                for (int l = 0; l < L; l++) {
                    if (what != 1)
                        hy_gdn_front<<<dim3(units + nv, n), dk>>>(qkv, c_dim, ring[l], taps, cq, n, c_dim, kk, nk, dk,
                                                                 xn, hidden, wa[l], wb[l], a_log, dt_bias, gb, nv);
                    if (what != 0)
                        hy_gdn_scan<<<(cols + 7) / 8 + ring_blocks, 256>>>(cq, gb, S[l], o, n, c_dim, nv, nk, dk, dv,
                                                                          ring[l], qkv, c_dim, kk);
                }
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                float ms;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                best = ms < best ? ms : best;
            }
            const char* names[] = {"front", "scan", "front+scan"};
            printf("n=%d %-10s %.3f ms for %d layers (%.1f us a layer)\n", n, names[what], best, L, best * 1e3f / L);
        }
    }
    // the candidates against hy_gdn_scan: one layer, the same inputs
    {
        const size_t sn = (size_t)nv * dk * dv;
        float* Sref = S[0];
        float *Sx, *ox, *oref, *sv;
        CK(cudaMalloc(&Sx, sn * 4)); CK(cudaMalloc(&ox, (size_t)NMAX * nv * dv * 4)); CK(cudaMalloc(&oref, (size_t)NMAX * nv * dv * 4));
        CK(cudaMalloc(&sv, sn * 4));
        std::vector<float> S0(sn);
        CK(cudaMemcpy(S0.data(), Sref, sn * 4, cudaMemcpyDeviceToHost));
        const int n = 5;
        hy_gdn_front<<<dim3(units + nv, n), dk>>>(qkv, c_dim, ring[0], taps, cq, n, c_dim, kk, nk, dk, xn, hidden, wa[0], wb[0], a_log, dt_bias, gb, nv);
        std::vector<float> Sr(sn), orf((size_t)n * nv * dv), Sc(sn), oc((size_t)n * nv * dv);
        CK(cudaMemcpy(Sx, S0.data(), sn * 4, cudaMemcpyHostToDevice));
        hy_gdn_scan<<<(cols + 7) / 8, 256>>>(cq, gb, Sx, oref, n, c_dim, nv, nk, dk, dv, nullptr, qkv, c_dim, kk);
        CK(cudaMemcpy(Sr.data(), Sx, sn * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(orf.data(), oref, orf.size() * 4, cudaMemcpyDeviceToHost));
        for (int v = 0; v < 4; v++) {
            CK(cudaMemcpy(Sx, S0.data(), sn * 4, cudaMemcpyHostToDevice));
            const int wpr = dv / 32;
            if (v == 0) gdn_scan_a<<<nv * wpr, 32>>>(cq, gb, Sx, ox, n, c_dim, nv, nk, dv, sv);
            if (v == 1) gdn_scan_b2<<<nv * wpr, 64>>>(cq, gb, Sx, ox, n, c_dim, nv, nk, dv, sv);
            if (v == 2) gdn_scan_b4<<<nv * wpr, 128>>>(cq, gb, Sx, ox, n, c_dim, nv, nk, dv, sv);
            if (v == 3) gdn_scan_b8<<<nv * wpr, 256>>>(cq, gb, Sx, ox, n, c_dim, nv, nk, dv, sv);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(Sc.data(), Sx, sn * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(oc.data(), ox, oc.size() * 4, cudaMemcpyDeviceToHost));
            std::vector<float> svh(sn);
            CK(cudaMemcpy(svh.data(), sv, sn * 4, cudaMemcpyDeviceToHost));
            double ds = 0, dso = 0, dss = 0, ms = 0, mo = 0;
            for (size_t i = 0; i < sn; i++) { ds = fmax(ds, fabs(Sc[i] - Sr[i])); ms = fmax(ms, fabs(Sr[i])); dss = fmax(dss, fabs(svh[i] - S0[i])); }
            for (size_t i = 0; i < oc.size(); i++) { dso = fmax(dso, fabs(oc[i] - orf[i])); mo = fmax(mo, fabs(orf[i])); }
            const char* nm[] = {"A", "B2", "B4", "B8"};
            printf("check %s: S max diff %.2e (max %.2e), o max diff %.2e (max %.2e), save diff %.1e\n", nm[v], ds, ms, dso, mo, dss);
        }
        // timing of the candidates over 48 layers, with and without the save
        for (int nn : {1, 5, 8}) {
            for (int v = 0; v < 4; v++) {
                for (int save = 0; save < 2; save++) {
                    float best = 1e9f;
                    for (int rep = 0; rep < 5; rep++) {
                        CK(cudaEventRecord(e0));
                        for (int l = 0; l < L; l++) {
                            const int wpr = dv / 32;
                            float* svp = save ? sv : nullptr;
                            if (v == 0) gdn_scan_a<<<nv * wpr, 32>>>(cq, gb, S[l], o, nn, c_dim, nv, nk, dv, svp);
                            if (v == 1) gdn_scan_b2<<<nv * wpr, 64>>>(cq, gb, S[l], o, nn, c_dim, nv, nk, dv, svp);
                            if (v == 2) gdn_scan_b4<<<nv * wpr, 128>>>(cq, gb, S[l], o, nn, c_dim, nv, nk, dv, svp);
                            if (v == 3) gdn_scan_b8<<<nv * wpr, 256>>>(cq, gb, S[l], o, nn, c_dim, nv, nk, dv, svp);
                        }
                        CK(cudaEventRecord(e1));
                        CK(cudaEventSynchronize(e1));
                        float ms2;
                        CK(cudaEventElapsedTime(&ms2, e0, e1));
                        best = ms2 < best ? ms2 : best;
                    }
                    const char* nm[] = {"A", "B2", "B4", "B8"};
                    printf("n=%d scan %-3s save=%d %.1f us a layer\n", nn, nm[v], save, best * 1e3f / L);
                }
            }
            // the gates: the front's conv blocks only, then the gate kernel
            for (int part = 0; part < 2; part++) {
                float best = 1e9f;
                for (int rep = 0; rep < 5; rep++) {
                    CK(cudaEventRecord(e0));
                    for (int l = 0; l < L; l++) {
                        if (part == 0)
                            hy_gdn_front<<<dim3(units, nn), dk>>>(qkv, c_dim, ring[l], taps, cq, nn, c_dim, kk, nk, dk, xn, hidden, wa[l], wb[l], a_log, dt_bias, gb, nv);
                        else
                            gdn_gates2<<<dim3(2 * nv, nn), 256>>>(xn, hidden, wa[l], wb[l], a_log, dt_bias, gb, nv);
                    }
                    CK(cudaEventRecord(e1));
                    CK(cudaEventSynchronize(e1));
                    float ms2;
                    CK(cudaEventElapsedTime(&ms2, e0, e1));
                    best = ms2 < best ? ms2 : best;
                }
                printf("n=%d %s %.1f us a layer\n", nn, part == 0 ? "front conv only" : "gates2", best * 1e3f / L);
            }
        }
    }
    CK(cudaGetLastError());
    return 0;
}
