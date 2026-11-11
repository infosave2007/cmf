// The small path's GDN kernels as hyb.cu has them: the front with row-batched
// gates and the column scan (`hy_gdn_scan4`), timed over 48 layers, and the
// restore checked to the bit: scan4 over n rows with the tape, then
// `hy_gdn_restore_scan4` keeping k rows, against scan4 over those k rows.
//   nvcc -O3 -arch=sm_86 -DHPK=6 -DHD=256 -w gdn_test2.cu -o gdn_test2
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include "hyb.cu"

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
    const int nv = 48, nk = 16, dk = 128, dv = 128, kk = 4, hidden = 5120, L = 48, NMAX = 16;
    const int c_dim = 2 * nk * dk + nv * dv, units = c_dim / dk;
    const size_t sn = (size_t)nv * dk * dv;
    float* qkv = dev_rand((size_t)NMAX * c_dim, 2.0f);
    float* taps = dev_rand((size_t)c_dim * kk, 0.5f);
    float* xn = dev_rand((size_t)NMAX * hidden, 2.0f);
    std::vector<float*> wa(L), wb(L), S(L), ring(L);
    for (int l = 0; l < L; l++) {
        wa[l] = dev_rand((size_t)nv * hidden, 0.02f);
        wb[l] = dev_rand((size_t)nv * hidden, 0.02f);
        S[l] = dev_rand(sn, 0.1f);
        ring[l] = dev_rand((size_t)(kk - 1) * c_dim, 1.0f);
    }
    float* a_log = dev_rand(nv, 1.0f);
    float* dt_bias = dev_rand(nv, 1.0f);
    float *cq, *gb, *o, *sv;
    CK(cudaMalloc(&cq, (size_t)NMAX * c_dim * 4));
    CK(cudaMalloc(&gb, (size_t)NMAX * 2 * nv * 4));
    CK(cudaMalloc(&o, (size_t)NMAX * nv * dv * 4));
    CK(cudaMalloc(&sv, sn * 4));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const int sblk = nv * dv / 32, rblk = (c_dim + 127) / 128;
    for (int n : {1, 5, 8}) {
        for (int what = 0; what < 3; what++) {
            float best = 1e9f;
            for (int rep = 0; rep < 5; rep++) {
                CK(cudaEventRecord(e0));
                for (int l = 0; l < L; l++) {
                    if (what != 1)
                        hy_gdn_front<<<dim3(units + 2 * nv, n), dk>>>(qkv, c_dim, ring[l], taps, cq, n, c_dim, kk, nk, dk,
                                                                     xn, hidden, wa[l], wb[l], a_log, dt_bias, gb, nv);
                    if (what != 0)
                        hy_gdn_scan4<<<sblk + rblk, 128>>>(cq, gb, S[l], o, n, c_dim, nv, nk, dv, ring[l], qkv, c_dim,
                                                           kk, what == 2 && n > 1 ? sv : nullptr);
                }
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                float ms;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                best = ms < best ? ms : best;
            }
            const char* names[] = {"front", "scan4", "front+scan4 (tape if n>1)"};
            printf("n=%d %-26s %.1f us a layer\n", n, names[what], best * 1e3f / L);
        }
    }
    // the gates against a host double reference, rows 0..4
    {
        const int n = 5;
        hy_gdn_front<<<dim3(units + 2 * nv, n), dk>>>(qkv, c_dim, ring[0], taps, cq, n, c_dim, kk, nk, dk, xn, hidden,
                                                     wa[0], wb[0], a_log, dt_bias, gb, nv);
        CK(cudaDeviceSynchronize());
        std::vector<float> hx((size_t)n * hidden), ha((size_t)nv * hidden), hb((size_t)nv * hidden), hal(nv), hdt(nv),
            hg((size_t)n * 2 * nv);
        CK(cudaMemcpy(hx.data(), xn, hx.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ha.data(), wa[0], ha.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hb.data(), wb[0], hb.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hal.data(), a_log, nv * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hdt.data(), dt_bias, nv * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hg.data(), gb, hg.size() * 4, cudaMemcpyDeviceToHost));
        double md = 0;
        for (int t = 0; t < n; t++)
            for (int h = 0; h < nv; h++) {
                double a = 0, b = 0;
                for (int c = 0; c < hidden; c++) {
                    a += (double)ha[(size_t)h * hidden + c] * hx[(size_t)t * hidden + c];
                    b += (double)hb[(size_t)h * hidden + c] * hx[(size_t)t * hidden + c];
                }
                const double x2 = a + hdt[h];
                const double sp = x2 > 20 ? x2 : log1p(exp(x2));
                const double g = exp(-exp((double)hal[h]) * sp), be = 1 / (1 + exp(-b));
                md = fmax(md, fabs(g - hg[(size_t)t * 2 * nv + h]));
                md = fmax(md, fabs(be - hg[(size_t)t * 2 * nv + nv + h]));
            }
        printf("gates vs host double, 5 rows: max diff %.2e\n", md);
    }
    // restore to the bit
    {
        const int n = 6;
        std::vector<float> S0(sn), Sk(sn), Sr(sn);
        CK(cudaMemcpy(S0.data(), S[1], sn * 4, cudaMemcpyDeviceToHost));
        float *Sa, *Sb;
        CK(cudaMalloc(&Sa, sn * 4));
        CK(cudaMalloc(&Sb, sn * 4));
        u64* tab;
        CK(cudaMalloc(&tab, 7 * 8));
        int bad = 0;
        for (int keep = 1; keep <= n; keep++) {
            // reference: keep rows from S0
            CK(cudaMemcpy(Sa, S0.data(), sn * 4, cudaMemcpyHostToDevice));
            hy_gdn_scan4<<<sblk, 128>>>(cq, gb, Sa, o, keep, c_dim, nv, nk, dv, nullptr, qkv, c_dim, kk, nullptr);
            // the window: n rows with the tape, then the restore to keep
            CK(cudaMemcpy(Sb, S0.data(), sn * 4, cudaMemcpyHostToDevice));
            hy_gdn_scan4<<<sblk, 128>>>(cq, gb, Sb, o, n, c_dim, nv, nk, dv, nullptr, qkv, c_dim, kk, sv);
            u64 t[7] = {(u64)sv, (u64)Sb, (u64)cq, (u64)gb, 0, 0, 0};
            CK(cudaMemcpy(tab, t, sizeof t, cudaMemcpyHostToDevice));
            hy_gdn_restore_scan4<<<dim3(sblk, 1), 128>>>(tab, keep, c_dim, nv, nk, dv);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(Sk.data(), Sa, sn * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(Sr.data(), Sb, sn * 4, cudaMemcpyDeviceToHost));
            const int diff = memcmp(Sk.data(), Sr.data(), sn * 4) != 0;
            bad += diff;
            printf("restore keep=%d of %d: %s\n", keep, n, diff ? "DIFFERS" : "bit-identical");
        }
        // the restore's own time over 48 layers (keep 2 of 5)
        std::vector<u64> tt((size_t)L * 7);
        for (int l = 0; l < L; l++) {
            u64 r[7] = {(u64)sv, (u64)S[l], (u64)cq, (u64)gb, 0, 0, 0};
            memcpy(&tt[(size_t)l * 7], r, sizeof r);
        }
        u64* tabL;
        CK(cudaMalloc(&tabL, tt.size() * 8));
        CK(cudaMemcpy(tabL, tt.data(), tt.size() * 8, cudaMemcpyHostToDevice));
        float best = 1e9f;
        for (int rep = 0; rep < 5; rep++) {
            CK(cudaEventRecord(e0));
            hy_gdn_restore_scan4<<<dim3(sblk, L), 128>>>(tabL, 2, c_dim, nv, nk, dv);
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            float ms;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            best = ms < best ? ms : best;
        }
        printf("restore4 of 48 layers (keep 2): %.3f ms\n", best);
        best = 1e9f;
        for (int rep = 0; rep < 5; rep++) {
            CK(cudaEventRecord(e0));
            hy_gdn_restore_scan<<<dim3((nv * dv + 7) / 8, L), 256>>>(tabL, 2, c_dim, nv, nk, dk, dv);
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            float ms;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            best = ms < best ? ms : best;
        }
        printf("old restore of 48 layers (keep 2): %.3f ms\n", best);
        printf(bad ? "RESTORE MISMATCH\n" : "restore OK\n");
    }
    CK(cudaGetLastError());
    return 0;
}
