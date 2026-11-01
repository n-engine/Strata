// ds4_attn_parity: DS4_FAST_ATTN's XMX kernel (ds4_kernels.dp.cpp xmxa, sparse_attn_xmx) against the scalar kernel it replaces
// (sparse_attn_packed_kernel<G>, through sparse_attn with heads_per_block G) and against host references in double precision
// on sampled outputs, at DeepSeek V4.1 Flash's attention shapes: head dim 512, 64 heads on one card (num_attention_heads),
// MQA (K = V), the window buffer's BF16 rows (sliding_window 128) plus up to index_topk = 512 selected compressed rows
// (kKvFp4: E2M1 codes, an E4M3 scale per 16), a learned sink logit per head, scale 1 / sqrt(512).  The rows a query reads
// are the engine's chunked-prompt rule (AttnRows gen = 1: query t at position p0 + t reads min(128, p + 1) window rows and
// min(512, (p + 1) / R) selected rows; R is the layer's compress ratio, 0, 1 or 2 in the model's compress_ratios).
// Inputs are random: window values N(0, 1.3) rounded to BF16 (|v| < 2^-14 set to 0, so every key is exact in FP16 as the real
// FP8-derived rows are), packed rows with random codes and E4M3 scales in [0.125, 0.875] (values up to 5.25, RMS about 1.3),
// queries N(0, qstd) (qstd 1: diffuse attention, 4: peaked), sinks N(1, 1.5), selected rows drawn from a pool.
// Both kernels are timed (host clock around `iters` launches and a wait, after one warm-up launch).
//
// Checks (a case passes when all pass).  Per sampled output o_c (query t, head h, dim c), from the exact decoded keys k_j (a
// window row's BF16, a packed row's E2M1 value times its E4M3 scale, written here from the formats' definitions) and the F32
// queries, in double: s_j = scale q.k_j, w_j the softmax weights with the sink in the denominator, o_c = sum w_j k_jc, and
//   a_j = scale sum_d |q_d k_jd|, U_c = sum w_j a_j |k_jc - o_c|, V_c = sum w_j |k_jc|, S_c = sum |k_jc|, n keys.
//   F_c = 2^-24 (514 U_c + sum w_j |s_j - max| |k_jc - o_c| + (n + 8) V_c + (n + 4) |o_c|): a first-order worst case of F32
//         arithmetic (a 512-term dot and its scale: 514 roundings of a_j, since do/ds_j = w_j (k_j - o); the exponent's
//         subtraction; n-term sums, rescales, exp's few ulps and the division).
//   H_c = 2^-11 (U_c + V_c) + 2^-25 S_c: what the XMX kernel adds - Q rounded to FP16 (|dq| <= 2^-11 |q|, so |ds_j| <=
//         2^-11 a_j), P rounded to FP16 (relative 2^-11; absolute 2^-25 where P is an FP16 subnormal).
//   F  scalar vs exact: |d| <= 2 F_c.   xmx vs exact: |d| <= 2 H_c + 2 F_c.  (2: room for the second-order terms.)
//   A  xmx vs "emu" - the XMX kernel's own algorithm in double (FP16 Q, the 16-key tiles' online softmax, P rounded to FP16
//      from each tile's running max, the denominator from the unrounded P): |d| <= 2 F_c + 2^-10 V_c + 2^-24 S_c (the last two:
//      a P whose F32 and double values straddle an FP16 rounding boundary).  About 40x tighter than xmx vs exact at these
//      shapes (a_j is 20-60 there): a dropped, doubled or misplaced key fails here first.
//   B  xmx vs scalar on EVERY output: the same outputs written (each finite, every output of the case, nothing past them), and
//      max |d| <= 2 x the largest (2 H_c + 4 F_c) of the samples (|x - s| <= |x - o| + |s - o| per output; the unsampled
//      outputs have the same distribution - a heuristic bound, unlike F and A).
// Each check prints its worst |d| / bound (pass: <= 1), and the max abs and relative (to the largest |reference|) errors.
// Upstream's gate for DS4_FAST_ATTN is a model-level one (ds4_kernels.dp.cpp: "KL 0.0003 against the 0.006 gate, ids
// identical"): the KL divergence of the engine's next-token distribution against the reference model may move by at most 0.006
// nats, and the greedy ids stay the same.  This test does not measure that; it checks the kernel against its own arithmetic.
// Usage: ds4_attn_parity [--t=16,64,256] [--p0=0,200,9000] [--ratio=0,1,2] [--qstd=1,4] [--heads=64] [--win=128]
//                        [--kcap=512] [--pool=4096] [--samples=32] [--g=8] [--iters=5] [--seed=1]
// --g: the scalar kernel's heads a block (the engine picks 8 down to 1 by the card's size; the bits are the same for any G).
// Exit 0: every case passed.
#include "ds4_kernels.cuh"
#include "ds4_sycl.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace ds4 = strata::ds4;

static std::vector<int> parse_list(const std::string& s) {
    std::vector<int> v;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) v.push_back(std::atoi(s.substr(i, j - i).c_str()));
        i = j + 1;
    }
    return v;
}
static std::vector<double> parse_dlist(const std::string& s) {
    std::vector<double> v;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) v.push_back(std::atof(s.substr(i, j - i).c_str()));
        i = j + 1;
    }
    return v;
}

template <class F>
static void par_for(int n, F f) {
    const int nt = std::max(1, std::min<int>(32, (int) std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t)
        th.emplace_back([&, t] {
            for (int i = t; i < n; i += nt) f(i);
        });
    for (auto& x : th) x.join();
}

// ---------------------------------------------------------------- the formats, from their definitions (not the device decoders)
static float bf16_val(uint16_t b) { uint32_t u = (uint32_t) b << 16; float f; std::memcpy(&f, &u, 4); return f; }
static uint16_t f2bf16_rne(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
static double e4m3_val(uint8_t q) {   // OCP E4M3: bias 7, no infinities, S.1111.111 NaN
    const int e = (q >> 3) & 15, m = q & 7;
    double v;
    if (e == 15 && m == 7) v = NAN;
    else if (e == 0) v = std::ldexp((double) m, -9);
    else v = std::ldexp(1.0 + m / 8.0, e - 7);
    return (q & 0x80) ? -v : v;
}
static double e2m1_val(int c) {   // OCP E2M1: bias 1; 0, 0.5, 1, 1.5, 2, 3, 4, 6
    const int e = (c >> 1) & 3, m = c & 1;
    const double v = e == 0 ? 0.5 * m : std::ldexp(1.0 + 0.5 * m, e - 1);
    return (c & 8) ? -v : v;
}
static double rh(double x) { return (double) (float) sycl::half((float) x); }   // to FP16 (round to nearest even) and back

int main(int argc, char** argv) {
    std::vector<int> Ts = {16, 64, 256}, P0s = {0, 200, 9000}, Rs = {0, 1, 2};
    std::vector<double> qstds = {1.0, 4.0};
    int heads = 64, W = 128, kcap = 512, pool = 4096, nsamp = 32, G = 8, iters = 5;
    unsigned seed = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char* k) { return a.rfind(k, 0) == 0 ? a.substr(std::strlen(k)) : std::string(); };
        if (!val("--t=").empty()) Ts = parse_list(val("--t="));
        else if (!val("--p0=").empty()) P0s = parse_list(val("--p0="));
        else if (!val("--ratio=").empty()) Rs = parse_list(val("--ratio="));
        else if (!val("--qstd=").empty()) qstds = parse_dlist(val("--qstd="));
        else if (!val("--heads=").empty()) heads = std::atoi(val("--heads=").c_str());
        else if (!val("--win=").empty()) W = std::atoi(val("--win=").c_str());
        else if (!val("--kcap=").empty()) kcap = std::atoi(val("--kcap=").c_str());
        else if (!val("--pool=").empty()) pool = std::max(1, std::atoi(val("--pool=").c_str()));
        else if (!val("--samples=").empty()) nsamp = std::max(1, std::atoi(val("--samples=").c_str()));
        else if (!val("--g=").empty()) G = std::atoi(val("--g=").c_str());
        else if (!val("--iters=").empty()) iters = std::max(1, std::atoi(val("--iters=").c_str()));
        else if (!val("--seed=").empty()) seed = (unsigned) std::atoi(val("--seed=").c_str());
        else {
            std::fprintf(stderr, "usage: %s [--t=16,64,256] [--p0=0,200,9000] [--ratio=0,1,2] [--qstd=1,4] [--heads=64] [--win=128] "
                                 "[--kcap=512] [--pool=4096] [--samples=32] [--g=8] [--iters=5] [--seed=1]\n", argv[0]);
            return 2;
        }
    }
    if (heads % 16 != 0 || heads <= 0 || (G != 1 && G != 2 && G != 4 && G != 8) || heads % G != 0) {
        std::fprintf(stderr, "heads must be a multiple of 16 and of --g (1, 2, 4 or 8)\n");
        return 2;
    }
    constexpr int D = 512;
    const int sel_stride = std::max(kcap, 32);   // (the engine's topk_stride)
    const size_t crow = ds4::kv_row_bytes(ds4::kKvFp4, D);
    const float scale = 1.0f / std::sqrt((float) D);
    sycl::queue& q = *strata::q_of(nullptr);
    std::printf("ds4_attn_parity on %s: %d heads, head dim %d, window %d, index_topk %d, %d pooled compressed rows, scalar G %d, "
                "%d timed launches a kernel, %d sampled outputs a case\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), heads, D, W, kcap, pool, G, iters, nsamp);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);

    // the compressed rows' pool (shared by the cases) and its exact values
    std::vector<uint8_t> hckv((size_t) pool * crow, 0);
    std::vector<float> ckv_val((size_t) pool * D);
    for (int r = 0; r < pool; ++r) {
        uint8_t* row = hckv.data() + (size_t) r * crow;
        for (int b = 0; b < D / 2; ++b) row[b] = (uint8_t) rng();
        for (int g = 0; g < D / 16; ++g) row[D / 2 + g] = (uint8_t) (((4 + rng() % 3) << 3) | (rng() & 7));   // [0.125, 0.875]
        for (int c = 0; c < D; ++c) {
            const int nib = (c & 1) ? row[c / 2] >> 4 : row[c / 2] & 15;
            ckv_val[(size_t) r * D + c] = (float) (e2m1_val(nib) * e4m3_val(row[D / 2 + c / 16]));
        }
    }
    uint8_t* dckv = sycl::malloc_device<uint8_t>(hckv.size(), q);
    q.memcpy(dckv, hckv.data(), hckv.size()).wait();

    int fails = 0, cases = 0;
    double t_s_all = 0, t_x_all = 0;
    for (int T : Ts)
        for (int p0 : P0s)
            for (int R : Rs)
                for (double qstd : qstds) {
                    if (T < 1 || p0 < 0) continue;
                    ++cases;
                    ds4::AttnRows ar;
                    ar.gen = 1;
                    ar.p0 = p0;
                    ar.t0 = 0;
                    ar.win = W;
                    ar.R = R;
                    ar.kcap = kcap;
                    // the rows each query reads (sparse_attn's gen rule)
                    std::vector<int> nwin(T), nsel(T), w0(T);
                    int kmin = 1 << 30, kmax = 0;
                    for (int t = 0; t < T; ++t) {
                        const long long ta = t, pa = (long long) p0 + ta;
                        const long long first = std::max((long long) W - p0, ta + 1);
                        w0[t] = (int) first;
                        nwin[t] = (int) (W + ta - first + 1);
                        nsel[t] = R > 0 ? (int) std::min((long long) kcap, (pa + 1) / R) : 0;
                        kmin = std::min(kmin, nwin[t] + nsel[t]);
                        kmax = std::max(kmax, nwin[t] + nsel[t]);
                    }
                    // the window buffer (W + T rows), the selections, the queries, the sinks
                    std::vector<uint16_t> hwin((size_t) (W + T) * D);
                    for (auto& v : hwin) {
                        float x = nd(rng) * 1.3f;
                        if (std::fabs(x) < std::ldexp(1.0f, -14)) x = 0.0f;
                        v = f2bf16_rne(x);
                    }
                    std::vector<int> hsel((size_t) T * sel_stride, 0);
                    std::uniform_int_distribution<int> pd(0, pool - 1);
                    for (int t = 0; t < T; ++t)
                        for (int i = 0; i < nsel[t]; ++i) hsel[(size_t) t * sel_stride + i] = pd(rng);
                    std::vector<float> hq((size_t) T * heads * D), hsink(heads);
                    for (auto& v : hq) v = nd(rng) * (float) qstd;
                    for (auto& v : hsink) v = 1.0f + 1.5f * nd(rng);
                    const size_t outn = (size_t) T * heads * D, pad = 256;
                    uint16_t* dwin = sycl::malloc_device<uint16_t>(hwin.size(), q);
                    int* dsel = sycl::malloc_device<int>(hsel.size(), q);
                    float* dq = sycl::malloc_device<float>(hq.size(), q);
                    float* dsink = sycl::malloc_device<float>(heads, q);
                    float* dout[2] = {sycl::malloc_device<float>(outn + pad, q), sycl::malloc_device<float>(outn + pad, q)};
                    q.memcpy(dwin, hwin.data(), 2 * hwin.size());
                    q.memcpy(dsel, hsel.data(), 4 * hsel.size());
                    q.memcpy(dq, hq.data(), 4 * hq.size());
                    q.memcpy(dsink, hsink.data(), 4 * heads);
                    for (int k = 0; k < 2; ++k) {
                        q.fill(dout[k], NAN, outn);
                        q.fill(dout[k] + outn, 1234.5f, pad);
                    }
                    q.wait();
                    // 0: the scalar kernel, 1: XMX; one launch, the outputs read, then the timed launches
                    bool launched = true;
                    double ms[2] = {0, 0};
                    std::vector<float> o[2];
                    for (int k = 0; k < 2; ++k) {
                        auto run = [&]() {
                            if (k == 0) {
                                ds4::sparse_attn(dq, dwin, dckv, dsel, sel_stride, ar, dsink, heads, D, T, scale, dout[0], nullptr,
                                                 ds4::kKvFp4, G, nullptr);
                                return true;
                            }
                            return ds4::sparse_attn_xmx(dq, dwin, dckv, dsel, sel_stride, ar, dsink, heads, T, scale, dout[1], nullptr);
                        };
                        launched = run() && launched;
                        q.wait();
                        o[k].resize(outn + pad);
                        q.memcpy(o[k].data(), dout[k], 4 * (outn + pad)).wait();
                        const auto c0 = std::chrono::steady_clock::now();
                        for (int r = 0; r < iters && launched; ++r) run();
                        q.wait();
                        ms[k] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count() / iters;
                    }
                    char head[160];
                    std::snprintf(head, sizeof head, "T %3d p0 %5d R %d qstd %.1f (keys %3d..%3d)", T, p0, R, qstd, kmin, kmax);
                    if (!launched) {
                        std::printf("%s: NOT LAUNCHED (sparse_attn_xmx refused)  FAIL\n", head);
                        ++fails;
                    } else {
                        // B: every output written, finite, nothing past them; the largest difference
                        size_t bad = 0, stray = 0;
                        double bmax = 0, smax = 0;
                        for (size_t i = 0; i < outn; ++i) {
                            if (!std::isfinite(o[0][i]) || !std::isfinite(o[1][i])) { ++bad; continue; }
                            bmax = std::max(bmax, std::fabs((double) o[1][i] - o[0][i]));
                            smax = std::max(smax, (double) std::fabs(o[0][i]));
                        }
                        for (size_t i = outn; i < outn + pad; ++i)
                            if (o[0][i] != 1234.5f || o[1][i] != 1234.5f) ++stray;
                        // the samples: (t, h) pairs, every dim; t = 0 and T - 1 among them
                        struct S { int t, h; };
                        std::vector<S> smp;
                        std::uniform_int_distribution<int> td(0, T - 1), hd(0, heads - 1);
                        for (int i = 0; i < nsamp; ++i) smp.push_back({i == 0 ? 0 : i == 1 ? T - 1 : td(rng), hd(rng)});
                        struct R8 { double fs = 0, fx = 0, ax = 0, bb = 0, ex_s = 0, ex_x = 0, ea_x = 0, refmax = 0; bool nf = false; };
                        std::vector<R8> res(smp.size());
                        par_for((int) smp.size(), [&](int si) {
                            const int t = smp[si].t, h = smp[si].h, n = nwin[t] + nsel[t];
                            std::vector<const float*> kc(n);   // (the exact keys; window rows decoded once below)
                            std::vector<float> wv((size_t) nwin[t] * D);
                            for (int j = 0; j < nwin[t]; ++j)
                                for (int c = 0; c < D; ++c) wv[(size_t) j * D + c] = bf16_val(hwin[(size_t) (w0[t] + j) * D + c]);
                            for (int j = 0; j < n; ++j)
                                kc[j] = j < nwin[t] ? wv.data() + (size_t) j * D
                                                    : ckv_val.data() + (size_t) hsel[(size_t) t * sel_stride + (j - nwin[t])] * D;
                            const float* qv = hq.data() + ((size_t) t * heads + h) * D;
                            const double sc = (double) scale, sink = hsink[h];
                            std::vector<double> s(n), a(n), s16(n);
                            double mx = sink;
                            for (int j = 0; j < n; ++j) {
                                double x = 0, ab = 0, x16 = 0;
                                for (int c = 0; c < D; ++c) {
                                    x += (double) qv[c] * kc[j][c];
                                    ab += std::fabs((double) qv[c] * kc[j][c]);
                                    x16 += rh(qv[c]) * kc[j][c];
                                }
                                s[j] = sc * x, a[j] = sc * ab, s16[j] = sc * x16;
                                mx = std::max(mx, s[j]);
                            }
                            std::vector<double> w(n);
                            double den = std::exp(sink - mx);
                            for (int j = 0; j < n; ++j) den += (w[j] = std::exp(s[j] - mx));
                            for (int j = 0; j < n; ++j) w[j] /= den;
                            // emu: the XMX kernel's tiles in double
                            std::vector<double> oe(D, 0.0);
                            double m_run = -1e30, dn = 0;
                            for (int t0 = 0; t0 < n; t0 += 16) {
                                const int nt = std::min(16, n - t0);
                                double m = -INFINITY;
                                for (int j = t0; j < t0 + nt; ++j) m = std::max(m, s16[j]);
                                const double mn = std::max(m_run, m), f = std::exp(m_run - mn);
                                double sum = 0;
                                for (int c = 0; c < D; ++c) oe[c] *= f;
                                for (int j = t0; j < t0 + nt; ++j) {
                                    const double p = std::exp(s16[j] - mn), p16 = rh((float) p);
                                    sum += p;
                                    for (int c = 0; c < D; ++c) oe[c] += p16 * kc[j][c];
                                }
                                dn = dn * f + sum;
                                m_run = mn;
                            }
                            const double dfin = dn + std::exp(sink - m_run);
                            R8 r;
                            const float* os = o[0].data() + ((size_t) t * heads + h) * D;
                            const float* ox = o[1].data() + ((size_t) t * heads + h) * D;
                            for (int c = 0; c < D; ++c) {
                                double oc = 0;
                                for (int j = 0; j < n; ++j) oc += w[j] * kc[j][c];
                                double U = 0, V = 0, Sa = 0, Ex = 0;
                                for (int j = 0; j < n; ++j) {
                                    const double u = std::fabs(kc[j][c] - oc);
                                    U += w[j] * a[j] * u;
                                    V += w[j] * std::fabs((double) kc[j][c]);
                                    Sa += std::fabs((double) kc[j][c]);
                                    Ex += w[j] * std::fabs(s[j] - mx) * u;
                                }
                                const double F = std::ldexp(514.0 * U + Ex + (n + 8.0) * V + (n + 4.0) * std::fabs(oc), -24);
                                const double H = std::ldexp(U + V, -11) + std::ldexp(Sa, -25);
                                const double bs = 2 * F, bx = 2 * H + 2 * F, ba = 2 * F + std::ldexp(V, -10) + std::ldexp(Sa, -24);
                                const double ec = oe[c] / dfin;
                                if (!std::isfinite(os[c]) || !std::isfinite(ox[c])) { r.nf = true; continue; }
                                const double ds = std::fabs(os[c] - oc), dx = std::fabs(ox[c] - oc), da = std::fabs(ox[c] - ec);
                                r.fs = std::max(r.fs, ds / bs);
                                r.fx = std::max(r.fx, dx / bx);
                                r.ax = std::max(r.ax, da / ba);
                                r.bb = std::max(r.bb, 2 * H + 4 * F);
                                r.ex_s = std::max(r.ex_s, ds);
                                r.ex_x = std::max(r.ex_x, dx);
                                r.ea_x = std::max(r.ea_x, da);
                                r.refmax = std::max(r.refmax, std::fabs(oc));
                            }
                            res[si] = r;
                        });
                        R8 m;
                        for (const auto& r : res) {
                            m.fs = std::max(m.fs, r.fs), m.fx = std::max(m.fx, r.fx), m.ax = std::max(m.ax, r.ax);
                            m.bb = std::max(m.bb, r.bb), m.ex_s = std::max(m.ex_s, r.ex_s), m.ex_x = std::max(m.ex_x, r.ex_x);
                            m.ea_x = std::max(m.ea_x, r.ea_x), m.refmax = std::max(m.refmax, r.refmax), m.nf = m.nf || r.nf;
                        }
                        const double bbound = 2 * m.bb;
                        const bool okB = bad == 0 && stray == 0 && bmax <= bbound;
                        const bool ok = okB && !m.nf && m.fs <= 1 && m.fx <= 1 && m.ax <= 1;
                        fails += !ok;
                        t_s_all += ms[0];
                        t_x_all += ms[1];
                        const double rm = std::max(m.refmax, 1e-30);
                        std::printf("%s: scalar %8.3f ms  xmx %8.3f ms (%5.2fx) | B max|d| %.2e (%.1e of max %.2e, %.2f of bound) | "
                                    "exact: scalar %.1e (rel %.1e, %.2f of bound) xmx %.1e (rel %.1e, %.2f of bound) | "
                                    "A xmx~emu %.1e (rel %.1e, %.2f of bound)  %s\n",
                                    head, ms[0], ms[1], ms[1] > 0 ? ms[0] / ms[1] : 0.0, bmax, bmax / std::max(smax, 1e-30), smax,
                                    bbound > 0 ? bmax / bbound : 0.0, m.ex_s, m.ex_s / rm, m.fs, m.ex_x, m.ex_x / rm, m.fx, m.ea_x,
                                    m.ea_x / rm, m.ax, ok ? "OK" : "FAIL");
                        if (bad) std::printf("  %zu outputs not finite (or not written) in a kernel\n", bad);
                        if (stray) std::printf("  %zu values past the outputs changed\n", stray);
                        if (m.nf) std::printf("  a sampled output is not finite\n");
                        std::fflush(stdout);
                    }
                    sycl::free(dwin, q);
                    sycl::free(dsel, q);
                    sycl::free(dq, q);
                    sycl::free(dsink, q);
                    sycl::free(dout[0], q);
                    sycl::free(dout[1], q);
                }
    sycl::free(dckv, q);
    std::printf("%d cases, %d failed; time in all: scalar %.1f ms, xmx %.1f ms (%.2fx)\n%s\n", cases, fails, t_s_all, t_x_all,
                t_x_all > 0 ? t_s_all / t_x_all : 0.0, fails ? "FAILED" : "all OK");
    return fails ? 1 : 0;
}
