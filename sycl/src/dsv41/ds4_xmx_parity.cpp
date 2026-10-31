// ds4_xmx_parity: the prompt sets' XMX tiles (ds4_iq.dp.cpp xmxs, DS4_XMX=1) against the FMA tiles they replace (set_kernel),
// on random but valid expert blocks of each type, random activations, several set shapes (most rows of an expert, experts in
// the set).  Both run through iq::set_tiles on the same inputs; each kernel is timed (host clock around `iters` launches and a
// wait, after one warm-up launch).  For IQ1_M, IQ2_XXS and IQ2_XS a host reference in double precision on sampled rows as well,
// from ggml's dequantize_row_* (written here from ggml's definitions, not from the device decoder):
//   - "emu": weights and activations rounded to FP16 first (what the XMX tiles multiply), sums in double;
//   - "f32": the F32 values (what set_kernel multiplies), sums in double.
// Pass / fail (a case passes when every check it runs passes):
//   B  XMX vs set_kernel, every output: max |d| <= 5e-3 max |set_kernel|.  The two differ by the FP16 rounding of both operands
//      (relative 2^-11 each, 2^-10 for a product at most).  The errors of K such products have random signs: a dot's error is
//      about 2^-10 / sqrt(3) of the root of its squared products, i.e. ~6e-4 of the outputs' RMS, ~1e-4 of their max with
//      5120 / 2304-long rows; the largest of millions of outputs ~5 sigma.  5e-3 leaves about 10x over that.
//   A  XMX vs "emu" on the sampled rows: |d| <= 1e-4 max |emu| (+ 2^-10 |emu| for gate / up, whose F16 output can round to
//      the neighbour of the reference's): the same products, only the order of the F32 sums differs.  set_kernel vs "f32" with
//      the same bound checks the reference itself (a failure there is the test's, not the kernel's: "ref?").
// Usage: ds4_xmx_parity [--types=29,16,17|all] [--rows=1,7,16,17,64,200,1000] [--experts=1,6,32] [--n-embd=5120]
//                       [--n-ff=2304] [--iters=5] [--no-ref] [--seed=1]
// (defaults: DeepSeek V4.1 Flash's expert shape, swiglu limit 10.)  The light kernels' split is the engine's: a set whose
// most rows exceed 16 runs its tiles with skip_le 16.  Exit 0: every case passed.
#include "ds4_iq.cuh"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

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

namespace iq = strata::ds4::gguf::iq;

static float h2f(uint16_t h) { return (float) sycl::bit_cast<sycl::half>(h); }
static uint16_t f2h(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }
static float rh(float f) { return (float) sycl::half(f); }   // F32 -> F16 (round to nearest even) -> F32

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

// ---------------------------------------------------------------- random valid blocks
// weights of about 0.03: |w| ~ factor d, so d = 0.03 / factor (times 0.5 .. 1.5 a block)
static double w_factor(int t) {
    switch (t) {
        case iq::kIQ2_XXS: case iq::kIQ2_XS: case iq::kIQ2_S: return 40;
        case iq::kIQ3_XXS: return 100;
        case iq::kIQ3_S: return 400;
        case iq::kIQ4_XS: return 800;
        case iq::kIQ4_NL: return 50;
        case iq::kIQ1_S: case iq::kIQ1_M: return 5;
        case iq::kQ4_K: return 240;
        case iq::kQ2_K: return 11;
        case iq::kQ3_K: return 32;
        default: return 40;
    }
}
static void put16(uint8_t* p, uint16_t v) { p[0] = (uint8_t) v; p[1] = (uint8_t) (v >> 8); }
// every bit pattern of these blocks is valid (grid indices in range, any sign or scale bits) except the F16 scales: set
static void fill_blocks(int t, uint8_t* p, size_t bytes, std::mt19937& rng) {
    const int B = iq::block_bytes(t);
    std::uniform_real_distribution<float> u(0.5f, 1.5f);
    for (size_t i = 0; i < bytes; ++i) p[i] = (uint8_t) rng();
    for (size_t o = 0; o + B <= bytes; o += B) {
        uint8_t* b = p + o;
        const uint16_t d = f2h((float) (0.03 / w_factor(t)) * u(rng));
        switch (t) {
            case iq::kIQ1_M: {   // the F16 scale is the top nibbles of the four u16 scales at 48
                for (int k = 0; k < 4; ++k) {
                    uint16_t sc = (uint16_t) (b[48 + 2 * k] | b[49 + 2 * k] << 8);
                    sc = (uint16_t) ((sc & 0x0fffu) | (((d >> (4 * k)) & 15u) << 12));
                    put16(b + 48 + 2 * k, sc);
                }
                break;
            }
            case iq::kQ4_K: put16(b, d); put16(b + 2, d); break;
            case iq::kQ2_K: put16(b + 80, d); put16(b + 82, d); break;
            case iq::kQ3_K: put16(b + 108, d); break;
            default: put16(b, d); break;   // the i-quants: d first
        }
    }
}

// ---------------------------------------------------------------- host decode (ggml's dequantize_row_*), F32 as ggml computes it
static bool host_type(int t) { return t == iq::kIQ2_XXS || t == iq::kIQ2_XS || t == iq::kIQ1_M; }
static void host_row(int t, const uint8_t* row, int k, float* y) {
    const int B = iq::block_bytes(t);
    for (int i = 0; i < k / 256; ++i, y += 256) {
        const uint8_t* b = row + (size_t) i * B;
        if (t == iq::kIQ2_XXS) {
            const float d = h2f((uint16_t) (b[0] | b[1] << 8));
            for (int ib = 0; ib < 8; ++ib) {
                uint32_t aux[2];
                std::memcpy(aux, b + 2 + 8 * ib, 8);
                const uint8_t* a8 = (const uint8_t*) &aux[0];
                const float db = d * (0.5f + (float) (aux[1] >> 28)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    const uint8_t* g = (const uint8_t*) (iq2xxs_grid + a8[l]);
                    const uint8_t s = ksigns_iq2xs[(aux[1] >> (7 * l)) & 127];
                    for (int j = 0; j < 8; ++j) y[32 * ib + 8 * l + j] = db * (float) g[j] * (s & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
        } else if (t == iq::kIQ2_XS) {
            const float d = h2f((uint16_t) (b[0] | b[1] << 8));
            for (int ib = 0; ib < 8; ++ib) {
                const uint8_t sc = b[66 + ib];
                const float db[2] = {d * (0.5f + (float) (sc & 15)) * 0.25f, d * (0.5f + (float) (sc >> 4)) * 0.25f};
                for (int l = 0; l < 4; ++l) {
                    const uint16_t q = (uint16_t) (b[2 + 2 * (4 * ib + l)] | b[3 + 2 * (4 * ib + l)] << 8);
                    const uint8_t* g = (const uint8_t*) (iq2xs_grid + (q & 511));
                    const uint8_t s = ksigns_iq2xs[q >> 9];
                    for (int j = 0; j < 8; ++j) y[32 * ib + 8 * l + j] = db[l / 2] * (float) g[j] * (s & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
        } else {   // IQ1_M: qs[32] at 0, qh[16] at 32, scales[8] at 48
            uint16_t sc[4];
            for (int j = 0; j < 4; ++j) sc[j] = (uint16_t) (b[48 + 2 * j] | b[49 + 2 * j] << 8);
            const uint16_t s16 = (uint16_t) ((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
            const float d = h2f(s16);
            const uint8_t* qs = b;
            const uint8_t* qh = b + 32;
            for (int ib = 0; ib < 8; ++ib, qs += 4, qh += 2) {
                const float dl1 = d * (float) (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 7) + 1);
                const float dl2 = d * (float) (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 7) + 1);
                const int idx[4] = {qs[0] | ((qh[0] << 8) & 0x700), qs[1] | ((qh[0] << 4) & 0x700), qs[2] | ((qh[1] << 8) & 0x700),
                                    qs[3] | ((qh[1] << 4) & 0x700)};
                const float delta[4] = {qh[0] & 0x08 ? -IQ1M_DELTA : IQ1M_DELTA, qh[0] & 0x80 ? -IQ1M_DELTA : IQ1M_DELTA,
                                        qh[1] & 0x08 ? -IQ1M_DELTA : IQ1M_DELTA, qh[1] & 0x80 ? -IQ1M_DELTA : IQ1M_DELTA};
                for (int l = 0; l < 4; ++l) {
                    const int8_t* g = (const int8_t*) (iq1s_grid + idx[l]);
                    for (int j = 0; j < 8; ++j) y[32 * ib + 8 * l + j] = (l < 2 ? dl1 : dl2) * ((float) g[j] + delta[l]);
                }
            }
        }
    }
}

static float swiglu_h(float g, float u, float limit) {   // ds4_iq.dp.cpp's swiglu
    if (limit > 0.0f) {
        g = std::fmin(g, limit);
        u = std::fmin(std::fmax(u, -limit), limit);
    }
    return (g / (1.0f + std::exp(-g))) * u;
}

struct Err {
    double max_abs = 0, max_ref = 0, max_rel = 0;
    size_t nonfin = 0, n = 0;
};
// d against r; rel with a floor of 1e-3 max |r| (set once max_ref is known: two passes)
static Err compare(const std::vector<float>& d, const std::vector<float>& r) {
    Err e;
    for (size_t i = 0; i < r.size(); ++i)
        if (std::isfinite(r[i])) e.max_ref = std::max(e.max_ref, (double) std::fabs(r[i]));
    const double floor = std::max(1e-3 * e.max_ref, 1e-30);
    for (size_t i = 0; i < r.size(); ++i) {
        const bool fd = std::isfinite(d[i]), fr = std::isfinite(r[i]);
        if (!fd || !fr) {
            if (fd != fr || std::memcmp(&d[i], &r[i], 4) != 0) ++e.nonfin;
            continue;
        }
        const double a = std::fabs((double) d[i] - (double) r[i]);
        e.max_abs = std::max(e.max_abs, a);
        e.max_rel = std::max(e.max_rel, a / std::max((double) std::fabs(r[i]), floor));
        ++e.n;
    }
    return e;
}

int main(int argc, char** argv) {
    std::vector<int> types = {iq::kIQ1_M, iq::kIQ2_XXS, iq::kIQ2_XS};
    std::vector<int> rows = {1, 7, 16, 17, 64, 200, 1000};
    std::vector<int> experts = {1, 6, 32};
    int n_embd = 5120, n_ff = 2304, iters = 5;
    bool ref = true;
    unsigned seed = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&](const char* k) { return a.rfind(k, 0) == 0 ? a.substr(std::strlen(k)) : std::string(); };
        if (a == "--types=all")
            types = {iq::kIQ1_M, iq::kIQ2_XXS, iq::kIQ2_XS, iq::kIQ2_S, iq::kIQ3_XXS, iq::kIQ3_S, iq::kIQ4_XS, iq::kIQ1_S,
                     iq::kIQ4_NL, iq::kQ4_K, iq::kQ2_K, iq::kQ3_K};
        else if (!val("--types=").empty()) types = parse_list(val("--types="));
        else if (!val("--rows=").empty()) rows = parse_list(val("--rows="));
        else if (!val("--experts=").empty()) experts = parse_list(val("--experts="));
        else if (!val("--n-embd=").empty()) n_embd = std::atoi(val("--n-embd=").c_str());
        else if (!val("--n-ff=").empty()) n_ff = std::atoi(val("--n-ff=").c_str());
        else if (!val("--iters=").empty()) iters = std::max(1, std::atoi(val("--iters=").c_str()));
        else if (!val("--seed=").empty()) seed = (unsigned) std::atoi(val("--seed=").c_str());
        else if (a == "--no-ref") ref = false;
        else {
            std::fprintf(stderr, "usage: %s [--types=29,16,17|all] [--rows=1,7,...] [--experts=1,6,32] [--n-embd=5120] "
                                 "[--n-ff=2304] [--iters=5] [--no-ref] [--seed=1]\n", argv[0]);
            return 2;
        }
    }
    sycl::queue& q = *strata::q_of(nullptr);
    std::printf("ds4_xmx_parity on %s: n_embd %d, n_ff %d, swiglu limit 10, %d timed launches a kernel\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), n_embd, n_ff, iters);
    const float limit = 10.0f;
    const int row0 = 3;   // (the sets' rows start past 0, as a chunk's do)
    std::mt19937 rng(seed);
    int fails = 0, cases = 0;
    double t_fma_all = 0, t_xmx_all = 0;
    for (int t : types) {
        if (!iq::supported(t) || iq::block_values(t) == 0) { std::printf("type %d: not a device type, skipped\n", t); continue; }
        for (int gu = 1; gu >= 0; --gu) {
            const int K = gu ? n_embd : n_ff, N = gu ? n_ff : n_embd;
            const size_t rb = iq::row_bytes(t, K), mat = rb * (size_t) N;
            const size_t stride = ((gu ? 2 : 1) * mat + 255) / 256 * 256;
            for (int E : experts) {
                // the experts' blobs: gate then up (gate / up), or down alone
                std::vector<uint8_t> hb(stride * (size_t) E);
                for (int e = 0; e < E; ++e) fill_blocks(t, hb.data() + (size_t) e * stride, (gu ? 2 : 1) * mat, rng);
                uint8_t* dw = sycl::malloc_device<uint8_t>(hb.size(), q);
                q.memcpy(dw, hb.data(), hb.size()).wait();
                std::vector<std::vector<float>> deq;   // (host decode, lazily: [e * 2 + m] = matrix m of expert e, N x K)
                for (int M : rows) {
                    if (M < 1) continue;
                    std::vector<int> cnt(E);
                    std::uniform_int_distribution<int> cd(0, M);
                    for (int e = 0; e < E; ++e) cnt[e] = e == 0 ? M : cd(rng);
                    std::vector<int> bounds(E + 1);
                    bounds[0] = row0;
                    for (int e = 0; e < E; ++e) bounds[e + 1] = bounds[e] + cnt[e];
                    const int total = bounds[E] - row0;
                    const int skip_le = M > 16 ? 16 : 0;
                    // activations: gate / up F32 rows (ntok of them, idx[local] picks one), down F16 rows (one a local row)
                    const int ntok = std::max(1, total), ldx = K + 16, ldo = N + 32;
                    std::normal_distribution<float> nd(0.f, 1.f);
                    std::vector<float> hxf;
                    std::vector<uint16_t> hxh;
                    std::vector<int> hidx(total);
                    if (gu) {
                        hxf.resize((size_t) ntok * ldx);
                        for (auto& v : hxf) v = nd(rng);
                        std::uniform_int_distribution<int> td(0, ntok - 1);
                        for (auto& v : hidx) v = td(rng);
                    } else {
                        hxh.resize((size_t) total * ldx);
                        for (auto& v : hxh) v = f2h(nd(rng));
                    }
                    int* dbounds = sycl::malloc_device<int>(E + 1, q);
                    int* didx = gu && total > 0 ? sycl::malloc_device<int>(total, q) : nullptr;
                    float* dxf = gu ? sycl::malloc_device<float>(hxf.size(), q) : nullptr;
                    uint16_t* dxh = gu ? nullptr : sycl::malloc_device<uint16_t>(std::max<size_t>(1, hxh.size()), q);
                    const size_t outn = (size_t) std::max(1, total) * ldo;
                    uint16_t* dh[2] = {nullptr, nullptr};
                    float* dy[2] = {nullptr, nullptr};
                    for (int k = 0; k < 2; ++k) {
                        if (gu) { dh[k] = sycl::malloc_device<uint16_t>(outn, q); q.fill(dh[k], (uint16_t) 0x5a5a, outn); }
                        else { dy[k] = sycl::malloc_device<float>(outn, q); q.fill(dy[k], 1234.5f, outn); }
                    }
                    q.memcpy(dbounds, bounds.data(), sizeof(int) * (E + 1));
                    if (didx) q.memcpy(didx, hidx.data(), sizeof(int) * total);
                    if (dxf) q.memcpy(dxf, hxf.data(), sizeof(float) * hxf.size());
                    if (dxh && !hxh.empty()) q.memcpy(dxh, hxh.data(), 2 * hxh.size());
                    q.wait();
                    iq::SetArgs a;
                    a.wbase = dw;
                    a.stride = stride;
                    a.tr[0] = 0;
                    a.tr[1] = gu ? mat : 0;
                    a.tr[2] = 0;
                    a.bounds = dbounds;
                    a.row0 = row0;
                    a.max_rows = M;
                    a.n_exp = E;
                    a.n_embd = n_embd;
                    a.n_ff = n_ff;
                    a.limit = limit;
                    a.xf = dxf;
                    a.idx = didx;
                    a.xh = dxh;
                    a.ldx = ldx;
                    a.ldo = ldo;
                    a.tg = t;
                    a.td = t;
                    a.skip_le = skip_le;
                    double ms[2] = {0, 0};
                    bool launched = true;
                    for (int k = 0; k < 2; ++k) {   // 0: set_kernel (FMA), 1: XMX
                        iq::SetArgs ak = a;
                        ak.h = dh[k];
                        ak.y = dy[k];
                        launched = launched && iq::set_tiles(ak, gu != 0, k == 1, nullptr);
                        q.wait();
                        const auto t0 = std::chrono::steady_clock::now();
                        for (int r = 0; r < iters && launched; ++r) iq::set_tiles(ak, gu != 0, k == 1, nullptr);
                        q.wait();
                        ms[k] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
                    }
                    ++cases;
                    if (!launched) {
                        std::printf("type %-7s %-7s rows %4d experts %2d: NOT LAUNCHED (set_tiles refused)\n", iq::name(t),
                                    gu ? "gate/up" : "down", M, E);
                        ++fails;
                    } else {
                        // B: every output
                        std::vector<float> o[2];
                        for (int k = 0; k < 2; ++k) {
                            o[k].resize(outn);
                            if (gu) {
                                std::vector<uint16_t> hh(outn);
                                q.memcpy(hh.data(), dh[k], 2 * outn).wait();
                                for (size_t i = 0; i < outn; ++i) o[k][i] = h2f(hh[i]);
                            } else {
                                q.memcpy(o[k].data(), dy[k], 4 * outn).wait();
                            }
                        }
                        // the rows the tiles write (experts of more than skip_le rows, N outputs each); everything else must
                        // keep the fill in both (bit for bit)
                        std::vector<float> wx, wf;
                        size_t stray = 0;
                        {
                            std::vector<char> live(std::max(1, total), 0);
                            for (int e = 0; e < E; ++e)
                                if (cnt[e] > skip_le)
                                    for (int r = bounds[e] - row0; r < bounds[e + 1] - row0; ++r) live[r] = 1;
                            for (int r = 0; r < total; ++r)
                                for (int c = 0; c < ldo; ++c) {
                                    const size_t i = (size_t) r * ldo + c;
                                    if (live[r] && c < N) { wx.push_back(o[1][i]); wf.push_back(o[0][i]); }
                                    else if (std::memcmp(&o[0][i], &o[1][i], 4) != 0 ||
                                             (gu ? o[0][i] != h2f(0x5a5a) : o[0][i] != 1234.5f)) ++stray;
                                }
                        }
                        const Err eb = compare(wx, wf);
                        bool ok = eb.nonfin == 0 && stray == 0 && eb.max_abs <= 5e-3 * eb.max_ref;
                        // A: sampled rows against the host references
                        std::string astr;
                        if (ref && host_type(t)) {
                            struct S { int e, lr; };
                            std::vector<S> smp;
                            for (int e = 0; e < E; ++e) {
                                if (cnt[e] == 0 || cnt[e] <= skip_le) continue;
                                const int lo = bounds[e] - row0;
                                smp.push_back({e, lo});
                                if (cnt[e] > 1) smp.push_back({e, lo + cnt[e] - 1});
                                if (cnt[e] > 2) smp.push_back({e, lo + 1 + (int) (rng() % (unsigned) (cnt[e] - 2))});
                            }
                            std::shuffle(smp.begin(), smp.end(), rng);
                            if (smp.size() > 24) smp.resize(24);
                            std::sort(smp.begin(), smp.end(), [](const S& x, const S& y) { return x.e < y.e; });
                            if (deq.empty()) deq.resize((size_t) E * 2);
                            std::vector<float> re((size_t) smp.size() * N), rf((size_t) smp.size() * N), ox(re.size()), of(re.size());
                            for (size_t si = 0; si < smp.size(); ++si) {
                                const int e = smp[si].e, lr = smp[si].lr;
                                for (int m = 0; m < (gu ? 2 : 1); ++m) {
                                    auto& dm = deq[(size_t) e * 2 + m];
                                    if (dm.empty()) {
                                        dm.resize((size_t) N * K);
                                        const uint8_t* base = hb.data() + (size_t) e * stride + (size_t) m * mat;
                                        par_for(N, [&](int r) { host_row(t, base + (size_t) r * rb, K, dm.data() + (size_t) r * K); });
                                    }
                                }
                                std::vector<float> xr(K);
                                for (int c = 0; c < K; ++c)
                                    xr[c] = gu ? hxf[(size_t) hidx[lr] * ldx + c] : h2f(hxh[(size_t) lr * ldx + c]);
                                par_for(N, [&](int c) {
                                    double se[2] = {0, 0}, sf[2] = {0, 0};
                                    for (int m = 0; m < (gu ? 2 : 1); ++m) {
                                        const float* w = deq[(size_t) e * 2 + m].data() + (size_t) c * K;
                                        for (int k = 0; k < K; ++k) {
                                            se[m] += (double) rh(w[k]) * (double) rh(xr[k]);
                                            sf[m] += (double) w[k] * (double) xr[k];
                                        }
                                    }
                                    const size_t at = si * N + c;
                                    if (gu) {
                                        re[at] = rh(swiglu_h((float) se[0], (float) se[1], limit));
                                        rf[at] = rh(swiglu_h((float) sf[0], (float) sf[1], limit));
                                    } else {
                                        re[at] = (float) se[0];
                                        rf[at] = (float) sf[0];
                                    }
                                    ox[at] = o[1][(size_t) lr * ldo + c];
                                    of[at] = o[0][(size_t) lr * ldo + c];
                                });
                            }
                            if (!smp.empty()) {
                                // bound: 1e-4 max |ref| (+ 2^-10 |ref| an element for the F16 outputs)
                                auto check = [&](const std::vector<float>& d, const std::vector<float>& r, Err& er) {
                                    er = compare(d, r);
                                    bool good = er.nonfin == 0;
                                    for (size_t i = 0; i < r.size(); ++i)
                                        if (std::isfinite(r[i]) && std::isfinite(d[i]) &&
                                            std::fabs((double) d[i] - r[i]) > 1e-4 * er.max_ref + (gu ? std::ldexp(std::fabs((double) r[i]), -10) : 0.0))
                                            good = false;
                                    return good;
                                };
                                Err ea, er, ex;
                                const bool okA = check(ox, re, ea), okR = check(of, rf, er);
                                ex = compare(ox, rf);
                                char buf[256];
                                std::snprintf(buf, sizeof buf, " | A xmx~emu %.1e%s fma~f32 %.1e%s xmx~f32 %.1e (%zu rows)",
                                              ea.max_abs / std::max(ea.max_ref, 1e-30), okA ? "" : "!", er.max_abs / std::max(er.max_ref, 1e-30),
                                              okR ? "" : " ref?", ex.max_abs / std::max(ex.max_ref, 1e-30), smp.size());
                                astr = buf;
                                ok = ok && okA && okR;
                            }
                        }
                        fails += !ok;
                        t_fma_all += ms[0];
                        t_xmx_all += ms[1];
                        std::printf("type %-7s %-7s rows %4d experts %2d (%6d rows): fma %8.3f ms  xmx %8.3f ms (%5.2fx) | "
                                    "B max|d| %.2e of max %.2e = %.1e, rel %.1e%s%s  %s\n",
                                    iq::name(t), gu ? "gate/up" : "down", M, E, total, ms[0], ms[1], ms[1] > 0 ? ms[0] / ms[1] : 0.0,
                                    eb.max_abs, eb.max_ref, eb.max_abs / std::max(eb.max_ref, 1e-30), eb.max_rel,
                                    eb.nonfin ? " NONFINITE-MISMATCH" : "", astr.c_str(), ok ? "OK" : "FAIL");
                        if (stray) std::printf("  %zu outputs outside the set's rows changed (or differ between the kernels)\n", stray);
                        std::fflush(stdout);
                    }
                    sycl::free(dbounds, q);
                    if (didx) sycl::free(didx, q);
                    if (dxf) sycl::free(dxf, q);
                    if (dxh) sycl::free(dxh, q);
                    for (int k = 0; k < 2; ++k) {
                        if (dh[k]) sycl::free(dh[k], q);
                        if (dy[k]) sycl::free(dy[k], q);
                    }
                }
                sycl::free(dw, q);
            }
        }
    }
    std::printf("%d cases, %d failed; time in all: fma %.1f ms, xmx %.1f ms (%.2fx)\n%s\n", cases, fails, t_fma_all, t_xmx_all,
                t_xmx_all > 0 ? t_fma_all / t_xmx_all : 0.0, fails ? "FAILED" : "all OK");
    return fails ? 1 : 0;
}
