// Copyright (c) 2026 Peasant Smith and the Project Maya contributors (MIT).
// Ported into Strata for Intel Arc Pro B70 (Xe2 / bmg-g31).
// The SYCL build has no AMD WMMA tiles (gfx11 / gfx12): launch() is never taken and the test hooks report no device.
#pragma once
#include <atomic>

namespace hipk {
inline std::atomic<int>& force_mode() { static std::atomic<int> f{-1}; return f; }
inline std::atomic<long long>& launches() { static std::atomic<long long> n{0}; return n; }
inline bool ok_device() { return false; }
template <int T, bool GU>
inline bool launch(const SetArgs& a, cudaStream_t s) {
    (void) a;
    (void) s;
    return false;
}
}  // namespace hipk
