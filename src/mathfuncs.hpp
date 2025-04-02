#pragma once
#pragma GCC target("avx", "fma")

#include <cmath>
#include <chrono>

#define _HIGH_PERF [[nodiscard]] [[using gnu: hot, always_inline]] inline
#define _HIGH_PERF_VOID [[using gnu: hot, always_inline]] inline

_HIGH_PERF double
sigmoid(double value) noexcept {
    return 1.0 / (1.0 + std::exp(-value));
}

_HIGH_PERF double
dx_sigmoid(double value) noexcept {
    return value * (1.0 - value);
}

_HIGH_PERF double
swish(double value) noexcept {
    return value * sigmoid(value);
}

_HIGH_PERF double
dx_swish(double value) noexcept {
    const double s = sigmoid(value);
    return s + value * s * (1.0 - s);
}

_HIGH_PERF double
genHe(int qtyInputs) {
    std::srand((unsigned int)std::time(0));
    return ((static_cast<double>(std::rand()) / RAND_MAX) * 2.0 - 1.0) * std::sqrt(2.0 / qtyInputs);
}