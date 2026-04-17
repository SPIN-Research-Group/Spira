#pragma once
#include <cstdint>
#include <type_traits>

template <typename CT>
struct Layout;

// =====================
// PACK 32 layout
// =====================
template <>
struct Layout<uint32_t> {
    static constexpr int Z_BITS = 8;
    static constexpr int Y_BITS = 12;
    static constexpr int X_BITS = 12;

    static constexpr int Y_SHIFT = Z_BITS;
    static constexpr int X_SHIFT = Z_BITS + Y_BITS;

    static constexpr uint32_t Z_ONES = (1u << Z_BITS) - 1u; 
    static constexpr uint32_t Y_ONES = (1u << Y_BITS) - 1u;  
    static constexpr uint32_t X_ONES = (1u << X_BITS) - 1u;  

    static constexpr int TOTAL_BITS = X_BITS + Y_BITS + Z_BITS;
};

// =====================
// PACK 64 layout
// =====================
template <>
struct Layout<uint64_t> {
    static constexpr int Z_BITS = 16;
    static constexpr int Y_BITS = 24;
    static constexpr int X_BITS = 24;

    static constexpr int Y_SHIFT = Z_BITS;
    static constexpr int X_SHIFT = Z_BITS + Y_BITS;

    static constexpr uint64_t Z_ONES = (1ull << Z_BITS) - 1ull;
    static constexpr uint64_t Y_ONES = (1ull << Y_BITS) - 1ull;
    static constexpr uint64_t X_ONES = (1ull << X_BITS) - 1ull;

    static constexpr int TOTAL_BITS = X_BITS + Y_BITS + Z_BITS;
};

// Compile-time safety
static_assert(Layout<uint32_t>::TOTAL_BITS == 32);
static_assert(Layout<uint64_t>::TOTAL_BITS == 64);