#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

#ifdef __CUDACC__
typedef short decimal_int16_t;
typedef int decimal_int32_t;
typedef long long decimal_int64_t;
typedef unsigned char decimal_uint8_t;
typedef struct { long long int low; long long int high; } decimal_int128_t;
#else
#include <cstdint>
typedef int16_t decimal_int16_t;
typedef int32_t decimal_int32_t;
typedef int64_t decimal_int64_t;
typedef uint8_t decimal_uint8_t;
typedef struct { long long int low; long long int high; } decimal_int128_t;
#endif

// namespace turbograph {
// namespace gpu {

static constexpr decimal_uint8_t MAX_WIDTH_INT16 = 4;
static constexpr decimal_uint8_t MAX_WIDTH_INT32 = 9;
static constexpr decimal_uint8_t MAX_WIDTH_INT64 = 18;
static constexpr decimal_uint8_t MAX_WIDTH_INT128 = 38;

struct DecimalMeta {
    decimal_uint8_t width;
    decimal_uint8_t scale;
    
    __host__ __device__ DecimalMeta() : width(0), scale(0) {}
    __host__ __device__ DecimalMeta(decimal_uint8_t w, decimal_uint8_t s) : width(w), scale(s) {}
};

__device__ __forceinline__ decimal_int16_t decimal_add_16(decimal_int16_t a, decimal_int16_t b, DecimalMeta meta) {
    return a + b;
}

__device__ __forceinline__ decimal_int32_t decimal_add_32(decimal_int32_t a, decimal_int32_t b, DecimalMeta meta) {
    return a + b;
}

__device__ __forceinline__ decimal_int64_t decimal_add_64(decimal_int64_t a, decimal_int64_t b, DecimalMeta meta) {
    return a + b;
}

__device__ __forceinline__ decimal_int128_t decimal_add_128(decimal_int128_t a, decimal_int128_t b, DecimalMeta meta) {
    decimal_int128_t result;
    result.low = a.low + b.low;
    unsigned long long carry = (unsigned long long)result.low < (unsigned long long)a.low ? 1 : 0;
    result.high = a.high + b.high + carry;
    return result;
}

__device__ __forceinline__ decimal_int16_t decimal_sub_16(decimal_int16_t a, decimal_int16_t b, DecimalMeta meta) {
    return a - b;
}

__device__ __forceinline__ decimal_int32_t decimal_sub_32(decimal_int32_t a, decimal_int32_t b, DecimalMeta meta) {
    return a - b;
}

__device__ __forceinline__ decimal_int64_t decimal_sub_64(decimal_int64_t a, decimal_int64_t b, DecimalMeta meta) {
    return a - b;
}

__device__ __forceinline__ decimal_int128_t decimal_sub_128(decimal_int128_t a, decimal_int128_t b, DecimalMeta meta) {
    decimal_int128_t result;
    result.low = a.low - b.low;
    unsigned long long borrow = (unsigned long long)a.low < (unsigned long long)b.low ? 1 : 0;
    result.high = a.high - b.high - borrow;
    return result;
}

__device__ __forceinline__ decimal_int16_t decimal_mul_16(decimal_int16_t a, decimal_int16_t b, DecimalMeta meta) {
    decimal_int32_t result = (decimal_int32_t)a * (decimal_int32_t)b;
    
    if (meta.scale > 0) {
        decimal_int32_t scale_divisor = 1;
        for (decimal_uint8_t i = 0; i < meta.scale; ++i) {
            scale_divisor *= 10;
        }
        result /= scale_divisor;
    }
    
    return (decimal_int16_t)result;
}

__device__ __forceinline__ decimal_int32_t decimal_mul_32(decimal_int32_t a, decimal_int32_t b, DecimalMeta meta) {
    decimal_int64_t result = (decimal_int64_t)a * (decimal_int64_t)b;
    
    if (meta.scale > 0) {
        decimal_int64_t scale_divisor = 1;
        for (decimal_uint8_t i = 0; i < meta.scale; ++i) {
            scale_divisor *= 10;
        }
        result /= scale_divisor;
    }
    
    return (decimal_int32_t)result;
}

__device__ __forceinline__ decimal_int64_t decimal_mul_64(decimal_int64_t a, decimal_int64_t b, DecimalMeta meta) {
    decimal_int64_t result = a * b;
    
    if (meta.scale > 0) {
        decimal_int64_t scale_divisor = 1;
        for (decimal_uint8_t i = 0; i < meta.scale; ++i) {
            scale_divisor *= 10;
        }
        result /= scale_divisor;
    }
    
    return result;
}

__device__ __forceinline__ decimal_int128_t decimal_mul_128(decimal_int128_t a, decimal_int128_t b, DecimalMeta meta) {
    decimal_int128_t result;
    
    unsigned long long a_low = (unsigned long long)a.low;
    unsigned long long b_low = (unsigned long long)b.low;
    unsigned long long prod = a_low * b_low;
    
    result.low = (long long)prod;
    result.high = 0;
    
    if (meta.scale > 0) {
        decimal_int64_t scale_divisor = 1;
        for (decimal_uint8_t i = 0; i < meta.scale; ++i) {
            scale_divisor *= 10;
        }
        result.low /= scale_divisor;
    }
    
    return result;
}

__device__ __forceinline__ decimal_int16_t decimal_div_16(decimal_int16_t a, decimal_int16_t b, DecimalMeta meta) {
    if (b == 0) return 0;

    decimal_int32_t scaled_a = a;
    if (meta.scale > 0) {
        for (decimal_uint8_t i = 0; i < meta.scale; ++i) {
            scaled_a *= 10;
        }
    }
    
    return (decimal_int16_t)(scaled_a / b);
}

__device__ __forceinline__ decimal_int32_t decimal_div_32(decimal_int32_t a, decimal_int32_t b, DecimalMeta meta) {
    if (b == 0) return 0;
    
    decimal_int64_t scaled_a = a;
    if (meta.scale > 0) {
        for (decimal_uint8_t i = 0; i < meta.scale; ++i) {
            scaled_a *= 10;
        }
    }
    
    return (decimal_int32_t)(scaled_a / b);
}

__device__ __forceinline__ decimal_int64_t decimal_div_64(decimal_int64_t a, decimal_int64_t b, DecimalMeta meta) {
    if (b == 0) return 0;
    
    decimal_int64_t scaled_a = a;
    if (meta.scale > 0) {
        for (decimal_uint8_t i = 0; i < meta.scale; ++i) {
            scaled_a *= 10;
        }
    }
    
    return scaled_a / b;
}

__device__ __forceinline__ decimal_int128_t decimal_div_128(decimal_int128_t a, decimal_int128_t b, DecimalMeta meta) {
    decimal_int128_t result;
    result.low = 0;
    result.high = 0;
    
    if (b.low == 0 && b.high == 0) {
        return result;
    }
    
    if (a.high == 0 && b.high == 0) {
        decimal_int64_t scaled_a = a.low;
        if (meta.scale > 0) {
            for (decimal_uint8_t i = 0; i < meta.scale; ++i) {
                scaled_a *= 10;
            }
        }
        result.low = scaled_a / b.low;
    }
    
    return result;
}

__device__ __forceinline__ bool decimal_eq_16(decimal_int16_t a, decimal_int16_t b, DecimalMeta meta) {
    return a == b;
}

__device__ __forceinline__ bool decimal_eq_32(decimal_int32_t a, decimal_int32_t b, DecimalMeta meta) {
    return a == b;
}

__device__ __forceinline__ bool decimal_eq_64(decimal_int64_t a, decimal_int64_t b, DecimalMeta meta) {
    return a == b;
}

__device__ __forceinline__ bool decimal_eq_128(decimal_int128_t a, decimal_int128_t b, DecimalMeta meta) {
    return (a.low == b.low) && (a.high == b.high);
}

__device__ __forceinline__ bool decimal_lt_16(decimal_int16_t a, decimal_int16_t b, DecimalMeta meta) {
    return a < b;
}

__device__ __forceinline__ bool decimal_lt_32(decimal_int32_t a, decimal_int32_t b, DecimalMeta meta) {
    return a < b;
}

__device__ __forceinline__ bool decimal_lt_64(decimal_int64_t a, decimal_int64_t b, DecimalMeta meta) {
    return a < b;
}

__device__ __forceinline__ bool decimal_lt_128(decimal_int128_t a, decimal_int128_t b, DecimalMeta meta) {
    if (a.high != b.high) {
        return a.high < b.high;
    }
    return (unsigned long long)a.low < (unsigned long long)b.low;
}

__device__ __forceinline__ bool decimal_lte_16(decimal_int16_t a, decimal_int16_t b, DecimalMeta meta) {
    return a <= b;
}

__device__ __forceinline__ bool decimal_lte_32(decimal_int32_t a, decimal_int32_t b, DecimalMeta meta) {
    return a <= b;
}

__device__ __forceinline__ bool decimal_lte_64(decimal_int64_t a, decimal_int64_t b, DecimalMeta meta) {
    return a <= b;
}

__device__ __forceinline__ bool decimal_lte_128(decimal_int128_t a, decimal_int128_t b, DecimalMeta meta) {
    if (a.high != b.high) {
        return a.high < b.high;
    }
    return (unsigned long long)a.low <= (unsigned long long)b.low;
}

__device__ __forceinline__ bool decimal_gt_16(decimal_int16_t a, decimal_int16_t b, DecimalMeta meta) {
    return a > b;
}

__device__ __forceinline__ bool decimal_gt_32(decimal_int32_t a, decimal_int32_t b, DecimalMeta meta) {
    return a > b;
}

__device__ __forceinline__ bool decimal_gt_64(decimal_int64_t a, decimal_int64_t b, DecimalMeta meta) {
    return a > b;
}

__device__ __forceinline__ bool decimal_gt_128(decimal_int128_t a, decimal_int128_t b, DecimalMeta meta) {
    if (a.high != b.high) {
        return a.high > b.high;
    }
    return (unsigned long long)a.low > (unsigned long long)b.low;
}

__device__ __forceinline__ bool decimal_gte_16(decimal_int16_t a, decimal_int16_t b, DecimalMeta meta) {
    return a >= b;
}

__device__ __forceinline__ bool decimal_gte_32(decimal_int32_t a, decimal_int32_t b, DecimalMeta meta) {
    return a >= b;
}

__device__ __forceinline__ bool decimal_gte_64(decimal_int64_t a, decimal_int64_t b, DecimalMeta meta) {
    return a >= b;
}

__device__ __forceinline__ bool decimal_gte_128(decimal_int128_t a, decimal_int128_t b, DecimalMeta meta) {
    if (a.high != b.high) {
        return a.high > b.high;
    }
    return (unsigned long long)a.low >= (unsigned long long)b.low;
}

__device__ __forceinline__ bool decimal_neq_16(decimal_int16_t a, decimal_int16_t b, DecimalMeta meta) {
    return a != b;
}

__device__ __forceinline__ bool decimal_neq_32(decimal_int32_t a, decimal_int32_t b, DecimalMeta meta) {
    return a != b;
}

__device__ __forceinline__ bool decimal_neq_64(decimal_int64_t a, decimal_int64_t b, DecimalMeta meta) {
    return a != b;
}

__device__ __forceinline__ bool decimal_neq_128(decimal_int128_t a, decimal_int128_t b, DecimalMeta meta) {
    return (a.low != b.low) || (a.high != b.high);
}

__device__ __forceinline__ decimal_int32_t scale_decimal_16_to_32(decimal_int16_t value, decimal_uint8_t old_scale, decimal_uint8_t new_scale) {
    decimal_int32_t result = value;
    if (new_scale > old_scale) {
        for (decimal_uint8_t i = 0; i < (new_scale - old_scale); ++i) {
            result *= 10;
        }
    } else if (old_scale > new_scale) {
        for (decimal_uint8_t i = 0; i < (old_scale - new_scale); ++i) {
            result /= 10;
        }
    }
    return result;
}

__device__ __forceinline__ decimal_int64_t scale_decimal_32_to_64(decimal_int32_t value, decimal_uint8_t old_scale, decimal_uint8_t new_scale) {
    decimal_int64_t result = value;
    if (new_scale > old_scale) {
        for (decimal_uint8_t i = 0; i < (new_scale - old_scale); ++i) {
            result *= 10;
        }
    } else if (old_scale > new_scale) {
        for (decimal_uint8_t i = 0; i < (old_scale - new_scale); ++i) {
            result /= 10;
        }
    }
    return result;
}

__device__ __forceinline__ decimal_int128_t scale_decimal_64_to_128(decimal_int64_t value, decimal_uint8_t old_scale, decimal_uint8_t new_scale) {
    decimal_int128_t result;
    result.low = value;
    result.high = (value < 0) ? -1 : 0;
    
    if (new_scale > old_scale) {
        for (decimal_uint8_t i = 0; i < (new_scale - old_scale); ++i) {
            decimal_int64_t old_low = result.low;
            result.low *= 10;
            if (result.low / 10 != old_low) {
                result.high = result.high * 10 + (result.low < 0 ? -1 : 1);
            } else {
                result.high *= 10;
            }
        }
    } else if (old_scale > new_scale) {
        for (decimal_uint8_t i = 0; i < (old_scale - new_scale); ++i) {
            result.low /= 10;
        }
    }
    return result;
}

#ifdef __CUDACC__
__device__ __forceinline__
void atomicAdd(decimal_int128_t* addr, decimal_int64_t val) {
    const unsigned long long add_lo = static_cast<unsigned long long>(val);

    const unsigned long long old_lo =
        atomicAdd(reinterpret_cast<unsigned long long *>(&addr->low), add_lo);
    const unsigned long long new_lo = old_lo + add_lo;

    const long long carry = (new_lo < old_lo) ? 1 : 0;

    atomicAdd(reinterpret_cast<unsigned long long *>(&addr->high),
              static_cast<unsigned long long>(carry));
}
#endif

// } // namespace gpu
// } // namespace turbograph

#ifdef __CUDACC__
__device__ __host__
#endif
inline decimal_int128_t make_decimal_int128(long long int low, long long int high) {
    decimal_int128_t result;
    result.low = low;
    result.high = high;
    return result;
}

#ifdef __CUDACC__
__device__ __host__
#endif
inline decimal_int128_t decimal_int128_zero() {
    decimal_int128_t result;
    result.low = 0;
    result.high = 0;
    return result;
}
