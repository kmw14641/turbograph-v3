#ifndef _TYPEDEF_CUH__
#define _TYPEDEF_CUH__

#ifdef __CUDACC_RTC__
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef int int32_t;
typedef unsigned uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
#endif

struct str_t {
    char *start;
    char *end;

    // bool operator==(const str_t &b)
    // {
    //     str_t &a = *this;
    //     int lena = end - start;
    //     int lenb = b.end - b.start;
    //     if (lena != lenb)
    //         return false;
    //     char *c = a.start;
    //     char *d = b.start;
    //     for (; c < a.end; c++, d++) {
    //         if (*c != *d)
    //             return false;
    //     }
    //     return true;
    // }
};

struct hugeint_t {
    uint64_t lower;
	int64_t upper;
};

__device__ inline bool operator==(const hugeint_t& a, const hugeint_t& b) {
    return a.upper == b.upper && a.lower == b.lower;
}
__device__ inline bool operator!=(const hugeint_t& a, const hugeint_t& b) {
    return !(a == b);
}
__device__ inline bool operator<(const hugeint_t& a, const hugeint_t& b) {
    if (a.upper != b.upper) return a.upper < b.upper;
    return a.lower < b.lower;
}
__device__ inline bool operator>(const hugeint_t& a, const hugeint_t& b) {
    return b < a;
}
__device__ inline bool operator<=(const hugeint_t& a, const hugeint_t& b) {
    return !(b < a);
}
__device__ inline bool operator>=(const hugeint_t& a, const hugeint_t& b) {
    return !(a < b);
}

__device__ inline void atomicAdd(hugeint_t* addr, int x) {
    const unsigned long long add_lo = static_cast<unsigned long long>(x);

    const unsigned long long old_lo =
        atomicAdd(reinterpret_cast<unsigned long long *>(&addr->lower), add_lo);
    const unsigned long long new_lo = old_lo + add_lo;

    const long long carry = (new_lo < old_lo) ? 1 : 0;

    atomicAdd(reinterpret_cast<unsigned long long *>(&addr->upper),
              static_cast<unsigned long long>(carry));
}

__device__ inline void atomicAdd(hugeint_t* addr, unsigned x) {
    const unsigned long long add_lo = static_cast<unsigned long long>(x);

    const unsigned long long old_lo =
        atomicAdd(reinterpret_cast<unsigned long long *>(&addr->lower), add_lo);
    const unsigned long long new_lo = old_lo + add_lo;

    const long long carry = (new_lo < old_lo) ? 1 : 0;

    atomicAdd(reinterpret_cast<unsigned long long *>(&addr->upper),
              static_cast<unsigned long long>(carry));
}

__device__ inline void atomicAdd(hugeint_t* addr, long x) {
    const unsigned long long add_lo = static_cast<unsigned long long>(x);

    const unsigned long long old_lo =
        atomicAdd(reinterpret_cast<unsigned long long *>(&addr->lower), add_lo);
    const unsigned long long new_lo = old_lo + add_lo;

    const long long carry = (new_lo < old_lo) ? 1 : 0;

    atomicAdd(reinterpret_cast<unsigned long long *>(&addr->upper),
              static_cast<unsigned long long>(carry));
}

__device__ inline void atomicAdd(hugeint_t* addr, unsigned long x) {
    const unsigned long long add_lo = static_cast<unsigned long long>(x);

    const unsigned long long old_lo =
        atomicAdd(reinterpret_cast<unsigned long long *>(&addr->lower), add_lo);
    const unsigned long long new_lo = old_lo + add_lo;

    const long long carry = (new_lo < old_lo) ? 1 : 0;

    atomicAdd(reinterpret_cast<unsigned long long *>(&addr->upper),
              static_cast<unsigned long long>(carry));
}

__device__ inline void atomicAdd(hugeint_t* addr, long long x) {
    const unsigned long long add_lo = static_cast<unsigned long long>(x);

    const unsigned long long old_lo =
        atomicAdd(reinterpret_cast<unsigned long long *>(&addr->lower), add_lo);
    const unsigned long long new_lo = old_lo + add_lo;

    const long long carry = (new_lo < old_lo) ? 1 : 0;

    atomicAdd(reinterpret_cast<unsigned long long *>(&addr->upper),
              static_cast<unsigned long long>(carry));
}

__device__ inline void atomicAdd(hugeint_t* addr, unsigned long long x) {
    const unsigned long long add_lo = static_cast<unsigned long long>(x);

    const unsigned long long old_lo =
        atomicAdd(reinterpret_cast<unsigned long long *>(&addr->lower), add_lo);
    const unsigned long long new_lo = old_lo + add_lo;

    const long long carry = (new_lo < old_lo) ? 1 : 0;

    atomicAdd(reinterpret_cast<unsigned long long *>(&addr->upper),
              static_cast<unsigned long long>(carry));
}

__device__ inline void atomicAdd(hugeint_t *addr, hugeint_t x) {
    const unsigned long long add_lo = static_cast<unsigned long long>(x.lower);

    const unsigned long long old_lo =
        atomicAdd(reinterpret_cast<unsigned long long *>(&addr->lower), add_lo);
    const unsigned long long new_lo = old_lo + add_lo;

    const long long carry = (new_lo < old_lo) ? 1 : 0;

    const long long hi_inc = x.upper + carry;
    atomicAdd(reinterpret_cast<unsigned long long *>(&addr->upper),
              static_cast<unsigned long long>(hi_inc));
}

#endif