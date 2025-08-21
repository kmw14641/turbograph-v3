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

#define USE_NEW_STR_T 1

#ifdef USE_NEW_STR_T
struct __align__(16) str_t {
    union {
        struct {
            uint32_t length;
            char prefix[4];
            char *ptr;
        } pointer;
        struct {
            uint32_t length;
            char prefix[4];
            uint64_t offset;
        } offset;
        struct {
            uint32_t length;
            char inlined[12];
        } inlined;
    } value;

    __host__ __device__ inline uint32_t size() const
    {
        return value.pointer.length;
    }

   private:
    __host__ __device__ static inline bool bytes_equal(const char *a,
                                                       const char *b,
                                                       uint32_t n)
    {
        for (uint32_t i = 0; i < n; ++i) {
            if (a[i] != b[i])
                return false;
        }
        return true;
    }

    __host__ __device__ static inline uint32_t cstrlen(const char *s)
    {
        if (!s)
            return 0;
        uint32_t n = 0;
        while (s[n] != '\0')
            ++n;
        return n;
    }

    __host__ __device__ inline bool equals_raw(const char *s, uint32_t n) const
    {
        const uint32_t len = size();
        if (len != n)
            return false;
        if (len == 0)
            return true;

        if (len <= 12) {
            return bytes_equal(value.inlined.inlined, s, len);
        }

        if (!bytes_equal(value.pointer.prefix, s, 4))
            return false;

        const char *tail = value.pointer.ptr + 4;
        const char *s_tail = s + 4;
        const uint32_t tailN = len - 4;

        return bytes_equal(tail, s_tail, tailN);
    }

   public:
    __host__ __device__ inline bool equals(const char *s) const
    {
        return equals_raw(s, cstrlen(s));
    }

    template <size_t N>
    __host__ __device__ inline bool equals_lit(const char (&lit)[N]) const
    {
        const uint32_t n = (N > 0) ? static_cast<uint32_t>(N - 1) : 0u;
        return equals_raw(lit, n);
    }
};

// str_t == "literal"
template <size_t N>
__host__ __device__ inline bool operator==(const str_t& a, const char (&lit)[N]) {
    return a.equals_lit(lit);
}
template <size_t N>
__host__ __device__ inline bool operator!=(const str_t& a, const char (&lit)[N]) {
    return !operator==(a, lit);
}

// "literal" == str_t
template <size_t N>
__host__ __device__ inline bool operator==(const char (&lit)[N], const str_t& a) {
    return a.equals_lit(lit);
}
template <size_t N>
__host__ __device__ inline bool operator!=(const char (&lit)[N], const str_t& a) {
    return !operator==(lit, a);
}

// str_t == const char*
__host__ __device__ inline bool operator==(const str_t& a, const char* s) {
    return a.equals(s);
}
__host__ __device__ inline bool operator!=(const str_t& a, const char* s) {
    return !operator==(a, s);
}

// const char* == str_t
__host__ __device__ inline bool operator==(const char* s, const str_t& a) {
    return a.equals(s);
}
__host__ __device__ inline bool operator!=(const char* s, const str_t& a) {
    return !operator==(s, a);
}
#else
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
#endif

struct hugeint_t {
    uint64_t lower;
	int64_t upper;
};

#ifdef __CUDACC_RTC__
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

#endif