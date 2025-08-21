#include "common/gpu/gpu_ctx.hpp"
#include <cstdio>

namespace duckdb {

static std::atomic<CUcontext> g_ctx{nullptr};
static CUdevice g_dev = 0;
static std::mutex g_mu;

bool EnsurePrimaryCudaContext(CUcontext *out_ctx, int device_index,
                              unsigned flags)
{
    CUcontext ctx = g_ctx.load(std::memory_order_acquire);
    if (ctx) {
        if (out_ctx)
            *out_ctx = ctx;
        return true;
    }

    std::lock_guard<std::mutex> lk(g_mu);

    ctx = g_ctx.load(std::memory_order_acquire);
    if (ctx) {
        if (out_ctx)
            *out_ctx = ctx;
        return true;
    }

    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "cuInit failed\n");
        return false;
    }

    r = cuDeviceGet(&g_dev, device_index);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "cuDeviceGet failed\n");
        return false;
    }

    int active = 0;
    unsigned int curFlags = 0;
    cuDevicePrimaryCtxGetState(g_dev, &curFlags, &active);

    if (!active) {
        CUresult s = cuDevicePrimaryCtxSetFlags(g_dev, flags);
        if (s != CUDA_SUCCESS && s != CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE) {
            fprintf(stderr, "cuDevicePrimaryCtxSetFlags failed\n");
        }
    }

    r = cuDevicePrimaryCtxRetain(&ctx, g_dev);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "cuDevicePrimaryCtxRetain failed\n");
        return false;
    }

    cuCtxSetCurrent(ctx);

    g_ctx.store(ctx, std::memory_order_release);
    if (out_ctx)
        *out_ctx = ctx;
    return true;
}

CUcontext GetPrimaryCudaContext()
{
    return g_ctx.load(std::memory_order_acquire);
}

CUdevice GetCudaDevice()
{
    return g_dev;
}

}  // namespace duckdb
