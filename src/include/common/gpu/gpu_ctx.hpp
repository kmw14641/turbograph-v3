#pragma once
#include <cuda.h>
#include <atomic>
#include <mutex>

namespace duckdb {

bool EnsurePrimaryCudaContext(CUcontext *out_ctx = nullptr,
                              int device_index = 0,
                              unsigned flags = CU_CTX_SCHED_AUTO);

CUcontext GetPrimaryCudaContext();
CUdevice GetCudaDevice();

struct CudaCtxGuard {
    CUcontext prev{nullptr};
    explicit CudaCtxGuard(CUcontext ctx)
    {
        cuCtxGetCurrent(&prev);
        if (ctx)
            cuCtxSetCurrent(ctx);
    }
    ~CudaCtxGuard()
    {
        if (prev)
            cuCtxSetCurrent(prev);
        else
            cuCtxSetCurrent(nullptr);
    }
};

} // namespace duckdb