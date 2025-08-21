#include "storage/cache/gpu_chunk_cache_manager.h"
#include <cuda_runtime.h>
#include "velox/experimental/wave/common/Cuda.h"
#include <nvrtc.h>
#include "common/gpu/gpu_ctx.hpp"
#include "common/gpu/gpu_utils.hpp"

namespace duckdb {

GpuChunkCacheManager *GpuChunkCacheManager::g_ccm;

GpuChunkCacheManager::GpuChunkCacheManager(const char *path,
                                           GpuCachePolicy policy,
                                           bool standalone)
    : policy_(policy)
{
    // initialize cpu cache manager
    if (policy == GpuCachePolicy::CPU_THEN_GPU) {
        cpu_cache_manager_ = new ChunkCacheManager(path, standalone);
    }
    else {
        cpu_cache_manager_ = nullptr;
    }

    if (!duckdb::EnsurePrimaryCudaContext()) {
        throw InternalException(
            "Failed to ensure primary CUDA context for GpuChunkCacheManager");
    }

    {
        duckdb::CudaCtxGuard guard(duckdb::GetPrimaryCudaContext());

        // initialize gpu arena
        const size_t gpu_arena_size = 1024 * 1024 * 1024;  // 1GB
        gpu_arena = new facebook::velox::wave::GpuArena(
            gpu_arena_size, facebook::velox::wave::getAllocator(
                                facebook::velox::wave::getDevice()));
    }

    InitSwizzleKernel();
}

GpuChunkCacheManager::~GpuChunkCacheManager()
{
    // delete gpu memory
    for (auto &pair : gpu_ptrs_) {
        if (pair.second) {
            gpu_arena->free(pair.second.get());
        }
    }
    gpu_ptrs_.clear();

    // delete gpu arena
    delete gpu_arena;

    // delete cpu cache manager
    if (cpu_cache_manager_) {
        delete cpu_cache_manager_;
    }
}

void GpuChunkCacheManager::InitSwizzleKernel() {
    if (!duckdb::EnsurePrimaryCudaContext()) {
        throw InternalException(
            "Failed to ensure primary CUDA context for swizzle kernel");
    }
    duckdb::CudaCtxGuard guard(duckdb::GetPrimaryCudaContext());

    static const char* kStrSwizzleKernelSrc_ = R"(
        struct __align__(16) str_t {
            union {
                struct { unsigned int length; char prefix[4]; char* ptr; } pointer;
                struct { unsigned int length; char prefix[4]; unsigned long long offset; } offset;
                struct { unsigned int length; char inlined[12]; } inlined;
            } value;
        };

        static __device__ __forceinline__ bool is_inlined_len(unsigned int len) {
            return len <= 12u;
        }

        extern "C" __global__
        void swizzle_offsets_to_ptr_inplace(str_t* __restrict__ rows,
                                            const char* __restrict__ chars_base,
                                            unsigned int n)
        {
            unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) return;

            unsigned int len = rows[i].value.offset.length;
            if (is_inlined_len(len)) return;

            unsigned long long off = rows[i].value.offset.offset;
            rows[i].value.pointer.ptr = (char*)(chars_base + off);
        }
    )";

    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, kStrSwizzleKernelSrc_, "swizzle.cu", 0, nullptr,
                       nullptr);
    std::string arch = makeNvrtcArchFlag_();
    const char *opts[] = { arch.c_str(), "--std=c++17" };

    nvrtcResult cres =
        nvrtcCompileProgram(prog, (int)(sizeof(opts) / sizeof(opts[0])), opts);

    size_t logSize = 0;
    nvrtcGetProgramLogSize(prog, &logSize);
    if (logSize > 1) {
        std::string log(logSize, '\0');
        nvrtcGetProgramLog(prog, &log[0]);
        fprintf(stderr, "[NVRTC LOG] %s\n", log.c_str());
    }
    if (cres != NVRTC_SUCCESS) {
        nvrtcDestroyProgram(&prog);
        fprintf(stderr, "[NVRTC] compile failed.\n");
        return;
    }

    size_t ptxSize = 0;
    nvrtcGetPTXSize(prog, &ptxSize);
    std::string ptx(ptxSize, '\0');
    nvrtcGetPTX(prog, &ptx[0]);
    nvrtcDestroyProgram(&prog);

    CUmodule mod = nullptr;
    cuModuleLoadData(&mod, ptx.c_str());

    CUfunction fn = nullptr;
    cuModuleGetFunction(&fn, mod, "swizzle_offsets_to_ptr_inplace");

    swizzle_module_ = mod;
    swizzle_kernel_ = fn;
    return;
}

ReturnStatus GpuChunkCacheManager::PinSegment(ChunkID cid,
                                              std::string file_path,
                                              uint8_t **gpu_ptr, size_t *size,
                                              bool read_data_async,
                                              bool is_initial_loading)
{
    if (CidValidityCheck(cid)) {
        return ReturnStatus::NOERROR;
    }

    // if already in GPU
    auto it = gpu_ptrs_.find(cid);
    if (it != gpu_ptrs_.end()) {
        *gpu_ptr = static_cast<uint8_t *>(it->second->as<void>());
        *size = GetSegmentSize(cid, file_path);
        return ReturnStatus::OK;
    }

    if (policy_ == GpuCachePolicy::CPU_THEN_GPU) {
        // load through CPU first
        uint8_t *cpu_ptr = nullptr;
        size_t cpu_size = 0;

        // load to CPU first
        ReturnStatus status =
            cpu_cache_manager_->PinSegment(cid, file_path, &cpu_ptr, &cpu_size,
                                           read_data_async, true);
        if (status != ReturnStatus::NOERROR) {
            return status;
        }

        // allocate and copy to GPU memory
        auto gpu_buffer = gpu_arena->allocateBytes(cpu_size);
        if (!gpu_buffer) {
            std::cerr << "GPU allocation failed for CID " << cid << ", size " << cpu_size << std::endl;
            cpu_cache_manager_->UnPinSegment(cid);
            return ReturnStatus::NOERROR;
        }

        // pin the GPU buffer and get the pointer
        gpu_buffer->pin();
        void *gpu_mem = gpu_buffer->as<void>();

        // copy from CPU to GPU
        cudaMemcpy(gpu_mem, cpu_ptr, cpu_size, cudaMemcpyHostToDevice);

        // swizzle data if necessary
        Swizzle(gpu_mem, cpu_ptr);

        // release CPU memory
        cpu_cache_manager_->UnPinSegment(cid);

        // store GPU pointer
        gpu_ptrs_[cid] = std::move(gpu_buffer);
        *gpu_ptr = static_cast<uint8_t *>(gpu_mem);
        *size = cpu_size;
    }
    else {  // GPU_DIRECT
        // load directly from file to GPU
        size_t file_size = GetFileSize(cid, file_path);
        auto gpu_buffer = gpu_arena->allocateBytes(file_size);
        if (!gpu_buffer) {
            return ReturnStatus::NOERROR;
        }
        void *gpu_mem = gpu_buffer->as<void>();

        // get file handler
        Turbo_bin_aio_handler *file_handler =
            cpu_cache_manager_->GetFileHandler(cid);
        if (!file_handler) {
            gpu_arena->free(gpu_buffer.get());
            return ReturnStatus::NOERROR;
        }

        // read directly from file to GPU
        // TODO: implement direct file to GPU reading
        // currently using temporary CPU buffer
        uint8_t *temp_buffer = new uint8_t[file_size];
        file_handler->Read(0, file_size, reinterpret_cast<char *>(temp_buffer),
                           nullptr, nullptr);
        cudaMemcpy(gpu_mem, temp_buffer, file_size, cudaMemcpyHostToDevice);
        delete[] temp_buffer;

        gpu_ptrs_[cid] = gpu_buffer.get();
        *gpu_ptr = static_cast<uint8_t *>(gpu_mem);
        *size = file_size;
    }

    return ReturnStatus::OK;
}

ReturnStatus GpuChunkCacheManager::UnPinSegment(ChunkID cid)
{
    if (CidValidityCheck(cid)) {
        return ReturnStatus::NOERROR;
    }

    auto it = gpu_ptrs_.find(cid);
    if (it != gpu_ptrs_.end()) {
        it->second->unpin();
        gpu_arena->free(it->second.get());
        gpu_ptrs_.erase(it);
    }

    return ReturnStatus::OK;
}

ReturnStatus GpuChunkCacheManager::SetDirty(ChunkID cid)
{
    if (CidValidityCheck(cid)) {
        return ReturnStatus::NOERROR;
    }

    // mark GPU memory as dirty
    // TODO: implement dirty state management
    return ReturnStatus::OK;
}

ReturnStatus GpuChunkCacheManager::CreateSegment(ChunkID cid,
                                                 std::string file_path,
                                                 size_t alloc_size,
                                                 bool can_destroy)
{
    if (CidValidityCheck(cid) || AllocSizeValidityCheck(alloc_size)) {
        return ReturnStatus::NOERROR;
    }

    // create segment in CPU cache manager
    ReturnStatus status = cpu_cache_manager_->CreateSegment(
        cid, file_path, alloc_size, can_destroy);
    if (status != ReturnStatus::OK) {
        return status;
    }

    // allocate GPU memory
    auto gpu_buffer = gpu_arena->allocateBytes(alloc_size);
    if (!gpu_buffer) {
        cpu_cache_manager_->DestroySegment(cid);
        return ReturnStatus::NOERROR;
    }
    void *gpu_mem = gpu_buffer->as<void>();

    gpu_ptrs_[cid] = gpu_buffer.get();
    return ReturnStatus::OK;
}

ReturnStatus GpuChunkCacheManager::DestroySegment(ChunkID cid)
{
    if (CidValidityCheck(cid)) {
        return ReturnStatus::NOERROR;
    }

    // free GPU memory
    auto it = gpu_ptrs_.find(cid);
    if (it != gpu_ptrs_.end()) {
        gpu_arena->free(it->second.get());
        gpu_ptrs_.erase(it);
    }

    // free CPU segment
    return cpu_cache_manager_->DestroySegment(cid);
}

ReturnStatus GpuChunkCacheManager::FinalizeIO(ChunkID cid, bool read,
                                              bool write)
{
    if (CidValidityCheck(cid)) {
        return ReturnStatus::NOERROR;
    }

    // wait for GPU operations to complete
    cudaDeviceSynchronize();

    // complete CPU IO
    return cpu_cache_manager_->FinalizeIO(cid, read, write);
}

ReturnStatus GpuChunkCacheManager::FlushDirtySegmentsAndDeleteFromcache(
    bool destroy_segment)
{
    // copy changes from GPU to CPU
    for (auto &pair : gpu_ptrs_) {
        ChunkID cid = pair.first;
        void *gpu_ptr = pair.second->as<void>();

        // TODO: check and handle dirty state
        // currently copying all segments to CPU
        uint8_t *cpu_ptr = nullptr;
        size_t size = 0;
        cpu_cache_manager_->PinSegment(cid, "", &cpu_ptr, &size);

        cudaMemcpy(cpu_ptr, gpu_ptr, size, cudaMemcpyDeviceToHost);

        cpu_cache_manager_->UnPinSegment(cid);
    }

    // flush dirty segments in CPU cache
    ReturnStatus status =
        cpu_cache_manager_->FlushDirtySegmentsAndDeleteFromcache(
            destroy_segment);

    if (destroy_segment) {
        // free GPU memory
        for (auto &pair : gpu_ptrs_) {
            gpu_arena->free(pair.second.get());
        }
        gpu_ptrs_.clear();
    }

    return status;
}

ReturnStatus GpuChunkCacheManager::GetRemainingMemoryUsage(
    size_t &remaining_memory_usage)
{
    // calculate GPU memory usage
    size_t gpu_used = 0;
    for (const auto &pair : gpu_ptrs_) {
        gpu_used += pair.second->size();
    }

    remaining_memory_usage = gpu_arena->maxCapacity() - gpu_used;
    return ReturnStatus::OK;
}

int GpuChunkCacheManager::GetRefCount(ChunkID cid)
{
    if (CidValidityCheck(cid)) {
        return 0;
    }
    return cpu_cache_manager_->GetRefCount(cid);
}

void GpuChunkCacheManager::SetPolicy(GpuCachePolicy policy)
{
    policy_ = policy;
}

GpuCachePolicy GpuChunkCacheManager::GetPolicy() const
{
    return policy_;
}

bool GpuChunkCacheManager::CidValidityCheck(ChunkID cid)
{
    return cpu_cache_manager_->CidValidityCheck(cid);
}

bool GpuChunkCacheManager::AllocSizeValidityCheck(size_t alloc_size)
{
    return cpu_cache_manager_->AllocSizeValidityCheck(alloc_size);
}

size_t GpuChunkCacheManager::GetSegmentSize(ChunkID cid, std::string file_path)
{
    return cpu_cache_manager_->GetSegmentSize(cid, file_path);
}

size_t GpuChunkCacheManager::GetFileSize(ChunkID cid, std::string file_path)
{
    return cpu_cache_manager_->GetFileSize(cid, file_path);
}

struct StrAbiHost { uint32_t length; char prefix[4]; uint64_t tail8; };
static_assert(sizeof(StrAbiHost)==16, "str_t stride must be 16");
constexpr size_t kStrStride = sizeof(StrAbiHost);

void GpuChunkCacheManager::Swizzle(void *gpu_ptr, void *cpu_ptr)
{
    CompressionHeader comp_header;
    memcpy(&comp_header, cpu_ptr, comp_header.GetSizeWoBitSet());
    if (comp_header.swizzle_type == SwizzlingType::SWIZZLE_NONE) {
        // No swizzling needed
        return;
    }

    const size_t header_bytes = comp_header.GetSizeWoBitSet();
    size_t size = comp_header.data_len;

    CUdeviceptr base = reinterpret_cast<CUdeviceptr>(gpu_ptr);
    CUdeviceptr rows_addr = base + static_cast<CUdeviceptr>(header_bytes);
    CUdeviceptr chars_addr =
        rows_addr + static_cast<CUdeviceptr>(size) * kStrStride;

    void *d_rows = reinterpret_cast<void *>(rows_addr);
    const void *d_chars_base = reinterpret_cast<const void *>(chars_addr);

    D_ASSERT(swizzle_kernel_ != nullptr);
    // Launch the swizzle kernel
    unsigned int n = static_cast<unsigned int>(size);
    const unsigned int block = 256;
    const unsigned int grid = (n + block - 1) / block;

    void *args[] = {&d_rows, (void *)&d_chars_base, &n};

    std::cerr << "Launching swizzle kernel for " << size
              << " rows with grid size: " << grid
              << ", block size: " << block << std::endl;
    CUresult r = cuLaunchKernel(swizzle_kernel_, grid, 1, 1, block, 1, 1,
                                /*sharedMemBytes*/ 0,
                                /*stream*/ 0, args, nullptr);

    if (r != CUDA_SUCCESS) {
        const char *err = nullptr;
        cuGetErrorString(r, &err);
        std::cerr << "swizzle kernel launch failed: " << (err ? err : "unknown")
                  << std::endl;
    }

    cuCtxSynchronize();
    std::cerr << "Swizzle kernel launched successfully for "
              << size << " rows." << std::endl;
}

}  // namespace duckdb
