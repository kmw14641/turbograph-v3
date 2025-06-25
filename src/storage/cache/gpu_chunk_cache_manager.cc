#include "storage/cache/gpu_chunk_cache_manager.h"
#include <cuda_runtime.h>

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

    // initialize gpu arena
    const size_t gpu_arena_size = 1024 * 1024 * 1024;  // 1GB
    // gpu_arena = new facebook::velox::wave::GpuArena(gpu_arena_size, nullptr);
}

GpuChunkCacheManager::~GpuChunkCacheManager()
{
    // delete gpu memory
    for (auto &pair : gpu_ptrs_) {
        if (pair.second) {
            // gpu_arena->free(
            //     static_cast<facebook::velox::wave::Buffer *>(pair.second));
        }
    }
    gpu_ptrs_.clear();

    // delete gpu arena
    // delete gpu_arena;

    // delete cpu cache manager
    if (cpu_cache_manager_) {
        delete cpu_cache_manager_;
    }
}

ReturnStatus GpuChunkCacheManager::PinSegment(ChunkID cid,
                                              std::string file_path,
                                              uint8_t **gpu_ptr, size_t *size,
                                              bool read_data_async,
                                              bool is_initial_loading)
{
    if (!CidValidityCheck(cid)) {
        return ReturnStatus::NOERROR;
    }

    // if already in GPU
    auto it = gpu_ptrs_.find(cid);
    if (it != gpu_ptrs_.end()) {
        *gpu_ptr = static_cast<uint8_t *>(it->second);
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
                                           read_data_async, is_initial_loading);
        if (status != ReturnStatus::OK) {
            return status;
        }

        // allocate and copy to GPU memory
        void *gpu_mem; //= gpu_arena->allocateBytes(cpu_size);
        if (!gpu_mem) {
            cpu_cache_manager_->UnPinSegment(cid);
            return ReturnStatus::NOERROR;
        }

        // copy from CPU to GPU
        cudaMemcpy(gpu_mem, cpu_ptr, cpu_size, cudaMemcpyHostToDevice);

        // release CPU memory
        cpu_cache_manager_->UnPinSegment(cid);

        // store GPU pointer
        gpu_ptrs_[cid] = gpu_mem;
        *gpu_ptr = static_cast<uint8_t *>(gpu_mem);
        *size = cpu_size;
    }
    else {  // GPU_DIRECT
        // load directly from file to GPU
        size_t file_size = GetFileSize(cid, file_path);
        void *gpu_mem; //= gpu_arena->allocateBytes(file_size);
        if (!gpu_mem) {
            return ReturnStatus::NOERROR;
        }

        // get file handler
        Turbo_bin_aio_handler *file_handler =
            cpu_cache_manager_->GetFileHandler(cid);
        if (!file_handler) {
            // gpu_arena->free(
            //     static_cast<facebook::velox::wave::Buffer *>(gpu_mem));
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

        gpu_ptrs_[cid] = gpu_mem;
        *gpu_ptr = static_cast<uint8_t *>(gpu_mem);
        *size = file_size;
    }

    return ReturnStatus::OK;
}

ReturnStatus GpuChunkCacheManager::UnPinSegment(ChunkID cid)
{
    if (!CidValidityCheck(cid)) {
        return ReturnStatus::NOERROR;
    }

    auto it = gpu_ptrs_.find(cid);
    if (it != gpu_ptrs_.end()) {
        // gpu_arena->free(
        //     static_cast<facebook::velox::wave::Buffer *>(it->second));
        gpu_ptrs_.erase(it);
    }

    return ReturnStatus::OK;
}

ReturnStatus GpuChunkCacheManager::SetDirty(ChunkID cid)
{
    if (!CidValidityCheck(cid)) {
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
    if (!CidValidityCheck(cid) || !AllocSizeValidityCheck(alloc_size)) {
        return ReturnStatus::NOERROR;
    }

    // create segment in CPU cache manager
    ReturnStatus status = cpu_cache_manager_->CreateSegment(
        cid, file_path, alloc_size, can_destroy);
    if (status != ReturnStatus::OK) {
        return status;
    }

    // allocate GPU memory
    void *gpu_mem; //= gpu_arena->allocateBytes(alloc_size);
    if (!gpu_mem) {
        cpu_cache_manager_->DestroySegment(cid);
        return ReturnStatus::NOERROR;
    }

    gpu_ptrs_[cid] = gpu_mem;
    return ReturnStatus::OK;
}

ReturnStatus GpuChunkCacheManager::DestroySegment(ChunkID cid)
{
    if (!CidValidityCheck(cid)) {
        return ReturnStatus::NOERROR;
    }

    // free GPU memory
    auto it = gpu_ptrs_.find(cid);
    if (it != gpu_ptrs_.end()) {
        // gpu_arena->free(
        //     static_cast<facebook::velox::wave::Buffer *>(it->second));
        gpu_ptrs_.erase(it);
    }

    // free CPU segment
    return cpu_cache_manager_->DestroySegment(cid);
}

ReturnStatus GpuChunkCacheManager::FinalizeIO(ChunkID cid, bool read,
                                              bool write)
{
    if (!CidValidityCheck(cid)) {
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
        void *gpu_ptr = pair.second;

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
            // gpu_arena->free(
            //     static_cast<facebook::velox::wave::Buffer *>(pair.second));
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
        gpu_used += GetSegmentSize(pair.first, "");
    }

    // remaining_memory_usage = gpu_arena->maxCapacity() - gpu_used;
    return ReturnStatus::OK;
}

int GpuChunkCacheManager::GetRefCount(ChunkID cid)
{
    if (!CidValidityCheck(cid)) {
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

}  // namespace duckdb
