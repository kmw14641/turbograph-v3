#ifndef GPU_CHUNK_CACHE_MANAGER_H
#define GPU_CHUNK_CACHE_MANAGER_H

#include <string>
#include <cuda_runtime.h>
#include <cuda.h>
#include <libaio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include "common/constants.hpp"
#include "common/unordered_map.hpp"
#include "storage/cache/chunk_cache_manager.h"
#include "storage/cache/client.h"
#include "storage/cache/common.h"
#include "storage/cache/disk_aio/Turbo_bin_aio_handler.hpp"

#include "velox/experimental/wave/common/GpuArena.h"

namespace duckdb {

enum class GpuCachePolicy { CPU_THEN_GPU, GPU_DIRECT };

class GpuChunkCacheManager {
   public:  
    static GpuChunkCacheManager *g_ccm;

   public:
    GpuChunkCacheManager(const char *path,
                         GpuCachePolicy policy = GpuCachePolicy::CPU_THEN_GPU,
                         bool standalone = false);
    ~GpuChunkCacheManager();

    // GPU ChunkCacheManager APIs
    ReturnStatus PinSegment(ChunkID cid, std::string file_path,
                            uint8_t **gpu_ptr, size_t *size,
                            bool read_data_async = false,
                            bool is_initial_loading = false);
    ReturnStatus UnPinSegment(ChunkID cid);
    ReturnStatus SetDirty(ChunkID cid);
    ReturnStatus CreateSegment(ChunkID cid, std::string file_path,
                               size_t alloc_size, bool can_destroy);
    ReturnStatus DestroySegment(ChunkID cid);
    ReturnStatus FinalizeIO(ChunkID cid, bool read = true, bool write = true);
    ReturnStatus FlushDirtySegmentsAndDeleteFromcache(
        bool destroy_segment = false);
    ReturnStatus GetRemainingMemoryUsage(size_t &remaining_memory_usage);

    // Debugging
    int GetRefCount(ChunkID cid);

    void SetPolicy(GpuCachePolicy policy);
    GpuCachePolicy GetPolicy() const;

   private:
    bool CidValidityCheck(ChunkID cid);
    bool AllocSizeValidityCheck(size_t alloc_size);
    size_t GetSegmentSize(ChunkID cid, std::string file_path);
    size_t GetFileSize(ChunkID cid, std::string file_path);

    facebook::velox::wave::GpuArena *gpu_arena;

    GpuCachePolicy policy_;

    ChunkCacheManager *cpu_cache_manager_;

    unordered_map<ChunkID, facebook::velox::wave::WaveBufferPtr> gpu_ptrs_;
};

}  // namespace duckdb

#endif  // GPU_CHUNK_CACHE_MANAGER_H
