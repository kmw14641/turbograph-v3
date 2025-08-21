#include "common/scoped_timer.hpp"
#include "execution/gpu_pipeline_executor.hpp"
#include "execution/execution_context.hpp"
#include "execution/schema_flow_graph.hpp"
#include "main/client_context.hpp"
#include "storage/cache/gpu_chunk_cache_manager.h"
#include "storage/extent/compression/compression_header.hpp"

#include <iostream>
#include <cuda_runtime.h>

namespace duckdb {

GPUPipelineExecutor::GPUPipelineExecutor(
    ExecutionContext *context, std::vector<CypherPipeline *> &pipelines,
    void *main_function, std::vector<CUfunction> &gpu_kernels,
    const std::vector<PointerMapping> &pointer_mappings,
    const std::vector<ScanColumnInfo> &scan_column_infos)
    : BasePipelineExecutor(),
      pipelines(pipelines),
      main_function(main_function),
      gpu_kernels(std::move(gpu_kernels)),
      pointer_mappings(pointer_mappings),
      scan_column_infos(scan_column_infos)
{
    this->context = context;
    this->sfg = nullptr;
}

GPUPipelineExecutor::~GPUPipelineExecutor()
{
    CleanupGPU();
}

void GPUPipelineExecutor::ExecutePipeline()
{
    if (main_function) {
        ExecuteGPUPipeline();
    }
    else {
        throw std::runtime_error("GPU pipeline executor not initialized");
    }
}

bool GPUPipelineExecutor::AllocateGPUMemory()
{
    spdlog::info("[AllocateGPUMemory] Allocating GPU memory for pipeline execution");
    auto *gpu_cache_manager = GpuChunkCacheManager::g_ccm;

    if (use_scan_column_infos) {
        for (const auto &scan_info : scan_column_infos) {
            uint64_t num_tuples_total = 0;
            for (const auto &num_tuples : scan_info.num_tuples_per_extent) {
                num_tuples_total += num_tuples;
            }
            if (scan_info.get_physical_id_column) {
                // generate physical id column
                uint64_t *d_physical_id = nullptr;
                size_t total_bytes = num_tuples_total * sizeof(uint64_t);
                cudaMalloc(reinterpret_cast<void **>(&d_physical_id),
                           total_bytes);
                // Store the physical ID values
                // TODO: currently, we generate data in host and copy to device.
                // This can be optimized by generating data directly on device.
                std::vector<uint64_t> h_physical_id(num_tuples_total);
                uint64_t current_idx = 0;
                for (uint64_t i = 0; i < scan_info.extent_ids.size(); i++) {
                    uint64_t eid_base =
                        (uint64_t(scan_info.extent_ids[i]) << 32);
                    for (uint64_t j = 0; j < scan_info.num_tuples_per_extent[i];
                         ++j)
                        h_physical_id[current_idx++] = eid_base + j;
                }
                cudaMemcpy(d_physical_id, h_physical_id.data(), total_bytes,
                           cudaMemcpyHostToDevice);
                
                PointerMapping physical_id_mapping;
                physical_id_mapping.name = "_id";
                physical_id_mapping.address = reinterpret_cast<void *>(d_physical_id);
                physical_id_mapping.cid = -1;
                input_pointer_mappings.push_back(physical_id_mapping);
            }
            for (auto i = 0; i < scan_info.col_name.size(); i++) {
                const auto &col_name = scan_info.col_name[i];
                auto col_pos = scan_info.col_position[i];
                auto col_type_size = scan_info.col_type_size[i];

                uint8_t *d_col_data = nullptr;
                size_t total_bytes = num_tuples_total * col_type_size;
                cudaMalloc(reinterpret_cast<void **>(&d_col_data), total_bytes);
                
                // Store the column values (TODO: this is temporary implementation)
                // uint64_t current_idx = 0;
                uint64_t current_byte_offset = 0;
                for (uint64_t j = 0; j < scan_info.extent_ids.size(); j++) {
                    auto extent_id = scan_info.extent_ids[j];
                    auto num_tuples = scan_info.num_tuples_per_extent[j];
                    auto cid_in_extent = scan_info.chunk_ids[i][j];

                    PointerMapping *target_mapping;
                    // TODO: below code is super inefficient
                    for (auto &mapping : pointer_mappings) {
                        if (mapping.cid == cid_in_extent) {
                            target_mapping = &mapping;
                            break;
                        }
                    }

                    uint8_t *gpu_ptr = nullptr;
                    size_t size = 0;
                    std::string file_path = DiskAioParameters::WORKSPACE +
                                            std::string("/chunk_") +
                                            std::to_string(cid_in_extent);

                    ReturnStatus status = gpu_cache_manager->PinSegment(
                        cid_in_extent, file_path, &gpu_ptr, &size, false, false);

                    if (status == ReturnStatus::OK) {
                        // Store the allocated GPU memory info
                        GPUMemory gpu_mem;
                        gpu_mem.data_ptr = gpu_ptr;
                        gpu_mem.size = size;
                        gpu_mem.is_allocated = true;
                        gpu_memory_pool.push_back(gpu_mem);

                        // advance gpu_ptr by size of compression header
                        gpu_ptr += CompressionHeader::GetSizeWoBitSet();

                        // Update the pointer mapping with the actual GPU address
                        target_mapping->address = gpu_ptr;

                        // Direct GPU-to-GPU memory copy (no host transfer!)
                        size_t extent_data_size = num_tuples * col_type_size;
                        cudaMemcpy(d_col_data + current_byte_offset, gpu_ptr,
                                   extent_data_size, cudaMemcpyDeviceToDevice);

                        current_byte_offset += extent_data_size;
                    }
                    else {
                        std::cerr << "Failed to allocate GPU memory for chunk "
                                  << cid_in_extent << std::endl;
                        return false;
                    }
                }
                D_ASSERT(current_byte_offset == total_bytes);
                
                PointerMapping col_mapping;
                col_mapping.name = col_name;
                col_mapping.address = reinterpret_cast<void *>(d_col_data);
                col_mapping.cid = -1;  // No chunk ID for virtual columns
                input_pointer_mappings.push_back(col_mapping);
            }
        }
    }
    return true;
}

bool GPUPipelineExecutor::LaunchKernel()
{
    if (main_function) {
        // Use the new function signature with pointer mappings
        using ExecuteQueryFn = void (*)(PointerMapping *, int);
        auto exec = reinterpret_cast<ExecuteQueryFn>(main_function);

        // Call the main function with pointer mappings
        exec(const_cast<PointerMapping *>(input_pointer_mappings.data()),
             input_pointer_mappings.size());
        return true;
    }
    return false;
}

bool GPUPipelineExecutor::TransferResultsToCPU()
{
    // TODO: Implement results transfer from GPU
    return true;
}

void GPUPipelineExecutor::CleanupGPU()
{
    spdlog::info("[CleanupGPU] Cleaning up GPU resources");
    auto *gpu_cache_manager = GpuChunkCacheManager::g_ccm;

    // Clean up allocated GPU memory through cache manager
    for (size_t i = 0;
         i < pointer_mappings.size() && i < gpu_memory_pool.size(); i++) {
        const auto &mapping = pointer_mappings[i];
        auto &gpu_mem = gpu_memory_pool[i];

        if (mapping.cid >= 0 && gpu_mem.is_allocated) {
            ReturnStatus status = gpu_cache_manager->UnPinSegment(mapping.cid);
            if (status != ReturnStatus::OK) {
                std::cerr << "Failed to unpin chunk " << mapping.cid
                          << " from GPU memory" << std::endl;
            }

            gpu_mem.is_allocated = false;
            gpu_mem.data_ptr = nullptr;
            gpu_mem.size = 0;
        }
    }

    // Clear memory pools
    gpu_memory_pool.clear();
}

void GPUPipelineExecutor::ExecuteGPUPipeline()
{
    spdlog::info("[ExecuteGPUPipeline] Executing GPU pipeline Started");
    SCOPED_TIMER_SIMPLE(ExecuteGPUPipeline, spdlog::level::info,
                        spdlog::level::info);
    
    // Allocate GPU memory
    SUBTIMER_START(ExecuteGPUPipeline, "AllocateGPUMemory");
    if (!AllocateGPUMemory()) {
        throw std::runtime_error("Failed to allocate GPU memory");
    }
    SUBTIMER_STOP(ExecuteGPUPipeline, "AllocateGPUMemory");

    // Launch kernel
    SUBTIMER_START(ExecuteGPUPipeline, "LaunchKernel");
    if (!LaunchKernel()) {
        throw std::runtime_error("Failed to launch GPU kernel");
    }
    SUBTIMER_STOP(ExecuteGPUPipeline, "LaunchKernel");

    // Transfer results back to CPU
    SUBTIMER_START(ExecuteGPUPipeline, "TransferResultsToCPU");
    if (!TransferResultsToCPU()) {
        throw std::runtime_error("Failed to transfer results from GPU");
    }
    SUBTIMER_STOP(ExecuteGPUPipeline, "TransferResultsToCPU");
    spdlog::info("[ExecuteGPUPipeline] GPU pipeline Execution Finished");
}

}  // namespace duckdb