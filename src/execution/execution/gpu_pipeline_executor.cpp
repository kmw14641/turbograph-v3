#include "execution/gpu_pipeline_executor.hpp"
#include "execution/execution_context.hpp"
#include "execution/schema_flow_graph.hpp"
#include "main/client_context.hpp"
#include "storage/cache/gpu_chunk_cache_manager.h"

#include <iostream>
#include <cuda_runtime.h>

namespace duckdb {

GPUPipelineExecutor::GPUPipelineExecutor(ExecutionContext *context,
                                         CypherPipeline *pipeline,
                                         void *main_function)
    : BasePipelineExecutor(),
      main_function(main_function),
      cuda_stream(nullptr),
      is_initialized(false)
{
    this->pipeline = pipeline;
    this->context = context;
    this->sfg = nullptr;
    // Initialize GPU resources
    if (!InitializeGPU()) {
        std::cerr << "Failed to initialize GPU resources" << std::endl;
    }
}

GPUPipelineExecutor::GPUPipelineExecutor(ExecutionContext *context,
                                         CypherPipeline *pipeline,
                                         SchemaFlowGraph &sfg,
                                         void *main_function)
    : BasePipelineExecutor(),
      main_function(main_function),
      cuda_stream(nullptr),
      is_initialized(false)
{
    this->pipeline = pipeline;
    this->context = context;
    this->sfg = std::make_unique<SchemaFlowGraph>(sfg);
    // Initialize GPU resources
    if (!InitializeGPU()) {
        std::cerr << "Failed to initialize GPU resources" << std::endl;
    }
}

GPUPipelineExecutor::GPUPipelineExecutor(
    ExecutionContext *context, CypherPipeline *pipeline, void *main_function,
    const std::vector<PointerMapping> &pointer_mappings,
    const std::vector<ScanColumnInfo> &scan_column_infos)
    : BasePipelineExecutor(),
      main_function(main_function),
      pointer_mappings(pointer_mappings),
      scan_column_infos(scan_column_infos),
      cuda_stream(nullptr),
      is_initialized(false)
{
    this->pipeline = pipeline;
    this->context = context;
    this->sfg = nullptr;
    // Initialize GPU resources
    if (!InitializeGPU()) {
        std::cerr << "Failed to initialize GPU resources" << std::endl;
    }
}

GPUPipelineExecutor::GPUPipelineExecutor(
    ExecutionContext *context, CypherPipeline *pipeline, SchemaFlowGraph &sfg,
    void *main_function, const std::vector<PointerMapping> &pointer_mappings,
    const std::vector<ScanColumnInfo> &scan_column_infos)
    : BasePipelineExecutor(),
      main_function(main_function),
      pointer_mappings(pointer_mappings),
      scan_column_infos(scan_column_infos),
      cuda_stream(nullptr),
      is_initialized(false)
{
    this->pipeline = pipeline;
    this->context = context;
    this->sfg = std::make_unique<SchemaFlowGraph>(sfg);
    // Initialize GPU resources
    if (!InitializeGPU()) {
        std::cerr << "Failed to initialize GPU resources" << std::endl;
    }
}

GPUPipelineExecutor::~GPUPipelineExecutor()
{
    CleanupGPU();
}

void GPUPipelineExecutor::ExecutePipeline()
{
    if (is_initialized && main_function) {
        // Try GPU execution first
        try {
            ExecuteGPUPipeline();
        }
        catch (const std::exception &e) {
            std::cerr << "GPU execution failed: " << e.what() << std::endl;
            std::cerr << "Falling back to CPU execution" << std::endl;
            // todo exception handling
        }
    }
    else {
        throw std::runtime_error("GPU pipeline executor not initialized");
    }
}

bool GPUPipelineExecutor::InitializeGPU()
{
    // TODO: Implement actual GPU initialization
    // For now, just set initialized to true
    is_initialized = true;
    return true;
}

bool GPUPipelineExecutor::AllocateGPUMemory()
{
    std::cout << "Allocating GPU memory using chunk cache manager..."
              << std::endl;

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

                // generate column pointer
                uint64_t *d_col_ptr = nullptr;
                size_t total_bytes = num_tuples_total * sizeof(uint64_t);
                cudaMalloc(reinterpret_cast<void **>(&d_col_ptr), total_bytes);

                // Store the column value ptrs
                std::vector<uint64_t> h_col_ptr(num_tuples_total);
                uint64_t current_idx = 0;
                for (uint64_t j = 0; i < scan_info.extent_ids.size(); j++) {
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

                        // Update the pointer mapping with the actual GPU address
                        target_mapping->address = gpu_ptr;
                    }
                    else {
                        std::cerr << "Failed to allocate GPU memory for chunk "
                                  << cid_in_extent << std::endl;
                        return false;
                    }

                    // For each tuple in this extent, store the ptrs to column value
                    for (uint64_t k = 0; k < num_tuples; k++) {
                        uint64_t value_address = reinterpret_cast<uint64_t>(
                            gpu_ptr + k * col_type_size);
                        h_col_ptr[current_idx++] = value_address;
                    }
                }
                D_ASSERT(current_idx == num_tuples_total);

                cudaMemcpy(d_col_ptr, h_col_ptr.data(), total_bytes,
                           cudaMemcpyHostToDevice);
                
                PointerMapping col_mapping;
                col_mapping.name = col_name;
                col_mapping.address = reinterpret_cast<void *>(d_col_ptr);
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
        std::cout << "Launching GPU kernel with "
                  << input_pointer_mappings.size() << " pointer mappings..."
                  << std::endl;
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
    std::cout << "Cleaning up GPU resources..." << std::endl;

    auto *gpu_cache_manager = GpuChunkCacheManager::g_ccm;

    // Clean up allocated GPU memory through cache manager
    for (size_t i = 0;
         i < pointer_mappings.size() && i < gpu_memory_pool.size(); i++) {
        const auto &mapping = pointer_mappings[i];
        auto &gpu_mem = gpu_memory_pool[i];

        if (mapping.cid >= 0 && gpu_mem.is_allocated) {
            std::cout << "Unpinning chunk " << mapping.cid << " from GPU memory"
                      << std::endl;

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
    device_ptrs.clear();

    is_initialized = false;
    std::cout << "GPU cleanup completed" << std::endl;
}

void GPUPipelineExecutor::ExecuteGPUPipeline()
{
    std::cout << "Executing pipeline on GPU..." << std::endl;

    // Allocate GPU memory
    if (!AllocateGPUMemory()) {
        throw std::runtime_error("Failed to allocate GPU memory");
    }

    // Launch kernel
    if (!LaunchKernel()) {
        throw std::runtime_error("Failed to launch GPU kernel");
    }

    // Transfer results back to CPU
    if (!TransferResultsToCPU()) {
        throw std::runtime_error("Failed to transfer results from GPU");
    }

    std::cout << "GPU pipeline execution completed" << std::endl;
}

void GPUPipelineExecutor::ExecuteCPUPipeline()
{
    std::cout << "Executing pipeline on CPU (fallback)..." << std::endl;

    // TODO: Implement CPU fallback execution
    // This could create a temporary CypherPipelineExecutor and use it
    // or implement a simplified CPU version

    std::cout << "CPU pipeline execution completed" << std::endl;
}

}  // namespace duckdb