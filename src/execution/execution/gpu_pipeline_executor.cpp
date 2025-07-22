#include "execution/gpu_pipeline_executor.hpp"
#include <iostream>
#include "execution/execution_context.hpp"
#include "execution/schema_flow_graph.hpp"
#include "main/client_context.hpp"
#include "storage/cache/gpu_chunk_cache_manager.h"

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
    const std::vector<PointerMapping> &pointer_mappings)
    : BasePipelineExecutor(),
      main_function(main_function),
      pointer_mappings(pointer_mappings),
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
    void *main_function, const std::vector<PointerMapping> &pointer_mappings)
    : BasePipelineExecutor(),
      main_function(main_function),
      pointer_mappings(pointer_mappings),
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

    for (const auto &mapping : pointer_mappings) {
        if (mapping.cid >= 0) {
            // This is a chunk that needs GPU memory allocation
            // std::cout << "Allocating GPU memory for chunk " << mapping.cid
            //           << " (name: " << mapping.name << ")" << std::endl;

            uint8_t *gpu_ptr = nullptr;
            size_t size = 0;
            std::string file_path = DiskAioParameters::WORKSPACE +
                                    std::string("/chunk_") +
                                    std::to_string(mapping.cid);

            ReturnStatus status = gpu_cache_manager->PinSegment(
                mapping.cid, file_path, &gpu_ptr, &size, false, false);

            if (status == ReturnStatus::OK) {
                // Store the allocated GPU memory info
                GPUMemory gpu_mem;
                gpu_mem.data_ptr = gpu_ptr;
                gpu_mem.size = size;
                gpu_mem.is_allocated = true;
                gpu_memory_pool.push_back(gpu_mem);

                // Update the pointer mapping with the actual GPU address
                const_cast<PointerMapping &>(mapping).address = gpu_ptr;

                // fprintf(stderr, "GPU memory address = %p\n", gpu_ptr);
            }
            else {
                std::cerr << "Failed to allocate GPU memory for chunk "
                          << mapping.cid << std::endl;
                return false;
            }
        }
    }

    // std::cout << "GPU memory allocation completed for "
    //           << gpu_memory_pool.size() << " chunks" << std::endl;
    return true;
}

bool GPUPipelineExecutor::LaunchKernel()
{
    if (main_function) {
        // Use the new function signature with pointer mappings
        using ExecuteQueryFn = void (*)(PointerMapping *, int);
        auto exec = reinterpret_cast<ExecuteQueryFn>(main_function);

        // Call the main function with pointer mappings
        exec(const_cast<PointerMapping *>(pointer_mappings.data()),
             pointer_mappings.size());
        std::cout << "Launching GPU kernel with " << pointer_mappings.size()
                  << " pointer mappings..." << std::endl;
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