#ifndef GPU_PIPELINE_EXECUTOR_HPP
#define GPU_PIPELINE_EXECUTOR_HPP

#include <memory>
#include "execution/base_pipeline_executor.hpp"
#include "execution/cypher_pipeline.hpp"
#include "planner/gpu/gpu_code_generator.hpp"

namespace duckdb {

class ExecutionContext;
class SchemaFlowGraph;

//! GPU Pipeline Executor for GPU-accelerated pipeline execution
class GPUPipelineExecutor : public BasePipelineExecutor {
   public:
    GPUPipelineExecutor(ExecutionContext *context,
                        std::vector<CypherPipeline *> &pipelines,
                        void *main_function,
                        std::vector<CUfunction> &gpu_kernels,
                        const std::vector<PointerMapping> &pointer_mappings,
                        const std::vector<ScanColumnInfo> &scan_column_infos);
    ~GPUPipelineExecutor();

    //! Fully execute a pipeline with GPU acceleration
    void ExecutePipeline() override;

    //! Get pipeline pointer
    CypherPipeline *GetPipeline() const override { return pipelines.back(); }

    //! Get context pointer
    ExecutionContext *GetContext() const override { return context; }

    //! Get schema flow graph
    SchemaFlowGraph *GetSchemaFlowGraph() override
    {
        return sfg ? &(*sfg) : nullptr;
    }

    std::string GetPipelineToString() const override {
        return "";
    }

   private:
    //! Initialize GPU resources
    bool InitializeGPU();

    //! Allocate GPU memory
    bool AllocateGPUMemory();

    //! Launch kernel
    bool LaunchKernel();

    //! Transfer results back to CPU
    bool TransferResultsToCPU();

    //! Clean up GPU resources
    void CleanupGPU();

    //! Execute pipeline using GPU kernel
    void ExecuteGPUPipeline();

    //! Execute pipeline using CPU fallback
    void ExecuteCPUPipeline();

   private:
    std::unique_ptr<SchemaFlowGraph> sfg;
    std::vector<CypherPipeline *> &pipelines;
    void *main_function;
    std::vector<CUfunction> gpu_kernels;

    // Pointer mappings for GPU execution
    std::vector<PointerMapping> pointer_mappings;
    std::vector<PointerMapping> input_pointer_mappings;

    // Scan column information for GPU execution
    std::vector<ScanColumnInfo> scan_column_infos;

    bool use_scan_column_infos = true;

    // GPU resources
    void *cuda_stream;  // CUstream 대신 void* 사용
    std::vector<void *> device_ptrs;
    bool is_initialized;

    // GPU memory management
    struct GPUMemory {
        void *data_ptr;
        size_t size;
        bool is_allocated;
    };
    std::vector<GPUMemory> gpu_memory_pool;
};

}  // namespace duckdb

#endif  // GPU_PIPELINE_EXECUTOR_HPP