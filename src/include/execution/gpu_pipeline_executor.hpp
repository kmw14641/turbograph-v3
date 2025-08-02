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
    GPUPipelineExecutor(ExecutionContext *context, CypherPipeline *pipeline,
                        void *kernel_function);
    GPUPipelineExecutor(ExecutionContext *context, CypherPipeline *pipeline,
                        SchemaFlowGraph &sfg, void *kernel_function);
    GPUPipelineExecutor(ExecutionContext *context, CypherPipeline *pipeline,
                        void *kernel_function,
                        const std::vector<PointerMapping> &pointer_mappings,
                        const std::vector<ScanColumnInfo> &scan_column_infos);
    GPUPipelineExecutor(ExecutionContext *context, CypherPipeline *pipeline,
                        SchemaFlowGraph &sfg, void *kernel_function,
                        const std::vector<PointerMapping> &pointer_mappings,
                        const std::vector<ScanColumnInfo> &scan_column_infos);
    ~GPUPipelineExecutor();

    //! Fully execute a pipeline with GPU acceleration
    void ExecutePipeline() override;

    //! Get pipeline pointer
    CypherPipeline *GetPipeline() const override { return pipeline; }

    //! Get context pointer
    ExecutionContext *GetContext() const override { return context; }

    //! Get schema flow graph
    SchemaFlowGraph *GetSchemaFlowGraph() override
    {
        return sfg ? &(*sfg) : nullptr;
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
    void *main_function;

    // Pointer mappings for GPU execution
    std::vector<PointerMapping> pointer_mappings;

    // Scan column information for GPU execution
    std::vector<ScanColumnInfo> scan_column_infos;

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