#ifndef GPU_CODE_GENERATOR_H
#define GPU_CODE_GENERATOR_H

#include <cuda.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "common/types.hpp"
#include "common/vector.hpp"
#include "execution/cypher_pipeline.hpp"
#include "execution/physical_operator/cypher_physical_operator.hpp"
#include "planner/gpu/gpu_jit_compiler.hpp"

#include <llvm/ExecutionEngine/Orc/LLJIT.h>

namespace duckdb {

// Structure for GPU kernel parameters
struct KernelParam {
    std::string name;
    std::string type;
    std::string value;
    bool is_device_ptr;
};

// Structure for GPU memory transfer information
struct MemoryTransferInfo {
    std::string src_name;
    std::string dst_name;
    size_t size;
    bool is_host_to_device;
};

class GpuCodeGenerator {
   public:
    GpuCodeGenerator(ClientContext &context);
    ~GpuCodeGenerator();

    void InitializeLLVMTargets();

    // Generate GPU code
    void GenerateGPUCode(CypherPipeline &pipeline);

    // Generate GPU kernel code
    void GenerateKernelCode(CypherPipeline &pipeline);

    // Generate GPU host code
    void GenerateHostCode();

    // Analyze pipeline dependencies
    void AnalyzeDependencies(const CypherPipeline &pipeline);

    // Analyze memory access patterns
    void AnalyzeMemoryAccess(const CypherPipeline &pipeline);

    // Generate kernel parameters
    std::vector<KernelParam> GenerateKernelParams(
        const CypherPipeline &pipeline);

    // Compile generated CUDA code using nvcc
    bool CompileGeneratedCode();

    // Get compiled host function
    void *GetCompiledHost();

    // Get kernel parameters
    const std::vector<KernelParam> &GetKernelParams() const
    {
        return kernel_params;
    }

    // Set whether this kernel needs to be repeatable
    void SetRepeatable(bool repeatable) { is_repeatable = repeatable; }

    // Cleanup resources
    void Cleanup();

   private:
    ClientContext &context;

    std::string generated_gpu_code;
    std::string generated_cpu_code;
    std::string current_code_hash;

    std::vector<KernelParam> kernel_params;
    std::vector<MemoryTransferInfo> memory_transfers;
    std::map<std::string, size_t> device_memory_sizes;

    CUmodule   gpu_module   = nullptr;
    CUfunction kernel_function = nullptr;

    std::unique_ptr<GpuJitCompiler> jit_compiler;
    
    bool is_compiled;
    bool is_repeatable;
};

}  // namespace duckdb

#endif  // GPU_CODE_GENERATOR_H