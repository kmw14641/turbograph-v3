#ifndef GPU_CODE_GENERATOR_H
#define GPU_CODE_GENERATOR_H

#include <cuda.h>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "common/constants.hpp"
#include "common/types.hpp"
#include "common/types/value.hpp"
#include "common/vector.hpp"
#include "execution/cypher_pipeline.hpp"
#include "execution/physical_operator/cypher_physical_operator.hpp"
#include "planner/expression.hpp"
#include "planner/expression/bound_constant_expression.hpp"
#include "planner/expression/bound_function_expression.hpp"
#include "planner/expression/bound_operator_expression.hpp"
#include "planner/expression/bound_reference_expression.hpp"
#include "planner/gpu/codegen_utils.hpp"
#include "planner/gpu/gpu_jit_compiler.hpp"

#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <sstream>

namespace duckdb {

// Forward declaration
class GpuCodeGenerator;

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

// Structure for pointer mapping
struct PointerMapping {
    std::string name;
    void *address;
    ChunkDefinitionID cid;  // Chunk ID for GPU chunk cache manager
};

// Strategy pattern for operator-specific code generation
class OperatorCodeGenerator {
   public:
    virtual ~OperatorCodeGenerator() = default;
    virtual void GenerateCode(CypherPhysicalOperator *op, CodeBuilder &code,
                              GpuCodeGenerator *code_gen,
                              ClientContext &context,
                              PipelineContext &pipeline_ctx,
                              bool is_main_loop = false) = 0;
};

class NodeScanCodeGenerator : public OperatorCodeGenerator {
   public:
    void GenerateCode(CypherPhysicalOperator *op, CodeBuilder &code,
                      GpuCodeGenerator *code_gen, ClientContext &context,
                      PipelineContext &pipeline_ctx,
                      bool is_main_loop = false) override;
};

class ProjectionCodeGenerator : public OperatorCodeGenerator {
   public:
    void GenerateCode(CypherPhysicalOperator *op, CodeBuilder &code,
                      GpuCodeGenerator *code_gen, ClientContext &context,
                      PipelineContext &pipeline_ctx,
                      bool is_main_loop = false) override;

   private:
    void GenerateProjectionExpressionCode(Expression *expr, size_t expr_idx,
                                          CodeBuilder &code,
                                          GpuCodeGenerator *code_gen,
                                          ClientContext &context,
                                          PipelineContext &pipeline_ctx);
    std::string ConvertLogicalTypeToCUDAType(LogicalType type);
    std::string ConvertValueToCUDALiteral(const Value &value);
    std::string ExpressionTypeToString(ExpressionType type);
    void GenerateFunctionCallCode(BoundFunctionExpression *func_expr,
                                  const std::string &output_var,
                                  CodeBuilder &code, GpuCodeGenerator *code_gen,
                                  ClientContext &context,
                                  PipelineContext &pipeline_ctx);
    void GenerateOperatorCode(BoundOperatorExpression *op_expr,
                              const std::string &output_var, CodeBuilder &code,
                              GpuCodeGenerator *code_gen,
                              ClientContext &context,
                              PipelineContext &pipeline_ctx);
};

class ProduceResultsCodeGenerator : public OperatorCodeGenerator {
   public:
    void GenerateCode(CypherPhysicalOperator *op, CodeBuilder &code,
                      GpuCodeGenerator *code_gen, ClientContext &context,
                      PipelineContext &pipeline_ctx,
                      bool is_main_loop = false) override;
};

class FilterCodeGenerator : public OperatorCodeGenerator {
   public:
    void GenerateCode(CypherPhysicalOperator *op, CodeBuilder &code,
                      GpuCodeGenerator *code_gen, ClientContext &context,
                      PipelineContext &pipeline_ctx,
                      bool is_main_loop = false) override;
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
    void GenerateHostCode(CypherPipeline &pipeline);

    // Analyze pipeline dependencies
    void AnalyzeDependencies(const CypherPipeline &pipeline);

    // Analyze memory access patterns
    void AnalyzeMemoryAccess(const CypherPipeline &pipeline);

    // Generate kernel parameters
    void GenerateKernelParams(const CypherPipeline &pipeline);

    // Compile generated CUDA code using nvcc
    bool CompileGeneratedCode();

    // Get compiled host function
    void *GetCompiledHost();

    // Get kernel parameters
    const std::vector<KernelParam> &GetKernelParams() const
    {
        return input_kernel_params;
    }

    std::string ConvertLogicalTypeToPrimitiveType(LogicalTypeId type_id);

    // Set whether this kernel needs to be repeatable
    void SetRepeatable(bool repeatable) { is_repeatable = repeatable; }

    // Set verbose mode for parameter naming
    void SetVerboseMode(bool verbose) { verbose_mode = verbose; }

    // Get verbose mode
    bool GetVerboseMode() const { return verbose_mode; }

    // Add pointer mapping
    void AddPointerMapping(const std::string &name, void *address,
                           ChunkDefinitionID cid);

    // Get pointer mappings
    const std::vector<PointerMapping> &GetPointerMappings() const
    {
        return pointer_mappings;
    }

    // Cleanup resources
    void Cleanup();

    // Generate code for a specific operator
    void GenerateOperatorCode(CypherPhysicalOperator *op, CodeBuilder &code,
                              PipelineContext &pipeline_ctx, bool is_main_loop);

    // Generate main scan loop with nested operators
    void GenerateMainScanLoop(CypherPipeline &pipeline, CodeBuilder &code);

    // Process remaining operators recursively
    void ProcessRemainingOperators(CypherPipeline &pipeline, int op_idx,
                                   CodeBuilder &code);

    // Pipeline context management
    void InitializePipelineContext(const CypherPipeline &pipeline);
    void MoveToOperator(int op_idx);
    void AnalyzeOperatorDependencies(CypherPhysicalOperator *op);

    // Schema analysis
    void ExtractInputSchema(CypherPhysicalOperator *op);
    void ExtractOutputSchema(CypherPhysicalOperator *op);
    void TrackColumnUsage(Expression *expr);

   private:
    ClientContext &context;

    std::string generated_gpu_code;
    std::string generated_cpu_code;
    std::string current_code_hash;

    std::vector<KernelParam> input_kernel_params;
    std::vector<KernelParam> output_kernel_params;
    std::vector<MemoryTransferInfo> memory_transfers;
    std::map<std::string, size_t> device_memory_sizes;

    CUmodule gpu_module = nullptr;
    CUfunction kernel_function = nullptr;

    std::unique_ptr<GpuJitCompiler> jit_compiler;

    bool is_compiled;
    bool is_repeatable;
    bool verbose_mode = true;  // Control parameter naming style

    std::vector<PointerMapping> pointer_mappings;

    // Strategy pattern for operator-specific code generation
    std::unordered_map<PhysicalOperatorType,
                       std::unique_ptr<OperatorCodeGenerator>>
        operator_generators;

    // Single pipeline context for the entire pipeline
    PipelineContext pipeline_context;

    // Initialize operator generators
    void InitializeOperatorGenerators();

    // Pipeline context management
    PipelineContext CreatePipelineContext(const CypherPipeline &pipeline,
                                          int op_idx);
    void UpdatePipelineContext(PipelineContext &ctx,
                               CypherPhysicalOperator *op);
    void AnalyzeOperatorDependencies(CypherPhysicalOperator *op,
                                     PipelineContext &ctx);

    // Schema analysis
    void ExtractInputSchema(CypherPhysicalOperator *op, PipelineContext &ctx);
    void ExtractOutputSchema(CypherPhysicalOperator *op, PipelineContext &ctx);
    void TrackColumnUsage(Expression *expr, PipelineContext &ctx);
};

}  // namespace duckdb

#endif  // GPU_CODE_GENERATOR_H