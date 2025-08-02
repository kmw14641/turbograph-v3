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

// GPU Kernel Configuration Constants
struct KernelConstants {
    // Block and Grid Configuration
    static constexpr int DEFAULT_BLOCK_SIZE = 128;
    static constexpr int DEFAULT_GRID_SIZE = 3280;
};

// GPU Kernel Configuration Arguments (equivalent to KernelCall.args)
struct GpuKernelArgs {
    // Database and experiment settings
    std::string sqlite3_path = "";
    std::string exp_id = "test";
    std::string mode = "timecheck";  // timecheck / profile / debug / sample / stats / fake
    int timeout = 10800;
    
    // Path settings
    std::string dbpath = "";
    std::string qpath = "";
    std::string system = "";
    std::string resultpath = "";
    
    // GPU configuration
    int gridsize = 3280;
    int blocksize = 128;
    int device = 0;
    int min_num_warps = 3936;
    int sm_num = 82;
    bool local_aggregation = false;
    bool lazy_materialization = false;
    
    // Inter-warp load balancing
    bool inter_warp_lb = false;
    std::string inter_warp_lb_method = "aws";  // aws / ws
    int inter_warp_lb_interval = 32;
    std::string inter_warp_lb_detection_method = "twolvlbitmaps";  // twolvlbitmaps / idqueue
    int inter_warp_lb_ws_threshold = 1024;
    
    // Additional settings
    int pyper_grid_threshold = 24;
    int max_interval = 16;  // TODO: get actual value
};

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

enum class PipeInputType : uint8_t {
    TYPE_0, // scan
    TYPE_1, // multi
    TYPE_2 // filter
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

    GpuKernelArgs &GetKernelArgs()
    {
        return kernel_args;
    }

    std::string ConvertLogicalTypeToPrimitiveType(LogicalType &type);
    std::string ConvertLogicalTypeIdToPrimitiveType(LogicalTypeId type_id,
                                                    uint16_t extra_info);

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

    // Generate pipeline code
    void GeneratePipelineCode(CypherPipeline &pipeline, CodeBuilder &code);

    // Generate sub-pipeline code
    void GenerateSubPipelineCode(CypherPipeline &pipeline, CodeBuilder &code);

    // Process remaining operators recursively
    void ProcessRemainingOperators(CypherPipeline &pipeline, int op_idx,
                                   CodeBuilder &code);

    // Generate input code for a specific type
    void GenerateInputCode(CypherPhysicalOperator *op, CodeBuilder &code,
                           PipelineContext &pipeline_ctx,
                           PipeInputType input_type);
    void GenerateInputCodeForType0(CypherPhysicalOperator *op,
                                   CodeBuilder &code,
                                   PipelineContext &pipeline_ctx);
    void GenerateInputCodeForType1(CypherPhysicalOperator *op,
                                   CodeBuilder &code,
                                   PipelineContext &pipeline_ctx);
    void GenerateInputCodeForType2(CypherPhysicalOperator *op,
                                   CodeBuilder &code,
                                   PipelineContext &pipeline_ctx);

    // Generate code for adaptive work sharing
    void GenerateCodeForAdaptiveWorkSharing(
        CypherPipeline &pipeline, CodeBuilder &code);
    void GenerateCodeForAdaptiveWorkSharingPull(
        CypherPipeline &pipeline, CodeBuilder &code);
    void GenerateCopyCodeForAdaptiveWorkSharingPull(
        CypherPipeline &pipeline, CodeBuilder &code);
    void GenerateCodeForAdaptiveWorkSharingPush(
        CypherPipeline &pipeline, CodeBuilder &code);
    void GenerateCopyCodeForAdaptiveWorkSharingPush(
        CypherPipeline &pipeline, CodeBuilder &code);

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
    GpuKernelArgs kernel_args;  // GPU kernel configuration

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

    bool do_inter_warp_lb = true;
    bool is_compiled;
    bool is_repeatable;
    bool verbose_mode = true;  // Control parameter naming style
    bool doWorkoadSizeTracking = false;
    int tsWidth = 32;
    std::string idleWarpDetectionType = "twolvlbitmaps";  // twolvlbitmaps / idqueue

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

    // Split pipeline into sub-pipelines
    void SplitPipelineIntoSubPipelines(CypherPipeline &pipeline);

    // Schema analysis
    void ExtractInputSchema(CypherPhysicalOperator *op, PipelineContext &ctx);
    void ExtractOutputSchema(CypherPhysicalOperator *op, PipelineContext &ctx);
    void TrackColumnUsage(Expression *expr, PipelineContext &ctx);
};

}  // namespace duckdb

#endif  // GPU_CODE_GENERATOR_H