#ifndef GPU_CODE_GENERATOR_H
#define GPU_CODE_GENERATOR_H

#include <cuda.h>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <fstream>
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

struct ScanColumnInfo {
    ScanColumnInfo() : get_physical_id_column(false), graphlet_id(0) {}
    bool get_physical_id_column = false;
    uint64_t graphlet_id;
    std::vector<ExtentID> extent_ids;
    std::vector<uint64_t> num_tuples_per_extent;
    std::vector<uint64_t> col_position;
    std::vector<uint64_t> col_type_size;
    std::vector<std::string> col_name;
    std::vector<std::vector<ChunkDefinitionID>> chunk_ids;
};

enum class PipeInputType : uint8_t {
    TYPE_0_FALSE, // scan
    TYPE_0_TRUE, // scan
    TYPE_1_FALSE, // multi
    TYPE_1_TRUE, // multi
    TYPE_2_FALSE, // filter
    TYPE_2_TRUE // filter
};

enum class PipeOutputType : uint8_t {
    TYPE_0_FALSE, // materialize, do not lb
    TYPE_0_TRUE, // materialize, do lb
    TYPE_1_FALSE, // multi, do not lb
    TYPE_1_TRUE, // multi, do lb
    TYPE_2_FALSE, // filter, do not lb
    TYPE_2_TRUE // filter, do lb
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
    virtual void GenerateGlobalDeclaration(CypherPhysicalOperator *op,
                                           CypherPhysicalOperator *prev_op,
                                           size_t op_idx, CodeBuilder &code,
                                           GpuCodeGenerator *code_gen,
                                           ClientContext &context,
                                           PipelineContext &pipeline_ctx)
    {
        return;
    }
    virtual void GenerateDeclarationInHostCode(CypherPhysicalOperator *op,
                                               size_t op_idx, CodeBuilder &code,
                                               GpuCodeGenerator *code_gen,
                                               PipelineContext &pipeline_ctx)
    {
        return;  // Default implementation does nothing
    }
    virtual void GenerateCodeForLocalVariable(CypherPhysicalOperator *op,
                                              size_t op_idx, CodeBuilder &code,
                                              GpuCodeGenerator *code_gen,
                                              PipelineContext &pipeline_ctx)
    {
        return;  // Default implementation does nothing
    }
    virtual void GenerateInputKernelParameters(
        CypherPhysicalOperator *op, GpuCodeGenerator *code_gen,
        ClientContext &context, PipelineContext &pipeline_context,
        std::vector<KernelParam> &input_kernel_params,
        std::vector<ScanColumnInfo> &scan_column_infos)
    {
        throw NotImplementedException(
            "GenerateInputKernelParameters not implemented for this operator");
    }
    virtual void GenerateOutputKernelParameters(
        CypherPhysicalOperator *op, GpuCodeGenerator *code_gen,
        ClientContext &context, PipelineContext &pipeline_context,
        std::vector<KernelParam> &output_kernel_params)
    {
        throw NotImplementedException(
            "GenerateInputKernelParameters not implemented for this operator");
    }
    virtual void AnalyzeOperatorForMaterialization(
        CypherPhysicalOperator *op, int sub_idx, int op_idx,
        PipelineContext &pipeline_context, GpuCodeGenerator *code_gen) = 0;
};

class NodeScanCodeGenerator : public OperatorCodeGenerator {
   public:
    void GenerateCode(CypherPhysicalOperator *op, CodeBuilder &code,
                      GpuCodeGenerator *code_gen, ClientContext &context,
                      PipelineContext &pipeline_ctx,
                      bool is_main_loop = false) override;
    void GenerateInputKernelParameters(
        CypherPhysicalOperator *op, GpuCodeGenerator *code_gen,
        ClientContext &context, PipelineContext &pipeline_context,
        std::vector<KernelParam> &input_kernel_params,
        std::vector<ScanColumnInfo> &scan_column_infos);
    void AnalyzeOperatorForMaterialization(CypherPhysicalOperator *op,
                                           int sub_idx, int op_idx,
                                           PipelineContext &pipeline_context,
                                           GpuCodeGenerator *code_gen) override;
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
    void AnalyzeOperatorForMaterialization(CypherPhysicalOperator *op,
                                           int sub_idx, int op_idx,
                                           PipelineContext &pipeline_context,
                                           GpuCodeGenerator *code_gen) override;
};

class ProduceResultsCodeGenerator : public OperatorCodeGenerator {
   public:
    void GenerateCode(CypherPhysicalOperator *op, CodeBuilder &code,
                      GpuCodeGenerator *code_gen, ClientContext &context,
                      PipelineContext &pipeline_ctx,
                      bool is_main_loop = false) override;
    void GenerateDeclarationInHostCode(CypherPhysicalOperator *op,
                                       size_t op_idx, CodeBuilder &code,
                                       GpuCodeGenerator *code_gen,
                                       PipelineContext &pipeline_ctx) override;
    void GenerateCodeForLocalVariable(CypherPhysicalOperator *op, size_t op_idx,
                                      CodeBuilder &code,
                                      GpuCodeGenerator *code_gen,
                                      PipelineContext &pipeline_ctx) override;
    void GenerateOutputKernelParameters(
        CypherPhysicalOperator *op, GpuCodeGenerator *code_gen,
        ClientContext &context, PipelineContext &pipeline_context,
        std::vector<KernelParam> &output_kernel_params) override;
    void AnalyzeOperatorForMaterialization(CypherPhysicalOperator *op,
                                           int sub_idx, int op_idx,
                                           PipelineContext &pipeline_context,
                                           GpuCodeGenerator *code_gen) override;
};

class FilterCodeGenerator : public OperatorCodeGenerator {
   public:
    void GenerateCode(CypherPhysicalOperator *op, CodeBuilder &code,
                      GpuCodeGenerator *code_gen, ClientContext &context,
                      PipelineContext &pipeline_ctx,
                      bool is_main_loop = false) override;
    void AnalyzeOperatorForMaterialization(CypherPhysicalOperator *op,
                                           int sub_idx, int op_idx,
                                           PipelineContext &pipeline_context,
                                           GpuCodeGenerator *code_gen) override;
};

class HashAggregateCodeGenerator : public OperatorCodeGenerator {
   public:
    void GenerateCode(CypherPhysicalOperator *op, CodeBuilder &code,
                      GpuCodeGenerator *code_gen, ClientContext &context,
                      PipelineContext &pipeline_ctx,
                      bool is_main_loop = false) override;
    void GenerateBuildSideCode(CypherPhysicalOperator *op, CodeBuilder &code,
                               GpuCodeGenerator *code_gen,
                               ClientContext &context,
                               PipelineContext &pipeline_ctx);
    void GenerateSourceSideCode(CypherPhysicalOperator *op, CodeBuilder &code,
                                GpuCodeGenerator *code_gen,
                                ClientContext &context,
                                PipelineContext &pipeline_ctx);
    void GenerateGlobalDeclaration(CypherPhysicalOperator *op,
                                   CypherPhysicalOperator *prev_op,
                                   size_t op_idx, CodeBuilder &code,
                                   GpuCodeGenerator *code_gen,
                                   ClientContext &context,
                                   PipelineContext &pipeline_ctx) override;
    void GenerateDeclarationInHostCode(CypherPhysicalOperator *op,
                                       size_t op_idx, CodeBuilder &code,
                                       GpuCodeGenerator *code_gen,
                                       PipelineContext &pipeline_ctx) override;
    void GenerateInputKernelParameters(
        CypherPhysicalOperator *op, GpuCodeGenerator *code_gen,
        ClientContext &context, PipelineContext &pipeline_context,
        std::vector<KernelParam> &input_kernel_params,
        std::vector<ScanColumnInfo> &scan_column_infos) override;
    void GenerateOutputKernelParameters(
        CypherPhysicalOperator *op, GpuCodeGenerator *code_gen,
        ClientContext &context, PipelineContext &pipeline_context,
        std::vector<KernelParam> &output_kernel_params) override;
    void AnalyzeOperatorForMaterialization(CypherPhysicalOperator *op,
                                           int sub_idx, int op_idx,
                                           PipelineContext &pipeline_context,
                                           GpuCodeGenerator *code_gen) override;
};

class GpuCodeGenerator {
   public:
    GpuCodeGenerator(ClientContext &context);
    ~GpuCodeGenerator();

    void InitializeLLVMTargets();

    // Generate global declarations for the query
    void GenerateGlobalDeclarations(std::vector<CypherPipeline *> &pipelines);

    // Generate GPU code
    void GenerateGPUCode(CypherPipeline &pipeline);

    // Generate CPU code
    void GenerateCPUCode(std::vector<CypherPipeline *> &pipelines);

    // Generate GPU kernel code
    void GenerateKernelCode(CypherPipeline &pipeline);

    // Generate GPU host code
    void GenerateHostCode(std::vector<CypherPipeline *> &pipelines);

    // Generate kernel parameters
    void GenerateKernelParams(const CypherPipeline &pipeline);

    // Compile generated CUDA code using nvcc
    bool CompileGeneratedCode();

    // Get compiled host function
    void *GetCompiledHost();

    GpuKernelArgs &GetKernelArgs()
    {
        return kernel_args;
    }

    LogicalType GetLogicalTypeFromId(LogicalTypeId type_id,
                                     uint16_t extra_info = 0);
    std::string ConvertLogicalTypeToPrimitiveType(LogicalType &type,
                                                  bool do_strip = false,
                                                  bool get_short_name = false);
    std::string ConvertLogicalTypeIdToPrimitiveType(LogicalTypeId type_id,
                                                    uint16_t extra_info);
    std::string GetValidVariableName(const std::string &name, size_t col_idx);
    std::string GetInitValueForAggregate(
        const BoundAggregateExpression *bound_agg);

    // Set whether this kernel needs to be repeatable
    void SetRepeatable(bool repeatable) { is_repeatable = repeatable; }

    // Set verbose mode for parameter naming
    void SetVerboseMode(bool verbose) { verbose_mode = verbose; }

    // Get verbose mode
    bool GetVerboseMode() const { return verbose_mode; }

    // Add pointer mapping
    void AddPointerMapping(const std::string &name, void *address,
                           ChunkDefinitionID cid);
    
    void AddInitFunctionName(const std::string &name)
    {
        initfn_names.push_back(name);
    }

    // Get pointer mappings
    const std::vector<PointerMapping> &GetPointerMappings() const
    {
        return pointer_mappings;
    }

    // Get scan column information
    const std::vector<ScanColumnInfo> &GetScanColumnInfos() const
    {
        return scan_column_infos;
    }

    // Cleanup resources
    void Cleanup();

    // Generate code for a specific operator
    void GenerateOperatorCode(CypherPhysicalOperator *op, CodeBuilder &code,
                              PipelineContext &pipeline_ctx, bool is_main_loop);

    // Generate global declaration code for a specific operator
    void GenerateGlobalDeclaration(CypherPhysicalOperator *op,
                                   CypherPhysicalOperator *prev_op,
                                   size_t op_idx, CodeBuilder &code,
                                   PipelineContext &pipeline_ctx);
    void GenerateCodeForLocalVariable(CypherPhysicalOperator *op, size_t op_idx,
                                      CodeBuilder &code,
                                      PipelineContext &pipeline_ctx);

    // Generate pipeline code
    void GeneratePipelineCode(CypherPipeline &pipeline, CodeBuilder &code);

    // Generate sub-pipeline code
    void GenerateSubPipelineCode(CypherPipeline &pipeline, CodeBuilder &code);

    // Process remaining operators recursively
    void ProcessRemainingOperators(CypherPipeline &pipeline, int op_idx,
                                   CodeBuilder &code);

    // Generate input code for a specific type
    void GenerateInputCode(CypherPipeline &sub_pipeline, CodeBuilder &code,
                           PipelineContext &pipeline_ctx,
                           PipeInputType input_type);
    void GenerateInputCodeForType0(CypherPipeline &sub_pipeline,
                                   CodeBuilder &code,
                                   PipelineContext &pipeline_ctx);
    void GenerateInputCodeForType1(CypherPipeline &sub_pipeline,
                                   CodeBuilder &code,
                                   PipelineContext &pipeline_ctx);
    void GenerateInputCodeForType2(CypherPipeline &sub_pipeline,
                                   CodeBuilder &code,
                                   PipelineContext &pipeline_ctx);
    
    // Generate output code for a specific type
    void GenerateOutputCode(CypherPipeline &sub_pipeline, CodeBuilder &code,
                            PipelineContext &pipeline_ctx,
                            PipeOutputType output_type);
    void GenerateOutputCodeForType0(CypherPipeline &sub_pipeline,
                                    CodeBuilder &code,
                                    PipelineContext &pipeline_ctx);
    void GenerateOutputCodeForType1(CypherPipeline &sub_pipeline,
                                    CodeBuilder &code,
                                    PipelineContext &pipeline_ctx);
    void GenerateOutputCodeForType2(CypherPipeline &sub_pipeline,
                                    CodeBuilder &code,
                                    PipelineContext &pipeline_ctx);
    int FindLowerLoopLvl(PipelineContext &pipeline_context);

    // Helper code for get pipe input/output type
    PipeInputType GetPipeInputType(CypherPipeline &sub_pipeline,
                                   int sub_pipe_idx);
    PipeOutputType GetPipeOutputType(CypherPipeline &sub_pipeline);

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

    void GenerateCodeForMaterialization(CodeBuilder &code,
                                        PipelineContext &pipeline_context);
    void GenerateCodeForLocalVariable(CodeBuilder &code,
                                      PipelineContext &pipeline_context);

    void GenerateDeclarationInHostCode(std::vector<CypherPipeline *> &pipelines,
                                       CodeBuilder &code);
    void GenerateKernelCallInHostCode(std::vector<CypherPipeline *> &pipelines,
                                      CodeBuilder &code);

    // Pipeline context management
    void InitializePipelineContext(CypherPipeline &pipeline);
    void AdvanceOperator();

    void GetReferencedColumns(Expression *expr,
                              std::vector<uint64_t> &referenced_columns);

    // Initialize operator generators
    void InitializeOperatorGenerators();

    // Split pipeline into sub-pipelines
    void SplitPipelineIntoSubPipelines(CypherPipeline &pipeline);

    // Analyze sub-pipelines for materialization
    void AnalyzeSubPipelinesForMaterialization();

    void ResolveAttributes();

    void ResolveBoundaryAttributes();

   private:
    ClientContext &context;
    GpuKernelArgs kernel_args;  // GPU kernel configuration

    std::string global_declarations;
    std::string gpu_include_header;
    std::string cpu_include_header;
    std::string generated_gpu_code;
    std::string generated_cpu_code;
    std::string current_code_hash;
    std::vector<std::string> initfn_names;

    std::vector<std::vector<KernelParam>> input_kernel_params;
    std::vector<std::vector<KernelParam>> output_kernel_params;
    std::vector<MemoryTransferInfo> memory_transfers;
    std::map<std::string, size_t> device_memory_sizes;

    std::unique_ptr<GpuJitCompiler> jit_compiler;

    bool is_compiled;
    bool is_repeatable;
    bool verbose_mode = false;  // Control parameter naming style
    bool do_inter_warp_lb = true;
    bool doWorkoadSizeTracking = true;
    bool generateInputPtrMapping = true;
    int tsWidth = 32;
    std::string idleWarpDetectionType = "twolvlbitmaps";  // twolvlbitmaps / idqueue

    std::vector<PointerMapping> pointer_mappings;
    std::vector<ScanColumnInfo> scan_column_infos;

    // Strategy pattern for operator-specific code generation
    std::unordered_map<PhysicalOperatorType,
                       std::unique_ptr<OperatorCodeGenerator>>
        operator_generators;

    // Pipeline context for the entire pipeline
    PipelineContext pipeline_context;
    int num_pipelines_compiled = 0;

    // for debugging
    std::ofstream gpu_code_file;
    std::ofstream cpu_code_file;
};

}  // namespace duckdb

#endif  // GPU_CODE_GENERATOR_H