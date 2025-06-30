#ifndef GPU_CODE_GENERATOR_H
#define GPU_CODE_GENERATOR_H

#include <cuda.h>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "common/constants.hpp"
#include "common/types.hpp"
#include "common/vector.hpp"
#include "execution/cypher_pipeline.hpp"
#include "execution/physical_operator/cypher_physical_operator.hpp"
#include "planner/gpu/gpu_jit_compiler.hpp"
#include "planner/expression.hpp"
#include "planner/expression/bound_reference_expression.hpp"
#include "planner/expression/bound_constant_expression.hpp"
#include "planner/expression/bound_function_expression.hpp"
#include "planner/expression/bound_operator_expression.hpp"
#include "planner/expression/bound_comparison_expression.hpp"
#include "common/types/value.hpp"

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

// Structure for tracking attribute access patterns
struct AttributeAccess {
    std::string table_name;
    std::string column_name;
    idx_t extent_id;
    idx_t chunk_id;
    bool is_loaded;
    std::string param_name;
};

// Structure for lazy materialization tracking
struct LazyMaterializationInfo {
    std::vector<AttributeAccess> required_attributes;
    std::vector<AttributeAccess> loaded_attributes;
    std::unordered_map<std::string, std::string> attribute_to_param_mapping;
};

// Helper class for code generation with indentation
class CodeBuilder {
public:
    void Add(int nesting_level, const std::string &line) {
        for (int i = 0; i < nesting_level; ++i) {
            ss << "    "; // 4 spaces per level
        }
        ss << line << "\n";
    }
    std::string str() const { return ss.str(); }
    void clear() { ss.str(""); ss.clear(); }
private:
    std::stringstream ss;
};

// Strategy pattern for operator-specific code generation
class OperatorCodeGenerator {
   public:
    virtual ~OperatorCodeGenerator() = default;
    virtual void GenerateCode(CypherPhysicalOperator *op,
                              CodeBuilder &code,
                              GpuCodeGenerator *code_gen,
                              ClientContext &context,
                              int nesting_level = 0,
                              bool is_main_loop = false) = 0;
};

class NodeScanCodeGenerator : public OperatorCodeGenerator {
   public:
    void GenerateCode(CypherPhysicalOperator *op, CodeBuilder &code,
                      GpuCodeGenerator *code_gen, ClientContext &context,
                      int nesting_level = 0,
                      bool is_main_loop = false) override;
};

class ProjectionCodeGenerator : public OperatorCodeGenerator {
   public:
    void GenerateCode(CypherPhysicalOperator *op, CodeBuilder &code,
                      GpuCodeGenerator *code_gen, ClientContext &context,
                      int nesting_level = 0,
                      bool is_main_loop = false) override;

   private:
    void AnalyzeExpressionForAttributes(Expression *expr, 
                                       CodeBuilder &code,
                                       GpuCodeGenerator *code_gen,
                                       ClientContext &context,
                                       int nesting_level = 0);
    void GenerateProjectionExpressionCode(Expression *expr, 
                                         size_t expr_idx,
                                         CodeBuilder &code,
                                         GpuCodeGenerator *code_gen,
                                         ClientContext &context,
                                         int nesting_level = 0);
    std::string ConvertLogicalTypeToCUDAType(LogicalType type);
    std::string ConvertValueToCUDALiteral(const Value &value);
    std::string ExpressionTypeToString(ExpressionType type);
    void GenerateFunctionCallCode(BoundFunctionExpression *func_expr,
                                 const std::string &output_var,
                                 CodeBuilder &code,
                                 GpuCodeGenerator *code_gen,
                                 ClientContext &context,
                                 int nesting_level = 0);
    void GenerateOperatorCode(BoundOperatorExpression *op_expr,
                             const std::string &output_var,
                             CodeBuilder &code,
                             GpuCodeGenerator *code_gen,
                             ClientContext &context,
                             int nesting_level = 0);
};

class ProduceResultsCodeGenerator : public OperatorCodeGenerator {
   public:
    void GenerateCode(CypherPhysicalOperator *op, CodeBuilder &code,
                      GpuCodeGenerator *code_gen, ClientContext &context,
                      int nesting_level = 0,
                      bool is_main_loop = false) override;
};

class FilterCodeGenerator : public OperatorCodeGenerator {
   public:
    void GenerateCode(CypherPhysicalOperator *op, CodeBuilder &code,
                      GpuCodeGenerator *code_gen, ClientContext &context,
                      int nesting_level = 0,
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
        return kernel_params;
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

    // Lazy materialization management
    void AddRequiredAttribute(const std::string &table_name, const std::string &column_name,
                             idx_t extent_id, idx_t chunk_id, const std::string &param_name);
    void MarkAttributeAsLoaded(const std::string &table_name, const std::string &column_name);
    bool IsAttributeLoaded(const std::string &table_name, const std::string &column_name) const;
    std::string GetAttributeParamName(const std::string &table_name, const std::string &column_name) const;
    void GenerateLazyLoadCode(const std::string &table_name, const std::string &column_name, 
                             CodeBuilder &code, GpuCodeGenerator *code_gen, int nesting_level = 0);
    void ClearLazyMaterializationInfo();

    // Cleanup resources
    void Cleanup();

    // Generate code for a specific operator
    void GenerateOperatorCode(CypherPhysicalOperator *op,
                              CodeBuilder &code,
                              int nesting_level = 0,
                              bool is_main_loop = false);

    // Generate main scan loop with nested operators
    void GenerateMainScanLoop(CypherPipeline &pipeline, CodeBuilder &code, int nesting_level = 0);

    // Process remaining operators recursively
    void ProcessRemainingOperators(CypherPipeline &pipeline, int op_idx, CodeBuilder &code, int nesting_level = 0);

   private:
    ClientContext &context;

    std::string generated_gpu_code;
    std::string generated_cpu_code;
    std::string current_code_hash;

    std::vector<KernelParam> kernel_params;
    std::vector<MemoryTransferInfo> memory_transfers;
    std::map<std::string, size_t> device_memory_sizes;

    CUmodule gpu_module = nullptr;
    CUfunction kernel_function = nullptr;

    std::unique_ptr<GpuJitCompiler> jit_compiler;

    bool is_compiled;
    bool is_repeatable;
    bool verbose_mode = true;  // Control parameter naming style

    std::vector<PointerMapping> pointer_mappings;

    // Lazy materialization tracking
    LazyMaterializationInfo lazy_materialization_info;

    // Strategy pattern for operator-specific code generation
    std::unordered_map<PhysicalOperatorType,
                       std::unique_ptr<OperatorCodeGenerator>>
        operator_generators;

    // Initialize operator generators
    void InitializeOperatorGenerators();
};

}  // namespace duckdb

#endif  // GPU_CODE_GENERATOR_H