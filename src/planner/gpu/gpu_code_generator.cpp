#include "planner/gpu/gpu_code_generator.hpp"
#include <cuda.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <nvrtc.h>
#include <sstream>
#include "catalog/catalog.hpp"
#include "catalog/catalog_entry/list.hpp"
#include "common/file_system.hpp"
#include "execution/physical_operator/cypher_physical_operator.hpp"
#include "execution/physical_operator/physical_filter.hpp"
#include "execution/physical_operator/physical_node_scan.hpp"
#include "execution/physical_operator/physical_produce_results.hpp"
#include "execution/physical_operator/physical_projection.hpp"
#include "llvm/Support/TargetSelect.h"
#include "main/database.hpp"

namespace duckdb {

GpuCodeGenerator::GpuCodeGenerator(ClientContext &context)
    : context(context), is_compiled(false), is_repeatable(false)
{
    InitializeLLVMTargets();
    jit_compiler = std::make_unique<GpuJitCompiler>();
    InitializeOperatorGenerators();
}

GpuCodeGenerator::~GpuCodeGenerator()
{
    Cleanup();
}

void GpuCodeGenerator::InitializeLLVMTargets()
{
    static bool is_llvm_targets_initialized = false;
    if (is_llvm_targets_initialized)
        return;
    is_llvm_targets_initialized = true;

    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
}

void GpuCodeGenerator::InitializeOperatorGenerators()
{
    operator_generators[PhysicalOperatorType::NODE_SCAN] =
        std::make_unique<NodeScanCodeGenerator>();
    operator_generators[PhysicalOperatorType::FILTER] =
        std::make_unique<FilterCodeGenerator>();
    operator_generators[PhysicalOperatorType::PROJECTION] =
        std::make_unique<ProjectionCodeGenerator>();
    operator_generators[PhysicalOperatorType::PRODUCE_RESULTS] =
        std::make_unique<ProduceResultsCodeGenerator>();
}

void GpuCodeGenerator::GenerateGPUCode(CypherPipeline &pipeline)
{
    // generate kernel code
    GenerateKernelCode(pipeline);

    // then, generate host code
    GenerateHostCode(pipeline);

    // for debug
    std::cout << generated_gpu_code << std::endl;
    std::cout << generated_cpu_code << std::endl;
}

void GpuCodeGenerator::GenerateKernelCode(CypherPipeline &pipeline)
{
    CodeBuilder code;
    int nesting_level = 0;

    // Generate kernel function
    code.Add(nesting_level, "extern \"C\" __global__ void gpu_kernel(");
    nesting_level++;

    // Add kernel parameters
    GenerateKernelParams(pipeline);
    for (size_t i = 0; i < kernel_params.size(); i++) {
        code.Add(nesting_level, kernel_params[i].type + kernel_params[i].name +
                                    (i < kernel_params.size() - 1 ? "," : ""));
    }
    nesting_level--;
    code.Add(nesting_level, ") {");
    nesting_level++;

    // Get thread and block indices
    code.Add(nesting_level, "int tid = blockIdx.x * blockDim.x + threadIdx.x;");
    code.Add(nesting_level, "int stride = blockDim.x * gridDim.x;\n");

    // Generate the main scan loop that will contain all operators
    GenerateMainScanLoop(pipeline, code, nesting_level);

    nesting_level--;
    code.Add(nesting_level, "}");

    generated_gpu_code = code.str();
}

void GpuCodeGenerator::GenerateHostCode(CypherPipeline &pipeline)
{
    CodeBuilder code;
    int nesting_level = 0;

    code.Add(nesting_level, "#include <cuda.h>");
    code.Add(nesting_level, "#include <cuda_runtime.h>");
    code.Add(nesting_level, "#include <cstdint>");
    code.Add(nesting_level, "#include <vector>");
    code.Add(nesting_level, "#include <iostream>");
    code.Add(nesting_level, "#include <string>");
    code.Add(nesting_level, "#include <unordered_map>\n");

    code.Add(nesting_level, "extern \"C\" CUfunction gpu_kernel;\n");

    // Define structure for pointer mapping
    code.Add(nesting_level, "struct PointerMapping {");
    nesting_level++;
    code.Add(nesting_level, "const char *name;");
    code.Add(nesting_level, "void *address;");
    code.Add(nesting_level,
             "int cid;  // Chunk ID for GPU chunk cache manager");
    nesting_level--;
    code.Add(nesting_level, "};\n");

    code.Add(nesting_level,
             "extern \"C\" void execute_query(PointerMapping *ptr_mappings, "
             "int num_mappings) {");
    nesting_level++;
    code.Add(nesting_level, "cudaError_t err;\n");

    // Generate variable declarations for each parameter
    int param_index = 0;
    for (const auto &p : kernel_params) {
        if (p.type.find('*') != std::string::npos) {
            // For pointer types, declare as void* and assign from ptr_mappings
            code.Add(nesting_level, "void *" + p.name + " = ptr_mappings[" +
                                        std::to_string(param_index) +
                                        "].address;");
            param_index++;
        }
        else {
            // For non-pointer types, declare as the actual type
            code.Add(nesting_level,
                     p.type + " " + p.name + " = " + p.value + ";");
        }
    }
    code.Add(nesting_level, "");

    code.Add(nesting_level, "const int blockSize = 128;");
    code.Add(nesting_level, "const int gridSize  = 3280;");
    std::string args_line = "void *args[] = {";
    for (size_t i = 0; i < kernel_params.size(); ++i) {
        const auto &p = kernel_params[i];
        args_line += "&" + p.name;
        if (i + 1 < kernel_params.size())
            args_line += ", ";
    }
    args_line += "};";
    code.Add(nesting_level, args_line + "\n");

    code.Add(nesting_level,
             "CUresult r = cuLaunchKernel(gpu_kernel, gridSize,1,1, "
             "blockSize,1,1, 0, 0, args, nullptr);");
    code.Add(nesting_level, "if (r != CUDA_SUCCESS) {");
    nesting_level++;
    code.Add(nesting_level, "const char *name = nullptr, *str = nullptr;");
    code.Add(nesting_level, "cuGetErrorName(r, &name);");
    code.Add(nesting_level, "cuGetErrorString(r, &str);");
    code.Add(nesting_level,
             "std::cerr << \"cuLaunchKernel failed: \" << (name?name:"
             ") << \" – \" << (str?str:"
             ") << std::endl;");
    code.Add(nesting_level,
             "throw std::runtime_error(\"cuLaunchKernel failed\");");
    nesting_level--;
    code.Add(nesting_level, "}");
    code.Add(nesting_level, "cudaError_t errSync = cudaDeviceSynchronize();");
    code.Add(nesting_level, "if (errSync != cudaSuccess) {");
    nesting_level++;
    code.Add(nesting_level,
             "std::cerr << \"sync error: \" << cudaGetErrorString(errSync) << "
             "std::endl;");
    code.Add(nesting_level,
             "throw std::runtime_error(\"cudaDeviceSynchronize failed\");");
    nesting_level--;
    code.Add(nesting_level, "}");

    code.Add(nesting_level,
             "std::cout << \"Query finished on GPU.\" << std::endl;");
    nesting_level--;
    code.Add(nesting_level, "}");

    generated_cpu_code = code.str();
}

bool GpuCodeGenerator::CompileGeneratedCode()
{
    if (generated_gpu_code.empty() || generated_cpu_code.empty())
        return false;

    if (!jit_compiler->CompileWithNVRTC(generated_gpu_code, "gpu_kernel",
                                        gpu_module, kernel_function))
        return false;

    if (!jit_compiler->CompileWithORCLLJIT(generated_cpu_code,
                                           kernel_function)) {
        return false;
    }

    is_compiled = true;
    return true;
}

void *GpuCodeGenerator::GetCompiledHost()
{
    if (!is_compiled) {
        return nullptr;
    }
    return jit_compiler->GetMainFunction();
}

void GpuCodeGenerator::Cleanup()
{
    if (!is_repeatable && is_compiled) {
        // If not repeatable and compiled, release the kernel
        // jit_compiler->ReleaseKernel(current_code_hash);
        is_compiled = false;
    }
}

void GpuCodeGenerator::AddPointerMapping(const std::string &name, void *address,
                                         ChunkDefinitionID cid)
{
    PointerMapping mapping;
    mapping.name = name;
    mapping.address = address;
    mapping.cid = cid;
    pointer_mappings.push_back(mapping);
}

void GpuCodeGenerator::GenerateKernelParams(const CypherPipeline &pipeline)
{
    // Clear existing parameters
    kernel_params.clear();

    // Only process the first operator (assuming it's a scan)
    if (pipeline.GetPipelineLength() == 0) {
        return;
    }

    auto first_op = pipeline.GetSource();

    // Only handle scan operators for now
    if (first_op->GetOperatorType() != PhysicalOperatorType::NODE_SCAN) {
        throw std::runtime_error("Only scan operators are supported for now");
    }

    // Handle scan operator
    auto scan_op = dynamic_cast<PhysicalNodeScan *>(first_op);
    if (!scan_op) {
        throw std::runtime_error("Only scan operators are supported for now");
    }

    // Process oids to get table/column information
    for (size_t oid_idx = 0; oid_idx < scan_op->oids.size(); oid_idx++) {
        idx_t oid = scan_op->oids[oid_idx];

        // Get property schema catalog entry using oid
        Catalog &catalog = context.db->GetCatalog();
        PropertySchemaCatalogEntry *property_schema_cat_entry =
            (PropertySchemaCatalogEntry *)catalog.GetEntry(context,
                                                           DEFAULT_SCHEMA, oid);

        if (property_schema_cat_entry) {
            // Generate table name (graphletX format)
            std::string table_name = "graphlet" + std::to_string(oid);
            std::string short_table_name = "gr" + std::to_string(oid);

            // Add count parameter for this table
            KernelParam count_param;
            count_param.name = table_name + "_count";
            count_param.type = "int ";
            count_param.value = std::to_string(10);  // TODO: tmp
            count_param.is_device_ptr = false;
            kernel_params.push_back(count_param);

            // Get extent IDs from the property schema
            for (size_t extent_idx = 0;
                 extent_idx < property_schema_cat_entry->extent_ids.size();
                 extent_idx++) {
                idx_t extent_id =
                    property_schema_cat_entry->extent_ids[extent_idx];

                // Get extent catalog entry to access chunks (columns)
                ExtentCatalogEntry *extent_cat_entry =
                    (ExtentCatalogEntry *)catalog.GetEntry(
                        context, CatalogType::EXTENT_ENTRY, DEFAULT_SCHEMA,
                        DEFAULT_EXTENT_PREFIX + std::to_string(extent_id));

                if (extent_cat_entry) {
                    // Each chunk in the extent represents a column
                    for (size_t chunk_idx = 0;
                         chunk_idx < extent_cat_entry->chunks.size();
                         chunk_idx++) {
                        ChunkDefinitionID cdf_id =
                            extent_cat_entry->chunks[chunk_idx];

                        // Generate column name based on chunk index
                        std::string col_name =
                            "col_" + std::to_string(chunk_idx);

                        // Generate parameter names based on verbose mode
                        std::string param_name;
                        if (this->GetVerboseMode()) {
                            param_name = table_name + "_" + col_name;
                        }
                        else {
                            param_name = short_table_name + "_" +
                                         std::to_string(chunk_idx);
                        }

                        // Add data buffer parameter for this column (chunk)
                        KernelParam data_param;
                        data_param.name = param_name + "_data";
                        data_param.type = "void *";
                        data_param.is_device_ptr = true;
                        kernel_params.push_back(data_param);

                        // Add pointer mapping for this chunk (column)
                        std::string chunk_name =
                            "chunk_" + std::to_string(cdf_id);
                        AddPointerMapping(chunk_name, nullptr, cdf_id);
                    }
                }
            }
        }
    }

    // Add output parameters based on sink operator
    auto sink_op = pipeline.GetSink();
    if (sink_op) {
        // Generate output table name
        std::string output_table_name = "output";
        std::string short_output_name = "out";

        // Add output count parameter
        KernelParam output_count_param;
        output_count_param.name = output_table_name + "_count";
        output_count_param.type = "int ";
        output_count_param.value = "0";
        output_count_param.is_device_ptr = true;
        kernel_params.push_back(output_count_param);

        // Add output data parameters based on sink schema
        auto &output_schema = sink_op->GetSchema();
        auto &output_column_names = output_schema.getStoredColumnNamesRef();
        for (size_t col_idx = 0; col_idx < output_column_names.size();
             col_idx++) {
            std::string col_name = output_column_names[col_idx];
            // Replace '.' with '_' for valid C/C++ variable names
            std::replace(col_name.begin(), col_name.end(), '.', '_');

            // Generate output parameter names
            std::string output_param_name;
            if (this->GetVerboseMode()) {
                output_param_name = output_table_name + "_" + col_name;
            }
            else {
                output_param_name = short_output_name + "_" + std::to_string(col_idx);
            }

            // Add output data buffer parameter
            KernelParam output_data_param;
            output_data_param.name = output_param_name + "_data";
            output_data_param.type =
                "void *";  // Always void* for CUDA compatibility
            output_data_param.is_device_ptr = true;
            kernel_params.push_back(output_data_param);
        }
    }
}

std::string GpuCodeGenerator::ConvertLogicalTypeToPrimitiveType(
    LogicalTypeId type_id)
{
    switch (type_id) {
        case LogicalTypeId::BOOLEAN:
            return "bool";
        case LogicalTypeId::TINYINT:
            return "int8_t";
        case LogicalTypeId::SMALLINT:
            return "int16_t";
        case LogicalTypeId::INTEGER:
            return "int32_t";
        case LogicalTypeId::BIGINT:
            return "int64_t";
        case LogicalTypeId::UBIGINT:
            return "uint64_t";
        case LogicalTypeId::UTINYINT:
            return "uint8_t";
        case LogicalTypeId::USMALLINT:
            return "uint16_t";
        case LogicalTypeId::UINTEGER:
            return "uint32_t";
        case LogicalTypeId::FLOAT:
            return "float";
        case LogicalTypeId::DOUBLE:
            return "double";
        case LogicalTypeId::VARCHAR:
            return "char*";
        case LogicalTypeId::DATE:
            return "int32_t";
        case LogicalTypeId::TIME:
            return "int32_t";
        case LogicalTypeId::TIMESTAMP:
            return "int64_t";
        case LogicalTypeId::INTERVAL:
            return "int64_t";
        case LogicalTypeId::UUID:
            return "int64_t";
        default:
            throw std::runtime_error("Unsupported logical type: " +
                                     std::to_string((uint8_t)type_id));
    }
}

void GpuCodeGenerator::GenerateMainScanLoop(CypherPipeline &pipeline,
                                            CodeBuilder &code,
                                            int &nesting_level)
{
    auto first_op = pipeline.GetSource();
    if (first_op->GetOperatorType() == PhysicalOperatorType::NODE_SCAN) {
        GenerateOperatorCode(first_op, code, nesting_level,
                             /*is_main_loop=*/true);

        ProcessRemainingOperators(pipeline, 1, code, nesting_level);

        code.Add(nesting_level, "}");
    }
}

void GpuCodeGenerator::ProcessRemainingOperators(CypherPipeline &pipeline,
                                                 int op_idx, CodeBuilder &code,
                                                 int &nesting_level)
{
    if (op_idx >= pipeline.GetPipelineLength()) {
        return;
    }

    auto op = pipeline.GetIdxOperator(op_idx);

    switch (op->GetOperatorType()) {
        case PhysicalOperatorType::FILTER:
            code.Add(nesting_level, "if (condition) {");
            nesting_level++;
            GenerateOperatorCode(op, code, nesting_level,
                                 /*is_main_loop=*/false);
            ProcessRemainingOperators(pipeline, op_idx + 1, code,
                                      nesting_level);
            nesting_level--;
            code.Add(nesting_level, "}");
            break;

            // case PhysicalOperatorType::JOIN:
            //     GenerateOperatorCode(op, code, /*is_main_loop=*/false);
            //     ProcessRemainingOperators(pipeline, op_idx + 1, code);
            //     break;

        case PhysicalOperatorType::PROJECTION:
            GenerateOperatorCode(op, code, nesting_level,
                                 /*is_main_loop=*/false);
            ProcessRemainingOperators(pipeline, op_idx + 1, code,
                                      nesting_level);
            break;

        case PhysicalOperatorType::PRODUCE_RESULTS:
            GenerateOperatorCode(op, code, nesting_level,
                                 /*is_main_loop=*/false);
            ProcessRemainingOperators(pipeline, op_idx + 1, code,
                                      nesting_level);
            break;

        default:
            GenerateOperatorCode(op, code, nesting_level,
                                 /*is_main_loop=*/false);
            ProcessRemainingOperators(pipeline, op_idx + 1, code,
                                      nesting_level);
            break;
    }
}

void GpuCodeGenerator::GenerateOperatorCode(CypherPhysicalOperator *op,
                                            CodeBuilder &code,
                                            int &nesting_level,
                                            bool is_main_loop)
{
    auto it = operator_generators.find(op->GetOperatorType());
    if (it != operator_generators.end()) {
        it->second->GenerateCode(op, code, this, context, nesting_level,
                                 is_main_loop);
    }
    else {
        // Default handling for unknown operators
        code.Add(nesting_level,
                 "// Unknown operator type: " +
                     std::to_string(static_cast<int>(op->GetOperatorType())));
    }
}

void NodeScanCodeGenerator::GenerateCode(CypherPhysicalOperator *op,
                                         CodeBuilder &code,
                                         GpuCodeGenerator *code_gen,
                                         ClientContext &context,
                                         int &nesting_level, bool is_main_loop)
{
    auto scan_op = dynamic_cast<PhysicalNodeScan *>(op);
    if (!scan_op)
        return;

    if (is_main_loop) {
        code.Add(nesting_level, "// Scan operator");

        // Process oids and scan_projection_mapping to get chunk IDs
        for (size_t oid_idx = 0; oid_idx < scan_op->oids.size(); oid_idx++) {
            idx_t oid = scan_op->oids[oid_idx];

            // Get property schema catalog entry using oid
            Catalog &catalog = context.db->GetCatalog();
            PropertySchemaCatalogEntry *property_schema_cat_entry =
                (PropertySchemaCatalogEntry *)catalog.GetEntry(
                    context, DEFAULT_SCHEMA, oid);

            if (property_schema_cat_entry) {
                // Generate table name (graphletX format)
                std::string table_name = "graphlet" + std::to_string(oid);
                std::string short_table_name = "gr" + std::to_string(oid);

                // Get extent IDs from the property schema
                for (size_t extent_idx = 0;
                     extent_idx < property_schema_cat_entry->extent_ids.size();
                     extent_idx++) {
                    idx_t extent_id =
                        property_schema_cat_entry->extent_ids[extent_idx];

                    // Get extent catalog entry to access chunks (columns)
                    ExtentCatalogEntry *extent_cat_entry =
                        (ExtentCatalogEntry *)catalog.GetEntry(
                            context, CatalogType::EXTENT_ENTRY, DEFAULT_SCHEMA,
                            DEFAULT_EXTENT_PREFIX + std::to_string(extent_id));

                    uint64_t tuple_id_base = extent_id;
                    tuple_id_base <<= 32;
                    if (extent_cat_entry) {
                        // Generate scan loop for this extent
                        std::string count_param_name = table_name + "_count";
                        code.Add(nesting_level, "// Process extent " +
                                                    std::to_string(extent_id) +
                                                    " (property " +
                                                    std::to_string(oid) + ")");
                        code.Add(nesting_level, "for (int i = tid; i < " +
                                                    count_param_name +
                                                    "; i += stride) {");
                        nesting_level++;

                        // lazy materialization
                        code.Add(nesting_level, "// lazy materialization");
                        code.Add(nesting_level,
                                 "unsigned long long tuple_id_base = " +
                                     std::to_string(tuple_id_base) + ";");
                        code.Add(nesting_level,
                                 "unsigned long long tuple_id = tuple_id_base "
                                 "+ tid;");

                        // Track all available attributes for lazy materialization
                        for (size_t chunk_idx = 0;
                             chunk_idx < extent_cat_entry->chunks.size();
                             chunk_idx++) {
                            ChunkDefinitionID cdf_id =
                                extent_cat_entry->chunks[chunk_idx];

                            // Generate column name based on chunk index
                            std::string col_name =
                                "col_" + std::to_string(chunk_idx);

                            // Generate parameter names based on verbose mode
                            std::string param_name;
                            if (code_gen->GetVerboseMode()) {
                                param_name = table_name + "_" + col_name;
                            }
                            else {
                                param_name = short_table_name + "_" +
                                             std::to_string(chunk_idx);
                            }

                            // Add to lazy materialization tracking
                            code_gen->AddRequiredAttribute(table_name, col_name,
                                                           extent_id, cdf_id,
                                                           param_name);
                        }

                        break;
                    }
                }
            }
        }
    }
    else {
        code.Add(nesting_level, "// Additional scan logic (if needed)");
    }
}

void ProjectionCodeGenerator::GenerateCode(CypherPhysicalOperator *op,
                                           CodeBuilder &code,
                                           GpuCodeGenerator *code_gen,
                                           ClientContext &context,
                                           int &nesting_level, bool is_main_loop)
{
    auto proj_op = dynamic_cast<PhysicalProjection *>(op);
    if (!proj_op) {
        return;
    }

    code.Add(nesting_level, "// Projection operator");

    // Process each projection expression
    for (size_t expr_idx = 0; expr_idx < proj_op->expressions.size();
         expr_idx++) {
        auto &expr = proj_op->expressions[expr_idx];
        // Only generate projection code (no analysis)
        GenerateProjectionExpressionCode(expr.get(), expr_idx, code, code_gen,
                                         context, nesting_level);
    }
}

void ProjectionCodeGenerator::GenerateProjectionExpressionCode(
    Expression *expr, size_t expr_idx, CodeBuilder &code,
    GpuCodeGenerator *code_gen, ClientContext &context, int &nesting_level)
{
    if (!expr)
        return;

    std::string output_var = "proj_result_" + std::to_string(expr_idx);

    switch (expr->expression_class) {
        case ExpressionClass::BOUND_REF: {
            // Do not generate any code for simple reference
            break;
        }
        case ExpressionClass::BOUND_CONSTANT: {
            // Constant expression
            auto const_expr = dynamic_cast<BoundConstantExpression *>(expr);
            if (const_expr) {
                code.Add(nesting_level, "// Constant value");
                code.Add(nesting_level,
                         ConvertLogicalTypeToCUDAType(expr->return_type) + " " +
                             output_var + " = " +
                             ConvertValueToCUDALiteral(const_expr->value) +
                             ";");
            }
            break;
        }
        case ExpressionClass::BOUND_FUNCTION: {
            // Function expression
            auto func_expr = dynamic_cast<BoundFunctionExpression *>(expr);
            if (func_expr) {
                code.Add(nesting_level,
                         "// Function call: " + func_expr->function.name);
                GenerateFunctionCallCode(func_expr, output_var, code, code_gen,
                                         context, nesting_level);
            }
            break;
        }
        case ExpressionClass::BOUND_OPERATOR: {
            // Operator expression
            auto op_expr = dynamic_cast<BoundOperatorExpression *>(expr);
            if (op_expr) {
                code.Add(
                    nesting_level,
                    "// Operator: " + ExpressionTypeToString(op_expr->type));
                GenerateOperatorCode(op_expr, output_var, code, code_gen,
                                     context, nesting_level);
            }
            break;
        }
        default:
            // Default case - just assign a placeholder
            code.Add(nesting_level, "// Unsupported expression type");
            code.Add(nesting_level,
                     ConvertLogicalTypeToCUDAType(expr->return_type) + " " +
                         output_var + " = 0; // TODO: implement");
            break;
    }

    // Store the result in output buffer, except for BOUND_REF
    if (expr->expression_class != ExpressionClass::BOUND_REF) {
        code.Add(nesting_level, "// Store projection result");
        code.Add(nesting_level, "output_col_" + std::to_string(expr_idx) + "[tid] = " + output_var + ";");
    }
}

std::string ProjectionCodeGenerator::ConvertLogicalTypeToCUDAType(
    LogicalType type)
{
    switch (type.id()) {
        case LogicalTypeId::BOOLEAN:
            return "bool";
        case LogicalTypeId::TINYINT:
            return "int8_t";
        case LogicalTypeId::SMALLINT:
            return "int16_t";
        case LogicalTypeId::INTEGER:
            return "int32_t";
        case LogicalTypeId::BIGINT:
            return "int64_t";
        case LogicalTypeId::UBIGINT:
            return "uint64_t";
        case LogicalTypeId::FLOAT:
            return "float";
        case LogicalTypeId::DOUBLE:
            return "double";
        case LogicalTypeId::VARCHAR:
            return "char*";
        default:
            return "uint64_t";  // Default to uint64_t for compatibility
    }
}

std::string ProjectionCodeGenerator::ConvertValueToCUDALiteral(
    const Value &value)
{
    switch (value.type().id()) {
        case LogicalTypeId::BOOLEAN:
            return value.GetValue<bool>() ? "true" : "false";
        case LogicalTypeId::TINYINT:
        case LogicalTypeId::SMALLINT:
        case LogicalTypeId::INTEGER:
        case LogicalTypeId::BIGINT:
            return std::to_string(value.GetValue<int64_t>());
        case LogicalTypeId::UBIGINT:
            return std::to_string(value.GetValue<uint64_t>());
        case LogicalTypeId::FLOAT:
            return std::to_string(value.GetValue<float>()) + "f";
        case LogicalTypeId::DOUBLE:
            return std::to_string(value.GetValue<double>());
        case LogicalTypeId::VARCHAR:
            return "\"" + value.GetValue<string>() + "\"";
        default:
            return "0";  // Default value
    }
}

std::string ProjectionCodeGenerator::ExpressionTypeToString(ExpressionType type)
{
    switch (type) {
        case ExpressionType::COMPARE_EQUAL:
            return "==";
        case ExpressionType::COMPARE_NOTEQUAL:
            return "!=";
        case ExpressionType::COMPARE_LESSTHAN:
            return "<";
        case ExpressionType::COMPARE_LESSTHANOREQUALTO:
            return "<=";
        case ExpressionType::COMPARE_GREATERTHAN:
            return ">";
        case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
            return ">=";
        default:
            return "unknown";
    }
}

void ProjectionCodeGenerator::GenerateFunctionCallCode(
    BoundFunctionExpression *func_expr, const std::string &output_var,
    CodeBuilder &code, GpuCodeGenerator *code_gen, ClientContext &context,
    int &nesting_level)
{
    // For now, we'll implement a simple approach
    // In the future, this should be expanded to handle different function types

    if (func_expr->children.size() == 1) {
        // Unary function
        code.Add(nesting_level,
                 ConvertLogicalTypeToCUDAType(func_expr->return_type) + " " +
                     output_var + " = ");

        // Generate lazy loading for the input
        auto child_expr = func_expr->children[0].get();
        if (child_expr->expression_class == ExpressionClass::BOUND_REF) {
            auto ref_expr =
                dynamic_cast<BoundReferenceExpression *>(child_expr);
            if (ref_expr) {
                std::string input_col_name =
                    "input_col_" + std::to_string(ref_expr->index);
                code.Add(nesting_level, "// Lazy load input column if needed");
                code.Add(nesting_level,
                         "if (!" + input_col_name + "_loaded) {");
                code.Add(nesting_level + 1,
                         input_col_name + "_ptr = static_cast<uint64_t*>(" +
                             input_col_name + "_data);");
                code.Add(nesting_level + 1, input_col_name + "_loaded = true;");
                code.Add(nesting_level, "}");

                if (func_expr->function.name == "abs") {
                    code.Add(nesting_level, output_var + " = abs(" +
                                                input_col_name + "_ptr[i]);");
                }
                else if (func_expr->function.name == "sqrt") {
                    code.Add(nesting_level, output_var + " = sqrt(" +
                                                input_col_name + "_ptr[i]);");
                }
                else {
                    code.Add(nesting_level,
                             output_var + " = " + input_col_name +
                                 "_ptr[i]; // TODO: implement function " +
                                 func_expr->function.name);
                }
            }
        }
        else {
            code.Add(nesting_level,
                     "input_col_0; // TODO: get actual input column");
        }
    }
    else {
        // Multi-argument function
        code.Add(nesting_level,
                 ConvertLogicalTypeToCUDAType(func_expr->return_type) + " " +
                     output_var +
                     " = 0; // TODO: implement multi-arg function");
    }
}

void ProjectionCodeGenerator::GenerateOperatorCode(
    BoundOperatorExpression *op_expr, const std::string &output_var,
    CodeBuilder &code, GpuCodeGenerator *code_gen, ClientContext &context,
    int &nesting_level)
{
    if (op_expr->children.size() == 2) {
        // Binary operator
        code.Add(nesting_level,
                 ConvertLogicalTypeToCUDAType(op_expr->return_type) + " " +
                     output_var + " = ");

        // Generate lazy loading for both operands
        std::string left_operand, right_operand;

        if (op_expr->children[0]->expression_class ==
            ExpressionClass::BOUND_REF) {
            auto ref_expr = dynamic_cast<BoundReferenceExpression *>(
                op_expr->children[0].get());
            if (ref_expr) {
                left_operand =
                    "input_col_" + std::to_string(ref_expr->index) + "_ptr[i]";
                std::string input_col_name =
                    "input_col_" + std::to_string(ref_expr->index);
                code.Add(nesting_level, "// Lazy load left operand if needed");
                code.Add(nesting_level,
                         "if (!" + input_col_name + "_loaded) {");
                code.Add(nesting_level + 1,
                         input_col_name + "_ptr = static_cast<uint64_t*>(" +
                             input_col_name + "_data);");
                code.Add(nesting_level + 1, input_col_name + "_loaded = true;");
                code.Add(nesting_level, "}");
            }
        }
        else {
            left_operand = "0";  // TODO: handle other expression types
        }

        if (op_expr->children[1]->expression_class ==
            ExpressionClass::BOUND_REF) {
            auto ref_expr = dynamic_cast<BoundReferenceExpression *>(
                op_expr->children[1].get());
            if (ref_expr) {
                right_operand =
                    "input_col_" + std::to_string(ref_expr->index) + "_ptr[i]";
                std::string input_col_name =
                    "input_col_" + std::to_string(ref_expr->index);
                code.Add(nesting_level, "// Lazy load right operand if needed");
                code.Add(nesting_level,
                         "if (!" + input_col_name + "_loaded) {");
                code.Add(nesting_level + 1,
                         input_col_name + "_ptr = static_cast<uint64_t*>(" +
                             input_col_name + "_data);");
                code.Add(nesting_level + 1, input_col_name + "_loaded = true;");
                code.Add(nesting_level, "}");
            }
        }
        else {
            right_operand = "0";  // TODO: handle other expression types
        }

        code.Add(nesting_level, output_var + " = " + left_operand + " " +
                                    ExpressionTypeToString(op_expr->type) +
                                    " " + right_operand + ";");
    }
    else {
        // Unary or other operator
        code.Add(nesting_level,
                 ConvertLogicalTypeToCUDAType(op_expr->return_type) + " " +
                     output_var + " = 0; // TODO: implement operator");
    }
}

void ProduceResultsCodeGenerator::GenerateCode(
    CypherPhysicalOperator *op, CodeBuilder &code, GpuCodeGenerator *code_gen,
    ClientContext &context, int &nesting_level, bool is_main_loop)
{
    auto results_op = dynamic_cast<PhysicalProduceResults *>(op);
    if (!results_op) {
        return;
    }

    code.Add(nesting_level, "// Produce results operator");

    // Get the output schema to determine what columns to write
    auto &output_schema = results_op->GetSchema();
    auto &output_column_names = output_schema.getStoredColumnNamesRef();

    code.Add(nesting_level, "// Write results to output buffers");
    for (size_t col_idx = 0; col_idx < output_column_names.size(); col_idx++) {
        std::string col_name = output_column_names[col_idx];
        // Replace '.' with '_' for valid C/C++ variable names
        std::replace(col_name.begin(), col_name.end(), '.', '_');
        std::string output_param_name;
        if (code_gen->GetVerboseMode()) {
            output_param_name = "output_" + col_name;
        } else {
            output_param_name = "out_" + std::to_string(col_idx);
        }
        std::string output_data_name = output_param_name + "_data";
        std::string output_ptr_name = output_param_name + "_ptr";
        code.Add(nesting_level, "// Write column " + std::to_string(col_idx) +
                                    " (" + col_name + ") to output");
        code.Add(nesting_level, "if (" + output_data_name + " != nullptr) {");
        code.Add(nesting_level + 1,
                 "uint64_t* " + output_ptr_name + " = static_cast<uint64_t*>(" + output_data_name + ");");
        code.Add(nesting_level + 1, output_ptr_name + "[tid] = proj_result_" + std::to_string(col_idx) + ";");
        code.Add(nesting_level, "}");
    }

    // Update output count
    code.Add(nesting_level, "// Update output count atomically");
    code.Add(nesting_level, "atomicAdd(output_count, 1);");

    code.Add(nesting_level, "// Results produced successfully");
}

void FilterCodeGenerator::GenerateCode(CypherPhysicalOperator *op,
                                       CodeBuilder &code,
                                       GpuCodeGenerator *code_gen,
                                       ClientContext &context,
                                       int &nesting_level, bool is_main_loop)
{
    // For now, we'll implement a simple filter
    // In the future, this should analyze the filter expression and generate appropriate code

    code.Add(nesting_level, "// Filter operator - check condition");
    code.Add(
        nesting_level,
        "bool condition = true; // TODO: implement actual filter condition");
    code.Add(nesting_level, "if (!condition) {");
    code.Add(nesting_level + 1,
             "continue; // Skip this tuple if condition is false");
    code.Add(nesting_level, "}");
    code.Add(nesting_level, "// Filter condition passed, continue processing");
}

// Lazy materialization implementation
void GpuCodeGenerator::AddRequiredAttribute(const std::string &table_name,
                                            const std::string &column_name,
                                            idx_t extent_id, idx_t chunk_id,
                                            const std::string &param_name)
{
    AttributeAccess attr;
    attr.table_name = table_name;
    attr.column_name = column_name;
    attr.extent_id = extent_id;
    attr.chunk_id = chunk_id;
    attr.is_loaded = false;
    attr.param_name = param_name;

    lazy_materialization_info.required_attributes.push_back(attr);
    lazy_materialization_info
        .attribute_to_param_mapping[table_name + "." + column_name] =
        param_name;
}

void GpuCodeGenerator::MarkAttributeAsLoaded(const std::string &table_name,
                                             const std::string &column_name)
{
    std::string key = table_name + "." + column_name;
    for (auto &attr : lazy_materialization_info.required_attributes) {
        if (attr.table_name == table_name && attr.column_name == column_name) {
            attr.is_loaded = true;
            lazy_materialization_info.loaded_attributes.push_back(attr);
            break;
        }
    }
}

bool GpuCodeGenerator::IsAttributeLoaded(const std::string &table_name,
                                         const std::string &column_name) const
{
    for (const auto &attr : lazy_materialization_info.loaded_attributes) {
        if (attr.table_name == table_name && attr.column_name == column_name) {
            return true;
        }
    }
    return false;
}

std::string GpuCodeGenerator::GetAttributeParamName(
    const std::string &table_name, const std::string &column_name) const
{
    std::string key = table_name + "." + column_name;
    auto it = lazy_materialization_info.attribute_to_param_mapping.find(key);
    if (it != lazy_materialization_info.attribute_to_param_mapping.end()) {
        return it->second;
    }
    return "";
}

void GpuCodeGenerator::GenerateLazyLoadCode(const std::string &table_name,
                                            const std::string &column_name,
                                            CodeBuilder &code,
                                            GpuCodeGenerator *code_gen,
                                            int &nesting_level)
{
    if (IsAttributeLoaded(table_name, column_name)) {
        return;  // Already loaded
    }

    std::string param_name = GetAttributeParamName(table_name, column_name);
    if (param_name.empty()) {
        return;  // No parameter mapping found
    }

    // Generate lazy load code
    code.Add(nesting_level,
             "// Lazy load for " + table_name + "." + column_name);
    code.Add(nesting_level, "if (!" + param_name + "_loaded) {");
    code.Add(nesting_level + 1, param_name + "_ptr = static_cast<uint64_t*>(" +
                                    param_name + "_data);");
    code.Add(nesting_level + 1, param_name + "_loaded = true;");
    code.Add(nesting_level, "}");

    MarkAttributeAsLoaded(table_name, column_name);
}

void GpuCodeGenerator::ClearLazyMaterializationInfo()
{
    lazy_materialization_info.required_attributes.clear();
    lazy_materialization_info.loaded_attributes.clear();
    lazy_materialization_info.attribute_to_param_mapping.clear();
}

}  // namespace duckdb