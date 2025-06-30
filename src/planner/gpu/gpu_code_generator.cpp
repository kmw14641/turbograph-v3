#include "planner/gpu/gpu_code_generator.hpp"
#include <cuda.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <nvrtc.h>
#include <sstream>
#include "main/database.hpp"
#include "catalog/catalog.hpp"
#include "catalog/catalog_entry/list.hpp"
#include "common/file_system.hpp"
#include "execution/physical_operator/cypher_physical_operator.hpp"
#include "execution/physical_operator/physical_node_scan.hpp"
#include "llvm/Support/TargetSelect.h"

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
    // Add more operator generators here as needed
    // operator_generators[PhysicalOperatorType::JOIN] = std::make_unique<JoinCodeGenerator>();
    // operator_generators[PhysicalOperatorType::FILTER] = std::make_unique<FilterCodeGenerator>();
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
    std::stringstream code;

    // Generate kernel function
    code << "extern \"C\" __global__ void gpu_kernel(\n";

    // Add kernel parameters
    GenerateKernelParams(pipeline);
    for (size_t i = 0; i < kernel_params.size(); i++) {
        code << "    " << kernel_params[i].type << kernel_params[i].name;
        if (i < kernel_params.size() - 1) {
            code << ",";
        }
        code << "\n";
    }
    code << ") {\n";

    // Get thread and block indices
    code << "    int tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
    code << "    int stride = blockDim.x * gridDim.x;\n\n";

    // Process each operator in the pipeline
    for (int i = 0; i < pipeline.GetPipelineLength(); i++) {
        auto op = pipeline.GetIdxOperator(i);

        // Use strategy pattern to generate operator-specific code
        GenerateOperatorCode(op, code);
    }

    code << "}\n";

    generated_gpu_code = code.str();
}

void GpuCodeGenerator::GenerateHostCode(CypherPipeline &pipeline)
{
    std::stringstream code;

    code << "#include <cuda.h>\n";
    code << "#include <cuda_runtime.h>\n";
    code << "#include <cstdint>\n";
    code << "#include <vector>\n";
    code << "#include <iostream>\n";
    code << "#include <string>\n";
    code << "#include <unordered_map>\n\n";

    code << "extern \"C\" CUfunction gpu_kernel;\n\n";

    // Define structure for pointer mapping
    code << "struct PointerMapping {\n";
    code << "    const char *name;\n";
    code << "    void *address;\n";
    code << "    int cid;  // Chunk ID for GPU chunk cache manager\n";
    code << "};\n\n";

    code << "extern \"C\" void execute_query(PointerMapping *ptr_mappings, int num_mappings) {\n";
    code << "    cudaError_t err;\n\n";
    
    // Generate variable declarations for each parameter
    int param_index = 0;
    for (const auto &p : kernel_params) {
        if (p.type.find('*') != std::string::npos) {
            // For pointer types, declare as void* and assign from ptr_mappings
            code << "    void *" << p.name << " = ptr_mappings[" << param_index << "].address;\n";
            param_index++;
        } else {
            // For non-pointer types, declare as the actual type
            code << "    " << p.type << " " << p.name << " = " << p.value << ";\n";
        }
    }
    code << "\n";

    code << "    const int blockSize = 128;\n";
    code << "    const int gridSize  = 3280;\n";
    code << "    void *args[] = {";
    for (size_t i = 0; i < kernel_params.size(); ++i) {
        const auto &p = kernel_params[i];
        code << "&" << p.name;
        if (i + 1 < kernel_params.size())
            code << ", ";
    }
    code << "};\n\n";

    // code << "    std::cerr << \"Hello from host code\" << std::endl;\n";
    // code << "    std::cerr << \"gpu_kernel ptr = \" << (void*)gpu_kernel << "
    //         "std::endl;\n";
    // code << "    std::cerr << \"grid=\" << gridSize << \"  block=\" << "
    //         "blockSize << std::endl;\n";

    code << "    CUresult r = cuLaunchKernel(gpu_kernel, gridSize,1,1, "
            "blockSize,1,1, 0, 0, args, nullptr);\n";
    code << "    if (r != CUDA_SUCCESS) {\n";
    code << "        const char *name = nullptr, *str = nullptr;\n";
    code << "        cuGetErrorName(r, &name);\n";
    code << "        cuGetErrorString(r, &str);\n";
    code << "        std::cerr << \"cuLaunchKernel failed: \" << "
            "(name?name:\"\") << \" – \" << (str?str:\"\") << std::endl;\n";
    code << "        throw std::runtime_error(\"cuLaunchKernel failed\");\n";
    code << "    }\n";
    code << "    cudaError_t errSync = cudaDeviceSynchronize();\n";
    code << "    if (errSync != cudaSuccess) {\n";
    code << "        std::cerr << \"sync error: \" << "
            "cudaGetErrorString(errSync) << std::endl;\n";
    code << "        throw std::runtime_error(\"cudaDeviceSynchronize "
            "failed\");\n";
    code << "    }\n";

    code << "    std::cout << \"Query finished on GPU.\" << std::endl;\n";
    code << "}\n";

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
            (PropertySchemaCatalogEntry *)catalog.GetEntry(
                context, DEFAULT_SCHEMA, oid);
                
        if (property_schema_cat_entry) {
            // Generate table name (graphletX format)
            std::string table_name = "graphlet" + std::to_string(oid);
            std::string short_table_name = "gr" + std::to_string(oid);
            
            // Add count parameter for this table
            KernelParam count_param;
            count_param.name = table_name + "_count";
            count_param.type = "int ";
            count_param.value = std::to_string(10); // TODO: tmp
            count_param.is_device_ptr = false;
            kernel_params.push_back(count_param);

            // Add data buffer parameters for each column
            auto column_names = property_schema_cat_entry->GetKeys();
            auto column_types = property_schema_cat_entry->GetTypes();
            for (size_t col_idx = 0;
                 col_idx < column_names->size();
                 col_idx++) {
                std::string col_name = column_names->at(col_idx);

                // Generate parameter names based on verbose mode
                std::string param_name;
                if (this->GetVerboseMode()) {
                    param_name = table_name + "_" + col_name;
                }
                else {
                    param_name =
                        short_table_name + "_" + std::to_string(col_idx);
                }

                // Add data buffer parameter
                KernelParam data_param;
                data_param.name = param_name + "_data";
                data_param.type = "void *";
                // data_param.type = ConvertLogicalTypeToPrimitiveType(
                //     column_types->at(col_idx)) + " *";
                data_param.is_device_ptr = true;
                kernel_params.push_back(data_param);
                
                // // Add buffer size parameter
                // KernelParam size_param;
                // size_param.name = param_name + "_size";
                // size_param.type = "int";
                // size_param.value = "0"; // Will be set dynamically
                // size_param.is_device_ptr = false;
                // kernel_params.push_back(size_param);
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
        for (size_t col_idx = 0; col_idx < output_column_names.size(); col_idx++) {
            std::string col_name = output_column_names[col_idx];
            
            // Generate output parameter names
            std::string output_param_name;
            if (this->GetVerboseMode()) {
                output_param_name = output_table_name + "_" + col_name;
            } else {
                output_param_name = short_output_name + "_" + std::to_string(col_idx);
            }
            
            // Add output data buffer parameter
            KernelParam output_data_param;
            output_data_param.name = output_param_name + "_data";
            output_data_param.type = "void *";  // Always void* for CUDA compatibility
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

void GpuCodeGenerator::GenerateOperatorCode(CypherPhysicalOperator *op,
                                            std::stringstream &code)
{
    auto it = operator_generators.find(op->GetOperatorType());
    if (it != operator_generators.end()) {
        it->second->GenerateCode(op, code, this, context);
    }
    else {
        // Default handling for unknown operators
        code << "    // Unknown operator type: "
             << static_cast<int>(op->GetOperatorType()) << "\n";
    }
}

void NodeScanCodeGenerator::GenerateCode(CypherPhysicalOperator *op,
                                         std::stringstream &code,
                                         GpuCodeGenerator *code_gen,
                                         ClientContext &context)
{
    auto scan_op = dynamic_cast<PhysicalNodeScan *>(op);
    if (!scan_op)
        return;

    code << "    // Scan operator\n";

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

            auto column_names = property_schema_cat_entry->GetKeys();
            
            // Get extent IDs from the property schema
            for (size_t extent_idx = 0;
                 extent_idx < property_schema_cat_entry->extent_ids.size();
                 extent_idx++) {
                idx_t extent_id =
                    property_schema_cat_entry->extent_ids[extent_idx];

                // Get extent catalog entry to access chunks
                ExtentCatalogEntry *extent_cat_entry =
                    (ExtentCatalogEntry *)catalog.GetEntry(
                        context, CatalogType::EXTENT_ENTRY, DEFAULT_SCHEMA,
                        DEFAULT_EXTENT_PREFIX + std::to_string(extent_id));

                if (extent_cat_entry) {
                    D_ASSERT(column_names->size() == extent_cat_entry->chunks.size());
                    // Process each chunk in the extent
                    for (size_t chunk_idx = 0;
                         chunk_idx < extent_cat_entry->chunks.size();
                         chunk_idx++) {
                        ChunkDefinitionID cdf_id =
                            extent_cat_entry->chunks[chunk_idx];

                        // Create a unique name for this chunk
                        std::string chunk_name = "graphlet" +
                                                 std::to_string(oid) + "_" +
                                                 std::string(column_names->at(chunk_idx)) +
                                                 "_" + std::to_string(cdf_id);

                        // Add pointer mapping for this chunk
                        code_gen->AddPointerMapping(chunk_name, nullptr,
                                                    cdf_id);

                        // Generate kernel code for this chunk
                        code << "    // Process chunk " << cdf_id << " (extent "
                             << extent_id << ", property " << oid << ")\n";
                        
                        // Generate scan loop using the new parameter naming
                        std::string count_param_name = table_name + "_count";
                        code << "    for (int i = tid; i < " << count_param_name << "; i += stride) {\n";
                        
                        // Add column access based on verbose mode
                        for (size_t col_idx = 0; col_idx < column_names->size(); col_idx++) {
                            std::string col_name = std::string(column_names->at(col_idx));
                            
                            // Generate parameter names based on verbose mode
                            std::string param_name;
                            if (code_gen->GetVerboseMode()) {
                                param_name = table_name + "_" + col_name;
                            } else {
                                param_name = short_table_name + "_" + std::to_string(col_idx);
                            }
                            
                            // Cast void* to uint64_t* for data access and print
                            code << "        uint64_t* " << param_name << "_ptr = static_cast<uint64_t*>(" << param_name << "_data);\n";
                            code << "        printf(\"Thread %d: " << param_name << "[%d] = %lu\\n\", tid, i, " << param_name << "_ptr[i]);\n";
                        }
                        
                        // Add output writing logic
                        code << "        // Write to output buffers\n";
                        code << "        if (tid == 0) {\n";  // Only first thread writes output count for now
                        code << "            *output_count = " << count_param_name << ";\n";
                        code << "        }\n";
                        
                        // Write input data to output (simple copy for now)
                        for (size_t col_idx = 0; col_idx < column_names->size(); col_idx++) {
                            std::string col_name = std::string(column_names->at(col_idx));
                            
                            // Generate input parameter names
                            std::string input_param_name;
                            if (code_gen->GetVerboseMode()) {
                                input_param_name = table_name + "_" + col_name;
                            } else {
                                input_param_name = short_table_name + "_" + std::to_string(col_idx);
                            }
                            
                            // Generate output parameter names
                            std::string output_param_name;
                            if (code_gen->GetVerboseMode()) {
                                output_param_name = "output_" + col_name;
                            } else {
                                output_param_name = "out_" + std::to_string(col_idx);
                            }
                            
                            code << "        uint64_t* " << output_param_name << "_ptr = static_cast<uint64_t*>(" << output_param_name << "_data);\n";
                            code << "        " << output_param_name << "_ptr[i] = " << input_param_name << "_ptr[i];\n";
                            code << "        printf(\"Thread %d: wrote " << output_param_name << "[%d] = %lu\\n\", tid, i, " << output_param_name << "_ptr[i]);\n";
                        }
                        
                        code << "    }\n";
                    }
                }
            }
        }
    }
}

}  // namespace duckdb