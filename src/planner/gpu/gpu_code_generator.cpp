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
    code << "extern \"C\" __global__ void gpu_kernel(";

    // Add kernel parameters
    kernel_params =
        GenerateKernelParams(pipeline);  // Store parameters for later use
    for (size_t i = 0; i < kernel_params.size(); i++) {
        code << "    " << kernel_params[i].type << " " << kernel_params[i].name;
        if (i < kernel_params.size() - 1) {
            code << ",";
        }
        code << "\n";
    }
    code << ") {\n";

    // Get thread and block indices
    code << "    int tid = blockIdx.x * blockDim.x + threadIdx.x;\n";
    code << "    int stride = blockDim.x * gridDim.x;\n\n";

    code << "    if (tid == 0) {\n";
    code << "        printf(\"Hello from GPU!\\n\");\n";
    code << "    }\n";

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
    code << "#include <vector>\n";
    code << "#include <iostream>\n";
    code << "#include <string>\n";
    code << "#include <unordered_map>\n\n";

    code << "extern \"C\" CUfunction gpu_kernel;\n\n";

    // Define structure for pointer mapping
    code << "struct PointerMapping {\n";
    code << "    const char* name;\n";
    code << "    void* address;\n";
    code << "    int cid;  // Chunk ID for GPU chunk cache manager\n";
    code << "};\n\n";

    code << "extern \"C\" void execute_query(size_t N, PointerMapping* "
            "ptr_mappings, int num_mappings) {\n";
    code << "    // Create lookup map for easy access\n";
    code << "    std::unordered_map<std::string, void*> ptr_map;\n";
    code << "    std::unordered_map<std::string, int> cid_map;\n";
    code << "    for (int i = 0; i < num_mappings; i++) {\n";
    code
        << "        ptr_map[ptr_mappings[i].name] = ptr_mappings[i].address;\n";
    code << "        cid_map[ptr_mappings[i].name] = ptr_mappings[i].cid;\n";
    code << "    }\n\n";
    // code << "    // Example: access cache manager and chunk data\n";
    // code << "    // auto* ccm = static_cast<void*>(ptr_map[\"gpu_cache_manager\"]);\n";
    // code << "    // int chunk_cid = cid_map[\"segment_data\"];\n";
    // code << "    // auto* chunk_data = static_cast<void*>(ptr_map[\"segment_data\"]);\n";
    code << "    cudaError_t err;\n";

    for (auto &p : kernel_params) {
        if (p.type.find('*') != std::string::npos) {
            std::string elem = p.type.substr(0, p.type.find('*'));
            std::string h = p.name + "_h";
            std::string d = p.name + "_d";

            code << "    std::vector<" << elem << "> " << h << "(N, 0);\n";
            code << "    " << elem << "* " << d << " = nullptr;\n";
            code << "    err = cudaMalloc(&" << d << ", N * sizeof(" << elem
                 << "));\n";
            code << "    if (err) throw std::runtime_error(\"cudaMalloc "
                    "failed\");\n";
            code << "    err = cudaMemcpy(" << d << ", " << h
                 << ".data(), N * sizeof(" << elem
                 << "), cudaMemcpyHostToDevice);\n";
            code << "    if (err) throw std::runtime_error(\"H2D memcpy "
                    "failed\");\n\n";
        }
    }

    code << "    const int blockSize = 256;\n";
    code << "    const int gridSize  = static_cast<int>((N + blockSize - 1) / "
            "blockSize);\n";
    code << "    void* args[] = {";
    for (size_t i = 0; i < kernel_params.size(); ++i) {
        const auto &p = kernel_params[i];
        code << (p.type.find('*') != std::string::npos ? "&" + p.name + "_d"
                                                       : "&" + p.name);
        if (i + 1 < kernel_params.size())
            code << ", ";
    }
    code << "};\n";

    code << "    std::cerr << \"Hello from host code\" << std::endl;\n";
    code << "    std::cerr << \"gpu_kernel ptr = \" << (void*)gpu_kernel << "
            "std::endl;\n";
    code << "    std::cerr << \"grid=\" << gridSize << \"  block=\" << "
            "blockSize << std::endl;\n";

    code << "    CUresult r = cuLaunchKernel(gpu_kernel, gridSize,1,1, "
            "blockSize,1,1, 0, 0, nullptr, nullptr);\n";
    // code << "    if (r != CUDA_SUCCESS) throw std::runtime_error(\"cuLaunchKernel failed\");\n";
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

    for (auto &p : kernel_params) {
        if (p.type.find('*') != std::string::npos) {
            std::string elem = p.type.substr(0, p.type.find('*'));
            std::string h = p.name + "_h";
            std::string d = p.name + "_d";
            code << "    cudaFree(" << d << ");\n";
        }
    }

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
    mapping.name = name.c_str();
    mapping.address = address;
    mapping.cid = cid;
    pointer_mappings.push_back(mapping);
}

std::vector<KernelParam> GpuCodeGenerator::GenerateKernelParams(
    const CypherPipeline &pipeline)
{
    if (!kernel_params.empty()) {
        return kernel_params;  // Return cached parameters if already generated
    }

    std::vector<KernelParam> params;

    // Process each operator in the pipeline
    for (size_t i = 0; i < pipeline.GetOperators().size(); i++) {
        auto op = pipeline.GetOperators()[i];

        // Handle Scan operator
        // if (auto scan_op = dynamic_cast<CypherPhysicalScan*>(op)) {
        //     auto table_name = scan_op->GetTableName();
        //     auto column_count = scan_op->GetColumnCount();

        //     // Add count parameter
        //     KernelParam count_param;
        //     count_param.name = table_name + "_count";
        //     count_param.type = "int";
        //     count_param.value = std::to_string(scan_op->GetRowCount());
        //     count_param.is_device_ptr = false;
        //     params.push_back(count_param);

        //     // Add data buffer parameters for each column
        //     for (int col = 0; col < column_count; col++) {
        //         KernelParam data_param;
        //         data_param.name = table_name + "_data[" + std::to_string(col) + "]";
        //         data_param.type = "void*";
        //         data_param.is_device_ptr = true;
        //         params.push_back(data_param);

        //         KernelParam buffer_param;
        //         buffer_param.name = table_name + "_buffer[" + std::to_string(col) + "]";
        //         buffer_param.type = "void*";
        //         buffer_param.is_device_ptr = true;
        //         params.push_back(buffer_param);
        //     }
        // }
    }

    return params;
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
        // This follows the logic from extent_iterator.cpp
        Catalog &catalog = context.db->GetCatalog();
        PropertySchemaCatalogEntry *property_schema_cat_entry =
            (PropertySchemaCatalogEntry *)catalog.GetEntry(
                context, DEFAULT_SCHEMA, oid);

        if (property_schema_cat_entry) {
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
                    // Process each chunk in the extent
                    for (size_t chunk_idx = 0;
                         chunk_idx < extent_cat_entry->chunks.size();
                         chunk_idx++) {
                        ChunkDefinitionID cdf_id =
                            extent_cat_entry->chunks[chunk_idx];

                        // Create a unique name for this chunk
                        std::string chunk_name =
                            "chunk_" + std::to_string(cdf_id);

                        // Add pointer mapping for this chunk
                        code_gen->AddPointerMapping(chunk_name, nullptr,
                                                    cdf_id);

                        // Generate kernel code for this chunk
                        // code << "    // Process chunk " << cdf_id << " (extent "
                        //      << extent_id << ", property " << oid << ")\n";
                        // code << "    for (int i = tid; i < chunk_" << cdf_id
                        //      << "_count; i += stride) {\n";
                        // code << "        // TODO: Add actual scan logic for "
                        //         "chunk "
                        //      << cdf_id << "\n";
                        // code << "    }\n";
                    }
                }
            }
        }
    }
}

}  // namespace duckdb