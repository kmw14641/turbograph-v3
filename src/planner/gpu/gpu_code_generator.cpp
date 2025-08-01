#include "planner/gpu/gpu_code_generator.hpp"
#include <cuda.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <nvrtc.h>
#include <fstream>
#include <sstream>
#include "catalog/catalog.hpp"
#include "catalog/catalog_entry/list.hpp"
#include "common/file_system.hpp"
#include "common/logger.hpp"
#include "execution/physical_operator/cypher_physical_operator.hpp"
#include "execution/physical_operator/physical_filter.hpp"
#include "execution/physical_operator/physical_node_scan.hpp"
#include "execution/physical_operator/physical_produce_results.hpp"
#include "execution/physical_operator/physical_projection.hpp"
#include "llvm/Support/TargetSelect.h"
#include "main/database.hpp"
#include "planner/gpu/expression_code_generator.hpp"

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
    SCOPED_TIMER_SIMPLE(GenerateGPUCode, spdlog::level::info,
                        spdlog::level::info);
    
    // Split pipeline into sub-pipelines based on filter operators
    SUBTIMER_START(GenerateGPUCode, "SplitPipeline");
    SplitPipelineIntoSubPipelines(pipeline);
    SUBTIMER_STOP(GenerateGPUCode, "SplitPipeline");

    // generate kernel code
    SUBTIMER_START(GenerateGPUCode, "GenerateKernelCode");
    GenerateKernelCode(pipeline);
    SUBTIMER_STOP(GenerateGPUCode, "GenerateKernelCode");

    // then, generate host code
    SUBTIMER_START(GenerateGPUCode, "GenerateHostCode");
    GenerateHostCode(pipeline);
    SUBTIMER_STOP(GenerateGPUCode, "GenerateHostCode");

    // for debug - write to files
    std::ofstream gpu_code_file("generated_gpu_code.cu");
    if (gpu_code_file.is_open()) {
        gpu_code_file << generated_gpu_code << std::endl;
        gpu_code_file.close();
    }

    std::ofstream cpu_code_file("generated_cpu_code.cpp");
    if (cpu_code_file.is_open()) {
        cpu_code_file << generated_cpu_code << std::endl;
        cpu_code_file.close();
    }
}

void GpuCodeGenerator::SplitPipelineIntoSubPipelines(CypherPipeline &pipeline)
{
    CypherPhysicalOperatorGroups groups;
    CypherPhysicalOperatorGroups &pipeline_groups = pipeline.GetOperatorGroups();
    
    for (int op_idx = 0; op_idx < pipeline.GetPipelineLength(); op_idx++) {
        auto op = pipeline.GetIdxOperator(op_idx);
        if (!op) continue;
        
        groups.GetGroups().push_back(pipeline_groups.GetGroups()[op_idx]);
        
        // TODO: implement other cases
        if (op->GetOperatorType() == PhysicalOperatorType::FILTER) {
            pipeline_context.sub_pipelines.emplace_back(groups);
            groups.GetGroups().clear();
            groups.GetGroups().push_back(pipeline_groups.GetGroups()[op_idx]);
        }
    }
    
    pipeline_context.sub_pipelines.emplace_back(groups);
}

void GpuCodeGenerator::GenerateKernelCode(CypherPipeline &pipeline)
{
    CodeBuilder code;

    code.Add("#include \"range.cuh\"");
    code.Add("#include \"themis.cuh\"");
    code.Add("#include \"work_sharing.cuh\"");
    code.Add("#include \"adaptive_work_sharing.cuh\"");
    code.Add(""); // new line

    // Generate kernel function
    code.Add("extern \"C\" __global__ void gpu_kernel(");
    code.IncreaseNesting();

    // Add kernel parameters
    GenerateKernelParams(pipeline);
    for (size_t i = 0; i < input_kernel_params.size(); i++) {
        code.Add(
            input_kernel_params[i].type + input_kernel_params[i].name +
            (i < input_kernel_params.size() - 1 || !output_kernel_params.empty()
                 ? ","
                 : ""));
    }
    for (size_t i = 0; i < output_kernel_params.size(); i++) {
        code.Add(output_kernel_params[i].type + output_kernel_params[i].name +
                 (i < output_kernel_params.size() - 1 ? "," : ""));
    }
    code.DecreaseNesting();
    code.Add(") {");
    code.IncreaseNesting();

    // 1. Generate pipeline initilization
    code.Add("__shared__ int active_thread_ids[" +
             std::to_string(KernelConstants::DEFAULT_BLOCK_SIZE) + "];");

    if (do_inter_warp_lb) {
        std::string max_num_warps =
            std::to_string((1024 / KernelConstants::DEFAULT_BLOCK_SIZE) * 82);
        std::string total_warp_num =
            std::to_string((KernelConstants::DEFAULT_BLOCK_SIZE / 32));
        code.Add("if (blockIdx.x > " + max_num_warps +
                 ") return; // Maximum number of warps a GPU can execute "
                 "concurrently");
        code.Add("int gpart_id = -1;");
        code.Add(
            "Themis::WarpsStatus* warp_status = (Themis::WarpsStatus*) "
            "global_num_idle_warps;");
        code.Add("if (threadIdx.x == 0) {");
        code.IncreaseNesting();
        code.Add("if (warp_status->isTerminated()) active_thread_ids[0] = -1;");
        code.Add("else active_thread_ids[0] = warp_status->addTotalWarpNum(" +
                 total_warp_num + ");");
        code.DecreaseNesting();
        code.Add("}");
        code.Add("__syncthreads();");
        code.Add("gpart_id = active_thread_ids[0];");
        code.Add("__syncthreads();");
        code.Add("if (gpart_id == -1) return;");
        code.Add("gpart_id = gpart_id + threadIdx.x / 32;");
    }

    // Get thread and block indices
    code.Add("int tid = threadIdx.x % 32;");
    code.Add("int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;");
    code.Add("unsigned prefixlanes = (0xffffffffu >> (32 - tid));");
    code.Add("int active = 0;\n");
    code.Add("int stride = blockDim.x * gridDim.x;");  // TODO: remove

    // TODO: refer to pipes2themis.py line 174 ~ 189

    // 2. Generate initial distribution
    code.Add("int inodes_cnts = 0; // the number of nodes per level");
    code.Add("Range ts_0_range_cached;");
    // code.Add("ts_0_range_cached.end = {scan.genTableSize()};')

    if (do_inter_warp_lb) {
        std::string num_warps =
            std::to_string(int(KernelConstants::DEFAULT_BLOCK_SIZE / 32) *
                           KernelConstants::DEFAULT_GRID_SIZE);
        code.Add("int local_scan_offset = 0;");
        code.Add("int global_scan_end = ts_0_range_cached.end;");
        code.Add("Themis::PullINodesAtZeroLvlDynamically<" + num_warps +
                 ">(thread_id, global_scan_offset, global_scan_end, "
                 "local_scan_offset, ts_0_range_cached, inodes_cnts);");
        // if self.interWarpLbMethod == 'aws':
        code.Add("Themis::LocalLevelAndOrderInfo local_info;");
        // if self.doWorkoadSizeTracking:
        code.Add(
            "Themis::WorkloadTracking::InitLocalWorkloadSizeAtZeroLvl(inodes_"
            "cnts, local_info, global_stats_per_lvl);");
        code.Add("unsigned interval = 1;");
        code.Add("unsigned loop = {self.maxInterval} - 1;");
    }
    else {
        code.Add(
            "Themis::PullINodesAtZeroLvlStatically(tid, ts_0_range_cached, "
            "inodes_cnts);");
    }

    code.Add(
        "unsigned mask_32 = 0; // a bit mask to indicate levels where more "
        "than 32 INodes exist");
    code.Add(
        "unsigned mask_1 = 0; // a bit mask to indicate levels where INodes "
        "exist");
    code.Add("int lvl = -1;");
    code.Add(
        "Themis::UpdateMaskAtZeroLvl(0, tid, ts_0_range_cached, mask_32, "
        "mask_1);");
    code.Add("Themis::ChooseLvl(tid, mask_32, mask_1, lvl);");

    // Generate the main scan loop that will contain all operators
    if (do_inter_warp_lb) {
        code.Add("do {");
        code.IncreaseNesting();
        GenerateMainScanLoop(pipeline, code);
        GenerateCodeForAdaptiveWorkSharing(pipeline, code);
        code.DecreaseNesting();
        code.Add("} while (true); // while loop");
    }
    else {
        GenerateMainScanLoop(pipeline, code);
    }

    code.DecreaseNesting();
    code.Add("}");

    generated_gpu_code = code.str();
}

void GpuCodeGenerator::GenerateHostCode(CypherPipeline &pipeline)
{
    CodeBuilder code;

    code.Add("#include <cuda.h>");
    code.Add("#include <cuda_runtime.h>");
    code.Add("#include <cstdint>");
    code.Add("#include <vector>");
    code.Add("#include <iostream>");
    code.Add("#include <string>");
    code.Add("#include <unordered_map>\n");

    code.Add("extern \"C\" CUfunction gpu_kernel;\n");

    // Define structure for pointer mapping
    code.Add("struct PointerMapping {");
    code.IncreaseNesting();
    code.Add("std::string name;");
    code.Add("void *address;");
    code.Add(
        "unsigned long long cid;  // Chunk ID for GPU chunk cache manager");
    code.DecreaseNesting();
    code.Add("};\n");

    code.Add(
        "extern \"C\" void execute_query(PointerMapping *ptr_mappings, "
        "int num_mappings) {");
    code.IncreaseNesting();
    code.Add("cudaError_t err;\n");

    // Generate variable declarations for each parameter
    int param_index = 0;
    for (const auto &p : input_kernel_params) {
        if (p.type.find('*') != std::string::npos) {
            // For pointer types, declare as void* and assign from ptr_mappings
            code.Add("void *" + p.name + " = ptr_mappings[" +
                     std::to_string(param_index) + "].address;");
            param_index++;
        }
        else {
            // For non-pointer types, declare as the actual type
            code.Add(p.type + " " + p.name + " = " + p.value + ";");
        }
    }
    for (const auto &p : output_kernel_params) {
        if (p.type.find('*') != std::string::npos) {
            code.Add("void *" + p.name + ";");
            code.Add("cudaMalloc(&" + p.name + ", 1024);");
        }
        else {
            // For non-pointer types, declare as the actual type
            code.Add(p.type + " " + p.name + " = " + p.value + ";");
        }
    }
    code.Add("");

    code.Add("const int blockSize = 128;");
    code.Add("const int gridSize  = 3280;");
    std::string args_line = "void *args[] = {";
    for (size_t i = 0; i < input_kernel_params.size(); ++i) {
        const auto &p = input_kernel_params[i];
        args_line += "&" + p.name;
        if (i + 1 < input_kernel_params.size())
            args_line += ", ";
    }
    if (output_kernel_params.size() > 0) {
        args_line += ", ";
        for (size_t i = 0; i < output_kernel_params.size(); ++i) {
            const auto &p = output_kernel_params[i];
            args_line += "&" + p.name;
            if (i + 1 < output_kernel_params.size())
                args_line += ", ";
        }
    }
    args_line += "};";
    code.Add(args_line + "\n");

    code.Add(
        "CUresult r = cuLaunchKernel(gpu_kernel, gridSize,1,1, "
        "blockSize,1,1, 0, 0, args, nullptr);");
    code.Add("if (r != CUDA_SUCCESS) {");
    code.IncreaseNesting();
    code.Add("const char *name = nullptr, *str = nullptr;");
    code.Add("cuGetErrorName(r, &name);");
    code.Add("cuGetErrorString(r, &str);");
    code.Add(
        "std::cerr << \"cuLaunchKernel failed: \" << (name?name:\"unknown\""
        ") << \" – \" << (str?str:\"unknown\""
        ") << std::endl;");
    code.Add("throw std::runtime_error(\"cuLaunchKernel failed\");");
    code.DecreaseNesting();
    code.Add("}");
    code.Add("cudaError_t errSync = cudaDeviceSynchronize();");
    code.Add("if (errSync != cudaSuccess) {");
    code.IncreaseNesting();
    code.Add(
        "std::cerr << \"sync error: \" << cudaGetErrorString(errSync) << "
        "std::endl;");
    code.Add("throw std::runtime_error(\"cudaDeviceSynchronize failed\");");
    code.DecreaseNesting();
    code.Add("}");

    code.Add("std::cout << \"Query finished on GPU.\" << std::endl;");
    code.DecreaseNesting();
    code.Add("}");

    generated_cpu_code = code.str();
}

bool GpuCodeGenerator::CompileGeneratedCode()
{
    SCOPED_TIMER_SIMPLE(CompileGeneratedCode, spdlog::level::info,
                        spdlog::level::info);
    if (generated_gpu_code.empty() || generated_cpu_code.empty())
        return false;

    // Compile the generated GPU code using nvrtc
    SUBTIMER_START(CompileGeneratedCode, "CompileWithNVRTC");
    auto success = jit_compiler->CompileWithNVRTC(
        generated_gpu_code, "gpu_kernel", gpu_module, kernel_function);
    SUBTIMER_STOP(CompileGeneratedCode, "CompileWithNVRTC");

    // Check if the GPU code compilation was successful
    if (!success)
        return false;

    // Compile the generated CPU code using ORC JIT
    SUBTIMER_START(CompileGeneratedCode, "CompileWithORCLLJIT");
    auto success_orc =
        jit_compiler->CompileWithORCLLJIT(generated_cpu_code, kernel_function);
    SUBTIMER_STOP(CompileGeneratedCode, "CompileWithORCLLJIT");

    // Check if the CPU code compilation was successful
    if (!success_orc)
        return false;

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
    input_kernel_params.clear();
    output_kernel_params.clear();

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

        auto *column_names = property_schema_cat_entry->GetKeys();

        if (property_schema_cat_entry) {
            // Get extent IDs from the property schema
            for (size_t extent_idx = 0;
                 extent_idx < property_schema_cat_entry->extent_ids.size();
                 extent_idx++) {
                idx_t extent_id =
                    property_schema_cat_entry->extent_ids[extent_idx];

                // Generate table name
                std::string table_name = "gr" + std::to_string(oid) + "_ext" +
                                         std::to_string(extent_id);

                // Get extent catalog entry to access chunks (columns)
                ExtentCatalogEntry *extent_cat_entry =
                    (ExtentCatalogEntry *)catalog.GetEntry(
                        context, CatalogType::EXTENT_ENTRY, DEFAULT_SCHEMA,
                        DEFAULT_EXTENT_PREFIX + std::to_string(extent_id));

                if (extent_cat_entry) {
                    uint64_t num_tuples_in_extent =
                        extent_cat_entry->GetNumTuplesInExtent();

                    // Each chunk in the extent represents a column
                    for (size_t chunk_idx = 0;
                         chunk_idx <
                         scan_op->scan_projection_mapping[oid_idx].size();
                         chunk_idx++) {
                        auto column_idx =
                            scan_op
                                ->scan_projection_mapping[oid_idx][chunk_idx];
                        if (column_idx == 0)
                            continue;  // _id column
                        ChunkDefinitionID cdf_id =
                            extent_cat_entry->chunks[column_idx - 1];

                        // Generate column name based on chunk index
                        std::string col_name =
                            "col_" + std::to_string(column_idx - 1);

                        // Generate parameter names based on verbose mode
                        std::string param_name;
                        param_name = table_name + "_" + col_name;

                        // Add data buffer parameter for this column (chunk)
                        KernelParam data_param;
                        data_param.name = param_name + "_data";
                        data_param.type = "void *";
                        data_param.is_device_ptr = true;
                        input_kernel_params.push_back(data_param);

                        // Add pointer mapping for this chunk (column)
                        std::string chunk_name =
                            "chunk_" + std::to_string(cdf_id);
                        AddPointerMapping(chunk_name, nullptr, cdf_id);

                        pipeline_context
                            .column_to_param_mapping[column_names->at(
                                column_idx - 1)] = param_name;
                    }

                    // Add count parameter for this table
                    KernelParam count_param;
                    count_param.name = table_name + "_count";
                    count_param.type = "unsigned long long ";
                    count_param.value = std::to_string(num_tuples_in_extent);
                    count_param.is_device_ptr = false;
                    input_kernel_params.push_back(count_param);
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
        output_kernel_params.push_back(output_count_param);

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
                output_param_name =
                    short_output_name + "_" + std::to_string(col_idx);
            }

            // Add output data buffer parameter
            KernelParam output_data_param;
            output_data_param.name = output_param_name + "_data";
            output_data_param.type =
                "void *";  // Always void* for CUDA compatibility
            output_data_param.is_device_ptr = true;
            output_kernel_params.push_back(output_data_param);
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
            return "char";
        case LogicalTypeId::SMALLINT:
            return "short";
        case LogicalTypeId::INTEGER:
            return "int";
        case LogicalTypeId::BIGINT:
            return "long long";
        case LogicalTypeId::ID:
        case LogicalTypeId::UBIGINT:
            return "unsigned long long";
        case LogicalTypeId::UTINYINT:
            return "unsigned char";
        case LogicalTypeId::USMALLINT:
            return "unsigned short";
        case LogicalTypeId::UINTEGER:
            return "unsigned int";
        case LogicalTypeId::FLOAT:
            return "float";
        case LogicalTypeId::DOUBLE:
            return "double";
        case LogicalTypeId::VARCHAR:
            return "char*";
        case LogicalTypeId::DATE:
            return "int";
        case LogicalTypeId::TIME:
            return "int";
        case LogicalTypeId::TIMESTAMP:
            return "long long";
        case LogicalTypeId::INTERVAL:
            return "long long";
        case LogicalTypeId::UUID:
            return "long long";
        default:
            throw std::runtime_error("Unsupported logical type: " +
                                     std::to_string((uint8_t)type_id));
    }
}

void GpuCodeGenerator::GenerateInputCode(CypherPhysicalOperator *op,
                                         CodeBuilder &code,
                                         PipelineContext &pipeline_ctx,
                                         PipeInputType input_type)
{
    switch (input_type) {
        case PipeInputType::TYPE_0:
            GenerateInputCodeForType0(op, code, pipeline_ctx);
            break;
        case PipeInputType::TYPE_1:
            GenerateInputCodeForType1(op, code, pipeline_ctx);
            break;
        case PipeInputType::TYPE_2:
            GenerateInputCodeForType2(op, code, pipeline_ctx);
            break;
        default:
            throw std::runtime_error("Unsupported input type");
    }
}

void GpuCodeGenerator::GenerateInputCodeForType0(CypherPhysicalOperator *op,
                                                 CodeBuilder &code,
                                                 PipelineContext &pipeline_ctx)
{
    int stepSize = KernelConstants::DEFAULT_BLOCK_SIZE *
                   KernelConstants::DEFAULT_GRID_SIZE;
    // tid = spSeq.getTid()
    std::string tid_name =
        "tid_" + std::to_string(pipeline_ctx.sub_pipelines[0].GetPipelineId());

    code.Add("while (lvl >= 0 && loop < " +
             std::to_string(kernel_args.inter_warp_lb_interval) +
             ") {");

    code.IncreaseNesting();

    code.Add("__syncwarp();");
    code.Add("if (lvl == 0) {");
    code.IncreaseNesting();
    code.Add("int " + tid_name + ";");
    code.Add("Themis::FillIPartAtZeroLvl(lvl, thread_id, active, " + tid_name +
             ", ts_0_range_cached, mask_32, mask_1, " +
             std::to_string(stepSize) + ");");

    // if self.doInterWarpLB and self.interWarpLbMethod == 'aws' and self.doWorkoadSizeTracking:
    code.Add(
        "Themis::WorkloadTracking::UpdateWorkloadSizeAtZeroLvl(thread_id, "
        "++loop, local_info, global_stats_per_lvl);");
}
void GpuCodeGenerator::GenerateInputCodeForType1(CypherPhysicalOperator *op,
                                                 CodeBuilder &code,
                                                 PipelineContext &pipeline_ctx)
{
    // tid = spSeq.getTid()
    // lastOp = spSeq.pipe.subpipeSeqs[spSeq.id-1].subpipes[-1].operators[-1]
    // c = Code()
    // if self.doInterWarpLB and self.interWarpLbMethod == 'aws':
    //     c.add(f'while (lvl >= {spSeq.id} && loop < {self.maxInterval}) ' + '{')
    // else:
    //     c.add(f'while (lvl >= {spSeq.id}) ' + '{')
    // c.add('__syncwarp();')
    // c.add(f'if (lvl == {spSeq.id}) ' + '{')
    // c.add(f'int loopvar{spSeq.id};')
    // c.add(f'Themis::FillIPartAtLoopLvl({spSeq.id}, thread_id, active, loopvar{spSeq.id}, ts_{spSeq.id}_range_cached, mask_32, mask_1);')
    
    // # Declare in-boundary attrs
    // c.add(f'//{list(attrsToDeclareAndMaterialize)}')
    // c.add(f'int {tid.id_name};')
    // for attrId, attr in spSeq.inBoundaryAttrs.items():
    //     if attrId == tid.id: continue
    //     c.add(f'{langType(attr.dataType)} {attr.id_name};')
        
    
    // _, attrsToLoadBeforeExec, _, _ = attrsToDeclareAndMaterialize[0]
    // c.add(self.genAttrDeclaration(spSeq.pipe, attrsToLoadBeforeExec))
    
    // c.add('if (active) {')

    // # Load in-boundary attrs
    // c.add(f'// last op generating Attrs: {lastOp.generatingAttrs}')
    // var = f'loopvar{spSeq.id}'
    // c.add(f'{tid.id_name} = {spSeq.convertTid(var)};')
    // for attrId, attr in spSeq.inBoundaryAttrs.items():
    //     if attrId == tid.id: continue
    //     if attrId in lastOp.generatingAttrs:
    //         c.add(f'{attr.id_name} = {spSeq.pipe.originExpr[attrId]};')
    //     else:
    //         c.add(f'{attr.id_name} = ts_{spSeq.id}_{attr.id_name}_cached;')
        
    // c.add(self.genAttrLoad(spSeq.pipe, attrsToLoadBeforeExec))

    // c.add('}')
    // c.add('int ts_src = 32;')
    // c.add(f'bool is_updated = Themis::DistributeFromPartToDPart(thread_id, {spSeq.id}, ts_src, ts_{spSeq.id}_range, ts_{spSeq.id}_range_cached, mask_32, mask_1);')
    // if self.doInterWarpLB and self.interWarpLbMethod == 'aws' and self.doWorkoadSizeTracking:
    //     c.add(f'Themis::WorkloadTracking::UpdateWorkloadSizeAtLoopLvl(thread_id, {spSeq.id}, ++loop, ts_{spSeq.id}_range, ts_{spSeq.id}_range_cached, mask_1, local_info, global_stats_per_lvl);')
    // c.add('if (is_updated) {')
    // for attrId, attr in spSeq.inBoundaryAttrs.items():
    //     if attrId == tid.id: continue
    //     if attrId in lastOp.generatingAttrs: continue
    //     c.add('{')
    //     name = f'ts_{spSeq.id}_{attr.id_name}'
    //     if attr.dataType == Type.STRING:
    //         c.add(f'char* start = (char*) __shfl_sync(ALL_LANES, (uint64_t) {name}.start, ts_src);')
    //         c.add(f'char* end = (char*) __shfl_sync(ALL_LANES, (uint64_t) {name}.end, ts_src);')
    //         c.add('if (ts_src < 32) {')
    //         c.add(f'{name}_cached.start = start;')
    //         c.add(f'{name}_cached.end = end;')
    //         c.add('}')
    //     elif attr.dataType == Type.PTR_INT:
    //         c.add(f'uint64_t cache = __shfl_sync(ALL_LANES, (uint64_t){name}, ts_src);')
    //         c.add(f'if (ts_src < 32) {name}_cached = (int*) cache;')
    //     else:
    //         c.add(f'{langType(attr.dataType)} cache = __shfl_sync(ALL_LANES, {name}, ts_src);')
    //         c.add(f'if (ts_src < 32) {name}_cached = cache;')
    //     c.add('}')
    // c.add('}')
    // return c
}
void GpuCodeGenerator::GenerateInputCodeForType2(CypherPhysicalOperator *op,
                                                 CodeBuilder &code,
                                                 PipelineContext &pipeline_ctx)
{
    // c.add(f'if (lvl == {spSeq.id}) ' + '{')
    // if len(spSeq.inBoundaryAttrs) > 0:
    //     c.add(f'if (!(mask_32 & (0x1u << {spSeq.id}))) ' + '{')
    //     for attrId, attr in spSeq.inBoundaryAttrs.items():
    //         c.add(f'ts_{spSeq.id}_{attr.id_name} = ts_{spSeq.id}_{attr.id_name}_flushed;')    
    //     c.add('}')
    // c.add(f'Themis::FillIPartAtIfLvl({spSeq.id}, thread_id, inodes_cnts, active, mask_32, mask_1);')
    // if self.doInterWarpLB and self.interWarpLbMethod == 'aws' and self.doWorkoadSizeTracking:
    //     c.add(f'Themis::WorkloadTracking::UpdateWorkloadSizeAtIfLvl(thread_id, {spSeq.id}, loop, inodes_cnts, mask_1, local_info, global_stats_per_lvl);')
    
    // for attrId, attr in spSeq.inBoundaryAttrs.items():
    //     c.add(f'{langType(attr.dataType)} {attr.id_name};')
    // c.add('if (active) {')
    // for attrId, attr in spSeq.inBoundaryAttrs.items():
    //     c.add(f'{attr.id_name} = ts_{spSeq.id}_{attr.id_name};')
    // c.add('}')
}

void GpuCodeGenerator::GenerateMainScanLoop(CypherPipeline &pipeline,
                                            CodeBuilder &code)
{
    // Initialize pipeline context once
    InitializePipelineContext(pipeline);

    auto first_op = pipeline.GetSource();
    if (first_op->GetOperatorType() == PhysicalOperatorType::NODE_SCAN) {
        MoveToOperator(0);
        ExtractOutputSchema(first_op);
        AnalyzeOperatorDependencies(first_op);
        GenerateOperatorCode(first_op, code, pipeline_context,
                             /*is_main_loop=*/true);

        ProcessRemainingOperators(pipeline, 1, code);

        code.DecreaseNesting();
        code.Add("}");
    }
}

void GpuCodeGenerator::ProcessRemainingOperators(CypherPipeline &pipeline,
                                                 int op_idx, CodeBuilder &code)
{
    if (op_idx >= pipeline.GetPipelineLength()) {
        return;
    }

    auto op = pipeline.GetIdxOperator(op_idx);

    // Move to current operator and update schemas
    MoveToOperator(op_idx);
    ExtractOutputSchema(op);
    AnalyzeOperatorDependencies(op);

    switch (op->GetOperatorType()) {
        case PhysicalOperatorType::FILTER:
            code.Add("if (condition) {");
            code.IncreaseNesting();
            GenerateOperatorCode(op, code, pipeline_context,
                                 /*is_main_loop=*/false);
            ProcessRemainingOperators(pipeline, op_idx + 1, code);
            code.DecreaseNesting();
            code.Add("}");
            break;

            // case PhysicalOperatorType::JOIN:
            //     GenerateOperatorCode(op, code, /*is_main_loop=*/false);
            //     ProcessRemainingOperators(pipeline, op_idx + 1, code);
            //     break;

        case PhysicalOperatorType::PROJECTION:
            GenerateOperatorCode(op, code, pipeline_context,
                                 /*is_main_loop=*/false);
            ProcessRemainingOperators(pipeline, op_idx + 1, code);
            break;

        case PhysicalOperatorType::PRODUCE_RESULTS:
            GenerateOperatorCode(op, code, pipeline_context,
                                 /*is_main_loop=*/false);
            ProcessRemainingOperators(pipeline, op_idx + 1, code);
            break;

        default:
            GenerateOperatorCode(op, code, pipeline_context,
                                 /*is_main_loop=*/false);
            ProcessRemainingOperators(pipeline, op_idx + 1, code);
            break;
    }
}

void GpuCodeGenerator::GenerateOperatorCode(CypherPhysicalOperator *op,
                                            CodeBuilder &code,
                                            PipelineContext &pipeline_ctx,
                                            bool is_main_loop)
{
    auto it = operator_generators.find(op->GetOperatorType());
    if (it != operator_generators.end()) {
        it->second->GenerateCode(op, code, this, context, pipeline_ctx,
                                 is_main_loop);
    }
    else {
        // Default handling for unknown operators
        code.Add("// Unknown operator type: " +
                 std::to_string(static_cast<int>(op->GetOperatorType())));
    }
}

void NodeScanCodeGenerator::GenerateCode(
    CypherPhysicalOperator *op, CodeBuilder &code, GpuCodeGenerator *code_gen,
    ClientContext &context, PipelineContext &pipeline_ctx, bool is_main_loop)
{
    auto scan_op = dynamic_cast<PhysicalNodeScan *>(op);
    if (!scan_op)
        return;

    if (is_main_loop) {
        code.Add("// Scan operator");

        // Process oids and scan_projection_mapping to get chunk IDs
        for (size_t oid_idx = 0; oid_idx < scan_op->oids.size(); oid_idx++) {
            idx_t oid = scan_op->oids[oid_idx];

            // Get property schema catalog entry using oid
            Catalog &catalog = context.db->GetCatalog();
            PropertySchemaCatalogEntry *property_schema_cat_entry =
                (PropertySchemaCatalogEntry *)catalog.GetEntry(
                    context, DEFAULT_SCHEMA, oid);
            D_ASSERT(property_schema_cat_entry != nullptr);

            auto *property_types_id = property_schema_cat_entry->GetTypes();

            // Get extent IDs from the property schema
            for (size_t extent_idx = 0;
                 extent_idx < property_schema_cat_entry->extent_ids.size();
                 extent_idx++) {
                idx_t extent_id =
                    property_schema_cat_entry->extent_ids[extent_idx];

                // Generate table name
                std::string table_name = "gr" + std::to_string(oid) + "_ext" +
                                         std::to_string(extent_id);

                // Get extent catalog entry to access chunks (columns)
                ExtentCatalogEntry *extent_cat_entry =
                    (ExtentCatalogEntry *)catalog.GetEntry(
                        context, CatalogType::EXTENT_ENTRY, DEFAULT_SCHEMA,
                        DEFAULT_EXTENT_PREFIX + std::to_string(extent_id));
                D_ASSERT(extent_cat_entry != nullptr);

                uint64_t tuple_id_base = extent_id;
                tuple_id_base <<= 32;
                // Generate scan loop for this extent
                std::string count_param_name = table_name + "_count";
                code.Add("// Process extent " + std::to_string(extent_id) +
                         " (property " + std::to_string(oid) + ")");

                code_gen->GenerateInputCode(
                    op, code, pipeline_ctx,
                    PipeInputType::TYPE_0);  // TYPE_0 for node scan

                // lazy materialization
                code.Add("// lazy materialization");
                code.Add("unsigned long long tuple_id_base = " +
                         std::to_string(tuple_id_base) + ";");
                code.Add(
                    "unsigned long long tuple_id = tuple_id_base "
                    "+ tid;");

                // Declare input column pointers for lazy materialization
                code.Add("// Declare input column pointers");

                for (size_t chunk_idx = 0;
                     chunk_idx <
                     scan_op->scan_projection_mapping[oid_idx].size();
                     chunk_idx++) {

                    auto column_idx =
                        scan_op->scan_projection_mapping[oid_idx][chunk_idx];
                    if (column_idx == 0)
                        continue;  // _id column
                    // type extract
                    LogicalTypeId type_id =
                        (LogicalTypeId)property_types_id->at(column_idx - 1);
                    std::string ctype =
                        code_gen->ConvertLogicalTypeToPrimitiveType(type_id);

                    // Generate column name based on chunk index
                    std::string col_name =
                        "col_" + std::to_string(column_idx - 1);

                    // Generate parameter names based on verbose mode
                    std::string param_name;
                    param_name = table_name + "_" + col_name;

                    // Declare the pointer
                    code.Add(ctype + "* " + param_name + "_ptr = static_cast<" +
                             ctype + "*>(" + param_name + "_data);");

                    // Add to lazy materialization tracking
                    pipeline_ctx.input_column_names.push_back(param_name);
                    pipeline_ctx.column_materialized[param_name] = false;
                }

                if (scan_op->is_filter_pushdowned) {
                    std::string predicate_string = "";
                    // Generate predicate string for filter pushdown
                    if (scan_op->filter_pushdown_type ==
                        FilterPushdownType::FP_EQ) {
                        for (auto i = 0;
                             i < scan_op->filter_pushdown_key_idxs.size();
                             i++) {
                            if (i > 0) {
                                predicate_string += " && ";
                            }
                            auto key_idx = scan_op->filter_pushdown_key_idxs[i];
                            std::string attr_name =
                                table_name + "_col_" +
                                std::to_string(key_idx - 1) + "_ptr";
                            if (key_idx >= 0) {
                                auto value_str =
                                    scan_op->eq_filter_pushdown_values[i]
                                        .ToString();
                                predicate_string +=
                                    (attr_name + "[tid] == " + value_str);
                            }
                        }
                    }
                    else if (scan_op->filter_pushdown_type ==
                             FilterPushdownType::FP_RANGE) {
                        for (auto i = 0;
                             i < scan_op->filter_pushdown_key_idxs.size();
                             i++) {
                            if (i > 0) {
                                predicate_string += " && ";
                            }
                            auto key_idx = scan_op->filter_pushdown_key_idxs[i];
                            std::string attr_name =
                                table_name + "_col_" +
                                std::to_string(key_idx - 1) + "_ptr";
                            if (key_idx >= 0) {
                                auto left_value_str =
                                    scan_op->range_filter_pushdown_values[i]
                                        .l_value.ToString();
                                auto right_value_str =
                                    scan_op->range_filter_pushdown_values[i]
                                        .r_value.ToString();
                                predicate_string +=
                                    scan_op->range_filter_pushdown_values[i]
                                            .l_inclusive
                                        ? (attr_name +
                                           "[tid] >= " + left_value_str)
                                        : (attr_name + "[tid] > " +
                                           left_value_str);
                                predicate_string += " && ";
                                predicate_string +=
                                    scan_op->range_filter_pushdown_values[i]
                                            .r_inclusive
                                        ? (attr_name +
                                           "[tid] <= " + right_value_str)
                                        : (attr_name + "[tid] < " +
                                           right_value_str);
                            }
                        }
                    }
                    else {  // FP_COMPLEX
                        ExpressionCodeGenerator expr_gen(context);
                        predicate_string = expr_gen.GenerateConditionCode(
                            scan_op->filter_expression.get(), code,
                            pipeline_ctx);
                    }

                    code.Add("active = (" + predicate_string + ");");
                }
            }
        }
    }
    else {
        code.Add("// Additional scan logic (if needed)");
    }
}

void ProjectionCodeGenerator::GenerateCode(
    CypherPhysicalOperator *op, CodeBuilder &code, GpuCodeGenerator *code_gen,
    ClientContext &context, PipelineContext &pipeline_ctx, bool is_main_loop)
{
    auto proj_op = dynamic_cast<PhysicalProjection *>(op);
    if (!proj_op) {
        return;
    }

    code.Add("// Projection operator");

    // Process each projection expression
    for (size_t expr_idx = 0; expr_idx < proj_op->expressions.size();
         expr_idx++) {
        auto &expr = proj_op->expressions[expr_idx];
        // Generate projection code with pipeline context
        GenerateProjectionExpressionCode(expr.get(), expr_idx, code, code_gen,
                                         context, pipeline_ctx);

        // Mark the output column as materialized
        if (expr_idx < pipeline_ctx.output_column_names.size()) {
            std::string output_col_name =
                pipeline_ctx.output_column_names[expr_idx];
            pipeline_ctx.column_materialized[output_col_name] = true;
        }
    }
}

void ProjectionCodeGenerator::GenerateProjectionExpressionCode(
    Expression *expr, size_t expr_idx, CodeBuilder &code,
    GpuCodeGenerator *code_gen, ClientContext &context,
    PipelineContext &pipeline_ctx)
{
    if (!expr)
        return;

    std::string output_var = "proj_result_" + std::to_string(expr_idx);

    switch (expr->expression_class) {
        case ExpressionClass::BOUND_REF: {
            // Handle reference expression with pipeline context
            // auto ref_expr = dynamic_cast<BoundReferenceExpression *>(expr);
            // if (ref_expr &&
            //     ref_expr->index < pipeline_ctx.input_column_names.size()) {
            //     std::string input_col_name =
            //         pipeline_ctx.input_column_names[ref_expr->index];

            //     // Check if this column is already materialized
            //     bool is_materialized =
            //         pipeline_ctx.column_materialized.find(input_col_name) !=
            //             pipeline_ctx.column_materialized.end() &&
            //         pipeline_ctx.column_materialized[input_col_name];

            //     if (!is_materialized) {
            //         // Generate lazy loading code
            //         code.Add(nesting_level,
            //                  "// Lazy load input column: " + input_col_name);
            //         code.Add(nesting_level,
            //                  "if (!" + input_col_name + "_loaded) {");
            //         code.Add(nesting_level + 1,
            //                  input_col_name + "_ptr = static_cast<uint64_t*>(" +
            //                      input_col_name + "_data);");
            //         code.Add(nesting_level + 1,
            //                  input_col_name + "_loaded = true;");
            //         code.Add("}");

            //         // Mark as materialized
            //         pipeline_ctx.column_materialized[input_col_name] = true;
            //     }

            //     // Create column mapping for output
            //     if (expr_idx < pipeline_ctx.output_column_names.size()) {
            //         std::string output_col_name =
            //             pipeline_ctx.output_column_names[expr_idx];
            //         pipeline_ctx.column_mapping[output_col_name] =
            //             input_col_name;
            //     }
            // }
            break;
        }
        case ExpressionClass::BOUND_CONSTANT: {
            // Constant expression
            auto const_expr = dynamic_cast<BoundConstantExpression *>(expr);
            if (const_expr) {
                code.Add("// Constant value");
                code.Add(ConvertLogicalTypeToCUDAType(expr->return_type) + " " +
                         output_var + " = " +
                         ConvertValueToCUDALiteral(const_expr->value) + ";");
            }
            break;
        }
        case ExpressionClass::BOUND_FUNCTION: {
            // Function expression
            auto func_expr = dynamic_cast<BoundFunctionExpression *>(expr);
            if (func_expr) {
                code.Add("// Function call: " + func_expr->function.name);
                GenerateFunctionCallCode(func_expr, output_var, code, code_gen,
                                         context, pipeline_ctx);
            }
            break;
        }
        case ExpressionClass::BOUND_OPERATOR: {
            // Operator expression
            auto op_expr = dynamic_cast<BoundOperatorExpression *>(expr);
            if (op_expr) {
                code.Add("// Operator: " +
                         ExpressionTypeToString(op_expr->type));
                GenerateOperatorCode(op_expr, output_var, code, code_gen,
                                     context, pipeline_ctx);
            }
            break;
        }
        default:
            // Default case - just assign a placeholder
            code.Add("// Unsupported expression type");
            code.Add(ConvertLogicalTypeToCUDAType(expr->return_type) + " " +
                     output_var + " = 0; // TODO: implement");
            break;
    }

    // Store the result in output buffer, except for BOUND_REF
    if (expr->expression_class != ExpressionClass::BOUND_REF) {
        code.Add("// Store projection result");
        code.Add("output_col_" + std::to_string(expr_idx) +
                 "[tid] = " + output_var + ";");
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
    PipelineContext &pipeline_ctx)
{
    // For now, we'll implement a simple approach
    // In the future, this should be expanded to handle different function types

    if (func_expr->children.size() == 1) {
        // Unary function
        code.Add(ConvertLogicalTypeToCUDAType(func_expr->return_type) + " " +
                 output_var + " = ");

        // Generate lazy loading for the input
        auto child_expr = func_expr->children[0].get();
        if (child_expr->expression_class == ExpressionClass::BOUND_REF) {
            auto ref_expr =
                dynamic_cast<BoundReferenceExpression *>(child_expr);
            if (ref_expr) {
                std::string input_col_name =
                    "input_col_" + std::to_string(ref_expr->index);
                code.Add("// Lazy load input column if needed");
                code.Add("if (!" + input_col_name + "_loaded) {");
                code.IncreaseNesting();
                code.Add(input_col_name + "_ptr = static_cast<uint64_t*>(" +
                         input_col_name + "_data);");
                code.Add(input_col_name + "_loaded = true;");
                code.DecreaseNesting();
                code.Add("}");

                if (func_expr->function.name == "abs") {
                    code.Add(output_var + " = abs(" + input_col_name +
                             "_ptr[i]);");
                }
                else if (func_expr->function.name == "sqrt") {
                    code.Add(output_var + " = sqrt(" + input_col_name +
                             "_ptr[i]);");
                }
                else {
                    code.Add(output_var + " = " + input_col_name +
                             "_ptr[i]; // TODO: implement function " +
                             func_expr->function.name);
                }
            }
        }
        else {
            code.Add("input_col_0; // TODO: get actual input column");
        }
    }
    else {
        // Multi-argument function
        code.Add(ConvertLogicalTypeToCUDAType(func_expr->return_type) + " " +
                 output_var + " = 0; // TODO: implement multi-arg function");
    }
}

void ProjectionCodeGenerator::GenerateOperatorCode(
    BoundOperatorExpression *op_expr, const std::string &output_var,
    CodeBuilder &code, GpuCodeGenerator *code_gen, ClientContext &context,
    PipelineContext &pipeline_ctx)
{
    if (op_expr->children.size() == 2) {
        // Binary operator
        code.Add(ConvertLogicalTypeToCUDAType(op_expr->return_type) + " " +
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
                code.Add("// Lazy load left operand if needed");
                code.Add("if (!" + input_col_name + "_loaded) {");
                code.IncreaseNesting();
                code.Add(input_col_name + "_ptr = static_cast<uint64_t*>(" +
                         input_col_name + "_data);");
                code.Add(input_col_name + "_loaded = true;");
                code.DecreaseNesting();
                code.Add("}");
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
                code.Add("// Lazy load right operand if needed");
                code.Add("if (!" + input_col_name + "_loaded) {");
                code.IncreaseNesting();
                code.Add(input_col_name + "_ptr = static_cast<uint64_t*>(" +
                         input_col_name + "_data);");
                code.Add(input_col_name + "_loaded = true;");
                code.DecreaseNesting();
                code.Add("}");
            }
        }
        else {
            right_operand = "0";  // TODO: handle other expression types
        }

        code.Add(output_var + " = " + left_operand + " " +
                 ExpressionTypeToString(op_expr->type) + " " + right_operand +
                 ";");
    }
    else {
        // Unary or other operator
        code.Add(ConvertLogicalTypeToCUDAType(op_expr->return_type) + " " +
                 output_var + " = 0; // TODO: implement operator");
    }
}

void ProduceResultsCodeGenerator::GenerateCode(
    CypherPhysicalOperator *op, CodeBuilder &code, GpuCodeGenerator *code_gen,
    ClientContext &context, PipelineContext &pipeline_ctx, bool is_main_loop)
{
    auto results_op = dynamic_cast<PhysicalProduceResults *>(op);
    if (!results_op) {
        return;
    }

    code.Add("// Produce results operator");

    // Declare output count pointer
    code.Add("int* output_count_ptr = static_cast<int*>(&output_count);");

    // Get the output schema to determine what columns to write
    auto &output_schema = results_op->GetSchema();
    auto &output_column_names = output_schema.getStoredColumnNamesRef();

    code.Add("// Write results to output buffers");
    for (size_t col_idx = 0; col_idx < output_column_names.size(); col_idx++) {
        std::string orig_col_name = output_column_names[col_idx];
        std::string col_name = orig_col_name;
        // Replace '.' with '_' for valid C/C++ variable names
        std::replace(col_name.begin(), col_name.end(), '.', '_');

        // type extract
        LogicalTypeId type_id = pipeline_ctx.output_column_types[col_idx].id();
        std::string ctype =
            code_gen->ConvertLogicalTypeToPrimitiveType(type_id);

        // Check if this column is materialized in the pipeline context
        bool is_materialized =
            pipeline_ctx.column_materialized.find(col_name) !=
                pipeline_ctx.column_materialized.end() &&
            pipeline_ctx.column_materialized[col_name];

        std::string output_param_name;
        if (code_gen->GetVerboseMode()) {
            output_param_name = "output_" + col_name;
        }
        else {
            output_param_name = "out_" + std::to_string(col_idx);
        }
        std::string output_data_name = output_param_name + "_data";
        std::string output_ptr_name = output_param_name + "_ptr";

        code.Add("// Write column " + std::to_string(col_idx) + " (" +
                 col_name + ") to output");

        if (is_materialized) {
            // Column is already materialized, use the projection result
            code.Add(ctype + "* " + output_ptr_name + " = static_cast<" +
                     ctype + "*>(" + output_data_name + ");");
            code.Add(output_ptr_name + "[tid] = proj_result_" +
                     std::to_string(col_idx) + ";");
        }
        else {
            // Column needs to be materialized from input
            // Find the corresponding input column

            std::string orig_col_name_wo_varname =
                orig_col_name.substr(orig_col_name.find_last_of('.') + 1);
            if (orig_col_name_wo_varname == "_id") {
                code.Add("// Special case for _id column, use tuple_id");
                code.Add(ctype + "* " + output_ptr_name + " = static_cast<" +
                         ctype + "*>(" + output_data_name + ");");
                code.Add(output_ptr_name + "[tid] = " + "tuple_id;");
            }
            else {
                auto it = pipeline_ctx.column_to_param_mapping.find(
                    orig_col_name_wo_varname);
                if (it != pipeline_ctx.column_to_param_mapping.end()) {
                    std::string input_col_name = it->second;
                    code.Add("// Materialize column from input: " +
                             input_col_name);
                    code.Add(ctype + "* " + output_ptr_name +
                             " = static_cast<" + ctype + "*>(" +
                             output_data_name + ");");
                    // code.Add(ctype + "* " + input_col_name +
                    //                             "_ptr" + " = static_cast<" +
                    //                             ctype + "*>(" + input_col_name +
                    //                             ");");
                    code.Add(output_ptr_name + "[tid] = " + input_col_name +
                             "_ptr" + "[i];");
                }
                else {
                    // No mapping found, check if it's a direct input column
                    auto input_it = std::find(
                        pipeline_ctx.input_column_names.begin(),
                        pipeline_ctx.input_column_names.end(), orig_col_name);
                    if (input_it != pipeline_ctx.input_column_names.end()) {
                        // Direct input column
                        code.Add("// Direct input column: " + orig_col_name);
                        code.Add(ctype + "* " + output_ptr_name +
                                 " = static_cast<" + ctype + "*>(" +
                                 output_data_name + ");");
                        code.Add(output_ptr_name + "[tid] = " + col_name +
                                 "_ptr[i];");
                    }
                    else {
                        // Fallback: use projection result
                        code.Add(
                            "// No mapping found, using projection result");
                        code.Add(ctype + "* " + output_ptr_name +
                                 " = static_cast<" + ctype + "*>(" +
                                 output_data_name + ");");
                        code.Add(output_ptr_name + "[tid] = proj_result_" +
                                 std::to_string(col_idx) + ";");
                    }
                }
            }
        }
    }

    // Update output count
    code.Add("// Update output count atomically");
    // code.Add("atomicAdd(output_count_ptr, 1);");

    code.Add("// Results produced successfully");
}

void FilterCodeGenerator::GenerateCode(
    CypherPhysicalOperator *op, CodeBuilder &code, GpuCodeGenerator *code_gen,
    ClientContext &context, PipelineContext &pipeline_ctx, bool is_main_loop)
{
    auto filter_op = dynamic_cast<PhysicalFilter *>(op);
    if (!filter_op) {
        code.Add("// Invalid filter operator");
        return;
    }

    code.Add("// Filter operator - analyze expression");

    // Create expression code generator
    ExpressionCodeGenerator expr_gen(context);

    // Generate condition code from the filter expression
    std::string condition_var = "filter_condition";
    if (filter_op->expression) {
        condition_var = expr_gen.GenerateConditionCode(
            filter_op->expression.get(), code, pipeline_ctx);
    }
    else {
        // Fallback for missing expression
        code.Add("bool " + condition_var +
                 " = true; // No filter expression found");
    }

    code.Add("// Apply filter condition");
    code.Add("if (!" + condition_var + ") {");
    code.IncreaseNesting();
    code.Add("continue; // Skip this tuple if condition is false");
    code.DecreaseNesting();
    code.Add("}");
    code.Add("// Filter condition passed, continue processing");
}

void GpuCodeGenerator::GenerateCodeForAdaptiveWorkSharing(
    CypherPipeline &pipeline, CodeBuilder &code)
{
    code.Add("loop = 0;");
    code.Add("if (lvl == -1) {");
    code.IncreaseNesting();
    GenerateCodeForAdaptiveWorkSharingPull(pipeline, code);
    code.DecreaseNesting();
    code.Add("} else { // if (lvl == -1)");
    code.IncreaseNesting();
    GenerateCodeForAdaptiveWorkSharingPush(pipeline, code);
    code.DecreaseNesting();
    code.Add("} // end of adaptive work sharing ");
}

void GpuCodeGenerator::GenerateCodeForAdaptiveWorkSharingPull(
    CypherPipeline &pipeline, CodeBuilder &code)
{
    int num_warps = int(KernelConstants::DEFAULT_BLOCK_SIZE / 32) *
                    KernelConstants::DEFAULT_GRID_SIZE;
    int min_num_warps = std::min(num_warps, kernel_args.min_num_warps);

    if (kernel_args.mode == "stats") {
        code.Add("unsigned long long current_tp = clock64();");
        code.Add(
            "if (current_status != -1) stat_counters[current_status] += "
            "current_tp - tp;");
        code.Add("tp = current_tp;");
        code.Add("current_status = TYPE_STATS_WAITING;");
        code.Add("stat_counters[TYPE_STATS_NUM_IDLE] += 1;");
    }

    code.Add(
        "if (thread_id == 0 && local_info.locally_lowest_lvl != -1) "
        "atomicSub(&global_stats_per_lvl[local_info.locally_lowest_lvl].num_"
        "warps, 1);");
    code.Add("inodes_cnts = 0;");
    code.Add("mask_32 = mask_1 = 0;");
    code.Add("unsigned int num_idle_warps = 0;");
    code.Add("int src_warp_id = -1;");
    code.Add("int lowest_lvl = -1;");
    code.Add("bool is_successful = Themis::PullINodesAtZeroLvlDynamically<" +
             std::to_string(num_warps) +
             ">(thread_id, global_scan_offset, global_scan_end, "
             "local_scan_offset, ts_0_range_cached, inodes_cnts);");
    code.Add("if (is_successful) {");
    code.IncreaseNesting();
    code.Add("loop = 0;");
    code.Add("lowest_lvl = 0;");
    code.DecreaseNesting();
    code.Add("} else { // adaptive work-sharing for all levels");
    code.IncreaseNesting();
    code.Add("Themis::Wait<" + std::to_string(num_warps) + ", " +
             std::to_string(min_num_warps) + ">(");
    code.IncreaseNesting();
    code.Add(
        "gpart_id, src_warp_id, warp_id, thread_id, lowest_lvl, warp_status, "
        "num_idle_warps, global_stats_per_lvl, gts, size_of_stack_per_warp");
    if (idleWarpDetectionType == "twolvlbitmaps") {
        code.Add(", global_bit1, global_bit2");
    }
    else if (idleWarpDetectionType == "idqueue") {
        code.Add(", global_id_stack");
    }
    code.DecreaseNesting();
    code.Add(");");

    // Termination code
    code.Add("if (src_warp_id == -2) {");
    code.IncreaseNesting();
    if (kernel_args.mode == "stats") {
        code.Add("stat_counters[TYPE_STATS_WAITING] += (clock64() - tp);");
    }
    code.Add(
        "if (blockIdx.x == 0 && threadIdx.x == 0) warp_status->terminate();");
    code.Add("break;");
    code.DecreaseNesting();
    code.Add("}");

    code.Add(
        "Themis::PushedParts::PushedPartsStack* stack = "
        "Themis::PushedParts::GetIthStack(gts, size_of_stack_per_warp, "
        "(size_t) src_warp_id);");

    GenerateCopyCodeForAdaptiveWorkSharingPull(pipeline, code);
    code.Add("if (thread_id == 0) {");
    code.IncreaseNesting();
    code.Add("__threadfence();");
    code.Add("stack->FreeLock();");
    code.DecreaseNesting();
    code.Add("}");
    code.Add("loop = " + std::to_string(kernel_args.inter_warp_lb_interval) +
             " - 1;");
    code.DecreaseNesting();
    code.Add("} // ~ adaptive work-sharing for all levels");

    if (doWorkoadSizeTracking) {
        code.Add(
            "Themis::WorkloadTracking::InitLocalWorkloadSize(lowest_lvl, "
            "inodes_cnts, local_info, global_stats_per_lvl);");
    }
    code.Add("lvl = lowest_lvl;");
    code.Add(
        "if (thread_id == lvl) mask_32 = inodes_cnts >= 32 ? 0x1 << lvl : 0;");
    code.Add("mask_1 = 0x1 << lvl;");
    code.Add("mask_32 = __shfl_sync(ALL_LANES, mask_32, lvl);");
}

void GpuCodeGenerator::GenerateCopyCodeForAdaptiveWorkSharingPull(
    CypherPipeline &pipeline, CodeBuilder &code)
{
    code.Add("switch (lowest_lvl) {");
    // for spSeqId, spSeq in enumerate(pipe.subpipeSeqs):
    //     code.Add('case {spSeqId}: ' + '{');
    //     if spSeq.inputType == 0:
    //         code.Add('Themis::PushedParts::PushedPartsAtZeroLvl* src_pparts = (Themis::PushedParts::PushedPartsAtZeroLvl*) stack->Top();')
    //         code.Add('Themis::PullINodesFromPPartAtZeroLvl(thread_id, src_pparts, ts_0_range_cached, inodes_cnts);')
    //         code.Add('if (thread_id == 0) stack->PopPartsAtZeroLvl();')
    //     elif spSeq.inputType == 1:
    //         code.Add('Themis::PushedParts::PushedPartsAtLoopLvl* src_pparts = (Themis::PushedParts::PushedPartsAtLoopLvl*) stack->Top();')
    //         code.Add(f'Themis::PullINodesFromPPartAtLoopLvl(thread_id, {spSeqId}, src_pparts, ts_{spSeqId}_range_cached, ts_{spSeqId}_range, inodes_cnts);')

    //         attrs = {}
    //         tid = spSeq.getTid()
    //         lastOp = spSeq.pipe.subpipeSeqs[spSeq.id-1].subpipes[-1].operators[-1]
    //         for attrId, attr in spSeq.inBoundaryAttrs.items():
    //             if attrId == tid.id: continue
    //             if attrId in lastOp.generatingAttrs: continue
    //             attrs[attrId] = attr

    //         speculated_size = 0
    //         if len(attrs) > 0:
    //             code.Add('volatile char* src_pparts_attrs = src_pparts->GetAttrsPtr();')

    //             for attrId, attr in attrs.items():
    //                 if attr.dataType == Type.STRING:
    //                     code.Add(f'Themis::PullStrAttributesAtLoopLvl(thread_id, ts_{spSeqId}_{attr.id_name}_cached, ts_{spSeqId}_{attr.id_name}, (volatile str_t*) (src_pparts_attrs + {speculated_size}));')
    //                 elif attr.dataType == Type.PTR_INT:
    //                     code.Add(f'Themis::PullPtrIntAttributesAtLoopLvl(thread_id, ts_{spSeqId}_{attr.id_name}_cached, ts_{spSeqId}_{attr.id_name}, (volatile int**) (src_pparts_attrs + {speculated_size}));')
    //                 else:
    //                     code.Add(f'Themis::PullAttributesAtLoopLvl<{langType(attr.dataType)}>(thread_id, ts_{spSeqId}_{attr.id_name}_cached, ts_{spSeqId}_{attr.id_name}, ({langType(attr.dataType)}*) (src_pparts_attrs + {speculated_size}));')
    //                 speculated_size += CType.size[langType(attr.dataType)] * (2 * tsWidth)
    //         code.Add(f'if (thread_id == 0) stack->PopPartsAtLoopLvl({speculated_size});')
    //     elif spSeq.inputType == 2:
    //         code.Add('Themis::PushedParts::PushedPartsAtIfLvl* src_pparts = (Themis::PushedParts::PushedPartsAtIfLvl*) stack->Top();')
    //         code.Add(f'Themis::PullINodesFromPPartAtIfLvl(thread_id, {spSeqId}, src_pparts, inodes_cnts);')
    //         if len(spSeq.inBoundaryAttrs) > 0:
    //             code.Add('volatile char* src_pparts_attrs = src_pparts->GetAttrsPtr();')
    //             speculated_size = 0
    //             for attrId, attr in spSeq.inBoundaryAttrs.items():
    //                 if attr.dataType == Type.STRING:
    //                     code.Add(f'Themis::PullStrAttributesAtIfLvl(thread_id, ts_{spSeqId}_{attr.id_name}_flushed, (volatile str_t*) (src_pparts_attrs + {speculated_size}));')
    //                 elif attr.dataType == Type.PTR_INT:
    //                     code.Add(f'Themis::PullPtrIntAttributesAtIfLvl(thread_id, ts_{spSeqId}_{attr.id_name}_flushed, (int**) (src_pparts_attrs + {speculated_size}));')
    //                 else:
    //                     code.Add(f'Themis::PullAttributesAtIfLvl<{langType(attr.dataType)}>(thread_id, ts_{spSeqId}_{attr.id_name}_flushed, ({langType(attr.dataType)}*) (src_pparts_attrs + {speculated_size}));')
    //                 speculated_size += CType.size[langType(attr.dataType)] * (2 * tsWidth)
    //             code.Add(f'if (thread_id == 0) stack->PopPartsAtIfLvl({speculated_size});')
    //     code.Add('} break;')
    code.Add("} // switch");
}

void GpuCodeGenerator::GenerateCodeForAdaptiveWorkSharingPush(
    CypherPipeline &pipeline, CodeBuilder &code)
{
    int num_warps = int(KernelConstants::DEFAULT_BLOCK_SIZE / 32) *
                    KernelConstants::DEFAULT_GRID_SIZE;
    int min_num_warps = std::min(num_warps, kernel_args.min_num_warps);

    if (kernel_args.mode == "stats") {
        code.Add("unsigned long long current_tp = clock64();");
        code.Add(
            "if (current_status != -1) stat_counters[current_status] "
            "+= current_tp - tp;");
        code.Add("tp = current_tp;");
        code.Add("current_status = TYPE_STATS_PUSHING;");
    }

    code.Add("int target_warp_id = -1;");
    code.Add("unsigned int num_idle_warps = 0;");
    code.Add("unsigned int num_warps = 0;");

    if (doWorkoadSizeTracking) {
        code.Add(
            "bool is_allowed = Themis::isPushingAllowed(thread_id, "
            "warp_status, num_idle_warps, num_warps, local_info, "
            "global_stats_per_lvl);");
    }
    else {
        code.Add("bool is_allowed = false;");
    }
    code.Add("if (is_allowed) {");
    code.IncreaseNesting();

    // code.Add(
    //          "Themis::FindIdleWarp<{len(pipe.subpipeSeqs)},{num_warps},{min_"
    //          "num_warps}>(");  // TODO: what is subpipeSeqs?
    code.IncreaseNesting();
    code.Add(
        "target_warp_id, warp_id, thread_id, warp_status, num_idle_warps, "
        "num_warps,gts, size_of_stack_per_warp");
    if (idleWarpDetectionType == "twolvlbitmaps") {
        code.Add(", global_bit1, global_bit2");
    }
    else if (idleWarpDetectionType == "idqueue") {
        code.Add(", global_id_stack");
    }
    code.DecreaseNesting();
    code.Add(");");
    code.DecreaseNesting();
    code.Add("}");

    if (kernel_args.mode == "sample") {
        code.Add(
            "if (tried) sample(locally_lowest_lvl, thread_id, samples, "
            "sampling_start, TYPE_SAMPLE_TRY_PUSHING);");
        code.Add(
            "sample(locally_lowest_lvl, thread_id, samples, sampling_start, "
            "TYPE_SAMPLE_DETECTING);");
    }

    code.Add("if (target_warp_id >= 0) {");
    code.IncreaseNesting();
    if (kernel_args.mode == "sample") {
        code.Add(
            "sample(locally_lowest_lvl, thread_id, samples, sampling_start, "
            "TYPE_SAMPLE_PUSHING);");
    }

    code.Add(
        "Themis::PushedParts::PushedPartsStack* stack = "
        "Themis::PushedParts::GetIthStack(gts, size_of_stack_per_warp, "
        "target_warp_id);");
    code.Add("int lvl_to_push = local_info.locally_lowest_lvl;");
    code.Add("int num_to_push = 0;");
    code.Add("int num_remaining = 0;");
    code.Add(
        "int num_nodes = local_info.num_nodes_at_locally_lowest_lvl > 0 ? "
        "local_info.num_nodes_at_locally_lowest_lvl : 0;");
    code.Add("unsigned m = 0x1u << lvl_to_push;");
    GenerateCopyCodeForAdaptiveWorkSharingPush(pipeline, code);
    code.Add(
        "Themis::WorkloadTracking::UpdateWorkloadSizeOfIdleWarpAfterPush("
        "thread_id, lvl_to_push, num_to_push, global_stats_per_lvl);");
    code.Add("if (thread_id == 0) stack->FreeLock();");

    // Recalculate current workload size of this busy warp
    code.Add("// Calculate the workload size of this busy warp");
    code.Add("int new_num_nodes_at_locally_lowest_lvl = num_remaining;");
    code.Add(
        "int8_t new_local_max_order = "
        "Themis::CalculateOrder(new_num_nodes_at_locally_lowest_lvl);");
    code.Add(
        "int new_local_lowest_lvl = new_num_nodes_at_locally_lowest_lvl > 0 ? "
        "lvl_to_push : -1;");

    // Find the new lowet level
    code.Add("if (new_num_nodes_at_locally_lowest_lvl == 0 && mask_1 != 0) {");
    code.IncreaseNesting();
    code.Add("new_local_lowest_lvl = __ffs(mask_1) - 1;");
    // if len(pipe.subpipeSeqs) > 1:
    //     code.Add("switch (new_local_lowest_lvl) {")
    //     for spSeqId, spSeq in enumerate(pipe.subpipeSeqs):
    //         if spSeqId == 0: continue
    //         code.Add(f"case {spSeqId}:" + "{")
    //         if spSeq.inputType == 1:
    //             code.Add(f"Themis::CountINodesAtLoopLvl(thread_id, {spSeqId}, ts_{spSeqId}_range_cached, ts_{spSeqId}_range, inodes_cnts);")
    //             code.Add(f"new_num_nodes_at_locally_lowest_lvl = __shfl_sync(ALL_LANES, inodes_cnts, {spSeqId});")
    //             code.Add(f"new_local_max_order = Themis::CalculateOrder(new_num_nodes_at_locally_lowest_lvl);")
    //         else:
    //             code.Add(f"new_num_nodes_at_locally_lowest_lvl = __shfl_sync(ALL_LANES, inodes_cnts, {spSeqId});")
    //             code.Add(f"new_local_max_order = 0;")
    //         code.Add("} break;")
    //     code.Add("}")
    code.DecreaseNesting();
    code.Add("}");
    code.Add(
        "Themis::WorkloadTracking::UpdateWorkloadSizeOfBusyWarpAfterPush("
        "thread_id, mask_1, new_num_nodes_at_locally_lowest_lvl, "
        "new_local_lowest_lvl, new_local_max_order, local_info, "
        "global_stats_per_lvl);");
    code.Add("interval = 0;");
    code.Add("Themis::ChooseLvl(thread_id, mask_32, mask_1, lvl);");
    code.DecreaseNesting();
    code.Add("} else {");
    code.IncreaseNesting();
    code.Add(
        "Themis::chooseNextIntervalAfterPush(interval, local_info, "
        "num_warps, num_idle_warps, is_allowed, " +
        std::to_string(kernel_args.inter_warp_lb_interval) + ");");
    code.DecreaseNesting();
    code.Add("} // else");
    code.Add("loop = " + std::to_string(kernel_args.inter_warp_lb_interval) +
             " - interval;");
}

void GpuCodeGenerator::GenerateCopyCodeForAdaptiveWorkSharingPush(
    CypherPipeline &pipeline, CodeBuilder &code)
{
    int stepSize = KernelConstants::DEFAULT_BLOCK_SIZE *
                   KernelConstants::DEFAULT_GRID_SIZE;

    code.Add("switch(lvl_to_push) {");

    // for spSeqId, spSeq in enumerate(pipe.subpipeSeqs):
    //     code.Add(f'case {spSeqId}: ' + '{')
    //     if spSeq.inputType == 0:
    //         code.Add(f'if (thread_id == 0) stack->PushPartsAtZeroLvl();')
    //         code.Add('Themis::PushedParts::PushedPartsAtZeroLvl* target_pparts = (Themis::PushedParts::PushedPartsAtZeroLvl*) stack->Top();')
    //         code.Add(f'num_to_push = Themis::PushINodesToPPartAtZeroLvl(thread_id, target_pparts, ts_0_range_cached, {stepSize});')
    //         code.Add(f'num_remaining = num_nodes - num_to_push;')
    //         code.Add(f'mask_32 = num_remaining >= 32 ? (m | mask_32) : ((~m) & mask_32);')
    //         code.Add(f'mask_1 = num_remaining > 0 ?  (m | mask_1) : ((~m) & mask_1);')

    //     elif spSeq.inputType == 1:

    //         attrs = {}
    //         tid = spSeq.getTid()
    //         lastOp = spSeq.pipe.subpipeSeqs[spSeq.id-1].subpipes[-1].operators[-1]
    //         for attrId, attr in spSeq.inBoundaryAttrs.items():
    //             if attrId == tid.id: continue
    //             if attrId in lastOp.generatingAttrs: continue
    //             attrs[attrId] = attr

    //         speculated_size = 0
    //         for attrId, attr in attrs.items():
    //             speculated_size += CType.size[langType(attr.dataType)] * (2 * tsWidth)
    //         code.Add(f'if (thread_id == 0) stack->PushPartsAtLoopLvl({spSeqId}, {speculated_size});')
    //         code.Add('Themis::PushedParts::PushedPartsAtLoopLvl* target_pparts = (Themis::PushedParts::PushedPartsAtLoopLvl*) stack->Top();')
    //         code.Add(f'num_to_push = Themis::PushINodesToPPartAtLoopLvl(thread_id, {spSeqId}, target_pparts, ts_{spSeqId}_range_cached, ts_{spSeqId}_range);')
    //         code.Add(f'num_remaining = num_nodes - num_to_push;')
    //         code.Add(f'mask_32 = num_remaining >= 32 ? (m | mask_32) : ((~m) & mask_32);')
    //         code.Add(f'mask_1 = num_remaining > 0 ?  (m | mask_1) : ((~m) & mask_1);')
    //         code.Add(f'int ts_src = 32;')
    //         code.Add(f'Themis::DistributeFromPartToDPart(thread_id, {spSeqId}, ts_src, ts_{spSeqId}_range, ts_{spSeqId}_range_cached, mask_32, mask_1);')
    //         if len(attrs) > 0:
    //             speculated_size = 0
    //             code.Add('volatile char* target_pparts_attrs = target_pparts->GetAttrsPtr();')
    //             for attrId, attr in attrs.items():
    //                 name = f'ts_{spSeqId}_{attr.id_name}'
    //                 code.Add('{')
    //                 if attr.dataType == Type.STRING:
    //                     code.Add(f'Themis::PushStrAttributesAtLoopLvl(thread_id, (volatile str_t*) (target_pparts_attrs + {speculated_size}), {name}_cached, {name});')
    //                     code.Add(f'char* start = (char*) __shfl_sync(ALL_LANES, (uint64_t) {name}.start, ts_src);')
    //                     code.Add(f'char* end = (char*) __shfl_sync(ALL_LANES, (uint64_t) {name}.end, ts_src);')
    //                     code.Add(f'if (ts_src < 32) {name}_cached.start = start;')
    //                     code.Add(f'if (ts_src < 32) {name}_cached.end = end;')
    //                 elif attr.dataType == Type.PTR_INT:
    //                     code.Add(f'Themis::PushPtrIntAttributesAtLoopLvl(thread_id, (volatile int**) (target_pparts_attrs + {speculated_size}), {name}_cached, {name});')
    //                     code.Add(f'uint64_t cache = __shfl_sync(ALL_LANES, (uint64_t){name}, ts_src);')
    //                     code.Add(f'if (ts_src < 32) {name}_cached = (int*) cache;')
    //                 else:
    //                     dtype = langType(attr.dataType)
    //                     code.Add(f'Themis::PushAttributesAtLoopLvl<{dtype}>(thread_id, (volatile {dtype}*) (target_pparts_attrs + {speculated_size}), {name}_cached, {name});')
    //                     code.Add(f'{dtype} cache = __shfl_sync(ALL_LANES, {name}, ts_src);')
    //                     code.Add(f'if (ts_src < 32) {name}_cached = cache;')
    //                 code.Add('}')
    //                 speculated_size += CType.size[langType(attr.dataType)] * (2 * tsWidth)
    //     else:
    //         speculated_size = 0
    //         for attrId, attr in spSeq.inBoundaryAttrs.items():
    //             speculated_size += CType.size[langType(attr.dataType)] * (2 * tsWidth)
    //         code.Add(f'if (thread_id == 0) stack->PushPartsAtIfLvl({spSeqId}, {speculated_size});')
    //         code.Add('Themis::PushedParts::PushedPartsAtIfLvl* target_pparts = (Themis::PushedParts::PushedPartsAtIfLvl*) stack->Top();')
    //         code.Add(f'num_to_push = Themis::PushINodesToPPartAtIfLvl(thread_id, {spSeqId}, target_pparts, inodes_cnts);')
    //         code.Add(f'mask_32 = ((~m) & mask_32);')
    //         code.Add(f'mask_1 = ((~m) & mask_1);')
    //         if len(spSeq.inBoundaryAttrs) > 0:
    //             code.Add('volatile char* target_pparts_attrs = target_pparts->GetAttrsPtr();')
    //             speculated_size = 0
    //             for attrId, attr in spSeq.inBoundaryAttrs.items():
    //                 name = f'ts_{spSeqId}_{attr.id_name}'
    //                 code.Add('{')
    //                 if attr.dataType == Type.STRING:
    //                     code.Add(f'Themis::PushStrAttributesAtIfLvl(thread_id, (volatile str_t*) (target_pparts_attrs + {speculated_size}), {name}_flushed);')
    //                 elif attr.dataType == Type.PTR_INT:
    //                     code.Add(f'Themis::PushPtrIntAttributesAtIfLvl(thread_id, (volatile int**) (target_pparts_attrs + {speculated_size}), {name}_flushed);')
    //                 else:
    //                     dtype = langType(attr.dataType)
    //                     code.Add(f'Themis::PushAttributesAtIfLvl<{dtype}>(thread_id, (volatile {dtype}*) (target_pparts_attrs + {speculated_size}), {name}_flushed);')
    //                 code.Add('}')
    //                 speculated_size += CType.size[langType(attr.dataType)] * (2 * tsWidth)
    //     code.Add('} break;')
    code.Add("}");
    int num_warps_per_block = int(KernelConstants::DEFAULT_BLOCK_SIZE / 32);
    code.Add("if ((target_warp_id / " + std::to_string(num_warps_per_block) +
             ") == (gpart_id / " + std::to_string(num_warps_per_block) +
             ")) __threadfence_block();");
    code.Add("else __threadfence();");
}

// Pipeline context management methods
void GpuCodeGenerator::InitializePipelineContext(const CypherPipeline &pipeline)
{
    pipeline_context.InitializePipeline(pipeline);
}

void GpuCodeGenerator::MoveToOperator(int op_idx)
{
    pipeline_context.MoveToOperator(op_idx);
}

void GpuCodeGenerator::AnalyzeOperatorDependencies(CypherPhysicalOperator *op)
{
    // Analyze expressions to track column usage and update mappings
    switch (op->GetOperatorType()) {
        case PhysicalOperatorType::PROJECTION: {
            auto proj_op = dynamic_cast<PhysicalProjection *>(op);
            if (proj_op) {
                // Clear existing column mappings for this operator
                pipeline_context.column_mapping.clear();

                for (size_t i = 0; i < proj_op->expressions.size(); i++) {
                    auto &expr = proj_op->expressions[i];
                    TrackColumnUsage(expr.get());

                    // Update column mapping based on expression type
                    if (expr->expression_class == ExpressionClass::BOUND_REF) {
                        auto ref_expr =
                            dynamic_cast<BoundReferenceExpression *>(
                                expr.get());
                        if (ref_expr &&
                            ref_expr->index <
                                pipeline_context.input_column_names.size()) {
                            std::string input_col =
                                pipeline_context
                                    .input_column_names[ref_expr->index];
                            if (i <
                                pipeline_context.output_column_names.size()) {
                                std::string output_col =
                                    pipeline_context.output_column_names[i];
                                pipeline_context.column_mapping[output_col] =
                                    input_col;
                            }
                        }
                    }
                }
            }
            break;
        }
        case PhysicalOperatorType::FILTER: {
            // For filter, maintain existing column mappings (pass-through)
            // TODO: Analyze filter expressions for column usage
            break;
        }
        default:
            // For other operators, maintain existing column mappings
            break;
    }
}

void GpuCodeGenerator::ExtractInputSchema(CypherPhysicalOperator *op)
{
    // For now, we'll extract from the operator's children
    // In a more complete implementation, this would analyze the actual input schema
    if (pipeline_context.current_operator_index > 0) {
        pipeline_context.input_column_names =
            *pipeline_context.operator_column_names
                 [pipeline_context.current_operator_index - 1];
        pipeline_context.input_column_types =
            *pipeline_context.operator_column_types
                 [pipeline_context.current_operator_index - 1];
    }
}

void GpuCodeGenerator::ExtractOutputSchema(CypherPhysicalOperator *op)
{
    auto &schema = op->GetSchema();
    auto &column_names = schema.getStoredColumnNamesRef();
    auto &column_types = schema.getStoredTypesRef();

    pipeline_context.output_column_names.clear();
    pipeline_context.output_column_types.clear();

    for (size_t i = 0; i < column_names.size(); i++) {
        pipeline_context.output_column_names.push_back(column_names[i]);
        pipeline_context.output_column_types.push_back(column_types[i]);
    }
}

void GpuCodeGenerator::TrackColumnUsage(Expression *expr)
{
    if (!expr)
        return;

    switch (expr->expression_class) {
        case ExpressionClass::BOUND_REF: {
            auto ref_expr = dynamic_cast<BoundReferenceExpression *>(expr);
            if (ref_expr &&
                ref_expr->index < pipeline_context.input_column_names.size()) {
                std::string col_name =
                    pipeline_context.input_column_names[ref_expr->index];
                pipeline_context.used_columns.insert(col_name);
            }
            break;
        }
        case ExpressionClass::BOUND_FUNCTION: {
            auto func_expr = dynamic_cast<BoundFunctionExpression *>(expr);
            if (func_expr) {
                for (auto &child : func_expr->children) {
                    TrackColumnUsage(child.get());
                }
            }
            break;
        }
        case ExpressionClass::BOUND_OPERATOR: {
            auto op_expr = dynamic_cast<BoundOperatorExpression *>(expr);
            if (op_expr) {
                for (auto &child : op_expr->children) {
                    TrackColumnUsage(child.get());
                }
            }
            break;
        }
        default:
            break;
    }
}

// PipelineContext method implementations
void PipelineContext::InitializePipeline(const CypherPipeline &pipeline)
{
    total_operators = pipeline.GetPipelineLength();
    current_operator_index = 0;

    // Clear existing data
    operator_column_names.clear();
    operator_column_types.clear();
    column_materialized.clear();
    column_mapping.clear();
    used_columns.clear();
    gpu_memory_loaded.clear();
    // column_to_param_mapping.clear();
    column_to_table_mapping.clear();
    column_to_extent_mapping.clear();
    column_to_chunk_mapping.clear();

    // Collect all operator schemas
    for (int i = 0; i < total_operators; i++) {
        auto op = pipeline.GetIdxOperator(i);
        if (op) {
            auto &schema = op->GetSchema();
            auto &column_names = schema.getStoredColumnNamesRef();
            auto &column_types = schema.getStoredTypesRef();

            operator_column_names.push_back(&column_names);
            operator_column_types.push_back(&column_types);
        }
        else {
            operator_column_names.push_back(nullptr);
            operator_column_types.push_back(nullptr);
        }
    }
}

void PipelineContext::MoveToOperator(int op_idx)
{
    if (op_idx >= 0 && op_idx < total_operators) {
        current_operator_index = op_idx;

        // Update input schema from previous operator
        if (op_idx > 0 && operator_column_names[op_idx - 1] &&
            operator_column_types[op_idx - 1]) {
            input_column_names = *operator_column_names[op_idx - 1];
            input_column_types = *operator_column_types[op_idx - 1];
        }
        else {
            input_column_names.clear();
            input_column_types.clear();
        }

        // Update output schema from current operator
        if (operator_column_names[op_idx] && operator_column_types[op_idx]) {
            output_column_names = *operator_column_names[op_idx];
            output_column_types = *operator_column_types[op_idx];
        }
        else {
            output_column_names.clear();
            output_column_types.clear();
        }

        // Update column mappings for pass-through columns
        // If a column exists in both input and output with the same name, maintain the mapping
        if (op_idx > 0) {
            std::unordered_map<std::string, std::string> new_column_mapping;

            for (const auto &output_col : output_column_names) {
                // Check if this output column exists in input (pass-through)
                auto it = std::find(input_column_names.begin(),
                                    input_column_names.end(), output_col);
                if (it != input_column_names.end()) {
                    // This is a pass-through column, maintain the mapping
                    auto mapping_it = column_mapping.find(output_col);
                    if (mapping_it != column_mapping.end()) {
                        new_column_mapping[output_col] = mapping_it->second;
                    }
                    else {
                        // Direct pass-through
                        new_column_mapping[output_col] = output_col;
                    }
                }
            }

            // Update column_mapping with new mappings
            column_mapping = new_column_mapping;
        }
    }
}

}  // namespace duckdb