#include "planner/gpu/gpu_code_generator.hpp"
#include <cuda.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <nvrtc.h>
#include <sstream>
#include <cctype>
#include <algorithm>
#include "catalog/catalog.hpp"
#include "catalog/catalog_entry/list.hpp"
#include "common/file_system.hpp"
#include "common/logger.hpp"
#include "execution/physical_operator/cypher_physical_operator.hpp"
#include "execution/physical_operator/physical_filter.hpp"
#include "execution/physical_operator/physical_node_scan.hpp"
#include "execution/physical_operator/physical_produce_results.hpp"
#include "execution/physical_operator/physical_projection.hpp"
#include "execution/physical_operator/physical_hash_aggregate.hpp"
#include "llvm/Support/TargetSelect.h"
#include "main/database.hpp"
#include "planner/gpu/expression_code_generator.hpp"
#include "planner/expression/bound_function_expression.hpp"
#include "planner/expression/bound_cast_expression.hpp"

namespace duckdb {

GpuCodeGenerator::GpuCodeGenerator(ClientContext &context)
    : context(context), is_compiled(false), is_repeatable(false)
{
    InitializeLLVMTargets();
    jit_compiler = std::make_unique<GpuJitCompiler>();
    InitializeOperatorGenerators();

    // for debugging
    gpu_code_file.open("generated_gpu_code.cu", std::ios::out | std::ios::trunc);
    cpu_code_file.open("generated_cpu_code.cpp", std::ios::out | std::ios::trunc);
    gpu_code_file.close();
    cpu_code_file.close();
}

GpuCodeGenerator::~GpuCodeGenerator()
{
    Cleanup();
    gpu_code_file.close();
    cpu_code_file.close();
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
    operator_generators[PhysicalOperatorType::HASH_AGGREGATE] =
        std::make_unique<HashAggregateCodeGenerator>();
}

void GpuCodeGenerator::GenerateGlobalDeclarations(
    std::vector<CypherPipeline *> &pipelines)
{
    CodeBuilder code;
    for (auto &pipeline : pipelines) {
        // Generate global declarations for each operator
        for (size_t i = 0; i < pipeline->GetPipelineLength(); i++) {
            auto *op = pipeline->GetIdxOperator(i);
            GenerateGlobalDeclarations(op, i, code, pipeline_context);
        }
    }
    global_declarations = code.str();

    // for debugging - write to file
    gpu_code_file.open("generated_gpu_code.cu", std::ios::app);
    gpu_code_file << global_declarations << std::endl;
    gpu_code_file.close();
}

void GpuCodeGenerator::GenerateGPUCode(CypherPipeline &pipeline)
{
    SCOPED_TIMER_SIMPLE(GenerateGPUCode, spdlog::level::info,
                        spdlog::level::info);

    // generate kernel code
    SUBTIMER_START(GenerateGPUCode, "GenerateKernelCode");
    GenerateKernelCode(pipeline);
    SUBTIMER_STOP(GenerateGPUCode, "GenerateKernelCode");

    // then, generate host code
    SUBTIMER_START(GenerateGPUCode, "GenerateHostCode");
    GenerateHostCode(pipeline);
    SUBTIMER_STOP(GenerateGPUCode, "GenerateHostCode");

    // for debug - write to files
    gpu_code_file.open("generated_gpu_code.cu", std::ios::app);
    gpu_code_file << generated_gpu_code << std::endl;
    gpu_code_file.close();
    cpu_code_file.open("generated_cpu_code.cpp", std::ios::app);
    cpu_code_file << generated_cpu_code << std::endl;
    cpu_code_file.close();
}

void GpuCodeGenerator::SplitPipelineIntoSubPipelines(CypherPipeline &pipeline)
{
    CypherPhysicalOperatorGroups groups;
    CypherPhysicalOperatorGroups &pipeline_groups = pipeline.GetOperatorGroups();
    pipeline_context.sub_pipelines.clear();
    
    for (int op_idx = 0; op_idx < pipeline.GetPipelineLength(); op_idx++) {
        auto op = pipeline.GetIdxOperator(op_idx);
        if (!op) continue;
        
        groups.GetGroups().push_back(pipeline_groups.GetGroups()[op_idx]);
        
        // TODO: implement other cases
        if (op->GetOperatorType() == PhysicalOperatorType::NODE_SCAN) {
            auto scan_op = dynamic_cast<PhysicalNodeScan *>(op);
            if (scan_op->is_filter_pushdowned) {
                // If the scan operator has filter pushdown, we need to
                // create a new sub-pipeline for the filter
                pipeline_context.sub_pipelines.emplace_back(groups);
                groups.GetGroups().clear();
                groups.GetGroups().push_back(pipeline_groups.GetGroups()[op_idx]);
            }
        }
        else if (op->GetOperatorType() == PhysicalOperatorType::FILTER) {
            pipeline_context.sub_pipelines.emplace_back(groups);
            groups.GetGroups().clear();
            groups.GetGroups().push_back(pipeline_groups.GetGroups()[op_idx]);
        }
    }
    
    pipeline_context.sub_pipelines.emplace_back(groups);
}

void GpuCodeGenerator::AnalyzeSubPipelinesForMaterialization()
{
    int num_sub_pipelines = pipeline_context.sub_pipelines.size();
    auto &columns_to_be_materialized =
        pipeline_context.columns_to_be_materialized;
    columns_to_be_materialized.clear();
    columns_to_be_materialized.resize(num_sub_pipelines);
    auto &sub_pipeline_tids = pipeline_context.sub_pipeline_tids;
    sub_pipeline_tids.clear();
    sub_pipeline_tids.resize(num_sub_pipelines);
    pipeline_context.attribute_tid_mapping.clear();
    pipeline_context.attribute_source_mapping.clear();

    // Analyze each sub-pipeline to determine which columns need to be
    // materialized
    for (int sub_idx = 0; sub_idx < num_sub_pipelines; sub_idx++) {
        auto &sub_pipeline = pipeline_context.sub_pipelines[sub_idx];
        // Iterate over each operator in the sub-pipeline
        int op_idx = sub_idx == 0 ? 0 : 1;
        for (; op_idx < sub_pipeline.GetPipelineLength(); op_idx++) {
            auto *op = sub_pipeline.GetIdxOperator(op_idx);
            auto &output_schema = op->GetSchema();
            auto &output_column_names = output_schema.getStoredColumnNamesRef();
            auto &output_column_types = output_schema.getStoredTypesRef();

            switch (op->GetOperatorType()) {
                case PhysicalOperatorType::NODE_SCAN: {
                    auto *scan_op = dynamic_cast<PhysicalNodeScan *>(op);

                    // Create tid mapping info
                    std::string tid = "tid_" + std::to_string(sub_idx) + "_" +
                                      std::to_string(op_idx) + "_" +
                                      std::to_string(scan_op->oids[0]);
                    for (const auto &col_name : output_column_names) {
                        D_ASSERT(pipeline_context.attribute_tid_mapping.find(
                                     col_name) ==
                                 pipeline_context.attribute_tid_mapping.end());
                        pipeline_context.attribute_tid_mapping[col_name] = tid;
                    }
                    sub_pipeline_tids[sub_idx].push_back(tid);

                    // Create source mapping info
                    for (size_t i = 0; i < scan_op->oids.size(); i++) {
                        uint64_t oid = scan_op->oids[i];
                        auto &scan_projection_mapping =
                            scan_op->scan_projection_mapping[i];
                        for (size_t j = 0; j < scan_projection_mapping.size();
                             j++) {
                            auto &col_name = output_column_names[j];
                            std::string source_name =
                                "gr" + std::to_string(oid) + "_col_" +
                                std::to_string(scan_projection_mapping[j] - 1) +
                                "_data";
                            D_ASSERT(
                                pipeline_context.attribute_source_mapping.find(
                                    col_name) ==
                                pipeline_context.attribute_source_mapping
                                    .end());
                            pipeline_context
                                .attribute_source_mapping[col_name] =
                                source_name;
                        }
                    }

                    // Check if the scan operator has filter pushdown
                    // If it does, we need to materialize the columns
                    if (!scan_op->is_filter_pushdowned) {
                        break;
                    }

                    // Get referenced columns from the filter pushdown
                    std::vector<uint64_t> referenced_columns;
                    if (scan_op->filter_pushdown_type ==
                            FilterPushdownType::FP_EQ ||
                        scan_op->filter_pushdown_type ==
                            FilterPushdownType::FP_RANGE) {
                        for (const auto &key_idx :
                             scan_op->filter_pushdown_key_idxs_in_output) {
                            referenced_columns.push_back(key_idx);
                        }
                    }
                    else {
                        GetReferencedColumns(scan_op->filter_expression.get(),
                                             referenced_columns);
                    }
                    // TODO handling multiple graphlets
                    D_ASSERT(scan_op->oids.size() == 1);
                    for (const auto &col_idx : referenced_columns) {
                        D_ASSERT(col_idx >= 0 &&
                                 col_idx < output_column_names.size());
                        // Mark the column as materialized
                        auto &col_name = output_column_names[col_idx];
                        columns_to_be_materialized[sub_idx].push_back(
                            Attr{col_name, output_column_types[col_idx], col_idx});
                    }
                    break;
                }
                case PhysicalOperatorType::FILTER: {
                    auto *filter_op = dynamic_cast<PhysicalFilter *>(op);
                    std::vector<uint64_t> referenced_columns;
                    GetReferencedColumns(filter_op->expression.get(),
                                         referenced_columns);
                    for (const auto &col_idx : referenced_columns) {
                        D_ASSERT(col_idx >= 0 &&
                                 col_idx < output_column_names.size());
                        // Mark the column as materialized
                        auto &col_name = output_column_names[col_idx];
                        columns_to_be_materialized[sub_idx].push_back(Attr{
                            col_name, output_column_types[col_idx], col_idx});
                    }
                    break;
                }
                case PhysicalOperatorType::HASH_AGGREGATE: {
                    if (op_idx != 0)
                        break;
                    // Create tid mapping info
                    std::string tid = "tid_" + std::to_string(sub_idx) + "_" +
                                      std::to_string(op_idx);
                    for (const auto &col_name : output_column_names) {
                        D_ASSERT(pipeline_context.attribute_tid_mapping.find(
                                     col_name) ==
                                 pipeline_context.attribute_tid_mapping.end());
                        pipeline_context.attribute_tid_mapping[col_name] = tid;
                    }
                    sub_pipeline_tids[sub_idx].push_back(tid);
                    break;
                }
                case PhysicalOperatorType::PROJECTION: {
                    auto *proj_op = dynamic_cast<PhysicalProjection *>(op);
                    for (const auto &expr : proj_op->expressions) {
                        std::vector<uint64_t> referenced_columns;
                        GetReferencedColumns(expr.get(), referenced_columns);
                        for (const auto &col_idx : referenced_columns) {
                            D_ASSERT(col_idx >= 0 &&
                                     col_idx < output_column_names.size());
                            auto &col_name = output_column_names[col_idx];

                            bool is_valid_column = true;
                            for (char c : col_name) {
                                if (c == '(' || c == ')' || c == ',' ||
                                    c == '-' || c == ' ') {
                                    is_valid_column = false;
                                    break;
                                }
                            }

                            if (is_valid_column) {
                                columns_to_be_materialized[sub_idx].push_back(
                                    Attr{col_name, output_column_types[col_idx],
                                         col_idx});
                            }
                        }
                    }
                    break;
                }
                default:
                    break;
            }
        }
    }
}

void GpuCodeGenerator::ResolveAttributes()
{
    throw NotImplementedException(
        "GpuCodeGenerator::ResolveAttributes is not implemented yet");
}

void GpuCodeGenerator::ResolveBoundaryAttributes()
{
    throw NotImplementedException(
        "GpuCodeGenerator::ResolveBoundaryAttributes is not implemented yet");
}

void GpuCodeGenerator::GetReferencedColumns(
    Expression *expr, std::vector<uint64_t> &referenced_columns)
{
    switch (expr->expression_class) {
        case ExpressionClass::BOUND_REF: {
            auto ref_expr = dynamic_cast<BoundReferenceExpression *>(expr);
            referenced_columns.push_back(ref_expr->index);
            break;
        }
        case ExpressionClass::BOUND_BETWEEN: {
            auto between_expr = dynamic_cast<BoundBetweenExpression *>(expr);
            GetReferencedColumns(between_expr->input.get(), referenced_columns);
            break;
        }
        case ExpressionClass::BOUND_COMPARISON: {
            auto comp_expr =
                dynamic_cast<BoundComparisonExpression *>(expr);
            GetReferencedColumns(comp_expr->left.get(), referenced_columns);
            GetReferencedColumns(comp_expr->right.get(), referenced_columns);
            break;
        }
        case ExpressionClass::BOUND_CONSTANT: {
            // Constants do not reference any columns
            break;
        }
        case ExpressionClass::BOUND_OPERATOR: {
            auto op_expr = dynamic_cast<BoundOperatorExpression *>(expr);
            if (op_expr) {
                for (auto &child : op_expr->children) {
                    GetReferencedColumns(child.get(), referenced_columns);
                }
            }
            break;
        }
        case ExpressionClass::BOUND_FUNCTION: {
            auto func_expr = dynamic_cast<BoundFunctionExpression *>(expr);
            if (func_expr) {
                for (auto &child : func_expr->children) {
                    GetReferencedColumns(child.get(), referenced_columns);
                }
            }
            break;
        }
        case ExpressionClass::BOUND_CAST: {
            auto cast_expr = dynamic_cast<BoundCastExpression *>(expr);
            if (cast_expr) {
                GetReferencedColumns(cast_expr->child.get(), referenced_columns);
            }
            break;
        }
        case ExpressionClass::BOUND_CONJUNCTION: {
            auto conj_expr =
                dynamic_cast<BoundConjunctionExpression *>(expr);
            for (size_t i = 0; i < conj_expr->children.size(); i++) {
                GetReferencedColumns(conj_expr->children[i].get(),
                                     referenced_columns);
            }
            break;
        }
        default:
            std::cerr << "Not implemented expression type for column reference "
                    "extraction: " << (uint8_t)(expr->expression_class) << std::endl;
            std::cerr << "Expression type enum value: " << static_cast<int>(expr->expression_class) << std::endl;
            throw NotImplementedException(
                "Not implemented expression type for column reference "
                "extraction");
            break;
    }
}

void GpuCodeGenerator::GenerateKernelCode(CypherPipeline &pipeline)
{
    CodeBuilder code;

    // Initialize pipeline context once
    InitializePipelineContext(pipeline);

    // Split pipeline into sub-pipelines based on filter operators
    SplitPipelineIntoSubPipelines(pipeline);

    // Analyze sub pipelines and get information for materialization
    AnalyzeSubPipelinesForMaterialization();

    // Generate kernel parameters
    GenerateKernelParams(pipeline);

    // include necessary headers
    code.Add("#include \"range.cuh\"");
    code.Add("#include \"themis.cuh\"");
    code.Add("#include \"work_sharing.cuh\"");
    code.Add("#include \"adaptive_work_sharing.cuh\"");
    code.Add(""); // new line

    // kernel function declaration
    if (do_inter_warp_lb) {
        code.Add(
            "extern \"C\" __global__ void __launch_bounds__(128, 8) "
            "gpu_kernel(");
    }
    else {
        code.Add("extern \"C\" __global__ void gpu_kernel(");
    }
    code.IncreaseNesting();
    code.Add("void **input_data, void **output_data, int *output_count,");

    code.Add("unsigned int *global_num_idle_warps, int *global_scan_offset,");
    //if self.interWarpLbMethod == 'aws':
    code.Add("Themis::PushedParts::PushedPartsStack* gts, size_t size_of_stack_per_warp,");
    code.Add("Themis::StatisticsPerLvl *global_stats_per_lvl,");
    //if self.idleWarpDetectionType == 'twolvlbitmaps':
    code.Add("unsigned long long *global_bit1, unsigned long long *global_bit2");
    code.DecreaseNesting();
    code.Add(") {");
    code.IncreaseNesting();
    int in_idx  = 0;
    int out_idx = 0;
    for (const auto &p : input_kernel_params) {
        if (p.type.find('*') != std::string::npos) {
            code.Add(p.type + p.name + " = (" + p.type +
                     ")input_data[" + std::to_string(in_idx) + "];");
            ++in_idx;
        } else {
            code.Add("const " + p.type + " " + p.name + " = " + p.value + ";");
        }
    }
    for (const auto &p : output_kernel_params) {
        if (p.type.find('*') != std::string::npos) {
            code.Add(p.type + p.name + " = (" + p.type +
                     ")output_data[" + std::to_string(out_idx) + "];");
            ++out_idx;
        } else {
            // code.Add(p.type + " " + p.name + " = " + p.value + ";");
        }
    }
    code.Add("");

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
    code.Add("int thread_id = threadIdx.x % 32;");
    code.Add("int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;");
    code.Add("unsigned prefixlanes = (0xffffffffu >> (32 - thread_id));");
    code.Add("int active = 0;");

    for (auto i = 0; i < pipeline_context.sub_pipelines.size(); i++) {
        auto &sub_pipeline = pipeline_context.sub_pipelines[i];
        PipeInputType input_type = GetPipeInputType(sub_pipeline);
        if (input_type == PipeInputType::TYPE_1) {
            code.Add("Range ts_" + std::to_string(i) + "_range;");
            code.Add("Range ts_" + std::to_string(i) + "_range_cached;");
            // tid = spSeq.getTid()
            // lastOp = spSeq.pipe.subpipeSeqs[spSeq.id-1].subpipes[-1].operators[-1]
            // for attrId, attr in spSeq.inBoundaryAttrs.items():
            //     if attrId == tid.id: continue
            //     if attrId in lastOp.generatingAttrs: continue
            //     code.Add(f'{langType(attr.dataType)} ts_{spSeqId}_{attr.id_name};')
            //     code.Add(f'{langType(attr.dataType)} ts_{spSeqId}_{attr.id_name}_cached;')
        }
        else if (input_type == PipeInputType::TYPE_2) {
            D_ASSERT(i > 0);
            auto &tids = pipeline_context.sub_pipeline_tids[i - 1];
            for (const auto &tid : tids) {
                code.Add("int ts_" + std::to_string(i) + "_" + tid + ";");
                code.Add("int ts_" + std::to_string(i) + "_" + tid +
                         "_flushed;");
            }
        }
    }

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
        code.Add("unsigned loop = " +
                 std::to_string(kernel_args.inter_warp_lb_interval) + " - 1;");
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
        "Themis::UpdateMaskAtZeroLvl(0, thread_id, ts_0_range_cached, mask_32, "
        "mask_1);");
    code.Add("Themis::ChooseLvl(thread_id, mask_32, mask_1, lvl);");

    // Generate the main scan loop that will contain all operators
    if (do_inter_warp_lb) {
        code.Add("do {");
        code.IncreaseNesting();
        GeneratePipelineCode(pipeline, code);
        GenerateCodeForAdaptiveWorkSharing(pipeline, code);
        // code.DecreaseNesting();
        // code.Add("}");
        code.DecreaseNesting();
        code.Add("} while (true); // while loop");
    }
    else {
        GeneratePipelineCode(pipeline, code);
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
    // code.Add("#include \"range.cuh\"");
    // code.Add("#include \"themis.cuh\"");
    // code.Add("#include \"work_sharing.cuh\"");
    code.Add("#include \"adaptive_work_sharing.cuh\"");
    code.Add("#include \"pushedparts.cuh\"");
    code.Add("");

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

    int input_ptr_count  = 0;
    for (const auto &p : input_kernel_params)
        if (p.type.find('*') != std::string::npos) ++input_ptr_count;

    int output_ptr_count = 0;
    for (const auto &p : output_kernel_params)
        if (p.type.find('*') != std::string::npos) ++output_ptr_count;

    code.Add("void *input_data[" + std::to_string(input_ptr_count)  + "];");
    code.Add("void *output_data[" + std::to_string(output_ptr_count) + "];");
    code.Add("int  output_count = 0;");
    code.Add("");

    int ptr_map_idx = 0;
    int in_idx      = 0;

    for (const auto &p : input_kernel_params) {
        if (p.type.find('*') != std::string::npos) {
            code.Add("input_data[" + std::to_string(in_idx) + "] = "
                    "ptr_mappings[" + std::to_string(ptr_map_idx) + "].address;");
            ++in_idx;
            ++ptr_map_idx;
        } else {
            // skip
            // code.Add(p.type + " " + p.name + " = " + p.value + ";");
        }
    }

    int out_idx = 0;
    for (const auto &p : output_kernel_params) {
        if (p.type.find('*') != std::string::npos) {
            code.Add("cudaMalloc(&output_data[" + std::to_string(out_idx) + "], 1024);");
            ++out_idx;
        } else {
            // skip
            // code.Add(p.type + " " + p.name + " = " + p.value + ";");
        }
    }
    code.Add("");

    code.Add("const int blockSize = 128;");
    code.Add("const int gridSize  = 3280;");

    GenerateDeclarationInHostCode(code);
    GenerateKernelCallInHostCode(code);
    
    code.Add("if (r != CUDA_SUCCESS) {");
    code.IncreaseNesting();
    code.Add("const char *name = nullptr, *str = nullptr;");
    code.Add("cuGetErrorName(r, &name);");
    code.Add("cuGetErrorString(r, &str);");
    code.Add(
        "std::cerr << \"cuLaunchKernel failed: \" << (name?name:\"unknown\""
        ") << \" - \" << (str?str:\"unknown\""
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

void GpuCodeGenerator::GenerateDeclarationInHostCode(CodeBuilder &code)
{
    if (!do_inter_warp_lb) {
        // If inter warp load balancing is not enabled, we can skip the
        // declaration of global variables for inter warp load balancing
        return;
    }

    int num_warps = int(KernelConstants::DEFAULT_BLOCK_SIZE / 32) *
                    KernelConstants::DEFAULT_GRID_SIZE;
    std::string num_warps_str = std::to_string(num_warps);
    std::string min_num_warps =
        std::to_string(std::min(num_warps, kernel_args.min_num_warps));
    int bitmapsize_1 = 16;
    int bitmapsize_2 = (int((num_warps - 1) / 64) + 1) * 16;
    std::string bitmapsize_str = std::to_string(bitmapsize_1 + bitmapsize_2);
    std::string bitmapsize_2_str = std::to_string(bitmapsize_2);

    // Declare the basic variables for inter warp load balancing
    code.Add("unsigned int *global_info;");
    code.Add(
        "cudaMalloc((void **)&global_info, sizeof(unsigned int) * 2 * 32);");
    code.Add("unsigned int *global_num_idle_warps = global_info;");
    code.Add("int *global_scan_offset = (int *)(global_info + 32);");

    code.Add("Themis::StatisticsPerLvl *global_stats_per_lvl = NULL;");
    code.Add("Themis::InitStatisticsPerLvl(global_stats_per_lvl, " +
             num_warps_str + ");");
    code.Add("Themis::PushedParts::PushedPartsStack *gts;");
    code.Add("size_t size_of_stack_per_warp;");
    code.Add(
        "Themis::PushedParts::InitPushedPartsStack(gts, "
        "size_of_stack_per_warp, 1 << 31, " +
        num_warps_str + ");");
    code.Add("unsigned long long *global_bit1;");
    code.Add("cudaMalloc((void **)&global_bit1, sizeof(unsigned long long) * " +
             bitmapsize_str + ");");
    code.Add("unsigned long long *global_bit2 = global_bit1 + 16;");
    code.Add("cudaMemset(global_bit1, 0, sizeof(unsigned long long) * 16);");
    code.Add("cudaMemset(global_bit2, 0, sizeof(unsigned long long) * " +
             bitmapsize_2_str + ");");
}

void GpuCodeGenerator::GenerateKernelCallInHostCode(CodeBuilder &code)
{
    if (!do_inter_warp_lb) {
        code.Add("void *args[] = { input_data, output_data, &output_count };");
        code.Add("");

        code.Add(
            "CUresult r = cuLaunchKernel(gpu_kernel, gridSize,1,1, "
            "blockSize,1,1, 0, 0, args, nullptr);");
        return;
    } else {
        int num_warps = int(KernelConstants::DEFAULT_BLOCK_SIZE / 32) *
                        KernelConstants::DEFAULT_GRID_SIZE;
        std::string num_subpipes =
            std::to_string(pipeline_context.sub_pipelines.size());
        code.Add("cudaMemset(global_info, 0, 64 * sizeof(unsigned int));");
        code.Add(
            "cudaMemset(global_stats_per_lvl, 0, "
            "sizeof(Themis::StatisticsPerLvl) * " +
            num_subpipes + ");");
        // tableSize = pipe.subpipes[0].operators[0].genTableSize()
        int tableSize = 100000; // TODO
        // if tableSize[0] == '*': 
        //     // tableSize = tableSize[1:]                
        //     code.Add(f'Themis::InitStatisticsPerLvlPtr(global_stats_per_lvl, {num_warps}, {tableSize}, {len(pipe.subpipeSeqs)});')
        // else:
        code.Add("Themis::InitStatisticsPerLvlHost(global_stats_per_lvl, " +
                 std::to_string(num_warps) + ", " + std::to_string(tableSize) +
                 ", " + num_subpipes + ");");

        int bitmapsize = (int((num_warps - 1) / 64) + 1);
        std::string bitmapsize_str = std::to_string(16 + bitmapsize * 16);
        code.Add("cudaMemset(global_bit1, 0, sizeof(unsigned long long) * " +
                 bitmapsize_str + ");");

        code.Add(
            "void *args[] = { input_data, output_data, &output_count, "
            "&global_num_idle_warps, &global_scan_offset, &gts, "
            "&size_of_stack_per_warp,"
            "&global_bit1, &global_bit2 };");
        code.Add("");

        code.Add(
            "CUresult r = cuLaunchKernel(gpu_kernel, gridSize,1,1, "
            "blockSize,1,1, 0, 0, args, nullptr);");
    }
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

    if (pipeline.GetPipelineLength() == 0) {
        return;
    }

    auto source_op = pipeline.GetSource();
    auto source_it = operator_generators.find(source_op->GetOperatorType());
    if (source_it != operator_generators.end()) {
        source_it->second->GenerateInputKernelParameters(
            source_op, this, context, pipeline_context, input_kernel_params,
            scan_column_infos);
    }
    else {
        throw NotImplementedException(
            "Source operator type not implemented: " +
            std::to_string(static_cast<int>(source_op->GetOperatorType())));
    }

    // Add output parameters based on sink operator
    auto sink_op = pipeline.GetSink();
    auto sink_it = operator_generators.find(sink_op->GetOperatorType());
    if (sink_it != operator_generators.end()) {
        sink_it->second->GenerateOutputKernelParameters(
            sink_op, this, context, pipeline_context, output_kernel_params);
    }
    else {
        throw NotImplementedException(
            "Sink operator type not implemented: " +
            std::to_string(static_cast<int>(sink_op->GetOperatorType())));
    }
}

LogicalType GpuCodeGenerator::GetLogicalTypeFromId(LogicalTypeId type_id,
                                                   uint16_t extra_info)
{
    if (type_id == LogicalTypeId::DECIMAL) {
        uint8_t width = (uint8_t)(extra_info >> 8);
        uint8_t scale = (uint8_t)(extra_info & 0xFF);
        return LogicalType::DECIMAL(width, scale);
    }
    return LogicalType(type_id);
}

std::string GpuCodeGenerator::ConvertLogicalTypeIdToPrimitiveType(
    LogicalTypeId type_id, uint16_t extra_info)
{
    LogicalType type = GetLogicalTypeFromId(type_id, extra_info);
    return ConvertLogicalTypeToPrimitiveType(type);
}

std::string GpuCodeGenerator::ConvertLogicalTypeToPrimitiveType(
    LogicalType &type)
{
    PhysicalType p_type = type.InternalType();
    switch (p_type) {
        case PhysicalType::BOOL:
            return "bool ";
        case PhysicalType::INT8:
            return "char ";
        case PhysicalType::INT16:
            return "short ";
        case PhysicalType::INT32:
            return "int ";
        case PhysicalType::INT64:
            return "long long ";
        case PhysicalType::UINT8:
            return "unsigned char ";
        case PhysicalType::UINT16:
            return "unsigned short ";
        case PhysicalType::UINT32:
            return "unsigned int ";
        case PhysicalType::UINT64:
            return "unsigned long long ";
        case PhysicalType::FLOAT:
            return "float ";
        case PhysicalType::DOUBLE:
            return "double ";
        case PhysicalType::VARCHAR:
            // return "char *";
            return "str_t "; // TODO string_t?
        default:
            throw std::runtime_error("Unsupported logical type: " +
                                     std::to_string((uint8_t)type.id()));
    }
}

void GpuCodeGenerator::GenerateInputCode(CypherPipeline &sub_pipeline,
                                         CodeBuilder &code,
                                         PipelineContext &pipeline_ctx,
                                         PipeInputType input_type)
{
    switch (input_type) {
        case PipeInputType::TYPE_0:
            GenerateInputCodeForType0(sub_pipeline, code, pipeline_ctx);
            break;
        case PipeInputType::TYPE_1:
            GenerateInputCodeForType1(sub_pipeline, code, pipeline_ctx);
            break;
        case PipeInputType::TYPE_2:
            GenerateInputCodeForType2(sub_pipeline, code, pipeline_ctx);
            break;
        default:
            throw std::runtime_error("Unsupported input type");
    }
}

void GpuCodeGenerator::GenerateInputCodeForType0(CypherPipeline &sub_pipeline,
                                                 CodeBuilder &code,
                                                 PipelineContext &pipeline_ctx)
{
    int stepSize = KernelConstants::DEFAULT_BLOCK_SIZE *
                   KernelConstants::DEFAULT_GRID_SIZE;

    D_ASSERT(
        pipeline_ctx.sub_pipeline_tids[pipeline_ctx.current_sub_pipeline_index]
            .size() == 1);
    std::string tid_name =
        pipeline_ctx
            .sub_pipeline_tids[pipeline_ctx.current_sub_pipeline_index][0];

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
void GpuCodeGenerator::GenerateInputCodeForType1(CypherPipeline &sub_pipeline,
                                                 CodeBuilder &code,
                                                 PipelineContext &pipeline_ctx)
{
    throw NotImplementedException(
        "GenerateInputCodeForType1 is not implemented yet");
    // tid = spSeq.getTid()
    // lastOp = spSeq.pipe.subpipeSeqs[spSeq.id-1].subpipes[-1].operators[-1]
    // c = Code()
    // if self.doInterWarpLB and self.interWarpLbMethod == 'aws':
    //     code.Add(f'while (lvl >= {spSeq.id} && loop < {self.maxInterval}) ' + '{')
    // else:
    //     code.Add(f'while (lvl >= {spSeq.id}) ' + '{')
    // code.Add('__syncwarp();')
    // code.Add(f'if (lvl == {spSeq.id}) ' + '{')
    // code.Add(f'int loopvar{spSeq.id};')
    // code.Add(f'Themis::FillIPartAtLoopLvl({spSeq.id}, thread_id, active, loopvar{spSeq.id}, ts_{spSeq.id}_range_cached, mask_32, mask_1);')
    
    // # Declare in-boundary attrs
    // code.Add(f'//{list(attrsToDeclareAndMaterialize)}')
    // code.Add(f'int {tid.id_name};')
    // for attrId, attr in spSeq.inBoundaryAttrs.items():
    //     if attrId == tid.id: continue
    //     code.Add(f'{langType(attr.dataType)} {attr.id_name};')
        
    
    // _, attrsToLoadBeforeExec, _, _ = attrsToDeclareAndMaterialize[0]
    // code.Add(self.genAttrDeclaration(spSeq.pipe, attrsToLoadBeforeExec))
    
    // code.Add('if (active) {')

    // # Load in-boundary attrs
    // code.Add(f'// last op generating Attrs: {lastOp.generatingAttrs}')
    // var = f'loopvar{spSeq.id}'
    // code.Add(f'{tid.id_name} = {spSeq.convertTid(var)};')
    // for attrId, attr in spSeq.inBoundaryAttrs.items():
    //     if attrId == tid.id: continue
    //     if attrId in lastOp.generatingAttrs:
    //         code.Add(f'{attr.id_name} = {spSeq.pipe.originExpr[attrId]};')
    //     else:
    //         code.Add(f'{attr.id_name} = ts_{spSeq.id}_{attr.id_name}_cached;')
        
    // code.Add(self.genAttrLoad(spSeq.pipe, attrsToLoadBeforeExec))

    // code.Add('}')
    // code.Add('int ts_src = 32;')
    // code.Add(f'bool is_updated = Themis::DistributeFromPartToDPart(thread_id, {spSeq.id}, ts_src, ts_{spSeq.id}_range, ts_{spSeq.id}_range_cached, mask_32, mask_1);')
    // if self.doInterWarpLB and self.interWarpLbMethod == 'aws' and self.doWorkoadSizeTracking:
    //     code.Add(f'Themis::WorkloadTracking::UpdateWorkloadSizeAtLoopLvl(thread_id, {spSeq.id}, ++loop, ts_{spSeq.id}_range, ts_{spSeq.id}_range_cached, mask_1, local_info, global_stats_per_lvl);')
    // code.Add('if (is_updated) {')
    // for attrId, attr in spSeq.inBoundaryAttrs.items():
    //     if attrId == tid.id: continue
    //     if attrId in lastOp.generatingAttrs: continue
    //     code.Add('{')
    //     name = f'ts_{spSeq.id}_{attr.id_name}'
    //     if attr.dataType == Type.STRING:
    //         code.Add(f'char* start = (char*) __shfl_sync(ALL_LANES, (uint64_t) {name}.start, ts_src);')
    //         code.Add(f'char* end = (char*) __shfl_sync(ALL_LANES, (uint64_t) {name}.end, ts_src);')
    //         code.Add('if (ts_src < 32) {')
    //         code.Add(f'{name}_cached.start = start;')
    //         code.Add(f'{name}_cached.end = end;')
    //         code.Add('}')
    //     elif attr.dataType == Type.PTR_INT:
    //         code.Add(f'uint64_t cache = __shfl_sync(ALL_LANES, (uint64_t){name}, ts_src);')
    //         code.Add(f'if (ts_src < 32) {name}_cached = (int*) cache;')
    //     else:
    //         code.Add(f'{langType(attr.dataType)} cache = __shfl_sync(ALL_LANES, {name}, ts_src);')
    //         code.Add(f'if (ts_src < 32) {name}_cached = cache;')
    //     code.Add('}')
    // code.Add('}')
    // return c
}
void GpuCodeGenerator::GenerateInputCodeForType2(CypherPipeline &sub_pipeline,
                                                 CodeBuilder &code,
                                                 PipelineContext &pipeline_ctx)
{
    int subpipe_id = pipeline_ctx.current_sub_pipeline_index;
    auto &tids = pipeline_ctx.sub_pipeline_tids[subpipe_id - 1];
    code.Add("if (lvl == " + std::to_string(subpipe_id) + ") {");
    code.IncreaseNesting();
    if (tids.size() > 0) {
        code.Add("if (!(mask_32 & (0x1u << " + std::to_string(subpipe_id) +
                 "))) {");
        code.IncreaseNesting();
        for (const auto &tid : tids) {
            code.Add("ts_" + std::to_string(subpipe_id) + "_" + tid + " = ts_" +
                     std::to_string(subpipe_id) + "_" + tid + "_flushed;");
        }
        code.DecreaseNesting();
        code.Add("}");
    }

    code.Add("Themis::FillIPartAtIfLvl(" + std::to_string(subpipe_id) +
             ", thread_id, inodes_cnts, active, mask_32, mask_1);");
    // if self.doInterWarpLB and self.interWarpLbMethod == 'aws' and self.doWorkoadSizeTracking:
    code.Add("Themis::WorkloadTracking::UpdateWorkloadSizeAtIfLvl(thread_id, " +
             std::to_string(subpipe_id) +
             ", loop, inodes_cnts, mask_1, local_info, "
             "global_stats_per_lvl);");

    for (const auto &tid : tids) {
        code.Add("int " + tid + ";");
    }
    code.Add("if (active) {");
    code.IncreaseNesting();
    for (const auto &tid : tids) {
        code.Add(tid + " = ts_" + std::to_string(subpipe_id) + "_" + tid + ";");
    }
    code.DecreaseNesting();
    code.Add("}");
}

void GpuCodeGenerator::GenerateOutputCode(CypherPipeline &sub_pipeline,
                                          CodeBuilder &code,
                                          PipelineContext &pipeline_ctx,
                                          PipeOutputType output_type)
{
    switch (output_type) {
        case PipeOutputType::TYPE_0:
            GenerateOutputCodeForType0(sub_pipeline, code, pipeline_ctx);
            break;
        case PipeOutputType::TYPE_1:
            GenerateOutputCodeForType1(sub_pipeline, code, pipeline_ctx);
            break;
        case PipeOutputType::TYPE_2:
            GenerateOutputCodeForType2(sub_pipeline, code, pipeline_ctx);
            break;
        default:
            throw std::runtime_error("Unsupported output type");
    }
}

void GpuCodeGenerator::GenerateOutputCodeForType0(CypherPipeline &sub_pipeline,
                                                  CodeBuilder &code,
                                                  PipelineContext &pipeline_ctx)
{
    std::string loop_lvl = std::to_string(FindLowerLoopLvl(pipeline_ctx));
    code.Add("if (mask_32 & (0x1 << " + loop_lvl + ")) lvl = " + loop_lvl +
             ";");
    code.Add("else Themis::ChooseLvl(thread_id, mask_32, mask_1, lvl);");
    code.DecreaseNesting();
    code.Add("}");
}

void GpuCodeGenerator::GenerateOutputCodeForType1(CypherPipeline &sub_pipeline,
                                                  CodeBuilder &code,
                                                  PipelineContext &pipeline_ctx)
{
    throw NotImplementedException(
        "GenerateOutputCodeForType1 is not implemented yet");
    // lastOp = spSeq.subpipes[-1].operators[-1]
    // opId = lastOp.opId
    // tid = lastOp.tid
    // #attrsGeneratedByLastOp = lastOp.generatingAttrs
    
    // currentMaterializedAttrs = attrsToDeclareAndMaterialize[-1][-1]

    // attrsToDeclare = {}
    // for attrId, attr in spSeq.outBoundaryAttrs.items():
    //     if attrId in lastOp.generatingAttrs: continue
    //     if attrId in currentMaterializedAttrs: continue
    //     attrsToDeclare[attrId] = attr
    
    // code.Add(self.genAttrDeclaration(spSeq.pipe, attrsToDeclare))
    
    // code.Add('unsigned push_active_mask = __ballot_sync(ALL_LANES, active);')
    
    // if self.doInterWarpLB and self.interWarpLbMethod == 'ws':
    //     # Find attributes to push
    //     attrs = {}
    //     for attrId, attr in spSeq.outBoundaryAttrs.items():
    //         if attrId == tid.id: continue
    //         if attrId in lastOp.generatingAttrs: continue
    //         attrs[attrId] = attr
    //     code.Add(self.genWorkSharingPushCode(spSeq, attrs))
    
    // code.Add('if (push_active_mask) {')
    // # generated by the last Op?
    
    // code.Add(f'//{currentMaterializedAttrs}')
    // code.Add('if (active) {')
    // code.Add(f'ts_{spSeq.id+1}_range.set(local{opId}_range);')
    
    // for attrId, attr in spSeq.outBoundaryAttrs.items():
    //     if attrId in lastOp.generatingAttrs: continue
    //     if attrId in currentMaterializedAttrs:
    //         code.Add(f'ts_{spSeq.id+1}_{attr.id_name} = {attr.id_name};')
    //     elif attrId in spSeq.inBoundaryAttrs:
    //         code.Add(f'ts_{spSeq.id+1}_{attr.id_name} = ts_{spSeq.id}_{attr.id_name};')
    //     else:
    //         code.Add(f'ts_{spSeq.id+1}_{attr.id_name} = {spSeq.pipe.originExpr[attrId]};')
    // code.Add('}')
    // code.Add('int ts_src = 32;')
    // code.Add(f'Themis::DistributeFromPartToDPart(thread_id, {spSeq.id+1}, ts_src, ts_{spSeq.id+1}_range, ts_{spSeq.id+1}_range_cached);')
    // code.Add(f'Themis::UpdateMaskAtLoopLvl({spSeq.id+1}, ts_{spSeq.id+1}_range_cached, mask_32, mask_1);')
    // for attrId, attr in spSeq.outBoundaryAttrs.items():
    //     if attrId in lastOp.generatingAttrs: continue
    //     code.Add('{')
    //     name = f'ts_{spSeq.id+1}_{attr.id_name}'
    //     if attr.dataType == Type.STRING:
    //         code.Add(f'char* start = (char*) __shfl_sync(ALL_LANES, (uint64_t) {name}.start, ts_src);')
    //         code.Add(f'char* end = (char*) __shfl_sync(ALL_LANES, (uint64_t) {name}.end, ts_src);')
    //         code.Add('if (ts_src < 32) {')
    //         code.Add(f'{name}_cached.start = start;')
    //         code.Add(f'{name}_cached.end = end;')
    //         code.Add('}')
    //     elif attr.dataType == Type.PTR_INT:
    //         code.Add(f'uint64_t cache = __shfl_sync(ALL_LANES, (uint64_t){name}, ts_src);')
    //         code.Add(f'if (ts_src < 32) {name}_cached = (int*) cache;')
    //     else:
    //         code.Add(f'{langType(attr.dataType)} cache = __shfl_sync(ALL_LANES, {name}, ts_src);')
    //         code.Add(f'if (ts_src < 32) {name}_cached = cache;')
    //     code.Add('}')
    // code.Add('}')
    
    // # Find the outermost loop level
    // loop_lvl = self.findLowerLoopLvl(spSeq)
    // code.Add(f'if (!(mask_32 & (0x1 << {spSeq.id+1})))' + '{')
    // code.Add(f'if (mask_32 & (0x1 << {loop_lvl})) lvl = {loop_lvl};')
    // code.Add('else Themis::ChooseLvl(thread_id, mask_32, mask_1, lvl);')
    // code.Add('continue;')
    // code.Add('}')
    // code.Add(f'lvl = {spSeq.id+1};')
    
    // code.Add('}')
}

void GpuCodeGenerator::GenerateOutputCodeForType2(CypherPipeline &sub_pipeline,
                                                  CodeBuilder &code,
                                                  PipelineContext &pipeline_ctx)
{
    std::string next_subpipe_id =
        std::to_string(pipeline_ctx.current_sub_pipeline_index + 1);
    auto &tids = pipeline_ctx.sub_pipeline_tids[
        pipeline_ctx.current_sub_pipeline_index];

    // if len(spSeq.outBoundaryAttrs) > 0:
    //     currentMaterializedAttrs = attrsToDeclareAndMaterialize[-1][-1]
    //     code.Add(f'// {currentMaterializedAttrs}')
    //     attrs = {}
    //     for attrId, attr in spSeq.outBoundaryAttrs.items():
    //         if attrId in currentMaterializedAttrs: continue
    //         #if isinstance(lastOp, HashJoin) and attrId == lastOp.tid.id: continue 
    //         attrs[attrId] = attr            
    //     code.Add(self.genAttrDeclaration(spSeq.pipe, attrs))
    //     if len(attrs) > 0:
    //         code.Add('if (active) {')
    //         for attrId, attr in attrs.items():
    //             if attrId in spSeq.inBoundaryAttrs:
    //                 code.Add(f'{attr.id_name} = ts_{spSeq.id}_{attr.id_name};')
    //             else:
    //                 code.Add(f'{attr.id_name} = {spSeq.pipe.originExpr[attrId]};')
    //         code.Add('}')

    code.Add("unsigned push_active_mask = __ballot_sync(ALL_LANES, active);");
    code.Add("if (push_active_mask) {");
    code.IncreaseNesting();
    code.Add("int old_ts_cnt = __shfl_sync(ALL_LANES, inodes_cnts, " +
             next_subpipe_id + ");");
    code.Add("int ts_cnt = old_ts_cnt + __popc(push_active_mask);");
    code.Add("if (thread_id == " + next_subpipe_id + ") inodes_cnts = ts_cnt;");
    code.Add("Themis::UpdateMaskAtIfLvlAfterPush(" + next_subpipe_id +
             ", ts_cnt, mask_32, mask_1);");
    if (tids.size() > 0) {
        code.Add("if (ts_cnt >=32) {");
        code.IncreaseNesting();
        for (const auto &tid : tids) {
            code.Add("ts_" + next_subpipe_id + "_" + tid + " = " + tid + ";");
        }
        code.Add("if (ts_cnt - old_ts_cnt < 32) {");
        code.IncreaseNesting();
        code.Add("unsigned ts_src = 32;");
        code.Add("if (!active) ts_src = old_ts_cnt - __popc((~push_active_mask) & prefixlanes) - 1;");
        for (const auto &tid : tids) {
            std::string name = "ts_" + next_subpipe_id + "_" + tid;
            code.Add("{");
            code.IncreaseNesting();
            // TODO additional types
            // if attr.dataType == Type.STRING:
            //     code.Add(f'char* start = (char*) __shfl_sync(ALL_LANES, (uint64_t) {name}_flushed.start, ts_src);')
            //     code.Add(f'char* end = (char*) __shfl_sync(ALL_LANES, (uint64_t) {name}_flushed.end, ts_src);')
            //     code.Add('if (ts_src < 32) {')
            //     code.Add(f'{name}.start = start;')
            //     code.Add(f'{name}.end = end;')
            //     code.Add('}')
            // elif attr.dataType == Type.PTR_INT:
            //     code.Add(f'uint64_t cache = __shfl_sync(ALL_LANES, (uint64_t){name}_flushed, ts_src);')
            //     code.Add(f'if (ts_src < 32) {name} = (int*) cache;')
            // else:
            //     code.Add(f'{langType(attr.dataType)} cache = __shfl_sync(ALL_LANES, {name}_flushed, ts_src);')
            //     code.Add(f'if (ts_src < 32) {name} = cache;')
            code.Add("int cache = __shfl_sync(ALL_LANES, " + name +
                     "_flushed, ts_src);");
            code.Add("if (ts_src < 32) " + name + " = cache;");
            code.DecreaseNesting();
            code.Add("}");
        }
        code.DecreaseNesting();
        code.Add("}");
        code.DecreaseNesting();
        code.Add("} else {");
        code.IncreaseNesting();
        code.Add("active_thread_ids[threadIdx.x] = 32;");
        code.Add(
            "int *src_thread_ids = active_thread_ids + ((threadIdx.x >> 5) << "
            "5);");
        code.Add(
            "if (active) src_thread_ids[__popc(push_active_mask & "
            "prefixlanes)] = thread_id;");
        code.Add(
            "unsigned ts_src = thread_id >= old_ts_cnt && thread_id < ts_cnt ? "
            "src_thread_ids[thread_id - old_ts_cnt] : 32;");
        for (const auto &tid : tids) {
            std::string name = "ts_" + next_subpipe_id + "_" + tid;
            code.Add("{");
            code.IncreaseNesting();
            // TODO additional types
            // if attr.dataType == Type.STRING:
            //     code.Add(f'char* start = (char*) __shfl_sync(ALL_LANES, (uint64_t) {attr.id_name}.start, ts_src);')
            //     code.Add(f'char* end = (char*) __shfl_sync(ALL_LANES, (uint64_t) {attr.id_name}.end, ts_src);')
            //     code.Add('if (ts_src < 32) {')
            //     code.Add(f'{name}_flushed.start = start;')
            //     code.Add(f'{name}_flushed.end = end;')
            //     code.Add('}')
            // elif attr.dataType == Type.PTR_INT:
            //     code.Add(f'uint64_t cache = __shfl_sync(ALL_LANES, (uint64_t){attr.id_name}, ts_src);')
            //     code.Add(f'if (ts_src < 32) {name}_flushed = (int*) cache;')
            // else:
            //     code.Add(f'{langType(attr.dataType)} cache = __shfl_sync(ALL_LANES, {attr.id_name}, ts_src);')
            //     code.Add(f'if (ts_src < 32) {name}_flushed = cache;')
            code.Add("int cache = __shfl_sync(ALL_LANES, " + tid +
                     ", ts_src);");
            code.Add("if (ts_src < 32) " + name + "_flushed = cache;");
            code.DecreaseNesting();
            code.Add("}");
        }
        code.DecreaseNesting();
        code.Add("}");
    }
    code.DecreaseNesting();
    code.Add("} // push active mask");

    // Find the outermost loop level
    std::string loop_lvl = std::to_string(FindLowerLoopLvl(pipeline_ctx));
    code.Add("if (!(mask_32 & (0x1 << " + next_subpipe_id + "))) {");
    code.IncreaseNesting();
    code.Add("if (mask_32 & (0x1 << " + loop_lvl + ")) lvl = " + loop_lvl +
             ";");
    code.Add("else Themis::ChooseLvl(thread_id, mask_32, mask_1, lvl);");
    code.Add("continue;");
    code.DecreaseNesting();
    code.Add("}");
    code.Add("lvl = " + next_subpipe_id + ";");
}

int GpuCodeGenerator::FindLowerLoopLvl(PipelineContext &pipeline_context)
{
    int loop_lvl = pipeline_context.current_sub_pipeline_index;
    while (true) {
        PipeInputType inputType =
            GetPipeInputType(pipeline_context.sub_pipelines[loop_lvl]);
        if (inputType == PipeInputType::TYPE_0 ||
            inputType == PipeInputType::TYPE_1) {
            return loop_lvl;
        }
        loop_lvl--;
    }
}

PipeInputType GpuCodeGenerator::GetPipeInputType(CypherPipeline &sub_pipeline)
{
    PipeInputType pipe_input_type;
    auto first_op = sub_pipeline.GetSource();
    if (first_op->GetOperatorType() == PhysicalOperatorType::NODE_SCAN) {
        auto scan_op = dynamic_cast<PhysicalNodeScan *>(first_op);
        if (scan_op->is_filter_pushdowned) {
            if (sub_pipeline.GetSink() == first_op) {
                pipe_input_type = PipeInputType::TYPE_0;
            }
            else {
                pipe_input_type = PipeInputType::TYPE_2;
            }
        }
        else {
            pipe_input_type = PipeInputType::TYPE_0;
        }
    }
    else if (first_op->GetOperatorType() == PhysicalOperatorType::FILTER) {
        pipe_input_type = PipeInputType::TYPE_2;
    }
    else if (first_op->GetOperatorType() == PhysicalOperatorType::HASH_AGGREGATE) {
        pipe_input_type = PipeInputType::TYPE_0;
    }
    else {
        throw NotImplementedException("Input type");
    }
    return pipe_input_type;
}

PipeOutputType GpuCodeGenerator::GetPipeOutputType(CypherPipeline &sub_pipeline)
{
    PipeOutputType pipe_output_type;
    auto last_op = sub_pipeline.GetSink();
    switch (last_op->GetOperatorType()) {
        case PhysicalOperatorType::PRODUCE_RESULTS:
            // TODO correctness check
            pipe_output_type = PipeOutputType::TYPE_0;
            break;
        case PhysicalOperatorType::FILTER:
            pipe_output_type = PipeOutputType::TYPE_2;
            break;
        case PhysicalOperatorType::NODE_SCAN:
            // Filter pushdowned scan case
            D_ASSERT(dynamic_cast<PhysicalNodeScan *>(last_op)
                         ->is_filter_pushdowned);
            pipe_output_type = PipeOutputType::TYPE_2;
            break;
        case PhysicalOperatorType::HASH_AGGREGATE:
            pipe_output_type = PipeOutputType::TYPE_0;
            break;
        default:
            throw NotImplementedException("Output type");
    }
    return pipe_output_type;
}

void GpuCodeGenerator::GenerateCodeForMaterialization(
    CodeBuilder &code, PipelineContext &pipeline_context)
{
    // Declaration only in this point
    int cur_subpipe_idx = pipeline_context.current_sub_pipeline_index;
    auto &cur_cols_to_be_materialized =
        pipeline_context.columns_to_be_materialized[cur_subpipe_idx];
    for (auto i = 0; i < cur_cols_to_be_materialized.size(); i++) {
        auto column = cur_cols_to_be_materialized[i].name;
        if (pipeline_context.column_materialized.find(column) !=
            pipeline_context.column_materialized.end()) {
            // If the column is already materialized, skip it
            continue;
        }
        code.Add("// Materialize column: " + column);
        std::string col_name = column;
        std::replace(col_name.begin(), col_name.end(), '.', '_');
        std::string ctype = ConvertLogicalTypeToPrimitiveType(
            cur_cols_to_be_materialized[i].type);
        code.Add(ctype + col_name + ";");
        // pipeline_context.column_materialized[column] = true;
    }
    // int cur_subpipe_idx = pipeline_context.current_sub_pipeline_index;
    // auto &cur_cols_to_be_materialized =
    //     pipeline_context.columns_to_be_materialized[cur_subpipe_idx];
    // auto &cur_col_types_to_be_materialized =
    //     pipeline_context.column_types_to_be_materialized[cur_subpipe_idx];
    // for (auto i = 0; i < cur_cols_to_be_materialized.size();
    //      i++) {
    //     auto column = cur_cols_to_be_materialized[i];
    //     if (pipeline_context.column_materialized.find(column) !=
    //         pipeline_context.column_materialized.end()) {
    //         // If the column is already materialized, skip it
    //         continue;
    //     }
    //     code.Add("// Materialize column: " + column);
    //     std::string col_name = column;
        
    //     std::string sanitized_col_name;
    //     for (char c : col_name) {
    //         if (std::isalnum(c) || c == '_') {
    //             sanitized_col_name += c;
    //         } else {
    //             sanitized_col_name += '_';
    //         }
    //     }
        
    //     if (!sanitized_col_name.empty() && std::isdigit(sanitized_col_name[0])) {
    //         sanitized_col_name = "col_" + sanitized_col_name;
    //     }
        
    //     if (sanitized_col_name.empty()) {
    //         sanitized_col_name = "col_unknown";
    //     }
        
    //     std::string ctype = ConvertLogicalTypeToPrimitiveType(
    //         cur_col_types_to_be_materialized[i]);
    //     code.Add(ctype + sanitized_col_name + ";");
    //     pipeline_context.column_materialized[column] = true;
    // }
}

void GpuCodeGenerator::GeneratePipelineCode(CypherPipeline &pipeline,
                                            CodeBuilder &code)
{
    for (size_t i = 0; i < pipeline_context.sub_pipelines.size(); i++) {
        std::cerr << "Generating code for sub-pipeline " << i << std::endl;
        auto &sub_pipeline = pipeline_context.sub_pipelines[i];
        pipeline_context.current_sub_pipeline_index = i;

        // Generate code for each sub-pipeline
        GenerateSubPipelineCode(sub_pipeline, code);
    }
}

void GpuCodeGenerator::GenerateSubPipelineCode(CypherPipeline &sub_pipeline,
                                               CodeBuilder &code)
{
    // Analyze pipeline input/output type
    PipeInputType pipe_input_type = GetPipeInputType(sub_pipeline);
    PipeOutputType pipe_output_type = GetPipeOutputType(sub_pipeline);
    
    // Advance to the next operator
    AdvanceOperator();

    // Generate input code based on the source operator type
    GenerateInputCode(sub_pipeline, code, pipeline_context, pipe_input_type);

    // Generate code for materialization if needed
    GenerateCodeForMaterialization(code, pipeline_context);
    
    // Generate code for sub-pipeline operators
    if (pipe_input_type == PipeInputType::TYPE_0) {
        auto first_op = sub_pipeline.GetSource();
        code.Add("if (active) {");
        code.IncreaseNesting();
        GenerateOperatorCode(first_op, code, pipeline_context,
                             /*is_main_loop=*/true);

        ProcessRemainingOperators(sub_pipeline, 1, code);
        code.DecreaseNesting();
        code.Add("} // end of active");
    } else {
        auto second_op = sub_pipeline.GetIdxOperator(1);
        code.Add("if (active) {");
        code.IncreaseNesting();
        GenerateOperatorCode(second_op, code, pipeline_context,
                             /*is_main_loop=*/false);

        ProcessRemainingOperators(sub_pipeline, 2, code);
        code.DecreaseNesting();
        code.Add("} // end of active");
    }

    // Generate output code based on the sink operator type
    GenerateOutputCode(sub_pipeline, code, pipeline_context, pipe_output_type);

    code.DecreaseNesting();
    code.Add("}");
}

void GpuCodeGenerator::ProcessRemainingOperators(CypherPipeline &pipeline,
                                                 int op_idx, CodeBuilder &code)
{
    if (op_idx >= pipeline.GetPipelineLength()) {
        return;
    }

    auto op = pipeline.GetIdxOperator(op_idx);

    // Move to current operator and update schemas
    AdvanceOperator();

    GenerateOperatorCode(op, code, pipeline_context,
                         /*is_main_loop=*/false);
    ProcessRemainingOperators(pipeline, op_idx + 1, code);
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

void GpuCodeGenerator::GenerateGlobalDeclarations(CypherPhysicalOperator *op,
                                                  size_t op_idx,
                                                  CodeBuilder &code,
                                                  PipelineContext &pipeline_ctx)
{
    auto it = operator_generators.find(op->GetOperatorType());
    if (it != operator_generators.end()) {
        it->second->GenerateGlobalDeclarations(op, op_idx, code, this, context,
                                               pipeline_ctx);
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

    // currently, handle main loop case only
    D_ASSERT(is_main_loop);
    code.Add("// Scan operator");

    // Process oids and scan_projection_mapping to get chunk IDs
    for (size_t oid_idx = 0; oid_idx < scan_op->oids.size(); oid_idx++) {
        idx_t oid = scan_op->oids[oid_idx];
        std::string graphlet_name = "gr" + std::to_string(oid);

        // Get property schema catalog entry using oid
        Catalog &catalog = context.db->GetCatalog();
        PropertySchemaCatalogEntry *property_schema_cat_entry =
            (PropertySchemaCatalogEntry *)catalog.GetEntry(context,
                                                           DEFAULT_SCHEMA, oid);
        D_ASSERT(property_schema_cat_entry != nullptr);

        auto *property_types_id = property_schema_cat_entry->GetTypes();
        auto *extra_infos = property_schema_cat_entry->GetExtraTypeInfos();

        // We need to generate code for materializing required columns
        auto &columns_to_materialize =
            pipeline_ctx.columns_to_be_materialized
                [pipeline_ctx.current_sub_pipeline_index];
        std::unordered_map<uint64_t, std::string> column_pos_to_name;
        for (size_t i = 0; i < columns_to_materialize.size(); i++) {
            auto &col_name = columns_to_materialize[i].name;
            auto col_type = columns_to_materialize[i].type;
            auto col_pos = columns_to_materialize[i].pos;
            auto col_id = scan_op->scan_projection_mapping[0][col_pos];

            // Generate code for materializing this column
            D_ASSERT(pipeline_ctx.attribute_tid_mapping.find(col_name) !=
                     pipeline_ctx.attribute_tid_mapping.end());
            std::string tid_name = pipeline_ctx.attribute_tid_mapping[col_name];

            std::string sanitized_col_name;
            for (char c : col_name) {
                if (std::isalnum(c) || c == '_') {
                    sanitized_col_name += c;
                }
                else {
                    sanitized_col_name += '_';
                }
            }

            if (!sanitized_col_name.empty() &&
                std::isdigit(sanitized_col_name[0])) {
                sanitized_col_name = "col_" + sanitized_col_name;
            }

            if (sanitized_col_name.empty()) {
                sanitized_col_name = "col_" + std::to_string(i);
            }

            code.Add(sanitized_col_name + " = " + graphlet_name + "_col_" +
                     std::to_string(col_id - 1) + "_data[" + tid_name + "];");
            column_pos_to_name[col_pos] = sanitized_col_name;
        }

        if (scan_op->is_filter_pushdowned) {
            code.Add("// Pushdowned filter for scan operator");
            std::string predicate_string = "";
            // Generate predicate string for filter pushdown
            if (scan_op->filter_pushdown_type == FilterPushdownType::FP_EQ) {
                for (auto i = 0;
                     i < scan_op->filter_pushdown_key_idxs_in_output.size();
                     i++) {
                    if (i > 0) {
                        predicate_string += " && ";
                    }
                    auto key_idx =
                        scan_op->filter_pushdown_key_idxs_in_output[i];
                    if (key_idx < 0)
                        continue;

                    auto it = column_pos_to_name.find(key_idx);
                    if (it == column_pos_to_name.end()) {
                        throw std::runtime_error(
                            "Column position not found in materialized "
                            "columns");
                    }
                    std::string attr_name = it->second;
                    auto value_str = scan_op->eq_filter_pushdown_values[i]
                                         .ToPhysicalTypeString();
                    predicate_string += (attr_name + " == " + value_str);
                }
            }
            else if (scan_op->filter_pushdown_type ==
                     FilterPushdownType::FP_RANGE) {
                for (auto i = 0;
                     i < scan_op->filter_pushdown_key_idxs_in_output.size();
                     i++) {
                    if (i > 0) {
                        predicate_string += " && ";
                    }
                    auto key_idx =
                        scan_op->filter_pushdown_key_idxs_in_output[i];
                    if (key_idx < 0)
                        continue;

                    auto it = column_pos_to_name.find(key_idx);
                    if (it == column_pos_to_name.end()) {
                        throw std::runtime_error(
                            "Column position not found in materialized "
                            "columns");
                    }
                    std::string attr_name = it->second;

                    auto left_value_str =
                        scan_op->range_filter_pushdown_values[i]
                            .l_value.ToPhysicalTypeString();
                    auto right_value_str =
                        scan_op->range_filter_pushdown_values[i]
                            .r_value.ToPhysicalTypeString();
                    predicate_string +=
                        scan_op->range_filter_pushdown_values[i].l_inclusive
                            ? (attr_name + " >= " + left_value_str)
                            : (attr_name + " > " + left_value_str);
                    predicate_string += " && ";
                    predicate_string +=
                        scan_op->range_filter_pushdown_values[i].r_inclusive
                            ? (attr_name + " <= " + right_value_str)
                            : (attr_name + " < " + right_value_str);
                }
            }
            else {  // FP_COMPLEX
                ExpressionCodeGenerator expr_gen(context);
                predicate_string = expr_gen.GenerateConditionCode(
                    scan_op->filter_expression.get(), code, pipeline_ctx,
                    column_pos_to_name);
            }

            code.Add("active = (" + predicate_string + ");");
        }
    }
}

void NodeScanCodeGenerator::GenerateGlobalDeclarations(
    CypherPhysicalOperator *op, size_t op_idx, CodeBuilder &code,
    GpuCodeGenerator *code_gen, ClientContext &context,
    PipelineContext &pipeline_ctx)
{
    return;
}

void NodeScanCodeGenerator::GenerateInputKernelParameters(
    CypherPhysicalOperator *op, GpuCodeGenerator *code_gen,
    ClientContext &context, PipelineContext &pipeline_context,
    std::vector<KernelParam> &input_kernel_params,
    std::vector<ScanColumnInfo> &scan_column_infos)
{
    auto scan_op = dynamic_cast<PhysicalNodeScan *>(op);

    // Process oids to get table/column information
    for (size_t oid_idx = 0; oid_idx < scan_op->oids.size(); oid_idx++) {
        idx_t oid = scan_op->oids[oid_idx];
        std::string graphlet_name = "gr" + std::to_string(oid);

        // Get property schema catalog entry using oid
        Catalog &catalog = context.db->GetCatalog();
        PropertySchemaCatalogEntry *property_schema_cat_entry =
            (PropertySchemaCatalogEntry *)catalog.GetEntry(context,
                                                           DEFAULT_SCHEMA, oid);
        D_ASSERT(property_schema_cat_entry != nullptr);

        auto *column_names = property_schema_cat_entry->GetKeys();
        auto *property_types_id = property_schema_cat_entry->GetTypes();
        auto *extra_infos = property_schema_cat_entry->GetExtraTypeInfos();
        scan_column_infos.push_back(ScanColumnInfo());
        ScanColumnInfo &scan_column_info = scan_column_infos.back();
        uint64_t num_extents = property_schema_cat_entry->extent_ids.size();
        bool is_first_time_to_get_column_info = true;

        scan_column_info.graphlet_id = oid;
        scan_column_info.extent_ids.reserve(num_extents);
        scan_column_info.num_tuples_per_extent.reserve(num_extents);

        // Get extent IDs from the property schema
        for (size_t extent_idx = 0; extent_idx < num_extents; extent_idx++) {
            idx_t extent_id = property_schema_cat_entry->extent_ids[extent_idx];
            scan_column_info.extent_ids.push_back((ExtentID)extent_id);

            // Generate table name
            std::string table_name =
                "gr" + std::to_string(oid) + "_ext" + std::to_string(extent_id);

            // Get extent catalog entry to access chunks (columns)
            ExtentCatalogEntry *extent_cat_entry =
                (ExtentCatalogEntry *)catalog.GetEntry(
                    context, CatalogType::EXTENT_ENTRY, DEFAULT_SCHEMA,
                    DEFAULT_EXTENT_PREFIX + std::to_string(extent_id));
            D_ASSERT(extent_cat_entry != nullptr);

            uint64_t num_tuples_in_extent =
                extent_cat_entry->GetNumTuplesInExtent();
            scan_column_info.num_tuples_per_extent.push_back(
                num_tuples_in_extent);

            if (is_first_time_to_get_column_info) {
                scan_column_info.chunk_ids.resize(
                    scan_op->scan_projection_mapping[oid_idx].size());
            }

            // Each chunk in the extent represents a column
            for (size_t chunk_idx = 0;
                 chunk_idx < scan_op->scan_projection_mapping[oid_idx].size();
                 chunk_idx++) {
                auto column_idx =
                    scan_op->scan_projection_mapping[oid_idx][chunk_idx];
                std::string col_name;
                if (column_idx == 0) {  // _id column
                    col_name = "col__id";
                    // col_name = "col_" + std::to_string(chunk_idx);
                    scan_column_info.get_physical_id_column = true;
                }
                else {
                    ChunkDefinitionID cdf_id =
                        extent_cat_entry->chunks[column_idx - 1];
                    // Generate column name based on chunk index
                    col_name = "col_" + std::to_string(column_idx - 1);
                    // col_name = "col_" + std::to_string(chunk_idx);

                    // Generate parameter names based on verbose mode
                    std::string param_name;
                    param_name = table_name + "_" + col_name;

                    // // Add data buffer parameter for this column (chunk)
                    // KernelParam data_param;
                    // data_param.name = param_name + "_data";
                    // data_param.type = "void *";
                    // data_param.is_device_ptr = true;
                    // input_kernel_params.push_back(data_param);

                    // Add pointer mapping for this chunk (column)
                    std::string chunk_name = "chunk_" + std::to_string(cdf_id);
                    code_gen->AddPointerMapping(chunk_name, nullptr, cdf_id);

                    scan_column_info.chunk_ids[chunk_idx].push_back(cdf_id);
                }

                if (is_first_time_to_get_column_info) {
                    // Store column information for the first time
                    scan_column_info.col_position.push_back(column_idx);
                    scan_column_info.col_name.push_back(col_name);

                    KernelParam data_param;
                    data_param.name = graphlet_name + "_" + col_name + "_data";

                    if (column_idx == 0) {
                        scan_column_info.col_type_size.push_back(
                            GetTypeIdSize(PhysicalType::UINT64));
                        pipeline_context
                            .column_to_param_mapping["_id"] =
                            graphlet_name + "_" + col_name;
                        data_param.type = "unsigned long long *";
                    }
                    else {
                        LogicalTypeId type_id =
                            (LogicalTypeId)property_types_id->at(column_idx -
                                                                 1);
                        uint16_t extra_info = extra_infos->at(column_idx - 1);
                        LogicalType type =
                            code_gen->GetLogicalTypeFromId(type_id, extra_info);
                        uint64_t type_size = GetTypeIdSize(type.InternalType());
                        scan_column_info.col_type_size.push_back(type_size);

                        pipeline_context
                            .column_to_param_mapping[column_names->at(
                                column_idx - 1)] =
                            graphlet_name + "_" + col_name;
                        data_param.type =
                            code_gen->ConvertLogicalTypeToPrimitiveType(type) +
                            "*";
                    }

                    // Add data buffer parameter for this column (chunk)
                    data_param.is_device_ptr = true;
                    input_kernel_params.push_back(data_param);
                }
            }

            is_first_time_to_get_column_info = false;

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

void ProjectionCodeGenerator::GenerateCode(
    CypherPhysicalOperator *op, CodeBuilder &code, GpuCodeGenerator *code_gen,
    ClientContext &context, PipelineContext &pipeline_ctx, bool is_main_loop)
{
    auto proj_op = dynamic_cast<PhysicalProjection *>(op);
    if (!proj_op) {
        return;
    }

    code.Add("// Projection operator");

    ExpressionCodeGenerator expr_gen(context);

    std::unordered_map<uint64_t, std::string> column_map;

    auto &columns_to_materialize =
        pipeline_ctx.columns_to_be_materialized
            [pipeline_ctx.current_sub_pipeline_index];

    if (pipeline_ctx.input_proj_vars_generated.find(
            pipeline_ctx.current_sub_pipeline_index) ==
        pipeline_ctx.input_proj_vars_generated.end()) {

        for (size_t i = 0; i < columns_to_materialize.size(); i++) {
            std::string &col_name = columns_to_materialize[i].name;
            LogicalType &col_type = columns_to_materialize[i].type;

            std::stringstream hex_stream;
            hex_stream << std::hex << (0x100000 + i);
            std::string hex_suffix = hex_stream.str().substr(1);
            std::string input_var_name = "input_proj_" + hex_suffix;

            std::string sanitized_col_name;
            for (char c : col_name) {
                if (std::isalnum(c) || c == '_') {
                    sanitized_col_name += c;
                }
                else {
                    sanitized_col_name += '_';
                }
            }

            if (!sanitized_col_name.empty() &&
                std::isdigit(sanitized_col_name[0])) {
                sanitized_col_name = "col_" + sanitized_col_name;
            }

            if (sanitized_col_name.empty()) {
                sanitized_col_name = "col_" + std::to_string(i);
            }

            std::string cuda_type =
                code_gen->ConvertLogicalTypeToPrimitiveType(col_type);
            code.Add(cuda_type + input_var_name + " = " + sanitized_col_name +
                     ";");

            column_map[i] = input_var_name;
        }
        pipeline_ctx.input_proj_vars_generated.insert(
            pipeline_ctx.current_sub_pipeline_index);
    }
    else {
        for (size_t i = 0; i < columns_to_materialize.size(); i++) {
            std::stringstream hex_stream;
            hex_stream << std::hex << (0x100000 + i);
            std::string hex_suffix = hex_stream.str().substr(1);
            std::string input_var_name = "input_proj_" + hex_suffix;
            column_map[i] = input_var_name;
        }
    }

    // Process each projection expression
    size_t output_var_counter = 0;
    for (size_t expr_idx = 0; expr_idx < proj_op->expressions.size();
         expr_idx++) {
        auto &expr = proj_op->expressions[expr_idx];

        std::stringstream hex_stream;
        hex_stream << std::hex << (0x100000 + output_var_counter);
        std::string hex_suffix = hex_stream.str().substr(1);
        std::string proj_var_name = "output_proj_" + hex_suffix;

        std::string result_expr = expr_gen.GenerateExpressionCode(
            expr.get(), code, pipeline_ctx, column_map);

        if (expr->expression_class == ExpressionClass::BOUND_REF) {
            pipeline_ctx.projection_variable_names[expr_idx] = result_expr;
        }
        else {
            std::string cuda_type =
                code_gen->ConvertLogicalTypeToPrimitiveType(expr->return_type);
            code.Add(cuda_type + proj_var_name + " = " + result_expr + ";");

            pipeline_ctx.projection_variable_names[expr_idx] = proj_var_name;
            output_var_counter++;
        }

        if (expr_idx < pipeline_ctx.output_column_names.size()) {
            std::string output_col_name =
                pipeline_ctx.output_column_names[expr_idx];
            // pipeline_ctx.column_materialized[output_col_name] = true;

            std::string final_var_name =
                pipeline_ctx.projection_variable_names[expr_idx];
            column_map[pipeline_ctx.input_column_names.size() + expr_idx] =
                final_var_name;
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
}

void ProjectionCodeGenerator::GenerateGlobalDeclarations(
    CypherPhysicalOperator *op, size_t op_idx, CodeBuilder &code,
    GpuCodeGenerator *code_gen, ClientContext &context,
    PipelineContext &pipeline_ctx)
{
    return;
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
    code.Add("int output_idx = atomicAdd(output_count, 1);");

    // Get the output schema to determine what columns to write
    auto &output_schema = results_op->GetSchema();
    auto &output_column_names = output_schema.getStoredColumnNamesRef();

    code.Add("// Write results to output buffers");
    
    std::cerr << "Output column count: " << output_column_names.size() << std::endl;
    for (size_t i = 0; i < output_column_names.size(); i++) {
        std::cerr << "Output column " << i << ": " << output_column_names[i] << std::endl;
    }
    
    for (size_t col_idx = 0; col_idx < output_column_names.size(); col_idx++) {
        std::string orig_col_name = output_column_names[col_idx];
        std::string tid_name = pipeline_ctx.attribute_tid_mapping[orig_col_name];
        // Replace '.' with '_' for valid C/C++ variable names
        // std::replace(col_name.begin(), col_name.end(), '.', '_');

        std::cerr << "Processing output column: " << orig_col_name
                  << std::endl;
        
        std::string col_name;
        for (char c : orig_col_name) {
            if (std::isalnum(c) || c == '_') {
                col_name += c;
            } else {
                col_name += '_';
            }
        }
        
        if (!col_name.empty() && std::isdigit(col_name[0])) {
            col_name = "col_" + col_name;
        }
        
        if (col_name.empty()) {
            col_name = "col_" + std::to_string(col_idx);
        }

        // type extract
        LogicalType type = pipeline_ctx.output_column_types[col_idx];
        std::string ctype =
            code_gen->ConvertLogicalTypeToPrimitiveType(type);

        // Check if this column is materialized in the pipeline context
        bool is_materialized =
            pipeline_ctx.column_materialized.find(col_name) !=
                pipeline_ctx.column_materialized.end() &&
            pipeline_ctx.column_materialized[col_name];

        std::cerr << "Column " << col_name << " is_materialized: " << is_materialized << std::endl;
        
        // Debug: print projection variable names
        std::cerr << "Projection variables available:" << std::endl;
        for (const auto& pv : pipeline_ctx.projection_variable_names) {
            std::cerr << "  [" << pv.first << "] = " << pv.second << std::endl;
        }

        std::string output_param_name;
        if (code_gen->GetVerboseMode()) {
            output_param_name = "output_" + col_name;
        }
        else {
            std::stringstream hex_stream;
            hex_stream << std::hex << (0x100000 + col_idx);
            std::string hex_suffix = hex_stream.str().substr(1);
            output_param_name = "output_proj_" + hex_suffix;
        }
        std::string output_data_name = output_param_name + "_data";
        std::string output_ptr_name = output_param_name + "_ptr";

        code.Add("// Write column " + std::to_string(col_idx) + " (" +
                 col_name + ") to output");

        if (is_materialized) {
            // Column is already materialized, use the projection result
            code.Add(ctype + "* " + output_ptr_name + " = static_cast<" +
                     ctype + "*>(" + output_data_name + ");");
            
            auto proj_var_it = pipeline_ctx.projection_variable_names.find(col_idx);
            if (proj_var_it != pipeline_ctx.projection_variable_names.end()) {
                code.Add(output_data_name + "[output_idx] = " + proj_var_it->second + ";");
            } else {
                code.Add(output_data_name + "[output_idx] = proj_result_" +
                         std::to_string(col_idx) + ";");
            }
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
                // code.Add(output_ptr_name + "[output_idx] = " + "tuple_id;");
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
                    code.Add(output_ptr_name + "[output_idx] = " +
                             input_col_name + "_ptr" + "[" + tid_name + "];");
                }
                else {
                    D_ASSERT(false); // TODO check
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
                        code.Add(output_ptr_name + "[output_idx] = " +
                                 col_name + "_ptr[" + tid_name + "];");
                    }
                    else {
                        // Fallback: use projection result
                        code.Add(
                            "// No mapping found, using projection result");
                        code.Add(ctype + "* " + output_ptr_name +
                                 " = static_cast<" + ctype + "*>(" +
                                 output_data_name + ");");
                        code.Add(output_ptr_name +
                                 "[output_idx] = proj_result_" +
                                 std::to_string(col_idx) + ";");
                    }
                }
            }
        }
    }
    
    for (const auto& proj_var : pipeline_ctx.projection_variable_names) {
        size_t var_idx = proj_var.first;
        const std::string& var_name = proj_var.second;
        
        std::stringstream hex_stream;
        hex_stream << std::hex << (0x100000 + var_idx);
        std::string hex_suffix = hex_stream.str().substr(1);
        std::string output_param_name = "output_proj_" + hex_suffix;
        std::string output_data_name = output_param_name + "_data";
        std::string output_ptr_name = output_param_name + "_ptr";
        
        if (var_idx < pipeline_ctx.output_column_types.size()) {
            LogicalType type = pipeline_ctx.output_column_types[var_idx];
            std::string ctype = code_gen->ConvertLogicalTypeToPrimitiveType(type);
            code.Add(ctype + "* " + output_ptr_name + " = static_cast<" + ctype + "*>(" + output_data_name + ");");
            code.Add(output_ptr_name + "[output_idx] = " + var_name + ";");
        } else {
            code.Add("double* " + output_ptr_name + " = static_cast<double*>(" + output_data_name + ");");
            code.Add(output_ptr_name + "[output_idx] = " + var_name + ";");
        }
    }

    // Update output count
    code.Add("// Update output count atomically");
    // code.Add("atomicAdd(output_count_ptr, 1);");

    code.Add("// Results produced successfully");
}

void ProduceResultsCodeGenerator::GenerateGlobalDeclarations(
    CypherPhysicalOperator *op, size_t op_idx, CodeBuilder &code,
    GpuCodeGenerator *code_gen, ClientContext &context,
    PipelineContext &pipeline_ctx)
{
    return;
}

void ProduceResultsCodeGenerator::GenerateOutputKernelParameters(
    CypherPhysicalOperator *op, GpuCodeGenerator *code_gen,
    ClientContext &context, PipelineContext &pipeline_context,
    std::vector<KernelParam> &output_kernel_params)
{
    auto sink_op = dynamic_cast<PhysicalProduceResults *>(op);
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
    for (size_t col_idx = 0; col_idx < output_column_names.size(); col_idx++) {
        std::string col_name = output_column_names[col_idx];
        // Replace '.' with '_' for valid C/C++ variable names
        // std::replace(col_name.begin(), col_name.end(), '.', '_');

        std::string sanitized_col_name;
        for (char c : col_name) {
            if (std::isalnum(c) || c == '_') {
                sanitized_col_name += c;
            } else {
                sanitized_col_name += '_';
            }
        }
        
        if (!sanitized_col_name.empty() && std::isdigit(sanitized_col_name[0])) {
            sanitized_col_name = "col_" + sanitized_col_name;
        }
        
        if (sanitized_col_name.empty()) {
            sanitized_col_name = "col_" + std::to_string(col_idx);
        }

        // Generate output parameter names
        std::string output_param_name;
        std::stringstream hex_stream;
        hex_stream << std::hex << (0x100000 + col_idx);
        std::string hex_suffix = hex_stream.str().substr(1);
        output_param_name = "output_proj_" + hex_suffix;

        // Add output data buffer parameter
        KernelParam output_data_param;
        output_data_param.name = output_param_name + "_data";
        output_data_param.type =
            "void *";  // Always void* for CUDA compatibility
        output_data_param.is_device_ptr = true;
        output_kernel_params.push_back(output_data_param);
    }
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
    std::unordered_map<uint64_t, std::string> column_pos_to_name;
    if (filter_op->expression) {
        condition_var =
            expr_gen.GenerateConditionCode(filter_op->expression.get(), code,
                                           pipeline_ctx, column_pos_to_name);
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

void FilterCodeGenerator::GenerateGlobalDeclarations(
    CypherPhysicalOperator *op, size_t op_idx, CodeBuilder &code,
    GpuCodeGenerator *code_gen, ClientContext &context,
    PipelineContext &pipeline_ctx)
{
    return;
}

void HashAggregateCodeGenerator::GenerateCode(
    CypherPhysicalOperator *op, CodeBuilder &code, GpuCodeGenerator *code_gen,
    ClientContext &context, PipelineContext &pipeline_ctx, bool is_main_loop)
{
    // Hashaggregate operator is either the first or the last operator
    D_ASSERT(pipeline_ctx.cur_op_idx == 0 ||
             pipeline_ctx.cur_op_idx == pipeline_ctx.total_operators - 1);

    if (pipeline_ctx.cur_op_idx == 0) {
        GenerateProbeSideCode(op, code, code_gen, context, pipeline_ctx);
    }
    else if (pipeline_ctx.cur_op_idx == pipeline_ctx.total_operators - 1) {
        GenerateBuildSideCode(op, code, code_gen, context, pipeline_ctx);
    }
    else {
        throw InvalidInputException("HashAggregate as intermediate operator");
    }
}

void HashAggregateCodeGenerator::GenerateProbeSideCode(
    CypherPhysicalOperator *op, CodeBuilder &code, GpuCodeGenerator *code_gen,
    ClientContext &context, PipelineContext &pipeline_ctx)
{
    // auto agg_op = dynamic_cast<PhysicalHashAggregate *>(op);
    // throw NotImplementedException(
    //     "HashAggregate probe side code generation not implemented yet");
}

void HashAggregateCodeGenerator::GenerateBuildSideCode(
    CypherPhysicalOperator *op, CodeBuilder &code, GpuCodeGenerator *code_gen,
    ClientContext &context, PipelineContext &pipeline_ctx)
{
    auto agg_op = dynamic_cast<PhysicalHashAggregate *>(op);
    if (agg_op->groups.size() == 0) {
        code.Add("// HashAggregate build side code without grouping");
        // if not self.touched and not self.doGroup and KernelCall.args.local_aggregation:
        //     for attrId, (inId, reductionType) in self.lop.aggregateTuples.items():
        //         attr, inputIdentifier, reductionType = self.lop.aggregateTuplesCreated[attrId]
        //         inAttr = self.lop.aggregateInAttributes.get(inId, None)
        //         if reductionType == Reduction.COUNT:
        //             code.Add(f'local_{attr.id_name} += 1;')
        //         elif reductionType == Reduction.SUM or reductionType == Reduction.AVG:
        //             code.Add(f'local_{attr.id_name} += {inAttr.id_name};')
        //         elif reductionType == Reduction.MAX:
        //             code.Add(f'local_{attr.id_name} = local_{attr.id_name} < {inAttr.id_name} ? {inAttr.id_name} : local_{attr.id_name};')
        //         elif reductionType == Reduction.MIN:
        //             code.Add(f'local_{attr.id_name} = local_{attr.id_name} > {inAttr.id_name} ? {inAttr.id_name} : local_{attr.id_name};')
        // else:        
        //     for attrId, (inId, _) in self.lop.aggregateTuples.items():
        //         attr, inputIdentifier, reductionType = self.lop.aggregateTuplesCreated[attrId]
        //         inAttr = self.lop.aggregateInAttributes.get(inId, None)
        //         dst = f"aht{self.opId}_{attr.id_name}[0]"
        //         if reductionType == Reduction.COUNT:
        //             code.Add(f"atomicAdd(&{dst},1);")
        //         elif reductionType == Reduction.SUM or reductionType == Reduction.AVG:
        //             code.Add(f"atomicAdd(&{dst},{inAttr.id_name});")
        //         elif reductionType == Reduction.MAX:
        //             code.Add(f"atomicMax(&{dst},{inAttr.id_name});")
        //         elif reductionType == Reduction.MIN:
        //             code.Add(f"atomicMin(&{dst},{inAttr.id_name});")
    } else {
        D_ASSERT(agg_op->groups.size() >= 1);
        std::string op_id = std::to_string(agg_op->GetOperatorId());
        code.Add("// HashAggregate build side code with grouping");
        code.Add("Payload" + op_id + " buf_payl;");
        code.Add("uint64_t hash_key = 0;");

        std::vector<uint64_t> group_key_idxs;
        for (auto &group : agg_op->groups) {
            code_gen->GetReferencedColumns(group.get(), group_key_idxs);
        }

        std::vector<std::string> group_key_names;
        for (auto &group_key_idx : group_key_idxs) {
            std::string &orig_col_name =
                pipeline_ctx.input_column_names[group_key_idx];
            std::string col_name = orig_col_name;
            std::replace(col_name.begin(), col_name.end(), '.', '_');
            group_key_names.push_back(col_name);

            // Check whether the group keys are materialized
            if (pipeline_ctx.column_materialized.find(orig_col_name) ==
                pipeline_ctx.column_materialized.end()) {
                code.Add("// Group key " + orig_col_name +
                         " is not materialized, materializing it now");

                std::string source_name =
                    pipeline_ctx.attribute_source_mapping[orig_col_name];
                std::string tid_name =
                    pipeline_ctx.attribute_tid_mapping[orig_col_name];
                code.Add(col_name + " = " + source_name + "[" + tid_name +
                         "];");
            }
        }

        for (size_t i = 0; i < group_key_idxs.size(); i++) {
            auto group_key_idx = group_key_idxs[i];
            std::string col_name = group_key_names[i];
            code.Add("buf_payl." + col_name + " = " + col_name + ";");
        }

        for (size_t i = 0; i < group_key_idxs.size(); i++) {
            auto group_key_idx = group_key_idxs[i];
            std::string col_name = group_key_names[i];
            if (pipeline_ctx.input_column_types[group_key_idx].id() ==
                LogicalTypeId::VARCHAR) {
                code.Add("hash_key = hash(hash_key + stringHash(" + col_name +
                         "));");
            }
            else {
                code.Add("hash_key = hash(hash_key + ((uint64_t) " + col_name +
                         "));");
            }
        }

        code.Add("int bucketFound = 0;");
        code.Add("int numLookups = 0;");
        code.Add("int bucketId = -1;");
        
        code.Add("while (!bucketFound) {");
        code.IncreaseNesting();
        code.Add("bucketId = -1;");
        code.Add("bool done = false;");
        
        code.Add("while (!done) {");
        code.IncreaseNesting();
        code.Add("bucketId = (hash_key + numLookups) % {int(self.size)};");
        code.Add("agg_ht<Payload" + op_id + ">& entry = aht" + op_id +
                 "[bucketId];");
        code.Add("numLookups++;");

        code.Add("if (entry.lock.enter()) {");
        code.IncreaseNesting();
        code.Add("entry.payload = buf_payl;");
        code.Add("entry.hash = hash_key;");
        code.Add("entry.lock.done();");
        code.Add("break;");
        code.DecreaseNesting();
        code.Add("} else {");
        code.IncreaseNesting();
        code.Add("entry.lock.wait();");
        code.Add("done = (entry.hash == hash_key);");
        code.DecreaseNesting();
        code.Add("}");
        code.DecreaseNesting();
        code.Add("}");

        code.Add("Payload" + op_id + " entry = aht" + op_id +
                 "[bucketId].payload;");
        code.Add("bucketFound = 1;");

        for (size_t i = 0; i < group_key_idxs.size(); i++) {
            auto group_key_idx = group_key_idxs[i];
            std::string col_name = group_key_names[i];
            if (pipeline_ctx.input_column_types[group_key_idx].id() ==
                LogicalTypeId::VARCHAR) {
                code.Add("bucketFound &= stringEquals(entry." + col_name +
                         ", " + col_name + ");");
            }
            else {
                code.Add("bucketFound &= entry." + col_name +
                         " == " + col_name + ";");
            }
        }

        code.DecreaseNesting();
        code.Add("}");

        std::vector<std::string> agg_function_names;
        for (auto &aggregate : agg_op->aggregates) {
            D_ASSERT(aggregate->GetExpressionType() ==
                     ExpressionType::BOUND_AGGREGATE);
            auto bound_agg =
                dynamic_cast<BoundAggregateExpression *>(aggregate.get());
            agg_function_names.push_back(bound_agg->function.name);
        }

        for (size_t i = 0; i < agg_function_names.size(); i++) {
            auto &agg_function_name = agg_function_names[i];
            std::string agg_var_name = "agg_" + std::to_string(i);
            std::string dst = "aht" + op_id + "_" + agg_var_name + "[bucketId]";
            std::string inattr = "todo";  // TODO
            if (agg_function_name == "count_star" ||
                agg_function_name == "count") {
                code.Add("atomicAdd(&" + dst + ", 1);");
            }
            else if (agg_function_name == "sum" || agg_function_name == "avg") {
                code.Add("atomicAdd(&" + dst + ", " + inattr + ");");
            }
            else if (agg_function_name == "max") {
                code.Add("atomicMax(&" + dst + ", " + inattr + ");");
            }
            else if (agg_function_name == "min") {
                code.Add("atomicMin(&" + dst + ", " + inattr + ");");
            }
            else {
                throw NotImplementedException(
                    "HashAggregate build side code generation for " +
                    agg_function_name + " not implemented yet");
            }
        }
    }
}

void HashAggregateCodeGenerator::GenerateGlobalDeclarations(
    CypherPhysicalOperator *op, size_t op_idx, CodeBuilder &code,
    GpuCodeGenerator *code_gen, ClientContext &context,
    PipelineContext &pipeline_ctx)
{
    // Hashaggregate appears twice in the pipelines, once for build side
    // and once for probe side. We need to generate payload structure only once
    if (op_idx == 0)
        return;

    // declare payload
    auto agg_op = dynamic_cast<PhysicalHashAggregate *>(op);
    uint64_t op_id = agg_op->GetOperatorId();
    code.Add("struct Payload" + std::to_string(op_id) + " {");
    code.IncreaseNesting();

    auto &input_schema = agg_op->GetSchema();
    auto &input_column_names = input_schema.getStoredColumnNamesRef();
    auto &input_column_types = input_schema.getStoredTypesRef();
    vector<uint64_t> group_key_idxs;
    for (auto &group : agg_op->groups) {
        code_gen->GetReferencedColumns(group.get(), group_key_idxs);
    }
    for (auto &group_key_idx : group_key_idxs) {
        std::string col_name = input_column_names[group_key_idx];
        std::replace(col_name.begin(), col_name.end(), '.', '_');
        auto col_type = input_column_types[group_key_idx];
        std::string ctype =
            code_gen->ConvertLogicalTypeToPrimitiveType(col_type);
        code.Add(ctype + col_name + ";");
    }
    code.DecreaseNesting();
    code.Add("}; // Payload" + std::to_string(op_id));
    return;
}

void HashAggregateCodeGenerator::GenerateInputKernelParameters(
    CypherPhysicalOperator *op, GpuCodeGenerator *code_gen,
    ClientContext &context, PipelineContext &pipeline_context,
    std::vector<KernelParam> &input_kernel_params,
    std::vector<ScanColumnInfo> &scan_column_infos)
{
    auto agg_op = dynamic_cast<PhysicalHashAggregate *>(op);

    // Add input parameters for the hash aggregate
    KernelParam input_param;
    input_param.name = "aht" + std::to_string(agg_op->GetOperatorId());
    input_param.type = "agg_ht<Payload" + std::to_string(agg_op->GetOperatorId()) +
                       "> *";
    input_param.is_device_ptr = true;
    input_kernel_params.push_back(input_param);
}

void HashAggregateCodeGenerator::GenerateOutputKernelParameters(
    CypherPhysicalOperator *op, GpuCodeGenerator *code_gen,
    ClientContext &context, PipelineContext &pipeline_context,
    std::vector<KernelParam> &output_kernel_params)
{
    auto agg_op = dynamic_cast<PhysicalHashAggregate *>(op);
    auto op_id = agg_op->GetOperatorId();

    // Add output parameters for the hash aggregate
    KernelParam output_param;
    output_param.name = "aht" + std::to_string(agg_op->GetOperatorId());
    output_param.type =
        "agg_ht<Payload" + std::to_string(agg_op->GetOperatorId()) + "> *";
    output_param.is_device_ptr = true;
    output_kernel_params.push_back(output_param);

    for (size_t i = 0; i < agg_op->aggregates.size(); i++) {
        auto &aggregate = agg_op->aggregates[i];
        D_ASSERT(aggregate->GetExpressionType() ==
                 ExpressionType::BOUND_AGGREGATE);
        auto bound_agg =
            dynamic_cast<BoundAggregateExpression *>(aggregate.get());
        std::string func_name = bound_agg->function.name;

        if (func_name == "count_star" || func_name == "count") {
            // Add output parameter for this aggregate function
            KernelParam agg_param;
            agg_param.name = "aht" + std::to_string(op_id) + "_" + "agg_" +
                             std::to_string(i);
            agg_param.type = "int *";
            agg_param.is_device_ptr = true;
            output_kernel_params.push_back(agg_param);
        }
        else {
            throw NotImplementedException(
                "HashAggregate output parameter generation for " + func_name +
                " not implemented yet");
        }
    }
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
    code.IncreaseNesting();
    for (size_t i = 0; i < pipeline_context.sub_pipelines.size(); i++) {
        std::string spid = std::to_string(i);
        auto &sub_pipeline = pipeline_context.sub_pipelines[i];
        PipeInputType input_type = GetPipeInputType(sub_pipeline);
        code.Add("case " + spid + ": {");
        code.IncreaseNesting();
        if (input_type == PipeInputType::TYPE_0) {
            code.Add(
                "Themis::PushedParts::PushedPartsAtZeroLvl *src_pparts = "
                "(Themis::PushedParts::PushedPartsAtZeroLvl*) stack->Top();");
            code.Add(
                "Themis::PullINodesFromPPartAtZeroLvl(thread_id, src_pparts, "
                "ts_0_range_cached, inodes_cnts);");
            code.Add("if (thread_id == 0) stack->PopPartsAtZeroLvl();");
        }
        else if (input_type == PipeInputType::TYPE_1) {
            throw NotImplementedException(
                "GenerateCopyCodeForAdaptiveWorkSharingPull for TYPE_1");
            code.Add(
                "Themis::PushedParts::PushedPartsAtLoopLvl* src_pparts = "
                "(Themis::PushedParts::PushedPartsAtLoopLvl*) stack->Top();");
            code.Add("Themis::PullINodesFromPPartAtLoopLvl(thread_id, " + spid +
                     ", "
                     "src_pparts, ts_" +
                     spid + "_range_cached, ts_" + spid +
                     "_range, "
                     "inodes_cnts);");

            // attrs = {}
            // tid = spSeq.getTid()
            // lastOp = spSeq.pipe.subpipeSeqs[spSeq.id-1].subpipes[-1].operators[-1]
            // for attrId, attr in spSeq.inBoundaryAttrs.items():
            //     if attrId == tid.id: continue
            //     if attrId in lastOp.generatingAttrs: continue
            //     attrs[attrId] = attr

            uint64_t speculated_size = 0;
            // if len(attrs) > 0:
            //     code.Add("volatile char* src_pparts_attrs = src_pparts->GetAttrsPtr();')

            //     for attrId, attr in attrs.items():
            //         if attr.dataType == Type.STRING:
            //             code.Add(f'Themis::PullStrAttributesAtLoopLvl(thread_id, ts_{spSeqId}_{attr.id_name}_cached, ts_{spSeqId}_{attr.id_name}, (volatile str_t*) (src_pparts_attrs + {speculated_size}));')
            //         elif attr.dataType == Type.PTR_INT:
            //             code.Add(f'Themis::PullPtrIntAttributesAtLoopLvl(thread_id, ts_{spSeqId}_{attr.id_name}_cached, ts_{spSeqId}_{attr.id_name}, (volatile int**) (src_pparts_attrs + {speculated_size}));')
            //         else:
            //             code.Add(f'Themis::PullAttributesAtLoopLvl<{langType(attr.dataType)}>(thread_id, ts_{spSeqId}_{attr.id_name}_cached, ts_{spSeqId}_{attr.id_name}, ({langType(attr.dataType)}*) (src_pparts_attrs + {speculated_size}));')
            //         speculated_size += CType.size[langType(attr.dataType)] * (2 * tsWidth)
            code.Add("if (thread_id == 0) stack->PopPartsAtLoopLvl(" +
                     std::to_string(speculated_size) + ");");
        }
        else if (input_type == PipeInputType::TYPE_2) {
            code.Add(
                "Themis::PushedParts::PushedPartsAtIfLvl* src_pparts = "
                "(Themis::PushedParts::PushedPartsAtIfLvl*) stack->Top();");
            code.Add("Themis::PullINodesFromPPartAtIfLvl(thread_id, " + spid +
                     ", src_pparts, inodes_cnts);");
            
            D_ASSERT(i > 0);
            auto &tids = pipeline_context.sub_pipeline_tids[i - 1];
            if (tids.size() > 0) {
                code.Add("volatile char *src_pparts_attrs = src_pparts->GetAttrsPtr();");
                uint64_t speculated_size = 0;
                uint64_t tsWidth = 32;
                for (auto &tid : tids) {
                    // TODO additional types
                    // if attr.dataType == Type.STRING:
                    //     code.Add(f'Themis::PullStrAttributesAtIfLvl(thread_id, ts_{spSeqId}_{attr.id_name}_flushed, (volatile str_t*) (src_pparts_attrs + {speculated_size}));')
                    // elif attr.dataType == Type.PTR_INT:
                    //     code.Add(f'Themis::PullPtrIntAttributesAtIfLvl(thread_id, ts_{spSeqId}_{attr.id_name}_flushed, (int**) (src_pparts_attrs + {speculated_size}));')
                    // else:
                    // code.Add("Themis::PullAttributesAtIfLvl<{langType(attr.dataType)}>(thread_id, ts_{spSeqId}_{attr.id_name}_flushed, ({langType(attr.dataType)}*) (src_pparts_attrs + {speculated_size}));");
                    code.Add(
                        "Themis::PullAttributesAtIfLvl<int>(thread_id, ts_" +
                        spid + "_" + tid +
                        "_flushed, (int *)(src_pparts_attrs + " +
                        std::to_string(speculated_size) + "));");
                    speculated_size += sizeof(int) * (2 * tsWidth);
                }
                code.Add("if (thread_id == 0) stack->PopPartsAtIfLvl(" +
                         std::to_string(speculated_size) + ");");
            }
        }
        code.DecreaseNesting();
        code.Add("}");
        code.Add("break;");
    }
    code.DecreaseNesting();
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

    code.Add("Themis::FindIdleWarp<" +
             std::to_string(pipeline_context.sub_pipelines.size()) + ", " +
             std::to_string(num_warps) + ", " + std::to_string(min_num_warps) +
             ">(");
    code.IncreaseNesting();
    code.Add(
        "target_warp_id, warp_id, thread_id, warp_status,");
    code.Add("num_idle_warps, "
        "num_warps,gts, size_of_stack_per_warp,");
    if (idleWarpDetectionType == "twolvlbitmaps") {
        code.Add("global_bit1, global_bit2);");
    }
    else if (idleWarpDetectionType == "idqueue") {
        code.Add("global_id_stack);");
    }
    code.DecreaseNesting();
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
    if (pipeline_context.sub_pipelines.size() > 1) {
        code.Add("switch (new_local_lowest_lvl) {");
        code.IncreaseNesting();
        for (size_t i = 0; i < pipeline_context.sub_pipelines.size(); i++) {
            if (i == 0) continue;
            std::string spid = std::to_string(i);
            auto &sub_pipeline = pipeline_context.sub_pipelines[i];
            PipeInputType input_type = GetPipeInputType(sub_pipeline);
            code.Add("case " + spid + ": {");
            code.IncreaseNesting();
            if (input_type == PipeInputType::TYPE_1) {
                code.Add("Themis::CountINodesAtLoopLvl(thread_id, " + spid +
                         ", ts_" + spid + "_range_cached, ts_" + spid +
                         "_range, inodes_cnts);");
                code.Add(
                    "new_num_nodes_at_locally_lowest_lvl = "
                    "__shfl_sync(ALL_LANES, inodes_cnts, " +
                    spid + ");");
                code.Add(
                    "new_local_max_order = "
                    "Themis::CalculateOrder(new_num_nodes_at_locally_"
                    "lowest_lvl);");
            }
            else {
                code.Add(
                    "new_num_nodes_at_locally_lowest_lvl = "
                    "__shfl_sync(ALL_LANES, inodes_cnts, " +
                    spid + ");");
                code.Add("new_local_max_order = 0;");
            }
            code.DecreaseNesting();
            code.Add("}");
            code.Add("break;");
        }       
        code.DecreaseNesting();     
        code.Add("}");
    }
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

    code.Add("switch (lvl_to_push) {");
    code.IncreaseNesting();
    for (size_t i = 0; i < pipeline_context.sub_pipelines.size(); i++) {
        std::string spid = std::to_string(i);
        auto &sub_pipeline = pipeline_context.sub_pipelines[i];
        PipeInputType input_type = GetPipeInputType(sub_pipeline);
        code.Add("case " + spid + ": {");
        code.IncreaseNesting();
        if (input_type == PipeInputType::TYPE_0) {
            code.Add("if (thread_id == 0) stack->PushPartsAtZeroLvl();");
            code.Add(
                "Themis::PushedParts::PushedPartsAtZeroLvl* target_pparts = "
                "(Themis::PushedParts::PushedPartsAtZeroLvl*) stack->Top();");
            code.Add(
                "num_to_push = Themis::PushINodesToPPartAtZeroLvl(thread_id, "
                "target_pparts, ts_0_range_cached, " +
                std::to_string(stepSize) + ");");
            code.Add("num_remaining = num_nodes - num_to_push;");
            code.Add(
                "mask_32 = num_remaining >= 32 ? (m | mask_32) : ((~m) & "
                "mask_32);");
            code.Add(
                "mask_1 = num_remaining > 0 ?  (m | mask_1) : ((~m) & "
                "mask_1);");
        }
        else if (input_type == PipeInputType::TYPE_1) {
            throw NotImplementedException(
                "GenerateCopyCodeForAdaptiveWorkSharingPush for TYPE_1");
            // attrs = {}
            // tid = spSeq.getTid()
            // lastOp = spSeq.pipe.subpipeSeqs[spSeq.id-1].subpipes[-1].operators[-1]
            // for attrId, attr in spSeq.inBoundaryAttrs.items():
            //     if attrId == tid.id: continue
            //     if attrId in lastOp.generatingAttrs: continue
            //     attrs[attrId] = attr

            // speculated_size = 0
            // for attrId, attr in attrs.items():
            //     speculated_size += CType.size[langType(attr.dataType)] * (2 * tsWidth)
            // code.Add(f'if (thread_id == 0) stack->PushPartsAtLoopLvl({spSeqId}, {speculated_size});')
            // code.Add("Themis::PushedParts::PushedPartsAtLoopLvl* target_pparts = (Themis::PushedParts::PushedPartsAtLoopLvl*) stack->Top();')
            // code.Add(f'num_to_push = Themis::PushINodesToPPartAtLoopLvl(thread_id, {spSeqId}, target_pparts, ts_{spSeqId}_range_cached, ts_{spSeqId}_range);')
            // code.Add(f'num_remaining = num_nodes - num_to_push;')
            // code.Add(f'mask_32 = num_remaining >= 32 ? (m | mask_32) : ((~m) & mask_32);')
            // code.Add(f'mask_1 = num_remaining > 0 ?  (m | mask_1) : ((~m) & mask_1);')
            // code.Add(f'int ts_src = 32;')
            // code.Add(f'Themis::DistributeFromPartToDPart(thread_id, {spSeqId}, ts_src, ts_{spSeqId}_range, ts_{spSeqId}_range_cached, mask_32, mask_1);')
            // if len(attrs) > 0:
            //     speculated_size = 0
            //     code.Add("volatile char* target_pparts_attrs = target_pparts->GetAttrsPtr();')
            //     for attrId, attr in attrs.items():
            //         name = f'ts_{spSeqId}_{attr.id_name}'
            //         code.Add("{')
            //         if attr.dataType == Type.STRING:
            //             code.Add(f'Themis::PushStrAttributesAtLoopLvl(thread_id, (volatile str_t*) (target_pparts_attrs + {speculated_size}), {name}_cached, {name});')
            //             code.Add(f'char* start = (char*) __shfl_sync(ALL_LANES, (uint64_t) {name}.start, ts_src);')
            //             code.Add(f'char* end = (char*) __shfl_sync(ALL_LANES, (uint64_t) {name}.end, ts_src);')
            //             code.Add(f'if (ts_src < 32) {name}_cached.start = start;')
            //             code.Add(f'if (ts_src < 32) {name}_cached.end = end;')
            //         elif attr.dataType == Type.PTR_INT:
            //             code.Add(f'Themis::PushPtrIntAttributesAtLoopLvl(thread_id, (volatile int**) (target_pparts_attrs + {speculated_size}), {name}_cached, {name});')
            //             code.Add(f'uint64_t cache = __shfl_sync(ALL_LANES, (uint64_t){name}, ts_src);')
            //             code.Add(f'if (ts_src < 32) {name}_cached = (int*) cache;')
            //         else:
            //             dtype = langType(attr.dataType)
            //             code.Add(f'Themis::PushAttributesAtLoopLvl<{dtype}>(thread_id, (volatile {dtype}*) (target_pparts_attrs + {speculated_size}), {name}_cached, {name});')
            //             code.Add(f'{dtype} cache = __shfl_sync(ALL_LANES, {name}, ts_src);')
            //             code.Add(f'if (ts_src < 32) {name}_cached = cache;')
            //         code.Add("}')
            //         speculated_size += CType.size[langType(attr.dataType)] * (2 * tsWidth)
        }
        else if (input_type == PipeInputType::TYPE_2) {
            uint64_t speculated_size = 0;
            D_ASSERT(i > 0);
            auto &tids = pipeline_context.sub_pipeline_tids[i - 1];
            for (auto &tid : tids) {
                // speculated_size += CType.size[langType(attr.dataType)] * (2 * tsWidth)
                speculated_size += sizeof(int) * (2 * tsWidth);
            }
            code.Add("if (thread_id == 0) stack->PushPartsAtIfLvl(" + spid +
                     ", " + std::to_string(speculated_size) + ");");
            code.Add(
                "Themis::PushedParts::PushedPartsAtIfLvl* target_pparts = "
                "(Themis::PushedParts::PushedPartsAtIfLvl*) stack->Top();");
            code.Add(
                "num_to_push = Themis::PushINodesToPPartAtIfLvl(thread_id, " +
                spid + ", target_pparts, inodes_cnts);");
            code.Add("mask_32 = ((~m) & mask_32);");
            code.Add("mask_1 = ((~m) & mask_1);");
            if (tids.size() > 0) {
                code.Add("volatile char* target_pparts_attrs = target_pparts->GetAttrsPtr();");
                speculated_size = 0;
                for (auto &tid : tids) {
                    std::string name = "ts_" + spid + "_" + tid;
                    code.Add("{");
                    code.IncreaseNesting();
                    // if attr.dataType == Type.STRING:
                    //     code.Add(f'Themis::PushStrAttributesAtIfLvl(thread_id, (volatile str_t*) (target_pparts_attrs + {speculated_size}), {name}_flushed);')
                    // elif attr.dataType == Type.PTR_INT:
                    //     code.Add(f'Themis::PushPtrIntAttributesAtIfLvl(thread_id, (volatile int**) (target_pparts_attrs + {speculated_size}), {name}_flushed);')
                    // else:
                    //     dtype = langType(attr.dataType)
                    //     code.Add(f'Themis::PushAttributesAtIfLvl<{dtype}>(thread_id, (volatile {dtype}*) (target_pparts_attrs + {speculated_size}), {name}_flushed);')
                    code.Add(
                        "Themis::PushAttributesAtIfLvl<int>(thread_id, "
                        "(volatile int*) (target_pparts_attrs + " +
                        std::to_string(speculated_size) + "), " + name +
                        "_flushed);");
                    code.DecreaseNesting();
                    code.Add("}");
                    // speculated_size += CType.size[langType(attr.dataType)] * (2 * tsWidth)
                    speculated_size += sizeof(int) * (2 * tsWidth);
                }
            }
        }
        code.DecreaseNesting();
        code.Add("}");
        code.Add("break;");
    }
    code.DecreaseNesting();
    code.Add("}");
    int num_warps_per_block = int(KernelConstants::DEFAULT_BLOCK_SIZE / 32);
    code.Add("if ((target_warp_id / " + std::to_string(num_warps_per_block) +
             ") == (gpart_id / " + std::to_string(num_warps_per_block) +
             ")) __threadfence_block();");
    code.Add("else __threadfence();");
}

// Pipeline context management methods
void GpuCodeGenerator::InitializePipelineContext(CypherPipeline &pipeline)
{
    pipeline_context.InitializePipeline(pipeline);
}

void GpuCodeGenerator::AdvanceOperator()
{
    pipeline_context.AdvanceOperator();
}

// PipelineContext method implementations
void PipelineContext::InitializePipeline(CypherPipeline &pipeline)
{
    total_operators = pipeline.GetPipelineLength();
    cur_op_idx = -1;

    // Clear existing data
    operator_column_names.clear();
    operator_column_types.clear();
    column_materialized.clear();
    used_columns.clear();
    // column_to_param_mapping.clear();

    current_pipeline = &pipeline;

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

void PipelineContext::AdvanceOperator()
{
    cur_op_idx++;
    D_ASSERT(cur_op_idx >= 0 && cur_op_idx < total_operators);

    auto *current_op = current_pipeline->GetIdxOperator(cur_op_idx);
    // auto &input_schema = current_op->GetSchema();
    // auto &input_column_names = input_schema.getStoredColumnNamesRef();
    // auto &input_column_types = input_schema.getStoredTypesRef();

    // Update input schema from previous operator
    if (cur_op_idx > 0 && operator_column_names[cur_op_idx - 1] &&
        operator_column_types[cur_op_idx - 1]) {
        input_column_names = *operator_column_names[cur_op_idx - 1];
        input_column_types = *operator_column_types[cur_op_idx - 1];
    }
    else {
        input_column_names.clear();
        input_column_types.clear();
    }

    // Update output schema from current operator
    if (operator_column_names[cur_op_idx] &&
        operator_column_types[cur_op_idx]) {
        output_column_names = *operator_column_names[cur_op_idx];
        output_column_types = *operator_column_types[cur_op_idx];
    }
    else {
        output_column_names.clear();
        output_column_types.clear();
    }
}

}  // namespace duckdb