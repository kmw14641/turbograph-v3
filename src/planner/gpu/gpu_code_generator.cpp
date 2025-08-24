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
#include "common/types/decimal.hpp"
#include "execution/physical_operator/cypher_physical_operator.hpp"
#include "execution/physical_operator/physical_filter.hpp"
#include "execution/physical_operator/physical_node_scan.hpp"
#include "execution/physical_operator/physical_produce_results.hpp"
#include "execution/physical_operator/physical_projection.hpp"
#include "execution/physical_operator/physical_hash_aggregate.hpp"
#include "execution/physical_operator/physical_adjidxjoin.hpp"
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
    operator_generators[PhysicalOperatorType::HASH_AGGREGATE] =
        std::make_unique<HashAggregateCodeGenerator>();
    operator_generators[PhysicalOperatorType::ADJ_IDX_JOIN] =
        std::make_unique<AdjIdxJoinCodeGenerator>();
    operator_generators[PhysicalOperatorType::ID_SEEK] =
        std::make_unique<IdSeekCodeGenerator>();
}

void GpuCodeGenerator::GenerateGlobalDeclarations(
    std::vector<CypherPipeline *> &pipelines)
{
    CodeBuilder code;

    // include necessary headers
    gpu_include_header = "#include \"range.cuh\"\n";
    gpu_include_header += "#include \"themis.cuh\"\n";
    gpu_include_header += "#include \"work_sharing.cuh\"\n";
    gpu_include_header += "#include \"adaptive_work_sharing.cuh\"\n";
    gpu_include_header += "#include \"decimal_device.cuh\"\n";
    gpu_include_header += "\n";

    // Generate global declarations for each operator
    for (auto &pipeline : pipelines) {
        for (size_t i = 0; i < pipeline->GetPipelineLength(); i++) {
            auto *prev_op = i == 0 ? nullptr : pipeline->GetIdxOperator(i - 1);
            auto *op = pipeline->GetIdxOperator(i);
            GenerateGlobalDeclaration(op, prev_op, i, code, pipeline_context);
        }
    }

    global_declarations = code.str();
}

void GpuCodeGenerator::GenerateGPUCode(CypherPipeline &pipeline)
{
    SCOPED_TIMER_SIMPLE(GenerateGPUCode, spdlog::level::info,
                        spdlog::level::info);

    // generate kernel code
    SUBTIMER_START(GenerateGPUCode, "GenerateKernelCode");
    GenerateKernelCode(pipeline);
    SUBTIMER_STOP(GenerateGPUCode, "GenerateKernelCode");

    num_pipelines_compiled++;
}

void GpuCodeGenerator::GenerateCPUCode(std::vector<CypherPipeline *> &pipelines)
{
    SCOPED_TIMER_SIMPLE(GenerateCPUCode, spdlog::level::info,
                        spdlog::level::info);

    // generate host code
    SUBTIMER_START(GenerateCPUCode, "GenerateHostCode");
    GenerateHostCode(pipelines);
    SUBTIMER_STOP(GenerateCPUCode, "GenerateHostCode");
}

void GpuCodeGenerator::SplitPipelineIntoSubPipelines(CypherPipeline &pipeline)
{
    CypherPhysicalOperatorGroups groups;
    CypherPhysicalOperatorGroups &pipeline_groups =
        pipeline.GetOperatorGroups();
    pipeline_context.sub_pipelines.clear();
    pipeline_context.do_lb.clear();

    for (int op_idx = 0; op_idx < pipeline.GetPipelineLength(); op_idx++) {
        auto op = pipeline.GetIdxOperator(op_idx);
        D_ASSERT(op != nullptr);

        // insert the operator into the current group
        groups.GetGroups().push_back(pipeline_groups.GetGroups()[op_idx]);

        bool do_lb = false;
        bool split = false;

        switch (op->GetOperatorType()) {
            case PhysicalOperatorType::NODE_SCAN: {
                auto scan_op = dynamic_cast<PhysicalNodeScan *>(op);
                if (scan_op->is_filter_pushdowned) {
                    // If the scan operator has filter pushdown, we need to
                    // create a new sub-pipeline for the filter
                    do_lb = true;
                    split = true;
                }
                break;
            }
            case PhysicalOperatorType::FILTER: {
                // Filter operators always create a new sub-pipeline
                do_lb = true;
                split = true;
                break;
            }
            case PhysicalOperatorType::HASH_AGGREGATE: {
                if (op_idx == 0) {
                    do_lb = false;
                    split = true;
                }
                break;
            }
            case PhysicalOperatorType::ADJ_IDX_JOIN: {
                auto adj_idx_join_op = dynamic_cast<PhysicalAdjIdxJoin *>(op);
                if (!adj_idx_join_op->IsTargetUnique()) {
                    // If the target is not unique, we need to create a new
                    // sub-pipeline for the next operators
                    do_lb = true;
                    split = true;
                }
                break;
            }
            case PhysicalOperatorType::PROJECTION:
            case PhysicalOperatorType::PRODUCE_RESULTS: {
                break;
            }
            case PhysicalOperatorType::ID_SEEK: {
                // temporary
                break;
            }
            default:
                throw NotImplementedException(
                    "GpuCodeGenerator::SplitPipelineIntoSubPipelines: "
                    "Unsupported operator type " +
                    std::to_string((uint8_t)op->GetOperatorType()) +
                    " for splitting pipeline");
        }

        if (split) {
            pipeline_context.sub_pipelines.emplace_back(groups);
            pipeline_context.do_lb.push_back(do_lb);
            groups.GetGroups().clear();
            groups.GetGroups().push_back(pipeline_groups.GetGroups()[op_idx]);
        }
    }

    if (groups.GetGroups().size() > 0) {
        pipeline_context.sub_pipelines.emplace_back(groups);
        pipeline_context.do_lb.push_back(true);
    }
}

void GpuCodeGenerator::AnalyzeSubPipelinesForMaterialization()
{
    // Initialize materialization structures
    int num_sub_pipelines = pipeline_context.sub_pipelines.size();
    pipeline_context.columns_to_be_materialized.clear();
    pipeline_context.columns_to_be_materialized.resize(num_sub_pipelines);
    pipeline_context.materialization_target_columns.clear();
    pipeline_context.materialization_target_columns.resize(num_sub_pipelines);
    pipeline_context.sub_pipeline_tids.clear();
    pipeline_context.sub_pipeline_tids.resize(num_sub_pipelines);
    pipeline_context.attribute_tid_mapping.clear();
    pipeline_context.attribute_source_mapping.clear();

    // Analyze each sub-pipeline to determine which columns need to be
    // materialized
    for (int sub_idx = 0; sub_idx < num_sub_pipelines; sub_idx++) {
        auto &sub_pipeline = pipeline_context.sub_pipelines[sub_idx];
        auto &mat_target_columns =
            pipeline_context.materialization_target_columns[sub_idx];
        // debugging
        std::cerr << "Sub-pipeline " << sub_idx << ":\n";
        std::cerr << sub_pipeline.toString() << std::endl;
        // Iterate over each operator in the sub-pipeline
        int op_idx = sub_idx == 0 ? 0 : 1;
        for (; op_idx < sub_pipeline.GetPipelineLength(); op_idx++) {
            auto *op = sub_pipeline.GetIdxOperator(op_idx);
            auto it = operator_generators.find(op->GetOperatorType());
            if (it != operator_generators.end()) {
                it->second->AnalyzeOperatorForMaterialization(
                    op, sub_idx, op_idx, pipeline_context, this);
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
        case ExpressionClass::BOUND_AGGREGATE: {
            auto agg_expr = dynamic_cast<BoundAggregateExpression *>(expr);
            for (auto &child : agg_expr->children) {
                GetReferencedColumns(child.get(), referenced_columns);
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

    // kernel function declaration
    if (do_inter_warp_lb) {
        code.Add(
            "extern \"C\" __global__ void __launch_bounds__(128, 8) "
            "gpu_kernel" +
            std::to_string(pipeline.GetPipelineId()) + "(");
    }
    else {
        code.Add("extern \"C\" __global__ void gpu_kernel" +
                 std::to_string(pipeline.GetPipelineId()) + "(");
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
    for (const auto &p : input_kernel_params.back()) {
        if (p.type.find('*') != std::string::npos) {
            code.Add(p.type + p.name + " = (" + p.type +
                     ")input_data[" + std::to_string(in_idx) + "];");
            ++in_idx;
        } else {
            code.Add("const " + p.type + " " + p.name + " = " + p.value + ";");
        }
    }
    for (const auto &p : output_kernel_params.back()) {
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
        PipeInputType input_type = GetPipeInputType(sub_pipeline, i);
        if (input_type == PipeInputType::TYPE_1_FALSE ||
            input_type == PipeInputType::TYPE_1_TRUE) {
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
        else if (input_type == PipeInputType::TYPE_2_FALSE ||
                 input_type == PipeInputType::TYPE_2_TRUE) {
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
    code.Add(
        "ts_0_range_cached.end = " +
        std::to_string(pipeline_context.per_pipeline_num_input_tuples.back()) +
        ";");

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
        code.DecreaseNesting();
        code.Add("} while (true); // while loop");
    }
    else {
        GeneratePipelineCode(pipeline, code);
    }

    code.DecreaseNesting();
    code.Add("} // end of kernel");
    code.Add("");

    generated_gpu_code += code.str();
}

void GpuCodeGenerator::GenerateHostCode(std::vector<CypherPipeline *> &pipelines)
{
    CodeBuilder code;

    cpu_include_header = "#include <cuda.h>\n";
    cpu_include_header += "#include <cuda_runtime.h>\n";
    cpu_include_header += "#include <cstdint>\n";
    cpu_include_header += "#include <vector>\n";
    cpu_include_header += "#include <iostream>\n";
    cpu_include_header += "#include <string>\n";
    cpu_include_header += "#include <unordered_map>\n";
    cpu_include_header += "#include \"adaptive_work_sharing.cuh\"\n";
    cpu_include_header += "#include \"pushedparts.cuh\"\n";
    cpu_include_header += "#include \"relation.cuh\"\n";
    cpu_include_header += "#include \"typedef.cuh\"\n";
    cpu_include_header += "#include \"decimal_device.cuh\"\n";
    cpu_include_header += "\n";

    // declare kernel functions
    for (int i = 0; i < num_pipelines_compiled; i++) {
        code.Add("extern \"C\" CUfunction gpu_kernel" + std::to_string(i) +
                 ";");
    }
    // declare initialization function
    code.Add("extern \"C\" CUfunction initAggHT;");
    code.Add("extern \"C\" CUfunction initArray;");
    code.Add("");

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

    code.Add("printf(\"Query started on GPU...\\n\");");
    code.Add("const int blockSize = " +
             std::to_string(KernelConstants::DEFAULT_BLOCK_SIZE) + ";");
    code.Add("const int gridSize  = " +
             std::to_string(KernelConstants::DEFAULT_GRID_SIZE) + ";");

    GenerateDeclarationInHostCode(pipelines, code);
    GenerateKernelCallInHostCode(pipelines, code);

    code.Add("std::cout << \"Query finished on GPU.\" << std::endl;");
    code.DecreaseNesting();
    code.Add("}");

    generated_cpu_code = code.str();
}

void GpuCodeGenerator::GenerateDeclarationInHostCode(
    std::vector<CypherPipeline *> &pipelines, CodeBuilder &code)
{
    // Generate declarations for operators
    code.Add("");
    code.Add("// Generate declarations for operators in host code");
    for (auto &pipeline : pipelines) {
        for (size_t i = 0; i < pipeline->GetPipelineLength(); i++) {
            auto *op = pipeline->GetIdxOperator(i);
            auto it = operator_generators.find(op->GetOperatorType());
            if (it != operator_generators.end()) {
                it->second->GenerateDeclarationInHostCode(
                    op, i, code, this, pipeline_context);
            }
        }
    }

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
    code.Add("");
    code.Add("// Declare variables for inter warp load balancing");
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

void GpuCodeGenerator::GenerateKernelCallInHostCode(
    std::vector<CypherPipeline *> &pipelines, CodeBuilder &code)
{
    if (!do_inter_warp_lb) {
        throw NotImplementedException(
            "GpuCodeGenerator::GenerateKernelCallInHostCode is not implemented "
            "for non-inter warp load balancing");
    } else {
        int num_warps = int(KernelConstants::DEFAULT_BLOCK_SIZE / 32) *
                        KernelConstants::DEFAULT_GRID_SIZE;
        std::string num_subpipes =
            std::to_string(pipeline_context.sub_pipelines.size());

        for (int pipe_idx = 0; pipe_idx < pipelines.size(); pipe_idx++) {
            auto &cur_input_params = input_kernel_params[pipe_idx];
            auto &cur_output_params = output_kernel_params[pipe_idx];

            code.Add("");
            code.Add("// Pipeline " + std::to_string(pipe_idx));
            code.Add("{");
            code.IncreaseNesting();

            code.Add("cudaMemset(global_info, 0, 64 * sizeof(unsigned int));");
            code.Add(
                "cudaMemset(global_stats_per_lvl, 0, "
                "sizeof(Themis::StatisticsPerLvl) * " +
                num_subpipes + ");");
            uint64_t tableSize =
                pipeline_context.per_pipeline_num_input_tuples[pipe_idx];
            // if tableSize[0] == '*':
            //     // tableSize = tableSize[1:]
            //     code.Add(f'Themis::InitStatisticsPerLvlPtr(global_stats_per_lvl, {num_warps}, {tableSize}, {len(pipe.subpipeSeqs)});')
            // else:
            code.Add("Themis::InitStatisticsPerLvlHost(global_stats_per_lvl, " +
                     std::to_string(num_warps) + ", " +
                     std::to_string(tableSize) + ", " + num_subpipes + ");");

            int bitmapsize = (int((num_warps - 1) / 64) + 1);
            std::string bitmapsize_str = std::to_string(16 + bitmapsize * 16);
            code.Add(
                "cudaMemset(global_bit1, 0, sizeof(unsigned long long) * " +
                bitmapsize_str + ");");

            int input_ptr_count = 0;
            for (const auto &p : cur_input_params)
                if (p.type.find('*') != std::string::npos)
                    ++input_ptr_count;

            int output_ptr_count = 0;
            for (const auto &p : cur_output_params)
                if (p.type.find('*') != std::string::npos)
                    ++output_ptr_count;

            code.Add("void **d_input_data, **d_output_data;");
            code.Add("cudaMalloc(&d_input_data, " +
                     std::to_string(input_ptr_count) + "* sizeof(void *));");
            code.Add("cudaMalloc(&d_output_data, " +
                     std::to_string(output_ptr_count) + "* sizeof(void *));");
            // code.Add("std::cerr << \"Allocated device memory for input and output "
            //          "data pointers.\" << std::endl;");

            int ptr_map_idx = 0;
            int in_idx = 0;
            code.Add("void *h_input_data[" + std::to_string(input_ptr_count) +
                     "] = {");
            code.IncreaseNesting();
            for (const auto &p : cur_input_params) {
                if (p.type.find('*') != std::string::npos) {
                    std::string delimiter =
                        in_idx == input_ptr_count - 1 ? "" : ",";
                    if (pipe_idx == 0) {
                        code.Add("ptr_mappings[" + std::to_string(ptr_map_idx) +
                                 "].address" + delimiter);
                    }
                    else {
                        code.Add(p.name + delimiter);
                    }
                    ++in_idx;
                    ++ptr_map_idx;
                }
                else {
                    // code.Add("// skipped " + p.type + " " + p.name + " = " +
                    //          p.value + ";");
                }
            }
            code.DecreaseNesting();
            code.Add("};");

            int out_idx = 0;
            code.Add("void *h_output_data[" + std::to_string(output_ptr_count) +
                     "] = {");
            code.IncreaseNesting();
            for (const auto &p : cur_output_params) {
                std::string delimiter =
                    out_idx == output_ptr_count - 1 ? "" : ",";
                if (p.type.find('*') != std::string::npos) {
                    code.Add(p.name + delimiter);
                    out_idx++;
                }
                else {
                    // code.Add("// skipped " + p.type + " " + p.name + " = " +
                    //          p.value + ";");
                }
            }
            code.DecreaseNesting();
            code.Add("};");

            code.Add("cudaMemcpy(d_input_data, h_input_data, " +
                     std::to_string(input_ptr_count) + "* sizeof(void *), "
                     "cudaMemcpyHostToDevice);");
            code.Add("cudaMemcpy(d_output_data, h_output_data, " +
                     std::to_string(output_ptr_count) + "* sizeof(void *), "
                     "cudaMemcpyHostToDevice);");
            code.Add("int *d_output_count = 0;");
            code.Add("cudaMalloc((void **)&d_output_count, sizeof(int));");
            code.Add("int output_count = 0;");
            code.Add("cudaMemcpy(d_output_count, &output_count, sizeof(int), "
                     "cudaMemcpyHostToDevice);");
            code.Add("");
            // code.Add(
            //     "std::cerr << \"Prepared input and output data pointers for "
            //     "kernel launch.\" << std::endl;");
            code.Add("int tmp = 0;");
            code.Add("CUresult fr = cuFuncGetAttribute(&tmp, CU_FUNC_ATTRIBUTE_NUM_REGS, gpu_kernel0);");
            code.Add("if (fr != CUDA_SUCCESS) {");
            code.IncreaseNesting();
            code.Add("const char *n,*s; cuGetErrorName(fr,&n); cuGetErrorString(fr,&s);");
            code.Add(
                "std::cerr << \"cuFuncGetAttribute failed: \" << "
                "(n?n:\"unknown\") << \" - \" << (s?s:\"unknown\") << "
                "std::endl;");
            code.DecreaseNesting();
            code.Add("}");

            code.Add(
                "void *args[] = { &d_input_data, &d_output_data, "
                "&d_output_count, &global_num_idle_warps, &global_scan_offset, "
                "&gts, &size_of_stack_per_warp, &global_stats_per_lvl, "
                "&global_bit1, &global_bit2 };");

            // code.Add("std::cerr << \"Launching kernel gpu_kernel" +
            //          std::to_string(pipe_idx) + "\" << std::endl;");
            code.Add(
                "CUresult r = cuLaunchKernel(gpu_kernel" +
                std::to_string(pipe_idx) +
                ", gridSize, 1, 1, blockSize, 1, 1, 0, 0, args, nullptr);");
            // code.Add("std::cerr << \"Kernel gpu_kernel" +
            //          std::to_string(pipe_idx) + " launched.\" << std::endl;");

            // error handling
            code.Add("if (r != CUDA_SUCCESS) {");
            code.IncreaseNesting();
            code.Add("const char *name = nullptr, *str = nullptr;");
            code.Add("cuGetErrorName(r, &name);");
            code.Add("cuGetErrorString(r, &str);");
            code.Add(
                "std::cerr << \"cuLaunchKernel failed: \" << "
                "(name?name:\"unknown\""
                ") << \" - \" << (str?str:\"unknown\""
                ") << std::endl;");
            code.Add("throw std::runtime_error(\"cuLaunchKernel failed\");");
            code.DecreaseNesting();
            code.Add("}");
            code.Add("cudaError_t errSync = cudaDeviceSynchronize();");
            code.Add("if (errSync != cudaSuccess) {");
            code.IncreaseNesting();
            code.Add(
                "std::cerr << \"sync error: \" << cudaGetErrorString(errSync) "
                "<< "
                "std::endl;");
            code.Add(
                "throw std::runtime_error(\"cudaDeviceSynchronize failed\");");
            code.DecreaseNesting();
            code.Add("}");

            code.Add("cudaMemcpy(&output_count, d_output_count, sizeof(int), "
                     "cudaMemcpyDeviceToHost);");
            code.Add("std::cerr << \"Pipeline " + std::to_string(pipe_idx) +
                     " execution finished. Output count: \" << output_count << "
                     "std::endl;");
            code.Add("result_count = output_count;");

            code.DecreaseNesting();
            code.Add("} // end of pipeline " + std::to_string(pipe_idx));
        }

        // Print output data
        GeneratePrintResultsInHostCode(pipelines, code);
    }
}

void GpuCodeGenerator::GeneratePrintResultsInHostCode(
    std::vector<CypherPipeline *> &pipelines, CodeBuilder &code)
{
    code.Add("// Print results");
    auto &output_schema = pipelines.back()->GetSink()->GetSchema();
    auto &output_column_names = output_schema.getStoredColumnNamesRef();
    auto &output_column_types = output_schema.getStoredTypesRef();

    // Declare output host variables
    int tmp_cnt = 256;  // hardcoded for now, should be dynamic
    std::string output_count = std::to_string(tmp_cnt);
    std::vector<std::string> ctypes;
    ctypes.reserve(output_column_types.size());
    for (size_t col_idx = 0; col_idx < output_column_names.size(); col_idx++) {
        LogicalType &type = output_column_types[col_idx];
        ctypes.push_back(ConvertLogicalTypeToPrimitiveType(type, true));
    }

    for (size_t col_idx = 0; col_idx < output_column_names.size(); col_idx++) {
        std::string &ctype = ctypes[col_idx];
        if (ctype == "str_t") {
            code.Add("std::vector<" + ctype + "> h_res" +
                     std::to_string(col_idx) + "_hdrs(" + output_count + ");");
            code.Add("std::vector<std::string> h_res" +
                     std::to_string(col_idx) + "(" + output_count + ");");
        }
        else {
            code.Add("std::vector<" + ctype + "> h_res" +
                     std::to_string(col_idx) + "(" + output_count + ");");
        }
    }

    for (size_t col_idx = 0; col_idx < output_column_names.size(); col_idx++) {
        std::string &ctype = ctypes[col_idx];

        std::stringstream hex_stream;
        hex_stream << std::hex << (0x100000 + col_idx);
        std::string hex_suffix = hex_stream.str().substr(1);
        std::string output_param_name = "result_" + hex_suffix + "_data";
        std::string destination = "h_res" + std::to_string(col_idx);
        if (ctype == "str_t") {
            code.Add("cudaMemcpy(" + destination + "_hdrs.data(), " +
                     output_param_name + ", sizeof(str_t) * " + output_count +
                     ", cudaMemcpyDeviceToHost);");
            code.Add("for (int i = 0; i < " + output_count + "; i++) {");
            code.IncreaseNesting();
            code.Add("const auto &s = " + destination + "_hdrs[i];");
            code.Add("const uint32_t len = s.value.inlined.length;");
            code.Add("if (len == 0) continue;");
            code.Add("if (len <= 12) {");
            code.IncreaseNesting();
            code.Add(destination +
                     "[i] = std::string(s.value.inlined.inlined, 0, len);");
            code.DecreaseNesting();
            code.Add("} else {");
            code.IncreaseNesting();
            code.Add("std::string dst;");
            code.Add("dst.resize(len);");
            code.Add("const uint32_t prefix_len = (len >= 4) ? 4u : len;");
            code.Add("memcpy(dst.data(), s.value.pointer.prefix, prefix_len);");
            code.Add("if (len > 4) {");
            code.IncreaseNesting();
            code.Add(
                "const void *d_tail = static_cast<const void "
                "*>(s.value.pointer.ptr);");
            code.Add(
                "cudaMemcpy(dst.data() + 4, d_tail, len - 4, "
                "cudaMemcpyDeviceToHost);");
            code.DecreaseNesting();
            code.Add("}");
            code.Add(destination + "[i] = std::move(dst);");
            code.DecreaseNesting();
            code.Add("}");
            code.DecreaseNesting();
            code.Add("}");  // end of for loop
        }
        else {
            code.Add("cudaMemcpy(h_res" + std::to_string(col_idx) +
                     ".data(), " + output_param_name + ", sizeof(" + ctype +
                     ") * " + output_count + ", cudaMemcpyDeviceToHost);");
        }
    }

    std::string format_str = "";
    std::string args_str = "";
    code.Add("std::cout << \"Query results: \" << std::endl;");
    for (size_t col_idx = 0; col_idx < output_column_names.size();
         col_idx++) {
        std::string &col_name = output_column_names[col_idx];
        auto &col_type = output_column_types[col_idx];
        format_str += ConvertLogicalTypeToFormatStr(col_type);
        args_str += ConvertLogicalTypeToArgStr(col_type, col_idx);
        if (col_idx < output_column_names.size() - 1) {
            code.Add("std::cout << \"" + col_name + ", \";");
            format_str += ", ";
            args_str += ", ";
        } else {
            code.Add("std::cout << \"" + col_name + "\";");
        }
    }
    code.Add("std::cout << std::endl;");
    code.Add("for (int i = 0; i < result_count && i < 256; i++) {");
    code.IncreaseNesting();
    code.Add("printf(\"" + format_str + "\\n\"," + args_str + ");");
    code.DecreaseNesting();
    code.Add("}");
}

bool GpuCodeGenerator::CompileGeneratedCode()
{
    SCOPED_TIMER_SIMPLE(CompileGeneratedCode, spdlog::level::info,
                        spdlog::level::info);
    if (generated_gpu_code.empty() || generated_cpu_code.empty())
        return false;

    // Compile the generated GPU code using nvrtc
    SUBTIMER_START(CompileGeneratedCode, "CompileWithNVRTC");
    generated_gpu_code =
        gpu_include_header + global_declarations + generated_gpu_code;
    // for debug - write to files
    gpu_code_file.open("generated_gpu_code.cu");
    gpu_code_file << generated_gpu_code << std::endl;
    gpu_code_file.close();

    CUmodule gpu_module = nullptr;
    std::vector<CUfunction> initfns;
    auto success = jit_compiler->CompileWithNVRTC(
        generated_gpu_code, "gpu_kernel", num_pipelines_compiled, initfn_names,
        gpu_module, kernels, initfns);
    SUBTIMER_STOP(CompileGeneratedCode, "CompileWithNVRTC");

    // Check if the GPU code compilation was successful
    if (!success)
        return false;

    // Compile the generated CPU code using ORC JIT
    SUBTIMER_START(CompileGeneratedCode, "CompileWithORCLLJIT");
    generated_cpu_code =
        cpu_include_header + global_declarations + generated_cpu_code;
    cpu_code_file.open("generated_cpu_code.cpp");
    cpu_code_file << generated_cpu_code << std::endl;
    cpu_code_file.close();
    auto success_orc = jit_compiler->CompileWithORCLLJIT(
        generated_cpu_code, kernels, initfn_names, initfns);
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
    input_kernel_params.push_back(std::vector<KernelParam>());
    output_kernel_params.push_back(std::vector<KernelParam>());

    if (pipeline.GetPipelineLength() == 0) {
        return;
    }

    auto source_op = pipeline.GetSource();
    auto source_it = operator_generators.find(source_op->GetOperatorType());
    if (source_it != operator_generators.end()) {
        source_it->second->GenerateInputKernelParameters(
            source_op, this, context, pipeline_context,
            input_kernel_params.back(), scan_column_infos);
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
            sink_op, this, context, pipeline_context,
            output_kernel_params.back());
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
    LogicalType &type, bool do_strip, bool get_short_name)
{
    PhysicalType p_type = type.InternalType();
    std::string type_name;
    if (get_short_name) {
        switch (p_type) {
            case PhysicalType::BOOL:
                type_name = "bool";
                break;
            case PhysicalType::INT8:
                type_name = "char";
                break;
            case PhysicalType::INT16:
                type_name = "short";
                break;
            case PhysicalType::INT32:
                type_name = "int";
                break;
            case PhysicalType::INT64:
                type_name = "int64";
                break;
            case PhysicalType::INT128:
                type_name = "hugeint";
                break;
            case PhysicalType::UINT8:
                type_name = "uchar";
                break;
            case PhysicalType::UINT16:
                type_name = "ushort";
                break;
            case PhysicalType::UINT32:
                type_name = "uint";
                break;
            case PhysicalType::UINT64:
                type_name = "uint64";
                break;
            case PhysicalType::FLOAT:
                type_name = "float";
                break;
            case PhysicalType::DOUBLE:
                type_name = "double";
                break;
            case PhysicalType::VARCHAR:
                type_name = "str";
                break;
            default:
                throw std::runtime_error("Unsupported physical type: " +
                                         std::to_string((uint8_t)p_type));
        }
    }
    else {
        if (type.id() == LogicalTypeId::DECIMAL) {
            switch (p_type) {
                case PhysicalType::INT16:
                    type_name = "decimal_int16_t ";
                    break;
                case PhysicalType::INT32:
                    type_name = "decimal_int32_t ";
                    break;
                case PhysicalType::INT64:
                    type_name = "decimal_int64_t ";
                    break;
                case PhysicalType::INT128:
                    type_name = "decimal_int128_t ";
                    break;
                default:
                    type_name = "decimal_int64_t ";  // fallback
            }
        }
        else {
            switch (p_type) {
                case PhysicalType::BOOL:
                    type_name = "bool ";
                    break;
                case PhysicalType::INT8:
                    type_name = "char ";
                    break;
                case PhysicalType::INT16:
                    type_name = "short ";
                    break;
                case PhysicalType::INT32:
                    type_name = "int ";
                    break;
                case PhysicalType::INT64:
                    type_name = "long long ";
                    break;
                case PhysicalType::INT128:
                    type_name = "hugeint_t ";
                    break;
                case PhysicalType::UINT8:
                    type_name = "unsigned char ";
                    break;
                case PhysicalType::UINT16:
                    type_name = "unsigned short ";
                    break;
                case PhysicalType::UINT32:
                    type_name = "unsigned int ";
                    break;
                case PhysicalType::UINT64:
                    type_name = "unsigned long long ";
                    break;
                case PhysicalType::FLOAT:
                    type_name = "float ";
                    break;
                case PhysicalType::DOUBLE:
                    type_name = "double ";
                    break;
                case PhysicalType::VARCHAR:
                    type_name = "str_t ";  // TODO string_t?
                    break;
                default:
                    throw std::runtime_error("Unsupported physical type: " +
                                             std::to_string((uint8_t)p_type));
            }
        }
    }
    
    if (do_strip) {
        // Strip the trailing space
        if (!type_name.empty() && type_name.back() == ' ') {
            type_name.pop_back();
        }
    }
    return type_name;
}

std::string GpuCodeGenerator::ConvertLogicalTypeToFormatStr(LogicalType &type)
{
    PhysicalType p_type = type.InternalType();
    std::string format_str;
    if (type.id() == LogicalTypeId::DECIMAL) {
        format_str = "%s";
    }
    else {
        switch (p_type) {
            case PhysicalType::BOOL:
                format_str = "%s";
                break;
            case PhysicalType::INT8:
                format_str = "%hhd";
                break;
            case PhysicalType::INT16:
                format_str = "%hd";
                break;
            case PhysicalType::INT32:
                format_str = "%d";
                break;
            case PhysicalType::INT64:
                format_str = "%lld";
                break;
            case PhysicalType::INT128:
                // format_str = "%s";
                format_str = "%llu";
                break;
            case PhysicalType::UINT8:
                format_str = "%hhu";
                break;
            case PhysicalType::UINT16:
                format_str = "%hu";
                break;
            case PhysicalType::UINT32:
                format_str = "%u";
                break;
            case PhysicalType::UINT64:
                format_str = "%llu";
                break;
            case PhysicalType::FLOAT:
                format_str = "%f";
                break;
            case PhysicalType::DOUBLE:
                format_str = "%f";
                break;
            case PhysicalType::VARCHAR:
                format_str = "%s";
                break;
            default:
                throw std::runtime_error("Unsupported physical type: " +
                                            std::to_string((uint8_t)p_type));
        }
    }
    return format_str;
}

std::string GpuCodeGenerator::ConvertLogicalTypeToArgStr(LogicalType &type, size_t col_idx)
{
    PhysicalType p_type = type.InternalType();
    std::string var_name = "h_res" + std::to_string(col_idx) + "[i]";
    std::string args_str;
    if (type.id() == LogicalTypeId::DECIMAL) {
        std::string scale = std::to_string(DecimalType::GetScale(type));
        args_str = "decimal_to_string(" + var_name + ", " + scale + ").c_str()";
    }
    else {
        switch (p_type) {
            case PhysicalType::BOOL:
                args_str = var_name + " ? \"true\" : \"false\"";
                break;
            case PhysicalType::INT8:
            case PhysicalType::INT16:
            case PhysicalType::INT32:
            case PhysicalType::INT64:
            case PhysicalType::UINT8:
            case PhysicalType::UINT16:
            case PhysicalType::UINT32:
            case PhysicalType::UINT64:
            case PhysicalType::FLOAT:
            case PhysicalType::DOUBLE:
                args_str = var_name;
                break;
            case PhysicalType::INT128:
                args_str = var_name + ".lower";
                break;
            case PhysicalType::VARCHAR:
                args_str = var_name + ".c_str()";
                break;
            default:
                throw std::runtime_error("Unsupported physical type: " +
                                            std::to_string((uint8_t)p_type));
        }
    }
    return args_str;
}

std::string GpuCodeGenerator::GetValidVariableName(const std::string &name,
                                                   size_t col_idx)
{
    auto it = pipeline_context.column_to_var_map.find(name);
    if (it != pipeline_context.column_to_var_map.end()) {
        return it->second;
    }

    std::stringstream hex_stream;
    hex_stream << std::hex
               << (0x100000 + pipeline_context.input_column_count++);
    std::string hex_suffix = hex_stream.str().substr(1);
    std::string col_name = "attr_" + hex_suffix;
    pipeline_context.column_to_var_map[name] = col_name;
    return col_name;
}

std::string GpuCodeGenerator::GetInitValueForAggregate(
    const BoundAggregateExpression *bound_agg)
{
    std::string init_value;
    auto &agg_func_name = bound_agg->function.name;

    if (agg_func_name == "count" || agg_func_name == "count_star" ||
        agg_func_name == "avg" || agg_func_name == "sum") {
        switch (bound_agg->return_type.InternalType()) {
            case PhysicalType::INT8:
            case PhysicalType::INT16:
            case PhysicalType::INT32:
            case PhysicalType::INT64:
            case PhysicalType::UINT8:
            case PhysicalType::UINT16:
            case PhysicalType::UINT32:
            case PhysicalType::UINT64:
                init_value = "0";
                break;
            case PhysicalType::INT128:
                init_value = "{0, 0}";
                break;
            case PhysicalType::FLOAT:
                init_value = "0.0f";
                break;
            case PhysicalType::DOUBLE:
                init_value = "0.0";
                break;
            default:
                throw NotImplementedException(
                    "Aggregate function not implemented for type: " +
                    bound_agg->return_type.ToString());
        }
    }
    else if (agg_func_name == "min") {
        init_value = Value::MaximumValue(bound_agg->return_type).ToString();
    }
    else if (agg_func_name == "max") {
        init_value = Value::MinimumValue(bound_agg->return_type).ToString();
    }
    else {
        throw NotImplementedException("Aggregate function not implemented: " +
                                      agg_func_name);
    }
    return init_value;
}

void GpuCodeGenerator::GenerateInputCode(CypherPipeline &sub_pipeline,
                                         CodeBuilder &code,
                                         PipelineContext &pipeline_ctx,
                                         PipeInputType input_type)
{
    switch (input_type) {
        case PipeInputType::TYPE_0_FALSE:
        case PipeInputType::TYPE_0_TRUE:
            GenerateInputCodeForType0(sub_pipeline, code, pipeline_ctx);
            break;
        case PipeInputType::TYPE_1_FALSE:
        case PipeInputType::TYPE_1_TRUE:
            GenerateInputCodeForType1(sub_pipeline, code, pipeline_ctx);
            break;
        case PipeInputType::TYPE_2_TRUE:
            GenerateInputCodeForType2(sub_pipeline, code, pipeline_ctx);
            break;
        default:
            break;
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
    if (!pipeline_ctx.do_lb[pipeline_ctx.current_sub_pipeline_index]) {
        return;
    }
    switch (output_type) {
        case PipeOutputType::TYPE_0_TRUE:
        case PipeOutputType::TYPE_0_FALSE:
            GenerateOutputCodeForType0(sub_pipeline, code, pipeline_ctx);
            break;
        case PipeOutputType::TYPE_1_TRUE:
            GenerateOutputCodeForType1(sub_pipeline, code, pipeline_ctx);
            break;
        case PipeOutputType::TYPE_2_TRUE:
            GenerateOutputCodeForType2(sub_pipeline, code, pipeline_ctx);
            break;
        default:
            break;
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
    std::string next_subpipe_id =
        std::to_string(pipeline_ctx.current_sub_pipeline_index + 1);
    auto *last_op = sub_pipeline.GetSink();
    std::string op_id = std::to_string(last_op->GetOperatorID());
    std::string tid_name;
    // #attrsGeneratedByLastOp = lastOp.generatingAttrs
    
    // currentMaterializedAttrs = attrsToDeclareAndMaterialize[-1][-1]

    // attrsToDeclare = {}
    // for attrId, attr in spSeq.outBoundaryAttrs.items():
    //     if attrId in lastOp.generatingAttrs: continue
    //     if attrId in currentMaterializedAttrs: continue
    //     attrsToDeclare[attrId] = attr
    
    // code.Add(self.genAttrDeclaration(spSeq.pipe, attrsToDeclare))
    
    code.Add("unsigned push_active_mask = __ballot_sync(ALL_LANES, active);");
    
    // We currently implement 'aws' case only
    // if self.doInterWarpLB and self.interWarpLbMethod == 'ws':
    //     # Find attributes to push
    //     attrs = {}
    //     for attrId, attr in spSeq.outBoundaryAttrs.items():
    //         if attrId == tid.id: continue
    //         if attrId in lastOp.generatingAttrs: continue
    //         attrs[attrId] = attr
    //     code.Add(self.genWorkSharingPushCode(spSeq, attrs))
    
    code.Add("if (push_active_mask) {");
    code.IncreaseNesting();
    // # generated by the last Op?
    
    // code.Add(f'//{currentMaterializedAttrs}')
    code.Add("if (active) {");
    code.IncreaseNesting();
    code.Add("ts_" + next_subpipe_id + "_range.set(local" + op_id + "_range);");

    // for attrId, attr in spSeq.outBoundaryAttrs.items():
    //     if attrId in lastOp.generatingAttrs: continue
    //     if attrId in currentMaterializedAttrs:
    //         code.Add(f'ts_{spSeq.id+1}_{attr.id_name} = {attr.id_name};')
    //     elif attrId in spSeq.inBoundaryAttrs:
    //         code.Add(f'ts_{spSeq.id+1}_{attr.id_name} = ts_{spSeq.id}_{attr.id_name};')
    //     else:
    //         code.Add(f'ts_{spSeq.id+1}_{attr.id_name} = {spSeq.pipe.originExpr[attrId]};')
    code.DecreaseNesting();
    code.Add("}");
    code.Add("int ts_src = 32;");
    code.Add("Themis::DistributeFromPartToDPart(thread_id, " + next_subpipe_id +
             ", ts_src, ts_" + next_subpipe_id + "_range, ts_" +
             next_subpipe_id + "_range_cached);");
    code.Add("Themis::UpdateMaskAtLoopLvl(" + next_subpipe_id + ", ts_" +
             next_subpipe_id + "_range_cached, mask_32, mask_1);");
    // for attrId, attr in spSeq.outBoundaryAttrs.items():
    for(;;) {
        // if attrId in lastOp.generatingAttrs: continue
        code.Add("{");
        code.IncreaseNesting();
        // name = f'ts_{spSeq.id+1}_{attr.id_name}'
        // if attr.dataType == Type.STRING:
        //     code.Add(f'char* start = (char*) __shfl_sync(ALL_LANES, (uint64_t) {name}.start, ts_src);')
        //     code.Add(f'char* end = (char*) __shfl_sync(ALL_LANES, (uint64_t) {name}.end, ts_src);')
        //     code.Add('if (ts_src < 32) {')
        //     code.Add(f'{name}_cached.start = start;')
        //     code.Add(f'{name}_cached.end = end;')
        //     code.Add('}')
        // elif attr.dataType == Type.PTR_INT:
        //     code.Add(f'uint64_t cache = __shfl_sync(ALL_LANES, (uint64_t){name}, ts_src);')
        //     code.Add(f'if (ts_src < 32) {name}_cached = (int*) cache;')
        // else:
        //     code.Add(f'{langType(attr.dataType)} cache = __shfl_sync(ALL_LANES, {name}, ts_src);')
        //     code.Add(f'if (ts_src < 32) {name}_cached = cache;')
        code.DecreaseNesting();
        code.Add("}");
    }
    code.DecreaseNesting();
    code.Add("}");
    
    // Find the outermost loop level
    std::string loop_lvl = std::to_string(FindLowerLoopLvl(pipeline_context));
    code.Add("if (!(mask_32 & (0x1 << " + next_subpipe_id + "))) {");
    code.IncreaseNesting();
    code.Add("if (mask_32 & (0x1 << " + loop_lvl + ")) lvl = " + loop_lvl +
             ";");
    code.Add("else Themis::ChooseLvl(thread_id, mask_32, mask_1, lvl);");
    code.Add("continue;");
    code.DecreaseNesting();
    code.Add("}");
    code.Add("lvl = "+next_subpipe_id+";");
    code.DecreaseNesting();
    code.Add("}");
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
        code.Add("if (ts_cnt >= 32) {");
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
        PipeInputType inputType = GetPipeInputType(
            pipeline_context.sub_pipelines[loop_lvl], loop_lvl);
        if (inputType == PipeInputType::TYPE_0_FALSE ||
            inputType == PipeInputType::TYPE_0_TRUE ||
            inputType == PipeInputType::TYPE_1_FALSE ||
            inputType == PipeInputType::TYPE_1_TRUE) {
            return loop_lvl;
        }
        loop_lvl--;
    }
}

PipeInputType GpuCodeGenerator::GetPipeInputType(CypherPipeline &sub_pipeline,
                                                 int sub_pipe_idx)
{
    PipeInputType pipe_input_type;
    auto first_op = sub_pipeline.GetSource();
    switch (first_op->GetOperatorType()) {
        case PhysicalOperatorType::NODE_SCAN: {
            auto scan_op = dynamic_cast<PhysicalNodeScan *>(first_op);
            if (scan_op->is_filter_pushdowned) {
                if (sub_pipeline.GetSink() == first_op) {
                    pipe_input_type = PipeInputType::TYPE_0_FALSE;
                }
                else {
                    pipe_input_type = PipeInputType::TYPE_2_TRUE;
                }
            }
            else {
                pipe_input_type = PipeInputType::TYPE_0_FALSE;
            }
            break;
        }
        case PhysicalOperatorType::FILTER: {
            pipe_input_type = PipeInputType::TYPE_2_TRUE;
            break;
        }
        case PhysicalOperatorType::HASH_AGGREGATE: {
            if (sub_pipe_idx == 0) {
                pipe_input_type = PipeInputType::TYPE_0_FALSE;
            }
            else {
                pipe_input_type = PipeInputType::TYPE_2_FALSE;
            }
            break;
        }
        case PhysicalOperatorType::ADJ_IDX_JOIN: {
            pipe_input_type = PipeInputType::TYPE_1_TRUE;
            break;
        }
        default:
            throw NotImplementedException(
                "Input type " +
                std::to_string((uint8_t)first_op->GetOperatorType()));
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
            pipe_output_type = PipeOutputType::TYPE_0_FALSE;
            break;
        case PhysicalOperatorType::FILTER:
            pipe_output_type = PipeOutputType::TYPE_2_TRUE;
            break;
        case PhysicalOperatorType::NODE_SCAN:
            // Filter pushdowned scan case
            D_ASSERT(dynamic_cast<PhysicalNodeScan *>(last_op)
                         ->is_filter_pushdowned);
            pipe_output_type = PipeOutputType::TYPE_2_TRUE;
            break;
        case PhysicalOperatorType::HASH_AGGREGATE:
            pipe_output_type = PipeOutputType::TYPE_0_FALSE;
            break;
        case PhysicalOperatorType::ADJ_IDX_JOIN:
            pipe_output_type = PipeOutputType::TYPE_1_TRUE;
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
        std::string col_name = GetValidVariableName(
            column, cur_cols_to_be_materialized[i].pos);
        std::string ctype = ConvertLogicalTypeToPrimitiveType(
            cur_cols_to_be_materialized[i].type);
        code.Add(ctype + col_name + ";");
    }
}

void GpuCodeGenerator::GenerateCodeForLocalVariable(
    CodeBuilder &code, PipelineContext &pipeline_context)
{
    // Declaration only in this point
    int cur_subpipe_idx = pipeline_context.current_sub_pipeline_index;
    auto &sub_pipeline = pipeline_context.sub_pipelines[cur_subpipe_idx];
    for (size_t i = 0; i < sub_pipeline.GetPipelineLength(); i++) {
        auto *op = sub_pipeline.GetIdxOperator(i);
        GenerateCodeForLocalVariable(op, i, code, pipeline_context);
    }
}

void GpuCodeGenerator::GeneratePipelineCode(CypherPipeline &pipeline,
                                            CodeBuilder &code)
{
    for (size_t i = 0; i < pipeline_context.sub_pipelines.size(); i++) {
        std::cerr << "Generating code for sub-pipeline " << i << std::endl;
        auto &sub_pipeline = pipeline_context.sub_pipelines[i];
        pipeline_context.current_sub_pipeline_index = i;

        // for debugging purposes
        std::cerr << sub_pipeline.toString() << std::endl;

        // Generate code for each sub-pipeline
        GenerateSubPipelineCode(sub_pipeline, code);
    }
}

void GpuCodeGenerator::GenerateSubPipelineCode(CypherPipeline &sub_pipeline,
                                               CodeBuilder &code)
{
    // Advance to the next operator
    AdvanceOperator();

    // Analyze pipeline input/output type
    PipeInputType pipe_input_type = GetPipeInputType(
        sub_pipeline, pipeline_context.current_sub_pipeline_index);
    PipeOutputType pipe_output_type = GetPipeOutputType(sub_pipeline);

    // Generate input code based on the source operator type
    GenerateInputCode(sub_pipeline, code, pipeline_context, pipe_input_type);

    // Generate code for materialization if needed
    GenerateCodeForMaterialization(code, pipeline_context);

    // Generate code for local variable
    GenerateCodeForLocalVariable(code, pipeline_context);
    
    // Generate code for sub-pipeline operators
    if (pipe_input_type == PipeInputType::TYPE_0_FALSE) {
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

    if (pipeline_context.do_lb[pipeline_context.current_sub_pipeline_index]) {
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

void GpuCodeGenerator::GenerateGlobalDeclaration(
    CypherPhysicalOperator *op, CypherPhysicalOperator *prev_op, size_t op_idx,
    CodeBuilder &code, PipelineContext &pipeline_ctx)
{
    auto it = operator_generators.find(op->GetOperatorType());
    if (it != operator_generators.end()) {
        it->second->GenerateGlobalDeclaration(op, prev_op, op_idx, code, this,
                                              context, pipeline_ctx);
    }
    else {
        // Default handling for unknown operators
        code.Add("// Unknown operator type: " +
                 std::to_string(static_cast<int>(op->GetOperatorType())));
    }
}

void GpuCodeGenerator::GenerateCodeForLocalVariable(
    CypherPhysicalOperator *op, size_t op_idx, CodeBuilder &code,
    PipelineContext &pipeline_ctx)
{
    auto it = operator_generators.find(op->GetOperatorType());
    if (it != operator_generators.end()) {
        it->second->GenerateCodeForLocalVariable(op, op_idx, code, this,
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
        // TODO utilize attr to source mapping
        for (size_t i = 0; i < columns_to_materialize.size(); i++) {
            auto &col_name = columns_to_materialize[i].name;
            auto col_type = columns_to_materialize[i].type;
            auto col_pos = columns_to_materialize[i].pos;
            auto col_id = scan_op->scan_projection_mapping[0][col_pos];

            // Generate code for materializing this column
            D_ASSERT(pipeline_ctx.attribute_tid_mapping.find(col_name) !=
                     pipeline_ctx.attribute_tid_mapping.end());
            std::string tid_name = pipeline_ctx.attribute_tid_mapping[col_name];
            std::string sanitized_col_name =
                code_gen->GetValidVariableName(col_name, col_pos);

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
                    if (scan_op->eq_filter_pushdown_values[i].type() ==
                        LogicalType::VARCHAR) {
                        // For string values, we need to add quotes
                        value_str = "\"" + value_str + "\"";
                    }
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

void NodeScanCodeGenerator::GenerateInputKernelParameters(
    CypherPhysicalOperator *op, GpuCodeGenerator *code_gen,
    ClientContext &context, PipelineContext &pipeline_context,
    std::vector<KernelParam> &input_kernel_params,
    std::vector<ScanColumnInfo> &scan_column_infos)
{
    auto scan_op = dynamic_cast<PhysicalNodeScan *>(op);

    uint64_t num_tuples_total = 0;
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
        scan_column_info.pipeline_id =
            pipeline_context.current_pipeline->GetPipelineId();
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
            num_tuples_total += num_tuples_in_extent;

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
        }
    }
    pipeline_context.per_pipeline_num_input_tuples.push_back(num_tuples_total);
}

void NodeScanCodeGenerator::AnalyzeOperatorForMaterialization(
    CypherPhysicalOperator *op, int sub_idx, int op_idx,
    PipelineContext &pipeline_context, GpuCodeGenerator *code_gen)
{
    auto *scan_op = dynamic_cast<PhysicalNodeScan *>(op);
    auto &output_schema = op->GetSchema();
    auto &output_column_names = output_schema.getStoredColumnNamesRef();
    auto &output_column_types = output_schema.getStoredTypesRef();
    auto &columns_to_be_materialized =
        pipeline_context.columns_to_be_materialized;
    auto &sub_pipeline_tids = pipeline_context.sub_pipeline_tids;
    auto &mat_target_columns =
        pipeline_context.materialization_target_columns[sub_idx];

    // Create tid mapping info
    std::string tid = "tid_" + std::to_string(sub_idx) + "_" +
                      std::to_string(op_idx) + "_" +
                      std::to_string(scan_op->oids[0]);
    for (const auto &col_name : output_column_names) {
        D_ASSERT(pipeline_context.attribute_tid_mapping.find(col_name) ==
                 pipeline_context.attribute_tid_mapping.end());
        pipeline_context.attribute_tid_mapping[col_name] = tid;
    }
    sub_pipeline_tids[sub_idx].push_back(tid);

    // Create source mapping info
    for (size_t i = 0; i < scan_op->oids.size(); i++) {
        uint64_t oid = scan_op->oids[i];
        auto &scan_projection_mapping = scan_op->scan_projection_mapping[i];
        for (size_t j = 0; j < scan_projection_mapping.size(); j++) {
            auto &col_name = output_column_names[j];
            std::string source_name =
                "gr" + std::to_string(oid) + "_col_" +
                std::to_string(scan_projection_mapping[j] - 1) + "_data[" +
                tid + "]";
            D_ASSERT(pipeline_context.attribute_source_mapping.find(col_name) ==
                     pipeline_context.attribute_source_mapping.end());
            pipeline_context.attribute_source_mapping[col_name] = source_name;
        }
    }

    // Check if the scan operator has filter pushdown
    // If it does, we need to materialize the columns
    if (!scan_op->is_filter_pushdowned) {
        return;
    }

    // Get referenced columns from the filter pushdown
    std::vector<uint64_t> referenced_columns;
    if (scan_op->filter_pushdown_type == FilterPushdownType::FP_EQ ||
        scan_op->filter_pushdown_type == FilterPushdownType::FP_RANGE) {
        for (const auto &key_idx :
             scan_op->filter_pushdown_key_idxs_in_output) {
            referenced_columns.push_back(key_idx);
        }
    }
    else {
        code_gen->GetReferencedColumns(scan_op->filter_expression.get(),
                             referenced_columns);
    }

    // TODO handling multiple graphlets
    D_ASSERT(scan_op->oids.size() == 1);
    for (const auto &col_idx : referenced_columns) {
        D_ASSERT(col_idx >= 0 && col_idx < output_column_names.size());
        auto &col_name = output_column_names[col_idx];
        if (mat_target_columns.find(col_name) == mat_target_columns.end()) {
            mat_target_columns.insert(col_name);
            columns_to_be_materialized[sub_idx].push_back(
                Attr{col_name, output_column_types[col_idx], col_idx});
        }
    }
}

void ProjectionCodeGenerator::GenerateCode(
    CypherPhysicalOperator *op, CodeBuilder &code, GpuCodeGenerator *code_gen,
    ClientContext &context, PipelineContext &pipeline_ctx, bool is_main_loop)
{
    auto proj_op = dynamic_cast<PhysicalProjection *>(op);
    D_ASSERT(proj_op != nullptr);

    code.Add("// Projection operator");
    ExpressionCodeGenerator expr_gen(context);
    std::unordered_map<uint64_t, std::string> column_map;
    auto &columns_to_materialize =
        pipeline_ctx.columns_to_be_materialized
            [pipeline_ctx.current_sub_pipeline_index];

    auto &output_schema = proj_op->GetSchema();
    auto &output_column_names = output_schema.getStoredColumnNamesRef();
    auto &output_column_types = output_schema.getStoredTypesRef();

    D_ASSERT(proj_op->children.size() == 1);
    auto &input_schema = proj_op->children[0]->GetSchema();
    auto &input_column_names = input_schema.getStoredColumnNamesRef();

    // Create column map (input column index -> variable name)
    for (size_t i = 0; i < input_column_names.size(); i++) {
        auto &col_name = input_column_names[i];
        std::string sanitized_col_name =
            code_gen->GetValidVariableName(col_name, i);
        column_map[i] = sanitized_col_name;
    }

    // if (pipeline_ctx.input_proj_vars_generated.find(
    //         pipeline_ctx.current_sub_pipeline_index) ==
    //     pipeline_ctx.input_proj_vars_generated.end()) {

    //     for (size_t i = 0; i < columns_to_materialize.size(); i++) {
    //         std::string &col_name = columns_to_materialize[i].name;
    //         auto &col_pos = columns_to_materialize[i].pos;
            
    //         std::string sanitized_col_name =
    //             code_gen->GetValidVariableName(col_name, col_pos);
    //         column_map[i] = sanitized_col_name;
    //     }
    //     pipeline_ctx.input_proj_vars_generated.insert(
    //         pipeline_ctx.current_sub_pipeline_index);
    // }
    // else {
    //     for (size_t i = 0; i < columns_to_materialize.size(); i++) {
    //         std::string &col_name = columns_to_materialize[i].name;
    //         auto &col_pos = columns_to_materialize[i].pos;
    //         std::string sanitized_col_name =
    //             code_gen->GetValidVariableName(col_name, col_pos);
    //         column_map[i] = sanitized_col_name;
    //     }
    // }

    // Process each projection expression
    size_t output_var_counter = 0;
    D_ASSERT(proj_op->expressions.size() == output_column_names.size());
    auto &mat_target_columns = pipeline_ctx.materialization_target_columns
                                   [pipeline_ctx.current_sub_pipeline_index];
    for (size_t expr_idx = 0; expr_idx < proj_op->expressions.size();
         expr_idx++) {
        auto &expr = proj_op->expressions[expr_idx];
        auto &col_name = output_column_names[expr_idx];
        if (expr->expression_class == ExpressionClass::BOUND_REF) {
            // check if the expression required materialization
            if (mat_target_columns.find(col_name) == mat_target_columns.end()) {
                // This column need not be materialized, skip it
                continue;
            }

            // check if the expression is already materialized
            bool is_materialized =
                pipeline_ctx.column_materialized.find(col_name) !=
                    pipeline_ctx.column_materialized.end() &&
                pipeline_ctx.column_materialized[col_name];
            if (!is_materialized) {
                // Column is not materialized, we need to generate code for it
                // and mark it as materialized
                auto it = pipeline_ctx.attribute_source_mapping.find(col_name);
                if (it != pipeline_ctx.attribute_source_mapping.end()) {
                    std::string source_var_name = it->second;
                    code.Add("// " + col_name +
                             " is mapped to source variable " +
                             source_var_name);
                    std::string sanitized_col_name =
                        code_gen->GetValidVariableName(col_name, expr_idx);
                    code.Add(sanitized_col_name + " = " + source_var_name +
                             ";");
                }
                pipeline_ctx.column_materialized[col_name] = true;
            }
            continue;
        }

        std::string proj_var_name;
        std::string result_expr = expr_gen.GenerateExpressionCode(
            expr.get(), code, pipeline_ctx, column_map);
        auto it = pipeline_ctx.column_to_var_map.find(col_name);
        if (it != pipeline_ctx.column_to_var_map.end()) {
            // If the column is already mapped to a variable, use it
            proj_var_name = it->second;
            code.Add(proj_var_name + " = " + result_expr + ";");
        }
        else {
            // TODO: this should not be happened
            std::stringstream hex_stream;
            hex_stream << std::hex << (0x100000 + output_var_counter);
            std::string hex_suffix = hex_stream.str().substr(1);
            proj_var_name = "result_" + hex_suffix;
            std::string cuda_type =
                code_gen->ConvertLogicalTypeToPrimitiveType(expr->return_type);
            code.Add(cuda_type + proj_var_name + " = " + result_expr + ";");
        }

        pipeline_ctx.projection_variable_names[expr_idx] = proj_var_name;
        output_var_counter++;
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

void ProjectionCodeGenerator::AnalyzeOperatorForMaterialization(
    CypherPhysicalOperator *op, int sub_idx, int op_idx,
    PipelineContext &pipeline_context, GpuCodeGenerator *code_gen)
{
    auto *proj_op = dynamic_cast<PhysicalProjection *>(op);
    D_ASSERT(proj_op->children.size() == 1);
    auto *prev_op = proj_op->children[0];
    auto &input_schema = prev_op->GetSchema();
    auto &input_column_names = input_schema.getStoredColumnNamesRef();
    auto &input_column_types = input_schema.getStoredTypesRef();
    auto &columns_to_be_materialized =
        pipeline_context.columns_to_be_materialized[sub_idx];
    auto &mat_target_columns =
        pipeline_context.materialization_target_columns[sub_idx];

    for (const auto &expr : proj_op->expressions) {
        if (expr->GetExpressionClass() == ExpressionClass::BOUND_REF) {
            continue;
        }
        std::vector<uint64_t> referenced_columns;
        code_gen->GetReferencedColumns(expr.get(), referenced_columns);
        for (const auto &col_idx : referenced_columns) {
            D_ASSERT(col_idx >= 0 && col_idx < input_column_names.size());
            auto &col_name = input_column_names[col_idx];

            if (mat_target_columns.find(col_name) == mat_target_columns.end()) {
                mat_target_columns.insert(col_name);
                columns_to_be_materialized.push_back(
                    Attr{col_name, input_column_types[col_idx], col_idx});
            }
        }
    }
}

void ProduceResultsCodeGenerator::GenerateCode(
    CypherPhysicalOperator *op, CodeBuilder &code, GpuCodeGenerator *code_gen,
    ClientContext &context, PipelineContext &pipeline_ctx, bool is_main_loop)
{
    auto results_op = dynamic_cast<PhysicalProduceResults *>(op);
    D_ASSERT(results_op != nullptr);

    code.Add("// Produce results operator");

    // Get the output schema to determine what columns to write
    auto &output_schema = results_op->GetSchema();
    auto &output_column_names = output_schema.getStoredColumnNamesRef();
    
    // Materialize columns if needed
    for (size_t col_idx = 0; col_idx < output_column_names.size(); col_idx++) {
        std::string orig_col_name = output_column_names[col_idx];
        std::string tid_name = pipeline_ctx.attribute_tid_mapping[orig_col_name];
        std::string col_name =
            code_gen->GetValidVariableName(orig_col_name, col_idx);

        // type extract
        LogicalType type;
        if (col_idx < pipeline_ctx.output_column_types.size()) {
            type = pipeline_ctx.output_column_types[col_idx];
        } else {
            type = LogicalType::DOUBLE;
            std::cerr << "[DEBUG] Using default DOUBLE type for column " << col_idx << std::endl;
        }
        std::string ctype =
            code_gen->ConvertLogicalTypeToPrimitiveType(type);

        // Check if this column is materialized in the pipeline context
        auto it = pipeline_ctx.column_materialized.find(orig_col_name);
        bool is_materialized =
            it != pipeline_ctx.column_materialized.end() && it->second;

        if (!is_materialized) {
            // Column needs to be materialized from input
            // Find the corresponding input column
            std::string orig_col_name_wo_varname =
                orig_col_name.substr(orig_col_name.find_last_of('.') + 1);
            if (orig_col_name_wo_varname == "_id") {
                throw NotImplementedException(
                    "ProduceResultsCodeGenerator does not support _id column "
                    "materialization yet.");
            }
            else {
                auto it =
                    pipeline_ctx.attribute_source_mapping.find(orig_col_name);
                if (it != pipeline_ctx.attribute_source_mapping.end()) {
                    std::string source_var_name = it->second;
                    code.Add("// " + orig_col_name +
                             " is mapped to source variable " +
                             source_var_name);
                    code.Add(col_name + " = " + source_var_name + ";");
                }
                else {
                    code.Add("// " + orig_col_name +
                             " is not found in attribute source mapping.");
                    // throw InvalidInputException(
                    //     "Output column " + orig_col_name +
                    //     " is not found in attribute source mapping.");
                }
            }
        }
    }

    for (size_t col_idx = 0; col_idx < output_column_names.size(); col_idx++) {
        std::string orig_col_name = output_column_names[col_idx];
        std::string col_name =
            code_gen->GetValidVariableName(orig_col_name, col_idx);
        // type extract
        LogicalType type = pipeline_ctx.output_column_types[col_idx];
        std::string ctype =
            code_gen->ConvertLogicalTypeToPrimitiveType(type);

        std::stringstream hex_stream;
        hex_stream << std::hex << (0x100000 + col_idx);
        std::string hex_suffix = hex_stream.str().substr(1);
        std::string output_param_name = "result_" + hex_suffix;
        std::string output_data_name = output_param_name + "_data";
        std::string output_ptr_name = output_param_name + "_ptr";
        code.Add(ctype + "*" + output_ptr_name + " = static_cast<" + ctype +
                 "*>(" + output_data_name + ");");
        code.Add(output_ptr_name + "[wp] = " + col_name + ";");
    }
}

void ProduceResultsCodeGenerator::GenerateDeclarationInHostCode(
    CypherPhysicalOperator *op, size_t op_idx, CodeBuilder &code,
    GpuCodeGenerator *code_gen, PipelineContext &pipeline_ctx)
{
    auto &output_schema = op->GetSchema();
    auto &output_column_names = output_schema.getStoredColumnNamesRef();
    auto &output_column_types = output_schema.getStoredTypesRef();

    std::string output_count = "256";  // TODO
    code.Add("");
    code.Add("// Declare variables for results");
    for (size_t col_idx = 0; col_idx < output_column_names.size(); col_idx++) {
        std::string &col_name = output_column_names[col_idx];
        LogicalType &type = output_column_types[col_idx];
        std::string ctype = code_gen->ConvertLogicalTypeToPrimitiveType(type);

        std::stringstream hex_stream;
        hex_stream << std::hex << (0x100000 + col_idx);
        std::string hex_suffix = hex_stream.str().substr(1);
        std::string output_param_name = "result_" + hex_suffix + "_data";
        code.Add(ctype + "*" + output_param_name + ";");
        code.Add("cudaMalloc((void**)&" + output_param_name + ", sizeof(" +
                 ctype + ") * " + output_count + ");");
    }
    code.Add("int result_count = 0;");
}

void ProduceResultsCodeGenerator::GenerateCodeForLocalVariable(
    CypherPhysicalOperator *op, size_t op_idx, CodeBuilder &code,
    GpuCodeGenerator *code_gen, PipelineContext &pipeline_ctx)
{
    code.Add("int wp, writeMask, numProj;");
    code.Add("writeMask = __ballot_sync(ALL_LANES, active);");
    code.Add("numProj = __popc(writeMask);");
    // TODO: nout_{self.tableName} -> output_count
    code.Add("if (thread_id == 0) wp = atomicAdd(output_count, numProj);");
    code.Add("wp = __shfl_sync(ALL_LANES, wp, 0);");
    code.Add("wp = wp + __popc(writeMask & prefixlanes);");
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
        std::string sanitized_col_name =
            code_gen->GetValidVariableName(col_name, col_idx);

        // Generate output parameter names
        std::string output_param_name;
        std::stringstream hex_stream;
        hex_stream << std::hex << (0x100000 + col_idx);
        std::string hex_suffix = hex_stream.str().substr(1);
        output_param_name = "result_" + hex_suffix;

        // Add output data buffer parameter
        KernelParam output_data_param;
        output_data_param.name = output_param_name + "_data";
        output_data_param.type =
            "void *";  // Always void* for CUDA compatibility
        output_data_param.is_device_ptr = true;
        output_kernel_params.push_back(output_data_param);
    }
}

void ProduceResultsCodeGenerator::AnalyzeOperatorForMaterialization(
    CypherPhysicalOperator *op, int sub_idx, int op_idx,
    PipelineContext &pipeline_context, GpuCodeGenerator *code_gen)
{
    auto &output_schema = op->GetSchema();
    auto &output_column_names = output_schema.getStoredColumnNamesRef();
    auto &output_column_types = output_schema.getStoredTypesRef();
    auto &columns_to_be_materialized =
        pipeline_context.columns_to_be_materialized[sub_idx];
    auto &mat_target_columns =
        pipeline_context.materialization_target_columns[sub_idx];

    for (size_t col_idx = 0; col_idx < output_column_names.size(); col_idx++) {
        auto &col_name = output_column_names[col_idx];
        auto &col_type = output_column_types[col_idx];
        if (mat_target_columns.find(col_name) == mat_target_columns.end()) {
            mat_target_columns.insert(col_name);
            columns_to_be_materialized.push_back(
                Attr{col_name, col_type, col_idx});
        }
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

void FilterCodeGenerator::AnalyzeOperatorForMaterialization(
    CypherPhysicalOperator *op, int sub_idx, int op_idx,
    PipelineContext &pipeline_context, GpuCodeGenerator *code_gen)
{
    auto *filter_op = dynamic_cast<PhysicalFilter *>(op);
    std::vector<uint64_t> referenced_columns;
    code_gen->GetReferencedColumns(filter_op->expression.get(),
                                   referenced_columns);
    auto &output_schema = op->GetSchema();
    auto &output_column_names = output_schema.getStoredColumnNamesRef();
    auto &output_column_types = output_schema.getStoredTypesRef();
    auto &columns_to_be_materialized =
        pipeline_context.columns_to_be_materialized[sub_idx];
    auto &mat_target_columns =
        pipeline_context.materialization_target_columns[sub_idx];

    for (const auto &col_idx : referenced_columns) {
        D_ASSERT(col_idx >= 0 && col_idx < output_column_names.size());
        auto &col_name = output_column_names[col_idx];
        if (mat_target_columns.find(col_name) == mat_target_columns.end()) {
            mat_target_columns.insert(col_name);
            columns_to_be_materialized.push_back(
                Attr{col_name, output_column_types[col_idx], col_idx});
        }
    }
}

void HashAggregateCodeGenerator::GenerateCode(
    CypherPhysicalOperator *op, CodeBuilder &code, GpuCodeGenerator *code_gen,
    ClientContext &context, PipelineContext &pipeline_ctx, bool is_main_loop)
{
    // Hashaggregate operator is either the first or the last operator
    D_ASSERT(pipeline_ctx.cur_op_idx == 0 ||
             pipeline_ctx.cur_op_idx == pipeline_ctx.total_operators - 1);

    if (pipeline_ctx.cur_op_idx == 0) {
        GenerateSourceSideCode(op, code, code_gen, context, pipeline_ctx);
    }
    else if (pipeline_ctx.cur_op_idx == pipeline_ctx.total_operators - 1) {
        GenerateBuildSideCode(op, code, code_gen, context, pipeline_ctx);
    }
    else {
        throw InvalidInputException("HashAggregate as intermediate operator");
    }
}

void HashAggregateCodeGenerator::GenerateSourceSideCode(
    CypherPhysicalOperator *op, CodeBuilder &code, GpuCodeGenerator *code_gen,
    ClientContext &context, PipelineContext &pipeline_ctx)
{
    auto agg_op = dynamic_cast<PhysicalHashAggregate *>(op);
    // TODO correctness check
    D_ASSERT(
        pipeline_ctx.sub_pipeline_tids[pipeline_ctx.current_sub_pipeline_index]
            .size() == 1);
    std::string tid_name =
        pipeline_ctx
            .sub_pipeline_tids[pipeline_ctx.current_sub_pipeline_index][0];
    code.Add("// HashAggregate source side code");
    if (agg_op->groups.size() != 0) {
        code.Add("active = aht" + std::to_string(agg_op->GetOperatorId()) +
                 "[" + tid_name + "].lock.lock == OnceLock::LOCK_DONE;");
    }
}

void HashAggregateCodeGenerator::GenerateBuildSideCode(
    CypherPhysicalOperator *op, CodeBuilder &code, GpuCodeGenerator *code_gen,
    ClientContext &context, PipelineContext &pipeline_ctx)
{
    auto agg_op = dynamic_cast<PhysicalHashAggregate *>(op);
    if (agg_op->groups.size() == 0) {
        code.Add("// HashAggregate build side code without grouping");
        std::string op_id = std::to_string(agg_op->GetOperatorId());
        if (code_gen->GetKernelArgs().local_aggregation) {
            throw NotImplementedException(
                "Local aggregation is not supported for HashAggregate yet.");
            // for attrId, (inId, reductionType) in self.lop.aggregateTuples.items():
            //     attr, inputIdentifier, reductionType = self.lop.aggregateTuplesCreated[attrId]
            //     inAttr = self.lop.aggregateInAttributes.get(inId, None)
            //     if reductionType == Reduction.COUNT:
            //         code.Add(f'local_{attr.id_name} += 1;')
            //     elif reductionType == Reduction.SUM or reductionType == Reduction.AVG:
            //         code.Add(f'local_{attr.id_name} += {inAttr.id_name};')
            //     elif reductionType == Reduction.MAX:
            //         code.Add(f'local_{attr.id_name} = local_{attr.id_name} < {inAttr.id_name} ? {inAttr.id_name} : local_{attr.id_name};')
            //     elif reductionType == Reduction.MIN:
            //         code.Add(f'local_{attr.id_name} = local_{attr.id_name} > {inAttr.id_name} ? {inAttr.id_name} : local_{attr.id_name};')
        }
        else {
            bool contain_count_func = false;
            for (auto &aggregate : agg_op->aggregates) {
                D_ASSERT(aggregate->GetExpressionType() ==
                        ExpressionType::BOUND_AGGREGATE);
                auto bound_agg =
                    dynamic_cast<BoundAggregateExpression *>(aggregate.get());
                if (bound_agg->function.name == "count_star" ||
                    bound_agg->function.name == "count") {
                    contain_count_func = true;
                    break;
                }
            }

            for (size_t i = 0; i < agg_op->aggregates.size(); i++) {
                auto &aggregate = agg_op->aggregates[i];
                D_ASSERT(aggregate->GetExpressionType() ==
                         ExpressionType::BOUND_AGGREGATE);
                auto bound_agg =
                    dynamic_cast<BoundAggregateExpression *>(aggregate.get());
                auto &agg_function_name = bound_agg->function.name;

                std::string agg_var_name = "agg_" + std::to_string(i);
                std::string dst = "aht" + op_id + "_" + agg_var_name + "[0]";
                std::string agg_type =
                    code_gen->ConvertLogicalTypeToPrimitiveType(
                        bound_agg->return_type, true);
                if (agg_function_name == "count_star" ||
                    agg_function_name == "count") {
                    code.Add("atomicAdd(&" + dst + ", (" + agg_type + ")1);");
                }
                else {
                    // get ref idxs
                    std::vector<uint64_t> ref_idxs;
                    D_ASSERT(bound_agg->children.size() == 1);
                    code_gen->GetReferencedColumns(bound_agg->children[0].get(),
                                                   ref_idxs);
                    D_ASSERT(ref_idxs.size() == 1);
                    std::string inattr = code_gen->GetValidVariableName(
                        pipeline_ctx.input_column_names[ref_idxs[0]],
                        ref_idxs[0]);
                    if (agg_function_name == "sum") {
                        code.Add("atomicAdd(&" + dst + ", " + inattr + ");");
                    }
                    else if (agg_function_name == "avg") {
                        code.Add("atomicAdd(&" + dst + ", " + inattr + ");");
                        if (!contain_count_func) {
                            // If there is no count function, we need to add a count
                            // variable for average calculation
                            std::string dst_cnt = "aht" + op_id + "_" +
                                                  agg_var_name +
                                                  "_cnt[0]";
                            code.Add("atomicAdd(&" + dst_cnt + ", (" +
                                     agg_type + ")1);");
                        }
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
        std::vector<std::string> group_key_sanitized_names;
        for (auto &group_key_idx : group_key_idxs) {
            std::string &orig_col_name =
                pipeline_ctx.input_column_names[group_key_idx];
            std::string col_name = orig_col_name;
            std::string sanitized_col_name =
                code_gen->GetValidVariableName(orig_col_name, group_key_idx);
            group_key_sanitized_names.push_back(sanitized_col_name);
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
                code.Add(sanitized_col_name + " = " + source_name + "[" +
                         tid_name + "];");
            }
        }

        for (size_t i = 0; i < group_key_idxs.size(); i++) {
            auto group_key_idx = group_key_idxs[i];
            std::string col_name = group_key_names[i];
            std::string sanitized_col_name = group_key_sanitized_names[i];
            code.Add("buf_payl." + col_name + " = " + sanitized_col_name + ";");
        }

        for (size_t i = 0; i < group_key_idxs.size(); i++) {
            auto group_key_idx = group_key_idxs[i];
            std::string col_name = group_key_names[i];
            std::string sanitized_col_name = group_key_sanitized_names[i];
            if (pipeline_ctx.input_column_types[group_key_idx].id() ==
                LogicalTypeId::VARCHAR) {
                code.Add("hash_key = hash(hash_key + stringHash(" +
                         sanitized_col_name + "));");
            }
            else {
                code.Add("hash_key = hash(hash_key + ((uint64_t) " +
                         sanitized_col_name + "));");
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
        // TODO int(self.size) how can be determined?
        code.Add("bucketId = (hash_key + numLookups) % 10000;");
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
            std::string sanitized_col_name = group_key_sanitized_names[i];
            if (pipeline_ctx.input_column_types[group_key_idx].id() ==
                LogicalTypeId::VARCHAR) {
                code.Add("bucketFound &= stringEquals(entry." + col_name +
                         ", " + sanitized_col_name + ");");
            }
            else {
                code.Add("bucketFound &= entry." + col_name +
                         " == " + sanitized_col_name + ";");
            }
        }

        code.DecreaseNesting();
        code.Add("}");

        bool contain_count_func = false;
        for (auto &aggregate : agg_op->aggregates) {
            D_ASSERT(aggregate->GetExpressionType() ==
                     ExpressionType::BOUND_AGGREGATE);
            auto bound_agg =
                dynamic_cast<BoundAggregateExpression *>(aggregate.get());
            if (bound_agg->function.name == "count_star" ||
                bound_agg->function.name == "count") {
                contain_count_func = true;
                break;
            }
        }

        for (size_t i = 0; i < agg_op->aggregates.size(); i++) {
            auto &aggregate = agg_op->aggregates[i];
            D_ASSERT(aggregate->GetExpressionType() ==
                     ExpressionType::BOUND_AGGREGATE);
            auto bound_agg =
                dynamic_cast<BoundAggregateExpression *>(aggregate.get());
            auto &agg_function_name = bound_agg->function.name;

            std::string agg_var_name = "agg_" + std::to_string(i);
            std::string dst = "aht" + op_id + "_" + agg_var_name + "[bucketId]";
            std::string agg_type = code_gen->ConvertLogicalTypeToPrimitiveType(
                bound_agg->return_type, true);
            if (agg_function_name == "count_star" ||
                agg_function_name == "count") {
                code.Add("atomicAdd(&" + dst + ", (" + agg_type + ")1);");
            }
            else {
                // get ref idxs
                std::vector<uint64_t> ref_idxs;
                D_ASSERT(bound_agg->children.size() == 1);
                code_gen->GetReferencedColumns(bound_agg->children[0].get(),
                                               ref_idxs);
                D_ASSERT(ref_idxs.size() == 1);
                std::string inattr = code_gen->GetValidVariableName(
                    pipeline_ctx.input_column_names[ref_idxs[0]], ref_idxs[0]);
                if (agg_function_name == "sum") {
                    code.Add("atomicAdd(&" + dst + ", " + inattr + ");");
                }
                else if (agg_function_name == "avg") {
                    code.Add("atomicAdd(&" + dst + ", " + inattr + ");");
                    if (!contain_count_func) {
                        // If there is no count function, we need to add a count
                        // variable for average calculation
                        std::string dst_cnt = "aht" + op_id + "_" +
                                              agg_var_name + "_cnt[bucketId]";
                        code.Add("atomicAdd(&" + dst_cnt + ", (" + agg_type +
                                 ")1);");
                    }
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
}

void HashAggregateCodeGenerator::GenerateGlobalDeclaration(
    CypherPhysicalOperator *op, CypherPhysicalOperator *prev_op, size_t op_idx,
    CodeBuilder &code, GpuCodeGenerator *code_gen, ClientContext &context,
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

    // Get the input schema from the previous operator
    D_ASSERT(prev_op != nullptr);
    auto &input_schema = prev_op->GetSchema();
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
    code.Add("");
    return;
}

void HashAggregateCodeGenerator::GenerateDeclarationInHostCode(
    CypherPhysicalOperator *op, size_t op_idx, CodeBuilder &code,
    GpuCodeGenerator *code_gen, PipelineContext &pipeline_ctx)
{
    // Hashaggregate appears twice in the pipelines, once for build side
    // and once for probe side. We need to generate declarations only once
    if (op_idx == 0)
        return;

    // declare agg hash table
    std::string ht_size;
    auto agg_op = dynamic_cast<PhysicalHashAggregate *>(op);
    if (agg_op->groups.size() == 0) {
        // If there are no groups, we can use a single bucket
        ht_size = "1";
    }
    else {
        ht_size = "10000";  // TODO: make this configurable or get from operator
    }
    uint64_t op_id = agg_op->GetOperatorId();
    std::string ht_name = "agg_ht<Payload" + std::to_string(op_id) + ">";
    code_gen->AddInitFunctionName("initAggHT<Payload" + std::to_string(op_id) +
                                  ">");
    code.Add("int ht_size = " + ht_size + ";");
    code.Add(ht_name + " *aht" + std::to_string(op_id) + ";");
    code.Add("cudaMalloc((void **)&aht" + std::to_string(op_id) + ", " +
             ht_size + " * sizeof(" + ht_name + "));");
    code.Add("{");
    code.IncreaseNesting();
    code.Add("void *args[] = { &aht" + std::to_string(op_id) + ", &ht_size };");
    code.Add(
        "cuLaunchKernel(initAggHT, gridSize, 1, 1, blockSize, 1, 1, 0, 0, "
        "args, nullptr);");
    code.DecreaseNesting();
    code.Add("} // initAggHT kernel launch");

    // declare aggregate variables
    bool contain_count_func = false;
    for (auto &aggregate : agg_op->aggregates) {
        D_ASSERT(aggregate->GetExpressionType() ==
                    ExpressionType::BOUND_AGGREGATE);
        auto bound_agg =
            dynamic_cast<BoundAggregateExpression *>(aggregate.get());
        if (bound_agg->function.name == "count_star" ||
            bound_agg->function.name == "count") {
            contain_count_func = true;
            break;
        }
    }

    code.Add("");
    code.Add("// Declare aggregate variables for hash aggregate");
    for (size_t i = 0; i < agg_op->aggregates.size(); i++) {
        auto &aggregate = agg_op->aggregates[i];
        D_ASSERT(aggregate->GetExpressionType() ==
                 ExpressionType::BOUND_AGGREGATE);
        auto bound_agg =
            dynamic_cast<BoundAggregateExpression *>(aggregate.get());
        auto &agg_function_name = bound_agg->function.name;
        std::string agg_var_name =
            "aht" + std::to_string(op_id) + "_agg_" + std::to_string(i);
        std::string agg_type =
            code_gen->ConvertLogicalTypeToPrimitiveType(bound_agg->return_type);
        std::string agg_type_short =
            code_gen->ConvertLogicalTypeToPrimitiveType(bound_agg->return_type,
                                                        false, true);
        std::string init_value = code_gen->GetInitValueForAggregate(bound_agg);
        // code_gen->AddInitFunctionName("initArray_" + agg_type_short);
        code_gen->AddInitFunctionName("initArray<" + agg_type + ">");
        code.Add(agg_type + "*" + agg_var_name + ";");
        code.Add("cudaMalloc((void **)&" + agg_var_name + ", " + ht_size +
                 " * sizeof(" + agg_type + "));");
        code.Add("{");
        code.IncreaseNesting();
        code.Add(agg_type + "init_value = " + init_value + ";");
        code.Add("void *args[] = { &" + agg_var_name +
                 ", &init_value, &ht_size };");
        // code.Add("cuLaunchKernel(initArray_" + agg_type_short +
        //          ", gridSize, 1, 1, blockSize, 1, 1, 0, 0, args, nullptr);");
        code.Add(
            "cuLaunchKernel(initArray, gridSize, 1, 1, blockSize, 1, 1, 0, 0, "
            "args, nullptr);");
        code.DecreaseNesting();
        code.Add("} // initArray kernel launch");
        if (agg_function_name == "avg" && !contain_count_func) {
            // If there is no count function, we need to add a count variable
            // for average calculation
            std::string count_var_name = agg_var_name + "_cnt";
            std::string count_init_value = "0";
            // code_gen->AddInitFunctionName("initArray_uint64");
            code_gen->AddInitFunctionName("initArray<unsigned long long>");
            code.Add("unsigned long long *" + count_var_name + ";");
            code.Add("cudaMalloc((void **)&" + count_var_name + ", " + ht_size +
                     " * sizeof(unsigned long long));");
            code.Add("{");
            code.IncreaseNesting();
            code.Add("unsigned long long init_value = " + count_init_value +
                     ";");
            code.Add("void *args[] = { &" + count_var_name +
                     ", &init_value, &ht_size };");
            code.Add(
                "cuLaunchKernel(initArray, gridSize, 1, 1, blockSize, "
                "1, 1, 0, 0, args, nullptr);");
            // code.Add(
            //     "cuLaunchKernel(initArray_uint64, gridSize, 1, 1, blockSize, "
            //     "1, 1, 0, 0, args, nullptr);");
            code.DecreaseNesting();
            code.Add("} // initArray kernel launch");
        }
    }

    return;
}

void HashAggregateCodeGenerator::GenerateInputKernelParameters(
    CypherPhysicalOperator *op, GpuCodeGenerator *code_gen,
    ClientContext &context, PipelineContext &pipeline_context,
    std::vector<KernelParam> &input_kernel_params,
    std::vector<ScanColumnInfo> &scan_column_infos)
{
    auto agg_op = dynamic_cast<PhysicalHashAggregate *>(op);
    auto op_id = agg_op->GetOperatorId();

    // Add input parameters for the hash aggregate
    KernelParam input_param;
    input_param.name = "aht" + std::to_string(op_id);
    input_param.type = "agg_ht<Payload" + std::to_string(op_id) + "> *";
    input_param.is_device_ptr = true;
    input_kernel_params.push_back(input_param);

    bool contain_count_func = false;
    for (auto &aggregate : agg_op->aggregates) {
        D_ASSERT(aggregate->GetExpressionType() ==
                 ExpressionType::BOUND_AGGREGATE);
        auto bound_agg =
            dynamic_cast<BoundAggregateExpression *>(aggregate.get());
        if (bound_agg->function.name == "count_star" ||
            bound_agg->function.name == "count") {
            contain_count_func = true;
            break;
        }
    }

    for (size_t i = 0; i < agg_op->aggregates.size(); i++) {
        auto &aggregate = agg_op->aggregates[i];
        D_ASSERT(aggregate->GetExpressionType() ==
                 ExpressionType::BOUND_AGGREGATE);
        auto bound_agg =
            dynamic_cast<BoundAggregateExpression *>(aggregate.get());
        std::string func_name = bound_agg->function.name;
        std::string ctype =
            code_gen->ConvertLogicalTypeToPrimitiveType(bound_agg->return_type);
        KernelParam agg_param;
        agg_param.name =
            "aht" + std::to_string(op_id) + "_" + "agg_" + std::to_string(i);
        agg_param.type = ctype + "*";
        agg_param.is_device_ptr = true;
        input_kernel_params.push_back(agg_param);
        if (func_name == "avg" && !contain_count_func) {
            // If there is no count function, we need to add a count variable
            // for average calculation
            KernelParam count_param;
            count_param.name = "aht" + std::to_string(op_id) + "_" + "agg_" +
                               std::to_string(i) + "_cnt";
            count_param.type = "unsigned long long *";
            count_param.is_device_ptr = true;
            input_kernel_params.push_back(count_param);
        }
    }

    if (agg_op->groups.size() == 0) {
        pipeline_context.per_pipeline_num_input_tuples.push_back(1);
    }
    else {
        pipeline_context.per_pipeline_num_input_tuples.push_back(
            10000);  // TODO: make this configurable
    }
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

    bool contain_count_func = false;
    for (auto &aggregate : agg_op->aggregates) {
        D_ASSERT(aggregate->GetExpressionType() ==
                 ExpressionType::BOUND_AGGREGATE);
        auto bound_agg =
            dynamic_cast<BoundAggregateExpression *>(aggregate.get());
        if (bound_agg->function.name == "count_star" ||
            bound_agg->function.name == "count") {
            contain_count_func = true;
            break;
        }
    }

    for (size_t i = 0; i < agg_op->aggregates.size(); i++) {
        auto &aggregate = agg_op->aggregates[i];
        D_ASSERT(aggregate->GetExpressionType() ==
                 ExpressionType::BOUND_AGGREGATE);
        auto bound_agg =
            dynamic_cast<BoundAggregateExpression *>(aggregate.get());
        std::string func_name = bound_agg->function.name;
        std::string ctype =
            code_gen->ConvertLogicalTypeToPrimitiveType(bound_agg->return_type);

        KernelParam agg_param;
        agg_param.name =
            "aht" + std::to_string(op_id) + "_" + "agg_" + std::to_string(i);
        agg_param.type = ctype + "*";
        agg_param.is_device_ptr = true;
        output_kernel_params.push_back(agg_param);

        if (func_name == "avg" && !contain_count_func) {
            // If there is no count function, we need to add a count variable
            // for average calculation
            KernelParam count_param;
            count_param.name = "aht" + std::to_string(op_id) + "_" + "agg_" +
                               std::to_string(i) + "_cnt";
            count_param.type = "unsigned long long *";
            count_param.is_device_ptr = true;
            output_kernel_params.push_back(count_param);
        }
    }
}

void HashAggregateCodeGenerator::AnalyzeOperatorForMaterialization(
    CypherPhysicalOperator *op, int sub_idx, int op_idx,
    PipelineContext &pipeline_context, GpuCodeGenerator *code_gen)
{
    auto &output_schema = op->GetSchema();
    auto &output_column_names = output_schema.getStoredColumnNamesRef();
    auto &output_column_types = output_schema.getStoredTypesRef();
    auto *agg_op = dynamic_cast<PhysicalHashAggregate *>(op);
    D_ASSERT(agg_op->children.size() == 1);
    auto &input_schema = agg_op->children[0]->GetSchema();
    auto &input_column_names = input_schema.getStoredColumnNamesRef();
    auto &input_column_types = input_schema.getStoredTypesRef();
    auto &columns_to_be_materialized =
        pipeline_context.columns_to_be_materialized[sub_idx];
    auto &mat_target_columns =
        pipeline_context.materialization_target_columns[sub_idx];
    auto &sub_pipeline_tids =
        pipeline_context.sub_pipeline_tids[sub_idx];
    std::string tid =
        "tid_" + std::to_string(sub_idx) + "_" + std::to_string(op_idx);

    // Create tid mapping info
    if (op_idx == 0) { // Source side of hash aggregate
        for (const auto &col_name : output_column_names) {
            D_ASSERT(pipeline_context.attribute_tid_mapping.find(col_name) ==
                     pipeline_context.attribute_tid_mapping.end());
            pipeline_context.attribute_tid_mapping[col_name] = tid;
        }
        sub_pipeline_tids.push_back(tid);
    }

    // Create source mapping info
    std::vector<uint64_t> group_key_idxs;
    for (auto &group : agg_op->groups) {
        code_gen->GetReferencedColumns(group.get(), group_key_idxs);
    }
    for (auto &group_key_idx : group_key_idxs) {
        std::string &orig_col_name = input_column_names[group_key_idx];
        std::string col_name = orig_col_name;
        std::replace(col_name.begin(), col_name.end(), '.', '_');
        if (op_idx == 0) {
            pipeline_context.attribute_source_mapping[orig_col_name] =
                "aht" + std::to_string(agg_op->GetOperatorId()) + "[" + tid +
                "].payload." + col_name;
        }
        else {
            if (mat_target_columns.find(orig_col_name) ==
                mat_target_columns.end()) {
                mat_target_columns.insert(orig_col_name);
                columns_to_be_materialized.push_back(
                    Attr{orig_col_name, input_column_types[group_key_idx],
                         group_key_idx});
            }
        }
    }

    bool contain_count_func = false;
    std::string count_var_name = "";
    for (size_t i = 0; i < agg_op->aggregates.size(); i++) {
        auto &aggregate = agg_op->aggregates[i];
        D_ASSERT(aggregate->GetExpressionType() ==
                 ExpressionType::BOUND_AGGREGATE);
        auto bound_agg =
            dynamic_cast<BoundAggregateExpression *>(aggregate.get());
        if (bound_agg->function.name == "count_star" ||
            bound_agg->function.name == "count") {
            contain_count_func = true;
            count_var_name = "aht" + std::to_string(agg_op->GetOperatorId()) +
                             "_agg_" + std::to_string(i) + "[" + tid + "]";
            break;
        }
    }

    for (size_t i = 0; i < agg_op->aggregates.size(); i++) {
        auto &aggregate = agg_op->aggregates[i];
        D_ASSERT(aggregate->GetExpressionType() ==
                 ExpressionType::BOUND_AGGREGATE);
        auto bound_agg =
            dynamic_cast<BoundAggregateExpression *>(aggregate.get());
        std::string agg_var_name = "agg_" + std::to_string(i);
        std::string col_name = output_column_names[group_key_idxs.size() + i];
        if (op_idx == 0) {
            if (bound_agg->function.name == "avg") {
                if (contain_count_func) {
                    pipeline_context.attribute_source_mapping[col_name] =
                        "aht" + std::to_string(agg_op->GetOperatorId()) + "_" +
                        agg_var_name + "[" + tid + "]/" + count_var_name;
                }
                else {
                    pipeline_context.attribute_source_mapping[col_name] =
                        "aht" + std::to_string(agg_op->GetOperatorId()) + "_" +
                        agg_var_name + "[" + tid + "]/aht" +
                        std::to_string(agg_op->GetOperatorId()) + "_" +
                        agg_var_name + "_cnt[" + tid + "]";
                }
            }
            else {
                pipeline_context.attribute_source_mapping[col_name] =
                    "aht" + std::to_string(agg_op->GetOperatorId()) + "_" +
                    agg_var_name + "[" + tid + "]";
            }
        }
        else {
            std::vector<uint64_t> referenced_columns;
            code_gen->GetReferencedColumns(aggregate.get(), referenced_columns);
            for (const auto &col_idx : referenced_columns) {
                auto &input_col_name = input_column_names[col_idx];
                if (mat_target_columns.find(input_col_name) ==
                    mat_target_columns.end()) {
                    mat_target_columns.insert(input_col_name);
                    columns_to_be_materialized.push_back(Attr{
                        input_col_name, input_column_types[col_idx], col_idx});
                }
            }
        }
    }
}

void AdjIdxJoinCodeGenerator::GenerateCode(
    CypherPhysicalOperator *op, CodeBuilder &code, GpuCodeGenerator *code_gen,
    ClientContext &context, PipelineContext &pipeline_ctx, bool is_main_loop)
{
    auto adj_op = dynamic_cast<PhysicalAdjIdxJoin *>(op);
    D_ASSERT(adj_op != nullptr);
    std::string op_id = std::to_string(adj_op->GetOperatorId());
    std::string rel_name = ""; // TODO
    std::string id_name; // TODO
    std::string tid; // TODO

    code.Add("// AdjIdxJoin operator");
    code.Add("active = indexProbeMulti(" + rel_name + "_offset, " + id_name +
             ", local" + op_id + "_range.start, local" + op_id + "_range.end);")
    
    if (adj_op->IsTargetUnique()) {
        std::string var = "local" + op_id + "_range.start";
        code.Add("indexGetPid(" + rel_name + "_position, " + var + ");");
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
        PipeInputType input_type = GetPipeInputType(sub_pipeline, i);
        code.Add("case " + spid + ": {");
        code.IncreaseNesting();
        if (input_type == PipeInputType::TYPE_0_FALSE) {
            code.Add(
                "Themis::PushedParts::PushedPartsAtZeroLvl *src_pparts = "
                "(Themis::PushedParts::PushedPartsAtZeroLvl*) stack->Top();");
            code.Add(
                "Themis::PullINodesFromPPartAtZeroLvl(thread_id, src_pparts, "
                "ts_0_range_cached, inodes_cnts);");
            code.Add("if (thread_id == 0) stack->PopPartsAtZeroLvl();");
        }
        else if (input_type == PipeInputType::TYPE_1_FALSE) {
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
        else if (input_type == PipeInputType::TYPE_2_TRUE) {
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
            PipeInputType input_type = GetPipeInputType(sub_pipeline, i);
            code.Add("case " + spid + ": {");
            code.IncreaseNesting();
            if (input_type == PipeInputType::TYPE_1_TRUE ||
                input_type == PipeInputType::TYPE_1_FALSE) {
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
        PipeInputType input_type = GetPipeInputType(sub_pipeline, i);
        code.Add("case " + spid + ": {");
        code.IncreaseNesting();
        if (input_type == PipeInputType::TYPE_0_FALSE ||
            input_type == PipeInputType::TYPE_0_TRUE) {
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
        else if (input_type == PipeInputType::TYPE_1_FALSE ||
                 input_type == PipeInputType::TYPE_1_TRUE) {
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
        else if (input_type == PipeInputType::TYPE_2_FALSE ||
                 input_type == PipeInputType::TYPE_2_TRUE) {
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