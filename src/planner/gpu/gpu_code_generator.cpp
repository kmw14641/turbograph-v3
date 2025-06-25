#include "common/file_system.hpp"
#include "planner/gpu/gpu_code_generator.hpp"
#include <cuda.h>
#include <nvrtc.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include "execution/physical_operator/cypher_physical_operator.hpp"
#include "llvm/Support/TargetSelect.h"
#include <sstream>

namespace duckdb {

GpuCodeGenerator::GpuCodeGenerator(ClientContext &context)
    : context(context),
      is_compiled(false),
      is_repeatable(false)
{
    InitializeLLVMTargets();
    jit_compiler = std::make_unique<GpuJitCompiler>();
}

GpuCodeGenerator::~GpuCodeGenerator()
{
    Cleanup();
}

void GpuCodeGenerator::InitializeLLVMTargets()
{
    static bool is_llvm_targets_initialized = false;
    if (is_llvm_targets_initialized) return;
    is_llvm_targets_initialized = true;

    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
}

void GpuCodeGenerator::GenerateGPUCode(CypherPipeline &pipeline) {
    // generate kernel code
    GenerateKernelCode(pipeline);

    // then, generate host code
    GenerateHostCode();

    // for debug
    std::cout << generated_gpu_code << std::endl;
    std::cout << generated_cpu_code << std::endl;
}


void GpuCodeGenerator::GenerateKernelCode(CypherPipeline &pipeline) {
    std::stringstream code;

    // // Add CUDA includes and definitions
    // code << "#include <cuda_runtime.h>\n";
    // code << "#include <device_launch_parameters.h>\n";
    // code << "#include <cstdio>\n";

    // code << "\n\n";
    
    // Generate kernel function
    code << "extern \"C\" __global__ void gpu_kernel(";
    
    // Add kernel parameters
    kernel_params = GenerateKernelParams(pipeline);  // Store parameters for later use
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
        
        // Handle Scan operator
        if (PhysicalOperatorType::NODE_SCAN == op->GetOperatorType()) {
            // Get scan parameters
            
            // auto column_count = op->GetColumnCount();
            
            // Generate scan code
            // code << "    // Scan operator\n";
            // code << "    for (int i = tid; i < " << table_name << "_count; i += stride) {\n";
            
            // Process each column
            // for (int col = 0; col < column_count; col++) {
            //     code << "        " << table_name << "_data[" << col << "][i] = " 
            //          << table_name << "_buffer[" << col << "][i];\n";
            // }
            
            // code << "    }\n";
        }
    }
    
    code << "}\n";
    
    generated_gpu_code = code.str();
}

void GpuCodeGenerator::GenerateHostCode() {
    std::stringstream code;

    code << "#include <cuda.h>\n";
    code << "#include <cuda_runtime.h>\n";
    code << "#include <vector>\n";
    code << "#include <iostream>\n\n";

    code << "extern \"C\" CUfunction gpu_kernel;\n\n";

    // code << "CUcontext g_ctx = nullptr;\n";
    // code << "void ensureCudaContextInit() {\n";
    // code << "    static bool done = false;\n";
    // code << "    if (done) return;\n";

    // code << "    cuInit(0);\n";
    // code << "    CUdevice dev;  cuDeviceGet(&dev, 0);\n";
    // code << "    cuCtxCreate(&g_ctx, 0, dev);\n";
    // code << "    cuCtxSetCurrent(g_ctx);\n";
    // code << "    done = true;\n";
    // code << "}\n";
    
    code << "extern \"C\" void execute_query(size_t N) {\n";
    // code << "    ensureCudaContextInit();\n";
    // code << "    cuCtxSetCurrent(g_ctx);\n";
    code << "    cudaError_t err;\n";

    for (auto &p : kernel_params) {
        if (p.type.find('*') != std::string::npos) {
            std::string elem = p.type.substr(0, p.type.find('*'));
            std::string h = p.name + "_h";
            std::string d = p.name + "_d";

            code << "    std::vector<" << elem << "> " << h << "(N, 0);\n";
            code << "    " << elem << "* " << d << " = nullptr;\n";
            code << "    err = cudaMalloc(&" << d << ", N * sizeof(" << elem << "));\n";
            code << "    if (err) throw std::runtime_error(\"cudaMalloc failed\");\n";
            code << "    err = cudaMemcpy(" << d << ", " << h << ".data(), N * sizeof(" << elem << "), cudaMemcpyHostToDevice);\n";
            code << "    if (err) throw std::runtime_error(\"H2D memcpy failed\");\n\n";
        }
    }

    // code << "    int* test_device_var = nullptr;\n";
    // code << "    int* test_host_var = nullptr;\n";
    // code << "    test_host_var = (int*)malloc(N * sizeof(int));\n";
    // code << "    err = cudaMalloc(&test_device_var, N * sizeof(int));\n";
    // code << "    if (err) throw std::runtime_error(\"cudaMalloc failed\");\n";
    // code << "    err = cudaMemcpy(test_device_var, test_host_var, N * sizeof(int), cudaMemcpyHostToDevice);\n";
    // code << "    if (err) throw std::runtime_error(\"H2D memcpy failed\");\n";

    code << "    const int blockSize = 256;\n";
    code << "    const int gridSize  = static_cast<int>((N + blockSize - 1) / blockSize);\n";
    code << "    void* args[] = {";
    for (size_t i = 0; i < kernel_params.size(); ++i) {
        const auto &p = kernel_params[i];
        code << (p.type.find('*') != std::string::npos ? "&" + p.name + "_d" : "&" + p.name);
        if (i + 1 < kernel_params.size()) code << ", ";
    }
    code << "};\n";

    code << "    std::cerr << \"Hello from host code\" << std::endl;\n";
    code << "    std::cerr << \"gpu_kernel ptr = \" << (void*)gpu_kernel << std::endl;\n";
    code << "    std::cerr << \"grid=\" << gridSize << \"  block=\" << blockSize << std::endl;\n";

    code << "    CUresult r = cuLaunchKernel(gpu_kernel, gridSize,1,1, blockSize,1,1, 0, 0, nullptr, nullptr);\n";
    // code << "    if (r != CUDA_SUCCESS) throw std::runtime_error(\"cuLaunchKernel failed\");\n";
    code << "    if (r != CUDA_SUCCESS) {\n";
    code << "        const char *name = nullptr, *str = nullptr;\n";
    code << "        cuGetErrorName(r, &name);\n";
    code << "        cuGetErrorString(r, &str);\n";
    code << "        std::cerr << \"cuLaunchKernel failed: \" << (name?name:\"\") << \" – \" << (str?str:\"\") << std::endl;\n";
    code << "        throw std::runtime_error(\"cuLaunchKernel failed\");\n";
    code << "    }\n";
    code << "    cudaError_t errSync = cudaDeviceSynchronize();\n";
    code << "    if (errSync != cudaSuccess) {\n";
    code << "        std::cerr << \"sync error: \" << cudaGetErrorString(errSync) << std::endl;\n";
    code << "        throw std::runtime_error(\"cudaDeviceSynchronize failed\");\n";
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

bool GpuCodeGenerator::CompileGeneratedCode() {
    if (generated_gpu_code.empty() || generated_cpu_code.empty()) return false;

    if (!jit_compiler->CompileWithNVRTC(generated_gpu_code, "gpu_kernel",
                                        gpu_module, kernel_function))
        return false;
    
    if (!jit_compiler->CompileWithORCLLJIT(generated_cpu_code, kernel_function)) {
        return false;
    }

    is_compiled = true;
    return true;
}

// bool GpuCodeGenerator::CompileGeneratedCode() {
//     if (generated_gpu_code.empty() || generated_cpu_code.empty()) {
//         return false;
//     }

//     // Use JIT compiler to compile and load the code
//     if (!jit_compiler->CompileAndLoad(generated_gpu_code)) {
//         return false;
//     }

//     is_compiled = true;
//     return true;
// }

void* GpuCodeGenerator::GetCompiledHost() {
    if (!is_compiled) {
        return nullptr;
    }
    return jit_compiler->GetMainFunction();
}

void GpuCodeGenerator::Cleanup() {
    if (!is_repeatable && is_compiled) {
        // If not repeatable and compiled, release the kernel
        // jit_compiler->ReleaseKernel(current_code_hash);
        is_compiled = false;
    }
}

std::vector<KernelParam> GpuCodeGenerator::GenerateKernelParams(const CypherPipeline& pipeline) {
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

} // namespace duckdb 