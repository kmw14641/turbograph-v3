#ifndef GPU_JIT_COMPILER_H
#define GPU_JIT_COMPILER_H

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"

namespace duckdb {

class GpuJitCompiler {
   public:
    GpuJitCompiler();
    ~GpuJitCompiler();

    // Compile generated CUDA code using NVRTC
    bool CompileWithNVRTC(const std::string &src, const char *kernel_name,
                          int num_pipelines_compiled, CUmodule &mod_out,
                          std::vector<CUfunction> &kernels_out);

    bool CompileWithORCLLJIT(const std::string &host_code,
                             std::vector<CUfunction> &kernels);

    // Compile and load CUDA code
    bool CompileAndLoad(const std::string &cuda_code);

    // Release a specific kernel
    void ReleaseKernel(const std::string &code_hash);

    // Cleanup all resources
    void Cleanup();

    // Get the compiled kernel function
    void *GetMainFunction() const { return main_function; }

    bool ensureCudaContext();

   private:
    // CUDA compiler configuration
    struct CudaCompilerConfig {
        std::string cuda_include_path;
        std::string cuda_lib_path;
        std::vector<std::string> cuda_compile_flags;
    };

    // Calculate hash for code caching
    std::string CalculateHash(const std::string &code);

    // Initialize Clang compiler
    bool InitializeCompiler();

    // Compile CUDA code to LLVM IR
    std::vector<std::unique_ptr<llvm::Module>> CompileToIR(
        const std::string &cuda_code);

    // Add CUDA runtime symbols to JIT
    bool AddCudaRuntimeSymbols();

    // Compiler configuration
    CudaCompilerConfig cuda_config;
    std::string project_include_path;  // Project include path for JIT compilation
    std::unique_ptr<clang::CompilerInstance> compiler;
    std::unique_ptr<llvm::orc::LLJIT> jit;
    llvm::orc::ThreadSafeContext ts_ctx;
    std::unordered_map<std::string, std::pair<void *, int>>
        compiled_cache;  // <hash, <function_ptr, ref_count>>
    void *main_function;

    std::string diag_buffer;
    std::unique_ptr<llvm::raw_string_ostream> diag_stream;
    bool cuda_context_initialized = false;
};

}  // namespace duckdb

#endif  // GPU_JIT_COMPILER_H