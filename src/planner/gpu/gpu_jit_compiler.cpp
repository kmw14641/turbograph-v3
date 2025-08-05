#include "planner/gpu/gpu_jit_compiler.hpp"
#include <cuda_runtime_api.h>
#include <functional>
#include <stdexcept>
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Tool.h"
#include "clang/Format/Format.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/VirtualFileSystem.h"

namespace duckdb {

extern "C" {
cudaError_t cudaConfigureCall(...);
void *__cudaRegisterFatBinary(...);
void __cudaUnregisterFatBinary(...);
void __cudaRegisterFunction(...);
void __cudaPopCallConfiguration(...);
void __cudaPushCallConfiguration(...);
}

GpuJitCompiler::GpuJitCompiler()
    : main_function(nullptr), ts_ctx(std::make_unique<llvm::LLVMContext>())
{
    // Initialize CUDA configuration // TODO get path, flags from config
    cuda_config.cuda_include_path = "/usr/local/cuda/include";
    cuda_config.cuda_lib_path = "/usr/local/cuda/lib64";
    cuda_config.cuda_compile_flags = {
        "-x", "cuda",
        "--cuda-gpu-arch=sm_75",  // Set GPU architecture
        "-O3"                     // Optimization level
    };

    // Calculate project include path once
#ifdef S62GDB_ROOT
    project_include_path = std::string(S62GDB_ROOT) + "/src/include";
#else
    const char* env_root = std::getenv("S62GDB_ROOT");
    const char* root_path = env_root ? env_root : "/turbograph-v3";
    project_include_path = std::string(root_path) + "/src/include";
#endif

    // Initialize JIT
    auto jb = llvm::orc::LLJITBuilder{};
    auto res = jb.create();
    if (!res) {
        llvm::handleAllErrors(res.takeError(), [](llvm::ErrorInfoBase &E) {
            E.log(llvm::errs());
        });
        throw std::runtime_error("failed to create LLJIT");
    }

    jit = std::move(*res);

    // Add CUDA runtime symbols
    if (!AddCudaRuntimeSymbols())
        throw std::runtime_error("failed to register CUDA runtime symbols");

    setenv("CLANG_PRINT_STATISTICS", "1", /*overwrite*/ 0);
    setenv("LLVM_ENABLE_STATS", "1", 0);
}

GpuJitCompiler::~GpuJitCompiler()
{
    Cleanup();
}

bool GpuJitCompiler::InitializeCompiler()
{
    compiler = std::make_unique<clang::CompilerInstance>();

    // Create diagnostics
    llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diag_opts(
        new clang::DiagnosticOptions());
    diag_opts->ShowColors = true;
    diag_opts->ShowCarets = true;
    diag_opts->ShowOptionNames = true;
    diag_opts->ShowFixits = true;
    diag_opts->ShowSourceRanges = true;

    auto diag_printer = std::make_unique<clang::TextDiagnosticPrinter>(
        llvm::errs(), &*diag_opts);

    compiler->createDiagnostics(diag_printer.release(),
                                /*ShouldOwnClient=*/true);

    // Set target options
    auto &target_opts = compiler->getTargetOpts();
    target_opts.Triple = llvm::sys::getProcessTriple();

    // Set CUDA options
    auto &header_opts = compiler->getHeaderSearchOpts();
    header_opts.AddPath(cuda_config.cuda_include_path, clang::frontend::Angled,
                        false, false);

    // Set compiler flags
    auto &lang_opts = compiler->getLangOpts();
    lang_opts.CUDA = true;
    // lang_opts.CUDAIsDevice = true;

    return true;
}

std::vector<std::unique_ptr<llvm::Module>> GpuJitCompiler::CompileToIR(
    const std::string &cuda_code)
{
    InitializeCompiler();

    auto realFS = llvm::vfs::getRealFileSystem();
    auto memFS = llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem>(
        new llvm::vfs::InMemoryFileSystem);

    auto overlayFS = llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem>(
        new llvm::vfs::OverlayFileSystem(realFS));
    overlayFS->pushOverlay(memFS);

    memFS->addFile(
        "gpu_kernel.cu",
        /*ModTime=*/0,
        llvm::MemoryBuffer::getMemBuffer(cuda_code, "gpu_kernel.cu"));

    clang::driver::Driver drv(
        "/usr/bin/clang++-17", llvm::sys::getDefaultTargetTriple(),
        compiler->getDiagnostics(), "clang LLVM compiler", memFS);

    std::vector<const char*> drv_argv = {
        "clang++",
        "-###", "-v",
        "--cuda-path=/usr/local/cuda-11.8",
        "-x", "cuda", "gpu_kernel.cu",
        "--cuda-gpu-arch=sm_75",
        "--cuda-host-only",
        "-O3",
        "-std=c++17",
        "-isystem", "/usr/include/c++/11",
        "-isystem", "/usr/include/x86_64-linux-gnu/c++/11",
        "-isystem", "/usr/local/cuda/include",
        "-isystem", "/usr/include",
        "-isystem", "/usr/include/x86_64-linux-gnu",
        "-isystem", "/turbograph-v3/src/include/planner/gpu/themis/",
        "-I", project_include_path.c_str(),  // Use dynamic include path
        "-emit-llvm", "-c"
    };

    std::unique_ptr<clang::driver::Compilation> comp(
        drv.BuildCompilation(drv_argv));
    if (!comp || comp->getJobs().empty()) {
        llvm::errs() << "[JIT] Driver fail to create job\n";
        throw std::runtime_error("");
    }

    // debug
    llvm::errs() << "=== driver jobs ===\n";
    comp->getJobs().Print(llvm::errs(), "\n", /*PrintInputFilenames=*/false);
    llvm::errs() << "===================\n";

    std::vector<std::unique_ptr<llvm::Module>> modules;

    for (auto &J : comp->getJobs()) {
        auto *cmd = llvm::dyn_cast<clang::driver::Command>(&J);
        if (!cmd)
            continue;

        llvm::errs() << cmd->getExecutable();
        for (auto A : cmd->getArguments())
            llvm::errs() << ' ' << A;
        llvm::errs() << '\n';

        const bool is_device = llvm::any_of(
            cmd->getArguments(),
            [](llvm::StringRef A) { return A == "-fcuda-is-device"; });
        
        if (is_device) continue;

        // ── 개별 CompilerInstance 생성 ────────────────────────────────
        auto CI = std::make_unique<clang::CompilerInstance>();

        auto &diag_opts = *new clang::DiagnosticOptions();
        auto diag_printer = std::make_unique<clang::TextDiagnosticPrinter>(
            llvm::errs(), &diag_opts);
        CI->createDiagnostics(diag_printer.release(), /*ShouldOwnClient=*/true);

        clang::CompilerInvocation::CreateFromArgs(
            CI->getInvocation(), cmd->getArguments(), CI->getDiagnostics());

        // 소스 파일 remap
        CI->getPreprocessorOpts().addRemappedFile(
            "gpu_kernel.cu",
            llvm::MemoryBuffer::getMemBuffer(cuda_code, "gpu_kernel.cu")
                .release());

        // ── IR 생성 ───────────────────────────────────────────────────
        clang::EmitLLVMOnlyAction act(ts_ctx.getContext());
        bool ok = false;
        llvm::CrashRecoveryContext CRC;
        CRC.RunSafelyOnThread([&] { ok = CI->ExecuteAction(act); });

        if (!ok || CI->getDiagnostics().hasErrorOccurred()) {
            llvm::errs() << "[JIT] " << (is_device ? "device" : "host")
                         << " compilation failed — abort\n";
            return {};
        }

        if (auto M = act.takeModule())
            modules.push_back(std::move(M));
    }

    return modules;
}

bool GpuJitCompiler::AddCudaRuntimeSymbols()
{
    auto to_void = [](auto fn) -> void * {
        return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(fn));
    };

    using FnMalloc = cudaError_t (*)(void **, size_t);
    using FnFree = cudaError_t (*)(void *);
    using FnSync = cudaError_t (*)();
    using FnErrStr = const char *(*)(cudaError_t);
    // using FnCfg = cudaError_t (*)(...);
    using FnMemcpy =
        cudaError_t (*)(void *, const void *, size_t, cudaMemcpyKind);
    using FnMemset = cudaError_t (*)(void *, int, size_t);

    struct Pair {
        const char *name;
        void *addr;
    } tbl[] = {{"cudaMalloc", to_void(static_cast<FnMalloc>(&cudaMalloc))},
               {"cudaFree", to_void(static_cast<FnFree>(&cudaFree))},
               {"cudaDeviceSynchronize",
                to_void(static_cast<FnSync>(&cudaDeviceSynchronize))},
               {"cudaGetErrorString",
                to_void(static_cast<FnErrStr>(&cudaGetErrorString))},
               {"cudaMemcpy", to_void(static_cast<FnMemcpy>(&cudaMemcpy))},
               {"cudaMemset", to_void(static_cast<FnMemset>(&cudaMemset))}};

    llvm::orc::SymbolMap smap;
    for (auto &p : tbl)
        smap[jit->mangleAndIntern(p.name)] = llvm::orc::ExecutorSymbolDef(
            llvm::orc::ExecutorAddr::fromPtr(p.addr),
            llvm::JITSymbolFlags::Exported);

    if (auto err =
            jit->getMainJITDylib().define(llvm::orc::absoluteSymbols(smap))) {
        llvm::errs() << err << '\n';
        return false;
    }
    return true;
}

// Helper: compile src → PTX → load → return (mod, fn)
bool GpuJitCompiler::CompileWithNVRTC(const std::string &src,
                      const char *kernel_name,
                      CUmodule &mod_out,
                      CUfunction &fn_out) {
    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, src.c_str(), "jit_kernel.cu", 0, nullptr, nullptr) != NVRTC_SUCCESS)
        return false;

    // TODO: get themis include path from config or something
    // const char *opts[] = {
    //     "--gpu-architecture=compute_75",
    //     "--std=c++17",
    //     "--include-path=/usr/local/cuda/targets/x86_64-linux/include",
    //     "--include-path=/turbograph-v3/src/include/planner/gpu/themis/",
    //     "--include-path=/usr/include/",
    //     "--include-path=/usr/include/x86_64-linux-gnu/",
    //     "-D__x86_64__=1",
    //     "-D__LP64__=1"};
    const char *opts[] = {
        "--gpu-architecture=compute_75",
        "--std=c++17",
        "--include-path=/usr/include/",
        "--include-path=/usr/include/x86_64-linux-gnu/",
        "--include-path=/turbograph-v3/src/include/planner/gpu/themis/"};
    nvrtcResult r = nvrtcCompileProgram(prog, 5, opts);
    if (r != NVRTC_SUCCESS) {
        size_t sz; nvrtcGetProgramLogSize(prog, &sz);
        std::string log(sz, '\0'); nvrtcGetProgramLog(prog, log.data());
        std::cerr << log << std::endl;
        nvrtcDestroyProgram(&prog);
        return false;
    }
    size_t ptxSize; nvrtcGetPTXSize(prog, &ptxSize);
    std::string ptx(ptxSize, '\0'); nvrtcGetPTX(prog, ptx.data());
    nvrtcDestroyProgram(&prog);

    if (!ensureCudaContext()) {
        std::cerr << "CUDA Driver/context init failed\n";
        return false;
    }

    CUresult loadResult = cuModuleLoadDataEx(&mod_out, ptx.data(), 0, nullptr, nullptr);
    if (loadResult != CUDA_SUCCESS) {
        const char *errName = nullptr, *errStr = nullptr;
        cuGetErrorName(loadResult, &errName);
        cuGetErrorString(loadResult, &errStr);
        std::cerr << "cuModuleLoadDataEx failed: ["
                << (errName ? errName : "unknown") << "] "
                << (errStr  ? errStr  : "") << '\n';

        int drvVer = 0; cuDriverGetVersion(&drvVer);
        std::cerr << "Driver version  : " << drvVer/1000 << '.' << (drvVer%1000)/10 << '\n';

        std::cerr << "PTX snippet:\n" 
                << std::string(ptx.begin(), ptx.begin() + std::min<size_t>(200, ptx.size()))
                << "\n----\n";

        return false;
    }

    if (cuModuleGetFunction(&fn_out, mod_out, kernel_name) != CUDA_SUCCESS) {
        std::cerr << "Failed to get function" << std::endl;
        return false;
    }
    return true;
}

bool GpuJitCompiler::CompileWithORCLLJIT(const std::string &host_code, CUfunction &kernel_function) {
    InitializeCompiler();

    auto realFS = llvm::vfs::getRealFileSystem();
    auto memFS = llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem>(
        new llvm::vfs::InMemoryFileSystem);

    auto overlayFS = llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem>(
        new llvm::vfs::OverlayFileSystem(realFS));
    overlayFS->pushOverlay(memFS);

    memFS->addFile(
        "gpu_host.cu",
        /*ModTime=*/0,
        llvm::MemoryBuffer::getMemBuffer(host_code, "gpu_host.cu"));

    clang::driver::Driver drv(
        "/usr/bin/clang++-17", llvm::sys::getDefaultTargetTriple(),
        compiler->getDiagnostics(), "clang LLVM compiler", memFS);

    std::vector<const char*> drv_argv = {
        "clang++",
        // "-###", "-v",
        "--cuda-path=/usr/local/cuda-11.8",
        "-x", "cuda", "gpu_host.cu",
        "--cuda-gpu-arch=sm_75",
        "--cuda-host-only",
        "-O3",
        "-std=c++17",
        "-isystem", "/usr/include/c++/11",
        "-isystem", "/usr/include/x86_64-linux-gnu/c++/11",
        "-isystem", "/usr/local/cuda/include",
        "-isystem", "/usr/include",
        "-isystem", "/usr/include/x86_64-linux-gnu",
        "-isystem", "/turbograph-v3/src/include/planner/gpu/themis/",
        "-I", project_include_path.c_str(),  // Use dynamic include path
        "-emit-llvm", "-c"
    };

    std::unique_ptr<clang::driver::Compilation> comp(
        drv.BuildCompilation(drv_argv));
    if (!comp || comp->getJobs().empty()) {
        llvm::errs() << "[JIT] Driver fail to create job\n";
        throw std::runtime_error("");
    }

    // // debug
    // llvm::errs() << "=== driver jobs ===\n";
    // comp->getJobs().Print(llvm::errs(), "\n", /*PrintInputFilenames=*/false);
    // llvm::errs() << "===================\n";

    std::vector<std::unique_ptr<llvm::Module>> modules;

    for (auto &J : comp->getJobs()) {
        auto *cmd = llvm::dyn_cast<clang::driver::Command>(&J);
        if (!cmd) continue;

        // llvm::errs() << cmd->getExecutable();
        // for (auto A : cmd->getArguments())
        //     llvm::errs() << ' ' << A;
        // llvm::errs() << '\n';

        const bool is_device = llvm::any_of(
            cmd->getArguments(),
            [](llvm::StringRef A) { return A == "-fcuda-is-device"; });
        
        if (is_device) continue;

        auto CI = std::make_unique<clang::CompilerInstance>();

        auto &diag_opts = *new clang::DiagnosticOptions();
        auto diag_printer = std::make_unique<clang::TextDiagnosticPrinter>(
            llvm::errs(), &diag_opts);
        CI->createDiagnostics(diag_printer.release(), /*ShouldOwnClient=*/true);

        clang::CompilerInvocation::CreateFromArgs(
            CI->getInvocation(), cmd->getArguments(), CI->getDiagnostics());

        CI->getPreprocessorOpts().addRemappedFile(
            "gpu_host.cu",
            llvm::MemoryBuffer::getMemBuffer(host_code, "gpu_host.cu")
                .release());

        clang::EmitLLVMOnlyAction act(ts_ctx.getContext());
        bool ok = false;
        llvm::CrashRecoveryContext CRC;
        CRC.RunSafelyOnThread([&] { ok = CI->ExecuteAction(act); });

        if (!ok || CI->getDiagnostics().hasErrorOccurred()) {
            llvm::errs() << "[JIT] " << (is_device ? "device" : "host")
                         << " compilation failed — abort\n";
            return {};
        }

        if (auto M = act.takeModule())
            modules.push_back(std::move(M));
    }

    {
        llvm::orc::SymbolMap m;
        m[jit->mangleAndIntern("gpu_kernel")] =
            llvm::orc::ExecutorSymbolDef(
                llvm::orc::ExecutorAddr::fromPtr(&kernel_function),
                llvm::JITSymbolFlags::Exported);

        if (auto err = jit->getMainJITDylib().define(
                llvm::orc::absoluteSymbols(m))) {
            llvm::errs() << err << '\n';
            return false;
        }
    }

    if (modules.empty())
        return false;

    for (auto &M : modules) {
        M->setDataLayout(jit->getDataLayout());
        M->setTargetTriple(jit->getTargetTriple().str());

        llvm::orc::ThreadSafeModule tsm(std::move(M), ts_ctx);
        if (auto err = jit->addIRModule(std::move(tsm))) {
            llvm::errs() << err << '\n';
            return false;
        }
    }

    // Look up kernel function symbol
    auto sym_or_err = jit->lookup("execute_query");
    if (!sym_or_err) {
        llvm::errs() << sym_or_err.takeError() << '\n';
        return false;
    }
    main_function = sym_or_err->toPtr<void *>();

    // compiled_cache.try_emplace(code_hash, main_function, 1);
    return true;
}

bool GpuJitCompiler::CompileAndLoad(const std::string &cuda_code)
{
    std::string code_hash = CalculateHash(cuda_code);

    // Check cache
    auto it = compiled_cache.find(code_hash);
    if (it != compiled_cache.end()) {
        it->second.second++;  // Increment ref count
        main_function = it->second.first;
        return true;
    }

    // Compile CUDA code to LLVM IR
    auto mods = CompileToIR(cuda_code);
    if (mods.empty())
        return false;

    // Add IR module to JIT
    for (auto &M : mods) {
        M->setDataLayout(jit->getDataLayout());
        M->setTargetTriple(jit->getTargetTriple().str());

        llvm::orc::ThreadSafeModule tsm(std::move(M), ts_ctx);
        if (auto err = jit->addIRModule(std::move(tsm))) {
            llvm::errs() << err << '\n';
            return false;
        }
    }

    // Look up kernel function symbol
    auto sym_or_err = jit->lookup("gpu_kernel");
    if (!sym_or_err) {
        llvm::errs() << sym_or_err.takeError() << '\n';
        return false;
    }
    main_function = sym_or_err->toPtr<void *>();

    compiled_cache.try_emplace(code_hash, main_function, 1);
    return true;
}

void GpuJitCompiler::ReleaseKernel(const std::string &code_hash)
{
    auto it = compiled_cache.find(code_hash);
    if (it != compiled_cache.end()) {
        it->second.second--;  // Decrement ref count
        if (it->second.second == 0) {
            compiled_cache.erase(it);
        }
    }
}

void GpuJitCompiler::Cleanup()
{
    compiled_cache.clear();
    jit.reset();
    compiler.reset();
    main_function = nullptr;
}

std::string GpuJitCompiler::CalculateHash(const std::string &code)
{
    return std::to_string(std::hash<std::string>{}(code));
}

bool GpuJitCompiler::ensureCudaContext() {
    if (cuda_context_initialized) return true;

    CUresult r;
    r = cuInit(0);
    if (r != CUDA_SUCCESS) return false;

    CUdevice dev;
    r = cuDeviceGet(&dev, 0);
    if (r != CUDA_SUCCESS) return false;

    CUcontext ctx;
    r = cuCtxCreate(&ctx, 0, dev);
    if (r != CUDA_SUCCESS) return false;

    cuda_context_initialized = true;
    return true;
}

}  // namespace duckdb