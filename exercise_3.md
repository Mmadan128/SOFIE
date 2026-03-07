# Exercise 3 - Play with the SOFIE alpaka Implementation

I built the experimental SOFIE alpaka repo and went through the code. These are the main things I noticed.

## Improvements / Extensions Found

### 1. Virtual Method Signature Mismatch

`ROperator.hxx` declares `Generate_GPU_Kernel_ALPAKA` with a `std::string opName` parameter, but operators like `ROperator_Relu`, `ROperator_Gather`, `ROperator_Tile`, `ROperator_Cast` and `ROperator_BasicUnary` define it without that parameter and without `override`. So C++ sees them as separate overloads and virtual dispatch always hits the base stub returning `""`, the kernel struct never gets emitted. You can see it in the generated `Tile5D_FromONNX_GPU_ALPAKA.hxx` which has a `tileKernel` member but no `struct TileKernel` anywhere, so it would fail to compile. Adding the parameter and `override` to all affected operators fixes it, and `-Wsuggest-override` would prevent this going forward.

### 2. Kernel Deduplication Broken due to UNDEFINED OperatorKind

`RModel::GenerateSessionCode_GPU_ALPAKA()` tracks which kernel structs have been emitted using `OperatorKind` to avoid duplicates. But nearly every operator just leaves `fKind` as `OperatorKind::UNDEFINED`. So after the first operator emits its struct, all the rest get skipped. In something like Sigmoid followed by Tanh, only Sigmoid's kernel ends up in the header. Either each operator needs a proper unique `OperatorKind`, or the deduplication should use the kernel struct type name instead.

### 3. Output Type Hardcoded to float

`GenerateOutput_GPU_ALPAKA()` always writes `float` as the output type no matter what the model actually outputs. A model with double/ int64 outputs will compile fine but produce wrong results. Models with more than one output just throw at runtime. It should read the type from `GetTensorType()` and use a `std::tuple` for multiple outputs.

### 4. No HIP Backend Test

CMake supports `-DALPAKA_BACKEND=hip` but the only test file `TestCustomModelsFromONNXForAlpakaCuda.cxx` is hardcoded to CUDA with `alpaka::PlatformCudaRt`, `cuda_runtime.h` and `nvml.h`. Since alpaka exists precisely to abstract over backends, the tests should work on AMD too with just a flag swap. A test parameterized on a compile time backend tag would cover this with minimal effort.

### 5. Missing GPU Implementations for Some Activation Operators

`ROperator_Selu.hxx`, `ROperator_Swish.hxx` and `ROperator_Erf.hxx` only have the CPU `Generate()` path, no `Generate_GPU_ALPAKA` at all. They silently produce an empty `infer()` on GPU. These are straightforward to add and would fill a noticeable gap in GPU operator coverage.