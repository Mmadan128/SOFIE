# Exercise 5 

What I implemented:

- Added BatchNorm GPU support in `ROperator_BatchNormalization.hxx`:
  - `Generate_GPU_Kernel_ALPAKA`
  - `Generate_GPU_Kernel_Definitions_ALPAKA`
  - `Generate_GPU_ALPAKA`
- Added a new test: `SofieAlpakaTest.BatchNorm`
- Added expected output file: `BatchNorm.ref.hxx`
- Re-ran the ONNX code generation step so generated GPU files were updated.

After that, BatchNorm worked in the Alpaka test pipeline, and the full test run was still passing all cases.
