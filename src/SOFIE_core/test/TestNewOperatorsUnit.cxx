#include "SOFIE/RModel.hxx"
#include "SOFIE/ROperator_Gelu.hxx"
#include "SOFIE/ROperator_HardSwish.hxx"

#include "gtest/gtest.h"

namespace {

SOFIE::RModel MakeModelWithInputX() {
   SOFIE::RModel model("UnitModel", "now");
   model.AddInputTensorInfo("X", SOFIE::ETensorType::FLOAT, std::vector<size_t>{2, 3});
   return model;
}

TEST(NewOperatorsUnit, GeluApproximateCpuAndGpuGeneration) {
   auto model = MakeModelWithInputX();
   SOFIE::ROperator_Gelu<float> op("X", "Y", true);
   op.Initialize(model);

   auto cpuCode = op.Generate("gelu_test");
   EXPECT_NE(cpuCode.find("GELU"), std::string::npos);
   EXPECT_NE(cpuCode.find("#pragma omp parallel for"), std::string::npos);
   EXPECT_NE(cpuCode.find("std::tanh"), std::string::npos);

   auto gpuKernel = op.Generate_GPU_Kernel_ALPAKA("");
   EXPECT_NE(gpuKernel.find("GELU_KERNEL_ALPAKA"), std::string::npos);
   EXPECT_NE(gpuKernel.find("alpaka::uniformElements"), std::string::npos);

   auto gpuLaunch = op.Generate_GPU_ALPAKA("gelu_test");
   EXPECT_NE(gpuLaunch.find("alpaka::getValidWorkDiv"), std::string::npos);
   EXPECT_NE(gpuLaunch.find("alpaka::exec<Acc>"), std::string::npos);
   EXPECT_NE(gpuLaunch.find(", true)"), std::string::npos);
}

TEST(NewOperatorsUnit, GeluExactCpuAndGpuGeneration) {
   auto model = MakeModelWithInputX();
   SOFIE::ROperator_Gelu<float> op("X", "Y", false);
   op.Initialize(model);

   auto cpuCode = op.Generate("gelu_exact_test");
   EXPECT_NE(cpuCode.find("std::erf"), std::string::npos);

   auto gpuLaunch = op.Generate_GPU_ALPAKA("gelu_exact_test");
   EXPECT_NE(gpuLaunch.find(", false)"), std::string::npos);
}

TEST(NewOperatorsUnit, HardSwishCpuAndGpuGeneration) {
   auto model = MakeModelWithInputX();
   SOFIE::ROperator_HardSwish<float> op("X", "Y");
   op.Initialize(model);

   auto cpuCode = op.Generate("hswish_test");
   EXPECT_NE(cpuCode.find("HARDSWISH"), std::string::npos);
   EXPECT_NE(cpuCode.find("#pragma omp parallel for"), std::string::npos);
   EXPECT_NE(cpuCode.find("x * t / 6.0f"), std::string::npos);

   auto gpuKernel = op.Generate_GPU_Kernel_ALPAKA("");
   EXPECT_NE(gpuKernel.find("HARDSWISH_KERNEL_ALPAKA"), std::string::npos);
   EXPECT_NE(gpuKernel.find("alpaka::uniformElements"), std::string::npos);

   auto gpuLaunch = op.Generate_GPU_ALPAKA("hardswish_test");
   EXPECT_NE(gpuLaunch.find("alpaka::getValidWorkDiv"), std::string::npos);
   EXPECT_NE(gpuLaunch.find("alpaka::exec<Acc>"), std::string::npos);
}

} // namespace
