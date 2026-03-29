#ifndef SOFIE_ROPERATOR_HARDSWISH
#define SOFIE_ROPERATOR_HARDSWISH

#include "SOFIE/SOFIE_common.hxx"
#include "SOFIE/ROperator.hxx"
#include "SOFIE/RModel.hxx"

#include <sstream>

namespace SOFIE {

template <typename T>
class ROperator_HardSwish final : public ROperator {
private:
   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShape;

public:
   ROperator_HardSwish() {}
   ROperator_HardSwish(std::string nameX, std::string nameY)
      : fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))
   {
      fInputTensorNames = {fNX};
      fOutputTensorNames = {fNY};
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override { return input; }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override
   {
      auto ret = input;
      return ret;
   }

   void Initialize(RModel &model) override
   {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("TMVA SOFIE HardSwish Op Input Tensor is not found in model");
      }
      fShape = model.GetTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
   }

   std::string Generate(std::string OpName) override
   {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator HardSwish called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t length = ConvertShapeToLength(fShape);
      out << "\n//------ HARDSWISH\n";
      out << SP << "#ifdef _OPENMP\n";
      out << SP << "#pragma omp parallel for\n";
      out << SP << "#endif\n";
      out << SP << "for (size_t id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "auto x = tensor_" << fNX << "[id];\n";
      out << SP << SP << "auto t = x + 3.0f;\n";
      out << SP << SP << "if (t < 0.0f) t = 0.0f;\n";
      out << SP << SP << "if (t > 6.0f) t = 6.0f;\n";
      out << SP << SP << "tensor_" << fNY << "[id] = x * t / 6.0f;\n";
      out << SP << "}\n";
      return out.str();
   }

   std::string Generate_GPU_Kernel_ALPAKA(std::string /*opName*/) override
   {
      std::string op;
      op = "\n//------ HARDSWISH_KERNEL_ALPAKA\n";
      op += "struct HardSwishKernel {\n";
      op += SP + "template<typename TAcc, typename T>\n";
      op += SP + "ALPAKA_FN_ACC void operator()(TAcc const & acc, T const* __restrict__ data, T* __restrict__ out, std::size_t numElements) const {\n";
      op += SP + SP + "for (auto idx : alpaka::uniformElements(acc, numElements)) {\n";
      op += SP + SP + SP + "T x = data[idx];\n";
      op += SP + SP + SP + "T t = x + T(3);\n";
      op += SP + SP + SP + "if (t < T(0)) t = T(0);\n";
      op += SP + SP + SP + "if (t > T(6)) t = T(6);\n";
      op += SP + SP + SP + "out[idx] = x * t / T(6);\n";
      op += SP + SP + "}\n";
      op += SP + "}\n";
      op += "};\n";
      return op;
   }

   std::string Generate_GPU_Kernel_Definitions_ALPAKA(std::string /*opName*/) override
   {
      return SP + "HardSwishKernel hardSwishKernel;\n";
   }

   std::string Generate_GPU_ALPAKA(std::string OpName) override
   {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator HardSwish called to Generate without being initialized first");
      }
      std::stringstream out;
      auto length = ConvertShapeToLength(fShape);
      out << "\n//------ HARDSWISH_GPU_ALPAKA\n";
      out << SP << "auto const elementsPerThread_" << fNX << " = Vec::all(static_cast<Idx>(1));\n";
      out << SP << "auto const elementsPerGrid_" << fNX << " = Vec::all(Idx{" << length << "});\n";
      out << SP << "alpaka::KernelCfg<Acc> const kernelCfg_" << fNX << " = {elementsPerGrid_" << fNX
          << ", elementsPerThread_" << fNX << "};\n";
      out << SP << "auto const workDiv_" << fNX << " = alpaka::getValidWorkDiv(kernelCfg_" << fNX
          << ", devAcc, hardSwishKernel, alpaka::getPtrNative(deviceBuf_" << fNX << "), alpaka::getPtrNative(deviceBuf_"
          << fNY << "), static_cast<Idx>(" << length << "));\n";
      out << SP << "alpaka::exec<Acc>(queue, workDiv_" << fNX
          << ", hardSwishKernel, alpaka::getPtrNative(deviceBuf_" << fNX << "), alpaka::getPtrNative(deviceBuf_"
          << fNY << "), static_cast<Idx>(" << length << "));\n";
      return out.str();
   }

   std::string GetFusableOutputTensorName() override { return fNY; }

   void UpdateFusableTensorName(std::string fusable_tensor_name,
                                const std::function<void(const std::string &)> &removal_func)
   {
      removal_func(fNX);
      removal_func(fNY);
      fNX = fusable_tensor_name;
      fNY = fusable_tensor_name;
      fInputTensorNames[0] = fNX;
      fOutputTensorNames[0] = fNY;
   }
};

} // namespace SOFIE

#endif // SOFIE_ROPERATOR_HARDSWISH
