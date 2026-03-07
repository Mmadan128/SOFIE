#include <numeric>
#include <cstddef>

#include "Linear_64_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/Linear_64.ref.hxx"

#include "AddBroadcast1_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/AddBroadcast1.ref.hxx"

#include "LinearWithLeakyRelu_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/LinearWithLeakyRelu.ref.hxx"

#include "LinearWithSigmoid_FromONNX_GPU_ALPAKA.hxx"
#include "input_models/references/LinearWithSigmoid.ref.hxx"

#include "Transpose_FromONNX_GPU_ALPAKA.hxx"
#include "Tanh_FromONNX_GPU_ALPAKA.hxx"

#include "Softmax2d_FromONNX_GPU_ALPAKA.hxx"
#include "Softmax3d_FromONNX_GPU_ALPAKA.hxx"
#include "Elu_FromONNX_GPU_ALPAKA.hxx"
#include "Softplus_FromONNX_GPU_ALPAKA.hxx"

#include "Concat_0D_FromONNX_GPU_ALPAKA.hxx"
#include "ScatterElements_FromONNX_GPU_ALPAKA.hxx"


#include <alpaka/alpaka.hpp>
#include <cuda_runtime.h>
#include <nvml.h>
#include "gtest/gtest.h"

constexpr float DEFAULT_TOLERANCE = 1e-3f;

using Idx = std::size_t;
using Dim = alpaka::DimInt<1>;
using Ext1D = alpaka::Vec<Dim, Idx>;

class SofieAlpakaTest : public ::testing::Test {
protected:
    // Shared devices and platforms
    alpaka::PlatformCpu hostPlatform;
    alpaka::DevCpu host;
    alpaka::PlatformCudaRt platform;
    alpaka::DevCudaRt device;
    alpaka::Queue<alpaka::DevCudaRt, alpaka::NonBlocking> queue;

    SofieAlpakaTest() 
        : hostPlatform{}
        , host(alpaka::getDevByIdx(hostPlatform, 0u))
        , platform{}
        , device(alpaka::getDevByIdx(platform, 0u))
        , queue(device)
    {
    }

    void SetUp() override {
        cudaDeviceSynchronize();
    }

    void TearDown() override {
        alpaka::wait(queue);
        cudaDeviceSynchronize();
    }

    ~SofieAlpakaTest() override {
        cudaDeviceSynchronize();
    }
};


// TEST_F(SofieAlpakaTest, Linear64)
// {
//    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

//    auto A = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{1600}));
//    float *A_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(A));

//    for (Idx i = 0; i < 1600; ++i) {
//       A_ptr[i] = 1.0;
//    }

//    auto A_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{1600}));
//    alpaka::memcpy(queue, A_d, A);
//    alpaka::wait(queue);

//    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{160}));
   
//    {
//       SOFIE_Linear_64::Session<alpaka::TagGpuCudaRt> session("Linear_64_FromONNX_GPU_ALPAKA.dat");
//       auto result = session.infer(A_d);
//       alpaka::wait(queue);
//       cudaDeviceSynchronize();

//       alpaka::memcpy(queue, result_h, result);
//       alpaka::wait(queue);
//    }
   
//    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
//    float *correct = Linear_64_ExpectedOutput::all_ones;

//    for (size_t i = 0; i < 160; ++i) {
//       EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE);
//    }
// }

TEST_F(SofieAlpakaTest, LinearWithLeakyRelu)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input({
      0.4369, -0.6882,  1.0309, -1.0263, -0.1519,  1.2237, -0.7054, -0.1762,
      -0.6811, -2.2597,  1.0388, -0.7993,  0.1468,  1.3257, -0.4714, -0.0958,
      0.7057, -0.3749, -0.3310,  0.0986, -0.1370,  0.0832, -1.6465, -0.2793
   });

   auto A = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float *A_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(A));

   for (Idx i = 0; i < input.size(); ++i) {
      A_ptr[i] = input[i];
   }

   auto A_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, A_d, A);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{24}));
   
   {
      SOFIE_LinearWithLeakyRelu::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(A_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();

      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }
   
   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float *correct = LinearWithLeakyRelu_ExpectedOutput::outputs;

   for (size_t i = 0; i < 24; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE);
   }
}

TEST_F(SofieAlpakaTest, LinearWithSigmoid)
{

   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   auto A = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{48}));
   float *A_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(A));

   for (Idx i = 0; i < 48; ++i) {
      A_ptr[i] = 1.0;
   }

   auto A_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{48}));
   alpaka::memcpy(queue, A_d, A);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{24}));
   
   {
      SOFIE_LinearWithSigmoid::Session<alpaka::TagGpuCudaRt> session("LinearWithSigmoid_FromONNX_GPU_ALPAKA.dat");
      auto result = session.infer(A_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();

      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float *correct = LinearWithSigmoid_ExpectedOutput::all_ones;
   for (size_t i = 0; i < 24; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE);
   }
}

TEST_F(SofieAlpakaTest, AddBroadcast1)
{

   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   auto A = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{5}));
   float *A_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(A));

   auto B = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{20}));
   float *B_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(B));

   std::vector<float> A_vec({-0.78023305, -1.34029483, -3.01482951, 0.53641361,
                 -1.22594789});
   std::vector<float> B_vec({1.0626695,  0.43842875,  1.22476468,  0.79763274,  0.98688211,
                 0.25267614, 0.44874883,  0.31516773,  -0.78771195, 0.64565664,
                 0.50450593, -0.41265227, -0.22474539, -0.22362374, 0.00509674,
                 0.16927211, 1.06756969,  -0.81634773, 0.88467744,  0.78902059});

   for (Idx i = 0; i < A_vec.size(); ++i) {
      A_ptr[i] = A_vec[i];
   }

   for (Idx i = 0; i < B_vec.size(); ++i) {
      B_ptr[i] = B_vec[i];
   }

   auto A_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{5}));
   alpaka::memcpy(queue, A_d, A);
   alpaka::wait(queue);

   auto B_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{20}));
   alpaka::memcpy(queue, B_d, B);
   alpaka::wait(queue);
   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{20}));
   
   {
       SOFIE_AddBroadcast1::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(A_d, B_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();

      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }  

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   float *correct = AddBroadcast1_ExpectedOutput::output;
   for (size_t i = 0; i < 20; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE);
   }
}

TEST_F(SofieAlpakaTest, Transpose)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    // Input shape: (2, 1, 3, 4) -> 24 elements
    constexpr Idx inputSize = 24;
    // Output shape: (2, 3, 4, 1) -> 24 elements
    constexpr Idx outputSize = 24;

    auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{inputSize}));
    float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));

    std::vector<float> input_vec({
        // shape (2, 1, 3, 4)
        0.f,  1.f,  2.f,  3.f,
        4.f,  5.f,  6.f,  7.f,
        8.f,  9.f, 10.f, 11.f,

       12.f, 13.f, 14.f, 15.f,
       16.f, 17.f, 18.f, 19.f,
       20.f, 21.f, 22.f, 23.f
    });

    for (Idx i = 0; i < inputSize; ++i)
        input_ptr[i] = input_vec[i];

    auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{inputSize}));
    alpaka::memcpy(queue, input_d, input_h);
    alpaka::wait(queue);

    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{outputSize}));

    {
        SOFIE_Transpose::Session<alpaka::TagGpuCudaRt> session;
        auto result = session.infer(input_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();

        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    std::vector<float> expected(outputSize);
    std::vector<size_t> inputShape  = {2, 1, 3, 4};
    std::vector<size_t> perm        = {0, 2, 3, 1};
    std::vector<size_t> outputShape = {2, 3, 4, 1};

    std::vector<size_t> inputStrides  = {12, 12, 4, 1};
    std::vector<size_t> outputStrides = {12,  4,  1, 1};

    for (size_t i = 0; i < outputSize; ++i)
    {
        size_t remaining = i;
        size_t inputIdx  = 0;
        for (size_t d = 0; d < 4; ++d)
        {
            size_t const coord = remaining / outputStrides[d];
            remaining          = remaining - coord * outputStrides[d];
            inputIdx          += coord * inputStrides[perm[d]];
        }
        expected[i] = input_vec[inputIdx];
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    for (size_t i = 0; i < outputSize; ++i)
        EXPECT_LE(std::abs(res_ptr[i] - expected[i]), TOLERANCE);
}

TEST_F(SofieAlpakaTest, Tanh)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input({
      -0.3896f, -0.3521f,  0.0363f,  1.0961f,  0.5085f, -0.8524f, -0.6766f,  0.2421f,
       1.5969f,  1.3874f, -0.2112f, -0.6894f, -0.5069f, -2.1401f, -0.7088f,  1.1657f,
       1.3494f,  0.8133f,  1.7153f, -0.8638f, -0.1971f,  0.0411f, -0.5661f, -0.2516f
   });

   std::vector<float> expected({
      -0.371015f, -0.338237f,  0.036284f,  0.799094f,  0.468776f, -0.692321f, -0.589305f,  0.237478f,
       0.921201f,  0.882598f, -0.208115f, -0.597596f, -0.467526f, -0.972698f, -0.609924f,  0.822889f,
       0.873912f,  0.671407f,  0.937295f, -0.698210f, -0.194587f,  0.041077f, -0.512489f, -0.246422f
   });

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < input.size(); ++i)
      input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{24}));

   {
      SOFIE_Tanh::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();

      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   for (size_t i = 0; i < 24; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - expected[i]), TOLERANCE)
         << "mismatch at index " << i
         << ": got " << res_ptr[i] << ", expected " << expected[i];
   }
}

TEST_F(SofieAlpakaTest, Softmax2d)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;
   constexpr Idx N = 3;

   std::vector<float> input({1.0f, 2.0f, 3.0f});
   std::vector<float> expected({0.09003057f, 0.24472847f, 0.66524096f});

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{N}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < N; ++i) input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{N}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{N}));

   {
      SOFIE_Softmax2d::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   for (size_t i = 0; i < N; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - expected[i]), TOLERANCE)
         << "mismatch at index " << i << ": got " << res_ptr[i] << ", expected " << expected[i];
   }
   // verify the output sums to 1
   float rowSum = 0.f;
   for (size_t i = 0; i < N; ++i) rowSum += res_ptr[i];
   EXPECT_LE(std::abs(rowSum - 1.0f), TOLERANCE) << "softmax output row does not sum to 1: " << rowSum;
}

TEST_F(SofieAlpakaTest, Softmax3d)
{
   // shape [2,3,4], axis=1
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;
   constexpr Idx N = 24;

   std::vector<float> input({
       0.5577f, -1.9000f, -0.8999f, -1.1072f,
       0.9459f,  0.7068f,  1.5687f, -1.6522f,
      -0.3123f, -1.8808f, -1.1254f,  0.0214f,
      -1.8939f, -1.2046f,  0.5995f,  0.1798f,
      -1.1182f,  0.3571f,  1.2377f, -1.9740f,
       1.2233f,  0.7926f, -0.6390f, -1.3781f
   });

   std::vector<float> expected({
      0.34562895f, 0.06420550f, 0.07350766f, 0.21407925f,
      0.50956929f, 0.87034428f, 0.86782461f, 0.12413209f,
      0.14480177f, 0.06545015f, 0.05866772f, 0.66178864f,
      0.03882715f, 0.07613065f, 0.31417996f, 0.75379521f,
      0.08433694f, 0.36290857f, 0.59476483f, 0.08747217f,
      0.87683588f, 0.56096071f, 0.09105522f, 0.15873255f,
   });

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{N}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < N; ++i) input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{N}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{N}));

   {
      SOFIE_Softmax3d::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   for (size_t i = 0; i < N; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - expected[i]), TOLERANCE)
         << "mismatch at index " << i << ": got " << res_ptr[i] << ", expected " << expected[i];
   }
   constexpr size_t nBatch = 2, nRows = 3, nCols = 4;
   for (size_t b = 0; b < nBatch; ++b) {
      for (size_t c = 0; c < nCols; ++c) {
         float colSum = 0.f;
         for (size_t r = 0; r < nRows; ++r) colSum += res_ptr[b * nRows * nCols + r * nCols + c];
         EXPECT_LE(std::abs(colSum - 1.0f), TOLERANCE) << "batch " << b << " col " << c << " does not sum to 1: " << colSum;
      }
   }
}

TEST_F(SofieAlpakaTest, Elu)
{
   // Elu.onnx: shape [6], alpha=1. f(x)=x if x>=0, else exp(x)-1
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;
   constexpr Idx N = 6;

   std::vector<float> input({0.5f, -0.5f, 1.2f, -1.5f, 0.0f, -2.0f});
   std::vector<float> expected({0.50000000f, -0.39346933f, 1.20000005f, -0.77686983f, 0.00000000f, -0.86466473f});

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{N}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < N; ++i) input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{N}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{N}));

   {
      SOFIE_Elu::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   for (size_t i = 0; i < N; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - expected[i]), TOLERANCE)
         << "mismatch at index " << i << ": got " << res_ptr[i] << ", expected " << expected[i];
   }
}

TEST_F(SofieAlpakaTest, Softplus)
{
   // Softplus.onnx: shape [8], f(x) = log(1 + exp(x))
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;
   constexpr Idx N = 8;

   std::vector<float> input({-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f});
   std::vector<float> expected({0.12692805f, 0.31326166f, 0.47407699f, 0.69314718f,
                                0.97407699f, 1.31326175f, 2.12692785f, 3.04858732f});

   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{N}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));
   for (Idx i = 0; i < N; ++i) input_ptr[i] = input[i];

   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{N}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{N}));

   {
      SOFIE_Softplus::Session<alpaka::TagGpuCudaRt> session;
      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();
      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
   for (size_t i = 0; i < N; ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - expected[i]), TOLERANCE)
         << "mismatch at index " << i << ": got " << res_ptr[i] << ", expected " << expected[i];
   }
}

TEST_F(SofieAlpakaTest, Concat0D)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   std::vector<float> input({1.40519865e+00, -2.87660856e-01});
   std::vector<float> expected_output({
      1.40519865e+00, -2.87660856e-01,
      1.40519865e+00, -2.87660856e-01
   });

   // Host input buffer
   auto input_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{input.size()}));
   float* input_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(input_h));

   for (Idx i = 0; i < input.size(); ++i)
      input_ptr[i] = input[i];

   // Device input buffer
   auto input_d = alpaka::allocBuf<float, Idx>(device, Ext1D::all(Idx{input.size()}));
   alpaka::memcpy(queue, input_d, input_h);
   alpaka::wait(queue);

   // Host output buffer
   auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{expected_output.size()}));

   {
      SOFIE_Concat_0D::Session<alpaka::TagGpuCudaRt> session("Concat_0D_FromONNX_GPU_ALPAKA.dat");

      auto result = session.infer(input_d);
      alpaka::wait(queue);
      cudaDeviceSynchronize();

      alpaka::memcpy(queue, result_h, result);
      alpaka::wait(queue);
   }

   float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));

   for (size_t i = 0; i < expected_output.size(); ++i) {
      EXPECT_LE(std::abs(res_ptr[i] - expected_output[i]), TOLERANCE);
   }
}

TEST_F(SofieAlpakaTest, ScatterElements)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;

    std::vector<float>   input   (9, 0.f);
    std::vector<int64_t> indices = { 1, 0, 2, 0, 2, 1 };
    std::vector<float>   updates = { 1.f, 1.1f, 1.2f, 2.f, 2.1f, 2.2f };
    std::vector<float>   correct = { 2.f, 1.1f, 0.f, 1.f, 0.f, 2.2f, 0.f, 2.1f, 1.2f };

    // Allocate and fill host buffers
    auto input_h   = alpaka::allocBuf<float,   Idx>(host, Ext1D::all(Idx{input.size()}));
    auto indices_h = alpaka::allocBuf<int64_t, Idx>(host, Ext1D::all(Idx{indices.size()}));
    auto updates_h = alpaka::allocBuf<float,   Idx>(host, Ext1D::all(Idx{updates.size()}));

    float*   input_ptr   = reinterpret_cast<float*>  (alpaka::getPtrNative(input_h));
    int64_t* indices_ptr = reinterpret_cast<int64_t*>(alpaka::getPtrNative(indices_h));
    float*   updates_ptr = reinterpret_cast<float*>  (alpaka::getPtrNative(updates_h));

    for (Idx i = 0; i < input.size();   ++i) input_ptr[i]   = input[i];
    for (Idx i = 0; i < indices.size(); ++i) indices_ptr[i] = indices[i];
    for (Idx i = 0; i < updates.size(); ++i) updates_ptr[i] = updates[i];

    // Allocate device buffers and copy
    auto input_d   = alpaka::allocBuf<float,   Idx>(device, Ext1D::all(Idx{input.size()}));
    auto indices_d = alpaka::allocBuf<int64_t, Idx>(device, Ext1D::all(Idx{indices.size()}));
    auto updates_d = alpaka::allocBuf<float,   Idx>(device, Ext1D::all(Idx{updates.size()}));

    alpaka::memcpy(queue, input_d,   input_h);
    alpaka::memcpy(queue, indices_d, indices_h);
    alpaka::memcpy(queue, updates_d, updates_h);
    alpaka::wait(queue);

    // Host result buffer
    auto result_h = alpaka::allocBuf<float, Idx>(host, Ext1D::all(Idx{correct.size()}));

    {
        SOFIE_ScatterElements::Session<alpaka::TagGpuCudaRt> session;
        auto result = session.infer(input_d, indices_d, updates_d);
        alpaka::wait(queue);
        cudaDeviceSynchronize();

        alpaka::memcpy(queue, result_h, result);
        alpaka::wait(queue);
    }

    float* res_ptr = reinterpret_cast<float*>(alpaka::getPtrNative(result_h));
    EXPECT_EQ(correct.size(), 9u);
    for (size_t i = 0; i < correct.size(); ++i){
        EXPECT_LE(std::abs(res_ptr[i] - correct[i]), TOLERANCE);
    }
}