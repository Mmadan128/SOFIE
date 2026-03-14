# Exercise 4

I implemented these ONNX operators on GPU with Alpaka:

1. Tanh
2. Softmax
3. Elu
4. Softplus

Main things I changed:

- Added GPU kernel functions for Tanh, Softmax, and Elu.
- Added Softplus as a new operator including its parsing and registration.
- Added tests for all of them in the Alpaka CUDA test file:
  - `SofieAlpakaTest.Tanh`
  - `SofieAlpakaTest.Softmax2d`
  - `SofieAlpakaTest.Softmax3d`
  - `SofieAlpakaTest.Elu`
  - `SofieAlpakaTest.Softplus`

I ran the full `TestCustomModelsFromONNXForAlpakaCuda`, and the code passed all test cases.
