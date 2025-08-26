# NeuralNetwork Class Documentation

## Overview

The `NeuralNetwork` class provides a complete implementation of a feedforward neural network with the following features:

- **Multiple layers**: Supports any number of hidden layers
- **ReLU activation**: All hidden layers use ReLU activation 
- **Softmax output**: Final layer uses softmax for classification
- **Cross-entropy loss**: Optimized for classification tasks
- **GPU acceleration**: Implemented using CUDA and cuTensor
- **Automatic differentiation**: Complete backward pass with parameter updates

## Class Interface

### Constructor
```cpp
NeuralNetwork(const std::vector<int>& layer_sizes);
```
- **Parameters**: Vector of layer sizes (e.g., `{784, 128, 64, 10}` for MNIST)
- **Behavior**: 
  - All layers except the last have ReLU activation
  - Final layer has no activation (for softmax + cross-entropy)

### Main Methods

#### Forward Pass
```cpp
float forward(const float* input, const float* target);
```
- **Parameters**:
  - `input`: Device pointer to input vector 
  - `target`: Device pointer to one-hot encoded target vector
- **Returns**: Cross-entropy loss value
- **Behavior**: 
  1. Passes input through all layers
  2. Applies softmax to final layer output
  3. Computes cross-entropy loss against target

#### Backward Pass
```cpp
void backward(const float* input, const float* target);
```
- **Parameters**: Same input and target used in forward pass
- **Behavior**:
  1. Computes gradients of loss w.r.t. all parameters
  2. Updates all layer parameters automatically
  3. No separate optimizer step needed

#### Prediction
```cpp
int predict(const float* input);
```
- **Parameters**: Device pointer to input vector
- **Returns**: Index of predicted class (argmax of softmax output)

### Utility Methods
```cpp
int get_num_layers() const;
const NeuralLayer* get_layer(int index) const;
```

## Usage Example

```cpp
#include "nn_utils.h"

// Create network: 784 input -> 128 hidden -> 64 hidden -> 10 output
std::vector<int> architecture = {784, 128, 64, 10};
NeuralNetwork network(architecture);

// Allocate device memory for data
float *input_d, *target_d;
cudaMalloc(&input_d, 784 * sizeof(float));
cudaMalloc(&target_d, 10 * sizeof(float));

// Copy your data to device...
// cudaMemcpy(input_d, input_host, 784 * sizeof(float), cudaMemcpyHostToDevice);
// cudaMemcpy(target_d, target_host, 10 * sizeof(float), cudaMemcpyHostToDevice);

// Training loop
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    float loss = network.forward(input_d, target_d);
    network.backward(input_d, target_d);
    
    if (epoch % 100 == 0) {
        printf("Epoch %d: Loss = %f\n", epoch, loss);
    }
}

// Make prediction
int predicted_class = network.predict(input_d);
```

## Implementation Details

### Memory Management
- All intermediate layer outputs are stored in GPU memory
- Automatic cleanup in destructor
- Efficient memory reuse during training

### cuTensor-Enhanced Softmax Implementation
- **Numerically Stable**: Max subtraction using cuTensor reductions
- **Hybrid Approach**: cuTensor for tensor management + CUDA kernels for exponentials
- **Optimized Memory**: cuTensor workspace management for temporary storage
- **Precision Handling**: Epsilon values and overflow prevention

### cuTensor-Enhanced Cross-Entropy Loss
- **Tensor Integration**: cuTensor descriptors for efficient data flow
- **Fused Operations**: Coordinated softmax + cross-entropy computation
- **Gradient Computation**: Optimized `∇L = softmax_output - target`
- **GPU Acceleration**: cuTensor-managed reduction operations

### Backpropagation
- Automatic gradient computation through all layers
- Immediate parameter updates (no separate optimizer needed)
- Efficient memory usage with temporary gradient storage

### cuTensor-Enhanced Operations
The implementation leverages cuTensor library with custom CUDA kernels for:
- **cuTensor Descriptors**: Efficient tensor metadata management
- **Hybrid Softmax**: cuTensor reductions + CUDA kernels for exponential operations
- **Cross-entropy Loss**: cuTensor-coordinated computation with numerical stability
- **Gradient Computation**: Optimized elementwise operations
- **Memory Management**: cuTensor workspace allocation and reuse

## Performance Considerations

1. **Memory Layout**: All tensors stored in contiguous GPU memory
2. **Kernel Fusion**: Softmax and loss computation are optimized
3. **Stream Usage**: Single CUDA stream for all operations
4. **Workspace Reuse**: Efficient temporary memory management

## Error Handling

- Comprehensive CUDA error checking with `HANDLE_CUDA_ERROR` macro
- cuTensor error handling with `HANDLE_CUTENSOR_ERROR` macro
- Validation of network architecture in constructor

## Limitations

1. **Fixed Architecture**: Network structure cannot be changed after creation
2. **Single Stream**: All operations use one CUDA stream
3. **Float Precision**: Currently supports only single-precision floating point
4. **Classification Only**: Designed specifically for classification tasks

## Dependencies

- CUDA Runtime
- cuTensor library
- C++11 or later
- NVIDIA GPU with compute capability 6.0+
