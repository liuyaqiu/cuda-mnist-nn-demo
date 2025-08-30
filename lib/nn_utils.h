#ifndef NN_UTILS_H
#define NN_UTILS_H

#include <cuda_runtime.h>
#include <cutensor.h>
#include <vector>
#include <cstdint>

// Error handling macros
#define HANDLE_CUTENSOR_ERROR(x)                                             \
{ const auto err = x;                                               \
    if( err != CUTENSOR_STATUS_SUCCESS )                              \
    { printf("cuTENSOR Error in %s at %s:%d: %s\n", __func__, __FILE__, __LINE__, cutensorGetErrorString(err)); exit(-1); } \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
    if( err != cudaSuccess )                                        \
    { printf("CUDA Error in %s at %s:%d: %s\n", __func__, __FILE__, __LINE__, cudaGetErrorString(err)); exit(-1); } \
};

/*
Important instructions:
    1. All deveice and host memory are stored as column major order tensor, such as (batch_size, input_elements)
    2. If it's possible, we should use CUDA managed memory to avoid manual memory allocation and deallocation.
    3. If it's possible, we should use cuTENSOR to calculate tensor permute(unary), element-wise(binary), and reduction(ternary) operations.
    4. When cuTENSOR doesn't have native support for a certain operation, such as add scalar or low dimension tensor to a tensor(which needs broadcast),
       we should use kernel function to implement it.
    5. We should add some important assertions to check the correctness of some operations, for example the sum of softmax logtis should be 1.
    6. We need to consider add unit test for some important implementations, such as softmax, cross-entropy, and ReLU.
*/


class NeuralLayer {
public:
    // Static function to set random seed for weight initialization
    static void set_random_seed(unsigned int seed);
    
    // Static function to get a time-based random seed
    static unsigned int get_time_based_seed();
    
    // Constructor
    NeuralLayer(int input_elements, int output_elements, bool non_linear_activate);
    
    // Destructor
    ~NeuralLayer();
    
    // Batch forward pass: Y = ReLU(X * W + B) or Y = X * W + B
    // input_batch: (batch_size, input_elements)
    // output_batch: (batch_size, output_elements)
    void forward_batch(const float* input_batch, float* output_batch, int batch_size);
    
    // Batch backward pass: compute gradients dW, db, dX given dY
    // input_batch: (batch_size, input_elements)
    // dy_batch: (batch_size, output_elements)
    // dW: (input_elements, output_elements)
    // db: (output_elements)
    // dx_batch: (batch_size, input_elements)
    void backward_batch(const float* input_batch, const float* dy_batch, float* dW, float* db, float* dx_batch, int batch_size);
    
    // Update parameters: W = W - learning_rate * dW, b = b - learning_rate * db (gradient descent)
    // max_gradient_norm: maximum allowed gradient norm (default 5.0 for stability)
    // weight_decay: L2 regularization coefficient (default 0.0)
    void update_parameters(const float* dW, const float* db, float learning_rate, 
                          float max_gradient_norm = 5.0f, float weight_decay = 0.0f);
    
    // Getters
    int get_input_elements() const { return input_elements_; }
    int get_output_elements() const { return output_elements_; }
    bool get_non_linear_activate() const { return non_linear_activate_; }
    
    // Get pointers to weights and biases (for inspection/testing)
    const float* get_weights() const { return W_d_; }
    const float* get_biases() const { return b_d_; }

private:
    // Layer dimensions
    int input_elements_;
    int output_elements_;
    bool non_linear_activate_;
    
    // Device memory for weights and biases
    float* W_d_;  // weights: (input_elements, output_elements)
    float* b_d_;  // biases: (output_elements)
    
    // Device memory for intermediate computations
    float* z_d_;  // linear output before activation: W * x + b
    float* zero_d_; // zero tensor for ReLU operation (max(x, 0))
    size_t zero_size_; // size of allocated zero tensor
    
    // Batch ReLU operations
    void apply_relu_batch_cutensor(cutensorHandle_t &handle, float* input_batch, float* output_batch, int batch_size, int size);
    void apply_relu_derivative_batch(const float* z_batch, const float* dy_batch, float* dz_batch, int batch_size, int size);
};

//=============================================================================
// cuTENSOR Helper Function Declarations
//=============================================================================

// General tensor contraction wrapper using cuTENSOR
// Performs: C = alpha * A * B + beta * C
// Users specify tensor extents and modes (labels) for each tensor
// Contracted dimensions should have the same mode labels
// Example: Matrix multiplication C[i,j] = sum_k A[i,k] * B[k,j]
//          extentA = {m, k}, extentB = {k, n}, extentC = {m, n}
//          modeA = {'i', 'k'}, modeB = {'k', 'j'}, modeC = {'i', 'j'}
void cutensor_contraction_wrapper(cutensorHandle_t &cutensor_handle,
                                 const float* A, const float* B, float* C,
                                 const std::vector<int64_t>& extentA,
                                 const std::vector<int64_t>& extentB,
                                 const std::vector<int64_t>& extentC,
                                 const std::vector<int32_t>& modeA,
                                 const std::vector<int32_t>& modeB,
                                 const std::vector<int32_t>& modeC,
                                 cudaStream_t stream = 0);

// General tensor reduction wrapper using cuTENSOR
// Performs: D = alpha * reduce_op(A) + beta * D
// Reduces tensor A along specified modes to produce tensor D
// Example: Sum reduction D[i,j] = sum_k A[i,j,k]
//          extentA = {m, n, k}, extentD = {m, n}
//          modeA = {'i', 'j', 'k'}, modeD = {'i', 'j'}
//          opReduce = CUTENSOR_OP_ADD
void cutensor_reduction_wrapper(cutensorHandle_t &cutensor_handle,
                               const float* A, float* D,
                               const std::vector<int64_t>& extentA,
                               const std::vector<int64_t>& extentD,
                               const std::vector<int32_t>& modeA,
                               const std::vector<int32_t>& modeD,
                               cutensorOperator_t opReduce,
                               float alpha = 1.0f,
                               float beta = 0.0f,
                               cudaStream_t stream = 0);

// General element-wise trinary operation wrapper using cuTENSOR
// Performs: D = op_ABC(op_AB(alpha * op_A(A), beta * op_B(B)), gamma * op_C(C))
// Supports broadcasting: modes can be omitted from A, B, or C for broadcasting
// This is more flexible than binary for complex operations
void cutensor_elementwise_trinary_wrapper(cutensorHandle_t &cutensor_handle,
                                         const float* A, const float* B, const float* C, float* D,
                                         const std::vector<int64_t>& extentA,
                                         const std::vector<int64_t>& extentB,
                                         const std::vector<int64_t>& extentC,
                                         const std::vector<int64_t>& extentD,
                                         const std::vector<int32_t>& modeA,
                                         const std::vector<int32_t>& modeB,
                                         const std::vector<int32_t>& modeC,
                                         const std::vector<int32_t>& modeD,
                                         cutensorOperator_t opA,
                                         cutensorOperator_t opB,
                                         cutensorOperator_t opC,
                                         cutensorOperator_t opAB,
                                         cutensorOperator_t opABC,
                                         float alpha = 1.0f,
                                         float beta = 1.0f,
                                         float gamma = 1.0f,
                                         cudaStream_t stream = 0);

class NeuralNetwork {
public:
    // Static function to set random seed for all weight initialization
    static void set_random_seed(unsigned int seed);
    
    // Static function to get a time-based random seed
    static unsigned int get_time_based_seed();
    
    // Constructor: takes a vector of layer sizes (e.g., {784, 128, 64, 10})
    // All layers except the last will have ReLU activation
    // The last layer will have no activation (for softmax + cross-entropy)
    NeuralNetwork(const std::vector<int>& layer_sizes);
    
    // Destructor
    ~NeuralNetwork();
    
    // Batch forward pass: computes average loss over batch
    // input_batch: (batch_size, input_size)
    // target_batch: (batch_size, num_classes)
    // Returns: average cross-entropy loss over batch
    float forward_batch(const float* input_batch, const float* target_batch, int batch_size);
    
    // Batch backward pass: computes gradients and updates all layer parameters
    // input_batch: same input batch used in forward pass
    // target_batch: same target batch used in forward pass
    // learning_rate: learning rate for parameter updates
    // max_gradient_norm: maximum allowed gradient norm (default 5.0 for stability)
    // weight_decay: L2 regularization coefficient (default 0.0)
    void backward_batch(const float* input_batch, const float* target_batch, 
                       float learning_rate, int batch_size,
                       float max_gradient_norm = 5.0f, float weight_decay = 0.0f);
    
    // Get predictions for batch (indices of maximum outputs)
    // predictions: output array of size batch_size
    void predict_batch(const float* input_batch, int* predictions, int batch_size);
    
    // Get number of layers
    int get_num_layers() const { return layers_.size(); }

   // Get layer at index
    const NeuralLayer* get_layer(int index) const { return layers_[index]; }

private:
    std::vector<NeuralLayer*> layers_;
    std::vector<float*> layer_outputs_;  // Device memory for intermediate outputs
    std::vector<float*> layer_gradients_; // Device memory for gradients
};

#endif // NN_UTILS_H
