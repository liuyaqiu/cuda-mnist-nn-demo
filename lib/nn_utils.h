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
    { printf("cuTENSOR Error: %s\n", cutensorGetErrorString(err)); exit(-1); } \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
    if( err != cudaSuccess )                                        \
    { printf("CUDA Error: %s\n", cudaGetErrorString(err)); exit(-1); } \
};

class NeuralLayer {
public:
    // Constructor
    NeuralLayer(int input_elements, int output_elements, bool non_linear_activate);
    
    // Destructor
    ~NeuralLayer();
    
    // Forward pass: y = ReLU(W * x + b) or y = W * x + b
    void forward(const float* input_vec, float* output_vec);
    
    // Backward pass: compute gradients dW, db, dx given dy
    void backward(const float* input_vec, const float* dy, float* dW, float* db, float* dx);
    
    // Update parameters: W = W + dW, b = b + db
    void update_parameters(const float* dW, const float* db);
    
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
    
    // cuTensor handle and descriptors
    cutensorHandle_t handle_;
    cutensorTensorDescriptor_t desc_W_;
    cutensorTensorDescriptor_t desc_x_;
    cutensorTensorDescriptor_t desc_b_;
    cutensorTensorDescriptor_t desc_z_;
    cutensorTensorDescriptor_t desc_y_;
    
    // cuTensor operation descriptors
    cutensorOperationDescriptor_t matmul_desc_;
    cutensorPlan_t matmul_plan_;
    
    // cuTensor elementwise operation descriptors for ReLU
    cutensorOperationDescriptor_t relu_desc_;
    cutensorPlan_t relu_plan_;
    
    // Workspace for cuTensor operations
    void* workspace_d_;
    size_t workspace_size_;
    
    // CUDA stream
    cudaStream_t stream_;
    
    // Helper methods
    void initialize_weights_and_biases();
    void setup_cutensor_descriptors();
    void setup_cutensor_operations();
    void setup_cutensor_elementwise_operations();
    void cleanup_cutensor_resources();
    void apply_relu_cutensor(float* input, float* output, int size);
    void apply_relu_derivative(const float* z, const float* dy, float* dz, int size);
};

class NeuralNetwork {
public:
    // Constructor: takes a vector of layer sizes (e.g., {784, 128, 64, 10})
    // All layers except the last will have ReLU activation
    // The last layer will have no activation (for softmax + cross-entropy)
    NeuralNetwork(const std::vector<int>& layer_sizes);
    
    // Destructor
    ~NeuralNetwork();
    
    // Forward pass: computes loss given input and target labels
    // input: input vector (flattened, e.g., 784 for MNIST)
    // target: one-hot encoded target vector (e.g., 10 for MNIST classes)
    // Returns: cross-entropy loss value
    float forward(const float* input, const float* target);
    
    // Backward pass: computes gradients and updates all layer parameters
    // input: same input vector used in forward pass
    // target: same target vector used in forward pass
    void backward(const float* input, const float* target);
    
    // Get prediction (index of maximum output)
    int predict(const float* input);
    
    // Get number of layers
    int get_num_layers() const { return layers_.size(); }
    
    // Get layer at index
    const NeuralLayer* get_layer(int index) const { return layers_[index]; }

private:
    std::vector<NeuralLayer*> layers_;
    std::vector<float*> layer_outputs_;  // Device memory for intermediate outputs
    std::vector<float*> layer_gradients_; // Device memory for gradients
    
    // Device memory for final softmax output and loss computation
    float* softmax_output_d_;
    float* loss_gradient_d_;
    float* temp_storage_d_;     // Temporary storage for intermediate computations
    float* max_values_d_;       // For storing max values in softmax
    float* sum_exp_d_;          // For storing sum of exponentials
    
    // CUDA stream
    cudaStream_t stream_;
    
    // cuTensor handle and descriptors for softmax/loss operations
    cutensorHandle_t cutensor_handle_;
    cutensorTensorDescriptor_t softmax_input_desc_;
    cutensorTensorDescriptor_t softmax_output_desc_;
    cutensorTensorDescriptor_t target_desc_;
    
    // cuTensor operation descriptors for softmax components
    cutensorOperationDescriptor_t exp_desc_;          // For exponential operation
    cutensorOperationDescriptor_t log_desc_;          // For logarithm operation
    cutensorOperationDescriptor_t sub_desc_;          // For subtraction operation
    cutensorOperationDescriptor_t div_desc_;          // For division operation
    cutensorOperationDescriptor_t mul_desc_;          // For multiplication operation
    cutensorOperationDescriptor_t reduce_max_desc_;   // For max reduction
    cutensorOperationDescriptor_t reduce_sum_desc_;   // For sum reduction
    
    // cuTensor plans
    cutensorPlan_t exp_plan_;
    cutensorPlan_t log_plan_;
    cutensorPlan_t sub_plan_;
    cutensorPlan_t div_plan_;
    cutensorPlan_t mul_plan_;
    cutensorPlan_t reduce_max_plan_;
    cutensorPlan_t reduce_sum_plan_;
    
    // Workspace for cuTensor operations
    void* cutensor_workspace_d_;
    size_t cutensor_workspace_size_;
    
    // Helper methods
    void setup_device_memory();
    void cleanup_device_memory();
    void setup_cutensor_softmax_operations();
    void cleanup_cutensor_softmax_operations();
    void apply_softmax_cutensor(const float* input, float* output, int size);
    float compute_cross_entropy_loss_cutensor(const float* softmax_output, const float* target, int size);
    void compute_cross_entropy_gradient_cutensor(const float* softmax_output, const float* target, float* gradient, int size);
};

#endif // NN_UTILS_H
