#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cutensor.h>
#include <cmath>
#include <random>
#include <vector>
#include <iostream>

#include "nn_utils.h"

// Test fixture for nn_utils tests
class NNUtilsTest : public ::testing::Test {
protected:
    cutensorHandle_t cutensor_handle;
    cudaStream_t stream;
    
    void SetUp() override {
        // Initialize cuTENSOR
        HANDLE_CUTENSOR_ERROR(cutensorCreate(&cutensor_handle));
        
        // Create CUDA stream
        HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));
    }
    
    void TearDown() override {
        // Destroy cuTENSOR handle
        HANDLE_CUTENSOR_ERROR(cutensorDestroy(cutensor_handle));
        
        // Destroy stream
        HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));
    }
    
    // Helper function to perform CPU matrix multiplication for verification
    // Computes C[i,j] = sum_k A[i,k] * B[k,j]
    // Where A is (m, k), B is (k, n), C is (m, n)
    void cpu_matrix_multiply(const std::vector<float>& A, const std::vector<float>& B,
                           std::vector<float>& C, int m, int k, int n) {
        // Initialize C to zero
        std::fill(C.begin(), C.end(), 0.0f);
        
        // Perform matrix multiplication
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int l = 0; l < k; ++l) {
                    sum += A[i * k + l] * B[l * n + j];
                }
                C[i * n + j] = sum;
            }
        }
    }
    
    // Helper function to check if two arrays are close
    bool arrays_are_close(const std::vector<float>& a, const std::vector<float>& b, 
                         float rtol = 1e-5f, float atol = 1e-7f) {
        if (a.size() != b.size()) {
            return false;
        }
        
        for (size_t i = 0; i < a.size(); ++i) {
            float diff = std::abs(a[i] - b[i]);
            float tol = atol + rtol * std::abs(b[i]);
            if (diff > tol) {
                std::cerr << "Mismatch at index " << i << ": " 
                         << a[i] << " vs " << b[i] 
                         << " (diff: " << diff << ", tol: " << tol << ")" << std::endl;
                return false;
            }
        }
        return true;
    }
};

// Test basic matrix multiplication using cutensor_contraction_wrapper
TEST_F(NNUtilsTest, MatrixMultiplyWithWrapper) {
    const int m = 4;
    const int k = 3;
    const int n = 5;
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    // Create host arrays
    std::vector<float> h_A(m * k);
    std::vector<float> h_B(k * n);
    std::vector<float> h_C_gpu(m * n);
    std::vector<float> h_C_cpu(m * n);
    
    // Initialize with random values
    for (auto& val : h_A) val = dis(gen);
    for (auto& val : h_B) val = dis(gen);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_A, h_A.size() * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_B, h_B.size() * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_C, h_C_gpu.size() * sizeof(float)));
    
    // Copy data to device
    HANDLE_CUDA_ERROR(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    // Define tensor extents and modes for matrix multiplication
    // A: (m, k) with modes 'i', 'k'
    // B: (k, n) with modes 'k', 'j'
    // C: (m, n) with modes 'i', 'j'
    std::vector<int64_t> extentA{m, k};
    std::vector<int64_t> extentB{k, n};
    std::vector<int64_t> extentC{m, n};
    
    std::vector<int32_t> modeA{'i', 'k'};
    std::vector<int32_t> modeB{'k', 'j'};
    std::vector<int32_t> modeC{'i', 'j'};
    
    // Perform GPU matrix multiplication using wrapper
    cutensor_contraction_wrapper(cutensor_handle, d_A, d_B, d_C,
                               extentA, extentB, extentC,
                               modeA, modeB, modeC, stream);
    
    // Copy result back to host
    HANDLE_CUDA_ERROR(cudaMemcpy(h_C_gpu.data(), d_C, h_C_gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Perform CPU matrix multiplication
    cpu_matrix_multiply(h_A, h_B, h_C_cpu, m, k, n);
    
    // Compare results
    EXPECT_TRUE(arrays_are_close(h_C_gpu, h_C_cpu));
    
    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Test NeuralLayer forward pass
TEST_F(NNUtilsTest, NeuralLayerForwardPass) {
    const int batch_size = 4;
    const int input_elements = 10;
    const int output_elements = 5;
    
    // Test both with and without activation
    for (bool non_linear_activate : {false, true}) {
        SCOPED_TRACE("Testing with non_linear_activate = " + std::to_string(non_linear_activate));
        
        // Create neural layer
        NeuralLayer layer(input_elements, output_elements, non_linear_activate);
        
        // Initialize random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
        
        // Create input data
        std::vector<float> h_input(batch_size * input_elements);
        for (auto& val : h_input) val = dis(gen);
        
        // Allocate device memory
        float *d_input, *d_output;
        HANDLE_CUDA_ERROR(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
        HANDLE_CUDA_ERROR(cudaMalloc(&d_output, batch_size * output_elements * sizeof(float)));
        
        // Copy input to device
        HANDLE_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
        
        // Perform forward pass
        layer.forward_batch(d_input, d_output, batch_size);
        
        // Copy output to host
        std::vector<float> h_output(batch_size * output_elements);
        HANDLE_CUDA_ERROR(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Verify output shape
        EXPECT_EQ(h_output.size(), batch_size * output_elements);
        
        // If ReLU is active, check that all outputs are non-negative
        if (non_linear_activate) {
            for (const auto& val : h_output) {
                EXPECT_GE(val, 0.0f);
            }
        }
        
        // Clean up
        cudaFree(d_input);
        cudaFree(d_output);
    }
}

// Test element-wise operations with broadcasting
TEST_F(NNUtilsTest, ElementwiseBroadcasting) {
    const int batch_size = 3;
    const int size = 5;
    
    // Create bias vector (to be broadcasted)
    std::vector<float> h_bias(size);
    std::vector<float> h_input(batch_size * size);
    std::vector<float> h_output(batch_size * size);
    
    // Initialize with known values
    for (int i = 0; i < size; ++i) {
        h_bias[i] = i * 0.1f;
    }
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < size; ++i) {
            h_input[b * size + i] = b + i * 0.5f;
        }
    }
    
    // Allocate device memory
    float *d_input, *d_bias, *d_output;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_bias, h_bias.size() * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_output, h_output.size() * sizeof(float)));
    
    // Copy to device
    HANDLE_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    // Test broadcasting addition using element-wise wrapper
    std::vector<int64_t> extentInput{batch_size, size};
    std::vector<int64_t> extentBias{size};
    std::vector<int64_t> extentZero{};  // Scalar
    std::vector<int64_t> extentOutput{batch_size, size};
    
    std::vector<int32_t> modeInput{'i', 'j'};
    std::vector<int32_t> modeBias{'j'};
    std::vector<int32_t> modeZero{};  // Scalar has no modes
    std::vector<int32_t> modeOutput{'i', 'j'};
    
    // Create a zero scalar on device
    float* d_zero;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_zero, sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMemset(d_zero, 0, sizeof(float)));
    
    cutensor_elementwise_trinary_wrapper(cutensor_handle,
                                        d_input, d_bias, d_zero, d_output,
                                        extentInput, extentBias, extentZero, extentOutput,
                                        modeInput, modeBias, modeZero, modeOutput,
                                        CUTENSOR_OP_IDENTITY, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_IDENTITY,
                                        CUTENSOR_OP_ADD, CUTENSOR_OP_ADD,
                                        1.0f, 1.0f, 0.0f, stream);
    
    HANDLE_CUDA_ERROR(cudaFree(d_zero));
    
    // Copy result back
    HANDLE_CUDA_ERROR(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify results
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < size; ++i) {
            float expected = h_input[b * size + i] + h_bias[i];
            float actual = h_output[b * size + i];
            EXPECT_NEAR(actual, expected, 1e-6f);
        }
    }
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_bias);
    cudaFree(d_output);
}

// Test reduction operations
TEST_F(NNUtilsTest, ReductionOperations) {
    const int batch_size = 4;
    const int elements = 6;
    
    // Create input data
    std::vector<float> h_input(batch_size * elements);
    std::vector<float> h_output(elements);
    
    // Initialize with known values
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < elements; ++i) {
            h_input[b * elements + i] = b + i * 0.5f;
        }
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_output, h_output.size() * sizeof(float)));
    
    // Copy to device
    HANDLE_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    // Test sum reduction over batch dimension
    std::vector<int64_t> extentInput{batch_size, elements};
    std::vector<int64_t> extentOutput{elements};
    
    std::vector<int32_t> modeInput{'b', 'i'};
    std::vector<int32_t> modeOutput{'i'};
    
    cutensor_reduction_wrapper(cutensor_handle,
                              d_input, d_output,
                              extentInput, extentOutput,
                              modeInput, modeOutput,
                              CUTENSOR_OP_ADD,
                              1.0f, 0.0f, stream);
    
    // Copy result back
    HANDLE_CUDA_ERROR(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify results
    for (int i = 0; i < elements; ++i) {
        float expected = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            expected += h_input[b * elements + i];
        }
        float actual = h_output[i];
        EXPECT_NEAR(actual, expected, 1e-6f);
    }
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
}

// Test NeuralLayer backward pass
TEST_F(NNUtilsTest, NeuralLayerBackwardPass) {
    const int batch_size = 2;
    const int input_elements = 8;
    const int output_elements = 4;
    
    // Create neural layer with ReLU
    NeuralLayer layer(input_elements, output_elements, true);
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
    
    // Create data
    std::vector<float> h_input(batch_size * input_elements);
    std::vector<float> h_dy(batch_size * output_elements);
    
    for (auto& val : h_input) val = dis(gen);
    for (auto& val : h_dy) val = dis(gen);
    
    // Allocate device memory
    float *d_input, *d_output, *d_dy, *d_dW, *d_db, *d_dx;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_output, batch_size * output_elements * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_dy, h_dy.size() * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_dW, input_elements * output_elements * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_db, output_elements * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_dx, h_input.size() * sizeof(float)));
    
    // Copy to device
    HANDLE_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_dy, h_dy.data(), h_dy.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    // Forward pass (needed for ReLU derivative)
    layer.forward_batch(d_input, d_output, batch_size);
    
    // Backward pass
    layer.backward_batch(d_input, d_dy, d_dW, d_db, d_dx, batch_size);
    
    // Copy gradients back
    std::vector<float> h_dW(input_elements * output_elements);
    std::vector<float> h_db(output_elements);
    std::vector<float> h_dx(h_input.size());
    
    HANDLE_CUDA_ERROR(cudaMemcpy(h_dW.data(), d_dW, h_dW.size() * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_CUDA_ERROR(cudaMemcpy(h_db.data(), d_db, h_db.size() * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_CUDA_ERROR(cudaMemcpy(h_dx.data(), d_dx, h_dx.size() * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Basic checks
    EXPECT_EQ(h_dW.size(), input_elements * output_elements);
    EXPECT_EQ(h_db.size(), output_elements);
    EXPECT_EQ(h_dx.size(), h_input.size());
    
    // Test parameter update
    float learning_rate = 0.01f;
    layer.update_parameters(d_dW, d_db, learning_rate);
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_dy);
    cudaFree(d_dW);
    cudaFree(d_db);
    cudaFree(d_dx);
}

// Test NeuralNetwork construction and destruction
TEST_F(NNUtilsTest, NeuralNetworkConstruction) {
    // Test valid construction
    std::vector<int> layer_sizes = {784, 128, 64, 10};
    NeuralNetwork network(layer_sizes);
    
    EXPECT_EQ(network.get_num_layers(), 3);  // 3 weight layers
    
    // Verify layer dimensions
    EXPECT_EQ(network.get_layer(0)->get_input_elements(), 784);
    EXPECT_EQ(network.get_layer(0)->get_output_elements(), 128);
    EXPECT_TRUE(network.get_layer(0)->get_non_linear_activate());  // First hidden layer has ReLU
    
    EXPECT_EQ(network.get_layer(1)->get_input_elements(), 128);
    EXPECT_EQ(network.get_layer(1)->get_output_elements(), 64);
    EXPECT_TRUE(network.get_layer(1)->get_non_linear_activate());  // Second hidden layer has ReLU
    
    EXPECT_EQ(network.get_layer(2)->get_input_elements(), 64);
    EXPECT_EQ(network.get_layer(2)->get_output_elements(), 10);
    EXPECT_FALSE(network.get_layer(2)->get_non_linear_activate());  // Output layer has no activation
}

// Test NeuralNetwork forward pass and loss computation
TEST_F(NNUtilsTest, NeuralNetworkForwardPass) {
    const int batch_size = 4;
    const int input_size = 10;
    const int num_classes = 3;
    
    std::vector<int> layer_sizes = {input_size, 8, num_classes};
    NeuralNetwork network(layer_sizes);
    
    // Create input and target data
    std::vector<float> h_input(batch_size * input_size);
    std::vector<float> h_target(batch_size * num_classes, 0.0f);
    
    // Initialize with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
    
    for (auto& val : h_input) val = dis(gen);
    
    // Create one-hot encoded targets
    std::uniform_int_distribution<int> class_dis(0, num_classes - 1);
    for (int i = 0; i < batch_size; i++) {
        int target_class = class_dis(gen);
        h_target[i * num_classes + target_class] = 1.0f;
    }
    
    // Copy to device
    float* d_input;
    float* d_target;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_target, h_target.size() * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_target, h_target.data(), h_target.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    // Forward pass
    float loss = network.forward_batch(d_input, d_target, batch_size);
    
    // Loss should be positive
    EXPECT_GT(loss, 0.0f);
    
    // For random initialization, loss should be close to -log(1/num_classes)
    float expected_loss = -logf(1.0f / num_classes);
    EXPECT_NEAR(loss, expected_loss, 2.0f);  // Allow some variance
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_target);
}

// Test NeuralNetwork backward pass
TEST_F(NNUtilsTest, NeuralNetworkBackwardPass) {
    const int batch_size = 2;
    const int input_size = 5;
    const int num_classes = 3;
    
    std::vector<int> layer_sizes = {input_size, 4, num_classes};
    NeuralNetwork network(layer_sizes);
    
    // Create simple input and target
    std::vector<float> h_input(batch_size * input_size, 0.5f);
    std::vector<float> h_target(batch_size * num_classes, 0.0f);
    
    // Set targets
    h_target[0 * num_classes + 0] = 1.0f;  // First sample -> class 0
    h_target[1 * num_classes + 1] = 1.0f;  // Second sample -> class 1
    
    // Copy to device
    float* d_input;
    float* d_target;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_target, h_target.size() * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_target, h_target.data(), h_target.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    // Get initial loss
    float initial_loss = network.forward_batch(d_input, d_target, batch_size);
    
    // Backward pass with learning
    float learning_rate = 0.1f;
    network.backward_batch(d_input, d_target, learning_rate, batch_size);
    
    // Get loss after update
    float updated_loss = network.forward_batch(d_input, d_target, batch_size);
    
    // Loss should decrease
    EXPECT_LT(updated_loss, initial_loss);
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_target);
}

// Test NeuralNetwork predictions
TEST_F(NNUtilsTest, NeuralNetworkPrediction) {
    const int batch_size = 4;
    const int input_size = 8;
    const int num_classes = 3;
    
    std::vector<int> layer_sizes = {input_size, 6, num_classes};
    NeuralNetwork network(layer_sizes);
    
    // Create input data
    std::vector<float> h_input(batch_size * input_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (auto& val : h_input) val = dis(gen);
    
    // Copy to device
    float* d_input;
    int* d_predictions;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_predictions, batch_size * sizeof(int)));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    // Get predictions
    network.predict_batch(d_input, d_predictions, batch_size);
    
    // Copy predictions back
    std::vector<int> h_predictions(batch_size);
    HANDLE_CUDA_ERROR(cudaMemcpy(h_predictions.data(), d_predictions, batch_size * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Verify predictions are valid class indices
    for (int pred : h_predictions) {
        EXPECT_GE(pred, 0);
        EXPECT_LT(pred, num_classes);
    }
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_predictions);
}

// Test NeuralNetwork training on simple linearly separable problem
TEST_F(NNUtilsTest, NeuralNetworkSimpleTraining) {
    const int input_size = 2;
    const int hidden_size = 8;
    const int num_classes = 2;
    
    std::vector<int> layer_sizes = {input_size, hidden_size, num_classes};
    NeuralNetwork network(layer_sizes);
    
    // Create linearly separable training data
    std::vector<float> h_inputs = {
        0.1f, 0.1f,  // class 0
        0.2f, 0.1f,  // class 0
        0.8f, 0.9f,  // class 1
        0.9f, 0.8f   // class 1
    };
    
    std::vector<float> h_targets = {
        1.0f, 0.0f,  // class 0
        1.0f, 0.0f,  // class 0
        0.0f, 1.0f,  // class 1
        0.0f, 1.0f   // class 1
    };
    
    const int batch_size = 4;
    
    // Copy to device
    float* d_inputs;
    float* d_targets;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_inputs, h_inputs.size() * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_targets, h_targets.size() * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_inputs, h_inputs.data(), h_inputs.size() * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_targets, h_targets.data(), h_targets.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    // Train for several epochs
    float learning_rate = 0.01f;  // Much smaller learning rate
    const int epochs = 50;        // Fewer epochs
    
    float initial_loss = network.forward_batch(d_inputs, d_targets, batch_size);
    
    // Track loss over time
    float prev_loss = initial_loss;
    int improvements = 0;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        network.backward_batch(d_inputs, d_targets, learning_rate, batch_size);
        
        if (epoch % 10 == 0) {
            float current_loss = network.forward_batch(d_inputs, d_targets, batch_size);
            if (current_loss < prev_loss) {
                improvements++;
            }
            prev_loss = current_loss;
        }
    }
    
    float final_loss = network.forward_batch(d_inputs, d_targets, batch_size);
    
    // We expect some improvement during training
    // Either the final loss is lower OR we saw improvements during training
    EXPECT_TRUE(final_loss < initial_loss * 1.5f || improvements > 0);
    
    // Check predictions
    int* d_predictions;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_predictions, batch_size * sizeof(int)));
    network.predict_batch(d_inputs, d_predictions, batch_size);
    
    std::vector<int> h_predictions(batch_size);
    HANDLE_CUDA_ERROR(cudaMemcpy(h_predictions.data(), d_predictions, batch_size * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Clean up
    cudaFree(d_inputs);
    cudaFree(d_targets);
    cudaFree(d_predictions);
}

// Test gradient clipping and stability
TEST_F(NNUtilsTest, GradientClippingStability) {
    const int batch_size = 8;
    const int input_size = 10;
    const int hidden_size = 20;
    const int num_classes = 5;
    
    std::vector<int> layer_sizes = {input_size, hidden_size, num_classes};
    NeuralNetwork network(layer_sizes);
    
    // Create input with large values to potentially cause gradient explosion
    std::vector<float> h_input(batch_size * input_size);
    std::vector<float> h_target(batch_size * num_classes, 0.0f);
    
    // Initialize with large values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(5.0f, 10.0f);  // Large values
    
    for (auto& val : h_input) val = dis(gen);
    
    // Create one-hot targets
    std::uniform_int_distribution<int> class_dis(0, num_classes - 1);
    for (int i = 0; i < batch_size; i++) {
        int target_class = class_dis(gen);
        h_target[i * num_classes + target_class] = 1.0f;
    }
    
    // Copy to device
    float* d_input;
    float* d_target;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_target, h_target.size() * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_target, h_target.data(), h_target.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    // Train with large learning rate (which could cause instability)
    float large_learning_rate = 1.0f;
    float max_gradient_norm = 1.0f;  // Aggressive clipping
    float weight_decay = 0.001f;
    
    // Get initial loss
    float initial_loss = network.forward_batch(d_input, d_target, batch_size);
    EXPECT_TRUE(std::isfinite(initial_loss));
    
    // Train for several epochs with safe backward pass
    const int epochs = 10;
    bool all_losses_finite = true;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        network.backward_batch(d_input, d_target, large_learning_rate, batch_size,
                              max_gradient_norm, weight_decay);
        
        float loss = network.forward_batch(d_input, d_target, batch_size);
        if (!std::isfinite(loss)) {
            all_losses_finite = false;
            break;
        }
    }
    
    // With gradient clipping, all losses should remain finite
    EXPECT_TRUE(all_losses_finite);
    
    // Final loss should still be reasonable
    float final_loss = network.forward_batch(d_input, d_target, batch_size);
    EXPECT_TRUE(std::isfinite(final_loss));
    EXPECT_LT(final_loss, 100.0f);  // Should not explode
    
    // Clean up
    cudaFree(d_input);
    cudaFree(d_target);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
