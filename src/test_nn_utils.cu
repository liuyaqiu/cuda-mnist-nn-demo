#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "nn_utils.h"

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
    if( err != cudaSuccess )                                        \
    { printf("CUDA Error: %s\n", cudaGetErrorString(err)); exit(-1); } \
};

int main() {
    printf("Testing NeuralLayer implementation with cuTensor\n");
    
    // Test parameters
    const int input_size = 4;
    const int output_size = 3;
    const bool use_relu = true;
    
    // Create neural layer
    NeuralLayer layer(input_size, output_size, use_relu);
    
    printf("Created NeuralLayer with %d inputs, %d outputs, ReLU: %s\n", 
           input_size, output_size, use_relu ? "enabled" : "disabled");
    
    // Prepare test input on host
    std::vector<float> input_h = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> output_h(output_size);
    std::vector<float> dy_h = {1.0f, 1.0f, 1.0f}; // gradient from next layer
    
    // Allocate device memory for test
    float *input_d, *output_d, *dy_d, *dW_d, *db_d, *dx_d;
    HANDLE_CUDA_ERROR(cudaMalloc(&input_d, input_size * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&output_d, output_size * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&dy_d, output_size * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&dW_d, input_size * output_size * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&db_d, output_size * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&dx_d, input_size * sizeof(float)));
    
    // Copy input to device
    HANDLE_CUDA_ERROR(cudaMemcpy(input_d, input_h.data(), input_size * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(dy_d, dy_h.data(), output_size * sizeof(float), cudaMemcpyHostToDevice));
    
    printf("\nTesting forward pass...\n");
    
    // Test forward pass
    layer.forward(input_d, output_d);
    
    printf("\nTesting forward pass...\n");
    // Copy output back to host
    HANDLE_CUDA_ERROR(cudaMemcpy(output_h.data(), output_d, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Input: [");
    for (int i = 0; i < input_size; i++) {
        printf("%.2f", input_h[i]);
        if (i < input_size - 1) printf(", ");
    }
    printf("]\n");
    
    printf("Output: [");
    for (int i = 0; i < output_size; i++) {
        printf("%.2f", output_h[i]);
        if (i < output_size - 1) printf(", ");
    }
    printf("]\n");
    
    printf("\nTesting backward pass...\n");
    
    // Test backward pass
    layer.backward(input_d, dy_d, dW_d, db_d, dx_d);
    
    // Copy gradients back to host for inspection
    std::vector<float> dW_h(input_size * output_size);
    std::vector<float> db_h(output_size);
    std::vector<float> dx_h(input_size);
    
    HANDLE_CUDA_ERROR(cudaMemcpy(dW_h.data(), dW_d, input_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_CUDA_ERROR(cudaMemcpy(db_h.data(), db_d, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_CUDA_ERROR(cudaMemcpy(dx_h.data(), dx_d, input_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Weight gradients (dW):\n");
    for (int i = 0; i < input_size; i++) {
        printf("  [");
        for (int j = 0; j < output_size; j++) {
            printf("%.3f", dW_h[i * output_size + j]);
            if (j < output_size - 1) printf(", ");
        }
        printf("]\n");
    }
    
    printf("Bias gradients (db): [");
    for (int i = 0; i < output_size; i++) {
        printf("%.3f", db_h[i]);
        if (i < output_size - 1) printf(", ");
    }
    printf("]\n");
    
    printf("Input gradients (dx): [");
    for (int i = 0; i < input_size; i++) {
        printf("%.3f", dx_h[i]);
        if (i < input_size - 1) printf(", ");
    }
    printf("]\n");
    
    printf("\nTesting parameter update...\n");
    
    // Test parameter update
    layer.update_parameters(dW_d, db_d);
    
    printf("Parameters updated successfully!\n");
    
    // Test forward pass again to see if results changed
    layer.forward(input_d, output_d);
    HANDLE_CUDA_ERROR(cudaMemcpy(output_h.data(), output_d, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Output after parameter update: [");
    for (int i = 0; i < output_size; i++) {
        printf("%.2f", output_h[i]);
        if (i < output_size - 1) printf(", ");
    }
    printf("]\n");
    
    // Cleanup
    cudaFree(input_d);
    cudaFree(output_d);
    cudaFree(dy_d);
    cudaFree(dW_d);
    cudaFree(db_d);
    cudaFree(dx_d);
    
    printf("\nNeuralLayer test completed successfully!\n");
    
    // Test NeuralNetwork class
    printf("\n==================================================\n");
    printf("Testing NeuralNetwork implementation\n");
    printf("==================================================\n");
    
    try {
        // Test network architecture: 4 input -> 6 hidden -> 3 output
        std::vector<int> layer_sizes = {4, 6, 3};
        
        printf("Creating neural network with architecture: ");
        for (size_t i = 0; i < layer_sizes.size(); ++i) {
            printf("%d", layer_sizes[i]);
            if (i < layer_sizes.size() - 1) printf(" -> ");
        }
        printf("\n");
        
        NeuralNetwork network(layer_sizes);
        
        // Create test data
        float input_host[4] = {0.5f, -0.2f, 1.0f, 0.0f};
        float target_host[3] = {0.0f, 1.0f, 0.0f}; // One-hot: class 1
        
        // Allocate device memory
        float *input_d, *target_d;
        HANDLE_CUDA_ERROR(cudaMalloc(&input_d, 4 * sizeof(float)));
        HANDLE_CUDA_ERROR(cudaMalloc(&target_d, 3 * sizeof(float)));
        
        HANDLE_CUDA_ERROR(cudaMemcpy(input_d, input_host, 4 * sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_CUDA_ERROR(cudaMemcpy(target_d, target_host, 3 * sizeof(float), cudaMemcpyHostToDevice));
        
        // Test forward pass
        printf("\nTesting forward pass (loss computation)...\n");
        float loss = network.forward(input_d, target_d);
        printf("Initial loss: %f\n", loss);
        
        // Test prediction
        printf("\nTesting prediction...\n");
        int prediction = network.predict(input_d);
        printf("Initial prediction: class %d\n", prediction);
        
        // Test training for a few iterations
        printf("\nTesting training loop...\n");
        for (int epoch = 0; epoch < 5; ++epoch) {
            float current_loss = network.forward(input_d, target_d);
            network.backward(input_d, target_d);
            printf("Epoch %d: loss = %f\n", epoch + 1, current_loss);
        }
        
        // Test final prediction
        int final_prediction = network.predict(input_d);
        printf("Final prediction: class %d\n", final_prediction);
        
        // Cleanup
        HANDLE_CUDA_ERROR(cudaFree(input_d));
        HANDLE_CUDA_ERROR(cudaFree(target_d));
        
        printf("\nNeuralNetwork test completed successfully!\n");
        
    } catch (const std::exception& e) {
        printf("NeuralNetwork test failed with exception: %s\n", e.what());
        return -1;
    } catch (...) {
        printf("NeuralNetwork test failed with unknown exception\n");
        return -1;
    }
    
    return 0;
}
