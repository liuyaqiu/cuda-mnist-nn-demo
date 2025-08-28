#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include "nn_utils.h"

int main() {
    printf("Testing NeuralLayer implementation with cuTensor\n");
    
    // Set random seed for reproducible results
    NeuralLayer::set_random_seed(42);
    
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
    
    // Test parameter update with learning rate
    float learning_rate = 0.01f;
    layer.update_parameters(dW_d, db_d, learning_rate);
    
    printf("Parameters updated successfully with learning rate %.3f!\n", learning_rate);
    
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
    
    // Test NeuralLayer batch processing
    printf("\n==================================================\n");
    printf("Testing NeuralLayer BATCH implementation\n");
    printf("==================================================\n");
    
    const int batch_size = 3;
    printf("Testing batch processing with batch_size = %d\n", batch_size);
    
    // Prepare batch input data
    std::vector<float> input_batch_h = {
        1.0f, 2.0f, 3.0f, 4.0f,  // Sample 1
        0.5f, 1.5f, 2.5f, 3.5f,  // Sample 2
        -1.0f, 0.0f, 1.0f, 2.0f  // Sample 3
    };
    std::vector<float> output_batch_h(batch_size * output_size);
    std::vector<float> dy_batch_h(batch_size * output_size, 1.0f); // All gradients = 1.0
    
    // Allocate device memory for batch test
    float *input_batch_d, *output_batch_d, *dy_batch_d, *dW_batch_d, *db_batch_d, *dx_batch_d;
    HANDLE_CUDA_ERROR(cudaMalloc(&input_batch_d, batch_size * input_size * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&output_batch_d, batch_size * output_size * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&dy_batch_d, batch_size * output_size * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&dW_batch_d, input_size * output_size * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&db_batch_d, output_size * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&dx_batch_d, batch_size * input_size * sizeof(float)));
    
    // Copy batch input to device
    HANDLE_CUDA_ERROR(cudaMemcpy(input_batch_d, input_batch_h.data(), batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(dy_batch_d, dy_batch_h.data(), batch_size * output_size * sizeof(float), cudaMemcpyHostToDevice));
    
    printf("\nTesting batch forward pass...\n");
    
    // Test batch forward pass
    layer.forward_batch(input_batch_d, output_batch_d, batch_size);
    
    // Copy batch output back to host
    HANDLE_CUDA_ERROR(cudaMemcpy(output_batch_h.data(), output_batch_d, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Display batch results
    for (int b = 0; b < batch_size; ++b) {
        printf("Batch %d - Input: [", b + 1);
        for (int i = 0; i < input_size; i++) {
            printf("%.2f", input_batch_h[b * input_size + i]);
            if (i < input_size - 1) printf(", ");
        }
        printf("] -> Output: [");
        for (int i = 0; i < output_size; i++) {
            printf("%.2f", output_batch_h[b * output_size + i]);
            if (i < output_size - 1) printf(", ");
        }
        printf("]\n");
    }
    
    printf("\nTesting batch backward pass...\n");
    
    // Test batch backward pass
    layer.backward_batch(input_batch_d, dy_batch_d, dW_batch_d, db_batch_d, dx_batch_d, batch_size);
    
    // Copy batch gradients back to host for inspection
    std::vector<float> dW_batch_h(input_size * output_size);
    std::vector<float> db_batch_h(output_size);
    std::vector<float> dx_batch_h(batch_size * input_size);
    
    HANDLE_CUDA_ERROR(cudaMemcpy(dW_batch_h.data(), dW_batch_d, input_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_CUDA_ERROR(cudaMemcpy(db_batch_h.data(), db_batch_d, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_CUDA_ERROR(cudaMemcpy(dx_batch_h.data(), dx_batch_d, batch_size * input_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Batch weight gradients (averaged across batch):\n");
    for (int i = 0; i < input_size; i++) {
        printf("  [");
        for (int j = 0; j < output_size; j++) {
            printf("%.3f", dW_batch_h[i * output_size + j]);
            if (j < output_size - 1) printf(", ");
        }
        printf("]\n");
    }
    
    printf("Batch bias gradients (summed across batch): [");
    for (int i = 0; i < output_size; i++) {
        printf("%.3f", db_batch_h[i]);
        if (i < output_size - 1) printf(", ");
    }
    printf("]\n");
    
    printf("Batch input gradients:\n");
    for (int b = 0; b < batch_size; ++b) {
        printf("  Batch %d: [", b + 1);
        for (int i = 0; i < input_size; i++) {
            printf("%.3f", dx_batch_h[b * input_size + i]);
            if (i < input_size - 1) printf(", ");
        }
        printf("]\n");
    }
    
    printf("\nTesting batch parameter update...\n");
    
    // Test batch parameter update
    layer.update_parameters(dW_batch_d, db_batch_d, learning_rate);
    
    printf("Batch parameters updated successfully!\n");
    
    // Test batch forward pass again to see if results changed
    layer.forward_batch(input_batch_d, output_batch_d, batch_size);
    HANDLE_CUDA_ERROR(cudaMemcpy(output_batch_h.data(), output_batch_d, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("Batch outputs after parameter update:\n");
    for (int b = 0; b < batch_size; ++b) {
        printf("  Batch %d: [", b + 1);
        for (int i = 0; i < output_size; i++) {
            printf("%.2f", output_batch_h[b * output_size + i]);
            if (i < output_size - 1) printf(", ");
        }
        printf("]\n");
    }
    
    // Cleanup batch memory
    cudaFree(input_batch_d);
    cudaFree(output_batch_d);
    cudaFree(dy_batch_d);
    cudaFree(dW_batch_d);
    cudaFree(db_batch_d);
    cudaFree(dx_batch_d);
    
    printf("\nNeuralLayer BATCH test completed successfully!\n");
    
    // Test NeuralNetwork class
    printf("\n==================================================\n");
    printf("Testing NeuralNetwork implementation\n");
    printf("==================================================\n");
    
    try {
        // Option 1: Set fixed seed for reproducible results
        NeuralNetwork::set_random_seed(NeuralNetwork::get_time_based_seed());
        
        // Option 2: Uncomment below for time-based random seed
        // unsigned int time_seed = NeuralNetwork::get_time_based_seed();
        // NeuralNetwork::set_random_seed(time_seed);
        
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
        float training_learning_rate = 0.01f;
        printf("Using learning rate: %.3f\n", training_learning_rate);
        for (int epoch = 0; epoch < 5; ++epoch) {
            float current_loss = network.forward(input_d, target_d);
            network.backward(input_d, target_d, training_learning_rate);
            printf("Epoch %d: loss = %f\n", epoch + 1, current_loss);
        }
        
        // Test final prediction
        int final_prediction = network.predict(input_d);
        printf("Final prediction: class %d\n", final_prediction);
        
        // Cleanup
        HANDLE_CUDA_ERROR(cudaFree(input_d));
        HANDLE_CUDA_ERROR(cudaFree(target_d));
        
        printf("\nNeuralNetwork test completed successfully!\n");
        
        // Test NeuralNetwork batch processing
        printf("\n==================================================\n");
        printf("Testing NeuralNetwork BATCH implementation\n");
        printf("==================================================\n");
        
        const int network_batch_size = 4;
        printf("Testing network batch processing with batch_size = %d\n", network_batch_size);
        
        // Create batch test data - 4 samples with 4 inputs each
        std::vector<float> input_batch_host = {
            0.5f, -0.2f, 1.0f, 0.0f,   // Sample 1
            1.0f,  0.5f, 0.0f, -0.5f,  // Sample 2
            -0.5f, 1.0f, 0.5f, 1.5f,   // Sample 3
            0.0f,  0.0f, 1.0f, 1.0f    // Sample 4
        };
        
        std::vector<float> target_batch_host = {
            0.0f, 1.0f, 0.0f,  // Sample 1: class 1
            1.0f, 0.0f, 0.0f,  // Sample 2: class 0
            0.0f, 0.0f, 1.0f,  // Sample 3: class 2
            0.0f, 1.0f, 0.0f   // Sample 4: class 1
        };
        
        // Allocate device memory for batch
        float *input_batch_d, *target_batch_d;
        HANDLE_CUDA_ERROR(cudaMalloc(&input_batch_d, network_batch_size * 4 * sizeof(float)));
        HANDLE_CUDA_ERROR(cudaMalloc(&target_batch_d, network_batch_size * 3 * sizeof(float)));
        
        HANDLE_CUDA_ERROR(cudaMemcpy(input_batch_d, input_batch_host.data(), network_batch_size * 4 * sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_CUDA_ERROR(cudaMemcpy(target_batch_d, target_batch_host.data(), network_batch_size * 3 * sizeof(float), cudaMemcpyHostToDevice));
        
        // Test batch forward pass
        printf("\nTesting batch forward pass (loss computation)...\n");
        float batch_loss = network.forward_batch(input_batch_d, target_batch_d, network_batch_size);
        printf("Batch loss (average): %f\n", batch_loss);
        
        // Test batch prediction
        printf("\nTesting batch prediction...\n");
        std::vector<int> batch_predictions(network_batch_size);
        network.predict_batch(input_batch_d, batch_predictions.data(), network_batch_size);
        
        printf("Batch predictions:\n");
        for (int i = 0; i < network_batch_size; ++i) {
            printf("  Sample %d: predicted class %d (target class ", i + 1, batch_predictions[i]);
            // Find target class
            for (int j = 0; j < 3; ++j) {
                if (target_batch_host[i * 3 + j] == 1.0f) {
                    printf("%d)\n", j);
                    break;
                }
            }
        }
        
        // Test batch training for a few iterations
        printf("\nTesting batch training loop...\n");
        float batch_learning_rate = 0.01f;
        printf("Using learning rate: %.3f\n", batch_learning_rate);
        for (int epoch = 0; epoch < 5; ++epoch) {
            float current_batch_loss = network.forward_batch(input_batch_d, target_batch_d, network_batch_size);
            network.backward_batch(input_batch_d, target_batch_d, batch_learning_rate, network_batch_size);
            printf("Batch Epoch %d: loss = %f\n", epoch + 1, current_batch_loss);
        }
        
        // Test final batch predictions
        printf("\nFinal batch predictions after training:\n");
        network.predict_batch(input_batch_d, batch_predictions.data(), network_batch_size);
        for (int i = 0; i < network_batch_size; ++i) {
            printf("  Sample %d: predicted class %d (target class ", i + 1, batch_predictions[i]);
            // Find target class
            for (int j = 0; j < 3; ++j) {
                if (target_batch_host[i * 3 + j] == 1.0f) {
                    printf("%d)\n", j);
                    break;
                }
            }
        }
        
        // Compare single vs batch processing
        printf("\n==================================================\n");
        printf("Comparing Single vs Batch Processing Performance\n");
        printf("==================================================\n");
        
        // Test single sample processing time
        printf("\nTesting single sample processing...\n");
        auto start_single = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < network_batch_size; ++i) {
            float* single_input = input_batch_d + i * 4;
            float* single_target = target_batch_d + i * 3;
            network.forward(single_input, single_target);
        }
        auto end_single = std::chrono::high_resolution_clock::now();
        auto duration_single = std::chrono::duration_cast<std::chrono::microseconds>(end_single - start_single);
        
        // Test batch processing time
        printf("Testing batch processing...\n");
        auto start_batch = std::chrono::high_resolution_clock::now();
        network.forward_batch(input_batch_d, target_batch_d, network_batch_size);
        auto end_batch = std::chrono::high_resolution_clock::now();
        auto duration_batch = std::chrono::duration_cast<std::chrono::microseconds>(end_batch - start_batch);
        
        printf("Single processing time: %ld microseconds\n", duration_single.count());
        printf("Batch processing time: %ld microseconds\n", duration_batch.count());
        printf("Speedup ratio: %.2fx\n", (float)duration_single.count() / duration_batch.count());
        
        // Cleanup batch memory
        HANDLE_CUDA_ERROR(cudaFree(input_batch_d));
        HANDLE_CUDA_ERROR(cudaFree(target_batch_d));
        
        printf("\nNeuralNetwork BATCH test completed successfully!\n");
        
    } catch (const std::exception& e) {
        printf("NeuralNetwork test failed with exception: %s\n", e.what());
        return -1;
    } catch (...) {
        printf("NeuralNetwork test failed with unknown exception\n");
        return -1;
    }
    
    return 0;
}
