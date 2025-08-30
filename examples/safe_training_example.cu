// Example demonstrating safe neural network training with gradient clipping
#include "../lib/nn_utils.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

// Helper function to generate synthetic data
void generate_synthetic_data(std::vector<float>& inputs, std::vector<float>& targets,
                            int batch_size, int input_size, int num_classes) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> input_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> class_dist(0, num_classes - 1);
    
    // Generate random inputs
    for (int i = 0; i < batch_size * input_size; i++) {
        inputs[i] = input_dist(gen);
    }
    
    // Generate one-hot targets
    for (int i = 0; i < batch_size; i++) {
        int target_class = class_dist(gen);
        for (int j = 0; j < num_classes; j++) {
            targets[i * num_classes + j] = (j == target_class) ? 1.0f : 0.0f;
        }
    }
}

int main() {
    // Network architecture
    const int input_size = 784;      // e.g., MNIST flattened
    const int hidden1_size = 256;
    const int hidden2_size = 128;
    const int num_classes = 10;
    const int batch_size = 64;
    
    // Training hyperparameters
    const float learning_rate = 0.01f;
    const float max_gradient_norm = 5.0f;  // Gradient clipping threshold
    const float weight_decay = 0.0001f;    // L2 regularization
    const int epochs = 100;
    
    std::cout << "Creating neural network with gradient clipping..." << std::endl;
    std::cout << "Architecture: " << input_size << " -> " << hidden1_size 
              << " -> " << hidden2_size << " -> " << num_classes << std::endl;
    
    // Create network
    std::vector<int> layer_sizes = {input_size, hidden1_size, hidden2_size, num_classes};
    NeuralNetwork network(layer_sizes);
    
    // Generate synthetic training data
    std::vector<float> h_inputs(batch_size * input_size);
    std::vector<float> h_targets(batch_size * num_classes);
    generate_synthetic_data(h_inputs, h_targets, batch_size, input_size, num_classes);
    
    // Allocate device memory
    float* d_inputs;
    float* d_targets;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_inputs, h_inputs.size() * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&d_targets, h_targets.size() * sizeof(float)));
    
    // Copy data to device
    HANDLE_CUDA_ERROR(cudaMemcpy(d_inputs, h_inputs.data(), 
                                h_inputs.size() * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_targets, h_targets.data(), 
                                h_targets.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    std::cout << "\nTraining with gradient clipping (max norm: " << max_gradient_norm 
              << ") and weight decay (" << weight_decay << ")..." << std::endl;
    
    // Training loop
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward pass to get loss
        float loss = network.forward_batch(d_inputs, d_targets, batch_size);
        
        // Safe backward pass with gradient clipping
        network.backward_batch(d_inputs, d_targets, learning_rate, batch_size,
                              max_gradient_norm, weight_decay);
        
        // Print progress
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss;
            
            // Check if loss is finite
            if (std::isfinite(loss)) {
                std::cout << " (stable)" << std::endl;
            } else {
                std::cout << " (NaN/Inf detected!)" << std::endl;
                break;
            }
        }
        
        // Generate new batch (in real training, you'd iterate through your dataset)
        if ((epoch + 1) % 10 == 0) {
            generate_synthetic_data(h_inputs, h_targets, batch_size, input_size, num_classes);
            HANDLE_CUDA_ERROR(cudaMemcpy(d_inputs, h_inputs.data(), 
                                        h_inputs.size() * sizeof(float), cudaMemcpyHostToDevice));
            HANDLE_CUDA_ERROR(cudaMemcpy(d_targets, h_targets.data(), 
                                        h_targets.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
    }
    
    // Test predictions
    int* d_predictions;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_predictions, batch_size * sizeof(int)));
    
    network.predict_batch(d_inputs, d_predictions, batch_size);
    
    std::vector<int> h_predictions(batch_size);
    HANDLE_CUDA_ERROR(cudaMemcpy(h_predictions.data(), d_predictions, 
                                batch_size * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Calculate accuracy on current batch
    int correct = 0;
    for (int i = 0; i < batch_size; i++) {
        int true_class = 0;
        for (int j = 0; j < num_classes; j++) {
            if (h_targets[i * num_classes + j] == 1.0f) {
                true_class = j;
                break;
            }
        }
        if (h_predictions[i] == true_class) {
            correct++;
        }
    }
    
    float accuracy = (float)correct / batch_size * 100.0f;
    std::cout << "\nFinal batch accuracy: " << accuracy << "%" << std::endl;
    
    // Cleanup
    cudaFree(d_inputs);
    cudaFree(d_targets);
    cudaFree(d_predictions);
    
    std::cout << "\nTraining completed successfully with gradient stability!" << std::endl;
    
    return 0;
}
