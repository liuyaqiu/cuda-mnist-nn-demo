#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "model.h"
#include "nn_utils.h"

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " code=" << error << " \"" << cudaGetErrorString(error) << "\"" << std::endl; \
        exit(1); \
    } \
} while(0)

// Structure to hold performance metrics
struct PerformanceMetrics {
    double total_flops;
    double training_time_seconds;
    double tflops;
    size_t total_memory_bytes;
    int total_iterations;
    
    PerformanceMetrics() : total_flops(0), training_time_seconds(0), 
                          tflops(0), total_memory_bytes(0), total_iterations(0) {}
};

// Helper function to parse command line arguments
void parseArguments(int argc, char** argv, int& num_epochs, int& batch_size, 
                   float& learning_rate, float& max_gradient_norm, float& weight_decay,
                   int& random_seed) {
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) {
            std::cerr << "Missing value for argument: " << argv[i] << std::endl;
            exit(1);
        }
        
        std::string arg = argv[i];
        if (arg == "--epochs" || arg == "-e") {
            num_epochs = std::atoi(argv[i + 1]);
        } else if (arg == "--batch_size" || arg == "-b") {
            batch_size = std::atoi(argv[i + 1]);
        } else if (arg == "--learning_rate" || arg == "-lr") {
            learning_rate = std::atof(argv[i + 1]);
        } else if (arg == "--max_gradient_norm" || arg == "-g") {
            max_gradient_norm = std::atof(argv[i + 1]);
        } else if (arg == "--weight_decay" || arg == "-wd") {
            weight_decay = std::atof(argv[i + 1]);
        } else if (arg == "--random_seed" || arg == "-s") {
            random_seed = std::atoi(argv[i + 1]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cerr << "Options:" << std::endl;
            std::cerr << "  --epochs, -e <int>              Number of epochs (default: 5)" << std::endl;
            std::cerr << "  --batch_size, -b <int>          Batch size (default: 100)" << std::endl;
            std::cerr << "  --learning_rate, -lr <float>    Learning rate (default: 0.001)" << std::endl;
            std::cerr << "  --max_gradient_norm, -g <float> Max gradient norm (default: 5.0)" << std::endl;
            std::cerr << "  --weight_decay, -wd <float>     Weight decay (default: 0.0)" << std::endl;
            std::cerr << "  --random_seed, -s <int>         Random seed for reproducibility (default: 42)" << std::endl;
            exit(1);
        }
    }
}

// Function to print model architecture details
void printModelArchitecture(const std::vector<int>& architecture) {
    std::cout << "\n=== Model Architecture ===" << std::endl;
    std::cout << "Layer Structure:" << std::endl;
    
    int total_params = 0;
    for (size_t i = 0; i < architecture.size() - 1; ++i) {
        int layer_params = architecture[i] * architecture[i + 1] + architecture[i + 1]; // weights + biases
        total_params += layer_params;
        
        std::cout << "  Layer " << i + 1 << ": " 
                  << architecture[i] << " -> " << architecture[i + 1] 
                  << " (params: " << layer_params << ")" << std::endl;
    }
    
    std::cout << "\nTotal Parameters: " << total_params << std::endl;
    std::cout << "Memory Required: " << std::fixed << std::setprecision(2) 
              << (total_params * sizeof(float)) / (1024.0 * 1024.0) << " MB" << std::endl;
}

// Function to calculate FLOPS for neural network operations
double calculateFlops(const std::vector<int>& architecture, int batch_size, int num_iterations) {
    double flops = 0;
    
    // Forward pass FLOPS
    for (size_t i = 0; i < architecture.size() - 1; ++i) {
        // Matrix multiplication: 2 * M * N * K - M * N
        // For batch_size x input_size * input_size x output_size
        double layer_flops = 2.0 * batch_size * architecture[i] * architecture[i + 1] - 
                            batch_size * architecture[i + 1];
        
        // Add activation function FLOPS (ReLU: 1 FLOP per element)
        if (i < architecture.size() - 2) { // Not the last layer
            layer_flops += batch_size * architecture[i + 1];
        }
        
        flops += layer_flops;
    }
    
    // Backward pass is approximately 2x forward pass
    flops *= 3;
    
    // Multiply by number of iterations
    flops *= num_iterations;
    
    return flops;
}

// Function to print detailed per-class metrics
void printDetailedMetrics(const EvaluationResults& results, int num_classes) {
    std::cout << "\n=== Detailed Test Metrics ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Overall Test Accuracy: " << results.accuracy * 100 << "%" << std::endl;
    std::cout << "Test Loss: " << results.loss << std::endl;
    std::cout << std::defaultfloat;
    std::cout << "Correct Predictions: " << results.correct_predictions 
              << "/" << results.total_samples << std::endl;
    
    std::cout << "\nPer-Class Accuracy:" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    for (int i = 0; i < num_classes; ++i) {
        std::cout << "  Class " << i << ": " << results.per_class_accuracy[i] * 100 << "%" << std::endl;
    }
    std::cout << std::defaultfloat;
    
    // Print confusion matrix
    std::cout << "\nConfusion Matrix:" << std::endl;
    std::cout << "     ";
    for (int i = 0; i < num_classes; ++i) {
        std::cout << std::setw(6) << "P" << i;
    }
    std::cout << std::endl;
    
    for (int i = 0; i < num_classes; ++i) {
        std::cout << "  T" << i << " ";
        for (int j = 0; j < num_classes; ++j) {
            std::cout << std::setw(6) << results.confusion_matrix[i * num_classes + j];
        }
        std::cout << std::endl;
    }
    std::cout << "(T = True label, P = Predicted label)" << std::endl;
}

// Function to get GPU memory usage
void printGPUMemoryInfo() {
    size_t free_memory, total_memory;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
    
    std::cout << "\n=== GPU Memory Usage ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total GPU Memory: " << total_memory / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Free GPU Memory: " << free_memory / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Used GPU Memory: " << (total_memory - free_memory) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << std::defaultfloat;
}

// Function to print CUDA device properties
void printCUDADeviceInfo() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    std::cout << "\n=== CUDA Device Information ===" << std::endl;
    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Memory Clock Rate: " << prop.memoryClockRate / 1000.0 << " MHz" << std::endl;
    std::cout << std::defaultfloat;
    std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Peak Memory Bandwidth: " << 
              2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << " GB/s" << std::endl;
    std::cout << std::defaultfloat;
}

int main(int argc, char** argv) {
    // Default parameters
    int num_epochs = 5;
    int batch_size = 100;
    float learning_rate = 0.001f;
    float max_gradient_norm = 5.0f;
    float weight_decay = 0.0f;
    int random_seed = 42;
    
    // Parse command line arguments
    parseArguments(argc, argv, num_epochs, batch_size, learning_rate, max_gradient_norm, weight_decay, random_seed);
    
    // Set random seed for reproducibility
    srand(random_seed);
    CUDA_CHECK(cudaSetDevice(0));
    
    // Set random seed for neural network weight initialization
    NeuralNetwork::set_random_seed(random_seed);
    
    // Print configuration
    std::cout << "=== CUDA MNIST Neural Network Demo ===" << std::endl;
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Epochs: " << num_epochs << std::endl;
    std::cout << "  Batch Size: " << batch_size << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Learning Rate: " << learning_rate << std::endl;
    std::cout << "  Max Gradient Norm: " << max_gradient_norm << std::endl;
    std::cout << "  Weight Decay: " << weight_decay << std::endl;
    std::cout << std::defaultfloat;
    std::cout << "  Random Seed: " << random_seed << std::endl;
    
    // Print CUDA device information
    printCUDADeviceInfo();
    
    // Define model architecture
    std::vector<int> architecture = {784, 512, 256, 128, 10}; // 4-layer network for MNIST
    
    // Print model architecture
    printModelArchitecture(architecture);
    
    // Data paths
    std::string train_data_path = "data/input/train_data";
    std::string test_data_path = "data/input/test_data";
    
    std::cout << "\n=== Initializing Model ===" << std::endl;
    std::cout << "Training data path: " << train_data_path << std::endl;
    std::cout << "Test data path: " << test_data_path << std::endl;
    
    try {
        // Create model with 20% validation split
        Model model(architecture, train_data_path, test_data_path, 0.2f);
        
        // Initialize model
        model.initialize();
        
        // Configure training
        TrainingConfig config;
        config.epochs = num_epochs;
        config.batch_size = batch_size;
        config.learning_rate = learning_rate;
        config.momentum = 0.9f;
        config.weight_decay = weight_decay;
        config.gradient_clip_value = max_gradient_norm;
        config.log_interval = 50; // Log every 50 batches
        config.use_validation = true;
        
        // Start timing
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Initial GPU memory state
        size_t free_before, total_before;
        CUDA_CHECK(cudaMemGetInfo(&free_before, &total_before));
        
        // Train the model
        float final_val_accuracy = model.train(config);
        
        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> training_duration = end_time - start_time;
        
        // Calculate performance metrics
        PerformanceMetrics perf;
        perf.training_time_seconds = training_duration.count();
        
        // Calculate total iterations (approximate)
        // Assuming we know the dataset size (60000 for MNIST training)
        int train_size = 48000; // 80% of 60000 (with 20% validation)
        perf.total_iterations = num_epochs * ((train_size + batch_size - 1) / batch_size);
        
        // Calculate FLOPS
        perf.total_flops = calculateFlops(architecture, batch_size, perf.total_iterations);
        perf.tflops = perf.total_flops / (perf.training_time_seconds * 1e12);
        
        // Memory usage
        size_t free_after, total_after;
        CUDA_CHECK(cudaMemGetInfo(&free_after, &total_after));
        perf.total_memory_bytes = free_before - free_after;
        
        // Print training summary
        std::cout << "\n=== Training Summary ===" << std::endl;
        std::cout << "Final Validation Accuracy: " << std::fixed << std::setprecision(4) 
                  << final_val_accuracy * 100 << "%" << std::endl;
        std::cout << "Total Training Time: " << std::fixed << std::setprecision(4) 
                  << perf.training_time_seconds << " seconds" << std::endl;
        std::cout << "Average Time per Epoch: " << std::fixed << std::setprecision(4) 
                  << perf.training_time_seconds / num_epochs << " seconds" << std::endl;
        
        // Test the model
        std::cout << "\n=== Evaluating on Test Set ===" << std::endl;
        EvaluationResults test_results = model.test(batch_size, true);
        
        // Print detailed metrics
        printDetailedMetrics(test_results, architecture.back());
        
        // Print performance metrics
        std::cout << "\n=== Performance Metrics ===" << std::endl;
        std::cout << "Total Training Iterations: " << perf.total_iterations << std::endl;
        std::cout << "Total FLOPS: " << std::scientific << std::setprecision(2) 
                  << perf.total_flops << std::endl;
        std::cout << "Training TFLOPS: " << std::fixed << std::setprecision(3) 
                  << perf.tflops << std::endl;
        std::cout << "GPU Memory Used: " << std::fixed << std::setprecision(2) 
                  << perf.total_memory_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
        
        // Calculate and print efficiency metrics
        cudaDeviceProp prop;
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        
        // Theoretical peak TFLOPS (approximate for modern GPUs)
        // For simplicity, assuming FP32: cores * 2 (FMA) * clock rate
        double theoretical_peak_tflops = prop.multiProcessorCount * 64 * 2 * 
                                       (prop.clockRate / 1e9) / 1000.0; // Rough estimate
        
        std::cout << "Estimated Peak TFLOPS (FP32): " << std::fixed << std::setprecision(1) 
                  << theoretical_peak_tflops << std::endl;
        std::cout << "CUDA Efficiency: " << std::fixed << std::setprecision(1) 
                  << (perf.tflops / theoretical_peak_tflops) * 100 << "%" << std::endl;
        
        // Print final GPU memory state
        printGPUMemoryInfo();
        
        // Save the trained model
        std::string model_file = "model_checkpoints/trained_model.bin";
        std::cout << "\n=== Saving Model ===" << std::endl;
        model.save_weights(model_file);
        std::cout << "Model saved to: " << model_file << std::endl;
        
        // Additional statistics
        std::cout << "\n=== Additional Statistics ===" << std::endl;
        std::cout << "Images per Second (Training): " << std::fixed << std::setprecision(1)
                  << (perf.total_iterations * batch_size) / perf.training_time_seconds << std::endl;
        std::cout << "Milliseconds per Batch: " << std::fixed << std::setprecision(2)
                  << (perf.training_time_seconds * 1000) / perf.total_iterations << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n=== Demo Complete ===" << std::endl;
    return 0;
}
