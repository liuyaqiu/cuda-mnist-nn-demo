#include "model.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cassert>

// CUDA kernel for computing predictions from output probabilities
__global__ void get_predictions_kernel(const float* output, int* predictions, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        const float* sample_output = output + idx * num_classes;
        int max_idx = 0;
        float max_val = sample_output[0];
        
        for (int i = 1; i < num_classes; i++) {
            if (sample_output[i] > max_val) {
                max_val = sample_output[i];
                max_idx = i;
            }
        }
        
        predictions[idx] = max_idx;
    }
}

// CUDA kernel for computing accuracy
__global__ void compute_accuracy_kernel(const float* predictions, const float* labels, 
                                       int* correct_count, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        const float* pred = predictions + idx * num_classes;
        const float* label = labels + idx * num_classes;
        
        // Find predicted class
        int pred_class = 0;
        float max_val = pred[0];
        for (int i = 1; i < num_classes; i++) {
            if (pred[i] > max_val) {
                max_val = pred[i];
                pred_class = i;
            }
        }
        
        // Find true class
        int true_class = 0;
        for (int i = 0; i < num_classes; i++) {
            if (label[i] > 0.5f) {  // One-hot encoded
                true_class = i;
                break;
            }
        }
        
        // Atomic add to correct count
        if (pred_class == true_class) {
            atomicAdd(correct_count, 1);
        }
    }
}

// Constructor
Model::Model(const std::vector<int>& architecture,
             const std::string& train_data_path,
             const std::string& test_data_path,
             float validation_ratio) 
    : architecture_(architecture),
      neural_network_(nullptr),
      train_data_loader_(nullptr),
      test_data_loader_(nullptr),
      current_loss_(0.0f),
      current_accuracy_(0.0f),
      current_epoch_(0),
      is_initialized_(false),
      d_input_batch_(nullptr),
      d_label_batch_(nullptr),
      d_output_batch_(nullptr),
      d_loss_(nullptr),
      h_input_batch_(nullptr),
      h_label_batch_(nullptr),
      h_output_batch_(nullptr),
      max_batch_size_(256) {
    
    // Validate architecture
    if (architecture_.size() < 2) {
        throw std::invalid_argument("Architecture must have at least 2 layers (input and output)");
    }
    
    input_size_ = architecture_.front();
    output_size_ = architecture_.back();
    
    // Calculate total parameters
    total_params_ = 0;
    for (size_t i = 1; i < architecture_.size(); ++i) {
        total_params_ += architecture_[i-1] * architecture_[i];  // Weights
        total_params_ += architecture_[i];  // Biases
    }
    
    // Create data loaders
    train_data_loader_ = new DataLoader(train_data_path, validation_ratio);
    
    if (!test_data_path.empty()) {
        test_data_loader_ = new DataLoader(test_data_path, 0.0f);  // No validation split for test data
    }
    
    std::cout << "Model created with architecture: ";
    for (size_t i = 0; i < architecture_.size(); ++i) {
        std::cout << architecture_[i];
        if (i < architecture_.size() - 1) std::cout << " -> ";
    }
    std::cout << std::endl;
    std::cout << "Total parameters: " << total_params_ << std::endl;
}

// Destructor
Model::~Model() {
    deallocate_memory();
    
    if (neural_network_) {
        delete neural_network_;
    }
    
    if (train_data_loader_) {
        delete train_data_loader_;
    }
    
    if (test_data_loader_) {
        delete test_data_loader_;
    }
}

// Initialize the model
void Model::initialize() {
    if (is_initialized_) {
        std::cout << "Model is already initialized." << std::endl;
        return;
    }
    
    std::cout << "Initializing model..." << std::endl;
    
    // Initialize data loaders
    train_data_loader_->initialize_input();
    if (test_data_loader_) {
        test_data_loader_->initialize_input();
    }
    
    // Verify data dimensions match model input
    int data_input_size = train_data_loader_->get_image_elements();
    if (data_input_size != input_size_) {
        throw std::runtime_error("Data input size (" + std::to_string(data_input_size) + 
                               ") does not match model input size (" + std::to_string(input_size_) + ")");
    }
    
    // Create neural network with the architecture
    neural_network_ = new NeuralNetwork(architecture_);
    
    // Allocate memory
    allocate_memory(max_batch_size_);
    
    is_initialized_ = true;
    
    std::cout << "Model initialized successfully." << std::endl;
    std::cout << "Training dataset: " << train_data_loader_->get_training_size() << " samples" << std::endl;
    std::cout << "Validation dataset: " << train_data_loader_->get_validation_size() << " samples" << std::endl;
    if (test_data_loader_) {
        std::cout << "Test dataset: " << test_data_loader_->get_total_size() << " samples" << std::endl;
    }
}

// Allocate GPU and host memory
void Model::allocate_memory(int max_batch_size) {
    max_batch_size_ = max_batch_size;
    
    // Allocate GPU memory
    HANDLE_ERROR(cudaMalloc(&d_input_batch_, max_batch_size_ * input_size_ * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_label_batch_, max_batch_size_ * output_size_ * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_output_batch_, max_batch_size_ * output_size_ * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&d_loss_, sizeof(float)));
    
    // Allocate host memory
    h_input_batch_ = new float[max_batch_size_ * input_size_];
    h_label_batch_ = new float[max_batch_size_ * output_size_];
    h_output_batch_ = new float[max_batch_size_ * output_size_];
}

// Deallocate memory
void Model::deallocate_memory() {
    // Free GPU memory
    if (d_input_batch_) cudaFree(d_input_batch_);
    if (d_label_batch_) cudaFree(d_label_batch_);
    if (d_output_batch_) cudaFree(d_output_batch_);
    if (d_loss_) cudaFree(d_loss_);
    
    // Free host memory
    delete[] h_input_batch_;
    delete[] h_label_batch_;
    delete[] h_output_batch_;
    
    d_input_batch_ = nullptr;
    d_label_batch_ = nullptr;
    d_output_batch_ = nullptr;
    d_loss_ = nullptr;
    h_input_batch_ = nullptr;
    h_label_batch_ = nullptr;
    h_output_batch_ = nullptr;
}

// Train the model
float Model::train(const TrainingConfig& config) {
    if (!is_initialized_) {
        throw std::runtime_error("Model must be initialized before training. Call initialize() first.");
    }
    
    std::cout << "\n=== Starting Training ===" << std::endl;
    std::cout << "Epochs: " << config.epochs << std::endl;
    std::cout << "Batch size: " << config.batch_size << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Learning rate: " << config.learning_rate << std::endl;
    std::cout << "Momentum: " << config.momentum << std::endl;
    std::cout << "Weight decay: " << config.weight_decay << std::endl;
    std::cout << "Gradient clipping: " << config.gradient_clip_value << std::endl;
    std::cout << std::defaultfloat;
    std::cout << "========================\n" << std::endl;
    
    float best_validation_accuracy = 0.0f;
    
    // Training loop
    for (int epoch = 1; epoch <= config.epochs; ++epoch) {
        current_epoch_ = epoch;
        auto epoch_start = std::chrono::high_resolution_clock::now();
        
        // Train for one epoch
        float avg_train_loss = train_epoch(config);
        
        // Evaluate on validation set
        if (config.use_validation && train_data_loader_->get_validation_size() > 0) {
            EvaluationResults val_results = evaluate_dataset(train_data_loader_, config.batch_size, true);
            
            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start);
            
            std::cout << "Epoch " << epoch << "/" << config.epochs 
                      << " - Time: " << epoch_duration.count() << "s"
                      << " - Train Loss: " << std::fixed << std::setprecision(4) << avg_train_loss
                      << " - Val Loss: " << val_results.loss
                      << " - Val Acc: " << std::setprecision(2) << val_results.accuracy * 100 << "%"
                      << std::endl;
            
            if (val_results.accuracy > best_validation_accuracy) {
                best_validation_accuracy = val_results.accuracy;
            }
            
            current_accuracy_ = val_results.accuracy;
        } else {
            // Evaluate on training set if no validation
            EvaluationResults train_results = evaluate_dataset(train_data_loader_, config.batch_size, false);
            
            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start);
            
            std::cout << "Epoch " << epoch << "/" << config.epochs 
                      << " - Time: " << epoch_duration.count() << "s"
                      << " - Train Loss: " << std::fixed << std::setprecision(4) << avg_train_loss
                      << " - Train Acc: " << std::setprecision(2) << train_results.accuracy * 100 << "%"
                      << std::endl;
            
            current_accuracy_ = train_results.accuracy;
        }
        
        // Reset data loader for next epoch
        train_data_loader_->reset_training_iterator();
    }
    
    std::cout << "\n=== Training Complete ===" << std::endl;
    if (config.use_validation && train_data_loader_->get_validation_size() > 0) {
        std::cout << "Best validation accuracy: " << std::fixed << std::setprecision(4) 
                  << best_validation_accuracy * 100 << "%" << std::endl;
        return best_validation_accuracy;
    }
    
    return current_accuracy_;
}

// Train for one epoch
float Model::train_epoch(const TrainingConfig& config) {
    int total_batches = (train_data_loader_->get_training_size() + config.batch_size - 1) / config.batch_size;
    float epoch_loss = 0.0f;
    int batch_count = 0;
    
    while (train_data_loader_->has_more_training_batches(config.batch_size)) {
        // Load batch
        train_data_loader_->get(config.batch_size, h_input_batch_, false, h_label_batch_);
        
        // Copy to GPU
        HANDLE_ERROR(cudaMemcpy(d_input_batch_, h_input_batch_, 
                                config.batch_size * input_size_ * sizeof(float), 
                                cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_label_batch_, h_label_batch_, 
                                config.batch_size * output_size_ * sizeof(float), 
                                cudaMemcpyHostToDevice));
        
        // Forward pass - returns average loss
        float batch_loss = neural_network_->forward_batch(d_input_batch_, d_label_batch_, config.batch_size);
        epoch_loss += batch_loss;
        
        // Backward pass
        neural_network_->backward_batch(d_input_batch_, d_label_batch_, 
                                       config.learning_rate, config.batch_size,
                                       config.gradient_clip_value, config.weight_decay);
        
        batch_count++;
        
        // Log progress
        if (config.log_interval > 0 && batch_count % config.log_interval == 0) {
            // Get predictions for accuracy calculation
            int* d_predictions;
            HANDLE_ERROR(cudaMalloc(&d_predictions, config.batch_size * sizeof(int)));
            neural_network_->predict_batch(d_input_batch_, d_predictions, config.batch_size);
            
            // Copy predictions to host
            int* h_predictions = new int[config.batch_size];
            HANDLE_ERROR(cudaMemcpy(h_predictions, d_predictions, 
                                   config.batch_size * sizeof(int), 
                                   cudaMemcpyDeviceToHost));
            
            // Compute accuracy
            float batch_accuracy = compute_accuracy_from_predictions(h_predictions, h_label_batch_, config.batch_size);
            
            delete[] h_predictions;
            cudaFree(d_predictions);
            
            print_progress(current_epoch_, batch_count, total_batches, batch_loss, batch_accuracy);
        }
    }
    
    return epoch_loss / batch_count;
}

// Test the model
EvaluationResults Model::test(int batch_size, bool verbose) {
    if (!is_initialized_) {
        throw std::runtime_error("Model must be initialized before testing. Call initialize() first.");
    }
    
    if (!test_data_loader_) {
        throw std::runtime_error("No test data loader available. Provide test_data_path in constructor.");
    }
    
    if (verbose) {
        std::cout << "\n=== Evaluating on Test Set ===" << std::endl;
    }
    
    EvaluationResults results = evaluate_dataset(test_data_loader_, batch_size, false);
    
    if (verbose) {
        std::cout << "Test Accuracy: " << std::fixed << std::setprecision(2) 
                  << results.accuracy * 100 << "%" << std::endl;
        std::cout << "Test Loss: " << std::setprecision(4) << results.loss << std::endl;
        std::cout << "Correct: " << results.correct_predictions << "/" << results.total_samples << std::endl;
        
        // Print per-class accuracy if available
        if (!results.per_class_accuracy.empty()) {
            std::cout << "\nPer-class accuracy:" << std::endl;
            for (size_t i = 0; i < results.per_class_accuracy.size(); ++i) {
                std::cout << "Class " << i << ": " << std::setprecision(2) 
                          << results.per_class_accuracy[i] * 100 << "%" << std::endl;
            }
        }
    }
    
    return results;
}

// Evaluate on a dataset
EvaluationResults Model::evaluate_dataset(DataLoader* data_loader, int batch_size, bool is_validation) {
    EvaluationResults results;
    results.per_class_accuracy.resize(output_size_, 0.0f);
    results.confusion_matrix.resize(output_size_ * output_size_, 0);
    
    std::vector<int> class_correct(output_size_, 0);
    std::vector<int> class_total(output_size_, 0);
    
    float total_loss = 0.0f;
    int batch_count = 0;
    
    // Reset iterator
    if (is_validation) {
        data_loader->reset_validation_iterator();
    } else {
        data_loader->reset_training_iterator();
    }
    
    // Evaluation loop
    while ((is_validation && data_loader->has_more_validation_batches(batch_size)) ||
           (!is_validation && data_loader->has_more_training_batches(batch_size))) {
        
        // Load batch
        data_loader->get(batch_size, h_input_batch_, is_validation, h_label_batch_);
        
        // Copy to GPU
        HANDLE_ERROR(cudaMemcpy(d_input_batch_, h_input_batch_, 
                                batch_size * input_size_ * sizeof(float), 
                                cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_label_batch_, h_label_batch_, 
                                batch_size * output_size_ * sizeof(float), 
                                cudaMemcpyHostToDevice));
        
        // Forward pass - returns average loss
        float batch_loss = neural_network_->forward_batch(d_input_batch_, d_label_batch_, batch_size);
        total_loss += batch_loss;
        
        // Get predictions for accuracy calculation
        int* d_predictions;
        HANDLE_ERROR(cudaMalloc(&d_predictions, batch_size * sizeof(int)));
        neural_network_->predict_batch(d_input_batch_, d_predictions, batch_size);
        
        // Copy predictions to host
        int* h_predictions = new int[batch_size];
        HANDLE_ERROR(cudaMemcpy(h_predictions, d_predictions, 
                               batch_size * sizeof(int), 
                               cudaMemcpyDeviceToHost));
        
        // Update metrics
        for (int i = 0; i < batch_size; ++i) {
            // Get predicted class from predictions array
            int pred_class = h_predictions[i];
            int true_class = get_predicted_class(h_label_batch_ + i * output_size_);
            
            // Skip if this is padding (all zeros in label)
            bool is_padding = true;
            for (int j = 0; j < output_size_; ++j) {
                if (h_label_batch_[i * output_size_ + j] != 0.0f) {
                    is_padding = false;
                    break;
                }
            }
            if (is_padding) continue;
            
            results.total_samples++;
            if (pred_class == true_class) {
                results.correct_predictions++;
                class_correct[true_class]++;
            }
            class_total[true_class]++;
            
            // Update confusion matrix
            results.confusion_matrix[true_class * output_size_ + pred_class]++;
        }
        
        // Clean up
        delete[] h_predictions;
        cudaFree(d_predictions);
        
        batch_count++;
    }
    
    // Calculate final metrics
    results.accuracy = static_cast<float>(results.correct_predictions) / results.total_samples;
    results.loss = total_loss / batch_count;
    
    // Calculate per-class accuracy
    for (int i = 0; i < output_size_; ++i) {
        if (class_total[i] > 0) {
            results.per_class_accuracy[i] = static_cast<float>(class_correct[i]) / class_total[i];
        }
    }
    
    return results;
}

// Predict on input data
std::vector<int> Model::predict(const float* input, int batch_size) {
    if (!is_initialized_) {
        throw std::runtime_error("Model must be initialized before prediction. Call initialize() first.");
    }
    
    if (batch_size > max_batch_size_) {
        throw std::invalid_argument("Batch size exceeds maximum allocated size");
    }
    
    // Copy input to GPU
    HANDLE_ERROR(cudaMemcpy(d_input_batch_, input, 
                            batch_size * input_size_ * sizeof(float), 
                            cudaMemcpyHostToDevice));
    
    // Get predictions directly
    int* d_predictions;
    HANDLE_ERROR(cudaMalloc(&d_predictions, batch_size * sizeof(int)));
    neural_network_->predict_batch(d_input_batch_, d_predictions, batch_size);
    
    // Copy predictions to host
    std::vector<int> predictions(batch_size);
    HANDLE_ERROR(cudaMemcpy(predictions.data(), d_predictions, 
                           batch_size * sizeof(int), 
                           cudaMemcpyDeviceToHost));
    
    cudaFree(d_predictions);
    
    return predictions;
}

// Save model weights
void Model::save_weights(const std::string& filepath) {
    if (!is_initialized_ || !neural_network_) {
        throw std::runtime_error("Model must be initialized before saving weights");
    }
    
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    
    // Save architecture
    size_t arch_size = architecture_.size();
    file.write(reinterpret_cast<const char*>(&arch_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(architecture_.data()), arch_size * sizeof(int));
    
    // Save weights for each layer
    for (int i = 0; i < neural_network_->get_num_layers(); ++i) {
        const NeuralLayer* layer = neural_network_->get_layer(i);
        
        // Get weight and bias dimensions from architecture
        int input_size = architecture_[i];
        int output_size = architecture_[i + 1];
        int weight_size = input_size * output_size;
        int bias_size = output_size;
        
        // Allocate host memory
        float* h_weights = new float[weight_size];
        float* h_biases = new float[bias_size];
        
        // Copy from device to host
        HANDLE_ERROR(cudaMemcpy(h_weights, layer->get_weights(), 
                                weight_size * sizeof(float), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(h_biases, layer->get_biases(), 
                                bias_size * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Write to file
        file.write(reinterpret_cast<const char*>(h_weights), weight_size * sizeof(float));
        file.write(reinterpret_cast<const char*>(h_biases), bias_size * sizeof(float));
        
        // Clean up
        delete[] h_weights;
        delete[] h_biases;
    }
    
    file.close();
    std::cout << "Model weights saved to: " << filepath << std::endl;
}

// Load model weights
void Model::load_weights(const std::string& filepath) {
    if (!is_initialized_) {
        throw std::runtime_error("Model must be initialized before loading weights");
    }
    
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }
    
    // Read and verify architecture
    size_t arch_size;
    file.read(reinterpret_cast<char*>(&arch_size), sizeof(size_t));
    
    std::vector<int> saved_architecture(arch_size);
    file.read(reinterpret_cast<char*>(saved_architecture.data()), arch_size * sizeof(int));
    
    // Verify architecture matches
    if (saved_architecture != architecture_) {
        throw std::runtime_error("Saved model architecture does not match current model");
    }
    
    // Load weights for each layer
    for (int i = 0; i < neural_network_->get_num_layers(); ++i) {
        const NeuralLayer* layer = neural_network_->get_layer(i);
        
        // Get weight and bias dimensions from architecture
        int input_size = architecture_[i];
        int output_size = architecture_[i + 1];
        int weight_size = input_size * output_size;
        int bias_size = output_size;
        
        // Allocate host memory
        float* h_weights = new float[weight_size];
        float* h_biases = new float[bias_size];
        
        // Read from file
        file.read(reinterpret_cast<char*>(h_weights), weight_size * sizeof(float));
        file.read(reinterpret_cast<char*>(h_biases), bias_size * sizeof(float));
        
        // Copy to device (cast away const for loading)
        HANDLE_ERROR(cudaMemcpy(const_cast<float*>(layer->get_weights()), h_weights, 
                                weight_size * sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(const_cast<float*>(layer->get_biases()), h_biases, 
                                bias_size * sizeof(float), cudaMemcpyHostToDevice));
        
        // Clean up
        delete[] h_weights;
        delete[] h_biases;
    }
    
    file.close();
    std::cout << "Model weights loaded from: " << filepath << std::endl;
}

// Helper function to compute accuracy
float Model::compute_accuracy(const float* predictions, const float* labels, int batch_size) {
    int correct = 0;
    for (int i = 0; i < batch_size; ++i) {
        int pred_class = get_predicted_class(predictions + i * output_size_);
        int true_class = get_predicted_class(labels + i * output_size_);
        
        // Skip if this is padding (all zeros in label)
        bool is_padding = true;
        for (int j = 0; j < output_size_; ++j) {
            if (labels[i * output_size_ + j] != 0.0f) {
                is_padding = false;
                break;
            }
        }
        
        if (!is_padding && pred_class == true_class) {
            correct++;
        }
    }
    
    // Count non-padding samples
    int valid_samples = 0;
    for (int i = 0; i < batch_size; ++i) {
        bool is_padding = true;
        for (int j = 0; j < output_size_; ++j) {
            if (labels[i * output_size_ + j] != 0.0f) {
                is_padding = false;
                break;
            }
        }
        if (!is_padding) valid_samples++;
    }
    
    return valid_samples > 0 ? static_cast<float>(correct) / valid_samples : 0.0f;
}

// Helper function to compute accuracy from prediction indices
float Model::compute_accuracy_from_predictions(const int* predictions, const float* labels, int batch_size) {
    int correct = 0;
    int valid_samples = 0;
    
    for (int i = 0; i < batch_size; ++i) {
        int true_class = get_predicted_class(labels + i * output_size_);
        
        // Skip if this is padding (all zeros in label)
        bool is_padding = true;
        for (int j = 0; j < output_size_; ++j) {
            if (labels[i * output_size_ + j] != 0.0f) {
                is_padding = false;
                break;
            }
        }
        
        if (!is_padding) {
            valid_samples++;
            if (predictions[i] == true_class) {
                correct++;
            }
        }
    }
    
    return valid_samples > 0 ? static_cast<float>(correct) / valid_samples : 0.0f;
}

// Get predicted class from output probabilities
int Model::get_predicted_class(const float* output) {
    int max_idx = 0;
    float max_val = output[0];
    
    for (int i = 1; i < output_size_; ++i) {
        if (output[i] > max_val) {
            max_val = output[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

// Convert one-hot labels to class indices
std::vector<int> Model::get_label_indices(const float* labels, int batch_size) {
    std::vector<int> indices(batch_size);
    
    for (int i = 0; i < batch_size; ++i) {
        const float* label = labels + i * output_size_;
        for (int j = 0; j < output_size_; ++j) {
            if (label[j] > 0.5f) {
                indices[i] = j;
                break;
            }
        }
    }
    
    return indices;
}

// Print training progress
void Model::print_progress(int epoch, int batch, int total_batches, float loss, float accuracy) {
    int bar_width = 50;
    float progress = static_cast<float>(batch) / total_batches;
    
    std::cout << "Epoch " << epoch << " [";
    int pos = static_cast<int>(bar_width * progress);
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100.0) << "% "
              << "Batch " << batch << "/" << total_batches
              << " - Loss: " << std::fixed << std::setprecision(4) << loss
              << " - Acc: " << std::setprecision(2) << accuracy * 100 << "%"
              << "\r" << std::flush;
    
    if (batch == total_batches) {
        std::cout << std::endl;
    }
    
    current_loss_ = loss;
}
