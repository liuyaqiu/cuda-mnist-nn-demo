#ifndef MODEL_H
#define MODEL_H

#include "nn_utils.h"
#include "data_loader.h"
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <memory>

// Reuse HANDLE_ERROR macro from data_loader.h
#ifndef HANDLE_ERROR
#define HANDLE_ERROR(x) HANDLE_CUDA_ERROR(x)
#endif

// Training configuration structure
struct TrainingConfig {
    int epochs;
    int batch_size;
    float learning_rate;
    float momentum;
    float weight_decay;
    float gradient_clip_value;
    int log_interval;  // Log every N batches
    bool use_validation;
    
    // Default constructor with sensible defaults
    TrainingConfig() : 
        epochs(10),
        batch_size(64),
        learning_rate(0.01f),
        momentum(0.9f),
        weight_decay(0.0001f),
        gradient_clip_value(5.0f),
        log_interval(100),
        use_validation(true) {}
};

// Test/evaluation results structure
struct EvaluationResults {
    float accuracy;
    float loss;
    int total_samples;
    int correct_predictions;
    std::vector<float> per_class_accuracy;
    std::vector<int> confusion_matrix;  // Flattened confusion matrix
    
    EvaluationResults() : accuracy(0.0f), loss(0.0f), total_samples(0), correct_predictions(0) {}
};

// Model class that encapsulates neural network, data loading, training, and inference
class Model {
public:
    // Constructor
    // architecture: vector of layer sizes (e.g., {784, 128, 64, 10} for MNIST)
    // train_data_path: path to training/validation data
    // test_data_path: path to test data (can be empty if only training)
    // validation_ratio: fraction of training data to use for validation
    Model(const std::vector<int>& architecture,
          const std::string& train_data_path,
          const std::string& test_data_path = "",
          float validation_ratio = 0.2f);
    
    // Destructor
    ~Model();
    
    // Initialize the model (must be called before training/testing)
    void initialize();
    
    // Train the model
    // config: training configuration
    // Returns: final validation accuracy (or training accuracy if no validation)
    float train(const TrainingConfig& config);
    
    // Test the model on test dataset
    // batch_size: batch size for evaluation
    // verbose: whether to print detailed results
    // Returns: evaluation results
    EvaluationResults test(int batch_size = 64, bool verbose = true);
    
    // Predict on a single input or batch
    // input: flattened input data (on host)
    // batch_size: number of samples in input
    // Returns: predicted class indices
    std::vector<int> predict(const float* input, int batch_size = 1);
    
    // Save model weights to file
    void save_weights(const std::string& filepath);
    
    // Load model weights from file
    void load_weights(const std::string& filepath);
    
    // Get model information
    int get_input_size() const { return input_size_; }
    int get_output_size() const { return output_size_; }
    const std::vector<int>& get_architecture() const { return architecture_; }
    
    // Get current training metrics
    float get_current_loss() const { return current_loss_; }
    float get_current_accuracy() const { return current_accuracy_; }
    int get_current_epoch() const { return current_epoch_; }
    
private:
    // Model architecture and components
    std::vector<int> architecture_;
    NeuralNetwork* neural_network_;
    DataLoader* train_data_loader_;
    DataLoader* test_data_loader_;
    
    // Model dimensions
    int input_size_;
    int output_size_;
    int total_params_;
    
    // Training state
    float current_loss_;
    float current_accuracy_;
    int current_epoch_;
    bool is_initialized_;
    
    // GPU memory for batch processing
    float* d_input_batch_;
    float* d_label_batch_;
    float* d_output_batch_;
    float* d_loss_;
    
    // Host memory for data transfer
    float* h_input_batch_;
    float* h_label_batch_;
    float* h_output_batch_;
    
    // Maximum batch size for memory allocation
    int max_batch_size_;
    
    // Helper functions
    void allocate_memory(int max_batch_size);
    void deallocate_memory();
    
    // Training helper functions
    float train_epoch(const TrainingConfig& config);
    EvaluationResults evaluate_dataset(DataLoader* data_loader, int batch_size, bool is_validation);
    
    // Compute accuracy from predictions and labels
    float compute_accuracy(const float* predictions, const float* labels, int batch_size);
    
    // Compute accuracy from prediction indices and labels
    float compute_accuracy_from_predictions(const int* predictions, const float* labels, int batch_size);
    
    // Get predicted class from output probabilities
    int get_predicted_class(const float* output);
    
    // Convert one-hot labels to class indices
    std::vector<int> get_label_indices(const float* labels, int batch_size);
    
    // Print training progress
    void print_progress(int epoch, int batch, int total_batches, float loss, float accuracy);
};

#endif // MODEL_H
