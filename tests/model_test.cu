#include <gtest/gtest.h>
#include "model.h"
#include <filesystem>
#include <random>
#include <FreeImage.h>
#include <cuda_runtime.h>
#include <cmath>

namespace fs = std::filesystem;

// Test fixture for Model tests
class ModelTest : public ::testing::Test {
protected:
    // Test data directory paths
    std::string test_data_dir_;
    std::string test_data_dir_2_;
    
    // Setup method called before each test
    void SetUp() override {
        // Create test directories
        test_data_dir_ = "model_test_data";
        test_data_dir_2_ = "model_test_data_2";
        
        // Initialize FreeImage
        FreeImage_Initialise();
        
        // Create test datasets
        createTestDataset(test_data_dir_, 20, 28, 28, 3);  // 20 images per class, 3 classes = 60 total
        createTestDataset(test_data_dir_2_, 10, 28, 28, 3); // 10 images per class for test set = 30 total
    }
    
    // Teardown method called after each test
    void TearDown() override {
        // Clean up test directories
        if (fs::exists(test_data_dir_)) {
            fs::remove_all(test_data_dir_);
        }
        if (fs::exists(test_data_dir_2_)) {
            fs::remove_all(test_data_dir_2_);
        }
        
        // De-initialize FreeImage
        FreeImage_DeInitialise();
    }
    
    // Helper function to create a test dataset
    void createTestDataset(const std::string& dir, int num_images_per_class, 
                          int width, int height, int num_classes) {
        fs::create_directories(dir);
        
        // Create directories for each class
        for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
            fs::create_directories(dir + "/" + std::to_string(class_idx));
            
            // Create images for this class
            for (int img_idx = 0; img_idx < num_images_per_class; ++img_idx) {
                createTestImage(dir + "/" + std::to_string(class_idx) + 
                              "/image_" + std::to_string(img_idx) + ".png",
                              width, height, class_idx);
            }
        }
    }
    
    // Helper function to create a single test image
    void createTestImage(const std::string& filepath, int width, int height, int class_label) {
        FIBITMAP* bitmap = FreeImage_Allocate(width, height, 8);
        if (!bitmap) {
            FAIL() << "Failed to allocate bitmap";
        }
        
        // Create a simple pattern based on class label
        for (int y = 0; y < height; ++y) {
            BYTE* scanline = FreeImage_GetScanLine(bitmap, y);
            for (int x = 0; x < width; ++x) {
                // Create different patterns for different classes
                BYTE value = 0;
                switch (class_label) {
                    case 0: // Horizontal gradient
                        value = static_cast<BYTE>((x * 255) / width);
                        break;
                    case 1: // Vertical gradient
                        value = static_cast<BYTE>((y * 255) / height);
                        break;
                    case 2: // Diagonal gradient
                        value = static_cast<BYTE>(((x + y) * 255) / (width + height));
                        break;
                    default: // Random pattern
                        value = static_cast<BYTE>((x * y) % 256);
                        break;
                }
                scanline[x] = value;
            }
        }
        
        if (!FreeImage_Save(FIF_PNG, bitmap, filepath.c_str())) {
            FreeImage_Unload(bitmap);
            FAIL() << "Failed to save test image: " << filepath;
        }
        
        FreeImage_Unload(bitmap);
    }
    
    // Helper function to create synthetic input data
    void createSyntheticInput(float* data, int batch_size, int input_size, float pattern_value = 0.5f) {
        for (int i = 0; i < batch_size * input_size; ++i) {
            data[i] = pattern_value + 0.1f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
        }
    }
};

// Test Model constructor and initialization
TEST_F(ModelTest, ConstructorAndInitialization) {
    // Test valid architecture
    std::vector<int> architecture = {784, 128, 64, 10};
    Model model(architecture, test_data_dir_, test_data_dir_2_, 0.2f);
    
    // Check architecture
    EXPECT_EQ(model.get_architecture(), architecture);
    EXPECT_EQ(model.get_input_size(), 784);
    EXPECT_EQ(model.get_output_size(), 10);
    
    // Model should not be initialized yet
    EXPECT_EQ(model.get_current_epoch(), 0);
    
    // Initialize model
    EXPECT_NO_THROW(model.initialize());
    
    // Try to initialize again (should be safe)
    EXPECT_NO_THROW(model.initialize());
}

// Test invalid architecture
TEST_F(ModelTest, InvalidArchitecture) {
    // Architecture with only one layer
    std::vector<int> invalid_arch = {784};
    EXPECT_THROW(Model model(invalid_arch, test_data_dir_), std::invalid_argument);
    
    // Empty architecture
    std::vector<int> empty_arch = {};
    EXPECT_THROW(Model model(empty_arch, test_data_dir_), std::invalid_argument);
}

// Test training with small dataset
TEST_F(ModelTest, BasicTraining) {
    // Create a simple 2-layer network for faster testing
    std::vector<int> architecture = {784, 32, 3};  // 3 classes to match test data
    Model model(architecture, test_data_dir_, "", 0.2f);
    
    model.initialize();
    
    // Configure training
    TrainingConfig config;
    config.epochs = 2;
    config.batch_size = 10;
    config.learning_rate = 0.01f;
    config.momentum = 0.9f;
    config.weight_decay = 0.0001f;
    config.gradient_clip_value = 5.0f;
    config.log_interval = 5;
    config.use_validation = true;
    
    // Train the model
    float final_accuracy = model.train(config);
    
    // Check that training happened
    EXPECT_EQ(model.get_current_epoch(), 2);
    EXPECT_GE(final_accuracy, 0.0f);
    EXPECT_LE(final_accuracy, 1.0f);
    
    // Loss should be positive
    EXPECT_GT(model.get_current_loss(), 0.0f);
}

// Test training without validation
TEST_F(ModelTest, TrainingWithoutValidation) {
    std::vector<int> architecture = {784, 32, 3};
    Model model(architecture, test_data_dir_, "", 0.0f);  // No validation split
    
    model.initialize();
    
    TrainingConfig config;
    config.epochs = 1;
    config.batch_size = 20;
    config.use_validation = false;
    
    float final_accuracy = model.train(config);
    
    EXPECT_GE(final_accuracy, 0.0f);
    EXPECT_LE(final_accuracy, 1.0f);
}

// Test model evaluation
TEST_F(ModelTest, ModelEvaluation) {
    std::vector<int> architecture = {784, 32, 3};
    Model model(architecture, test_data_dir_, test_data_dir_2_, 0.1f);
    
    model.initialize();
    
    // Train briefly
    TrainingConfig config;
    config.epochs = 1;
    config.batch_size = 10;
    model.train(config);
    
    // Test evaluation
    EvaluationResults results = model.test(10, true);
    
    // Check results structure
    EXPECT_GE(results.accuracy, 0.0f);
    EXPECT_LE(results.accuracy, 1.0f);
    EXPECT_GT(results.loss, 0.0f);
    EXPECT_GT(results.total_samples, 0);
    EXPECT_GE(results.correct_predictions, 0);
    EXPECT_LE(results.correct_predictions, results.total_samples);
    
    // Check per-class accuracy
    EXPECT_EQ(results.per_class_accuracy.size(), 3);  // 3 classes
    for (float acc : results.per_class_accuracy) {
        EXPECT_GE(acc, 0.0f);
        EXPECT_LE(acc, 1.0f);
    }
    
    // Check confusion matrix
    EXPECT_EQ(results.confusion_matrix.size(), 9);  // 3x3 matrix
}

// Test prediction functionality
TEST_F(ModelTest, Prediction) {
    std::vector<int> architecture = {784, 32, 3};
    Model model(architecture, test_data_dir_, "", 0.0f);
    
    model.initialize();
    
    // Create synthetic input
    const int batch_size = 5;
    float* input_data = new float[batch_size * 784];
    createSyntheticInput(input_data, batch_size, 784);
    
    // Make predictions
    std::vector<int> predictions = model.predict(input_data, batch_size);
    
    // Check predictions
    EXPECT_EQ(predictions.size(), batch_size);
    for (int pred : predictions) {
        EXPECT_GE(pred, 0);
        EXPECT_LT(pred, 3);  // 3 classes
    }
    
    delete[] input_data;
}

// Test single sample prediction
TEST_F(ModelTest, SinglePrediction) {
    std::vector<int> architecture = {784, 32, 3};
    Model model(architecture, test_data_dir_, "", 0.0f);
    
    model.initialize();
    
    // Create single input
    float* input_data = new float[784];
    createSyntheticInput(input_data, 1, 784);
    
    // Make prediction
    std::vector<int> predictions = model.predict(input_data, 1);
    
    EXPECT_EQ(predictions.size(), 1);
    EXPECT_GE(predictions[0], 0);
    EXPECT_LT(predictions[0], 3);
    
    delete[] input_data;
}

// Test model saving and loading
TEST_F(ModelTest, SaveAndLoadWeights) {
    std::vector<int> architecture = {784, 32, 3};
    Model model1(architecture, test_data_dir_, "", 0.0f);
    
    model1.initialize();
    
    // Train the model briefly
    TrainingConfig config;
    config.epochs = 1;
    config.batch_size = 10;
    model1.train(config);
    
    // Save weights
    std::string weight_file = "test_model_weights.bin";
    EXPECT_NO_THROW(model1.save_weights(weight_file));
    
    // Create a new model with same architecture
    Model model2(architecture, test_data_dir_, "", 0.0f);
    model2.initialize();
    
    // Load weights
    EXPECT_NO_THROW(model2.load_weights(weight_file));
    
    // Test that predictions are the same
    float* test_input = new float[784];
    createSyntheticInput(test_input, 1, 784, 0.7f);
    
    std::vector<int> pred1 = model1.predict(test_input, 1);
    std::vector<int> pred2 = model2.predict(test_input, 1);
    
    EXPECT_EQ(pred1[0], pred2[0]);
    
    delete[] test_input;
    
    // Clean up weight file
    fs::remove(weight_file);
}

// Test loading weights with mismatched architecture
TEST_F(ModelTest, LoadWeightsMismatchedArchitecture) {
    // Create and save model with one architecture
    std::vector<int> arch1 = {784, 32, 3};
    Model model1(arch1, test_data_dir_, "", 0.0f);
    model1.initialize();
    
    std::string weight_file = "test_model_weights_mismatch.bin";
    model1.save_weights(weight_file);
    
    // Try to load with different architecture
    std::vector<int> arch2 = {784, 64, 3};  // Different hidden layer size
    Model model2(arch2, test_data_dir_, "", 0.0f);
    model2.initialize();
    
    EXPECT_THROW(model2.load_weights(weight_file), std::runtime_error);
    
    // Clean up
    fs::remove(weight_file);
}

// Test training with different batch sizes
TEST_F(ModelTest, DifferentBatchSizes) {
    // Test with various batch sizes - create new model for each to avoid state issues
    std::vector<int> batch_sizes = {1, 5, 10, 20};  // Max 20 since we have limited data
    
    for (int batch_size : batch_sizes) {
        std::vector<int> architecture = {784, 32, 3};
        Model model(architecture, test_data_dir_, "", 0.0f);  // No validation to have more training data
        model.initialize();
        
        TrainingConfig config;
        config.epochs = 1;
        config.batch_size = batch_size;
        config.log_interval = 0;  // Disable logging for cleaner test output
        config.use_validation = false;
        
        float accuracy = model.train(config);
        EXPECT_GE(accuracy, 0.0f);
        EXPECT_LE(accuracy, 1.0f);
    }
}

// Test error handling for uninitialized model
TEST_F(ModelTest, UninitializedModelErrors) {
    std::vector<int> architecture = {784, 32, 3};
    Model model(architecture, test_data_dir_, test_data_dir_2_, 0.2f);
    
    // Don't initialize
    
    // Training should throw
    TrainingConfig config;
    EXPECT_THROW(model.train(config), std::runtime_error);
    
    // Testing should throw
    EXPECT_THROW(model.test(10), std::runtime_error);
    
    // Prediction should throw
    float* input = new float[784];
    EXPECT_THROW(model.predict(input, 1), std::runtime_error);
    delete[] input;
    
    // Saving weights should throw
    EXPECT_THROW(model.save_weights("test.bin"), std::runtime_error);
}

// Test with no test data loader
TEST_F(ModelTest, NoTestDataLoader) {
    std::vector<int> architecture = {784, 32, 3};
    Model model(architecture, test_data_dir_, "", 0.2f);  // Empty test path
    
    model.initialize();
    
    // Testing should throw since no test data loader
    EXPECT_THROW(model.test(10), std::runtime_error);
}

// Test prediction with batch size exceeding maximum
TEST_F(ModelTest, PredictionBatchSizeLimit) {
    std::vector<int> architecture = {784, 32, 3};
    Model model(architecture, test_data_dir_, "", 0.0f);
    
    model.initialize();
    
    // Try to predict with batch size larger than max (256 by default)
    const int large_batch_size = 300;
    float* input_data = new float[large_batch_size * 784];
    
    EXPECT_THROW(model.predict(input_data, large_batch_size), std::invalid_argument);
    
    delete[] input_data;
}

// Test training configuration edge cases
TEST_F(ModelTest, TrainingConfigurationEdgeCases) {
    std::vector<int> architecture = {784, 32, 3};
    Model model(architecture, test_data_dir_, "", 0.2f);
    
    model.initialize();
    
    // Test with zero epochs (should complete immediately)
    TrainingConfig config1;
    config1.epochs = 0;
    float acc1 = model.train(config1);
    EXPECT_EQ(model.get_current_epoch(), 0);
    
    // Test with very small learning rate
    TrainingConfig config2;
    config2.epochs = 1;
    config2.learning_rate = 1e-6f;
    float acc2 = model.train(config2);
    EXPECT_GE(acc2, 0.0f);
    
    // Test with no weight decay or momentum
    TrainingConfig config3;
    config3.epochs = 1;
    config3.momentum = 0.0f;
    config3.weight_decay = 0.0f;
    float acc3 = model.train(config3);
    EXPECT_GE(acc3, 0.0f);
}

// Test model with larger architecture
TEST_F(ModelTest, LargerArchitecture) {
    std::vector<int> architecture = {784, 256, 128, 64, 32, 3};  // 5-layer network
    Model model(architecture, test_data_dir_, "", 0.1f);
    
    EXPECT_NO_THROW(model.initialize());
    
    // Quick training test
    TrainingConfig config;
    config.epochs = 1;
    config.batch_size = 10;
    config.log_interval = 0;
    
    float accuracy = model.train(config);
    EXPECT_GE(accuracy, 0.0f);
    EXPECT_LE(accuracy, 1.0f);
}

// Test memory management with multiple predictions
TEST_F(ModelTest, MemoryManagementStressTest) {
    std::vector<int> architecture = {784, 32, 3};
    Model model(architecture, test_data_dir_, "", 0.0f);
    
    model.initialize();
    
    // Make many predictions to test memory management
    const int num_iterations = 100;
    const int batch_size = 10;
    float* input_data = new float[batch_size * 784];
    
    for (int i = 0; i < num_iterations; ++i) {
        createSyntheticInput(input_data, batch_size, 784, i * 0.01f);
        std::vector<int> predictions = model.predict(input_data, batch_size);
        EXPECT_EQ(predictions.size(), batch_size);
    }
    
    delete[] input_data;
}

// Test concurrent training and evaluation
TEST_F(ModelTest, TrainingMetrics) {
    std::vector<int> architecture = {784, 64, 3};
    Model model(architecture, test_data_dir_, test_data_dir_2_, 0.2f);
    
    model.initialize();
    
    // Train and check metrics update
    TrainingConfig config;
    config.epochs = 3;
    config.batch_size = 10;
    config.log_interval = 5;
    
    float initial_loss = model.get_current_loss();
    EXPECT_EQ(initial_loss, 0.0f);  // Should be 0 before training
    
    float final_accuracy = model.train(config);
    
    // Check that metrics were updated
    EXPECT_GT(model.get_current_loss(), 0.0f);
    // Note: get_current_accuracy() returns the last epoch's accuracy, not the best
    EXPECT_GE(model.get_current_accuracy(), 0.0f);
    EXPECT_LE(model.get_current_accuracy(), 1.0f);
    // final_accuracy is the best validation accuracy achieved
    EXPECT_GE(final_accuracy, 0.0f);
    EXPECT_LE(final_accuracy, 1.0f);
    EXPECT_EQ(model.get_current_epoch(), 3);
}
