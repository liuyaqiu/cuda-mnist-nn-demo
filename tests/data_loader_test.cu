#include <gtest/gtest.h>
#include "data_loader.h"
#include <filesystem>
#include <fstream>
#include <FreeImage.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

namespace fs = std::filesystem;

// Test fixture for DataLoader tests
class DataLoaderTest : public ::testing::Test {
protected:
    // Test data directory paths
    std::string test_data_dir_;
    std::string empty_dir_;
    std::string invalid_dir_;
    
    // Setup method called before each test
    void SetUp() override {
        // Create test directories
        test_data_dir_ = "test_data_temp";
        empty_dir_ = "empty_test_dir";
        invalid_dir_ = "non_existent_dir";
        
        // Create test directory structure
        fs::create_directories(test_data_dir_);
        fs::create_directories(empty_dir_);
        
        // Create mock image directory structure with labels
        for (int label = 0; label < 3; ++label) {
            fs::create_directories(test_data_dir_ + "/" + std::to_string(label));
        }
        
        // Initialize FreeImage
        FreeImage_Initialise();
        
        // Create test images
        createTestImages();
    }
    
    // Teardown method called after each test
    void TearDown() override {
        // Clean up test directories
        if (fs::exists(test_data_dir_)) {
            fs::remove_all(test_data_dir_);
        }
        if (fs::exists(empty_dir_)) {
            fs::remove_all(empty_dir_);
        }
        
        // De-initialize FreeImage
        FreeImage_DeInitialise();
    }
    
    // Helper function to create test images
    void createTestImages() {
        const int width = 28;
        const int height = 28;
        const int bpp = 8; // bits per pixel for grayscale
        
        // Create 3 images per label (total 9 images)
        for (int label = 0; label < 3; ++label) {
            for (int img_idx = 0; img_idx < 3; ++img_idx) {
                // Create a simple test pattern for each image
                FIBITMAP* bitmap = FreeImage_Allocate(width, height, bpp);
                if (!bitmap) {
                    FAIL() << "Failed to allocate bitmap";
                }
                
                // Fill with a pattern unique to each image
                for (int y = 0; y < height; ++y) {
                    BYTE* scanline = FreeImage_GetScanLine(bitmap, y);
                    for (int x = 0; x < width; ++x) {
                        // Create a gradient pattern with label influence
                        BYTE value = static_cast<BYTE>((x + y + label * 50 + img_idx * 30) % 256);
                        scanline[x] = value;
                    }
                }
                
                // Save the image
                std::string filename = test_data_dir_ + "/" + std::to_string(label) + 
                                     "/test_image_" + std::to_string(img_idx) + ".png";
                
                if (!FreeImage_Save(FIF_PNG, bitmap, filename.c_str())) {
                    FreeImage_Unload(bitmap);
                    FAIL() << "Failed to save test image: " << filename;
                }
                
                FreeImage_Unload(bitmap);
            }
        }
    }
    
    // Helper function to create a single test image
    void createSingleTestImage(const std::string& filepath, int width, int height, BYTE fill_value) {
        FIBITMAP* bitmap = FreeImage_Allocate(width, height, 8);
        if (!bitmap) {
            FAIL() << "Failed to allocate bitmap";
        }
        
        for (int y = 0; y < height; ++y) {
            BYTE* scanline = FreeImage_GetScanLine(bitmap, y);
            std::fill(scanline, scanline + width, fill_value);
        }
        
        if (!FreeImage_Save(FIF_PNG, bitmap, filepath.c_str())) {
            FreeImage_Unload(bitmap);
            FAIL() << "Failed to save test image: " << filepath;
        }
        
        FreeImage_Unload(bitmap);
    }
};

// Test DataLoader constructor
TEST_F(DataLoaderTest, ConstructorValidation) {
    // Test with valid parameters
    DataLoader loader1(test_data_dir_, 0.2f);
    EXPECT_EQ(loader1.get_total_size(), 0); // Not initialized yet
    
    // Test with edge case validation ratios
    DataLoader loader2(test_data_dir_, 0.0f); // No validation
    DataLoader loader3(test_data_dir_, 1.0f); // All validation
    DataLoader loader4(test_data_dir_, -0.1f); // Should clamp to 0.0
    DataLoader loader5(test_data_dir_, 1.5f); // Should clamp to 1.0
}

// Test image discovery
TEST_F(DataLoaderTest, ImageDiscovery) {
    DataLoader loader(test_data_dir_, 0.2f);
    loader.initialize_input();
    
    // Should find 9 images (3 labels × 3 images each)
    EXPECT_EQ(loader.get_total_size(), 9);
    
    // Check image dimensions
    EXPECT_EQ(loader.get_image_width(), 28);
    EXPECT_EQ(loader.get_image_height(), 28);
    EXPECT_EQ(loader.get_image_channels(), 1);
    EXPECT_EQ(loader.get_image_elements(), 28 * 28 * 1);
}

// Test empty directory handling
TEST_F(DataLoaderTest, EmptyDirectory) {
    DataLoader loader(empty_dir_, 0.2f);
    loader.initialize_input();
    
    EXPECT_EQ(loader.get_total_size(), 0);
    EXPECT_EQ(loader.get_training_size(), 0);
    EXPECT_EQ(loader.get_validation_size(), 0);
}

// Test non-existent directory handling
TEST_F(DataLoaderTest, NonExistentDirectory) {
    DataLoader loader(invalid_dir_, 0.2f);
    loader.initialize_input();
    
    EXPECT_EQ(loader.get_total_size(), 0);
}

// Test dataset splitting
TEST_F(DataLoaderTest, DatasetSplitting) {
    // Test various validation ratios
    std::vector<float> ratios = {0.0f, 0.2f, 0.5f, 0.8f, 1.0f};
    
    for (float ratio : ratios) {
        DataLoader loader(test_data_dir_, ratio);
        loader.initialize_input();
        
        int total = loader.get_total_size();
        int validation = loader.get_validation_size();
        int training = loader.get_training_size();
        
        // Check split correctness
        EXPECT_EQ(training + validation, total);
        
        // Check validation ratio (allowing for rounding)
        if (ratio == 0.0f) {
            EXPECT_EQ(validation, 0);
        } else if (ratio == 1.0f) {
            EXPECT_EQ(training, 0);
        } else {
            float actual_ratio = static_cast<float>(validation) / total;
            EXPECT_NEAR(actual_ratio, ratio, 0.15f); // Allow some tolerance due to rounding
        }
    }
}

// Test batch loading
TEST_F(DataLoaderTest, BatchLoading) {
    DataLoader loader(test_data_dir_, 0.3f);
    loader.initialize_input();
    
    const int batch_size = 4;
    const int image_elements = loader.get_image_elements();
    
    // Allocate memory for batch
    std::vector<float> batch_data(batch_size * image_elements);
    std::vector<float> batch_labels(batch_size * LABEL_CATEGORIES);
    
    // Test training batch
    loader.get(batch_size, batch_data.data(), false, batch_labels.data());
    
    // Verify data is loaded (not all zeros)
    bool has_non_zero = false;
    for (float val : batch_data) {
        if (val != 0.0f) {
            has_non_zero = true;
            break;
        }
    }
    EXPECT_TRUE(has_non_zero);
    
    // Verify labels are one-hot encoded
    for (int i = 0; i < batch_size; ++i) {
        float* label = batch_labels.data() + i * LABEL_CATEGORIES;
        float sum = 0.0f;
        int non_zero_count = 0;
        
        for (int j = 0; j < LABEL_CATEGORIES; ++j) {
            sum += label[j];
            if (label[j] != 0.0f) {
                EXPECT_FLOAT_EQ(label[j], 1.0f); // Should be exactly 1.0
                non_zero_count++;
            }
        }
        
        // Each label should have exactly one 1.0 and sum to 1.0
        EXPECT_EQ(non_zero_count, 1);
        EXPECT_FLOAT_EQ(sum, 1.0f);
    }
}

// Test data normalization
TEST_F(DataLoaderTest, DataNormalization) {
    DataLoader loader(test_data_dir_, 0.0f);
    loader.initialize_input();
    
    const int batch_size = 1;
    const int image_elements = loader.get_image_elements();
    
    std::vector<float> batch_data(batch_size * image_elements);
    std::vector<float> batch_labels(batch_size * LABEL_CATEGORIES);
    
    loader.get(batch_size, batch_data.data(), false, batch_labels.data());
    
    // Check all values are in [0, 1] range
    for (float val : batch_data) {
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 1.0f);
    }
}

// Test iterator functionality
TEST_F(DataLoaderTest, IteratorBehavior) {
    DataLoader loader(test_data_dir_, 0.3f);
    loader.initialize_input();
    
    const int batch_size = 2;
    const int image_elements = loader.get_image_elements();
    
    std::vector<float> batch_data(batch_size * image_elements);
    std::vector<float> batch_labels(batch_size * LABEL_CATEGORIES);
    
    // Count total batches we can get
    int training_batches = 0;
    while (loader.has_more_training_batches(batch_size)) {
        loader.get(batch_size, batch_data.data(), false, batch_labels.data());
        training_batches++;
        
        // Prevent infinite loop in case of bug
        if (training_batches > 100) break;
    }
    
    // Verify we got expected number of batches
    int expected_batches = (loader.get_training_size() + batch_size - 1) / batch_size;
    EXPECT_EQ(training_batches, expected_batches);
    
    // Test reset functionality
    loader.reset_training_iterator();
    EXPECT_TRUE(loader.has_more_training_batches(batch_size));
}

// Test validation data loading
TEST_F(DataLoaderTest, ValidationDataLoading) {
    DataLoader loader(test_data_dir_, 0.4f);
    loader.initialize_input();
    
    const int batch_size = 2;
    const int image_elements = loader.get_image_elements();
    
    std::vector<float> batch_data(batch_size * image_elements);
    std::vector<float> batch_labels(batch_size * LABEL_CATEGORIES);
    
    // Test validation batch loading
    if (loader.get_validation_size() > 0) {
        loader.get(batch_size, batch_data.data(), true, batch_labels.data());
        
        // Verify data is loaded
        bool has_non_zero = false;
        for (float val : batch_data) {
            if (val != 0.0f) {
                has_non_zero = true;
                break;
            }
        }
        EXPECT_TRUE(has_non_zero);
    }
}

// Test partial batch handling
TEST_F(DataLoaderTest, PartialBatchHandling) {
    DataLoader loader(test_data_dir_, 0.0f);
    loader.initialize_input();
    
    // Request batch larger than dataset
    const int batch_size = 20; // We only have 9 images
    const int image_elements = loader.get_image_elements();
    
    std::vector<float> batch_data(batch_size * image_elements);
    std::vector<float> batch_labels(batch_size * LABEL_CATEGORIES);
    
    loader.get(batch_size, batch_data.data(), false, batch_labels.data());
    
    // First 9 should have data, rest should be zeros
    // Check that images after the 9th are all zeros
    for (int i = 9; i < batch_size; ++i) {
        float* img_start = batch_data.data() + i * image_elements;
        float* img_end = img_start + image_elements;
        
        bool all_zeros = std::all_of(img_start, img_end, 
                                     [](float val) { return val == 0.0f; });
        EXPECT_TRUE(all_zeros);
        
        // Check labels are also all zeros
        float* label_start = batch_labels.data() + i * LABEL_CATEGORIES;
        float* label_end = label_start + LABEL_CATEGORIES;
        
        all_zeros = std::all_of(label_start, label_end,
                               [](float val) { return val == 0.0f; });
        EXPECT_TRUE(all_zeros);
    }
}

// Test label parsing
TEST_F(DataLoaderTest, LabelParsing) {
    DataLoader loader(test_data_dir_, 0.0f);
    loader.initialize_input();
    
    const int batch_size = 9; // Get all images
    const int image_elements = loader.get_image_elements();
    
    std::vector<float> batch_data(batch_size * image_elements);
    std::vector<float> batch_labels(batch_size * LABEL_CATEGORIES);
    
    loader.get(batch_size, batch_data.data(), false, batch_labels.data());
    
    // Collect all labels
    std::vector<int> parsed_labels;
    for (int i = 0; i < batch_size; ++i) {
        float* label = batch_labels.data() + i * LABEL_CATEGORIES;
        for (int j = 0; j < LABEL_CATEGORIES; ++j) {
            if (label[j] == 1.0f) {
                parsed_labels.push_back(j);
                break;
            }
        }
    }
    
    // We should have labels 0, 1, 2 (each appearing 3 times)
    std::sort(parsed_labels.begin(), parsed_labels.end());
    
    int count_0 = std::count(parsed_labels.begin(), parsed_labels.end(), 0);
    int count_1 = std::count(parsed_labels.begin(), parsed_labels.end(), 1);
    int count_2 = std::count(parsed_labels.begin(), parsed_labels.end(), 2);
    
    EXPECT_EQ(count_0, 3);
    EXPECT_EQ(count_1, 3);
    EXPECT_EQ(count_2, 3);
}

// Test shuffling behavior
TEST_F(DataLoaderTest, ShufflingBehavior) {
    DataLoader loader(test_data_dir_, 0.0f);
    loader.initialize_input();
    
    const int batch_size = loader.get_training_size();
    const int image_elements = loader.get_image_elements();
    
    // Get first epoch
    std::vector<float> epoch1_data(batch_size * image_elements);
    std::vector<float> epoch1_labels(batch_size * LABEL_CATEGORIES);
    loader.get(batch_size, epoch1_data.data(), false, epoch1_labels.data());
    
    // Reset and get second epoch
    loader.reset_training_iterator();
    std::vector<float> epoch2_data(batch_size * image_elements);
    std::vector<float> epoch2_labels(batch_size * LABEL_CATEGORIES);
    loader.get(batch_size, epoch2_data.data(), false, epoch2_labels.data());
    
    // Data should be shuffled (very unlikely to be in same order)
    bool is_different = false;
    for (size_t i = 0; i < epoch1_data.size(); ++i) {
        if (epoch1_data[i] != epoch2_data[i]) {
            is_different = true;
            break;
        }
    }
    
    // With 9 images, there's a 1/9! = 1/362880 chance they're in the same order
    // So this test should almost always pass
    EXPECT_TRUE(is_different);
}

// Test with different image formats
TEST_F(DataLoaderTest, DifferentImageFormats) {
    // Create a temporary directory for format tests
    std::string format_test_dir = "format_test_dir";
    fs::create_directories(format_test_dir + "/0");
    
    // Test different supported formats
    std::vector<std::pair<std::string, FREE_IMAGE_FORMAT>> formats = {
        {".bmp", FIF_BMP},
        {".jpg", FIF_JPEG},
        {".png", FIF_PNG}
    };
    
    // Create test images in different formats
    for (size_t i = 0; i < formats.size(); ++i) {
        FIBITMAP* bitmap = FreeImage_Allocate(16, 16, 8);
        if (!bitmap) continue;
        
        // Fill with some data
        for (int y = 0; y < 16; ++y) {
            BYTE* scanline = FreeImage_GetScanLine(bitmap, y);
            std::fill(scanline, scanline + 16, static_cast<BYTE>(i * 50));
        }
        
        std::string filename = format_test_dir + "/0/test" + formats[i].first;
        FreeImage_Save(formats[i].second, bitmap, filename.c_str());
        FreeImage_Unload(bitmap);
    }
    
    // Test loading
    DataLoader loader(format_test_dir, 0.0f);
    loader.initialize_input();
    
    // Should find all format files
    EXPECT_EQ(loader.get_total_size(), formats.size());
    
    // Cleanup
    fs::remove_all(format_test_dir);
}

// Test error handling for corrupted images
TEST_F(DataLoaderTest, CorruptedImageHandling) {
    // Create a directory with a corrupted "image" file
    std::string corrupt_test_dir = "corrupt_test_dir";
    fs::create_directories(corrupt_test_dir + "/0");
    
    // Create a fake PNG file with invalid data
    std::string corrupt_file = corrupt_test_dir + "/0/corrupt.png";
    std::ofstream ofs(corrupt_file, std::ios::binary);
    ofs << "This is not a valid PNG file";
    ofs.close();
    
    DataLoader loader(corrupt_test_dir, 0.0f);
    loader.initialize_input();
    
    // The corrupted file might not be properly initialized
    // The DataLoader may fail to determine dimensions from the corrupted file
    // In this case, get_total_size() could be 0 or the loader might not have valid dimensions
    
    // If the loader couldn't initialize properly, it should have 0 images
    // or if it found the file but couldn't read it, initialization would fail
    if (loader.get_total_size() == 0 || loader.get_image_width() == 0) {
        // This is expected behavior - loader couldn't initialize with corrupted image
        SUCCEED();
    } else {
        // If somehow the loader initialized, test that get() handles errors gracefully
        const int batch_size = 1;
        const int image_elements = loader.get_image_elements();
        
        std::vector<float> batch_data(batch_size * image_elements);
        std::vector<float> batch_labels(batch_size * LABEL_CATEGORIES);
        
        // The get method should handle the error gracefully without crashing
        // It may print error messages but shouldn't throw
        EXPECT_NO_THROW({
            loader.get(batch_size, batch_data.data(), false, batch_labels.data());
        });
    }
    
    // Cleanup
    fs::remove_all(corrupt_test_dir);
}

// Performance test for large batch loading
TEST_F(DataLoaderTest, LargeBatchPerformance) {
    DataLoader loader(test_data_dir_, 0.0f);
    loader.initialize_input();
    
    const int large_batch_size = 100;
    const int image_elements = loader.get_image_elements();
    
    std::vector<float> batch_data(large_batch_size * image_elements);
    std::vector<float> batch_labels(large_batch_size * LABEL_CATEGORIES);
    
    // Measure time for loading
    auto start = std::chrono::high_resolution_clock::now();
    loader.get(large_batch_size, batch_data.data(), false, batch_labels.data());
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Just ensure it completes without hanging
    // Actual performance requirements would depend on system
    EXPECT_LT(duration.count(), 5000); // Should complete within 5 seconds
}
