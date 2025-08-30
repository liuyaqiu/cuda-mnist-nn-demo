#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <cstdint>
#include <random>

// Constants
#define LABEL_CATEGORIES 10  // Number of classes for MNIST (0-9)

// Error handling macros for DataLoader
#define HANDLE_CUDA_ERROR_DATALOADER(x)                                      \
{ const auto err = x;                                             \
    if( err != cudaSuccess )                                        \
    { printf("CUDA Error in DataLoader %s at %s:%d: %s\n", __func__, __FILE__, __LINE__, cudaGetErrorString(err)); exit(-1); } \
};

struct ImageMeta {
    std::string filepath;
};

class DataLoader {
public:
    // Constructor
    // input_path: path to directory containing images
    // validation_ratio: ratio of data to use for validation (0.0 to 1.0)
    DataLoader(const std::string& input_path, float validation_ratio = 0.0f);
    
    // Destructor
    ~DataLoader();
    
    // Initialize input: iterates all files under the directory input_path
    // Discovers all image files and determines dataset structure
    void initialize_input();
    
    // Get batch of images and labels
    // batch_size: number of images to load
    // host_mem: caller-provided host memory buffer (must be at least batch_size * image_elements * sizeof(float))
    // is_validation: whether to get images from validation set (true) or training set (false)
    // labels: caller-provided buffer for one-hot encoded labels (must be at least batch_size * LABEL_CATEGORIES * sizeof(float))
    // Note: Images are loaded on-demand and normalized to [0,1]. Labels are one-hot encoded.
    void get(int batch_size, float* host_mem, bool is_validation, float* labels);
    
    // Getters for dataset information
    int get_training_size() const;
    int get_validation_size() const;
    int get_total_size() const { return images_.size(); }
    int get_image_width() const { return image_width_; }
    int get_image_height() const { return image_height_; }
    int get_image_channels() const { return image_channels_; }
    int get_image_elements() const { return image_width_ * image_height_ * image_channels_; }
    
    // Reset batch iterators to beginning
    void reset_training_iterator();
    void reset_validation_iterator();
    
    // Check if more batches are available
    bool has_more_training_batches(int batch_size) const;
    bool has_more_validation_batches(int batch_size) const;

private:
    std::string input_path_;
    float validation_ratio_;
    
    // Dataset storage
    std::vector<ImageMeta> images_;
    std::vector<int> training_indices_;
    std::vector<int> validation_indices_;
    
    // Image dimensions (assumed uniform across dataset)
    int image_width_;
    int image_height_;
    int image_channels_;
    
    // Batch iterators
    int training_iterator_;
    int validation_iterator_;
    
    // Random number generator for shuffling
    std::mt19937 rng_;
    
    // Helper methods
    void load_image_to_buffer(const std::string& filepath, float* buffer, int& width, int& height, int& channels);
    int parse_label(const ImageMeta& image_meta);
    void split_dataset();
    void shuffle_training_data();
    void discover_image_files();
    bool is_image_file(const std::string& filename) const;
    std::string get_file_extension(const std::string& filename) const;
};

#endif // DATA_LOADER_H
