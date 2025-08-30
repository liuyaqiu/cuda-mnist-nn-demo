#include "data_loader.h"
#include "ImageIO.h"
#include "ImagesCPU.h"
#include "ImagesNPP.h"
#include "Exceptions.h"
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <random>
#include <cstring>

namespace fs = std::filesystem;

DataLoader::DataLoader(const std::string& input_path, float validation_ratio)
    : input_path_(input_path), validation_ratio_(validation_ratio),
      image_width_(0), image_height_(0), image_channels_(1),
      training_iterator_(0), validation_iterator_(0),
      rng_(std::random_device{}()) {
    
    // Clamp validation ratio to valid range
    validation_ratio_ = std::max(0.0f, std::min(1.0f, validation_ratio_));
    
    std::cout << "DataLoader initialized with input_path: " << input_path_ 
              << ", validation_ratio: " << validation_ratio_ << std::endl;
}

DataLoader::~DataLoader() {
    // No cleanup needed since we don't store image data in memory
}

void DataLoader::initialize_input() {
    std::cout << "Discovering image files in: " << input_path_ << std::endl;
    
    // Clear existing data
    images_.clear();
    training_indices_.clear();
    validation_indices_.clear();
    
    // Discover all image files
    discover_image_files();
    
    if (images_.empty()) {
        std::cerr << "No image files found in: " << input_path_ << std::endl;
        return;
    }
    
    std::cout << "Found " << images_.size() << " image files" << std::endl;
    
    // Load the first image to determine dimensions
    if (!images_.empty()) {
        try {
            int width, height, channels;
            float* temp_buffer = new float[28 * 28]; // Assume max size for temp buffer
            load_image_to_buffer(images_[0].filepath, temp_buffer, width, height, channels);
            
            image_width_ = width;
            image_height_ = height;
            image_channels_ = channels;
            
            std::cout << "Image dimensions: " << image_width_ << "x" << image_height_ 
                      << "x" << image_channels_ << std::endl;
            
            delete[] temp_buffer;
            
        } catch (const std::exception& e) {
            std::cerr << "Failed to determine image dimensions from: " << images_[0].filepath << std::endl;
            return;
        }
    }
    
    // All images assumed to have uniform dimensions (stored in class members)
    
    // Split dataset into training and validation
    split_dataset();
    
    // Shuffle training data
    shuffle_training_data();
    
    std::cout << "Dataset initialized with " << get_training_size() 
              << " training images and " << get_validation_size() 
              << " validation images" << std::endl;
}



void DataLoader::get(int batch_size, float* host_mem, bool is_validation, float* labels) {
    if (images_.empty()) {
        std::cerr << "Error: No images found. Call initialize_input() first." << std::endl;
        return;
    }
    
    const std::vector<int>& indices = is_validation ? validation_indices_ : training_indices_;
    int& iterator = is_validation ? validation_iterator_ : training_iterator_;
    
    if (indices.empty()) {
        std::cerr << "Error: No " << (is_validation ? "validation" : "training") 
                  << " images available." << std::endl;
        return;
    }
    
    // Check if we have enough images for the batch
    int available_images = static_cast<int>(indices.size()) - iterator;
    if (available_images <= 0) {
        std::cout << "No more " << (is_validation ? "validation" : "training") 
                  << " batches available. Resetting iterator." << std::endl;
        iterator = 0;
        available_images = static_cast<int>(indices.size());
    }
    
    int actual_batch_size = std::min(batch_size, available_images);
    int image_elements = get_image_elements();
    
    // Load images on-demand
    for (int i = 0; i < actual_batch_size; ++i) {
        int img_idx = indices[iterator + i];
        const ImageMeta& image_meta = images_[img_idx];
        
        // Calculate memory offset for this image in the batch
        float* image_buffer = host_mem + i * image_elements;
        
        // Load image directly into the batch buffer
        int width, height, channels;
        load_image_to_buffer(image_meta.filepath, image_buffer, width, height, channels);
        
        // Parse label from filepath and convert to one-hot encoding
        int label_class = parse_label(image_meta);
        float* label_buffer = labels + i * LABEL_CATEGORIES;
        
        // Create one-hot encoding: set all to 0.0, then set correct class to 1.0
        std::fill(label_buffer, label_buffer + LABEL_CATEGORIES, 0.0f);
        label_buffer[label_class] = 1.0f;
    }
    
    // Update iterator
    iterator += actual_batch_size;
    
    // Fill remaining images with zeros if batch is smaller than requested
    if (actual_batch_size < batch_size) {
        for (int i = actual_batch_size; i < batch_size; ++i) {
            float* image_buffer = host_mem + i * image_elements;
            std::fill(image_buffer, image_buffer + image_elements, 0.0f);
            
            // Fill label with zeros (all classes = 0.0)
            float* label_buffer = labels + i * LABEL_CATEGORIES;
            std::fill(label_buffer, label_buffer + LABEL_CATEGORIES, 0.0f);
        }
    }
}

int DataLoader::get_training_size() const {
    return static_cast<int>(training_indices_.size());
}

int DataLoader::get_validation_size() const {
    return static_cast<int>(validation_indices_.size());
}

void DataLoader::reset_training_iterator() {
    training_iterator_ = 0;
    shuffle_training_data();
}

void DataLoader::reset_validation_iterator() {
    validation_iterator_ = 0;
}

bool DataLoader::has_more_training_batches(int batch_size) const {
    return training_iterator_ < static_cast<int>(training_indices_.size());
}

bool DataLoader::has_more_validation_batches(int batch_size) const {
    return validation_iterator_ < static_cast<int>(validation_indices_.size());
}

// Private helper methods implementation

void DataLoader::load_image_to_buffer(const std::string& filepath, float* buffer, int& width, int& height, int& channels) {
    try {
        // Load image using NPP ImageIO
        npp::ImageCPU_8u_C1 npp_image;
        npp::loadImage(filepath, npp_image);
        
        width = static_cast<int>(npp_image.width());
        height = static_cast<int>(npp_image.height());
        channels = 1; // Grayscale
        
        // Convert NPP 8u data to float and normalize to [0, 1]
        const Npp8u* npp_data = npp_image.data();
        unsigned int npp_pitch = npp_image.pitch();
        
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int dst_idx = y * width + x;
                int src_idx = y * (npp_pitch / sizeof(Npp8u)) + x;
                buffer[dst_idx] = static_cast<float>(npp_data[src_idx]) / 255.0f;
            }
        }
        
    } catch (const npp::Exception& e) {
        throw std::runtime_error("Failed to load image " + filepath + ": " + e.message());
    }
}

int DataLoader::parse_label(const ImageMeta& image_meta) {
    // Extract label from directory structure
    // For format: data/input/train_data/2/image_00007.png -> label is "2"
    fs::path path(image_meta.filepath);
    
    // Get the parent directory name (which should be the label)
    std::string parent_dir = path.parent_path().filename().string();
    
    try {
        int label = std::stoi(parent_dir);
        // Validate label is in reasonable range (0 to LABEL_CATEGORIES-1 for MNIST)
        if (label >= 0 && label < LABEL_CATEGORIES) {
            return label;
        } else {
            throw std::runtime_error("Invalid label " + std::to_string(label) + 
                                   " (must be 0-" + std::to_string(LABEL_CATEGORIES-1) + 
                                   ") in directory structure for file: " + image_meta.filepath);
        }
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("Cannot parse label from directory name '" + parent_dir + 
                               "' in file path: " + image_meta.filepath + 
                               ". Expected format: .../[0-" + std::to_string(LABEL_CATEGORIES-1) + "]/filename.png");
    } catch (const std::out_of_range&) {
        throw std::runtime_error("Label number out of range in directory name '" + parent_dir + 
                               "' in file path: " + image_meta.filepath);
    }
}

void DataLoader::split_dataset() {
    training_indices_.clear();
    validation_indices_.clear();
    
    int total_images = static_cast<int>(images_.size());
    int validation_count = static_cast<int>(total_images * validation_ratio_);
    
    // Create indices for all images
    std::vector<int> all_indices(total_images);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    
    // Shuffle indices for random split
    std::shuffle(all_indices.begin(), all_indices.end(), rng_);
    
    // Split into validation and training
    validation_indices_.assign(all_indices.begin(), all_indices.begin() + validation_count);
    training_indices_.assign(all_indices.begin() + validation_count, all_indices.end());
}

void DataLoader::shuffle_training_data() {
    std::shuffle(training_indices_.begin(), training_indices_.end(), rng_);
}

void DataLoader::discover_image_files() {
    if (!fs::exists(input_path_) || !fs::is_directory(input_path_)) {
        std::cerr << "Error: Input path does not exist or is not a directory: " 
                  << input_path_ << std::endl;
        return;
    }
    
    for (const auto& entry : fs::recursive_directory_iterator(input_path_)) {
        if (entry.is_regular_file() && is_image_file(entry.path().string())) {
            ImageMeta img_meta;
            img_meta.filepath = entry.path().string();
            images_.push_back(img_meta);
        }
    }
    
    // Sort by filepath for consistent ordering
    std::sort(images_.begin(), images_.end(), 
              [](const ImageMeta& a, const ImageMeta& b) {
                  return a.filepath < b.filepath;
              });
}

bool DataLoader::is_image_file(const std::string& filename) const {
    std::string ext = get_file_extension(filename);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    // Support formats that FreeImage/NPP can handle
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || 
           ext == ".bmp" || ext == ".tiff" || ext == ".tif" ||
           ext == ".pgm" || ext == ".ppm" || ext == ".raw";
}



std::string DataLoader::get_file_extension(const std::string& filename) const {
    size_t dot_pos = filename.find_last_of('.');
    if (dot_pos != std::string::npos) {
        return filename.substr(dot_pos);
    }
    return "";
}


