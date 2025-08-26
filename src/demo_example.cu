#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <exception>
#include <cctype>

// NPP Image utilities
#include "ImageIO.h"
#include "ImagesCPU.h"
#include "ImagesNPP.h"
#include "Exceptions.h"

// Error checking macro for CUDA operations
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Type alias for convenience
using GrayImage = npp::ImageCPU_8u_C1;
namespace fs = std::filesystem;

// Supported image file extensions
const std::vector<std::string> SUPPORTED_EXTENSIONS = {
    ".pgm", ".bmp", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".raw"
};

// Function to check if a file has a supported image extension
bool isSupportedImageFile(const std::string& filename) {
    std::string ext = fs::path(filename).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    return std::find(SUPPORTED_EXTENSIONS.begin(), SUPPORTED_EXTENSIONS.end(), ext) 
           != SUPPORTED_EXTENSIONS.end();
}

// Function to recursively find all image files in a directory (like os.walk)
std::vector<std::string> walkDirectory(const std::string& directory_path) {
    std::vector<std::string> image_files;
    
    try {
        if (!fs::exists(directory_path)) {
            std::cerr << "Error: Directory does not exist: " << directory_path << std::endl;
            return image_files;
        }
        
        if (!fs::is_directory(directory_path)) {
            std::cerr << "Error: Path is not a directory: " << directory_path << std::endl;
            return image_files;
        }
        
        // Recursively iterate through all files and subdirectories
        for (const auto& entry : fs::recursive_directory_iterator(directory_path)) {
            if (entry.is_regular_file()) {
                std::string file_path = entry.path().string();
                if (isSupportedImageFile(file_path)) {
                    image_files.push_back(file_path);
                }
            }
        }
        
        // Sort files for consistent ordering
        std::sort(image_files.begin(), image_files.end());
        
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error walking directory: " << e.what() << std::endl;
    }
    
    return image_files;
}

// Function to load image using NPP ImageIO
bool loadImageFile(const std::string& filename, GrayImage& image) {
    try {
        // Use NPP's loadImage function
        npp::loadImage(filename, image);
        
        std::cout << "Reading image using NPP ImageIO:" << std::endl;
        std::cout << "  File: " << filename << std::endl;
        std::cout << "  Dimensions: " << image.width() << "x" << image.height() << std::endl;
        std::cout << "  Total pixels: " << image.width() * image.height() << std::endl;
        
        return true;
    } catch (const npp::Exception& e) {
        std::cerr << "NPP Exception: " << e.message() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Unknown exception occurred while loading image" << std::endl;
        return false;
    }
}

// Function to read raw grayscale image file (fallback for unsupported formats)
bool readRawGray(const std::string& filename, int width, int height, GrayImage& image) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }
    
    // Create a new image with specified dimensions
    GrayImage tempImage(width, height);
    
    file.read(reinterpret_cast<char*>(tempImage.data()), width * height);
    
    if (file.gcount() != width * height) {
        std::cerr << "Error: Could not read expected amount of data from raw file" << std::endl;
        std::cerr << "Expected: " << width * height << " bytes, got: " << file.gcount() << " bytes" << std::endl;
        return false;
    }
    
    file.close();
    
    // Swap with the output image
    image.swap(tempImage);
    
    std::cout << "Reading raw grayscale image:" << std::endl;
    std::cout << "  Dimensions: " << width << "x" << height << std::endl;
    std::cout << "  Total pixels: " << width * height << std::endl;
    
    return true;
}

// Function to print image statistics
void printImageStats(const GrayImage& image) {
    if (!image.data()) {
        std::cerr << "Error: No image data to analyze" << std::endl;
        return;
    }
    
    // Calculate statistics
    int min_val = 255, max_val = 0;
    long long sum = 0;
    
    int total_pixels = image.width() * image.height();
    const Npp8u* pixel_data = image.data();
    
    for (int i = 0; i < total_pixels; i++) {
        int pixel = pixel_data[i];
        min_val = std::min(min_val, pixel);
        max_val = std::max(max_val, pixel);
        sum += pixel;
    }
    
    double mean = static_cast<double>(sum) / total_pixels;
    
    std::cout << "\nImage Statistics:" << std::endl;
    std::cout << "  Min value: " << min_val << std::endl;
    std::cout << "  Max value: " << max_val << std::endl;
    std::cout << "  Mean value: " << std::fixed << std::setprecision(2) << mean << std::endl;
}

// Function to print a sample of pixel values
void printPixelSample(const GrayImage& image, int sample_size = 10) {
    if (!image.data()) {
        std::cerr << "Error: No image data to display" << std::endl;
        return;
    }
    
    std::cout << "\nPixel Sample (first " << sample_size << "x" << sample_size << " region):" << std::endl;
    
    int rows_to_show = std::min(sample_size, static_cast<int>(image.height()));
    int cols_to_show = std::min(sample_size, static_cast<int>(image.width()));
    
    const Npp8u* pixel_data = image.data();
    
    for (int y = 0; y < rows_to_show; y++) {
        std::cout << "Row " << y << ": ";
        for (int x = 0; x < cols_to_show; x++) {
            int pixel_value = pixel_data[y * image.width() + x];
            std::cout << std::setw(3) << pixel_value << " ";
        }
        std::cout << std::endl;
    }
}

// Function to print corner samples
void printCornerSamples(const GrayImage& image, int corner_size = 5) {
    if (!image.data()) {
        std::cerr << "Error: No image data to display" << std::endl;
        return;
    }
    
    std::cout << "\nCorner Samples (" << corner_size << "x" << corner_size << " each):" << std::endl;
    
    const Npp8u* pixel_data = image.data();
    int width = image.width();
    int height = image.height();
    
    // Top-left corner
    std::cout << "\nTop-left corner:" << std::endl;
    for (int y = 0; y < std::min(corner_size, height); y++) {
        for (int x = 0; x < std::min(corner_size, width); x++) {
            std::cout << std::setw(3) << static_cast<int>(pixel_data[y * width + x]) << " ";
        }
        std::cout << std::endl;
    }
    
    // Top-right corner
    if (width > corner_size) {
        std::cout << "\nTop-right corner:" << std::endl;
        for (int y = 0; y < std::min(corner_size, height); y++) {
            for (int x = width - corner_size; x < width; x++) {
                std::cout << std::setw(3) << static_cast<int>(pixel_data[y * width + x]) << " ";
            }
            std::cout << std::endl;
        }
    }
    
    // Bottom-left corner
    if (height > corner_size) {
        std::cout << "\nBottom-left corner:" << std::endl;
        for (int y = height - corner_size; y < height; y++) {
            for (int x = 0; x < std::min(corner_size, width); x++) {
                std::cout << std::setw(3) << static_cast<int>(pixel_data[y * width + x]) << " ";
            }
            std::cout << std::endl;
        }
    }
    
    // Bottom-right corner
    if (width > corner_size && height > corner_size) {
        std::cout << "\nBottom-right corner:" << std::endl;
        for (int y = height - corner_size; y < height; y++) {
            for (int x = width - corner_size; x < width; x++) {
                std::cout << std::setw(3) << static_cast<int>(pixel_data[y * width + x]) << " ";
            }
            std::cout << std::endl;
        }
    }
}

// CUDA kernel to process image on GPU
__global__ void analyzeImageKernel(unsigned char* image, int width, int height, 
                                  int* min_val, int* max_val, long long* sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    
    if (idx >= total_pixels) return;
    
    unsigned char pixel = image[idx];
    
    // Use atomic operations to update global statistics
    atomicMin(min_val, static_cast<int>(pixel));
    atomicMax(max_val, static_cast<int>(pixel));
    atomicAdd((unsigned long long*)sum, static_cast<unsigned long long>(pixel));
}

// Function to demonstrate GPU processing
void demonstrateGPUProcessing(const GrayImage& image) {
    if (!image.data()) {
        std::cerr << "Error: No image data for GPU processing" << std::endl;
        return;
    }
    
    std::cout << "\n=== GPU Processing Demo ===" << std::endl;
    
    // Allocate GPU memory
    unsigned char* d_image;
    int* d_min_val;
    int* d_max_val;
    long long* d_sum;
    
    int width = image.width();
    int height = image.height();
    size_t image_size = width * height * sizeof(unsigned char);
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_image, image_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_min_val, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_max_val, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_sum, sizeof(long long)));
    
    // Copy image to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_image, image.data(), image_size, cudaMemcpyHostToDevice));
    
    // Initialize values
    int initial_min = 255, initial_max = 0;
    long long initial_sum = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(d_min_val, &initial_min, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_max_val, &initial_max, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sum, &initial_sum, sizeof(long long), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int threads_per_block = 256;
    int blocks = (width * height + threads_per_block - 1) / threads_per_block;
    
    std::cout << "Launching GPU kernel with " << blocks << " blocks and " << threads_per_block << " threads per block" << std::endl;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    
    analyzeImageKernel<<<blocks, threads_per_block>>>(d_image, width, height, 
                                                     d_min_val, d_max_val, d_sum);
    
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    
    // Check for kernel errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Get timing
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    
    // Copy results back
    int gpu_min, gpu_max;
    long long gpu_sum;
    CHECK_CUDA_ERROR(cudaMemcpy(&gpu_min, d_min_val, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&gpu_max, d_max_val, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(&gpu_sum, d_sum, sizeof(long long), cudaMemcpyDeviceToHost));
    
    // Display GPU results
    double gpu_mean = static_cast<double>(gpu_sum) / (width * height);
    
    std::cout << "GPU Analysis Results:" << std::endl;
    std::cout << "  Min value: " << gpu_min << std::endl;
    std::cout << "  Max value: " << gpu_max << std::endl;
    std::cout << "  Mean value: " << std::fixed << std::setprecision(2) << gpu_mean << std::endl;
    std::cout << "  Processing time: " << milliseconds << " ms" << std::endl;
    
    // Cleanup
    CHECK_CUDA_ERROR(cudaFree(d_image));
    CHECK_CUDA_ERROR(cudaFree(d_min_val));
    CHECK_CUDA_ERROR(cudaFree(d_max_val));
    CHECK_CUDA_ERROR(cudaFree(d_sum));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
}

// Function to process a single image file
bool processImageFile(const std::string& filename, bool show_detailed_output = true) {
    GrayImage image;
    bool success = false;
    
    if (show_detailed_output) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Processing: " << filename << std::endl;
        std::cout << std::string(60, '=') << std::endl;
    }
    
    // Check file extension
    if (filename.find(".raw") != std::string::npos) {
        // Raw files are more complex - for now, skip them in batch processing
        // unless dimensions are known
        if (show_detailed_output) {
            std::cout << "Skipping raw file (requires width/height): " << filename << std::endl;
        }
        return false;
    } else {
        // Use NPP ImageIO for all other formats
        success = loadImageFile(filename, image);
    }
    
    if (!success) {
        std::cerr << "Failed to read image file: " << filename << std::endl;
        return false;
    }
    
    if (show_detailed_output) {
        std::cout << "Image loaded successfully!" << std::endl;
        
        // Print image statistics (CPU version)
        std::cout << "\n=== CPU Analysis ===" << std::endl;
        printImageStats(image);
        
        // Print pixel samples (smaller sample for batch processing)
        printPixelSample(image, 5);
        
        // Demonstrate GPU processing
        demonstrateGPUProcessing(image);
    } else {
        // Brief output for batch processing
        std::cout << "Processed: " << fs::path(filename).filename().string() 
                  << " (" << image.width() << "x" << image.height() << ")" << std::endl;
    }
    
    return true;
}

void printUsage(const char* program_name) {
    std::cout << "Usage:" << std::endl;
    std::cout << "  " << program_name << " <path> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  path          Path to an image file or directory containing images" << std::endl;
    std::cout << std::endl;
    std::cout << "Options for single raw files:" << std::endl;
    std::cout << "  " << program_name << " <raw_file> <width> <height>" << std::endl;
    std::cout << "    width       Width of the raw image" << std::endl;
    std::cout << "    height      Height of the raw image" << std::endl;
    std::cout << std::endl;
    std::cout << "Options for directories:" << std::endl;
    std::cout << "  --detailed    Show detailed analysis for each image (default: brief)" << std::endl;
    std::cout << std::endl;
    std::cout << "Supported formats (via NPP ImageIO + FreeImage):" << std::endl;
    std::cout << "  .pgm          Portable GrayMap (P2/P5 format)" << std::endl;
    std::cout << "  .bmp, .jpg, .jpeg, .png, .tiff, .tif  Standard image formats" << std::endl;
    std::cout << "  .raw          Raw grayscale data (requires width and height for single files)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " Common/data/teapot512.pgm" << std::endl;
    std::cout << "  " << program_name << " Common/data/teapot_512x512_8u_Gray.raw 512 512" << std::endl;
    std::cout << "  " << program_name << " data/input/train_data/" << std::endl;
    std::cout << "  " << program_name << " data/input/train_data/ --detailed" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== Grayscale Image Reader Demo (using NPP ImageIO) ===" << std::endl;
    
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string path = argv[1];
    bool detailed_output = false;
    
    // Check for --detailed flag
    for (int i = 2; i < argc; i++) {
        if (std::string(argv[i]) == "--detailed") {
            detailed_output = true;
            break;
        }
    }
    
    // Check if path is a directory or file
    if (fs::is_directory(path)) {
        // Process directory
        std::cout << "\nScanning directory: " << path << std::endl;
        
        std::vector<std::string> image_files = walkDirectory(path);
        
        if (image_files.empty()) {
            std::cout << "No supported image files found in directory." << std::endl;
            std::cout << "Supported extensions: ";
            for (size_t i = 0; i < SUPPORTED_EXTENSIONS.size(); i++) {
                std::cout << SUPPORTED_EXTENSIONS[i];
                if (i < SUPPORTED_EXTENSIONS.size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
            return 1;
        }
        
        std::cout << "Found " << image_files.size() << " image file(s)" << std::endl;
        
        int successful = 0;
        int failed = 0;
        
        for (const auto& file : image_files) {
            if (processImageFile(file, detailed_output)) {
                successful++;
            } else {
                failed++;
            }
        }
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Batch Processing Summary:" << std::endl;
        std::cout << "  Total files: " << image_files.size() << std::endl;
        std::cout << "  Successful: " << successful << std::endl;
        std::cout << "  Failed: " << failed << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
    } else if (fs::is_regular_file(path)) {
        // Process single file
        std::string filename = path;
        
        // Check if it's a raw file that needs dimensions
        if (filename.find(".raw") != std::string::npos) {
            if (argc < 4) {
                std::cerr << "Error: Raw files require width and height parameters" << std::endl;
                printUsage(argv[0]);
                return 1;
            }
            
            int width = std::atoi(argv[2]);
            int height = std::atoi(argv[3]);
            
            if (width <= 0 || height <= 0) {
                std::cerr << "Error: Invalid width or height values" << std::endl;
                return 1;
            }
            
            GrayImage image;
            bool success = readRawGray(filename, width, height, image);
            
            if (!success) {
                std::cerr << "Failed to read raw image file: " << filename << std::endl;
                return 1;
            }
            
            std::cout << "\nImage loaded successfully!" << std::endl;
            
            // Print image statistics (CPU version)
            std::cout << "\n=== CPU Analysis ===" << std::endl;
            printImageStats(image);
            
            // Print pixel samples
            printPixelSample(image, 8);
            printCornerSamples(image, 5);
            
            // Demonstrate GPU processing
            demonstrateGPUProcessing(image);
            
        } else {
            // Regular image file
            if (!processImageFile(filename, true)) {
                return 1;
            }
        }
        
    } else {
        std::cerr << "Error: Path does not exist or is not a file/directory: " << path << std::endl;
        return 1;
    }
    
    std::cout << "\n=== Demo completed successfully! ===" << std::endl;
    
    return 0;
}
