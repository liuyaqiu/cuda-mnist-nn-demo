################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
################################################################################
#
# Makefile project only supported on Mac OS X and Linux Platforms)
#
################################################################################

# Define the compiler and flags
NVCC = /usr/local/cuda/bin/nvcc
CXX = g++

# Auto-detect GPU architecture
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1 | sed 's/\.//g')
ARCH_FLAG := -arch=sm_$(GPU_ARCH)

# CUDA compilation flags - added support for device code compilation
NVCCFLAGS = $(ARCH_FLAG) --std=c++17 \
	-I/usr/local/cuda/include \
	-Iinclude -ICommon -ICommon/UtilNPP -ICommon/GL -I$(LIB_DIR) \
	--relocatable-device-code=true \
	--compile

# Google Test configuration
GTEST_VERSION = 1.14.0
GTEST_DIR = third_party/googletest
GTEST_BUILD_DIR = $(GTEST_DIR)/build
GTEST_INSTALL_DIR = $(GTEST_DIR)/install
GTEST_INCLUDE = $(GTEST_INSTALL_DIR)/include
GTEST_LIB = $(GTEST_INSTALL_DIR)/lib

# Google Test compilation flags for test files (NVCC compatible)
GTEST_FLAGS = -I$(GTEST_INCLUDE)

# Additional flags for linking CUDA device code
NVCC_LINK_FLAGS = $(ARCH_FLAG) --std=c++17 \
	-I/usr/local/cuda/include \
	-Iinclude -ICommon -ICommon/UtilNPP -ICommon/GL -I$(LIB_DIR)

CXXFLAGS = -std=c++17 -I/usr/local/cuda/include -Iinclude -ICommon -ICommon/UtilNPP -ICommon/GL -I$(LIB_DIR)

# Updated library flags - added math library for CUDA math functions
LDFLAGS = -L/usr/local/cuda/lib64 -L$(LIB_DIR) -LCommon/lib \
	-lcudart -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc \
	-lfreeimage -lcutensor -lm

# Google Test library flags
GTEST_LIBS = -L$(GTEST_LIB) -lgtest -lgtest_main -lpthread

# Define directories
SRC_DIR = src
BIN_DIR = bin
OBJ_DIR = obj
LIB_DIR = lib
DATA_DIR = data
TESTS_DIR = tests
INPUT_DIR = data/input
OUTPUT_DIR = data/output

# Training data download parameters
MNIST_TRAIN_IMAGES_URL = http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
MNIST_TRAIN_LABELS_URL = http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
MNIST_TEST_IMAGES_URL = http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
MNIST_TEST_LABELS_URL = http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

# MNIST checksums (MD5)
MNIST_TRAIN_IMAGES_CHECKSUM = md5::8d4fb7e6c68d591d4c3dfef9ec88bf0d
MNIST_TRAIN_LABELS_CHECKSUM = md5::25c81989df183df01b3e8a0aad5dffbe
MNIST_TEST_IMAGES_CHECKSUM = md5::bef4ecab320f06d8554ea6380940ec79
MNIST_TEST_LABELS_CHECKSUM = md5::bb300cfdad3c16e7a12a480ee83cd310

# Download script
DOWNLOAD_SCRIPT = scripts/download_artifacts.sh

# Define source files and target executables
SRC_CUTENSOR = $(SRC_DIR)/cutensor_example.cu
SRC_DEMO = $(SRC_DIR)/demo_example.cu
SRC_NN_UTILS = $(LIB_DIR)/nn_utils.cu
SRC_DATA_LOADER = $(LIB_DIR)/data_loader.cu
SRC_NN_UTILS_TEST = $(TESTS_DIR)/nn_utils_test.cu
SRC_DATA_LOADER_TEST = $(TESTS_DIR)/data_loader_test.cu

TARGET_CUTENSOR = $(BIN_DIR)/cutensor_example
TARGET_DEMO = $(BIN_DIR)/demo_example
TARGET_NN_UTILS_TEST = $(BIN_DIR)/nn_utils_test
TARGET_DATA_LOADER_TEST = $(BIN_DIR)/data_loader_test

OBJ_CUTENSOR = $(OBJ_DIR)/cutensor_example.o
OBJ_DEMO = $(OBJ_DIR)/demo_example.o
OBJ_NN_UTILS = $(OBJ_DIR)/nn_utils.o
OBJ_DATA_LOADER = $(OBJ_DIR)/data_loader.o
OBJ_TEST_DATA_LOADER = $(OBJ_DIR)/test_data_loader.o
OBJ_NN_UTILS_TEST = $(OBJ_DIR)/nn_utils_test.o
OBJ_DATA_LOADER_TEST = $(OBJ_DIR)/data_loader_test.o

# Build targets
build-cutensor-example: $(TARGET_CUTENSOR)
build-demo-example: $(TARGET_DEMO)
build-nn-utils: $(OBJ_NN_UTILS)
build-data-loader: $(OBJ_DATA_LOADER)
build-nn-utils-test: $(TARGET_NN_UTILS_TEST)
build-data-loader-test: $(TARGET_DATA_LOADER_TEST)
build-all: $(TARGET_CUTENSOR) $(TARGET_DEMO) $(OBJ_DATA_LOADER) $(TARGET_NN_UTILS_TEST) $(TARGET_DATA_LOADER_TEST)

# Create necessary directories
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(DATA_DIR):
	mkdir -p $(DATA_DIR)

$(INPUT_DIR): | $(DATA_DIR)
	mkdir -p $(INPUT_DIR)

$(OUTPUT_DIR): | $(DATA_DIR)
	mkdir -p $(OUTPUT_DIR)

# Google Test installation targets
$(GTEST_DIR):
	@echo "Creating Google Test directory..."
	mkdir -p $(GTEST_DIR)

$(GTEST_INSTALL_DIR)/lib/libgtest.a: | $(GTEST_DIR)
	@echo "Downloading and building Google Test $(GTEST_VERSION)..."
	@if [ ! -d "$(GTEST_DIR)/googletest-$(GTEST_VERSION)" ]; then \
		echo "Downloading Google Test source..."; \
		cd $(GTEST_DIR) && \
		wget -q https://github.com/google/googletest/archive/refs/tags/v$(GTEST_VERSION).tar.gz && \
		tar -xzf v$(GTEST_VERSION).tar.gz && \
		rm v$(GTEST_VERSION).tar.gz; \
	fi
	@echo "Building Google Test..."
	@mkdir -p $(GTEST_BUILD_DIR)
	@cd $(GTEST_BUILD_DIR) && \
		cmake ../googletest-$(GTEST_VERSION) \
			-DCMAKE_INSTALL_PREFIX=../install \
			-DCMAKE_BUILD_TYPE=Release \
			-DBUILD_GMOCK=OFF \
			-DINSTALL_GTEST=ON && \
		make -j$(shell nproc) && \
		make install
	@echo "Google Test installed successfully in $(GTEST_INSTALL_DIR)"

# Install Google Test dependency
install-gtest: $(GTEST_INSTALL_DIR)/lib/libgtest.a

# Check if Google Test is installed
check-gtest:
	@if [ -f "$(GTEST_INSTALL_DIR)/lib/libgtest.a" ]; then \
		echo "Google Test is installed in $(GTEST_INSTALL_DIR)"; \
		echo "Version: $(GTEST_VERSION)"; \
	else \
		echo "Google Test is NOT installed. Run 'make install-gtest' to install."; \
		exit 1; \
	fi

# Clean Google Test installation
clean-gtest:
	@echo "Cleaning Google Test installation..."
	rm -rf $(GTEST_DIR)
	@echo "Google Test cleaned."

# Rules for compiling CUDA source to object files
$(OBJ_CUTENSOR): $(SRC_CUTENSOR) | $(OBJ_DIR)
	@echo "Compiling cuTENSOR example with detected GPU architecture: sm_$(GPU_ARCH)"
	$(NVCC) $(NVCCFLAGS) $(SRC_CUTENSOR) -o $(OBJ_CUTENSOR)

$(OBJ_DEMO): $(SRC_DEMO) | $(OBJ_DIR)
	@echo "Compiling demo example with detected GPU architecture: sm_$(GPU_ARCH)"
	$(NVCC) $(NVCCFLAGS) $(SRC_DEMO) -o $(OBJ_DEMO)

$(OBJ_NN_UTILS): $(SRC_NN_UTILS) | $(OBJ_DIR)
	@echo "Compiling neural network utils with detected GPU architecture: sm_$(GPU_ARCH)"
	$(NVCC) $(NVCCFLAGS) $(SRC_NN_UTILS) -o $(OBJ_NN_UTILS)

$(OBJ_DATA_LOADER): $(SRC_DATA_LOADER) | $(OBJ_DIR)
	@echo "Compiling data loader with detected GPU architecture: sm_$(GPU_ARCH)"
	$(NVCC) $(NVCCFLAGS) $(SRC_DATA_LOADER) -o $(OBJ_DATA_LOADER)

$(OBJ_TEST_DATA_LOADER): $(SRC_TEST_DATA_LOADER) | $(OBJ_DIR)
	@echo "Compiling data loader test with detected GPU architecture: sm_$(GPU_ARCH)"
	$(NVCC) $(NVCCFLAGS) $(SRC_TEST_DATA_LOADER) -o $(OBJ_TEST_DATA_LOADER)

$(OBJ_NN_UTILS_TEST): $(SRC_NN_UTILS_TEST) $(GTEST_INSTALL_DIR)/lib/libgtest.a | $(OBJ_DIR)
	@echo "Compiling neural network unit tests with detected GPU architecture: sm_$(GPU_ARCH)"
	$(NVCC) $(NVCCFLAGS) $(GTEST_FLAGS) $(SRC_NN_UTILS_TEST) -o $(OBJ_NN_UTILS_TEST)

$(OBJ_DATA_LOADER_TEST): $(SRC_DATA_LOADER_TEST) $(GTEST_INSTALL_DIR)/lib/libgtest.a | $(OBJ_DIR)
	@echo "Compiling data loader unit tests with detected GPU architecture: sm_$(GPU_ARCH)"
	$(NVCC) $(NVCCFLAGS) $(GTEST_FLAGS) $(SRC_DATA_LOADER_TEST) -o $(OBJ_DATA_LOADER_TEST)

# Rules for linking to create final executables
$(TARGET_CUTENSOR): $(OBJ_CUTENSOR) | $(BIN_DIR)
	@echo "Linking cuTENSOR executable with CUDA device code"
	$(NVCC) $(NVCC_LINK_FLAGS) $(OBJ_CUTENSOR) -o $(TARGET_CUTENSOR) $(LDFLAGS)

$(TARGET_DEMO): $(OBJ_DEMO) | $(BIN_DIR)
	@echo "Linking demo executable with CUDA device code and image libraries"
	$(NVCC) $(NVCC_LINK_FLAGS) $(OBJ_DEMO) -o $(TARGET_DEMO) $(LDFLAGS)

$(TARGET_NN_UTILS_TEST): $(OBJ_NN_UTILS_TEST) $(OBJ_NN_UTILS) | $(BIN_DIR)
	@echo "Linking neural network unit tests with Google Test"
	$(NVCC) $(NVCC_LINK_FLAGS) $(OBJ_NN_UTILS_TEST) $(OBJ_NN_UTILS) -o $(TARGET_NN_UTILS_TEST) $(LDFLAGS) $(GTEST_LIBS)

$(TARGET_DATA_LOADER_TEST): $(OBJ_DATA_LOADER_TEST) $(OBJ_DATA_LOADER) | $(BIN_DIR)
	@echo "Linking data loader unit tests with Google Test"
	$(NVCC) $(NVCC_LINK_FLAGS) $(OBJ_DATA_LOADER_TEST) $(OBJ_DATA_LOADER) -o $(TARGET_DATA_LOADER_TEST) $(LDFLAGS) $(GTEST_LIBS)

# Rules for running the applications
run-cutensor-example: $(TARGET_CUTENSOR)
	@echo "Running cuTENSOR example"
	./$(TARGET_CUTENSOR)

run-demo: $(TARGET_DEMO)
	@echo "Running image demo example"
	@echo "Usage: ./$(TARGET_DEMO) <image_file> [width] [height]"
	@echo "Example: ./$(TARGET_DEMO) Common/data/teapot512.pgm"
	@echo "Example: ./$(TARGET_DEMO) Common/data/teapot_512x512_8u_Gray.raw 512 512"

run-test-nn: $(TARGET_TEST_NN)
	@echo "Running neural network test"
	./$(TARGET_TEST_NN)

run-nn-utils-test: $(TARGET_NN_UTILS_TEST)
	@echo "Running neural network unit tests"
	./$(TARGET_NN_UTILS_TEST)

run-data-loader-test: $(TARGET_DATA_LOADER_TEST)
	@echo "Running data loader unit tests"
	./$(TARGET_DATA_LOADER_TEST)

# Download training data and show dataset info
download-data: | $(INPUT_DIR)
	@echo "Downloading MNIST training dataset..."
	@echo "This will download ~47MB of training data from Yann LeCun's MNIST database"
	@echo ""
	@echo "Downloading training images..."
	$(DOWNLOAD_SCRIPT) "$(MNIST_TRAIN_IMAGES_URL)" "$(MNIST_TRAIN_IMAGES_CHECKSUM)" "$(INPUT_DIR)/train-images-idx3-ubyte.gz"
	@echo ""
	@echo "Downloading training labels..."
	$(DOWNLOAD_SCRIPT) "$(MNIST_TRAIN_LABELS_URL)" "$(MNIST_TRAIN_LABELS_CHECKSUM)" "$(INPUT_DIR)/train-labels-idx1-ubyte.gz"
	@echo ""
	@echo "Downloading test images..."
	$(DOWNLOAD_SCRIPT) "$(MNIST_TEST_IMAGES_URL)" "$(MNIST_TEST_IMAGES_CHECKSUM)" "$(INPUT_DIR)/t10k-images-idx3-ubyte.gz"
	@echo ""
	@echo "Downloading test labels..."
	$(DOWNLOAD_SCRIPT) "$(MNIST_TEST_LABELS_URL)" "$(MNIST_TEST_LABELS_CHECKSUM)" "$(INPUT_DIR)/t10k-labels-idx1-ubyte.gz"
	@echo ""
	@echo "MNIST dataset download completed successfully!"
	@echo "Files automatically extracted to: $(INPUT_DIR)/"
	@echo ""
	@echo "=== Dataset Information ==="
	@python3 scripts/extract_mnist_images.py $(INPUT_DIR)/train-images-idx3-ubyte $(INPUT_DIR)/train-labels-idx1-ubyte --info
	@echo ""
	@echo "Ready for training!"

# Show dataset information
show-data:
	@echo "=== MNIST Dataset Information ==="
	@if [ ! -f "$(INPUT_DIR)/train-images-idx3-ubyte" ] || [ ! -f "$(INPUT_DIR)/train-labels-idx1-ubyte" ]; then \
		echo "Error: MNIST dataset files not found. Run 'make download-data' first."; \
		exit 1; \
	fi
	@python3 scripts/extract_mnist_images.py $(INPUT_DIR)/train-images-idx3-ubyte $(INPUT_DIR)/train-labels-idx1-ubyte --info

# Visualize sample images from dataset
visualize-data:
	@echo "=== Visualizing MNIST Sample Images ==="
	@if [ ! -f "$(INPUT_DIR)/train-images-idx3-ubyte" ] || [ ! -f "$(INPUT_DIR)/train-labels-idx1-ubyte" ]; then \
		echo "Error: MNIST dataset files not found. Run 'make download-data' first."; \
		exit 1; \
	fi
	@echo "Opening matplotlib window with sample images..."
	@python3 scripts/extract_mnist_images.py $(INPUT_DIR)/train-images-idx3-ubyte $(INPUT_DIR)/train-labels-idx1-ubyte --visualize

# Extract sample images as PNG files
extract-samples:
	@echo "=== Extracting Sample MNIST Images ==="
	@if [ ! -f "$(INPUT_DIR)/train-images-idx3-ubyte" ] || [ ! -f "$(INPUT_DIR)/train-labels-idx1-ubyte" ]; then \
		echo "Error: MNIST dataset files not found. Run 'make download-data' first."; \
		exit 1; \
	fi
	@python3 scripts/extract_mnist_images.py $(INPUT_DIR)/train-images-idx3-ubyte $(INPUT_DIR)/train-labels-idx1-ubyte --extract --output $(INPUT_DIR)/train_data
	@echo "Sample images extracted to $(INPUT_DIR)/train_data/"
	@python3 scripts/extract_mnist_images.py $(INPUT_DIR)/t10k-images-idx3-ubyte $(INPUT_DIR)/t10k-labels-idx1-ubyte --extract --output $(INPUT_DIR)/test_data
	@echo "Sample images extracted to $(INPUT_DIR)/test_data/"

# Download single MNIST file (internal helper target)
download-mnist-file:
	@if [ -z "$(URL)" ] || [ -z "$(CHECKSUM)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Error: Missing parameters. Use: make download-mnist-file URL=<url> CHECKSUM=<checksum> OUTPUT=<path>"; \
		exit 1; \
	fi
	$(DOWNLOAD_SCRIPT) "$(URL)" "$(CHECKSUM)" "$(OUTPUT)"

# Clean up
clean:
	rm -rf $(BIN_DIR)/*
	rm -rf $(OBJ_DIR)/*
	rm -rf $(OUTPUT_DIR)/*

# Clean downloaded data files
clean-data:
	@echo "Cleaning downloaded training data..."
	rm -rf $(INPUT_DIR)/*
	@echo "Training data cleaned from $(INPUT_DIR)/"

# Clean everything including data and dependencies
clean-all: clean clean-data clean-gtest
	@echo "All build files, data, and dependencies cleaned."

# Show GPU information
gpu-info:
	@echo "GPU Information:"
	@nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
	@echo "Detected architecture flag: $(ARCH_FLAG)"

# Test CUDA compilation
test-cuda:
	@echo "Testing CUDA compilation..."
	@echo "NVCC path: $(NVCC)"
	@$(NVCC) --version
	@echo "GPU Architecture: $(GPU_ARCH)"
	@echo "Architecture flag: $(ARCH_FLAG)"

# Generate compile_commands.json for VS Code IntelliSense using bear
compile-commands:
	@echo "Generating compile_commands.json using bear..."
	@rm -f compile_commands.json
	bear --output compile_commands.json -- make clean build
	@echo "compile_commands.json generated successfully!"

# Help command
help:
	@echo "Available make commands:"
	@echo "  make              - Build the cuTENSOR example (default)."
	@echo "  make build        - Build the cuTENSOR example."
	@echo "  make build-demo   - Build the image demo example."
	@echo "  make build-nn-utils     - Build the neural network utilities."
	@echo "  make build-data-loader - Build the data loader utilities."
	@echo "  make build-test-data-loader - Build the data loader test program."
	@echo "  make build-nn-utils-test - Build the neural network unit test program."
	@echo "  make build-data-loader-test - Build the data loader unit test program."
	@echo "  make build-all    - Build all examples and utilities."
	@echo "  make run-cutensor-example          - Run the cuTENSOR example."
	@echo "  make run-demo-example     - Show usage for image demo example."
	@echo "  make run-test-data-loader - Run the data loader test program."
	@echo "  make run-nn-utils-test - Run the neural network unit tests."
	@echo "  make run-data-loader-test - Run the data loader unit tests."
	@echo "  make download-data - Download MNIST training dataset (~47MB) and show info."
	@echo "  make show-data    - Display MNIST dataset information."
	@echo "  make visualize-data - Visualize sample MNIST images (requires matplotlib)."
	@echo "  make extract-samples - Extract 100 sample images as PNG files."
	@echo "  make clean        - Clean up the build files."
	@echo "  make clean-data   - Clean downloaded training data."
	@echo "  make clean-all    - Clean build files and training data."
	@echo "  make gpu-info     - Show detected GPU information."
	@echo "  make test-cuda    - Test CUDA compilation setup."
	@echo "  make compile-commands - Generate compile_commands.json using bear."
	@echo "  make help         - Display this help message."
	@echo ""
	@echo "Google Test Management:"
	@echo "  make install-gtest - Download and build Google Test locally."
	@echo "  make check-gtest  - Check if Google Test is installed."
	@echo "  make clean-gtest  - Remove Google Test installation."
	@echo ""
	@echo "Data Management:"
	@echo "  make download-data - Downloads and extracts MNIST dataset to data/input/"
	@echo "    - Training images: train-images-idx3-ubyte (9.9MB)"
	@echo "    - Training labels: train-labels-idx1-ubyte (29KB)"
	@echo "    - Test images: t10k-images-idx3-ubyte (1.6MB)"
	@echo "    - Test labels: t10k-labels-idx1-ubyte (5KB)"
	@echo "    - All files are verified with MD5 checksums and auto-extracted"
	@echo ""
	@echo "MNIST Dataset Processing:"
	@echo "  Install Python dependencies: pip install -r requirements.txt"
	@echo "  View dataset info:"
	@echo "    python scripts/extract_mnist_images.py data/input/train-images-idx3-ubyte data/input/train-labels-idx1-ubyte --info"
	@echo "  Visualize samples:"
	@echo "    python scripts/extract_mnist_images.py data/input/train-images-idx3-ubyte data/input/train-labels-idx1-ubyte --visualize"
	@echo "  Extract images as PNG:"
	@echo "    python scripts/extract_mnist_images.py data/input/train-images-idx3-ubyte data/input/train-labels-idx1-ubyte --extract --count 100"
	@echo ""
	@echo "Image Demo Examples:"
	@echo "  ./bin/demo_example Common/data/teapot512.pgm"
	@echo "  ./bin/demo_example Common/data/teapot_512x512_8u_Gray.raw 512 512"
	@echo ""
	@echo "Note: The project contains CUDA device code (__global__ kernels)."
