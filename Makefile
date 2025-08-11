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
NVCCFLAGS = $(ARCH_FLAG) --std=c++11 \
	-I/usr/local/cuda/include \
	-Iinclude -ICommon -ICommon/UtilNPP -ICommon/GL -I$(LIB_DIR) \
	--relocatable-device-code=true \
	--compile

# Additional flags for linking CUDA device code
NVCC_LINK_FLAGS = $(ARCH_FLAG) --std=c++11 \
	-I/usr/local/cuda/include \
	-Iinclude -ICommon -ICommon/UtilNPP -ICommon/GL -I$(LIB_DIR)

CXXFLAGS = -std=c++11 -I/usr/local/cuda/include -Iinclude -ICommon -ICommon/UtilNPP -ICommon/GL

# Updated library flags - added math library for CUDA math functions
LDFLAGS = -L/usr/local/cuda/lib64 -L$(LIB_DIR) -LCommon/lib \
	-lcudart -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc \
	-lfreeimage -lm

# Define directories
SRC_DIR = src
BIN_DIR = bin
OBJ_DIR = obj
LIB_DIR = lib
INPUT_DIR = data/input
OUTPUT_DIR = data/output

# Default input/output files (can be overridden)
INPUT_FILE ?= $(INPUT_DIR)/Splash_gray.png
OUTPUT_FILE ?= $(OUTPUT_DIR)/Splash_gray_edge.png

# Define source files and target executable
SRC = $(SRC_DIR)/imageEdgeDetection.cu
TARGET = $(BIN_DIR)/imageEdgeDetection
OBJ = $(OBJ_DIR)/imageEdgeDetection.o

# Build target
build: $(TARGET)

# Create necessary directories
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Rule for compiling CUDA source to object file
$(OBJ): $(SRC) | $(OBJ_DIR)
	@echo "Compiling CUDA source with detected GPU architecture: sm_$(GPU_ARCH)"
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(OBJ)

# Rule for linking to create final executable
$(TARGET): $(OBJ) | $(BIN_DIR)
	@echo "Linking executable with CUDA device code"
	$(NVCC) $(NVCC_LINK_FLAGS) $(OBJ) -o $(TARGET) $(LDFLAGS)

# Rule for running the application
run: $(TARGET)
	@echo "Running with input: $(INPUT_FILE), output: $(OUTPUT_FILE)"
	./$(TARGET) --input $(INPUT_FILE) --output $(OUTPUT_FILE)

# Clean up
clean:
	rm -rf $(BIN_DIR)/*
	rm -rf $(OBJ_DIR)/*
	rm -rf $(OUTPUT_DIR)/*

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
	@echo "  make              - Build the project (2-step: compile then link)."
	@echo "  make build        - Build the project (2-step: compile then link)."
	@echo "  make run          - Run the project with default files."
	@echo "  make clean        - Clean up the build files."
	@echo "  make gpu-info     - Show detected GPU information."
	@echo "  make test-cuda    - Test CUDA compilation setup."
	@echo "  make compile-commands - Generate compile_commands.json using bear."
	@echo "  make help         - Display this help message."
	@echo ""
	@echo "You can override input/output files:"
	@echo "  make run INPUT_FILE=path/to/input.png OUTPUT_FILE=path/to/output.png"
	@echo ""
	@echo "Default files:"
	@echo "  INPUT_FILE  = $(INPUT_FILE)"
	@echo "  OUTPUT_FILE = $(OUTPUT_FILE)"
	@echo ""
	@echo "Note: The project contains CUDA device code (__global__ kernels)."
	@echo "If you encounter compilation issues, try 'make build-simple' instead."
