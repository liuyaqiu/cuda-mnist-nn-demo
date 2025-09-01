# MNIST image classification using cuTENSOR with CUDA

## Overview

This project demonstrates the use of [cuTENSOR](https://docs.nvidia.com/cuda/cutensor/latest/getting_started.html) library with CUDA to implement an simple neural network for image classification.

This demo project only cares about the following aspects:
1. the basic concepts and pipeline of cuTENSOR and CUDA programming
2. the basic concepts and algorithm of neural network, and the implementations with cuTENSOR
3. the end to end solution for the image classification task on the simple MNIST dataset, contains neural network definition, forward and backward propagation, validation and testing.

The following are not goals of this demo project:
1. high performance and optimization of the programming of cuTENSOR with CUDA.
2. advanced and frontier technique and knack of deep neural network and image classification.

## Code Organization

```bin/```
This folder should hold all binary/executable code that is built automatically or manually. Executable code should have use the .exe extension or programming language-specific extension.

```data/```
This folder should hold all example data in any format. If the original data is rather large or can be brought in via scripts, this can be left blank in the respository, so that it doesn't require major downloads when all that is desired is the code/structure.

```Common/```
Any external libraries that are not installed via the Operating System-specific package manager should be placed here, so that it is easier for inclusion/linking.

Now these dependent libraries is copied from [cuda-examples](https://github.com/NVIDIA/cuda-samples).

```lib/```
Internal libraries, some custom utils functions.

```src/```
The source code should be placed here in a hierarchical fashion, as appropriate.

```scripts/```
Some helpful scripts to download data.

```README.md```
This file should hold the description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.

```Makefile```
How to build and run code.

```run.sh```
An optional script used to run your executable code, either with or without command-line arguments.

```compile_commands.json```
The compilation database for vscode code navigation.
If you want to use it on Ubuntu, please install `bear`:
```bash
sudo apt install bear
```

## Key Concepts

Performance Strategies, Image Processing, NPP Library

## Supported SM Architectures

[SM 3.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 3.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux

## Supported CPU Architecture

x86_64 

## CUDA APIs involved
cuTENSOR

## Dependencies needed to build/run

### FreeImage
Project: [FreeImage](https://github.com/danoli3/FreeImage)

You could install it on Ubuntu by:
```
sudo apt install libfreeimage-dev
```

### CMAKE
You should install cmake to build the gtest library.
```
sudo apt install cmake
```

### Google Test (for unit testing)
Google Test is now automatically managed by the Makefile. You don't need to install it manually.

To install Google Test locally within the project:
```
make install-gtest
```

This will:
- Download Google Test v1.14.0 source code
- Build it with CMake
- Install it to `third_party/googletest/install/`

To check if Google Test is installed:
```
make check-gtest
```

To clean the Google Test installation:
```
make clean-gtest
```

### Python dependency
python >= 3.10
```
pip3 install -r requirements.txt
```

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

Download and install the [cuTENSOR](https://developer.nvidia.com/cutensor-downloads) for your corresponding platform.

## Build and Run
Now the `make build` will detect GPU model automatically and set the corresponding arch tag for nvcc compiling.

### Basic Commands
* `make build`:  build the binary
* `make run`: run the binary with example data
* `make clean`: delete the binary
* `make gpu-info`: show the GPU information.
* `make test-cuda`: show cuda version and nvcc path.
* `make compile-commands`: generate the `compile_commands.json`

### Unit Testing
* `make install-gtest`: Install Google Test dependency (required for unit tests)
* `make build-nn-utils-test`: Build the neural network unit tests
* `make run-nn-utils-test`: Run the neural network unit tests
* `make run-unit-tests`: Run all the unit tests.

### Google Test Management
* `make check-gtest`: Check if Google Test is installed
* `make clean-gtest`: Remove Google Test installation
* `make clean-all`: Clean everything including dependencies

### Download dataset
The dataset is [fashion-mnist](https://github.com/zalandoresearch/fashion-mnist), an upgraded version for [mnist](https://huggingface.co/datasets/ylecun/mnist).

* `make download-data`: Download the mnist dataset into `data/input`.
* `make extract-data`: Extract images from zipped data files.

## Demo output
Run the entrypoint shell script will download dataset automatically, build and run the unit tests, and then run the demo
Neural Network training and testing.

```bash
./run.sh
```

The input data path:
* `data/input/train_data`: the extracted training data images and labels.
* `data/input/test_data`: the extracted test data images and labels.

Then the output logs are:
* [unittesting.log](data/output/unittesting.log)
* [model_training.log](data/output/model_training.log)

You could take a look at the logs for the demo Neural Network details.

