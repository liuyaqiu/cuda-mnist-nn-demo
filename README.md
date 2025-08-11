# Image Edge dection using NVIDIA NPP with CUDA

## Overview

This project demonstrates the use of NVIDIA Performance Primitives (NPP) library with CUDA to perform image edge detection. The goal is to utilize GPU acceleration to efficiently rotate a given image by a specified angle, leveraging the computational power of modern GPUs. The project is a part of the CUDA at Scale for the Enterprise course and serves as a template for understanding how to implement basic image processing operations using CUDA and NPP.

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

## Dependencies needed to build/run

### FreeImage
Project: [FreeImage](https://github.com/danoli3/FreeImage)

You could install it on Ubuntu by:
```
sudo apt install libfreeimage-dev
```

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## Build and Run
Now the `make build` will detect GPU model automatically and set the corresponding arch tag for nvcc compiling.

* `make build`:  build the binary
* `make run`: run the binary with example data
* `make clean`: delete the binary
* `make gpu-info`: show the GPU information.
* `make test-cuda`: show cuda version and nvcc path.
* `make compile-commands`: generate the `compile_commands.json`

## Demo output
If you execute the script, it will clean and rebuild the binary, then execute with the demo input file `data/input/Splash_gray.png`:
```bash
./run.sh
```

Then if will output the following in your terminal:
```
rm -rf bin/*
rm -rf obj/*
rm -rf data/output/*
Compiling CUDA source with detected GPU architecture: sm_86
/usr/local/cuda/bin/nvcc -arch=sm_86 --std=c++11 -I/usr/local/cuda/include -Iinclude -ICommon -ICommon/UtilNPP -ICommon/GL -Ilib --relocatable-device-code=true --compile src/imageEdgeDetection.cu -o obj/imageEdgeDetection.o
Linking executable with CUDA device code
/usr/local/cuda/bin/nvcc -arch=sm_86 --std=c++11 -I/usr/local/cuda/include -Iinclude -ICommon -ICommon/UtilNPP -ICommon/GL -Ilib obj/imageEdgeDetection.o -o bin/imageEdgeDetection -L/usr/local/cuda/lib64 -Llib -LCommon/lib -lcudart -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lfreeimage -lm
Running with input: data/input/Splash_gray.png, output: data/output/Splash_gray_edge.png
./bin/imageEdgeDetection --input data/input/Splash_gray.png --output data/output/Splash_gray_edge.png
./bin/imageEdgeDetection Starting...

GPU Device 0: "Ampere" with compute capability 8.6

NPP Library Version 12.4.1
  CUDA Driver  Version: 12.2
  CUDA Runtime Version: 12.9
  Device 0: <          Ampere >, Compute SM 8.6 detected
nppiRotate opened: <data/input/Splash_gray.png> successfully!
Save to data/output/Splash_gray_edge_noise_reduced.png
Debug: Image size = 200x200
Debug: oSizeROI = 200x200
Save to data/output/Splash_gray_edge_magnitude.png
Running NMS...
Saved output image to data/output/Splash_gray_edge.png
```

You could see the output files:
* `data/output/Splash_gray_edge_noise_reduced.png`: The output image after Gauss filter.
* `data/output/Splash_gray_edge_magnitude.png`: the output image after gradient calculation.
* `data/output/Splash_gray_edge.png`: the final edge style outpt image.
