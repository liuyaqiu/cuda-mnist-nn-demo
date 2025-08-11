#include <cstdio>
#include <iostream>
#include <cmath>

// CUDA headers - must be included before CUDA kernel definitions
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <npp.h>

#include <Exceptions.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <helper_string.h>

#include <utils.h>

// CUDA kernel for non-maximum suppression
__global__ void nonMaximumSuppression(const Npp16s* magnitude, const Npp16s* gradX, const Npp16s* gradY,
                                      Npp8u* output, int width, int height, 
                                      int magStep, int outputStep) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds
    if (x >= width || y >= height) {
        return;
    }
    
    // Handle border pixels by setting them to 0
    if (x < 1 || y < 1 || x >= width-1 || y >= height-1) {
        output[y * outputStep + x] = 0;
        return;
    }

    // Calculate indices using proper step sizes
    int magIdx = y * (magStep / sizeof(Npp16s)) + x;
    int outIdx = y * outputStep + x;
    
    float mag = (float)magnitude[magIdx];
    float gx = (float)gradX[magIdx];
    float gy = (float)gradY[magIdx];

    if (mag < 128) {
        output[outIdx] = 0;
        return;
    }
   
    // Calculate gradient direction
    float angle = atan2f(gy, gx) * 180.0f / M_PI;
    
    // Normalize angle to 0-180 degrees
    if (angle < 0) angle += 180.0f;
    
    // Get step size for magnitude array indexing
    int magStepElements = magStep / sizeof(Npp16s);
    
    // Determine interpolation direction based on gradient angle
    float q = 0.0f, r = 0.0f;
    
    if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
        // Horizontal direction (0 degrees)
        q = (float)magnitude[magIdx + 1];           // East
        r = (float)magnitude[magIdx - 1];           // West
    }
    else if (angle >= 22.5 && angle < 67.5) {
        // Diagonal direction (45 degrees)
        q = (float)magnitude[(y-1) * magStepElements + (x+1)]; // Northeast
        r = (float)magnitude[(y+1) * magStepElements + (x-1)]; // Southwest
    }
    else if (angle >= 67.5 && angle < 112.5) {
        // Vertical direction (90 degrees)
        q = (float)magnitude[(y-1) * magStepElements + x];     // North
        r = (float)magnitude[(y+1) * magStepElements + x];     // South
    }
    else if (angle >= 112.5 && angle < 157.5) {
        // Diagonal direction (135 degrees)
        q = (float)magnitude[(y-1) * magStepElements + (x-1)]; // Northwest
        r = (float)magnitude[(y+1) * magStepElements + (x+1)]; // Southeast
    }
    
    // Non-maximum suppression: keep pixel only if it's the maximum along gradient direction
    if (mag >= q && mag >= r) {
        output[outIdx] = mag;
    } else {
        output[outIdx] = 0;
    }
}

int main(int argc, char *argv[]) {
    printf("%s Starting...\n\n", argv[0]);

    std::string input_file;
    std::string output_file;
    char *file_path;

    findCudaDevice(argc, (const char **)argv);
    if (printfNPPinfo(argc, argv) == false) {
        exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "input")) {
        getCmdLineArgumentString(argc, (const char **)argv, "input", &file_path);
    }
    else {
        printf("Usage: %s --input=<input_file>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    input_file = file_path;

    if (checkCmdLineFlag(argc, (const char **)argv, "output")) {
        getCmdLineArgumentString(argc, (const char **)argv, "output", &file_path);
    }
    else {
        printf("Usage: %s --output=<output_file>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    output_file = file_path;

    if (checkFileExists(input_file)) {
        std::cout << "nppiRotate opened: <" << input_file.data() << "> successfully!" << std::endl;
    }
    else {
        std::cout << "nppiRotate unable to open: <" << input_file.data() << ">" << std::endl;
        exit(EXIT_FAILURE);
    }

    // declare a host image object for an 8-bit grayscale image
    npp::ImageCPU_8u_C1 oHostSrc;
    // load gray-scale image from disk
    npp::loadImage(input_file, oHostSrc);
    // declare a device image and copy construct from the host image,
    // i.e. upload host to device
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

    // Stage1: noise reduction
    NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    NppiMaskSize oMaskSize = NPP_MASK_SIZE_3_X_3;
    NPP_CHECK_NPP(nppiFilterGauss_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(), oDeviceSrc.data(), oDeviceSrc.pitch(), oSizeROI, oMaskSize));

    // save the noise reduced image
    std::string noise_reduced_file = output_file;
    noise_reduced_file.replace(noise_reduced_file.find(".png"), 4, "_noise_reduced.png");
    npp::saveImage(noise_reduced_file, oDeviceSrc);
    printf("Save to %s\n", noise_reduced_file.c_str());

    // Stage2: gradient calculation
    // Create intermediate 16-bit images for gradient calculations to avoid overflow
    npp::ImageNPP_16s_C1 oDeviceGradX(oDeviceSrc.size());
    npp::ImageNPP_16s_C1 oDeviceGradY(oDeviceSrc.size());
    npp::ImageNPP_16s_C1 oDeviceMagnitude(oDeviceSrc.size());
    
    printf("Debug: Image size = %dx%d\n", (int)oDeviceSrc.width(), (int)oDeviceSrc.height());
    printf("Debug: oSizeROI = %dx%d\n", oSizeROI.width, oSizeROI.height);
    
    Npp32s nGradStep = oDeviceGradX.pitch();
    // Calculate horizontal gradients (Sobel X)
    NPP_CHECK_NPP(nppiFilterSobelHoriz_8u16s_C1R(oDeviceSrc.data(), nGradStep, 
                                                  oDeviceGradX.data(), nGradStep, oSizeROI, oMaskSize));
    
    // Calculate vertical gradients (Sobel Y)
    NPP_CHECK_NPP(nppiFilterSobelVert_8u16s_C1R(oDeviceSrc.data(), nGradStep, 
                                                 oDeviceGradY.data(), nGradStep, oSizeROI, oMaskSize));
   
    // Calculate gradient magnitude: sqrt(Gx^2 + Gy^2)
    // We need to preserve the original gradients for direction calculation in NMS
    npp::ImageNPP_16s_C1 oDeviceGradXSquared(oDeviceSrc.size());
    npp::ImageNPP_16s_C1 oDeviceGradYSquared(oDeviceSrc.size());
    
    // Copy gradients to separate arrays for squaring
    NPP_CHECK_NPP(nppiCopy_16s_C1R(oDeviceGradX.data(), nGradStep,
                                   oDeviceGradXSquared.data(), nGradStep, oSizeROI));
    NPP_CHECK_NPP(nppiCopy_16s_C1R(oDeviceGradY.data(), nGradStep,
                                   oDeviceGradYSquared.data(), nGradStep, oSizeROI));
    
    // Square the copied gradients: Gx^2 and Gy^2
    NPP_CHECK_NPP(nppiMul_16s_C1IRSfs(oDeviceGradXSquared.data(), nGradStep,
                                      oDeviceGradXSquared.data(), nGradStep, oSizeROI, 0));
    
    NPP_CHECK_NPP(nppiMul_16s_C1IRSfs(oDeviceGradYSquared.data(), nGradStep,
                                      oDeviceGradYSquared.data(), nGradStep, oSizeROI, 0));
    
    // Add Gx^2 + Gy^2
    NPP_CHECK_NPP(nppiAdd_16s_C1RSfs(oDeviceGradXSquared.data(), nGradStep,
                                     oDeviceGradYSquared.data(), nGradStep,
                                     oDeviceMagnitude.data(), nGradStep, oSizeROI, 0));
    
    // Take square root to get magnitude: sqrt(Gx^2 + Gy^2)
    NPP_CHECK_NPP(nppiSqrt_16s_C1RSfs(oDeviceMagnitude.data(), nGradStep,
                                      oDeviceMagnitude.data(), nGradStep, oSizeROI, 0));

    // convert to 8-bit and save
    npp::ImageNPP_8u_C1 oDeviceMagnitude8u(oDeviceMagnitude.size());
    NPP_CHECK_NPP(nppiConvert_16s8u_C1R(oDeviceMagnitude.data(), nGradStep,
                                        oDeviceMagnitude8u.data(), nGradStep, oSizeROI));
    std::string magnitude_file = output_file;
    magnitude_file.replace(magnitude_file.find(".png"), 4, "_magnitude.png");
    npp::saveImage(magnitude_file, oDeviceMagnitude8u);
    printf("Save to %s\n", magnitude_file.c_str());
    
    // Stage3: non-maximum suppression
    // Non-Maximum Suppression (NMS): This is the key step to "thin" the edges. It scans along the gradient direction and suppresses any pixel that is not the maximum in its local neighborhood. This results in single-pixel wide edges.
    
    // Create output image for non-maximum suppressed result
    npp::ImageNPP_8u_C1 oDeviceNMS(oDeviceSrc.size());
    Npp32s nNMSStep = oDeviceNMS.pitch(); // Use actual device pitch instead of width
    
    // Launch CUDA kernel for non-maximum suppression
    dim3 blockSize(16, 16);
    dim3 gridSize((oDeviceSrc.width() + blockSize.x - 1) / blockSize.x,
                  (oDeviceSrc.height() + blockSize.y - 1) / blockSize.y);
    
    NPP_CHECK_CUDA(cudaGetLastError());
    NPP_CHECK_CUDA(cudaDeviceSynchronize());
    
    npp::ImageCPU_8u_C1 oHostNMS(oDeviceNMS.size());
    oDeviceNMS.copyTo(oHostNMS.data(), oHostNMS.pitch());

    // Now run the actual NMS
    printf("Running NMS...\n");
    nonMaximumSuppression<<<gridSize, blockSize>>>(
        oDeviceMagnitude.data(), oDeviceGradX.data(), oDeviceGradY.data(),
        oDeviceNMS.data(), oDeviceSrc.width(), oDeviceSrc.height(), 
        nGradStep, nNMSStep);
    
    // Check for kernel launch errors
    NPP_CHECK_CUDA(cudaGetLastError());
    NPP_CHECK_CUDA(cudaDeviceSynchronize());

    // save to output_file
    oDeviceNMS.copyTo(oHostNMS.data(), oHostNMS.pitch());
    npp::saveImage(output_file, oHostNMS);
    printf("Saved output image to %s\n", output_file.c_str());

    return 0;
}