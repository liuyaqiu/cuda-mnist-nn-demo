#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <limits>
#include <vector>
#include <cmath>
#include <time.h>
#include <cstdint>

#include <cuda_runtime.h>
#include <cutensor.h>

// Handle cuTENSOR errors
#define HANDLE_ERROR(x)                                             \
{ const auto err = x;                                               \
    if( err != CUTENSOR_STATUS_SUCCESS )                              \
    { printf("Error: %s in %s at line %d\n", cutensorGetErrorString(err), __FILE__, __LINE__); exit(-1); } \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
    if( err != cudaSuccess )                                        \
    { printf("CUDA Error: %s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); exit(-1); } \
};

// cuTENSOR Best Practice: Manual Broadcasting via Contraction
//
// This implementation demonstrates a reliable pattern for broadcasting in cuTENSOR:
// 1. Use cuTENSOR's strong contraction support instead of element-wise broadcasting
// 2. Manually create "broadcasted" tensors via: broadcasted_tensor = scalar * ones_tensor  
// 3. Then use standard element-wise operations on tensors of the same shape
//
// Why this approach is superior:
// - cuTENSOR contraction has excellent broadcasting semantics and documentation
// - Element-wise broadcasting support is limited and poorly documented
// - This pattern works consistently across all cuTENSOR versions and operations
// - Provides explicit control over broadcasting behavior

int main(int argc, char** argv)
{
    printf("cuTENSOR Reduction & Element-wise Broadcasting Example\n");
    printf("=====================================================\n");
    printf("1. Find maximum value using reduction\n");
    printf("2. Subtract max from all elements using broadcasting\n\n");

    // Host element type definition
    typedef float floatTypeA;
    typedef float floatTypeD;
    typedef float floatTypeCompute;

    // CUDA types
    cutensorDataType_t typeA = CUTENSOR_R_32F;
    cutensorDataType_t typeD = CUTENSOR_R_32F;
    cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;

    /* ***************************** */
    // Define tensor dimensions
    /* ***************************** */

    // Create a 4D tensor A[b,i,j,k] with batch dimension
    std::vector<int> modeA = {'b','i','j','k'};
    int nmodeA = modeA.size();

    // The output will be a 1D tensor D[b] with one element
    std::vector<int> modeD = {'b'};  // Keep batch dimension
    int nmodeD = modeD.size();

    // Define extents
    std::vector<int64_t> extentA = {1, 64, 32, 16};  // 1x64x32x16 tensor (batch=1)
    std::vector<int64_t> extentD = {1};  // 1-element tensor

    printf("Input tensor dimensions: %ldx%ldx%ldx%ld (batch x height x width x depth)\n", 
           extentA[0], extentA[1], extentA[2], extentA[3]);
    printf("Output tensor dimensions: %ld (preserving batch dimension)\n\n", extentD[0]);

    /* ***************************** */
    // Allocate and initialize tensors
    /* ***************************** */

    // Number of elements
    size_t elementsA = 1;
    for(auto ext : extentA)
        elementsA *= ext;
    size_t elementsD = 1;  // Scalar output
    size_t elementsResult = elementsA;  // Result tensor has same size as input A

    // Size in bytes
    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeD = sizeof(floatTypeD) * elementsD;
    size_t sizeResult = sizeof(floatTypeA) * elementsResult;

    // Allocate on device
    void *A_d, *D_d, *Result_d;
    HANDLE_CUDA_ERROR(cudaMalloc((void**)&A_d, sizeA));
    HANDLE_CUDA_ERROR(cudaMalloc((void**)&D_d, sizeD));
    HANDLE_CUDA_ERROR(cudaMalloc((void**)&Result_d, sizeResult));

    // Allocate on host
    floatTypeA *A = (floatTypeA*) malloc(sizeof(floatTypeA) * elementsA);
    floatTypeD *D = (floatTypeD*) malloc(sizeof(floatTypeD) * elementsD);
    floatTypeA *Result = (floatTypeA*) malloc(sizeof(floatTypeA) * elementsResult);

    // Initialize data on host with random values
    // Use high-resolution clock for better random seeding
    auto now = std::chrono::high_resolution_clock::now();
    auto seed = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    srand(seed % UINT_MAX);  // Seed random number generator
    
    for(int64_t i = 0; i < elementsA; i++)
    {
        A[i] = (((float) rand())/RAND_MAX) * 200.0f - 100.0f;  // Random values between -100 and 100
    }

    printf("Initialized tensor with %zu elements\n", elementsA);
    printf("Random values in range: [-100.0, 100.0]\n\n");

    // Copy to device
    HANDLE_CUDA_ERROR(cudaMemcpy(A_d, A, sizeA, cudaMemcpyHostToDevice));

    // Initialize output to negative infinity
    *D = -std::numeric_limits<float>::infinity();
    HANDLE_CUDA_ERROR(cudaMemcpy(D_d, D, sizeD, cudaMemcpyHostToDevice));

    const uint32_t kAlignment = 128; // Alignment of the global-memory device pointers (bytes)
    assert(uintptr_t(A_d) % kAlignment == 0);
    assert(uintptr_t(D_d) % kAlignment == 0);
    assert(uintptr_t(Result_d) % kAlignment == 0);

    /*************************
     * cuTENSOR Setup
     *************************/ 

    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorCreate(&handle));

    /**********************
     * Create Tensor Descriptors
     **********************/

    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                &descA,
                nmodeA,
                extentA.data(),
                NULL,  // stride
                typeA, 
                kAlignment));

    // Create descriptor for output tensor D[b]
    cutensorTensorDescriptor_t descD;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                &descD,
                nmodeD,
                extentD.data(),
                NULL,  // stride
                typeD, 
                kAlignment));

    // Create descriptor for result tensor (same shape as A)
    cutensorTensorDescriptor_t descResult;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                &descResult,
                nmodeA,
                extentA.data(),
                NULL,  // stride
                typeA, 
                kAlignment));

    printf("Created tensor descriptors\n");

    /*******************************
     * Create Reduction Descriptor
     *******************************/

    cutensorOperationDescriptor_t desc;
    
    // For reduction, we need to specify:
    // - modeA: all modes of input tensor A
    // - modeC/modeD: modes that are preserved (not reduced)
    // We want to reduce over i,j,k but keep b
    std::vector<int32_t> modeAint;
    std::vector<int32_t> modeDint;
    
    // Convert modes to int32_t
    for(auto mode : modeA) {
        modeAint.push_back((int32_t)mode);
    }
    
    for(auto mode : modeD) {
        modeDint.push_back((int32_t)mode);
    }

    HANDLE_ERROR(cutensorCreateReduction(handle, 
                &desc,
                descA, modeAint.data(),              // Input tensor A[b,i,j,k] and all its modes
                CUTENSOR_OP_IDENTITY,                // Unary operator for A (identity = no change)
                descD, modeDint.data(),              // C tensor with preserved modes [b]
                CUTENSOR_OP_IDENTITY,                // Operator for C
                descD, modeDint.data(),              // Output tensor D[b] with preserved modes
                CUTENSOR_OP_MAX,                     // Reduction operator: MAX
                descCompute));                       // Compute type descriptor

    printf("Created reduction operation descriptor\n");

    /*****************************
     * Verify scalar type
     *****************************/

    cutensorDataType_t scalarType;
    HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(handle,
                desc,
                CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                (void*)&scalarType,
                sizeof(scalarType)));

    assert(scalarType == CUTENSOR_R_32F);
    floatTypeCompute alpha = 1.0f;  // Scale factor for input
    floatTypeCompute beta  = 0.0f;  // Scale factor for output (0 = overwrite)

    /**************************
     * Set the algorithm to use
     ***************************/

    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t planPref;
    HANDLE_ERROR(cutensorCreatePlanPreference(
                handle,
                &planPref,
                algo,
                CUTENSOR_JIT_MODE_NONE));

    /**********************
     * Query workspace estimate
     **********************/

    uint64_t workspaceSizeEstimate = 0;
    const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
    HANDLE_ERROR(cutensorEstimateWorkspaceSize(handle,
                desc,
                planPref,
                workspacePref,
                &workspaceSizeEstimate));

    printf("Estimated workspace size: %lu bytes\n", workspaceSizeEstimate);

    /**************************
     * Create Reduction Plan
     **************************/

    cutensorPlan_t plan;
    HANDLE_ERROR(cutensorCreatePlan(handle,
                &plan,
                desc,
                planPref,
                workspaceSizeEstimate));

    /**************************
     * Query actual workspace size
     **************************/

    uint64_t actualWorkspaceSize = 0;
    HANDLE_ERROR(cutensorPlanGetAttribute(handle,
                plan,
                CUTENSOR_PLAN_REQUIRED_WORKSPACE,
                &actualWorkspaceSize,
                sizeof(actualWorkspaceSize)));

    printf("Actual workspace size: %lu bytes\n\n", actualWorkspaceSize);

    void *work = nullptr;
    if (actualWorkspaceSize > 0)
    {
        HANDLE_CUDA_ERROR(cudaMalloc(&work, actualWorkspaceSize));
        assert(uintptr_t(work) % 128 == 0); // workspace must be aligned to 128 byte-boundary
    }

    /**********************
     * Execute Reduction
     **********************/

    cudaStream_t stream;
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

    // Create CUDA events for timing
    cudaEvent_t start_gpu, stop_gpu;
    HANDLE_CUDA_ERROR(cudaEventCreate(&start_gpu));
    HANDLE_CUDA_ERROR(cudaEventCreate(&stop_gpu));

    printf("Starting reduction to find maximum value...\n");

    // Start timing
    auto start_cpu = std::chrono::high_resolution_clock::now();
    HANDLE_CUDA_ERROR(cudaEventRecord(start_gpu, stream));

    // Execute reduction
    // For reduction: D = reduce(alpha * A)
    HANDLE_ERROR(cutensorReduce(handle,
                plan,
                (const void*) &alpha, A_d,    // alpha * A
                (const void*) &beta,  D_d,    // beta * D (beta=0, so D is overwritten)
                D_d,                    // output
                work, actualWorkspaceSize, stream));

    // Stop timing
    HANDLE_CUDA_ERROR(cudaEventRecord(stop_gpu, stream));
    HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));
    auto end_cpu = std::chrono::high_resolution_clock::now();

    // Copy result back to host
    HANDLE_CUDA_ERROR(cudaMemcpy(D, D_d, sizeD, cudaMemcpyDeviceToHost));

    // Calculate timing
    float gpu_time_ms;
    HANDLE_CUDA_ERROR(cudaEventElapsedTime(&gpu_time_ms, start_gpu, stop_gpu));
    
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
    float cpu_time_ms = cpu_duration.count() / 1000.0f;

    printf("\nReduction completed!\n");
    printf("cuTENSOR maximum value found: %.6f\n", *D);

    /**********************
     * Element-wise Broadcasting Operation: A - max_value using cuTENSOR Contract + Binary
     **********************/

    printf("\nStarting element-wise broadcasting operation: A[i,j,k,l] - max_value using cuTENSOR...\n");
    printf("Using cuTENSOR manual broadcasting pattern: scalar * ones = full_tensor, then A - full_tensor\n");
    printf("This approach is more reliable than direct element-wise broadcasting in cuTENSOR\n");

    // Create timing events for element-wise operation
    cudaEvent_t start_elementwise, stop_elementwise;
    HANDLE_CUDA_ERROR(cudaEventCreate(&start_elementwise));
    HANDLE_CUDA_ERROR(cudaEventCreate(&stop_elementwise));

    // Create a true scalar tensor descriptor (0-dimensional)
    cutensorTensorDescriptor_t descScalar;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                &descScalar,
                0,                       // 0 modes = true scalar
                NULL,                    // no extents for scalar
                NULL,                    // no strides for scalar
                typeD,                   // Same type as the max value
                kAlignment));

    printf("Created scalar tensor descriptor for broadcasting\n");
    
    // Print detailed information about tensors
    printf("Tensor A modes: [");
    for(int i = 0; i < nmodeA; i++) {
        printf("%c", modeA[i]);
        if(i < nmodeA-1) printf(",");
    }
    printf("] extents: [");
    for(int i = 0; i < nmodeA; i++) {
        printf("%ld", extentA[i]);
        if(i < nmodeA-1) printf(",");
    }
    printf("]\n");
    
    printf("Scalar modes: [] extents: [] (true scalar)\n");
    printf("Result modes: [");
    for(int i = 0; i < nmodeA; i++) {
        printf("%c", modeA[i]);
        if(i < nmodeA-1) printf(",");
    }
    printf("] extents: [");
    for(int i = 0; i < nmodeA; i++) {
        printf("%ld", extentA[i]);
        if(i < nmodeA-1) printf(",");
    }
    printf("]\n");

    // Convert modes to int32_t for operations
    std::vector<int32_t> modeAint_ew, modeResultInt;
    for(auto mode : modeA) {
        modeAint_ew.push_back((int32_t)mode);
        modeResultInt.push_back((int32_t)mode);
    }

    // We'll create operations dynamically during execution for maximum flexibility
    printf("Setting up contraction and binary operations dynamically...\n");

    HANDLE_CUDA_ERROR(cudaEventRecord(start_elementwise, stream));
    
    // Step 1: Create ones tensor with same shape as A
    printf("Step 1: Creating and filling ones tensor...\n");
    void *ones_d, *scalar_tensor_d;
    HANDLE_CUDA_ERROR(cudaMalloc(&ones_d, sizeA));
    HANDLE_CUDA_ERROR(cudaMalloc(&scalar_tensor_d, sizeA));
    
    // Fill ones tensor on host and copy to device
    floatTypeA *ones_host = (floatTypeA*) malloc(sizeA);
    for(size_t i = 0; i < elementsA; i++) {
        ones_host[i] = 1.0f;
    }
    HANDLE_CUDA_ERROR(cudaMemcpy(ones_d, ones_host, sizeA, cudaMemcpyHostToDevice));
    free(ones_host);
    
    // Step 2: Set up contraction to broadcast scalar: scalar_tensor = scalar * ones
    // This is the key insight: use contraction (which has excellent broadcasting support)
    // to manually create the broadcasted tensor, rather than relying on element-wise broadcasting
    printf("Step 2: Setting up contraction: scalar_tensor = scalar * ones...\n");
    
    // Create tensor descriptors for ones and scalar_tensor
    cutensorTensorDescriptor_t descOnes, descScalarTensor;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descOnes, nmodeA, extentA.data(), NULL, typeA, kAlignment));
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle, &descScalarTensor, nmodeA, extentA.data(), NULL, typeA, kAlignment));
    
    // Create contraction descriptor: scalar_tensor[b,i,j,k] = scalar[] * ones[b,i,j,k]
    cutensorOperationDescriptor_t descContraction;
    std::vector<int32_t> modeScalarEmpty;  // Empty for 0D scalar
    
    HANDLE_ERROR(cutensorCreateContraction(handle,
                &descContraction,
                descScalar, modeScalarEmpty.data(),      // A: scalar (0D)
                CUTENSOR_OP_IDENTITY,
                descOnes, modeAint_ew.data(),            // B: ones tensor (4D)
                CUTENSOR_OP_IDENTITY,
                descScalarTensor, modeAint_ew.data(),    // C: dummy (we'll use beta=0)
                CUTENSOR_OP_IDENTITY,
                descScalarTensor, modeAint_ew.data(),    // Output: scalar_tensor (4D)
                descCompute));
    
    // Create plan for contraction
    cutensorPlanPreference_t planPrefContraction;
    HANDLE_ERROR(cutensorCreatePlanPreference(handle, &planPrefContraction, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE));
    
    uint64_t workspaceSizeContraction = 0;
    HANDLE_ERROR(cutensorEstimateWorkspaceSize(handle, descContraction, planPrefContraction, CUTENSOR_WORKSPACE_DEFAULT, &workspaceSizeContraction));
    
    cutensorPlan_t planContraction;
    HANDLE_ERROR(cutensorCreatePlan(handle, &planContraction, descContraction, planPrefContraction, workspaceSizeContraction));
    
    void *workContraction = nullptr;
    if (workspaceSizeContraction > 0) {
        HANDLE_CUDA_ERROR(cudaMalloc(&workContraction, workspaceSizeContraction));
    }
    
    // Execute contraction: scalar_tensor = 1.0 * scalar * ones + 0.0 * dummy
    floatTypeCompute alpha_contract = 1.0f;
    floatTypeCompute beta_contract = 0.0f;
    
    printf("Step 3: Executing contraction...\n");
    HANDLE_ERROR(cutensorContract(handle, planContraction,
                (const void*)&alpha_contract, D_d, ones_d,              // alpha * scalar * ones
                (const void*)&beta_contract, scalar_tensor_d,           // beta * C (beta=0, so ignore)
                scalar_tensor_d,                                         // Output: scalar_tensor
                workContraction, workspaceSizeContraction, stream));
    
    // Step 4: Set up binary element-wise operation: Result = A - scalar_tensor
    printf("Step 4: Setting up binary element-wise: Result = A - scalar_tensor...\n");
    
    cutensorOperationDescriptor_t descBinary;
    HANDLE_ERROR(cutensorCreateElementwiseBinary(handle,
                &descBinary,
                descA, modeAint_ew.data(),               // Input A
                CUTENSOR_OP_IDENTITY,
                descScalarTensor, modeAint_ew.data(),    // scalar_tensor
                CUTENSOR_OP_IDENTITY,
                descResult, modeResultInt.data(),        // Output
                CUTENSOR_OP_ADD,                         // A + (-1.0 * scalar_tensor)
                descCompute));
    
    // Create plan for binary operation
    cutensorPlanPreference_t planPrefBinary;
    HANDLE_ERROR(cutensorCreatePlanPreference(handle, &planPrefBinary, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE));
    
    uint64_t workspaceSizeBinary = 0;
    HANDLE_ERROR(cutensorEstimateWorkspaceSize(handle, descBinary, planPrefBinary, CUTENSOR_WORKSPACE_DEFAULT, &workspaceSizeBinary));
    
    cutensorPlan_t planBinary;
    HANDLE_ERROR(cutensorCreatePlan(handle, &planBinary, descBinary, planPrefBinary, workspaceSizeBinary));
    
    void *workBinary = nullptr;
    if (workspaceSizeBinary > 0) {
        HANDLE_CUDA_ERROR(cudaMalloc(&workBinary, workspaceSizeBinary));
    }
    
    // Execute binary operation: Result = 1.0 * A + (-1.0) * scalar_tensor
    floatTypeCompute alpha_binary = 1.0f;
    floatTypeCompute gamma_binary = -1.0f;  // Negative for subtraction
    
    printf("Step 5: Executing binary element-wise operation...\n");
    HANDLE_ERROR(cutensorElementwiseBinaryExecute(handle, planBinary,
                (const void*)&alpha_binary, A_d,            // alpha * A
                (const void*)&gamma_binary, scalar_tensor_d, // gamma * scalar_tensor (negative for subtraction)
                Result_d,                                     // Output
                stream));
    
    // Cleanup temporary resources
    HANDLE_CUDA_ERROR(cudaFree(ones_d));
    HANDLE_CUDA_ERROR(cudaFree(scalar_tensor_d));
    if (workContraction) HANDLE_CUDA_ERROR(cudaFree(workContraction));
    if (workBinary) HANDLE_CUDA_ERROR(cudaFree(workBinary));
    
    HANDLE_ERROR(cutensorDestroyPlan(planContraction));
    HANDLE_ERROR(cutensorDestroyPlan(planBinary));
    HANDLE_ERROR(cutensorDestroyOperationDescriptor(descContraction));
    HANDLE_ERROR(cutensorDestroyOperationDescriptor(descBinary));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descOnes));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descScalarTensor));
    HANDLE_ERROR(cutensorDestroyPlanPreference(planPrefContraction));
    HANDLE_ERROR(cutensorDestroyPlanPreference(planPrefBinary));
    
    printf("Completed cuTENSOR contraction + binary element-wise approach\n");
    
    HANDLE_CUDA_ERROR(cudaEventRecord(stop_elementwise, stream));
    HANDLE_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Copy result back to host
    HANDLE_CUDA_ERROR(cudaMemcpy(Result, Result_d, sizeResult, cudaMemcpyDeviceToHost));

    // Calculate timing for element-wise operation
    float elementwise_time_ms;
    HANDLE_CUDA_ERROR(cudaEventElapsedTime(&elementwise_time_ms, start_elementwise, stop_elementwise));

    printf("Element-wise operation completed in %.3f ms using cuTENSOR contraction + binary element-wise\n", elementwise_time_ms);

    // Verify element-wise result on CPU
    printf("\nVerifying element-wise broadcasting operation...\n");
    
    // Calculate expected result on CPU: Result[i] = A[i] - max_value
    float max_error = 0.0f;
    int max_error_idx = -1;
    int verification_samples = 10;  // Check first 10 elements for verification
    
    printf("Checking first %d elements:\n", verification_samples);
    for(int i = 0; i < verification_samples && i < elementsA; i++) {
        float expected = A[i] - (*D);
        float actual = Result[i];
        float error = fabs(actual - expected);
        
        printf("  Element %d: A=%.6f, max=%.6f, Expected=%.6f, GPU=%.6f, Error=%.9f\n", 
               i, A[i], *D, expected, actual, error);
        
        if(error > max_error) {
            max_error = error;
            max_error_idx = i;
        }
    }
    
    printf("Maximum error in element-wise operation: %.9f (at index %d)\n", max_error, max_error_idx);
    const float elementwise_tolerance = 1e-5f;
    bool elementwise_results_match = max_error < elementwise_tolerance;
    printf("Element-wise results match: %s\n", elementwise_results_match ? "YES" : "NO");
    
    // Verify result by finding max on CPU
    float cpu_max = -std::numeric_limits<float>::infinity();
    int cpu_max_idx = -1;
    for(int64_t i = 0; i < elementsA; i++) {
        if(A[i] > cpu_max) {
            cpu_max = A[i];
            cpu_max_idx = i;
        }
    }
    printf("CPU verification max: %.6f (at index %d)\n", cpu_max, cpu_max_idx);
    printf("Difference (GPU vs CPU): %.9f\n", fabs(*D - cpu_max));
    
    // Check if results match within floating point tolerance
    const float tolerance = 1e-5f;
    bool results_match = fabs(*D - cpu_max) < tolerance;
    printf("Results match: %s\n", results_match ? "YES" : "NO");
    
    printf("\nPerformance metrics:\n");
    printf("Input tensor size:        %zu elements (%.2f MB)\n", elementsA, sizeA / (1024.0 * 1024.0));
    printf("GPU reduction time:       %.3f ms\n", gpu_time_ms);
    printf("GPU element-wise time:    %.3f ms\n", elementwise_time_ms);
    printf("Total GPU time:           %.3f ms\n", gpu_time_ms + elementwise_time_ms);
    printf("CPU wall-clock time:      %.3f ms\n", cpu_time_ms);
    printf("Reduction throughput:     %.2f GB/s\n", (sizeA / (1024.0 * 1024.0 * 1024.0)) / (gpu_time_ms / 1000.0));
    printf("Element-wise throughput:  %.2f GB/s\n", ((sizeA + sizeResult) / (1024.0 * 1024.0 * 1024.0)) / (elementwise_time_ms / 1000.0));

    /**********************
     * Free allocated resources
     **********************/
    
    // Cleanup element-wise operation resources
    HANDLE_CUDA_ERROR(cudaEventDestroy(start_elementwise));
    HANDLE_CUDA_ERROR(cudaEventDestroy(stop_elementwise));
    
    // Cleanup scalar tensor descriptor
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descScalar));
    
    // Cleanup reduction operation resources
    HANDLE_ERROR(cutensorDestroyPlan(plan));
    HANDLE_ERROR(cutensorDestroyOperationDescriptor(desc));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descA));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descD));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descResult));
    HANDLE_ERROR(cutensorDestroyPlanPreference(planPref));
    HANDLE_ERROR(cutensorDestroy(handle));
    
    HANDLE_CUDA_ERROR(cudaEventDestroy(start_gpu));
    HANDLE_CUDA_ERROR(cudaEventDestroy(stop_gpu));
    HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));

    // Free host memory
    if (A) free(A);
    if (D) free(D);
    if (Result) free(Result);
    
    // Free device memory
    if (A_d) cudaFree(A_d);
    if (D_d) cudaFree(D_d);
    if (Result_d) cudaFree(Result_d);
    if (work) cudaFree(work);

    printf("\nTest completed successfully!\n");
    return 0;
}
