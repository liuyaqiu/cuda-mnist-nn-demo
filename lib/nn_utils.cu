#include "nn_utils.h"
#include <curand_kernel.h>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <algorithm>

//=============================================================================
// CUDA Kernels for Neural Layer Operations
//=============================================================================

// Kernel for initializing weights using Xavier/He initialization
__global__ void init_weights_kernel(float* weights, int size, float scale, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        weights[idx] = curand_normal(&state) * scale;
    }
}

// Kernel for ReLU derivative
__global__ void relu_derivative_kernel(const float* z, const float* dy, float* dz, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dz[idx] = (z[idx] > 0.0f) ? dy[idx] : 0.0f;
    }
}

// Kernel to compute L2 norm of a tensor
__global__ void compute_norm_kernel(const float* data, float* partial_norms, int size) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load and square the values
    float val = (idx < size) ? data[idx] : 0.0f;
    sdata[tid] = val * val;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        partial_norms[blockIdx.x] = sdata[0];
    }
}

// Kernel to clip gradients by value
__global__ void clip_gradients_kernel(float* gradients, int size, float max_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = gradients[idx];
        gradients[idx] = fminf(fmaxf(val, -max_value), max_value);
    }
}

// Kernel to clip gradients by norm
__global__ void clip_gradients_by_norm_kernel(float* gradients, int size, float max_norm, float current_norm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && current_norm > max_norm) {
        float scale = max_norm / current_norm;
        gradients[idx] *= scale;
    }
}

// Kernel to check for NaN or Inf values
__global__ void check_finite_kernel(const float* data, int* has_nan_inf, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        if (isnan(val) || isinf(val)) {
            atomicExch(has_nan_inf, 1);
        }
    }
}

// Kernel to apply weight decay (L2 regularization)
__global__ void apply_weight_decay_kernel(float* gradients, const float* weights, 
                                         int size, float weight_decay) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gradients[idx] += weight_decay * weights[idx];
    }
}

// Helper function to add bias using cuTENSOR element-wise operation with broadcasting
// input: (batch_size, output_elements) 
// bias: (output_elements)
// output: (batch_size, output_elements)
void add_bias_cutensor(cutensorHandle_t &handle, 
                      const float* input, const float* bias, float* output,
                      int batch_size, int output_elements) {
    
    // Use trinary operation: output = input + bias + 0
    // This allows us to use the more flexible trinary API for broadcasting
    
    // Create a zero scalar on device
    float* d_zero;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_zero, sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMemset(d_zero, 0, sizeof(float)));
    
    std::vector<int64_t> extentInput{batch_size, output_elements};
    std::vector<int64_t> extentBias{output_elements};      // 1D tensor
    std::vector<int64_t> extentZero{};                     // Scalar (empty extent)
    std::vector<int64_t> extentOutput{batch_size, output_elements};
    
    std::vector<int32_t> modeInput{'i', 'j'};
    std::vector<int32_t> modeBias{'j'};      // Only 'j' mode - broadcasts over 'i'
    std::vector<int32_t> modeZero{};         // Scalar has no modes
    std::vector<int32_t> modeOutput{'i', 'j'};
    
    // output = (input + bias) + 0
    cutensor_elementwise_trinary_wrapper(handle,
                                        input, bias, d_zero, output,
                                        extentInput, extentBias, extentZero, extentOutput,
                                        modeInput, modeBias, modeZero, modeOutput,
                                        CUTENSOR_OP_IDENTITY,  // opA
                                        CUTENSOR_OP_IDENTITY,  // opB
                                        CUTENSOR_OP_IDENTITY,  // opC
                                        CUTENSOR_OP_ADD,       // opAB
                                        CUTENSOR_OP_ADD,       // opABC
                                        1.0f,                  // alpha
                                        1.0f,                  // beta
                                        1.0f,                  // gamma
                                        0);                    // stream
    
    // Clean up
    HANDLE_CUDA_ERROR(cudaFree(d_zero));
}



// Helper function for parameter update using cuTENSOR
// params = params - learning_rate * gradients
void update_parameters_cutensor(cutensorHandle_t &handle, 
                               float* params, const float* gradients,
                               float learning_rate, int size) {
    // Use trinary operation: params = params + (-learning_rate * gradients) + 0
    
    // Create a zero scalar on device
    float* d_zero;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_zero, sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMemset(d_zero, 0, sizeof(float)));
    
    // Both params and gradients are 1D tensors of same size
    std::vector<int64_t> extent{size};
    std::vector<int64_t> extentZero{};  // Scalar
    std::vector<int32_t> mode{'i'};
    std::vector<int32_t> modeZero{};    // Scalar has no modes
    
    // We want: params = 1.0 * params + (-learning_rate) * gradients + 0.0 * zero
    float alpha = 1.0f;
    float beta = -learning_rate;
    float gamma = 0.0f;
    
    // Use element-wise trinary operation for in-place update
    cutensor_elementwise_trinary_wrapper(handle,
                                        params, gradients, d_zero, params,
                                        extent, extent, extentZero, extent,
                                        mode, mode, modeZero, mode,
                                        CUTENSOR_OP_IDENTITY,  // opA
                                        CUTENSOR_OP_IDENTITY,  // opB
                                        CUTENSOR_OP_IDENTITY,  // opC
                                        CUTENSOR_OP_ADD,       // opAB
                                        CUTENSOR_OP_ADD,       // opABC
                                        alpha,                 // alpha
                                        beta,                  // beta
                                        gamma,                 // gamma
                                        0);                    // stream
    
    // Clean up
    HANDLE_CUDA_ERROR(cudaFree(d_zero));
}

//=============================================================================
// cuTENSOR Wrapper Implementations
//=============================================================================

void cutensor_contraction_wrapper(cutensorHandle_t &cutensor_handle,
                             const float* A, const float* B, float* C, 
                                 const std::vector<int64_t>& extentA,
                                 const std::vector<int64_t>& extentB,
                                 const std::vector<int64_t>& extentC,
                                 const std::vector<int32_t>& modeA,
                                 const std::vector<int32_t>& modeB,
                                 const std::vector<int32_t>& modeC,
                             cudaStream_t stream) {
    // General tensor contraction: C = A * B
    // Users specify the modes (labels) for each tensor
    // Contracted dimensions should have the same mode labels
    // Example: Matrix multiplication C[i,j] = sum_k A[i,k] * B[k,j]
    //          modeA = {'i', 'k'}, modeB = {'k', 'j'}, modeC = {'i', 'j'}
    
    // Compute strides for row-major layout
    auto computeStrides = [](const std::vector<int64_t>& extent) {
        std::vector<int64_t> strides(extent.size());
        strides[extent.size() - 1] = 1;
        for (int i = extent.size() - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * extent[i + 1];
        }
        return strides;
    };
    
    std::vector<int64_t> strideA = computeStrides(extentA);
    std::vector<int64_t> strideB = computeStrides(extentB);
    std::vector<int64_t> strideC = computeStrides(extentC);
    
    // Data types
    cutensorDataType_t typeA = CUTENSOR_R_32F;
    cutensorDataType_t typeB = CUTENSOR_R_32F;
    cutensorDataType_t typeC = CUTENSOR_R_32F;
    cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;
    
    const uint32_t kAlignment = 128; // Alignment for device pointers
    
    // Create tensor descriptors
    cutensorTensorDescriptor_t descA, descB, descC;
    
    HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(
        cutensor_handle,
        &descA,
        modeA.size(),
        extentA.data(),
        strideA.data(),
        typeA, 
        kAlignment));
    
    HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(
        cutensor_handle,
        &descB,
        modeB.size(),
        extentB.data(),
        strideB.data(),
        typeB,
        kAlignment));
    
    HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(
        cutensor_handle,
        &descC,
        modeC.size(),
        extentC.data(),
        strideC.data(),
        typeC,
        kAlignment));
    
    // Create contraction descriptor
    // C = alpha * A * B + beta * C
    cutensorOperationDescriptor_t desc;
    HANDLE_CUTENSOR_ERROR(cutensorCreateContraction(
        cutensor_handle,
        &desc,
        descA, modeA.data(), CUTENSOR_OP_IDENTITY,
        descB, modeB.data(), CUTENSOR_OP_IDENTITY,
        descC, modeC.data(), CUTENSOR_OP_IDENTITY,
        descC, modeC.data(),
        descCompute));
    
    // Set alpha and beta
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Set algorithm
    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
    
    // Create plan preference
    cutensorPlanPreference_t planPref;
    HANDLE_CUTENSOR_ERROR(cutensorCreatePlanPreference(
        cutensor_handle,
        &planPref,
        algo,
        CUTENSOR_JIT_MODE_NONE));
    
    // Query workspace size
    uint64_t workspaceSizeEstimate = 0;
    const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
    HANDLE_CUTENSOR_ERROR(cutensorEstimateWorkspaceSize(
        cutensor_handle,
        desc,
        planPref,
        workspacePref,
        &workspaceSizeEstimate));
    
    // Create plan
    cutensorPlan_t plan;
    HANDLE_CUTENSOR_ERROR(cutensorCreatePlan(
        cutensor_handle,
        &plan,
        desc,
        planPref,
        workspaceSizeEstimate));
    
    // Query actual workspace size
    uint64_t actualWorkspaceSize = 0;
    HANDLE_CUTENSOR_ERROR(cutensorPlanGetAttribute(
        cutensor_handle,
        plan,
        CUTENSOR_PLAN_REQUIRED_WORKSPACE,
        &actualWorkspaceSize,
        sizeof(actualWorkspaceSize)));
    
    // Allocate workspace if needed
    void* workspace = nullptr;
    if (actualWorkspaceSize > 0) {
        HANDLE_CUDA_ERROR(cudaMalloc(&workspace, actualWorkspaceSize));
    }
    
    // Execute contraction
    HANDLE_CUTENSOR_ERROR(cutensorContract(
        cutensor_handle,
        plan,
        (void*)&alpha, A, B,
        (void*)&beta, C, C,
        workspace, actualWorkspaceSize, stream));
    
    // Clean up
    if (workspace) {
        HANDLE_CUDA_ERROR(cudaFree(workspace));
    }
    
    HANDLE_CUTENSOR_ERROR(cutensorDestroyPlan(plan));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyPlanPreference(planPref));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyOperationDescriptor(desc));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyTensorDescriptor(descA));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyTensorDescriptor(descB));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyTensorDescriptor(descC));
}

//=============================================================================
// cuTENSOR Reduction Wrapper Implementation
//=============================================================================

void cutensor_reduction_wrapper(cutensorHandle_t &cutensor_handle,
                               const float* A, float* D,
                               const std::vector<int64_t>& extentA,
                               const std::vector<int64_t>& extentD,
                               const std::vector<int32_t>& modeA,
                               const std::vector<int32_t>& modeD,
                               cutensorOperator_t opReduce,
                               float alpha,
                               float beta,
                               cudaStream_t stream) {
    // General tensor reduction: D = alpha * reduce_op(A) + beta * D
    // Modes that appear in D must also appear in A
    // Modes that only appear in A are reduced
    
    // Compute strides for row-major layout
    auto computeStrides = [](const std::vector<int64_t>& extent) {
        std::vector<int64_t> strides(extent.size());
        if (extent.size() > 0) {
            strides[extent.size() - 1] = 1;
            for (int i = extent.size() - 2; i >= 0; i--) {
                strides[i] = strides[i + 1] * extent[i + 1];
            }
        }
        return strides;
    };
    
    std::vector<int64_t> strideA = computeStrides(extentA);
    std::vector<int64_t> strideD = computeStrides(extentD);
    
    // Data types
    cutensorDataType_t typeA = CUTENSOR_R_32F;
    cutensorDataType_t typeD = CUTENSOR_R_32F;
    cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;
    
    const uint32_t kAlignment = 128;
    
    // Create tensor descriptors
    cutensorTensorDescriptor_t descA, descD;
    
    HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(
        cutensor_handle,
        &descA,
        modeA.size(),
        extentA.data(),
        strideA.data(),
        typeA,
        kAlignment));
    
    HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(
        cutensor_handle,
        &descD,
        modeD.size(),
        extentD.data(),
        strideD.data(),
        typeD,
        kAlignment));
    
    // Create reduction descriptor
    cutensorOperationDescriptor_t desc;
    HANDLE_CUTENSOR_ERROR(cutensorCreateReduction(
        cutensor_handle,
        &desc,
        descA, modeA.data(), CUTENSOR_OP_IDENTITY,
        descD, modeD.data(), CUTENSOR_OP_IDENTITY,
        descD, modeD.data(),
        opReduce,
        descCompute));
    
    // Set algorithm
    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
    
    // Create plan preference
    cutensorPlanPreference_t planPref;
    HANDLE_CUTENSOR_ERROR(cutensorCreatePlanPreference(
        cutensor_handle,
        &planPref,
        algo,
        CUTENSOR_JIT_MODE_NONE));
    
    // Query workspace size
    uint64_t workspaceSizeEstimate = 0;
    const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
    HANDLE_CUTENSOR_ERROR(cutensorEstimateWorkspaceSize(
        cutensor_handle,
        desc,
        planPref,
        workspacePref,
        &workspaceSizeEstimate));
    
    // Create plan
    cutensorPlan_t plan;
    HANDLE_CUTENSOR_ERROR(cutensorCreatePlan(
        cutensor_handle,
        &plan,
        desc,
        planPref,
        workspaceSizeEstimate));
    
    // Query actual workspace size
    uint64_t actualWorkspaceSize = 0;
    HANDLE_CUTENSOR_ERROR(cutensorPlanGetAttribute(
        cutensor_handle,
        plan,
        CUTENSOR_PLAN_REQUIRED_WORKSPACE,
        &actualWorkspaceSize,
        sizeof(actualWorkspaceSize)));
    
    // Allocate workspace if needed
    void* workspace = nullptr;
    if (actualWorkspaceSize > 0) {
        HANDLE_CUDA_ERROR(cudaMalloc(&workspace, actualWorkspaceSize));
    }
    
    // Execute reduction
    HANDLE_CUTENSOR_ERROR(cutensorReduce(
        cutensor_handle,
        plan,
        (void*)&alpha, A,
        (void*)&beta, D,
        D,
        workspace, actualWorkspaceSize, stream));
    
    // Clean up
    if (workspace) {
        HANDLE_CUDA_ERROR(cudaFree(workspace));
    }
    
    HANDLE_CUTENSOR_ERROR(cutensorDestroyPlan(plan));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyPlanPreference(planPref));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyOperationDescriptor(desc));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyTensorDescriptor(descA));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyTensorDescriptor(descD));
}


//=============================================================================
// cuTENSOR Element-wise Trinary Wrapper Implementation
//=============================================================================

void cutensor_elementwise_trinary_wrapper(cutensorHandle_t &cutensor_handle,
                                         const float* A, const float* B, const float* C, float* D,
                                         const std::vector<int64_t>& extentA,
                                         const std::vector<int64_t>& extentB,
                                         const std::vector<int64_t>& extentC,
                                         const std::vector<int64_t>& extentD,
                                         const std::vector<int32_t>& modeA,
                                         const std::vector<int32_t>& modeB,
                                         const std::vector<int32_t>& modeC,
                                         const std::vector<int32_t>& modeD,
                                         cutensorOperator_t opA,
                                         cutensorOperator_t opB,
                                         cutensorOperator_t opC,
                                         cutensorOperator_t opAB,
                                         cutensorOperator_t opABC,
                                         float alpha,
                                         float beta,
                                         float gamma,
                                         cudaStream_t stream) {
    // Helper function to validate and check if broadcasting is needed
    auto validateAndCheckBroadcasting = [&](const std::vector<int32_t>& modeTensor, 
                                           const std::vector<int64_t>& extentTensor,
                                           const char* tensorName) -> bool {
        // First validate: all modes in tensor must exist in D
        for (const auto& mode : modeTensor) {
            auto it = std::find(modeD.begin(), modeD.end(), mode);
            if (it == modeD.end()) {
                fprintf(stderr, "Error: Mode '%c' in tensor %s not found in output tensor D\n", 
                        mode, tensorName);
                fprintf(stderr, "%s modes: ", tensorName);
                for (auto m : modeTensor) fprintf(stderr, "'%c' ", m);
                fprintf(stderr, "\nD modes: ");
                for (auto m : modeD) fprintf(stderr, "'%c' ", m);
                fprintf(stderr, "\n");
                exit(1);
            }
        }
        
        // If tensor has fewer modes than output, it needs broadcasting
        if (modeTensor.size() < modeD.size()) return true;
        
        // If same number of modes, they should match exactly (no broadcasting needed)
        if (modeTensor.size() == modeD.size()) {
            for (size_t i = 0; i < modeTensor.size(); ++i) {
                if (modeTensor[i] != modeD[i]) {
                    fprintf(stderr, "Error: Mode mismatch for tensor %s with same dimensions as D\n", 
                            tensorName);
                    exit(1);
                }
            }
            return false;
        }
        
        // Tensor has more modes than D - invalid
        fprintf(stderr, "Error: Tensor %s has more modes than output tensor D\n", tensorName);
        exit(1);
    };
    
    // Helper to create broadcast shape info
    auto createBroadcastInfo = [&](const std::vector<int32_t>& modeTensor,
                                   const std::vector<int64_t>& extentTensor) {
        std::vector<int64_t> onesExtent;
        std::vector<int32_t> onesModes;
        
        // Find missing modes that need to be broadcasted
        for (size_t i = 0; i < modeD.size(); ++i) {
            auto it = std::find(modeTensor.begin(), modeTensor.end(), modeD[i]);
            if (it == modeTensor.end()) {
                onesExtent.push_back(extentD[i]);
                onesModes.push_back(modeD[i]);
            }
        }
        
        return std::make_pair(onesExtent, onesModes);
    };
    
    // Pointers for potentially broadcasted tensors
    const float* A_use = A;
    const float* B_use = B;
    const float* C_use = C;
    float* A_broadcast = nullptr;
    float* B_broadcast = nullptr;
    float* C_broadcast = nullptr;
    
    // Check and handle broadcasting for A
    if (validateAndCheckBroadcasting(modeA, extentA, "A")) {
        auto broadcastInfo = createBroadcastInfo(modeA, extentA);
        std::vector<int64_t> onesExtent = broadcastInfo.first;
        std::vector<int32_t> onesModes = broadcastInfo.second;
        
        if (!onesExtent.empty()) {
            // Create ones tensor for broadcasting
            size_t onesSize = 1;
            for (auto e : onesExtent) onesSize *= e;
            
            float* d_ones;
            HANDLE_CUDA_ERROR(cudaMalloc(&d_ones, onesSize * sizeof(float)));
            std::vector<float> h_ones(onesSize, 1.0f);
            HANDLE_CUDA_ERROR(cudaMemcpy(d_ones, h_ones.data(), onesSize * sizeof(float), cudaMemcpyHostToDevice));
            
            // Allocate broadcasted tensor
            size_t broadcastSize = 1;
            for (auto e : extentD) broadcastSize *= e;
            HANDLE_CUDA_ERROR(cudaMalloc(&A_broadcast, broadcastSize * sizeof(float)));
            
            // Special case: if A is a scalar (empty modes), we need to fill the output with the scalar value
            if (modeA.empty() && extentA.empty()) {
                // A is a scalar - we need to fill the entire output tensor with this value
                float h_scalar;
                HANDLE_CUDA_ERROR(cudaMemcpy(&h_scalar, A, sizeof(float), cudaMemcpyDeviceToHost));
                std::vector<float> h_broadcast(broadcastSize, h_scalar);
                HANDLE_CUDA_ERROR(cudaMemcpy(A_broadcast, h_broadcast.data(), broadcastSize * sizeof(float), cudaMemcpyHostToDevice));
            } else {
                // Perform broadcasting via contraction
                cutensor_contraction_wrapper(cutensor_handle, d_ones, A, A_broadcast,
                                            onesExtent, extentA, extentD,
                                            onesModes, modeA, modeD, stream);
            }
            
            A_use = A_broadcast;
            HANDLE_CUDA_ERROR(cudaFree(d_ones));
        }
    }
    
    // Check and handle broadcasting for B
    if (validateAndCheckBroadcasting(modeB, extentB, "B")) {
        auto broadcastInfo = createBroadcastInfo(modeB, extentB);
        std::vector<int64_t> onesExtent = broadcastInfo.first;
        std::vector<int32_t> onesModes = broadcastInfo.second;
        
        if (!onesExtent.empty()) {
            // Create ones tensor for broadcasting
            size_t onesSize = 1;
            for (auto e : onesExtent) onesSize *= e;
            
            float* d_ones;
            HANDLE_CUDA_ERROR(cudaMalloc(&d_ones, onesSize * sizeof(float)));
            std::vector<float> h_ones(onesSize, 1.0f);
            HANDLE_CUDA_ERROR(cudaMemcpy(d_ones, h_ones.data(), onesSize * sizeof(float), cudaMemcpyHostToDevice));
            
            // Allocate broadcasted tensor
            size_t broadcastSize = 1;
            for (auto e : extentD) broadcastSize *= e;
            HANDLE_CUDA_ERROR(cudaMalloc(&B_broadcast, broadcastSize * sizeof(float)));
            
            // Special case: if B is a scalar (empty modes), we need to fill the output with the scalar value
            if (modeB.empty() && extentB.empty()) {
                // B is a scalar - we need to fill the entire output tensor with this value
                float h_scalar;
                HANDLE_CUDA_ERROR(cudaMemcpy(&h_scalar, B, sizeof(float), cudaMemcpyDeviceToHost));
                std::vector<float> h_broadcast(broadcastSize, h_scalar);
                HANDLE_CUDA_ERROR(cudaMemcpy(B_broadcast, h_broadcast.data(), broadcastSize * sizeof(float), cudaMemcpyHostToDevice));
            } else {
                // Perform broadcasting via contraction
                cutensor_contraction_wrapper(cutensor_handle, d_ones, B, B_broadcast,
                                            onesExtent, extentB, extentD,
                                            onesModes, modeB, modeD, stream);
            }
            
            B_use = B_broadcast;
            HANDLE_CUDA_ERROR(cudaFree(d_ones));
        }
    }
    
    // Check and handle broadcasting for C
    if (validateAndCheckBroadcasting(modeC, extentC, "C")) {
        auto broadcastInfo = createBroadcastInfo(modeC, extentC);
        std::vector<int64_t> onesExtent = broadcastInfo.first;
        std::vector<int32_t> onesModes = broadcastInfo.second;
        
        if (!onesExtent.empty()) {
            // Create ones tensor for broadcasting
            size_t onesSize = 1;
            for (auto e : onesExtent) onesSize *= e;
            
            float* d_ones;
            HANDLE_CUDA_ERROR(cudaMalloc(&d_ones, onesSize * sizeof(float)));
            std::vector<float> h_ones(onesSize, 1.0f);
            HANDLE_CUDA_ERROR(cudaMemcpy(d_ones, h_ones.data(), onesSize * sizeof(float), cudaMemcpyHostToDevice));
            
            // Allocate broadcasted tensor
            size_t broadcastSize = 1;
            for (auto e : extentD) broadcastSize *= e;
            HANDLE_CUDA_ERROR(cudaMalloc(&C_broadcast, broadcastSize * sizeof(float)));
            
            // Special case: if C is a scalar (empty modes), we need to fill the output with the scalar value
            if (modeC.empty() && extentC.empty()) {
                // C is a scalar - we need to fill the entire output tensor with this value
                float h_scalar;
                HANDLE_CUDA_ERROR(cudaMemcpy(&h_scalar, C, sizeof(float), cudaMemcpyDeviceToHost));
                std::vector<float> h_broadcast(broadcastSize, h_scalar);
                HANDLE_CUDA_ERROR(cudaMemcpy(C_broadcast, h_broadcast.data(), broadcastSize * sizeof(float), cudaMemcpyHostToDevice));
            } else {
                // Perform broadcasting via contraction
                cutensor_contraction_wrapper(cutensor_handle, d_ones, C, C_broadcast,
                                            onesExtent, extentC, extentD,
                                            onesModes, modeC, modeD, stream);
            }
            
            C_use = C_broadcast;
            HANDLE_CUDA_ERROR(cudaFree(d_ones));
        }
    }
    
    // Now perform the trinary operation with all tensors having the same shape
    // Compute strides for row-major layout
    auto computeStrides = [](const std::vector<int64_t>& extent) {
        std::vector<int64_t> strides(extent.size());
        if (extent.size() > 0) {
            strides[extent.size() - 1] = 1;
            for (int i = extent.size() - 2; i >= 0; i--) {
                strides[i] = strides[i + 1] * extent[i + 1];
            }
        }
        return strides;
    };
    
    std::vector<int64_t> strideD = computeStrides(extentD);
    
    // Data types
    cutensorDataType_t typeA = CUTENSOR_R_32F;
    cutensorDataType_t typeB = CUTENSOR_R_32F;
    cutensorDataType_t typeC = CUTENSOR_R_32F;
    cutensorDataType_t typeD = CUTENSOR_R_32F;
    cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;
    
    const uint32_t kAlignment = 128;
    
    // Create tensor descriptors - all with same shape as D
    cutensorTensorDescriptor_t descA, descB, descC, descD;
    
    HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(
        cutensor_handle,
        &descA,
        modeD.size(),
        extentD.data(),
        strideD.data(),
        typeA,
        kAlignment));
    
    HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(
        cutensor_handle,
        &descB,
        modeD.size(),
        extentD.data(),
        strideD.data(),
        typeB,
        kAlignment));
    
    HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(
        cutensor_handle,
        &descC,
        modeD.size(),
        extentD.data(),
        strideD.data(),
        typeC,
        kAlignment));
    
    HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(
        cutensor_handle,
        &descD,
        modeD.size(),
        extentD.data(),
        strideD.data(),
        typeD,
        kAlignment));
    
    // Create element-wise trinary operation descriptor
    cutensorOperationDescriptor_t desc;
    HANDLE_CUTENSOR_ERROR(cutensorCreateElementwiseTrinary(
        cutensor_handle,
        &desc,
        descA, modeD.data(), opA,
        descB, modeD.data(), opB,
        descC, modeD.data(), opC,
        descD, modeD.data(),
        opAB,
        opABC,
        descCompute));
    
    // Set algorithm
    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
    
    // Create plan preference
    cutensorPlanPreference_t planPref;
    HANDLE_CUTENSOR_ERROR(cutensorCreatePlanPreference(
        cutensor_handle,
        &planPref,
        algo,
        CUTENSOR_JIT_MODE_NONE));
    
    // Query workspace size
    uint64_t workspaceSizeEstimate = 0;
    const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
    HANDLE_CUTENSOR_ERROR(cutensorEstimateWorkspaceSize(
        cutensor_handle,
        desc,
        planPref,
        workspacePref,
        &workspaceSizeEstimate));
    
    // Create plan
    cutensorPlan_t plan;
    HANDLE_CUTENSOR_ERROR(cutensorCreatePlan(
        cutensor_handle,
        &plan,
        desc,
        planPref,
        workspaceSizeEstimate));
    
    // Query actual workspace size
    uint64_t actualWorkspaceSize = 0;
    HANDLE_CUTENSOR_ERROR(cutensorPlanGetAttribute(
        cutensor_handle,
        plan,
        CUTENSOR_PLAN_REQUIRED_WORKSPACE,
        &actualWorkspaceSize,
        sizeof(actualWorkspaceSize)));
    
    // Allocate workspace if needed
    void* workspace = nullptr;
    if (actualWorkspaceSize > 0) {
        HANDLE_CUDA_ERROR(cudaMalloc(&workspace, actualWorkspaceSize));
    }
    
    // Execute element-wise trinary operation
    HANDLE_CUTENSOR_ERROR(cutensorElementwiseTrinaryExecute(
        cutensor_handle,
        plan,
        (void*)&alpha, A_use,
        (void*)&beta, B_use,
        (void*)&gamma, C_use,
        D,
        stream));
    
    // Clean up
    if (workspace) {
        HANDLE_CUDA_ERROR(cudaFree(workspace));
    }
    
    if (A_broadcast) HANDLE_CUDA_ERROR(cudaFree(A_broadcast));
    if (B_broadcast) HANDLE_CUDA_ERROR(cudaFree(B_broadcast));
    if (C_broadcast) HANDLE_CUDA_ERROR(cudaFree(C_broadcast));
    
    HANDLE_CUTENSOR_ERROR(cutensorDestroyPlan(plan));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyPlanPreference(planPref));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyOperationDescriptor(desc));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyTensorDescriptor(descA));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyTensorDescriptor(descB));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyTensorDescriptor(descC));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyTensorDescriptor(descD));
}

//=============================================================================
// NeuralLayer Implementation
//=============================================================================

// Random seed management
static unsigned int g_random_seed = 1234;

void NeuralLayer::set_random_seed(unsigned int seed) {
    g_random_seed = seed;
}

unsigned int NeuralLayer::get_time_based_seed() {
    return static_cast<unsigned int>(time(nullptr));
}

// Constructor
NeuralLayer::NeuralLayer(int input_elements, int output_elements, bool non_linear_activate)
    : input_elements_(input_elements), 
      output_elements_(output_elements),
      non_linear_activate_(non_linear_activate),
      zero_size_(0) {
    
    // Allocate device memory for weights and biases
    size_t weight_size = input_elements_ * output_elements_ * sizeof(float);
    size_t bias_size = output_elements_ * sizeof(float);
    
    HANDLE_CUDA_ERROR(cudaMalloc(&W_d_, weight_size));
    HANDLE_CUDA_ERROR(cudaMalloc(&b_d_, bias_size));
    
    // Initialize weights using He initialization for ReLU, Xavier for linear
    int total_weights = input_elements_ * output_elements_;
    int block_size = 256;
    int grid_size = (total_weights + block_size - 1) / block_size;
    
    float scale;
    if (non_linear_activate_) {
        // He initialization for ReLU: sqrt(2/fan_in)
        scale = sqrtf(2.0f / input_elements_);
    } else {
        // Xavier initialization for linear: sqrt(1/fan_in)
        scale = sqrtf(1.0f / input_elements_);
    }
    
    init_weights_kernel<<<grid_size, block_size>>>(W_d_, total_weights, scale, g_random_seed);
    
    // Initialize biases to zero
    HANDLE_CUDA_ERROR(cudaMemset(b_d_, 0, bias_size));
    
    // Initialize pointers
    z_d_ = nullptr;
    zero_d_ = nullptr;
    
    // Note: z_d_ will be allocated dynamically in forward pass based on batch size
    // zero_d_ will be allocated in apply_relu_batch_cutensor when needed
}

// Destructor
NeuralLayer::~NeuralLayer() {
    if (W_d_) cudaFree(W_d_);
    if (b_d_) cudaFree(b_d_);
    if (z_d_) cudaFree(z_d_);
    if (zero_d_) cudaFree(zero_d_);
}

// Forward pass: Y = ReLU(X * W + B) or Y = X * W + B
void NeuralLayer::forward_batch(const float* input_batch, float* output_batch, int batch_size) {
    
    // First compute X * W
    // input_batch: (batch_size, input_elements)
    // W_d_: (input_elements, output_elements)
    // output_batch: (batch_size, output_elements)
    
    // Create cuTENSOR handle
    cutensorHandle_t handle;
    HANDLE_CUTENSOR_ERROR(cutensorCreate(&handle));
    
    // Allocate temporary storage for the linear output
    float* linear_output = output_batch;
    if (non_linear_activate_) {
        // We need z_d_ to store linear output for the entire batch
        size_t batch_output_size = batch_size * output_elements_ * sizeof(float);
        if (z_d_) {
            HANDLE_CUDA_ERROR(cudaFree(z_d_));
        }
        HANDLE_CUDA_ERROR(cudaMalloc(&z_d_, batch_output_size));
        linear_output = z_d_;
    }
    
    // Perform matrix multiplication using cuTENSOR: linear_output = input_batch * W_d_
    // input_batch: (batch_size, input_elements) with modes 'i', 'k'
    // W_d_: (input_elements, output_elements) with modes 'k', 'j'
    // linear_output: (batch_size, output_elements) with modes 'i', 'j'
    
    std::vector<int64_t> extentInput{batch_size, input_elements_};
    std::vector<int64_t> extentW{input_elements_, output_elements_};
    std::vector<int64_t> extentOutput{batch_size, output_elements_};
    
    std::vector<int32_t> modeInput{'i', 'k'};
    std::vector<int32_t> modeW{'k', 'j'};
    std::vector<int32_t> modeOutput{'i', 'j'};
    
    cutensor_contraction_wrapper(handle, input_batch, W_d_, linear_output,
                                extentInput, extentW, extentOutput,
                                modeInput, modeW, modeOutput, 0);
    
    // Add bias using cuTENSOR element-wise operation with broadcasting
    if (non_linear_activate_) {
        // Add bias to linear_output in-place
        add_bias_cutensor(handle, linear_output, b_d_, linear_output,
                         batch_size, output_elements_);
        // Apply ReLU activation
        apply_relu_batch_cutensor(handle, linear_output, output_batch, batch_size, output_elements_);
    } else {
        // Add bias directly to output
        add_bias_cutensor(handle, linear_output, b_d_, output_batch,
                         batch_size, output_elements_);
    }
    
    // Synchronize to ensure completion
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Destroy cuTENSOR handle
    HANDLE_CUTENSOR_ERROR(cutensorDestroy(handle));
}

// Backward pass: compute gradients
void NeuralLayer::backward_batch(const float* input_batch, const float* dy_batch, 
                                float* dW, float* db, float* dx_batch, int batch_size) {
    // Backward pass computes:
    // 1. dW = input^T * dy
    // 2. db = sum(dy, axis=0)
    // 3. dx = dy * W^T
    
    // If we have activation, we need to apply ReLU derivative first
    float* grad_output = const_cast<float*>(dy_batch);
    if (non_linear_activate_) {
        // Allocate temporary storage for ReLU derivative
        float* relu_grad;
        size_t grad_size = batch_size * output_elements_ * sizeof(float);
        HANDLE_CUDA_ERROR(cudaMalloc(&relu_grad, grad_size));
        
        // Apply ReLU derivative: grad = dy * (z > 0)
        apply_relu_derivative_batch(z_d_, dy_batch, relu_grad, batch_size, output_elements_);
        grad_output = relu_grad;
    }
    
    // Create cuTENSOR handle
    cutensorHandle_t handle;
    HANDLE_CUTENSOR_ERROR(cutensorCreate(&handle));
    
    // 1. Compute dW = input^T * grad_output
    // input_batch: (batch_size, input_elements) with modes 'b', 'i'
    // grad_output: (batch_size, output_elements) with modes 'b', 'j'
    // dW: (input_elements, output_elements) with modes 'i', 'j'
    // This is a contraction over the batch dimension 'b'
    
    std::vector<int64_t> extentInput{batch_size, input_elements_};
    std::vector<int64_t> extentGrad{batch_size, output_elements_};
    std::vector<int64_t> extentDW{input_elements_, output_elements_};
    
    std::vector<int32_t> modeInput{'b', 'i'};
    std::vector<int32_t> modeGrad{'b', 'j'};
    std::vector<int32_t> modeDW{'i', 'j'};
    
    cutensor_contraction_wrapper(handle, input_batch, grad_output, dW,
                                extentInput, extentGrad, extentDW,
                                modeInput, modeGrad, modeDW, 0);
    
    // 2. Compute db = sum(grad_output, axis=0) using cuTENSOR reduction
    // grad_output: (batch_size, output_elements) with modes 'b', 'j'
    // db: (output_elements) with mode 'j'
    // We reduce over the batch dimension 'b'
    std::vector<int64_t> extentGradForDb{batch_size, output_elements_};
    std::vector<int64_t> extentDb{output_elements_};
    std::vector<int32_t> modeGradForDb{'b', 'j'};
    std::vector<int32_t> modeDb{'j'};
    
    cutensor_reduction_wrapper(handle, grad_output, db,
                              extentGradForDb, extentDb,
                              modeGradForDb, modeDb,
                              CUTENSOR_OP_ADD,  // Sum reduction
                              1.0f,             // alpha
                              0.0f,             // beta (set to 0 to overwrite)
                              0);               // stream
    
    // 3. Compute dx = grad_output * W^T if needed
    if (dx_batch) {
        // grad_output: (batch_size, output_elements) with modes 'i', 'k'
        // W_d_: (input_elements, output_elements) with modes 'j', 'k'
        // dx_batch: (batch_size, input_elements) with modes 'i', 'j'
        // This is a contraction over the output dimension 'k'
        
        std::vector<int64_t> extentGradForDx{batch_size, output_elements_};
        std::vector<int64_t> extentW{input_elements_, output_elements_};
        std::vector<int64_t> extentDx{batch_size, input_elements_};
        
        std::vector<int32_t> modeGradForDx{'i', 'k'};
        std::vector<int32_t> modeW{'j', 'k'};
        std::vector<int32_t> modeDx{'i', 'j'};
        
        cutensor_contraction_wrapper(handle, grad_output, W_d_, dx_batch,
                                    extentGradForDx, extentW, extentDx,
                                    modeGradForDx, modeW, modeDx, 0);
    }
    
    // Clean up
    if (non_linear_activate_) {
        HANDLE_CUDA_ERROR(cudaFree(grad_output));
    }
    
    // Destroy cuTENSOR handle
    HANDLE_CUTENSOR_ERROR(cutensorDestroy(handle));
}

// Helper function to compute L2 norm of a gradient tensor
float compute_gradient_norm(const float* gradients, int size) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    
    // Allocate memory for partial norms
    float* d_partial_norms;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_partial_norms, grid_size * sizeof(float)));
    
    // Compute partial norms
    compute_norm_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(
        gradients, d_partial_norms, size);
    HANDLE_CUDA_ERROR(cudaGetLastError());
    
    // Reduce partial norms on CPU (could be optimized with another kernel)
    std::vector<float> h_partial_norms(grid_size);
    HANDLE_CUDA_ERROR(cudaMemcpy(h_partial_norms.data(), d_partial_norms, 
                                grid_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    float total_norm_squared = 0.0f;
    for (float norm : h_partial_norms) {
        total_norm_squared += norm;
    }
    
    HANDLE_CUDA_ERROR(cudaFree(d_partial_norms));
    return sqrtf(total_norm_squared);
}

// Helper function to check if tensor contains NaN or Inf
bool check_finite(const float* data, int size) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;
    
    int* d_has_nan_inf;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_has_nan_inf, sizeof(int)));
    HANDLE_CUDA_ERROR(cudaMemset(d_has_nan_inf, 0, sizeof(int)));
    
    check_finite_kernel<<<grid_size, block_size>>>(data, d_has_nan_inf, size);
    HANDLE_CUDA_ERROR(cudaGetLastError());
    
    int has_nan_inf;
    HANDLE_CUDA_ERROR(cudaMemcpy(&has_nan_inf, d_has_nan_inf, sizeof(int), cudaMemcpyDeviceToHost));
    HANDLE_CUDA_ERROR(cudaFree(d_has_nan_inf));
    
    return has_nan_inf == 0;
}

// Helper function to clip gradients
void clip_gradients(float* gradients, int size, float max_norm) {
    // First compute the current norm
    float current_norm = compute_gradient_norm(gradients, size);
    
    // Only clip if norm exceeds threshold
    if (current_norm > max_norm) {
        const int block_size = 256;
        const int grid_size = (size + block_size - 1) / block_size;
        
        clip_gradients_by_norm_kernel<<<grid_size, block_size>>>(
            gradients, size, max_norm, current_norm);
        HANDLE_CUDA_ERROR(cudaGetLastError());
    }
}

// Update parameters with gradient clipping and stability checks
void NeuralLayer::update_parameters(const float* dW, const float* db, float learning_rate,
                                   float max_gradient_norm, float weight_decay) {
    // Check gradients for NaN/Inf
    int weight_size = input_elements_ * output_elements_;
    if (!check_finite(dW, weight_size)) {
        fprintf(stderr, "Warning: Weight gradients contain NaN or Inf! Skipping update.\n");
        return;
    }
    if (!check_finite(db, output_elements_)) {
        fprintf(stderr, "Warning: Bias gradients contain NaN or Inf! Skipping update.\n");
        return;
    }
    
    // Create copies of gradients for clipping
    float* dW_clipped;
    float* db_clipped;
    HANDLE_CUDA_ERROR(cudaMalloc(&dW_clipped, weight_size * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMalloc(&db_clipped, output_elements_ * sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMemcpy(dW_clipped, dW, weight_size * sizeof(float), cudaMemcpyDeviceToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(db_clipped, db, output_elements_ * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Apply weight decay if requested
    if (weight_decay > 0.0f) {
        int block_size = 256;
        int grid_size = (weight_size + block_size - 1) / block_size;
        apply_weight_decay_kernel<<<grid_size, block_size>>>(dW_clipped, W_d_, weight_size, weight_decay);
        HANDLE_CUDA_ERROR(cudaGetLastError());
    }
    
    // Clip gradients by norm
    if (max_gradient_norm > 0.0f) {
        // Compute combined norm of weight and bias gradients
        float weight_norm = compute_gradient_norm(dW_clipped, weight_size);
        float bias_norm = compute_gradient_norm(db_clipped, output_elements_);
        float total_norm = sqrtf(weight_norm * weight_norm + bias_norm * bias_norm);
        
        if (total_norm > max_gradient_norm) {
            float scale = max_gradient_norm / total_norm;
            int block_size = 256;
            
            // Scale weight gradients
            int grid_size = (weight_size + block_size - 1) / block_size;
            clip_gradients_by_norm_kernel<<<grid_size, block_size>>>(
                dW_clipped, weight_size, max_gradient_norm, total_norm);
            
            // Scale bias gradients
            grid_size = (output_elements_ + block_size - 1) / block_size;
            clip_gradients_by_norm_kernel<<<grid_size, block_size>>>(
                db_clipped, output_elements_, max_gradient_norm, total_norm);
            
            HANDLE_CUDA_ERROR(cudaGetLastError());
        }
    }
    
    // Use cuTENSOR to update parameters with clipped gradients
    cutensorHandle_t handle;
    HANDLE_CUTENSOR_ERROR(cutensorCreate(&handle));
    
    // Update weights using cuTENSOR
    int weight_size_update = input_elements_ * output_elements_;
    update_parameters_cutensor(handle, W_d_, dW_clipped, learning_rate, weight_size_update);
    
    // Update biases using cuTENSOR
    update_parameters_cutensor(handle, b_d_, db_clipped, learning_rate, output_elements_);
    
    // Synchronize to ensure updates are complete
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Destroy cuTENSOR handle
    HANDLE_CUTENSOR_ERROR(cutensorDestroy(handle));
    
    // Check parameters after update
    if (!check_finite(W_d_, weight_size)) {
        fprintf(stderr, "Error: Weights contain NaN or Inf after update!\n");
    }
    if (!check_finite(b_d_, output_elements_)) {
        fprintf(stderr, "Error: Biases contain NaN or Inf after update!\n");
    }
    
    // Clean up
    HANDLE_CUDA_ERROR(cudaFree(dW_clipped));
    HANDLE_CUDA_ERROR(cudaFree(db_clipped));
}

// ReLU implementations stubs
void NeuralLayer::apply_relu_batch_cutensor(cutensorHandle_t &handle,
                                           float* input_batch, float* output_batch, 
                                           int batch_size, int size) {
    // ReLU(x) = max(x, 0)
    // Use trinary wrapper with a dummy third tensor
    
    // Define tensor extents and modes
    std::vector<int64_t> extent{batch_size, size};
    std::vector<int32_t> mode{'i', 'j'};
    
    // Allocate zero tensor if not already allocated or if size changed
    size_t zero_size = batch_size * size * sizeof(float);
    if (!zero_d_ || zero_size_ < zero_size) {
        if (zero_d_) {
            HANDLE_CUDA_ERROR(cudaFree(zero_d_));
        }
        HANDLE_CUDA_ERROR(cudaMalloc(&zero_d_, zero_size));
        HANDLE_CUDA_ERROR(cudaMemset(zero_d_, 0, zero_size));
        zero_size_ = zero_size;
    }
    
    // Create a dummy scalar for the third input
    float* d_dummy;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_dummy, sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMemset(d_dummy, 0, sizeof(float)));
    
    std::vector<int64_t> extentDummy{};  // Scalar
    std::vector<int32_t> modeDummy{};    // Scalar has no modes
    
    // Use element-wise trinary wrapper: output = max(input, 0) + 0
    cutensor_elementwise_trinary_wrapper(handle,
                                        input_batch, zero_d_, d_dummy, output_batch,
                                        extent, extent, extentDummy, extent,
                                        mode, mode, modeDummy, mode,
                                        CUTENSOR_OP_IDENTITY,  // opA
                                        CUTENSOR_OP_IDENTITY,  // opB
                                        CUTENSOR_OP_IDENTITY,  // opC
                                        CUTENSOR_OP_MAX,       // opAB
                                        CUTENSOR_OP_ADD,       // opABC
                                        1.0f,                  // alpha
                                        1.0f,                  // beta
                                        0.0f,                  // gamma
                                        0);                    // stream
    
    // Clean up
    HANDLE_CUDA_ERROR(cudaFree(d_dummy));
}

void NeuralLayer::apply_relu_derivative_batch(const float* z_batch, const float* dy_batch, 
                                             float* dz_batch, int batch_size, int size) {
    // ReLU derivative: dz = dy if z > 0, else 0
    int total_elements = batch_size * size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    
    relu_derivative_kernel<<<grid_size, block_size>>>(z_batch, dy_batch, dz_batch, total_elements);
    HANDLE_CUDA_ERROR(cudaGetLastError());
}

//=============================================================================
// CUDA Kernels for Softmax and Cross-Entropy
//=============================================================================

// Kernel to compute softmax in-place
// Each thread handles one element in the batch
__global__ void softmax_kernel(float* logits, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    float* row = logits + idx * num_classes;
    
    // Find max for numerical stability
    float max_val = row[0];
    for (int i = 1; i < num_classes; i++) {
        max_val = fmaxf(max_val, row[i]);
    }
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        row[i] = expf(row[i] - max_val);
        sum += row[i];
    }
    
    // Normalize
    for (int i = 0; i < num_classes; i++) {
        row[i] /= sum;
    }
}

// Kernel to compute cross-entropy loss
__global__ void cross_entropy_loss_kernel(const float* predictions, const float* targets, 
                                         float* losses, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    const float* pred_row = predictions + idx * num_classes;
    const float* target_row = targets + idx * num_classes;
    
    float loss = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        // Avoid log(0) by adding small epsilon
        loss -= target_row[i] * logf(fmaxf(pred_row[i], 1e-7f));
    }
    
    losses[idx] = loss;
}

// Kernel to compute softmax cross-entropy gradient
// gradient = predictions - targets (for softmax + cross-entropy combined)
__global__ void softmax_cross_entropy_gradient_kernel(const float* predictions, const float* targets,
                                                     float* gradients, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * num_classes;
    
    if (idx >= total_elements) return;
    
    gradients[idx] = predictions[idx] - targets[idx];
}

// Kernel to find argmax (for predictions)
__global__ void argmax_kernel(const float* input, int* output, int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    const float* row = input + idx * num_classes;
    int max_idx = 0;
    float max_val = row[0];
    
    for (int i = 1; i < num_classes; i++) {
        if (row[i] > max_val) {
            max_val = row[i];
            max_idx = i;
        }
    }
    
    output[idx] = max_idx;
}

//=============================================================================
// NeuralNetwork Implementation
//=============================================================================

// Random seed management
void NeuralNetwork::set_random_seed(unsigned int seed) {
    NeuralLayer::set_random_seed(seed);
}

unsigned int NeuralNetwork::get_time_based_seed() {
    return NeuralLayer::get_time_based_seed();
}

// Constructor
NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes) {
    if (layer_sizes.size() < 2) {
        throw std::invalid_argument("Network must have at least 2 layers (input and output)");
    }
    
    // Create layers
    for (size_t i = 0; i < layer_sizes.size() - 1; i++) {
        int input_size = layer_sizes[i];
        int output_size = layer_sizes[i + 1];
        bool use_relu = (i < layer_sizes.size() - 2);  // ReLU for all except last layer
        
        layers_.push_back(new NeuralLayer(input_size, output_size, use_relu));
    }
    
    // Allocate device memory for intermediate outputs and gradients
    // We need one output buffer per layer (including input as layer 0 output)
    // and one gradient buffer per layer (excluding input)
    for (size_t i = 0; i < layer_sizes.size(); i++) {
        // Outputs: allocate for max batch size (we'll use dynamic allocation based on actual batch size)
        layer_outputs_.push_back(nullptr);
        
        if (i > 0) {
            // Gradients: only for layers after input
            layer_gradients_.push_back(nullptr);
        }
    }
}

// Destructor
NeuralNetwork::~NeuralNetwork() {
    // Delete all layers
    for (auto layer : layers_) {
        delete layer;
    }
    
    // Free device memory for outputs
    for (auto output : layer_outputs_) {
        if (output) {
            cudaFree(output);
        }
    }
    
    // Free device memory for gradients
    for (auto gradient : layer_gradients_) {
        if (gradient) {
            cudaFree(gradient);
        }
    }
}

// Forward pass
float NeuralNetwork::forward_batch(const float* input_batch, const float* target_batch, int batch_size) {
    // Ensure we have allocated buffers for this batch size
    // Calculate sizes for each layer
    std::vector<int> layer_sizes;
    layer_sizes.push_back(layers_[0]->get_input_elements());  // Input size
    for (auto layer : layers_) {
        layer_sizes.push_back(layer->get_output_elements());
    }
    
    // Allocate/reallocate output buffers as needed
    for (size_t i = 0; i < layer_outputs_.size(); i++) {
        size_t size = batch_size * layer_sizes[i] * sizeof(float);
        if (!layer_outputs_[i]) {
            HANDLE_CUDA_ERROR(cudaMalloc(&layer_outputs_[i], size));
        }
    }
    
    // Copy input to first output buffer
    size_t input_size = batch_size * layer_sizes[0] * sizeof(float);
    HANDLE_CUDA_ERROR(cudaMemcpy(layer_outputs_[0], input_batch, input_size, cudaMemcpyDeviceToDevice));
    
    // Forward pass through all layers
    for (size_t i = 0; i < layers_.size(); i++) {
        layers_[i]->forward_batch(layer_outputs_[i], layer_outputs_[i + 1], batch_size);
    }
    
    // Last layer output contains logits - apply softmax for probability
    int num_classes = layer_sizes.back();
    float* final_output = layer_outputs_.back();
    
    // Create a copy for softmax (to preserve logits for backward pass)
    float* softmax_output;
    size_t output_size = batch_size * num_classes * sizeof(float);
    HANDLE_CUDA_ERROR(cudaMalloc(&softmax_output, output_size));
    HANDLE_CUDA_ERROR(cudaMemcpy(softmax_output, final_output, output_size, cudaMemcpyDeviceToDevice));
    
    // Apply softmax
    int block_size = 256;
    int grid_size = (batch_size + block_size - 1) / block_size;
    softmax_kernel<<<grid_size, block_size>>>(softmax_output, batch_size, num_classes);
    HANDLE_CUDA_ERROR(cudaGetLastError());
    
    // Compute cross-entropy loss
    float* d_losses;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_losses, batch_size * sizeof(float)));
    
    cross_entropy_loss_kernel<<<grid_size, block_size>>>(softmax_output, target_batch, d_losses, 
                                                         batch_size, num_classes);
    HANDLE_CUDA_ERROR(cudaGetLastError());
    
    // Reduce to get average loss using cuTENSOR reduction
    float* d_total_loss;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_total_loss, sizeof(float)));
    
    cutensorHandle_t handle;
    HANDLE_CUTENSOR_ERROR(cutensorCreate(&handle));
    
    std::vector<int64_t> extentA{batch_size};
    std::vector<int64_t> extentD{};  // Scalar output
    std::vector<int32_t> modeA{'i'};
    std::vector<int32_t> modeD{};     // Scalar has no modes
    
    cutensor_reduction_wrapper(handle, d_losses, d_total_loss, 
                              extentA, extentD, modeA, modeD,
                              CUTENSOR_OP_ADD,  // Sum reduction
                              1.0f,             // alpha
                              0.0f,             // beta
                              0);               // stream
    
    // Copy result to host and compute average
    float total_loss;
    HANDLE_CUDA_ERROR(cudaMemcpy(&total_loss, d_total_loss, sizeof(float), cudaMemcpyDeviceToHost));
    float avg_loss = total_loss / batch_size;
    
    // Cleanup
    HANDLE_CUDA_ERROR(cudaFree(softmax_output));
    HANDLE_CUDA_ERROR(cudaFree(d_losses));
    HANDLE_CUDA_ERROR(cudaFree(d_total_loss));
    HANDLE_CUTENSOR_ERROR(cutensorDestroy(handle));
    
    return avg_loss;
}

// Backward pass with gradient clipping and stability checks
void NeuralNetwork::backward_batch(const float* input_batch, const float* target_batch, 
                                       float learning_rate, int batch_size,
                                       float max_gradient_norm, float weight_decay) {
    // Calculate layer sizes
    std::vector<int> layer_sizes;
    layer_sizes.push_back(layers_[0]->get_input_elements());
    for (auto layer : layers_) {
        layer_sizes.push_back(layer->get_output_elements());
    }
    
    // Allocate gradient buffers if needed
    for (size_t i = 0; i < layer_gradients_.size(); i++) {
        size_t size = batch_size * layer_sizes[i + 1] * sizeof(float);
        if (!layer_gradients_[i]) {
            HANDLE_CUDA_ERROR(cudaMalloc(&layer_gradients_[i], size));
        }
    }
    
    // Compute gradient for last layer (softmax + cross-entropy)
    int num_classes = layer_sizes.back();
    float* final_output = layer_outputs_.back();
    float* final_gradient = layer_gradients_.back();
    
    // First, apply softmax to get predictions
    size_t output_size = batch_size * num_classes * sizeof(float);
    HANDLE_CUDA_ERROR(cudaMemcpy(final_gradient, final_output, output_size, cudaMemcpyDeviceToDevice));
    
    int block_size = 256;
    int grid_size = (batch_size + block_size - 1) / block_size;
    softmax_kernel<<<grid_size, block_size>>>(final_gradient, batch_size, num_classes);
    HANDLE_CUDA_ERROR(cudaGetLastError());
    
    // Compute gradient: predictions - targets
    int total_elements = batch_size * num_classes;
    grid_size = (total_elements + block_size - 1) / block_size;
    softmax_cross_entropy_gradient_kernel<<<grid_size, block_size>>>(final_gradient, target_batch, 
                                                                     final_gradient, batch_size, num_classes);
    HANDLE_CUDA_ERROR(cudaGetLastError());
    
    // Scale gradient by 1/batch_size for average loss
    cutensorHandle_t handle;
    HANDLE_CUTENSOR_ERROR(cutensorCreate(&handle));
    
    float scale = 1.0f / batch_size;
    float* d_zero;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_zero, sizeof(float)));
    HANDLE_CUDA_ERROR(cudaMemset(d_zero, 0, sizeof(float)));
    
    std::vector<int64_t> extent{batch_size, num_classes};
    std::vector<int32_t> mode{'i', 'j'};
    std::vector<int64_t> extentScalar{};
    std::vector<int32_t> modeScalar{};
    
    cutensor_elementwise_trinary_wrapper(handle,
                                        final_gradient, d_zero, d_zero, final_gradient,
                                        extent, extentScalar, extentScalar, extent,
                                        mode, modeScalar, modeScalar, mode,
                                        CUTENSOR_OP_IDENTITY, CUTENSOR_OP_IDENTITY, CUTENSOR_OP_IDENTITY,
                                        CUTENSOR_OP_ADD, CUTENSOR_OP_ADD,
                                        scale, 0.0f, 0.0f, 0);
    
    // Backward pass through layers with safe updates
    for (int i = layers_.size() - 1; i >= 0; i--) {
        float* input_to_layer = layer_outputs_[i];
        float* grad_output = layer_gradients_[i];
        float* grad_input = (i > 0) ? layer_gradients_[i - 1] : nullptr;
        
        // Allocate temporary buffers for weight and bias gradients
        int input_size = layers_[i]->get_input_elements();
        int output_size = layers_[i]->get_output_elements();
        
        float* dW;
        float* db;
        HANDLE_CUDA_ERROR(cudaMalloc(&dW, input_size * output_size * sizeof(float)));
        HANDLE_CUDA_ERROR(cudaMalloc(&db, output_size * sizeof(float)));
        
        // Backward pass to compute gradients
        layers_[i]->backward_batch(input_to_layer, grad_output, dW, db, grad_input, batch_size);
        
        // Update parameters with gradient clipping
        layers_[i]->update_parameters(dW, db, learning_rate, max_gradient_norm, weight_decay);
        
        // Free temporary buffers
        HANDLE_CUDA_ERROR(cudaFree(dW));
        HANDLE_CUDA_ERROR(cudaFree(db));
    }
    
    // Cleanup
    HANDLE_CUDA_ERROR(cudaFree(d_zero));
    HANDLE_CUTENSOR_ERROR(cutensorDestroy(handle));
}

// Predict batch - get class predictions
void NeuralNetwork::predict_batch(const float* input_batch, int* predictions, int batch_size) {
    // Calculate sizes
    std::vector<int> layer_sizes;
    layer_sizes.push_back(layers_[0]->get_input_elements());
    for (auto layer : layers_) {
        layer_sizes.push_back(layer->get_output_elements());
    }
    
    // Ensure output buffers are allocated
    for (size_t i = 0; i < layer_outputs_.size(); i++) {
        size_t size = batch_size * layer_sizes[i] * sizeof(float);
        if (!layer_outputs_[i]) {
            HANDLE_CUDA_ERROR(cudaMalloc(&layer_outputs_[i], size));
        }
    }
    
    // Copy input to first output buffer
    size_t input_size = batch_size * layer_sizes[0] * sizeof(float);
    HANDLE_CUDA_ERROR(cudaMemcpy(layer_outputs_[0], input_batch, input_size, cudaMemcpyDeviceToDevice));
    
    // Forward pass through all layers
    for (size_t i = 0; i < layers_.size(); i++) {
        layers_[i]->forward_batch(layer_outputs_[i], layer_outputs_[i + 1], batch_size);
    }
    
    // Get argmax of final layer output (logits)
    int num_classes = layer_sizes.back();
    float* final_output = layer_outputs_.back();
    
    int block_size = 256;
    int grid_size = (batch_size + block_size - 1) / block_size;
    argmax_kernel<<<grid_size, block_size>>>(final_output, predictions, batch_size, num_classes);
    HANDLE_CUDA_ERROR(cudaGetLastError());
}
