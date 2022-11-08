#include "cutlass_wgrad_grouped.h"
// #include "cutlass_error.h"
// #include "base_operation.h"
#include "cuda_runtime.h"
// #include "cutlass_error.h"
// #include "cuda_error_helper.h"
#include <vector>
#include <sstream>
#include <string>
#include <iostream>

#include "cudnn.h"

#define FatalError(s) {                                                \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;\
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(EXIT_FAILURE);                                                \
}

#define checkCudaErrors(status) {                                      \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status                            \
      << "(" << cudaGetErrorString(status) << ")";                    \
      FatalError(_error.str());                                        \
    }                                                                   \
}

#define checkCUTLASS(status) {                                      \
    std::stringstream _error;                                          \
    if (status != cutlass_wgrad_grouped::Status::kSuccess) {             \
      _error << "CUTLASS failure: " << static_cast<int>(status)             \
      << "(" << cutlass_wgrad_grouped::cutlassGetStatusString(status) << ")";                    \
      FatalError(_error.str());                                        \
    }                                                                   \
}

void test_cutlass() {

}

int main() {

    using namespace cutlass_wgrad_grouped;

    printf("Initialize...\n");
    cutlass_wgrad_grouped::initialize_float();

    // Set problems
    std::vector<Conv2dConfig> configs;
    configs.push_back({1, 8, 8, 64, 64, 1, 1, 8, 8, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 8, 8, 64, 64, 3, 3, 8, 8, 1, 1, 1, 1, 1, 1, 1});

    std::vector<void *> host_ptr_A[2];
    std::vector<void *> host_ptr_B[2];
    std::vector<void *> host_ptr_D[2];
    std::vector<void *> host_ptr_D_ref[2];

    for (int i = 0; i < configs.size(); ++i) {
        checkCudaErrors(cudaMalloc((void **)&host_ptr_A[i], sizeof(float)*configs[i].N*configs[i].K*configs[i].P*configs[i].Q));
        checkCudaErrors(cudaMalloc((void **)&host_ptr_B[i], sizeof(float)*configs[i].N*configs[i].C*configs[i].H*configs[i].W));
        checkCudaErrors(cudaMalloc((void **)&host_ptr_D[i], sizeof(float)*configs[i].C*configs[i].K*configs[i].R*configs[i].S));
        checkCudaErrors(cudaMalloc((void **)&host_ptr_D_ref[i], sizeof(float)*configs[i].C*configs[i].K*configs[i].R*configs[i].S));

        float * temp_A = (float *)malloc(sizeof(float)*configs[i].N*configs[i].K*configs[i].P*configs[i].Q);
    }



    void ** device_ptr_A;
    void ** device_ptr_B;
    void ** device_ptr_C;
    void ** device_ptr_D;
    checkCudaErrors(cudaMalloc(&device_ptr_A, sizeof(float *) * 2));
    checkCudaErrors(cudaMalloc(&device_ptr_B, sizeof(float *) * 2));
    checkCudaErrors(cudaMalloc(&device_ptr_C, sizeof(float *) * 2));
    checkCudaErrors(cudaMalloc(&device_ptr_D, sizeof(float *) * 2));

    checkCudaErrors(cudaMemcpy(device_ptr_A, host_ptr_A, sizeof(float *) * 2,
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_ptr_B, host_ptr_B, sizeof(float *) * 2,
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_ptr_C, host_ptr_C, sizeof(float *) * 2,
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_ptr_D, host_ptr_D, sizeof(float *) * 2,
                                cudaMemcpyHostToDevice));

    printf("Initialize problems...\n");
    cutlass_wgrad_grouped::initialize_problems<float>(configs);

    printf("Get best opeartion...");
    OperationWithWorkspace best_operation = get_best_operation(device_ptr_A, device_ptr_B, device_ptr_C, device_ptr_D);

    printf("%p\n", best_operation.operation);

    Status status = run(best_operation);

    printf("%s\n", cutlassGetStatusString(status));

    checkCudaErrors(cudaDeviceSynchronize());

    cutlass_wgrad_grouped::finalize();


    // Verify with cuDNN

    
    return 0;
}