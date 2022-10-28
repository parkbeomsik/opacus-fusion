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


int main() {

    using namespace cutlass_wgrad_grouped;

    printf("Initialize...\n");
    cutlass_wgrad_grouped::initialize_int_tensorop();

    std::vector<Conv2dConfig> configs;

    configs.push_back({1, 8, 8, 64, 64, 1, 1, 8, 8, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 8, 8, 64, 64, 3, 3, 8, 8, 1, 1, 1, 1, 1, 1, 1});
    configs.push_back({1, 8, 8, 64, 256, 1, 1, 8, 8, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 8, 8, 64, 256, 1, 1, 8, 8, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 8, 8, 256, 64, 1, 1, 8, 8, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 8, 8, 64, 64, 3, 3, 8, 8, 1, 1, 1, 1, 1, 1, 1});
    configs.push_back({1, 8, 8, 64, 256, 1, 1, 8, 8, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 8, 8, 256, 64, 1, 1, 8, 8, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 8, 8, 64, 64, 3, 3, 8, 8, 1, 1, 1, 1, 1, 1, 1});
    configs.push_back({1, 8, 8, 64, 256, 1, 1, 8, 8, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 8, 8, 256, 128, 1, 1, 8, 8, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 8, 8, 128, 128, 3, 3, 4, 4, 1, 1, 2, 2, 1, 1, 1});
    configs.push_back({1, 4, 4, 128, 512, 1, 1, 4, 4, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 8, 8, 256, 512, 1, 1, 4, 4, 0, 0, 2, 2, 1, 1, 1});
    configs.push_back({1, 4, 4, 512, 128, 1, 1, 4, 4, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 4, 4, 128, 128, 3, 3, 4, 4, 1, 1, 1, 1, 1, 1, 1});
    configs.push_back({1, 4, 4, 128, 512, 1, 1, 4, 4, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 4, 4, 512, 128, 1, 1, 4, 4, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 4, 4, 128, 128, 3, 3, 4, 4, 1, 1, 1, 1, 1, 1, 1});
    configs.push_back({1, 4, 4, 128, 512, 1, 1, 4, 4, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 4, 4, 512, 128, 1, 1, 4, 4, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 4, 4, 128, 128, 3, 3, 4, 4, 1, 1, 1, 1, 1, 1, 1});
    configs.push_back({1, 4, 4, 128, 512, 1, 1, 4, 4, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 4, 4, 512, 256, 1, 1, 4, 4, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 4, 4, 256, 256, 3, 3, 2, 2, 1, 1, 2, 2, 1, 1, 1});
    configs.push_back({1, 2, 2, 256, 1024, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 4, 4, 512, 1024, 1, 1, 2, 2, 0, 0, 2, 2, 1, 1, 1});
    configs.push_back({1, 2, 2, 1024, 256, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 2, 2, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1});
    configs.push_back({1, 2, 2, 256, 1024, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 2, 2, 1024, 256, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 2, 2, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1});
    configs.push_back({1, 2, 2, 256, 1024, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 2, 2, 1024, 256, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 2, 2, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1});
    configs.push_back({1, 2, 2, 256, 1024, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 2, 2, 1024, 256, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 2, 2, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1});
    configs.push_back({1, 2, 2, 256, 1024, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 2, 2, 1024, 256, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 2, 2, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1});
    configs.push_back({1, 2, 2, 256, 1024, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 2, 2, 1024, 512, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 2, 2, 512, 512, 3, 3, 1, 1, 1, 1, 2, 2, 1, 1, 1});
    configs.push_back({1, 1, 1, 512, 2048, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 2, 2, 1024, 2048, 1, 1, 1, 1, 0, 0, 2, 2, 1, 1, 1});
    configs.push_back({1, 1, 1, 2048, 512, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 1, 1, 512, 512, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    configs.push_back({1, 1, 1, 512, 2048, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 1, 1, 2048, 512, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1});
    configs.push_back({1, 1, 1, 512, 512, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    configs.push_back({1, 1, 1, 512, 2048, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1});

    void * host_ptr_A[53];
    void * host_ptr_B[53];
    void * host_ptr_C[53];
    void * host_ptr_D[53];

    for (int i = 0; i < configs.size(); ++i) {
        checkCudaErrors(cudaMalloc(&host_ptr_A[i], sizeof(int8_t)*configs[i].N*configs[i].K*configs[i].P*configs[i].Q));
        checkCudaErrors(cudaMalloc(&host_ptr_B[i], sizeof(int8_t)*configs[i].N*configs[i].C*configs[i].H*configs[i].W));
        checkCudaErrors(cudaMalloc(&host_ptr_C[i], sizeof(float)*configs[i].C*configs[i].K*configs[i].R*configs[i].S));
        checkCudaErrors(cudaMalloc(&host_ptr_D[i], sizeof(float)*configs[i].C*configs[i].K*configs[i].R*configs[i].S));
    }

    void ** device_ptr_A;
    void ** device_ptr_B;
    void ** device_ptr_C;
    void ** device_ptr_D;
    checkCudaErrors(cudaMalloc(&device_ptr_A, sizeof(int8_t *) * 53));
    checkCudaErrors(cudaMalloc(&device_ptr_B, sizeof(int8_t *) * 53));
    checkCudaErrors(cudaMalloc(&device_ptr_C, sizeof(float *) * 53));
    checkCudaErrors(cudaMalloc(&device_ptr_D, sizeof(float *) * 53));

    checkCudaErrors(cudaMemcpy(device_ptr_A, host_ptr_A, sizeof(int8_t *) * 53,
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_ptr_B, host_ptr_B, sizeof(int8_t *) * 53,
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_ptr_C, host_ptr_C, sizeof(float *) * 53,
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_ptr_D, host_ptr_D, sizeof(float *) * 53,
                                cudaMemcpyHostToDevice));

    printf("Initialize problems...\n");
    cutlass_wgrad_grouped::initialize_problems<int8_t>(configs);

    printf("Get best opeartion...");
    OperationWithWorkspace best_operation = get_best_operation(device_ptr_A, device_ptr_B, device_ptr_C, device_ptr_D);

    printf("%p\n", best_operation.operation);

    Status status = run(best_operation);

    printf("%s\n", cutlassGetStatusString(status));

    checkCudaErrors(cudaDeviceSynchronize());

    cutlass_wgrad_grouped::finalize();
    
    return 0;
}