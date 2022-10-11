#include "cutlass_wgrad_grouped.h"
// #include "cutlass_error.h"
// #include "base_operation.h"
#include "cuda_runtime.h"
// #include "cutlass_error.h"
#include "cuda_error_helper.h"
#include <vector>

int main() {

    using namespace cutlass_wgrad_grouped;

    printf("Initialize...\n");
    cutlass_wgrad_grouped::initialize();

    std::vector<Conv2dConfig> configs(52);

    configs.at(0) = {1, 8, 8, 64, 64, 1, 1, 8, 8, 0, 0, 1, 1, 1, 1};
    configs.at(1) = {1, 8, 8, 64, 64, 3, 3, 8, 8, 1, 1, 1, 1, 1, 1};
    configs.at(2) = {1, 8, 8, 64, 256, 1, 1, 8, 8, 0, 0, 1, 1, 1, 1};
    configs.at(3) = {1, 8, 8, 64, 256, 1, 1, 8, 8, 0, 0, 1, 1, 1, 1};
    configs.at(4) = {1, 8, 8, 256, 64, 1, 1, 8, 8, 0, 0, 1, 1, 1, 1};
    configs.at(5) = {1, 8, 8, 64, 64, 3, 3, 8, 8, 1, 1, 1, 1, 1, 1};
    configs.at(6) = {1, 8, 8, 64, 256, 1, 1, 8, 8, 0, 0, 1, 1, 1, 1};
    configs.at(7) = {1, 8, 8, 256, 64, 1, 1, 8, 8, 0, 0, 1, 1, 1, 1};
    configs.at(8) = {1, 8, 8, 64, 64, 3, 3, 8, 8, 1, 1, 1, 1, 1, 1};
    configs.at(10-1) = {1, 8, 8, 64, 256, 1, 1, 8, 8, 0, 0, 1, 1, 1, 1};
    configs.at(11-1) = {1, 8, 8, 256, 128, 1, 1, 8, 8, 0, 0, 1, 1, 1, 1};
    configs.at(12-1) = {1, 8, 8, 128, 128, 3, 3, 4, 4, 1, 1, 2, 2, 1, 1};
    configs.at(13-1) = {1, 4, 4, 128, 512, 1, 1, 4, 4, 0, 0, 1, 1, 1, 1};
    configs.at(14-1) = {1, 8, 8, 256, 512, 1, 1, 4, 4, 0, 0, 2, 2, 1, 1};
    configs.at(15-1) = {1, 4, 4, 512, 128, 1, 1, 4, 4, 0, 0, 1, 1, 1, 1};
    configs.at(16-1) = {1, 4, 4, 128, 128, 3, 3, 4, 4, 1, 1, 1, 1, 1, 1};
    configs.at(17-1) = {1, 4, 4, 128, 512, 1, 1, 4, 4, 0, 0, 1, 1, 1, 1};
    configs.at(18-1) = {1, 4, 4, 512, 128, 1, 1, 4, 4, 0, 0, 1, 1, 1, 1};
    configs.at(19-1) = {1, 4, 4, 128, 128, 3, 3, 4, 4, 1, 1, 1, 1, 1, 1};
    configs.at(20-1) = {1, 4, 4, 128, 512, 1, 1, 4, 4, 0, 0, 1, 1, 1, 1};
    configs.at(21-1) = {1, 4, 4, 512, 128, 1, 1, 4, 4, 0, 0, 1, 1, 1, 1};
    configs.at(22-1) = {1, 4, 4, 128, 128, 3, 3, 4, 4, 1, 1, 1, 1, 1, 1};
    configs.at(23-1) = {1, 4, 4, 128, 512, 1, 1, 4, 4, 0, 0, 1, 1, 1, 1};
    configs.at(24-1) = {1, 4, 4, 512, 256, 1, 1, 4, 4, 0, 0, 1, 1, 1, 1};
    configs.at(25-1) = {1, 4, 4, 256, 256, 3, 3, 2, 2, 1, 1, 2, 2, 1, 1};
    configs.at(26-1) = {1, 2, 2, 256, 1024, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1};
    configs.at(27-1) = {1, 4, 4, 512, 1024, 1, 1, 2, 2, 0, 0, 2, 2, 1, 1};
    configs.at(28-1) = {1, 2, 2, 1024, 256, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1};
    configs.at(29-1) = {1, 2, 2, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1};
    configs.at(30-1) = {1, 2, 2, 256, 1024, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1};
    configs.at(31-1) = {1, 2, 2, 1024, 256, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1};
    configs.at(32-1) = {1, 2, 2, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1};
    configs.at(33-1) = {1, 2, 2, 256, 1024, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1};
    configs.at(34-1) = {1, 2, 2, 1024, 256, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1};
    configs.at(35-1) = {1, 2, 2, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1};
    configs.at(36-1) = {1, 2, 2, 256, 1024, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1};
    configs.at(37-1) = {1, 2, 2, 1024, 256, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1};
    configs.at(38-1) = {1, 2, 2, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1};
    configs.at(39-1) = {1, 2, 2, 256, 1024, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1};
    configs.at(40-1) = {1, 2, 2, 1024, 256, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1};
    configs.at(41-1) = {1, 2, 2, 256, 256, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1};
    configs.at(42-1) = {1, 2, 2, 256, 1024, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1};
    configs.at(43-1) = {1, 2, 2, 1024, 512, 1, 1, 2, 2, 0, 0, 1, 1, 1, 1};
    configs.at(44-1) = {1, 2, 2, 512, 512, 3, 3, 1, 1, 1, 1, 2, 2, 1, 1};
    configs.at(45-1) = {1, 1, 1, 512, 2048, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1};
    configs.at(46-1) = {1, 2, 2, 1024, 2048, 1, 1, 1, 1, 0, 0, 2, 2, 1, 1};
    configs.at(47-1) = {1, 1, 1, 2048, 512, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1};
    configs.at(48-1) = {1, 1, 1, 512, 512, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1};
    configs.at(49-1) = {1, 1, 1, 512, 2048, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1};
    configs.at(50-1) = {1, 1, 1, 2048, 512, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1};
    configs.at(51-1) = {1, 1, 1, 512, 512, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1};
    configs.at(52-1) = {1, 1, 1, 512, 2048, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1};

    printf("Initialize problems...\n");
    cutlass_wgrad_grouped::initialize_problems(configs);

    void * host_ptr_A[52];
    void * host_ptr_B[52];
    void * host_ptr_C[52];
    void * host_ptr_D[52];

    for (int i = 0; i < configs.size(); ++i) {
        checkCudaErrors(cudaMalloc(&host_ptr_A[i], sizeof(float)*configs[i].N*configs[i].K*configs[i].P*configs[i].Q));
        checkCudaErrors(cudaMalloc(&host_ptr_B[i], sizeof(float)*configs[i].N*configs[i].C*configs[i].H*configs[i].W));
        checkCudaErrors(cudaMalloc(&host_ptr_C[i], sizeof(float)*configs[i].C*configs[i].K*configs[i].R*configs[i].S));
        checkCudaErrors(cudaMalloc(&host_ptr_D[i], sizeof(float)*configs[i].C*configs[i].K*configs[i].R*configs[i].S));
    }

    void ** device_ptr_A;
    void ** device_ptr_B;
    void ** device_ptr_C;
    void ** device_ptr_D;
    checkCudaErrors(cudaMalloc(&device_ptr_A, sizeof(float *) * 52));
    checkCudaErrors(cudaMalloc(&device_ptr_B, sizeof(float *) * 52));
    checkCudaErrors(cudaMalloc(&device_ptr_C, sizeof(float *) * 52));
    checkCudaErrors(cudaMalloc(&device_ptr_D, sizeof(float *) * 52));

    checkCudaErrors(cudaMemcpy(device_ptr_A, host_ptr_A, sizeof(float *) * 52,
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_ptr_B, host_ptr_B, sizeof(float *) * 52,
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_ptr_C, host_ptr_C, sizeof(float *) * 52,
                                cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_ptr_D, host_ptr_D, sizeof(float *) * 52,
                                cudaMemcpyHostToDevice));

    printf("Get best opeartion...");
    OperationWithWorkspace best_operation = get_best_operation(device_ptr_A, device_ptr_B, device_ptr_C, device_ptr_D);

    printf("%p\n", best_operation.operation);

    Status status = run(best_operation);

    printf("%s\n", cutlassGetStatusString(status));

    checkCudaErrors(cudaDeviceSynchronize());

    cutlass_wgrad_grouped::finalize();
    
    return 0;
}